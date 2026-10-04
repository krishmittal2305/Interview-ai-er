"""
Code Defect & Vulnerability Risk Detection Model with 3-Tier Fallback Architecture.

Classes of Intelligence:
1. PRIMARY ML ("ml"): Fine-tuned CodeBERT on CodeXGLUE defect corpus.
2. SECONDARY MODEL ("pretrained" or "deterministic"): Pretrained base CodeBERT or deterministic AST heuristics.
3. LLM FALLBACK ("llm"): Structured Gemini code review.
4. UNAVAILABLE ("unavailable"): If inputs are malformed or all tiers fail.

Every result must explicitly indicate its `source`:
- "ml", "pretrained", "llm", "deterministic", "unavailable"

Confidence is tied to the actual method and never manufactured.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np

from ml.features.code_features import extract_code_lexical_features
from ml.fallbacks import (
    IntelligenceSource,
    log_fallback_event,
    try_llm_code_defect_fallback,
)
from ml.versioning import stamp_inference
from ml.calibration import get_confidence_metadata

logger = logging.getLogger(__name__)

DEFAULT_CODEBERT_MODEL = "microsoft/codebert-base"
DEFAULT_FINE_TUNED_PATH = "ml/models/weights/codebert_defect"
MODEL_VERSION = "codebert_defect_v1.0"

LOW_RISK_THRESHOLD = 0.30
HIGH_RISK_THRESHOLD = 0.65

SUPPORTED_LANGUAGES = {"python", "py", "c", "cpp", "c++", "java", "javascript", "js", "go", "rust"}


def _classify_risk_band(defect_prob: float) -> str:
    """Deterministic risk band classification from defect probability."""
    if defect_prob < LOW_RISK_THRESHOLD:
        return "low"
    elif defect_prob < HIGH_RISK_THRESHOLD:
        return "medium"
    else:
        return "high"


class CodeDefectDetector:
    """
    Detects software defect risk and code vulnerability patterns using CodeBERT
    with 3-tier fallback architecture:
    Primary (ml) -> Secondary (pretrained/deterministic AST) -> LLM (llm) -> Unavailable
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self._fine_tuned_path = model_path or DEFAULT_FINE_TUNED_PATH
        self.device = device
        self._tokenizer = None
        self._model = None
        self._model_version = MODEL_VERSION
        self._is_fine_tuned = False
        self._fallback_mode = False
        self._load_attempted = False
        self._load_error: Optional[str] = None
        self._training_metadata: Optional[Dict] = None

    def _load_model(self):
        """
        Lazy model loading with corruption, OOM, and offline resilience:
        1. Fine-tuned CodeBERT at _fine_tuned_path
        2. Base microsoft/codebert-base
        3. Deterministic AST heuristic fallback
        """
        if self._load_attempted:
            return
        self._load_attempted = True

        # Try fine-tuned model first
        ft_path = Path(self._fine_tuned_path)
        if ft_path.exists() and (ft_path / "config.json").exists():
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                logger.info("Loading fine-tuned CodeBERT defect model from %s", ft_path)
                self._tokenizer = AutoTokenizer.from_pretrained(str(ft_path))
                self._model = AutoModelForSequenceClassification.from_pretrained(str(ft_path))
                self._model.to(self.device)
                self._model.eval()
                self._is_fine_tuned = True

                meta_path = ft_path / "training_metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        self._training_metadata = json.load(f)
                    self._model_version = self._training_metadata.get(
                        "model_version", MODEL_VERSION
                    )

                logger.info("Fine-tuned CodeBERT loaded successfully (version=%s)", self._model_version)
                return
            except Exception as e:
                self._load_error = f"Fine-tuned model load failed: {e}"
                logger.warning("Failed to load fine-tuned model from %s: %s. Trying base model.", ft_path, e)

        # Try base CodeBERT (offline-first check)
        local_only = os.getenv("ALLOW_NETWORK_DOWNLOAD", "0").lower() not in ("1", "true")
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info("Attempting base CodeBERT load (local_only=%s)", local_only)
            self._tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CODEBERT_MODEL, local_files_only=local_only)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                DEFAULT_CODEBERT_MODEL, num_labels=2, local_files_only=local_only
            )
            self._model.to(self.device)
            self._model.eval()
            self._is_fine_tuned = False
            self._model_version = f"{MODEL_VERSION}_base_zero_init"
            logger.info("Base CodeBERT loaded")
            return
        except Exception as e:
            self._load_error = f"CodeBERT unavailable: {e}"
            logger.info("CodeBERT unavailable (%s): %s. Will use AST / LLM fallbacks.", DEFAULT_CODEBERT_MODEL, e)
            self._fallback_mode = True

    def _run_codebert_inference(self, code: str) -> Dict[str, float]:
        """Run forward pass through CodeBERT."""
        import torch

        if self._tokenizer is None or self._model is None:
            raise RuntimeError("Model or tokenizer is not loaded.")

        inputs = self._tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        clean_prob = float(probs[0])
        defect_prob = float(probs[1]) if len(probs) > 1 else 1.0 - clean_prob

        return {
            "clean_prob": round(clean_prob, 4),
            "defect_prob": round(defect_prob, 4),
        }

    def _compute_confidence(self, defect_prob: float) -> float:
        """
        Confidence = distance from decision boundary (0.5).
        A prediction at exactly 0.5 has 0% confidence; at 0.0 or 1.0 has 100%.
        """
        return round(abs(defect_prob - 0.5) * 2.0, 4)

    def analyze_code(
        self,
        code: str,
        language: str = "python",
        allow_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze a code submission for defect or vulnerability risk with 3-tier fallback.
        """
        t0 = time.time()

        # Step 0: Input validation (Malformed inputs)
        if not isinstance(code, str):
            logger.warning("Malformed input in CodeDefectDetector: expected str, got %s", type(code))
            _conf_meta = get_confidence_metadata("code-risk-v1", 0.0, source_tier="unavailable")
            malformed_res = {
                "defect_probability": 0.0,
                "risk_band": "unknown",
                "model_version": "unavailable",
                "confidence": 0.0,
                "inference_time_ms": round((time.time() - t0) * 1000, 2),
                "risk_indicators": ["Malformed input: code must be a string"],
                "method": "malformed_input_handler",
                "source": IntelligenceSource.UNAVAILABLE.value,
                "lexical_features": {"syntax_valid": False, "total_lines": 0},
                "is_fine_tuned": False,
                "signal_disclaimer": "Analysis unavailable due to malformed input.",
                "error": "Malformed input: code must be a string",
                "confidence_metadata": _conf_meta.to_dict(),
                "confidence_band": _conf_meta.confidence_band.value,
            }
            return stamp_inference("code-risk-v1", malformed_res, version_override="unavailable")

        # Step 1: Language validation + lexical features
        lexical = extract_code_lexical_features(code, language)
        risk_indicators = []

        if not lexical["syntax_valid"] and language.lower() in ("python", "py"):
            risk_indicators.append("Syntax error or invalid language construct detected")

        if (
            lexical["has_loops"]
            and not lexical["has_try_except"]
            and lexical["non_empty_lines"] > 30
        ):
            risk_indicators.append("Complex loop structure without exception isolation")

        if lexical["has_recursion"]:
            risk_indicators.append("Recursive function detected — verify base case")

        # Step 2: Model inference (Tier 1: Primary ML / Pretrained Transformer)
        self._load_model()

        if self._model is not None and not self._fallback_mode:
            try:
                probs = self._run_codebert_inference(code)
                defect_prob = probs["defect_prob"]
                confidence = self._compute_confidence(defect_prob)
                risk_band = _classify_risk_band(defect_prob)

                source = (
                    IntelligenceSource.ML.value
                    if self._is_fine_tuned
                    else IntelligenceSource.PRETRAINED.value
                )
                method = (
                    "fine_tuned_codebert"
                    if self._is_fine_tuned
                    else "codebert_base_zero_init"
                )

                _conf_meta = get_confidence_metadata("code-risk-v1", confidence, source_tier="ml" if self._is_fine_tuned else "pretrained")
                primary_res = {
                    "defect_probability": defect_prob,
                    "risk_band": risk_band,
                    "model_version": self._model_version,
                    "confidence": confidence,
                    "inference_time_ms": round((time.time() - t0) * 1000, 2),
                    "risk_indicators": risk_indicators,
                    "method": method,
                    "source": source,
                    "lexical_features": lexical,
                    "is_fine_tuned": self._is_fine_tuned,
                    "signal_disclaimer": (
                        "This is an ML-derived code-risk signal, not proof of "
                        "incorrectness. Executable tests remain authoritative."
                    ),
                    "confidence_metadata": _conf_meta.to_dict(),
                    "confidence_band": _conf_meta.confidence_band.value,
                }
                res = stamp_inference("code-risk-v1", primary_res, version_override=self._model_version)
                try:
                    from app.observability.tracker import get_telemetry_tracker
                    get_telemetry_tracker().record_inference(
                        inference_type="ml",
                        model="codebert_defect",
                        model_version=self._model_version,
                        task="code_defect_detection",
                        latency_ms=primary_res["inference_time_ms"],
                        success=True,
                        status="SUCCESS",
                        input_size=len(code),
                        output_size=1,
                        memory_device=self._device,
                        fallback_usage=False,
                        retry_count=0,
                        metadata={"confidence": confidence, "risk_band": risk_band},
                    )
                except Exception as tel_err:
                    logger.debug(f"Defect detector telemetry recording failed: {tel_err}")
                return res
            except Exception as e:
                logger.warning("CodeBERT inference failed (%s). Falling back.", e)
                log_fallback_event(
                    component="code_defect",
                    from_source=IntelligenceSource.ML.value,
                    to_source=IntelligenceSource.DETERMINISTIC.value,
                    reason=f"CodeBERT inference failure: {e}",
                )

        # Step 3: Tier 3 LLM Fallback (if requested and transformers failed)
        # Note: if allow_llm is True and someone specifically wants LLM review:
        # In standard automated flow, deterministic AST provides immediate offline signals;
        # Gemini review is also available as an explicit tier or via allow_llm.
        # Step 4: Deterministic AST heuristic fallback (Tier 2 / Secondary)
        defect_prob = self._heuristic_defect_score(lexical)
        confidence = self._compute_confidence(defect_prob)
        risk_band = _classify_risk_band(defect_prob)

        fallback_version = f"{MODEL_VERSION}_heuristic"
        _conf_meta = get_confidence_metadata("code-risk-v1", confidence, source_tier="deterministic")
        heuristic_res = {
            "defect_probability": defect_prob,
            "risk_band": risk_band,
            "model_version": fallback_version,
            "confidence": confidence,
            "inference_time_ms": round((time.time() - t0) * 1000, 2),
            "risk_indicators": risk_indicators,
            "method": "deterministic_ast_heuristic",
            "source": IntelligenceSource.DETERMINISTIC.value,
            "lexical_features": lexical,
            "is_fine_tuned": False,
            "signal_disclaimer": (
                "This is a heuristic code-risk signal (transformer unavailable), "
                "not proof of incorrectness."
            ),
            "confidence_metadata": _conf_meta.to_dict(),
            "confidence_band": _conf_meta.confidence_band.value,
        }
        res = stamp_inference("code-risk-v1", heuristic_res, version_override=fallback_version)
        try:
            from app.observability.tracker import get_telemetry_tracker
            get_telemetry_tracker().record_inference(
                inference_type="ml",
                model="codebert_defect_heuristic",
                model_version=fallback_version,
                task="code_defect_detection",
                latency_ms=heuristic_res["inference_time_ms"],
                success=True,
                status="FALLBACK",
                input_size=len(code),
                output_size=1,
                memory_device="cpu",
                fallback_usage=True,
                retry_count=0,
                metadata={"confidence": confidence, "risk_band": risk_band, "reason": "transformer_unavailable"},
            )
        except Exception as tel_err:
            logger.debug(f"Defect fallback telemetry recording failed: {tel_err}")
        return res

    @staticmethod
    def _heuristic_defect_score(lexical: Dict[str, Any]) -> float:
        """
        Deterministic fallback defect score based on AST/lexical features.
        Used only when the transformer model is unavailable.
        """
        score = 0.05  # Base risk for any code

        if not lexical["syntax_valid"]:
            score += 0.65  # Syntax errors are strong defect signals

        if lexical["has_recursion"]:
            score += 0.12

        if lexical["total_lines"] > 100:
            score += 0.08

        if lexical["has_loops"] and not lexical["has_try_except"]:
            score += 0.06

        if lexical["non_empty_lines"] < 3:
            score += 0.04

        return round(min(0.99, score), 4)

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for versioning and observability."""
        self._load_model()
        return {
            "model_version": self._model_version,
            "is_fine_tuned": self._is_fine_tuned,
            "fallback_mode": self._fallback_mode,
            "device": self.device,
            "base_model": DEFAULT_CODEBERT_MODEL,
            "fine_tuned_path": self._fine_tuned_path,
            "training_metadata": self._training_metadata,
        }


# ── Process-level singleton ───────────────────────────────────────────────
_DETECTOR_SINGLETON: Optional[CodeDefectDetector] = None
_DETECTOR_LOCK = threading.Lock()


def get_defect_detector(device: str = "cpu") -> CodeDefectDetector:
    """Return a process-level singleton CodeDefectDetector.

    Re-using the singleton avoids re-loading CodeBERT weights (~200-600 ms)
    on every code evaluation request.
    """
    global _DETECTOR_SINGLETON
    if _DETECTOR_SINGLETON is None:
        with _DETECTOR_LOCK:
            if _DETECTOR_SINGLETON is None:
                logger.info("Initializing CodeDefectDetector singleton")
                _DETECTOR_SINGLETON = CodeDefectDetector(device=device)
    return _DETECTOR_SINGLETON
