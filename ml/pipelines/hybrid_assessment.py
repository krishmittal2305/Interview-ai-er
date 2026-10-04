"""
Hybrid Assessment Pipeline.

Orchestrates the ML-derived concept coverage signal and feeds structured
evidence into the Gemini evaluator, so Gemini explains evidence rather than
inventing its own.

Pipeline flow:
    1. Extract testable concepts from the question rubric
    2. Run ConceptCoverageAnalyzer (DeBERTa NLI + MiniLM similarity)
    3. Format structured ML evidence
    4. Inject evidence into Gemini prompt
    5. Gemini reasons over the evidence and produces the final evaluation

This module does NOT replace the existing Gemini evaluator.
It augments it with an ML evidence layer.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

from ml.models.concept_coverage import ConceptCoverageAnalyzer

logger = logging.getLogger(__name__)

# ── Singleton cache ────────────────────────────────────────────────────
# The ConceptCoverageAnalyzer loads DeBERTa NLI + MiniLM on first use.
# Keeping a process-level singleton avoids re-loading weights on every
# evaluate_answer() call (typically 500-1500 ms saved per request).
_PIPELINE_SINGLETON: Optional["HybridAssessmentPipeline"] = None
_PIPELINE_LOCK = threading.Lock()


def get_hybrid_pipeline(device: str = "cpu") -> "HybridAssessmentPipeline":
    """Return a process-level singleton HybridAssessmentPipeline."""
    global _PIPELINE_SINGLETON
    if _PIPELINE_SINGLETON is None:
        with _PIPELINE_LOCK:
            if _PIPELINE_SINGLETON is None:
                logger.info("Initializing HybridAssessmentPipeline singleton")
                _PIPELINE_SINGLETON = HybridAssessmentPipeline(device=device)
    return _PIPELINE_SINGLETON


class HybridAssessmentPipeline:
    """
    Produces structured ML evidence for answer evaluation and formats it
    for injection into the Gemini evaluator prompt.
    """

    def __init__(self, device: str = "cpu"):
        self.concept_analyzer = ConceptCoverageAnalyzer(device=device)
        self._initialized = False

    def _ensure_initialized(self):
        """Lazy initialization — only load models when first needed."""
        if not self._initialized:
            logger.info("HybridAssessmentPipeline: warming up ML models on first use")
            self._initialized = True

    def generate_ml_evidence(
        self,
        candidate_answer: str,
        question_text: str,
        evaluation_rubric: str,
        expected_concepts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate ML-derived concept coverage evidence for a candidate answer.

        Args:
            candidate_answer: The candidate's response text.
            question_text: The original interview question.
            evaluation_rubric: The rubric text from which concepts are extracted.
            expected_concepts: Optional explicit concept list. If not provided,
                concepts are auto-extracted from the rubric.

        Returns:
            Structured evidence dict including per-concept NLI results,
            similarity scores, and overall coverage signal.
        """
        self._ensure_initialized()
        t0 = time.time()

        # Step 1: Determine concepts
        if expected_concepts:
            concepts = expected_concepts
        else:
            concepts = self.concept_analyzer.extract_concepts_from_rubric(
                evaluation_rubric
            )

        if not concepts:
            # Fall back to extracting from question text itself
            concepts = self.concept_analyzer.extract_concepts_from_rubric(
                question_text
            )

        # Step 2: Run concept coverage analysis
        coverage_result = self.concept_analyzer.evaluate_concepts(
            candidate_answer, concepts
        )

        # Step 3: Package as ML evidence
        evidence = {
            "ml_concept_coverage": coverage_result,
            "concepts_evaluated": len(concepts),
            "concepts_list": concepts,
            "pipeline_version": "hybrid_v1.0",
            "total_pipeline_time_ms": round((time.time() - t0) * 1000, 2),
            "disclaimer": (
                "This is an ML-derived concept coverage signal, not a "
                "scientifically validated interview score. It should be "
                "interpreted as structured evidence for the evaluator."
            ),
        }

        return evidence

    def format_evidence_for_gemini(
        self, ml_evidence: Dict[str, Any]
    ) -> str:
        """
        Format ML evidence into a structured text block suitable for
        injection into a Gemini evaluation prompt.

        The formatted output instructs Gemini to:
        1. Explain and reason over the provided evidence
        2. NOT invent its own concept coverage findings
        3. Account for model limitations
        """
        coverage = ml_evidence.get("ml_concept_coverage", {})
        concept_results = coverage.get("concept_results", [])
        overall_pct = coverage.get("overall_coverage_pct", "N/A")
        method = coverage.get("method", "unknown")
        formula = coverage.get("aggregation_formula", "N/A")

        if not concept_results:
            return ""

        lines = [
            "",
            "═══════════════════════════════════════════════════════════════",
            "ML-DERIVED CONCEPT COVERAGE SIGNAL (Structured Evidence)",
            "═══════════════════════════════════════════════════════════════",
            f"Method: {method}",
            f"Aggregation: {formula}",
            f"Overall Coverage: {overall_pct}%",
            "",
            "Per-Concept Evidence:",
            "─────────────────────",
        ]

        for r in concept_results:
            status_emoji = {
                "covered": "✅",
                "partially_covered": "⚠️",
                "missing": "❌",
                "contradicted": "🚫",
            }.get(r["status"], "❓")

            lines.append(
                f"  {status_emoji} [{r['status'].upper()}] \"{r['concept']}\""
            )
            lines.append(
                f"     Entailment: {r['entailment_probability']:.3f}  |  "
                f"Contradiction: {r['contradiction_probability']:.3f}  |  "
                f"Neutral: {r['neutral_probability']:.3f}"
            )
            lines.append(
                f"     Semantic Similarity: {r['semantic_similarity']:.3f}  |  "
                f"Composite Score: {r['concept_score']:.3f}"
            )
            lines.append("")

        covered = coverage.get("covered_concepts", [])
        missing = coverage.get("missing_concepts", [])
        contradicted = coverage.get("contradicted_concepts", [])
        partial = coverage.get("partially_covered_concepts", [])

        lines.append("Summary:")
        lines.append(f"  Covered: {len(covered)}  |  Partial: {len(partial)}  |  "
                      f"Missing: {len(missing)}  |  Contradicted: {len(contradicted)}")
        lines.append("")
        lines.append(
            "IMPORTANT INSTRUCTIONS FOR EVALUATOR:"
        )
        lines.append(
            "  1. You MUST explain and reason over the above ML evidence."
        )
        lines.append(
            "  2. Do NOT invent your own concept coverage findings — use the evidence above."
        )
        lines.append(
            "  3. You MAY override individual concept assessments if you have strong reason,"
        )
        lines.append(
            "     but you must explain why the ML signal was incorrect."
        )
        lines.append(
            "  4. This is an ML-derived signal, not a scientifically validated score."
        )
        lines.append(
            "═══════════════════════════════════════════════════════════════"
        )

        return "\n".join(lines)

    def augment_evaluation_prompt(
        self,
        base_prompt: str,
        ml_evidence: Dict[str, Any],
    ) -> str:
        """
        Augment an existing Gemini evaluation prompt with ML evidence.

        Returns the enriched prompt string.
        """
        evidence_block = self.format_evidence_for_gemini(ml_evidence)
        if not evidence_block:
            return base_prompt

        return f"{base_prompt}\n{evidence_block}"
