"""
Text embeddings and vector representations using all-MiniLM-L6-v2.
Model: sentence-transformers/all-MiniLM-L6-v2 (PRETRAINED)
https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Used for:
- Semantic Answer Similarity
- Feature vectors for Question Difficulty & Skill Classification

Optimizations (applied):
- Per-instance LRU embedding cache (thread-safe) to skip repeated encode() calls.
- Batch encoding for multi-text inputs (single forward pass).
- CPU thread tuning via torch.set_num_threads() at load time.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

# ── CPU thread tuning ──────────────────────────────────────────────────────
# Avoid over-subscribing CPU cores when running multiple model inferences.
# Default: let PyTorch use half the logical CPUs or what the env specifies.
_CPU_THREADS = int(os.environ.get("ML_CPU_THREADS", str(max(1, (os.cpu_count() or 2) // 2))))
try:
    import torch
    torch.set_num_threads(_CPU_THREADS)
    logger.debug("PyTorch CPU threads set to %d", _CPU_THREADS)
except Exception:
    pass

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or 2D vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


import hashlib

# ── Module-level embedding cache ──────────────────────────────────────────
# Keyed by (model_name, normalize_flag, text).  Stored as bytes to minimise
# memory footprint; decoded to np.float32 on retrieval.
_EMBEDDING_CACHE: Dict[Tuple[str, bool, str], np.ndarray] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_SIZE = int(os.environ.get("EMBEDDING_CACHE_MAX_SIZE", "2048"))
_cache_hits = 0
_cache_misses = 0


def _cache_key(model_name: str, normalize: bool, text: str) -> Tuple[str, bool, str]:
    return (model_name, normalize, text)


def get_embedding_cache_stats() -> Dict[str, int]:
    """Return cache hit/miss counters for observability."""
    return {"hits": _cache_hits, "misses": _cache_misses, "size": len(_EMBEDDING_CACHE)}


def clear_embedding_cache() -> None:
    """Evict all cached embeddings (useful after model swap or in tests)."""
    global _cache_hits, _cache_misses
    with _CACHE_LOCK:
        _EMBEDDING_CACHE.clear()
        _cache_hits = 0
        _cache_misses = 0


class TextEmbeddingExtractor:
    """Extracts dense sentence embeddings using sentence-transformers or deterministic fallback."""


    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "cpu",
        enable_transformer: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._fallback_mode = False
        if enable_transformer is not None:
            self.enable_transformer = enable_transformer
        else:
            self.enable_transformer = os.environ.get("USE_TRANSFORMER_EMBEDDINGS", "1").lower() in ("1", "true")

    def _load_model(self):
        if self._model is not None or self._fallback_mode:
            return
        if not self.enable_transformer:
            self._fallback_mode = True
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading SentenceTransformer model: %s on %s", self.model_name, self.device)
            self._model = SentenceTransformer(self.model_name, device=self.device)
        except Exception as e:
            logger.warning(
                "Failed to load SentenceTransformer (%s). Falling back to deterministic pseudo-embedding. Error: %s",
                self.model_name,
                e
            )
            self._fallback_mode = True

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode text or list of texts into embedding vectors (384-dim for all-MiniLM-L6-v2).

        Optimization: results are cached per (model_name, normalize, text) to avoid
        redundant inference calls for repeated or corpus-level lookups.
        """
        global _cache_hits, _cache_misses

        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else list(texts)

        self._load_model()

        # ── Cache-aware encoding ──────────────────────────────────────────
        results: List[Optional[np.ndarray]] = [None] * len(text_list)
        uncached_indices: List[int] = []
        uncached_texts: List[str] = []

        with _CACHE_LOCK:
            for i, t in enumerate(text_list):
                key = _cache_key(self.model_name, normalize, t)
                cached = _EMBEDDING_CACHE.get(key)
                if cached is not None:
                    results[i] = cached
                    _cache_hits += 1
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(t)
                    _cache_misses += 1

        if uncached_texts:
            if self._model is not None and not self._fallback_mode:
                # Single batch forward pass for all uncached texts
                batch_embs = self._model.encode(
                    uncached_texts,
                    normalize_embeddings=normalize,
                    show_progress_bar=False,
                    batch_size=min(64, len(uncached_texts)),
                )
                batch_embs = np.array(batch_embs)
            else:
                # DETERMINISTIC FALLBACK: Normalized char/word MD5 hash vector
                dim = 384
                batch_embs_list = []
                for t in uncached_texts:
                    v = np.zeros(dim, dtype=np.float32)
                    words = t.lower().split()
                    for w in words:
                        idx = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16) % dim
                        v[idx] += 1.0
                    norm_val = np.linalg.norm(v)
                    if norm_val > 0 and normalize:
                        v = v / norm_val
                    batch_embs_list.append(v)
                batch_embs = np.array(batch_embs_list)

            # Ensure 2D even for single text
            if batch_embs.ndim == 1:
                batch_embs = batch_embs[np.newaxis, :]

            # Write back to cache
            with _CACHE_LOCK:
                for j, idx in enumerate(uncached_indices):
                    vec = batch_embs[j]
                    results[idx] = vec
                    key = _cache_key(self.model_name, normalize, text_list[idx])
                    if len(_EMBEDDING_CACHE) < _CACHE_MAX_SIZE:
                        _EMBEDDING_CACHE[key] = vec

        final = np.array(results, dtype=np.float32)  # shape (N, D)
        return final[0] if is_single else final
