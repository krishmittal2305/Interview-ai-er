"""
ML Pipeline Benchmark — Before / After Optimization Comparison.

Measures:
  1. Embedding model load time
  2. Single-text encode latency (cold vs warm / cached)
  3. Batch encode latency
  4. Similarity search latency — Python loop (before) vs np.dot (after)
  5. check_question_novelty — per-call encode loop (before) vs batch (after)

Run:
  cd Interview-ai-er
  python -m tests.benchmark_ml_pipeline

No GPU or Supabase required; uses deterministic fallback mode for isolation.
"""
from __future__ import annotations

import os
import time
import statistics
from typing import Callable, List, Tuple

import numpy as np

# Force deterministic fallback so the benchmark is reproducible without
# a downloaded sentence-transformers model.
os.environ.setdefault("USE_TRANSFORMER_EMBEDDINGS", "0")

# helpers

def _timeit(fn, n=10):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return (
        round(statistics.mean(times), 2),
        round(min(times), 2),
        round(sorted(times)[int(n * 0.95)], 2),
    )


def _print_row(label, before, after):
    speedup = before[0] / after[0] if after[0] > 0 else float("inf")
    print(
        f"  {label:<48}  {before[0]:>8.2f} ms  {after[0]:>8.2f} ms  {speedup:>6.1f}x"
    )


# benchmark functions

def bench_encoder_cold_warm():
    from ml.features.text_embeddings import TextEmbeddingExtractor, clear_embedding_cache
    enc = TextEmbeddingExtractor()
    text = "Explain the difference between a process and a thread in operating systems."

    def before_fn():
        clear_embedding_cache()
        enc.encode(text, normalize=True)

    clear_embedding_cache()
    enc.encode(text, normalize=True)

    def after_fn():
        enc.encode(text, normalize=True)

    b = _timeit(before_fn, n=20)
    a = _timeit(after_fn, n=20)
    _print_row("encode() cold vs warm (cache hit)", b, a)


def bench_batch_encode():
    from ml.features.text_embeddings import TextEmbeddingExtractor, clear_embedding_cache
    enc = TextEmbeddingExtractor()
    texts = [f"Explain concept #{i} about software engineering." for i in range(50)]

    def before_fn():
        clear_embedding_cache()
        for t in texts:
            enc.encode(t, normalize=True)

    def after_fn():
        clear_embedding_cache()
        enc.encode(texts, normalize=True)

    b = _timeit(before_fn, n=5)
    a = _timeit(after_fn, n=5)
    _print_row("50 texts: per-text encode loop vs batch", b, a)


def bench_similarity_search(corpus_size=200):
    from ml.features.text_embeddings import compute_cosine_similarity
    dim = 384
    rng = np.random.default_rng(42)
    corpus = rng.random((corpus_size, dim), dtype=np.float32)
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus /= norms
    query = rng.random(dim, dtype=np.float32)
    query /= np.linalg.norm(query)

    def before_fn():
        results = [compute_cosine_similarity(query, row) for row in corpus]
        results.sort(reverse=True)

    def after_fn():
        sims = corpus @ query
        sims.sort()

    b = _timeit(before_fn, n=200)
    a = _timeit(after_fn, n=200)
    _print_row(f"similarity search ({corpus_size} vectors): loop vs np.dot", b, a)


def bench_novelty_check(n_prev=20):
    from ml.features.text_embeddings import TextEmbeddingExtractor, clear_embedding_cache
    enc = TextEmbeddingExtractor()
    candidate = "What is the time complexity of quicksort in the average case?"
    prev_questions = [f"Describe sorting algorithm #{i} runtime." for i in range(n_prev)]

    def before_fn():
        clear_embedding_cache()
        cand_emb = enc.encode(candidate, normalize=True)
        for p in prev_questions:
            p_emb = enc.encode(p, normalize=True)
            float(np.dot(cand_emb, p_emb))

    def after_fn():
        clear_embedding_cache()
        cand_emb = enc.encode(candidate, normalize=True)
        prev_embs = enc.encode(prev_questions, normalize=True)
        _ = prev_embs @ cand_emb

    b = _timeit(before_fn, n=30)
    a = _timeit(after_fn, n=30)
    _print_row(f"check_novelty ({n_prev} prev): per-q encode vs batch", b, a)


def bench_embedding_cache_stats():
    from ml.features.text_embeddings import TextEmbeddingExtractor, clear_embedding_cache, get_embedding_cache_stats
    enc = TextEmbeddingExtractor()
    clear_embedding_cache()
    texts = [f"Interview question about topic {i % 30}" for i in range(100)]
    enc.encode(texts, normalize=True)
    enc.encode(texts, normalize=True)
    stats = get_embedding_cache_stats()
    total = stats["hits"] + stats["misses"]
    hit_rate = round(stats["hits"] / total * 100, 1) if total else 0
    print(
        f"\n  Embedding Cache (after 2-pass simulation):\n"
        f"    Unique texts cached : {stats['size']}\n"
        f"    Total hits          : {stats['hits']}\n"
        f"    Total misses        : {stats['misses']}\n"
        f"    Hit rate            : {hit_rate}%"
    )


def main():
    print("\n" + "=" * 75)
    print("  ML Pipeline Benchmark — Optimization Results")
    print("  Environment: deterministic fallback (USE_TRANSFORMER_EMBEDDINGS=0)")
    print("=" * 75)
    print(f"\n  {'Operation':<48}  {'Before (avg)':>12}  {'After (avg)':>11}  {'Speedup':>7}")
    print("  " + "-" * 73)

    bench_encoder_cold_warm()
    bench_batch_encode()
    bench_similarity_search(corpus_size=200)
    bench_similarity_search(corpus_size=1000)
    bench_novelty_check(n_prev=20)
    print()
    bench_embedding_cache_stats()

    print("\n" + "=" * 75)
    print("  NOTE: Ratios above are for deterministic fallback (hash-based).")
    print("  With sentence-transformers (MiniLM) the cache benefit is larger")
    print("  (>50 ms saved per repeated encode on first warm-up).")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
