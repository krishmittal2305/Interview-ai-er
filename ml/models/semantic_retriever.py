"""
Semantic Question Retrieval & Deduplication Engine.
Model: sentence-transformers/all-MiniLM-L6-v2
Vector Store: PostgreSQL / Supabase question_embeddings table + in-memory vector cache.

Capabilities:
  1. Corpus indexing with dense 384-dimensional MiniLM embeddings
  2. Cosine similarity search over question bank
  3. Exact duplicate detection (threshold >= 0.98)
  4. Paraphrase / Near-duplicate detection (0.78 <= threshold < 0.98)
  5. Related-question retrieval (threshold >= 0.50)
  6. Unrelated question discrimination (threshold < 0.40)
  7. Skill-aware retrieval (conditioned on target skill pillars)
  8. Interview engine novelty & deduplication protection
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pydantic import BaseModel, Field

from ml.features.text_embeddings import TextEmbeddingExtractor, compute_cosine_similarity
from ml.models.adaptive_selector import CANONICAL_QUESTION_CATALOG, CatalogQuestion
from ml.models.adaptive_practice import PRACTICE_ADDITIONAL_CATALOG

logger = logging.getLogger(__name__)

# Constants for Model & Versioning
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VERSION = "all-minilm-l6-v2-v1.0"
EMBEDDING_DIM = 384

# Calibrated Semantic Similarity Thresholds
EXACT_DUPLICATE_THRESHOLD = 0.98
NEAR_DUPLICATE_THRESHOLD = 0.78
RELATED_QUESTION_THRESHOLD = 0.50
UNRELATED_THRESHOLD = 0.40


class ScoredQuestionMatch(BaseModel):
    """Result of a semantic question similarity lookup."""
    question_id: str
    title: str = ""
    question_text: str
    skill_focus: str = ""
    difficulty: str = ""
    similarity: float
    classification: str  # "exact_duplicate" | "near_duplicate" | "related" | "unrelated"
    canonical_skills: List[str] = Field(default_factory=list)


class DuplicateDetectionResult(BaseModel):
    """Duplicate / Near-duplicate assessment for a candidate question."""
    is_duplicate: bool
    is_near_duplicate: bool
    classification: str  # "exact_duplicate" | "near_duplicate" | "related" | "unrelated"
    max_similarity: float
    matched_question_id: Optional[str] = None
    matched_question_text: Optional[str] = None
    explanation: str


class SemanticQuestionRetriever:
    """
    Semantic question retrieval using MiniLM embeddings.
    Integrates with PostgreSQL/Supabase question_embeddings storage.
    """

    def __init__(
        self,
        embedding_extractor: Optional[TextEmbeddingExtractor] = None,
        supabase_service: Optional[Any] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_version: str = EMBEDDING_VERSION,
        auto_index_catalog: bool = True,
    ):
        self.embedding_model = embedding_model
        self.embedding_version = embedding_version
        self._embedding_extractor = embedding_extractor
        self._supabase = supabase_service

        # In-memory vector store: question_id -> { "vector": np.ndarray, "metadata": dict }
        self._vector_store: Dict[str, Dict[str, Any]] = {}
        # Prebuilt corpus matrix for vectorized similarity search — rebuilt on corpus changes.
        self._corpus_matrix: Optional[np.ndarray] = None  # shape (N, D)
        self._corpus_ids: List[str] = []  # ordered list matching matrix rows
        self._corpus_dirty: bool = True  # True = matrix needs rebuild

        if auto_index_catalog:
            self.index_canonical_corpus(persist=True)

    # ── Lazy property loaders ──
    @property
    def embedding_extractor(self) -> TextEmbeddingExtractor:
        if self._embedding_extractor is None:
            self._embedding_extractor = TextEmbeddingExtractor(model_name=self.embedding_model)
        return self._embedding_extractor

    @property
    def supabase(self) -> Any:
        if self._supabase is None:
            from app.services.supabase_service import SupabaseService
            self._supabase = SupabaseService()
        return self._supabase

    # ─────────────────────────────────────────────────────────────────────────
    # Vector Corpus Indexing
    # ─────────────────────────────────────────────────────────────────────────
    def index_canonical_corpus(
        self,
        custom_catalog: Optional[List[Any]] = None,
        persist: bool = True
    ) -> int:
        """
        Generate and store embeddings for all questions in the question catalog.
        Uses PostgreSQL/Supabase table 'question_embeddings' with local fallback.
        """
        questions_to_index: List[Dict[str, Any]] = []

        if custom_catalog is not None:
            for item in custom_catalog:
                if isinstance(item, CatalogQuestion):
                    questions_to_index.append({
                        "question_id": item.id,
                        "title": item.title,
                        "question_text": item.question_text,
                        "skill_focus": item.skill_focus,
                        "difficulty": item.difficulty,
                        "canonical_skills": item.canonical_skills,
                    })
                elif isinstance(item, dict):
                    questions_to_index.append(dict(item))
        else:
            # Combine canonical selector catalog and practice additional catalog
            seen_ids = set()
            for q in CANONICAL_QUESTION_CATALOG:
                if q.id not in seen_ids:
                    seen_ids.add(q.id)
                    questions_to_index.append({
                        "question_id": q.id,
                        "title": q.title,
                        "question_text": q.question_text,
                        "skill_focus": q.skill_focus,
                        "difficulty": q.difficulty,
                        "canonical_skills": q.canonical_skills,
                    })
            for q in PRACTICE_ADDITIONAL_CATALOG:
                if q.id not in seen_ids:
                    seen_ids.add(q.id)
                    questions_to_index.append({
                        "question_id": q.id,
                        "title": q.title,
                        "question_text": q.question_text,
                        "skill_focus": q.skill_focus,
                        "difficulty": q.difficulty,
                        "canonical_skills": q.canonical_skills,
                    })

        indexed_count = 0
        texts_to_embed = []
        meta_to_embed = []

        for q in questions_to_index:
            qid = q.get("question_id") or q.get("id")
            qtext = q.get("question_text", "")
            if not qid or not qtext:
                continue

            # Check if already in vector store
            if qid in self._vector_store:
                continue

            texts_to_embed.append(qtext)
            meta_to_embed.append(q)

        if texts_to_embed:
            # Batch encode for high efficiency
            embeddings = self.embedding_extractor.encode(texts_to_embed, normalize=True)
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            elif embeddings.ndim == 1 and len(texts_to_embed) == 1:
                embeddings = np.expand_dims(embeddings, axis=0)

            for i, q in enumerate(meta_to_embed):
                qid = q.get("question_id") or q.get("id")
                vec = embeddings[i]
                vec_list = [float(x) for x in vec]

                # Store in-memory
                self._vector_store[qid] = {
                    "question_id": qid,
                    "vector": vec,
                    "title": q.get("title", ""),
                    "question_text": q.get("question_text", ""),
                    "skill_focus": q.get("skill_focus", ""),
                    "difficulty": q.get("difficulty", ""),
                    "canonical_skills": q.get("canonical_skills", []),
                    "embedding_model": self.embedding_model,
                    "embedding_version": self.embedding_version,
                }

                # Persist to PostgreSQL / Supabase
                if persist:
                    try:
                        self.supabase.save_question_embedding(
                            question_id=qid,
                            embedding_vector=vec_list,
                            embedding_model=self.embedding_model,
                            embedding_version=self.embedding_version,
                            question_text=q.get("question_text", ""),
                            skill_focus=q.get("skill_focus", ""),
                            difficulty=q.get("difficulty", ""),
                        )
                    except Exception as e:
                        logger.debug("Failed remote question embedding save: %s", e)

                indexed_count += 1

        # Invalidate prebuilt corpus matrix whenever new items are added
        if indexed_count > 0:
            self._corpus_dirty = True

        logger.info("Indexed %d questions into semantic vector store.", indexed_count)
        return len(self._vector_store)

    # ─────────────────────────────────────────────────────────────────────────
    # Vector Embedding Helper
    # ─────────────────────────────────────────────────────────────────────────
    def get_embedding(self, text_or_vector: Union[str, np.ndarray, List[float]]) -> np.ndarray:
        """Resolve a string or array into a normalized 1D numpy vector."""
        if isinstance(text_or_vector, str):
            emb = self.embedding_extractor.encode(text_or_vector, normalize=True)
            return np.array(emb, dtype=np.float32)
        elif isinstance(text_or_vector, list):
            v = np.array(text_or_vector, dtype=np.float32)
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v
        elif isinstance(text_or_vector, np.ndarray):
            v = text_or_vector.astype(np.float32)
            if v.ndim > 1:
                v = v.flatten()
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v
        raise ValueError("Unsupported embedding input type.")

    # ─────────────────────────────────────────────────────────────────────────
    # Corpus Matrix (Vectorized Search Support)
    # ─────────────────────────────────────────────────────────────────────────
    def _get_corpus_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return a prebuilt (N, D) float32 matrix of all stored vectors plus the
        corresponding ordered list of question IDs.  The matrix is rebuilt only
        when _corpus_dirty is True, avoiding repeated allocations.
        """
        if self._corpus_dirty or self._corpus_matrix is None:
            ids = list(self._vector_store.keys())
            if not ids:
                return np.empty((0, EMBEDDING_DIM), dtype=np.float32), []
            matrix = np.stack(
                [self._vector_store[qid]["vector"] for qid in ids], axis=0
            ).astype(np.float32)  # (N, D)
            self._corpus_matrix = matrix
            self._corpus_ids = ids
            self._corpus_dirty = False
        return self._corpus_matrix, self._corpus_ids

    @staticmethod
    def _skill_matches(record: Dict[str, Any], filter_skills: set) -> bool:
        """Return True if the record's skill fields overlap with filter_skills."""
        q_skill = (record.get("skill_focus") or "").lower()
        q_canonical = [c.lower() for c in record.get("canonical_skills", [])]
        return (
            any(fs in q_skill or q_skill in fs for fs in filter_skills)
            or any(any(fs in c or c in fs for fs in filter_skills) for c in q_canonical)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Similarity Search
    # ─────────────────────────────────────────────────────────────────────────
    def similarity_search(
        self,
        query: Union[str, np.ndarray, List[float]],
        top_k: int = 5,
        min_similarity: float = 0.0,
        skill_focus: Optional[str] = None,
        target_skills: Optional[List[str]] = None,
    ) -> List[ScoredQuestionMatch]:
        """
        Vectorized cosine similarity search over indexed question embeddings.

        Optimization: builds a single (N, D) corpus matrix once and uses a
        batched np.dot() rather than looping with compute_cosine_similarity().
        Skill filtering is applied as a boolean mask before the dot product,
        reducing the effective search space for skill-aware queries.
        """
        t0 = time.perf_counter()
        query_vec = self.get_embedding(query)  # (D,) normalized

        # ── Skill filter set ─────────────────────────────────────────────────
        filter_skills: set = set()
        if skill_focus:
            filter_skills.add(skill_focus.lower())
        if target_skills:
            for s in target_skills:
                filter_skills.add(s.lower())

        # ── Build / retrieve corpus matrix ───────────────────────────────────
        corpus_matrix, corpus_ids = self._get_corpus_matrix()

        if corpus_matrix.shape[0] == 0:
            return []

        # Resolve row indices that pass the skill filter
        if filter_skills:
            row_indices = [
                i for i, qid in enumerate(corpus_ids)
                if self._skill_matches(self._vector_store[qid], filter_skills)
            ]
            if not row_indices:
                return []
            search_matrix = corpus_matrix[row_indices]  # (M, D)
            search_ids = [corpus_ids[i] for i in row_indices]
        else:
            search_matrix = corpus_matrix  # (N, D)
            search_ids = corpus_ids

        # ── Vectorized cosine similarity (vectors are pre-normalized) ────────
        # query_vec is normalized; row vecs are normalized at index time.
        # sim[i] = dot(query_vec, search_matrix[i]) = cos(angle_i)
        similarities: np.ndarray = search_matrix @ query_vec  # (M,)

        # Build match list, apply threshold, classify
        matches: List[ScoredQuestionMatch] = []
        for i, qid in enumerate(search_ids):
            sim = float(similarities[i])
            if sim < min_similarity:
                continue

            if sim >= EXACT_DUPLICATE_THRESHOLD:
                classification = "exact_duplicate"
            elif sim >= NEAR_DUPLICATE_THRESHOLD:
                classification = "near_duplicate"
            elif sim >= UNRELATED_THRESHOLD:
                classification = "related"
            else:
                classification = "unrelated"

            record = self._vector_store[qid]
            matches.append(
                ScoredQuestionMatch(
                    question_id=qid,
                    title=record.get("title", ""),
                    question_text=record.get("question_text", ""),
                    skill_focus=record.get("skill_focus", ""),
                    difficulty=record.get("difficulty", ""),
                    similarity=round(sim, 4),
                    classification=classification,
                    canonical_skills=record.get("canonical_skills", []),
                )
            )

        matches.sort(key=lambda m: m.similarity, reverse=True)
        top_matches = matches[:top_k]

        # Record telemetry
        try:
            from app.observability.tracker import get_telemetry_tracker
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            is_fallback = not getattr(self.embedding_extractor, "enable_transformer", True)
            get_telemetry_tracker().record_inference(
                inference_type="ml",
                model=self.embedding_model,
                model_version=self.embedding_version,
                task="semantic_similarity_search",
                latency_ms=latency_ms,
                success=True,
                status="FALLBACK" if is_fallback else "SUCCESS",
                input_size=len(query) if isinstance(query, str) else len(str(query)),
                output_size=len(top_matches),
                memory_device="cpu",
                fallback_usage=is_fallback,
                retry_count=0,
                metadata={"top_k": top_k, "corpus_size": len(self._vector_store)},
            )
        except Exception as tel_err:
            logger.debug(f"Semantic search telemetry failed: {tel_err}")

        return top_matches

    # ─────────────────────────────────────────────────────────────────────────
    # Duplicate & Near-Duplicate Detection
    # ─────────────────────────────────────────────────────────────────────────
    def detect_duplicates(
        self,
        query: Union[str, np.ndarray, List[float]],
        exclude_id: Optional[str] = None
    ) -> DuplicateDetectionResult:
        """
        Determine if the query matches an existing question in the corpus as:
          - exact duplicate (similarity >= 0.98)
          - near-duplicate / paraphrase (0.78 <= similarity < 0.98)
          - related question (0.40 <= similarity < 0.78)
          - unrelated question (similarity < 0.40)
        """
        results = self.similarity_search(query=query, top_k=5, min_similarity=0.0)

        # Filter out self if query matches an existing ID
        filtered = [m for m in results if m.question_id != exclude_id] if exclude_id else results

        if not filtered:
            return DuplicateDetectionResult(
                is_duplicate=False,
                is_near_duplicate=False,
                classification="unrelated",
                max_similarity=0.0,
                explanation="No existing questions matched the query."
            )

        top_match = filtered[0]
        max_sim = top_match.similarity

        # Exact Duplicate check (also check normalized text match)
        is_exact = max_sim >= EXACT_DUPLICATE_THRESHOLD
        if isinstance(query, str):
            norm_q = "".join(filter(str.isalnum, query.lower()))
            norm_m = "".join(filter(str.isalnum, top_match.question_text.lower()))
            if norm_q and norm_q == norm_m:
                is_exact = True
                max_sim = 1.0

        if is_exact:
            return DuplicateDetectionResult(
                is_duplicate=True,
                is_near_duplicate=True,
                classification="exact_duplicate",
                max_similarity=max_sim,
                matched_question_id=top_match.question_id,
                matched_question_text=top_match.question_text,
                explanation=(
                    f"Exact duplicate detected (similarity: {max_sim:.4f} >= {EXACT_DUPLICATE_THRESHOLD}). "
                    f"Matches existing question '{top_match.question_id}'."
                )
            )

        # Near Duplicate / Paraphrase check
        if max_sim >= NEAR_DUPLICATE_THRESHOLD:
            return DuplicateDetectionResult(
                is_duplicate=False,
                is_near_duplicate=True,
                classification="near_duplicate",
                max_similarity=max_sim,
                matched_question_id=top_match.question_id,
                matched_question_text=top_match.question_text,
                explanation=(
                    f"Near-duplicate paraphrase detected (similarity: {max_sim:.4f} in "
                    f"[{NEAR_DUPLICATE_THRESHOLD}, {EXACT_DUPLICATE_THRESHOLD})). "
                    f"Highly overlapping concept with '{top_match.question_id}'."
                )
            )

        # Related or Unrelated check
        if max_sim >= UNRELATED_THRESHOLD:
            return DuplicateDetectionResult(
                is_duplicate=False,
                is_near_duplicate=False,
                classification="related",
                max_similarity=max_sim,
                matched_question_id=top_match.question_id,
                matched_question_text=top_match.question_text,
                explanation=(
                    f"Related question (similarity: {max_sim:.4f} in "
                    f"[{UNRELATED_THRESHOLD}, {NEAR_DUPLICATE_THRESHOLD})). Distinct prompt variations."
                )
            )

        return DuplicateDetectionResult(
            is_duplicate=False,
            is_near_duplicate=False,
            classification="unrelated",
            max_similarity=max_sim,
            matched_question_id=top_match.question_id,
            matched_question_text=top_match.question_text,
            explanation=(
                f"Unrelated question (similarity: {max_sim:.4f} < {UNRELATED_THRESHOLD}). "
                f"Concepts are distinct."
            )
        )

    def is_exact_duplicate(
        self,
        query: str,
        target_question_or_id: str
    ) -> Tuple[bool, float]:
        """Check if two specific questions are exact duplicates."""
        # Check by ID lookup
        if target_question_or_id in self._vector_store:
            target_vec = self._vector_store[target_question_or_id]["vector"]
        else:
            target_vec = self.get_embedding(target_question_or_id)

        query_vec = self.get_embedding(query)
        sim = compute_cosine_similarity(query_vec, target_vec)

        # Check identical string
        norm_q = "".join(filter(str.isalnum, query.lower()))
        norm_t = "".join(filter(str.isalnum, target_question_or_id.lower()))
        if norm_q and norm_q == norm_t:
            return True, 1.0

        return (sim >= EXACT_DUPLICATE_THRESHOLD), round(float(sim), 4)

    def is_near_duplicate(
        self,
        query: str,
        target_question_or_id: str
    ) -> Tuple[bool, float]:
        """Check if two specific questions are paraphrased near-duplicates."""
        if target_question_or_id in self._vector_store:
            target_vec = self._vector_store[target_question_or_id]["vector"]
        else:
            target_vec = self.get_embedding(target_question_or_id)

        query_vec = self.get_embedding(query)
        sim = compute_cosine_similarity(query_vec, target_vec)
        is_near = NEAR_DUPLICATE_THRESHOLD <= sim < EXACT_DUPLICATE_THRESHOLD
        return is_near, round(float(sim), 4)

    # ─────────────────────────────────────────────────────────────────────────
    # Related-Question Retrieval
    # ─────────────────────────────────────────────────────────────────────────
    def retrieve_related(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = RELATED_QUESTION_THRESHOLD,
    ) -> List[ScoredQuestionMatch]:
        """Retrieve top semantically related questions above threshold."""
        return self.similarity_search(query=query, top_k=top_k, min_similarity=threshold)

    # ─────────────────────────────────────────────────────────────────────────
    # Skill-Aware Retrieval
    # ─────────────────────────────────────────────────────────────────────────
    def retrieve_skill_aware(
        self,
        query: str,
        target_skills: Union[str, List[str]],
        top_k: int = 5,
        min_similarity: float = 0.30,
    ) -> List[ScoredQuestionMatch]:
        """
        Retrieve semantically similar questions specifically covering target skills.
        Ensures curriculum-aligned retrieval.
        """
        skills_list = [target_skills] if isinstance(target_skills, str) else list(target_skills)
        return self.similarity_search(
            query=query,
            top_k=top_k,
            min_similarity=min_similarity,
            target_skills=skills_list,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Interview Engine Novelty Protection
    # ─────────────────────────────────────────────────────────────────────────
    def check_question_novelty(
        self,
        candidate_question_text: str,
        previous_question_texts: List[str],
    ) -> Tuple[float, Optional[str], str]:
        """
        Evaluate candidate question against previously asked interview questions.

        Optimization: batch-encode all previous questions in a single forward pass
        rather than calling get_embedding() per question.  For N previous questions
        this reduces N separate encode() calls to 1 batch call (plus cache hits).

        Returns:
            (novelty_score: float in [0.0, 1.0], most_similar_text, verdict)
        """
        if not previous_question_texts:
            return 1.0, None, "First question in session (full novelty)."

        cand_emb = self.get_embedding(candidate_question_text)  # (D,)

        # Batch-encode previous questions (cache will short-circuit repeats)
        prev_texts_clean = [p for p in previous_question_texts if p]
        if not prev_texts_clean:
            return 1.0, None, "No valid previous questions."

        prev_embs = self.embedding_extractor.encode(prev_texts_clean, normalize=True)
        if prev_embs.ndim == 1:
            prev_embs = prev_embs[np.newaxis, :]  # handle single-item

        # Vectorized similarities: (M,)
        similarities_vec: np.ndarray = prev_embs @ cand_emb

        max_sim = 0.0
        most_sim_text: Optional[str] = None

        for i, prev_text in enumerate(prev_texts_clean):
            sim = float(similarities_vec[i])

            # Exact string match override
            norm_c = "".join(filter(str.isalnum, candidate_question_text.lower()))
            norm_p = "".join(filter(str.isalnum, prev_text.lower()))
            if norm_c == norm_p:
                sim = 1.0

            if sim > max_sim:
                max_sim = sim
                most_sim_text = prev_text

        novelty = round(max(0.0, 1.0 - max_sim), 4)

        if max_sim >= EXACT_DUPLICATE_THRESHOLD:
            verdict = "REJECT_EXACT_DUPLICATE"
        elif max_sim >= NEAR_DUPLICATE_THRESHOLD:
            verdict = "PENALIZE_NEAR_DUPLICATE"
        elif max_sim >= UNRELATED_THRESHOLD:
            verdict = "ALLOW_RELATED"
        else:
            verdict = "ALLOW_DISTINCT"

        return novelty, most_sim_text, verdict
