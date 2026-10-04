# 🧠 AI Interview Intelligence Platform (`Interview-ai-er`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14+](https://img.shields.io/badge/frontend-Next.js%2014%20App%20Router-black.svg)](https://nextjs.org/)
[![Database](https://img.shields.io/badge/database-Supabase%20%7C%20PostgreSQL%20%7C%20pgvector-emerald.svg)](https://supabase.com/)
[![AI Providers](https://img.shields.io/badge/LLM-Groq%20%7C%20Google%20Gemini-orange.svg)](https://groq.com/)
[![ML Stack](https://img.shields.io/badge/ML-PyTorch%20%7C%20Transformers%20%7C%20scikit--learn-red.svg)](https://huggingface.co/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

An enterprise-grade, end-to-end technical and behavioral interview intelligence platform. Rather than delegating candidate assessment to opaque, non-deterministic generative prompts, this platform integrates **calibrated local machine learning models**, **Item Response Theory (2PL-IRT) adaptive testing**, **Natural Language Inference (NLI) rubric entailment**, **isolated sandboxed code execution**, **browser proctoring telemetry**, and **deep longitudinal analytics** with structured LLM synthesis.

---

## 📑 Table of Contents

- [1. Executive System Architecture](#1-executive-system-architecture)
- [2. The 5-Tier Intelligence Classification](#2-the-5-tier-intelligence-classification)
- [3. Machine Learning Engine & Model Registry](#3-machine-learning-engine--model-registry)
  - [3.1 Model Catalog & Benchmark Performance](#31-model-catalog--benchmark-performance)
  - [3.2 Hybrid Assessment Pipeline (ML + LLM)](#32-hybrid-assessment-pipeline-ml--llm)
  - [3.3 3-Tier Fallback Resilience Architecture](#33-3-tier-fallback-resilience-architecture)
  - [3.4 Computerized Adaptive Testing (CAT) & 2PL-IRT](#34-computerized-adaptive-testing-cat--2pl-irt)
- [4. Backend Architecture & Domain Organization](#4-backend-architecture--domain-organization)
  - [4.1 Repository Layout](#41-repository-layout)
  - [4.2 Domain Entity Model](#42-domain-entity-model)
  - [4.3 State Machine Orchestrator (FSM)](#43-state-machine-orchestrator-fsm)
  - [4.4 Longitudinal Analytics & Methodology](#44-longitudinal-analytics--methodology)
  - [4.5 Code Execution Sandbox Infrastructure](#45-code-execution-sandbox-infrastructure)
  - [4.6 Security & Identity Architecture](#46-security--identity-architecture)
- [5. Frontend Architecture & User Experience](#5-frontend-architecture--user-experience)
  - [5.1 Application Routes & Workspaces](#51-application-routes--workspaces)
  - [5.2 Candidate IDE & Real-Time Runner](#52-candidate-ide--real-time-runner)
  - [5.3 Multi-Signal Anti-Cheat & Proctoring Engine](#53-multi-signal-anti-cheat--proctoring-engine)
  - [5.4 Admin & System Observability Dashboard](#54-admin--system-observability-dashboard)
- [6. Authoritative Database Schema & Migrations](#6-authoritative-database-schema--migrations)
  - [6.1 Relational Architecture (001)](#61-relational-architecture-001)
  - [6.2 Vector Storage & pgvector (002)](#62-vector-storage--pgvector-002)
  - [6.3 Telemetry & Inference Logging (003)](#63-telemetry--inference-logging-003)
- [7. Complete REST API Catalog (40+ Endpoints)](#7-complete-rest-api-catalog-40-endpoints)
- [8. Telemetry, Cost Modeling & Observability](#8-telemetry-cost-modeling--observability)
- [9. Installation, Setup & Environment Configuration](#9-installation-setup--environment-configuration)
  - [9.1 Prerequisites](#91-prerequisites)
  - [9.2 Backend Installation](#92-backend-installation)
  - [9.3 Frontend Installation](#93-frontend-installation)
  - [9.4 Environment Variable Matrix](#94-environment-variable-matrix)
  - [9.5 Supabase Database Setup](#95-supabase-database-setup)
- [10. Testing, Benchmarks & Validation](#10-testing-benchmarks--validation)
- [11. Engineering Documentation Directory](#11-engineering-documentation-directory)

---

## 1. Executive System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   CLIENT TIER (Next.js 14)                                       │
│                                                                                                  │
│   ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────┐   │
│   │ Interview Studio │   │  Monaco Web IDE  │   │  Adaptive Drills │   │ Analytics / Reports │   │
│   │ (/interview)     │   │  (/ide)          │   │  (/practice)     │   │ (/analytics)        │   │
│   └─────────┬────────┘   └─────────┬────────┘   └─────────┬────────┘   └──────────┬──────────┘   │
│             │                      │                      │                       │              │
│   ┌─────────▼──────────────────────▼──────────────────────▼───────────────────────▼───────────┐   │
│   │ Client Anti-Cheat Sensor (Blur/Tab, DevTools, Keystroke Cadence, Paste, WebCam/Mic)       │   │
│   │ Supabase Auth Client (JWT Bearer Token Management & Auto-Refresh)                         │   │
│   └─────────────────────────────────────────┬────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────┼────────────────────────────────────────────────────┘
                                              │ Authenticated HTTP / Bearer JWT
                                              ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 SERVER TIER (Flask Application Factory)                          │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │  Auth Middleware & Context Extraction: Server-side claims validation, user isolation     │   │
│   └─────────────────────────────────────────┬────────────────────────────────────────────────┘   │
│                                             │                                                    │
│   ┌────────────────────────┐  ┌─────────────▼──────────────┐  ┌──────────────────────────────┐   │
│   │ Interview Orchestrator │  │   Hybrid Assessment        │  │ Longitudinal Analytics       │   │
│   │ - Session Lifecycle FSM│  │   Pipeline                 │  │ - OLS Trajectory Slope       │   │
│   │ - Dynamic Question Q   │  │   - Rubric Concept Decomp. │  │ - Score Dispersion / IQR     │   │
│   │ - CAT Item Selection   │  │   - ML Entailment Evidence │  │ - Weakness Regex Clustering  │   │
│   │ - Anti-Duplicate Cache │  │   - Structured LLM Prompts │  │ - Recommendation Lift Metric │   │
│   └───────────┬────────────┘  └─────────────┬──────────────┘  └──────────────┬───────────────┘   │
│               │                             │                                │                   │
│   ┌───────────▼────────────┐  ┌─────────────▼──────────────┐  ┌──────────────▼───────────────┐   │
│   │ Sandboxed Code Runner  │  │ Inference Telemetry Tracker│  │ Proctoring Integrity Engine  │   │
│   │ - Subprocess Isolation │  │ - Latency & Memory RSS     │  │ - Multi-signal Fusion Score  │   │
│   │ - Wall-Clock Timeout   │  │ - Token & Cost Estimator   │  │ - Tamper Anomaly Auditing    │   │
│   │ - Multi-language (Py/JS│  │ - Dual Persistence (DB+Mem)│  │ - Severity Classification    │   │
│   └────────────────────────┘  └────────────────────────────┘  └──────────────────────────────┘   │
└─────────────────────────────────────────────┬────────────────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
┌───────────────────────────────────────┐   ┌──────────────────────────────────────────────────┐
│        LOCAL & EMBEDDED ML TIER       │   │               EXTERNAL INFRASTRUCTURE            │
│                                       │   │                                                  │
│ • question-difficulty-v1 (TACO+APPS)  │   │ • Supabase PostgreSQL                            │
│ • question-skill-v1 (21 taxonomy tags)│   │   - 14 Normalized Domain Tables                  │
│ • answer-nli-v1 (DeBERTa-v3 + MiniLM) │   │   - Row Level Security (RLS) Policies            │
│ • code-risk-v1 (CodeBERT Devign)      │   │   - pgvector Dense Similarity Embeddings         │
│ • mastery-v1 (2PL-IRT Ability Engine) │   │ • Generative AI Cloud (Primary & Fallback)       │
│ • faster-whisper (CTranslate2 STT)    │   │   - Groq API (openai/gpt-oss-120b, LLaMA-3.3-70b)│
│ • MiniLM Dense Semantic Retriever     │   │   - Google Gemini API (gemini-2.5-flash / pro)   │
└───────────────────────────────────────┘   └──────────────────────────────────────────────────┘
```

---

## 2. The 5-Tier Intelligence Classification

To prevent hallucinated scoring, ensure predictable performance, and maintain auditability, every single decision and computational task in the platform belongs to one of five explicit tiers:

| Tier | Name | Technology | Responsibilities | Latency Profile |
|---|---|---|---|---|
| **1** | `TRAINED BY US` | Scikit-learn, Ridge, MultiOutput Logistic Regression, 2PL-IRT parameters | Question difficulty prediction, 21-category skill tagging, candidate latent ability estimation ($\theta$), Fisher Information adaptive selection. | **< 25 ms** |
| **2** | `FINE-TUNED BY US` | `microsoft/codebert-base` on CodeXGLUE Devign | Static code vulnerability detection, boundary bug identification, AST defect risk scoring prior to execution. | **100–250 ms** |
| **3** | `PRETRAINED` | `cross-encoder/nli-deberta-v3-base`, `sentence-transformers/all-MiniLM-L6-v2`, `faster-whisper` | Zero-shot rubric entailment ($P(\text{entailment})$), 384-dimensional question deduplication, offline audio speech-to-text. | **15–400 ms** |
| **4** | `DETERMINISTIC` | Python Standard Library, Subprocess Sandbox, PostgreSQL Constraints | Session finite state machine transitions, strict process isolation, timeout enforcement, token hashing, running average updates. | **< 2 ms** |
| **5** | `LLM-GENERATED` | Groq (`openai/gpt-oss-120b`), Google Gemini (`gemini-2.5-flash`) | Contextual question drafting, conversational synthesis, qualitative feedback explanations based on ML-supplied evidence. | **400–1200 ms** |

---

## 3. Machine Learning Engine & Model Registry

All production machine learning models are cataloged in [`ml/model_registry.json`](ml/model_registry.json) with cryptographic hashes, training run IDs, dataset provenance, and runtime latency targets.

### 3.1 Model Catalog & Benchmark Performance

```
ml/
├── artifacts/              # Serialized model weights (.joblib, ONNX, PyTorch checkpoints)
│   ├── difficulty/         # question-difficulty-v1 weights + metadata
│   └── skill_tagger/       # question-skill-v1 OneVsRest weights
├── calibration/            # Probability calibration curves & isotonic regressors
├── configs/                # Hyperparameters, feature schemas, training configs
├── evaluation/             # Test harness, confusion matrices, ROC/PR curves
├── features/               # Dense vector extractors, TF-IDF n-gram tokenizers
├── inference/              # Production inference workers and ONNX wrappers
├── mastery/                # IRT item parameter tables & latent update equations
├── models/                 # Model definitions & inference classes
├── pipelines/              # Hybrid assessment & end-to-end interview pipelines
├── retrieval/              # Vector search, pgvector integration, cosine deduplication
├── serving/                # Dynamic model loader, warming singleton, health checks
├── training/               # Offline training pipelines (TACO, APPS, CodeXGLUE, EdNet)
├── fallbacks.py            # Three-tier fallback framework & Pydantic schemas
└── versioning.py           # Model version registry, deprecation lifecycle manager
```

| Model ID | Version | Category | Base Backbone | Dataset | Primary Metric | CPU Latency |
|---|---|---|---|---|---|---|
| `question-difficulty-v1` | `1.1.0` | `TRAINED BY US` | `all-MiniLM-L6-v2` + TF-IDF Logistic Regression | BAAI/TACO + codeparrot/apps | Test Acc: **65.7%** / Macro F1: **0.656** | 15–25 ms |
| `question-skill-v1` | `1.1.0` | `TRAINED BY US` | `all-MiniLM-L6-v2` + OneVsRest Logistic Head | BAAI/TACO (21 classes) | Micro F1: **0.645** / Hamming Loss: **0.096** | 15–20 ms |
| `answer-nli-v1` | `1.0.0` | `PRETRAINED` | `cross-encoder/nli-deberta-v3-base` | MNLI + SNLI | Entailment ECE: **0.082** | 80–150 ms |
| `code-risk-v1` | `1.0.0` | `FINE-TUNED BY US`| `microsoft/codebert-base` Sequence Classifier | CodeXGLUE Defect (Devign) | Accuracy: **64.8%** / Macro F1: **0.639** | 120–220 ms |
| `mastery-v1` | `1.0.0` | `TRAINED BY US` | 2PL-IRT / Bayesian Knowledge Tracing | mgor/EDNet (KT1) | Latent $\theta \in [-3.0, +3.0]$, Prof $\in [0, 100]$ | < 2 ms |
| `semantic-retriever` | `1.0.0` | `PRETRAINED` | `sentence-transformers/all-MiniLM-L6-v2` | SBERT benchmark | 384-d Cosine Similarity | < 15 ms |
| `speech-to-text` | `1.0.0` | `PRETRAINED` | `openai/whisper-base.en` (`faster-whisper`) | LibriSpeech / CommonVoice | Word Error Rate (WER) < 8.5% | ~400 ms / 5s chunk |

### 3.2 Hybrid Assessment Pipeline (ML + LLM)

Standard AI interview apps suffer from hallucinations and score inflation when an unconstrained LLM generates arbitrary scores. **Interview-ai-er** solves this with a **Hybrid Assessment Pipeline** ([`ml/pipelines/hybrid_assessment.py`](ml/pipelines/hybrid_assessment.py)):

1. **Rubric Concept Extraction**: The question rubric is split into discrete testable technical assertions (e.g., *"Handles cycle detection via two pointers or visited set"*, *"Analyzes worst-case space complexity as O(V+E)"*).
2. **Deterministic NLI Cross-Encoder**: Each concept is fed through `cross-encoder/nli-deberta-v3-base`:
   $$\text{Premise} = \text{Candidate Answer Text}$$
   $$\text{Hypothesis} = \text{"The candidate correctly explains and addresses } \{\text{concept}\}."$$
   The model computes true entailment probability $P(\text{entailment}) \in [0, 1]$.
3. **Semantic Similarity Validation**: `all-MiniLM-L6-v2` extracts dense embeddings to calculate semantic distance from canonical references.
4. **Structured ML Evidence Injection**: The exact per-concept entailment scores, similarity ratios, and defect risk probabilities are formatted into an immutable **ML Evidence Block**.
5. **Constrained LLM Synthesis**: Groq or Gemini receives the ML evidence and is instructed to **explain and synthesize** the findings rather than invent its own scores.

### 3.3 3-Tier Fallback Resilience Architecture

Every ML and AI component implements the resilience contract in [`ml/fallbacks.py`](ml/fallbacks.py):

```
┌────────────────────────────────────────────────────────┐
│  Tier 1: Primary ML (Trained / Fine-Tuned Models)       │
│  e.g., Local DeBERTa cross-encoder, CodeBERT, MiniLM   │
└──────────────────────────┬─────────────────────────────┘
                           │ Exception / OOM / Missing Weights
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 2: Pretrained Transformer or Deterministic Pass  │
│  e.g., Sentence-Transformers Cosine Similarity, AST    │
│  regex keyword taxonomy matching, token heuristics     │
└──────────────────────────┬─────────────────────────────┘
                           │ Exception / Degradation
                           ▼
┌────────────────────────────────────────────────────────┐
│  Tier 3: Structured Generative LLM Fallback            │
│  e.g., Gemini / Groq with strict Pydantic JSON schemas │
│  (DifficultyLLMSchema, ConceptCoverageLLMSchema)       │
└────────────────────────────────────────────────────────┘
```

Every inference result explicitly tags its `source` (`"ml"`, `"pretrained"`, `"deterministic"`, or `"llm"`), ensuring complete transparency.

### 3.4 Computerized Adaptive Testing (CAT) & 2PL-IRT

The platform implements two-parameter logistic item response theory ([`ml/models/skill_mastery.py`](ml/models/skill_mastery.py) and [`ml/models/adaptive_selector.py`](ml/models/adaptive_selector.py)):

$$P(Y_i = 1 \mid \theta) = \frac{1}{1 + e^{-a_i (\theta - b_i)}}$$

Where:
- $\theta$: Candidate latent ability on the skill.
- $b_i$: Question difficulty parameter.
- $a_i$: Question discrimination parameter.

**Fisher Information Maximization**:
The question selector selects the next candidate question $i$ that maximizes Fisher Information at the candidate's current ability $\hat{\theta}$:
$$I_i(\theta) = a_i^2 P_i(\theta) [1 - P_i(\theta)]$$
Combined with **Thompson Sampling** to balance knowledge exploration vs. ability exploitation within the candidate's **Zone of Proximal Development (ZPD)**.

---

## 4. Backend Architecture & Domain Organization

The backend is written in Python 3.10+ using Flask application factories, strict Pydantic validation schemas, clean domain dataclasses, and dedicated repository layers.

### 4.1 Repository Layout

```
app/
├── __init__.py                 # Flask application factory, CORS, env validation, blueprints
├── api/                        # HTTP transport controllers (Blueprints)
│   ├── interview.py            # Session start/end, question delivery, answer/code submission
│   ├── intelligence.py         # Skill profiles, targeted recommendations, activity updates
│   ├── analytics.py            # Trajectory trends, weakness clustering, consistency metrics
│   ├── logging.py              # Telemetry event ingestion, browser anomalies, health check
│   ├── ml_health.py            # Model registry inspection, artifact status, hardware profiling
│   └── observability.py        # Inference logs, latency quantiles, token & cost tracking
├── domain/                     # Framework-agnostic pure dataclasses
│   └── __init__.py             # InterviewSession, InterviewQuestion, Evaluation, SkillEvidence, etc.
├── schemas/                    # Pydantic request/response validation schemas
│   └── __init__.py             # StartSessionRequest, SubmitAnswerRequest, SubmitCodeRequest, etc.
├── services/                   # Orchestrators and domain services
│   ├── orchestrator/           # Backend-authoritative interview FSM & adaptive engine
│   │   ├── service.py          # State transitions, question queue, follow-up generator
│   │   ├── models.py           # OrchestratorState, InterviewPhase, SessionPolicy
│   │   ├── intelligence.py     # SkillEvidenceAggregator & proficiency updating
│   │   ├── recommendations.py  # RecommendationEngine & prescriptive activity generator
│   │   └── skills.py           # Taxonomy role-to-skill mappings
│   ├── ai/                     # AI provider integration & execution engine
│   │   ├── engine.py           # AssessmentEngine with telemetry context & AIRun audit logging
│   │   ├── schemas.py          # AI response contracts (EvaluationSchema, QuestionSchema)
│   │   └── providers/          # GroqProvider, GeminiProvider, BaseAIProvider
│   ├── supabase_service.py     # Typed Supabase PostgreSQL data access service
│   ├── question_difficulty_service.py # Model loader for difficulty predictor
│   ├── question_skill_service.py      # Model loader for skill tagger
│   ├── security_service.py     # Session integrity aggregator
│   └── transcription.py        # Faster-whisper audio transcription runner
├── analytics/                  # Longitudinal statistical computation engine
│   ├── methodology.py          # OLS regression, IQR, weakness clusters, lift calculation
│   ├── models.py               # Typed analytics response structures
│   └── service.py              # LongitudinalAnalyticsService
├── security/                   # Authentication & proctoring integrity engine
│   ├── auth.py                 # Supabase JWT extraction, caller identity, ownership checks
│   └── integrity.py            # Tamper anomaly scoring, review recommendation engine
├── observability/              # Telemetry tracking & metrics aggregator
│   └── tracker.py              # TelemetryTracker singleton, memory RSS, token cost modeling
└── infrastructure/             # External adapters & system wrappers
    ├── supabase.py             # SupabaseClient singleton
    └── sandbox.py              # Process-isolated code execution engine with strict timeouts
```

### 4.2 Domain Entity Model

The core entities in [`app/domain/__init__.py`](app/domain/__init__.py) model the complete interview lifecycle:

```
┌──────────────────┐       1:N       ┌──────────────────┐       1:N       ┌──────────────────┐
│ InterviewSession ├─────────────────► InterviewQuestion├─────────────────►CandidateResponse│
└────────┬─────────┘                 └──────────────────┘                 └────────┬─────────┘
         │                                                                         │ 1:1
         │ 1:N                                                                     ▼
         │                           ┌──────────────────┐       1:N       ┌──────────────────┐
         ├───────────────────────────►  IntegrityEvent  │                 │    Evaluation    │
         │                           └──────────────────┘                 └────────┬─────────┘
         │ 1:N                                                                     │ 1:N
         │                           ┌──────────────────┐                          ▼
         ├───────────────────────────►  Recommendation │                 ┌──────────────────┐
         │                           └──────────────────┘                 │  SkillEvidence   │
         │ 1:N                                                            └────────┬─────────┘
         ▼                                                                         │ Aggregates
┌──────────────────┐                                                               ▼
│      AIRun       │                                                      ┌──────────────────┐
└──────────────────┘                                                      │SkillProfile (User│
                                                                          └──────────────────┘
```

- `InterviewSession`: Tracks session status (`INITIALIZING`, `ACTIVE`, `PAUSED`, `COMPLETED`, `ABANDONED`), time boundaries, and cumulative weighted score.
- `InterviewQuestion`: Contains question prompt, role target, difficulty band, expected reasoning, and detailed evaluation rubric.
- `CandidateResponse`: Records candidate text or source code submission with submission timestamp.
- `Evaluation`: Stores multi-criteria ratings (technical accuracy, conceptual depth, problem solving, communication), feedback, and structured rubric coverage.
- `SkillEvidence`: Discrete atomic observations with signal strength and difficulty weighting.
- `CandidateSkillProfile`: User-level longitudinal Bayesian estimate of proficiency (0–100), confidence level, and trend.
- `Recommendation`: Prescriptive, targeted exercises addressing identified weakness clusters.
- `IntegrityEvent`: Tampering or focus anomalies classified by severity (`LOW`, `MEDIUM`, `HIGH`).
- `CodeSubmission` & `ExecutionRun`: Sandboxed execution results (stdout, stderr, exit code, execution time in ms).
- `AIRun`: Audit log recording model name, prompt tokens, completion tokens, latency, status, and cost.

### 4.3 State Machine Orchestrator (FSM)

The interview progression is strictly governed by the backend state machine ([`app/services/orchestrator/service.py`](app/services/orchestrator/service.py)):

```
                      ┌──────────────┐
                      │ INITIALIZING │
                      └──────┬───────┘
                             │
                             ▼
                      ┌──────────────┐
     ┌───────────────►│  QUESTIONING │◄──────────────┐
     │                └──────┬───────┘               │
     │                       │                       │
     │                       ▼                       │
     │                ┌──────────────┐               │
     │                │  EVALUATING  │               │
     │                └──────┬───────┘               │
     │                       │                       │
     │                       ▼                       │
     │          ┌─────────────────────────┐          │
     │          │  DIFFICULTY_ADJUSTMENT  │          │
     │          └──────┬───────────┬──────┘          │
     │                 │           │                 │
     │                 │           ▼                 │
     │                 │    ┌──────────────┐         │
     │                 │    │  FOLLOW_UP   ├─────────┘
     │                 │    └──────────────┘
     │ Next Question   │
     └─────────────────┤ Max Questions / Time Reached
                       ▼
             ┌──────────────────┐
             │ FINAL_ASSESSMENT │
             └─────────┬────────┘
                       │
                       ▼
             ┌──────────────────┐
             │    COMPLETED     │
             └──────────────────┘
```

Any unauthorized state skip (e.g., attempting to submit an evaluation while in `INITIALIZING` or requesting a follow-up after `COMPLETED`) is rejected with an HTTP 400 state transition fault.

### 4.4 Longitudinal Analytics & Methodology

Rather than generating fake progress bars, the platform performs real statistical calculations in [`app/analytics/methodology.py`](app/analytics/methodology.py):

- **Ordinary Least Squares (OLS) Regression**:
  Calculates the candidate's exact score trajectory slope ($\beta_1$) over time:
  $$\beta_1 = \frac{\sum (t_i - \bar{t})(S_i - \bar{S})}{\sum (t_i - \bar{t})^2}$$
  A positive slope with high $R^2$ indicates verifiable skill mastery improvement.
- **Score Dispersion & Consistency Index**:
  Computes sample standard deviation ($\sigma$) and Interquartile Range (IQR = $Q_3 - Q_1$). High variance flags performance volatility under specific problem domains.
- **Canonical Weakness Clustering**:
  Evaluations and feedback are classified across 9 canonical weakness clusters via compiled regex scanners:
  1. `edge_case_handling`: Boundary condition flaws, null/empty checks, zero/overflow bugs.
  2. `time_complexity`: Suboptimal algorithms, $O(n^2)$ vs $O(n \log n)$ efficiency.
  3. `space_complexity`: Excessive auxiliary space, memory leakage.
  4. `recursion_and_backtracking`: Missing base cases, stack overflow risks.
  5. `code_modularity`: Single responsibility violations, spaghetti logic.
  6. `scalability_architecture`: Single points of failure, unscalable bottlenecks.
  7. `error_handling`: Missing exception handling, unvalidated inputs.
  8. `communication_clarity`: Unclear assumptions, unstructured explanation.
  9. `data_structure_selection`: Suboptimal choice (e.g., array instead of hash map/heap).
- **Recommendation Lift Analysis**:
  Measures the delta in proficiency before and after completing a prescribed practice drill:
  $$\text{Lift} = \text{Proficiency}_{\text{post}} - \text{Proficiency}_{\text{baseline}}$$

### 4.5 Code Execution Sandbox Infrastructure

Arbitrary user-submitted code in the technical interview and standalone IDE is executed inside isolated subprocesses ([`app/infrastructure/sandbox.py`](app/infrastructure/sandbox.py)):
- Dedicated temporary execution directories with isolated environment variables.
- Strict wall-clock execution timeouts (default: 5.0 seconds).
- Buffer limits on stdout/stderr to prevent memory overflow from infinite loops.
- Support for Python 3 (`python`) and JavaScript / TypeScript (`node`).

### 4.6 Security & Identity Architecture

The security model ([`app/security/auth.py`](app/security/auth.py)) enforces zero-trust boundaries:
1. **Server-Side Identity Verification**: Client-provided `user_id` values in request payloads are never trusted. The caller's real identity is cryptographically decoded from the Supabase JWT (`Authorization: Bearer <token>`).
2. **Resource Ownership Verification**: Every session, question, code submission, and analytics lookup verifies that `session.user_id == caller_id`. Cross-user tampering attempts receive HTTP 403 Forbidden.
3. **Proctoring Integrity Fusion**: The proctoring engine aggregates tab-switch signals, devtools triggers, paste velocity anomalies, and camera focus events into a unified risk index with audit logs.

---

## 5. Frontend Architecture & User Experience

The frontend is built with **Next.js 14 (App Router)**, **TypeScript**, **Tailwind CSS**, **Shadcn UI**, and **Radix UI Primitives**.

```
frontend/
├── app/
│   ├── page.tsx                     # Landing page with interactive feature showcase
│   ├── login/page.tsx               # Supabase Auth login
│   ├── signup/page.tsx              # Supabase Auth registration
│   ├── interview/page.tsx           # Stateful interview workspace
│   ├── ide/page.tsx                 # Full-screen candidate IDE & code analysis
│   ├── practice/page.tsx            # Adaptive practice & targeted skill drills
│   ├── dashboard/page.tsx           # Candidate progress, radar charts, recommendations
│   ├── analytics/page.tsx           # Deep longitudinal reports & regression trends
│   ├── history/page.tsx             # Past session archive & drill-down replays
│   ├── observability/page.tsx       # Live platform health & inference telemetry
│   ├── admin/
│   │   └── observability/page.tsx   # Detailed admin telemetry dashboard
│   ├── layout.tsx                   # Root HTML shell & global font providers
│   └── globals.css                  # CSS variables, dark-mode tokens, animations
├── components/
│   ├── interview/                   # Session timer, audio visualizer, question viewer
│   ├── ide/                         # Enhanced Monaco code editor, stdin/stdout console
│   ├── anti-cheat/                  # Webcam proctoring card, anomaly status indicators
│   ├── navigation.tsx               # Responsive application navbar
│   └── ui/                          # 50+ Shadcn/ui components (cards, dialogs, badges)
├── lib/
│   ├── api-client.ts                # Fully typed TypeScript HTTP client for Flask API
│   ├── supabase.ts                  # Supabase browser & server SSR client helpers
│   └── utils.ts                     # Tailwind class merge utility (cn)
├── hooks/                           # Custom React hooks (useAuth, useInterview, etc.)
└── middleware.ts                    # Edge route protection & token propagation
```

### 5.1 Application Routes & Workspaces

- **Landing Page (`/`)**: Dynamic hero showcasing platform capabilities, sample evaluation breakdowns, and feature deep dives.
- **Interview Studio (`/interview`)**: The live session interface featuring synchronous question display, time remaining countdown, speech-to-text audio input, text answer submission, and instant multi-dimensional score feedback.
- **Standalone IDE (`/ide`)**: Multi-language developer environment powered by the Monaco Editor, featuring syntax highlighting, sample test case execution, and static ML defect risk analysis.
- **Adaptive Practice (`/practice`)**: Personalized drill generator targeting a candidate's specific identified weakness clusters (e.g., dynamic programming, edge-case validation).
- **Executive Dashboard (`/dashboard`)**: Unified overview of candidate readiness, recent session scores, skill radar breakdown, and pending recommendations.
- **Longitudinal Analytics (`/analytics`)**: Detailed statistical visualizations displaying score regression trajectories, weakness cluster frequencies, and before-and-after recommendation lift.
- **System Observability (`/observability` and `/admin/observability`)**: Live telemetry dashboard tracking ML vs. LLM inference latency, token counts, estimated dollar costs, device utilization, and model health.

### 5.2 Candidate IDE & Real-Time Runner

The IDE incorporates:
- **Monaco Editor**: IntelliSense, syntax validation, bracket matching, and theme customization.
- **Multi-Language Support**: Python, JavaScript, TypeScript, C++, Java, Go, Rust.
- **Defect Detection Badge**: Displays CodeBERT vulnerability analysis, flagging potential edge-case failures, unhandled exceptions, or boundary bugs before code execution.
- **Execution Terminal**: Real-time display of process stdout, stderr, execution time in milliseconds, and exit status.

### 5.3 Multi-Signal Anti-Cheat & Proctoring Engine

The browser proctoring component ([`frontend/components/anti-cheat/`](frontend/components/anti-cheat/)) passively monitors:
- **Tab & Window Visibility**: Listens to `visibilitychange` and `window.blur` events to track candidate tab switching.
- **Developer Tools Heuristic**: Analyzes window outer-to-inner dimensional discrepancies.
- **Paste Event Analysis**: Intercepts `paste` events and measures character payload volume.
- **Keystroke Cadence Variance**: Flags unnatural copy-paste bursts vs. authentic human typing.
- **Webcam Feed**: Displays continuous client-side camera stream to discourage multi-person collusion.

All events are posted to `/api/log_anomaly` and `/api/log_event`, building a cryptographic integrity report.

---

## 6. Authoritative Database Schema & Migrations

The database is built on **Supabase (PostgreSQL 15+)** with Row Level Security (RLS) enabled across every table. All tables use UUID primary keys and ISO-8601 UTC timestamps.

### 6.1 Relational Architecture (`001_authoritative_schema.sql`)

```sql
-- 1. Profiles (Candidate profile extending Supabase auth.users)
CREATE TABLE profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Interview Sessions
CREATE TABLE interview_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    interview_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'INITIALIZING',
    start_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    end_time TIMESTAMPTZ,
    score NUMERIC(5, 2) DEFAULT 0.0,
    session_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Interview Questions
CREATE TABLE interview_questions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    question_type VARCHAR(50) NOT NULL,
    difficulty VARCHAR(50) NOT NULL DEFAULT 'intermediate',
    skill_focus VARCHAR(100),
    domain VARCHAR(100),
    expected_reasoning TEXT,
    evaluation_rubric TEXT,
    question_order INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Responses (Candidate answers)
CREATE TABLE responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id UUID NOT NULL REFERENCES interview_questions(id) ON DELETE CASCADE,
    response_text TEXT NOT NULL,
    audio_url VARCHAR(500),
    is_code BOOLEAN NOT NULL DEFAULT false,
    programming_language VARCHAR(50),
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Evaluations (Structured scoring)
CREATE TABLE evaluations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    response_id UUID NOT NULL REFERENCES responses(id) ON DELETE CASCADE,
    score NUMERIC(5, 2) NOT NULL,
    feedback TEXT NOT NULL,
    evaluation_details JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(50) NOT NULL DEFAULT 'COMPLETED',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. Skill Evidence (Atomic observations)
CREATE TABLE skill_evidence (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    evaluation_id UUID NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    skill_name VARCHAR(100) NOT NULL,
    signal_strength NUMERIC(5, 2) NOT NULL,
    evidence_text TEXT,
    confidence VARCHAR(50) DEFAULT 'medium',
    difficulty VARCHAR(50) DEFAULT 'intermediate',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 7. Candidate Skill Profiles (Longitudinal proficiency)
CREATE TABLE candidate_skill_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    skill_name VARCHAR(100) NOT NULL,
    estimated_proficiency NUMERIC(5, 2) NOT NULL DEFAULT 50.0,
    confidence VARCHAR(50) NOT NULL DEFAULT 'insufficient evidence',
    evidence_count INTEGER NOT NULL DEFAULT 0,
    recent_performance NUMERIC(5, 2) DEFAULT 0.0,
    historical_performance NUMERIC(5, 2) DEFAULT 0.0,
    improvement_trend VARCHAR(50) DEFAULT 'stable',
    last_evaluated_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, skill_name)
);

-- 8. Recommendations (Targeted practice plans)
CREATE TABLE recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
    session_id UUID REFERENCES interview_sessions(id) ON DELETE SET NULL,
    target_skill VARCHAR(100) NOT NULL,
    strategy VARCHAR(100) NOT NULL,
    reason TEXT NOT NULL,
    recommended_activity JSONB NOT NULL DEFAULT '{}',
    priority VARCHAR(50) NOT NULL DEFAULT 'MEDIUM',
    expected_learning_objective TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING',
    baseline_proficiency NUMERIC(5, 2) DEFAULT 0.0,
    post_outcome_proficiency NUMERIC(5, 2),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

-- 9. Session Logs & Anomalies
CREATE TABLE session_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    event_type VARCHAR(100) NOT NULL,
    event_payload JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE session_anomalies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES interview_sessions(id) ON DELETE CASCADE,
    anomaly_type VARCHAR(100) NOT NULL,
    severity VARCHAR(50) NOT NULL DEFAULT 'LOW',
    details JSONB DEFAULT '{}',
    confidence NUMERIC(3, 2) DEFAULT 1.0,
    source VARCHAR(50) DEFAULT 'browser_api',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 10. AI Runs & Code Execution Auditing
CREATE TABLE ai_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES interview_sessions(id) ON DELETE SET NULL,
    operation VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    latency_seconds NUMERIC(6, 3) NOT NULL,
    status VARCHAR(50) NOT NULL,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE code_submissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_id UUID NOT NULL REFERENCES interview_questions(id) ON DELETE CASCADE,
    code_text TEXT NOT NULL,
    programming_language VARCHAR(50) NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE execution_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    code_submission_id UUID NOT NULL REFERENCES code_submissions(id) ON DELETE CASCADE,
    status VARCHAR(50) NOT NULL,
    stdout TEXT DEFAULT '',
    stderr TEXT DEFAULT '',
    exit_code INTEGER,
    execution_time_ms INTEGER,
    sandbox_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### 6.2 Vector Storage & pgvector (`002_question_embeddings.sql`)

Stores dense 384-dimensional question representations extracted by `all-MiniLM-L6-v2` to support semantic retrieval and near-duplicate suppression:

```sql
CREATE TABLE question_embeddings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    question_id VARCHAR(100) NOT NULL UNIQUE,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    embedding_version VARCHAR(50) NOT NULL DEFAULT 'all-minilm-l6-v2',
    embedding_vector JSONB NOT NULL, -- 384-dimensional float vector
    question_text TEXT,
    skill_focus VARCHAR(100),
    difficulty VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### 6.3 Telemetry & Inference Logging (`003_observability_telemetry.sql`)

Unified tracking table recording every local ML inference and external LLM call:

```sql
CREATE TABLE ai_inference_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    inference_type VARCHAR(16) NOT NULL, -- 'ml' or 'llm'
    model VARCHAR(128) NOT NULL,
    model_version VARCHAR(64) DEFAULT 'v1.0',
    task VARCHAR(128) NOT NULL,
    latency_ms NUMERIC(10, 2) NOT NULL,
    success BOOLEAN NOT NULL DEFAULT true,
    status VARCHAR(32) NOT NULL DEFAULT 'SUCCESS', -- 'SUCCESS', 'FAILED', 'FALLBACK', 'TIMEOUT'
    error_message TEXT,
    input_size INTEGER DEFAULT 0,
    output_size INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    memory_device VARCHAR(64) DEFAULT 'cpu',
    fallback_usage BOOLEAN NOT NULL DEFAULT false,
    retry_count INTEGER NOT NULL DEFAULT 0,
    cost_estimate_usd NUMERIC(10, 6) DEFAULT 0.000000,
    metadata JSONB DEFAULT '{}'
);
```

---

## 7. Complete REST API Catalog (40+ Endpoints)

All endpoints accept and return `application/json` (unless handling multipart audio). Protected endpoints require `Authorization: Bearer <token>`.

### 7.1 Interview Session Management

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `POST` | `/api/start_session` | **Required** | Initialize stateful interview session & state machine | `{"interview_type": "Software Engineer"}` | `{"session_id": "...", "status": "ACTIVE"}` |
| `POST` | `/api/end_session/<session_id>` | **Required** | Conclude session, compute cumulative score & generate summary | None | `{"session_id": "...", "final_score": 84.5}` |
| `GET` | `/api/session/<session_id>` | **Required** | Retrieve session details, questions & evaluations | None | `{"session": {...}}` |
| `GET` | `/api/session/<session_id>/state` | **Required** | Inspect active state machine phase, difficulty & question index | None | `{"phase": "QUESTIONING", "difficulty": "intermediate"}` |
| `GET` | `/api/user/<user_id>/sessions` | **Required** | List historical interview sessions for candidate | None | `{"sessions": [...]}` |

### 7.2 Question Delivery & Follow-ups

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `GET` | `/api/get_question` | **Required** | Retrieve next adaptive question based on candidate ability | `?session_id=<uuid>` | `{"id": "...", "question_text": "...", "rubric": "..."}` |
| `POST` | `/api/follow_up_question` | **Required** | Generate contextual follow-up probing candidate edge-case gaps | `{"session_id": "...", "question_id": "..."}` | `{"question": {...}}` |
| `GET` | `/api/session/<session_id>/questions/<question_id>/selection_rationale` | **Required** | Explain why question was selected (Fisher Info, CAT criteria) | None | `{"rationale": "Selected to maximize Fisher info at theta=0.4"}` |

### 7.3 Candidate Answers & Evaluation

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `POST` | `/api/submit_answer` | **Required** | Submit text answer; triggers Hybrid ML+LLM evaluation | `{"session_id": "...", "question_id": "...", "answer_text": "..."}` | `{"score": 82.0, "feedback": "...", "details": {...}}` |
| `POST` | `/api/submit_code` | **Required** | Submit source code; triggers CodeBERT risk + execution + eval | `{"session_id": "...", "question_id": "...", "code_text": "...", "language": "python"}` | `{"score": 90.0, "code_quality": 88, "ml_defect_detection": {...}}` |
| `POST` | `/api/submit_voice_answer`| **Required** | Upload multipart audio file, transcribe via Whisper & evaluate | `FormData: {session_id, question_id, audio: file}` | `{"transcription": "...", "evaluation": {...}}` |
| `POST` | `/api/transcribe` | **Required** | Transcribe arbitrary candidate audio recording via Whisper | `FormData: {audio: file}` | `{"text": "...", "language": "en"}` |

### 7.4 Sandboxed Code Execution

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `POST` | `/api/run_code` | Optional | Run code in isolated subprocess with 5-second wall clock limit | `{"code": "print(42)", "language": "python"}` | `{"stdout": "42\n", "stderr": "", "exit_code": 0, "execution_time_ms": 45}` |

### 7.5 Adaptive Practice Mode

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `POST` | `/api/practice/coding` | Optional | Generate targeted coding exercise tailored to specific skill | `{"skill": "dynamic-programming", "difficulty": "intermediate"}` | `{"question": {...}}` |
| `POST` | `/api/practice/evaluate`| Optional | Score standalone practice problem submission | `{"question_id": "...", "code": "...", "language": "python"}` | `{"score": 85, "feedback": "..."}` |
| `POST` | `/api/practice/sequence`| **Required** | Generate multi-step curriculum sequence targeting ZPD | `{"user_id": "...", "skill": "algorithms"}` | `{"sequence": [...]}` |

### 7.6 Skill Intelligence & Prescriptive Recommendations

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `GET` | `/api/intelligence/skills/<user_id>` | **Required** | Retrieve complete multi-skill proficiency profile | None | `{"skills": [...]}` |
| `GET` | `/api/intelligence/skills/<user_id>/<skill_name>` | **Required** | Drill down into specific skill proficiency & historical evidence | None | `{"skill": {...}}` |
| `GET` | `/api/intelligence/recommendations/<user_id>` | **Required** | Fetch pending and completed practice recommendations | None | `{"recommendations": [...]}` |
| `POST` | `/api/intelligence/recommendations/<user_id>/generate` | **Required** | Force generation of targeted recommendations from weaknesses | None | `{"recommendations": [...]}` |
| `PATCH` | `/api/intelligence/recommendations/<recommendation_id>/status` | **Required** | Update recommendation state (`ACCEPTED`, `COMPLETED`, `DISMISSED`) | `{"status": "COMPLETED"}` | `{"updated": true}` |

### 7.7 Longitudinal Analytics & Trajectories

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `GET` | `/api/analytics/<user_id>` | **Required** | Comprehensive candidate analytics roll-up | None | `{"summary": {...}, "trends": {...}}` |
| `GET` | `/api/analytics/trends/scores/<user_id>` | **Required** | OLS regression score trajectory over time | None | `{"points": [...], "slope": 1.42, "r_squared": 0.88}` |
| `GET` | `/api/analytics/trends/skills/<user_id>` | **Required** | Multi-skill radar breakdown & trajectory metrics | None | `{"skill_trends": [...]}` |
| `GET` | `/api/analytics/weaknesses/<user_id>` | **Required** | Ranked frequency distribution across 9 weakness clusters | None | `{"weaknesses": [...]}` |
| `GET` | `/api/analytics/consistency/<user_id>` | **Required** | Score dispersion, standard deviation & Interquartile Range | None | `{"variance": 12.4, "iqr": 8.0, "consistency_index": 0.82}` |
| `GET` | `/api/analytics/next-practice/<user_id>` | **Required** | Algorithmic next-best-practice recommendation | None | `{"recommended_drill": {...}}` |
| `GET` | `/api/analytics/filter-options/<user_id>` | **Required** | Available filters (roles, dates, difficulty levels) | None | `{"roles": [...], "difficulties": [...]}` |

### 7.8 Proctoring, Integrity & Telemetry Ingestion

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `POST` | `/api/security/check` | Optional | Post browser heartbeat & focus signal | `{"session_id": "...", "signal": "tab_hidden"}` | `{"status": "recorded"}` |
| `GET` | `/api/security/report/<session_id>` | **Required** | Generate composite anti-cheat audit report | None | `{"review_recommended": false, "risk_score": 12.5, "anomalies": [...]}` |
| `POST` | `/api/log_event` | Optional | Record structured session telemetry event | `{"session_id": "...", "event_type": "button_click"}` | `{"status": "ok"}` |
| `POST` | `/api/log_anomaly` | Optional | Report proctoring anomaly (paste, blur, devtools) | `{"session_id": "...", "anomaly_type": "tab_switch", "severity": "MEDIUM"}` | `{"status": "ok"}` |
| `GET` | `/api/events/<session_id>` | **Required** | Query event audit log for session | None | `{"events": [...]}` |
| `GET` | `/api/anomalies/<session_id>` | **Required** | Query anomaly audit log for session | None | `{"anomalies": [...]}` |

### 7.9 Platform Observability & Model Health

| Method | Endpoint | Auth | Description | Request Body / Params | Response |
|---|---|---|---|---|---|
| `GET` | `/api/health` | Public | System liveness probe & blueprint count | None | `{"status": "healthy", "service": "interview-ai-backend"}` |
| `GET` | `/api/metrics` | Public | System uptime, active session count & memory RSS | None | `{"uptime_seconds": 3600, "memory_mb": 142.5}` |
| `GET` | `/api/ml/health` | Public | Inspect ML model registry, artifact presence & versions | None | `{"registry_version": "1.0.0", "models": {...}}` |
| `GET` | `/api/observability/stats` | Public | Latency percentiles (P50, P90, P99), token totals, dollar cost | None | `{"stats": {...}}` |
| `GET` | `/api/observability/events`| Public | Recent real-time AI inference event stream | None | `{"events": [...]}` |
| `POST` | `/api/observability/ping` | Public | Observability heartbeat and diagnostics ping | `{"client": "frontend"}` | `{"pong": true, "timestamp": "..."}` |

---

## 8. Telemetry, Cost Modeling & Observability

Observability is implemented in [`app/observability/tracker.py`](app/observability/tracker.py) and surfaced through `/api/observability/stats` and the Next.js admin portal:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        AI OBSERVABILITY MONITOR                        │
│                                                                        │
│ Total Inferences: 1,482   │ Total Tokens: 482,109 │ Est Cost: $0.241   │
│ ML Latency P50:   18 ms   │ LLM Latency P50: 640 ms│ Fallback Rate: 1.2%│
├────────────────────────────────────────────────────────────────────────┤
│ Model Breakdown:                                                       │
│ • question-difficulty-v1  (Local ML): 412 calls │ 19ms avg │ $0.000000 │
│ • question-skill-v1       (Local ML): 412 calls │ 16ms avg │ $0.000000 │
│ • answer-nli-v1           (DeBERTa):  284 calls │112ms avg │ $0.000000 │
│ • code-risk-v1            (CodeBERT): 140 calls │148ms avg │ $0.000000 │
│ • gpt-oss-120b            (Groq LLM): 198 calls │590ms avg │ $0.184200 │
│ • gemini-2.5-flash        (Google):    36 calls │720ms avg │ $0.056800 │
└────────────────────────────────────────────────────────────────────────┘
```

### Precise Cost Estimation Rates

- **Local Machine Learning**: \$0.000000 (executed on CPU/GPU hardware).
- **Groq Llama 3.3 70B**: \$0.59 / 1M prompt tokens, \$0.79 / 1M completion tokens.
- **Google Gemini 2.5 Flash**: \$0.10 / 1M prompt tokens, \$0.40 / 1M completion tokens.
- **Google Gemini 1.5 Pro**: \$1.25 / 1M prompt tokens, \$5.00 / 1M completion tokens.

Dual Persistence ensures resilience: telemetry events are simultaneously written to an in-memory thread-safe circular ring buffer and asynchronously flushed to the Supabase `ai_inference_logs` table.

---

## 9. Installation, Setup & Environment Configuration

### 9.1 Prerequisites

- **Python**: Version 3.10, 3.11, or 3.12
- **Node.js**: Version 18.17+ or 20+
- **Package Managers**: `pip` (Python) and `npm` or `yarn` (Node.js)
- **Supabase Account**: A cloud Supabase project or local Docker Supabase instance
- **AI Provider Key**: At least one API key from [Groq Console](https://console.groq.com/) or [Google AI Studio](https://aistudio.google.com/)

### 9.2 Backend Installation

```bash
# 1. Navigate to project root
cd "Interview-ai-er"

# 2. Create and activate a Python virtual environment
python -m venv venv

# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd.exe):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy environment template
cp env.example .env
```

### 9.3 Frontend Installation

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install Node dependencies
npm install

# 3. Create local environment configuration
cp .env.example .env.local
```

### 9.4 Environment Variable Matrix

#### Backend Configuration (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `AI_PROVIDER` | No | `groq` | Default generative LLM provider (`groq` or `gemini`). |
| `GROQ_API_KEY` | **Conditional** | — | API key for Groq Cloud. Required if `AI_PROVIDER=groq`. |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Model ID hosted on Groq (e.g., `llama-3.3-70b-versatile`). |
| `GEMINI_API_KEY` | **Conditional** | — | Google Gemini API key. Required if `AI_PROVIDER=gemini` or fallback. |
| `SUPABASE_URL` | **Yes** | — | Supabase project URL (`https://<project-id>.supabase.co`). |
| `SUPABASE_KEY` | **Yes** | — | Supabase Service Role Key (grants administrative backend access). |
| `FLASK_SECRET_KEY` | No | `dev-secret-key`| Flask session cryptographic signature secret. |
| `FLASK_ENV` | No | `development` | Runtime environment mode (`development` or `production`). |
| `FLASK_DEBUG` | No | `False` | Enable/disable Flask debug server mode. |
| `FRONTEND_URL` | No | `http://localhost:3000` | Allowed CORS origin for frontend client requests. |

#### Frontend Configuration (`frontend/.env.local`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `NEXT_PUBLIC_API_URL` | **Yes** | `http://localhost:5000` | Base URL of the running Flask backend. |
| `NEXT_PUBLIC_API_BASE_URL` | **Yes** | `http://localhost:5000/api` | API path prefix for backend routes. |
| `NEXT_PUBLIC_SUPABASE_URL` | **Yes** | — | Public Supabase project URL. |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | **Yes** | — | Supabase anonymous public key for client-side auth. |

### 9.5 Supabase Database Setup

Follow the authoritative step-by-step checklist in [`SUPABASE_SETUP_CHECKLIST.md`](SUPABASE_SETUP_CHECKLIST.md):

1. Open your Supabase Dashboard -> **SQL Editor**.
2. Execute [`001_authoritative_schema.sql`](001_authoritative_schema.sql) to provision all 14 domain tables, primary keys, foreign keys, and RLS policies.
3. Execute [`002_question_embeddings.sql`](002_question_embeddings.sql) to provision dense question vector storage.
4. Execute [`003_observability_telemetry.sql`](003_observability_telemetry.sql) to provision the `ai_inference_logs` table.
5. In Supabase Dashboard -> **Authentication** -> **Providers**, confirm **Email / Password** provider is enabled.

### 9.6 Launching the Services

**Terminal 1: Backend Server**
```bash
# In project root with active venv:
python app.py
# Server starts on http://127.0.0.1:5000
```

**Terminal 2: Frontend Next.js Server**
```bash
# In frontend/ directory:
npm run dev
# Frontend starts on http://localhost:3000
```

---

## 10. Testing, Benchmarks & Validation

The platform features an extensive test harness built on `pytest` across 23 test suites:

```bash
# Run the complete test suite
pytest

# Run tests with short tracebacks and verbose output
pytest -v --tb=short

# Run specific domain test suites
pytest tests/test_adaptive_selector.py
pytest tests/test_concept_coverage.py
pytest tests/test_defect_detector.py
pytest tests/test_analytics.py
pytest tests/test_authorization.py
pytest tests/test_fallback_resilience.py

# Run latency and inference benchmarks
pytest tests/benchmark_ml_pipeline.py
```

### Test Suite Coverage Catalog

- [`tests/test_adaptive_selector.py`](tests/test_adaptive_selector.py): Validates Fisher Information calculations, item difficulty parameters, and CAT sequence selection.
- [`tests/test_concept_coverage.py`](tests/test_concept_coverage.py): Tests DeBERTa NLI cross-encoder rubric entailment and concept decomposition.
- [`tests/test_defect_detector.py`](tests/test_defect_detector.py): Validates CodeBERT sequence classification and vulnerability detection on source code.
- [`tests/test_fallback_resilience.py`](tests/test_fallback_resilience.py): Simulates local model failure, memory exhaustion, and verifies graceful degradation to Tier 3 LLM schemas.
- [`tests/test_analytics.py`](tests/test_analytics.py): Tests OLS regression slope calculation, IQR consistency metrics, and weakness cluster regexes.
- [`tests/test_authorization.py`](tests/test_authorization.py): Enforces that cross-user access attempts receive HTTP 403 Forbidden.
- [`tests/test_domain_models.py`](tests/test_domain_models.py): Verifies domain entity dataclass immutability and serialization.
- [`tests/test_speech_pipeline.py`](tests/test_speech_pipeline.py): Tests audio file ingestion, chunking, and Whisper transcription pipelines.

---

## 11. Engineering Documentation Directory

For deeper architectural studies, consult the specialized documentation in the [`docs/`](docs/) directory:

- 📐 **[System Rebuild Plan](docs/REBUILD_PLAN.md)**: Original technical roadmap and architecture specification.
- 🔬 **[ML Architecture & Technical Feasibility](docs/ML_ARCHITECTURE.md)**: Theoretical foundation, datasets (TACO, APPS, EdNet, CodeXGLUE), and model selection rationale.
- 🗄️ **[Database Architecture](docs/DATABASE_ARCHITECTURE.md)**: Relational schema design, entity relationship diagrams, and RLS policies.
- 🔄 **[Interview Orchestration Engine](docs/INTERVIEW_ORCHESTRATION.md)**: State machine lifecycle transitions and policy engine.
- 🎯 **[Recommendation Engine](docs/RECOMMENDATION_ENGINE.md)**: Weakness clustering, prescriptive drill generation, and lift estimation.
- 🤖 **[AI Pipeline Specification](docs/AI_PIPELINE.md)**: Prompt contracts, Pydantic schemas, and structured output parsing.
- 🛡️ **[Auth & Security Audit](docs/AUTH_SECURITY_AUDIT.md)**: Threat modeling, JWT verification, and sandbox security isolation.
- 📊 **[Confidence Glossary](docs/CONFIDENCE_GLOSSARY.md)**: Mathematical definitions of confidence bands across ML inferences.
- 🔍 **[Defect Detection Report](docs/DEFECT_DETECTION_REPORT.md)**: CodeBERT fine-tuning evaluation on CodeXGLUE Devign.
- 🏷️ **[Dataset Credits & Licensing](docs/DATASET_CREDITS.md)**: Provenance, academic citations, and open-source licenses for all training datasets.
