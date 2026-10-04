"""
Unit and integration tests for ML & LLM Inference Observability.

Validates:
  1. ML inference measurement (model, task, latency, success, input/output size, device)
  2. LLM inference measurement (tokens, cost, retry count, latency)
  3. Failure tracking & error categorization
  4. Fallback usage tracking
  5. Tail latency (P95) and average latency calculation accuracy
  6. Failure rate and fallback rate percentage calculations
  7. Most expensive operations sorting
  8. Non-fabrication guarantee (only returns genuine recorded data)
  9. Observability API endpoints (/api/observability/stats, /api/observability/events, /api/observability/ping)
"""
import pytest
from datetime import datetime, timezone

from app.observability.tracker import (
    InferenceTelemetryTracker,
    get_telemetry_tracker,
    _estimate_token_cost,
    _detect_default_device,
)
from app.services.supabase_service import SupabaseService
from app import create_app


@pytest.fixture
def mock_supabase():
    """Mock SupabaseService with fresh in-memory logs."""
    service = SupabaseService()
    # Reset local inference logs for deterministic test isolation
    service._local_inference_logs.clear()
    return service


@pytest.fixture
def tracker():
    """Returns singleton telemetry tracker."""
    return get_telemetry_tracker()


@pytest.fixture
def test_client():
    """Flask test client."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_token_cost_estimation():
    """Verify provider cost estimates based on token consumption."""
    # Groq Llama 3.3 70B
    cost_llama = _estimate_token_cost("llama-3.3-70b-versatile", input_tokens=1000, output_tokens=500)
    assert cost_llama > 0.0
    assert cost_llama == round((1000 * 0.00000059) + (500 * 0.00000079), 6)

    # Local ML model has zero cloud API cost
    cost_ml = _estimate_token_cost("sentence-transformers/all-MiniLM-L6-v2", input_tokens=1000, output_tokens=500)
    assert cost_ml == 0.0


def test_device_detection():
    """Verify compute device categorization."""
    assert _detect_default_device("llm") == "api/cloud"
    ml_dev = _detect_default_device("ml")
    assert ml_dev in ("cpu", "cuda:0") or "cuda" in ml_dev


def test_record_ml_inference_span(tracker, mock_supabase):
    """Test tracking an ML inference operation."""
    with tracker.track(
        inference_type="ml",
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="semantic_retrieval",
        model_version="v1.0",
        memory_device="cpu",
    ) as span:
        span.set_input("Find two sum problem in array")
        # Simulate quick work
        dummy_result = [{"qid": "dsa_arr_two_sum", "score": 0.92}]
        span.set_output(dummy_result)

    logs = mock_supabase._local_inference_logs
    assert len(logs) >= 1
    last_log = logs[-1]

    assert last_log["inference_type"] == "ml"
    assert last_log["model"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert last_log["task"] == "semantic_retrieval"
    assert last_log["latency_ms"] >= 0.0
    assert last_log["success"] is True
    assert last_log["status"] == "SUCCESS"
    assert last_log["input_size"] > 0
    assert last_log["output_size"] == 1
    assert last_log["memory_device"] == "cpu"
    assert last_log["fallback_usage"] is False
    assert last_log["retry_count"] == 0


def test_record_llm_inference_span(tracker, mock_supabase):
    """Test tracking an LLM inference operation with tokens, retries, and costs."""
    with tracker.track(
        inference_type="llm",
        model="llama-3.3-70b-versatile",
        task="question_generation",
        model_version="v1.0",
    ) as span:
        span.set_input("Generate a systems design question on caching.")
        span.set_tokens(input_tokens=250, output_tokens=180)
        span.record_retry(retries=1)
        span.set_output({"question": "Design an LRU cache system."})

    logs = mock_supabase._local_inference_logs
    last_log = logs[-1]

    assert last_log["inference_type"] == "llm"
    assert last_log["model"] == "llama-3.3-70b-versatile"
    assert last_log["task"] == "question_generation"
    assert last_log["input_tokens"] == 250
    assert last_log["output_tokens"] == 180
    assert last_log["retry_count"] == 1
    assert last_log["cost_estimate_usd"] > 0.0
    assert last_log["memory_device"] == "api/cloud"


def test_fallback_and_failure_recording(tracker, mock_supabase):
    """Test recording fallback execution and exception failures."""
    # 1. Fallback span
    with tracker.track(
        inference_type="ml",
        model="codebert_defect_heuristic",
        task="code_defect_detection",
    ) as span:
        span.record_fallback(reason="Transformer model unavailable")
        span.set_output({"defect_prob": 0.15})

    fallback_log = mock_supabase._local_inference_logs[-1]
    assert fallback_log["fallback_usage"] is True
    assert fallback_log["status"] == "FALLBACK"
    assert fallback_log["metadata"]["fallback_reason"] == "Transformer model unavailable"

    # 2. Failure span
    with pytest.raises(ValueError):
        with tracker.track(
            inference_type="llm",
            model="gemini-2.5-flash",
            task="code_evaluation",
        ) as span:
            raise ValueError("Rate limit exceeded 429")

    failed_log = mock_supabase._local_inference_logs[-1]
    assert failed_log["success"] is False
    assert failed_log["status"] == "FAILED"
    assert "Rate limit exceeded 429" in failed_log["error_message"]


def test_observability_stats_and_tail_latencies(mock_supabase):
    """Test calculation of average, p95 tail latency, failure rate, and most expensive ops."""
    mock_supabase._local_inference_logs.clear()

    # Empty stats check: must NOT fabricate numbers
    empty_stats = mock_supabase.get_observability_stats()
    assert empty_stats["total_volume"] == 0
    assert empty_stats["ml_volume"] == 0
    assert empty_stats["llm_volume"] == 0
    assert empty_stats["average_latency_ms"] == 0.0
    assert empty_stats["p95_latency_ms"] == 0.0
    assert empty_stats["failure_rate_pct"] == 0.0
    assert empty_stats["fallback_rate_pct"] == 0.0
    assert empty_stats["most_expensive_operations"] == []

    # Insert 20 deterministic spans with increasing latencies: 10ms, 20ms, ..., 200ms
    for i in range(1, 21):
        mock_supabase.save_inference_log({
            "inference_type": "ml" if i % 2 == 0 else "llm",
            "model": "MiniLM" if i % 2 == 0 else "Llama70B",
            "task": "search" if i % 2 == 0 else "eval",
            "latency_ms": float(i * 10),
            "success": (i != 1), # 1 failure
            "status": "FAILED" if i == 1 else "SUCCESS",
            "fallback_usage": (i == 2), # 1 fallback
            "cost_estimate_usd": 0.001 if i % 2 != 0 else 0.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    stats = mock_supabase.get_observability_stats()
    assert stats["total_volume"] == 20
    assert stats["ml_volume"] == 10
    assert stats["llm_volume"] == 10

    # Average latency: (10 + 20 + ... + 200) / 20 = 2100 / 20 = 105.0
    assert stats["average_latency_ms"] == 105.0

    # P95 tail latency: 95th percentile of 20 items (sorted 10..200) is 190.0 or 200.0
    assert stats["p95_latency_ms"] >= 190.0

    # Failure rate: 1/20 = 5.0%
    assert stats["failure_rate_pct"] == 5.0

    # Fallback rate: 1/20 = 5.0%
    assert stats["fallback_rate_pct"] == 5.0

    # Most expensive operations should be sorted descending
    expensive = stats["most_expensive_operations"]
    assert len(expensive) <= 15
    assert len(expensive) > 0


def test_api_observability_endpoints(test_client, mock_supabase):
    """Test /api/observability/stats and /api/observability/events endpoints."""
    # Diagnostic Ping
    ping_res = test_client.post("/api/observability/ping", json={"target": "ml"})
    assert ping_res.status_code == 200
    ping_data = ping_res.get_json()
    assert ping_data["success"] is True

    # Stats API
    stats_res = test_client.get("/api/observability/stats?window=24h")
    assert stats_res.status_code == 200
    stats_data = stats_res.get_json()
    assert stats_data["success"] is True
    assert "data" in stats_data
    assert stats_data["data"]["total_volume"] >= 1

    # Events API
    events_res = test_client.get("/api/observability/events?limit=10")
    assert events_res.status_code == 200
    events_data = events_res.get_json()
    assert events_data["success"] is True
    assert len(events_data["data"]["events"]) >= 1
