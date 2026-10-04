"""
app/observability/tracker.py — Production-grade observability for ML and LLM inference.

Measures:
  - model
  - task
  - latency (ms)
  - success/failure
  - input size (characters / tokens / vector length)
  - output size (characters / tokens / prediction items)
  - memory / device (cpu, cuda:0, process RSS memory)
  - fallback usage (boolean)
  - retry count (integer)
  - cost estimate (USD)

Emits structured JSON logs and persists telemetry records to PostgreSQL / Supabase and local cache.
Never creates fabricated monitoring numbers — records only actual executions.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Generator, List, Optional, Union

# Structured logger for inference telemetry
telemetry_logger = logging.getLogger("telemetry.inference")
logger = logging.getLogger(__name__)

# Device and hardware detection
HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

HAS_PSUTIL = False
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    pass


def _get_process_memory_mb() -> Optional[float]:
    """Returns current process Resident Set Size (RSS) memory in megabytes."""
    if HAS_PSUTIL:
        try:
            process = psutil.Process(os.getpid())
            return round(process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            pass
    return None


def _detect_default_device(inference_type: str) -> str:
    """Detects active compute device or runtime context."""
    if inference_type.lower() == "llm":
        return "api/cloud"
    if HAS_TORCH and torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            return f"cuda:0 ({device_name})"
        except Exception:
            return "cuda:0"
    return "cpu"


def _estimate_token_cost(
    model: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int]
) -> float:
    """Estimates cost in USD based on model provider published rates. Local ML models cost $0.00."""
    if not input_tokens and not output_tokens:
        return 0.0

    in_tok = input_tokens or 0
    out_tok = output_tokens or 0
    m = model.lower()

    if "llama-3.3-70b" in m:
        # Groq Llama 3.3 70B: ~$0.59 / 1M input, ~$0.79 / 1M output
        return round((in_tok * 0.00000059) + (out_tok * 0.00000079), 6)
    elif "gemini-2.5-flash" in m or "gemini-1.5-flash" in m:
        # Gemini Flash: ~$0.10 / 1M input, ~$0.40 / 1M output
        return round((in_tok * 0.00000010) + (out_tok * 0.00000040), 6)
    elif "gemini-1.5-pro" in m:
        # Gemini Pro: ~$1.25 / 1M input, ~$5.00 / 1M output
        return round((in_tok * 0.00000125) + (out_tok * 0.00000500), 6)
    elif "gpt-4" in m or "openai" in m:
        # General LLM tier
        return round((in_tok * 0.00000250) + (out_tok * 0.00001000), 6)
    
    # Local ML inference (MiniLM, scikit-learn, rule-based) has no cloud API cost
    return 0.0


class InferenceSpan:
    """
    Active execution span for tracking an inference invocation.
    Supports setting token counts, output sizes, retries, and fallback flags.
    """
    def __init__(
        self,
        inference_type: str,
        model: str,
        task: str,
        model_version: str = "v1.0",
        input_size: int = 0,
        memory_device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.span_id: str = str(uuid.uuid4())
        self.inference_type: str = inference_type.lower()
        self.model: str = model
        self.model_version: str = model_version
        self.task: str = task
        self.start_perf: float = time.perf_counter()
        self.start_wall: str = datetime.now(timezone.utc).isoformat()
        
        self.input_size: int = input_size
        self.output_size: Optional[int] = None
        self.input_tokens: Optional[int] = None
        self.output_tokens: Optional[int] = None
        
        self.memory_device: str = memory_device or _detect_default_device(self.inference_type)
        self.start_memory_mb: Optional[float] = _get_process_memory_mb()
        
        self.success: bool = True
        self.status: str = "SUCCESS"
        self.error_message: Optional[str] = None
        self.fallback_usage: bool = False
        self.retry_count: int = 0
        self.cost_estimate_usd: float = 0.0
        self.metadata: Dict[str, Any] = metadata or {}
        self.latency_ms: float = 0.0

    def set_input(self, input_obj: Any) -> InferenceSpan:
        """Calculate input size from string, dict, or list."""
        if isinstance(input_obj, str):
            self.input_size = len(input_obj)
        elif isinstance(input_obj, (list, tuple)):
            self.input_size = len(input_obj)
        elif isinstance(input_obj, dict):
            self.input_size = len(json.dumps(input_obj, default=str))
        return self

    def set_output(self, output_obj: Any) -> InferenceSpan:
        """Calculate output size from string, dict, or list."""
        if output_obj is None:
            self.output_size = 0
        elif isinstance(output_obj, str):
            self.output_size = len(output_obj)
        elif isinstance(output_obj, (list, tuple)):
            self.output_size = len(output_obj)
        elif isinstance(output_obj, dict):
            self.output_size = len(json.dumps(output_obj, default=str))
        else:
            self.output_size = len(str(output_obj))
        return self

    def set_tokens(
        self,
        input_tokens: Optional[int],
        output_tokens: Optional[int]
    ) -> InferenceSpan:
        """Record token usage for LLM tasks."""
        if input_tokens is not None:
            self.input_tokens = int(input_tokens)
            if self.input_size == 0:
                self.input_size = self.input_tokens
        if output_tokens is not None:
            self.output_tokens = int(output_tokens)
            if self.output_size is None:
                self.output_size = self.output_tokens
        return self

    def record_retry(self, retries: int = 1) -> InferenceSpan:
        """Increment retry count."""
        self.retry_count = max(self.retry_count, retries)
        return self

    def record_fallback(self, reason: Optional[str] = None) -> InferenceSpan:
        """Mark that a fallback strategy or backup model was invoked."""
        self.fallback_usage = True
        self.status = "FALLBACK"
        if reason:
            self.metadata["fallback_reason"] = reason
        return self

    def record_error(self, error: Union[str, Exception]) -> InferenceSpan:
        """Record failure status and error details."""
        self.success = False
        self.status = "FAILED"
        self.error_message = str(error)
        return self

    def close(self) -> Dict[str, Any]:
        """Finalize timing, memory delta, and cost computation."""
        end_perf = time.perf_counter()
        self.latency_ms = round((end_perf - self.start_perf) * 1000.0, 2)
        
        # Calculate memory device note if process memory changed
        end_memory_mb = _get_process_memory_mb()
        if self.start_memory_mb is not None and end_memory_mb is not None:
            delta_mb = round(end_memory_mb - self.start_memory_mb, 2)
            self.metadata["process_memory_mb"] = end_memory_mb
            self.metadata["memory_delta_mb"] = delta_mb

        self.cost_estimate_usd = _estimate_token_cost(
            self.model,
            self.input_tokens,
            self.output_tokens
        )

        record = {
            "id": self.span_id,
            "timestamp": self.start_wall,
            "inference_type": self.inference_type,
            "model": self.model,
            "model_version": self.model_version,
            "task": self.task,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "status": self.status,
            "error_message": self.error_message,
            "input_size": self.input_size,
            "output_size": self.output_size,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "memory_device": self.memory_device,
            "fallback_usage": self.fallback_usage,
            "retry_count": self.retry_count,
            "cost_estimate_usd": self.cost_estimate_usd,
            "metadata": self.metadata,
        }

        # 1. Emit structured log
        try:
            telemetry_logger.info(json.dumps(record, default=str))
        except Exception:
            pass

        # 2. Persist to storage layer
        _persist_span_record(record)
        return record


def _persist_span_record(record: Dict[str, Any]):
    """Safely persist span record via SupabaseService."""
    try:
        from app.services.supabase_service import SupabaseService
        service = SupabaseService()
        service.save_inference_log(record)
    except Exception as e:
        logger.debug(f"Telemetry persistence warning: {e}")


class InferenceTelemetryTracker:
    """
    Central singleton coordinator for capturing, aggregating, and querying inference telemetry.
    """
    _instance: Optional[InferenceTelemetryTracker] = None

    @classmethod
    def get_instance(cls) -> InferenceTelemetryTracker:
        if cls._instance is None:
            cls._instance = InferenceTelemetryTracker()
        return cls._instance

    @contextmanager
    def track(
        self,
        inference_type: str,
        model: str,
        task: str,
        model_version: str = "v1.0",
        input_size: int = 0,
        memory_device: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[InferenceSpan, None, None]:
        """Context manager to measure latency, sizes, errors, and metadata around an inference operation."""
        span = InferenceSpan(
            inference_type=inference_type,
            model=model,
            task=task,
            model_version=model_version,
            input_size=input_size,
            memory_device=memory_device,
            metadata=metadata,
        )
        try:
            yield span
        except Exception as e:
            span.record_error(e)
            raise
        finally:
            span.close()

    def record_inference(
        self,
        inference_type: str,
        model: str,
        task: str,
        latency_ms: float,
        success: bool = True,
        status: str = "SUCCESS",
        model_version: str = "v1.0",
        error_message: Optional[str] = None,
        input_size: int = 0,
        output_size: Optional[int] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        memory_device: Optional[str] = None,
        fallback_usage: bool = False,
        retry_count: int = 0,
        cost_estimate_usd: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an already-completed inference event."""
        cost = cost_estimate_usd
        if cost is None:
            cost = _estimate_token_cost(model, input_tokens, output_tokens)

        device = memory_device or _detect_default_device(inference_type)
        record = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "inference_type": inference_type.lower(),
            "model": model,
            "model_version": model_version,
            "task": task,
            "latency_ms": round(latency_ms, 2),
            "success": success,
            "status": status,
            "error_message": error_message,
            "input_size": input_size,
            "output_size": output_size,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "memory_device": device,
            "fallback_usage": fallback_usage,
            "retry_count": retry_count,
            "cost_estimate_usd": cost,
            "metadata": metadata or {},
        }

        try:
            telemetry_logger.info(json.dumps(record, default=str))
        except Exception:
            pass

        _persist_span_record(record)
        return record


# Global singleton accessors
def get_telemetry_tracker() -> InferenceTelemetryTracker:
    return InferenceTelemetryTracker.get_instance()


def track_inference(
    inference_type: str,
    model: str,
    task: str,
    model_version: str = "v1.0",
    memory_device: Optional[str] = None,
) -> Callable:
    """Decorator for profiling inference functions."""
    def decorator(fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            tracker = get_telemetry_tracker()
            with tracker.track(
                inference_type=inference_type,
                model=model,
                task=task,
                model_version=model_version,
                memory_device=memory_device,
            ) as span:
                result = fn(*args, **kwargs)
                span.set_output(result)
                return result
        return wrapper
    return decorator
