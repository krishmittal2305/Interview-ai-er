"""
Inference Service providing structured execution, timing, logging, concurrency control,
and failure mode resilience (timeout, OOM, corrupted artifact, and invalid tokenizer handling).
"""
import time
import logging
import concurrent.futures
from typing import Any, Callable, Dict, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .model_manager import ModelManager
from .model_registry import MLModelRegistry

logger = logging.getLogger(__name__)


class InferenceService:
    _manager = ModelManager()

    @classmethod
    def get_model(cls, model_id: str) -> Any:
        """Retrieve a model via the ModelManager."""
        return cls._manager.get_model(model_id)

    @classmethod
    def execute(
        cls,
        model_id: str,
        inference_func: Callable[[Any], Any],
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes an inference function with the given model.
        Handles concurrency, execution timeouts, out-of-memory, and structured logging.
        """
        metadata = MLModelRegistry.get(model_id)
        if not metadata:
            raise ValueError(f"Model {model_id} not registered.")

        # Concurrency control: acquire the lock for this model
        lock = cls._manager.acquire_lock(model_id)
        start_time = time.perf_counter()
        status = "success"
        error_name = None

        try:
            with lock:
                # Lazy load the model inside the locked section if needed
                model = cls._manager.get_model(model_id)

                if timeout_seconds and timeout_seconds > 0:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(inference_func, model)
                        try:
                            result = future.result(timeout=timeout_seconds)
                        except concurrent.futures.TimeoutError:
                            status = "timeout"
                            error_name = "TimeoutError"
                            raise TimeoutError(f"Inference timed out after {timeout_seconds}s for {model_id}")
                else:
                    result = inference_func(model)

        except MemoryError as e:
            status = "oom"
            error_name = "MemoryError"
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.error(f"OOM during inference for {model_id}: {e}", exc_info=True)
            raise
        except TimeoutError as e:
            status = "timeout"
            error_name = "TimeoutError"
            logger.error(f"Inference timeout for {model_id}: {e}")
            raise
        except Exception as e:
            # Check for torch CUDA OOM
            if HAS_TORCH and hasattr(torch.cuda, "OutOfMemoryError") and isinstance(e, torch.cuda.OutOfMemoryError):
                status = "oom"
                error_name = "CUDAOutOfMemoryError"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                status = "error"
                error_name = type(e).__name__
            logger.error(f"Inference failed for {model_id}: {e}", exc_info=True)
            result = None
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000.0

            # Structured logging
            log_payload = {
                "event": "inference_execution",
                "model_id": model_id,
                "version": metadata.version,
                "device": metadata.device,
                "status": status,
                "duration_ms": round(duration_ms, 2),
                "expected_latency_ms": metadata.expected_latency_ms,
            }
            if error_name:
                log_payload["error_type"] = error_name
            logger.info(log_payload)

            # Record ML inference in unified observability telemetry
            try:
                from app.observability.tracker import get_telemetry_tracker
                get_telemetry_tracker().record_inference(
                    inference_type="ml",
                    model=model_id,
                    model_version=metadata.version,
                    task=metadata.task,
                    latency_ms=round(duration_ms, 2),
                    success=(status == "success"),
                    status=status.upper(),
                    error_message=error_name,
                    input_size=0,
                    output_size=len(str(result)) if result is not None else 0,
                    memory_device=metadata.device,
                    fallback_usage=False,
                    retry_count=0,
                )
            except Exception as tel_err:
                logger.debug(f"ML serving telemetry record failed: {tel_err}")

        return {
            "result": result,
            "metadata": {
                "model_id": model_id,
                "duration_ms": duration_ms,
                "device": metadata.device,
                "status": status,
            },
        }

    @classmethod
    def get_health(cls) -> Dict[str, Any]:
        """Retrieve overall health and loaded status of the serving layer."""
        return cls._manager.get_health()
