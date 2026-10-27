"""
Groq LLM provider supporting structured JSON and streaming with models like openai/gpt-oss-120b.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, Iterator, Optional, Tuple, Type, TypeVar

try:
    from groq import Groq
except ImportError:
    Groq = None

from pydantic import BaseModel


from app.ai.providers.base import AIProvider

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class GroqProvider(AIProvider):
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        reasoning_effort: str = "medium",
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        if Groq is None:
            raise RuntimeError("The 'groq' package is not installed. Install with: pip install groq")
        self.model_name = model_name or os.getenv("GROQ_MODEL") or "openai/gpt-oss-120b"
        self.reasoning_effort = reasoning_effort
        self.client = Groq(api_key=self.api_key)

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        max_retries: int = 3,
        timeout_seconds: int = 45,
    ) -> Tuple[T, Dict[str, Any]]:
        schema_json = json.dumps(schema.model_json_schema())
        system_message = (
            "You are an authoritative, calibrated technical interview evaluation and assessment engine.\n"
            "You MUST respond ONLY with a valid JSON object matching the following JSON Schema.\n"
            "Do NOT include backticks, markdown code blocks, or conversational preamble.\n"
            f"JSON Schema:\n{schema_json}"
        )

        models_to_try = [self.model_name]
        for fallback in ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        current_model_idx = 0

        for attempt in range(max_retries):
            current_model = models_to_try[current_model_idx]
            try:
                completion = self.client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_completion_tokens=2048,
                    top_p=1,
                    reasoning_effort=self.reasoning_effort,
                    response_format={"type": "json_object"},
                    timeout=timeout_seconds,
                )

                content = completion.choices[0].message.content
                if not content or not content.strip():
                    raise ValueError("Empty response received from Groq")

                cleaned = content.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)

                parsed = schema.model_validate_json(cleaned)
                
                # Extract telemetry
                telemetry = {
                    "input_tokens": completion.usage.prompt_tokens if completion.usage else None,
                    "output_tokens": completion.usage.completion_tokens if completion.usage else None,
                    "retry_count": attempt,
                    "schema_validation": True,
                    "model_used": current_model,
                }
                
                return parsed, telemetry

            except Exception as e:
                err_str = str(e)
                logger.warning("Groq structured attempt %d/%d (model=%s) failed: %s", attempt + 1, max_retries, current_model, e)
                # If model not found or 404, advance to next model immediately
                if ("model_not_found" in err_str or "404" in err_str or "does not exist" in err_str) and (current_model_idx + 1 < len(models_to_try)):
                    current_model_idx += 1
                    logger.info("Switching Groq fallback model to: %s", models_to_try[current_model_idx])
                    continue
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Groq generation failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)

    def stream_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
    ) -> Iterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        models_to_try = [self.model_name, "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
        for m in models_to_try:
            try:
                stream = self.client.chat.completions.create(
                    model=m,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_tokens,
                    top_p=1,
                    reasoning_effort=self.reasoning_effort,
                    stream=True,
                    stop=None,
                )
                for chunk in stream:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        yield token
                return
            except Exception as e:
                err_str = str(e)
                if "model_not_found" in err_str or "404" in err_str or "does not exist" in err_str:
                    logger.warning("Groq stream model %s not available, trying next fallback: %s", m, e)
                    continue
                raise

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "provider": "groq",
            "model_name": self.model_name,
            "reasoning_effort": self.reasoning_effort,
        }
