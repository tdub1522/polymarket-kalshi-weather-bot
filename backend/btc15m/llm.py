"""Anthropic client with fast/smart model tiers and JSON tool-call helper.

Mirrors lib/llm-client.ts from sentient-market-reader, but Anthropic-only.
Falls back to a deterministic stub if no API key is configured so the
pipeline still runs end-to-end in dev / paper mode.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("trading_bot")

# Model routing — matches the repo's tier mapping for AI_PROVIDER=anthropic.
# Smart tier handles decomposition + synthesis; fast tier handles classification.
SMART_MODEL = "claude-sonnet-4-5-20250929"   # latest available Sonnet
FAST_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class LlmCall:
    """Result of a single LLM call. Used for logging + cost tracking."""
    text: str
    model: str
    tier: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    parsed_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class Llm:
    """Minimal async wrapper around the Anthropic SDK with tier routing."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            from backend.config import settings
            self.api_key = getattr(settings, "ANTHROPIC_API_KEY", None)
        if not self.api_key:
            return None
        try:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            logger.error("anthropic package not installed — pipeline will use deterministic fallbacks")
            return None
        return self._client

    async def complete(
        self,
        prompt: str,
        *,
        tier: str = "fast",
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LlmCall:
        """Plain text completion. Tier picks the model (fast | smart)."""
        model = SMART_MODEL if tier == "smart" else FAST_MODEL
        start = time.monotonic()

        client = self._get_client()
        if client is None:
            # Deterministic fallback — caller must be tolerant of generic answers.
            return LlmCall(
                text="",
                model=model,
                tier=tier,
                latency_ms=0.0,
                error="no_api_key",
            )

        try:
            kwargs: Dict[str, Any] = dict(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            if system:
                kwargs["system"] = system
            msg = await client.messages.create(**kwargs)
            text = "".join(getattr(b, "text", "") for b in msg.content)
            return LlmCall(
                text=text,
                model=model,
                tier=tier,
                latency_ms=(time.monotonic() - start) * 1000,
                input_tokens=getattr(msg.usage, "input_tokens", 0),
                output_tokens=getattr(msg.usage, "output_tokens", 0),
            )
        except Exception as exc:
            logger.warning(f"LLM call failed ({tier}): {exc}")
            return LlmCall(
                text="",
                model=model,
                tier=tier,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )

    async def tool_call_json(
        self,
        prompt: str,
        *,
        schema: Dict[str, Any],
        tier: str = "smart",
        system: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.2,
    ) -> LlmCall:
        """Structured-output call — the model is told to return ONLY a JSON object
        matching ``schema``. We don't use Anthropic's tools API formally because we
        only need a single structured result; this keeps the surface area small.
        """
        model = SMART_MODEL if tier == "smart" else FAST_MODEL
        sys_msg = (system or "") + (
            "\n\nYou MUST respond with a single JSON object that conforms to this schema: "
            + json.dumps(schema)
            + "\nReturn JSON only — no markdown, no commentary."
        )
        start = time.monotonic()
        client = self._get_client()
        if client is None:
            return LlmCall(text="", model=model, tier=tier, latency_ms=0.0, error="no_api_key")

        try:
            msg = await client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=sys_msg,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in msg.content).strip()

            # Strip ```json fences if the model added them anyway.
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:]
                text = text.strip()

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Last-ditch: extract first {...} block.
                start_brace = text.find("{")
                end_brace = text.rfind("}")
                parsed = json.loads(text[start_brace:end_brace + 1]) if start_brace != -1 else None

            return LlmCall(
                text=text,
                model=model,
                tier=tier,
                latency_ms=(time.monotonic() - start) * 1000,
                input_tokens=getattr(msg.usage, "input_tokens", 0),
                output_tokens=getattr(msg.usage, "output_tokens", 0),
                parsed_json=parsed,
            )
        except Exception as exc:
            logger.warning(f"LLM tool_call failed ({tier}): {exc}")
            return LlmCall(
                text="",
                model=model,
                tier=tier,
                latency_ms=(time.monotonic() - start) * 1000,
                error=str(exc),
            )
