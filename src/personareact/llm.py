"""Chat-model construction, structured output, retries and cost accounting.

Everything that talks to OpenAI goes through here so that token usage, cost and
failures are counted in one place. The original code constructed ``ChatOpenAI``
in three separate modules with three different parameter sets and counted
nothing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from langchain_openai import ChatOpenAI

from .config import ModelConfig

T = TypeVar("T")


class LLMError(RuntimeError):
    """Raised when a call keeps failing after the configured retries."""


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    calls: int = 0
    retries: int = 0

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.usd += other.usd
        self.calls += other.calls
        self.retries += other.retries

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "usd": round(self.usd, 6),
            "calls": self.calls,
            "retries": self.retries,
        }


@dataclass
class LLMClient:
    """A model plus its accounting.

    ``structured`` uses ``include_raw=True`` so the underlying ``AIMessage`` --
    and therefore ``usage_metadata`` -- survives the structured-output wrapper.
    Without it the token counts are simply unavailable, which is why the
    original implementation could not report cost at all.
    """

    config: ModelConfig
    api_key: str | None = None
    max_retries: int = 3
    retry_base_delay: float = 1.5
    usage: Usage = field(default_factory=Usage)
    _client: ChatOpenAI | None = field(default=None, repr=False)

    def _build(self) -> ChatOpenAI:
        if self._client is None:
            kwargs: dict[str, Any] = {"model": self.config.model}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            kwargs.update(self.config.sampling_kwargs())
            self._client = ChatOpenAI(**kwargs)
        return self._client

    @property
    def raw(self) -> ChatOpenAI:
        return self._build()

    def _record(self, message: Any) -> None:
        meta = getattr(message, "usage_metadata", None) or {}
        inp = int(meta.get("input_tokens", 0) or 0)
        out = int(meta.get("output_tokens", 0) or 0)
        self.usage.input_tokens += inp
        self.usage.output_tokens += out
        self.usage.usd += (
            inp * self.config.usd_per_1m_input + out * self.config.usd_per_1m_output
        ) / 1_000_000
        self.usage.calls += 1

    def structured(self, schema: type[T] | dict[str, Any], prompt: str) -> T:
        """Invoke with a structured-output schema, retrying transient failures.

        Retries also cover the *shape* of the response, not just transport
        errors. In the original code ``result['thought']`` raised ``KeyError``
        whenever the model omitted the key, the exception was caught at the top
        level, and the string ``[ERRO: 'thought']`` was written into the answer
        field and later graded by the judge as if it were an answer.
        """
        model = self._build().with_structured_output(schema, include_raw=True)
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                envelope = model.invoke(prompt)
            except Exception as exc:  # transport, rate limit, timeout
                last_error = exc
                self.usage.retries += 1
                time.sleep(self.retry_base_delay * (2**attempt))
                continue

            raw = envelope.get("raw") if isinstance(envelope, dict) else None
            if raw is not None:
                self._record(raw)

            parsing_error = envelope.get("parsing_error") if isinstance(envelope, dict) else None
            parsed = envelope.get("parsed") if isinstance(envelope, dict) else envelope

            if parsing_error is not None or parsed is None:
                last_error = LLMError(f"structured output failed to parse: {parsing_error}")
                self.usage.retries += 1
                time.sleep(self.retry_base_delay * (2**attempt))
                continue

            return parsed  # type: ignore[return-value]

        raise LLMError(
            f"{self.config.model}: {self.max_retries} attempts failed ({last_error})"
        ) from last_error


def build_client(config: ModelConfig, api_key: str | None = None, **kwargs: Any) -> LLMClient:
    return LLMClient(config=config, api_key=api_key, **kwargs)
