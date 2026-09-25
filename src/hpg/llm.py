"""Thin Ollama client wrapper that records tokens and timings for every call (plan §5.4)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import ollama


@dataclass
class LLMCall:
    content: str
    tool_calls: list[Any]
    prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    compute_ms: float  # prompt_eval + eval (excludes Ollama per-request scheduling/load overhead)
    load_ms: float
    logprobs: list[Any] | None = None
    message: Any = None


@dataclass
class LLM:
    model: str
    host: str = "http://localhost:11434"
    num_ctx: int = 8192
    temperature: float = 0.0
    keep_alive: str = "30m"
    seed: int = 1234
    _client: ollama.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ollama.Client(host=self.host)

    @classmethod
    def from_config(cls, cfg: dict) -> LLM:
        c = cfg["llm"]
        return cls(
            c["model"], c["host"], c["num_ctx"], c["temperature"], c["keep_alive"], cfg.get("seed", 1234)
        )

    def chat(
        self,
        messages: list[dict],
        *,
        format: dict | None = None,
        tools: list | None = None,
        think: bool = False,
        logprobs: bool = False,
        max_tokens: int | None = None,
    ) -> LLMCall:
        opts: dict[str, Any] = {"num_ctx": self.num_ctx, "temperature": self.temperature, "seed": self.seed}
        if max_tokens:
            opts["num_predict"] = max_tokens
        t0 = time.perf_counter()
        r = self._client.chat(
            model=self.model,
            messages=messages,
            format=format,
            tools=tools,
            think=think,
            options=opts,
            keep_alive=self.keep_alive,
            logprobs=logprobs or None,
            top_logprobs=5 if logprobs else None,
        )
        wall = (time.perf_counter() - t0) * 1000
        ns = lambda x: (x or 0) / 1e6  # noqa: E731
        return LLMCall(
            content=r.message.content or "",
            tool_calls=list(r.message.tool_calls or []),
            prompt_tokens=r.prompt_eval_count or 0,
            completion_tokens=r.eval_count or 0,
            wall_ms=wall,
            compute_ms=ns(r.prompt_eval_duration) + ns(r.eval_duration),
            load_ms=ns(r.load_duration),
            logprobs=list(r.logprobs) if r.logprobs else None,
            message=r.message,
        )
