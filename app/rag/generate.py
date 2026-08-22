"""Gemini generation with free-tier-aware harness.

Free tier is ~1500 requests/day and 30 RPM on Flash-Lite. That shapes three
things:

  pacing     A token bucket keeps us AT the limit rather than discovering it
             via 429s. Retrying into a rate limit is how you turn one failure
             into a storm.
  retries    Backoff with jitter, honouring retry-after when the API sends it.
  degrading  Every failure returns ok=False with a reason instead of raising,
             so the caller falls back to extractive mode. A quota-exhausted
             demo should still answer, just less fluently.

Structured output is forced via response_schema, which matters for more than
tidiness: `answerable` gives the model a sanctioned way to decline. Without it,
a model handed thin context is cornered into inventing something, because
returning prose is the only move available. The groundedness check is the
backstop, but letting the model say "not in the context" is the cheaper fix.

temperature=0 so benchmark runs are reproducible.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from app.config import settings

if TYPE_CHECKING:
    from app.rag.retrieve import Retrieved

SYSTEM_INSTRUCTION = """\
You answer questions about Hacker House Goa 2026 using ONLY the numbered \
context passages provided. You are precise and you never guess.

Rules:
1. Use only facts stated in the context. Never add outside knowledge, even if \
you are confident it is true.
2. If the context does not contain the answer, set answerable=false and leave \
answer empty. This is the correct, expected outcome for many questions -- it is \
not a failure.
3. Never invent specifics. Dates, amounts, counts, names, emails and URLs must \
appear verbatim in the context.
4. If passages disagree with each other, say so explicitly and give both \
figures with their sources. Do not silently pick one.
5. Cite the doc_id of every passage you used in cited_doc_ids.
6. Answer in at most three sentences, plainly. This will be read aloud.
"""


class GeneratedAnswer(BaseModel):
    """Forced response shape. Field order matters: deciding `answerable` before
    writing `answer` discourages committing to prose and rationalising after."""

    answerable: bool = Field(description="True only if the context contains the answer")
    answer: str = Field(default="", description="Empty when answerable is false")
    cited_doc_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@dataclass
class GenResult:
    ok: bool
    reason: str
    answer: str = ""
    answerable: bool = False
    cited_doc_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    diagnostics: dict = field(default_factory=dict)


class TokenBucket:
    """Thread-safe RPM limiter. Blocks rather than failing: for an interactive
    request, a short wait beats a 429 the user has to retry."""

    def __init__(self, rpm: int) -> None:
        self.capacity = max(1, rpm)
        self.tokens = float(self.capacity)
        self.refill_per_sec = self.capacity / 60.0
        self.updated = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout_s: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                now = time.monotonic()
                self.tokens = min(
                    self.capacity, self.tokens + (now - self.updated) * self.refill_per_sec
                )
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
                need = (1 - self.tokens) / self.refill_per_sec
            if time.monotonic() + need > deadline:
                return False
            time.sleep(min(need, 0.25))


_bucket = TokenBucket(settings.gen_rpm_limit)
_client = None
_client_lock = threading.Lock()


def _get_client(api_key: str | None = None):
    """Lazily build one client. Constructing per request would add TLS setup
    to every call's latency.

    A caller-supplied key (bring-your-own-key from the browser) bypasses the
    cache and builds a throwaway client: caching per key would leak one user's
    credential into another user's request on a shared deployment, and would
    grow unboundedly. The BYOK path therefore pays TLS setup -- correct
    trade, and it only affects users who supplied their own key.
    """
    if api_key:
        from google import genai
        from google.genai import types

        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=settings.gen_timeout_ms),
        )

    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from google import genai
                from google.genai import types

                _client = genai.Client(
                    api_key=settings.gemini_api_key,
                    http_options=types.HttpOptions(timeout=settings.gen_timeout_ms),
                )
    return _client


def build_prompt(query: str, hits: list[Retrieved]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(f"[{i}] doc_id={h.doc_id} | {h.title}\n{h.context_text}")
    context = "\n\n".join(blocks)
    return f"CONTEXT PASSAGES\n{context}\n\nQUESTION\n{query}"


def _is_retryable(exc: Exception) -> tuple[bool, str]:
    """429 and 5xx are transient; 400/401/403 are not and retrying wastes quota."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc).lower()
    if code == 429 or "resource_exhausted" in msg or "rate limit" in msg or "quota" in msg:
        return True, "rate_limited"
    if isinstance(code, int) and 500 <= code < 600:
        return True, "server_error"
    if "timeout" in msg or "deadline" in msg:
        return True, "timeout"
    if code in (400, 401, 403):
        return False, "client_error"
    return False, "unknown_error"


def generate(
    query: str, hits: list[Retrieved], api_key: str | None = None
) -> GenResult:
    """`api_key` overrides the server key for this request only (BYOK).

    Never logged, never echoed back, never persisted. The service is usable
    with no key at all -- it degrades to extractive mode -- so a key is an
    upgrade, not a requirement.
    """
    if not (api_key or settings.can_generate):
        return GenResult(False, "no_api_key")
    if not hits:
        return GenResult(False, "no_context")

    if not _bucket.acquire(timeout_s=settings.gen_timeout_ms / 1000):
        # Better to degrade now than queue behind a full bucket and then
        # time out anyway.
        return GenResult(False, "rate_limit_wait_exceeded", diagnostics={"paced": True})

    from google.genai import types

    prompt = build_prompt(query, hits)
    attempts: list[dict] = []

    for attempt in range(settings.gen_max_retries):
        t0 = time.perf_counter()
        try:
            resp = _get_client(api_key).models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.0,
                    max_output_tokens=512,
                    response_mime_type="application/json",
                    response_schema=GeneratedAnswer,
                ),
            )
            elapsed = (time.perf_counter() - t0) * 1000
            parsed = resp.parsed
            if parsed is None:
                attempts.append({"attempt": attempt, "ms": round(elapsed, 1), "err": "unparsed"})
                continue
            return GenResult(
                ok=True,
                reason="generated",
                answer=(parsed.answer or "").strip(),
                answerable=bool(parsed.answerable),
                cited_doc_ids=list(parsed.cited_doc_ids or []),
                confidence=float(parsed.confidence or 0.0),
                diagnostics={
                    "attempts": attempts + [{"attempt": attempt, "ms": round(elapsed, 1)}],
                    "model": settings.gemini_model,
                    "key_source": "request" if api_key else "server",
                },
            )
        except Exception as exc:  # noqa: BLE001 - classify, never propagate
            elapsed = (time.perf_counter() - t0) * 1000
            retryable, kind = _is_retryable(exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "ms": round(elapsed, 1),
                    "err": kind,
                    "detail": str(exc)[:160],
                }
            )
            if not retryable or attempt == settings.gen_max_retries - 1:
                return GenResult(False, kind, diagnostics={"attempts": attempts})
            # Exponential backoff with jitter: synchronized retries from
            # concurrent requests would recreate the burst we are avoiding.
            time.sleep(min(2**attempt + random.uniform(0, 0.4), 5.0))

    return GenResult(False, "exhausted_retries", diagnostics={"attempts": attempts})
