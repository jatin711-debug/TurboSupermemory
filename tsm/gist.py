"""Gist summarizers for compress-instead-of-delete.

When memory is over budget, eviction victims can be reduced to a short gist
instead of being dropped outright (forget detail, keep gist). A summarizer is
any callable mapping a list of evicted fact texts to one gist string; pass an
instance as ``Memory(gist_summarizer=...)``::

    from tsm import Memory
    from tsm.gist import OpenAIGistSummarizer

    mem = Memory("./db", max_records=500, gist_summarizer=OpenAIGistSummarizer())

Input facts may carry a ``[user] `` / ``[assistant] `` / ``[system] `` prefix;
user-asserted facts are prioritized and generic assistant advice is dropped.
"""

import logging
import os
from typing import Optional, Sequence

from ._retry import call_with_retries
from .budget import fit_complete_facts_to_budget

logger = logging.getLogger("tsm.gist")

GIST_SYSTEM_PROMPT = (
    "Compress memory facts into complete, atomic, queryable facts. Input facts may "
    "start with [user], [assistant], or [system]. Prioritize facts asserted by the "
    "user, especially names, numbers, dates, purchases, preferences, locations, "
    "changes, and counts. Drop generic assistant advice unless the user explicitly "
    "adopted it. Output only terse bullet facts, one complete fact per line. Never "
    "merge separate countable events and never end with an incomplete fact."
)

_ROLE_PREFIXES = ("[user] ", "[assistant] ", "[system] ")


def strip_role(text: str) -> str:
    """Drop a leading ``[role] `` label."""
    for prefix in _ROLE_PREFIXES:
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


def single_fact(text: str) -> str:
    """Gist of a one-fact chunk: the fact itself, or nothing for assistant chatter."""
    if text.lower().startswith("[assistant] "):
        return ""
    return strip_role(text)


class OpenAIGistSummarizer:
    """LLM gist summarizer over an OpenAI-compatible chat API.

    Reads the key from ``OPENAI_API_KEY`` unless a preconfigured ``client``
    (any OpenAI-compatible endpoint) is passed. ``calls`` / ``input_tokens`` /
    ``output_tokens`` accumulate usage for cost accounting.
    """

    def __init__(self, model: str = "gpt-4.1-nano", max_retries: int = 6,
                 request_timeout: float = 30.0, max_tokens: int = 120,
                 client=None, extra_body: Optional[dict] = None):
        if client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. OpenAIGistSummarizer needs it; "
                    "pass client=... for another OpenAI-compatible endpoint, or "
                    "use ExtractiveGistSummarizer for a model-free gist."
                )
            from openai import OpenAI

            client = OpenAI(timeout=request_timeout)
        self._client = client
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.calls = 0
        self.extra_body = extra_body
        self.input_tokens = 0
        self.output_tokens = 0

    def __call__(self, texts: Sequence[str]) -> str:
        return self.summarize(texts)

    def summarize(self, texts: Sequence[str], max_tokens: Optional[int] = None) -> str:
        facts = [t for t in texts if t and t.strip()]
        if not facts:
            return ""
        if len(facts) == 1:
            return single_fact(facts[0])
        joined = "\n".join(f"- {t}" for t in facts)

        def request():
            self.calls += 1
            kwargs = {
                "model": self.model,
                "messages": [{"role": "system", "content": GIST_SYSTEM_PROMPT},
                             {"role": "user", "content": f"Facts:\n{joined}\n\nGist:"}],
                "temperature": 0.0,
                "max_tokens": max_tokens or self.max_tokens,
            }
            if self.extra_body:
                kwargs["extra_body"] = self.extra_body
            r = self._client.chat.completions.create(
                **kwargs,
            )
            if r.usage:
                self.input_tokens += r.usage.prompt_tokens or 0
                self.output_tokens += r.usage.completion_tokens or 0
            content = (r.choices[0].message.content or "").strip()
            if getattr(r.choices[0], "finish_reason", None) == "length":
                # Cut off mid-fact: keep only the complete lines.
                lines = content.splitlines()
                content = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""
            return content

        return call_with_retries(request, "gist summarization", self.max_retries, logger)


class ExtractiveGistSummarizer:
    """Model-free gist: keep as many complete facts as fit the token budget,
    user facts first, assistant facts dropped. No API key, no network."""

    model = "extractive"
    calls = 0

    def __init__(self, max_tokens: int = 120):
        self.max_tokens = max_tokens

    def __call__(self, texts: Sequence[str]) -> str:
        return self.summarize(texts)

    def summarize(self, texts: Sequence[str], max_tokens: Optional[int] = None) -> str:
        facts = [text.strip() for text in texts if text and text.strip()]
        facts = [fact for fact in facts if not fact.lower().startswith("[assistant] ")]
        facts.sort(key=lambda text: 0 if text.lower().startswith("[user] ") else 1)
        joined = "\n".join(f"- {strip_role(fact)}" for fact in facts)
        return fit_complete_facts_to_budget(joined, max_tokens or self.max_tokens)
