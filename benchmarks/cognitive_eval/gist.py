"""Gist summarizer for compress-instead-of-delete (B4).

When memory is over budget, eviction victims can be REDUCED to a short gist
rather than deleted outright (rate-distortion: forget detail, keep gist —
"What to Keep, What to Forget", 2026). This makes those gists via the same
OpenAI-compatible client the judge/extractor use. Key from the environment only.
"""

import logging
import os
import time

logger = logging.getLogger("cognitive_eval.gist")

_SYS = (
    "Compress memory facts into complete, atomic, queryable facts. Input facts may "
    "start with [user], [assistant], or [system]. Prioritize facts asserted by the "
    "user, especially names, numbers, dates, purchases, preferences, locations, "
    "changes, and counts. Drop generic assistant advice unless the user explicitly "
    "adopted it. Output only terse bullet facts, one complete fact per line. Never "
    "merge separate countable events and never end with an incomplete fact."
)


def _strip_role(text):
    for prefix in ("[user] ", "[assistant] ", "[system] "):
        if text.lower().startswith(prefix):
            return text[len(prefix):]
    return text


def _single_fact(text):
    if text.lower().startswith("[assistant] "):
        return ""
    return _strip_role(text)


class OpenAIGister:
    def __init__(self, model="gpt-4.1-nano", max_retries=6, request_timeout=30.0, max_tokens=120):
        from ._secrets import ensure_openai_key, key_file_hint
        if not ensure_openai_key():
            raise RuntimeError("No OpenAI key. " + key_file_hint())
        from openai import OpenAI
        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.calls = 0
        self.extra_body = None
        self.input_tokens = 0
        self.output_tokens = 0

    def summarize(self, texts, max_tokens=None):
        facts = [t for t in texts if t and t.strip()]
        if not facts:
            return ""
        if len(facts) == 1:
            return _single_fact(facts[0])
        joined = "\n".join(f"- {t}" for t in facts)
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": _SYS},
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
                    lines = content.splitlines()
                    content = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""
                return content
            except Exception as e:  # noqa: BLE001
                wait = min(5.0 * (2 ** attempt), 120.0)
                logger.warning("gist failed (attempt %d/%d): %s; retry %.0fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("gist summarization failed after retries")


class MiniMaxGister(OpenAIGister):
    def __init__(
        self,
        model="MiniMax-M3",
        base_url="https://api.minimax.io/v1",
        max_retries=6,
        request_timeout=60.0,
        max_tokens=120,
    ):
        from ._secrets import ensure_minimax_key, minimax_key_file_hint

        if not ensure_minimax_key():
            raise RuntimeError("No MiniMax key. " + minimax_key_file_hint())
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url=base_url,
            timeout=request_timeout,
        )
        self.model = model
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.extra_body = {
            "thinking": {"type": "disabled"},
            "reasoning_split": True,
        }


def create_gister(name="openai", model=None):
    if name == "openai":
        return OpenAIGister(model=model or "gpt-4.1-nano")
    if name == "minimax":
        return MiniMaxGister(model=model or "MiniMax-M3")
    if name == "ollama":
        # Minimal ollama gister mirroring OpenAIGister.
        import ollama
        client = ollama.Client(host="http://localhost:11434")

        class _OllamaGister:
            def __init__(self):
                self.model = model or "qwen2.5:3b"
                self.calls = 0

            def summarize(self, texts, max_tokens=None):
                facts = [t for t in texts if t and t.strip()]
                if len(facts) <= 1:
                    return _single_fact(facts[0]) if facts else ""
                self.calls += 1
                joined = "\n".join(f"- {t}" for t in facts)
                r = client.chat(model=self.model,
                                messages=[{"role": "system", "content": _SYS},
                                          {"role": "user", "content": f"Facts:\n{joined}\n\nGist:"}],
                                options={"temperature": 0.0,
                                         "num_predict": max_tokens or 120})
                content = (r.message.content or "").strip()
                if getattr(r, "done_reason", None) == "length":
                    lines = content.splitlines()
                    content = "\n".join(lines[:-1]).strip() if len(lines) > 1 else ""
                return content
        return _OllamaGister()
    if name == "extractive":
        class _ExtractiveGister:
            model = "extractive-smoke"
            calls = 0

            @staticmethod
            def summarize(texts, max_tokens=None):
                from .budgeting import fit_complete_facts_to_budget

                facts = [text.strip() for text in texts if text and text.strip()]
                facts = [fact for fact in facts if not fact.lower().startswith("[assistant] ")]
                facts.sort(key=lambda text: 0 if text.lower().startswith("[user] ") else 1)
                joined = "\n".join(f"- {_strip_role(fact)}" for fact in facts)
                return fit_complete_facts_to_budget(joined, max_tokens or 120)

        return _ExtractiveGister()
    raise ValueError(f"unknown gister '{name}'")
