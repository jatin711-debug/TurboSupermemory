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
    "You compress a list of a user's remembered facts into a SHORT gist that "
    "preserves the concrete, queryable details (names, numbers, dates, places, "
    "preferences) and drops filler. Write 1-3 terse sentences, no preamble."
)


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

    def summarize(self, texts):
        facts = [t for t in texts if t and t.strip()]
        if not facts:
            return ""
        if len(facts) == 1:
            return facts[0]
        joined = "\n".join(f"- {t}" for t in facts)
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                r = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": _SYS},
                              {"role": "user", "content": f"Facts:\n{joined}\n\nGist:"}],
                    temperature=0.0, max_tokens=self.max_tokens,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001
                wait = min(5.0 * (2 ** attempt), 120.0)
                logger.warning("gist failed (attempt %d/%d): %s; retry %.0fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("gist summarization failed after retries")


def create_gister(name="openai", model=None):
    if name == "openai":
        return OpenAIGister(model=model or "gpt-4.1-nano")
    if name == "ollama":
        # Minimal ollama gister mirroring OpenAIGister.
        import ollama
        client = ollama.Client(host="http://localhost:11434")

        class _OllamaGister:
            def __init__(self):
                self.model = model or "qwen2.5:3b"
                self.calls = 0

            def summarize(self, texts):
                facts = [t for t in texts if t and t.strip()]
                if len(facts) <= 1:
                    return facts[0] if facts else ""
                self.calls += 1
                joined = "\n".join(f"- {t}" for t in facts)
                r = client.chat(model=self.model,
                                messages=[{"role": "system", "content": _SYS},
                                          {"role": "user", "content": f"Facts:\n{joined}\n\nGist:"}],
                                options={"temperature": 0.0, "num_predict": 120})
                return (r.message.content or "").strip()
        return _OllamaGister()
    raise ValueError(f"unknown gister '{name}'")
