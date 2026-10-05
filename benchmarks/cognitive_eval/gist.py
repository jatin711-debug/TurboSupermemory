"""Gist summarizer for compress-instead-of-delete (B4).

When memory is over budget, eviction victims can be REDUCED to a short gist
rather than deleted outright (rate-distortion: forget detail, keep gist —
"What to Keep, What to Forget", 2026). This makes those gists via the same
OpenAI-compatible client the judge/extractor use. Key from the environment only.

The summarizers themselves are the shipped ones in ``tsm.gist``; this module
adds the harness's key-file lookup and the eval-only providers (MiniMax,
ollama) behind ``create_gister``.
"""

import os

from tsm.gist import GIST_SYSTEM_PROMPT as _SYS
from tsm.gist import ExtractiveGistSummarizer, OpenAIGistSummarizer
from tsm.gist import single_fact as _single_fact


class OpenAIGister(OpenAIGistSummarizer):
    """The shipped OpenAI gist summarizer, with the harness's key-file lookup."""

    def __init__(self, model="gpt-4.1-nano", max_retries=6, request_timeout=30.0, max_tokens=120):
        from ._secrets import ensure_openai_key, key_file_hint
        if not ensure_openai_key():
            raise RuntimeError("No OpenAI key. " + key_file_hint())
        super().__init__(model=model, max_retries=max_retries,
                         request_timeout=request_timeout, max_tokens=max_tokens)


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

        client = OpenAI(
            api_key=os.environ["MINIMAX_API_KEY"],
            base_url=base_url,
            timeout=request_timeout,
        )
        OpenAIGistSummarizer.__init__(
            self,
            model=model,
            max_retries=max_retries,
            request_timeout=request_timeout,
            max_tokens=max_tokens,
            client=client,
            extra_body={
                "thinking": {"type": "disabled"},
                "reasoning_split": True,
            },
        )


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
        class _ExtractiveGister(ExtractiveGistSummarizer):
            model = "extractive-smoke"

        return _ExtractiveGister()
    raise ValueError(f"unknown gister '{name}'")
