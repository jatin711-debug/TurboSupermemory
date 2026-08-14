"""OpenAI-backed answer + judge for the gold-standard LongMemEval metric (W6).

Two calls per question, matching the LongMemEval protocol:
  1. answer(question, memories) -> the model answers using ONLY the retrieved
     memories (or "NO ANSWER" if they don't contain it). It never sees the gold.
  2. judge(question, gold, prediction) -> CORRECT / INCORRECT, semantic match.

Keeping them separate preserves integrity (the answerer can't peek at gold). Both
calls are tiny (short max_tokens, temperature 0) so a full subset run costs cents
on gpt-4o-mini. The API key is read from OPENAI_API_KEY in the environment — this
module never receives, stores, or logs the raw key.
"""

import logging
import os
import random
import threading
import time

logger = logging.getLogger("cognitive_eval.judge.openai")


def _retry_wait(exc, attempt):
    """Backoff that RESPECTS the server's rate-limit hint (retry-after header or
    'try again in Xs' message), else jittered exponential. Critical for GPT-4o's
    low TPM ceiling — a fixed exp backoff crashes when the bucket stays saturated."""
    hinted = None
    try:
        ra = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after")
        if ra:
            hinted = float(ra)
    except Exception:  # noqa: BLE001
        hinted = None
    if hinted is None:
        import re
        m = re.search(r"try again in ([\d.]+)s", str(exc))
        if m:
            hinted = float(m.group(1))
    base = hinted if hinted is not None else min(4.0 * (2 ** attempt), 60.0)
    return base + random.uniform(0.5, 3.0)  # jitter to de-sync concurrent workers


_ANSWER_SYS = (
    "You answer a question using ONLY the numbered memory snippets provided. "
    "The memories are facts a user told an assistant over time.\n"
    "- Chronological Context: Snippets may include tags like [Turn N] or [YYYY-MM-DD] indicating "
    "the sequence in which the user spoke (e.g. Turn 1 occurred before Turn 5). "
    "Use these chronological tags to determine sequence, timing, and what occurred first.\n"
    "- Advice/Preference Queries: If the question asks for advice, tips, or reasons, tailor "
    "the response directly around the user's specific past plans, items being replaced or acquired, "
    "preferences, and component upgrades found in the memories.\n"
    "- Output: Give a concise, direct answer. If no relevant information is present in the memories, "
    "reply exactly: NO ANSWER."
)

_JUDGE_SYS = (
    "You grade whether a predicted response matches the gold answer or rubric for a question. "
    "They match if the predicted response addresses the question by incorporating the key facts, "
    "components, chronological sequence, or user preferences/aesthetics specified in the gold answer. "
    "If the predicted response is NO ANSWER, irrelevant, or factual wrong, reply INCORRECT. "
    "Reply with exactly one word: CORRECT or INCORRECT."
)


class OpenAIJudge:
    def __init__(self, model="gpt-4o-mini", max_retries=10, request_timeout=60.0):
        from .._secrets import ensure_openai_key, key_file_hint
        if not ensure_openai_key():
            raise RuntimeError("No OpenAI key. " + key_file_hint())
        from openai import OpenAI

        # The client reads OPENAI_API_KEY from the environment itself; the key is
        # never passed through or logged here.
        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self.max_retries = max_retries
        self.calls = 0
        self._calls_lock = threading.Lock()
        self.extra_body = None
        self.input_tokens = 0
        self.output_tokens = 0

    def _chat(self, system, user, max_tokens):
        for attempt in range(self.max_retries):
            try:
                with self._calls_lock:
                    self.calls += 1
                kwargs = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": system},
                                 {"role": "user", "content": user}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                }
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                resp = self._client.chat.completions.create(**kwargs)
                if resp.usage:
                    with self._calls_lock:
                        self.input_tokens += resp.usage.prompt_tokens or 0
                        self.output_tokens += resp.usage.completion_tokens or 0
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 — transient API errors: backoff + retry
                wait = _retry_wait(e, attempt)
                logger.warning("OpenAI call failed (attempt %d/%d): %s; retrying in %.1fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("OpenAI call failed after retries")

    def answer(self, question, memories):
        """Answer `question` using only `memories` (list of snippet strings)."""
        if not memories:
            return "NO ANSWER"
        ctx = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(memories) if m)
        user = f"Memories:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        return self._chat(_ANSWER_SYS, user, max_tokens=100)

    def judge(self, question, gold, prediction):
        """True if `prediction` matches `gold` for `question`."""
        if not prediction or prediction.strip().upper().startswith("NO ANSWER"):
            return False
        user = (f"Question: {question}\nGold answer: {gold}\n"
                f"Predicted answer: {prediction}\nGrade:")
        verdict = self._chat(_JUDGE_SYS, user, max_tokens=4)
        return verdict.strip().upper().startswith("CORRECT")

    def answer_and_judge(self, question, memories, gold):
        """Convenience: retrieve-grounded answer, then grade it. Returns bool."""
        pred = self.answer(question, memories)
        return self.judge(question, gold, pred)
