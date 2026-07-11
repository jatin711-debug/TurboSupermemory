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
import time

logger = logging.getLogger("cognitive_eval.judge.openai")

_ANSWER_SYS = (
    "You answer a question using ONLY the numbered memory snippets provided. "
    "The memories are facts a user told an assistant over time. Give the shortest "
    "possible answer (a name, value, date, or phrase). If the answer is not present "
    "in the memories, reply exactly: NO ANSWER."
)
_JUDGE_SYS = (
    "You grade whether a predicted answer matches the gold answer for a question. "
    "They match if they refer to the same fact, even if worded differently or with "
    "extra detail. Reply with exactly one word: CORRECT or INCORRECT."
)


class OpenAIJudge:
    def __init__(self, model="gpt-4o-mini", max_retries=4, request_timeout=30.0):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set in the environment. Set it (do not paste "
                "it into chat) and re-run, e.g. PowerShell: $env:OPENAI_API_KEY='sk-...'"
            )
        from openai import OpenAI

        # The client reads OPENAI_API_KEY from the environment itself; the key is
        # never passed through or logged here.
        self._client = OpenAI(timeout=request_timeout)
        self.model = model
        self.max_retries = max_retries
        self.calls = 0

    def _chat(self, system, user, max_tokens):
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 — transient API errors: backoff + retry
                wait = 2.0 * (2 ** attempt)
                logger.warning("OpenAI call failed (attempt %d/%d): %s; retrying in %.0fs",
                               attempt + 1, self.max_retries, e, wait)
                time.sleep(wait)
        raise RuntimeError("OpenAI call failed after retries")

    def answer(self, question, memories):
        """Answer `question` using only `memories` (list of snippet strings)."""
        if not memories:
            return "NO ANSWER"
        ctx = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(memories) if m)
        user = f"Memories:\n{ctx}\n\nQuestion: {question}\nAnswer:"
        return self._chat(_ANSWER_SYS, user, max_tokens=64)

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
