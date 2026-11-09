"""Claude (Anthropic API) as reader and judge, with ``OpenAIJudge``'s interface.

``answer`` and ``judge`` take the same arguments and prompts as
``OpenAIJudge``, so ``shipped_stack_eval.py --judge anthropic`` and
``reread.py --provider anthropic`` change only who answers and grades.

Current models take no ``temperature`` (a non-default value is a 400), so a
reading is not repeatable at the API level either: compare arms with several
reads (``reread.py``). The judge runs with thinking off and low effort (a
one-word verdict). The reader thinks (adaptive, ``reader_effort``, by default
the model's own), which counts against ``max_tokens``, so it gets room for
that on top of the reply.

The key comes from ``ANTHROPIC_API_KEY`` or a gitignored ``claude_key.txt``
(``_secrets.ensure_anthropic_key``); it is never logged.
"""

import logging
import threading

logger = logging.getLogger("cognitive_eval.judge.anthropic")

# Room for the reader's thinking on top of its reply.
THINKING_ROOM = 8000


class AnthropicJudge:
    def __init__(self, model="claude-haiku-5-5", max_retries=8, request_timeout=120.0,
                 reader_model=None, reader_effort=None):
        import anthropic

        from cognitive_eval._secrets import ensure_anthropic_key

        if not ensure_anthropic_key():
            raise RuntimeError("no Anthropic key: set ANTHROPIC_API_KEY or create the "
                               "gitignored claude_key.txt at the repo root")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=max_retries, timeout=request_timeout)
        self.model = model
        self.reader_model = reader_model
        self.reader_effort = reader_effort
        self.calls = 0
        self._calls_lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.reader_input_tokens = 0
        self.reader_output_tokens = 0

    def _chat(self, system, user, max_tokens, reader):
        model = (self.reader_model or self.model) if reader else self.model
        kwargs = {"model": model, "system": system,
                  "messages": [{"role": "user", "content": user}]}
        if reader:
            kwargs["max_tokens"] = max_tokens + THINKING_ROOM
            if self.reader_effort:
                kwargs["extra_body"] = {"output_config": {"effort": self.reader_effort}}
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["thinking"] = {"type": "disabled"}
            kwargs["extra_body"] = {"output_config": {"effort": "low"}}
        with self._calls_lock:
            self.calls += 1
        try:
            resp = self._client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as e:
            if _out_of_credit(e):
                from tsm._retry import QuotaExhausted

                raise QuotaExhausted("Anthropic request failed: the account is out of "
                                     "credit") from e
            raise
        with self._calls_lock:
            self.input_tokens += resp.usage.input_tokens or 0
            self.output_tokens += resp.usage.output_tokens or 0
            if reader:
                self.reader_input_tokens += resp.usage.input_tokens or 0
                self.reader_output_tokens += resp.usage.output_tokens or 0
        if resp.stop_reason == "refusal":
            logger.warning("refusal from %s", model)
            return ""
        return "".join(b.text for b in resp.content if b.type == "text").strip()

    def answer(self, question, memories, today=None, reader="plain", turns=None,
               system=None):
        """As ``OpenAIJudge.answer`` (``reader`` "plain" or "notes")."""
        from .openai_judge import _ANSWER_SYS

        if not memories:
            return "NO ANSWER"
        if reader == "notes":
            from tsm.reading import answer_prompt, final_answer

            if turns and len(turns) == len(memories):
                memories = [{"context": m, "turn_index": t} for m, t in zip(memories, turns)]
            system, user = (m["content"] for m in
                            answer_prompt(question, memories, today, system=system))
            return final_answer(self._chat(system, user, 800, reader=True)) or "NO ANSWER"
        ctx = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(memories) if m)
        when = f"Current date: {today}\n" if today else ""
        user = f"Memories:\n{ctx}\n\n{when}Question: {question}\nAnswer:"
        return self._chat(_ANSWER_SYS, user, 100, reader=True)

    def judge(self, question, gold, prediction):
        """True if ``prediction`` matches ``gold``: the OpenAI judge's prompt."""
        from .openai_judge import _JUDGE_SYS

        if not prediction or prediction.strip().upper().startswith("NO ANSWER"):
            return False
        user = (f"Question: {question}\nGold answer: {gold}\n"
                f"Predicted answer: {prediction}\nGrade:")
        verdict = self._chat(_JUDGE_SYS, user, 16, reader=False)
        return verdict.strip().upper().startswith("CORRECT")


def _out_of_credit(error) -> bool:
    body = getattr(error, "body", None)
    text = str(body if body is not None else "").lower()
    return getattr(error, "status_code", None) in (400, 402, 403) and (
        "credit balance" in text or "billing" in text)
