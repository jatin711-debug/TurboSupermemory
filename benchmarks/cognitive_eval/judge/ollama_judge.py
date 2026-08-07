"""Ollama-backed answer + judge for the gold-standard LongMemEval metric (W6).

Same contract as OpenAIJudge (answer / judge / answer_and_judge) but via a local
Ollama model — free and private. Preferred when an Ollama server is reachable;
falls back to OpenAI otherwise (see judge.create_judge).
"""

import logging
import threading

logger = logging.getLogger("cognitive_eval.judge.ollama")

_ANSWER_SYS = (
    "You answer a question using ONLY the numbered memory snippets provided (facts "
    "a user told an assistant over time). Give the shortest possible answer (a name, "
    "value, date, or phrase). If the answer is not in the memories, reply exactly: NO ANSWER."
)
_JUDGE_SYS = (
    "You grade whether a predicted answer matches the gold answer for a question. They "
    "match if they refer to the same fact, even if worded differently or with extra "
    "detail. Reply with exactly one word: CORRECT or INCORRECT."
)


class OllamaJudge:
    def __init__(self, model="qwen2.5:3b", host="http://localhost:11434"):
        import ollama

        self._client = ollama.Client(host=host)
        self.model = model
        self.calls = 0
        self._calls_lock = threading.Lock()

    def _chat(self, system, user):
        with self._calls_lock:
            self.calls += 1
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            options={"temperature": 0.0, "num_predict": 64},
        )
        return (resp.message.content or "").strip()

    def answer(self, question, memories):
        if not memories:
            return "NO ANSWER"
        ctx = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(memories) if m)
        return self._chat(_ANSWER_SYS, f"Memories:\n{ctx}\n\nQuestion: {question}\nAnswer:")

    def judge(self, question, gold, prediction):
        if not prediction or prediction.strip().upper().startswith("NO ANSWER"):
            return False
        verdict = self._chat(_JUDGE_SYS,
                             f"Question: {question}\nGold answer: {gold}\n"
                             f"Predicted answer: {prediction}\nGrade:")
        return "CORRECT" in verdict.strip().upper()

    def answer_and_judge(self, question, memories, gold):
        return self.judge(question, gold, self.answer(question, memories))

    def health_check(self):
        try:
            r = self._client.list()
            raw = r.models if hasattr(r, "models") else r.get("models", [])
            names = [getattr(m, "model", None) or (m.get("model") or m.get("name")
                     if isinstance(m, dict) else None) for m in raw]
            return any(n and (n == self.model or n.startswith(self.model)) for n in names)
        except Exception as e:  # noqa: BLE001
            logger.warning("Ollama judge health check failed: %s", e)
            return False
