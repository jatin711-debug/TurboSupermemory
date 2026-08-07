"""Gold-standard LLM-judge scoring for LongMemEval (W6).

Everything else in this suite is a RETRIEVAL proxy (does a top-k memory contain
the gold token?). The real benchmark metric is: retrieve -> an LLM answers the
question using ONLY those memories -> an LLM judges the answer against gold.

`create_judge` selects a backend with the same ollama-else-openai flexibility as
extraction:
  - "auto"   : Ollama if its server is reachable, else OpenAI (needs
               OPENAI_API_KEY), else a clear error.
  - "ollama" / "openai" : force a backend.
"""

import logging
import os

from .openai_judge import OpenAIJudge

logger = logging.getLogger("cognitive_eval.judge")

__all__ = ["OpenAIJudge", "OllamaJudge", "MiniMaxJudge", "create_judge"]


def create_judge(
    name="auto",
    ollama_model="qwen2.5:3b",
    openai_model="gpt-4o-mini",
    minimax_model="MiniMax-M3",
):
    if name == "openai":
        return OpenAIJudge(model=openai_model)
    if name == "ollama":
        from .ollama_judge import OllamaJudge
        return OllamaJudge(model=ollama_model)
    if name == "minimax":
        from .minimax_judge import MiniMaxJudge

        return MiniMaxJudge(model=minimax_model)
    if name == "auto":
        try:
            from .ollama_judge import OllamaJudge
            j = OllamaJudge(model=ollama_model)
            if j.health_check():
                logger.info("auto judge -> Ollama (%s)", ollama_model)
                return j
        except Exception as e:  # noqa: BLE001 — ollama missing/unavailable is expected
            logger.info("Ollama judge unavailable (%s); trying OpenAI.", e)
        if os.environ.get("OPENAI_API_KEY"):
            logger.info("auto judge -> OpenAI (%s)", openai_model)
            return OpenAIJudge(model=openai_model)
        raise RuntimeError(
            "auto judge: no reachable Ollama server and no OPENAI_API_KEY. "
            "Start Ollama (`ollama serve`) or set OPENAI_API_KEY."
        )
    raise ValueError(f"unknown judge '{name}'")


# Backwards/forwards helper name.
from .ollama_judge import OllamaJudge  # noqa: E402
