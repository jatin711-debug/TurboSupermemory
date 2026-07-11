"""Gold-standard LLM-judge scoring for LongMemEval (W6).

Everything else in this suite is a RETRIEVAL proxy (does a top-k memory contain
the gold token?). The real benchmark metric is: retrieve -> an LLM answers the
question using ONLY those memories -> an LLM judges the answer against gold. This
package provides that judge over the OpenAI API (any provider with the same
chat-completions shape works). The API key is read from the environment
(OPENAI_API_KEY) and never handled in code or logs.
"""

from .openai_judge import OpenAIJudge

__all__ = ["OpenAIJudge"]
