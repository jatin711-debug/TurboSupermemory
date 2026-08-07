"""MiniMax-M3 answerer and judge using concise, thinking-disabled calls."""

import os
import threading

from .openai_judge import OpenAIJudge


class MiniMaxJudge(OpenAIJudge):
    def __init__(
        self,
        model="MiniMax-M3",
        base_url="https://api.minimax.io/v1",
        max_retries=10,
        request_timeout=60.0,
    ):
        from .._secrets import ensure_minimax_key, minimax_key_file_hint

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
        self.calls = 0
        self._calls_lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.extra_body = {
            "thinking": {"type": "disabled"},
            "reasoning_split": True,
        }
