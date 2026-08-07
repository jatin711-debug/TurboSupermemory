"""MiniMax-M3 atomic-fact extraction through OpenAI-compatible chat."""

import json
import logging
import os
import pathlib
import time
from typing import List, Optional

from .openai_extractor import OpenAIExtractor, _SYS

logger = logging.getLogger("cognitive_eval.extraction.minimax")


class MiniMaxExtractor(OpenAIExtractor):
    def __init__(
        self,
        model="MiniMax-M3",
        base_url="https://api.minimax.io/v1",
        max_retries=6,
        request_timeout=60.0,
        cache_dir=None,
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
        self.base_url = base_url
        self.max_retries = max_retries
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        cache_root = pathlib.Path(cache_dir) if cache_dir else pathlib.Path(__file__).parent / "_cache"
        cache_root.mkdir(exist_ok=True)
        safe_model = model.replace("/", "_").replace(":", "_")
        self._cache_path = cache_root / f"extract_minimax_{safe_model}.json"
        self._cache = {}
        self._dirty = 0
        if self._cache_path.exists():
            try:
                self._cache = json.loads(self._cache_path.read_text(encoding="utf-8"))
                logger.info("Loaded %d cached MiniMax extractions", len(self._cache))
            except (OSError, json.JSONDecodeError):
                self._cache = {}

    def _chat_json(self, message: str, context: Optional[List[str]]) -> str:
        context_text = ""
        if context:
            context_text = "Recent context:\n" + "\n".join(
                f"- {item}" for item in context[-3:]
            ) + "\n\n"
        user = f'{context_text}Message:\n"{message}"'
        for attempt in range(self.max_retries):
            try:
                self.calls += 1
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": _SYS},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=400,
                    response_format={"type": "json_object"},
                    extra_body={
                        "thinking": {"type": "disabled"},
                        "reasoning_split": True,
                    },
                )
                usage = response.usage
                if usage:
                    self.input_tokens += usage.prompt_tokens or 0
                    self.output_tokens += usage.completion_tokens or 0
                return response.choices[0].message.content or "{}"
            except Exception as exc:  # noqa: BLE001 - provider retries are intentional
                wait = min(5.0 * (2**attempt), 120.0)
                logger.warning(
                    "MiniMax extract failed (attempt %d/%d): %s; retry in %.0fs",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError("MiniMax extraction failed after retries")
