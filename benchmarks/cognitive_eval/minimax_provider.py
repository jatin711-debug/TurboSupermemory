"""MiniMax-M3 provider shim for Mem0's OpenAI-compatible LLM wrapper."""

from mem0.llms.openai import OpenAILLM


class MiniMaxMem0LLM(OpenAILLM):
    """Force concise final output so Mem0 never parses reasoning as JSON."""

    def generate_response(
        self,
        messages,
        response_format=None,
        tools=None,
        tool_choice="auto",
        **kwargs,
    ):
        extra_body = dict(kwargs.pop("extra_body", {}) or {})
        extra_body.update(
            {
                "thinking": {"type": "disabled"},
                "reasoning_split": True,
            }
        )
        return super().generate_response(
            messages,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            extra_body=extra_body,
            **kwargs,
        )
