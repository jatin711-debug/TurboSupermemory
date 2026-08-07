#!/usr/bin/env python3
"""Probe MiniMax-M3 for TSM extraction, gisting, answers, and Mem0.

The probe uses MiniMax's OpenAI-compatible Chat Completions endpoint with
thinking disabled for short memory operations. It also tests direct function
calling and stock Mem0 ingestion/update/search. The API key is loaded only from
MINIMAX_API_KEY or a gitignored key file and is never printed.

Example:
    python benchmarks/cognitive_eval/probe_minimax_compat.py
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

DEFAULT_BASE_URL = "https://api.minimax.io/v1"
DEFAULT_MODEL = "MiniMax-M3"
DEFAULT_EMBEDDING_MODEL = "qllama/bge-large-en-v1.5:f16"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_FILES = (
    REPO_ROOT / "minimax_key.txt",
    REPO_ROOT / ".minimax_key",
    Path.home() / ".minimax_key",
)


def load_api_key():
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value.upper().startswith("MINIMAX_API_KEY"):
            value = value.split("=", 1)[-1].strip().strip('"').strip("'")
        if value:
            os.environ["MINIMAX_API_KEY"] = value
            return value
    return None


def extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
        raise


def usage_dict(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


class Probe:
    def __init__(self, client, model, timeout, temperature):
        self.client = client
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.results = []

    def record(self, name, passed, detail, elapsed=0.0, usage=None, required=True):
        result = {
            "name": name,
            "passed": bool(passed),
            "required": bool(required),
            "detail": str(detail)[:500],
            "elapsed_seconds": round(elapsed, 3),
            "usage": usage or {},
        }
        self.results.append(result)
        status = "PASS" if passed else ("FAIL" if required else "WARN")
        print(f"[{status}] {name}: {result['detail']}")

    def run_case(self, name, callback, required=True):
        try:
            passed, detail, elapsed, usage = callback()
            self.record(name, passed, detail, elapsed, usage, required)
        except Exception as exc:  # noqa: BLE001 - report capabilities independently
            self.record(name, False, f"{type(exc).__name__}: {exc}", required=required)

    def chat(self, system, user, max_tokens=128, response_format=None):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout,
            "extra_body": {
                "thinking": {"type": "disabled"},
                "reasoning_split": True,
            },
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        content = (choice.message.content or "").strip()
        if not content:
            reasoning = getattr(choice.message, "reasoning_content", "") or ""
            raise RuntimeError(
                "empty final content "
                f"(finish_reason={choice.finish_reason!r}, reasoning_chars={len(reasoning)})"
            )
        return content, elapsed, usage_dict(response)


def run_tool_probe(client, model, timeout, temperature):
    tool = {
        "type": "function",
        "function": {
            "name": "record_memory_actions",
            "description": "Record memory additions, updates, and deletions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "event": {"type": "string", "enum": ["ADD", "UPDATE", "DELETE"]},
                                "memory": {"type": "string"},
                            },
                            "required": ["event", "memory"],
                        },
                    }
                },
                "required": ["actions"],
            },
        },
    }
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Use the provided tool to update durable user memories.",
            },
            {
                "role": "user",
                "content": (
                    "Existing memory: The user lives in Berlin.\n"
                    "New statement: I moved to Lisbon and no longer live in Berlin.\n"
                    "Record the required memory action."
                ),
            },
        ],
        tools=[tool],
        tool_choice="required",
        temperature=temperature,
        max_tokens=512,
        timeout=timeout,
        extra_body={"thinking": {"type": "disabled"}, "reasoning_split": True},
    )
    elapsed = time.perf_counter() - started
    tool_calls = response.choices[0].message.tool_calls or []
    arguments = [extract_json(call.function.arguments) for call in tool_calls]
    serialized = json.dumps(arguments).lower()
    passed = bool(arguments) and "lisbon" in serialized and (
        "update" in serialized or "add" in serialized
    )
    return passed, f"tool_calls={len(tool_calls)}, arguments={arguments}", elapsed, usage_dict(response)


def run_mem0_probe(
    api_key,
    base_url,
    model,
    timeout,
    temperature,
    embedding_model,
    embedding_dims,
    ollama_base_url,
):
    from mem0 import Memory
    from mem0.llms.openai import OpenAIConfig
    from mem0.utils.factory import LlmFactory

    LlmFactory.register_provider(
        "openai",
        "minimax_provider.MiniMaxMem0LLM",
        OpenAIConfig,
    )

    temp_dir = tempfile.mkdtemp(prefix="minimax_mem0_probe_")
    try:
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": model,
                    "temperature": temperature,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "max_tokens": 4000,
                },
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": embedding_model,
                    "embedding_dims": embedding_dims,
                    "ollama_base_url": ollama_base_url,
                },
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": os.path.join(temp_dir, "chroma"),
                    "collection_name": "minimax_compat_probe",
                },
            },
            "history_db_path": os.path.join(temp_dir, "history.db"),
        }
        started = time.perf_counter()
        memory = Memory.from_config(config)
        user_id = "minimax-compat-user"
        first = memory.add(
            [{"role": "user", "content": "I live in Berlin and my favorite color is teal."}],
            user_id=user_id,
        )
        second = memory.add(
            [{"role": "user", "content": "I moved to Lisbon and no longer live in Berlin."}],
            user_id=user_id,
        )
        all_memories = memory.get_all(user_id=user_id, limit=100)
        search = memory.search(
            "Where does the user live now?", user_id=user_id, limit=10, rerank=False
        )
        elapsed = time.perf_counter() - started
        all_items = all_memories.get("results", all_memories) if isinstance(all_memories, dict) else all_memories
        search_items = search.get("results", search) if isinstance(search, dict) else search
        texts = [
            item.get("memory", "") if isinstance(item, dict) else str(item)
            for item in list(all_items or []) + list(search_items or [])
        ]
        lisbon_found = any("lisbon" in text.lower() for text in texts)
        passed = bool(first) and bool(second) and lisbon_found
        detail = (
            f"initial_add={bool(first)}, update_add={bool(second)}, "
            f"stored={len(all_items or [])}, retrieved={len(search_items or [])}, "
            f"lisbon_found={lisbon_found}"
        )
        return passed, detail, elapsed, {}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Test MiniMax-M3 for TSM and Mem0 use")
    parser.add_argument("--model", default=os.environ.get("MINIMAX_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--base-url", default=os.environ.get("MINIMAX_BASE_URL", DEFAULT_BASE_URL)
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("MINIMAX_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    )
    parser.add_argument("--embedding-dims", type=int, default=1024)
    parser.add_argument(
        "--ollama-base-url",
        default=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
    )
    parser.add_argument("--skip-mem0", action="store_true")
    parser.add_argument(
        "--mem0-only",
        action="store_true",
        help="run only the end-to-end Mem0 check after direct capabilities have passed",
    )
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    if args.skip_mem0 and args.mem0_only:
        parser.error("--skip-mem0 and --mem0-only cannot be used together")

    api_key = load_api_key()
    if not api_key:
        locations = ", ".join(str(path) for path in KEY_FILES)
        parser.error(f"set MINIMAX_API_KEY or place the key in one of: {locations}")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    probe = Probe(client, args.model, args.timeout, args.temperature)

    def basic_chat():
        text, elapsed, usage = probe.chat(
            "Follow the output instruction exactly.",
            "Reply with exactly MINIMAX_OK and nothing else.",
            max_tokens=32,
        )
        return text == "MINIMAX_OK", f"response={text!r}", elapsed, usage

    if not args.mem0_only:
        probe.run_case("chat_completions", basic_chat)

    def json_mode():
        text, elapsed, usage = probe.chat(
            "Return one JSON object and no prose.",
            'Return {"compatible": true}.',
            max_tokens=64,
            response_format={"type": "json_object"},
        )
        parsed = extract_json(text)
        return parsed.get("compatible") is True, f"parsed={parsed}", elapsed, usage

    if not args.mem0_only:
        probe.run_case("json_response_format", json_mode, required=False)

    def fact_extraction():
        text, elapsed, usage = probe.chat(
            "Extract complete atomic facts. Return only a JSON object with a facts array. "
            "Every facts element must be a self-contained JSON string, never an object.",
            "Message: I moved to Lisbon on March 3, and my favorite color is teal.",
            max_tokens=160,
        )
        facts = extract_json(text).get("facts", [])
        joined = " ".join(str(fact) for fact in facts).lower()
        passed = (
            isinstance(facts, list)
            and bool(facts)
            and all(isinstance(fact, str) for fact in facts)
            and "lisbon" in joined
            and "teal" in joined
        )
        return passed, f"facts={facts}", elapsed, usage

    if not args.mem0_only:
        probe.run_case("tsm_fact_extraction", fact_extraction)

    def gisting():
        text, elapsed, usage = probe.chat(
            "Compress into complete atomic bullet facts using at most 60 words. "
            "Preserve every separate dated event. Drop generic assistant advice unless "
            "the user explicitly adopted it.",
            "Facts:\n- The user visited Dr. Smith on March 3.\n"
            "- The user attended a follow-up with Dr. Thompson on March 20.\n"
            "- The assistant suggested generic relaxation exercises.\n\nGist:",
            max_tokens=100,
        )
        lowered = text.lower()
        estimated_tokens = max(1, len(text) // 4)
        passed = (
            "march 3" in lowered
            and "march 20" in lowered
            and "relaxation" not in lowered
            and estimated_tokens <= 80
        )
        return passed, f"estimated_tokens={estimated_tokens}, gist={text!r}", elapsed, usage

    if not args.mem0_only:
        probe.run_case("tsm_gisting", gisting)

    def answering():
        text, elapsed, usage = probe.chat(
            "Answer using only the supplied memories. Give the shortest possible answer.",
            "Memories:\n1. The user's internet plan provides 500 Mbps.\n"
            "2. The user lives in Lisbon.\n\nQuestion: What is the user's internet speed?",
            max_tokens=32,
        )
        return "500" in text, f"answer={text!r}", elapsed, usage

    if not args.mem0_only:
        probe.run_case("memory_answer_generation", answering)

    def judging():
        text, elapsed, usage = probe.chat(
            "Grade semantic equivalence. Reply exactly CORRECT or INCORRECT.",
            "Question: What rice does the user prefer?\n"
            "Gold: Japanese short-grain rice\nPrediction: short-grain Japanese rice",
            max_tokens=16,
        )
        return text.upper() == "CORRECT", f"verdict={text!r}", elapsed, usage

    if not args.mem0_only:
        probe.run_case("answer_judging", judging)
        probe.run_case(
            "mem0_action_tool_call",
            lambda: run_tool_probe(client, args.model, args.timeout, args.temperature),
        )

    if not args.skip_mem0:
        probe.run_case(
            "mem0_ingest_update_search",
            lambda: run_mem0_probe(
                api_key,
                args.base_url,
                args.model,
                args.timeout,
                args.temperature,
                args.embedding_model,
                args.embedding_dims,
                args.ollama_base_url,
            ),
        )

    required = [result for result in probe.results if result["required"]]
    optional = [result for result in probe.results if not result["required"]]
    report = {
        "provider": "minimax",
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "embedding_model": args.embedding_model,
        "embedding_dims": args.embedding_dims,
        "thinking": "disabled for direct calls and Mem0 provider calls",
        "mem0_tested": not args.skip_mem0,
        "mem0_only": args.mem0_only,
        "compatible": all(result["passed"] for result in required),
        "required_passed": sum(result["passed"] for result in required),
        "required_total": len(required),
        "optional_passed": sum(result["passed"] for result in optional),
        "optional_total": len(optional),
        "usage": {
            "input_tokens": sum(result["usage"].get("input_tokens", 0) for result in probe.results),
            "output_tokens": sum(result["usage"].get("output_tokens", 0) for result in probe.results),
            "total_tokens": sum(result["usage"].get("total_tokens", 0) for result in probe.results),
        },
        "results": probe.results,
    }
    print("\nSummary:")
    print(json.dumps(report, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {output_path}")
    return 0 if report["compatible"] else 1


if __name__ == "__main__":
    sys.exit(main())
