#!/usr/bin/env python3
"""Probe Kimi's OpenAI compatibility for the cognitive evaluation pipeline.

The probe exercises ordinary chat, JSON fact extraction, bounded gisting,
memory-grounded answering, judging, and an optional real Mem0 update/search.
It reads the API key from KIMI_API_KEY or a gitignored key file and never prints
the key.

Examples:
    python benchmarks/cognitive_eval/probe_kimi_compat.py --model MODEL_ID
    python benchmarks/cognitive_eval/probe_kimi_compat.py --model MODEL_ID --skip-mem0
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

DEFAULT_BASE_URL = "https://api.kimi.com/coding/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
KEY_FILES = (
    REPO_ROOT / "kimi_key.txt",
    REPO_ROOT / ".kimi_key",
    Path.home() / ".kimi_key",
)


def load_api_key():
    key = os.environ.get("KIMI_API_KEY", "").strip()
    if key:
        return key
    for path in KEY_FILES:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value.upper().startswith("KIMI_API_KEY"):
            value = value.split("=", 1)[-1].strip().strip('"').strip("'")
        if value:
            os.environ["KIMI_API_KEY"] = value
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
    def __init__(self, client, model, timeout, temperature, completion_token_floor):
        self.client = client
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.completion_token_floor = completion_token_floor
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

    def chat(self, system, user, max_tokens=128, response_format=None):
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max(max_tokens, self.completion_token_floor),
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if response_format is not None:
            kwargs["response_format"] = response_format
        started = time.perf_counter()
        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - started
        choice = response.choices[0]
        message = choice.message
        content = (message.content or "").strip()
        if not content:
            reasoning = getattr(message, "reasoning_content", "") or ""
            raise RuntimeError(
                "empty final content "
                f"(finish_reason={choice.finish_reason!r}, reasoning_chars={len(reasoning)}, "
                f"max_tokens={kwargs['max_tokens']})"
            )
        return content, elapsed, usage_dict(response)

    def run_case(self, name, callback, required=True):
        try:
            passed, detail, elapsed, usage = callback()
            self.record(name, passed, detail, elapsed, usage, required)
        except Exception as exc:  # noqa: BLE001 - each capability must fail independently
            self.record(name, False, f"{type(exc).__name__}: {exc}", required=required)


def run_mem0_probe(api_key, base_url, model, timeout, temperature, max_tokens):
    from mem0 import Memory

    temp_dir = tempfile.mkdtemp(prefix="kimi_mem0_probe_")
    try:
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": model,
                    "temperature": temperature if temperature is not None else 1.0,
                    "api_key": api_key,
                    "openai_base_url": base_url,
                    "max_tokens": max_tokens,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            },
            "vector_store": {
                "provider": "chroma",
                "config": {
                    "path": os.path.join(temp_dir, "chroma"),
                    "collection_name": "kimi_compat_probe",
                },
            },
            "history_db_path": os.path.join(temp_dir, "history.db"),
        }
        started = time.perf_counter()
        memory = Memory.from_config(config)
        user_id = "kimi-compat-user"
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
        passed = bool(first) and bool(second) and any("lisbon" in text.lower() for text in texts)
        detail = (
            f"initial_add={bool(first)}, update_add={bool(second)}, "
            f"stored={len(all_items or [])}, retrieved={len(search_items or [])}, "
            f"lisbon_found={any('lisbon' in text.lower() for text in texts)}"
        )
        return passed, detail, elapsed, {}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Test Kimi for TSM and Mem0 evaluation use")
    parser.add_argument(
        "--model",
        default=os.environ.get("KIMI_MODEL"),
        help="exact Kimi model ID; defaults to KIMI_MODEL",
    )
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature; omitted by default as recommended for Kimi coding models",
    )
    parser.add_argument(
        "--completion-token-floor",
        type=int,
        default=16_000,
        help="minimum max_tokens for always-thinking Kimi coding models",
    )
    parser.add_argument("--skip-mem0", action="store_true")
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()

    if not args.model:
        parser.error("provide --model or set KIMI_MODEL to the exact API model ID")
    api_key = load_api_key()
    if not api_key:
        locations = ", ".join(str(path) for path in KEY_FILES)
        parser.error(f"set KIMI_API_KEY or place the key in one of: {locations}")

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    probe = Probe(
        client,
        args.model,
        args.timeout,
        args.temperature,
        args.completion_token_floor,
    )

    def basic_chat():
        text, elapsed, usage = probe.chat(
            "Follow the user's output-format instruction exactly.",
            "Reply with exactly KIMI_OK and nothing else.",
            max_tokens=16,
        )
        return text == "KIMI_OK", f"response={text!r}", elapsed, usage

    probe.run_case("chat_completions", basic_chat)

    def json_mode():
        text, elapsed, usage = probe.chat(
            "Return a JSON object and no prose.",
            'Return {"compatible": true}.',
            max_tokens=32,
            response_format={"type": "json_object"},
        )
        parsed = extract_json(text)
        return parsed.get("compatible") is True, f"parsed={parsed}", elapsed, usage

    probe.run_case("json_response_format", json_mode, required=False)

    def fact_extraction():
        text, elapsed, usage = probe.chat(
            "Extract complete atomic facts. Return only a JSON object with a facts array.",
            "Message: I moved to Lisbon on March 3, and my favorite color is teal.",
            max_tokens=120,
        )
        parsed = extract_json(text)
        facts = parsed.get("facts", [])
        joined = " ".join(str(fact) for fact in facts).lower()
        passed = isinstance(facts, list) and "lisbon" in joined and "teal" in joined
        return passed, f"facts={facts}", elapsed, usage

    probe.run_case("tsm_fact_extraction", fact_extraction)

    def gisting():
        text, elapsed, usage = probe.chat(
            "Compress into complete atomic bullet facts using at most 60 words. "
            "Preserve every separate dated event.",
            "Facts:\n- The user visited Dr. Smith on March 3.\n"
            "- The user attended a follow-up with Dr. Thompson on March 20.\n"
            "- The assistant suggested generic relaxation exercises.\n\nGist:",
            max_tokens=80,
        )
        lowered = text.lower()
        estimated_tokens = max(1, len(text) // 4)
        passed = "march 3" in lowered and "march 20" in lowered and estimated_tokens <= 80
        return passed, f"estimated_tokens={estimated_tokens}, gist={text!r}", elapsed, usage

    probe.run_case("tsm_gisting", gisting)

    def answering():
        text, elapsed, usage = probe.chat(
            "Answer using only the supplied memories. Give the shortest possible answer.",
            "Memories:\n1. The user's internet plan provides 500 Mbps.\n"
            "2. The user lives in Lisbon.\n\nQuestion: What is the user's internet speed?",
            max_tokens=32,
        )
        return "500" in text, f"answer={text!r}", elapsed, usage

    probe.run_case("memory_answer_generation", answering)

    def judging():
        text, elapsed, usage = probe.chat(
            "Grade semantic equivalence. Reply exactly CORRECT or INCORRECT.",
            "Question: What rice does the user prefer?\n"
            "Gold: Japanese short-grain rice\nPrediction: short-grain Japanese rice",
            max_tokens=8,
        )
        return text.upper() == "CORRECT", f"verdict={text!r}", elapsed, usage

    probe.run_case("answer_judging", judging)

    if not args.skip_mem0:
        probe.run_case(
            "mem0_ingest_update_search",
            lambda: run_mem0_probe(
                api_key,
                args.base_url,
                args.model,
                args.timeout,
                args.temperature,
                args.completion_token_floor,
            ),
        )

    required = [result for result in probe.results if result["required"]]
    optional = [result for result in probe.results if not result["required"]]
    report = {
        "provider": "kimi",
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "completion_token_floor": args.completion_token_floor,
        "mem0_tested": not args.skip_mem0,
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
