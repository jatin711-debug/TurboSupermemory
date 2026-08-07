#!/usr/bin/env python3
"""Bounded-memory head-to-head: TSM compression vs deletion vs Mem0.

The primary mode caps every system by the same approximate active-memory text
tokens. This avoids treating one long gist as equivalent to one short atomic
fact. The historical equal-slot mode remains available through ``--budget``.

Paid quality run:
    python benchmarks/cognitive_eval/bounded_head_to_head.py --limit 120 \
        --storage-budgets 64,128,256 --token-budget 150 \
        --extractor openai --gister openai --judge openai

Offline harness smoke run:
    python benchmarks/cognitive_eval/bounded_head_to_head.py --smoke
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cognitive_eval.adapters.tsm_adapter import TSMAdapter
from cognitive_eval.benchmark_datasets.longmemeval import load_longmemeval
from cognitive_eval.budgeting import (
    BoundedStores,
    build_slot_bounded_stores,
    build_token_bounded_stores,
    pack_recent,
    total_tokens,
    truncate_to_budget,
)
from cognitive_eval.compress_eval import insert_facts
from cognitive_eval.diagnostics import classify_failure, summarize_diagnostics, write_diagnostics
from cognitive_eval.head_to_head_eval import conv_facts_with_roles, make_mem0, mem0_ingest
from cognitive_eval.run_belief_longmemeval import hit_at, key_tokens, prewarm_extraction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bounded_head_to_head")

SYSTEMS = ("naive", "tsm", "mem0")


class RetrievalProxyJudge:
    """No-cost smoke metric. This is not a substitute for an LLM-judged run."""

    model = "retrieval-proxy"

    def __init__(self):
        self.calls = 0

    def answer_and_judge(self, _query, texts, gold):
        self.calls += 1
        return hit_at(gold, texts, len(texts))


def score_task(judge, task):
    if isinstance(judge, RetrievalProxyJudge):
        correct = judge.answer_and_judge(task["query"], task["retrieved"], task["gold"])
        return ("RETRIEVAL HIT" if correct else "NO ANSWER"), correct
    prediction = judge.answer(task["query"], task["retrieved"])
    return prediction, judge.judge(task["query"], task["gold"], prediction)


def parse_positive_csv(value, name):
    try:
        parsed = sorted({int(item) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError(f"{name} must be comma-separated integers") from exc
    if not parsed or any(item < 2 for item in parsed):
        raise ValueError(f"{name} entries must be at least 2")
    return parsed


def _mem0_all(mem0, user_id):
    try:
        result = mem0.get_all(user_id=user_id, limit=10_000)
        return result.get("results", result) if isinstance(result, dict) else (result or [])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"failed to enumerate Mem0 memories for {user_id}") from exc


def _memory_text(memory):
    return memory.get("memory", "") if isinstance(memory, dict) else str(memory)


def ordered_memory_texts(memories):
    """Return oldest-to-newest memory text, using timestamps when available."""
    memories = list(memories)
    if memories and all(
        isinstance(memory, dict) and (memory.get("created_at") or memory.get("updated_at"))
        for memory in memories
    ):
        memories.sort(key=lambda memory: memory.get("created_at") or memory.get("updated_at"))
    return [text for text in (_memory_text(memory) for memory in memories) if text and text.strip()]


def bound_memory_texts(texts, limit, unit):
    if unit == "tokens":
        return pack_recent(texts, limit)[0]
    return list(texts[-limit:])


def overflow_memory_texts(texts, limit, unit):
    if unit == "tokens":
        return pack_recent(texts, limit)[1]
    return list(texts[:-limit])


def plain_text_retrieve(texts, query, model, pool_k, context_budget):
    """Search already-bounded texts with the same embedding model as TSM/naive."""
    if not texts:
        return []
    vectors = np.asarray(model.encode(texts), dtype=np.float32)
    query_vector = np.asarray(model.encode(query), dtype=np.float32)
    vector_norms = np.linalg.norm(vectors, axis=1) + 1e-9
    query_norm = np.linalg.norm(query_vector) + 1e-9
    scores = (vectors @ query_vector) / (vector_norms * query_norm)
    order = np.argsort(-scores)[:pool_k]
    return truncate_to_budget([texts[index] for index in order], context_budget)


def plain_retrieve(adapter, query, pool_k, context_budget):
    results = adapter.search(query, top_k=pool_k, use_cognitive=False)
    return truncate_to_budget([result["text"] for result in results], context_budget)


def answer_in_overflow(answer, overflow, active_store):
    tokens = key_tokens(answer)
    if not tokens:
        return False
    token = tokens[0]
    return any(token in text.lower() for text in overflow) and not any(
        token in text.lower() for text in active_store
    )


def mean(values):
    return sum(values) / len(values) if values else 0.0


def gold_token_in_texts(gold, texts):
    tokens = key_tokens(gold)
    if not tokens:
        return None, False
    token = tokens[0]
    return token, any(token in (text or "").lower() for text in texts)


def main():
    parser = argparse.ArgumentParser(description="Bounded-memory head-to-head")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument(
        "--storage-budgets",
        type=str,
        default=None,
        help="active-memory text-token caps, comma-separated; enables equal-token mode",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=8,
        help="historical memory-slot cap, used only without --storage-budgets",
    )
    parser.add_argument(
        "--gist-share",
        type=float,
        default=0.5,
        help="fraction of each storage-token budget reserved for the overflow gist",
    )
    parser.add_argument(
        "--compression-policy",
        choices=["recency", "role-aware"],
        default="role-aware",
        help="TSM survivor/gist policy; role-aware prioritizes user facts over assistant advice",
    )
    parser.add_argument(
        "--gist-chunk-tokens",
        type=int,
        default=32,
        help="target token allocation per chronological mini-gist",
    )
    parser.add_argument(
        "--max-gist-chunks",
        type=int,
        default=4,
        help="maximum separately embedded mini-gists per active store",
    )
    parser.add_argument("--token-budget", type=int, default=150, help="answer-context token cap")
    parser.add_argument("--pool-k", type=int, default=20)
    parser.add_argument("--local-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--tsm-embedder", choices=["local", "openai", "ollama"], default="openai")
    parser.add_argument("--embed-model", type=str, default="text-embedding-3-small")
    parser.add_argument("--embedding-dims", type=int, default=None)
    parser.add_argument("--ollama-base-url", default="http://localhost:11434")
    parser.add_argument("--extractor", choices=["mock", "ollama", "openai", "minimax", "auto"], default="openai")
    parser.add_argument("--extractor-model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--gister", choices=["extractive", "ollama", "openai", "minimax"], default="openai")
    parser.add_argument("--gist-model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--mem0-model", type=str, default="gpt-4.1-nano")
    parser.add_argument("--mem0-llm-provider", choices=["openai", "minimax"], default="openai")
    parser.add_argument("--mem0-embed-provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument(
        "--mem0-path",
        type=str,
        default=None,
        help="persistent Chroma path used to resume previously ingested Mem0 conversations",
    )
    parser.add_argument("--systems", type=str, default="naive,tsm,mem0")
    parser.add_argument("--judge", choices=["none", "auto", "ollama", "openai", "minimax"], default="openai")
    parser.add_argument("--judge-model", type=str, default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--diagnostics-path",
        type=str,
        default=None,
        help="write per-query stores, retrievals, predictions, and failure stages to JSON",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="offline 5-conversation harness check with mock/extractive/proxy components",
    )
    args = parser.parse_args()

    if args.smoke:
        args.limit = min(args.limit, 5)
        args.storage_budgets = args.storage_budgets or "32,64"
        args.systems = "naive,tsm"
        args.extractor = "mock"
        args.gister = "extractive"
        args.judge = "none"
        args.tsm_embedder = "local"
        args.workers = 1
        args.diagnostics_path = args.diagnostics_path or os.path.join(
            tempfile.gettempdir(), "tsm-bounded-smoke-diagnostics.json"
        )
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    systems = [name.strip() for name in args.systems.split(",") if name.strip() in SYSTEMS]
    if not systems:
        parser.error("--systems must include at least one of naive,tsm,mem0")
    if args.storage_budgets:
        try:
            limits = parse_positive_csv(args.storage_budgets, "--storage-budgets")
        except ValueError as exc:
            parser.error(str(exc))
        budget_defs = [(f"{limit}-tokens", limit, "tokens") for limit in limits]
    else:
        if args.budget < 2:
            parser.error("--budget must be at least 2")
        budget_defs = [(f"{args.budget}-slots", args.budget, "slots")]
    if not 0.0 < args.gist_share < 1.0:
        parser.error("--gist-share must be between 0 and 1")
    if args.gist_chunk_tokens < 1 or args.max_gist_chunks < 1:
        parser.error("gist chunk settings must be positive")

    if args.judge == "none":
        judge = RetrievalProxyJudge()
    else:
        from cognitive_eval.judge import create_judge

        judge_kwargs = (
            {
                "ollama_model": args.judge_model,
                "openai_model": args.judge_model,
                "minimax_model": args.judge_model,
            }
            if args.judge_model
            else {}
        )
        judge = create_judge(args.judge, **judge_kwargs)

    from cognitive_eval.extraction import create_extractor
    from cognitive_eval.gist import create_gister

    extractor_kwargs = (
        {"openai_model": args.extractor_model, "minimax_model": args.extractor_model}
        if args.extractor_model
        else {}
    )
    extractor = create_extractor(args.extractor, **extractor_kwargs)
    gister = create_gister(args.gister, model=args.gist_model)

    embed_model = None
    if args.tsm_embedder == "openai":
        from cognitive_eval.openai_embedder import OpenAIEmbedder

        embed_model = OpenAIEmbedder(model=args.embed_model)
    elif args.tsm_embedder == "ollama":
        from cognitive_eval.ollama_embedder import OllamaEmbedder

        embed_model = OllamaEmbedder(
            model=args.embed_model,
            dim=args.embedding_dims or 1024,
            host=args.ollama_base_url,
        )
    elif systems == ["mem0"]:
        from cognitive_eval.embedding import SimpleEmbeddingProvider

        embed_model = SimpleEmbeddingProvider(args.local_model)

    conversations = load_longmemeval(args.data_dir)[: args.limit]
    logger.info(
        "Loaded %d conversations. storage=%s context=%d systems=%s embedder=%s judge=%s",
        len(conversations),
        ",".join(key for key, _limit, _unit in budget_defs),
        args.token_budget,
        systems,
        args.tsm_embedder,
        type(judge).__name__,
    )
    prewarm_extraction(
        extractor,
        conversations,
        workers=max(args.workers, 8),
        contextual=True,
    )
    for name in (
        "cognitive_eval.adapters.tsm",
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "mem0",
        "huggingface_hub",
        "sentence_transformers",
        "transformers",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)

    mem0 = None
    completed_ids = set()
    completed_path = None
    if "mem0" in systems:
        mem0 = make_mem0(
            args.mem0_model,
            args.embed_model,
            path=args.mem0_path,
            llm_provider=args.mem0_llm_provider,
            embed_provider=args.mem0_embed_provider,
            embed_dim=args.embedding_dims or (1024 if args.mem0_embed_provider == "ollama" else 1536),
            ollama_base_url=args.ollama_base_url,
        )
        if args.mem0_path:
            completed_path = os.path.join(args.mem0_path, "_completed.json")
            if os.path.exists(completed_path):
                try:
                    with open(completed_path, encoding="utf-8") as completed_file:
                        completed_ids = set(json.load(completed_file))
                    logger.info("Mem0 resume: %d conversations already ingested", len(completed_ids))
                except (OSError, ValueError):
                    completed_ids = set()

    effective_compression_policy = (
        args.compression_policy if any(unit == "tokens" for _key, _limit, unit in budget_defs)
        else "recency"
    )
    tasks = []
    pressured = defaultdict(int)
    sizes = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    shared_local_model = embed_model

    def summarize(texts, max_tokens):
        return gister.summarize(texts, max_tokens=max_tokens)

    for conversation_index, conversation in enumerate(conversations):
        attributed_facts = conv_facts_with_roles(extractor, conversation)
        facts = [fact["text"] for fact in attributed_facts]
        fact_roles = [fact["role"] for fact in attributed_facts]
        stores_by_budget: dict[str, BoundedStores] = {}
        for key, limit, unit in budget_defs:
            if unit == "tokens":
                stores = build_token_bounded_stores(
                    facts,
                    limit,
                    summarize,
                    args.gist_share,
                    roles=fact_roles,
                    role_aware=args.compression_policy == "role-aware",
                    gist_chunk_tokens=args.gist_chunk_tokens,
                    max_gist_chunks=args.max_gist_chunks,
                )
            else:
                stores = build_slot_bounded_stores(facts, limit, summarize)
            if stores is not None:
                stores_by_budget[key] = stores
        if not stores_by_budget:
            continue

        if mem0 is not None and conversation.conv_id not in completed_ids:
            if not mem0_ingest(mem0, conversation, "incremental"):
                logger.warning("Mem0 ingest failed for %s; skipping all arms", conversation.conv_id)
                continue
            completed_ids.add(conversation.conv_id)
            if completed_path:
                try:
                    with open(completed_path, "w", encoding="utf-8") as completed_file:
                        json.dump(sorted(completed_ids), completed_file)
                except OSError:
                    pass

        native_mem0_texts = ordered_memory_texts(_mem0_all(mem0, conversation.conv_id)) if mem0 else []
        for key, stores in stores_by_budget.items():
            pressured[key] += 1
            mem0_active = []
            if mem0 is not None:
                mem0_active = bound_memory_texts(native_mem0_texts, stores.limit, stores.unit)
                mem0_overflow = overflow_memory_texts(
                    native_mem0_texts, stores.limit, stores.unit
                )
                sizes[key]["mem0"]["slots"].append(len(mem0_active))
                sizes[key]["mem0"]["tokens"].append(total_tokens(mem0_active))
                sizes[key]["mem0"]["native_slots"].append(len(native_mem0_texts))
                sizes[key]["mem0"]["native_tokens"].append(total_tokens(native_mem0_texts))

            active_stores = {"naive": stores.naive, "tsm": stores.compressed}
            adapters = {}
            for system in ("naive", "tsm"):
                if system not in systems:
                    continue
                db_path = tempfile.mkdtemp(prefix=f"tsm_bounded_{system}_")
                adapter = TSMAdapter(
                    db_path=db_path,
                    embedding_model=args.local_model,
                    extractor="mock",  # unused: insert_facts bypasses add()/extraction
                    extractor_instance=extractor,  # share the runner-level extractor
                    cognitive_features=False,
                    belief_revision=False,
                    model=shared_local_model,
                )
                if shared_local_model is None:
                    shared_local_model = adapter.model
                insert_facts(adapter, active_stores[system], conversation.conv_id)
                adapters[system] = (adapter, db_path)
                sizes[key][system]["slots"].append(len(active_stores[system]))
                sizes[key][system]["tokens"].append(total_tokens(active_stores[system]))

            try:
                for query in conversation.queries:
                    if query.is_abstention:
                        continue
                    in_overflow = answer_in_overflow(
                        query.answer_text, stores.naive_overflow, stores.naive
                    )
                    question_type = query.question_type or "?"
                    for system, (adapter, _db_path) in adapters.items():
                        texts = plain_retrieve(
                            adapter, query.query_text, args.pool_k, args.token_budget
                        )
                        tasks.append({
                            "budget": key,
                            "budget_limit": stores.limit,
                            "budget_unit": stores.unit,
                            "conversation_id": conversation.conv_id,
                            "query_id": query.query_id,
                            "question_type": question_type,
                            "system": system,
                            "answer_in_evicted": in_overflow,
                            "query": query.query_text,
                            "gold": query.answer_text,
                            "active_store": list(active_stores[system]),
                            "source_store": list(facts),
                            "source_roles": list(fact_roles),
                            "benchmark_overflow": list(
                                stores.naive_overflow
                                if system == "naive"
                                else stores.compressed_tail
                            ),
                            "compression_tail": list(
                                stores.compressed_tail if system == "tsm" else []
                            ),
                            "gists": list(stores.gists if system == "tsm" else []),
                            "retrieved": texts,
                        })
                    if mem0 is not None:
                        texts = plain_text_retrieve(
                            mem0_active,
                            query.query_text,
                            shared_local_model,
                            args.pool_k,
                            args.token_budget,
                        )
                        tasks.append({
                            "budget": key,
                            "budget_limit": stores.limit,
                            "budget_unit": stores.unit,
                            "conversation_id": conversation.conv_id,
                            "query_id": query.query_id,
                            "question_type": question_type,
                            "system": "mem0",
                            "answer_in_evicted": in_overflow,
                            "query": query.query_text,
                            "gold": query.answer_text,
                            "active_store": list(mem0_active),
                            "source_store": list(native_mem0_texts),
                            "source_roles": [],
                            "benchmark_overflow": list(mem0_overflow),
                            "compression_tail": [],
                            "gists": [],
                            "retrieved": texts,
                        })
            finally:
                for adapter, db_path in adapters.values():
                    adapter.close()
                    shutil.rmtree(db_path, ignore_errors=True)

        if (conversation_index + 1) % 20 == 0:
            logger.info("Processed %d/%d conversations", conversation_index + 1, len(conversations))

    if not tasks:
        raise RuntimeError(
            "benchmark produced no scored tasks; increase --limit or lower the storage budget"
        )
    logger.info("Scoring %d answers with %s (%d workers)", len(tasks), type(judge).__name__, args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        scored = list(executor.map(lambda task: score_task(judge, task), tasks))

    aggregation = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0])))
    diagnostic_records = []
    for task, (prediction, correct) in zip(tasks, scored):
        key = task["budget"]
        system = task["system"]
        in_overflow = task["answer_in_evicted"]
        subsets = ("all", "answer_in_evicted" if in_overflow else "answer_in_survivors")
        for subset in subsets:
            aggregation[key][subset][system][0] += 1
            aggregation[key][subset][system][1] += int(correct)

        gold_token, token_in_source = gold_token_in_texts(task["gold"], task["source_store"])
        _gold_token, token_in_active = gold_token_in_texts(task["gold"], task["active_store"])
        _gold_token, token_in_retrieved = gold_token_in_texts(task["gold"], task["retrieved"])
        diagnostic_records.append({
            **task,
            "active_slots": len(task["active_store"]),
            "active_tokens": total_tokens(task["active_store"]),
            "retrieved_slots": len(task["retrieved"]),
            "retrieved_tokens": total_tokens(task["retrieved"]),
            "prediction": prediction,
            "correct": bool(correct),
            "gold_token": gold_token,
            "gold_token_in_source": token_in_source,
            "gold_token_in_active": token_in_active,
            "gold_token_in_retrieved": token_in_retrieved,
            "failure_stage": classify_failure(
                bool(correct),
                gold_token is not None,
                token_in_source,
                token_in_active,
                token_in_retrieved,
            ),
        })

    diagnostic_summary = summarize_diagnostics(diagnostic_records)
    labels = {
        (item["budget"], item["conversation_id"], item["query_id"]): item["label"]
        for item in diagnostic_summary["query_labels"]
    }
    for record in diagnostic_records:
        record["disagreement"] = labels[
            (record["budget"], record["conversation_id"], record["query_id"])
        ]

    diagnostics_payload = {
        "metadata": {
            "storage_budgets": [
                {"key": key, "limit": limit, "unit": unit}
                for key, limit, unit in budget_defs
            ],
            "context_token_budget": args.token_budget,
            "systems": systems,
            "embedder": (
                args.embed_model
                if args.tsm_embedder in ("openai", "ollama")
                else args.local_model
            ),
            "judge_model": getattr(judge, "model", "?"),
            "compression_policy": effective_compression_policy,
            "gist_share": args.gist_share,
            "gist_chunk_tokens": args.gist_chunk_tokens,
            "max_gist_chunks": args.max_gist_chunks,
            "smoke": args.smoke,
        },
        "summary": diagnostic_summary,
        "records": diagnostic_records,
    }
    if args.diagnostics_path:
        write_diagnostics(args.diagnostics_path, diagnostics_payload)
        logger.info("Wrote per-query diagnostics to %s", args.diagnostics_path)

    def accuracy(key, subset, system):
        count, correct = aggregation[key][subset][system]
        return (correct / count if count else 0.0), count

    by_budget = {}
    for key, limit, unit in budget_defs:
        logger.info("=" * 100)
        logger.info(
            "BOUNDED HEAD-TO-HEAD @ %d %s active memory, %d-token context (pressured=%d)",
            limit,
            unit,
            args.token_budget,
            pressured[key],
        )
        for system in systems:
            system_sizes = sizes[key][system]
            extra = ""
            if system == "mem0" and system_sizes["native_slots"]:
                extra = " native=%.1f slots/%.1f tok" % (
                    mean(system_sizes["native_slots"]),
                    mean(system_sizes["native_tokens"]),
                )
            logger.info(
                "  active %-6s avg=%.1f slots / %.1f tok%s",
                system,
                mean(system_sizes["slots"]),
                mean(system_sizes["tokens"]),
                extra,
            )

        overall = {}
        subsets_summary = {}
        for subset in ("all", "answer_in_evicted", "answer_in_survivors"):
            cells = []
            subset_summary = {}
            for system in systems:
                score, count = accuracy(key, subset, system)
                subset_summary[system] = round(score, 4)
                if subset == "all":
                    overall[system] = score
                cells.append(f"{system}={score:.3f}")
            count = max((aggregation[key][subset][system][0] for system in systems), default=0)
            logger.info("  %-22s | %s (n=%d)", subset, "  ".join(cells), count)
            subsets_summary[subset] = subset_summary

        if "tsm" in overall and "mem0" in overall:
            logger.info("  TSM-compress vs Mem0  : %+.3f", overall["tsm"] - overall["mem0"])
        if "tsm" in overall and "naive" in overall:
            logger.info("  TSM-compress vs naive : %+.3f", overall["tsm"] - overall["naive"])

        by_budget[key] = {
            "limit": limit,
            "unit": unit,
            "pressured": pressured[key],
            "overall": {system: round(overall.get(system, 0.0), 4) for system in systems},
            "subsets": subsets_summary,
            "active_slots_avg": {
                system: round(mean(sizes[key][system]["slots"]), 2) for system in systems
            },
            "active_tokens_avg": {
                system: round(mean(sizes[key][system]["tokens"]), 2) for system in systems
            },
        }

    logger.info("=" * 100)
    summary = {
        "storage_budgets": by_budget,
        "context_token_budget": args.token_budget,
        "systems": systems,
        "embedder": args.embed_model if args.tsm_embedder in ("openai", "ollama") else args.local_model,
        "judge_model": getattr(judge, "model", "?"),
        "judge_calls": judge.calls,
        "judge_input_tokens": getattr(judge, "input_tokens", 0),
        "judge_output_tokens": getattr(judge, "output_tokens", 0),
        "gist_calls": gister.calls,
        "gist_input_tokens": getattr(gister, "input_tokens", 0),
        "gist_output_tokens": getattr(gister, "output_tokens", 0),
        "extractor_calls": getattr(extractor, "calls", 0),
        "extractor_input_tokens": getattr(extractor, "input_tokens", 0),
        "extractor_output_tokens": getattr(extractor, "output_tokens", 0),
        "mem0_llm_provider": args.mem0_llm_provider,
        "mem0_embed_provider": args.mem0_embed_provider,
        "mem0_usage": getattr(mem0, "_provider_usage", {}) if mem0 else {},
        "compression_policy": effective_compression_policy,
        "gist_share": args.gist_share,
        "gist_chunk_tokens": args.gist_chunk_tokens,
        "max_gist_chunks": args.max_gist_chunks,
        "smoke": args.smoke,
        "diagnostics_path": args.diagnostics_path,
        "diagnostics": diagnostic_summary,
    }
    if len(budget_defs) == 1 and budget_defs[0][2] == "slots":
        key, limit, _unit = budget_defs[0]
        historical = by_budget[key]
        summary.update(
            {
                "budget": limit,
                "token_budget": args.token_budget,
                "pressured": historical["pressured"],
                "overall": historical["overall"],
                "mem_size_avg": historical["active_slots_avg"],
                "evicted": historical["subsets"]["answer_in_evicted"],
            }
        )
    logger.info("GATE_SUMMARY: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
