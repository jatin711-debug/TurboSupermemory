"""Failure-stage and disagreement classification for memory evaluations."""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path


def classify_failure(
    correct,
    has_gold_token,
    token_in_source,
    token_in_active,
    token_in_retrieved,
):
    if correct:
        return "correct"
    if not has_gold_token:
        return "unclassified_no_gold_token"
    if not token_in_source:
        return "no_lexical_signal_in_source"
    if not token_in_active:
        return "budget_removed_lexical_signal"
    if not token_in_retrieved:
        return "retrieval_missed_lexical_signal"
    return "answer_failed_with_lexical_signal"


def classify_disagreement(correct_by_system):
    tsm = bool(correct_by_system.get("tsm"))
    mem0 = bool(correct_by_system.get("mem0"))
    if "tsm" in correct_by_system and "mem0" in correct_by_system:
        if tsm and mem0:
            return "both"
        if tsm:
            return "tsm_only"
        if mem0:
            return "mem0_only"
        return "neither"
    correct = sorted(system for system, value in correct_by_system.items() if value)
    return "correct:" + ",".join(correct) if correct else "none_correct"


def summarize_diagnostics(records):
    disagreements = defaultdict(Counter)
    failures = defaultdict(lambda: defaultdict(Counter))
    grouped = defaultdict(dict)
    for record in records:
        budget = record["budget"]
        failures[budget][record["system"]][record["failure_stage"]] += 1
        key = (budget, record["conversation_id"], record["query_id"])
        grouped[key][record["system"]] = record["correct"]

    query_labels = []
    for key, correct_by_system in grouped.items():
        label = classify_disagreement(correct_by_system)
        query_labels.append({
            "budget": key[0],
            "conversation_id": key[1],
            "query_id": key[2],
            "label": label,
        })
        disagreements[key[0]][label] += 1

    return {
        "disagreements": {
            budget: dict(sorted(counts.items())) for budget, counts in sorted(disagreements.items())
        },
        "failure_stages": {
            budget: {
                system: dict(sorted(counts.items()))
                for system, counts in sorted(systems.items())
            }
            for budget, systems in sorted(failures.items())
        },
        "query_labels": query_labels,
    }


def write_diagnostics(path, payload):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, output)
