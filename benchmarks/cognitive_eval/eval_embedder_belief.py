#!/usr/bin/env python3
"""Comparative Evaluation: MiniLM-L6-v2 vs EmbeddingGemma-2 on Belief Revision.

Measures:
1. Ingestion latency, throughput, and VRAM footprint.
2. Pairwise cosine distributions (update vs coexist vs same).
3. Semantic separation margin across categories.
4. Candidate retrieval recall in a mixed pool of all 260 pairs.
5. End-to-end belief revision outcomes (Recall, Wrong %, Precision, F1) via memory.consolidate().
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from tsm import Memory
from tsm.embedders import SentenceTransformerEmbedder

PAIRS_FILE = os.path.join(HERE, "belief_pairs.jsonl")


def load_pairs(split="all"):
    rows = [json.loads(line) for line in open(PAIRS_FILE, encoding="utf-8") if line.strip()]
    position = Counter()
    out = []
    for row in rows:
        idx = position[row["category"]]
        position[row["category"]] += 1
        half = "dev" if idx % 2 == 0 else "test"
        if split in ("all", half):
            out.append(row)
    return out


def evaluate_embedder(model_name: str, pairs: list, split: str):
    print(f"\n=======================================================")
    print(f"Evaluating: {model_name}")
    print(f"=======================================================")

    # 1. Load Model & Measure VRAM
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    t0 = time.perf_counter()
    embedder = SentenceTransformerEmbedder(model_name)
    load_time = time.perf_counter() - t0
    vram_after = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
    vram_used = vram_after - vram_before

    print(f"  Load time:      {load_time:.2f} s")
    print(f"  Dimension:      {embedder.dimension}")
    print(f"  VRAM footprint: {vram_used:.1f} MB (Peak allocated: {vram_after:.1f} MB)")

    # 2. Encode all statements & Measure Throughput
    texts = []
    for p in pairs:
        texts.append(p["older"])
        texts.append(p["newer"])
    unique_texts = list(dict.fromkeys(texts))

    t0 = time.perf_counter()
    embs = embedder.encode(unique_texts)
    encode_time = time.perf_counter() - t0
    throughput = len(unique_texts) / encode_time
    print(f"  Encoding:       {len(unique_texts)} texts in {encode_time:.3f} s ({throughput:.1f} texts/s)")

    emb_map = {t: embs[i] for i, t in enumerate(unique_texts)}

    # 3. Cosine Separation Analysis
    cat_stats = defaultdict(lambda: {"update": [], "coexist": [], "same": []})
    label_cosines = {"update": [], "coexist": [], "same": []}

    for p in pairs:
        v_old = emb_map[p["older"]]
        v_new = emb_map[p["newer"]]
        cos = float(np.dot(v_old, v_new) / (np.linalg.norm(v_old) * np.linalg.norm(v_new) + 1e-9))
        lbl = p["label"]
        label_cosines[lbl].append(cos)
        cat_stats[p["category"]][lbl].append(cos)

    print("\n--- Cosine Similarities by Label ---")
    for lbl in ("update", "coexist", "same"):
        vals = label_cosines[lbl]
        if vals:
            print(f"  {lbl.upper():<8}: count={len(vals):3d}  mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  min={np.min(vals):.3f}  max={np.max(vals):.3f}")

    margin = np.mean(label_cosines["update"]) - np.mean(label_cosines["coexist"])
    print(f"\n  >> Semantic Separation Margin (mean[update] - mean[coexist]): {margin:+.4f} <<")

    print("\n--- Per-Category Cosine Separation (Update vs Coexist) ---")
    print(f"  {'Category':<32} | {'Update Mean':<11} | {'Coexist Mean':<12} | {'Margin':<7}")
    print("  " + "-" * 70)
    for cat, data in sorted(cat_stats.items()):
        u_mean = np.mean(data["update"]) if data["update"] else float('nan')
        c_mean = np.mean(data["coexist"]) if data["coexist"] else float('nan')
        m_cat = u_mean - c_mean if not (np.isnan(u_mean) or np.isnan(c_mean)) else float('nan')
        u_str = f"{u_mean:.3f}" if not np.isnan(u_mean) else "N/A"
        c_str = f"{c_mean:.3f}" if not np.isnan(c_mean) else "N/A"
        m_str = f"{m_cat:+.3f}" if not np.isnan(m_cat) else "N/A"
        print(f"  {cat:<32} | {u_str:<11} | {c_str:<12} | {m_str:<7}")

    # 4. Storage Engine Candidate Retrieval & Consolidation in Mixed Layout
    db_dir = tempfile.mkdtemp(prefix=f"tsm_embed_eval_{model_name.replace('/', '_')}_")
    try:
        mem = Memory(db_dir, embedder=embedder, extractor="passthrough", profile=None,
                     keep_messages=False)
        
        # Ingest all older facts first, then all newer facts
        t0 = time.perf_counter()
        n = len(pairs)
        for i, pair in enumerate(pairs):
            mem.add([{"role": "user", "content": pair["older"]}], user_id="u")
        for i, pair in enumerate(pairs):
            mem.add([{"role": "user", "content": pair["newer"]}], user_id="u")
        
        insert_time = time.perf_counter() - t0
        print(f"\n--- Storage Engine Ingestion ---")
        print(f"  Ingested {2*n} records (older + newer) in {insert_time:.3f} s ({(2*n/insert_time):.1f} rec/s)")

        old_ids = [f"u_{i + 1}" for i in range(n)]
        new_ids = [f"u_{n + i + 1}" for i in range(n)]

        # Candidate Search: For each newer statement, check if its corresponding older statement is in Top-1, Top-3, Top-5
        top1_hits = 0
        top3_hits = 0
        top5_hits = 0
        update_top1 = 0
        update_top3 = 0
        update_top5 = 0
        update_count = sum(1 for p in pairs if p["label"] == "update")

        for i, pair in enumerate(pairs):
            q_vec = emb_map[pair["newer"]]
            results = mem.engine.search_ann(q_vec, 10, scope="u")
            target_id = old_ids[i]
            
            # Find rank among older memories
            older_rank = None
            rank_counter = 0
            for r_id, _score in results:
                if r_id.startswith("u_") and int(r_id[2:]) <= n:
                    rank_counter += 1
                    if r_id == target_id:
                        older_rank = rank_counter
                        break
            
            if older_rank is not None:
                if older_rank <= 1:
                    top1_hits += 1
                if older_rank <= 3:
                    top3_hits += 1
                if older_rank <= 5:
                    top5_hits += 1
                
                if pair["label"] == "update":
                    if older_rank <= 1:
                        update_top1 += 1
                    if older_rank <= 3:
                        update_top3 += 1
                    if older_rank <= 5:
                        update_top5 += 1

        print("\n--- Candidate Proposal Recall (Mixed Pool of All Pairs) ---")
        print(f"  All Pairs ({len(pairs)} pairs):")
        print(f"    Top-1 Recall: {top1_hits/len(pairs)*100:5.1f}% ({top1_hits}/{len(pairs)})")
        print(f"    Top-3 Recall: {top3_hits/len(pairs)*100:5.1f}% ({top3_hits}/{len(pairs)})")
        print(f"    Top-5 Recall: {top5_hits/len(pairs)*100:5.1f}% ({top5_hits}/{len(pairs)})")
        print(f"  Update Pairs Only ({update_count} updates):")
        print(f"    Top-1 Recall: {update_top1/update_count*100:5.1f}% ({update_top1}/{update_count})")
        print(f"    Top-3 Recall: {update_top3/update_count*100:5.1f}% ({update_top3}/{update_count})")
        print(f"    Top-5 Recall: {update_top5/update_count*100:5.1f}% ({update_top5}/{update_count})")

        # 5. Consolidation (Unverified Lexical Gate Baseline)
        committed = mem.consolidate()
        resolved = {r["id"]: r["current_id"] for r in mem.engine.resolve_beliefs(old_ids + new_ids)}
        
        true_hides = 0
        false_hides = 0
        coexist_count = sum(1 for p in pairs if p["label"] == "coexist")
        
        for i, pair in enumerate(pairs):
            current = resolved[old_ids[i]]
            hidden = current != old_ids[i]
            if pair["label"] == "update" and hidden:
                true_hides += 1
            elif pair["label"] == "coexist" and hidden:
                false_hides += 1

        recall = true_hides / update_count if update_count else 0
        wrong_pct = false_hides / coexist_count if coexist_count else 0
        precision = true_hides / (true_hides + false_hides) if (true_hides + false_hides) else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

        print("\n--- End-to-End Consolidation (Lexical Gate Built-in Baseline) ---")
        print(f"  Committed:  {committed} edges")
        print(f"  Recall:     {recall:.3f} ({true_hides}/{update_count} updates detected)")
        print(f"  Wrong:      {wrong_pct*100:.1f}% ({false_hides}/{coexist_count} coexisting hidden)")
        print(f"  Precision:  {precision:.3f}")
        print(f"  F1:         {f1:.3f}")

        mem.close()
    finally:
        shutil.rmtree(db_dir, ignore_errors=True)

    return {
        "model": model_name,
        "dim": embedder.dimension,
        "load_time": load_time,
        "vram_mb": vram_used,
        "throughput": throughput,
        "margin": margin,
        "update_top1": update_top1 / update_count,
        "update_top3": update_top3 / update_count,
        "update_top5": update_top5 / update_count,
        "consol_recall": recall,
        "consol_wrong": wrong_pct,
        "consol_prec": precision,
        "consol_f1": f1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "test", "all"), default="all")
    args = ap.parse_args()

    pairs = load_pairs(args.split)
    print(f"Loaded {len(pairs)} pairs from split='{args.split}'")

    res_minilm = evaluate_embedder("sentence-transformers/all-MiniLM-L6-v2", pairs, args.split)
    res_gemma = evaluate_embedder("google/embeddinggemma-2", pairs, args.split)

    print("\n" + "=" * 75)
    print("SUMMARY COMPARISON: MiniLM-L6-v2 vs EmbeddingGemma-2")
    print("=" * 75)
    print(f"{'Metric':<35} | {'MiniLM-L6-v2':<16} | {'EmbeddingGemma-2':<16}")
    print("-" * 75)
    print(f"{'Embedding Dimension':<35} | {res_minilm['dim']:<16} | {res_gemma['dim']:<16}")
    print(f"{'VRAM Footprint (MB)':<35} | {res_minilm['vram_mb']:<16.1f} | {res_gemma['vram_mb']:<16.1f}")
    print(f"{'Throughput (texts/sec)':<35} | {res_minilm['throughput']:<16.1f} | {res_gemma['throughput']:<16.1f}")
    print(f"{'Semantic Separation Margin':<35} | {res_minilm['margin']:<+16.4f} | {res_gemma['margin']:<+16.4f}")
    print(f"{'Update Candidate Top-1 Recall':<35} | {res_minilm['update_top1']*100:<15.1f}% | {res_gemma['update_top1']*100:<15.1f}%")
    print(f"{'Update Candidate Top-3 Recall':<35} | {res_minilm['update_top3']*100:<15.1f}% | {res_gemma['update_top3']*100:<15.1f}%")
    print(f"{'Update Candidate Top-5 Recall':<35} | {res_minilm['update_top5']*100:<15.1f}% | {res_gemma['update_top5']*100:<15.1f}%")
    print(f"{'Consolidation Recall (Lexical Gate)':<35} | {res_minilm['consol_recall']:<16.3f} | {res_gemma['consol_recall']:<16.3f}")
    print(f"{'Consolidation Wrong % (False Stale)':<35} | {res_minilm['consol_wrong']*100:<15.1f}% | {res_gemma['consol_wrong']*100:<15.1f}%")
    print(f"{'Consolidation Precision':<35} | {res_minilm['consol_prec']:<16.3f} | {res_gemma['consol_prec']:<16.3f}")
    print(f"{'Consolidation F1':<35} | {res_minilm['consol_f1']:<16.3f} | {res_gemma['consol_f1']:<16.3f}")
    print("=" * 75)


if __name__ == "__main__":
    main()
