"""Experience: how earlier tasks went, as the user reported them.

``Memory.feedback`` stores an **episode** for every outcome a caller reports:
the task, the answer given, the outcome, what the user said about it (a
correction), who says so (``source``) and the memories that were in context.
``Memory.episodes`` returns the episodes nearest a new task as they were, and
``episode_block`` puts them in front of a reader under a token cap. Nothing is
distilled, scored or rewritten: what the user said is shown word for word.

Measured 2026-11-13 (``benchmarks/executive/lessons_eval.py`` on a task world
written blind to this module, every answer checked by code): on tasks with a
house rule the reader did not know, the three nearest episodes took the second
half of a stream from 3 to 45 of 66 right (the rules stated: 65), and cost one
of 48 controls. Curated lessons (items distilled from episodes by a model,
with helpful / harmful counters and a probation / active / deprecated status)
got 11 of 66 and were removed (``LEARNINGS.md``; the code is at
``git show 6825fd5:tsm/experience.py``): the distiller wrote general advice
("confirm the date format") in place of the rule the correction stated, and
the counters credited every lesson that was shown when a task went right.

An episode with ``source="self"`` (the agent grading itself) is kept and
returned like any other, and ``episode_block`` marks it as self-reported.

Episodes live in their own engine (``<db_path>/tsm_experience``): the
memories' engine is untouched and no recall changes.
"""

import json
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

SOURCES = frozenset({"user", "verifier", "judge", "self"})


def normalize_outcome(outcome) -> str:
    """``"success"`` or ``"failure"`` from a word, a bool, or a score in
    [0, 1] (0.5 and above is a success)."""
    if isinstance(outcome, bool):
        return "success" if outcome else "failure"
    if isinstance(outcome, (int, float)):
        return "success" if float(outcome) >= 0.5 else "failure"
    word = str(outcome).strip().lower()
    if word in ("success", "succeeded", "correct", "good", "helpful", "pass", "yes", "right"):
        return "success"
    if word in ("failure", "failed", "wrong", "incorrect", "bad", "harmful", "fail", "no"):
        return "failure"
    raise ValueError(f"unknown outcome {outcome!r}: use 'success' / 'failure', a bool or a score")


def episode_block(episodes: Sequence[Dict], token_cap: int = 200) -> str:
    """Past episodes as text for a reader's prompt (``Memory.episodes``),
    best first, within ``token_cap`` (about four characters a token); ``""``
    without any."""
    lines, used = [], 0
    for ep in episodes:
        line = f"- Task: {ep['text']}"
        if ep.get("answer"):
            line += f"\n  Answer given: {ep['answer']}"
        line += f"\n  Outcome: {ep.get('outcome', '?')}"
        if ep.get("source") == "self":
            line += " (self-reported)"
        if ep.get("note"):
            line += f"\n  Feedback: {ep['note']}"
        cost = max(1, len(line) // 4)
        if used + cost > token_cap:
            break
        lines.append(line)
        used += cost
    return "Earlier attempts at similar tasks:\n" + "\n".join(lines) if lines else ""


def _filter(*parts: Dict) -> str:
    return json.dumps(parts[0] if len(parts) == 1 else {"op": "and", "filters": list(parts)})


def _eq(field: str, value) -> Dict:
    return {"field": field, "op": "eq", "value": value}


class ExperienceLedger:
    """The experience engine: episodes in, the nearest episodes out."""

    def __init__(self, engine, embed: Callable[[List[str]], np.ndarray]):
        self.engine = engine
        self._embed = embed

    def _next_id(self, prefix: str) -> str:
        seq = int(self.engine.next_insert_seq())
        while self.engine.contains_id(f"{prefix}_{seq}"):
            seq += 1
        return f"{prefix}_{seq}"

    def _items(self, ids: Sequence[str]) -> Dict[str, Dict]:
        out = {}
        for rec in self.engine.get_records(list(ids)):
            if rec is None:
                continue
            try:
                payload = json.loads(rec["payload"]) if rec["payload"] else {}
            except ValueError:
                payload = {}
            out[rec["id"]] = {"id": rec["id"], "text": rec["text"], "scope": rec["scope"],
                              **payload}
        return out

    def record(self, task: str, outcome, memory_ids: Sequence[str] = (),
               user_id: Optional[str] = None, source: str = "user",
               at: Optional[str] = None, vector=None, answer: Optional[str] = None,
               note: Optional[str] = None) -> str:
        """Store one episode; returns its id. ``answer`` is what was
        answered, ``note`` what the user said about it. ``vector`` is the
        task's embedding when the caller has it (else it is embedded)."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")
        if not (task or "").strip():
            raise ValueError("task must not be empty")
        payload = {"kind": "episode", "outcome": normalize_outcome(outcome), "source": source,
                   "ids": [str(m) for m in memory_ids],
                   "answer": (answer or "").strip() or None,
                   "note": (note or "").strip() or None, "user_id": user_id, "at": at}
        vec = (np.asarray(vector, dtype=np.float32)[None, :] if vector is not None
               else np.ascontiguousarray(self._embed([task]), dtype=np.float32))
        mid = self._next_id("ep")
        self.engine.insert_batch([mid], [task], vec, [1.0], [[]], [json.dumps(payload)],
                                 [user_id] if user_id is not None else None, ["episode"])
        return mid

    def episodes(self, vector, user_id: Optional[str] = None, k: int = 3) -> List[Dict]:
        """The past episodes nearest a task's vector, of ``user_id`` or
        global, best first (as they were reported, whatever their source)."""
        hits = self.engine.search_ann(np.asarray(vector, dtype=np.float32), max(4 * k, 12),
                                      filter=_filter(_eq("kind", "episode")))
        scores = dict(hits)
        items = self._items([mid for mid, _ in hits])
        out = []
        for mid, _ in hits:
            item = items.get(mid)
            if item is None or item.get("scope") not in (None, user_id):
                continue
            item["score"] = float(scores[mid])
            out.append(item)
            if len(out) == k:
                break
        return out

    def forget_content(self, forgotten, scopes: Sequence[Optional[str]]) -> int:
        """Delete the records of ``scopes`` (``None`` is the global scope)
        whose task, answer or feedback carries forgotten content
        (``forgotten`` is a ``tsm.forgetting.Forgotten``). The fields of
        lessons a store written before 2026-11-13 may still hold are swept
        too. Returns how many went."""
        gone = []
        for scope in set(scopes):
            ids = self.engine.ids_where(None, scope) if scope is not None else [
                mid for mid in self.engine.ids_where(None, None)]
            for mid, item in self._items(ids).items():
                if item.get("scope") != scope:
                    continue
                texts = (item.get("text"), item.get("answer"), item.get("note"),
                         item.get("content"), item.get("lesson"), item.get("trigger"))
                if any(forgotten.carries(t) for t in texts if t):
                    gone.append(mid)
        for mid in gone:
            self.engine.delete(mid)
        return len(gone)

    def forget_references(self, memory_ids: Sequence[str]) -> int:
        """Remove ``memory_ids`` from the episodes that list them."""
        gone = set(memory_ids)
        hits = set()
        for mid in gone:
            hits.update(self.engine.ids_where(_filter(_eq("ids", mid))))
        if not hits:
            return 0
        items = self._items(sorted(hits))
        ids, payloads = [], []
        for mid, item in items.items():
            payload = {key: v for key, v in item.items() if key not in ("id", "text", "scope")}
            payload["ids"] = [m for m in payload.get("ids", []) if m not in gone]
            ids.append(mid)
            payloads.append(json.dumps(payload))
        return int(self.engine.set_payloads(ids, payloads))
