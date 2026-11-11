"""Experience: learning from outcomes, and only from outcomes.

The replicated gains from experiential memory (ReasoningBank, ACE) come from
items distilled from what worked and what failed, curated by small
deterministic changes. The measured failures come from the opposite: a
self-generated skill scored below no skill at all (SkillsBench, -1.3), one
wholesale LLM rewrite collapsed a playbook from 18,282 tokens to 122 and its
accuracy below the baseline (ACE), and an attacker who only asks questions
got its own outputs stored in 98% of attempts (MINJA). So this store:

- records an **episode** for every outcome a caller reports
  (``Memory.feedback``): the task, the memories that were in context, the
  outcome, who says so (``source``), and optionally the lesson in words;
- in ``consolidate()`` only (never on a read), curates **items**, each a
  short lesson keyed by the task it applies to, with ``helpful`` /
  ``harmful`` counters and a status: ``probation`` -> ``active`` once it has
  helped twice more often than it hurt, ``deprecated`` once it hurt at
  least twice as often as it helped. Every change is a delta (add an item,
  bump a counter, change a status), written as one batch of payload
  patches; nothing is ever rewritten wholesale;
- trusts an outcome only from ``user``, ``verifier`` or ``judge``. An
  episode with ``source="self"`` (the agent grading itself) is kept, and
  can be retrieved as a past episode, but never creates an item or moves a
  counter;
- returns, for a new task, the active items nearest it (``Memory.lessons``,
  kept apart from the memories and under its own token cap) or the raw
  nearest episodes (``Memory.episodes``: retrieving past experiences as
  they are, ExpRAG, the baseline this has to beat).

A lesson comes from the caller (``lesson=``) or from a ``distiller``
(``(task, outcome, note) -> str``, e.g. ``LLMDistiller``) run on trusted
episodes at consolidation. Without either, an episode only moves the
counters of an item that already matches it.

Episodes and items live in their own engine (``<db_path>/tsm_experience``):
the memories' engine is untouched and no recall changes.

Status (2026-11-11): a prototype with unit tests, not yet measured. Its
evaluation (StreamBench, MemoryAgentBench test-time learning, Evo-Memory, five
stream orders; kill it if it does not beat ExpRAG) is Phase 4 of
``reports/Cognitive architecture for TurboSuperMemory.md``.
"""

import json
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

TRUSTED_SOURCES = frozenset({"user", "verifier", "judge"})
SOURCES = TRUSTED_SOURCES | {"self"}
# An episode updates an existing item when its task is at least this close
# to the item's trigger (embedder-dependent; untuned).
ITEM_MATCH_COSINE = 0.75
# Status rules (counts of trusted outcomes).
PROMOTE_AT = 2
DEPRECATE_AT = 2

DISTILL_SYSTEM = (
    "You turn one outcome of an assistant's work into a reusable lesson. You get "
    "the task, whether the outcome was a success or a failure, and any note about "
    "it. Write one or two sentences (at most 40 words) that would help with a "
    "similar task next time: what to do, or what to avoid. Do not restate the task. "
    "If there is nothing general to learn, write NONE."
)


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


def status_for(helpful: int, harmful: int, current: str) -> str:
    """An item's status from its counters (``PROMOTE_AT``, ``DEPRECATE_AT``)."""
    if harmful >= DEPRECATE_AT and harmful >= 2 * max(helpful, 1):
        return "deprecated"
    if current == "probation" and helpful >= PROMOTE_AT and helpful >= 2 * harmful:
        return "active"
    if current == "deprecated":
        return "deprecated"
    return current


def lesson_block(items: Sequence[Dict], token_cap: int = 200) -> str:
    """The lessons as text for a reader's prompt, best first, within
    ``token_cap`` (about four characters a token); ``""`` without any."""
    lines, used = [], 0
    for item in items:
        line = f"- {item['content']}"
        cost = max(1, len(line) // 4)
        if used + cost > token_cap:
            break
        lines.append(line)
        used += cost
    return "Lessons from earlier tasks:\n" + "\n".join(lines) if lines else ""


class LLMDistiller:
    """``(task, outcome, note) -> lesson`` with a chat model
    (``DISTILL_SYSTEM``); ``""`` when the model finds nothing general."""

    def __init__(self, model: str = "gpt-4o-mini", client=None,
                 base_url: Optional[str] = None, api_key: Optional[str] = None,
                 cache_dir: Optional[str] = None, **chat_kwargs):
        from ._chat import CachedChat

        self._chat = CachedChat(DISTILL_SYSTEM, model=model, client=client,
                                base_url=base_url, api_key=api_key, cache_dir=cache_dir,
                                cache_name="distill", max_tokens=120,
                                what="lesson distillation", **chat_kwargs)

    def forget_texts(self, texts) -> int:
        """Remove cached replies that carry ``texts`` (``Memory.forget``)."""
        return self._chat.forget_texts(texts)

    def __call__(self, task: str, outcome: str, note: Optional[str] = None) -> str:
        reply = self._chat.ask(f"Task: {task}\nOutcome: {outcome}\nNote: {note or '(none)'}")
        text = (reply or "").strip()
        return "" if not text or text.upper().startswith("NONE") else text

    def flush_cache(self) -> None:
        self._chat.flush_cache()


def _filter(*parts: Dict) -> str:
    return json.dumps(parts[0] if len(parts) == 1 else {"op": "and", "filters": list(parts)})


def _eq(field: str, value) -> Dict:
    return {"op": "eq", "field": field, "value": value}


class ExperienceLedger:
    """The experience engine: episodes in, curated items out."""

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
               lesson: Optional[str] = None, at: Optional[str] = None,
               vector=None) -> str:
        """Store one episode; returns its id. ``vector`` is the task's
        embedding when the caller has it (else it is embedded)."""
        if source not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")
        if not (task or "").strip():
            raise ValueError("task must not be empty")
        payload = {"kind": "episode", "status": "new", "outcome": normalize_outcome(outcome),
                   "source": source, "ids": [str(m) for m in memory_ids],
                   "lesson": (lesson or "").strip() or None, "user_id": user_id, "at": at,
                   "seq": int(self.engine.next_insert_seq())}
        vec = (np.asarray(vector, dtype=np.float32)[None, :] if vector is not None
               else np.ascontiguousarray(self._embed([task]), dtype=np.float32))
        mid = self._next_id("ep")
        self.engine.insert_batch([mid], [task], vec, [1.0], [[]], [json.dumps(payload)],
                                 [user_id] if user_id is not None else None, ["episode"])
        return mid

    def curate(self, distiller: Optional[Callable[..., str]] = None,
               match_cosine: float = ITEM_MATCH_COSINE) -> Dict[str, int]:
        """Process the new episodes, oldest first. For a trusted one:

        - each item that was in front of the reader (its id among the
          episode's ``ids``, as ``Memory.lessons`` returned it) is credited:
          ``helpful`` on success, ``harmful`` on failure. An item no one used
          is never blamed for an outcome;
        - its lesson (given, or from ``distiller``) supports the nearest item
          of its scope that says much the same (``helpful`` + 1, provenance
          kept), or else is added as a ``probation`` item that starts with
          its own episode as one helpful outcome.

        Statuses then follow the counters. The changes are one batch of
        payload patches, plus the inserted items."""
        episodes = self._items(self.engine.ids_where(
            _filter(_eq("kind", "episode"), _eq("status", "new"))))
        if not episodes:
            return {"episodes": 0, "added": 0, "updated": 0}
        order = sorted(episodes, key=lambda m: episodes[m].get("seq", 0))
        vectors, live = self.engine.get_vectors(order)
        patches: Dict[str, Dict] = {}
        counters: Dict[str, Dict] = {}   # item id -> its payload as it now stands
        added = updated = 0
        new_items: List[Dict] = []        # inserted this pass

        def credit(item_id: str, key: str, episode_id: str) -> None:
            payload = counters.get(item_id)
            if payload is None:
                payload = {k: v for k, v in self._items([item_id]).get(item_id, {}).items()
                           if k not in ("id", "text", "scope")}
                counters[item_id] = payload
            payload[key] = int(payload.get(key, 0)) + 1
            payload["status"] = status_for(int(payload.get("helpful", 0)),
                                           int(payload.get("harmful", 0)),
                                           payload.get("status", "probation"))
            provenance = payload.setdefault("provenance", [])
            if episode_id not in provenance:
                provenance.append(episode_id)

        for k, mid in enumerate(order):
            ep = episodes[mid]
            patches[mid] = {"status": "curated"}
            if not live[k] or ep.get("source") not in TRUSTED_SOURCES:
                continue
            success = ep.get("outcome") == "success"
            used = [i for i, item in self._items(ep.get("ids") or []).items()
                    if item.get("kind") == "item"]
            for item_id in used:
                credit(item_id, "helpful" if success else "harmful", mid)
                updated += 1
            lesson = ep.get("lesson")
            if not lesson and distiller is not None:
                lesson = distiller(ep["text"], ep.get("outcome"), None) or None
            if not lesson:
                continue
            target = self._nearest_item(vectors[k], ep.get("scope"), match_cosine, new_items)
            if target is not None:
                if target not in used:
                    credit(target, "helpful", mid)
                    updated += 1
                patches[mid]["item"] = target
                continue
            item_id = self._next_id("item")
            payload = {"kind": "item", "status": "probation", "trigger": ep["text"],
                       "content": lesson, "item_kind": "strategy" if success else "warning",
                       "helpful": 1, "harmful": 0, "provenance": [mid],
                       "user_id": ep.get("user_id")}
            self.engine.insert_batch([item_id], [lesson], vectors[k][None, :], [1.0], [[]],
                                     [json.dumps(payload)],
                                     [ep["scope"]] if ep.get("scope") is not None else None,
                                     ["item"])
            new_items.append({"id": item_id, "vector": vectors[k], "scope": ep.get("scope")})
            patches[mid]["item"] = item_id
            added += 1
        for item_id, payload in counters.items():
            patches[item_id] = payload
        ids = list(patches)
        merged = []
        current = self._items(ids)
        for mid in ids:
            base = {key: v for key, v in current.get(mid, {}).items()
                    if key not in ("id", "text", "scope")}
            base.update(patches[mid])
            merged.append(json.dumps(base))
        self.engine.set_payloads(ids, merged)
        return {"episodes": len(order), "added": added, "updated": updated}

    def _nearest_item(self, vector, scope, floor: float,
                      new_items: Sequence[Dict]) -> Optional[str]:
        best, best_score = None, floor
        hits = self.engine.search_ann(vector, 3, scope=scope, filter=_filter(
            _eq("kind", "item"), {"op": "not", "filter": _eq("status", "deprecated")}))
        for mid, score in hits:
            if score >= best_score:
                best, best_score = mid, float(score)
        for item in new_items:  # inserted this pass: already searchable, but be sure
            if item["scope"] != scope:
                continue
            score = float(np.dot(vector, item["vector"]))
            if score >= best_score:
                best, best_score = item["id"], score
        return best

    def lessons(self, vector, user_id: Optional[str] = None, k: int = 3,
                include_probation: bool = False) -> List[Dict]:
        """Items nearest a task's vector, of ``user_id`` or global, best
        first: active ones (and probation ones when asked)."""
        statuses = ["active"] + (["probation"] if include_probation else [])
        flt = _filter(_eq("kind", "item"),
                      {"op": "or", "filters": [_eq("status", s) for s in statuses]})
        hits = self.engine.search_ann(np.asarray(vector, dtype=np.float32), max(4 * k, 12),
                                      filter=flt)
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

    def episodes(self, vector, user_id: Optional[str] = None, k: int = 3) -> List[Dict]:
        """The past episodes nearest a task's vector, of ``user_id`` or
        global, best first (raw experiences, whatever their source)."""
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
        """Delete the episodes and lessons of ``scopes`` (``None`` is the
        global scope) whose task, lesson or trigger carries forgotten content
        (``forgotten`` is a ``tsm.forgetting.Forgotten``). Returns how many
        went."""
        gone = []
        for scope in set(scopes):
            ids = self.engine.ids_where(None, scope) if scope is not None else [
                mid for mid in self.engine.ids_where(None, None)]
            for mid, item in self._items(ids).items():
                if item.get("scope") != scope:
                    continue
                texts = (item.get("text"), item.get("content"), item.get("lesson"),
                         item.get("trigger"))
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
