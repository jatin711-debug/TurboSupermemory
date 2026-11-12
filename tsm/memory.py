"""``tsm.Memory`` — the shipped conversational-memory facade.

A thin, pure-Python layer over the compiled ``turbomemory`` engine that
packages the mechanisms proven in the cognitive evaluations as a one-flag
preset (``profile="conversational"``):

  - role-tagged, scope-filtered fact storage (user-scoped memories),
  - belief revision with refinement/contradiction thresholds 0.85 / 0.75,
  - cognitive search, calibrated: the concept graph may move a memory by up
    to half the spread of similarity among the candidates, enough to settle
    close calls and never enough to outvote the embedding,
  - belief revision that keeps history: a superseded fact is ranked lower
    and flagged, never removed. With a ``Verifier`` (consolidation proposes,
    the verifier vets, only accepted pairs are committed) recall also marks
    it as earlier in the text handed to a model and brings the fact that
    replaced it in beside it,
  - access-aware eviction and importance auto-scoring,
  - concept extraction (bigram ngrams) for the memory graph,
  - MMR best-set recall under a token budget,
  - results that say when each memory is from (a date or a turn in front of
    the text handed to a model),
  - the messages themselves kept next to the facts (``keep_messages``), so
    what the extractor missed can still be found,
  - optionally, each user's memory kept under a token budget
    (``max_user_tokens``): the newest facts stay as they are and everything
    older is folded into a few short gists.

Pluggable backends: pass any ``Embedder`` / ``Extractor`` / ``Verifier``
(see ``tsm.interfaces``) to use local models or other APIs. Defaults are
OpenAI-backed and read the key from ``OPENAI_API_KEY``.
"""

import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ._loader import load_turbomemory
from .budget import MAX_REDUNDANCY, NEW_TURN_BONUS, select_under_budget, total_tokens
from .chunks import is_message, message_role, occasion, passages, speaker, spoken_by
from .compaction import POLICIES, is_gist, plan_compaction
from .concepts import extract_concepts
from .identity import bind_embedder
from .interfaces import (  # noqa: F401  (Embedder, Extractor, Verifier: re-exported types)
    Embedder, Extractor, Verifier, embed_documents, embed_query, embedder_fingerprint)
from .experience import ExperienceLedger
from .forgetting import Forgotten, redact
from .intentions import IntentionLedger, detect_closure, detect_intentions
from .ranking import apply_role_prior, asks_what_assistant_said, packing_lam
from .when import as_date, question_windows, said_on

logger = logging.getLogger("tsm.memory")

# Put in front of a superseded memory's text in a result's "context" when its
# supersession was verified: it was true, and something newer replaced it.
STALE_MARK = "[earlier, since changed] "
# The same when the date of the memory that replaced it is known: the change
# happened by then (it may have happened earlier and been mentioned then).
STALE_BY = "[earlier, changed by {date}] "


def stale_mark(valid_until) -> str:
    """What goes in front of a verified-superseded memory: ``STALE_BY`` with
    the date its replacement was said, or ``STALE_MARK`` without one."""
    until = said_on(valid_until)
    return STALE_BY.format(date=f"{until:%Y-%m-%d}") if until else STALE_MARK


def span_tag(span) -> str:
    """When the memories a gist summarizes were said: ``"[2023-01-05 to
    2023-03-02] "``, one date when they were all said on one day, ``""``
    without a span."""
    if not span or len(span) != 2:
        return ""
    first, last = said_on(span[0]), said_on(span[1])
    if first is None or last is None:
        return ""
    return f"[{first:%Y-%m-%d}] " if first == last else f"[{first:%Y-%m-%d} to {last:%Y-%m-%d}] "


def _span_of(payload) -> Optional[Tuple[str, str]]:
    """A stored memory's span for compaction: a gist's own, or the day a fact
    was said."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload) if payload else {}
        except ValueError:
            payload = {}
    if not isinstance(payload, dict):
        return None
    span = payload.get("span")
    if isinstance(span, (list, tuple)) and len(span) == 2 and all(said_on(s) for s in span):
        return (f"{said_on(span[0]):%Y-%m-%d}", f"{said_on(span[1]):%Y-%m-%d}")
    day = said_on(payload.get("timestamp"))
    return (f"{day:%Y-%m-%d}",) * 2 if day else None


def _level_of(payload) -> int:
    """A stored gist's merge level (0 for anything else)."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload) if payload else {}
        except ValueError:
            return 0
    level = payload.get("level") if isinstance(payload, dict) else 0
    return int(level) if isinstance(level, int) and level >= 0 else 0

# A date, then optionally a weekday in brackets and a time of day:
# "2024-01-15", "2024-01-15T10:00:00Z", "2023/05/20 (Sat) 02:21".
_DATE = re.compile(
    r"\s*(\d{4})[-/](\d{2})[-/](\d{2})(?:\s*\(\w+\))?(?:[T\s]\s*(\d{1,2}):(\d{2}))?")


def time_tag(timestamp, turn_index, minutes: bool = False) -> str:
    """When a memory was said, as a prefix for the text shown to a model.

    ``"[2024-01-15] "`` when the message's timestamp carries a date (a string
    starting ``YYYY-MM-DD``, or epoch seconds or milliseconds); otherwise
    ``"[turn 12] "`` from the turn index, which only orders memories; ``""``
    when neither is known. Without one of these a reader cannot tell which of
    two memories came first.

    With ``minutes`` the time of day is added when the timestamp has one
    (``"[2024-01-15 10:00] "``, UTC for an epoch timestamp), which tells two
    memories of one day apart for two more tokens each.
    """
    if isinstance(timestamp, str):
        match = _DATE.match(timestamp)
        if match:
            year, month, day, hour, minute = match.groups()
            if minutes and hour is not None:
                return f"[{year}-{month}-{day} {int(hour):02d}:{minute}] "
            return f"[{year}-{month}-{day}] "
        try:
            timestamp = float(timestamp)
        except ValueError:
            timestamp = None
    if (isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool)
            and math.isfinite(timestamp) and timestamp >= 1e8):
        seconds = timestamp / 1000.0 if timestamp >= 1e11 else float(timestamp)
        try:
            return datetime.fromtimestamp(seconds, timezone.utc).strftime(
                "[%Y-%m-%d %H:%M] " if minutes else "[%Y-%m-%d] ")
        except (OverflowError, OSError, ValueError):
            pass
    if (isinstance(turn_index, (int, float)) and not isinstance(turn_index, bool)
            and math.isfinite(turn_index) and turn_index >= 0):
        return f"[turn {int(turn_index)}] "
    return ""


# How long close() waits for an idle consolidation pass that is still running
# (a verifier request can take a minute) before it releases the engine anyway.
IDLE_CLOSE_TIMEOUT_SECS = 30.0
# The least an idle pass waits before trying again after it failed.
IDLE_RETRY_FLOOR_SECS = 1.0

# Asked what the assistant said, recall() returns the reply it found whole
# (its passages joined in order), in at most this share of the token budget.
REPLY_SHARE = 0.75
# How many of the best-scored replies come back whole.
REPLIES_WHOLE = 2
# How many ids after a passage are read to find the rest of its message when
# the passage was stored without the message's passage count.
REPLY_WINDOW = 32

# A question that points at a time ("ten days ago", "last week") given
# recall(now=...): the memories said then take up to this share of the budget,
# the best of them packed before the rest.
WINDOW_SHARE = 0.5
# How many of them are offered to that packing, best-scored first.
WINDOW_POOL = 30
# How many of the user's memories nearest the question are looked through for
# them (the memories of a few days are a small part of a long history).
WINDOW_SCAN = 1000
# A thing is mentioned the day it happens or soon after ("I got it
# yesterday"): a window ends this many days after the time it points to.
WINDOW_LAG_DAYS = 3
# ... and starts a day before it: the clock that dates a question and the one
# that dated a message need not agree on the day (time zones).
WINDOW_LEAD_DAYS = 1

# Engine methods this SDK cannot work without. The extension and the SDK ship
# together; this guards against a stale locally-built turbomemory.pyd/.so.
_REQUIRED_ENGINE_METHODS = ("get_records", "next_insert_seq", "recovery_report")
# ... and those the executive's stores (intentions, experience), retract() and
# forget() need: payload patches, lookups by payload, filtered vector search.
_EXECUTIVE_ENGINE_METHODS = ("set_payloads", "ids_where", "get_vectors",
                             "retract_supersessions")

# The proven conversational configuration, from the evaluation wins. Every key
# is a MemoryEngine kwarg; explicit engine_kwargs passed to Memory() override
# these. `defer_supersession_commit` is added by Memory depending on whether a
# verifier is installed. Superseded facts stay in recall (pass
# `exclude_superseded=True` to drop them): a question about what was true
# earlier needs them, and removing them lost more judged answers than it won.
#
# Cognitive search is weighted against the query's own spread of cosine
# (`graph_weight`), without the per-word concept bonus (`query_concept_boost`:
# BM25 already scores word overlap) and with what a concept passes on split
# among the memories it points at (`fan_exponent`). The profile used a fixed
# boost before (`cognitive_alpha=0.5`, up to 0.5 on top of a cosine, where
# neighbouring candidates differ by about 0.01): that ranked by word overlap
# and lost questions against plain vector search. Passing `cognitive_alpha`
# to Memory() brings the fixed boost back.
CONVERSATIONAL_PROFILE = {
    "refinement_cosine_threshold": 0.85,
    "contradiction_cosine_threshold": 0.75,
    "graph_weight": 0.5,
    "query_concept_boost": 0.0,
    "fan_exponent": 1.0,
    "importance_auto_scoring": True,
    "concept_max_ngram_len": 2,
    "max_concepts": 10,
    "belief_source_roles": ["user"],     # only user-sourced facts supersede
    "access_aware_eviction": True,
    "auto_consolidation_secs": 0,        # manual consolidation (deterministic)
}

class Memory:
    """Scoped, self-correcting conversational memory.

    Usage::

        with Memory("./my_db") as mem:                # OPENAI_API_KEY required
            mem.add([{"role": "user", "content": "I moved to Lisbon."}],
                    user_id="alice")
            results = mem.recall("Where does Alice live?", user_id="alice")
            mem.consolidate()                          # verified belief revision

    Durability: the engine is the only store. Text, role, and scope are
    read back from it at recall time and ids are minted from its durable
    insert sequence, so a database behaves the same after it is reopened —
    in this process or another — as it did when it was written: ``add``
    keeps appending, the role prior still applies, and a verifier installed
    later can vet facts stored by an earlier session.

    Scope guard: ``recall`` drops any hit whose stored scope is neither
    ``user_id`` nor global (unscoped). This is belt-and-suspenders over the
    engine's own scope filter.

    ``close()`` releases the database (lock, mmaps, worker threads); the
    same ``db_path`` can be reopened immediately afterwards.
    """

    def __init__(
        self,
        db_path: str,
        dimension: Optional[int] = None,
        profile: Optional[str] = "conversational",
        embedder: Optional[Embedder] = None,
        extractor: Optional[Extractor] = None,
        verifier: Optional[Verifier] = None,
        gist_summarizer: Optional[Callable[[List[str]], str]] = None,
        max_user_tokens: Optional[int] = None,
        compaction: str = "refold",
        keep_messages: Optional[bool] = None,
        allow_embedder_change: bool = False,
        idle_consolidation_secs: Optional[float] = None,
        intentions: bool = False,
        experience: bool = False,
        readiness_judge=None,
        distiller=None,
        topics=None,
        intention_tracker=None,
        **engine_kwargs,
    ):
        """
        Args:
            db_path: Database directory for the engine.
            dimension: Embedding dimension. Defaults to the embedder's
                ``dimension``, else 1536 (OpenAI text-embedding-3-small).
            profile: ``"conversational"`` applies the proven preset;
                ``None`` leaves engine defaults (plain vector store).
                Explicit ``engine_kwargs`` override the profile.
            embedder: ``Embedder`` implementation. Default: ``OpenAIEmbedder``.
                ``"gemma"`` is EmbeddingGemma 2 run locally (text model,
                768 dimensions, no API key); ``"local"`` / ``"minilm"`` is
                all-MiniLM-L6-v2. An embedder with ``encode_query`` /
                ``encode_document`` is asked for the right one of the two
                (``tsm.interfaces.AsymmetricEmbedder``). The database
                remembers which embedder wrote it (``tsm_embedder.json``)
                and refuses another: two models of one dimension are not
                comparable and search would return wrong results silently.
            extractor: ``Extractor`` implementation. Default: ``OpenAIExtractor``.
                ``"none"`` extracts nothing: with ``keep_messages`` the store
                then holds the messages as said, and ``add`` asks no model.
            verifier: ``Verifier`` implementation, or ``"nli"`` /``"llm"``
                for the two in ``tsm.verification`` (``NLIVerifier``: a small
                local cross-encoder; ``LLMVerifier``: a chat model, far more
                accurate, one short request per few candidate pairs). When
                installed, ``consolidate()`` runs propose -> verify -> commit.
                A superseded fact always stays in recall, ranked lower and
                flagged (``superseded_by``). With a verifier its supersession
                is trusted enough to act on: ``recall()`` marks it as earlier
                in ``"context"`` and brings in the fact that replaced it.
                Pass ``exclude_superseded=True`` to remove superseded facts
                from recall instead.
            gist_summarizer: optional callable mapping a list of evicted fact
                texts to a single gist string (typically an LLM call); the
                ones in ``tsm.gist`` are also told how long the gist may be.
            max_user_tokens: keep each user's memory under this many tokens.
                ``consolidate()`` then compacts the users written to since
                the last pass: the newest facts stay as they are (what the
                user said before what the assistant said) and everything
                older is folded into a few short gists by
                ``gist_summarizer``, earlier gists included, so the store
                stays within the budget however long the history grows.
                Without a summarizer the older facts are simply removed.
                ``compaction="merge_oldest"`` keeps earlier gists as they are
                and merges only the two oldest when the gists outgrow their
                share (``tsm.compaction.plan_compaction``); the default
                ``"refold"`` summarizes every earlier gist again on each pass.
                This is per user and by tokens; the engine's ``max_records``
                is one count for the whole store and keeps whatever is
                newest.
            keep_messages: also store every message as it was said, cut
                into passages of about two sentences (``tsm.chunks``), each a
                memory of its own beside the facts taken from the message.
                A message the extractor took nothing from can then still be
                found, and ``recall`` shows a passage with who said it
                (``[2024-01-15] user: ...``). Passages are stored under the
                roles ``"message"`` / ``"message-<role>"``, which belief
                revision leaves alone and the role prior ranks like their
                speaker. They cost no model call, only an embedding each.
                Default (None): on, unless ``max_user_tokens`` is set
                (compaction does not fold passages yet). On LongMemEval-S
                (29 questions, judged) it answered 27 where the facts alone
                answered 19 at 2,000 tokens of context, and 25 against 20 at
                1,200 (``e79d75f:benchmarks/PHASE_PROGRESS.md``, 2026-10-29). With
                ``extractor="none"`` it is the whole store: 26 of the 29.
            allow_embedder_change: open a database with an embedder other
                than the one that wrote it (``EmbedderMismatchError``
                otherwise). Only for an embedder that produces the same
                vectors as the recorded one; the record is then replaced.
            idle_consolidation_secs: run ``consolidate()`` in a background
                thread once nothing has called ``add()`` or ``recall()`` for
                this many seconds and something was written since the last
                pass (also once after opening a store that already holds
                memories). Verification, supersession commits and
                ``max_user_tokens`` compaction then happen away from the
                request path. A pass is not interrupted when a call arrives:
                the engine serves it meanwhile, and only a manual
                ``consolidate()`` waits for the pass. A failed pass, or one
                a verifier could not finish (``verifier.unresolved``), is
                tried again after another idle period. ``close()`` stops the
                thread first. Default ``None``: consolidate when you call it.
            intentions: keep what the user asks to be reminded of or commits
                to doing at a future time (``tsm.intentions``), in an engine
                of its own under ``db_path``: ``add()`` forms them from user
                messages, ``consolidate()`` closes and replaces them, and
                ``due()`` says which have come due. Changes no recall result.
                Off by default (a prototype, not yet measured).
            experience: keep the outcomes ``feedback()`` reports and curate
                lessons from them (``tsm.experience``), in an engine of its
                own under ``db_path``; ``lessons()`` returns those that apply
                to a task. Changes no recall result. Off by default (a
                prototype, not yet measured).
            readiness_judge: what ``due()`` asks to judge its candidates
                (``tsm.intentions.LLMReadinessJudge``, or ``"llm"``).
                ``None``: due by time is "act", brought to mind by the
                context is "observe".
            distiller: turns a trusted outcome without a lesson into one at
                consolidation (``tsm.experience.LLMDistiller``, or ``"llm"``).
            topics: also offer the verifier facts that give a value for the
                same property (``user.residence``), however differently they
                are worded (``tsm.topics.LLMTopicKeyer``, or ``"llm"``). Needs
                a verifier that judges meaning (``verifier="llm"``). Facts are
                keyed at ``consolidate()``, about 10 to a request, once each.
                Off by default (not yet judged on answers).
            intention_tracker: with ``intentions``, form, close and
                reschedule intentions with a chat model instead of the
                rules (``tsm.intentions.LLMIntentionTracker``, or
                ``"llm"``): ``add()`` keeps each user message in the
                intentions store and ``consolidate()`` reads the new ones
                with the user's open intentions, about 8 messages to a
                request. Held out it found 34 of 53 due items against 6
                for the rules, but with 13 wrong against 8 for plain
                ``recall()`` read by a model (``LEARNINGS.md``).
            **engine_kwargs: forwarded to ``turbomemory.MemoryEngine``.
        """
        # Backend names are resolved here; anything else that is a string is a
        # typo, and is reported now rather than at the first add().
        for kind, value, known in (
            ("embedder", embedder, ("openai", "sentence_transformer", "local", "minilm", "gemma")),
            ("extractor", extractor, ("openai", "gliner", "passthrough", "none")),
            ("verifier", verifier, ("nli", "llm")),
            ("readiness_judge", readiness_judge, ("llm",)),
            ("distiller", distiller, ("llm",)),
            ("topics", topics, ("llm",)),
            ("intention_tracker", intention_tracker, ("llm",)),
        ):
            if isinstance(value, str) and value not in known:
                raise ValueError(
                    f"unknown {kind} {value!r}: use one of {', '.join(known)} "
                    f"or pass an instance")
        if embedder == "openai":
            embedder = None
        cache_dir = None
        if embedder is None or extractor is None or extractor == "openai":
            cache_dir = os.path.join(db_path, "tsm_cache")
        if embedder is None:
            from .embedders import OpenAIEmbedder

            embedder = OpenAIEmbedder(cache_dir=cache_dir)
        elif embedder in ("sentence_transformer", "local", "minilm"):
            from .embedders import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder()
        elif embedder == "gemma":
            from .embedders import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder("google/embeddinggemma-2")
        if extractor is None or extractor == "openai":
            from .extractors import OpenAIExtractor

            extractor = OpenAIExtractor(cache_dir=cache_dir)
        elif extractor == "gliner":
            from .extractors import GlinerExtractor

            extractor = GlinerExtractor(cache_dir=cache_dir)
        elif extractor == "passthrough" or extractor is False:
            from .extractors import PassthroughExtractor

            extractor = PassthroughExtractor()
        elif extractor == "none":
            from .extractors import NullExtractor

            extractor = NullExtractor()
        if verifier == "nli":
            from .verification import NLIVerifier

            verifier = NLIVerifier()
        elif verifier == "llm":
            from .verification import LLMVerifier

            verifier = LLMVerifier(cache_dir=os.path.join(db_path, "tsm_cache"))
        if readiness_judge == "llm":
            from .intentions import LLMReadinessJudge

            readiness_judge = LLMReadinessJudge(cache_dir=os.path.join(db_path, "tsm_cache"))
        if distiller == "llm":
            from .experience import LLMDistiller

            distiller = LLMDistiller(cache_dir=os.path.join(db_path, "tsm_cache"))
        if topics is not None and getattr(verifier, "candidate_min_cosine", None) is None:
            # The keys only propose; a verifier that cannot judge meaning
            # would be handed whatever two facts share a key.
            raise ValueError('topics needs a verifier that judges meaning (verifier="llm")')
        if topics == "llm":
            from .topics import LLMTopicKeyer

            topics = LLMTopicKeyer(cache_dir=os.path.join(db_path, "tsm_cache"))
        self.topic_keyer = topics
        if intention_tracker is not None and not intentions:
            raise ValueError("intention_tracker needs intentions=True")
        if intention_tracker == "llm":
            from .intentions import LLMIntentionTracker

            intention_tracker = LLMIntentionTracker(cache_dir=os.path.join(db_path, "tsm_cache"))
        self.intention_tracker = intention_tracker
        # Scopes with facts not yet keyed (None: every scope, on the first
        # pass after opening).
        self._topic_scopes: Optional[set] = None
        self.readiness_judge = readiness_judge
        self.distiller = distiller
        self.embedder = embedder
        self.extractor = extractor
        self.verifier = verifier
        # Supersessions a verifier accepted are reliable enough to show a
        # model; the engine's unchecked ones stay metadata (on its own the
        # detector also fires on facts that are both still true, "my sister
        # lives in Vancouver" / "my brother lives in Vancouver").
        self._verified = verifier is not None

        self.dim = int(dimension or getattr(embedder, "dimension", None) or 1536)

        config: Dict = {}
        if profile == "conversational":
            config.update(CONVERSATIONAL_PROFILE)
            # Cosine thresholds depend on the embedder's geometry (see
            # tsm.interfaces.Embedder); explicit engine_kwargs still win.
            config.update(getattr(embedder, "engine_defaults", None) or {})
        elif profile is not None:
            raise ValueError(f"unknown profile: {profile!r} (use 'conversational' or None)")
        # A verifier only gets to vet supersessions if the engine does not
        # commit them first, whatever the profile.
        config["defer_supersession_commit"] = verifier is not None
        if gist_summarizer is not None:
            config["gist_before_evict"] = True
        if "cognitive_alpha" in engine_kwargs and "graph_weight" not in engine_kwargs:
            # A caller who sets the fixed boost means it: the profile's
            # spread-scaled weight would otherwise take its place.
            config.pop("graph_weight", None)
        config.update(engine_kwargs)  # explicit kwargs win over the profile
        self.profile = profile
        self.engine_config = dict(config)  # what the engine was opened with
        self._gist_summarizer = gist_summarizer
        if max_user_tokens is not None and int(max_user_tokens) < 16:
            raise ValueError("max_user_tokens must be at least 16")
        self.max_user_tokens = None if max_user_tokens is None else int(max_user_tokens)
        if compaction not in POLICIES:
            raise ValueError(f"compaction must be one of {POLICIES}")
        self.compaction = compaction
        if keep_messages is None:
            keep_messages = self.max_user_tokens is None
        self.keep_messages = bool(keep_messages)
        if not self.keep_messages and getattr(self.extractor, "stores_nothing", False):
            raise ValueError('extractor="none" stores nothing without keep_messages')
        if self.keep_messages and self.max_user_tokens is not None:
            # Compaction folds by speaker and does not know passages yet: it
            # would summarize the user's own messages with the assistant's.
            raise ValueError("keep_messages cannot be combined with max_user_tokens yet")
        if idle_consolidation_secs is not None and not float(idle_consolidation_secs) > 0:
            raise ValueError("idle_consolidation_secs must be positive")
        self._idle_secs = (None if idle_consolidation_secs is None
                           else float(idle_consolidation_secs))
        # Idle consolidation: one pass at a time (also against a manual
        # consolidate()), and what decides whether the worker has work.
        self._maintenance_lock = threading.Lock()
        self._activity = time.monotonic()
        self._pending = False       # written to since the last pass
        self._retry = False         # the last pass left something to try again
        self._retry_after = 0.0
        self._stop_worker = threading.Event()
        self._worker: Optional[threading.Thread] = None
        # Users written to since their last compaction (this process only;
        # compact() with no argument covers every user in the store).
        self._touched: set = set()

        # The executive's own engines (opened below when enabled); set first
        # so close() works whatever fails during construction.
        self._intentions: Optional[IntentionLedger] = None
        self._experience: Optional[ExperienceLedger] = None
        self._experience_lock = threading.Lock()
        turbomemory = load_turbomemory()
        self.engine = turbomemory.MemoryEngine(
            db_path=db_path, dimension=self.dim, **config
        )
        self._closed = False
        # One writer at a time: id minting and the insert it feeds must not
        # interleave between threads sharing this Memory.
        self._write_lock = threading.Lock()
        required = _REQUIRED_ENGINE_METHODS + (
            ("scope_ids", "scopes") if self.max_user_tokens is not None else ()) + (
            _EXECUTIVE_ENGINE_METHODS if intentions or experience else ()) + (
            ("set_payloads", "get_vectors", "scope_ids", "scopes", "superseded_ids")
            if self.topic_keyer is not None else ())
        missing = [m for m in required if not hasattr(self.engine, m)]
        if missing:
            self.close()
            raise RuntimeError(
                "the compiled turbomemory extension is older than this tsm SDK "
                f"(missing: {', '.join(missing)}); rebuild it with 'make build-python'"
            )
        try:
            bind_embedder(db_path, embedder_fingerprint(embedder, self.dim), self.dim,
                          records=int(self.engine.record_count()),
                          allow_change=allow_embedder_change)
        except BaseException:
            self.close()  # do not leave the database locked behind the error
            raise
        if gist_summarizer is not None:
            set_compressor = getattr(self.engine, "set_gist_compressor", None)
            if set_compressor is None:
                self.close()  # do not leave the database locked behind the error
                raise ValueError(
                    "gist_summarizer requires an engine build with gist-before-evict "
                    "support (set_gist_compressor); rebuild the turbomemory extension"
                )
            set_compressor(self._compress_gist)

        try:
            # Derived records go in engines of their own: an engine's word
            # index and concept graph are shared by its scopes, so anything
            # stored beside the memories changes what recall returns.
            if intentions:
                self._intentions = IntentionLedger(
                    turbomemory.MemoryEngine(db_path=os.path.join(db_path, "tsm_intentions"),
                                             dimension=self.dim), self._embed_documents)
            if experience:
                self._experience = ExperienceLedger(
                    turbomemory.MemoryEngine(db_path=os.path.join(db_path, "tsm_experience"),
                                             dimension=self.dim), self._embed_documents)
        except BaseException:
            self.close()  # do not leave the databases locked behind the error
            raise

        # Ids are `{user_id}_{n}`. Seeding n from the engine's durable insert
        # sequence (which starts at 1 and is never reused) keeps ids unique
        # when an existing database is reopened.
        self._insert_counter = max(0, int(self.engine.next_insert_seq()) - 1)

        if self._idle_secs is not None:
            # A store written by an earlier session may never have been
            # consolidated: the first idle period does it.
            self._pending = int(self.engine.record_count()) > 0
            self._worker = threading.Thread(target=self._idle_loop,
                                            name="tsm-idle-consolidation", daemon=True)
            self._worker.start()

    # writes ----------------------------------------------------------------------
    def add(self, messages: List[Dict], user_id: str) -> int:
        """Extract facts from conversation messages and store them.

        Args:
            messages: list of ``{"role": ..., "content": ...}`` dicts
                (optional ``"timestamp"`` and ``"turn_index"`` are carried
                into the payload). Facts extracted from one message share a
                turn index, which budget recall uses to spread its selection
                across turns; when the caller gives none, one is assigned.
            user_id: scope the facts are stored under (recall is scoped too).

        Returns:
            The number of memories stored: facts, and with ``keep_messages``
            the passages of the messages as well. Exact-text duplicates
            within the same batch are skipped (write gate); cross-batch
            near-duplicates are the engine's job (dedup config / belief
            revision).

        The facts of one call are stored as a single batch that the engine
        validates before writing anything, so a call either stores all of its
        facts or raises having stored none. Safe to call from several threads.
        """
        self._require_open()
        self._activity = time.monotonic()
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]

        facts: List[str] = []
        metas: List[Dict] = []
        seen_in_batch = set()
        context: List[str] = []
        for turn, msg in enumerate(messages):
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            role = msg.get("role", "user")
            said = [(fact, None) for fact in self.extractor.extract_facts(content, context)]
            parts = passages(content) if self.keep_messages else []
            said += [(text, n) for n, text in enumerate(parts)]
            for fact, passage in said:
                norm = " ".join(fact.lower().split())
                if not norm or norm in seen_in_batch:
                    continue  # write gate: exact duplicate within this batch
                seen_in_batch.add(norm)
                facts.append(fact)
                metas.append({
                    "role": role if passage is None else message_role(role),
                    "timestamp": msg.get("timestamp", ""),
                    "content": content,
                    "turn": turn,
                    "turn_index": msg.get("turn_index"),
                    "passage": passage,
                    "passages": len(parts),
                })
            context.append(content)

        if not facts:
            if self._intentions is not None:
                with self._write_lock:
                    self._form_intentions(messages, {}, user_id)
                    self._pending = True
            return 0

        embeddings = np.ascontiguousarray(embed_documents(self.embedder, facts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape != (len(facts), self.dim):
            raise ValueError(
                f"embedder returned shape {embeddings.shape} for {len(facts)} facts; "
                f"expected ({len(facts)}, {self.dim})"
            )
        with self._write_lock:
            assigned_turns: Dict[int, int] = {}
            ids: List[str] = []
            payloads: List[str] = []
            for meta in metas:
                ids.append(self._next_id(user_id))
                # A caller-supplied turn index wins. Otherwise the turn is keyed
                # by the sequence number of its first stored fact: unique per
                # message and durable across restarts, like the ids themselves.
                turn_index = meta["turn_index"]
                if turn_index is None:
                    turn_index = assigned_turns.setdefault(meta["turn"], self._insert_counter)
                payload = {
                    "timestamp": meta["timestamp"],
                    "role": meta["role"],
                    "user_id": user_id,
                    "turn_index": turn_index,
                }
                if meta["passage"] is None:
                    payload["original_message"] = meta["content"]
                else:
                    # A passage is the message: its place in it is enough.
                    payload["passage"] = meta["passage"]
                    payload["passages"] = meta["passages"]
                payloads.append(json.dumps(payload))
            self.engine.insert_batch(
                ids,
                facts,
                embeddings,
                [1.0] * len(facts),
                [extract_concepts(fact) for fact in facts],
                payloads,
                [user_id] * len(facts) if user_id is not None else None,
                [str(meta["role"]) for meta in metas],
            )
            if self.max_user_tokens is not None and user_id is not None:
                self._touched.add(user_id)
            if self._topic_scopes is not None and user_id is not None:
                self._topic_scopes.add(user_id)
            if self._intentions is not None:
                by_turn: Dict[int, List[str]] = {}
                for mid, meta in zip(ids, metas):
                    by_turn.setdefault(meta["turn"], []).append(mid)
                self._form_intentions(messages, by_turn, user_id)
            self._pending = True
        self._activity = time.monotonic()
        return len(facts)

    def _embed_documents(self, texts: List[str]) -> np.ndarray:
        """``texts`` as document vectors, checked for shape."""
        vectors = np.ascontiguousarray(embed_documents(self.embedder, list(texts)),
                                       dtype=np.float32)
        if vectors.shape != (len(texts), self.dim):
            raise ValueError(f"embedder returned shape {vectors.shape} for {len(texts)} texts; "
                             f"expected ({len(texts)}, {self.dim})")
        return vectors

    def _form_intentions(self, messages: List[Dict], by_turn: Dict[int, List[str]],
                         user_id: Optional[str]) -> None:
        """Store the intentions and closure statements in the user messages
        of one ``add()`` (``tsm.intentions``), linked to the memories stored
        from the same message. Called under the write lock, after the
        memories are stored: a failure here is logged and leaves the
        memories as they are (``add`` stores all of its facts or none, and
        has already stored them)."""
        for turn, msg in enumerate(messages):
            content = (msg.get("content") or "").strip()
            if not content or msg.get("role", "user") != "user":
                continue
            stamp = msg.get("timestamp")
            said_at = stamp if as_date(stamp) else datetime.now().date().isoformat()
            sources = by_turn.get(turn, [])
            try:
                if self.intention_tracker is not None:
                    # Read by the tracker at consolidation, never here.
                    self._intentions.record_inbox(user_id, content, sources, said_at)
                    continue
                drafts = detect_intentions(content, said_at)
                if drafts:
                    self._intentions.record(user_id, drafts, sources, said_at)
                closure = detect_closure(content)
                if closure:
                    self._intentions.record_closure(user_id, closure[0], closure[1],
                                                    sources, said_at)
            except Exception as e:  # noqa: BLE001 — the memories are stored; see above
                logger.warning("intentions of a message not stored: %s: %s",
                               type(e).__name__, e)

    def _compress_gist(self, texts: List[str]):
        """Gist-compressor callback handed to the engine (B4).

        Summarizes one chunk of eviction victims with the user-supplied
        ``gist_summarizer`` and embeds the gist with this Memory's embedder.
        Returns ``(gist_text, embedding)``, or ``None`` when the summarizer
        produced nothing (it found nothing worth keeping, so the chunk is
        dropped without a gist).

        A failure is NOT an abstention: if the summarizer or the embedder
        raises, the exception is passed on to the engine, which then keeps
        this chunk's memories and tries again on the next eviction. Swallowing
        it would delete them with no gist in their place.
        """
        try:
            gist = (self._gist_summarizer(texts) or "").strip()
            if not gist:
                return None
            emb = np.asarray(embed_documents(self.embedder, [gist]), dtype=np.float32)[0]
            return gist, emb.tolist()
        except Exception as e:  # noqa: BLE001 — logged here, handled by the engine
            logger.warning("gist summarizer failed for %d texts (kept, will retry): %s",
                           len(texts), e)
            raise

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Memory is closed")

    def _next_id(self, user_id: Optional[str]) -> str:
        """Mint a ``{user_id}_{n}`` id that is not live in the engine."""
        while True:
            self._insert_counter += 1
            memory_id = (f"{user_id}_{self._insert_counter}" if user_id
                         else f"mem_{self._insert_counter}")
            # Only reachable for ids this SDK did not mint (e.g. records
            # written straight through the engine); costs one lookup per fact.
            if not self.engine.contains_id(memory_id):
                return memory_id

    def _records(self, ids: List[str]) -> Dict[str, Dict]:
        """Stored record metadata for ``ids``, read back from the engine.

        Maps id -> ``{"text", "scope", "source_role", "payload", ...}`` with
        ``payload`` parsed to a dict. Ids that are no longer live are omitted.
        """
        out: Dict[str, Dict] = {}
        for rec in self.engine.get_records(list(ids)):
            if rec is None:
                continue
            try:
                rec["payload"] = json.loads(rec["payload"]) if rec["payload"] else {}
            except (TypeError, ValueError):
                rec["payload"] = {}
            out[rec["id"]] = rec
        return out

    # reads -----------------------------------------------------------------------
    def recall(
        self,
        query: str,
        user_id: str,
        token_budget: Optional[int] = None,
        top_k: int = 10,
        pool_k: int = 20,
        lam: Optional[float] = None,
        resolve_beliefs: bool = True,
        max_items: Optional[int] = None,
        time_tags: Union[bool, str] = True,
        now=None,
        search: str = "cognitive",
    ) -> List[Dict]:
        """Search memories under ``user_id``'s scope.

        Returns a list of dicts, best first, each with at least ``"id"``,
        ``"text"``, ``"context"`` (the text to put in a prompt: ``"text"``
        with when it was said and, see below, whether it has since changed
        in front), ``"score"``, ``"role"`` (the stored source role),
        ``"turn_index"`` and ``"timestamp"``. With ``token_budget`` set,
        the result is instead the best *set* that fits the budget
        (``tsm.budget.select_under_budget``: greedy MMR over a candidate
        pool), in selection order. The pool is ``pool_k`` candidates or more:
        it grows with the budget so a large budget can be filled.
        ``max_items`` limits the number of results as well; by default only
        the token budget does. ``lam`` weighs relevance against variety in
        that choice. By default it is 0.7, and 0.9 for a question that counts
        or adds up ("how many", "in total"), whose answer needs every mention
        of the thing, however alike they read (``tsm.ranking.packing_lam``).

        With ``resolve_beliefs`` (default True), a memory that has been
        superseded stays in the results, ranked lower, with
        ``"superseded_by"`` (the current belief's id) and ``"chain"`` (the
        full supersession chain, oldest first, head last). A result without
        ``"superseded_by"`` is current. When a verifier is installed, two
        more things happen, because its supersessions can be relied on:
        the superseded memory's ``"context"`` starts with ``STALE_MARK``, so
        a model reading it knows it was true earlier and has since changed;
        and the memory that replaced it is ranked where the old one would
        have been without its demotion, and added if the search did not find
        it at all (an update is often worded differently from the fact it
        replaces, so a question in the old wording finds only the old fact).

        With ``time_tags`` (default True), ``"context"`` starts with when the
        memory was said (``time_tag``): its date when the message carried a
        timestamp, otherwise its turn, as in ``[turn 12] ...``. A model
        asked which of two things happened first, or what was true before
        something changed, has nothing else to go on. A gist is tagged with
        the span of what it summarizes (``[2023-01-05 to 2023-03-02] ...``)
        when that was dated, and carries no tag otherwise. The tags count
        against ``token_budget``.
        ``time_tags="minute"`` adds the time of day when the timestamp has
        one (``[2024-01-15 10:00] ...``), so two memories of one day can be
        ordered; ``"date"`` is the same as True.

        Every result carries ``"source"``, the message it was taken from
        (``None`` for gists, for message passages, which are the message,
        and for records stored before messages were kept). A
        verified-superseded memory says by when it changed (``STALE_BY``:
        the date its replacement was said) whenever that date is known.

        ``now`` is when the question is asked (a date, a datetime, or a
        timestamp as stored with messages); by default today's date on this
        machine's clock, and ``now=False`` turns what follows off. A question
        that points at a time before it ("what did I buy ten days ago", "last
        weekend", "in the past two months": ``tsm.when.question_windows``) is
        also matched against the memories said in that time (from
        ``WINDOW_LEAD_DAYS`` before it to ``WINDOW_LAG_DAYS`` after), and the
        best of those are packed first, in up to ``WINDOW_SHARE`` of the
        budget. Such a question often names nothing else to search for, and
        in a long history the memories of a few days are outscored by
        everything else on the topic. Without a budget they are among the
        candidates, ranked by score. Without such an expression, or with
        nothing said in that time, nothing changes. Pass ``now`` whenever the
        question is not being asked today (a replayed conversation, a
        benchmark): on LongMemEval-S it brought all the evidence of 3 more of
        127 temporal questions into a 2,000-token context and lost none
        (``e79d75f:benchmarks/PHASE_PROGRESS.md``, 2026-11-06).

        ``recall()`` is a read: asking does not change what the next call
        returns. Use ``reinforce()`` to tell the memory which results were
        actually used.

        ``search`` picks the candidate search: ``"cognitive"`` (default) is
        the engine's graph-expanded search, fused with the concept graph's
        boost; ``"vector"`` is plain cosine search over the same store, with
        the same supersession demotion. Everything after the search (role
        prior, belief resolution, time windows, packing, whole replies) is
        the same for both.
        """
        return self._recall_core(query, user_id, token_budget, top_k, pool_k, lam,
                                 resolve_beliefs, max_items, time_tags, now, search)[0]

    def _recall_core(self, query: str, user_id: Optional[str], token_budget: Optional[int],
                     top_k: int = 10, pool_k: int = 20, lam: Optional[float] = None,
                     resolve_beliefs: bool = True, max_items: Optional[int] = None,
                     time_tags: Union[bool, str] = True,
                     now=None, search: str = "cognitive") -> Tuple[List[Dict], List[Dict], int]:
        """``recall()``, also returning every candidate it chose from (with
        its score and ``context``) and how many time windows the question
        pointed at: ``(results, pool, windows)``."""
        self._require_open()
        self._activity = time.monotonic()
        if time_tags not in (True, False, "date", "minute"):
            raise ValueError("time_tags must be True, False, 'date' or 'minute'")
        if search not in ("cognitive", "vector"):
            raise ValueError("search must be 'cognitive' or 'vector'")
        query_embedding = np.asarray(embed_query(self.embedder, query), dtype=np.float32)

        if token_budget is not None:
            # Enough candidates to fill the budget even with short memories.
            fetch_k = max(pool_k, 30, token_budget // 8)
        else:
            fetch_k = top_k
        if search == "vector":
            results = self.engine.search_ann(query_embedding, top_k=fetch_k, scope=user_id)
        else:
            results = self.engine.search(
                query_text=query,
                query_embedding=query_embedding,
                top_k=fetch_k,
                scope=user_id,
            )
        if not results:  # nothing in this scope matched
            return [], [], 0

        records = self._records([mid for mid, _ in results])
        pool = []
        for mid, score in results:
            rec = records.get(mid)
            if rec is None:
                continue  # deleted or evicted since the search returned it
            # Scope guard: only this user's memories and global (unscoped) ones.
            if user_id is not None and rec["scope"] not in (None, user_id):
                continue
            pool.append(self._pool_item(mid, rec, float(score)))
        if not pool:
            return [], [], 0
        if search == "vector":
            self._demote_superseded(pool)
        windows = [] if now is False else question_windows(
            query, datetime.now() if now is None else now)
        if windows:
            self._add_window(pool, windows, query_embedding, user_id)
        if resolve_beliefs:
            self._resolve_beliefs(pool, user_id)
        for p in pool:
            # Set before packing: what is shown is what the budget counts.
            shown = p["text"] or ""
            if is_message(p["role"]):
                shown = speaker(spoken_by(p["role"])) + shown
            if self._verified and p.get("superseded_by"):
                shown = stale_mark(p.get("valid_until")) + shown
            if time_tags and not is_gist(p["role"]):
                shown = time_tag(p["timestamp"], p["turn_index"],
                                 minutes=time_tags == "minute") + shown
            elif time_tags:
                shown = span_tag(p.get("span")) + shown
            p["context"] = shown

        apply_role_prior(pool, query)

        # Best first by the score that is returned (the role prior and a
        # brought-in current belief both move a memory's place).
        # On a tie the current belief goes before the one it replaced.
        pool.sort(key=lambda p: (-p["score"], "superseded_by" in p))
        if token_budget is None:
            return pool[:top_k], pool, len(windows)
        budget = max(1, token_budget)
        lam = packing_lam(query, lam)
        then = [p for p in pool if p.get("in_window")][:WINDOW_POOL]
        chosen = []
        if then:
            chosen = self._pack(then, max(1, int(budget * WINDOW_SHARE)), lam, max_items)
        if not max_items or len(chosen) < max_items:
            taken = {p["id"] for p in chosen}
            rest = [p for p in pool if p["id"] not in taken]
            left = budget - total_tokens([p["context"] for p in chosen])
            if rest and left > 0:
                chosen += self._pack(rest, left, lam,
                                     max_items - len(chosen) if max_items else None)
        if asks_what_assistant_said(query):
            chosen = self._whole_reply(chosen, budget)
        return chosen, pool, len(windows)

    def _add_window(self, pool: List[Dict], windows, query_embedding,
                    user_id: Optional[str]) -> None:
        """Mark the memories in ``pool`` said in one of ``windows`` (each
        widened by ``WINDOW_LEAD_DAYS`` and ``WINDOW_LAG_DAYS``) with
        ``in_window``, and add
        the best-scored others said then, ``WINDOW_POOL`` in all at most,
        from the ``WINDOW_SCAN`` memories nearest the question."""
        lead, lag = timedelta(days=WINDOW_LEAD_DAYS), timedelta(days=WINDOW_LAG_DAYS)
        spans = [(first - lead, last + lag) for first, last in windows]

        def said_then(item: Dict) -> bool:
            if is_gist(item["role"]):
                return False  # a gist's date is a span, not when one thing was said
            day = said_on(item["timestamp"])
            return day is not None and any(first <= day <= last for first, last in spans)

        found = 0
        for p in pool:
            if said_then(p):
                p["in_window"] = True
                found += 1
        if found >= WINDOW_POOL:
            return
        have = {p["id"] for p in pool}
        hits = self.engine.search_ann(query_embedding, top_k=WINDOW_SCAN, scope=user_id)
        scores = dict(hits)
        ids = [mid for mid, _ in hits if mid not in have]
        for start in range(0, len(ids), 200):  # read in rank order; stop when enough
            chunk = ids[start:start + 200]
            records = self._records(chunk)
            for mid in chunk:
                rec = records.get(mid)
                if rec is None or (user_id is not None and rec["scope"] not in (None, user_id)):
                    continue
                item = self._pool_item(mid, rec, float(scores[mid]))
                if not said_then(item):
                    continue
                item["in_window"] = True
                pool.append(item)
                found += 1
                if found >= WINDOW_POOL:
                    return

    def _whole_reply(self, chosen: List[Dict], token_budget: int) -> List[Dict]:
        """For a question about what the assistant said: each of the
        ``REPLIES_WHOLE`` best-scored assistant replies with a passage in
        ``chosen`` comes back whole, its passages joined in order into one
        result in that passage's place.

        A long reply is stored as several passages, and the packer took the
        ones that read most like the question, not the whole list or the
        second of two names the question asks about ("the two companies you
        mentioned"). The replies take at most ``REPLY_SHARE`` of the budget
        together, the passages nearest the chosen one first. The rest of
        ``chosen`` keeps its order, and the last chosen are dropped until it
        fits.
        """
        anchors = sorted((p for p in chosen if is_message(p["role"])
                          and spoken_by(p["role"]) != "user" and p.get("passage") is not None),
                         key=lambda p: -p["score"])
        left = max(1, int(token_budget * REPLY_SHARE))
        replies: Dict[str, Dict] = {}
        taken = set()
        seen = set()
        for anchor in anchors:
            if len(replies) == REPLIES_WHOLE:
                break
            if anchor["turn_index"] in seen:
                continue
            seen.add(anchor["turn_index"])
            group, prefix = self._reply_around(anchor, left)
            if len(group) < 2:
                continue
            text = " ".join(p["text"] for p in group)
            replies[anchor["id"]] = dict(anchor, text=text, context=prefix + text,
                                         ids=[p["id"] for p in group])
            taken.update(p["id"] for p in group)
            left -= max(1, len(prefix + text) // 4)
        if not replies:
            return chosen
        out = []
        for p in chosen:
            if p["id"] in replies:
                out.append(replies[p["id"]])
            elif p["id"] not in taken:
                out.append(p)
        whole = {id(r) for r in replies.values()}
        while len(out) > len(whole) and total_tokens([p["context"] for p in out]) > token_budget:
            out.pop(max(k for k, p in enumerate(out) if id(p) not in whole))
        return out

    def _reply_around(self, anchor: Dict, cap: int) -> Tuple[List[Dict], str]:
        """The passages of ``anchor``'s message around it, in order, that fit
        ``cap`` tokens with the anchor's tags in front (returned with them)."""
        context, text = anchor.get("context") or "", anchor["text"] or ""
        if not text or not context.endswith(text):
            return [], ""  # what is shown no longer ends with the stored text
        prefix = context[:len(context) - len(text)]
        parts = self._reply_passages(anchor)
        at = next((k for k, p in enumerate(parts) if p["id"] == anchor["id"]), None)
        if at is None:
            return [], prefix

        def size(lo, hi):
            return max(1, len(prefix + " ".join(p["text"] for p in parts[lo:hi])) // 4)

        lo, hi = at, at + 1
        if size(lo, hi) > cap:
            return [], prefix
        grown = True
        while grown:
            grown = False
            for nlo, nhi in ((lo - 1, hi), (lo, hi + 1)):
                if 0 <= nlo and nhi <= len(parts) and size(nlo, nhi) <= cap:
                    lo, hi, grown = nlo, nhi, True
        return parts[lo:hi], prefix

    def _reply_passages(self, anchor: Dict) -> List[Dict]:
        """The stored passages of the message ``anchor`` is a passage of, in
        order. A message's passages are written in one batch, one after the
        other, so their ids are consecutive; ``passages`` (the count stored
        with each) bounds the ids to read. A passage stored before the count
        was kept is read in a window of ``REPLY_WINDOW``."""
        prefix, _, seq = anchor["id"].rpartition("_")
        if not seq.isdigit():
            return [anchor]
        seq, at = int(seq), int(anchor["passage"])
        count = anchor.get("passages")
        last = seq + (int(count) - 1 - at if count else REPLY_WINDOW)
        ids = [f"{prefix}_{k}" for k in range(max(0, seq - at), last + 1)]
        parts = []
        for mid, rec in self._records(ids).items():
            payload = rec["payload"]
            if (rec["source_role"] == anchor["role"] and rec["scope"] == anchor["scope"]
                    and payload.get("turn_index") == anchor["turn_index"]
                    and payload.get("passage") is not None):
                parts.append(dict(self._pool_item(mid, rec, anchor["score"])))
        return sorted(parts, key=lambda p: p["passage"])

    def _pack(self, pool: List[Dict], token_budget: int, lam: float,
              max_items: Optional[int]) -> List[Dict]:
        """``select_under_budget`` on ``pool``. Every candidate is a stored
        record, so the engine makes the selection on the embeddings it
        already holds (``MemoryEngine.pack_under_budget``) instead of the
        candidates' text being embedded a second time. The Python packer is
        the fallback for an older extension without it, and for turn indexes
        the engine cannot take (anything but an integer)."""
        pack = getattr(self.engine, "pack_under_budget", None)
        turns = [p.get("turn_index") for p in pool]
        if pack is not None and all(
                t is None or (isinstance(t, int) and not isinstance(t, bool)) for t in turns):
            ids = [p["id"] for p in pool]
            index = {mid: k for k, mid in enumerate(ids)}
            pairs = [(k, index[p["superseded_by"]]) for k, p in enumerate(pool)
                     if p.get("superseded_by") in index]
            tokens = [max(1, len(p.get("context") or p["text"] or "") // 4) for p in pool]
            chosen = pack(ids, [float(p["score"]) for p in pool], tokens, turns,
                          int(token_budget), lam=float(lam), max_items=max_items or None,
                          never_redundant=pairs, max_redundancy=MAX_REDUNDANCY,
                          new_turn_bonus=NEW_TURN_BONUS,
                          occasions=[p.get("occasion") for p in pool])
            return [pool[i] for i in chosen]
        return select_under_budget(
            pool, token_budget, embed=lambda texts: embed_documents(self.embedder, texts),
            lam=lam, max_items=max_items)

    @staticmethod
    def _pool_item(mid: str, rec: Dict, score: float) -> Dict:
        """A search candidate as ``recall()`` hands it on: the stored record
        with what its payload says about where and when it was said."""
        payload = rec["payload"]
        return {"id": mid, "text": rec["text"], "score": score,
                "role": rec["source_role"] or "", "scope": rec["scope"],
                "turn_index": payload.get("turn_index"),
                "timestamp": payload.get("timestamp"),
                "source": payload.get("original_message"),
                "span": payload.get("span"),
                "passage": payload.get("passage"),
                "passages": payload.get("passages"),
                "occasion": occasion(rec["source_role"], payload.get("timestamp"))}

    def _demote_superseded(self, pool: List[Dict]) -> None:
        """Give a superseded memory in ``pool`` the demotion the engine's
        cognitive search applies (``resolve_beliefs`` reports it), so a plain
        vector search ranks it the same way apart from the graph. Under
        ``exclude_superseded`` the engine's plain search has dropped it
        already."""
        resolve = getattr(self.engine, "resolve_beliefs", None)
        if resolve is None:
            return
        by_id = {p["id"]: p for p in pool}
        for res in resolve(list(by_id)):
            if res["current_id"] != res["id"] and res["id"] in by_id:
                by_id[res["id"]]["score"] *= float(res.get("demotion") or 1.0)

    def _resolve_beliefs(self, pool: List[Dict], user_id: Optional[str]) -> None:
        """Mark the superseded memories in ``pool`` and, when supersessions are
        verified, add the current belief of each one that is missing.

        Every superseded memory gets ``superseded_by`` and ``chain``, also
        when its current belief is in the pool: with both in front of it, a
        reader still has to be told which of the two is the stale one. Older
        engines lacking ``resolve_beliefs`` leave the pool as it is.
        """
        resolve = getattr(self.engine, "resolve_beliefs", None)
        if resolve is None or not pool:
            return
        by_id = {p["id"]: p for p in pool}
        # current belief -> the best score one of the memories it replaced
        # had before its demotion: the place the current belief is owed when
        # the query is worded like the old fact.
        owed: Dict[str, float] = {}
        successor: Dict[str, str] = {}  # stale memory -> the one that replaced it
        for res in resolve(list(by_id)):
            current = res["current_id"]
            if current == res["id"]:
                continue
            stale = by_id[res["id"]]
            stale["superseded_by"] = current
            stale["chain"] = list(res["chain"])
            if not self._verified:
                continue
            chain = stale["chain"]
            position = chain.index(res["id"]) if res["id"] in chain else -1
            successor[res["id"]] = (chain[position + 1] if 0 <= position < len(chain) - 1
                                    else current)
            before = stale["score"] / max(float(res.get("demotion") or 1.0), 1e-6)
            owed[current] = max(owed.get(current, 0.0), before)
        if successor:
            # When each was replaced: the date its replacement was said.
            unknown = [mid for mid in set(successor.values()) if mid not in by_id]
            said = {mid: rec["payload"].get("timestamp")
                    for mid, rec in (self._records(unknown).items() if unknown else ())}
            said.update({mid: p.get("timestamp") for mid, p in by_id.items()})
            for mid, after in successor.items():
                by_id[mid]["valid_until"] = said.get(after)
        missing = [mid for mid in owed if mid not in by_id]
        for mid, rec in (self._records(missing).items() if missing else ()):
            if user_id is not None and rec["scope"] not in (None, user_id):
                continue
            by_id[mid] = self._pool_item(mid, rec, owed[mid])
            pool.append(by_id[mid])
        for mid, score in owed.items():
            head = by_id.get(mid)
            if head is not None:
                head["score"] = max(head["score"], score)
                head["replaces"] = [p["id"] for p in pool if p.get("superseded_by") == mid]

    # maintenance -----------------------------------------------------------------
    def consolidate(self) -> int:
        """Run consolidation; with a verifier installed, vet supersessions.

        The engine runs its consolidation cycle (dedup, importance, belief
        detection). When a ``Verifier`` is installed, supersession commitment
        is deferred: candidates are proposed, vetted against their stored
        texts, and only accepted pairs are committed. ``recall()`` then marks
        those stale facts and serves their replacements with them.

        Which candidates are proposed depends on the verifier. One that can
        judge meaning says so with a ``candidate_min_cosine`` attribute
        (``LLMVerifier``) and is given every new fact paired with its nearest
        older facts (the closest one and any about as close). Any other
        verifier (``NLIVerifier``) only sees the pairs that already passed the
        engine's own lexical gates.

        With ``max_user_tokens``, the users written to since the last pass
        are then brought back under their budget (see ``compact``). A
        summarizer failure there leaves that user's memories as they are, to
        be tried again on the next pass.

        Returns:
            The number of supersession edges committed (0 without a verifier).
        """
        self._require_open()
        with self._maintenance_lock:
            self._require_open()  # closed while this call waited for a running pass
            # Writes that arrive during the pass set these again.
            self._pending = False
            self._retry = False
            try:
                self.engine.trigger_consolidation()
                committed = self._verify_supersessions()
                if self.max_user_tokens is not None:
                    for user in sorted(self._touched):
                        try:
                            self._compact_user(user)
                        except Exception as e:  # noqa: BLE001 — nothing was removed; retried later
                            self._retry = True
                            logger.warning(
                                "memory of %r not compacted (kept as it is, will retry): %s",
                                user, type(e).__name__)
                self._maintain_executive()
            except BaseException:
                self._pending = True
                raise
        return committed

    def _maintain_executive(self) -> None:
        """The executive's share of a consolidation pass: close and replace
        intentions, curate lessons from new outcomes. Each is one batch of
        payload patches; a failure is logged and tried again next pass
        (nothing is deleted)."""
        if self._intentions is not None and self.intention_tracker is not None:
            self._track_intentions()
        elif self._intentions is not None:
            try:
                with self._write_lock:
                    done = self._intentions.maintain()
                if done["closed"] or done["replaced"]:
                    logger.info("intentions: %d closed, %d replaced", done["closed"],
                                done["replaced"])
            except Exception as e:  # noqa: BLE001 — retried on the next pass
                self._retry = True
                logger.warning("intention maintenance failed (will retry): %s: %s",
                               type(e).__name__, e)
        if self._experience is not None:
            try:
                # Not under the write lock (a distiller may call a model);
                # under the lock forget() takes to patch the same episodes.
                with self._experience_lock:
                    done = self._experience.curate(self.distiller)
                if done["episodes"]:
                    logger.info("experience: %d episodes, %d lessons added, %d updated",
                                done["episodes"], done["added"], done["updated"])
            except Exception as e:  # noqa: BLE001 — retried on the next pass
                self._retry = True
                logger.warning("experience curation failed (will retry): %s: %s",
                               type(e).__name__, e)

    def _track_intentions(self) -> None:
        """The tracker's pass: each user's unread messages, a batch at a
        time with the user's open intentions, outside the write lock (a
        model is asked); each batch's changes are applied under it in one
        insert and one batch of payload patches. A failed batch stays
        unread and is tried on the next pass."""
        tracker = self.intention_tracker
        size = max(1, int(getattr(tracker, "batch_size", 8)))
        totals = Counter()
        for user, messages in self._intentions.inbox().items():
            for start in range(0, len(messages), size):
                chunk = messages[start:start + size]
                open_items = self._intentions.open_items(user)
                shown = open_items[-int(getattr(tracker, "max_open", 60)):]
                try:
                    ops = tracker.track(shown, [{"text": m["text"], "said_on": m.get("said_on")}
                                                for m in chunk])
                    with self._write_lock:
                        totals.update(self._intentions.apply_track(
                            user, chunk, ops, [item["id"] for item in shown]))
                except Exception as e:  # noqa: BLE001 — unread; retried on the next pass
                    self._retry = True
                    logger.warning("intention tracking failed (will retry): %s: %s",
                                   type(e).__name__, e)
                    break  # later messages wait: they may close what these create
        flush = getattr(tracker, "flush_cache", None)
        if callable(flush):
            flush()
        if any(totals.values()):
            logger.info("intentions: %d created, %d closed, %d rescheduled",
                        totals["created"], totals["closed"], totals["rescheduled"])

    def _idle_loop(self) -> None:
        """Background thread of ``idle_consolidation_secs``: consolidate when
        there is something to do and nobody has been using the memory."""
        idle = self._idle_secs
        poll = min(max(idle / 4.0, 0.05), 5.0)
        while not self._stop_worker.wait(poll):
            if not (self._pending or self._retry):
                continue
            now = time.monotonic()
            if now - self._activity < idle or now < self._retry_after:
                continue
            try:
                self.consolidate()
            except Exception as e:  # noqa: BLE001 — the worker must outlive one bad pass
                if self._closed or self._stop_worker.is_set():
                    return  # the engine was closed under this pass
                self._retry_after = time.monotonic() + max(idle, IDLE_RETRY_FLOOR_SECS)
                logger.warning("idle consolidation failed (memories are kept, will retry): %s: %s",
                               type(e).__name__, e)

    def _stop_idle_worker(self) -> None:
        worker = self._worker
        if worker is None:
            return
        self._stop_worker.set()
        if worker is not threading.current_thread():
            worker.join(IDLE_CLOSE_TIMEOUT_SECS)
            if worker.is_alive():
                logger.warning("an idle consolidation pass is still running after %.0f s; "
                               "closing anyway", IDLE_CLOSE_TIMEOUT_SECS)

    def _verify_supersessions(self) -> int:
        """Propose, vet and commit supersessions; 0 without a verifier."""
        if self.verifier is None:
            return 0
        min_cosine = getattr(self.verifier, "candidate_min_cosine", None)
        wide = getattr(self.engine, "propose_supersession_candidates", None)
        if min_cosine is not None and wide is not None:
            proposed = wide(float(min_cosine),
                            int(getattr(self.verifier, "candidates_per_record", 2)),
                            getattr(self.verifier, "candidate_margin", 0.1))
        else:
            proposed = self.engine.propose_supersessions()  # (old, new, kind, cosine)
        if self.topic_keyer is not None:
            seen = {(pair[0], pair[1]) for pair in proposed}
            proposed = list(proposed) + self._topic_candidates(seen)
        proposed = self._drop_affirmed(proposed)
        if not proposed:
            return 0
        pair_ids = sorted({mid for old, new, *_ in proposed for mid in (old, new)})
        id_to_text = {mid: rec["text"] for mid, rec in self._records(pair_ids).items()}
        accepted = self.verifier.verify(proposed, id_to_text)
        # A verifier that could not judge some pairs (a failed request) says
        # so; the idle worker then asks again later.
        self._retry = self._retry or bool(getattr(self.verifier, "unresolved", 0))
        if not accepted:
            return 0
        committed = self.engine.commit_supersessions(accepted)
        logger.info("verified supersession: proposed=%d accepted=%d committed=%d",
                    len(proposed), len(accepted), committed)
        return committed

    def _drop_affirmed(self, pairs):
        """Candidate pairs without those ``retract`` ruled out: a fact someone
        said still holds is not replaced by a pair they took back
        (``retracted_by``), nor by anything said before they said so
        (``affirmed_seq``: a retraction answers "does it still hold?" for
        everything said up to then). Something said later still can."""
        if not pairs:
            return pairs
        olds = self._records(sorted({pair[0] for pair in pairs}))
        guarded = {mid: rec["payload"] for mid, rec in olds.items()
                   if rec["payload"].get("retracted_by") or
                   rec["payload"].get("affirmed_seq") is not None}
        if not guarded:
            return pairs
        news = sorted({pair[1] for pair in pairs if pair[0] in guarded})
        seqs = {r["id"]: r.get("insert_seq") for r in self.engine.get_records(news) if r}
        kept = []
        for pair in pairs:
            payload = guarded.get(pair[0])
            if payload is not None:
                if pair[1] in (payload.get("retracted_by") or ()):
                    continue
                affirmed, seq = payload.get("affirmed_seq"), seqs.get(pair[1])
                if affirmed is not None and seq is not None and int(seq) < int(affirmed):
                    continue
            kept.append(pair)
        return kept

    def _topic_candidates(self, seen) -> List[Tuple[str, str, str, float]]:
        """Pairs of current facts that give a value for the same
        single-valued property (``tsm.topics``), after keying the facts of
        the scopes written to since the last pass. Proposals only: the
        verifier decides. A keying failure is logged and those scopes are
        keyed again on the next pass; facts keyed earlier still pair."""
        from .topics import topic_pairs

        scopes = self._topic_scopes
        if scopes is None:
            scopes = set(self.engine.scopes())
        self._topic_scopes = set()  # writes from here on are keyed next pass
        roles = self.engine_config.get("belief_source_roles")
        records: List[Dict] = []
        for scope in sorted(s for s in scopes if s is not None):
            for rec in self._records(self.engine.scope_ids(scope)).values():
                if roles is None or rec.get("source_role") in roles:
                    records.append(rec)
        unkeyed = sorted((r for r in records if "topic" not in r["payload"]),
                         key=lambda r: int(r["insert_seq"]))
        if unkeyed:
            counts: Dict[str, int] = {}
            for rec in records:
                key = rec["payload"].get("topic")
                if key:
                    counts[key] = counts.get(key, 0) + 1
            known = sorted(counts, key=lambda k: -counts[k])
            try:
                answers = self.topic_keyer.topics([r["text"] or "" for r in unkeyed], known)
            except Exception as e:  # noqa: BLE001 — retried on the next pass
                logger.warning("topic keying failed (will retry): %s: %s", type(e).__name__, e)
                answers = [None] * len(unkeyed)
            keyed = {}
            for rec, answer in zip(unkeyed, answers):
                if answer is None:
                    self._retry = True
                    self._topic_scopes.add(rec.get("scope"))
                    continue
                keyed[rec["id"]] = (answer[0], bool(answer[1]))
                rec["payload"]["topic"], rec["payload"]["topic_one"] = keyed[rec["id"]]
            if keyed:
                with self._write_lock:
                    # Read again under the lock: another patch (retract,
                    # forget) may have changed a payload since.
                    ids, payloads = [], []
                    for mid, rec in self._records(sorted(keyed)).items():
                        rec["payload"]["topic"], rec["payload"]["topic_one"] = keyed[mid]
                        ids.append(mid)
                        payloads.append(json.dumps(rec["payload"]))
                    if ids:
                        self.engine.set_payloads(ids, payloads)
            flush = getattr(self.topic_keyer, "flush_cache", None)
            if callable(flush):
                flush()
        superseded = set(self.engine.superseded_ids())
        current = [r for r in records if r["id"] not in superseded]
        per_record = int(getattr(self.topic_keyer, "per_record", 1))
        pairs = [pair for pair in topic_pairs(current, per_record) if pair not in seen]
        if not pairs:
            return []
        ids = sorted({mid for pair in pairs for mid in pair})
        vectors, found = self.engine.get_vectors(ids)
        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1)
        row = {mid: k for k, mid in enumerate(ids) if found[k] and norms[k] > 0}
        out = []
        for old, new in pairs:
            if old in row and new in row:
                a, b = row[old], row[new]
                cosine = float(vectors[a] @ vectors[b] / (norms[a] * norms[b]))
                out.append((old, new, "contradiction", cosine))
        logger.info("topic candidates: %d pairs (%d facts keyed this pass)",
                    len(out), len(unkeyed))
        return out

    def compact(self, user_id: Optional[str] = None) -> int:
        """Bring one user's memory (or, with no argument, every user's) back
        under ``max_user_tokens``.

        The newest facts are kept as they are, user facts first. What does
        not fit, together with any earlier gist, is rewritten by
        ``gist_summarizer`` as a few short gists stored under the same user
        (``source_role`` ``"summary"``); facts a newer one has replaced are
        folded first. The gists are written before anything is removed, so an
        interruption can leave extra memories behind but never lose one, and
        a summarizer that fails removes nothing: the error is raised.

        Returns:
            The number of memories folded away.
        """
        self._require_open()
        if self.max_user_tokens is None:
            raise ValueError("compact() needs Memory(max_user_tokens=...)")
        users = [user_id] if user_id is not None else list(self.engine.scopes())
        return sum(self._compact_user(user) for user in users)

    def _summarize_chunk(self, texts, max_tokens: int) -> str:
        summarize = getattr(self._gist_summarizer, "summarize", None)
        if callable(summarize):
            return summarize(list(texts), max_tokens=max_tokens) or ""
        return self._gist_summarizer(list(texts)) or ""

    def _compact_user(self, user: str) -> int:
        records = [r for r in self.engine.get_records(self.engine.scope_ids(user)) if r]
        texts = [r["text"] or "" for r in records]
        if total_tokens(texts) <= self.max_user_tokens:
            self._touched.discard(user)
            return 0
        superseded = set(self.engine.superseded_ids())
        plan = plan_compaction(
            texts,
            [r["source_role"] or "user" for r in records],
            self.max_user_tokens,
            self._summarize_chunk if self._gist_summarizer is not None else None,
            fold_first=[i for i, r in enumerate(records) if r["id"] in superseded],
            spans=[_span_of(r.get("payload")) for r in records],
            levels=[_level_of(r.get("payload")) for r in records],
            policy=self.compaction,
        )
        if plan is None:
            self._touched.discard(user)
            return 0
        embeddings = None
        if plan.gists:
            embeddings = np.ascontiguousarray(embed_documents(self.embedder, plan.gists),
                                              dtype=np.float32)
            if embeddings.shape != (len(plan.gists), self.dim):
                raise ValueError(
                    f"embedder returned shape {embeddings.shape} for {len(plan.gists)} gists; "
                    f"expected ({len(plan.gists)}, {self.dim})")
        folded = [records[i]["id"] for i in plan.fold]
        with self._write_lock:
            # The plan was made without the lock (it may have called a model).
            # If a memory it folds is gone, something else changed the store:
            # leave it for the next pass.
            if not all(self.engine.contains_id(mid) for mid in folded):
                return 0
            if plan.gists:
                gist_ids, payloads = [], []
                spans = plan.gist_spans or [None] * len(plan.gists)
                levels = plan.gist_levels or [0] * len(plan.gists)
                for span, level in zip(spans, levels):
                    gist_ids.append(self._next_id(user))
                    # Each gist is a turn of its own, so budget recall spreads
                    # its selection over gists as it does over turns. Its span
                    # says when what it summarizes was said; its level, how
                    # many merges it is the result of.
                    payload = {"kind": "gist", "user_id": user, "folded": len(folded),
                               "turn_index": self._insert_counter, "level": int(level)}
                    if span:
                        payload["span"] = list(span)
                    payloads.append(json.dumps(payload))
                self.engine.insert_batch(
                    gist_ids,
                    plan.gists,
                    embeddings,
                    [1.0] * len(plan.gists),
                    [extract_concepts(gist) for gist in plan.gists],
                    payloads,
                    [user] * len(plan.gists),
                    list(plan.gist_roles),
                )
            for mid in folded:
                self.engine.delete(mid)
            self._touched.discard(user)
        logger.info("compacted %r: %d memories folded into %d gists", user, len(folded),
                    len(plan.gists))
        return len(folded)

    def reinforce(self, memory_ids: Iterable[str]) -> int:
        """Tell the memory which memories were actually used.

        Their links in the concept graph get stronger, so cognitive search
        ranks them, and what is connected to them, higher from then on.
        ``recall()`` itself reinforces nothing (unless the engine is opened
        with ``reinforce_on_search=True``): having been retrieved is a weak
        signal, and reinforcing on every read made the same question return
        a different list the second time it was asked.

        Returns:
            How many of the ids exist and were reinforced.
        """
        self._require_open()
        reinforce = getattr(self.engine, "reinforce", None)
        if reinforce is None:
            raise RuntimeError(
                "the compiled turbomemory extension is older than this tsm SDK "
                "(missing: reinforce); rebuild it with 'make build-python'")
        return int(reinforce([str(mid) for mid in memory_ids]))

    # executive: intentions ----------------------------------------------------------
    def _require_intentions(self) -> IntentionLedger:
        self._require_open()
        if self._intentions is None:
            raise RuntimeError("intentions are off: open the Memory with intentions=True")
        return self._intentions

    def intentions(self, user_id: Optional[str]) -> List[Dict]:
        """The user's pending intentions, in the order they were formed."""
        return self._require_intentions().pending(user_id)

    def due(self, user_id: Optional[str], now=None, context: Optional[str] = None,
            top_k: int = 8, cue_min_cosine: Optional[float] = None) -> List[Dict]:
        """The user's pending intentions that need attention now: those
        whose time has come (``verdict="act"``, ``overdue`` once their last
        day has passed) and, given what is happening (``context``, any
        text), those it brings to mind (``verdict="observe"``). At most
        ``top_k``. ``cue_min_cosine``: how close the context must be (the
        embedder's ``cue_min_cosine`` if it has one, else
        ``tsm.intentions.CUE_MIN_COSINE``). With a ``readiness_judge`` each is judged again as act /
        wait / observe / satisfied / invalidated (``judged=True``).

        Each carries its ``evidence``: the stored memories of the message it
        was formed from. A read: close an intention with
        ``resolve_intention``. Costs no model call (one embedding with a
        ``context``) unless a judge is installed and something is due.
        """
        from .intentions import CUE_MIN_COSINE

        ledger = self._require_intentions()
        when = now if now not in (None, False) else datetime.now().date()
        vector = (np.asarray(embed_query(self.embedder, context), dtype=np.float32)
                  if context else None)
        if cue_min_cosine is None:
            # A cosine floor belongs to the embedder's geometry.
            cue_min_cosine = getattr(self.embedder, "cue_min_cosine", None) or CUE_MIN_COSINE
        items = ledger.due(user_id, when, vector, float(cue_min_cosine), top_k)
        if items and self.readiness_judge is not None:
            today = as_date(when)
            try:
                verdicts = self.readiness_judge.judge(items, context,
                                                      today.isoformat() if today else "")
                for k, verdict in verdicts.items():
                    items[k]["verdict"] = verdict
                    items[k]["judged"] = True
            except Exception as e:  # noqa: BLE001 — the unjudged verdicts stand
                logger.warning("readiness check failed (time and cue verdicts kept): %s", e)
        sources = sorted({mid for item in items for mid in item.get("source_ids") or ()})
        records = self._records(sources) if sources else {}
        for item in items:
            item["evidence"] = [{"id": mid, "text": records[mid]["text"],
                                 "timestamp": records[mid]["payload"].get("timestamp")}
                                for mid in item.get("source_ids") or () if mid in records]
        return items

    def resolve_intention(self, intention_id: str, status: str = "done",
                          snooze_until=None) -> bool:
        """Close an intention (``"done"``, ``"cancelled"``), or put it back
        (``"pending"``; with ``snooze_until``, a date, it comes due then).
        A payload patch: the intention keeps its place. Returns whether it
        exists."""
        if status not in ("done", "cancelled", "pending"):
            raise ValueError("status must be 'done', 'cancelled' or 'pending'")
        fields: Dict = {"status": status}
        if snooze_until is not None:
            day = as_date(snooze_until)
            if day is None:
                raise ValueError(f"cannot read a date from {snooze_until!r}")
            fields.update(status="pending", due_from=day.isoformat(),
                          due_from_day=day.toordinal())
        ledger = self._require_intentions()
        with self._write_lock:
            return ledger.set_status([(intention_id, fields)]) > 0

    # executive: experience ----------------------------------------------------------
    def _require_experience(self) -> ExperienceLedger:
        self._require_open()
        if self._experience is None:
            raise RuntimeError("experience is off: open the Memory with experience=True")
        return self._experience

    def feedback(self, task: str, outcome, memory_ids: Iterable[str] = (),
                 user_id: Optional[str] = None, source: str = "user",
                 lesson: Optional[str] = None) -> str:
        """Report how a task went: ``outcome`` is ``"success"`` /
        ``"failure"`` (or a bool, or a score in [0, 1]); ``memory_ids`` what
        was in front of the reader, lesson ids from ``lessons()`` included
        (only those are credited or blamed); ``source`` who says so
        (``"user"``, ``"verifier"``, ``"judge"``, or ``"self"``, which is
        kept but never learned from); ``lesson`` what to do or avoid next
        time, in words. Stored as an episode; lessons are curated from
        episodes in ``consolidate()`` (``tsm.experience``). Returns the
        episode's id."""
        ledger = self._require_experience()
        vector = np.asarray(embed_query(self.embedder, task), dtype=np.float32)
        with self._write_lock:
            mid = ledger.record(task, outcome, list(memory_ids), user_id, source, lesson,
                                at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                vector=vector)
            self._pending = True
        self._activity = time.monotonic()
        return mid

    def lessons(self, task: str, user_id: Optional[str] = None, k: int = 3,
                include_probation: bool = False) -> List[Dict]:
        """The curated lessons nearest ``task``, of ``user_id`` or global,
        best first: ``active`` ones (and ``probation`` ones when asked). Put
        them in front of the reader apart from the memories
        (``tsm.experience.lesson_block``), and pass their ids back with
        ``feedback`` so they are credited or blamed."""
        ledger = self._require_experience()
        return ledger.lessons(np.asarray(embed_query(self.embedder, task), dtype=np.float32),
                              user_id, k, include_probation)

    def episodes(self, task: str, user_id: Optional[str] = None, k: int = 3) -> List[Dict]:
        """The past episodes nearest ``task``, as they were reported (the
        plain "retrieve past experiences" baseline lessons must beat)."""
        ledger = self._require_experience()
        return ledger.episodes(np.asarray(embed_query(self.embedder, task), dtype=np.float32),
                               user_id, k)

    # executive: belief ledger -------------------------------------------------------
    def retract(self, old_id: str, new_id: str) -> bool:
        """Take back the supersession of ``old_id`` by ``new_id`` (a wrong
        verdict, or a user who says the old fact still holds): the old fact
        is no longer demoted or flagged, and later consolidations do not
        commit the pair again (it is kept in the old fact's payload as
        ``retracted_by``), nor replace the old fact with anything else said
        before the retraction (``affirmed_seq``); what is said later still
        can. Returns whether there was a supersession to take back."""
        self._require_open()
        with self._write_lock:
            removed = int(self.engine.retract_supersessions([(old_id, new_id)]))
            rec = self._records([old_id]).get(old_id)
            if rec is not None:
                payload = dict(rec["payload"])
                retracted = list(payload.get("retracted_by") or [])
                if new_id not in retracted:
                    retracted.append(new_id)
                payload["retracted_by"] = retracted
                payload["affirmed_seq"] = int(self.engine.next_insert_seq())
                self.engine.set_payloads([old_id], [json.dumps(payload)])
        return removed > 0

    def history(self, memory_id: str) -> List[Dict]:
        """The supersession chain ``memory_id`` belongs to, oldest first:
        each memory with ``valid_from`` (when it was said) and
        ``valid_until`` (when what replaced it was said; ``None`` for the
        current one) and ``current``."""
        self._require_open()
        resolved = self.engine.resolve_beliefs([memory_id])
        chain = list(resolved[0]["chain"]) if resolved else [memory_id]
        records = self._records(chain)
        out = []
        for k, mid in enumerate(chain):
            rec = records.get(mid)
            if rec is None:
                continue
            after = records.get(chain[k + 1]) if k + 1 < len(chain) else None
            out.append({"id": mid, "text": rec["text"],
                        "valid_from": rec["payload"].get("timestamp"),
                        "valid_until": after["payload"].get("timestamp") if after else None,
                        "current": k == len(chain) - 1})
        return out

    def forget(self, memory_ids: Iterable[str] = (), user_id: Optional[str] = None,
               cascade: bool = True, gists: str = "redact",
               purge_caches: bool = True, texts: Iterable[str] = ()) -> Dict:
        """Delete memories, and everything that still says what they said.

        A forget request is a promise that the content is not served again,
        so what was derived from the memories goes with them:

        - with ``cascade`` (default), everything stored from the same message
          (its other facts and its passages: one message, one
          ``turn_index``) and the intentions formed from it;
        - a gist that carries the forgotten content (``tsm.forgetting``: the
          text itself, a code or number from it, or most of its salient
          words) is **redacted**: its sentences that carry it are removed,
          the rest is stored as a new gist (same span, level and role)
          *before* the old one is deleted, and a gist left with nothing is
          deleted. A gist is lossy already and its sources are gone, so
          rebuilding it with a model would have nothing to rebuild from and
          could not be checked; removing sentences can. ``gists="keep"``
          leaves every gist as it is and lists the ones that may hold the
          content in ``gists_may_hold``;
        - intentions and the experience store's episodes and lessons that
          carry it are deleted, and episodes stop listing the forgotten ids;
        - with ``purge_caches`` (default), the backends' on-disk caches under
          ``tsm_cache/`` (extractions, embeddings, judges' replies) lose
          every entry that carries it, and are rewritten at once.

        With ``user_id``, memories of another user are refused
        (``skipped``) and only that user's gists, intentions and lessons are
        looked at.

        ``texts`` (needs ``user_id``): more content to forget, for what no
        stored memory holds any more but something derived still may (a fact
        folded into a gist, or dropped by it, whose intention or lesson
        remains): the user's gists, intentions, lessons and the caches are
        swept for it as for a forgotten memory. Memories themselves are only
        deleted by id (find them with ``recall``).

        Returns ``{"forgotten", "skipped", "gists_redacted", "gists_deleted",
        "gists_may_hold", "intentions_removed", "experience_removed",
        "episodes_unlinked", "cache_entries_purged", "verified"}``.
        ``verified`` says that no forgotten id is stored or found by a search
        near its own vector, and that no redacted gist still carries the
        content.
        """
        if gists not in ("redact", "keep"):
            raise ValueError("gists must be 'redact' or 'keep'")
        extra = [str(t) for t in texts if t and str(t).strip()]
        if extra and user_id is None:
            raise ValueError("forget(texts=...) needs a user_id")
        self._require_open()
        wanted = [str(m) for m in memory_ids]
        with self._write_lock:
            records = self._records(wanted)
            skipped = [m for m in wanted if m not in records]
            targets: Dict[str, Dict] = {}
            for mid, rec in records.items():
                if user_id is not None and rec["scope"] not in (None, user_id):
                    skipped.append(mid)
                    continue
                targets[mid] = rec
            if cascade and targets and hasattr(self.engine, "ids_where"):
                for rec in list(targets.values()):
                    turn = rec["payload"].get("turn_index")
                    if turn is None or is_gist(rec["source_role"] or ""):
                        continue
                    flt = json.dumps({"op": "eq", "field": "turn_index", "value": turn})
                    for mid, sib in self._records(self.engine.ids_where(flt, rec["scope"])).items():
                        if mid not in targets and not is_gist(sib["source_role"] or ""):
                            targets[mid] = sib
            ids = sorted(targets)
            said = [t["text"] for t in targets.values()]
            said += [t["payload"].get("original_message") for t in targets.values()
                     if t["payload"].get("original_message")]
            forgotten = Forgotten(said, phrases=extra)
            scopes = {t["scope"] for t in targets.values()}
            if extra:
                scopes.add(user_id)
            sweep = bool(ids) or bool(extra)
            vectors, found = (self.engine.get_vectors(ids)
                              if ids and hasattr(self.engine, "get_vectors") else (None, []))
            # Gists first: a redacted gist is stored before anything is deleted.
            gist_report = self._forget_in_gists(targets, forgotten, scopes, redact_them=(
                gists == "redact"))
            for mid in ids:
                self.engine.delete(mid)
            intentions_removed = 0
            if self._intentions is not None and sweep:
                doomed = set()
                for mid in ids:
                    doomed.update(self._intentions.engine.ids_where(json.dumps(
                        {"op": "eq", "field": "source_ids", "value": mid})))
                for scope in scopes:
                    for iid, item in self._intentions._items(
                            self._intentions.engine.ids_where(None, scope)).items():
                        if item.get("scope") == scope and forgotten.carries(item["text"]):
                            doomed.add(iid)
                for iid in doomed:
                    intentions_removed += bool(self._intentions.engine.delete(iid))
            experience_removed = episodes_unlinked = 0
            if self._experience is not None and sweep:
                with self._experience_lock:
                    experience_removed = self._experience.forget_content(forgotten, list(scopes))
                    episodes_unlinked = self._experience.forget_references(ids)
        purged = 0
        if purge_caches and forgotten:
            for backend in (self.extractor, self.embedder, self.verifier,
                            self.readiness_judge, self.distiller, self.topic_keyer,
                            self.intention_tracker):
                purge = getattr(backend, "forget_texts", None)
                if callable(purge):
                    try:
                        purged += int(purge(forgotten) or 0)
                    except Exception as e:  # noqa: BLE001 — reported, not fatal
                        logger.warning("cache purge failed for %s: %s",
                                       type(backend).__name__, e)
        verified = not any(self.engine.contains_id(mid) for mid in ids)
        gone = set(ids)
        for k, mid in enumerate(ids):
            if verified and k < len(found) and found[k]:
                hits = self.engine.search_ann(vectors[k], 5, scope=targets[mid]["scope"])
                verified = not any(hit in gone for hit, _ in hits)
        if verified and gist_report["redacted"]:
            texts = self._records(gist_report["redacted"])
            verified = not any(forgotten.carries(rec["text"]) for rec in texts.values())
        return {"forgotten": ids, "skipped": skipped,
                "gists_redacted": gist_report["redacted"],
                "gists_deleted": gist_report["deleted"],
                "gists_may_hold": gist_report["may_hold"],
                "intentions_removed": intentions_removed,
                "experience_removed": experience_removed,
                "episodes_unlinked": episodes_unlinked,
                "cache_entries_purged": purged, "verified": verified}

    def _forget_in_gists(self, targets: Dict[str, Dict], forgotten: "Forgotten",
                         scopes, redact_them: bool) -> Dict[str, List[str]]:
        """The gists of ``scopes`` against the forgotten content: those that
        carry it are redacted (a new gist, then the old one deleted) or, with
        nothing left, deleted; without ``redact_them`` they are only listed.
        A gist whose span covers a forgotten memory's day but that carries
        nothing detectable is listed in ``may_hold``. Under the write lock."""
        out: Dict[str, List[str]] = {"redacted": [], "deleted": [], "may_hold": []}
        if not hasattr(self.engine, "scope_ids"):
            return out
        days = {said_on(t["payload"].get("timestamp")) for t in targets.values()} - {None}
        for scope in scopes:
            if scope is None:
                continue
            for mid, rec in sorted(self._records(self.engine.scope_ids(scope)).items()):
                role = rec["source_role"] or ""
                if mid in targets or not is_gist(role):
                    continue
                text = rec["text"] or ""
                if not forgotten.carries(text):
                    span = rec["payload"].get("span")
                    if span and len(span) == 2:
                        first, last = said_on(span[0]), said_on(span[1])
                        if first and last and any(first <= d <= last for d in days):
                            out["may_hold"].append(mid)
                    continue
                if not redact_them:
                    out["may_hold"].append(mid)
                    continue
                kept, removed = redact(text, forgotten)
                if kept and not forgotten.carries(kept):
                    payload = dict(rec["payload"])
                    payload["redacted"] = int(payload.get("redacted", 0)) + len(removed)
                    new_id = self._next_id(scope)
                    self.engine.insert_batch(
                        [new_id], [kept], self._embed_documents([kept]), [1.0],
                        [extract_concepts(kept)], [json.dumps(payload)], [scope], [role])
                    out["redacted"].append(new_id)
                else:
                    out["deleted"].append(mid)
                self.engine.delete(mid)
        return out

    def flush(self) -> None:
        """Durably persist all pending writes."""
        self._require_open()
        self.engine.flush()
        for ledger in (self._intentions, self._experience):
            if ledger is not None:
                ledger.engine.flush()

    def close(self) -> None:
        """Flush and release the engine. Idempotent.

        Releases the database lock, mmaps, and worker threads, so ``db_path``
        can be reopened right away. Any later call on this object raises
        ``RuntimeError``.
        """
        if not self._closed:
            self._stop_idle_worker()
            try:
                for ledger in (self._intentions, self._experience):
                    if ledger is not None:
                        try:
                            ledger.engine.close()
                        except Exception as e:  # noqa: BLE001 — the main engine still closes
                            logger.warning("closing a derived store failed: %s", e)
                self.engine.close()
            finally:
                # The engine handle is released even when its final flush
                # raises, so this object is closed either way.
                self._closed = True
                # Persist the backends' paid-for caches (extractions,
                # embeddings); they otherwise only reach disk every few
                # hundred entries or at interpreter exit.
                for backend, method in ((self.extractor, "flush_cache"),
                                        (self.embedder, "flush"),
                                        (self.verifier, "flush_cache"),
                                        (self.readiness_judge, "flush_cache"),
                                        (self.distiller, "flush_cache"),
                                        (self.topic_keyer, "flush_cache"),
                                        (self.intention_tracker, "flush_cache")):
                    flush = getattr(backend, method, None)
                    if callable(flush):
                        try:
                            flush()
                        except Exception as e:  # noqa: BLE001 — never fail close()
                            logger.warning("cache flush failed on close: %s", e)

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
