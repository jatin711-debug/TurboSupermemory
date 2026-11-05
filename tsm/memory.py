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
import re
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ._loader import load_turbomemory
from .budget import (MAX_REDUNDANCY, NEW_TURN_BONUS, estimate_tokens, select_under_budget,
                     total_tokens)
from .chunks import is_message, message_role, occasion, passages, spoken_by
from .compaction import POLICIES, is_gist, plan_compaction
from .concepts import extract_concepts
from .events import EVENT_DIR, EVENT_POOL, EVENT_ROLE, EVENT_SHARE, event_note, event_text
from .observe import (OBSERVATION_DIR, OBSERVATION_ROLE, conversations,
                      observation_context)
from .identity import bind_embedder
from .interfaces import (  # noqa: F401  (Embedder, Extractor, Verifier: re-exported types)
    Embedder, Extractor, Verifier, embed_documents, embed_query, embedder_fingerprint)
from .ranking import apply_role_prior, asks_what_assistant_said, is_counting_query, packing_lam
from .sources import SOURCE_TOKENS, speaker, turn_units
from .when import annotate, said_on

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

# Engine methods this SDK cannot work without. The extension and the SDK ship
# together; this guards against a stale locally-built turbomemory.pyd/.so.
_REQUIRED_ENGINE_METHODS = ("get_records", "next_insert_seq", "recovery_report")

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
        reranker: Optional[object] = None,
        max_user_tokens: Optional[int] = None,
        compaction: str = "refold",
        keep_messages: Optional[bool] = None,
        allow_embedder_change: bool = False,
        idle_consolidation_secs: Optional[float] = None,
        events: Optional[object] = None,
        observer: Optional[object] = None,
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
                1,200 (``benchmarks/PHASE_PROGRESS.md``, 2026-10-29). With
                ``extractor="none"`` it is the whole store: 26 of the 29.
            reranker: optional ``Reranker`` implementation or ``"colbert"``
                to enable Stage-2 MultiVector late-interaction MaxSim precision
                reranking (e.g. ``LFM2.5-ColBERT-350M``).
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
            events: also list the events each user message reports (what
                the user did, bought, attended, owns or plans, with amounts
                and when; ``tsm.events``) and store each as a short record
                in an engine of its own (``<db_path>/tsm_events``), so the
                user's memories and their search are exactly as without
                events. ``recall()`` of a question that
                counts or totals then gives the best of them a share of its
                budget (``EVENT_SHARE``), one dated line each, so the
                instances a count needs fit where a few passages would.
                ``True`` / ``"openai"`` is ``OpenAIEventExtractor``; any
                object with ``extract_events(message, context)`` works (and
                ``extract_many(items)`` is used when it has one). One request
                per user message. Default ``None``: no events. A store that
                has events uses them in ``recall()`` whether or not an
                extractor is passed when it is reopened.
            observer: consolidate conversations into observations
                (``tsm.observe``): on each ``consolidate()`` every message not
                yet observed is handed, a day's conversation at a time, to the
                observer, which writes down everything in it worth keeping as
                short self-contained lines (asides included, one occasion per
                line, what the assistant gave briefly). They are stored in an
                engine of their own (``<db_path>/tsm_observations``) and read
                with ``observations()``, or by ``recall(observations=...)``.
                ``"anthropic"`` is ``AnthropicObserver`` (claude-haiku-5-5),
                ``"openai"`` is ``OpenAIObserver``; any object with
                ``observe(messages)`` (and optionally ``observe_many``) works.
                Needs the messages (``keep_messages``). Default ``None``.
            **engine_kwargs: forwarded to ``turbomemory.MemoryEngine``.
        """
        # Backend names are resolved here; anything else that is a string is a
        # typo, and is reported now rather than at the first add().
        for kind, value, known in (
            ("embedder", embedder, ("openai", "sentence_transformer", "local", "minilm", "gemma")),
            ("extractor", extractor, ("openai", "gliner", "passthrough", "none")),
            ("reranker", reranker, ("colbert",)),
            ("verifier", verifier, ("nli", "llm")),
        ):
            if isinstance(value, str) and value not in known:
                raise ValueError(
                    f"unknown {kind} {value!r}: use one of {', '.join(known)} "
                    f"or pass an instance")
        if embedder == "openai":
            embedder = None
        cache_dir = None
        if embedder is None or extractor is None or extractor == "openai":
            import os

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
        if events is True or events == "openai":
            import os

            from .events import OpenAIEventExtractor

            events = OpenAIEventExtractor(cache_dir=os.path.join(db_path, "tsm_cache"))
        elif isinstance(events, str) or events is False:
            if events is not False:
                raise ValueError(f"unknown events {events!r}: use True, 'openai' or an instance")
            events = None
        self.events = events
        if observer == "anthropic":
            import os

            from .observe import AnthropicObserver

            observer = AnthropicObserver(cache_dir=os.path.join(db_path, "tsm_cache"))
        elif observer == "openai":
            import os

            from .observe import OpenAIObserver

            observer = OpenAIObserver(cache_dir=os.path.join(db_path, "tsm_cache"))
        elif isinstance(observer, str):
            raise ValueError(f"unknown observer {observer!r}: use 'anthropic', 'openai' "
                             "or an instance")
        self.observer = observer
        if reranker == "colbert":
            from .rerankers import ColBertReranker

            reranker = ColBertReranker()
        if verifier == "nli":
            from .verification import NLIVerifier

            verifier = NLIVerifier()
        elif verifier == "llm":
            import os

            from .verification import LLMVerifier

            verifier = LLMVerifier(cache_dir=os.path.join(db_path, "tsm_cache"))
        self.embedder = embedder
        self.extractor = extractor
        self.verifier = verifier
        # Supersessions a verifier accepted are reliable enough to show a
        # model; the engine's unchecked ones stay metadata (on its own the
        # detector also fires on facts that are both still true, "my sister
        # lives in Vancouver" / "my brother lives in Vancouver").
        self._verified = verifier is not None
        self.reranker = reranker

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
        if self.observer is not None and not self.keep_messages:
            raise ValueError("an observer reads the messages: it needs keep_messages")
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

        turbomemory = load_turbomemory()
        self.engine = turbomemory.MemoryEngine(
            db_path=db_path, dimension=self.dim, **config
        )
        self._closed = False
        # One writer at a time: id minting and the insert it feeds must not
        # interleave between threads sharing this Memory.
        self._write_lock = threading.Lock()
        required = _REQUIRED_ENGINE_METHODS + (
            ("scope_ids", "scopes") if self.max_user_tokens is not None else ())
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

        # Ids are `{user_id}_{n}`. Seeding n from the engine's durable insert
        # sequence (which starts at 1 and is never reused) keeps ids unique
        # when an existing database is reopened.
        self._insert_counter = max(0, int(self.engine.next_insert_seq()) - 1)

        # Events live in an engine of their own: the word index and concept
        # graph are shared by every scope of an engine, so event records in
        # the main one changed the scores of every ordinary search.
        import os

        self._event_engine = None
        self._event_counter = 0
        event_dir = os.path.join(db_path, EVENT_DIR)
        if self.events is not None or os.path.isdir(event_dir):
            try:
                self._event_engine = turbomemory.MemoryEngine(db_path=event_dir,
                                                              dimension=self.dim)
            except BaseException:
                self.close()
                raise
            self._event_counter = max(0, int(self._event_engine.next_insert_seq()) - 1)

        # Observations, too, have an engine of their own: the long-term
        # record, searched apart from the messages it was written from.
        self._obs_engine = None
        self._obs_counter = 0
        self._observed: Dict[Optional[str], int] = {}  # user -> last message seq observed
        obs_dir = os.path.join(db_path, OBSERVATION_DIR)
        if self.observer is not None or os.path.isdir(obs_dir):
            try:
                self._obs_engine = turbomemory.MemoryEngine(db_path=obs_dir, dimension=self.dim)
            except BaseException:
                self.close()
                raise
            self._obs_counter = max(0, int(self._obs_engine.next_insert_seq()) - 1)

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
        reported = self._events_of(messages)
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
            for event in reported.get(turn, ()):
                text = event_text(event)
                # Two mentions of one kind of thing are what a count needs:
                # the write gate only skips a repeat within one message.
                norm = (turn, " ".join(text.lower().split()))
                if norm in seen_in_batch:
                    continue
                seen_in_batch.add(norm)
                facts.append(text)
                metas.append({"role": EVENT_ROLE, "timestamp": msg.get("timestamp", ""),
                              "content": content, "turn": turn,
                              "turn_index": msg.get("turn_index"), "passage": None,
                              "passages": 0, "event": event})
            context.append(content)

        if not facts:
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
                if meta.get("event") is not None:
                    self._event_counter += 1
                    ids.append(f"{user_id}_e{self._event_counter}" if user_id
                               else f"e{self._event_counter}")
                else:
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
                if meta.get("event") is not None:
                    payload["event"] = meta["event"]
                if meta["passage"] is None:
                    payload["original_message"] = meta["content"]
                else:
                    # A passage is the message: its place in it is enough.
                    payload["passage"] = meta["passage"]
                    payload["passages"] = meta["passages"]
                payloads.append(json.dumps(payload))
            mine = [k for k, meta in enumerate(metas) if meta.get("event") is None]
            reported = [k for k, meta in enumerate(metas) if meta.get("event") is not None]
            # The memories first, then their events: a failure on the events
            # leaves the memories stored and raises.
            for engine, rows in ((self.engine, mine), (self._event_engine, reported)):
                if not rows:
                    continue
                engine.insert_batch(
                    [ids[k] for k in rows],
                    [facts[k] for k in rows],
                    np.ascontiguousarray(embeddings[rows]),
                    [1.0] * len(rows),
                    [extract_concepts(facts[k]) for k in rows],
                    [payloads[k] for k in rows],
                    [user_id] * len(rows) if user_id is not None else None,
                    [str(metas[k]["role"]) for k in rows],
                )
            if self.max_user_tokens is not None and user_id is not None:
                self._touched.add(user_id)
            self._pending = True
        self._activity = time.monotonic()
        return len(facts)

    def _events_of(self, messages: List[Dict]) -> Dict[int, List[Dict]]:
        """The events each user message in ``messages`` reports, by its
        position, asked for all messages at once when the extractor can."""
        if self.events is None:
            return {}
        asked, items, context = [], [], []
        for turn, msg in enumerate(messages):
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if msg.get("role", "user") == "user":
                asked.append(turn)
                items.append((content, list(context)))
            context.append(content)
        many = getattr(self.events, "extract_many", None)
        found = (many(items) if many is not None
                 else [self.events.extract_events(m, c) for m, c in items])
        return {turn: list(events or []) for turn, events in zip(asked, found)}

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

    def _records(self, ids: List[str], engine=None) -> Dict[str, Dict]:
        """Stored record metadata for ``ids``, read back from the engine
        (``engine``, by default the memories' one).

        Maps id -> ``{"text", "scope", "source_role", "payload", ...}`` with
        ``payload`` parsed to a dict. Ids that are no longer live are omitted.
        """
        out: Dict[str, Dict] = {}
        for rec in (engine or self.engine).get_records(list(ids)):
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
        rerank: bool = False,
        reranker: Optional[object] = None,
        max_items: Optional[int] = None,
        time_tags: Union[bool, str] = True,
        sources: Optional[str] = None,
        source_tokens: int = SOURCE_TOKENS,
        event_dates: bool = False,
        events: Optional[float] = None,
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
        (``None`` for gists and for records stored before messages were
        kept). With ``sources="turn"`` the facts are the keys and the turns
        are what is returned (``tsm.sources.turn_units``): the facts found
        for one message become one result whose ``"text"`` is that message
        as it was said, cut to its sentences that bear on the facts and the
        question when it is longer than ``source_tokens``, with ``"facts"``
        and ``"ids"`` listing the facts it stands for. Its ``"context"``
        names the speaker after the time tag (``[2024-01-15] user: ...``).
        A fact does not always say what it is about ("The workshop was
        free"); the message it came from does.

        With ``event_dates``, a relative time in what is shown is followed by
        the date it points to, counted from the date the memory was said
        (``tsm.when.annotate``): "two weeks ago [week of 2023-05-08]". When
        something happened is then written out instead of left to the
        reader's arithmetic. A verified-superseded memory says by when it
        changed (``STALE_BY``: the date its replacement was said) whenever
        that date is known.

        With a token budget and a question that counts or totals
        (``tsm.ranking.is_counting_query``), the events stored with
        ``Memory(events=...)`` are searched too and the best of them take up
        to ``events`` of the budget (default ``EVENT_SHARE``; 0 turns this
        off), each one dated line with role ``"event"``, after the memories
        packed into the rest. Without stored events nothing changes.

        ``recall()`` is a read: asking does not change what the next call
        returns. Use ``reinforce()`` to tell the memory which results were
        actually used.

        With ``rerank=True`` or an active ``reranker`` (e.g. ``ColBertReranker``),
        retrieved candidate shortlists from TSM's cognitive graph are reranked
        using token-level MaxSim late interaction.
        """
        self._require_open()
        self._activity = time.monotonic()
        if time_tags not in (True, False, "date", "minute"):
            raise ValueError("time_tags must be True, False, 'date' or 'minute'")
        if sources not in (None, "turn"):
            raise ValueError("sources must be None or 'turn'")
        query_embedding = np.asarray(embed_query(self.embedder, query), dtype=np.float32)

        active_reranker = reranker or (self.reranker if rerank else None)
        if rerank and active_reranker is None:
            from .rerankers import ColBertReranker
            active_reranker = ColBertReranker()

        if token_budget is not None:
            # Enough candidates to fill the budget even with short memories.
            fetch_k = max(pool_k, 30, token_budget // 8)
        else:
            fetch_k = top_k * 3 if active_reranker else top_k
        results = self.engine.search(
            query_text=query,
            query_embedding=query_embedding,
            top_k=fetch_k,
            scope=user_id,
        )
        if not results:  # nothing in this scope matched
            return []

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
            return []
        if resolve_beliefs:
            self._resolve_beliefs(pool, user_id)
        if sources == "turn":
            pool = turn_units(pool, query, source_tokens)
        for p in pool:
            # Set before packing: what is shown is what the budget counts.
            shown = p["text"] or ""
            if event_dates and not is_gist(p["role"]):
                shown = annotate(shown, said_on(p["timestamp"]))
            if sources == "turn" and p.get("facts"):
                shown = speaker(p["role"]) + shown
            elif is_message(p["role"]):
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

        # Stage-2 MultiVector / ColBERT Late-Interaction Precision Reranking
        if active_reranker is not None and len(pool) > 1:
            candidate_texts = [p["text"] or "" for p in pool]
            rerank_scores = np.asarray(active_reranker.rerank(query, candidate_texts), dtype=np.float32)
            if len(rerank_scores) == len(pool):
                exp_scores = np.exp(rerank_scores - np.max(rerank_scores))
                norm_sim = exp_scores / np.sum(exp_scores)
                for idx, p in enumerate(pool):
                    p["maxsim_score"] = float(rerank_scores[idx])
                    p["score"] = float(p["score"]) * (1.0 + float(norm_sim[idx]) * len(pool))

        # Best first by the score that is returned (the role prior and a
        # brought-in current belief both move a memory's place).
        # On a tie the current belief goes before the one it replaced.
        pool.sort(key=lambda p: (-p["score"], "superseded_by" in p))
        if token_budget is None:
            return pool[:top_k]
        notes = []
        share = EVENT_SHARE if events is None else float(events)
        if share > 0 and is_counting_query(query):
            notes = self._event_notes(query, query_embedding, user_id,
                                      int(token_budget * min(share, 1.0)), time_tags)
        budget = max(1, token_budget - sum(estimate_tokens(n["context"]) for n in notes))
        chosen = self._pack(pool, budget, packing_lam(query, lam), max_items,
                            stored=sources is None)
        if sources is None and asks_what_assistant_said(query):
            chosen = self._whole_reply(chosen, budget)
        return chosen + notes

    def _event_notes(self, query: str, query_embedding, user_id: Optional[str],
                     budget: int, time_tags) -> List[Dict]:
        """The user's stored events that best match ``query``, best first, as
        recall results, within ``budget`` tokens. Empty when none are stored."""
        if budget <= 0 or self._event_engine is None:
            return []
        results = self._event_engine.search(query_text=query, query_embedding=query_embedding,
                                            top_k=EVENT_POOL, scope=user_id)
        if not results:
            return []
        records = self._records([mid for mid, _ in results], self._event_engine)
        notes, used = [], 0
        for mid, score in results:
            rec = records.get(mid)
            if rec is None or rec["scope"] != user_id:
                continue
            item = self._pool_item(mid, rec, float(score))
            event = rec["payload"].get("event") or {"what": rec["text"] or ""}
            shown = event_note(event, item["timestamp"])
            if time_tags:
                shown = time_tag(item["timestamp"], item["turn_index"],
                                 minutes=time_tags == "minute") + shown
            cost = estimate_tokens(shown)
            if used + cost > budget:
                continue  # a shorter one further down may still fit
            used += cost
            item.update(context=shown, event=event, source=rec["payload"].get("original_message"))
            notes.append(item)
        return notes

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
            return [], ""  # the text was rewritten for display (event_dates)
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
              max_items: Optional[int], stored: bool) -> List[Dict]:
        """``select_under_budget`` on ``pool``. When every candidate is a
        stored record read as stored (``stored``), the engine makes the same
        selection on the embeddings it already holds
        (``MemoryEngine.pack_under_budget``) instead of the candidates' text
        being embedded a second time."""
        pack = getattr(self.engine, "pack_under_budget", None)
        turns = [p.get("turn_index") for p in pool]
        if pack is not None and stored and all(
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

        With an ``observer``, every message not yet observed is then
        observed (``observe_pending``). A failure there stores nothing for
        that pass; the messages are observed on the next one.

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
                if self.observer is not None:
                    try:
                        self.observe_pending()
                    except Exception as e:  # noqa: BLE001 — nothing stored; retried later
                        self._retry = True
                        logger.warning("messages not observed (will retry): %s",
                                       type(e).__name__)
                if self.max_user_tokens is not None:
                    for user in sorted(self._touched):
                        try:
                            self._compact_user(user)
                        except Exception as e:  # noqa: BLE001 — nothing was removed; retried later
                            self._retry = True
                            logger.warning(
                                "memory of %r not compacted (kept as it is, will retry): %s",
                                user, type(e).__name__)
            except BaseException:
                self._pending = True
                raise
        return committed

    def observe_pending(self) -> int:
        """Observe every message not yet observed, for every user: a day's
        conversation at a time, all of them at once when the observer can.
        A user's observations of one pass are stored in one batch, so a pass
        either records all of them or (raising) none. Returns how many
        observations were stored."""
        self._require_open()
        if self.observer is None or self._obs_engine is None:
            return 0
        users = [u for u in (self.engine.scopes() if hasattr(self.engine, "scopes") else [])]
        work = []  # (user, last seq, conversations)
        for user in users:
            last, messages = self._unobserved(user)
            if messages:
                work.append((user, last, conversations(messages)))
        convs = [c for _, _, cs in work for c in cs]
        if not convs:
            return 0
        many = getattr(self.observer, "observe_many", None)
        found = many(convs) if many is not None else [self.observer.observe(c) for c in convs]
        stored, k = 0, 0
        for user, last, cs in work:
            texts, payloads = [], []
            for conv in cs:
                lines = found[k]
                k += 1
                first = conv[0]
                for line in lines:
                    texts.append(line)
                    payloads.append({"timestamp": first.get("timestamp", ""),
                                     "turn_index": first.get("turn_index"),
                                     "role": OBSERVATION_ROLE, "user_id": user,
                                     "through": last,
                                     "observes": [m.get("turn_index") for m in conv]})
            if not texts:
                # Nothing worth keeping: remember that it was read.
                self._observed[user] = last
                continue
            vectors = np.ascontiguousarray(embed_documents(self.embedder, texts),
                                           dtype=np.float32)
            with self._write_lock:
                ids = []
                for _ in texts:
                    self._obs_counter += 1
                    ids.append(f"{user}_o{self._obs_counter}" if user
                               else f"o{self._obs_counter}")
                self._obs_engine.insert_batch(
                    ids, texts, vectors, [1.0] * len(texts),
                    [extract_concepts(t) for t in texts],
                    [json.dumps(p) for p in payloads],
                    [user] * len(texts) if user is not None else None,
                    [OBSERVATION_ROLE] * len(texts))
            self._observed[user] = last
            stored += len(texts)
        return stored

    def _unobserved(self, user: Optional[str]) -> Tuple[int, List[Dict]]:
        """``(last message seq, messages)`` of ``user`` not yet observed, the
        messages rebuilt from their stored passages in the order said."""
        if user not in self._observed:
            through = 0
            for rec in self._obs_engine.get_records(self._obs_engine.scope_ids(user)):
                if rec is None:
                    continue
                try:
                    through = max(through, int(json.loads(rec["payload"]).get("through") or 0))
                except (TypeError, ValueError):
                    continue
            self._observed[user] = through
        through = self._observed[user]
        fresh = []
        for mid in self.engine.scope_ids(user):
            head, _, tail = str(mid).rpartition("_")
            if tail.isdigit() and int(tail) > through:
                fresh.append((int(tail), mid))
        if not fresh:
            return through, []
        fresh.sort()
        records = self._records([mid for _, mid in fresh])
        turns: Dict[object, Dict] = {}
        for seq, mid in fresh:
            rec = records.get(mid)
            if rec is None or not is_message(rec["source_role"]):
                continue
            payload = rec["payload"]
            turn = payload.get("turn_index")
            msg = turns.setdefault(turn, {"role": spoken_by(rec["source_role"]),
                                          "timestamp": payload.get("timestamp", ""),
                                          "turn_index": turn, "parts": [], "seq": seq})
            msg["parts"].append((payload.get("passage") or 0, rec["text"] or ""))
        messages = []
        for msg in sorted(turns.values(), key=lambda m: m["seq"]):
            msg["content"] = " ".join(text for _, text in sorted(msg.pop("parts")))
            messages.append(msg)
        return fresh[-1][0], messages

    def observations(self, user_id: Optional[str], query: Optional[str] = None,
                     token_budget: Optional[int] = None, time_tags: bool = True) -> List[Dict]:
        """The observations of ``user_id`` in the order said, each with
        ``"context"``: the date of its conversation, then the line, with each
        relative time followed by the date it points to. With ``query`` and
        ``token_budget``, only the observations that best match the query
        and fit the budget, still in the order said."""
        self._require_open()
        if self._obs_engine is None:
            return []
        if query is not None and token_budget is not None:
            vector = np.asarray(embed_query(self.embedder, query), dtype=np.float32)
            found = self._obs_engine.search(query_text=query, query_embedding=vector,
                                            top_k=max(50, int(token_budget) // 8),
                                            scope=user_id) or []
            scores = {mid: float(score) for mid, score in found}
            ids = [mid for mid, _ in found]
        else:
            ids, scores = list(self._obs_engine.scope_ids(user_id)), {}
        records = self._records(ids, self._obs_engine)
        items = []
        for mid in ids:
            rec = records.get(mid)
            if rec is None or rec["scope"] != user_id:
                continue
            item = self._pool_item(mid, rec, scores.get(mid, 0.0))
            item["context"] = observation_context(rec["text"] or "", item["timestamp"],
                                                  time_tags)
            items.append(item)
        if token_budget is not None:
            kept, used = [], 0
            for item in items:  # best first when searched
                cost = estimate_tokens(item["context"])
                if used + cost <= token_budget:
                    kept.append(item)
                    used += cost
            items = kept

        def order(item):
            seq = str(item["id"]).rpartition("_o")[2]
            return (str(said_on(item["timestamp"]) or ""), int(seq) if seq.isdigit() else 0)

        return sorted(items, key=order)

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

    def flush(self) -> None:
        """Durably persist all pending writes."""
        self._require_open()
        self.engine.flush()
        if self._event_engine is not None:
            self._event_engine.flush()
        if self._obs_engine is not None:
            self._obs_engine.flush()

    def close(self) -> None:
        """Flush and release the engine. Idempotent.

        Releases the database lock, mmaps, and worker threads, so ``db_path``
        can be reopened right away. Any later call on this object raises
        ``RuntimeError``.
        """
        if not self._closed:
            self._stop_idle_worker()
            try:
                for side in ("_obs_engine", "_event_engine"):
                    engine = getattr(self, side, None)
                    if engine is not None:
                        try:
                            engine.close()
                        except Exception as e:  # noqa: BLE001 — the main engine still closes
                            logger.warning("%s close failed: %s", side, e)
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
                                        (self.events, "flush_cache"),
                                        (self.observer, "flush_cache")):
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
