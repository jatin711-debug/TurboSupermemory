//! Top-level storage engine combining durable metadata, tiered vector segments,
//! and the cognitive graph.
//!
//! Durability model:
//!   1. Full embeddings are written to the mmap-backed `VectorStore` first.
//!   2. A metadata-only WAL entry is appended; it is the source of truth for
//!      record metadata and ordering.
//!   3. `redb` (via `MetadataStore`) is a lazy snapshot; it is flushed only on
//!      explicit `flush()` / background consolidation.
//!   4. On open we replay any un-flushed WAL entries, persist a snapshot, then
//!      rebuild the id index, graph, and tiered segments from the snapshot.
//!
//! Recovery never depends on a clean shutdown. A WAL insert carries a
//! checksum of its vector, and replay reads the vector straight from its slot
//! in the vector file, so everything a writer was told succeeded is back after
//! a kill. A torn or corrupt WAL tail is cut off at the last good frame.
//! Segment files are derived data: one that cannot be loaded is discarded and
//! its records are indexed again. What `open` had to repair is reported by
//! [`StorageEngine::recovery_report`].
//!
//! `StorageEngine` is one type whose methods are grouped by concern:
//!   - this file: the struct, `open`, record lookup, the consolidation cycle,
//!     `flush`, and `shutdown`;
//!   - [`write`]: insert / batch insert / delete / update (the WAL-ordered
//!     write path);
//!   - [`search`]: ANN, cognitive, filtered, and batch search;
//!   - [`retention`]: eviction, gist-before-evict, dedup, importance scoring;
//!   - [`belief`]: refinement / contradiction detection and supersession.

use crate::access_counters::AccessCounters;
use crate::config::StoreConfig;
use crate::gpu_exact::{GpuExactIndex, GpuSearchStats};
use crate::metadata_store::MetadataStore;
use crate::optimizer::BackgroundOptimizer;
use crate::payload_index::PayloadIndex;
use crate::record::{MetaRecord, PointOffset, Record};
use crate::scope_index::ScopeIndex;
use crate::segment_holder::{SegmentHolder, SegmentSnapshot};
use crate::text_index::TextIndex;
use crate::update_worker::UpdateWorker;
use crate::vector_store::VectorStore;
use crate::wal::{Wal, WalOp};
use ahash::HashMap as AHashMap;
use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use turbomemory_graph::{
    step_session_with_compressor, CognitiveCompressor, CompressedCognitiveState,
    DeterministicCompressor, MemoryGraph, SpreadingActivation, SpreadingConfig,
};

mod belief;
mod retention;
mod search;
mod write;

pub use belief::{BeliefResolution, ProposedSupersession, SupersessionKind};

const WAL_DIR: &str = "wal";

/// Compresses one chunk of eviction-victim texts into a single gist plus the
/// embedding to store it under (B4 gist-before-evict). Install via
/// `StorageEngine::set_gist_compressor`; only consulted when
/// `TierConfig::gist_before_evict` is enabled. Typically backed by an LLM
/// call plus an embedder — both live outside the engine, so the callback
/// supplies the vector, mirroring how `insert` takes caller embeddings.
///
/// The three outcomes mean different things to eviction:
/// - `Ok(Some((gist, embedding)))`: the gist is stored, then the victims are
///   deleted.
/// - `Ok(None)`: the compressor abstains (nothing in the chunk is worth
///   keeping); the victims are deleted without a gist.
/// - `Err(reason)`: the compressor failed (model unreachable, malformed
///   output). The victims are kept and tried again on the next eviction,
///   because deleting them now would lose the memories with nothing in their
///   place.
pub trait GistCompressor: Send + Sync {
    fn compress(&self, texts: &[String]) -> Result<Option<(String, Vec<f32>)>, String>;
}

/// What [`StorageEngine::open`] had to repair. Every field is zero for a store
/// that was shut down cleanly.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    /// Unflushed operations replayed from the write-ahead log.
    pub wal_ops_replayed: usize,
    /// Logged inserts dropped because their vector never reached the vector
    /// file (the process died between the two writes, or power was lost).
    pub wal_inserts_without_vector: usize,
    /// Bytes cut from the end of the log: a torn or corrupt tail.
    pub wal_bytes_discarded: u64,
    /// Segments that could not be loaded and were discarded. Their records
    /// are searchable again immediately and are re-indexed in the background.
    pub segments_discarded: usize,
    /// Segment directories removed because they were incomplete (a build
    /// that never finished) or superseded (already merged or compacted).
    pub segment_dirs_removed: usize,
    /// Graph memory nodes dropped because their record no longer exists.
    pub graph_nodes_pruned: usize,
}

impl RecoveryReport {
    /// True when the store opened without any repair.
    pub fn is_clean(&self) -> bool {
        *self == Self::default()
    }
}

/// The main storage engine.
pub struct StorageEngine {
    config: Arc<StoreConfig>,
    meta: Arc<MetadataStore>,
    pub(crate) vectors: Arc<VectorStore>,
    pub(crate) segments: Arc<RwLock<SegmentHolder>>,
    segment_snapshot: Arc<arc_swap::ArcSwap<SegmentSnapshot>>,
    graph: Arc<RwLock<SpreadingActivation>>,
    ccs: Arc<Mutex<Option<CompressedCognitiveState>>>,
    /// The cognitive compressor used by `step_session`. Defaults to
    /// `DeterministicCompressor`; callers can install an `LlmCompressor`
    /// via `set_compressor` to get LLM-driven working-memory compression.
    /// Stored behind `Arc<RwLock<Arc<...>>>` so it can be replaced at
    /// runtime and shared across engine clones. A `RwLock` is used instead
    /// of `ArcSwap` because `ArcSwap` does not support unsized `dyn Trait`
    /// types without additional wrapper boilerplate. The compressor is
    /// swapped rarely (once at setup), so the lock overhead is negligible.
    compressor: Arc<RwLock<Arc<dyn CognitiveCompressor>>>,
    /// Optional gist compressor for B4 gist-before-evict. `None` (default)
    /// means eviction victims are dropped outright even when
    /// `TierConfig::gist_before_evict` is on. Same `Arc<RwLock<...>>` pattern
    /// as `compressor`: installed once at setup, shared across engine clones.
    gist_compressor: Arc<RwLock<Option<Arc<dyn GistCompressor>>>>,
    /// Monotonic counter for gist record ids (`gist:{scope}:{now}:{n}`), so
    /// ids are unique within a process run; `now` disambiguates across runs.
    gist_seq: Arc<AtomicU64>,
    id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>,
    payload_index: Arc<RwLock<PayloadIndex>>,
    scope_index: Arc<RwLock<ScopeIndex>>,
    text_index: Arc<TextIndex>,
    wal: Arc<Mutex<Wal>>,
    /// Serializes `flush()` against in-flight writes. WAL-writing ops
    /// (insert / batch insert / delete) hold a read guard across
    /// `seq allocate → vectors.put → wal.append → meta apply`; `flush()`
    /// holds the write guard for its whole body, so its
    /// `{snapshot meta → wal.clear()}` sequence can never interleave with a
    /// write whose WAL entry it is about to truncate (which would silently
    /// drop the record on restart).
    flush_barrier: Arc<RwLock<()>>,
    /// Serializes the commit section of every write (id check, offset and
    /// sequence allocation, vector write, WAL append, index apply). Without
    /// it two writers could both pass the "id is free" check and both insert,
    /// and WAL order could differ from sequence order.
    write_lock: Arc<Mutex<()>>,
    /// What `open` had to repair.
    recovery: Arc<RecoveryReport>,
    optimizer: Arc<BackgroundOptimizer>,
    update_worker: Arc<UpdateWorker>,
    access_counters: Arc<AccessCounters>,
    /// Optional GPU backend for accelerated distance computation.
    /// Initialized lazily on first use; CPU fallback if CUDA unavailable.
    gpu: Arc<Mutex<Option<Arc<dyn turbomemory_gpu::GpuBackend>>>>,
    /// The store's vectors mirrored on the GPU for exact search. Decided once,
    /// at the first search that qualifies: `None` when the feature is off or
    /// there is no usable device.
    gpu_exact: Arc<OnceLock<Option<GpuExactIndex>>>,
    /// Seq-cursor for incremental supersession detection (W7): the `insert_seq`
    /// at/after which records still need supersession checking. Advanced past
    /// the current max seq after each consolidation. Only consulted when
    /// `incremental_supersession_detection` is enabled.
    supersession_watermark: Arc<AtomicU64>,
}

impl Clone for StorageEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            meta: self.meta.clone(),
            vectors: self.vectors.clone(),
            segments: self.segments.clone(),
            segment_snapshot: self.segment_snapshot.clone(),
            graph: self.graph.clone(),
            ccs: self.ccs.clone(),
            compressor: self.compressor.clone(),
            gist_compressor: self.gist_compressor.clone(),
            gist_seq: self.gist_seq.clone(),
            id_index: self.id_index.clone(),
            payload_index: self.payload_index.clone(),
            scope_index: self.scope_index.clone(),
            text_index: self.text_index.clone(),
            wal: self.wal.clone(),
            flush_barrier: self.flush_barrier.clone(),
            write_lock: self.write_lock.clone(),
            recovery: self.recovery.clone(),
            optimizer: self.optimizer.clone(),
            update_worker: self.update_worker.clone(),
            access_counters: self.access_counters.clone(),
            gpu: self.gpu.clone(),
            gpu_exact: self.gpu_exact.clone(),
            supersession_watermark: self.supersession_watermark.clone(),
        }
    }
}

impl StorageEngine {
    pub fn open(db_path: impl AsRef<Path>, config: StoreConfig) -> crate::Result<Arc<Self>> {
        let db_path = db_path.as_ref();

        // Fail fast on a TurboQuant tier configured for a non-power-of-two
        // dimension. TurboQuant relies on the in-place FWHT preconditioner
        // (lib.rs:fwht), which asserts that the vector length is a power of
        // two. The default `dimension = 768` is NOT a power of two, so
        // selecting `turbo_mse`/`turbo_prod` with the default config would
        // otherwise panic inside the quantizer constructor. Surface this as a
        // recoverable `InvalidArgument` error with an actionable message.
        let dim = config.dimension;
        for (name, kind) in [
            ("warm_quantizer", config.tier.warm_quantizer),
            ("cold_quantizer", config.tier.cold_quantizer),
        ] {
            if kind.requires_pow2_dim() && !dim.is_power_of_two() {
                return Err(crate::StorageError::InvalidArgument(format!(
                    "{name} {:?} requires a power-of-two dimension, but dimension is {dim}. \
                     Use a power-of-two dimension (e.g. 256, 512, 1024) or switch to a \
                     non-FWHT quantizer (scalar / sign).",
                    kind
                )));
            }
        }

        let meta = MetadataStore::open(db_path)?;
        let vectors = VectorStore::open(
            db_path.join("vectors.bin"),
            config.dimension,
            config.initial_capacity,
        )?;
        let wal_path = db_path.join(WAL_DIR);
        let mut wal = Wal::open(&wal_path)?;

        // Both must be read before replay changes them.
        let metadata_was_fresh = meta.is_fresh();
        let vectors_at_open = vectors.count();

        // Replay any un-flushed WAL entries into the metadata cache and vector store.
        let mut report = RecoveryReport::default();
        let last_applied = meta.last_applied_seq().unwrap_or(0);
        let mut max_seq = last_applied;
        let mut max_offset = 0u64;
        let mut replayed = false;
        let mut wal_iter = wal.iter()?;
        for op in wal_iter.by_ref() {
            match op? {
                WalOp::Insert {
                    offset,
                    seq,
                    meta: meta_rec,
                    vector_crc,
                } => {
                    if seq > last_applied {
                        // The embedding lives in the VectorStore mmap, in a
                        // slot the header may not count yet (the count is
                        // only stamped by flush). Read the slot itself and
                        // check it against the logged checksum; a mismatch
                        // means the vector never reached the file.
                        match vectors.recover(offset, vector_crc) {
                            Some(vec) => {
                                meta.put(offset, &meta_rec.with_embedding(Arc::from(vec)))?;
                                report.wal_ops_replayed += 1;
                            }
                            None => report.wal_inserts_without_vector += 1,
                        }
                        max_offset = max_offset.max(offset);
                        max_seq = max_seq.max(seq);
                        replayed = true;
                    }
                }
                WalOp::Replace {
                    old_offset,
                    offset,
                    seq,
                    meta: meta_rec,
                    vector_crc,
                } => {
                    if seq > last_applied {
                        // An update: swap old for new only if the new vector
                        // is really there, otherwise the old record stays.
                        match vectors.recover(offset, Some(vector_crc)) {
                            Some(vec) => {
                                meta.remove(old_offset)?;
                                meta.put(offset, &meta_rec.with_embedding(Arc::from(vec)))?;
                                report.wal_ops_replayed += 1;
                            }
                            None => report.wal_inserts_without_vector += 1,
                        }
                        max_offset = max_offset.max(offset);
                        max_seq = max_seq.max(seq);
                        replayed = true;
                    }
                }
                WalOp::Delete { offset } => {
                    meta.remove(offset)?;
                    // Vector data is left in place; it will be ignored because
                    // the metadata record is gone.
                    report.wal_ops_replayed += 1;
                    replayed = true;
                }
                WalOp::Flush { .. } => {}
            }
        }
        // Anything after the last readable frame is a torn or corrupt tail.
        // Cut it off so new entries are appended behind a good frame, not
        // behind garbage that the next replay would stop at.
        report.wal_bytes_discarded = wal_iter.discarded_bytes();
        let wal_valid_end = wal_iter.valid_end();
        drop(wal_iter);
        if report.wal_bytes_discarded > 0 {
            log::warn!(
                "WAL: discarded {} unreadable bytes after the last complete record",
                report.wal_bytes_discarded
            );
            wal.truncate_to(wal_valid_end)?;
        }

        if replayed || wal.needs_upgrade() {
            meta.advance_offset_past(max_offset);
            meta.advance_seq_past(max_seq);
            // Persist the recovered snapshot and discard the now-redundant WAL.
            vectors.flush()?;
            meta.flush(max_seq)?;
            wal.flush()?;
            wal.clear()?;
        }

        // The reverse of the check below: vectors that were flushed, but no
        // metadata at all and nothing in the log to rebuild it from. The
        // snapshot file was deleted or replaced by an empty one. Opening would
        // silently present an empty store on top of the old files.
        if metadata_was_fresh && !replayed && vectors_at_open > 0 {
            return Err(crate::StorageError::Corrupted(format!(
                "vectors.bin holds {vectors_at_open} vectors but memory.redb has no records:                  the metadata file is missing or was replaced (restore it from a backup)"
            )));
        }

        // Every live record must have its vector. The header count is stamped
        // before the metadata snapshot on every flush, and replay counts what
        // it recovers, so a record beyond the count means the vector file was
        // truncated, replaced, or deleted. Opening anyway would report the
        // records as present while no search could ever return them.
        if let Some(max_live) = meta.max_live_offset() {
            if max_live as usize >= vectors.count() {
                return Err(crate::StorageError::Corrupted(format!(
                    "metadata holds a record at offset {max_live} but vectors.bin only has \
                     {} vectors: the vector file is missing or truncated (restore it from a \
                     backup; the record texts are intact in memory.redb)",
                    vectors.count()
                )));
            }
        }

        // Collect metadata records once (no full HashMap clone) and rebuild
        // derived indexes from that collection.
        let mut records_meta: Vec<(PointOffset, MetaRecord)> =
            Vec::with_capacity(meta.record_count());
        meta.for_each_record(|offset, rec| records_meta.push((offset, rec.clone())))?;

        // Rebuild the payload index from the metadata snapshot.
        let payload_index = Arc::new(RwLock::new(PayloadIndex::from_meta_records_iter(
            records_meta.iter().map(|(o, m)| (*o, m)),
        )));

        // Rebuild the scope index from the metadata snapshot.
        let scope_index = Arc::new(RwLock::new(ScopeIndex::new()));
        {
            let mut sidx = scope_index.write();
            for (offset, meta_rec) in &records_meta {
                sidx.add(*offset, meta_rec.scope.as_deref());
            }
        }

        // Rebuild the full-text index from the metadata snapshot.
        let text_index = Arc::new(TextIndex::open(db_path.join("text_index"))?);
        for (offset, meta_rec) in &records_meta {
            text_index.add(*offset, &meta_rec.text)?;
        }
        text_index.commit()?;

        records_meta.sort_by(|a, b| {
            a.1.created_at
                .cmp(&b.1.created_at)
                .then(a.1.insert_seq.cmp(&b.1.insert_seq))
        });

        let view = vectors.read_view();
        let records_vec: Vec<(PointOffset, Record)> = records_meta
            .into_iter()
            .filter_map(|(offset, meta_rec)| {
                view.get(offset)
                    .map(|v| (offset, meta_rec.with_embedding(Arc::from(Vec::from(v)))))
            })
            .collect();
        drop(view);

        let id_index: AHashMap<Arc<str>, PointOffset> = records_vec
            .iter()
            .map(|(offset, rec)| (Arc::from(rec.id.as_str()), *offset))
            .collect();
        // Load the persisted graph (if any) so learned edge weights and
        // abstraction nodes survive restart. Binary snapshots are preferred;
        // legacy JSON snapshots written by older builds still load via
        // fallback. Falls back to a full rebuild when no snapshot parses.
        let saved_graph = meta
            .load_meta_bytes("graph")
            .or_else(|| meta.load_meta_str("graph").map(String::into_bytes));
        let graph = rebuild_graph(
            &records_vec,
            saved_graph,
            &config.spreading,
            &mut report.graph_nodes_pruned,
        );
        let ccs = meta
            .load_meta_str("ccs")
            .and_then(|s| serde_json::from_str::<CompressedCognitiveState>(&s).ok());

        // Load any sealed Hot, Warm, and Cold segments that were persisted before
        // the last flush.  Their offsets are excluded from the rebuilt Hot segment.
        // Segments are derived from the records and vectors, so one that cannot
        // be loaded is discarded rather than failing the open: its records
        // simply land in the Hot segment below and are indexed again.
        let segments_dir = db_path.join("segments");
        let sealed_loaded = load_segment_dirs(
            &segments_dir.join(crate::segment_holder::SEALED_HOT_DIR),
            &mut report,
            |path| crate::segments::sealed_hot::SealedHotSegment::open(path, &config),
        )?;
        let warm_loaded = load_segment_dirs(
            &segments_dir.join(crate::config::Tier::Warm.name()),
            &mut report,
            |path| crate::segments::warm::WarmSegment::open(path),
        )?;
        let cold_loaded = load_segment_dirs(
            &segments_dir.join(crate::config::Tier::Cold.name()),
            &mut report,
            |path| crate::segments::cold::ColdSegment::open(path),
        )?;

        let mut sealed_offsets: HashSet<PointOffset> = HashSet::new();
        let mut cold_segments = Vec::with_capacity(cold_loaded.len());
        for (_, seg) in cold_loaded {
            sealed_offsets.extend(seg.offsets().iter().copied());
            cold_segments.push(seg);
        }
        // A Warm segment whose records are all in a Cold segment was already
        // compacted (the process stopped before its directory was removed).
        let cold_offsets = sealed_offsets.clone();
        let mut warm_segments = Vec::with_capacity(warm_loaded.len());
        for (path, seg) in warm_loaded {
            if seg.offsets().iter().all(|o| cold_offsets.contains(o)) {
                drop(seg);
                remove_superseded_dir(&path, &mut report);
            } else {
                sealed_offsets.extend(seg.offsets().iter().copied());
                warm_segments.push(seg);
            }
        }
        // Likewise a sealed Hot segment fully covered by larger ones is a
        // leftover from a merge. Largest first, so the merged segment wins.
        let mut sealed_loaded = sealed_loaded;
        sealed_loaded.sort_by(|a, b| {
            b.1.point_count()
                .cmp(&a.1.point_count())
                .then_with(|| a.0.cmp(&b.0))
        });
        let mut hot_offsets: HashSet<PointOffset> = HashSet::new();
        let mut sealed_segments = Vec::with_capacity(sealed_loaded.len());
        for (path, seg) in sealed_loaded {
            if seg.offsets().iter().all(|o| hot_offsets.contains(o)) {
                drop(seg);
                remove_superseded_dir(&path, &mut report);
            } else {
                hot_offsets.extend(seg.offsets().iter().copied());
                sealed_segments.push(seg);
            }
        }
        sealed_offsets.extend(hot_offsets);

        let segments = SegmentHolder::from_records(
            config.clone(),
            segments_dir,
            &records_vec,
            &sealed_offsets,
            &vectors,
        )?;
        if !report.is_clean() {
            log::warn!("store recovered on open: {report:?}");
        }
        for seg in sealed_segments {
            segments.add_sealed_hot(seg);
        }
        for seg in warm_segments {
            segments.add_warm(seg);
        }
        for seg in cold_segments {
            segments.add_cold(seg);
        }

        let interval = config.auto_consolidation_interval;

        let meta = Arc::new(meta);
        let vectors = Arc::new(vectors);
        let segment_snapshot = segments.snapshot_handle();
        let segments = Arc::new(RwLock::new(segments));
        let graph = Arc::new(RwLock::new(graph));
        let id_index = Arc::new(RwLock::new(id_index));
        let budget = Arc::new(crate::optimizer::ResourceBudget::new(
            config.optimizer_budget.clone(),
        ));
        let access_counters = Arc::new(AccessCounters::new(config.tier.actr_history));
        let compressor: Arc<RwLock<Arc<dyn CognitiveCompressor>>> =
            Arc::new(RwLock::new(Arc::new(DeterministicCompressor)));
        // Snapshot before `meta` is moved into the cyclic closure.
        let meta_next_seq = meta.next_seq();

        Ok(Arc::new_cyclic(move |weak| {
            let optimizer = BackgroundOptimizer::new(weak.clone(), interval, budget);
            let applier = Arc::new(crate::update_worker::IndexApplier {
                meta: meta.clone(),
                vectors: vectors.clone(),
                segments: segments.clone(),
                graph: graph.clone(),
                id_index: id_index.clone(),
                payload_index: payload_index.clone(),
                scope_index: scope_index.clone(),
                text_index: text_index.clone(),
            });
            let update_worker = UpdateWorker::new(applier, 1024);
            Self {
                config: Arc::new(config),
                meta,
                vectors,
                segments,
                segment_snapshot,
                graph,
                ccs: Arc::new(Mutex::new(ccs)),
                compressor,
                gist_compressor: Arc::new(RwLock::new(None)),
                gist_seq: Arc::new(AtomicU64::new(0)),
                id_index,
                payload_index,
                scope_index,
                text_index,
                wal: Arc::new(Mutex::new(wal)),
                flush_barrier: Arc::new(RwLock::new(())),
                write_lock: Arc::new(Mutex::new(())),
                recovery: Arc::new(report),
                optimizer: Arc::new(optimizer),
                update_worker: Arc::new(update_worker),
                access_counters,
                gpu: Arc::new(Mutex::new(None)),
                gpu_exact: Arc::new(OnceLock::new()),
                // Records already present at open (reloaded history) are treated
                // as already-checked; only inserts after this point need
                // incremental supersession detection.
                supersession_watermark: Arc::new(AtomicU64::new(meta_next_seq)),
            }
        }))
    }

    /// Lazily initialize the GPU backend if not already done.
    /// Returns the backend, or CPU fallback if CUDA is unavailable.
    fn gpu_backend(&self) -> Arc<dyn turbomemory_gpu::GpuBackend> {
        let mut gpu = self.gpu.lock();
        if gpu.is_none() {
            let backend = turbomemory_gpu::init_backend();
            *gpu = Some(backend);
        }
        gpu.as_ref().unwrap().clone()
    }

    /// Check if the GPU backend is actually GPU-accelerated (not CPU fallback).
    pub fn is_gpu_accelerated(&self) -> bool {
        turbomemory_gpu::is_gpu_accelerated(&self.gpu_backend())
    }

    /// The GPU exact-search mirror, if this engine uses one. The decision is
    /// made once: the feature must be on and a real device must be present
    /// (or host emulation requested, for tests).
    pub(crate) fn gpu_exact(&self) -> Option<&GpuExactIndex> {
        self.gpu_exact
            .get_or_init(|| {
                let tier = &self.config.tier;
                if !tier.gpu_exact_search {
                    return None;
                }
                let backend = self.gpu_backend();
                if !turbomemory_gpu::is_gpu_accelerated(&backend) && !tier.gpu_exact_on_cpu_backend
                {
                    return None;
                }
                Some(GpuExactIndex::new(
                    backend,
                    self.config.dimension,
                    tier.gpu_memory_budget_mb,
                ))
            })
            .as_ref()
    }

    /// What the GPU search mirror holds. `None` when this engine has not
    /// used one: the feature is off, there is no device, or no search has
    /// qualified yet (the store is below `gpu_exact_min_records`).
    pub fn gpu_search_stats(&self) -> Option<GpuSearchStats> {
        self.gpu_exact.get()?.as_ref().map(GpuExactIndex::stats)
    }

    /// What the `open` call that produced this engine had to repair.
    pub fn recovery_report(&self) -> &RecoveryReport {
        &self.recovery
    }

    /// The `insert_seq` the next inserted record will receive. Durable and
    /// monotonically increasing: it is never reused across restarts, deletes,
    /// or eviction, so callers can derive collision-free ids from it.
    pub fn next_insert_seq(&self) -> u64 {
        self.meta.next_seq()
    }

    /// Whether a record with this id is currently live (present in the id
    /// index, i.e. not evicted/deleted).
    pub fn contains_id(&self, id: &str) -> bool {
        self.id_index.read().contains_key(id)
    }

    /// Return the JSON payload attached to a record, if any.
    pub fn get_payload(&self, id: &str) -> crate::Result<Option<String>> {
        let idx = self.id_index.read();
        let Some(&offset) = idx.get(id) else {
            return Ok(None);
        };
        drop(idx);
        match self.meta.get(offset)? {
            Some(meta) => Ok(meta.payload.clone()),
            None => Ok(None),
        }
    }

    /// Hydrate a full `Record` from the metadata cache + vector store.
    fn get_record(&self, offset: PointOffset) -> Option<Record> {
        let meta = self.meta.get(offset).ok().flatten()?;
        let view = self.vectors.read_view();
        let vec = view.get(offset)?;
        Some(meta.with_embedding(Arc::from(Vec::from(vec))))
    }

    /// Look up a live record by its caller-supplied id, hydrated with its
    /// embedding. `None` when the id is unknown or the record was deleted.
    pub fn find_record_by_id(&self, id: &str) -> Option<Record> {
        let idx = self.id_index.read();
        idx.get(id)
            .copied()
            .and_then(|offset| self.get_record(offset))
    }

    /// Look up metadata for a record by its caller-supplied id (without embedding).
    pub fn find_meta_by_id(&self, id: &str) -> Option<MetaRecord> {
        let idx = self.id_index.read();
        idx.get(id)
            .copied()
            .and_then(|offset| self.meta.get(offset).ok().flatten())
    }

    /// Bump the access score for the record with the given offset.
    ///
    /// Writes go to the fast in-memory `AccessCounters` instead of the metadata
    /// cache so that searches do not contend on the metadata write lock.
    fn bump_access(&self, offset: PointOffset) {
        self.access_counters.bump(offset, now_secs());
    }

    fn bump_access_by_id(&self, id: &str) {
        let idx = self.id_index.read();
        if let Some(&offset) = idx.get(id) {
            drop(idx);
            self.bump_access(offset);
        }
    }

    /// Reinforce the cognitive-graph edges of a retrieved memory (rehearsal).
    /// Called alongside `bump_access_by_id` on every cognitive-search result
    /// so that frequently-recalled memories get stronger graph links over
    /// time. This is the "retain what matters" learning loop: retrieval
    /// itself is the signal that a memory was useful.
    fn reinforce_graph_ids(&self, ids: &[&str]) {
        if ids.is_empty() {
            return;
        }
        // One write lock for the whole batch instead of one per hit: a
        // cognitive search previously serialized against every other search
        // and against consolidation top_k times.
        let now = now_secs();
        let mut graph = self.graph.write();
        for id in ids {
            graph.reinforce(id, now);
        }
    }

    pub fn step_session(
        &self,
        user_input: &str,
        assistant_response: &str,
    ) -> crate::Result<String> {
        let ccs_json = self.ccs.lock().as_ref().map(|c| c.to_json());
        let compressor = self.compressor.read().clone();
        let json = step_session_with_compressor(
            compressor.as_ref(),
            ccs_json.as_deref(),
            user_input,
            assistant_response,
        );
        *self.ccs.lock() = serde_json::from_str(&json).ok();
        self.save_ccs()?;
        Ok(json)
    }

    /// Install a custom cognitive compressor (e.g. an `LlmCompressor`).
    ///
    /// The default compressor is `DeterministicCompressor`. Call this to
    /// replace it with an LLM-backed compressor so that `step_session`
    /// distills turns using an external model instead of the deterministic
    /// keyword extractor.
    pub fn set_compressor(&self, compressor: Arc<dyn CognitiveCompressor>) {
        *self.compressor.write() = compressor;
    }

    /// Install (or clear) the gist compressor used by B4 gist-before-evict.
    /// Only consulted by `evict` when `TierConfig::gist_before_evict` is on;
    /// `None` restores drop-outright eviction.
    pub fn set_gist_compressor(&self, compressor: Option<Arc<dyn GistCompressor>>) {
        *self.gist_compressor.write() = compressor;
    }

    fn save_ccs(&self) -> crate::Result<()> {
        if let Some(ccs) = self.ccs.lock().as_ref() {
            self.meta.save_meta("ccs", &ccs.to_json())?;
        }
        Ok(())
    }

    fn save_graph(&self) -> crate::Result<()> {
        let bytes = self.graph.read().graph().to_snapshot_bytes()?;
        self.meta.save_meta_bytes("graph", &bytes)?;
        // Reclaim the legacy JSON snapshot (pre-binary databases) once the
        // binary snapshot is durable. No-op when already gone.
        self.meta.remove_meta("graph")?;
        Ok(())
    }

    /// Gracefully shut down the engine.
    ///
    /// Stops the background optimizer (waiting for an in-flight cycle), then
    /// flushes the WAL, vector store, metadata snapshot, and segment files.
    /// Stopping first means no worker can still hold a strong engine reference
    /// afterwards, so dropping the caller's last `Arc` releases the database
    /// lock, mmaps, and index files immediately. The engine stays usable for
    /// foreground calls; only automatic consolidation is off from here on.
    pub fn shutdown(&self) -> crate::Result<()> {
        self.optimizer.stop();
        self.flush()
    }

    pub fn trigger_consolidation(&self) -> crate::Result<(usize, usize, usize)> {
        // Make sure recent access counts are visible to the promotion scorer.
        self.access_counters.drain_into(&self.meta)?;
        let segments = self.segments.read();
        let (sealed, compacted, promoted) =
            segments.trigger_consolidation(&self.meta, &self.vectors)?;
        drop(segments);

        // Drain the background optimizer so sealed/merged segments are fully
        // materialized before the caller begins searching.
        self.optimizer.drain(self);

        // Automatic importance scoring: adjust each record's importance based
        // on retrieval patterns + connectivity, then sync the graph. Runs
        // before dedup/eviction so recomputed importance participates in
        // dedup tiebreaking and eviction ranking. Opt-in (no-op when off).
        self.recompute_importance()?;

        // Semantic dedup first (merges duplicates), then bounded-storage
        // eviction (drops low-salience records). Both are opt-in and no-op
        // when their config thresholds are unset.
        self.deduplicate()?;
        self.evict()?;

        // Memory evolution: detect refinements + contradictions and commit the
        // resulting Refines/Contradicts edges + demotion so retrieval surfaces
        // the current belief. Refinements run before contradictions so a
        // high-text-overlap pair is treated as a refinement, not double-counted
        // as a contradiction. Both opt-in.
        //
        // When `defer_supersession_commit` is set, consolidation does NOT commit
        // here — the caller drives `propose_supersessions` → verify →
        // `commit_supersessions` so a verifier can vet each demotion (W3).
        if !self.config.tier.defer_supersession_commit {
            self.check_refinements()?;
            self.check_contradictions()?;
        }
        // Advance the incremental watermark past everything inserted so far, so
        // the next cycle only checks records added after this one. (When the
        // flag is off the watermark is never read, so this is harmless.)
        // Only when detection actually ran here: with a deferred commit the
        // caller's `propose_supersessions` does the detecting, and advancing
        // the cursor first would leave it nothing to propose.
        if !self.config.tier.defer_supersession_commit {
            self.supersession_watermark
                .store(self.meta.next_seq(), Ordering::Relaxed);
        }

        // Cognitive-layer learning: decay stale reinforced edges and build
        // abstraction hierarchies from concept co-occurrence. Both are opt-in
        // (no-op when their config is 0) so the default behavior is unchanged.
        let now = now_secs();
        let half_life = self.config.tier.edge_decay_half_life_secs;
        let abstraction_threshold = self.config.tier.abstraction_co_occurrence_threshold;
        {
            let mut graph = self.graph.write();
            if half_life > 0 {
                graph.decay_edges(now, half_life);
            }
            if abstraction_threshold > 0 {
                graph.build_abstractions(abstraction_threshold);
            }
        }

        // Online concept vocabulary evolution: merge similar concept nodes
        // and suppress over-general hubs. Opt-in (no-op when disabled).
        self.evolve_concept_vocabulary()?;

        self.save_graph()?;
        Ok((sealed, compacted, promoted))
    }

    /// Run one pass of online concept vocabulary evolution.
    ///
    /// Merges concept nodes whose associated memory sets overlap strongly
    /// (Jaccard >= `concept_merge_overlap_threshold`) and suppresses base
    /// concepts whose degree exceeds `concept_hub_degree_fraction` of all
    /// memories. Work is capped by `concept_evolution_max_pairs_per_cycle`.
    ///
    /// Returns `(merged, newly_suppressed, examined_pairs)`. No-op when
    /// `concept_evolution_enabled` is false.
    /// Run one pass of online concept vocabulary evolution.
    ///
    /// Merges concept nodes whose associated memory sets overlap strongly
    /// (Jaccard >= `concept_merge_overlap_threshold`) and suppresses base
    /// concepts whose degree exceeds `concept_hub_degree_fraction` of all
    /// memories. Work is capped by `concept_evolution_max_pairs_per_cycle`.
    ///
    /// Returns `(merged, newly_suppressed, examined_pairs)`. No-op when
    /// `concept_evolution_enabled` is false.
    pub fn evolve_concept_vocabulary(&self) -> crate::Result<(usize, usize, usize)> {
        let tier = &self.config.tier;
        if !tier.concept_evolution_enabled {
            return Ok((0, 0, 0));
        }
        let overlap = tier.concept_merge_overlap_threshold.clamp(0.0, 1.0);
        let hub = tier.concept_hub_degree_fraction.max(0.0);
        let max_pairs = tier.concept_evolution_max_pairs_per_cycle;
        let stats = {
            let mut graph = self.graph.write();
            graph.evolve_vocabulary(overlap, hub, max_pairs)
        };
        if stats.merged > 0 || stats.suppressed > 0 {
            self.save_graph()?;
        }
        Ok((stats.merged, stats.suppressed, stats.examined_pairs))
    }

    pub fn flush(&self) -> crate::Result<()> {
        // Exclusive flush barrier: wait for in-flight writes to finish and
        // hold off new ones until the snapshot is durable and the WAL is
        // truncated. Without this, an insert racing steps 4–7 could have its
        // WAL entry cleared without ever reaching the redb snapshot — silent
        // record loss on restart.
        let _flush_guard = self.flush_barrier.write();

        // 1. Build any pending plain segments so the durable snapshot captures
        //    them as persisted HNSW / quantized segments rather than in-memory
        //    plain indexes. A build failure must not stop the durable part of
        //    the flush: segments are rebuildable, the records are not. The
        //    plain segment stays searchable and the error is returned once
        //    everything else is safely on disk.
        let mut seal_error = None;
        loop {
            match self.optimizer.process_one_seal(self) {
                Ok(true) => {}
                Ok(false) => break,
                Err(e) => {
                    seal_error = Some(e);
                    break;
                }
            }
        }

        // 2. Drain access counters into the metadata cache before snapshotting it.
        self.access_counters.drain_into(&self.meta)?;

        // 3. Durably sync the WAL.
        {
            let mut wal = self.wal.lock();
            wal.flush()?;
        }

        // 4. Persist the vector snapshot, metadata snapshot, and text index.
        self.vectors.flush()?;
        self.text_index.flush()?;
        let last_applied_seq = self.meta.next_seq().saturating_sub(1);
        self.meta.flush(last_applied_seq)?;

        // 5. Flush tiered segment files.
        let segments = self.segments.read();
        segments.flush()?;
        drop(segments);

        // 6. Persist graph / CCS metadata.
        self.save_graph()?;
        self.save_ccs()?;

        // 7. WAL is now fully captured by the redb snapshot; truncate it.
        {
            let mut wal = self.wal.lock();
            wal.clear()?;
        }

        match seal_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    pub fn record_count(&self) -> usize {
        self.meta.record_count()
    }

    /// Read-only access to the cognitive graph's learned state (concept
    /// nodes, edge weights, refinement/contradiction edges, abstraction
    /// hierarchy). Holds a `parking_lot` read lock on the graph for the
    /// lifetime of the returned guard — callers should drop it promptly.
    ///
    /// Intended for the introspection API (`graph_stats`, `get_concepts`,
    /// `get_memory_concepts`, `get_refinements`, `get_contradictions`) and
    /// debugging. Acquire, call `MemoryGraph` methods on `.graph()`, drop.
    pub fn read_graph(&self) -> parking_lot::RwLockReadGuard<'_, SpreadingActivation> {
        self.graph.read()
    }

    pub fn config(&self) -> &StoreConfig {
        &self.config
    }

    /// Flush only the vector store to disk.
    ///
    /// This is exposed primarily for crash-recovery tests that need embeddings
    /// to be durable while leaving the WAL un-snapshoted.
    pub fn flush_vectors(&self) -> crate::Result<()> {
        self.vectors.flush()
    }

    /// Flush only the WAL to disk.
    ///
    /// Exposed for crash-recovery tests that need the WAL durable without
    /// snapshoting metadata.
    pub fn flush_wal(&self) -> crate::Result<()> {
        let mut wal = self.wal.lock();
        wal.flush()
    }
}

fn build_graph(records: &[(PointOffset, Record)], config: &SpreadingConfig) -> SpreadingActivation {
    let mut graph = MemoryGraph::new();
    for (_, rec) in records {
        graph.add_memory_scoped(
            &rec.id,
            &rec.text,
            &rec.concepts,
            rec.importance,
            rec.scope.as_deref(),
        );
    }
    SpreadingActivation::new(graph, config.clone())
}

/// Rebuild the cognitive graph, preserving learned edge weights and
/// abstraction nodes from a previously-saved graph when available.
///
/// The persisted graph snapshot (written by `save_graph`) captures learned
/// state that is not reconstructable from records alone: reinforced edge
/// weights, reinforcement timestamps, and abstraction (parent concept) nodes.
/// Snapshots are binary (magic-prefixed bincode); snapshots without the magic
/// are treated as legacy JSON from pre-binary databases. If the persisted
/// graph is present, we load it and add only records that are not already
/// memory nodes in it (using `insert_seq` ordering to decide which records
/// are new). If the persisted graph is absent or fails to parse, we fall
/// back to a full rebuild from records — this preserves the pre-learning
/// behavior and is always correct, just without learned state.
fn rebuild_graph(
    records: &[(PointOffset, Record)],
    saved_graph: Option<Vec<u8>>,
    config: &SpreadingConfig,
    pruned: &mut usize,
) -> SpreadingActivation {
    let Some(bytes) = saved_graph else {
        return build_graph(records, config);
    };
    let parsed = if MemoryGraph::is_snapshot_bytes(&bytes) {
        MemoryGraph::from_snapshot_bytes(&bytes).ok()
    } else {
        std::str::from_utf8(&bytes)
            .ok()
            .and_then(|json| MemoryGraph::from_json(json).ok())
    };
    let Some(mut graph) = parsed else {
        return build_graph(records, config);
    };
    // Add records that are not already memory nodes in the persisted graph.
    // Records are sorted by (created_at, insert_seq) by the caller, so we
    // process them in insertion order, preserving temporal chaining for the
    // new tail. We track the last memory id seen so temporal edges chain
    // correctly from the last persisted memory to the first new one.
    let mut existing_mem_ids: HashSet<String> = graph
        .iter_memory_nodes()
        .map(|(k, _)| k.strip_prefix("mem:").unwrap_or(&k).to_string())
        .collect();
    // Drop memory nodes whose record is gone. The snapshot is saved at
    // consolidation and flush, while deletes take effect immediately, so after
    // an unclean stop it can still hold deleted memories: their text, and any
    // supersession edge through which a deleted record would keep hiding a
    // live one.
    let live_ids: HashSet<&str> = records.iter().map(|(_, r)| r.id.as_str()).collect();
    let stale: Vec<String> = existing_mem_ids
        .iter()
        .filter(|id| !live_ids.contains(id.as_str()))
        .cloned()
        .collect();
    for id in &stale {
        graph.remove_memory(id);
        existing_mem_ids.remove(id);
    }
    *pruned += stale.len();
    // Reset last_memory_id so new temporal edges chain from the most recent
    // persisted memory (if any) rather than from an arbitrary one. We find
    // the last memory by scanning the existing set — the graph stores nodes
    // in a BTreeMap keyed by "mem:{id}", so we pick the lexicographically
    // largest. This is a heuristic; exact temporal ordering of persisted
    // memories is not recoverable from the graph alone, but the chain only
    // matters for the *new* tail, and any persisted memory as the chain
    // anchor is sufficient for that.
    if let Some((last_key, _)) = graph.iter_memory_nodes().last() {
        let last_id = last_key
            .strip_prefix("mem:")
            .unwrap_or(&last_key)
            .to_string();
        graph_reset_last_memory(&mut graph, &last_id);
    }
    let mut added_any = false;
    for (_, rec) in records {
        if existing_mem_ids.contains(&rec.id) {
            continue;
        }
        graph.add_memory_scoped(
            &rec.id,
            &rec.text,
            &rec.concepts,
            rec.importance,
            rec.scope.as_deref(),
        );
        added_any = true;
    }
    let _ = added_any; // suppress unused warning when no new records
    SpreadingActivation::new(graph, config.clone())
}

/// Helper to set the `last_memory_id` of a `MemoryGraph` so that the next
/// `add_memory` call chains temporally from the given id. We do this by
/// inserting a no-op: since `last_memory_id` is private, we exploit the fact
/// that calling `add_memory` on an existing id re-inserts it and re-chains.
/// Actually, the cleanest approach is to not fight the encapsulation: if no
/// new records are added, the temporal chain doesn't matter. If new records
/// are added, they chain from whatever `last_memory_id` the deserialized
/// graph carries. Since the graph was serialized after potentially many
/// adds, `last_memory_id` is already the last-added memory's id. So this
/// function is a no-op — we keep it as a documented placeholder.
fn graph_reset_last_memory(_graph: &mut MemoryGraph, _id: &str) {
    // No-op: the deserialized graph already carries `last_memory_id` from the
    // last `add_memory` call before serialization. New records will chain
    // from it naturally. See `rebuild_graph` for the rationale.
}

/// Load every segment directory under `dir`, sorted by path.
///
/// A directory without a manifest is a build that never finished and is
/// removed. A directory whose segment fails to load is discarded too; the
/// caller indexes its records again.
fn load_segment_dirs<T>(
    dir: &Path,
    report: &mut RecoveryReport,
    open: impl Fn(&Path) -> crate::Result<T>,
) -> crate::Result<Vec<(PathBuf, T)>> {
    let mut loaded = Vec::new();
    if !dir.exists() {
        return Ok(loaded);
    }
    let mut paths: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            paths.push(path);
        }
    }
    paths.sort();
    for path in paths {
        if !path.join(crate::segments::MANIFEST_FILE).exists() {
            if std::fs::remove_dir_all(&path).is_ok() {
                report.segment_dirs_removed += 1;
            }
            continue;
        }
        match open(&path) {
            Ok(segment) => loaded.push((path, segment)),
            Err(e) => {
                log::warn!(
                    "discarding unreadable segment {}: {e}; its records will be indexed again",
                    path.display()
                );
                report.segments_discarded += 1;
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }
    Ok(loaded)
}

fn remove_superseded_dir(path: &Path, report: &mut RecoveryReport) {
    if std::fs::remove_dir_all(path).is_ok() {
        report.segment_dirs_removed += 1;
    }
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests;
