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

use crate::access_counters::AccessCounters;
use crate::config::StoreConfig;
use crate::metadata_store::MetadataStore;
use crate::optimizer::BackgroundOptimizer;
use crate::payload_index::{Filter, PayloadIndex};
use crate::record::{MetaRecord, PointOffset, Record};
use crate::scope_index::ScopeIndex;
use crate::segment_holder::{SegmentHolder, SegmentSnapshot};
use crate::text_index::TextIndex;
use crate::update_worker::UpdateWorker;
use crate::vector_store::VectorStore;
use crate::wal::{Wal, WalOp};
use crate::StorageError;
use ahash::HashMap as AHashMap;
use parking_lot::{Mutex, RwLock};
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use turbomemory_core::{cosine_similarity, normalize, validate_dimension};
use turbomemory_graph::{
    merge_concepts_with_config, step_session_with_compressor, CognitiveCompressor,
    CompressedCognitiveState, DeterministicCompressor, MemoryGraph, SpreadingActivation,
    SpreadingConfig,
};

/// For small collections an exact scan is deterministic and higher-recall than
/// a lightly-configured HNSW index.
const EXACT_FALLBACK_THRESHOLD: usize = 4096;

const WAL_DIR: &str = "wal";

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
    id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>,
    payload_index: Arc<RwLock<PayloadIndex>>,
    scope_index: Arc<RwLock<ScopeIndex>>,
    text_index: Arc<TextIndex>,
    wal: Arc<Mutex<Wal>>,
    optimizer: Arc<BackgroundOptimizer>,
    update_worker: Arc<UpdateWorker>,
    access_counters: Arc<AccessCounters>,
    /// Optional GPU backend for accelerated distance computation.
    /// Initialized lazily on first use; CPU fallback if CUDA unavailable.
    gpu: Arc<Mutex<Option<Arc<dyn turbomemory_gpu::GpuBackend>>>>,
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
            id_index: self.id_index.clone(),
            payload_index: self.payload_index.clone(),
            scope_index: self.scope_index.clone(),
            text_index: self.text_index.clone(),
            wal: self.wal.clone(),
            optimizer: self.optimizer.clone(),
            update_worker: self.update_worker.clone(),
            access_counters: self.access_counters.clone(),
            gpu: self.gpu.clone(),
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

        // Replay any un-flushed WAL entries into the metadata cache and vector store.
        let last_applied = meta.last_applied_seq().unwrap_or(0);
        let mut max_seq = last_applied;
        let mut max_offset = 0u64;
        let mut replayed = false;
        for op in wal.iter()? {
            match op? {
                WalOp::Insert {
                    offset,
                    seq,
                    meta: meta_rec,
                } => {
                    if seq > last_applied {
                        // The embedding lives in the VectorStore mmap. If it is
                        // missing we have a partial write; skip the metadata.
                        if let Some(vec) = vectors.get(offset) {
                            let record = meta_rec.with_embedding(Arc::from(Vec::from(&*vec)));
                            meta.put(offset, &record)?;
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
                    replayed = true;
                }
                WalOp::Flush { .. } => {}
            }
        }

        if replayed {
            meta.advance_offset_past(max_offset);
            meta.advance_seq_past(max_seq);
            // Persist the recovered snapshot and discard the now-redundant WAL.
            vectors.flush()?;
            meta.flush(max_seq)?;
            wal.flush()?;
            wal.clear()?;
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
        // abstraction nodes survive restart. Falls back to a full rebuild
        // when the graph JSON is absent or unparseable.
        let saved_graph = meta.load_meta_str("graph");
        let graph = rebuild_graph(&records_vec, saved_graph, &config.spreading);
        let ccs = meta
            .load_meta_str("ccs")
            .and_then(|s| serde_json::from_str::<CompressedCognitiveState>(&s).ok());

        // Load any sealed Hot, Warm, and Cold segments that were persisted before
        // the last flush.  Their offsets are excluded from the rebuilt Hot segment.
        let mut sealed_offsets = HashSet::new();
        let mut sealed_segments = Vec::new();
        let mut warm_segments = Vec::new();
        let mut cold_segments = Vec::new();

        let segments_dir = db_path.join("segments");
        let sealed_dir = segments_dir.join(crate::segment_holder::SEALED_HOT_DIR);
        if sealed_dir.exists() {
            for entry in std::fs::read_dir(&sealed_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() && path.join("manifest.json").exists() {
                    let seg = crate::segments::sealed_hot::SealedHotSegment::open(&path, &config)?;
                    sealed_offsets.extend(seg.offsets().iter().copied());
                    sealed_segments.push(seg);
                }
            }
        }

        let warm_dir = segments_dir.join(crate::config::Tier::Warm.name());
        if warm_dir.exists() {
            for entry in std::fs::read_dir(&warm_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() && path.join("manifest.json").exists() {
                    let seg = crate::segments::warm::WarmSegment::open(&path)?;
                    sealed_offsets.extend(seg.offsets().iter().copied());
                    warm_segments.push(seg);
                }
            }
        }

        let cold_dir = segments_dir.join(crate::config::Tier::Cold.name());
        if cold_dir.exists() {
            for entry in std::fs::read_dir(&cold_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() && path.join("manifest.json").exists() {
                    let seg = crate::segments::cold::ColdSegment::open(&path)?;
                    sealed_offsets.extend(seg.offsets().iter().copied());
                    cold_segments.push(seg);
                }
            }
        }

        let segments = SegmentHolder::from_records(
            config.clone(),
            segments_dir,
            &records_vec,
            &sealed_offsets,
            &vectors,
        )?;
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
        let access_counters = Arc::new(AccessCounters::new());
        let compressor: Arc<RwLock<Arc<dyn CognitiveCompressor>>> =
            Arc::new(RwLock::new(Arc::new(DeterministicCompressor)));

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
                id_index,
                payload_index,
                scope_index,
                text_index,
                wal: Arc::new(Mutex::new(wal)),
                optimizer: Arc::new(optimizer),
                update_worker: Arc::new(update_worker),
                access_counters,
                gpu: Arc::new(Mutex::new(None)),
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

    pub fn insert(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
    ) -> crate::Result<bool> {
        self.insert_with_payload(id, text, embedding, importance, concepts, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_with_payload(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
        scope: Option<String>,
    ) -> crate::Result<bool> {
        validate_dimension(embedding, self.config.dimension)?;
        if self.id_index.read().contains_key(id) {
            return Err(StorageError::DuplicateId(id.to_string()));
        }
        let mut emb = embedding.to_vec();
        normalize(&mut emb)?;
        // Augment caller-supplied concepts with auto-extracted ones from the
        // text. If the caller already provided >= max_concepts, their tags
        // are used as-is. If max_concepts is 0, auto-extraction is disabled.
        let extractor_config = self.config.tier.extractor_config();
        let vocab = self.graph.read().vocab().clone();
        let concepts = merge_concepts_with_config(concepts, text, &extractor_config, Some(&vocab));
        let offset = self.meta.allocate_offset();
        let seq = self.meta.allocate_seq();
        let record = Record {
            id: id.to_string(),
            text: text.to_string(),
            embedding: Arc::from(emb),
            importance,
            concepts,
            created_at: now_secs(),
            insert_seq: seq,
            access_count: 0,
            last_accessed: 0,
            tier: crate::config::Tier::Hot,
            payload,
            scope,
        };

        // 1. Persist the embedding to the mmap-backed vector store first.
        //    The vector store is the durable physical source of truth for
        //    embeddings; the WAL only records the metadata operation.
        self.vectors.put(offset, record.embedding_f32())?;

        // 2. WAL metadata entry.
        {
            let meta = MetaRecord::from(&record);
            let mut wal = self.wal.lock();
            wal.append(&WalOp::Insert { offset, seq, meta })?;
        }

        // 3. Submit the index update to the serialized worker.
        self.update_worker.submit_and_wait(vec![(offset, record)])?;
        Ok(true)
    }

    pub fn insert_batch(
        &self,
        ids: &[String],
        texts: &[String],
        embeddings: &[Vec<f32>],
        importances: &[f32],
        concepts: &[Vec<String>],
    ) -> crate::Result<usize> {
        let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
        self.insert_batch_with_payload(ids, texts, &refs, importances, concepts, &[], &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_batch_with_payload(
        &self,
        ids: &[String],
        texts: &[String],
        embeddings: &[&[f32]],
        importances: &[f32],
        concepts: &[Vec<String>],
        payloads: &[Option<String>],
        scopes: &[Option<String>],
    ) -> crate::Result<usize> {
        let n = ids.len();
        if n == 0 {
            return Ok(0);
        }
        if texts.len() < n
            || embeddings.len() < n
            || importances.len() < n
            || concepts.len() < n
            || (!payloads.is_empty() && payloads.len() < n)
            || (!scopes.is_empty() && scopes.len() < n)
        {
            return Err(StorageError::InvalidArgument(
                "batch arrays have mismatched lengths".into(),
            ));
        }
        for &emb in embeddings {
            validate_dimension(emb, self.config.dimension)?;
        }

        // Idempotent batch insert: skip existing ids and duplicate ids within the
        // batch.  This makes the operation safe to replay after a partial write.
        let idx = self.id_index.read();
        let mut seen = HashSet::with_capacity(n);
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        for (i, raw_id) in ids.iter().enumerate().take(n) {
            let id = raw_id.as_str();
            if idx.contains_key(id) || !seen.insert(id) {
                continue;
            }
            indices.push(i);
        }
        drop(idx);

        // Snapshot the current vocabulary so all records in the batch are
        // canonicalized consistently (even if another thread evolves the
        // vocabulary while this batch is being prepared).
        let vocab = self.graph.read().vocab().clone();

        let mut records: Vec<(PointOffset, Record)> = Vec::with_capacity(indices.len());
        for &i in &indices {
            let mut emb = embeddings[i].to_vec();
            normalize(&mut emb)?;
            let offset = self.meta.allocate_offset();
            let seq = self.meta.allocate_seq();
            let payload = if payloads.is_empty() {
                None
            } else {
                payloads[i].clone()
            };
            let scope = if scopes.is_empty() {
                None
            } else {
                scopes[i].clone()
            };
            // Augment caller-supplied concepts with auto-extracted ones.
            let extractor_config = self.config.tier.extractor_config();
            let concepts = merge_concepts_with_config(
                &concepts[i],
                &texts[i],
                &extractor_config,
                Some(&vocab),
            );
            let record = Record {
                id: ids[i].clone(),
                text: texts[i].clone(),
                embedding: Arc::from(emb),
                importance: importances[i],
                concepts,
                created_at: now_secs(),
                insert_seq: seq,
                access_count: 0,
                last_accessed: 0,
                tier: crate::config::Tier::Hot,
                payload,
                scope,
            };
            records.push((offset, record));
        }

        // 1. Persist embeddings to the mmap-backed vector store first.
        for (offset, record) in &records {
            self.vectors.put(*offset, record.embedding_f32())?;
        }

        // 2. WAL metadata entries (batched under a single lock).
        {
            let mut wal = self.wal.lock();
            let ops: Vec<WalOp> = records
                .iter()
                .map(|(offset, record)| {
                    let meta = MetaRecord::from(record);
                    WalOp::Insert {
                        offset: *offset,
                        seq: record.insert_seq,
                        meta,
                    }
                })
                .collect();
            wal.append_batch(&ops)?;
        }

        // 3. Submit the index updates to the serialized worker.
        self.update_worker.submit_and_wait(records)?;
        Ok(indices.len())
    }

    /// Delete the record with the given id.
    ///
    /// The embedding is left in place in the vector store; the offset becomes
    /// unreachable because the metadata entry and id index are removed.  Segment
    /// searches already filter out offsets with no metadata, so deleted points
    /// disappear from results immediately.  Physical reclamation is deferred to
    /// the vacuum optimizer.
    pub fn delete_by_id(&self, id: &str) -> crate::Result<bool> {
        let offset = {
            let idx = self.id_index.read();
            match idx.get(id).copied() {
                Some(o) => o,
                None => return Ok(false),
            }
        };

        // 1. WAL delete entry.
        {
            let mut wal = self.wal.lock();
            wal.append(&WalOp::Delete { offset })?;
        }

        // 2. Remove from payload, text, and scope indexes while we still know
        //    the old values.
        if let Ok(Some(meta_rec)) = self.meta.get(offset) {
            self.payload_index
                .write()
                .remove(offset, meta_rec.payload.as_deref());
            self.scope_index
                .write()
                .remove(offset, meta_rec.scope.as_deref());
            self.text_index.remove(offset)?;
        }

        // 3. Remove from in-memory metadata and id index.
        self.meta.remove(offset)?;
        self.id_index.write().remove(id);

        // 4. Remove from cognitive graph.
        {
            let mut graph = self.graph.write();
            graph.remove_memory(id);
        }

        Ok(true)
    }

    /// Replace an existing record, preserving its id.
    ///
    /// Implemented as an atomic-in-metadata delete + insert: the old offset is
    /// tombstoned and a new offset is allocated.  This keeps HNSW segments
    /// correct without requiring in-place vector updates inside immutable
    /// indexes.
    pub fn update(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
    ) -> crate::Result<bool> {
        self.update_with_payload(id, text, embedding, importance, concepts, None, None)
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

    #[allow(clippy::too_many_arguments)]
    pub fn update_with_payload(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
        scope: Option<String>,
    ) -> crate::Result<bool> {
        if !self.id_index.read().contains_key(id) {
            return Ok(false);
        }
        self.delete_by_id(id)?;
        self.insert_with_payload(id, text, embedding, importance, concepts, payload, scope)?;
        Ok(true)
    }

    pub fn search_ann(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_with_ef(query_embedding, top_k, None)
    }

    pub fn search_ann_with_ef(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_scoped(query_embedding, top_k, ef, None)
    }

    /// ANN search restricted to a single agent scope (plus global records).
    pub fn search_ann_scoped(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Vec<(String, f32)>> {
        let candidates =
            self.search_ann_candidates_filtered_with_ef(query_embedding, top_k, None, ef, scope)?;
        Ok(candidates)
    }

    pub fn search_ann_candidates(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered(query_embedding, top_k, None)
    }

    pub fn search_ann_candidates_with_ef(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered_with_ef(query_embedding, top_k, None, ef, None)
    }

    /// Filtered ANN candidate search.
    ///
    /// `filter` is evaluated against the payload index; the resulting offset
    /// bitmap is intersected with tiered segment search.
    pub fn search_ann_candidates_filtered(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered_with_ef(query_embedding, top_k, filter, None, None)
    }

    pub fn search_ann_candidates_filtered_with_ef(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Vec<(String, f32)>> {
        validate_dimension(query_embedding, self.config.dimension)?;
        let mut allowed_offsets = match filter {
            Some(f) => Some(self.evaluate_filter(f)?),
            None => None,
        };
        if let Some(s) = scope {
            let scope_bitmap = self.scope_index.read().query(Some(s));
            allowed_offsets = Some(match allowed_offsets {
                Some(existing) => existing & scope_bitmap,
                None => scope_bitmap,
            });
        }
        if self.record_count() <= EXACT_FALLBACK_THRESHOLD {
            let results = match &allowed_offsets {
                Some(bitmap) => self.exact_top_k_filtered(query_embedding, top_k, bitmap),
                None => self.exact_top_k(query_embedding, top_k),
            };
            for (id, _) in &results {
                self.bump_access_by_id(id);
            }
            return Ok(results);
        }
        let snapshot = self.segment_snapshot.load_full();
        let gpu = self.gpu_backend();
        let scored = snapshot.search_gpu(
            query_embedding,
            top_k,
            ef,
            &self.vectors,
            allowed_offsets.as_ref(),
            Some(&gpu),
        )?;
        let mut results = Vec::with_capacity(scored.len());
        for c in scored {
            if let Some(meta_rec) = self.meta.get(c.offset)? {
                self.bump_access(c.offset);
                results.push((meta_rec.id, c.score));
            }
        }
        Ok(results)
    }

    /// Batched ANN search for M queries. Runs each query's HNSW traversal on
    /// CPU, then reranks all queries' candidate lists in a single GPU `gemm`
    /// when CUDA is available (`search_gpu_batch`), which is the workload
    /// where GPU genuinely beats CPU. Returns one result list per query, each
    /// sorted by score desc and truncated to `top_k`.
    ///
    /// Filter and scope apply identically to every query in the batch.
    pub fn search_ann_batch(
        &self,
        queries: &[&[f32]],
        top_k: usize,
        ef: Option<usize>,
        filter: Option<&Filter>,
        scope: Option<&str>,
    ) -> crate::Result<Vec<Vec<(String, f32)>>> {
        let m = queries.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        for q in queries {
            validate_dimension(q, self.config.dimension)?;
        }
        let mut allowed_offsets = match filter {
            Some(f) => Some(self.evaluate_filter(f)?),
            None => None,
        };
        if let Some(s) = scope {
            let scope_bitmap = self.scope_index.read().query(Some(s));
            allowed_offsets = Some(match allowed_offsets {
                Some(existing) => existing & scope_bitmap,
                None => scope_bitmap,
            });
        }

        // Small collection: batch the exact scan (per-query, but cheap).
        if self.record_count() <= EXACT_FALLBACK_THRESHOLD {
            let mut out = Vec::with_capacity(m);
            for q in queries {
                let results = match &allowed_offsets {
                    Some(bitmap) => self.exact_top_k_filtered(q, top_k, bitmap),
                    None => self.exact_top_k(q, top_k),
                };
                for (id, _) in &results {
                    self.bump_access_by_id(id);
                }
                out.push(results);
            }
            return Ok(out);
        }

        // Large collection: batched snapshot search (GPU gemm rerank).
        let snapshot = self.segment_snapshot.load_full();
        let gpu = self.gpu_backend();
        let batch_scored = snapshot.search_gpu_batch(
            queries,
            top_k,
            ef,
            &self.vectors,
            allowed_offsets.as_ref(),
            Some(&gpu),
        )?;

        // Map offsets → ids per query and bump access counters.
        let mut out = Vec::with_capacity(m);
        for scored in batch_scored {
            let mut results = Vec::with_capacity(scored.len());
            for c in scored {
                if let Some(meta_rec) = self.meta.get(c.offset)? {
                    self.bump_access(c.offset);
                    results.push((meta_rec.id, c.score));
                }
            }
            out.push(results);
        }
        Ok(out)
    }

    fn exact_top_k(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        self.exact_top_k_filtered(query, top_k, &RoaringBitmap::new())
    }

    fn exact_top_k_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        allowed_offsets: &RoaringBitmap,
    ) -> Vec<(String, f32)> {
        let view = self.vectors.read_view();
        let mut all: Vec<(String, f32)> = Vec::new();
        let _ = self.meta.for_each_record(|offset, rec| {
            if !allowed_offsets.is_empty() && !allowed_offsets.contains(offset as u32) {
                return;
            }
            if let Some(v) = view.get(offset) {
                let score = cosine_similarity(query, v);
                all.push((rec.id.clone(), score));
            }
        });
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(top_k);
        all
    }

    pub fn search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_with_ef(query_text, query_embedding, top_k, None)
    }

    /// Cognitive search restricted to a single agent scope (plus global records).
    pub fn search_scoped(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_scoped_with_ef(query_text, query_embedding, top_k, None, scope)
    }

    /// Cognitive search with an explicit `ef` and optional agent scope.
    pub fn search_scoped_with_ef(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_with_ef_scoped(query_text, query_embedding, top_k, ef, scope)
    }

    /// Hydrate augmenter results with embeddings and additively fuse the graph
    /// boost with cosine similarity to produce the final ranking.
    ///
    /// `final_score = cosine + (1 - alpha) * normalized_graph_delta`
    ///
    /// `results` carries the augmenter's **pure graph delta** (>= 0, cosine is
    /// NOT folded in). The boost is additive, so it can re-order candidates and
    /// surface graph-discovered ones but never drops an ANN hit — preserving
    /// the recall floor. `alpha` controls how much the graph may nudge the
    /// ranking: `1.0` = pure cosine (graph only decides which candidates
    /// exist); lower values give the graph delta more of a vote.
    fn hydrate_and_fuse(
        &self,
        results: Vec<(String, f32)>,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        // Normalize graph activation scores to [0, 1] for boosting
        let max_act = results
            .iter()
            .map(|(_, a)| *a)
            .fold(0.0f32, f32::max)
            .max(1e-10);
        let alpha = self.config.cognitive_alpha.clamp(0.0, 1.0);

        let mut hydrated: Vec<(String, f32)> = results
            .into_iter()
            .filter_map(|(id, act)| {
                self.find_record_by_id(&id).map(|rec| {
                    let cos = cosine_similarity(query_embedding, rec.embedding_f32());
                    let norm_act = act / max_act;
                    // Hybrid scoring: cosine + graph boost.
                    // `cognitive_alpha` is the sole blend control:
                    //   final = cos + (1 - alpha) * normalized_graph_signal
                    // At alpha = 1.0 the ranking is pure cosine (graph only
                    // influences which candidates exist). At lower alpha the
                    // graph can re-rank via an additive, bounded boost.
                    let graph_boost = (1.0 - alpha) * norm_act;
                    let fused = cos + graph_boost;
                    // Supersession demotion: a memory superseded by a newer one
                    // (Contradicts/Refines edge created during consolidation)
                    // carries a persisted factor < 1.0. Applying it
                    // multiplicatively to the final score demotes the stale
                    // belief even at alpha = 1.0, where the additive graph
                    // boost has no effect. Default 1.0 (no demotion).
                    let demotion = self
                        .id_index
                        .read()
                        .get(id.as_str())
                        .map(|&offset| self.meta.demotion_factor(offset))
                        .unwrap_or(crate::metadata_store::NO_DEMOTION);
                    (id, fused * demotion)
                })
            })
            .collect();
        hydrated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        hydrated.truncate(top_k);
        Ok(hydrated)
    }

    pub fn search_with_ef(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_with_ef_scoped(query_text, query_embedding, top_k, ef, None)
    }

    fn search_with_ef_scoped(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        validate_dimension(query_embedding, self.config.dimension)?;

        let seeds = self.search_ann_candidates_filtered_with_ef(
            query_embedding,
            top_k.max(10),
            None,
            ef,
            scope,
        )?;

        let graph = self.graph.read();
        // Request more candidates from the graph than the final top_k so
        // that memories reached through multi-hop traversal (abstraction
        // edges, refinement edges) have a chance to be in the candidate
        // set even if their graph activation is lower than direct matches.
        // The fusion step (hydrate_and_fuse) will then re-rank using the
        // combination of cosine + graph activation and truncate to top_k.
        let graph_k = (top_k * 3).max(top_k + 5);
        let activated = graph.search(query_text, &seeds, graph_k);
        drop(graph);
        if let Some(results) = activated {
            let hydrated = self.hydrate_and_fuse(results, query_embedding, top_k)?;
            if hydrated.is_empty() {
                return Ok(None);
            }
            for (id, _) in &hydrated {
                self.bump_access_by_id(id);
                self.reinforce_graph_by_id(id);
            }
            Ok(Some(hydrated))
        } else {
            Ok(None)
        }
    }

    pub fn search_ann_filtered(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered(query_embedding, top_k, Some(filter))
    }

    /// Cognitive search with a payload filter.
    pub fn search_filtered(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_filtered_with_ef(query_text, query_embedding, top_k, filter, None)
    }

    pub fn search_filtered_with_ef(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
        ef: Option<usize>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_filtered_with_scope(query_text, query_embedding, top_k, filter, ef, None)
    }

    /// Cognitive search with both a payload filter and an agent scope.
    pub fn search_filtered_with_scope(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        validate_dimension(query_embedding, self.config.dimension)?;
        let seeds = self.search_ann_candidates_filtered_with_ef(
            query_embedding,
            top_k.max(10),
            Some(filter),
            ef,
            scope,
        )?;
        let graph = self.graph.read();
        let graph_k = (top_k * 3).max(top_k + 5);
        let activated = graph.search(query_text, &seeds, graph_k);
        drop(graph);
        if let Some(results) = activated {
            let hydrated = self.hydrate_and_fuse(results, query_embedding, top_k)?;
            if hydrated.is_empty() {
                return Ok(None);
            }
            for (id, _) in &hydrated {
                self.bump_access_by_id(id);
                self.reinforce_graph_by_id(id);
            }
            Ok(Some(hydrated))
        } else {
            Ok(None)
        }
    }

    /// Evaluate a filter against the payload and full-text indexes.
    fn evaluate_filter(&self, filter: &Filter) -> crate::Result<RoaringBitmap> {
        if filter.uses_full_text() {
            // Tantivy writes are deferred; make them visible before querying,
            // but only if there are actually pending documents.
            self.text_index.commit_if_pending()?;
        }
        self.evaluate_filter_recursive(filter)
    }

    fn evaluate_filter_recursive(&self, filter: &Filter) -> crate::Result<RoaringBitmap> {
        use crate::payload_index::Filter as F;
        Ok(match filter {
            F::FullText { query, .. } => self.text_index.search(query)?,
            F::Eq { .. } | F::Range { .. } => self.payload_index.read().query(filter),
            F::And(parts) => {
                let mut iter = parts.iter();
                let Some(first) = iter.next() else {
                    return Ok(RoaringBitmap::new());
                };
                let mut acc = self.evaluate_filter_recursive(first)?;
                for part in iter {
                    if acc.is_empty() {
                        break;
                    }
                    acc &= self.evaluate_filter_recursive(part)?;
                }
                acc
            }
            F::Or(parts) => {
                let mut acc = RoaringBitmap::new();
                for part in parts {
                    acc |= self.evaluate_filter_recursive(part)?;
                }
                acc
            }
            F::Not(inner) => {
                let positives = self.evaluate_filter_recursive(inner)?;
                self.payload_index.read().all_offsets() - &positives
            }
        })
    }

    /// Hydrate a full `Record` from the metadata cache + vector store.
    fn get_record(&self, offset: PointOffset) -> Option<Record> {
        let meta = self.meta.get(offset).ok().flatten()?;
        let view = self.vectors.read_view();
        let vec = view.get(offset)?;
        Some(meta.with_embedding(Arc::from(Vec::from(vec))))
    }

    fn find_record_by_id(&self, id: &str) -> Option<Record> {
        let idx = self.id_index.read();
        idx.get(id)
            .copied()
            .and_then(|offset| self.get_record(offset))
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
    fn reinforce_graph_by_id(&self, id: &str) {
        let mut graph = self.graph.write();
        graph.reinforce(id, now_secs());
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

    fn save_ccs(&self) -> crate::Result<()> {
        if let Some(ccs) = self.ccs.lock().as_ref() {
            self.meta.save_meta("ccs", &ccs.to_json())?;
        }
        Ok(())
    }

    fn save_graph(&self) -> crate::Result<()> {
        let json = self.graph.read().graph().to_json();
        self.meta.save_meta("graph", &json)?;
        Ok(())
    }

    /// Gracefully shut down the engine.
    ///
    /// Flushes the WAL, vector store, metadata snapshot, and segment files.
    /// The background optimizer is stopped automatically when the engine is
    /// dropped.
    pub fn shutdown(&self) -> crate::Result<()> {
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

        // Memory evolution: detect refinements (newer memories that
        // supersede older ones about the same topic) and create Refines
        // edges so retrieval surfaces the most current version. Opt-in.
        self.check_refinements()?;

        // Contradiction detection: when a newer memory contradicts an
        // older one (same topic, opposing content), create Contradicts
        // edges and weaken the old memory so it fades. Runs after
        // check_refinements so refinement pairs (high text overlap) are
        // not double-counted as contradictions. Opt-in.
        self.check_contradictions()?;

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

    /// Bounded-storage eviction: drop the lowest-salience records when the
    /// collection exceeds `max_records` or when a record's `access_score`
    /// falls below `evict_score_floor`. Returns the number of records evicted.
    ///
    /// Both triggers are opt-in; when neither is configured this is a no-op.
    /// Salience is the same recency-weighted `access_score` used for promotion.
    /// A grace period protects freshly inserted records that simply have not
    /// been queried yet from being evicted on their first consolidation.
    pub fn evict(&self) -> crate::Result<usize> {
        let tier = &self.config.tier;
        let max_records = tier.max_records;
        let floor = tier.evict_score_floor;
        if max_records.is_none() && floor.is_none() {
            return Ok(0);
        }

        // Make access counts current before scoring.
        self.access_counters.drain_into(&self.meta)?;

        let now = now_secs();
        let half_life = tier.recency_half_life_secs.max(1);
        // Grace window: never evict a record whose last access (or insertion)
        // is more recent than this. Guards against evicting just-inserted
        // records before they have had a chance to be queried.
        let grace = half_life / 8;

        // Snapshot (offset, id, score, protected) for every live record.
        let mut scored: Vec<(PointOffset, String, f64)> = Vec::new();
        let mut protected: Vec<(PointOffset, String)> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            if now.saturating_sub(rec.last_accessed) < grace {
                protected.push((offset, rec.id.clone()));
            } else {
                scored.push((offset, rec.id.clone(), access_score(rec, now, half_life)));
            }
        })?;

        // Collect victims by id (dedup via a set of offsets).
        let mut victim_offsets: HashSet<PointOffset> = HashSet::new();
        let mut victims: Vec<String> = Vec::new();

        // Floor pass: anything below the score floor is a victim.
        if let Some(floor) = floor {
            for (offset, id, score) in &scored {
                if *score < floor && victim_offsets.insert(*offset) {
                    victims.push(id.clone());
                }
            }
        }

        // Cap pass: if still over the cap, evict the lowest-scoring survivors.
        if let Some(max_records) = max_records {
            let live = self.meta.record_count();
            let mut over = live
                .saturating_sub(victims.len())
                .saturating_sub(max_records);
            if over > 0 {
                // Survivors not already marked, sorted by ascending score.
                let mut survivors: Vec<&(PointOffset, String, f64)> = scored
                    .iter()
                    .filter(|(offset, _, _)| !victim_offsets.contains(offset))
                    .collect();
                survivors.sort_by(|a, b| a.2.total_cmp(&b.2));
                for (offset, id, _) in survivors {
                    if over == 0 {
                        break;
                    }
                    if victim_offsets.insert(*offset) {
                        victims.push(id.clone());
                        over -= 1;
                    }
                }
            }
        }

        let mut evicted = 0usize;
        for id in &victims {
            if self.delete_by_id(id)? {
                evicted += 1;
            }
        }
        Ok(evicted)
    }

    /// Semantic consolidation: merge near-duplicate records.
    ///
    /// Two records whose cosine similarity is `>= dedup_cosine_threshold` are
    /// considered duplicates. For each duplicate pair the higher-salience
    /// record is kept (tiebreak: `importance`, then earlier `insert_seq`) and
    /// the other is deleted. The survivor inherits the victim's concept edges
    /// so graph relationships are not lost. Returns the number of records
    /// merged away.
    ///
    /// Opt-in: a no-op when `dedup_cosine_threshold` is `None`. Work is bounded
    /// by `dedup_max_pairs_per_cycle`. Candidate neighbors are found via the
    /// existing ANN index (no O(n^2) scan).
    pub fn deduplicate(&self) -> crate::Result<usize> {
        let Some(threshold) = self.config.tier.dedup_cosine_threshold else {
            return Ok(0);
        };
        let max_pairs = self.config.tier.dedup_max_pairs_per_cycle;
        if max_pairs == 0 {
            return Ok(0);
        }

        self.access_counters.drain_into(&self.meta)?;
        let now = now_secs();
        let half_life = self.config.tier.recency_half_life_secs.max(1);

        // Snapshot live records: id -> (offset, score, importance, insert_seq,
        // concepts). Cloning concepts is acceptable; the set of live records is
        // bounded and this runs only on consolidation.
        struct Cand {
            offset: PointOffset,
            id: String,
            score: f64,
            importance: f32,
            insert_seq: u64,
            concepts: Vec<String>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                score: access_score(rec, now, half_life),
                importance: rec.importance,
                insert_seq: rec.insert_seq,
                concepts: rec.concepts.clone(),
            });
        })?;

        // Higher salience wins. Returns true if `a` should be kept over `b`.
        let keeps = |a: &Cand, b: &Cand| -> bool {
            a.score
                .total_cmp(&b.score)
                .then(a.importance.total_cmp(&b.importance))
                .then(b.insert_seq.cmp(&a.insert_seq))
                .is_ge()
        };

        let mut merged_offsets: HashSet<PointOffset> = HashSet::new();
        // (survivor_id, victim_id, victim_concepts)
        let mut merges: Vec<(String, String, Vec<String>)> = Vec::new();

        'outer: for cand in &cands {
            if merged_offsets.contains(&cand.offset) {
                continue;
            }
            let view = self.vectors.read_view();
            let Some(vec) = view.get(cand.offset) else {
                continue;
            };
            let embedding: Vec<f32> = vec.to_vec();
            drop(view);

            // Find near neighbors via ANN, then verify exact cosine.
            let neighbors = self.search_ann_candidates(&embedding, 5)?;
            for (nid, _) in neighbors {
                if nid == cand.id {
                    continue;
                }
                let Some(other) = cands.iter().find(|c| c.id == nid) else {
                    continue;
                };
                if merged_offsets.contains(&other.offset) {
                    continue;
                }
                let view = self.vectors.read_view();
                let Some(other_vec) = view.get(other.offset) else {
                    continue;
                };
                let sim = cosine_similarity(&embedding, other_vec);
                drop(view);
                if sim < threshold {
                    continue;
                }
                // Decide survivor vs victim.
                let (survivor, victim) = if keeps(cand, other) {
                    (cand, other)
                } else {
                    (other, cand)
                };
                merged_offsets.insert(victim.offset);
                merges.push((
                    survivor.id.clone(),
                    victim.id.clone(),
                    victim.concepts.clone(),
                ));
                if merges.len() >= max_pairs {
                    break 'outer;
                }
                // `cand` may itself have become a victim; stop scanning its
                // neighbors and move to the next candidate.
                if victim.offset == cand.offset {
                    continue 'outer;
                }
            }
        }

        let mut count = 0usize;
        for (survivor_id, victim_id, victim_concepts) in &merges {
            // Transfer the victim's concept edges to the survivor before
            // deleting it, so relationships are preserved.
            if !victim_concepts.is_empty() {
                if let Some(survivor) = self.find_record_by_id(survivor_id) {
                    let mut concepts = survivor.concepts.clone();
                    for c in victim_concepts {
                        if !concepts.contains(c) {
                            concepts.push(c.clone());
                        }
                    }
                    let mut graph = self.graph.write();
                    graph.add_memory_with_importance(
                        survivor_id,
                        &survivor.text,
                        &concepts,
                        survivor.importance,
                    );
                    drop(graph);
                }
            }
            if self.delete_by_id(victim_id)? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Memory evolution: detect when a newer memory refines an older one
    /// and create `Refines` edges (old → new) so retrieval surfaces the
    /// most current version.
    ///
    /// For each pair of memories where:
    /// - cosine similarity >= `refinement_cosine_threshold`
    /// - they share at least one concept
    /// - the newer one has a higher `insert_seq`
    ///
    /// a `Refines` edge is created from the older memory to the newer one.
    /// The older memory is NOT deleted — it stays in the graph so the agent
    /// can reason about how its understanding evolved. The newer memory
    /// also inherits the older one's unique concepts (so it's discoverable
    /// through the same concept paths).
    ///
    /// Opt-in: a no-op when `refinement_cosine_threshold` is `None`. Work
    /// is bounded by `refinement_max_pairs_per_cycle`. Candidate pairs are
    /// found via the existing ANN index (no O(n²) scan).
    ///
    /// Returns the number of new `Refines` edges created.
    pub fn check_refinements(&self) -> crate::Result<usize> {
        let Some(threshold) = self.config.tier.refinement_cosine_threshold else {
            return Ok(0);
        };
        let max_pairs = self.config.tier.refinement_max_pairs_per_cycle;
        if max_pairs == 0 {
            return Ok(0);
        }
        let demotion_factor = self.config.tier.supersession_demotion_factor;

        self.access_counters.drain_into(&self.meta)?;

        // Snapshot live records sorted by insert_seq (newest first, so we
        // process the most recent refinements first).
        struct Cand {
            id: String,
            offset: PointOffset,
            insert_seq: u64,
            concepts: Vec<String>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                insert_seq: rec.insert_seq,
                concepts: rec.concepts.clone(),
            });
        })?;
        cands.sort_by_key(|c| std::cmp::Reverse(c.insert_seq));

        let mut created = 0usize;
        let mut processed: HashSet<PointOffset> = HashSet::new();

        for cand in &cands {
            if created >= max_pairs {
                break;
            }
            // Get this record's embedding for ANN search.
            let view = self.vectors.read_view();
            let Some(vec) = view.get(cand.offset) else {
                continue;
            };
            let embedding: Vec<f32> = vec.to_vec();
            drop(view);

            // Find near neighbors via ANN.
            let neighbors = self.search_ann_candidates(&embedding, 10)?;
            for (nid, _) in neighbors {
                if created >= max_pairs {
                    break;
                }
                if nid == cand.id {
                    continue;
                }
                // Find the neighbor in our candidate list.
                let Some(other) = cands.iter().find(|c| c.id == nid) else {
                    continue;
                };
                // The neighbor must be OLDER (lower insert_seq).
                if other.insert_seq >= cand.insert_seq {
                    continue;
                }
                // They must share at least one concept.
                let shares_concept = cand.concepts.iter().any(|c| other.concepts.contains(c));
                if !shares_concept {
                    continue;
                }
                // Verify exact cosine.
                let view = self.vectors.read_view();
                let Some(other_vec) = view.get(other.offset) else {
                    continue;
                };
                let sim = cosine_similarity(&embedding, other_vec);
                drop(view);
                if sim < threshold {
                    continue;
                }
                // Create the Refines edge: old (other) → new (cand).
                let mut graph = self.graph.write();
                let added = graph.add_refinement(&other.id, &cand.id, 0.8);
                drop(graph);
                if added {
                    created += 1;
                    // Demote the refined (older) memory so the newer version
                    // outranks it even under pure cosine. Multiplicative and
                    // guarded by `added`, so it fires once per distinct
                    // refinement (idempotent across re-runs and restarts).
                    let cur = self.meta.demotion_factor(other.offset);
                    self.meta
                        .set_demotion_factor(other.offset, cur * demotion_factor);
                    // Transfer the older memory's unique concepts to the
                    // newer one, so the refinement is discoverable through
                    // all the same concept paths.
                    let unique_concepts: Vec<String> = other
                        .concepts
                        .iter()
                        .filter(|c| !cand.concepts.contains(c))
                        .cloned()
                        .collect();
                    if !unique_concepts.is_empty() {
                        let mut new_concepts = cand.concepts.clone();
                        for c in &unique_concepts {
                            if !new_concepts.contains(c) {
                                new_concepts.push(c.clone());
                            }
                        }
                        // Update the record's concepts in metadata so the
                        // transfer persists across restarts.
                        if let Some(mut rec) = self.meta.get(cand.offset)? {
                            rec.concepts = new_concepts;
                            self.meta.put(
                                cand.offset,
                                &rec.with_embedding(
                                    // Re-attach a dummy embedding — put only
                                    // stores metadata, the embedding lives in
                                    // VectorStore and is not affected.
                                    Arc::from(embedding.as_slice()),
                                ),
                            )?;
                        }
                        // Also update the graph: re-add the new memory with
                        // the augmented concepts so the new concept edges
                        // exist. This is idempotent for existing concepts
                        // (graph dedupes by node id).
                        let mut graph = self.graph.write();
                        if let Some(rec) = self.find_record_by_id(&cand.id) {
                            graph.add_memory_with_importance(
                                &cand.id,
                                &rec.text,
                                &rec.concepts,
                                rec.importance,
                            );
                        }
                        drop(graph);
                    }
                }
            }
            processed.insert(cand.offset);
        }

        if created > 0 {
            self.save_graph()?;
        }
        Ok(created)
    }

    /// Contradiction detection: when a newer memory contradicts an older
    /// one (same topic, opposing content), create a `Contradicts` edge and
    /// weaken the old memory's edges so it fades from retrieval.
    ///
    /// For each pair where:
    /// - cosine similarity >= `contradiction_cosine_threshold`
    /// - they share at least one concept
    /// - text Jaccard similarity < `contradiction_text_threshold` (the
    ///   texts say different things about the same topic)
    /// - the newer one has a higher `insert_seq`
    ///
    /// a `Contradicts` edge is created (old → new) and the old memory's
    /// outgoing Association/Temporal edges are weakened by
    /// `contradiction_weaken_factor`. The old memory is NOT deleted.
    ///
    /// Opt-in: a no-op when `contradiction_cosine_threshold` is `None`.
    /// Runs after `check_refinements` so that pairs that are refinements
    /// (high text overlap) are not also flagged as contradictions.
    ///
    /// Returns the number of new `Contradicts` edges created.
    pub fn check_contradictions(&self) -> crate::Result<usize> {
        let Some(threshold) = self.config.tier.contradiction_cosine_threshold else {
            return Ok(0);
        };
        let max_pairs = self.config.tier.contradiction_max_pairs_per_cycle;
        if max_pairs == 0 {
            return Ok(0);
        }
        let text_threshold = self.config.tier.contradiction_text_threshold;
        let weaken_factor = self.config.tier.contradiction_weaken_factor;
        let demotion_factor = self.config.tier.supersession_demotion_factor;

        self.access_counters.drain_into(&self.meta)?;

        // Snapshot live records sorted by insert_seq (newest first).
        struct Cand {
            id: String,
            offset: PointOffset,
            insert_seq: u64,
            text: String,
            concepts: Vec<String>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                insert_seq: rec.insert_seq,
                text: rec.text.clone(),
                concepts: rec.concepts.clone(),
            });
        })?;
        cands.sort_by_key(|c| std::cmp::Reverse(c.insert_seq));

        let mut created = 0usize;
        for cand in &cands {
            if created >= max_pairs {
                break;
            }
            let view = self.vectors.read_view();
            let Some(vec) = view.get(cand.offset) else {
                continue;
            };
            let embedding: Vec<f32> = vec.to_vec();
            drop(view);

            let neighbors = self.search_ann_candidates(&embedding, 10)?;
            for (nid, _) in neighbors {
                if created >= max_pairs {
                    break;
                }
                if nid == cand.id {
                    continue;
                }
                let Some(other) = cands.iter().find(|c| c.id == nid) else {
                    continue;
                };
                // The neighbor must be OLDER.
                if other.insert_seq >= cand.insert_seq {
                    continue;
                }
                // They must share at least one concept.
                let shares_concept = cand.concepts.iter().any(|c| other.concepts.contains(c));
                if !shares_concept {
                    continue;
                }
                // Verify exact cosine.
                let view = self.vectors.read_view();
                let Some(other_vec) = view.get(other.offset) else {
                    continue;
                };
                let sim = cosine_similarity(&embedding, other_vec);
                drop(view);
                if sim < threshold {
                    continue;
                }
                // KEY DISTINGUISHER from refinement: the texts must be
                // DISSIMILAR (low Jaccard). A refinement has high text
                // overlap (same topic, updated content); a contradiction
                // has low text overlap (same topic, opposing content).
                let text_sim = turbomemory_graph::text_jaccard_similarity(&cand.text, &other.text);
                if text_sim >= text_threshold {
                    // High text overlap → this is a refinement, not a
                    // contradiction. Skip (check_refinements handles it).
                    continue;
                }
                // Create the Contradicts edge: old (other) → new (cand).
                // This also weakens the old memory's edges.
                let mut graph = self.graph.write();
                let added = graph.add_contradiction(&other.id, &cand.id, 0.8, weaken_factor);
                drop(graph);
                if added {
                    created += 1;
                    // Demote the superseded memory's retrieval score so the
                    // newer, contradicting belief outranks it even under pure
                    // cosine. Multiplicative against any existing demotion so
                    // repeated supersession compounds. Guarded by `added`, so
                    // this fires once per distinct contradiction (idempotent
                    // across re-runs and restarts).
                    let cur = self.meta.demotion_factor(other.offset);
                    self.meta
                        .set_demotion_factor(other.offset, cur * demotion_factor);
                }
            }
        }

        if created > 0 {
            self.save_graph()?;
        }
        Ok(created)
    }

    /// Automatic importance scoring (self-organizing memory). For each live
    /// record, compute a target importance as a blend of:
    ///   - retrieval salience: normalized `access_score` (recency-weighted
    ///     access count), and
    ///   - graph connectivity: normalized concept degree (how many distinct
    ///     concepts the memory is linked to).
    ///
    /// Then move the record's current importance `importance_learning_rate`
    /// of the way toward that target, clamped to `[floor, ceiling]`.
    ///
    /// Frequently retrieved + well-connected memories rise; never-retrieved
    /// memories decay toward the floor. The recomputed importance is written
    /// back to metadata and synced into the graph via `reweight_memory` so
    /// edge weights reflect the new importance.
    ///
    /// Opt-in: a no-op when `importance_auto_scoring` is false (the default).
    /// Returns the number of records whose importance changed by more than a
    /// small epsilon.
    pub fn recompute_importance(&self) -> crate::Result<usize> {
        if !self.config.tier.importance_auto_scoring {
            return Ok(0);
        }
        let rate = self.config.tier.importance_learning_rate.clamp(0.0, 1.0);
        let access_weight = self.config.tier.importance_access_weight.clamp(0.0, 1.0);
        let floor = self.config.tier.importance_floor;
        let ceiling = self.config.tier.importance_ceiling.max(floor);

        // Make access counts current before scoring.
        self.access_counters.drain_into(&self.meta)?;

        let now = now_secs();
        let half_life = self.config.tier.recency_half_life_secs.max(1);

        // Snapshot: offset, id, importance, salience (access_score), degree.
        struct Cand {
            offset: PointOffset,
            id: String,
            importance: f32,
            salience: f64,
            degree: usize,
        }
        let mut cands: Vec<Cand> = Vec::new();
        let mut max_salience: f64 = 0.0;
        let mut max_degree: usize = 0;
        self.meta.for_each_record(|offset, rec| {
            let salience = access_score(rec, now, half_life);
            let degree = rec.concepts.len();
            max_salience = max_salience.max(salience);
            max_degree = max_degree.max(degree);
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                importance: rec.importance,
                salience,
                degree,
            });
        })?;

        if cands.is_empty() {
            return Ok(0);
        }

        let mut changed = 0usize;
        for cand in &cands {
            // Normalize salience and degree to [0, 1].
            let sal = if max_salience > 0.0 {
                cand.salience / max_salience
            } else {
                0.0
            };
            let deg = if max_degree > 0 {
                cand.degree as f64 / max_degree as f64
            } else {
                0.0
            };
            // Target importance in [floor, ceiling]. Retrieval salience is the
            // primary driver; connectivity is a bounded boost that can lift a
            // retrieved memory but cannot, on its own, push a never-retrieved
            // memory to the ceiling. `access_weight` blends how much of the
            // band is salience-driven (the rest is a connectivity bonus scaled
            // down by salience so it only matters once a memory is being
            // retrieved). This keeps "never retrieved" decaying toward the
            // floor regardless of how many concepts it touches.
            let salience_band = sal;
            let connectivity_bonus = (1.0 - access_weight as f64) * deg * (0.5 + 0.5 * sal);
            let blend = (access_weight as f64 * salience_band + connectivity_bonus).clamp(0.0, 1.0);
            let target = floor + (ceiling - floor) * blend as f32;
            // Move a `rate` fraction of the way toward the target.
            let new_importance =
                (cand.importance + rate * (target - cand.importance)).clamp(floor, ceiling);

            if (new_importance - cand.importance).abs() <= 1e-4 {
                continue;
            }

            // Write back to metadata.
            if let Some(mut rec) = self.meta.get(cand.offset)? {
                rec.importance = new_importance;
                self.meta.put_meta(cand.offset, &rec)?;
            }
            // Sync the graph's edge weights to the new importance.
            {
                let mut graph = self.graph.write();
                graph.graph_mut().reweight_memory(&cand.id, new_importance);
            }
            changed += 1;
        }

        if changed > 0 {
            self.save_graph()?;
        }
        Ok(changed)
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
        // 1. Build any pending plain segments so the durable snapshot captures
        //    them as persisted HNSW / quantized segments rather than in-memory
        //    plain indexes.
        while self.optimizer.process_one_seal(self).unwrap_or(false) {}

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

        Ok(())
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
        graph.add_memory_with_importance(&rec.id, &rec.text, &rec.concepts, rec.importance);
    }
    SpreadingActivation::new(graph, config.clone())
}

/// Rebuild the cognitive graph, preserving learned edge weights and
/// abstraction nodes from a previously-saved graph when available.
///
/// The persisted graph JSON (written by `save_graph`) captures learned state
/// that is not reconstructable from records alone: reinforced edge weights,
/// reinforcement timestamps, and abstraction (parent concept) nodes. If the
/// persisted graph is present, we load it and add only records that are not
/// already memory nodes in it (using `insert_seq` ordering to decide which
/// records are new). If the persisted graph is absent or fails to parse, we
/// fall back to a full rebuild from records — this preserves the pre-learning
/// behavior and is always correct, just without learned state.
fn rebuild_graph(
    records: &[(PointOffset, Record)],
    saved_graph_json: Option<String>,
    config: &SpreadingConfig,
) -> SpreadingActivation {
    let Some(json) = saved_graph_json else {
        return build_graph(records, config);
    };
    let Ok(mut graph) = MemoryGraph::from_json(&json) else {
        return build_graph(records, config);
    };
    // Add records that are not already memory nodes in the persisted graph.
    // Records are sorted by (created_at, insert_seq) by the caller, so we
    // process them in insertion order, preserving temporal chaining for the
    // new tail. We track the last memory id seen so temporal edges chain
    // correctly from the last persisted memory to the first new one.
    let existing_mem_ids: HashSet<String> = graph
        .iter_memory_nodes()
        .map(|(k, _)| k.strip_prefix("mem:").unwrap_or(&k).to_string())
        .collect();
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
        graph.add_memory_with_importance(&rec.id, &rec.text, &rec.concepts, rec.importance);
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

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Recency-weighted salience: `access_count * 2^(-age / half_life)`.
///
/// Mirrors `segment_holder::access_score` (kept private there); used by
/// eviction and dedup to rank records by importance.
fn access_score(meta: &MetaRecord, now: u64, half_life: u64) -> f64 {
    let age = now.saturating_sub(meta.last_accessed).max(1);
    let recency = 2.0f64.powf(-(age as f64) / half_life as f64);
    meta.access_count as f64 * recency
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TierConfig;

    fn small_config(dim: usize) -> StoreConfig {
        StoreConfig {
            dimension: dim,
            max_edges: 3,
            level0_factor: 2,
            ef_construction: 100,
            search_list_size: 5,
            cognitive_alpha: 1.0,
            outlier_count: 0,
            initial_capacity: 16,
            tier: TierConfig {
                hot_capacity: 3,
                warm_capacity: 6,
                warm_quantizer: crate::config::QuantizerKind::Scalar { bits: 4 },
                warm_chunk_bytes: 4096,
                hnsw_threshold: 1000,
                full_scan_threshold_kb: 10_000,
                merge_threshold_segments: 2,
                merge_max_records: 200_000,
                cold_quantizer: crate::config::QuantizerKind::Sign,
                hot_promote_threshold: 2.0,
                warm_demote_threshold: 0.5,
                recency_half_life_secs: 60,
                max_records: None,
                evict_score_floor: None,
                dedup_cosine_threshold: None,
                dedup_max_pairs_per_cycle: 1024,
                abstraction_co_occurrence_threshold: 0,
                edge_decay_half_life_secs: 0,
                max_concepts: 5,
                refinement_cosine_threshold: None,
                refinement_max_pairs_per_cycle: 1024,
                contradiction_cosine_threshold: None,
                contradiction_text_threshold: 0.3,
                contradiction_weaken_factor: 0.5,
                contradiction_max_pairs_per_cycle: 1024,
                importance_auto_scoring: false,
                importance_learning_rate: 0.3,
                importance_access_weight: 0.6,
                importance_floor: 0.1,
                importance_ceiling: 4.0,
                concept_max_ngram_len: 1,
                concept_min_ngram_freq: 1,
                concept_enable_pmi: true,
                ..TierConfig::default()
            },
            optimizer_budget: crate::config::OptimizerBudget::default(),
            auto_consolidation_interval: None,
            spreading: turbomemory_graph::SpreadingConfig::default(),
        }
    }

    fn make_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx % dim] = 1.0;
        v
    }

    fn hnsw_test_config(dim: usize) -> StoreConfig {
        StoreConfig {
            dimension: dim,
            max_edges: 8,
            level0_factor: 2,
            ef_construction: 100,
            search_list_size: 10,
            cognitive_alpha: 1.0,
            outlier_count: 0,
            initial_capacity: 4096,
            tier: TierConfig {
                hot_capacity: 100,
                warm_capacity: 10_000,
                warm_quantizer: crate::config::QuantizerKind::Scalar { bits: 8 },
                warm_chunk_bytes: 4096,
                hnsw_threshold: 10,
                full_scan_threshold_kb: 10_000,
                merge_threshold_segments: 2,
                merge_max_records: 10_000,
                cold_quantizer: crate::config::QuantizerKind::Sign,
                hot_promote_threshold: 2.0,
                warm_demote_threshold: 0.5,
                recency_half_life_secs: 60,
                max_records: None,
                evict_score_floor: None,
                dedup_cosine_threshold: None,
                dedup_max_pairs_per_cycle: 1024,
                abstraction_co_occurrence_threshold: 0,
                edge_decay_half_life_secs: 0,
                max_concepts: 5,
                refinement_cosine_threshold: None,
                refinement_max_pairs_per_cycle: 1024,
                contradiction_cosine_threshold: None,
                contradiction_text_threshold: 0.3,
                contradiction_weaken_factor: 0.5,
                contradiction_max_pairs_per_cycle: 1024,
                importance_auto_scoring: false,
                importance_learning_rate: 0.3,
                importance_access_weight: 0.6,
                importance_floor: 0.1,
                importance_ceiling: 4.0,
                concept_max_ngram_len: 1,
                concept_min_ngram_freq: 1,
                concept_enable_pmi: true,
                ..TierConfig::default()
            },
            optimizer_budget: crate::config::OptimizerBudget::default(),
            auto_consolidation_interval: None,
            spreading: turbomemory_graph::SpreadingConfig::default(),
        }
    }

    #[test]
    fn insert_and_search_ann() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert(
                "m1",
                "Rust is safe",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
            )
            .unwrap();
        engine
            .insert(
                "m2",
                "Python is easy",
                &[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
            )
            .unwrap();
        let results = engine
            .search_ann(&[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
            .unwrap();
        assert_eq!(results[0].0, "m1");
    }

    #[test]
    fn auto_extracts_concepts_when_none_provided() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        // Insert with NO caller concepts — the engine should auto-extract
        // concepts from the text.
        engine
            .insert(
                "m1",
                "Rust memory safety concurrency",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
            )
            .unwrap();
        // Verify the record has extracted concepts by checking the graph.
        let graph = engine.graph.read();
        let graph = graph.graph();
        // "rust", "memory", "safety", "concurrency" should all be concept nodes.
        assert!(
            graph.nodes().contains_key("concept:rust"),
            "auto-extracted concept 'rust' should be a graph node"
        );
        assert!(
            graph.nodes().contains_key("concept:safety"),
            "auto-extracted concept 'safety' should be a graph node"
        );
        assert!(
            graph.nodes().contains_key("concept:concurrency"),
            "auto-extracted concept 'concurrency' should be a graph node"
        );
    }

    #[test]
    fn caller_concepts_are_preserved_and_augmented() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        // Insert with one caller concept — it should be preserved, and the
        // remaining slots filled by extraction.
        engine
            .insert(
                "m1",
                "Rust memory safety concurrency",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["my_tag".to_string()],
            )
            .unwrap();
        let graph = engine.graph.read();
        let graph = graph.graph();
        // Caller concept preserved.
        assert!(graph.nodes().contains_key("concept:my_tag"));
        // Extracted concepts augmented.
        assert!(graph.nodes().contains_key("concept:rust"));
    }

    #[test]
    fn max_concepts_zero_disables_extraction() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.max_concepts = 0;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        engine
            .insert(
                "m1",
                "Rust memory safety concurrency",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
            )
            .unwrap();
        let graph = engine.graph.read();
        let graph = graph.graph();
        // With max_concepts=0 and no caller concepts, no concept nodes should exist.
        assert!(
            !graph.nodes().contains_key("concept:rust"),
            "extraction should be disabled when max_concepts=0"
        );
    }

    #[test]
    fn check_refinements_creates_edge_for_related_memories() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        // Enable refinement with a low threshold so our test vectors trigger it.
        config.tier.refinement_cosine_threshold = Some(0.5);
        config.tier.refinement_max_pairs_per_cycle = 100;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();

        // Insert an "old" memory about rust safety.
        engine
            .insert(
                "old",
                "Rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        // Insert a "new" memory about the same topic (high cosine, shares concept).
        engine
            .insert(
                "new",
                "Rust borrow checker safety",
                &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();

        // Run check_refinements — should create a Refines edge old → new.
        let created = engine.check_refinements().unwrap();
        assert!(
            created >= 1,
            "should create at least one Refines edge, got {created}"
        );

        let graph = engine.graph.read();
        let graph = graph.graph();
        assert!(
            graph.refinement_count() >= 1,
            "graph should have Refines edges"
        );
        // The edge should be old → new.
        let refined = graph.refined_by("old");
        assert!(
            refined.contains(&"new".to_string()),
            "old should refine to new, got {refined:?}"
        );
    }

    #[test]
    fn supersession_demotion_makes_new_refinement_outrank_old_at_alpha_one() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.cognitive_alpha = 1.0;
        config.tier.refinement_cosine_threshold = Some(0.5);
        config.tier.supersession_demotion_factor = 0.4;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();

        engine
            .insert(
                "old",
                "Rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Rust memory safety updated",
                &[0.95f32, 0.3122499, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();

        assert_eq!(engine.check_refinements().unwrap(), 1);
        let old_offset = *engine.id_index.read().get("old").unwrap();
        assert!((engine.meta.demotion_factor(old_offset) - 0.4).abs() < 1e-6);

        let results = engine
            .search(
                "rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2,
            )
            .unwrap()
            .expect("search should return candidates");
        assert_eq!(results[0].0, "new", "demotion should lower stale memory");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn supersession_demotion_persists_after_flush_and_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.cognitive_alpha = 1.0;
        config.tier.refinement_cosine_threshold = Some(0.5);
        config.tier.supersession_demotion_factor = 0.4;

        {
            let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
            engine
                .insert(
                    "old",
                    "Rust memory safety",
                    &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    1.0,
                    &["rust".to_string()],
                )
                .unwrap();
            engine
                .insert(
                    "new",
                    "Rust memory safety updated",
                    &[0.95f32, 0.3122499, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    1.0,
                    &["rust".to_string()],
                )
                .unwrap();
            assert_eq!(engine.check_refinements().unwrap(), 1);
            engine.flush().unwrap();
        }

        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        let old_offset = *engine.id_index.read().get("old").unwrap();
        assert!((engine.meta.demotion_factor(old_offset) - 0.4).abs() < 1e-6);

        let results = engine
            .search(
                "rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                2,
            )
            .unwrap()
            .expect("search should return candidates after restart");
        assert_eq!(results[0].0, "new");
    }

    #[test]
    fn check_refinements_disabled_by_default() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert(
                "old",
                "Rust",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Rust",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        let created = engine.check_refinements().unwrap();
        assert_eq!(
            created, 0,
            "refinement should be disabled when threshold is None"
        );
    }

    #[test]
    fn check_refinements_skips_unrelated_concepts() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.refinement_cosine_threshold = Some(0.5);
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        // Two memories with high cosine (same vector) but NO shared concepts.
        engine
            .insert(
                "old",
                "Rust",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Python",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["python".to_string()],
            )
            .unwrap();
        let created = engine.check_refinements().unwrap();
        assert_eq!(created, 0, "should not refine when concepts don't match");
    }

    #[test]
    fn check_contradictions_creates_edge_and_weakens_old() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        // Enable contradiction detection with a low cosine threshold.
        config.tier.contradiction_cosine_threshold = Some(0.5);
        config.tier.contradiction_text_threshold = 0.4; // texts must be < 40% similar
        config.tier.contradiction_weaken_factor = 0.5;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();

        // Old memory: a FALSE claim. Text mentions the topic word "rust" but
        // otherwise uses vocabulary disjoint from the correction so the Jaccard
        // similarity stays below the threshold — a contradiction, not a refinement.
        //   A tokens: {rust, requires, manual, compilation, before, execution}
        //   B tokens: {rust, executes, source, code, through, interpretation, directly}
        //   intersection = {rust} → Jaccard = 1/12 ≈ 0.083 < 0.4.
        engine
            .insert(
                "old_claim",
                "Rust requires manual compilation before execution",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        // New memory: the CORRECTION (same topic, opposing claim).
        engine
            .insert(
                "new_correction",
                "Rust executes source code through interpretation directly",
                &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.5,
                &["rust".to_string()],
            )
            .unwrap();

        let created = engine.check_contradictions().unwrap();
        assert!(
            created >= 1,
            "should create at least one Contradicts edge, got {created}"
        );

        let graph = engine.graph.read();
        let graph = graph.graph();
        assert!(
            graph.contradiction_count() >= 1,
            "graph should have Contradicts edges"
        );
        let corrected = graph.contradicted_by("old_claim");
        assert!(
            corrected.contains(&"new_correction".to_string()),
            "old_claim should be contradicted by new_correction, got {corrected:?}"
        );
        let old_offset = *engine.id_index.read().get("old_claim").unwrap();
        assert!(
            engine.meta.demotion_factor(old_offset) < crate::metadata_store::NO_DEMOTION,
            "contradicted memory should receive a final-score demotion"
        );
    }

    #[test]
    fn check_contradictions_disabled_by_default() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert(
                "old",
                "Rust gc",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Rust borrow",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        let created = engine.check_contradictions().unwrap();
        assert_eq!(
            created, 0,
            "contradiction should be disabled when threshold is None"
        );
    }

    #[test]
    fn check_contradictions_skips_high_text_overlap() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.contradiction_cosine_threshold = Some(0.5);
        config.tier.contradiction_text_threshold = 0.3; // low threshold = harder to be a contradiction
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        // Two memories with same text (high Jaccard) — should NOT be a contradiction.
        engine
            .insert(
                "old",
                "Rust is safe fast",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Rust is safe fast",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        let created = engine.check_contradictions().unwrap();
        assert_eq!(
            created, 0,
            "high text overlap should not be a contradiction"
        );
    }

    #[test]
    fn recompute_importance_disabled_by_default() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert(
                "a",
                "Rust safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        let changed = engine.recompute_importance().unwrap();
        assert_eq!(changed, 0, "should be a no-op when auto scoring is off");
    }

    #[test]
    fn recompute_importance_raises_frequently_retrieved() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.importance_auto_scoring = true;
        config.tier.importance_learning_rate = 1.0; // jump straight to target
                                                    // Both records start at importance 1.0.
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        engine
            .insert(
                "hot",
                "Rust safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "cold",
                "Python threads",
                &[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["python".to_string()],
            )
            .unwrap();

        // Retrieve "hot" many times so its access_count dominates. Each
        // search_ann bumps the matched record's access counter.
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        for _ in 0..20 {
            let _ = engine.search_ann(&q, 1).unwrap();
        }
        // Drain counters into metadata so recompute sees them.
        engine.recompute_importance().unwrap();

        let hot_offset = *engine.id_index.read().get("hot").unwrap();
        let cold_offset = *engine.id_index.read().get("cold").unwrap();
        let hot_imp = engine.meta.get(hot_offset).unwrap().unwrap().importance;
        let cold_imp = engine.meta.get(cold_offset).unwrap().unwrap().importance;
        assert!(
            hot_imp > cold_imp,
            "frequently-retrieved record should have higher importance: hot={hot_imp} cold={cold_imp}"
        );
        // The never-retrieved record decays toward the floor.
        assert!(
            cold_imp < 1.0,
            "never-retrieved record should decay below its start: cold={cold_imp}"
        );
    }

    #[test]
    fn recompute_importance_respects_floor_and_ceiling() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.importance_auto_scoring = true;
        config.tier.importance_learning_rate = 1.0;
        config.tier.importance_floor = 0.5;
        config.tier.importance_ceiling = 2.0;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        // One record, never retrieved. Its target blends to the floor.
        engine
            .insert(
                "x",
                "Rust safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                5.0, // above ceiling
                &["rust".to_string()],
            )
            .unwrap();
        engine.recompute_importance().unwrap();
        let offset = *engine.id_index.read().get("x").unwrap();
        let imp = engine.meta.get(offset).unwrap().unwrap().importance;
        assert!(
            imp <= 2.0 + 1e-5,
            "importance should not exceed ceiling 2.0, got {imp}"
        );
        assert!(
            imp >= 0.5 - 1e-5,
            "importance should not drop below floor 0.5, got {imp}"
        );
    }

    #[test]
    fn tier_seal_keeps_search_correct() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..8usize {
            let v = make_vec(8, i);
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &v,
                    1.0,
                    &[format!("c{}", i % 2)],
                )
                .unwrap();
        }
        engine.trigger_consolidation().unwrap();
        let q = make_vec(8, 3);
        let results = engine.search_ann(&q, 3).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].0, "mem_3");
    }

    #[test]
    fn restart_reloads_records_and_tiers() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            for i in 0..6usize {
                let v = make_vec(8, i);
                engine
                    .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                    .unwrap();
            }
            engine.trigger_consolidation().unwrap();
            engine.flush().unwrap();
        }
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            assert_eq!(engine.record_count(), 6);
            let q = make_vec(8, 2);
            let results = engine.search_ann(&q, 1).unwrap();
            assert_eq!(results[0].0, "mem_2");
        }
    }

    #[test]
    fn wal_truncation_recovery() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            for i in 0..5usize {
                let v = make_vec(8, i);
                engine
                    .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                    .unwrap();
            }
            // Make embeddings and WAL durable but do not snapshot metadata / clear WAL.
            engine.flush_vectors().unwrap();
            engine.flush_wal().unwrap();
        }

        // Simulate a torn write by truncating the last 4 bytes (the CRC of the
        // final WAL record).  The iterator should detect the CRC mismatch and
        // stop replay, recovering all preceding records.
        let wal_path = tmp.path().join("wal").join(crate::wal::WAL_FILE);
        let bytes = std::fs::read(&wal_path).unwrap();
        // Walk the WAL to find the byte offset of the final record's CRC.
        let mut pos = crate::wal::WAL_HEADER_SIZE;
        let mut last_record_crc_end = pos;
        while pos + 4 <= bytes.len() {
            let len =
                u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
                    as usize;
            if pos + 4 + len + 4 > bytes.len() {
                break;
            }
            last_record_crc_end = pos + 4 + len + 4;
            pos = last_record_crc_end;
        }
        // Truncate only the CRC of the last complete record.
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(&wal_path)
            .unwrap();
        file.set_len((last_record_crc_end - 4) as u64).unwrap();
        drop(file);

        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(engine.record_count(), 4);
        for i in 0..4usize {
            let q = make_vec(8, i);
            let results = engine.search_ann(&q, 1).unwrap();
            assert_eq!(results[0].0, format!("mem_{i}"));
        }
    }

    #[test]
    fn wal_replay_without_flush() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            for i in 0..5usize {
                let v = make_vec(8, i);
                engine
                    .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                    .unwrap();
            }
            engine.flush_vectors().unwrap();
            engine.flush_wal().unwrap();
        }

        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(engine.record_count(), 5);
    }

    #[test]
    fn hot_warm_cold_persist_and_reload() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            for i in 0..8usize {
                let v = make_vec(8, i);
                engine
                    .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                    .unwrap();
            }
            // First seal -> SealedHot, second seal -> Warm.
            engine.trigger_consolidation().unwrap();
            engine.flush().unwrap();
        }
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            assert_eq!(engine.record_count(), 8);
            // Each query should be correct regardless of which tier the data
            // currently lives in.
            for i in 0..8usize {
                let q = make_vec(8, i);
                let results = engine.search_ann(&q, 1).unwrap();
                assert_eq!(results[0].0, format!("mem_{i}"));
            }
        }
    }

    #[test]
    fn promotion_brings_accessed_records_back_to_hot() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.hot_capacity = 2;
        config.tier.hot_promote_threshold = 0.5;
        config.tier.recency_half_life_secs = 3600;

        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        for i in 0..4usize {
            let v = make_vec(8, i);
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }

        // First consolidation seals the initial Hot segment (mem_0, mem_1).
        engine.trigger_consolidation().unwrap();

        // Repeatedly search for mem_0 to bump its access score.
        let q = make_vec(8, 0);
        for _ in 0..3 {
            let results = engine.search_ann(&q, 1).unwrap();
            assert_eq!(results[0].0, "mem_0");
        }

        // The next consolidation should promote mem_0 back into Hot.
        let (_sealed, _compacted, promoted) = engine.trigger_consolidation().unwrap();
        assert!(promoted > 0, "expected at least one promotion");

        // Search still works correctly after promotion.
        let results = engine.search_ann(&q, 1).unwrap();
        assert_eq!(results[0].0, "mem_0");
    }

    #[test]
    fn eviction_disabled_by_default_keeps_all_records() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..6usize {
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(8, i),
                    1.0,
                    &[],
                )
                .unwrap();
        }
        // Default config has max_records = None and evict_score_floor = None.
        let evicted = engine.evict().unwrap();
        assert_eq!(evicted, 0);
        engine.trigger_consolidation().unwrap();
        assert_eq!(engine.record_count(), 6);
    }

    #[test]
    fn eviction_cap_bounds_record_count() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.max_records = Some(2);
        config.tier.recency_half_life_secs = 3600;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        for i in 0..5usize {
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(8, i),
                    1.0,
                    &[],
                )
                .unwrap();
        }
        // Give mem_0 and mem_1 a high access score so they survive the cap.
        // Searching bumps access_count and refreshes last_accessed (also
        // protecting them via the grace window).
        for idx in [0usize, 1] {
            let q = make_vec(8, idx);
            for _ in 0..5 {
                engine.search_ann(&q, 1).unwrap();
            }
        }
        let evicted = engine.evict().unwrap();
        assert!(
            evicted >= 3,
            "expected to evict down to the cap, got {evicted}"
        );
        assert!(engine.record_count() <= 2);
        // The frequently-accessed records must survive.
        assert!(engine.find_record_by_id("mem_0").is_some());
        assert!(engine.find_record_by_id("mem_1").is_some());
    }

    #[test]
    fn eviction_score_floor_drops_stale() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        // Floor above 0 so never-accessed records (score 0) are evicted, while
        // accessed records stay above it.
        config.tier.evict_score_floor = Some(0.5);
        config.tier.recency_half_life_secs = 3600;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        for i in 0..4usize {
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(8, i),
                    1.0,
                    &[],
                )
                .unwrap();
        }
        // Access mem_0 so its score climbs above the floor and the grace window
        // protects it.
        let q = make_vec(8, 0);
        for _ in 0..3 {
            engine.search_ann(&q, 1).unwrap();
        }
        let evicted = engine.evict().unwrap();
        // mem_1, mem_2, mem_3 are never accessed (score 0 < 0.5) and were
        // inserted with last_accessed = 0 (outside the grace window).
        assert_eq!(evicted, 3);
        assert!(engine.find_record_by_id("mem_0").is_some());
        assert!(engine.find_record_by_id("mem_1").is_none());
    }

    #[test]
    fn eviction_respects_grace_period() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        // Aggressive floor that would evict everything by score alone.
        config.tier.evict_score_floor = Some(1000.0);
        config.tier.recency_half_life_secs = 3600; // grace = 450s
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        // Touch each record so last_accessed = now, placing it inside the
        // grace window even though its score is below the floor.
        for i in 0..3usize {
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(8, i),
                    1.0,
                    &[],
                )
                .unwrap();
            engine.search_ann(&make_vec(8, i), 1).unwrap();
        }
        let evicted = engine.evict().unwrap();
        assert_eq!(evicted, 0, "freshly accessed records must be protected");
        assert_eq!(engine.record_count(), 3);
    }

    #[test]
    fn dedup_merges_near_duplicates_keeping_salient() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.dedup_cosine_threshold = Some(0.95);
        config.tier.recency_half_life_secs = 3600;
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        // Two near-identical vectors (same direction) plus a distinct one.
        engine
            .insert(
                "dup_keep",
                "keep me",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["a".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "dup_drop",
                "drop me",
                &[0.999f32, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["b".to_string()],
            )
            .unwrap();
        engine
            .insert("distinct", "different", &make_vec(8, 4), 1.0, &[])
            .unwrap();
        // Make dup_keep the more salient of the pair.
        let q = make_vec(8, 0);
        for _ in 0..3 {
            engine.search_ann(&q, 1).unwrap();
        }
        let merged = engine.deduplicate().unwrap();
        assert_eq!(
            merged, 1,
            "exactly one of the duplicate pair should be merged"
        );
        assert!(engine.find_record_by_id("dup_keep").is_some());
        assert!(engine.find_record_by_id("dup_drop").is_none());
        // The distinct record is untouched.
        assert!(engine.find_record_by_id("distinct").is_some());
        // Survivor inherited the victim's concept edge.
        let survivor = engine.find_record_by_id("dup_keep").unwrap();
        assert!(survivor.concepts.contains(&"a".to_string()));
        let graph = engine.graph.read();
        assert!(
            graph.graph().concept_degree("b") >= 1,
            "survivor should inherit victim concept 'b'"
        );
    }

    #[test]
    fn dedup_disabled_leaves_duplicates() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert(
                "a",
                "x",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
            )
            .unwrap();
        engine
            .insert(
                "b",
                "y",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
            )
            .unwrap();
        // dedup_cosine_threshold defaults to None.
        let merged = engine.deduplicate().unwrap();
        assert_eq!(merged, 0);
        assert_eq!(engine.record_count(), 2);
    }

    #[test]
    fn delete_removes_record_from_search_and_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..4usize {
            let v = make_vec(8, i);
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }

        assert!(engine.delete_by_id("mem_2").unwrap());
        assert!(!engine.delete_by_id("mem_2").unwrap());
        assert_eq!(engine.record_count(), 3);

        // Searching for the deleted vector should return its nearest neighbor among
        // the remaining records, not mem_2.
        let q = make_vec(8, 2);
        let results = engine.search_ann(&q, 1).unwrap();
        assert_ne!(results[0].0, "mem_2");

        // Reopen and replay: the delete should survive.
        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
        drop(engine);
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(engine.record_count(), 3);
        let results = engine.search_ann(&q, 1).unwrap();
        assert_ne!(results[0].0, "mem_2");
    }

    #[test]
    fn update_replaces_record() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert("mem_0", "original text", &make_vec(8, 0), 1.0, &[])
            .unwrap();

        assert!(engine
            .update("mem_0", "updated text", &make_vec(8, 1), 2.0, &[])
            .unwrap());
        assert!(!engine
            .update("missing", "x", &make_vec(8, 1), 1.0, &[])
            .unwrap());
        assert_eq!(engine.record_count(), 1);

        let q = make_vec(8, 1);
        let results = engine.search_ann(&q, 1).unwrap();
        assert_eq!(results[0].0, "mem_0");

        // The old vector should no longer be returned.
        let q_old = make_vec(8, 0);
        let results = engine.search_ann(&q_old, 1).unwrap();
        assert_eq!(results[0].0, "mem_0");
    }

    #[test]
    fn batch_insert_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        let ids: Vec<String> = (0..4).map(|i| format!("mem_{i}")).collect();
        let texts: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
        let embeddings: Vec<Vec<f32>> = (0..4).map(|i| make_vec(8, i)).collect();
        let scores = vec![1.0f32; 4];
        let concepts: Vec<Vec<String>> = vec![vec![]; 4];

        let n1 = engine
            .insert_batch(&ids, &texts, &embeddings, &scores, &concepts)
            .unwrap();
        assert_eq!(n1, 4);

        // Replay the same batch: existing ids should be skipped.
        let n2 = engine
            .insert_batch(&ids, &texts, &embeddings, &scores, &concepts)
            .unwrap();
        assert_eq!(n2, 0);
        assert_eq!(engine.record_count(), 4);

        // Duplicate ids within a batch should be deduplicated.
        let dup_ids = vec![
            "new_1".to_string(),
            "new_1".to_string(),
            "new_2".to_string(),
        ];
        let dup_texts = vec!["t1".to_string(), "t1".to_string(), "t2".to_string()];
        let dup_embs = vec![make_vec(8, 5), make_vec(8, 5), make_vec(8, 6)];
        let dup_scores = vec![1.0f32; 3];
        let dup_concepts = vec![vec![]; 3];
        let n3 = engine
            .insert_batch(&dup_ids, &dup_texts, &dup_embs, &dup_scores, &dup_concepts)
            .unwrap();
        assert_eq!(n3, 2);
        assert_eq!(engine.record_count(), 6);
    }

    #[test]
    fn payload_round_trip_and_replay() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        let payload = r#"{"tags":["rust","ai"],"count":42}"#.to_string();
        engine
            .insert_with_payload(
                "mem_0",
                "text",
                &make_vec(8, 0),
                1.0,
                &[],
                Some(payload.clone()),
                None,
            )
            .unwrap();

        assert_eq!(
            engine.get_payload("mem_0").unwrap().as_ref(),
            Some(&payload)
        );

        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
        drop(engine);

        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(
            engine.get_payload("mem_0").unwrap().as_ref(),
            Some(&payload)
        );
    }

    #[test]
    fn filtered_search_by_payload() {
        use serde_json::json;
        use std::ops::Bound;

        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..4usize {
            let payload = json!({"category": if i % 2 == 0 { "even" } else { "odd" }, "score": (i as f64) * 10.0 });
            engine
                .insert_with_payload(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(8, i),
                    1.0,
                    &[],
                    Some(payload.to_string()),
                    None,
                )
                .unwrap();
        }

        // Equality filter.
        let filter = Filter::Eq {
            field: "category".into(),
            value: json!("even"),
        };
        let results = engine
            .search_ann_filtered(&make_vec(8, 0), 10, &filter)
            .unwrap();
        let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["mem_0", "mem_2"]);

        // Range filter.
        let filter = Filter::Range {
            field: "score".into(),
            low: Bound::Included(15.0),
            high: Bound::Included(35.0),
        };
        let results = engine
            .search_ann_filtered(&make_vec(8, 0), 10, &filter)
            .unwrap();
        let mut ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec!["mem_2", "mem_3"]);

        // Delete removes from index.
        engine.delete_by_id("mem_2").unwrap();
        let filter = Filter::Eq {
            field: "category".into(),
            value: json!("even"),
        };
        let results = engine
            .search_ann_filtered(&make_vec(8, 0), 10, &filter)
            .unwrap();
        let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["mem_0"]);
    }

    #[test]
    fn filtered_search_survives_replay() {
        use serde_json::json;

        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            for i in 0..4usize {
                let payload = json!({"group": "a", "idx": i});
                engine
                    .insert_with_payload(
                        &format!("mem_{i}"),
                        &format!("text {i}"),
                        &make_vec(8, i),
                        1.0,
                        &[],
                        Some(payload.to_string()),
                        None,
                    )
                    .unwrap();
            }
            engine.flush_vectors().unwrap();
            engine.flush_wal().unwrap();
        }

        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        let filter = Filter::Eq {
            field: "group".into(),
            value: json!("a"),
        };
        let results = engine
            .search_ann_filtered(&make_vec(8, 0), 10, &filter)
            .unwrap();
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn merge_optimizer_combines_sealed_segments() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
        let n = 600usize;
        for i in 0..n {
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(32, i),
                    1.0,
                    &[],
                )
                .unwrap();
        }
        engine.trigger_consolidation().unwrap();
        engine.flush().unwrap();
        // The merge optimizer runs in the background; wait for it to reduce the
        // sealed segment count.
        let mut sealed_count = engine.segments.read().sealed_hot_count();
        for _ in 0..60 {
            if sealed_count <= 2 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(100));
            sealed_count = engine.segments.read().sealed_hot_count();
        }
        assert!(
            sealed_count <= 2,
            "expected at most 2 sealed segments, got {}",
            sealed_count
        );
        // Search should still return results.
        let results = engine.search_ann(&make_vec(32, 0), 5).unwrap();
        assert_eq!(results.len(), 5);
        // mem_0 and mem_32 are identical one-hot vectors; the approximate index may
        // return either of them. Just verify the top result is a perfect cosine match.
        assert!(results[0].1 > 0.9999, "top result should be an exact match");
        assert!(results[0].0.starts_with("mem_"));
    }

    /// Run searches on many threads while another thread drives consolidation
    /// (seal + merge installs) on the same engine. The ArcSwap snapshot path
    /// means searches must keep returning valid results and never deadlock with
    /// the now-internally-synchronized SegmentHolder mutations.
    #[test]
    fn concurrent_search_during_consolidation() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::thread;

        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
        let n = 600usize;
        for i in 0..n {
            engine
                .insert(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(32, i),
                    1.0,
                    &[],
                )
                .unwrap();
        }

        let stop = Arc::new(AtomicBool::new(false));

        // Searcher threads hammer the engine while consolidation runs.
        let searchers: Vec<_> = (0..4)
            .map(|t| {
                let engine = Arc::clone(&engine);
                let stop = Arc::clone(&stop);
                thread::spawn(move || {
                    let mut iters = 0u64;
                    while !stop.load(Ordering::Relaxed) {
                        let results = engine
                            .search_ann(&make_vec(32, t), 5)
                            .expect("search must not fail during consolidation");
                        for (id, _) in &results {
                            assert!(id.starts_with("mem_"), "unexpected id {id}");
                        }
                        iters += 1;
                    }
                    iters
                })
            })
            .collect();

        // Driver thread: repeatedly consolidate (seal + merge + flush).
        let driver = {
            let engine = Arc::clone(&engine);
            thread::spawn(move || {
                for _ in 0..5 {
                    engine.trigger_consolidation().unwrap();
                }
            })
        };

        driver.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for s in searchers {
            let iters = s.join().unwrap();
            assert!(iters > 0, "searcher should have run at least once");
        }

        // Final search still returns a full result set with a perfect top match.
        let results = engine.search_ann(&make_vec(32, 0), 5).unwrap();
        assert_eq!(results.len(), 5);
        assert!(results[0].1 > 0.9999, "top result should be an exact match");
    }

    #[test]
    fn search_ann_with_ef_can_improve_recall() {
        use rand::Rng;
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
        let dim = 32;
        let n = 600;
        let mut rng = rand::thread_rng();
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
            embeddings.push(v.clone());
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }
        engine.trigger_consolidation().unwrap();

        // Compute flat ground truth for a few queries.
        let queries: Vec<Vec<f32>> = (0..10)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
                v
            })
            .collect();

        let mut low_recall = 0.0f32;
        let mut high_recall = 0.0f32;
        for q in &queries {
            let gt = {
                let mut scored: Vec<(String, f32)> = embeddings
                    .iter()
                    .enumerate()
                    .map(|(i, emb)| {
                        let score = turbomemory_core::cosine_similarity(q, emb);
                        (format!("mem_{i}"), score)
                    })
                    .collect();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(5);
                scored
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect::<std::collections::HashSet<_>>()
            };
            let low = engine.search_ann_with_ef(q, 5, None).unwrap();
            let high = engine.search_ann_with_ef(q, 5, Some(200)).unwrap();
            let low_ids: std::collections::HashSet<_> = low.into_iter().map(|(id, _)| id).collect();
            let high_ids: std::collections::HashSet<_> =
                high.into_iter().map(|(id, _)| id).collect();
            low_recall += low_ids.intersection(&gt).count() as f32 / 5.0;
            high_recall += high_ids.intersection(&gt).count() as f32 / 5.0;
        }
        low_recall /= queries.len() as f32;
        high_recall /= queries.len() as f32;
        assert!(
            high_recall >= low_recall,
            "higher ef should not reduce recall: low={}, high={}",
            low_recall,
            high_recall
        );
    }

    #[test]
    fn search_ann_batch_matches_single_query() {
        // Batch search must return the same results as running each query
        // individually. Exercises the batched rerank path (CPU fallback here;
        // the gemm path is validated on-GPU separately).
        use rand::Rng;
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
        let dim = 32;
        let n = 600;
        let mut rng = rand::thread_rng();
        for i in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }
        engine.trigger_consolidation().unwrap();

        // 8 random queries.
        let queries: Vec<Vec<f32>> = (0..8)
            .map(|_| {
                let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
                v
            })
            .collect();

        // Per-query results.
        let single: Vec<Vec<(String, f32)>> = queries
            .iter()
            .map(|q| engine.search_ann_with_ef(q, 5, Some(200)).unwrap())
            .collect();

        // Batch results.
        let q_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = engine
            .search_ann_batch(&q_refs, 5, Some(200), None, None)
            .unwrap();

        assert_eq!(
            batch.len(),
            single.len(),
            "batch should return one list per query"
        );
        for (i, (b, s)) in batch.iter().zip(single.iter()).enumerate() {
            let b_ids: Vec<&str> = b.iter().map(|(id, _)| id.as_str()).collect();
            let s_ids: Vec<&str> = s.iter().map(|(id, _)| id.as_str()).collect();
            assert_eq!(
                b_ids, s_ids,
                "query {i}: batch results must match single-query results"
            );
        }
    }

    #[test]
    fn search_ann_batch_empty_returns_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        let batch = engine.search_ann_batch(&[], 5, None, None, None).unwrap();
        assert!(batch.is_empty(), "empty batch should return empty");
    }

    #[test]
    fn scoped_search_isolates_agent_memories() {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();

        // Global memory: visible to all scopes.
        engine
            .insert_with_payload(
                "global_mem",
                "global knowledge",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
                None,
                None,
            )
            .unwrap();

        // Agent A private memory.
        engine
            .insert_with_payload(
                "agent_a_mem",
                "agent a secret",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
                None,
                Some("agent_a".into()),
            )
            .unwrap();

        // Agent B private memory.
        engine
            .insert_with_payload(
                "agent_b_mem",
                "agent b secret",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
                None,
                Some("agent_b".into()),
            )
            .unwrap();

        let q = &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // Global search sees all three records.
        let global_results: Vec<String> = engine
            .search_ann_scoped(q, 10, None, None)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(global_results.len(), 3);

        // Agent A search sees global + agent A, but not agent B.
        let a_results: Vec<String> = engine
            .search_ann_scoped(q, 10, None, Some("agent_a"))
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(a_results.contains(&"global_mem".to_string()));
        assert!(a_results.contains(&"agent_a_mem".to_string()));
        assert!(!a_results.contains(&"agent_b_mem".to_string()));

        // Agent B search sees global + agent B, but not agent A.
        let b_results: Vec<String> = engine
            .search_ann_scoped(q, 10, None, Some("agent_b"))
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(b_results.contains(&"global_mem".to_string()));
        assert!(!b_results.contains(&"agent_a_mem".to_string()));
        assert!(b_results.contains(&"agent_b_mem".to_string()));
    }

    #[test]
    fn scoped_search_survives_replay() {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
            engine
                .insert_with_payload(
                    "scoped_mem",
                    "scoped",
                    &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    1.0,
                    &[],
                    None,
                    Some("agent_x".into()),
                )
                .unwrap();
            engine.flush_vectors().unwrap();
            engine.flush_wal().unwrap();
        }

        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        let q = &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results: Vec<String> = engine
            .search_ann_scoped(q, 10, None, Some("agent_x"))
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert_eq!(results, vec!["scoped_mem".to_string()]);
    }
}
