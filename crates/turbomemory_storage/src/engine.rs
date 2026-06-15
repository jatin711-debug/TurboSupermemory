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

use crate::config::StoreConfig;
use crate::metadata_store::MetadataStore;
use crate::payload_index::{Filter, PayloadIndex};
use crate::record::{MetaRecord, PointOffset, Record};
use crate::optimizer::BackgroundOptimizer;
use crate::text_index::TextIndex;
use crate::segment_holder::SegmentHolder;
use crate::vector_store::VectorStore;
use crate::wal::{Wal, WalOp};
use crate::StorageError;
use ahash::HashMap as AHashMap;
use parking_lot::{Mutex, RwLock};
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use turbomemory_core::{cosine_similarity, normalize, validate_dimension};
use turbomemory_graph::{
    step_session, CompressedCognitiveState, MemoryGraph, SpreadingActivation, SpreadingConfig,
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
    graph: Arc<RwLock<SpreadingActivation>>,
    ccs: Arc<Mutex<Option<CompressedCognitiveState>>>,
    id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>,
    payload_index: Arc<RwLock<PayloadIndex>>,
    text_index: Arc<TextIndex>,
    wal: Arc<Mutex<Wal>>,
    optimizer: Arc<BackgroundOptimizer>,
}

impl Clone for StorageEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            meta: self.meta.clone(),
            vectors: self.vectors.clone(),
            segments: self.segments.clone(),
            graph: self.graph.clone(),
            ccs: self.ccs.clone(),
            id_index: self.id_index.clone(),
            payload_index: self.payload_index.clone(),
            text_index: self.text_index.clone(),
            wal: self.wal.clone(),
            optimizer: self.optimizer.clone(),
        }
    }
}

impl StorageEngine {
    pub fn open(db_path: impl AsRef<Path>, config: StoreConfig) -> crate::Result<Arc<Self>> {
        let db_path = db_path.as_ref();
        let meta = MetadataStore::open(db_path)?;
        let vectors = VectorStore::open(db_path.join("vectors.bin"), config.dimension)?;
        let wal_path = db_path.join(WAL_DIR);
        let mut wal = Wal::open(&wal_path)?;

        // Replay any un-flushed WAL entries into the metadata cache and vector store.
        let last_applied = meta.last_applied_seq().unwrap_or(0);
        let mut max_seq = last_applied;
        let mut max_offset = 0u64;
        let mut replayed = false;
        for op in wal.iter()? {
            match op? {
                WalOp::Insert { offset, seq, meta: meta_rec } => {
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

        let records = meta.records()?;

        // Rebuild the payload index from the metadata snapshot.  WAL replay above
        // already added any records that were not yet flushed.
        let payload_index = Arc::new(RwLock::new(PayloadIndex::from_meta_records(&records)));

        // Rebuild the full-text index from the metadata snapshot.
        let text_index = Arc::new(TextIndex::open(db_path.join("text_index"))?);
        for (offset, meta_rec) in &records {
            text_index.add(*offset, &meta_rec.text)?;
        }
        text_index.commit()?;

        let view = vectors.read_view();
        let mut records_vec: Vec<(PointOffset, Record)> = records
            .into_iter()
            .filter_map(|(offset, meta_rec)| {
                view.get(offset)
                    .map(|v| (offset, meta_rec.with_embedding(Arc::from(Vec::from(v)))))
            })
            .collect();
        drop(view);
        records_vec.sort_by(|a, b| {
            a.1.created_at
                .cmp(&b.1.created_at)
                .then(a.1.insert_seq.cmp(&b.1.insert_seq))
        });

        let id_index: AHashMap<Arc<str>, PointOffset> = records_vec
            .iter()
            .map(|(offset, rec)| (Arc::from(rec.id.as_str()), *offset))
            .collect();
        let graph = build_graph(&records_vec);
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

        let mut segments = SegmentHolder::from_records(
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

        let interval = config
            .auto_consolidation_interval
            .unwrap_or_else(|| Duration::from_secs(60));

        Ok(Arc::new_cyclic(move |weak| {
            let optimizer = BackgroundOptimizer::new(weak.clone(), interval);
            Self {
                config: Arc::new(config),
                meta: Arc::new(meta),
                vectors: Arc::new(vectors),
                segments: Arc::new(RwLock::new(segments)),
                graph: Arc::new(RwLock::new(graph)),
                ccs: Arc::new(Mutex::new(ccs)),
                id_index: Arc::new(RwLock::new(id_index)),
                payload_index,
                text_index,
                wal: Arc::new(Mutex::new(wal)),
                optimizer: Arc::new(optimizer),
            }
        }))
    }

    pub fn insert(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
    ) -> crate::Result<bool> {
        self.insert_with_payload(id, text, embedding, importance, concepts, None)
    }

    pub fn insert_with_payload(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
    ) -> crate::Result<bool> {
        validate_dimension(embedding, self.config.dimension)?;
        if self.id_index.read().contains_key(id) {
            return Err(StorageError::DuplicateId(id.to_string()));
        }
        let mut emb = embedding.to_vec();
        normalize(&mut emb)?;
        let offset = self.meta.allocate_offset();
        let seq = self.meta.allocate_seq();
        let record = Record {
            id: id.to_string(),
            text: text.to_string(),
            embedding: Arc::from(emb),
            importance,
            concepts: concepts.to_vec(),
            created_at: now_secs(),
            insert_seq: seq,
            access_count: 0,
            last_accessed: 0,
            tier: crate::config::Tier::Hot,
            payload,
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

        // 3. Update in-memory metadata cache and indexes.
        self.meta.put(offset, &record)?;
        self.id_index.write().insert(Arc::from(id), offset);
        self.payload_index
            .write()
            .add(offset, record.payload.as_deref())?;
        self.text_index.add(offset, text)?;

        {
            let mut graph = self.graph.write();
            graph.add_memory(id, text, concepts);
        }
        let needs_seal = {
            let segments = self.segments.read();
            segments.insert(offset, &record, &self.vectors)?
        };
        if needs_seal {
            let mut segments = self.segments.write();
            segments.seal_hot(&self.vectors)?;
        }
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
        self.insert_batch_with_payload(ids, texts, embeddings, importances, concepts, &[])
    }

    pub fn insert_batch_with_payload(
        &self,
        ids: &[String],
        texts: &[String],
        embeddings: &[Vec<f32>],
        importances: &[f32],
        concepts: &[Vec<String>],
        payloads: &[Option<String>],
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
        {
            return Err(StorageError::InvalidArgument(
                "batch arrays have mismatched lengths".into(),
            ));
        }
        for emb in embeddings {
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

        let mut records: Vec<(PointOffset, Record)> = Vec::with_capacity(indices.len());
        for &i in &indices {
            let mut emb = embeddings[i].clone();
            normalize(&mut emb)?;
            let offset = self.meta.allocate_offset();
            let seq = self.meta.allocate_seq();
            let payload = if payloads.is_empty() {
                None
            } else {
                payloads[i].clone()
            };
            let record = Record {
                id: ids[i].clone(),
                text: texts[i].clone(),
                embedding: Arc::from(emb),
                importance: importances[i],
                concepts: concepts[i].clone(),
                created_at: now_secs(),
                insert_seq: seq,
                access_count: 0,
                last_accessed: 0,
                tier: crate::config::Tier::Hot,
                payload,
            };
            records.push((offset, record));
        }

        // 1. Persist embeddings to the mmap-backed vector store first.
        for (offset, record) in &records {
            self.vectors.put(*offset, record.embedding_f32())?;
        }

        // 2. WAL metadata entries.
        {
            let mut wal = self.wal.lock();
            for (offset, record) in &records {
                let meta = MetaRecord::from(record);
                wal.append(&WalOp::Insert {
                    offset: *offset,
                    seq: record.insert_seq,
                    meta,
                })?;
            }
        }

        // 3. In-memory metadata cache and indexes.
        self.meta.put_batch(&records)?;
        {
            let mut idx = self.id_index.write();
            let mut pidx = self.payload_index.write();
            for (offset, rec) in &records {
                idx.insert(Arc::from(rec.id.as_str()), *offset);
                pidx.add(*offset, rec.payload.as_deref())?;
                self.text_index.add(*offset, &rec.text)?;
            }
        }

        {
            let mut graph = self.graph.write();
            for (_, rec) in &records {
                graph.add_memory(&rec.id, &rec.text, &rec.concepts);
            }
        }
        let mut needs_seal = false;
        {
            let segments = self.segments.read();
            for (offset, record) in &records {
                if segments.insert(*offset, record, &self.vectors)? {
                    needs_seal = true;
                }
            }
        }
        if needs_seal {
            let mut segments = self.segments.write();
            segments.seal_hot(&self.vectors)?;
        }
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

        // 2. Remove from payload and text indexes while we still know the old values.
        if let Ok(Some(meta_rec)) = self.meta.get(offset) {
            self.payload_index
                .write()
                .remove(offset, meta_rec.payload.as_deref());
            self.text_index.remove(offset)?;
        }

        // 3. Remove from in-memory metadata and id index.
        self.meta.remove(offset)?;
        self.id_index.write().remove(id);

        // 3. Remove from cognitive graph.
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
        self.update_with_payload(id, text, embedding, importance, concepts, None)
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

    pub fn update_with_payload(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
    ) -> crate::Result<bool> {
        if !self.id_index.read().contains_key(id) {
            return Ok(false);
        }
        self.delete_by_id(id)?;
        self.insert_with_payload(id, text, embedding, importance, concepts, payload)?;
        Ok(true)
    }

    pub fn search_ann(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        let candidates = self.search_ann_candidates(query_embedding, top_k)?;
        Ok(candidates)
    }

    pub fn search_ann_candidates(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered(query_embedding, top_k, None)
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
        validate_dimension(query_embedding, self.config.dimension)?;
        let allowed_offsets = match filter {
            Some(f) => Some(self.evaluate_filter(f)?),
            None => None,
        };
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
        let segments = self.segments.read();
        let scored = segments.search(
            query_embedding,
            top_k,
            &self.vectors,
            allowed_offsets.as_ref(),
        )?;
        drop(segments);
        let mut results = Vec::with_capacity(scored.len());
        for c in scored {
            if let Some(meta_rec) = self.meta.get(c.offset)? {
                self.bump_access(c.offset);
                results.push((meta_rec.id, c.score));
            }
        }
        Ok(results)
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
        let mut all: Vec<(String, f32)> = self
            .meta
            .records()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(offset, rec)| {
                if !allowed_offsets.is_empty() && !allowed_offsets.contains(offset as u32) {
                    return None;
                }
                view.get(offset).map(|v| {
                    let score = cosine_similarity(query, v);
                    (rec.id.clone(), score)
                })
            })
            .collect();
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
        validate_dimension(query_embedding, self.config.dimension)?;
        let seeds = self.search_ann_candidates(query_embedding, top_k.max(10))?;
        let graph = self.graph.read();
        let activated = graph.search(query_text, &seeds, top_k);
        drop(graph);
        if let Some(results) = activated {
            let mut hydrated: Vec<(String, f32)> = results
                .into_iter()
                .filter_map(|(id, _)| {
                    self.find_record_by_id(&id).map(|rec| {
                        let score = cosine_similarity(query_embedding, rec.embedding_f32());
                        (id, score)
                    })
                })
                .collect();
            hydrated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            hydrated.truncate(top_k);
            if hydrated.is_empty() {
                return Ok(None);
            }
            // Bump access counts.
            for (id, _) in &hydrated {
                self.bump_access_by_id(id);
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
        validate_dimension(query_embedding, self.config.dimension)?;
        let seeds =
            self.search_ann_candidates_filtered(query_embedding, top_k.max(10), Some(filter))?;
        let graph = self.graph.read();
        let activated = graph.search(query_text, &seeds, top_k);
        drop(graph);
        if let Some(results) = activated {
            let mut hydrated: Vec<(String, f32)> = results
                .into_iter()
                .filter_map(|(id, _)| {
                    self.find_record_by_id(&id).map(|rec| {
                        let score = cosine_similarity(query_embedding, rec.embedding_f32());
                        (id, score)
                    })
                })
                .collect();
            hydrated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            hydrated.truncate(top_k);
            if hydrated.is_empty() {
                return Ok(None);
            }
            for (id, _) in &hydrated {
                self.bump_access_by_id(id);
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
        idx.get(id).copied().and_then(|offset| self.get_record(offset))
    }

    /// Bump the access score for the record with the given offset.
    fn bump_access(&self, offset: PointOffset) {
        let Some(mut meta_rec) = self.meta.get(offset).ok().flatten() else {
            return;
        };
        meta_rec.access_count += 1;
        meta_rec.last_accessed = now_secs();
        let _ = self.meta.put_meta(offset, &meta_rec);
    }

    fn bump_access_by_id(&self, id: &str) {
        let idx = self.id_index.read();
        if let Some(&offset) = idx.get(id) {
            drop(idx);
            self.bump_access(offset);
        }
    }

    pub fn step_session(
        &self,
        user_input: &str,
        assistant_response: &str,
    ) -> crate::Result<String> {
        let ccs_json = self.ccs.lock().as_ref().map(|c| c.to_json());
        let json = step_session(ccs_json.as_deref(), user_input, assistant_response);
        *self.ccs.lock() = serde_json::from_str(&json).ok();
        self.save_ccs()?;
        Ok(json)
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
    /// Callers that own a background `UpdateHandler` should stop it before
    /// calling shutdown.
    pub fn shutdown(&self) -> crate::Result<()> {
        self.flush()
    }

    pub fn trigger_consolidation(&self) -> crate::Result<(usize, usize, usize)> {
        let mut segments = self.segments.write();
        let (sealed, compacted, promoted) = segments.trigger_consolidation(&self.meta, &self.vectors)?;
        drop(segments);
        self.save_graph()?;
        Ok((sealed, compacted, promoted))
    }

    pub fn flush(&self) -> crate::Result<()> {
        // 1. Build any pending plain segments so the durable snapshot captures
        //    them as persisted HNSW / quantized segments rather than in-memory
        //    plain indexes.
        while crate::optimizer::BackgroundOptimizer::process_one_seal(self).unwrap_or(false) {}

        // 2. Durably sync the WAL.
        {
            let mut wal = self.wal.lock();
            wal.flush()?;
        }

        // 3. Persist the vector snapshot, metadata snapshot, and text index.
        self.vectors.flush()?;
        self.text_index.flush()?;
        let last_applied_seq = self.meta.next_seq().saturating_sub(1);
        self.meta.flush(last_applied_seq)?;

        // 3. Flush tiered segment files.
        let segments = self.segments.read();
        segments.flush()?;
        drop(segments);

        // 4. Persist graph / CCS metadata.
        self.save_graph()?;
        self.save_ccs()?;

        // 5. WAL is now fully captured by the redb snapshot; truncate it.
        {
            let mut wal = self.wal.lock();
            wal.clear()?;
        }

        Ok(())
    }

    pub fn record_count(&self) -> usize {
        self.meta.records().map(|m| m.len()).unwrap_or(0)
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

fn build_graph(records: &[(PointOffset, Record)]) -> SpreadingActivation {
    let mut graph = MemoryGraph::new();
    for (_, rec) in records {
        graph.add_memory(&rec.id, &rec.text, &rec.concepts);
    }
    SpreadingActivation::new(graph, SpreadingConfig::default())
}

pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TierConfig;

    fn small_config(dim: usize) -> StoreConfig {
        StoreConfig {
            dimension: dim,
            max_edges: 3,
            search_list_size: 5,
            outlier_count: 0,
            initial_capacity: 16,
            tier: TierConfig {
                hot_capacity: 3,
                warm_capacity: 6,
                warm_bits: 4,
                warm_chunk_bytes: 4096,
                cold_sign: true,
                hot_promote_threshold: 2.0,
                warm_demote_threshold: 0.5,
                recency_half_life_secs: 60,
            },
            auto_consolidation_interval: None,
        }
    }

    fn make_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx % dim] = 1.0;
        v
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
            let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
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

        assert!(
            engine
                .update("mem_0", "updated text", &make_vec(8, 1), 2.0, &[])
                .unwrap()
        );
        assert!(!engine.update("missing", "x", &make_vec(8, 1), 1.0, &[]).unwrap());
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
        let dup_ids = vec!["new_1".to_string(), "new_1".to_string(), "new_2".to_string()];
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
            )
            .unwrap();

        assert_eq!(engine.get_payload("mem_0").unwrap().as_ref(), Some(&payload));

        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
        drop(engine);

        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(engine.get_payload("mem_0").unwrap().as_ref(), Some(&payload));
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
                )
                .unwrap();
        }

        // Equality filter.
        let filter = Filter::Eq {
            field: "category".into(),
            value: json!("even"),
        };
        let results = engine.search_ann_filtered(&make_vec(8, 0), 10, &filter).unwrap();
        let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
        assert_eq!(ids, vec!["mem_0", "mem_2"]);

        // Range filter.
        let filter = Filter::Range {
            field: "score".into(),
            low: Bound::Included(15.0),
            high: Bound::Included(35.0),
        };
        let results = engine.search_ann_filtered(&make_vec(8, 0), 10, &filter).unwrap();
        let mut ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
        ids.sort();
        assert_eq!(ids, vec!["mem_2", "mem_3"]);

        // Delete removes from index.
        engine.delete_by_id("mem_2").unwrap();
        let filter = Filter::Eq {
            field: "category".into(),
            value: json!("even"),
        };
        let results = engine.search_ann_filtered(&make_vec(8, 0), 10, &filter).unwrap();
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
        let results = engine.search_ann_filtered(&make_vec(8, 0), 10, &filter).unwrap();
        assert_eq!(results.len(), 4);
    }
}
