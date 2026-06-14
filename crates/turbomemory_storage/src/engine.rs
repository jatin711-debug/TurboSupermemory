//! Top-level storage engine combining durable metadata, tiered vector segments,
//! and the cognitive graph.
//!
//! Durability model:
//!   1. WAL is the source of truth.
//!   2. Writes append a framed WAL entry first, then update in-memory state.
//!   3. `redb` (via `MetadataStore`) is a lazy snapshot; it is flushed only on
//!      explicit `flush()` / background consolidation.
//!   4. On open we replay any un-flushed WAL entries, persist a snapshot, then
//!      rebuild the id index, graph, and tiered segments from the snapshot.

use crate::config::StoreConfig;
use crate::metadata_store::MetadataStore;
use crate::record::{PointOffset, Record};
use crate::segment_holder::SegmentHolder;
use crate::wal::{Wal, WalOp};
use crate::StorageError;
use ahash::HashMap as AHashMap;
use parking_lot::{Mutex, RwLock};
use std::path::Path;
use std::sync::Arc;
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
    segments: Arc<RwLock<SegmentHolder>>,
    graph: Arc<RwLock<SpreadingActivation>>,
    ccs: Arc<Mutex<Option<CompressedCognitiveState>>>,
    id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>,
    wal: Arc<Mutex<Wal>>,
}

impl Clone for StorageEngine {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            meta: self.meta.clone(),
            segments: self.segments.clone(),
            graph: self.graph.clone(),
            ccs: self.ccs.clone(),
            id_index: self.id_index.clone(),
            wal: self.wal.clone(),
        }
    }
}

impl StorageEngine {
    pub fn open(db_path: impl AsRef<Path>, config: StoreConfig) -> crate::Result<Self> {
        let db_path = db_path.as_ref();
        let meta = MetadataStore::open(db_path)?;
        let wal_path = db_path.join(WAL_DIR);
        let mut wal = Wal::open(&wal_path)?;

        // Replay any un-flushed WAL entries into the metadata cache.
        let last_applied = meta.last_applied_seq().unwrap_or(0);
        let mut max_seq = last_applied;
        let mut max_offset = 0u64;
        let mut replayed = false;
        for op in wal.iter()? {
            match op? {
                WalOp::Insert { offset, seq, record } => {
                    if seq > last_applied {
                        meta.put(offset, &record)?;
                        max_offset = max_offset.max(offset);
                        max_seq = max_seq.max(seq);
                        replayed = true;
                    }
                }
                WalOp::Delete { offset } => {
                    meta.remove(offset)?;
                    replayed = true;
                }
                WalOp::Flush { .. } => {}
            }
        }

        if replayed {
            meta.advance_offset_past(max_offset);
            meta.advance_seq_past(max_seq);
            // Persist the recovered snapshot and discard the now-redundant WAL.
            meta.flush(max_seq)?;
            wal.flush()?;
            wal.clear()?;
        }

        let records = meta.records()?;
        let mut records_vec: Vec<(PointOffset, Record)> = records.into_iter().collect();
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

        let segments = SegmentHolder::from_records(
            config.clone(),
            db_path.join("segments"),
            &records_vec,
            &meta,
        )?;

        Ok(Self {
            config: Arc::new(config),
            meta: Arc::new(meta),
            segments: Arc::new(RwLock::new(segments)),
            graph: Arc::new(RwLock::new(graph)),
            ccs: Arc::new(Mutex::new(ccs)),
            id_index: Arc::new(RwLock::new(id_index)),
            wal: Arc::new(Mutex::new(wal)),
        })
    }

    pub fn insert(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
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
            tier: crate::config::Tier::Hot,
        };

        // 1. WAL is the source of truth.
        {
            let mut wal = self.wal.lock();
            wal.append(&WalOp::Insert {
                offset,
                seq,
                record: record.clone(),
            })?;
        }

        // 2. Update in-memory state.
        self.meta.put(offset, &record)?;
        self.id_index.write().insert(Arc::from(id), offset);

        {
            let mut graph = self.graph.write();
            graph.add_memory(id, text, concepts);
        }
        {
            let mut segments = self.segments.write();
            segments.insert(offset, &record, &self.meta)?;
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
        let n = ids.len();
        if n == 0 {
            return Ok(0);
        }
        if texts.len() < n || embeddings.len() < n || importances.len() < n || concepts.len() < n {
            return Err(StorageError::InvalidArgument(
                "batch arrays have mismatched lengths".into(),
            ));
        }
        for emb in embeddings {
            validate_dimension(emb, self.config.dimension)?;
        }
        {
            let idx = self.id_index.read();
            for id in ids {
                if idx.contains_key(id.as_str()) {
                    return Err(StorageError::DuplicateId(id.clone()));
                }
            }
        }

        let mut records: Vec<(PointOffset, Record)> = Vec::with_capacity(n);
        for i in 0..n {
            let mut emb = embeddings[i].clone();
            normalize(&mut emb)?;
            let offset = self.meta.allocate_offset();
            let seq = self.meta.allocate_seq();
            let record = Record {
                id: ids[i].clone(),
                text: texts[i].clone(),
                embedding: Arc::from(emb),
                importance: importances[i],
                concepts: concepts[i].clone(),
                created_at: now_secs(),
                insert_seq: seq,
                access_count: 0,
                tier: crate::config::Tier::Hot,
            };
            records.push((offset, record));
        }

        // 1. WAL first.
        {
            let mut wal = self.wal.lock();
            for (offset, record) in &records {
                wal.append(&WalOp::Insert {
                    offset: *offset,
                    seq: record.insert_seq,
                    record: record.clone(),
                })?;
            }
        }

        // 2. In-memory state.
        self.meta.put_batch(&records)?;
        {
            let mut idx = self.id_index.write();
            for (offset, rec) in &records {
                idx.insert(Arc::from(rec.id.as_str()), *offset);
            }
        }

        {
            let mut graph = self.graph.write();
            for (_, rec) in &records {
                graph.add_memory(&rec.id, &rec.text, &rec.concepts);
            }
        }
        {
            let mut segments = self.segments.write();
            for (offset, record) in &records {
                segments.insert(*offset, record, &self.meta)?;
            }
        }
        Ok(n)
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
        validate_dimension(query_embedding, self.config.dimension)?;
        if self.record_count() <= EXACT_FALLBACK_THRESHOLD {
            return Ok(self.exact_top_k(query_embedding, top_k));
        }
        let segments = self.segments.read();
        let scored = segments.search(query_embedding, top_k, &self.meta)?;
        Ok(scored
            .into_iter()
            .filter_map(|c| {
                self.meta
                    .get(c.offset)
                    .ok()
                    .flatten()
                    .map(|r| (r.id, c.score))
            })
            .collect())
    }

    fn exact_top_k(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        let mut all: Vec<(String, f32)> = self
            .meta
            .records()
            .unwrap_or_default()
            .values()
            .map(|rec| {
                let score = cosine_similarity(query, rec.embedding_f32());
                (rec.id.clone(), score)
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
                if let Some((offset, mut rec)) = self.find_record_entry_by_id(id) {
                    rec.access_count += 1;
                    let _ = self.meta.put(offset, &rec);
                }
            }
            Ok(Some(hydrated))
        } else {
            Ok(None)
        }
    }

    fn find_record_by_id(&self, id: &str) -> Option<Record> {
        let idx = self.id_index.read();
        idx.get(id)
            .and_then(|&offset| self.meta.get(offset).ok().flatten())
    }

    fn find_record_entry_by_id(&self, id: &str) -> Option<(PointOffset, Record)> {
        let idx = self.id_index.read();
        idx.get(id)
            .copied()
            .and_then(|offset| self.meta.get(offset).ok().flatten().map(|r| (offset, r)))
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

    pub fn trigger_consolidation(&self) -> crate::Result<(usize, usize)> {
        let mut segments = self.segments.write();
        let (sealed, compacted) = segments.trigger_consolidation(&self.meta)?;
        drop(segments);
        self.save_graph()?;
        Ok((sealed, compacted))
    }

    pub fn flush(&self) -> crate::Result<()> {
        // 1. Durably sync the WAL.
        {
            let mut wal = self.wal.lock();
            wal.flush()?;
        }

        // 2. Persist a snapshot of dirty records + sequence counters to redb.
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
}

fn build_graph(records: &[(PointOffset, Record)]) -> SpreadingActivation {
    let mut graph = MemoryGraph::new();
    for (_, rec) in records {
        graph.add_memory(&rec.id, &rec.text, &rec.concepts);
    }
    SpreadingActivation::new(graph, SpreadingConfig::default())
}

fn now_secs() -> u64 {
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
}
