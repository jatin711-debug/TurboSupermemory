//! Owns the tiered vector segments and decides when to roll them.

use crate::config::{StoreConfig, Tier};
use crate::metadata_store::MetadataStore;
use crate::record::{MetaRecord, PointOffset, Record};
use crate::segments::cold::ColdSegment;
use crate::segments::hot::HotSegment;
use crate::segments::sealed_hot::SealedHotSegment;
use crate::segments::warm::WarmSegment;
use crate::segments::{merge_candidates, ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SEALED_HOT_DIR: &str = "sealed_hot";

/// Owns Hot/Warm/Cold segments for a single collection.
pub struct SegmentHolder {
    config: StoreConfig,
    hot: Arc<RwLock<HotSegment>>,
    sealed_hot: Vec<Arc<RwLock<dyn VectorSegment>>>,
    warm: Vec<Arc<RwLock<dyn VectorSegment>>>,
    cold: Vec<Arc<RwLock<dyn VectorSegment>>>,
    next_segment_id: u64,
    base_path: PathBuf,
}

impl SegmentHolder {
    pub fn new(config: StoreConfig, base_path: impl AsRef<Path>) -> crate::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;
        Ok(Self {
            hot: Arc::new(RwLock::new(HotSegment::new(&config)?)),
            sealed_hot: Vec::new(),
            warm: Vec::new(),
            cold: Vec::new(),
            next_segment_id: 0,
            base_path,
            config,
        })
    }

    /// Create a holder from records, skipping offsets that are already captured
    /// by sealed segments on disk.
    pub fn from_records(
        config: StoreConfig,
        base_path: impl AsRef<Path>,
        records: &[(PointOffset, Record)],
        sealed_offsets: &HashSet<PointOffset>,
        vectors: &VectorStore,
    ) -> crate::Result<Self> {
        let mut holder = Self::new(config, base_path)?;
        for (offset, rec) in records {
            if !sealed_offsets.contains(offset) && holder.insert(*offset, rec, vectors)? {
                holder.seal_hot(vectors)?;
            }
        }
        Ok(holder)
    }

    fn segment_path(&mut self, tier: Tier) -> PathBuf {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        self.base_path
            .join(tier.name())
            .join(format!("segment_{id}"))
    }

    fn sealed_hot_path(&mut self) -> PathBuf {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        self.base_path.join(SEALED_HOT_DIR).join(format!("segment_{id}"))
    }

    pub fn add_sealed_hot(&mut self, segment: SealedHotSegment) {
        self.sealed_hot
            .push(Arc::new(RwLock::new(segment)) as Arc<RwLock<dyn VectorSegment>>);
    }

    pub fn add_warm(&mut self, segment: WarmSegment) {
        self.warm
            .push(Arc::new(RwLock::new(segment)) as Arc<RwLock<dyn VectorSegment>>);
    }

    pub fn add_cold(&mut self, segment: ColdSegment) {
        self.cold
            .push(Arc::new(RwLock::new(segment)) as Arc<RwLock<dyn VectorSegment>>);
    }

    /// Insert a record into the mutable Hot segment.
    ///
    /// Returns `true` if the Hot segment reached capacity and should be sealed.
    /// Callers that hold the holder lock should upgrade to a write lock and call
    /// [`SegmentHolder::seal_hot`] when this returns `true`.
    pub fn insert(
        &self,
        offset: PointOffset,
        record: &Record,
        _vectors: &VectorStore,
    ) -> crate::Result<bool> {
        let mut hot = self.hot.write();
        hot.insert(offset, record)?;
        Ok(hot.point_count() >= self.config.tier.hot_capacity)
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        vectors: &VectorStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        let pool_k = top_k.saturating_mul(4).max(top_k);
        let mut lists = Vec::with_capacity(2 + self.sealed_hot.len() + self.warm.len() + self.cold.len());
        lists.push(self.hot.read().search(query, pool_k, vectors)?);
        for sealed in &self.sealed_hot {
            lists.push(sealed.read().search(query, pool_k, vectors)?);
        }
        for warm in &self.warm {
            lists.push(warm.read().search(query, pool_k, vectors)?);
        }
        for cold in &self.cold {
            lists.push(cold.read().search(query, pool_k, vectors)?);
        }
        let candidates = merge_candidates(lists, pool_k);

        // Final rerank with full f32 embeddings from the vector store.
        let view = vectors.read_view();
        let mut reranked: Vec<ScoredPoint> = candidates
            .chunks(64)
            .flat_map(|chunk| {
                let mut pairs = Vec::with_capacity(chunk.len());
                for c in chunk {
                    if let Some(v) = view.get(c.offset) {
                        pairs.push((c, v));
                    }
                }
                let refs: Vec<&[f32]> = pairs.iter().map(|(_, v)| *v).collect();
                let scores = turbomemory_core::cosine_similarity_batch(query, &refs);
                pairs
                    .into_iter()
                    .zip(scores)
                    .map(|((c, _), score)| ScoredPoint {
                        offset: c.offset,
                        score,
                        tier: c.tier,
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    /// Move the current Hot segment into a persisted SealedHot/Warm segment and
    /// create a fresh Hot segment.
    pub fn seal_hot(&mut self, vectors: &VectorStore) -> crate::Result<()> {
        // Re-check capacity: another thread may have sealed while we waited for
        // the holder write lock.
        {
            let hot = self.hot.read();
            if hot.point_count() < self.config.tier.hot_capacity {
                return Ok(());
            }
        }

        // Collect the offsets currently in the Hot segment.
        let offsets: Vec<PointOffset> = {
            let hot = self.hot.read();
            hot.search(
                &vec![0.0f32; self.config.dimension],
                hot.point_count(),
                vectors,
            )?
            .into_iter()
            .map(|c| c.offset)
            .collect()
        };
        if offsets.is_empty() {
            return Ok(());
        }

        // Bulk-read vectors from the mmap store without cloning into `Record`s.
        let view = vectors.read_view();
        let mut vectors_for_build: Vec<(PointOffset, Vec<f32>)> =
            Vec::with_capacity(offsets.len());
        for &offset in &offsets {
            if let Some(v) = view.get(offset) {
                vectors_for_build.push((offset, Vec::from(v)));
            }
        }
        drop(view);
        if vectors_for_build.is_empty() {
            return Ok(());
        }
        let borrowed: Vec<(PointOffset, &[f32])> = vectors_for_build
            .iter()
            .map(|(offset, v)| (*offset, v.as_slice()))
            .collect();

        if self.sealed_hot.is_empty() {
            // First seal: keep as fast HNSW (SealedHot).
            let path = self.sealed_hot_path();
            let sealed = SealedHotSegment::from_vectors(&path, &self.config, &borrowed)?;
            self.sealed_hot
                .push(Arc::new(RwLock::new(sealed)) as Arc<RwLock<dyn VectorSegment>>);
        } else {
            // Subsequent seals: quantize to Warm to save memory.
            let path = self.segment_path(Tier::Warm);
            let records: Vec<(PointOffset, Record)> = vectors_for_build
                .iter()
                .map(|(offset, v)| {
                    (*offset, Record {
                        id: String::new(),
                        text: String::new(),
                        embedding: Arc::from(v.clone()),
                        importance: 0.0,
                        concepts: Vec::new(),
                        created_at: 0,
                        insert_seq: 0,
                        access_count: 0,
                        last_accessed: 0,
                        tier: Tier::Warm,
                    })
                })
                .collect();
            let warm = WarmSegment::from_records(&path, &records, self.config.tier.warm_bits)?;
            self.warm
                .push(Arc::new(RwLock::new(warm)) as Arc<RwLock<dyn VectorSegment>>);
        }
        *self.hot.write() = HotSegment::new(&self.config)?;
        // Roll warm to cold if over capacity.
        self.compact_warm(vectors)?;
        Ok(())
    }

    /// If total warm records exceed the warm capacity, merge all warm segments
    /// into a single Cold segment.
    fn compact_warm(&mut self, vectors: &VectorStore) -> crate::Result<()> {
        let total_warm: usize = self.warm.iter().map(|s| s.read().point_count()).sum();
        if total_warm <= self.config.tier.warm_capacity || self.warm.is_empty() {
            return Ok(());
        }
        let mut offsets = Vec::with_capacity(total_warm);
        for warm in &self.warm {
            offsets.extend_from_slice(warm.read().offsets());
        }
        offsets.sort_unstable();
        offsets.dedup();
        let records = self.build_records(&offsets, vectors)?;
        if !records.is_empty() {
            let path = self.segment_path(Tier::Cold);
            let cold = ColdSegment::from_records(&path, &records)?;
            self.cold
                .push(Arc::new(RwLock::new(cold)) as Arc<RwLock<dyn VectorSegment>>);
        }
        self.warm.clear();
        Ok(())
    }

    /// Build full `Record`s from metadata + vectors for the given offsets.
    /// This is used during sealing/compaction where we need an embedding.
    fn build_records(
        &self,
        offsets: &[PointOffset],
        vectors: &VectorStore,
    ) -> crate::Result<Vec<(PointOffset, Record)>> {
        // TODO: plumb MetadataStore into SegmentHolder so we can populate id/text
        // and other metadata. For now we only need the embedding for segment
        // construction, so the rest of the fields are left empty/default.
        let mut records = Vec::with_capacity(offsets.len());
        for &offset in offsets {
            if let Some(vec_guard) = vectors.get(offset) {
                let embedding = Arc::from(Vec::from(&*vec_guard));
                let record = Record {
                    id: String::new(),
                    text: String::new(),
                    embedding,
                    importance: 0.0,
                    concepts: Vec::new(),
                    created_at: 0,
                    insert_seq: 0,
                    access_count: 0,
                    last_accessed: 0,
                    tier: Tier::Warm,
                };
                records.push((offset, record));
            }
        }
        Ok(records)
    }

    pub fn point_count(&self) -> usize {
        self.hot.read().point_count()
            + self.sealed_hot.iter().map(|s| s.read().point_count()).sum::<usize>()
            + self.warm.iter().map(|s| s.read().point_count()).sum::<usize>()
            + self.cold.iter().map(|s| s.read().point_count()).sum::<usize>()
    }

    pub fn memory_bytes(&self) -> usize {
        self.hot.read().memory_bytes()
            + self.sealed_hot.iter().map(|s| s.read().memory_bytes()).sum::<usize>()
            + self.warm.iter().map(|s| s.read().memory_bytes()).sum::<usize>()
            + self.cold.iter().map(|s| s.read().memory_bytes()).sum::<usize>()
    }

    pub fn flush(&self) -> crate::Result<()> {
        (self.hot.read().flusher())()?;
        for sealed in &self.sealed_hot {
            (sealed.read().flusher())()?;
        }
        for warm in &self.warm {
            (warm.read().flusher())()?;
        }
        for cold in &self.cold {
            (cold.read().flusher())()?;
        }
        Ok(())
    }

    pub fn trigger_consolidation(
        &mut self,
        meta: &MetadataStore,
        vectors: &VectorStore,
    ) -> crate::Result<(usize, usize, usize)> {
        let mut sealed = 0usize;
        let mut compacted = 0usize;
        if self.hot.read().point_count() >= self.config.tier.hot_capacity {
            self.seal_hot(vectors)?;
            sealed += 1;
        }
        let total_warm: usize = self.warm.iter().map(|s| s.read().point_count()).sum();
        if total_warm > self.config.tier.warm_capacity {
            self.compact_warm(vectors)?;
            compacted += 1;
        }
        let promoted = self.promote_hot(meta, vectors)?;
        Ok((sealed, compacted, promoted))
    }

    /// Promote frequently-accessed records from sealed/warm/cold tiers back
    /// into the mutable Hot segment.
    pub fn promote_hot(
        &mut self,
        meta: &MetadataStore,
        vectors: &VectorStore,
    ) -> crate::Result<usize> {
        let now = crate::engine::now_secs();
        let threshold = self.config.tier.hot_promote_threshold;
        let half_life = self.config.tier.recency_half_life_secs.max(1);

        let mut candidate_offsets = Vec::new();
        for sealed in &self.sealed_hot {
            for &offset in sealed.read().offsets() {
                candidate_offsets.push(offset);
            }
        }
        for warm in &self.warm {
            for &offset in warm.read().offsets() {
                candidate_offsets.push(offset);
            }
        }
        for cold in &self.cold {
            for &offset in cold.read().offsets() {
                candidate_offsets.push(offset);
            }
        }

        let mut promoted = 0usize;
        for offset in candidate_offsets {
            let Some(meta_rec) = meta.get(offset)? else { continue };
            if access_score(&meta_rec, now, half_life) < threshold {
                continue;
            }
            let view = vectors.read_view();
            let Some(vec) = view.get(offset) else { continue };
            let embedding = Arc::from(Vec::from(vec));
            drop(view);
            let record = meta_rec
                .with_embedding(embedding)
                .with_tier(Tier::Hot);
            // Persist the tier change in metadata.
            meta.put(offset, &record)?;
            {
                let mut hot = self.hot.write();
                hot.insert(offset, &record)?;
                if hot.point_count() >= self.config.tier.hot_capacity {
                    drop(hot);
                    self.seal_hot(vectors)?;
                }
            }
            promoted += 1;
        }
        Ok(promoted)
    }
}

fn access_score(meta: &MetaRecord, now: u64, half_life: u64) -> f64 {
    let age = now.saturating_sub(meta.last_accessed).max(1);
    let recency = 2.0f64.powf(-(age as f64) / half_life as f64);
    meta.access_count as f64 * recency
}
