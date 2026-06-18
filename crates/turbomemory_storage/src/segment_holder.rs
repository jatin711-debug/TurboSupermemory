//! Owns the tiered vector segments and decides when to roll them.

use crate::config::{StoreConfig, Tier};
use crate::metadata_store::MetadataStore;
use crate::record::{MetaRecord, PointOffset, Record};
use crate::segments::cold::ColdSegment;
use crate::segments::hot::HotSegment;
use crate::segments::sealed_hot::SealedHotSegment;
use crate::segments::warm::WarmSegment;
use crate::segments::{kway_merge_topk, merge_candidates, ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use parking_lot::RwLock;
use rayon::prelude::*;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const SEALED_HOT_DIR: &str = "sealed_hot";

/// Owns Hot/Warm/Cold segments for a single collection.
pub struct SegmentHolder {
    config: StoreConfig,
    hot: Arc<RwLock<HotSegment>>,
    /// Plain segments that have been swapped out of the Hot tier and are waiting
    /// for the background optimizer to build their HNSW replacements. They remain
    /// searchable as exact segments until the build completes.
    sealing_plain: Vec<Arc<RwLock<HotSegment>>>,
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
            sealing_plain: Vec::new(),
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

    pub(crate) fn segment_path(&mut self, tier: Tier) -> PathBuf {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        self.base_path
            .join(tier.name())
            .join(format!("segment_{id}"))
    }

    pub(crate) fn sealed_hot_path(&mut self) -> PathBuf {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        self.base_path
            .join(SEALED_HOT_DIR)
            .join(format!("segment_{id}"))
    }

    pub(crate) fn pop_sealing_plain(&mut self) -> Option<Arc<RwLock<HotSegment>>> {
        self.sealing_plain.pop()
    }

    pub(crate) fn push_sealing_plain(&mut self, plain: Arc<RwLock<HotSegment>>) {
        self.sealing_plain.push(plain);
    }

    pub(crate) fn remove_sealing_plain(&mut self, target: &Arc<RwLock<HotSegment>>) {
        self.sealing_plain.retain(|p| !Arc::ptr_eq(p, target));
    }

    pub(crate) fn push_sealed_hot(&mut self, segment: SealedHotSegment) {
        self.add_sealed_hot(segment);
    }

    pub(crate) fn push_warm(&mut self, segment: WarmSegment) {
        self.add_warm(segment);
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

    #[allow(dead_code)]
    pub(crate) fn sealed_hot_count(&self) -> usize {
        self.sealed_hot.len()
    }

    /// Choose a group of sealed HNSW segments to merge.
    ///
    /// Returns clones of the segment Arcs (smallest first) while keeping the
    /// total point count under `hard_cap`. Segments are sorted by ascending
    /// point count so small segments are merged first. `hard_cap` allows
    /// exceeding `max_records` up to 3x when the optimizer is stuck above the
    /// target segment count, so the merge can always converge.
    pub(crate) fn sealed_hot_merge_candidates(
        &self,
        threshold: usize,
        max_records: usize,
    ) -> Option<Vec<Arc<RwLock<dyn VectorSegment>>>> {
        if self.sealed_hot.len() < threshold {
            return None;
        }
        // Sort by point count ascending so small segments are merged first.
        let mut sorted: Vec<Arc<RwLock<dyn VectorSegment>>> = self.sealed_hot.clone();
        sorted.sort_by_key(|s| s.read().point_count());

        let hard_cap = max_records.saturating_mul(2).max(max_records);
        let mut candidates = Vec::with_capacity(self.sealed_hot.len());
        let mut total = 0usize;
        for seg in sorted {
            let count = seg.read().point_count();
            if total > 0 && total.saturating_add(count) > hard_cap {
                break;
            }
            candidates.push(seg);
            total += count;
        }
        if candidates.len() >= threshold {
            Some(candidates)
        } else {
            None
        }
    }

    /// Remove sealed hot segments whose `Arc` pointer matches one of `targets`.
    ///
    /// Returns the number of segments removed. Paths of removed segments should
    /// be collected by the caller for later deletion.
    pub(crate) fn remove_sealed_hot_segments(
        &mut self,
        targets: &[Arc<RwLock<dyn VectorSegment>>],
    ) -> usize {
        let mut removed = 0usize;
        self.sealed_hot.retain(|seg| {
            if targets.iter().any(|t| Arc::ptr_eq(t, seg)) {
                removed += 1;
                false
            } else {
                true
            }
        });
        removed
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
        ef: Option<usize>,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<ScoredPoint>> {
        // Use Qdrant-style ef semantics: floor the per-segment candidate pool at
        // the caller-provided `ef` (or the configured search list size), then
        // apply an over-fetch multiplier that grows with filter strictness and
        // the number of segments.
        let base_ef = ef.unwrap_or(self.config.search_list_size);
        let base_multiplier = if allowed_offsets.is_some() { 8 } else { 4 };
        let segment_count = 1
            + self.sealing_plain.len()
            + self.sealed_hot.len()
            + self.warm.len()
            + self.cold.len();
        let selectivity = allowed_offsets
            .map(|b| {
                let total = self.point_count().max(1);
                let allowed = b.len() as usize;
                (allowed as f32) / (total as f32)
            })
            .unwrap_or(1.0f32);
        let multiplier = if selectivity < 0.01 {
            // Very selective filters rely on the exact fallback in each segment.
            base_multiplier
        } else {
            let selectivity_factor = (1.0f32 / selectivity.sqrt()).clamp(1.0f32, 16.0f32);
            // Cap the segment-factor growth so high segment counts do not
            // explode the candidate pool and rerank cost.
            let segment_factor =
                (1.0f32 + (segment_count.saturating_sub(1)) as f32 * 0.25f32).clamp(1.0f32, 2.5f32);
            (base_multiplier as f32 * selectivity_factor * segment_factor) as usize
        };
        let pool_k = top_k
            .saturating_mul(multiplier)
            .max(base_ef)
            .min(top_k.saturating_mul(48));

        // Collect all searchable segments and query them in parallel.
        let mut segments: Vec<Arc<RwLock<dyn VectorSegment>>> = Vec::with_capacity(
            2 + self.sealing_plain.len()
                + self.sealed_hot.len()
                + self.warm.len()
                + self.cold.len(),
        );
        segments.push(self.hot.clone());
        for plain in &self.sealing_plain {
            segments.push(plain.clone());
        }
        for sealed in &self.sealed_hot {
            segments.push(sealed.clone());
        }
        for warm in &self.warm {
            segments.push(warm.clone());
        }
        for cold in &self.cold {
            segments.push(cold.clone());
        }

        let lists: Vec<Vec<ScoredPoint>> = if segments.len() <= 1 {
            // Avoid Rayon's thread-pool overhead when there is only one segment
            // (the common small-collection case).
            segments
                .into_iter()
                .map(|seg| seg.read().search(query, pool_k, vectors, allowed_offsets))
                .collect::<crate::Result<Vec<_>>>()?
        } else {
            segments
                .into_par_iter()
                .map(|seg| seg.read().search(query, pool_k, vectors, allowed_offsets))
                .collect::<crate::Result<Vec<_>>>()?
        };
        // Merge per-segment candidates with a k-way heap merge (PR-B).
        // Fall back to the old sort-based merge for very small result sets.
        let candidates = if lists.len() >= 4 && pool_k >= 64 {
            kway_merge_topk(&lists, pool_k)
        } else {
            merge_candidates(lists, pool_k)
        };

        // Final rerank with full f32 embeddings from the vector store.
        let view = vectors.read_view();
        let reranked: Vec<ScoredPoint> = if candidates.len() >= 256 {
            let chunks: Vec<&[ScoredPoint]> = candidates.chunks(64).collect();
            chunks
                .into_par_iter()
                .flat_map(|chunk| {
                    let mut pairs = Vec::with_capacity(chunk.len());
                    for c in chunk {
                        if let Some(v) = view.get(c.offset) {
                            pairs.push((*c, v));
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
                .collect()
        } else {
            candidates
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
                .collect()
        };

        let mut reranked = reranked;
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    /// Move the current Hot segment into the `sealing_plain` queue and create a
    /// fresh Hot segment. The actual HNSW build happens later in the background
    /// optimizer, so this method is fast enough to run on the insert hot path.
    pub fn seal_hot(&mut self, _vectors: &VectorStore) -> crate::Result<()> {
        // Re-check capacity: another thread may have sealed while we waited for
        // the holder write lock.
        {
            let hot = self.hot.read();
            if hot.point_count() < self.config.tier.hot_capacity {
                return Ok(());
            }
        }

        // Fast swap: capture the full plain Hot segment and replace it with a
        // fresh one. No HNSW construction happens here.
        let old_hot = {
            let mut hot = self.hot.write();
            if hot.point_count() < self.config.tier.hot_capacity {
                return Ok(());
            }
            std::mem::replace(&mut *hot, HotSegment::new(&self.config)?)
        };
        self.sealing_plain.push(Arc::new(RwLock::new(old_hot)));
        Ok(())
    }

    /// Read full vectors for a list of offsets. Used by the background optimizer
    /// to build persisted segments while NOT holding the segment holder write lock.
    pub fn read_vectors_for_offsets(
        &self,
        offsets: &[PointOffset],
        vectors: &VectorStore,
    ) -> Vec<(PointOffset, Vec<f32>)> {
        let view = vectors.read_view();
        let mut out: Vec<(PointOffset, Vec<f32>)> = Vec::with_capacity(offsets.len());
        for &offset in offsets {
            if let Some(v) = view.get(offset) {
                out.push((offset, Vec::from(v)));
            }
        }
        out
    }

    /// If total warm records exceed the warm capacity, merge all warm segments
    /// into a single Cold segment.
    pub(crate) fn compact_warm(&mut self, vectors: &VectorStore) -> crate::Result<()> {
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
            let cold = ColdSegment::from_records(&path, &records, self.config.tier.cold_quantizer)?;
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
                    payload: None,
                };
                records.push((offset, record));
            }
        }
        Ok(records)
    }

    pub fn point_count(&self) -> usize {
        self.hot.read().point_count()
            + self
                .sealing_plain
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + self
                .sealed_hot
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + self
                .warm
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + self
                .cold
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
    }

    pub fn memory_bytes(&self) -> usize {
        self.hot.read().memory_bytes()
            + self
                .sealing_plain
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + self
                .sealed_hot
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + self
                .warm
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + self
                .cold
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
    }

    pub fn flush(&self) -> crate::Result<()> {
        (self.hot.read().flusher())()?;
        for plain in &self.sealing_plain {
            (plain.read().flusher())()?;
        }
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
        for plain in &self.sealing_plain {
            for &offset in plain.read().offsets() {
                candidate_offsets.push(offset);
            }
        }
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
            let Some(meta_rec) = meta.get(offset)? else {
                continue;
            };
            if access_score(&meta_rec, now, half_life) < threshold {
                continue;
            }
            let view = vectors.read_view();
            let Some(vec) = view.get(offset) else {
                continue;
            };
            let embedding = Arc::from(Vec::from(vec));
            drop(view);
            let record = meta_rec.with_embedding(embedding).with_tier(Tier::Hot);
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
