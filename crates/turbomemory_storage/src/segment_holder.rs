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
use arc_swap::ArcSwap;
use parking_lot::RwLock;
use rayon::prelude::*;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

pub const SEALED_HOT_DIR: &str = "sealed_hot";

/// Immutable, atomically-published view of the searchable segment set.
///
/// Searches load one of these snapshots and never need to hold the
/// `SegmentHolder` lock while segment scoring, candidate merging, or reranking
/// runs. Segment mutation still happens under the holder write lock, then
/// publishes a fresh snapshot with an atomic pointer swap.
pub struct SegmentSnapshot {
    config: StoreConfig,
    segments: Vec<Arc<RwLock<dyn VectorSegment>>>,
}

impl SegmentSnapshot {
    fn empty(config: StoreConfig) -> Self {
        Self {
            config,
            segments: Vec::new(),
        }
    }

    fn point_count(&self) -> usize {
        self.segments
            .iter()
            .map(|s| s.read().point_count())
            .sum::<usize>()
    }

    pub(crate) fn search(
        &self,
        query: &[f32],
        top_k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<ScoredPoint>> {
        self.search_gpu(query, top_k, ef, vectors, allowed_offsets, None)
    }

    /// GPU-accelerated search with optional backend.
    pub(crate) fn search_gpu(
        &self,
        query: &[f32],
        top_k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
        gpu: Option<&Arc<dyn turbomemory_gpu::GpuBackend>>,
    ) -> crate::Result<Vec<ScoredPoint>> {
        // Use Qdrant-style ef semantics: floor the per-segment candidate pool at
        // the caller-provided `ef` (or the configured search list size), then
        // apply an over-fetch multiplier that grows with filter strictness and
        // the number of segments.
        let base_ef = ef.unwrap_or(self.config.search_list_size);
        let base_multiplier = if allowed_offsets.is_some() { 8 } else { 4 };
        let segment_count = self.segments.len().max(1);
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
        // The automatic multiplier is capped to bound rerank cost, but an
        // explicit caller `ef` must always be honored as the pool floor. Apply
        // the cap to the multiplier-derived width first, then floor at base_ef,
        // so a large caller-provided ef widens the HNSW beam instead of being
        // silently clamped back down to top_k*48.
        let pool_k = top_k
            .saturating_mul(multiplier)
            .min(top_k.saturating_mul(48))
            .max(base_ef);

        let segments = self.segments.clone();
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
        // Use GPU for large rerank pools if available.
        let reranked = if let Some(backend) = gpu {
            if turbomemory_gpu::is_gpu_accelerated(backend) && candidates.len() >= 256 {
                match gpu_rerank_candidates(query, &candidates, vectors, backend) {
                    Ok(results) => results,
                    Err(e) => {
                        log::warn!("GPU rerank failed ({}), falling back to CPU", e);
                        cpu_rerank_candidates(query, &candidates, vectors)
                    }
                }
            } else {
                cpu_rerank_candidates(query, &candidates, vectors)
            }
        } else {
            cpu_rerank_candidates(query, &candidates, vectors)
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

    /// Batched search over the snapshot for M queries at once. Per-query HNSW
    /// traversal runs on CPU (where it's fast for single queries); the
    /// candidate rerank is batched into ONE `gemm` call via
    /// [`gpu_rerank_candidates_batch`] when a GPU backend is supplied, which is
    /// where the GPU genuinely beats CPU. Falls back to per-query CPU rerank
    /// when no GPU is available.
    ///
    /// Returns `m` result lists, each truncated to `top_k`, sorted by score desc.
    pub(crate) fn search_gpu_batch(
        &self,
        queries: &[&[f32]],
        top_k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
        gpu: Option<&Arc<dyn turbomemory_gpu::GpuBackend>>,
    ) -> crate::Result<Vec<Vec<ScoredPoint>>> {
        let m = queries.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        // Single query: defer to the established single-query path.
        if m == 1 {
            let r = self.search_gpu(queries[0], top_k, ef, vectors, allowed_offsets, gpu)?;
            return Ok(vec![r]);
        }

        // Compute the per-segment candidate pool width once (same logic as
        // search_gpu, independent of the query).
        let base_ef = ef.unwrap_or(self.config.search_list_size);
        let base_multiplier = if allowed_offsets.is_some() { 8 } else { 4 };
        let segment_count = self.segments.len().max(1);
        let selectivity = allowed_offsets
            .map(|b| {
                let total = self.point_count().max(1);
                (b.len() as usize as f32) / (total as f32)
            })
            .unwrap_or(1.0f32);
        let multiplier = if selectivity < 0.01 {
            base_multiplier
        } else {
            let selectivity_factor = (1.0f32 / selectivity.sqrt()).clamp(1.0f32, 16.0f32);
            let segment_factor =
                (1.0f32 + (segment_count.saturating_sub(1)) as f32 * 0.25f32).clamp(1.0f32, 2.5f32);
            (base_multiplier as f32 * selectivity_factor * segment_factor) as usize
        };
        let pool_k = top_k
            .saturating_mul(multiplier)
            .min(top_k.saturating_mul(48))
            .max(base_ef);

        let segments = self.segments.clone();

        // Per-query: run CPU HNSW search across all segments and merge, giving
        // each query its own candidate list. Queries are independent, so this
        // parallelizes across queries (and within, across segments).
        let per_query_candidates: Vec<Vec<ScoredPoint>> = queries
            .iter()
            .map(|q| -> crate::Result<Vec<ScoredPoint>> {
                let lists: Vec<Vec<ScoredPoint>> = if segments.len() <= 1 {
                    segments
                        .iter()
                        .map(|seg| seg.read().search(q, pool_k, vectors, allowed_offsets))
                        .collect::<crate::Result<Vec<_>>>()?
                } else {
                    segments
                        .par_iter()
                        .map(|seg| seg.read().search(q, pool_k, vectors, allowed_offsets))
                        .collect::<crate::Result<Vec<_>>>()?
                };
                Ok(if lists.len() >= 4 && pool_k >= 64 {
                    kway_merge_topk(&lists, pool_k)
                } else {
                    merge_candidates(lists, pool_k)
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;

        // Batched rerank. GPU path needs a meaningful candidate union to be
        // worth the upload; otherwise fall back to per-query CPU rerank.
        let use_gpu = gpu
            .map(turbomemory_gpu::is_gpu_accelerated)
            .unwrap_or(false);
        let total_candidates: usize = per_query_candidates.iter().map(|c| c.len()).sum();
        if use_gpu && total_candidates >= 256 {
            if let Some(backend) = gpu {
                match gpu_rerank_candidates_batch(
                    queries,
                    &per_query_candidates,
                    top_k,
                    vectors,
                    backend,
                ) {
                    Ok(results) => return Ok(results),
                    Err(e) => {
                        log::warn!("GPU batch rerank failed ({}), falling back to CPU", e);
                    }
                }
            }
        }

        // CPU fallback: per-query rerank + truncate.
        let mut results: Vec<Vec<ScoredPoint>> = Vec::with_capacity(m);
        for (q, cands) in queries.iter().zip(per_query_candidates.iter()) {
            let mut reranked = cpu_rerank_candidates(q, cands, vectors);
            reranked.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            reranked.truncate(top_k);
            results.push(reranked);
        }
        Ok(results)
    }
}

/// CPU rerank of candidate points with full f32 embeddings.
fn cpu_rerank_candidates(
    query: &[f32],
    candidates: &[ScoredPoint],
    vectors: &VectorStore,
) -> Vec<ScoredPoint> {
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
    reranked
}

/// GPU-accelerated rerank of candidate points with full f32 embeddings.
fn gpu_rerank_candidates(
    query: &[f32],
    candidates: &[ScoredPoint],
    vectors: &VectorStore,
    backend: &Arc<dyn turbomemory_gpu::GpuBackend>,
) -> turbomemory_gpu::Result<Vec<ScoredPoint>> {
    let dim = vectors.dimension();
    let view = vectors.read_view();

    // Collect candidate vectors into a contiguous flat buffer
    let mut flat_vectors: Vec<f32> = Vec::with_capacity(candidates.len() * dim);
    let mut valid_candidates: Vec<ScoredPoint> = Vec::with_capacity(candidates.len());
    for c in candidates {
        if let Some(v) = view.get(c.offset) {
            flat_vectors.extend_from_slice(v);
            valid_candidates.push(*c);
        }
    }
    drop(view);

    if valid_candidates.is_empty() {
        return Ok(Vec::new());
    }

    // Upload to GPU and compute batched cosine similarity
    let device_buf = backend.upload_vectors(&flat_vectors, dim)?;
    let scores = backend.batch_cosine_similarity(query, &device_buf)?;

    // Build reranked points preserving tier info
    let reranked: Vec<ScoredPoint> = valid_candidates
        .into_iter()
        .zip(scores)
        .map(|(c, score)| ScoredPoint {
            offset: c.offset,
            score,
            tier: c.tier,
        })
        .collect();
    Ok(reranked)
}

/// Batched GPU rerank for M queries at once. This is where the GPU genuinely
/// wins over CPU: M queries × N candidate vectors are scored in a SINGLE
/// `gemm` call (via `batch_cosine_similarity_matrix`) instead of M separate
/// `gemv` calls, amortizing kernel-launch and host→device upload overhead.
///
/// To do that with per-query candidate lists (which may differ), we build the
/// UNION of candidate offsets across all queries, upload it once, score every
/// query against the whole union (M × union_size gemm), then each query keeps
/// only its own candidates and truncates to `top_k`. When queries' candidate
/// sets overlap (common for clustered query batches) this also avoids
/// re-uploading/re-scoring the same vectors.
fn gpu_rerank_candidates_batch(
    queries: &[&[f32]],
    per_query_candidates: &[Vec<ScoredPoint>],
    top_k: usize,
    vectors: &VectorStore,
    backend: &Arc<dyn turbomemory_gpu::GpuBackend>,
) -> turbomemory_gpu::Result<Vec<Vec<ScoredPoint>>> {
    use std::collections::HashMap;

    let dim = vectors.dimension();
    let m = queries.len();
    debug_assert_eq!(per_query_candidates.len(), m);

    // 1. Build the union of candidate offsets, deduplicated.
    let mut union_offsets: Vec<PointOffset> = Vec::new();
    let mut offset_to_union_idx: HashMap<PointOffset, usize> = HashMap::new();
    for cands in per_query_candidates {
        for c in cands {
            if offset_to_union_idx
                .insert(c.offset, union_offsets.len())
                .is_some()
            {
                // already present
                continue;
            }
            union_offsets.push(c.offset);
        }
    }
    if union_offsets.is_empty() {
        return Ok((0..m).map(|_| Vec::new()).collect());
    }

    // 2. Gather union vectors into one flat buffer and upload once.
    let view = vectors.read_view();
    let n = union_offsets.len();
    let mut flat_vectors: Vec<f32> = Vec::with_capacity(n * dim);
    for &off in &union_offsets {
        if let Some(v) = view.get(off) {
            flat_vectors.extend_from_slice(v);
        } else {
            // missing vector — zero-fill to keep indexing intact
            flat_vectors.extend(std::iter::repeat_n(0.0f32, dim));
        }
    }
    drop(view);

    // 3. Flatten the M queries into one M×dim buffer.
    let mut flat_queries: Vec<f32> = Vec::with_capacity(m * dim);
    for q in queries {
        flat_queries.extend_from_slice(q);
    }

    // 4. ONE gemm: M queries × N union vectors → M·N row-major scores.
    let device_buf = backend.upload_vectors(&flat_vectors, dim)?;
    let scores = backend.batch_cosine_similarity_matrix(&flat_queries, m, &device_buf)?;

    // 5. For each query, select its own candidates' scores, sort, truncate.
    let mut results: Vec<Vec<ScoredPoint>> = Vec::with_capacity(m);
    for (qi, cands) in per_query_candidates.iter().enumerate() {
        let mut scored: Vec<ScoredPoint> = cands
            .iter()
            .map(|c| {
                let uidx = offset_to_union_idx[&c.offset];
                ScoredPoint {
                    offset: c.offset,
                    score: scores[qi * n + uidx],
                    tier: c.tier,
                }
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
        results.push(scored);
    }
    Ok(results)
}

/// Mutable, internally-locked set of non-Hot segment lists.
///
/// All list mutations take this lock briefly for the swap itself; expensive
/// work (vector reads, segment builds) happens *outside* the guard so writers
/// never block searches (which read the published `SegmentSnapshot`) for long.
#[derive(Default)]
struct SegmentLists {
    /// Plain segments that have been swapped out of the Hot tier and are waiting
    /// for the background optimizer to build their HNSW replacements. They remain
    /// searchable as exact segments until the build completes.
    sealing_plain: Vec<Arc<RwLock<HotSegment>>>,
    /// Plain segments currently being converted by an optimizer worker. These
    /// are no longer pending work, but they must remain visible to search until
    /// their replacement is published.
    building_plain: Vec<Arc<RwLock<HotSegment>>>,
    sealed_hot: Vec<Arc<RwLock<dyn VectorSegment>>>,
    warm: Vec<Arc<RwLock<dyn VectorSegment>>>,
    cold: Vec<Arc<RwLock<dyn VectorSegment>>>,
}

/// Owns Hot/Warm/Cold segments for a single collection.
///
/// Internally synchronized: mutation methods take `&self`. The mutable Hot
/// segment has its own `RwLock`; all other segment lists live behind a single
/// `RwLock<SegmentLists>`. Searches never touch these locks — they read the
/// atomically-published `SegmentSnapshot` instead.
pub struct SegmentHolder {
    config: StoreConfig,
    hot: Arc<RwLock<HotSegment>>,
    lists: RwLock<SegmentLists>,
    next_segment_id: AtomicU64,
    base_path: PathBuf,
    snapshot: Arc<ArcSwap<SegmentSnapshot>>,
}

impl SegmentHolder {
    pub fn new(config: StoreConfig, base_path: impl AsRef<Path>) -> crate::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;
        let holder = Self {
            hot: Arc::new(RwLock::new(HotSegment::new(&config)?)),
            lists: RwLock::new(SegmentLists::default()),
            next_segment_id: AtomicU64::new(0),
            base_path,
            snapshot: Arc::new(ArcSwap::from_pointee(SegmentSnapshot::empty(
                config.clone(),
            ))),
            config,
        };
        holder.publish_snapshot();
        Ok(holder)
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
        let holder = Self::new(config, base_path)?;
        for (offset, rec) in records {
            if !sealed_offsets.contains(offset) && holder.insert(*offset, rec, vectors)? {
                holder.seal_hot(vectors)?;
            }
        }
        Ok(holder)
    }

    pub(crate) fn snapshot_handle(&self) -> Arc<ArcSwap<SegmentSnapshot>> {
        Arc::clone(&self.snapshot)
    }

    pub(crate) fn publish_snapshot(&self) {
        self.snapshot.store(Arc::new(self.build_snapshot()));
    }

    fn build_snapshot(&self) -> SegmentSnapshot {
        SegmentSnapshot {
            config: self.config.clone(),
            segments: self.searchable_segments(),
        }
    }

    fn searchable_segments(&self) -> Vec<Arc<RwLock<dyn VectorSegment>>> {
        let lists = self.lists.read();
        let mut segments: Vec<Arc<RwLock<dyn VectorSegment>>> = Vec::with_capacity(
            2 + lists.sealing_plain.len()
                + lists.building_plain.len()
                + lists.sealed_hot.len()
                + lists.warm.len()
                + lists.cold.len(),
        );
        segments.push(self.hot.clone());
        for plain in &lists.sealing_plain {
            segments.push(plain.clone());
        }
        for plain in &lists.building_plain {
            segments.push(plain.clone());
        }
        for sealed in &lists.sealed_hot {
            segments.push(sealed.clone());
        }
        for warm in &lists.warm {
            segments.push(warm.clone());
        }
        for cold in &lists.cold {
            segments.push(cold.clone());
        }
        segments
    }

    pub(crate) fn segment_path(&self, tier: Tier) -> PathBuf {
        let id = self.next_segment_id.fetch_add(1, Ordering::Relaxed);
        self.base_path
            .join(tier.name())
            .join(format!("segment_{id}"))
    }

    pub(crate) fn sealed_hot_path(&self) -> PathBuf {
        let id = self.next_segment_id.fetch_add(1, Ordering::Relaxed);
        self.base_path
            .join(SEALED_HOT_DIR)
            .join(format!("segment_{id}"))
    }

    pub(crate) fn pop_sealing_plain(&self) -> Option<Arc<RwLock<HotSegment>>> {
        let mut lists = self.lists.write();
        let plain = lists.sealing_plain.pop()?;
        lists.building_plain.push(plain.clone());
        Some(plain)
    }

    pub(crate) fn push_sealing_plain(&self, plain: Arc<RwLock<HotSegment>>) {
        {
            let mut lists = self.lists.write();
            lists.building_plain.retain(|p| !Arc::ptr_eq(p, &plain));
            lists.sealing_plain.push(plain);
        }
        self.publish_snapshot();
    }

    pub(crate) fn remove_sealing_plain(&self, target: &Arc<RwLock<HotSegment>>) {
        let mut lists = self.lists.write();
        lists.sealing_plain.retain(|p| !Arc::ptr_eq(p, target));
        lists.building_plain.retain(|p| !Arc::ptr_eq(p, target));
    }

    pub(crate) fn push_sealed_hot(&self, segment: SealedHotSegment) {
        self.add_sealed_hot(segment);
    }

    pub(crate) fn push_warm(&self, segment: WarmSegment) {
        self.add_warm(segment);
    }

    pub fn add_sealed_hot(&self, segment: SealedHotSegment) {
        self.lists
            .write()
            .sealed_hot
            .push(Arc::new(RwLock::new(segment)) as Arc<RwLock<dyn VectorSegment>>);
        self.publish_snapshot();
    }

    pub fn add_warm(&self, segment: WarmSegment) {
        self.lists
            .write()
            .warm
            .push(Arc::new(RwLock::new(segment)) as Arc<RwLock<dyn VectorSegment>>);
        self.publish_snapshot();
    }

    pub fn add_cold(&self, segment: ColdSegment) {
        self.lists
            .write()
            .cold
            .push(Arc::new(RwLock::new(segment)) as Arc<RwLock<dyn VectorSegment>>);
        self.publish_snapshot();
    }

    #[allow(dead_code)]
    pub(crate) fn sealed_hot_count(&self) -> usize {
        self.lists.read().sealed_hot.len()
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
        // Snapshot the sealed-hot list under the lock, then release it before
        // scoring so we never hold the lists lock while reading segment counts.
        let sealed_snapshot: Vec<Arc<RwLock<dyn VectorSegment>>> = {
            let lists = self.lists.read();
            if lists.sealed_hot.len() < threshold {
                return None;
            }
            lists.sealed_hot.clone()
        };
        // Sort by point count ascending so small segments are merged first.
        let mut sorted = sealed_snapshot;
        sorted.sort_by_key(|s| s.read().point_count());

        let hard_cap = max_records.saturating_mul(2).max(max_records);
        let mut candidates = Vec::with_capacity(sorted.len());
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
        &self,
        targets: &[Arc<RwLock<dyn VectorSegment>>],
    ) -> usize {
        let mut removed = 0usize;
        self.lists.write().sealed_hot.retain(|seg| {
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
    /// Callers should call [`SegmentHolder::seal_hot`] when this returns `true`.
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
        self.snapshot
            .load_full()
            .search(query, top_k, ef, vectors, allowed_offsets)
    }

    /// Batched search for M queries. See `SegmentSnapshot::search_gpu_batch`.
    #[allow(dead_code)]
    pub(crate) fn search_gpu_batch(
        &self,
        queries: &[&[f32]],
        top_k: usize,
        ef: Option<usize>,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
        gpu: Option<&Arc<dyn turbomemory_gpu::GpuBackend>>,
    ) -> crate::Result<Vec<Vec<ScoredPoint>>> {
        self.snapshot.load_full().search_gpu_batch(
            queries,
            top_k,
            ef,
            vectors,
            allowed_offsets,
            gpu,
        )
    }

    /// Move the current Hot segment into the `sealing_plain` queue and create a
    /// fresh Hot segment. The actual HNSW build happens later in the background
    /// optimizer, so this method is fast enough to run on the insert hot path.
    pub fn seal_hot(&self, _vectors: &VectorStore) -> crate::Result<()> {
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
        self.lists
            .write()
            .sealing_plain
            .push(Arc::new(RwLock::new(old_hot)));
        self.publish_snapshot();
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
    ///
    /// The expensive Cold build runs *outside* the lists lock. We capture the
    /// exact set of warm segments to compact, build the Cold segment from their
    /// offsets, then under the write lock remove only those specific warm Arcs
    /// (by pointer identity) and install the Cold segment. Warm segments added
    /// concurrently during the build are left untouched.
    pub(crate) fn compact_warm(&self, vectors: &VectorStore) -> crate::Result<()> {
        // 1. Snapshot the warm segments to compact and release the lock.
        let warm_targets: Vec<Arc<RwLock<dyn VectorSegment>>> = {
            let lists = self.lists.read();
            let total_warm: usize = lists.warm.iter().map(|s| s.read().point_count()).sum();
            if total_warm <= self.config.tier.warm_capacity || lists.warm.is_empty() {
                return Ok(());
            }
            lists.warm.clone()
        };

        // 2. Collect the union of offsets and build the Cold segment without
        //    holding the lists lock.
        let mut offsets = Vec::new();
        for warm in &warm_targets {
            offsets.extend_from_slice(warm.read().offsets());
        }
        offsets.sort_unstable();
        offsets.dedup();
        let records = self.build_records(&offsets, vectors)?;
        let cold = if records.is_empty() {
            None
        } else {
            let path = self.segment_path(Tier::Cold);
            Some(ColdSegment::from_records(
                &path,
                &records,
                self.config.tier.cold_quantizer,
            )?)
        };

        // 3. Install the Cold segment and remove exactly the compacted warm
        //    segments by pointer identity.
        {
            let mut lists = self.lists.write();
            if let Some(cold) = cold {
                lists
                    .cold
                    .push(Arc::new(RwLock::new(cold)) as Arc<RwLock<dyn VectorSegment>>);
            }
            lists
                .warm
                .retain(|seg| !warm_targets.iter().any(|t| Arc::ptr_eq(t, seg)));
        }
        self.publish_snapshot();
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
                    scope: None,
                };
                records.push((offset, record));
            }
        }
        Ok(records)
    }

    pub fn point_count(&self) -> usize {
        let lists = self.lists.read();
        self.hot.read().point_count()
            + lists
                .sealing_plain
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + lists
                .building_plain
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + lists
                .sealed_hot
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + lists
                .warm
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
            + lists
                .cold
                .iter()
                .map(|s| s.read().point_count())
                .sum::<usize>()
    }

    pub fn memory_bytes(&self) -> usize {
        let lists = self.lists.read();
        self.hot.read().memory_bytes()
            + lists
                .sealing_plain
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + lists
                .building_plain
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + lists
                .sealed_hot
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + lists
                .warm
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
            + lists
                .cold
                .iter()
                .map(|s| s.read().memory_bytes())
                .sum::<usize>()
    }

    pub fn flush(&self) -> crate::Result<()> {
        (self.hot.read().flusher())()?;
        let lists = self.lists.read();
        for plain in &lists.sealing_plain {
            (plain.read().flusher())()?;
        }
        for plain in &lists.building_plain {
            (plain.read().flusher())()?;
        }
        for sealed in &lists.sealed_hot {
            (sealed.read().flusher())()?;
        }
        for warm in &lists.warm {
            (warm.read().flusher())()?;
        }
        for cold in &lists.cold {
            (cold.read().flusher())()?;
        }
        Ok(())
    }

    pub fn trigger_consolidation(
        &self,
        meta: &MetadataStore,
        vectors: &VectorStore,
    ) -> crate::Result<(usize, usize, usize)> {
        let mut sealed = 0usize;
        let mut compacted = 0usize;
        if self.hot.read().point_count() >= self.config.tier.hot_capacity {
            self.seal_hot(vectors)?;
            sealed += 1;
        }
        let total_warm: usize = {
            let lists = self.lists.read();
            lists.warm.iter().map(|s| s.read().point_count()).sum()
        };
        if total_warm > self.config.tier.warm_capacity {
            self.compact_warm(vectors)?;
            compacted += 1;
        }
        let promoted = self.promote_hot(meta, vectors)?;
        Ok((sealed, compacted, promoted))
    }

    /// Promote frequently-accessed records from sealed/warm/cold tiers back
    /// into the mutable Hot segment.
    pub fn promote_hot(&self, meta: &MetadataStore, vectors: &VectorStore) -> crate::Result<usize> {
        let now = crate::engine::now_secs();
        let threshold = self.config.tier.hot_promote_threshold;
        let half_life = self.config.tier.recency_half_life_secs.max(1);

        // Snapshot candidate offsets under the lists lock, then release it so
        // the (potentially slow) per-offset promotion runs lock-free.
        let mut candidate_offsets = Vec::new();
        {
            let lists = self.lists.read();
            for plain in &lists.sealing_plain {
                candidate_offsets.extend_from_slice(plain.read().offsets());
            }
            for plain in &lists.building_plain {
                candidate_offsets.extend_from_slice(plain.read().offsets());
            }
            for sealed in &lists.sealed_hot {
                candidate_offsets.extend_from_slice(sealed.read().offsets());
            }
            for warm in &lists.warm {
                candidate_offsets.extend_from_slice(warm.read().offsets());
            }
            for cold in &lists.cold {
                candidate_offsets.extend_from_slice(cold.read().offsets());
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
        if promoted > 0 {
            self.publish_snapshot();
        }
        Ok(promoted)
    }
}

fn access_score(meta: &MetaRecord, now: u64, half_life: u64) -> f64 {
    let age = now.saturating_sub(meta.last_accessed).max(1);
    let recency = 2.0f64.powf(-(age as f64) / half_life as f64);
    meta.access_count as f64 * recency
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn make_record(offset: PointOffset, dim: usize, idx: usize) -> (PointOffset, Record) {
        let mut v = vec![0.0f32; dim];
        v[idx % dim] = 1.0;
        (
            offset,
            Record {
                id: format!("id-{idx}"),
                text: String::new(),
                embedding: Arc::from(v),
                importance: 1.0,
                concepts: Vec::new(),
                created_at: 0,
                insert_seq: idx as u64,
                access_count: 0,
                last_accessed: 0,
                tier: Tier::Hot,
                payload: None,
                scope: None,
            },
        )
    }

    #[test]
    fn snapshot_keeps_building_plain_searchable() {
        let dim = 8;
        let tmp = tempfile::tempdir().unwrap();
        let mut config = StoreConfig::default_for_dimension(dim);
        config.tier.hot_capacity = 2;

        let vectors = VectorStore::new_with_capacity(tmp.path().join("vectors"), dim, 8).unwrap();
        let holder = SegmentHolder::new(config, tmp.path().join("segments")).unwrap();
        let records: Vec<_> = (0..2)
            .map(|i| make_record(i as PointOffset + 1, dim, i))
            .collect();
        for (offset, record) in &records {
            vectors.put(*offset, record.embedding_f32()).unwrap();
            holder.insert(*offset, record, &vectors).unwrap();
        }
        holder.seal_hot(&vectors).unwrap();

        let query = records[0].1.embedding_f32().to_vec();
        let snapshot = holder.snapshot_handle();
        let before = snapshot
            .load_full()
            .search(&query, 1, None, &vectors, None)
            .unwrap();
        assert_eq!(before[0].offset, records[0].0);

        let building = holder.pop_sealing_plain().unwrap();
        // Simulate an unrelated segment-list publish while the optimizer is
        // still building a replacement for the popped plain segment.
        holder.publish_snapshot();

        let during = snapshot
            .load_full()
            .search(&query, 1, None, &vectors, None)
            .unwrap();
        assert_eq!(during[0].offset, records[0].0);

        holder.remove_sealing_plain(&building);
        holder.publish_snapshot();
        let after = snapshot
            .load_full()
            .search(&query, 1, None, &vectors, None)
            .unwrap();
        assert!(after.is_empty());
    }

    /// Hammer the published snapshot from many reader threads while a writer
    /// thread inserts and repeatedly seals the Hot segment. Because searches go
    /// through the lock-free `SegmentSnapshot`, they must never block, panic, or
    /// observe an invalid offset regardless of concurrent seal swaps.
    #[test]
    fn concurrent_search_during_seal_is_consistent() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Barrier;
        use std::thread;

        let dim = 8;
        let tmp = tempfile::tempdir().unwrap();
        let mut config = StoreConfig::default_for_dimension(dim);
        config.tier.hot_capacity = 4;

        let total = 200usize;
        let vectors = Arc::new(
            VectorStore::new_with_capacity(tmp.path().join("vectors"), dim, total).unwrap(),
        );
        let holder = Arc::new(SegmentHolder::new(config, tmp.path().join("segments")).unwrap());
        let snapshot = holder.snapshot_handle();

        let stop = Arc::new(AtomicBool::new(false));
        // Release all threads together so readers are guaranteed at least one
        // search before the writer can finish and set `stop`.
        let barrier = Arc::new(Barrier::new(5));

        // Writer: insert records and seal whenever the Hot segment fills.
        let writer = {
            let holder = Arc::clone(&holder);
            let vectors = Arc::clone(&vectors);
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                for i in 0..total {
                    let (offset, record) = make_record(i as PointOffset + 1, dim, i);
                    vectors.put(offset, record.embedding_f32()).unwrap();
                    if holder.insert(offset, &record, &vectors).unwrap() {
                        holder.seal_hot(&vectors).unwrap();
                    }
                }
            })
        };

        // Readers: continuously search the published snapshot.
        let readers: Vec<_> = (0..4)
            .map(|r| {
                let snapshot = Arc::clone(&snapshot);
                let vectors = Arc::clone(&vectors);
                let stop = Arc::clone(&stop);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let query = {
                        let mut q = vec![0.0f32; dim];
                        q[r % dim] = 1.0;
                        q
                    };
                    barrier.wait();
                    let mut iters = 0u64;
                    // Guarantee at least one search, then keep going until the
                    // writer signals completion.
                    loop {
                        let results = snapshot
                            .load_full()
                            .search(&query, 5, None, &vectors, None)
                            .expect("search must not fail during seal");
                        for sp in &results {
                            assert!(
                                sp.offset >= 1 && sp.offset <= total as PointOffset,
                                "search returned an invalid offset {}",
                                sp.offset
                            );
                        }
                        iters += 1;
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                    }
                    iters
                })
            })
            .collect();

        writer.join().unwrap();
        stop.store(true, Ordering::Relaxed);
        for reader in readers {
            let iters = reader.join().unwrap();
            assert!(iters > 0, "reader thread should have run at least once");
        }

        // After all inserts the holder must hold every record across its tiers.
        assert_eq!(holder.point_count(), total);
    }
}
