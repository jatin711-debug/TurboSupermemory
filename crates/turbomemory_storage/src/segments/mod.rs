//! Tiered vector segments.

pub mod cold;
pub mod gpu_hnsw_index;
pub mod hot;
pub mod mmap_array;
pub mod sealed_hot;
pub mod usearch_index;
pub mod vector_index;
pub mod warm;

use crate::config::{Flusher, Tier};
use crate::record::{PointOffset, Record};
use crate::vector_store::VectorStore;
use crate::StorageError;
use ahash::AHashSet;
use roaring::RoaringBitmap;
use smallvec::SmallVec;
use std::path::Path;
use std::sync::Arc;
use turbomemory_core::{cosine_similarity_batch, validate_dimension};

pub use cold::ColdSegment;
pub use hot::HotSegment;
pub use mmap_array::MmapBuffer;
pub use sealed_hot::SealedHotSegment;
pub use usearch_index::UsearchIndex;
pub use vector_index::{VectorIndex, VectorIndexManifest};
pub use warm::WarmSegment;

pub type Result<T> = std::result::Result<T, StorageError>;

/// Quantized-tier rerank oversampling floor. The quantized shortlist is sized
/// to `max(top_k * rerank_oversample(dim), MIN_RERANK_SHORTLIST)` so the
/// full-f32 rerank can recover true neighbors that quantization noise pushed
/// past `top_k`.
pub const RERANK_OVERSAMPLE: usize = 8;
pub const MIN_RERANK_SHORTLIST: usize = 64;

/// Dimension-aware rerank oversampling for quantized tiers (Warm/Cold).
///
/// At low dimension 8-bit scalar quantization preserves cosine ordering well,
/// so a small shortlist (8x) is plenty. At high dimension the quantization
/// noise grows relative to the tiny cosine gaps between near-orthogonal
/// vectors, so a true top-k neighbor can land well outside the quantized
/// top-k and be lost before the f32 rerank. Scaling the shortlist with
/// dimension recovers them: empirically ~16x at 768-dim lifts recall from
/// ~77% to ~92% at n=10k without touching the HNSW beam width.
///
/// For dimensions up to 256, sqrt scaling is sufficient. For high dimension
/// (384+) we switch to linear scaling because quantization noise accumulates
/// across coordinates and the true-neighbor gap becomes comparable to the
/// noise floor — a much larger shortlist is needed to ensure true neighbors
/// survive the quantized candidate selection.
///
/// Examples: 8x at 128-dim, ~14x at 384-dim, ~48x at 768-dim, ~128x at 2048-dim.
pub fn rerank_oversample(dim: usize) -> usize {
    let base = RERANK_OVERSAMPLE as f64; // 8
    let factor = if dim <= 256 {
        // Sub-linear (sqrt) scaling for low-dim: noise is manageable.
        base * ((dim as f64) / 128.0).sqrt()
    } else {
        // Linear scaling for high-dim: quantization noise dominates.
        // factor = 8 * (dim / 128) = dim / 16.
        base * (dim as f64) / 128.0
    };
    let raw = factor.round() as usize;
    // Clamp: 8 minimum, 512 maximum. The 512 cap is for very high dimensions
    // (e.g. 4096-dim would request 256x) to avoid reranking thousands of
    // candidates per segment while still being far more aggressive than the
    // old 64x cap that lost true neighbors at 50k x 768.
    raw.clamp(8, 512)
}

/// A scored point returned by a segment search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredPoint {
    pub offset: PointOffset,
    pub score: f32,
    pub tier: Tier,
}

/// Common interface implemented by every tier.
pub trait VectorSegment: Send + Sync {
    fn tier(&self) -> Tier;
    /// Insert a record.  Only the Hot segment is appendable in this design.
    fn insert(&mut self, offset: PointOffset, record: &Record) -> Result<()>;
    /// Search the segment and return scored candidates.  Reranking with the
    /// full f32 embedding is done by the caller using `VectorStore`.
    ///
    /// If `allowed_offsets` is provided, only offsets contained in the bitmap
    /// are returned.  This is used for payload-filtered ANN.
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> Result<Vec<ScoredPoint>>;
    fn point_count(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn flusher(&self) -> Flusher;
    /// On-disk directory for persisted segments.  Hot segments are not persisted
    /// directly, so they return `None`.
    fn segment_path(&self) -> Option<&Path> {
        None
    }
    /// Offsets stored in this segment.  Hot segments return an empty slice
    /// because they do not track a stable offset list.
    fn offsets(&self) -> &[PointOffset] {
        &[]
    }
}

/// Wrapper that makes [`ScoredPoint`] usable in a binary heap.
///
/// Ordering is by score descending; combined with [`std::cmp::Reverse`] this
/// yields a min-heap over scores, i.e. the heap always keeps the top-k highest
/// scoring points.
#[derive(Debug, Clone, Copy)]
struct HeapScored(ScoredPoint);

impl PartialEq for HeapScored {
    fn eq(&self, other: &Self) -> bool {
        self.0.score.to_bits() == other.0.score.to_bits()
    }
}

impl Eq for HeapScored {}

impl PartialOrd for HeapScored {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapScored {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .score
            .partial_cmp(&other.0.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Collect the top-k highest-scoring points from an iterator without sorting
/// the whole set. This turns the O(N log N) Warm/Cold scans into O(N log k).
pub fn top_k_minheap(scored: impl Iterator<Item = ScoredPoint>, k: usize) -> Vec<ScoredPoint> {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    if k == 0 {
        return Vec::new();
    }

    let mut heap: BinaryHeap<Reverse<HeapScored>> = BinaryHeap::with_capacity(k);
    for point in scored {
        if heap.len() < k {
            heap.push(Reverse(HeapScored(point)));
        } else if let Some(Reverse(min)) = heap.peek() {
            if point.score > min.0.score {
                heap.pop();
                heap.push(Reverse(HeapScored(point)));
            }
        }
    }

    let mut out: Vec<ScoredPoint> = heap.into_sorted_vec().into_iter().map(|r| r.0 .0).collect();
    // `into_sorted_vec` produces ascending order; reverse to get highest first.
    out.reverse();
    out
}

/// Exact brute-force search over a specific set of offsets.
///
/// This is used by index implementations (e.g. `UsearchIndex`) as a fallback
/// for very selective filters where approximate search would be unreliable.
pub fn exact_search_over_offsets(
    query: &[f32],
    top_k: usize,
    vectors: &VectorStore,
    offsets: &[PointOffset],
    tier: Tier,
) -> crate::Result<Vec<ScoredPoint>> {
    exact_search_over_offsets_gpu(query, top_k, vectors, offsets, tier, None)
}

/// GPU-accelerated exact brute-force search over a specific set of offsets.
///
/// If `gpu` is provided and the segment is large enough to justify GPU
/// transfer overhead, vectors are uploaded to GPU and scored with cuBLAS.
/// Otherwise falls back to CPU SIMD batch scoring.
pub fn exact_search_over_offsets_gpu(
    query: &[f32],
    top_k: usize,
    vectors: &VectorStore,
    offsets: &[PointOffset],
    tier: Tier,
    gpu: Option<&Arc<dyn turbomemory_gpu::GpuBackend>>,
) -> crate::Result<Vec<ScoredPoint>> {
    validate_dimension(query, vectors.dimension())?;

    // Only use GPU for sufficiently large segments to amortize transfer overhead.
    // Threshold: ~1,000 vectors at 768-dim = ~3 MiB of data.
    const GPU_THRESHOLD: usize = 1024;
    let use_gpu = gpu.map(|g| {
        turbomemory_gpu::is_gpu_accelerated(g) && offsets.len() >= GPU_THRESHOLD
    }).unwrap_or(false);

    if use_gpu {
        if let Some(backend) = gpu {
            match gpu_exact_search(query, top_k, vectors, offsets, tier, backend) {
                Ok(results) => return Ok(results),
                Err(e) => {
                    log::warn!("GPU exact search failed ({}), falling back to CPU", e);
                }
            }
        }
    }

    // CPU fallback path
    let view = vectors.read_view();

    const CHUNK: usize = 64;
    let mut scored: Vec<ScoredPoint> = Vec::with_capacity(offsets.len());
    for chunk in offsets.chunks(CHUNK) {
        let mut pairs = Vec::with_capacity(chunk.len());
        for &offset in chunk {
            if let Some(v) = view.get(offset) {
                pairs.push((offset, v));
            }
        }
        let refs: Vec<&[f32]> = pairs.iter().map(|(_, v)| *v).collect();
        let scores = cosine_similarity_batch(query, &refs);
        for ((offset, _), score) in pairs.into_iter().zip(scores) {
            scored.push(ScoredPoint {
                offset,
                score,
                tier,
            });
        }
    }
    drop(view);

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    Ok(scored)
}

/// GPU-accelerated exact search using cuBLAS batched dot product.
fn gpu_exact_search(
    query: &[f32],
    top_k: usize,
    vectors: &VectorStore,
    offsets: &[PointOffset],
    tier: Tier,
    backend: &Arc<dyn turbomemory_gpu::GpuBackend>,
) -> turbomemory_gpu::Result<Vec<ScoredPoint>> {
    let dim = vectors.dimension();
    let view = vectors.read_view();

    // Collect vectors into a contiguous flat buffer for GPU upload
    let mut flat_vectors: Vec<f32> = Vec::with_capacity(offsets.len() * dim);
    let mut valid_offsets: Vec<PointOffset> = Vec::with_capacity(offsets.len());
    for &offset in offsets {
        if let Some(v) = view.get(offset) {
            flat_vectors.extend_from_slice(v);
            valid_offsets.push(offset);
        }
    }
    drop(view);

    if valid_offsets.is_empty() {
        return Ok(Vec::new());
    }

    // Upload to GPU and compute batched cosine similarity
    let device_buf = backend.upload_vectors(&flat_vectors, dim)?;
    let scores = backend.batch_cosine_similarity(query, &device_buf)?;

    // Build scored points
    let mut scored: Vec<ScoredPoint> = valid_offsets
        .into_iter()
        .zip(scores)
        .map(|(offset, score)| ScoredPoint { offset, score, tier })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(top_k);
    Ok(scored)
}

/// Merge candidate lists from multiple segments, preserving the highest scores.
/// If the same offset appears in multiple lists (e.g. after promotion), keep
/// the highest score and drop the duplicates.
pub fn merge_candidates(lists: Vec<Vec<ScoredPoint>>, top_k: usize) -> Vec<ScoredPoint> {
    let mut merged: SmallVec<[ScoredPoint; 64]> = SmallVec::with_capacity(top_k * lists.len());
    for list in lists {
        merged.extend(list);
    }
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate by offset, keeping the first (highest-scoring) occurrence.
    let mut seen = AHashSet::with_capacity(merged.len());
    let mut deduped: SmallVec<[ScoredPoint; 64]> = SmallVec::with_capacity(top_k);
    for candidate in merged {
        if seen.insert(candidate.offset) {
            deduped.push(candidate);
            if deduped.len() >= top_k {
                break;
            }
        }
    }
    deduped.into_vec()
}

/// K-way heap merge of sorted per-segment candidate lists.
///
/// Each input list is assumed to be sorted by descending score.  Returns the
/// top-`k` unique offsets across all lists.  This avoids the `O(N log N)` full
/// sort in `merge_candidates` when many segments are involved.
pub fn kway_merge_topk(lists: &[Vec<ScoredPoint>], k: usize) -> Vec<ScoredPoint> {
    if k == 0 {
        return Vec::new();
    }
    if lists.is_empty() {
        return Vec::new();
    }
    if lists.len() == 1 {
        let mut v = lists[0].clone();
        v.truncate(k);
        return v;
    }

    use std::collections::BinaryHeap;

    // Heap item: (reverse score so max-heap, list index, element index).
    struct Item {
        score: f32,
        list: usize,
        idx: usize,
    }
    impl PartialEq for Item {
        fn eq(&self, other: &Self) -> bool {
            self.score.to_bits() == other.score.to_bits()
        }
    }
    impl Eq for Item {}
    impl PartialOrd for Item {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Item {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.score
                .partial_cmp(&other.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }
    }

    let mut heap: BinaryHeap<Item> = BinaryHeap::with_capacity(lists.len());
    for (i, list) in lists.iter().enumerate() {
        if !list.is_empty() {
            heap.push(Item {
                score: list[0].score,
                list: i,
                idx: 0,
            });
        }
    }

    let mut seen = AHashSet::with_capacity(k.min(1024));
    let mut out = Vec::with_capacity(k);
    while let Some(item) = heap.pop() {
        let candidate = lists[item.list][item.idx];
        if seen.insert(candidate.offset) {
            out.push(candidate);
            if out.len() >= k {
                break;
            }
        }
        let next_idx = item.idx + 1;
        if next_idx < lists[item.list].len() {
            heap.push(Item {
                score: lists[item.list][next_idx].score,
                list: item.list,
                idx: next_idx,
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::rerank_oversample;

    #[test]
    fn rerank_oversample_scales_with_dimension() {
        // Low dim: baseline 8x.
        assert_eq!(rerank_oversample(64), 8);
        assert_eq!(rerank_oversample(128), 8);
        // Mid dim grows sub-linearly.
        let d384 = rerank_oversample(384);
        assert!((9..32).contains(&d384), "384-dim oversample = {d384}");
        // 768-dim (the regression case): linear scaling gives 48x.
        let d768 = rerank_oversample(768);
        assert!((40..=60).contains(&d768), "768-dim oversample = {d768}");
        // Very high dim clamps at 512 (updated cap).
        assert_eq!(rerank_oversample(8192), 512);
    }
}
