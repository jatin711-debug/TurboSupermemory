//! Tiered vector segments.

pub mod cold;
pub mod hot;
pub mod mmap_array;
pub mod sealed_hot;
pub mod warm;

use crate::config::{Flusher, Tier};
use crate::record::{PointOffset, Record};
use crate::vector_store::VectorStore;
use crate::StorageError;
use ahash::AHashSet;
use roaring::RoaringBitmap;
use smallvec::SmallVec;

pub use cold::ColdSegment;
pub use hot::HotSegment;
pub use mmap_array::MmapBuffer;
pub use sealed_hot::SealedHotSegment;
pub use warm::WarmSegment;

pub type Result<T> = std::result::Result<T, StorageError>;

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
pub fn top_k_minheap(
    scored: impl Iterator<Item = ScoredPoint>,
    k: usize,
) -> Vec<ScoredPoint> {
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

    let mut out: Vec<ScoredPoint> = heap.into_sorted_vec().into_iter().map(|r| r.0.0).collect();
    // `into_sorted_vec` produces ascending order; reverse to get highest first.
    out.reverse();
    out
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
