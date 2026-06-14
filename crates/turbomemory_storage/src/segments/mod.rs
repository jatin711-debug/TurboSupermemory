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
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        vectors: &VectorStore,
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
