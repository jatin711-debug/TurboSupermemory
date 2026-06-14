//! Tiered vector segments.

pub mod cold;
pub mod hot;
pub mod mmap_array;
pub mod warm;

use crate::config::{Flusher, Tier};
use crate::metadata_store::MetadataStore;
use crate::record::{PointOffset, Record};
use crate::StorageError;

pub use cold::ColdSegment;
pub use hot::HotSegment;
pub use mmap_array::MmapBuffer;
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
    /// full f32 embedding is done by the caller using `MetadataStore`.
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        records: &MetadataStore,
    ) -> Result<Vec<ScoredPoint>>;
    fn point_count(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn flusher(&self) -> Flusher;
}

/// Merge candidate lists from multiple segments, preserving the highest scores.
pub fn merge_candidates(lists: Vec<Vec<ScoredPoint>>, top_k: usize) -> Vec<ScoredPoint> {
    let mut merged: Vec<ScoredPoint> = Vec::with_capacity(top_k * lists.len());
    for list in lists {
        merged.extend(list);
    }
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    merged.truncate(top_k);
    merged
}
