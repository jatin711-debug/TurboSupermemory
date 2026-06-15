//! Algorithm-agnostic vector index trait used by immutable segments.
//!
//! `VectorIndex` abstracts the search algorithm (HNSW, brute-force, quantized,
//! etc.) from segment lifecycle concerns such as persistence and tiering.

use crate::record::PointOffset;
use crate::segments::ScoredPoint;
use crate::vector_store::VectorStore;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Common interface for the search index inside an immutable segment.
pub trait VectorIndex: Send + Sync {
    /// Search the index and return scored candidates.
    ///
    /// `allowed_offsets` is an optional filter bitmap. The implementation may
    /// either apply it during the search (filter-aware traversal) or as a
    /// post-filter, falling back to an exact scan when the filter is very
    /// selective.
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        allowed_offsets: Option<&RoaringBitmap>,
        vectors: &VectorStore,
    ) -> crate::Result<Vec<ScoredPoint>>;

    /// Offsets stored in this index, in insertion order.
    fn offsets(&self) -> &[PointOffset];

    /// Number of indexed vectors.
    fn point_count(&self) -> usize;

    /// Current memory footprint in bytes.
    fn memory_bytes(&self) -> usize;

    /// Persist any in-memory state to the segment directory.
    fn save(&self) -> crate::Result<()>;
}

/// Manifest shared by all `VectorIndex` implementations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorIndexManifest {
    pub version: u32,
    /// Identifier used to dispatch to the correct implementation on load.
    pub index_type: String,
    pub dimension: usize,
    pub offsets: Vec<PointOffset>,
}
