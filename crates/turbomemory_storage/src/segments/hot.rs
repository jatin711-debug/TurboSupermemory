//! Hot tier: full-precision f32 vectors indexed by HNSW.
//!
//! The HNSW implementation is provided by `usearch`, a C++ HNSW library with
//! Rust bindings and built-in SIMD distance kernels.

use crate::config::{Flusher, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use crate::StorageError;
use turbomemory_core::validate_dimension;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

/// Appendable Hot segment backed by a `usearch` HNSW index.
pub struct HotSegment {
    dim: usize,
    index: Index,
    count: usize,
    config: crate::config::StoreConfig,
}

impl HotSegment {
    fn index_options(dim: usize, config: &crate::config::StoreConfig) -> IndexOptions {
        IndexOptions {
            dimensions: dim,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: config.max_edges,
            expansion_add: config.ef_construction(),
            expansion_search: config.search_list_size,
            multi: false,
        }
    }

    pub fn new(config: &crate::config::StoreConfig) -> crate::Result<Self> {
        let options = Self::index_options(config.dimension, config);
        let index = Index::new(&options)
            .map_err(|e| StorageError::IndexError(format!("usearch index creation failed: {e}")))?;
        index
            .reserve(config.tier.hot_capacity.max(1))
            .map_err(|e| StorageError::IndexError(format!("usearch reserve failed: {e}")))?;
        Ok(Self {
            dim: config.dimension,
            index,
            count: 0,
            config: config.clone(),
        })
    }

    pub fn from_records(
        config: &crate::config::StoreConfig,
        records: &[(PointOffset, Record)],
    ) -> crate::Result<Self> {
        let mut seg = Self::new(config)?;
        for (offset, rec) in records {
            seg.insert(*offset, rec)?;
        }
        Ok(seg)
    }

    /// Rebuild the HNSW index from the given records.  Used after a promotion
    /// or demotion operation.
    pub fn rebuild(&mut self, records: &[(PointOffset, Record)]) -> crate::Result<()> {
        *self = Self::new(&self.config)?;
        for (offset, rec) in records {
            self.insert(*offset, rec)?;
        }
        Ok(())
    }
}

impl VectorSegment for HotSegment {
    fn tier(&self) -> Tier {
        Tier::Hot
    }

    fn insert(&mut self, offset: PointOffset, record: &Record) -> crate::Result<()> {
        validate_dimension(record.embedding_f32(), self.dim)?;
        self.index
            .add(offset, record.embedding_f32())
            .map_err(|e| StorageError::IndexError(format!("usearch insert failed: {e}")))?;
        self.count += 1;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _vectors: &VectorStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;
        // `usearch` returns cosine *distance* (1 - similarity).  Convert back
        // to the similarity score used by the rest of the engine.
        let matches = self
            .index
            .search(query, top_k)
            .map_err(|e| StorageError::IndexError(format!("usearch search failed: {e}")))?;
        Ok(matches
            .keys
            .into_iter()
            .zip(matches.distances)
            .map(|(offset, distance)| ScoredPoint {
                offset,
                score: (1.0 - distance).clamp(-1.0, 1.0),
                tier: Tier::Hot,
            })
            .collect())
    }

    fn point_count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        self.index.memory_usage()
    }

    fn flusher(&self) -> Flusher {
        // Hot tier is rebuilt on open from durable metadata.
        Box::new(|| Ok(()))
    }
}
