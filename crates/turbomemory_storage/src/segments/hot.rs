//! Hot tier: full-precision f32 vectors indexed by HNSW.

use crate::config::{Flusher, Tier};
use crate::metadata_store::MetadataStore;
use crate::record::{PointOffset, Record};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::StorageError;
use turbomemory_core::{cosine_similarity, validate_dimension};
use vector_index::{HnswConfig, HnswIndex, Metric, Neighbor};

/// Cosine distance metric for the HNSW index.
#[derive(Clone, Default)]
pub struct CosineMetric;

impl Metric for CosineMetric {
    type Point = Vec<f32>;

    fn distance(&self, a: &Self::Point, b: &Self::Point) -> f32 {
        let sim = cosine_similarity(a, b);
        (1.0 - sim).max(0.0)
    }

    fn dim(&self, point: &Self::Point) -> usize {
        point.len()
    }
}

/// Appendable Hot segment backed by an in-memory HNSW index.
pub struct HotSegment {
    dim: usize,
    index: HnswIndex<Vec<f32>, CosineMetric>,
    count: usize,
    config: crate::config::StoreConfig,
}

impl HotSegment {
    pub fn new(config: &crate::config::StoreConfig) -> crate::Result<Self> {
        let hnsw_cfg = HnswConfig {
            m: config.max_edges,
            m_max0: config.max_edges * 2,
            ef_construction: config.ef_construction(),
            ef_search: config.search_list_size,
            level_lambda: 1.0 / (config.max_edges as f32).ln(),
        };
        let index = HnswIndex::new(hnsw_cfg, CosineMetric)
            .map_err(|e| StorageError::IndexError(e.to_string()))?;
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
            .insert(offset, record.embedding_f32().to_vec())
            .map_err(|e| StorageError::IndexError(e.to_string()))?;
        self.count += 1;
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _records: &MetadataStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;
        let q = query.to_vec();
        let neighbors: Vec<Neighbor> = self.index.search(&q, top_k);
        Ok(neighbors
            .into_iter()
            .map(|n| ScoredPoint {
                offset: n.id,
                score: (1.0 - n.distance).clamp(-1.0, 1.0),
                tier: Tier::Hot,
            })
            .collect())
    }

    fn point_count(&self) -> usize {
        self.count
    }

    fn memory_bytes(&self) -> usize {
        // HNSW graph overhead dominates; this is a rough lower bound.
        self.count * self.dim * std::mem::size_of::<f32>() * 2
    }

    fn flusher(&self) -> Flusher {
        // Hot tier is rebuilt on open from durable metadata.
        Box::new(|| Ok(()))
    }
}
