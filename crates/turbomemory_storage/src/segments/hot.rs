//! Hot tier: full-precision f32 vectors stored in an appendable plain segment.
//!
//! The mutable Hot segment uses an exact brute-force scan instead of an HNSW
//! index. This mirrors Qdrant's appendable plain segment and Chroma's
//! brute-force buffer: inserts are O(1) appends, and the expensive HNSW graph
//! construction happens offline when the segment is sealed.

use crate::config::{Flusher, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use ahash::AHashSet;
use roaring::RoaringBitmap;
use turbomemory_core::{cosine_similarity_batch, validate_dimension};

/// Appendable Hot segment backed by a plain list of offsets.
///
/// Vectors are not duplicated here; they are read from the shared `VectorStore`
/// during search, so the Hot segment only tracks which offsets it owns.
pub struct HotSegment {
    dim: usize,
    offsets: Vec<PointOffset>,
    seen: AHashSet<PointOffset>,
}

impl HotSegment {
    pub fn new(config: &crate::config::StoreConfig) -> crate::Result<Self> {
        Ok(Self {
            dim: config.dimension,
            offsets: Vec::with_capacity(config.tier.hot_capacity.clamp(1, 1024)),
            seen: AHashSet::with_capacity(config.tier.hot_capacity.clamp(1, 1024)),
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

    /// Rebuild the segment from the given records. Used after promotion/demotion.
    pub fn rebuild(&mut self, records: &[(PointOffset, Record)]) -> crate::Result<()> {
        self.offsets.clear();
        self.seen.clear();
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
        if self.seen.insert(offset) {
            self.offsets.push(offset);
        }
        Ok(())
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;
        let view = vectors.read_view();

        // Filter offsets first if a payload bitmap is supplied.
        let filtered: Vec<PointOffset> = if let Some(bitmap) = allowed_offsets {
            self.offsets
                .iter()
                .copied()
                .filter(|o| bitmap.contains(*o as u32))
                .collect()
        } else {
            self.offsets.clone()
        };

        // Score in chunks using the batched SIMD kernel to amortize call overhead.
        const CHUNK: usize = 64;
        let mut scored: Vec<ScoredPoint> = Vec::with_capacity(filtered.len());
        for chunk in filtered.chunks(CHUNK) {
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
                    tier: Tier::Hot,
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

    fn point_count(&self) -> usize {
        self.offsets.len()
    }

    fn memory_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<PointOffset>()
            + self.seen.capacity() * (std::mem::size_of::<PointOffset>() + 1)
    }

    fn flusher(&self) -> Flusher {
        // Hot tier is rebuilt on open from durable metadata.
        Box::new(|| Ok(()))
    }

    fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OptimizerBudget, StoreConfig, TierConfig};
    use crate::record::Record;
    use std::sync::Arc;

    fn test_config(dim: usize) -> StoreConfig {
        StoreConfig {
            dimension: dim,
            max_edges: 8,
            level0_factor: 2,
            ef_construction: 100,
            search_list_size: 16,
            outlier_count: 0,
            initial_capacity: 16,
            tier: TierConfig {
                hot_capacity: 100,
                warm_capacity: 1000,
                warm_quantizer: crate::config::QuantizerKind::Scalar { bits: 8 },
                warm_chunk_bytes: 4096,
                hnsw_threshold: 1000,
                full_scan_threshold_kb: 10_000,
                merge_threshold_segments: 4,
                merge_max_records: 200_000,
                cold_quantizer: crate::config::QuantizerKind::Sign,
                hot_promote_threshold: 2.0,
                warm_demote_threshold: 0.5,
                recency_half_life_secs: 60,
            },
            optimizer_budget: OptimizerBudget::default(),
            auto_consolidation_interval: None,
        }
    }

    fn make_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx % dim] = 1.0;
        v
    }

    fn make_record(id: &str, dim: usize, idx: usize) -> Record {
        Record {
            id: id.into(),
            text: String::new(),
            embedding: Arc::from(make_vec(dim, idx)),
            importance: 1.0,
            concepts: Vec::new(),
            created_at: idx as u64,
            insert_seq: idx as u64,
            access_count: 0,
            last_accessed: 0,
            tier: Tier::Hot,
            payload: None,
        }
    }

    #[test]
    fn plain_hot_exact_search() {
        let tmp = tempfile::tempdir().unwrap();
        let vectors = VectorStore::new(tmp.path().join("vectors.bin"), 4).unwrap();
        let config = test_config(4);
        let mut seg = HotSegment::new(&config).unwrap();

        for i in 0..4usize {
            vectors.put(i as u64, &make_vec(4, i)).unwrap();
            let rec = make_record(&format!("m{i}"), 4, i);
            seg.insert(i as u64, &rec).unwrap();
        }

        let results = seg
            .search(&[1.0f32, 0.0, 0.0, 0.0], 2, &vectors, None)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].offset, 0);
        assert_eq!(results[1].offset, 1);
    }

    #[test]
    fn plain_hot_filters_allowed_offsets() {
        let tmp = tempfile::tempdir().unwrap();
        let vectors = VectorStore::new(tmp.path().join("vectors.bin"), 4).unwrap();
        let config = test_config(4);
        let mut seg = HotSegment::new(&config).unwrap();

        for i in 0..4usize {
            vectors.put(i as u64, &make_vec(4, i)).unwrap();
            let rec = make_record(&format!("m{i}"), 4, i);
            seg.insert(i as u64, &rec).unwrap();
        }

        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(2);
        bitmap.insert(3);

        let results = seg
            .search(&[1.0f32, 0.0, 0.0, 0.0], 2, &vectors, Some(&bitmap))
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].offset, 2);
        assert_eq!(results[1].offset, 3);
    }
}
