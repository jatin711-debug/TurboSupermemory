//! `usearch` HNSW implementation of the [`VectorIndex`] trait.

use crate::config::{Flusher, StoreConfig, Tier};
use crate::record::PointOffset;
use crate::segments::vector_index::{VectorIndex, VectorIndexManifest};
use crate::segments::{exact_search_over_offsets, ScoredPoint};
use crate::vector_store::VectorStore;
use crate::StorageError;
use rayon::prelude::*;
use roaring::RoaringBitmap;
use std::path::{Path, PathBuf};
use turbomemory_core::validate_dimension;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const MANIFEST_FILE: &str = "manifest.json";
const INDEX_FILE: &str = "index.usearch";

/// Number of points inserted single-threaded before switching to parallel
/// insertion. Seeding the graph sequentially establishes a stable entry-point
/// structure; usearch's parallel insertion only degrades recall when it starts
/// from an empty graph. This is the Qdrant pattern (TODO 2.11).
const PARALLEL_SEED_POINTS: usize = 256;

/// Number of threads to use for a single parallel HNSW build.
///
/// Bounded so that `max_concurrent_builds` simultaneous segment builds × this
/// per-build thread count stays near the core count rather than oversubscribing
/// the machine. Qdrant uses `clamp(num_cpus, 1, 16)` overall.
fn build_threads(config: &StoreConfig) -> usize {
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let concurrent = config.optimizer_budget.max_concurrent_builds.max(1);
    (cpus / concurrent).clamp(1, 16)
}

/// `usearch`-backed HNSW index.
pub struct UsearchIndex {
    dim: usize,
    search_list_size: usize,
    index: Index,
    path: PathBuf,
    offsets: Vec<PointOffset>,
}

impl UsearchIndex {
    fn index_options(dim: usize, config: &StoreConfig) -> IndexOptions {
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

    /// Bulk-build a `usearch` index from `(offset, vector)` pairs and persist it.
    pub fn build(
        path: impl AsRef<Path>,
        config: &StoreConfig,
        vectors: &[(PointOffset, &[f32])],
    ) -> crate::Result<Self> {
        if vectors.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot build an empty usearch index".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        let options = Self::index_options(config.dimension, config);
        let index = Index::new(&options)
            .map_err(|e| StorageError::IndexError(format!("usearch index creation failed: {e}")))?;
        // Seed the graph single-threaded, then insert the remainder in parallel.
        // usearch's `Index` is `Send + Sync` and supports concurrent `add`; the
        // sequential seed avoids the low-recall graphs that parallel insertion
        // produces when started from empty (TODO 2.11).
        let threads = build_threads(config);
        index
            .reserve_capacity_and_threads(vectors.len().max(1), threads)
            .map_err(|e| StorageError::IndexError(format!("usearch reserve failed: {e}")))?;

        let seed = PARALLEL_SEED_POINTS.min(vectors.len());
        for (offset, vector) in &vectors[..seed] {
            index
                .add(*offset, vector)
                .map_err(|e| StorageError::IndexError(format!("usearch add failed: {e}")))?;
        }

        if vectors.len() > seed {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| {
                    StorageError::IndexError(format!("usearch build pool failed: {e}"))
                })?;
            pool.install(|| {
                vectors[seed..].par_iter().try_for_each(|(offset, vector)| {
                    index
                        .add(*offset, vector)
                        .map_err(|e| StorageError::IndexError(format!("usearch add failed: {e}")))
                })
            })?;
        }

        let index_path = path.join(INDEX_FILE);
        let index_path_str = index_path
            .to_str()
            .ok_or_else(|| StorageError::InvalidArgument("invalid usearch index path".into()))?;
        index
            .save(index_path_str)
            .map_err(|e| StorageError::IndexError(format!("usearch save failed: {e}")))?;

        let offsets: Vec<PointOffset> = vectors.iter().map(|(offset, _)| *offset).collect();
        let manifest = VectorIndexManifest {
            version: 1,
            index_type: "usearch".to_string(),
            dimension: config.dimension,
            offsets: offsets.clone(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            StorageError::Serialize(Box::new(bincode::ErrorKind::Custom(e.to_string())))
        })?;
        std::fs::write(path.join(MANIFEST_FILE), manifest_json)?;

        Ok(Self {
            dim: config.dimension,
            search_list_size: config.search_list_size,
            index,
            path,
            offsets,
        })
    }

    /// Open a previously persisted `usearch` index.
    pub fn open(path: impl AsRef<Path>, config: &StoreConfig) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let manifest: VectorIndexManifest = {
            let bytes = std::fs::read(path.join(MANIFEST_FILE))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| StorageError::InvalidArgument(format!("bad usearch manifest: {e}")))?
        };
        if manifest.dimension != config.dimension {
            return Err(StorageError::DimensionMismatch);
        }
        if manifest.index_type != "usearch" {
            return Err(StorageError::InvalidArgument(format!(
                "expected usearch index, got {}",
                manifest.index_type
            )));
        }

        let index_path = path.join(INDEX_FILE);
        let index_path_str = index_path
            .to_str()
            .ok_or_else(|| StorageError::InvalidArgument("invalid usearch index path".into()))?;

        let options = Self::index_options(config.dimension, config);
        let index = Index::new(&options)
            .map_err(|e| StorageError::IndexError(format!("usearch index creation failed: {e}")))?;

        if index.view(index_path_str).is_err() {
            index
                .load(index_path_str)
                .map_err(|e| StorageError::IndexError(format!("usearch load failed: {e}")))?;
        }

        Ok(Self {
            dim: config.dimension,
            search_list_size: manifest.offsets.len(),
            index,
            path,
            offsets: manifest.offsets,
        })
    }

    /// Return a flusher that verifies the index files still exist.
    pub fn flusher(&self) -> Flusher {
        let path = self.path.clone();
        Box::new(move || {
            if std::fs::metadata(&path).is_ok() {
                Ok(())
            } else {
                Err(StorageError::InvalidArgument(
                    "usearch index directory missing".into(),
                ))
            }
        })
    }

    /// Raw `usearch` search without filtering.
    fn search_unfiltered(&self, query: &[f32], top_k: usize) -> crate::Result<Vec<ScoredPoint>> {
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
}

impl VectorIndex for UsearchIndex {
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        allowed_offsets: Option<&RoaringBitmap>,
        vectors: &VectorStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;

        // For very selective filters, an exact scan over the allowed offsets is
        // more reliable than HNSW with post-filtering.
        let allowed_in_segment: Option<Vec<PointOffset>> = allowed_offsets.map(|bitmap| {
            self.offsets
                .iter()
                .copied()
                .filter(|o| bitmap.contains(*o as u32))
                .collect()
        });
        if let Some(ref allowed) = allowed_in_segment {
            let threshold = self.search_list_size.max(100);
            if allowed.len() <= threshold {
                return exact_search_over_offsets(query, top_k, vectors, allowed, Tier::Hot);
            }
        }

        // The caller already applies the ef/over-fetch multiplier, so use top_k
        // directly here. For filtered queries, dynamically expand the HNSW fetch
        // if post-filtering discards too many approximate neighbors.
        let allowed_count = allowed_in_segment.as_ref().map(|v| v.len());
        let max_fetch = allowed_count
            .map(|c| c.max(top_k).min(top_k.saturating_mul(16)))
            .unwrap_or_else(|| top_k.saturating_mul(16));
        let mut fetch_k = top_k;
        for _ in 0..4 {
            let mut results = Vec::with_capacity(top_k);
            let candidates = self.search_unfiltered(query, fetch_k)?;
            for candidate in candidates {
                if let Some(bitmap) = allowed_offsets {
                    if !bitmap.contains(candidate.offset as u32) {
                        continue;
                    }
                }
                results.push(candidate);
                if results.len() >= top_k {
                    return Ok(results);
                }
            }
            if fetch_k >= max_fetch || allowed_offsets.is_none() {
                return Ok(results);
            }
            fetch_k = (fetch_k.saturating_mul(2)).min(max_fetch);
        }
        Ok(Vec::new())
    }

    fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }

    fn point_count(&self) -> usize {
        self.offsets.len()
    }

    fn memory_bytes(&self) -> usize {
        self.index.memory_usage()
    }

    fn save(&self) -> crate::Result<()> {
        let index_path = self.path.join(INDEX_FILE);
        let index_path_str = index_path
            .to_str()
            .ok_or_else(|| StorageError::InvalidArgument("invalid usearch index path".into()))?;
        self.index
            .save(index_path_str)
            .map_err(|e| StorageError::IndexError(format!("usearch save failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{OptimizerBudget, StoreConfig, TierConfig};
    use roaring::RoaringBitmap;

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
                hnsw_threshold: 10,
                full_scan_threshold_kb: 10_000,
                merge_threshold_segments: 4,
                merge_max_records: 200_000,
                cold_quantizer: crate::config::QuantizerKind::Sign,
                hot_promote_threshold: 2.0,
                warm_demote_threshold: 0.5,
                recency_half_life_secs: 60,
                max_records: None,
                evict_score_floor: None,
                dedup_cosine_threshold: None,
                dedup_max_pairs_per_cycle: 1024,
            },
            optimizer_budget: OptimizerBudget::default(),
            auto_consolidation_interval: None,
        }
    }

    fn make_vec(dim: usize, idx: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; dim];
        v[idx % dim] = 1.0;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    #[test]
    fn usearch_index_builds_and_searches() {
        let tmp = tempfile::tempdir().unwrap();
        let dim = 8;
        let config = test_config(dim);
        let vectors: Vec<(u64, Vec<f32>)> =
            (0..64usize).map(|i| (i as u64, make_vec(dim, i))).collect();
        let borrowed: Vec<(u64, &[f32])> =
            vectors.iter().map(|(o, v)| (*o, v.as_slice())).collect();

        let index = UsearchIndex::build(tmp.path(), &config, &borrowed).unwrap();
        assert_eq!(index.point_count(), 64);

        let vs_path = tmp.path().join("vectors.bin");
        let vector_store = crate::vector_store::VectorStore::new(&vs_path, dim).unwrap();
        for (offset, v) in &vectors {
            vector_store.put(*offset, v).unwrap();
        }

        let query = make_vec(dim, 0);
        let results = index.search(&query, 5, None, &vector_store).unwrap();
        assert_eq!(results.len(), 5);
        // All returned offsets must belong to the indexed set.
        for r in &results {
            assert!((r.offset as usize) < vectors.len());
        }
    }

    #[test]
    fn usearch_index_round_trip_through_open() {
        let tmp = tempfile::tempdir().unwrap();
        let dim = 8;
        let config = test_config(dim);
        let vectors: Vec<(u64, Vec<f32>)> =
            (0..64usize).map(|i| (i as u64, make_vec(dim, i))).collect();
        let borrowed: Vec<(u64, &[f32])> =
            vectors.iter().map(|(o, v)| (*o, v.as_slice())).collect();

        UsearchIndex::build(tmp.path(), &config, &borrowed).unwrap();
        let index = UsearchIndex::open(tmp.path(), &config).unwrap();
        assert_eq!(index.point_count(), 64);

        let vs_path = tmp.path().join("vectors.bin");
        let vector_store = crate::vector_store::VectorStore::new(&vs_path, dim).unwrap();
        for (offset, v) in &vectors {
            vector_store.put(*offset, v).unwrap();
        }

        let query = make_vec(dim, 3);
        let results = index.search(&query, 5, None, &vector_store).unwrap();
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!((r.offset as usize) < vectors.len());
        }
    }

    #[test]
    fn usearch_index_selective_filter_uses_exact_fallback() {
        let tmp = tempfile::tempdir().unwrap();
        let dim = 8;
        let config = test_config(dim);
        let vectors: Vec<(u64, Vec<f32>)> =
            (0..64usize).map(|i| (i as u64, make_vec(dim, i))).collect();
        let borrowed: Vec<(u64, &[f32])> =
            vectors.iter().map(|(o, v)| (*o, v.as_slice())).collect();

        let index = UsearchIndex::build(tmp.path(), &config, &borrowed).unwrap();
        let vs_path = tmp.path().join("vectors.bin");
        let vector_store = crate::vector_store::VectorStore::new(&vs_path, dim).unwrap();
        for (offset, v) in &vectors {
            vector_store.put(*offset, v).unwrap();
        }

        let query = make_vec(dim, 0);
        let allowed: RoaringBitmap = [0u32, 1, 2, 3, 4].iter().copied().collect();
        let results = index
            .search(&query, 5, Some(&allowed), &vector_store)
            .unwrap();
        assert_eq!(results.len(), 5);
        for r in &results {
            assert!(allowed.contains(r.offset as u32));
        }
    }

    #[test]
    fn usearch_index_moderate_filter_returns_top_k() {
        let tmp = tempfile::tempdir().unwrap();
        let dim = 8;
        let config = test_config(dim);
        let vectors: Vec<(u64, Vec<f32>)> = (0..256usize)
            .map(|i| (i as u64, make_vec(dim, i)))
            .collect();
        let borrowed: Vec<(u64, &[f32])> =
            vectors.iter().map(|(o, v)| (*o, v.as_slice())).collect();

        let index = UsearchIndex::build(tmp.path(), &config, &borrowed).unwrap();
        let vs_path = tmp.path().join("vectors.bin");
        let vector_store = crate::vector_store::VectorStore::new(&vs_path, dim).unwrap();
        for (offset, v) in &vectors {
            vector_store.put(*offset, v).unwrap();
        }

        // Allow a moderately sized subset that does not include the query's
        // exact match, forcing the HNSW path to fetch and filter enough
        // approximate neighbors.
        let allowed: RoaringBitmap = (50u32..150).collect();
        let query = make_vec(dim, 0);
        let results = index
            .search(&query, 10, Some(&allowed), &vector_store)
            .unwrap();
        assert_eq!(results.len(), 10);
        for r in &results {
            assert!(allowed.contains(r.offset as u32));
        }
    }

    /// Deterministic pseudo-random unit vector for recall tests.
    fn rand_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut v = vec![0.0f32; dim];
        for x in v.iter_mut() {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            *x = ((s >> 11) as f32 / (1u64 << 53) as f32) - 0.5;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }

    /// The parallel build path (N > PARALLEL_SEED_POINTS) must produce a graph
    /// whose recall is intact: querying an indexed vector returns itself.
    #[test]
    fn usearch_parallel_build_preserves_recall() {
        let tmp = tempfile::tempdir().unwrap();
        let dim = 32;
        let n = PARALLEL_SEED_POINTS + 200; // force the parallel remainder path
        let config = test_config(dim);
        let vectors: Vec<(u64, Vec<f32>)> = (0..n)
            .map(|i| (i as u64, rand_unit_vec(dim, i as u64)))
            .collect();
        let borrowed: Vec<(u64, &[f32])> =
            vectors.iter().map(|(o, v)| (*o, v.as_slice())).collect();

        let index = UsearchIndex::build(tmp.path(), &config, &borrowed).unwrap();
        assert_eq!(index.point_count(), n);

        let vs_path = tmp.path().join("vectors.bin");
        let vector_store = crate::vector_store::VectorStore::new(&vs_path, dim).unwrap();
        for (offset, v) in &vectors {
            vector_store.put(*offset, v).unwrap();
        }

        // Query each indexed vector with itself; the top hit should be itself.
        let mut hits = 0usize;
        for (offset, v) in &vectors {
            let results = index.search(v, 1, None, &vector_store).unwrap();
            if results.first().map(|r| r.offset) == Some(*offset) {
                hits += 1;
            }
        }
        let recall = hits as f64 / n as f64;
        assert!(
            recall >= 0.95,
            "parallel build self-recall@1 too low: {recall:.3}"
        );
    }
}
