//! GPU-accelerated HNSW implementation of the [`VectorIndex`] trait.
//!
//! Uses the `turbomemory_gpu` crate's `GpuBackend` for:
//! - Batched distance computation during graph construction
//! - GPU-accelerated candidate search
//!
//! Falls back to CPU `UsearchIndex` if GPU is unavailable or fails.

use crate::config::{StoreConfig, Tier};
use crate::record::PointOffset;
use crate::segments::vector_index::{VectorIndex, VectorIndexManifest};
use crate::segments::{exact_search_over_offsets, ScoredPoint};
use crate::vector_store::VectorStore;
use crate::StorageError;
use roaring::RoaringBitmap;
use std::path::{Path, PathBuf};
use turbomemory_core::validate_dimension;

const MANIFEST_FILE: &str = "manifest.json";

/// GPU-accelerated HNSW index using `turbomemory_gpu`.
pub struct GpuHnswIndex {
    dim: usize,
    #[allow(dead_code)]
    search_list_size: usize,
    path: PathBuf,
    offsets: Vec<PointOffset>,
    // The GPU ANN index (stored on host, searched with GPU assistance)
    #[cfg(feature = "cuda")]
    gpu_index: Option<turbomemory_gpu::CudaAnnIndex>,
    // Fallback CPU index if GPU fails
    fallback: Option<crate::segments::UsearchIndex>,
}

impl GpuHnswIndex {
    /// Bulk-build a GPU HNSW index from `(offset, vector)` pairs and persist it.
    pub fn build(
        path: impl AsRef<Path>,
        config: &StoreConfig,
        vectors: &[(PointOffset, &[f32])],
    ) -> crate::Result<Self> {
        if vectors.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot build an empty GPU HNSW index".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        let dim = config.dimension;
        let offsets: Vec<PointOffset> = vectors.iter().map(|(o, _)| *o).collect();

        // Try to build on GPU if CUDA feature is enabled. We always also build
        // a usearch fallback on disk for reload compatibility (GPU indices are
        // not persisted across restarts).
        #[cfg(feature = "cuda")]
        let (gpu_index, fallback) = {
            let fallback = crate::segments::UsearchIndex::build(&path, config, vectors)?;
            let flat_vectors: Vec<f32> = vectors
                .iter()
                .flat_map(|(_, v)| v.iter().copied())
                .collect();
            match try_build_gpu(&flat_vectors, vectors.len(), dim, config) {
                Ok(index) => {
                    log::info!(
                        "GPU HNSW: successfully built index for {} vectors",
                        vectors.len()
                    );
                    (Some(index), Some(fallback))
                }
                Err(e) => {
                    log::warn!("GPU HNSW build failed ({}), using usearch", e);
                    (None, Some(fallback))
                }
            }
        };

        #[cfg(not(feature = "cuda"))]
        let (_gpu_index, fallback): (
            Option<turbomemory_gpu::CudaAnnIndex>,
            Option<crate::segments::UsearchIndex>,
        ) = {
            let fallback = crate::segments::UsearchIndex::build(&path, config, vectors)?;
            (None, Some(fallback))
        };

        // Use the fallback from the cfg block above. (Previously this
        // unconditionally rebuilt the usearch index a second time, which
        // shadowed the GPU build result and doubled the build cost.)
        let fallback = fallback.unwrap();

        // Save manifest
        let manifest = VectorIndexManifest {
            version: 1,
            index_type: "gpu_hnsw".into(),
            dimension: dim,
            offsets: offsets.clone(),
        };
        let manifest_json = serde_json::to_vec(&manifest).map_err(|e| {
            StorageError::InvalidArgument(format!("manifest serialization failed: {e}"))
        })?;
        std::fs::write(path.join(MANIFEST_FILE), &manifest_json)?;

        Ok(Self {
            dim,
            search_list_size: config.search_list_size,
            path,
            offsets,
            #[cfg(feature = "cuda")]
            gpu_index,
            fallback: Some(fallback),
        })
    }

    /// Open a previously built GPU HNSW index from disk.
    /// Currently rebuilds the GPU index from vectors (GPU indices are not persisted).
    pub fn open(
        path: impl AsRef<Path>,
        config: &StoreConfig,
        vectors: &[(PointOffset, &[f32])],
    ) -> crate::Result<Self> {
        Self::build(path, config, vectors)
    }
}

impl VectorIndex for GpuHnswIndex {
    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        allowed_offsets: Option<&RoaringBitmap>,
        vectors: &VectorStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;

        // For very selective filters, an exact scan over ONLY the allowed
        // offsets is more reliable than HNSW with post-filtering (mirrors
        // `UsearchIndex::search`). The bitmap must be applied here — scanning
        // the whole segment unfiltered would leak filtered-out records.
        if let Some(bitmap) = allowed_offsets {
            let allowed_in_segment: Vec<PointOffset> = self
                .offsets
                .iter()
                .copied()
                .filter(|o| bitmap.contains(*o as u32))
                .collect();
            let selectivity = allowed_in_segment.len() as f32 / self.offsets.len() as f32;
            if selectivity < 0.05 {
                return exact_search_over_offsets(
                    query,
                    top_k,
                    vectors,
                    &allowed_in_segment,
                    Tier::Hot,
                );
            }
        }

        // GPU-native HNSW search is deliberately not implemented
        // (`GpuBackend::ann_search` returns `BackendNotCompiled`; see
        // turbomemory_gpu). Do NOT call `init_backend()` here: it builds a
        // fresh CUDA context per query, and the search always fell back
        // anyway. The usearch index built alongside the GPU graph serves all
        // searches.
        if let Some(ref fallback) = self.fallback {
            fallback.search(query, top_k, allowed_offsets, vectors)
        } else {
            exact_search_over_offsets(query, top_k, vectors, &self.offsets, Tier::Hot)
        }
    }

    fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }

    fn point_count(&self) -> usize {
        self.offsets.len()
    }

    fn memory_bytes(&self) -> usize {
        #[cfg(feature = "cuda")]
        let gpu_bytes = self.gpu_index.as_ref().map(|i| i.memory_bytes).unwrap_or(0);
        #[cfg(not(feature = "cuda"))]
        let gpu_bytes = 0;
        let fallback_bytes = self
            .fallback
            .as_ref()
            .map(|i| i.memory_bytes())
            .unwrap_or(0);
        gpu_bytes + fallback_bytes + self.offsets.len() * std::mem::size_of::<PointOffset>()
    }

    fn save(&self) -> crate::Result<()> {
        // GPU index is rebuilt on load, so just save the manifest
        let manifest = VectorIndexManifest {
            version: 1,
            index_type: "gpu_hnsw".into(),
            dimension: self.dim,
            offsets: self.offsets.clone(),
        };
        let manifest_json = serde_json::to_vec(&manifest).map_err(|e| {
            StorageError::InvalidArgument(format!("manifest serialization failed: {e}"))
        })?;
        std::fs::write(self.path.join(MANIFEST_FILE), &manifest_json)?;
        Ok(())
    }
}

/// Try to build a GPU HNSW index.
#[cfg(feature = "cuda")]
fn try_build_gpu(
    flat_vectors: &[f32],
    _n: usize,
    dim: usize,
    config: &StoreConfig,
) -> Result<turbomemory_gpu::CudaAnnIndex, turbomemory_gpu::GpuError> {
    let backend = turbomemory_gpu::init_backend();
    if !turbomemory_gpu::is_gpu_accelerated(&backend) {
        return Err(turbomemory_gpu::GpuError::CudaNotAvailable(
            "No GPU backend available".into(),
        ));
    }

    let device_buf = backend.upload_vectors(flat_vectors, dim)?;
    let ann_config = turbomemory_gpu::AnnBuildConfig {
        max_edges: config.max_edges,
        ef_construction: config.ef_construction(),
        ef_search: config.search_list_size,
        target_recall: 0.95,
        max_build_memory: 0,
    };

    let index = backend.build_ann_index(&device_buf, dim, &ann_config)?;
    // Downcast to concrete type to extract data
    index
        .as_any()
        .downcast_ref::<turbomemory_gpu::CudaAnnIndex>()
        .ok_or_else(|| {
            turbomemory_gpu::GpuError::InvalidArgument("Failed to downcast GPU index".into())
        })
        .map(|i| {
            // Clone the index data (vectors + layers)
            turbomemory_gpu::CudaAnnIndex {
                n: i.n,
                dim: i.dim,
                memory_bytes: i.memory_bytes,
                layers: i.layers.clone(),
                vectors: i.vectors.clone(),
            }
        })
}
