//! Sealed Hot segment: an immutable, disk-persisted `usearch` HNSW index.
//!
//! When the mutable Hot segment reaches capacity it is rebuilt as a sealed
//! segment, persisted to disk, and reopened on subsequent engine starts.  This
//! removes the expensive "rebuild HNSW from metadata on every open" path.

use crate::config::{Flusher, StoreConfig, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use crate::StorageError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use turbomemory_core::validate_dimension;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

const MANIFEST_FILE: &str = "manifest.json";
const INDEX_FILE: &str = "index.usearch";

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dimension: usize,
    max_edges: usize,
    search_list_size: usize,
    offsets: Vec<PointOffset>,
}

/// Immutable sealed Hot segment backed by a persisted `usearch` HNSW index.
pub struct SealedHotSegment {
    dim: usize,
    index: Index,
    path: PathBuf,
    offsets: Vec<PointOffset>,
}

impl SealedHotSegment {
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

    /// Build a sealed segment from records and persist it to disk.
    pub fn from_records(
        path: impl AsRef<Path>,
        config: &StoreConfig,
        records: &[(PointOffset, Record)],
    ) -> crate::Result<Self> {
        if records.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot seal an empty hot segment".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        let options = Self::index_options(config.dimension, config);
        let index = Index::new(&options)
            .map_err(|e| StorageError::IndexError(format!("usearch index creation failed: {e}")))?;
        index
            .reserve(records.len().max(1))
            .map_err(|e| StorageError::IndexError(format!("usearch reserve failed: {e}")))?;

        for (offset, rec) in records {
            index
                .add(*offset, rec.embedding_f32())
                .map_err(|e| StorageError::IndexError(format!("usearch add failed: {e}")))?;
        }

        let index_path = path.join(INDEX_FILE);
        let index_path_str = index_path
            .to_str()
            .ok_or_else(|| StorageError::InvalidArgument("invalid sealed hot path".into()))?;
        index
            .save(index_path_str)
            .map_err(|e| StorageError::IndexError(format!("usearch save failed: {e}")))?;

        let offsets: Vec<PointOffset> = records.iter().map(|(offset, _)| *offset).collect();
        let manifest = Manifest {
            version: 1,
            dimension: config.dimension,
            max_edges: config.max_edges,
            search_list_size: config.search_list_size,
            offsets: offsets.clone(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| StorageError::Serialize(Box::new(bincode::ErrorKind::Custom(e.to_string()))))?;
        std::fs::write(path.join(MANIFEST_FILE), manifest_json)?;

        Ok(Self {
            dim: config.dimension,
            index,
            path,
            offsets,
        })
    }

    /// Open a previously sealed segment.  Prefer `view` (mmap) and fall back to
    /// `load` if the platform doesn't support it.
    pub fn open(path: impl AsRef<Path>, config: &StoreConfig) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let manifest: Manifest = {
            let bytes = std::fs::read(path.join(MANIFEST_FILE))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| StorageError::InvalidArgument(format!("bad sealed hot manifest: {e}")))?
        };
        if manifest.dimension != config.dimension {
            return Err(StorageError::DimensionMismatch);
        }

        let index_path = path.join(INDEX_FILE);
        let index_path_str = index_path
            .to_str()
            .ok_or_else(|| StorageError::InvalidArgument("invalid sealed hot path".into()))?;

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
            index,
            path,
            offsets: manifest.offsets,
        })
    }

    pub fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }

    pub fn point_count(&self) -> usize {
        self.offsets.len()
    }
}

impl VectorSegment for SealedHotSegment {
    fn tier(&self) -> Tier {
        Tier::Hot
    }

    fn insert(&mut self, _offset: PointOffset, _record: &Record) -> crate::Result<()> {
        Err(StorageError::InvalidArgument(
            "sealed hot segments are immutable".into(),
        ))
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        _vectors: &VectorStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;
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
        self.point_count()
    }

    fn memory_bytes(&self) -> usize {
        self.index.memory_usage()
    }

    fn flusher(&self) -> Flusher {
        // The index file is immutable after sealing.
        let path = self.path.clone();
        Box::new(move || {
            if std::fs::metadata(&path).is_ok() {
                Ok(())
            } else {
                Err(StorageError::InvalidArgument(
                    "sealed hot segment missing".into(),
                ))
            }
        })
    }
}
