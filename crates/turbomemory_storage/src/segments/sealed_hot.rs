//! Sealed Hot segment: an immutable segment that delegates search to a
//! pluggable [`VectorIndex`] implementation.
//!
//! Today the only implementation is `UsearchIndex`; the segment itself is
//! index-agnostic so future indexes can be swapped in without changing the
//! optimizer or segment holder.

use crate::config::{Flusher, StoreConfig, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::vector_index::{VectorIndex, VectorIndexManifest};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use crate::StorageError;
use roaring::RoaringBitmap;
use std::path::{Path, PathBuf};

const MANIFEST_FILE: &str = "manifest.json";

/// Immutable sealed Hot segment backed by a persisted [`VectorIndex`].
pub struct SealedHotSegment {
    index: Box<dyn VectorIndex>,
    path: PathBuf,
    offsets: Vec<PointOffset>,
}

impl SealedHotSegment {
    /// Build a sealed segment from records and persist it to disk.
    pub fn from_records(
        path: impl AsRef<Path>,
        config: &StoreConfig,
        records: &[(PointOffset, Record)],
    ) -> crate::Result<Self> {
        let vectors: Vec<(PointOffset, &[f32])> = records
            .iter()
            .map(|(offset, rec)| (*offset, rec.embedding_f32()))
            .collect();
        Self::from_vectors(path, config, &vectors)
    }

    /// Bulk-build a sealed segment from `(offset, vector)` pairs.
    pub fn from_vectors(
        path: impl AsRef<Path>,
        config: &StoreConfig,
        vectors: &[(PointOffset, &[f32])],
    ) -> crate::Result<Self> {
        if vectors.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot seal an empty hot segment".into(),
            ));
        }
        let path = path.as_ref().to_path_buf();
        let index = Box::new(crate::segments::UsearchIndex::build(
            &path, config, vectors,
        )?);
        let offsets = index.offsets().to_vec();

        Ok(Self {
            index,
            path,
            offsets,
        })
    }

    /// Open a previously sealed segment from disk.
    pub fn open(path: impl AsRef<Path>, config: &StoreConfig) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let manifest: VectorIndexManifest = {
            let bytes = std::fs::read(path.join(MANIFEST_FILE))?;
            serde_json::from_slice(&bytes).map_err(|e| {
                StorageError::InvalidArgument(format!("bad sealed hot manifest: {e}"))
            })?
        };
        if manifest.dimension != config.dimension {
            return Err(StorageError::DimensionMismatch);
        }

        let index: Box<dyn VectorIndex> = match manifest.index_type.as_str() {
            "usearch" => Box::new(crate::segments::UsearchIndex::open(&path, config)?),
            other => {
                return Err(StorageError::InvalidArgument(format!(
                    "unsupported sealed hot index type: {other}"
                )))
            }
        };
        let offsets = index.offsets().to_vec();

        Ok(Self {
            index,
            path,
            offsets,
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
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<ScoredPoint>> {
        self.index.search(query, top_k, allowed_offsets, vectors)
    }

    fn point_count(&self) -> usize {
        self.point_count()
    }

    fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }

    fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }

    fn segment_path(&self) -> Option<&std::path::Path> {
        Some(&self.path)
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
