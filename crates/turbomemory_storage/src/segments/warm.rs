//! Warm tier: scalar-quantized vectors stored in a memory-mapped file.

use crate::config::{Flusher, Tier};
use crate::metadata_store::MetadataStore;
use crate::record::{PointOffset, Record};
use crate::segments::mmap_array::{MmapBuffer, MmapFileWriter};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::StorageError;
use std::path::{Path, PathBuf};
use turbomemory_core::quantization::{Quantizer, ScalarQuantizer};
use turbomemory_core::quantized_search::{EncodedQuery, QuantizedStore};
use turbomemory_core::{cosine_similarity, validate_dimension};

/// Immutable Warm segment.
pub struct WarmSegment {
    dim: usize,
    quantizer: ScalarQuantizer,
    offsets: Vec<PointOffset>,
    buffer: MmapBuffer,
    path: PathBuf,
}

impl WarmSegment {
    pub fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }

    pub fn from_records(
        path: impl AsRef<Path>,
        records: &[(PointOffset, Record)],
        bits: u8,
    ) -> crate::Result<Self> {
        if records.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot build empty warm segment".into(),
            ));
        }
        let dim = records[0].1.embedding_f32().len();
        let path = path.as_ref().to_path_buf();

        let embeddings: Vec<_> = records
            .iter()
            .map(|(_, r)| r.embedding_f32().to_vec())
            .collect();
        let quantizer =
            ScalarQuantizer::calibrate(&embeddings, bits).map_err(StorageError::Core)?;

        let bytes_per_vec = quantizer.encoded_bytes_per_vector();
        let mut writer = MmapFileWriter::new(&path);
        let mut offsets = Vec::with_capacity(records.len());
        for (offset, rec) in records {
            let encoded = quantizer
                .encode(rec.embedding_f32())
                .map_err(StorageError::Core)?;
            debug_assert_eq!(encoded.len(), bytes_per_vec);
            writer.write(&encoded);
            offsets.push(*offset);
        }
        let buffer = writer.finish()?;
        Ok(Self {
            dim,
            quantizer,
            offsets,
            buffer,
            path,
        })
    }

    pub fn open(_path: impl AsRef<Path>) -> crate::Result<Self> {
        // Warm segments are rebuilt from records on open in this iteration,
        // so a standalone open path is not required.  This method is reserved
        // for future persistent segment metadata.
        Err(StorageError::InvalidArgument(
            "WarmSegment::open is not implemented".into(),
        ))
    }

    fn encoded_vector(&self, idx: usize) -> &[u8] {
        let bytes_per_vec = self.quantizer.encoded_bytes_per_vector();
        let start = idx * bytes_per_vec;
        &self.buffer.as_bytes()[start..start + bytes_per_vec]
    }
}

impl VectorSegment for WarmSegment {
    fn tier(&self) -> Tier {
        Tier::Warm
    }

    fn insert(&mut self, _offset: PointOffset, _record: &Record) -> crate::Result<()> {
        Err(StorageError::InvalidArgument(
            "warm segments are immutable; insert into hot".into(),
        ))
    }

    fn search(
        &self,
        query: &[f32],
        top_k: usize,
        records: &MetadataStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;
        let eq = self
            .quantizer
            .encode_query(query)
            .map_err(StorageError::Core)?;

        let mut all: Vec<ScoredPoint> = self
            .offsets
            .iter()
            .enumerate()
            .map(|(idx, &offset)| {
                let encoded = self.encoded_vector(idx);
                let score = eq.score(encoded);
                ScoredPoint {
                    offset,
                    score,
                    tier: Tier::Warm,
                }
            })
            .collect();
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let candidates: Vec<_> = all.into_iter().take(top_k).collect();

        // Rerank with full f32 embeddings from metadata.
        let mut reranked: Vec<ScoredPoint> = candidates
            .into_iter()
            .filter_map(|c| {
                records.get(c.offset).ok().flatten().map(|rec| ScoredPoint {
                    offset: c.offset,
                    score: cosine_similarity(query, rec.embedding_f32()),
                    tier: Tier::Warm,
                })
            })
            .collect();
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    fn point_count(&self) -> usize {
        self.offsets.len()
    }

    fn memory_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn flusher(&self) -> Flusher {
        // The mmap file is already synced to disk during construction.
        let path = self.path.clone();
        Box::new(move || {
            if std::fs::metadata(&path).is_ok() {
                Ok(())
            } else {
                Err(StorageError::InvalidArgument(
                    "warm segment file missing".into(),
                ))
            }
        })
    }
}
