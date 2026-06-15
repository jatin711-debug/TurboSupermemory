//! Warm tier: scalar-quantized vectors stored in a memory-mapped file.

use crate::config::{Flusher, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::mmap_array::{MmapBuffer, MmapFileWriter};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use crate::StorageError;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use turbomemory_core::quantization::{Quantizer, ScalarQuantizer};
use turbomemory_core::quantized_search::{EncodedQuery, QuantizedStore};
use turbomemory_core::{cosine_similarity, validate_dimension};

const DATA_FILE: &str = "data.bin";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dimension: usize,
    quantizer: ScalarQuantizer,
    offsets: Vec<PointOffset>,
}

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
        fs::create_dir_all(&path)?;

        let embeddings: Vec<_> = records
            .iter()
            .map(|(_, r)| r.embedding_f32().to_vec())
            .collect();
        let quantizer =
            ScalarQuantizer::calibrate(&embeddings, bits).map_err(StorageError::Core)?;

        let bytes_per_vec = quantizer.encoded_bytes_per_vector();
        let data_path = path.join(DATA_FILE);
        let mut writer = MmapFileWriter::new(&data_path);
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

        let manifest = Manifest {
            version: 1,
            dimension: dim,
            quantizer: quantizer.clone(),
            offsets: offsets.clone(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest).map_err(|e| {
            StorageError::Serialize(Box::new(bincode::ErrorKind::Custom(e.to_string())))
        })?;
        fs::write(path.join(MANIFEST_FILE), manifest_json)?;

        Ok(Self {
            dim,
            quantizer,
            offsets,
            buffer,
            path,
        })
    }

    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let manifest: Manifest = {
            let bytes = fs::read(path.join(MANIFEST_FILE))?;
            serde_json::from_slice(&bytes)
                .map_err(|e| StorageError::InvalidArgument(format!("bad warm manifest: {e}")))?
        };

        let data_path = path.join(DATA_FILE);
        let buffer = MmapBuffer::open(&data_path)?;

        let expected_bytes = manifest
            .offsets
            .len()
            .checked_mul(manifest.quantizer.encoded_bytes_per_vector())
            .ok_or_else(|| StorageError::InvalidArgument("warm segment size overflow".into()))?;
        if buffer.len() < expected_bytes {
            return Err(StorageError::InvalidArgument(
                "warm segment data file is too small".into(),
            ));
        }

        Ok(Self {
            dim: manifest.dimension,
            quantizer: manifest.quantizer,
            offsets: manifest.offsets,
            buffer,
            path,
        })
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
        vectors: &VectorStore,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<ScoredPoint>> {
        validate_dimension(query, self.dim)?;
        let eq = self
            .quantizer
            .encode_query(query)
            .map_err(StorageError::Core)?;

        // Score vectors in SIMD-friendly batches to reduce iterator/closure
        // overhead and let the quantized kernels amortize query setup.
        const CHUNK: usize = 64;
        let bytes_per_vec = self.quantizer.encoded_bytes_per_vector();
        let mut candidates = Vec::with_capacity(top_k * 2);
        for (chunk_idx, chunk) in self.offsets.chunks(CHUNK).enumerate() {
            let base = chunk_idx * CHUNK;
            let mut chunk_bytes = Vec::with_capacity(chunk.len() * bytes_per_vec);
            let mut chunk_offsets = Vec::with_capacity(chunk.len());
            for (local, &offset) in chunk.iter().enumerate() {
                if let Some(bitmap) = allowed_offsets {
                    if !bitmap.contains(offset as u32) {
                        continue;
                    }
                }
                chunk_bytes.extend_from_slice(self.encoded_vector(base + local));
                chunk_offsets.push(offset);
            }
            if chunk_offsets.is_empty() {
                continue;
            }
            let scores = eq.score_batch(&chunk_bytes);
            candidates.extend(
                chunk_offsets
                    .into_iter()
                    .zip(scores)
                    .map(|(offset, score)| ScoredPoint {
                        offset,
                        score,
                        tier: Tier::Warm,
                    }),
            );
        }
        let candidates = crate::segments::top_k_minheap(candidates.into_iter(), top_k);

        // Rerank with full f32 embeddings from the vector store.
        let view = vectors.read_view();
        let mut reranked: Vec<ScoredPoint> = candidates
            .into_iter()
            .filter_map(|c| {
                view.get(c.offset).map(|v| ScoredPoint {
                    offset: c.offset,
                    score: cosine_similarity(query, v),
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

    fn offsets(&self) -> &[PointOffset] {
        &self.offsets
    }

    fn memory_bytes(&self) -> usize {
        self.buffer.len()
    }

    fn segment_path(&self) -> Option<&Path> {
        Some(&self.path)
    }

    fn flusher(&self) -> Flusher {
        // The mmap file is already synced to disk during construction.
        let path = self.path.clone();
        Box::new(move || {
            if fs::metadata(&path).is_ok() {
                Ok(())
            } else {
                Err(StorageError::InvalidArgument(
                    "warm segment file missing".into(),
                ))
            }
        })
    }
}
