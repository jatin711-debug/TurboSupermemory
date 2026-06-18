//! Warm tier: scalar-quantized vectors stored in a memory-mapped file.

use crate::config::{Flusher, QuantizerKind, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::mmap_array::{MmapBuffer, MmapFileWriter};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use crate::StorageError;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use turbomemory_core::quantization::{Quantizer, ScalarQuantizer, SignQuantizer, VectorQuantizer};
use turbomemory_core::quantized_search::{EncodedQuery, QuantizedStore};
use turbomemory_core::turbo_quant::{TurboQuantMseQuantizer, TurboQuantProdQuantizer};
use turbomemory_core::{cosine_similarity, validate_dimension};

const DATA_FILE: &str = "data.bin";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dimension: usize,
    quantizer: VectorQuantizer,
    offsets: Vec<PointOffset>,
}

/// Immutable Warm segment.
pub struct WarmSegment {
    dim: usize,
    quantizer: VectorQuantizer,
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
        kind: QuantizerKind,
    ) -> crate::Result<Self> {
        if records.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot build empty warm segment".into(),
            ));
        }
        let dim = records[0].1.embedding_f32().len();
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        let quantizer = match kind {
            QuantizerKind::Scalar { bits } => {
                let embeddings: Vec<_> = records
                    .iter()
                    .map(|(_, r)| r.embedding_f32().to_vec())
                    .collect();
                VectorQuantizer::Scalar(
                    ScalarQuantizer::calibrate(&embeddings, bits).map_err(StorageError::Core)?,
                )
            }
            QuantizerKind::Sign => VectorQuantizer::Sign(SignQuantizer::new(dim)),
            QuantizerKind::TurboQuantMse { bits } => VectorQuantizer::TurboQuantMse(
                TurboQuantMseQuantizer::new(dim, bits, QuantizerKind::ROTATION_SEED)
                    .map_err(StorageError::Core)?,
            ),
            QuantizerKind::TurboQuantProd { bits } => VectorQuantizer::TurboQuantProd(
                TurboQuantProdQuantizer::new(
                    dim,
                    bits,
                    QuantizerKind::ROTATION_SEED,
                    QuantizerKind::QJL_SEED,
                )
                .map_err(StorageError::Core)?,
            ),
        };

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
        const CHUNK: usize = 1024;
        let bytes_per_vec = self.quantizer.encoded_bytes_per_vector();
        let mut candidates = Vec::with_capacity(top_k * 2);
        let mut filtered_bytes = Vec::with_capacity(CHUNK * bytes_per_vec);
        let mut filtered_offsets = Vec::with_capacity(CHUNK);
        for (chunk_idx, chunk) in self.offsets.chunks(CHUNK).enumerate() {
            let base = chunk_idx * CHUNK;
            if let Some(bitmap) = allowed_offsets {
                // Filtered path: copy only the vectors that pass the bitmap.
                filtered_bytes.clear();
                filtered_offsets.clear();
                for (local, &offset) in chunk.iter().enumerate() {
                    if bitmap.contains(offset as u32) {
                        filtered_bytes.extend_from_slice(self.encoded_vector(base + local));
                        filtered_offsets.push(offset);
                    }
                }
                if filtered_offsets.is_empty() {
                    continue;
                }
                let scores = eq.score_batch(&filtered_bytes);
                candidates.extend(filtered_offsets.iter().copied().zip(scores).map(
                    |(offset, score)| ScoredPoint {
                        offset,
                        score,
                        tier: Tier::Warm,
                    },
                ));
            } else {
                // Fast path: contiguous mmap slice, no copy.
                let start = base * bytes_per_vec;
                let end = start + chunk.len() * bytes_per_vec;
                let scores = eq.score_batch(&self.buffer.as_bytes()[start..end]);
                candidates.extend(chunk.iter().copied().zip(scores).map(|(offset, score)| {
                    ScoredPoint {
                        offset,
                        score,
                        tier: Tier::Warm,
                    }
                }));
            }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::Record;
    use std::sync::Arc;

    fn make_record(offset: PointOffset, dim: usize, idx: usize) -> (PointOffset, Record) {
        let mut v = vec![0.0f32; dim];
        v[idx % dim] = 1.0;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm);
        (
            offset,
            Record {
                id: format!("id-{idx}"),
                text: String::new(),
                embedding: Arc::from(v),
                importance: 1.0,
                concepts: Vec::new(),
                created_at: 0,
                insert_seq: 0,
                access_count: 0,
                last_accessed: 0,
                tier: Tier::Warm,
                payload: None,
            },
        )
    }

    #[test]
    fn warm_segment_turbo_quant_prod_search() {
        let dim = 64;
        let tmp = tempfile::tempdir().unwrap();
        let records: Vec<_> = (0..20).map(|i| make_record(i as u64 + 1, dim, i)).collect();
        let segment = WarmSegment::from_records(
            tmp.path().join("warm"),
            &records,
            QuantizerKind::TurboQuantProd { bits: 3 },
        )
        .unwrap();
        assert_eq!(segment.point_count(), 20);

        // Build a minimal VectorStore so search can rerank.
        let vectors_path = tmp.path().join("vectors");
        let vectors = VectorStore::new_with_capacity(&vectors_path, dim, 32).unwrap();
        for (offset, rec) in &records {
            vectors.put(*offset, rec.embedding_f32()).unwrap();
        }

        let query = {
            let mut v = vec![0.0f32; dim];
            v[0] = 1.0;
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm);
            v
        };
        let results = segment.search(&query, 5, &vectors, None).unwrap();
        assert!(!results.is_empty());
        // The top result should be the vector that has mass on dimension 0.
        assert_eq!(results[0].offset, 1);
    }

    #[test]
    fn warm_segment_scalar_still_works() {
        let dim = 32;
        let tmp = tempfile::tempdir().unwrap();
        let records: Vec<_> = (0..10).map(|i| make_record(i as u64 + 1, dim, i)).collect();
        let segment = WarmSegment::from_records(
            tmp.path().join("warm"),
            &records,
            QuantizerKind::Scalar { bits: 4 },
        )
        .unwrap();
        assert_eq!(segment.point_count(), 10);
    }
}
