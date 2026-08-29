//! Cold tier: 1-bit sign-quantized vectors stored in a memory-mapped file.

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
use turbomemory_core::{cosine_similarity_batch, validate_dimension};

const DATA_FILE: &str = "data.bin";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dimension: usize,
    quantizer: VectorQuantizer,
    offsets: Vec<PointOffset>,
}

/// Immutable Cold segment.
pub struct ColdSegment {
    dim: usize,
    quantizer: VectorQuantizer,
    offsets: Vec<PointOffset>,
    buffer: MmapBuffer,
    path: PathBuf,
}

impl ColdSegment {
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
                "cannot build empty cold segment".into(),
            ));
        }
        let dim = records[0].1.embedding_f32().len();
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
            QuantizerKind::RaBitQ { bits } => VectorQuantizer::RaBitQ(
                turbomemory_core::rabitq::RaBitQuantizer::new(
                    dim,
                    bits,
                    QuantizerKind::ROTATION_SEED,
                )
                .map_err(StorageError::Core)?,
            ),
        };
        let path = path.as_ref().to_path_buf();
        fs::create_dir_all(&path)?;

        let data_path = path.join(DATA_FILE);
        let mut writer = MmapFileWriter::new(&data_path);
        let mut offsets = Vec::with_capacity(records.len());
        for (offset, rec) in records {
            let encoded = quantizer
                .encode(rec.embedding_f32())
                .map_err(StorageError::Core)?;
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
                .map_err(|e| StorageError::InvalidArgument(format!("bad cold manifest: {e}")))?
        };

        let data_path = path.join(DATA_FILE);
        let buffer = MmapBuffer::open(&data_path)?;

        let expected_bytes = manifest
            .offsets
            .len()
            .checked_mul(manifest.quantizer.encoded_bytes_per_vector())
            .ok_or_else(|| StorageError::InvalidArgument("cold segment size overflow".into()))?;
        if buffer.len() < expected_bytes {
            return Err(StorageError::InvalidArgument(
                "cold segment data file is too small".into(),
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

impl VectorSegment for ColdSegment {
    fn tier(&self) -> Tier {
        Tier::Cold
    }

    fn insert(&mut self, _offset: PointOffset, _record: &Record) -> crate::Result<()> {
        Err(StorageError::InvalidArgument(
            "cold segments are immutable; insert into hot".into(),
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

        // Score vectors in batches so the sign-quantized batch kernel can
        // amortize the query lookup tables over many vectors.
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
                        tier: Tier::Cold,
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
                        tier: Tier::Cold,
                    }
                }));
            }
        }
        // Oversample the quantized shortlist before reranking: quantization
        // noise reshuffles near-orthogonal high-dim rankings, so widen the
        // candidate pool (scaled by dimension) so the full-f32 rerank can
        // recover true neighbors that landed past top_k under the quantized
        // score.
        let shortlist = (top_k.saturating_mul(crate::segments::rerank_oversample(self.dim)))
            .max(crate::segments::MIN_RERANK_SHORTLIST);
        let candidates = crate::segments::top_k_minheap(candidates.into_iter(), shortlist);

        let view = vectors.read_view();
        let mut offsets = Vec::with_capacity(candidates.len());
        let mut refs: Vec<&[f32]> = Vec::with_capacity(candidates.len());
        for c in &candidates {
            if let Some(v) = view.get(c.offset) {
                offsets.push(c.offset);
                refs.push(v);
            }
        }
        let scores = cosine_similarity_batch(query, &refs);
        let mut reranked: Vec<ScoredPoint> = offsets
            .into_iter()
            .zip(scores)
            .map(|(offset, score)| ScoredPoint {
                offset,
                score,
                tier: Tier::Cold,
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
        let path = self.path.clone();
        Box::new(move || {
            if fs::metadata(&path).is_ok() {
                Ok(())
            } else {
                Err(StorageError::InvalidArgument(
                    "cold segment file missing".into(),
                ))
            }
        })
    }
}
