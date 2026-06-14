//! Cold tier: 1-bit sign-quantized vectors stored in a memory-mapped file.

use crate::config::{Flusher, Tier};
use crate::record::{PointOffset, Record};
use crate::segments::mmap_array::{MmapBuffer, MmapFileWriter};
use crate::segments::{ScoredPoint, VectorSegment};
use crate::vector_store::VectorStore;
use crate::StorageError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use turbomemory_core::quantization::{Quantizer, SignQuantizer};
use turbomemory_core::quantized_search::{EncodedQuery, QuantizedStore};
use turbomemory_core::{cosine_similarity, validate_dimension};

const DATA_FILE: &str = "data.bin";
const MANIFEST_FILE: &str = "manifest.json";

#[derive(Debug, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    dimension: usize,
    quantizer: SignQuantizer,
    offsets: Vec<PointOffset>,
}

/// Immutable Cold segment.
pub struct ColdSegment {
    dim: usize,
    quantizer: SignQuantizer,
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
    ) -> crate::Result<Self> {
        if records.is_empty() {
            return Err(StorageError::InvalidArgument(
                "cannot build empty cold segment".into(),
            ));
        }
        let dim = records[0].1.embedding_f32().len();
        let quantizer = SignQuantizer::new(dim);
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
        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| StorageError::Serialize(Box::new(bincode::ErrorKind::Custom(e.to_string()))))?;
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
                    tier: Tier::Cold,
                }
            })
            .collect();
        all.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let candidates: Vec<_> = all.into_iter().take(top_k).collect();

        let view = vectors.read_view();
        let mut reranked: Vec<ScoredPoint> = candidates
            .into_iter()
            .filter_map(|c| {
                view.get(c.offset).map(|v| ScoredPoint {
                    offset: c.offset,
                    score: cosine_similarity(query, v),
                    tier: Tier::Cold,
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
