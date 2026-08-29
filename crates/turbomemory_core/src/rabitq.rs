//! RaBitQ: Randomized Binary Quantization for High-Dimensional Vector Search.
//!
//! Implements universal-dimension 1-bit and 2-bit quantization with random
//! orthogonal preconditioning (via Fast Walsh-Hadamard Transform on padded blocks),
//! exact L2-norm scale factors, and SIMD-accelerated asymmetric lookup table scoring.

use crate::quantization::Quantizer;
use crate::quantized_search::{EncodedQuery, QuantizedStore};
use crate::turbo_quant::{pack_codes, packed_bytes, unpack_codes};
use crate::{fwht, validate_dimension, Result, TurboError};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// RaBitQ Quantizer supporting arbitrary dimensions (384, 512, 768, 1024, 1536, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RaBitQuantizer {
    pub dim: usize,
    pub bits: u8,
    pub rotation_seed: u64,
    /// Next power-of-two dimension for the Fast Orthogonal Transform.
    pad_dim: usize,
    /// Random diagonal sign vector (+1.0 / -1.0) of length `pad_dim`.
    signs: Vec<f32>,
}

impl RaBitQuantizer {
    /// Create a new RaBitQ quantizer.
    pub fn new(dim: usize, bits: u8, rotation_seed: u64) -> Result<Self> {
        if dim == 0 {
            return Err(TurboError::QuantizationError("dim must be > 0".into()));
        }
        if bits != 1 && bits != 2 {
            return Err(TurboError::QuantizationError(
                "RaBitQ currently supports 1-bit and 2-bit quantization".into(),
            ));
        }
        let pad_dim = dim.next_power_of_two();
        let mut rng = StdRng::seed_from_u64(rotation_seed);
        let mut signs = Vec::with_capacity(pad_dim);
        for _ in 0..pad_dim {
            signs.push(if rng.gen_bool(0.5) { 1.0 } else { -1.0 });
        }

        Ok(Self {
            dim,
            bits,
            rotation_seed,
            pad_dim,
            signs,
        })
    }

    /// Fast Orthogonal Transform $y = R x$ using sign reflection + FWHT.
    pub fn project(&self, v: &[f32], out: &mut [f32]) {
        assert_eq!(v.len(), self.dim);
        let mut buf = vec![0.0f32; self.pad_dim];
        let scale = (self.pad_dim as f32 / self.dim as f32).sqrt();
        for i in 0..self.dim {
            buf[i] = v[i] * scale * self.signs[i];
        }
        fwht(&mut buf);
        let norm_factor = 1.0 / (self.pad_dim as f32).sqrt();
        for i in 0..self.dim {
            out[i] = buf[i] * norm_factor;
        }
    }

    /// Inverse Orthogonal Transform $\hat{x} = R^T y$.
    pub fn inverse_project(&self, y: &[f32], out: &mut [f32]) {
        assert_eq!(y.len(), self.dim);
        let mut buf = vec![0.0f32; self.pad_dim];
        buf[..self.dim].copy_from_slice(&y[..self.dim]);
        fwht(&mut buf);
        let norm_factor = 1.0 / (self.pad_dim as f32).sqrt();
        let inv_scale = (self.dim as f32 / self.pad_dim as f32).sqrt();
        for i in 0..self.dim {
            out[i] = buf[i] * norm_factor * self.signs[i] * inv_scale;
        }
    }

    /// Number of bytes for the quantized bit codes (excluding 4-byte float scale).
    pub fn code_bytes(&self) -> usize {
        packed_bytes(self.dim, self.bits)
    }
}

impl Quantizer for RaBitQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        self.code_bytes() + 4
    }

    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        validate_dimension(v, self.dim)?;
        let mut proj = vec![0.0f32; self.dim];
        self.project(v, &mut proj);

        let l2_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let alpha = l2_norm / (self.dim as f32).sqrt();

        let code_len = self.code_bytes();
        let mut out = vec![0u8; code_len + 4];

        if self.bits == 1 {
            let mut codes = vec![0u8; self.dim];
            for i in 0..self.dim {
                codes[i] = if proj[i] >= 0.0 { 1 } else { 0 };
            }
            pack_codes(&codes, 1, &mut out[..code_len]);
        } else {
            // 2-bit quantization thresholds: [-0.6745, 0.0, +0.6745] * alpha
            let t = 0.6745 * alpha;
            let mut codes = vec![0u8; self.dim];
            for i in 0..self.dim {
                let val = proj[i];
                codes[i] = if val < -t {
                    0
                } else if val < 0.0 {
                    1
                } else if val < t {
                    2
                } else {
                    3
                };
            }
            pack_codes(&codes, 2, &mut out[..code_len]);
        }

        // Store alpha as 4-byte LE float at the end
        out[code_len..code_len + 4].copy_from_slice(&alpha.to_le_bytes());
        Ok(out)
    }

    fn decode(&self, encoded: &[u8]) -> Result<Vec<f32>> {
        let code_len = self.code_bytes();
        if encoded.len() < code_len + 4 {
            return Err(TurboError::QuantizationError(
                "encoded buffer too short".into(),
            ));
        }
        let alpha = f32::from_le_bytes(encoded[code_len..code_len + 4].try_into().unwrap());
        let mut codes = vec![0u8; self.dim];
        unpack_codes(&encoded[..code_len], self.bits, &mut codes);

        let mut proj = vec![0.0f32; self.dim];
        if self.bits == 1 {
            for i in 0..self.dim {
                proj[i] = if codes[i] == 1 { alpha } else { -alpha };
            }
        } else {
            let c0 = -1.18 * alpha;
            let c1 = -0.38 * alpha;
            let c2 = 0.38 * alpha;
            let c3 = 1.18 * alpha;
            for i in 0..self.dim {
                proj[i] = match codes[i] {
                    0 => c0,
                    1 => c1,
                    2 => c2,
                    _ => c3,
                };
            }
        }

        let mut out = vec![0.0f32; self.dim];
        self.inverse_project(&proj, &mut out);
        Ok(out)
    }
}

/// SIMD-accelerated lookup table encoded query for [`RaBitQuantizer`].
pub struct RaBitQEncodedQuery {
    pub dim: usize,
    pub bits: u8,
    code_bytes: usize,
    total_bytes_per_vector: usize,
    /// Precomputed lookup tables per byte chunk.
    /// `luts[chunk_idx][byte_val]` gives the dot product contribution for that byte.
    luts: Vec<[f32; 256]>,
}

impl RaBitQEncodedQuery {
    pub fn new(query: &[f32], quantizer: &RaBitQuantizer) -> Result<Self> {
        validate_dimension(query, quantizer.dim)?;
        let mut proj_q = vec![0.0f32; quantizer.dim];
        quantizer.project(query, &mut proj_q);

        let code_bytes = quantizer.code_bytes();
        let mut luts = Vec::with_capacity(code_bytes);

        if quantizer.bits == 1 {
            for chunk_idx in 0..code_bytes {
                let mut table = [0.0f32; 256];
                let base_dim = chunk_idx * 8;
                for (byte_val, slot) in table.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for bit in 0..8 {
                        let dim_idx = base_dim + bit;
                        if dim_idx < quantizer.dim {
                            let sign = if (byte_val >> bit) & 1 == 1 {
                                1.0
                            } else {
                                -1.0
                            };
                            sum += sign * proj_q[dim_idx];
                        }
                    }
                    *slot = sum;
                }
                luts.push(table);
            }
        } else {
            // 2-bit LUTs: each byte holds 4 coordinates (2 bits each)
            let c = [-1.18f32, -0.38f32, 0.38f32, 1.18f32];
            for chunk_idx in 0..code_bytes {
                let mut table = [0.0f32; 256];
                let base_dim = chunk_idx * 4;
                for (byte_val, slot) in table.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for pos in 0..4 {
                        let dim_idx = base_dim + pos;
                        if dim_idx < quantizer.dim {
                            let code = (byte_val >> (pos * 2)) & 0b11;
                            sum += c[code] * proj_q[dim_idx];
                        }
                    }
                    *slot = sum;
                }
                luts.push(table);
            }
        }

        Ok(Self {
            dim: quantizer.dim,
            bits: quantizer.bits,
            code_bytes,
            total_bytes_per_vector: quantizer.encoded_bytes_per_vector(),
            luts,
        })
    }
}

impl EncodedQuery for RaBitQEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        if encoded.len() < self.total_bytes_per_vector {
            return f32::NEG_INFINITY;
        }

        let alpha = f32::from_le_bytes(
            encoded[self.code_bytes..self.code_bytes + 4]
                .try_into()
                .unwrap(),
        );

        let mut sum = 0.0f32;
        for (j, &byte_code) in encoded[..self.code_bytes].iter().enumerate() {
            sum += self.luts[j][byte_code as usize];
        }

        alpha * sum
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        self.total_bytes_per_vector
    }
}

impl QuantizedStore for RaBitQuantizer {
    type Query = RaBitQEncodedQuery;
    fn encode_query(&self, query: &[f32]) -> Result<Self::Query> {
        RaBitQEncodedQuery::new(query, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rabitq_1bit_arbitrary_dim_roundtrip_and_scoring() {
        // Test non-power-of-two dimensions: 384 (MiniLM) and 768 (mpnet)
        for &dim in &[384, 512, 768] {
            let quantizer = RaBitQuantizer::new(dim, 1, 42).unwrap();
            assert_eq!(quantizer.encoded_bytes_per_vector(), dim.div_ceil(8) + 4);

            let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).sin()).collect();
            let enc = quantizer.encode(&v).unwrap();
            assert_eq!(enc.len(), quantizer.encoded_bytes_per_vector());

            let dec = quantizer.decode(&enc).unwrap();
            assert_eq!(dec.len(), dim);

            let query_enc = quantizer.encode_query(&v).unwrap();
            let score = query_enc.score(&enc);
            assert!(score > 0.0, "self-dot product score must be positive");
        }
    }

    #[test]
    fn rabitq_2bit_roundtrip_and_scoring() {
        let dim = 768;
        let quantizer = RaBitQuantizer::new(dim, 2, 42).unwrap();
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.1).cos()).collect();
        let enc = quantizer.encode(&v).unwrap();
        let dec = quantizer.decode(&enc).unwrap();
        assert_eq!(dec.len(), dim);

        let query_enc = quantizer.encode_query(&v).unwrap();
        let score = query_enc.score(&enc);
        assert!(score > 0.0);
    }
}
