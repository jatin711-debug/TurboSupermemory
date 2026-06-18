//! Quantized distance computation via lookup tables and SIMD kernels.

use crate::quantization::{Quantizer, ScalarQuantizer, SignQuantizer, VectorQuantizer};
use crate::turbo_quant::{TurboQuantMseEncodedQuery, TurboQuantProdEncodedQuery};
use crate::Result;

/// An encoded query that can score a quantized vector without decoding it.
pub trait EncodedQuery: Send + Sync {
    /// Score a single encoded vector.
    fn score(&self, encoded: &[u8]) -> f32;

    /// Score a contiguous block of encoded vectors.
    ///
    /// The default implementation falls back to [`score`] in a loop. Implementors
    /// should override this when a batch kernel is available.
    fn score_batch(&self, encoded: &[u8]) -> Vec<f32> {
        let bytes_per_vec = self.encoded_bytes_per_vector();
        encoded
            .chunks_exact(bytes_per_vec)
            .map(|v| self.score(v))
            .collect()
    }

    /// Number of bytes per encoded vector. Needed by the default batch loop.
    fn encoded_bytes_per_vector(&self) -> usize;
}

/// SIMD-accelerated encoded query for [`ScalarQuantizer`].
pub struct ScalarEncodedQuery {
    dim: usize,
    query: Vec<f32>,
    min: f32,
    scale: f32,
}

impl ScalarEncodedQuery {
    pub fn new(query: &[f32], quantizer: &ScalarQuantizer) -> Result<Self> {
        crate::validate_dimension(query, quantizer.dim)?;
        let levels = quantizer.levels();
        let scale = if levels < f32::EPSILON {
            0.0
        } else {
            (quantizer.max - quantizer.min) / levels
        };
        Ok(Self {
            dim: quantizer.dim,
            query: query.to_vec(),
            min: quantizer.min,
            scale,
        })
    }
}

impl EncodedQuery for ScalarEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        crate::metrics_quantized::scalar_quantized_dot(
            &self.query,
            &encoded[..self.dim],
            self.min,
            self.scale,
        )
    }

    fn score_batch(&self, encoded: &[u8]) -> Vec<f32> {
        crate::metrics_quantized::scalar_quantized_dot_batch(
            &self.query,
            encoded,
            self.min,
            self.scale,
        )
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        self.dim
    }
}

/// Encoded query for [`SignQuantizer`] using a per-byte lookup table.
///
/// The table lets us score a 1-bit sign-quantized vector in O(dim/8) memory
/// accesses instead of the O(dim) bit-extraction loop used by the scalar
/// fallback. The table is built once per query and reused for every encoded
/// vector in the segment.
pub struct SignEncodedQuery {
    /// Per-byte sign mask derived from the query (bit set == query sign >= 0).
    mask: Vec<u8>,
    /// For each byte position, a 256-entry LUT that maps an XOR bit-pattern to
    /// the sum of |query_i| for the bits that are set in that pattern.
    weights: Vec<[f32; 256]>,
    /// Sum of absolute query values; used to convert weighted differing bits
    /// into the final signed dot product.
    sum_abs: f32,
}

impl SignEncodedQuery {
    pub fn new(query: &[f32], quantizer: &SignQuantizer) -> Result<Self> {
        crate::validate_dimension(query, quantizer.dim)?;
        let dim = quantizer.dim;
        let bytes = quantizer.encoded_bytes_per_vector();
        let mut mask = vec![0u8; bytes];
        let mut abs_per_dim = vec![0.0f32; dim];
        let mut sum_abs = 0.0f32;
        for (i, &q) in query.iter().enumerate() {
            let abs_q = q.abs();
            abs_per_dim[i] = abs_q;
            sum_abs += abs_q;
            if q >= 0.0 {
                mask[i / 8] |= 1 << (i % 8);
            }
        }

        // Build per-byte LUTs. For byte b and bit-pattern p, the entry is the
        // sum of abs(query_i) for all bits j of p that are set.
        let mut weights = vec![[0.0f32; 256]; bytes];
        for (b, row) in weights.iter_mut().enumerate().take(bytes) {
            let base = b * 8;
            for (pattern, cell) in row.iter_mut().enumerate() {
                let mut s = 0.0f32;
                for bit in 0..8usize {
                    let idx = base + bit;
                    if idx < dim && (pattern & (1 << bit)) != 0 {
                        s += abs_per_dim[idx];
                    }
                }
                *cell = s;
            }
        }

        Ok(Self {
            mask,
            weights,
            sum_abs,
        })
    }
}

impl EncodedQuery for SignEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        let mut sum_diff = 0.0f32;
        for ((mask_byte, encoded_byte), weight_row) in self
            .mask
            .iter()
            .zip(encoded.iter())
            .zip(self.weights.iter())
        {
            let xor = mask_byte ^ encoded_byte;
            sum_diff += weight_row[xor as usize];
        }
        self.sum_abs - 2.0 * sum_diff
    }

    fn score_batch(&self, encoded: &[u8]) -> Vec<f32> {
        crate::metrics_quantized::sign_quantized_score_batch(
            &self.mask,
            &self.weights,
            encoded,
            self.sum_abs,
        )
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        self.mask.len()
    }
}

/// A quantized vector store that can encode a query for fast scoring.
pub trait QuantizedStore {
    type Query: EncodedQuery;
    fn encode_query(&self, query: &[f32]) -> Result<Self::Query>;
}

impl QuantizedStore for ScalarQuantizer {
    type Query = ScalarEncodedQuery;
    fn encode_query(&self, query: &[f32]) -> Result<Self::Query> {
        ScalarEncodedQuery::new(query, self)
    }
}

impl QuantizedStore for SignQuantizer {
    type Query = SignEncodedQuery;
    fn encode_query(&self, query: &[f32]) -> Result<Self::Query> {
        SignEncodedQuery::new(query, self)
    }
}

/// Encoded-query enum that matches [`VectorQuantizer`].
pub enum AnyEncodedQuery {
    Scalar(ScalarEncodedQuery),
    Sign(SignEncodedQuery),
    TurboQuantMse(TurboQuantMseEncodedQuery),
    TurboQuantProd(TurboQuantProdEncodedQuery),
}

impl EncodedQuery for AnyEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        match self {
            Self::Scalar(q) => q.score(encoded),
            Self::Sign(q) => q.score(encoded),
            Self::TurboQuantMse(q) => q.score(encoded),
            Self::TurboQuantProd(q) => q.score(encoded),
        }
    }

    fn score_batch(&self, encoded: &[u8]) -> Vec<f32> {
        match self {
            Self::Scalar(q) => q.score_batch(encoded),
            Self::Sign(q) => q.score_batch(encoded),
            Self::TurboQuantMse(q) => EncodedQuery::score_batch(q, encoded),
            Self::TurboQuantProd(q) => q.score_batch(encoded),
        }
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        match self {
            Self::Scalar(q) => q.encoded_bytes_per_vector(),
            Self::Sign(q) => q.encoded_bytes_per_vector(),
            Self::TurboQuantMse(q) => q.encoded_bytes_per_vector(),
            Self::TurboQuantProd(q) => q.encoded_bytes_per_vector(),
        }
    }
}

impl QuantizedStore for VectorQuantizer {
    type Query = AnyEncodedQuery;

    fn encode_query(&self, query: &[f32]) -> Result<Self::Query> {
        match self {
            Self::Scalar(q) => Ok(AnyEncodedQuery::Scalar(ScalarEncodedQuery::new(query, q)?)),
            Self::Sign(q) => Ok(AnyEncodedQuery::Sign(SignEncodedQuery::new(query, q)?)),
            Self::TurboQuantMse(q) => Ok(AnyEncodedQuery::TurboQuantMse(q.encode_query(query)?)),
            Self::TurboQuantProd(q) => Ok(AnyEncodedQuery::TurboQuantProd(q.encode_query(query)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_lut_matches_decode_dot() {
        let vectors: Vec<Vec<f32>> = (0..5)
            .map(|i| vec![i as f32 * 0.2, -(i as f32 * 0.1)])
            .collect();
        let q = ScalarQuantizer::calibrate(&vectors, 4).unwrap();
        let query = vec![0.5f32, -0.3];
        let encoded = q.encode(&query).unwrap();
        let decoded = q.decode(&encoded).unwrap();
        let expected: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();
        let eq = q.encode_query(&query).unwrap();
        let actual = eq.score(&encoded);
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }

    #[test]
    fn sign_lut_matches_decode_dot() {
        let q = SignQuantizer::new(8);
        let query = vec![0.5f32, -0.3, 0.2, 0.9, -0.1, 0.4, -0.6, 0.7];
        let encoded = q.encode(&query).unwrap();
        let decoded = q.decode(&encoded).unwrap();
        let expected: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();
        let eq = q.encode_query(&query).unwrap();
        let actual = eq.score(&encoded);
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }
}
