//! Quantization primitives for tiered vector storage.

use crate::turbo_quant::{TurboQuantMseQuantizer, TurboQuantProdQuantizer};
use crate::{validate_dimension, Result, TurboError};
use serde::{Deserialize, Serialize};

/// A quantizer maps full-precision vectors to compact byte representations.
pub trait Quantizer: Send + Sync + Clone {
    /// Original vector dimension.
    fn dim(&self) -> usize;
    /// Number of bytes required for one encoded vector.
    fn encoded_bytes_per_vector(&self) -> usize;
    /// Encode a vector. `v.len()` must equal `dim()`.
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>>;
    /// Decode an encoded vector back to full precision.
    fn decode(&self, q: &[u8]) -> Result<Vec<f32>>;
}

/// Lloyd-Max centroids for a standard normal distribution at 1-8 bits.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LloydMaxTable {
    pub bits: u8,
    pub centroids: Vec<f32>,
}

impl LloydMaxTable {
    pub fn new(bits: u8) -> Result<Self> {
        if bits == 0 || bits > 8 {
            return Err(TurboError::QuantizationError(
                "bits must be in 1..=8".into(),
            ));
        }
        #[allow(clippy::excessive_precision)]
        let centroids = match bits {
            1 => vec![-0.7978846f32, 0.7978846f32],
            2 => vec![-1.510417, -0.452780, 0.452780, 1.510417],
            3 => vec![
                -2.151386, -1.343909, -0.799502, -0.366454, 0.366454, 0.799502, 1.343909, 2.151386,
            ],
            4 => vec![
                -2.732240, -2.069575, -1.618534, -1.256179, -0.942934, -0.659760, -0.393128,
                -0.135174, 0.135174, 0.393128, 0.659760, 0.942934, 1.256179, 1.618534, 2.069575,
                2.732240,
            ],
            _ => {
                let levels = 1usize << bits;
                let step = 6.0 / (levels as f32);
                (0..levels)
                    .map(|i| -3.0 + (i as f32 + 0.5) * step)
                    .collect()
            }
        };
        Ok(Self { bits, centroids })
    }
}

/// Per-calibration scalar quantizer.
///
/// After calibration it stores a global min/max and maps each coordinate linearly
/// to an unsigned integer.  This is used by the Warm tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    pub bits: u8,
    pub dim: usize,
    pub min: f32,
    pub max: f32,
}

impl ScalarQuantizer {
    pub fn calibrate(vectors: &[Vec<f32>], bits: u8) -> Result<Self> {
        if bits == 0 || bits > 8 {
            return Err(TurboError::QuantizationError(
                "bits must be in 1..=8".into(),
            ));
        }
        if vectors.is_empty() {
            return Err(TurboError::QuantizationError(
                "cannot calibrate on empty set".into(),
            ));
        }
        let dim = vectors[0].len();
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for v in vectors {
            validate_dimension(v, dim)?;
            for &x in v {
                if x < min {
                    min = x;
                }
                if x > max {
                    max = x;
                }
            }
        }
        Ok(Self {
            bits,
            dim,
            min,
            max,
        })
    }

    pub fn levels(&self) -> f32 {
        (1usize << self.bits).saturating_sub(1).max(1) as f32
    }

    pub fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        validate_dimension(v, self.dim)?;
        let levels = self.levels();
        let range = self.max - self.min;
        let scale = if range == 0.0 { 1.0 } else { levels / range };
        let mut out = vec![0u8; self.dim];
        for (i, &x) in v.iter().enumerate() {
            let q = ((x - self.min) * scale).round().clamp(0.0, levels) as u8;
            out[i] = q;
        }
        Ok(out)
    }

    pub fn decode(&self, q: &[u8]) -> Result<Vec<f32>> {
        if q.len() != self.dim {
            return Err(TurboError::DimensionMismatch {
                expected: self.dim,
                got: q.len(),
            });
        }
        let levels = self.levels();
        let range = self.max - self.min;
        let scale = if levels == 0.0 { 0.0 } else { range / levels };
        Ok(q.iter().map(|&x| self.min + (x as f32) * scale).collect())
    }

    /// Decode a single quantized coordinate.
    pub fn decode_coordinate(&self, q: u8) -> f32 {
        let levels = self.levels();
        let range = self.max - self.min;
        let scale = if levels == 0.0 { 0.0 } else { range / levels };
        self.min + (q as f32) * scale
    }
}

impl Quantizer for ScalarQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }
    fn encoded_bytes_per_vector(&self) -> usize {
        self.dim
    }
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        self.encode(v)
    }
    fn decode(&self, q: &[u8]) -> Result<Vec<f32>> {
        self.decode(q)
    }
}

/// Unified quantizer enum used by storage manifests.
///
/// Lets Warm/Cold segments switch between scalar, sign, and TurboQuant
/// quantizers without being generic over the concrete type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorQuantizer {
    Scalar(ScalarQuantizer),
    Sign(SignQuantizer),
    TurboQuantMse(TurboQuantMseQuantizer),
    TurboQuantProd(TurboQuantProdQuantizer),
}

impl Quantizer for VectorQuantizer {
    fn dim(&self) -> usize {
        match self {
            Self::Scalar(q) => q.dim(),
            Self::Sign(q) => q.dim(),
            Self::TurboQuantMse(q) => q.dim(),
            Self::TurboQuantProd(q) => q.dim(),
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

    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        match self {
            Self::Scalar(q) => q.encode(v),
            Self::Sign(q) => q.encode(v),
            Self::TurboQuantMse(q) => q.encode(v),
            Self::TurboQuantProd(q) => q.encode(v),
        }
    }

    fn decode(&self, encoded: &[u8]) -> Result<Vec<f32>> {
        match self {
            Self::Scalar(q) => q.decode(encoded),
            Self::Sign(q) => q.decode(encoded),
            Self::TurboQuantMse(q) => q.decode(encoded),
            Self::TurboQuantProd(q) => q.decode(encoded),
        }
    }
}

/// 1-bit sign quantizer.
///
/// Each dimension is reduced to its sign (+1 / -1) and packed into bits.
/// This is used by the Cold tier for maximum compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignQuantizer {
    pub dim: usize,
}

impl SignQuantizer {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }

    pub fn bytes_per_vector(dim: usize) -> usize {
        dim.div_ceil(8)
    }

    pub fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        validate_dimension(v, self.dim)?;
        let bytes = Self::bytes_per_vector(self.dim);
        let mut out = vec![0u8; bytes];
        for (i, &x) in v.iter().enumerate() {
            if x >= 0.0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
        Ok(out)
    }

    pub fn decode(&self, q: &[u8]) -> Result<Vec<f32>> {
        let expected = Self::bytes_per_vector(self.dim);
        if q.len() != expected {
            return Err(TurboError::DimensionMismatch {
                expected,
                got: q.len(),
            });
        }
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let byte = q[i / 8];
            let bit = byte & (1 << (i % 8)) != 0;
            out.push(if bit { 1.0 } else { -1.0 });
        }
        Ok(out)
    }

    /// Decode a single dimension from an encoded byte slice.
    pub fn decode_coordinate(q: &[u8], i: usize) -> f32 {
        let byte = q[i / 8];
        let bit = byte & (1 << (i % 8)) != 0;
        if bit {
            1.0
        } else {
            -1.0
        }
    }
}

impl Quantizer for SignQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }
    fn encoded_bytes_per_vector(&self) -> usize {
        Self::bytes_per_vector(self.dim)
    }
    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        self.encode(v)
    }
    fn decode(&self, q: &[u8]) -> Result<Vec<f32>> {
        self.decode(q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_quantizer_roundtrip() {
        let vectors: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 * 0.1, -(i as f32 * 0.1)])
            .collect();
        let q = ScalarQuantizer::calibrate(&vectors, 4).unwrap();
        let original = vec![0.35f32, -0.25];
        let quantized = q.encode(&original).unwrap();
        let reconstructed = q.decode(&quantized).unwrap();
        for (a, b) in original.iter().zip(&reconstructed) {
            assert!(
                (a - b).abs() < 0.1,
                "large reconstruction error: {a} vs {b}"
            );
        }
    }

    #[test]
    fn sign_quantizer_roundtrip() {
        let q = SignQuantizer::new(10);
        let v: Vec<f32> = (0..10)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let encoded = q.encode(&v).unwrap();
        assert_eq!(encoded.len(), 2);
        let decoded = q.decode(&encoded).unwrap();
        for (a, b) in v.iter().zip(&decoded) {
            assert_eq!(a.signum(), b.signum());
        }
    }

    #[test]
    fn sign_quantizer_packing() {
        let q = SignQuantizer::new(9);
        let v = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
        let encoded = q.encode(&v).unwrap();
        assert_eq!(encoded.len(), 2);
        let decoded = q.decode(&encoded).unwrap();
        assert_eq!(decoded[8], 1.0);
    }
}
