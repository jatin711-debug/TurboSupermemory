//! Quantized distance computation via lookup tables and SIMD kernels.

use crate::quantization::{ScalarQuantizer, SignQuantizer};
use crate::Result;

/// An encoded query that can score a quantized vector without decoding it.
pub trait EncodedQuery: Send + Sync {
    fn score(&self, encoded: &[u8]) -> f32;
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
}

/// SIMD-accelerated encoded query for [`SignQuantizer`].
pub struct SignEncodedQuery {
    dim: usize,
    query: Vec<f32>,
}

impl SignEncodedQuery {
    pub fn new(query: &[f32], quantizer: &SignQuantizer) -> Result<Self> {
        crate::validate_dimension(query, quantizer.dim)?;
        Ok(Self {
            dim: quantizer.dim,
            query: query.to_vec(),
        })
    }
}

impl EncodedQuery for SignEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        crate::metrics_quantized::sign_quantized_dot(&self.query, &encoded[..self.dim.div_ceil(8)])
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
