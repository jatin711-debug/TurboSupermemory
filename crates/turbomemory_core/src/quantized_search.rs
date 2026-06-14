//! Quantized distance computation via lookup tables.

use crate::quantization::{ScalarQuantizer, SignQuantizer};
use crate::Result;

/// An encoded query that can score a quantized vector without decoding it.
pub trait EncodedQuery: Send + Sync {
    fn score(&self, encoded: &[u8]) -> f32;
}

/// Lookup-table encoded query for [`ScalarQuantizer`].
///
/// For each dimension `i` and each possible code `c`, `lut[i * 256 + c]` stores
/// `query_i * dequantized(c)`.  The score is the sum over dimensions.
pub struct ScalarEncodedQuery {
    dim: usize,
    lut: Vec<f32>,
}

impl ScalarEncodedQuery {
    pub fn new(query: &[f32], quantizer: &ScalarQuantizer) -> Result<Self> {
        crate::validate_dimension(query, quantizer.dim)?;
        let dim = quantizer.dim;
        let mut lut = vec![0.0f32; dim * 256];
        for (i, &qi) in query.iter().enumerate() {
            let base = i * 256;
            for c in 0..256u32 {
                let value = quantizer.decode_coordinate(c as u8);
                lut[base + c as usize] = qi * value;
            }
        }
        Ok(Self { dim, lut })
    }
}

impl EncodedQuery for ScalarEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (i, &code) in encoded.iter().enumerate().take(self.dim) {
            sum += self.lut[i * 256 + code as usize];
        }
        sum
    }
}

/// Encoded query for [`SignQuantizer`].
pub struct SignEncodedQuery {
    dim: usize,
    pos: Vec<f32>,
    neg: Vec<f32>,
}

impl SignEncodedQuery {
    pub fn new(query: &[f32], quantizer: &SignQuantizer) -> Result<Self> {
        crate::validate_dimension(query, quantizer.dim)?;
        let dim = quantizer.dim;
        let mut pos = Vec::with_capacity(dim);
        let mut neg = Vec::with_capacity(dim);
        for &q in query {
            pos.push(q);
            neg.push(-q);
        }
        Ok(Self { dim, pos, neg })
    }
}

impl EncodedQuery for SignEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for i in 0..self.dim {
            let byte = encoded[i / 8];
            let bit = byte & (1 << (i % 8)) != 0;
            sum += if bit { self.pos[i] } else { self.neg[i] };
        }
        sum
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
