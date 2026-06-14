//! Core primitives for TurboSuperMemory.
//!
//! Provides vector math, distance functions, and compression/quantization
//! building blocks (FWHT, Lloyd-Max tables, scalar/sign quantization).

pub mod metrics;
pub mod metrics_quantized;
pub mod quantization;
pub mod quantized_search;

pub use metrics::{
    cosine_distance, cosine_similarity, cosine_similarity_batch, dot_and_norms, dot_product,
    l2_distance_sq, CosineMetric, DotProductMetric, EuclideanMetric, Metric,
};
pub use quantization::{LloydMaxTable, Quantizer, ScalarQuantizer, SignQuantizer};
pub use quantized_search::{EncodedQuery, QuantizedStore};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Vector = Vec<f32>;

#[derive(Debug, Error)]
pub enum TurboError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("vector is zero-norm")]
    ZeroNorm,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("quantization error: {0}")]
    QuantizationError(String),
}

pub type Result<T> = std::result::Result<T, TurboError>;

/// Validates that a slice has the expected dimension.
pub fn validate_dimension(vec: &[f32], dim: usize) -> Result<()> {
    if vec.len() != dim {
        Err(TurboError::DimensionMismatch {
            expected: dim,
            got: vec.len(),
        })
    } else {
        Ok(())
    }
}

/// In-place L2 normalization.
pub fn normalize(v: &mut [f32]) -> Result<()> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm == 0.0 {
        return Err(TurboError::ZeroNorm);
    }
    for x in v.iter_mut() {
        *x /= norm;
    }
    Ok(())
}

/// Fast Walsh-Hadamard Transform in O(d log d). `v.len()` must be a power of two.
pub fn fwht(v: &mut [f32]) {
    let n = v.len();
    assert!(n.is_power_of_two(), "FWHT requires power-of-two dimension");
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = v[j];
                let y = v[j + h];
                v[j] = x + y;
                v[j + h] = x - y;
            }
        }
        h *= 2;
    }
}

/// Optional randomized diagonal preconditioner used before FWHT.
pub fn random_diagonal_precondition(v: &mut [f32], seed: u64) {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(seed);
    for x in v.iter_mut() {
        *x *= if rng.gen::<bool>() { 1.0 } else { -1.0 };
    }
}

/// Preconditioning: random diagonal + FWHT. This spreads coordinate variance uniformly
/// and is the first step of the TurboQuant pipeline.
pub fn precondition(v: &mut [f32], seed: u64) {
    random_diagonal_precondition(v, seed);
    fwht(v);
    let inv = 1.0 / (v.len() as f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// A lightweight owned-or-borrowed vector view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorRef {
    Owned(Vec<f32>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_fwht_invertible() {
        let mut v = vec![1.0f32, 0.0, 0.0, 0.0];
        precondition(&mut v, 42);
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }
}
