//! Core primitives for TurboSuperMemory.
//!
//! Provides vector math, distance functions, and compression/quantization
//! building blocks (FWHT, Lloyd-Max tables, scalar/sign quantization).

pub mod metrics;
pub mod metrics_quantized;
pub mod quantization;
pub mod quantized_search;
pub mod rabitq;
pub mod turbo_quant;

pub use metrics::{
    cosine_distance, cosine_similarity, cosine_similarity_batch, dot_and_norms, dot_product,
    l2_distance_sq, maxsim_score, maxsim_score_batch, CosineMetric, DotProductMetric,
    EuclideanMetric, Metric,
};
pub use quantization::{LloydMaxTable, Quantizer, ScalarQuantizer, SignQuantizer, VectorQuantizer};
pub use quantized_search::{AnyEncodedQuery, EncodedQuery, QuantizedStore};
pub use rabitq::{RaBitQEncodedQuery, RaBitQuantizer};
pub use turbo_quant::{TurboQuantMseQuantizer, TurboQuantProdQuantizer};

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub type Vector = Vec<f32>;

#[derive(Debug, Error)]
pub enum TurboError {
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("vector is zero-norm")]
    ZeroNorm,
    #[error("vector contains NaN or infinite values")]
    NonFinite,
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

/// Validates that every component is finite (no NaN, no infinity).
///
/// A single non-finite component makes every similarity against the vector
/// NaN, which poisons ranking for every later query, so it is rejected at the
/// boundary instead of being stored.
pub fn validate_finite(vec: &[f32]) -> Result<()> {
    if vec.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(TurboError::NonFinite)
    }
}

/// Validates a query vector: expected dimension and all components finite.
pub fn validate_query(vec: &[f32], dim: usize) -> Result<()> {
    validate_dimension(vec, dim)?;
    validate_finite(vec)
}

/// In-place L2 normalization.
///
/// Returns [`TurboError::NonFinite`] for a vector containing NaN or infinity
/// and [`TurboError::ZeroNorm`] for an all-zero vector. Finite inputs whose
/// sum of squares would overflow or underflow are rescaled by their largest
/// magnitude first, so they still normalize to the right unit vector.
pub fn normalize(v: &mut [f32]) -> Result<()> {
    let mut max_abs = 0.0f32;
    for x in v.iter() {
        if !x.is_finite() {
            return Err(TurboError::NonFinite);
        }
        max_abs = max_abs.max(x.abs());
    }
    if max_abs == 0.0 {
        return Err(TurboError::ZeroNorm);
    }
    // Ordinary magnitudes: the plain norm is exact enough and keeps results
    // bit-identical to what earlier builds stored.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm.is_normal() {
        for x in v.iter_mut() {
            *x /= norm;
        }
        return Ok(());
    }
    // The sum of squares overflowed or underflowed: rescale first. Divide
    // (rather than multiply by a reciprocal) so a subnormal `max_abs` cannot
    // overflow the scale factor.
    let norm: f32 = v
        .iter()
        .map(|x| {
            let s = x / max_abs;
            s * s
        })
        .sum::<f32>()
        .sqrt();
    // `norm` is in [1, sqrt(len)] here: finite and non-zero by construction.
    for x in v.iter_mut() {
        *x = (*x / max_abs) / norm;
    }
    Ok(())
}

/// Total order for similarity scores, highest first, with NaN last.
///
/// Sorting scores with `partial_cmp(..).unwrap_or(Equal)` is not a total order
/// once a NaN is present, and the standard sort may panic on that. Use this
/// comparator for every "best first" sort so a stray NaN (for example from a
/// vector stored by an older build) sinks to the bottom instead.
pub fn cmp_score_desc(a: f32, b: f32) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (false, false) => b.total_cmp(&a),
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
    }
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

/// Preconditioning: random diagonal + FWHT, normalized to preserve L2 norm.
///
/// This implements a fast approximate random rotation: for a unit-norm input
/// the output is also unit-norm, and each coordinate is approximately
/// `N(0, 1/d)` distributed.  TurboQuant scales the Lloyd-Max centroids by
/// `1/sqrt(d)` to match this distribution.
pub fn precondition(v: &mut [f32], seed: u64) {
    random_diagonal_precondition(v, seed);
    fwht(v);
    let inv = 1.0 / (v.len() as f32).sqrt();
    for x in v.iter_mut() {
        *x *= inv;
    }
}

/// Rotate a vector to the TurboQuant coordinate system and return its original norm.
///
/// The input is L2-normalized, then a fast approximate random rotation is
/// applied.  The rotation is norm-preserving, so output coordinates are
/// approximately `N(0, 1/d)`.  The original norm is returned so callers can
/// rescale the decoded reconstruction.
pub fn rotate_to_turbo_quant_domain(v: &mut [f32], seed: u64) -> f32 {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
    precondition(v, seed);
    norm
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
    fn normalize_rejects_non_finite_and_zero() {
        let mut nan = vec![1.0f32, f32::NAN, 0.5];
        assert!(matches!(normalize(&mut nan), Err(TurboError::NonFinite)));
        let mut inf = vec![f32::INFINITY, 1.0];
        assert!(matches!(normalize(&mut inf), Err(TurboError::NonFinite)));
        let mut zero = vec![0.0f32; 4];
        assert!(matches!(normalize(&mut zero), Err(TurboError::ZeroNorm)));
        assert!(validate_query(&[1.0, f32::NAN], 2).is_err());
        assert!(validate_query(&[1.0, 2.0], 3).is_err());
        assert!(validate_query(&[1.0, 2.0], 2).is_ok());
    }

    #[test]
    fn normalize_survives_extreme_magnitudes() {
        // Squaring these overflows f32; the result must still be the unit vector.
        let mut big = vec![3.0e19f32, 4.0e19];
        normalize(&mut big).unwrap();
        assert!((big[0] - 0.6).abs() < 1e-6 && (big[1] - 0.8).abs() < 1e-6);
        // Squaring these underflows to zero.
        let mut tiny = vec![3.0e-30f32, 4.0e-30];
        normalize(&mut tiny).unwrap();
        assert!((tiny[0] - 0.6).abs() < 1e-6 && (tiny[1] - 0.8).abs() < 1e-6);
        // Subnormal input: a reciprocal of the maximum would overflow.
        let mut sub = vec![1.0e-45f32, 0.0];
        normalize(&mut sub).unwrap();
        assert!((sub[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cmp_score_desc_is_total_with_nan() {
        let mut scores = [0.2f32, f32::NAN, 0.9, -0.5, f32::NAN, 0.9];
        scores.sort_by(|a, b| cmp_score_desc(*a, *b));
        assert_eq!(&scores[..4], &[0.9, 0.9, 0.2, -0.5]);
        assert!(scores[4].is_nan() && scores[5].is_nan());
    }

    #[test]
    fn test_precondition_preserves_norm() {
        let mut v = vec![1.0f32, 0.0, 0.0, 0.0];
        precondition(&mut v, 42);
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_rotate_to_turbo_quant_domain() {
        let mut v = vec![3.0f32, 4.0, 0.0, 0.0];
        let norm = rotate_to_turbo_quant_domain(&mut v, 42);
        assert!((norm - 5.0).abs() < 1e-5);
        let out_norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((out_norm_sq - 1.0).abs() < 1e-5);
    }
}
