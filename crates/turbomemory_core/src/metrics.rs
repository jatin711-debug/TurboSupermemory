//! SIMD-accelerated vector distance functions and metric abstractions.
//!
//! Provides runtime CPU feature dispatch for dot product, squared L2 distance,
//! cosine similarity, and cosine distance on `f32` slices. Architectures:
//!
//! * x86_64: AVX2/FMA, SSE fallbacks
//! * aarch64: NEON
//! * scalar fallback everywhere else

/// A metric over dense `f32` vectors.
///
/// `preprocess` is called once on stored vectors and queries when the metric
/// requires normalization (e.g. cosine).  Similarity scores are in `[-1, 1]`
/// for cosine and unbounded for dot / negative-squared-L2.
pub trait Metric: Send + Sync + 'static {
    /// In-place normalization required before similarity/distance calls.
    fn preprocess(v: &mut [f32]);

    /// Similarity between two already-preprocessed vectors.
    fn similarity(a: &[f32], b: &[f32]) -> f32;

    /// Distance derived from similarity.  Lower is closer.
    fn distance(a: &[f32], b: &[f32]) -> f32 {
        1.0 - Self::similarity(a, b)
    }
}

/// Cosine metric.  Vectors are L2-normalized by `preprocess`; afterwards
/// similarity is just the dot product.
#[derive(Clone, Copy, Debug, Default)]
pub struct CosineMetric;

impl Metric for CosineMetric {
    fn preprocess(v: &mut [f32]) {
        let _ = crate::normalize(v);
    }

    fn similarity(a: &[f32], b: &[f32]) -> f32 {
        dot_product(a, b)
    }
}

/// Dot-product (inner product) metric.  Assumes vectors are already in the
/// desired space; `preprocess` is a no-op.
#[derive(Clone, Copy, Debug, Default)]
pub struct DotProductMetric;

impl Metric for DotProductMetric {
    fn preprocess(_: &mut [f32]) {}

    fn similarity(a: &[f32], b: &[f32]) -> f32 {
        dot_product(a, b)
    }
}

/// Negative squared Euclidean metric.  Larger similarity means closer vectors.
#[derive(Clone, Copy, Debug, Default)]
pub struct EuclideanMetric;

impl Metric for EuclideanMetric {
    fn preprocess(_: &mut [f32]) {}

    fn similarity(a: &[f32], b: &[f32]) -> f32 {
        -l2_distance_sq(a, b)
    }
}

/// Dot product with runtime SIMD dispatch.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection above.
            return unsafe { dot_product_avx2(a, b) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection above.
            return unsafe { dot_product_sse(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON.
        return unsafe { dot_product_neon(a, b) };
    }
    dot_product_scalar(a, b)
}

/// Squared L2 distance with runtime SIMD dispatch.
#[inline]
pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { l2_distance_sq_avx2(a, b) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { l2_distance_sq_sse(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l2_distance_sq_neon(a, b) };
    }
    l2_distance_sq_scalar(a, b)
}

/// Dot product plus per-vector squared norms, used by cosine similarity.
#[inline]
pub fn dot_and_norms(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    debug_assert_eq!(a.len(), b.len());
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            return unsafe { dot_and_norms_avx2(a, b) };
        }
        if std::is_x86_feature_detected!("sse") {
            return unsafe { dot_and_norms_sse(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_and_norms_neon(a, b) };
    }
    dot_and_norms_scalar(a, b)
}

/// Cosine similarity in `[-1, 1]`.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (dot, norm_sq_a, norm_sq_b) = dot_and_norms(a, b);
    let denom = (norm_sq_a * norm_sq_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Cosine distance = `1 - cosine_similarity`.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_distance_sq_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn dot_and_norms_scalar(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }
    (dot, norm_a, norm_b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut dot = tmp.iter().sum::<f32>();
        while i < a.len() {
            dot += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        dot
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn l2_distance_sq_avx2(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
            i += 8;
        }
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut sum = tmp.iter().sum::<f32>();
        while i < a.len() {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += d * d;
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_and_norms_avx2(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let mut dot_acc = _mm256_setzero_ps();
        let mut na_acc = _mm256_setzero_ps();
        let mut nb_acc = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= a.len() {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
            na_acc = _mm256_fmadd_ps(va, va, na_acc);
            nb_acc = _mm256_fmadd_ps(vb, vb, nb_acc);
            i += 8;
        }
        let mut dot_tmp = [0.0f32; 8];
        let mut na_tmp = [0.0f32; 8];
        let mut nb_tmp = [0.0f32; 8];
        _mm256_storeu_ps(dot_tmp.as_mut_ptr(), dot_acc);
        _mm256_storeu_ps(na_tmp.as_mut_ptr(), na_acc);
        _mm256_storeu_ps(nb_tmp.as_mut_ptr(), nb_acc);
        let mut dot = dot_tmp.iter().sum::<f32>();
        let mut na = na_tmp.iter().sum::<f32>();
        let mut nb = nb_tmp.iter().sum::<f32>();
        while i < a.len() {
            let x = *a.get_unchecked(i);
            let y = *b.get_unchecked(i);
            dot += x * y;
            na += x * x;
            nb += y * y;
            i += 1;
        }
        (dot, na, nb)
    }

    #[inline]
    #[target_feature(enable = "sse")]
    pub unsafe fn dot_product_sse(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
            i += 4;
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut dot = tmp.iter().sum::<f32>();
        while i < a.len() {
            dot += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        dot
    }

    #[inline]
    #[target_feature(enable = "sse")]
    pub unsafe fn l2_distance_sq_sse(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
            i += 4;
        }
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), acc);
        let mut sum = tmp.iter().sum::<f32>();
        while i < a.len() {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += d * d;
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "sse")]
    pub unsafe fn dot_and_norms_sse(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let mut dot_acc = _mm_setzero_ps();
        let mut na_acc = _mm_setzero_ps();
        let mut nb_acc = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            dot_acc = _mm_add_ps(dot_acc, _mm_mul_ps(va, vb));
            na_acc = _mm_add_ps(na_acc, _mm_mul_ps(va, va));
            nb_acc = _mm_add_ps(nb_acc, _mm_mul_ps(vb, vb));
            i += 4;
        }
        let mut dot_tmp = [0.0f32; 4];
        let mut na_tmp = [0.0f32; 4];
        let mut nb_tmp = [0.0f32; 4];
        _mm_storeu_ps(dot_tmp.as_mut_ptr(), dot_acc);
        _mm_storeu_ps(na_tmp.as_mut_ptr(), na_acc);
        _mm_storeu_ps(nb_tmp.as_mut_ptr(), nb_acc);
        let mut dot = dot_tmp.iter().sum::<f32>();
        let mut na = na_tmp.iter().sum::<f32>();
        let mut nb = nb_tmp.iter().sum::<f32>();
        while i < a.len() {
            let x = *a.get_unchecked(i);
            let y = *b.get_unchecked(i);
            dot += x * y;
            na += x * x;
            nb += y * y;
            i += 1;
        }
        (dot, na, nb)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use x86::*;

#[cfg(target_arch = "aarch64")]
mod aarch64 {
    use std::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            acc = vfmaq_f32(acc, va, vb);
            i += 4;
        }
        let mut dot = vaddvq_f32(acc);
        while i < a.len() {
            dot += *a.get_unchecked(i) * *b.get_unchecked(i);
            i += 1;
        }
        dot
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn l2_distance_sq_neon(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, diff, diff);
            i += 4;
        }
        let mut sum = vaddvq_f32(acc);
        while i < a.len() {
            let d = *a.get_unchecked(i) - *b.get_unchecked(i);
            sum += d * d;
            i += 1;
        }
        sum
    }

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_and_norms_neon(a: &[f32], b: &[f32]) -> (f32, f32, f32) {
        let mut dot_acc = vdupq_n_f32(0.0);
        let mut na_acc = vdupq_n_f32(0.0);
        let mut nb_acc = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= a.len() {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            dot_acc = vfmaq_f32(dot_acc, va, vb);
            na_acc = vfmaq_f32(na_acc, va, va);
            nb_acc = vfmaq_f32(nb_acc, vb, vb);
            i += 4;
        }
        let mut dot = vaddvq_f32(dot_acc);
        let mut na = vaddvq_f32(na_acc);
        let mut nb = vaddvq_f32(nb_acc);
        while i < a.len() {
            let x = *a.get_unchecked(i);
            let y = *b.get_unchecked(i);
            dot += x * y;
            na += x * x;
            nb += y * y;
            i += 1;
        }
        (dot, na, nb)
    }
}

#[cfg(target_arch = "aarch64")]
use aarch64::*;

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f32;
        let mut na = 0.0f32;
        let mut nb = 0.0f32;
        for (x, y) in a.iter().zip(b) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        let denom = na.sqrt() * nb.sqrt();
        if denom == 0.0 {
            0.0
        } else {
            (dot / denom).clamp(-1.0, 1.0)
        }
    }

    #[test]
    fn cosine_against_reference_across_dims() {
        let dims = [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129];
        for &dim in &dims {
            let a: Vec<f32> = (0..dim).map(|i| (i as f32).sin()).collect();
            let b: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0).cos()).collect();
            let expected = reference_cosine(&a, &b);
            let got = cosine_similarity(&a, &b);
            assert!(
                (got - expected).abs() < 1e-4,
                "dim={dim} expected={expected} got={got}"
            );
        }
    }

    #[test]
    fn dot_and_l2_against_reference() {
        let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..64).map(|i| (63 - i) as f32 * 0.1).collect();
        let dot_ref: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let l2_ref: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).powi(2)).sum();
        assert!((dot_product(&a, &b) - dot_ref).abs() < 1e-3);
        assert!((l2_distance_sq(&a, &b) - l2_ref).abs() < 1e-3);
    }

    #[test]
    fn cosine_metric_preprocess() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        CosineMetric::preprocess(&mut v);
        let norm_sq: f32 = v.iter().map(|x| x * x).sum();
        assert!((norm_sq - 1.0).abs() < 1e-5);
    }
}
