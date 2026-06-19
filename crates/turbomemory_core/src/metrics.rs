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

/// Batched cosine similarity between one query and many vectors.
///
/// All slices must have the same length.  Returns a vector of scores in
/// `[-1, 1]` parallel to `vectors`.
///
/// The query's squared norm is computed once for the whole batch, and vectors
/// are scored four at a time so each query SIMD load is reused across the
/// block (a blocked AVX2/FMA matrix-multiply with a four-accumulator unroll).
pub fn cosine_similarity_batch(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(vectors.len());
    if vectors.is_empty() {
        return out;
    }
    // ||query||^2 computed once instead of once per vector.
    let na = dot_product(query, query);

    let mut chunks = vectors.chunks_exact(4);
    for chunk in &mut chunks {
        let dn = dot_and_nb_x4(query, chunk[0], chunk[1], chunk[2], chunk[3]);
        for (dot, nb) in dn {
            out.push(finalize_cosine(dot, na, nb));
        }
    }
    for v in chunks.remainder() {
        let dot = dot_product(query, v);
        let nb = dot_product(v, v);
        out.push(finalize_cosine(dot, na, nb));
    }
    out
}

/// Combine a dot product and the two squared norms into a clamped cosine score.
#[inline]
fn finalize_cosine(dot: f32, norm_sq_a: f32, norm_sq_b: f32) -> f32 {
    let denom = (norm_sq_a * norm_sq_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        (dot / denom).clamp(-1.0, 1.0)
    }
}

/// Dot product and squared norm of four vectors against one query, in lockstep.
///
/// Returns `[(dot, ||v||^2); 4]`.  All five slices must have the same length.
#[inline]
fn dot_and_nb_x4(q: &[f32], v0: &[f32], v1: &[f32], v2: &[f32], v3: &[f32]) -> [(f32, f32); 4] {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            // SAFETY: guarded by runtime feature detection above.
            return unsafe { dot_and_nb_x4_avx2(q, v0, v1, v2, v3) };
        }
        if std::is_x86_feature_detected!("sse") {
            // SAFETY: guarded by runtime feature detection above.
            return unsafe { dot_and_nb_x4_sse(q, v0, v1, v2, v3) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: AArch64 guarantees NEON.
        return unsafe { dot_and_nb_x4_neon(q, v0, v1, v2, v3) };
    }
    dot_and_nb_x4_scalar(q, v0, v1, v2, v3)
}

fn dot_and_nb_x4_scalar(
    q: &[f32],
    v0: &[f32],
    v1: &[f32],
    v2: &[f32],
    v3: &[f32],
) -> [(f32, f32); 4] {
    let vs = [v0, v1, v2, v3];
    let mut dot = [0.0f32; 4];
    let mut nb = [0.0f32; 4];
    for (k, v) in vs.iter().enumerate() {
        let mut d = 0.0f32;
        let mut n = 0.0f32;
        for (x, y) in q.iter().zip(v.iter()) {
            d += x * y;
            n += y * y;
        }
        dot[k] = d;
        nb[k] = n;
    }
    [
        (dot[0], nb[0]),
        (dot[1], nb[1]),
        (dot[2], nb[2]),
        (dot[3], nb[3]),
    ]
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

    #[inline]
    unsafe fn hsum256(v: __m256) -> f32 {
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), v);
        tmp.iter().sum::<f32>()
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_and_nb_x4_avx2(
        q: &[f32],
        v0: &[f32],
        v1: &[f32],
        v2: &[f32],
        v3: &[f32],
    ) -> [(f32, f32); 4] {
        let len = q.len();
        let mut d0 = _mm256_setzero_ps();
        let mut d1 = _mm256_setzero_ps();
        let mut d2 = _mm256_setzero_ps();
        let mut d3 = _mm256_setzero_ps();
        let mut n0 = _mm256_setzero_ps();
        let mut n1 = _mm256_setzero_ps();
        let mut n2 = _mm256_setzero_ps();
        let mut n3 = _mm256_setzero_ps();
        let mut i = 0usize;
        while i + 8 <= len {
            let qv = _mm256_loadu_ps(q.as_ptr().add(i));
            let a0 = _mm256_loadu_ps(v0.as_ptr().add(i));
            let a1 = _mm256_loadu_ps(v1.as_ptr().add(i));
            let a2 = _mm256_loadu_ps(v2.as_ptr().add(i));
            let a3 = _mm256_loadu_ps(v3.as_ptr().add(i));
            d0 = _mm256_fmadd_ps(qv, a0, d0);
            d1 = _mm256_fmadd_ps(qv, a1, d1);
            d2 = _mm256_fmadd_ps(qv, a2, d2);
            d3 = _mm256_fmadd_ps(qv, a3, d3);
            n0 = _mm256_fmadd_ps(a0, a0, n0);
            n1 = _mm256_fmadd_ps(a1, a1, n1);
            n2 = _mm256_fmadd_ps(a2, a2, n2);
            n3 = _mm256_fmadd_ps(a3, a3, n3);
            i += 8;
        }
        let mut dot = [hsum256(d0), hsum256(d1), hsum256(d2), hsum256(d3)];
        let mut nb = [hsum256(n0), hsum256(n1), hsum256(n2), hsum256(n3)];
        while i < len {
            let qx = *q.get_unchecked(i);
            let x0 = *v0.get_unchecked(i);
            let x1 = *v1.get_unchecked(i);
            let x2 = *v2.get_unchecked(i);
            let x3 = *v3.get_unchecked(i);
            dot[0] += qx * x0;
            dot[1] += qx * x1;
            dot[2] += qx * x2;
            dot[3] += qx * x3;
            nb[0] += x0 * x0;
            nb[1] += x1 * x1;
            nb[2] += x2 * x2;
            nb[3] += x3 * x3;
            i += 1;
        }
        [
            (dot[0], nb[0]),
            (dot[1], nb[1]),
            (dot[2], nb[2]),
            (dot[3], nb[3]),
        ]
    }

    #[inline]
    unsafe fn hsum128(v: __m128) -> f32 {
        let mut tmp = [0.0f32; 4];
        _mm_storeu_ps(tmp.as_mut_ptr(), v);
        tmp.iter().sum::<f32>()
    }

    #[inline]
    #[target_feature(enable = "sse")]
    pub unsafe fn dot_and_nb_x4_sse(
        q: &[f32],
        v0: &[f32],
        v1: &[f32],
        v2: &[f32],
        v3: &[f32],
    ) -> [(f32, f32); 4] {
        let len = q.len();
        let mut d0 = _mm_setzero_ps();
        let mut d1 = _mm_setzero_ps();
        let mut d2 = _mm_setzero_ps();
        let mut d3 = _mm_setzero_ps();
        let mut n0 = _mm_setzero_ps();
        let mut n1 = _mm_setzero_ps();
        let mut n2 = _mm_setzero_ps();
        let mut n3 = _mm_setzero_ps();
        let mut i = 0usize;
        while i + 4 <= len {
            let qv = _mm_loadu_ps(q.as_ptr().add(i));
            let a0 = _mm_loadu_ps(v0.as_ptr().add(i));
            let a1 = _mm_loadu_ps(v1.as_ptr().add(i));
            let a2 = _mm_loadu_ps(v2.as_ptr().add(i));
            let a3 = _mm_loadu_ps(v3.as_ptr().add(i));
            d0 = _mm_add_ps(d0, _mm_mul_ps(qv, a0));
            d1 = _mm_add_ps(d1, _mm_mul_ps(qv, a1));
            d2 = _mm_add_ps(d2, _mm_mul_ps(qv, a2));
            d3 = _mm_add_ps(d3, _mm_mul_ps(qv, a3));
            n0 = _mm_add_ps(n0, _mm_mul_ps(a0, a0));
            n1 = _mm_add_ps(n1, _mm_mul_ps(a1, a1));
            n2 = _mm_add_ps(n2, _mm_mul_ps(a2, a2));
            n3 = _mm_add_ps(n3, _mm_mul_ps(a3, a3));
            i += 4;
        }
        let mut dot = [hsum128(d0), hsum128(d1), hsum128(d2), hsum128(d3)];
        let mut nb = [hsum128(n0), hsum128(n1), hsum128(n2), hsum128(n3)];
        while i < len {
            let qx = *q.get_unchecked(i);
            let x0 = *v0.get_unchecked(i);
            let x1 = *v1.get_unchecked(i);
            let x2 = *v2.get_unchecked(i);
            let x3 = *v3.get_unchecked(i);
            dot[0] += qx * x0;
            dot[1] += qx * x1;
            dot[2] += qx * x2;
            dot[3] += qx * x3;
            nb[0] += x0 * x0;
            nb[1] += x1 * x1;
            nb[2] += x2 * x2;
            nb[3] += x3 * x3;
            i += 1;
        }
        [
            (dot[0], nb[0]),
            (dot[1], nb[1]),
            (dot[2], nb[2]),
            (dot[3], nb[3]),
        ]
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

    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_and_nb_x4_neon(
        q: &[f32],
        v0: &[f32],
        v1: &[f32],
        v2: &[f32],
        v3: &[f32],
    ) -> [(f32, f32); 4] {
        let len = q.len();
        let mut d0 = vdupq_n_f32(0.0);
        let mut d1 = vdupq_n_f32(0.0);
        let mut d2 = vdupq_n_f32(0.0);
        let mut d3 = vdupq_n_f32(0.0);
        let mut n0 = vdupq_n_f32(0.0);
        let mut n1 = vdupq_n_f32(0.0);
        let mut n2 = vdupq_n_f32(0.0);
        let mut n3 = vdupq_n_f32(0.0);
        let mut i = 0usize;
        while i + 4 <= len {
            let qv = vld1q_f32(q.as_ptr().add(i));
            let a0 = vld1q_f32(v0.as_ptr().add(i));
            let a1 = vld1q_f32(v1.as_ptr().add(i));
            let a2 = vld1q_f32(v2.as_ptr().add(i));
            let a3 = vld1q_f32(v3.as_ptr().add(i));
            d0 = vfmaq_f32(d0, qv, a0);
            d1 = vfmaq_f32(d1, qv, a1);
            d2 = vfmaq_f32(d2, qv, a2);
            d3 = vfmaq_f32(d3, qv, a3);
            n0 = vfmaq_f32(n0, a0, a0);
            n1 = vfmaq_f32(n1, a1, a1);
            n2 = vfmaq_f32(n2, a2, a2);
            n3 = vfmaq_f32(n3, a3, a3);
            i += 4;
        }
        let mut dot = [
            vaddvq_f32(d0),
            vaddvq_f32(d1),
            vaddvq_f32(d2),
            vaddvq_f32(d3),
        ];
        let mut nb = [
            vaddvq_f32(n0),
            vaddvq_f32(n1),
            vaddvq_f32(n2),
            vaddvq_f32(n3),
        ];
        while i < len {
            let qx = *q.get_unchecked(i);
            let x0 = *v0.get_unchecked(i);
            let x1 = *v1.get_unchecked(i);
            let x2 = *v2.get_unchecked(i);
            let x3 = *v3.get_unchecked(i);
            dot[0] += qx * x0;
            dot[1] += qx * x1;
            dot[2] += qx * x2;
            dot[3] += qx * x3;
            nb[0] += x0 * x0;
            nb[1] += x1 * x1;
            nb[2] += x2 * x2;
            nb[3] += x3 * x3;
            i += 1;
        }
        [
            (dot[0], nb[0]),
            (dot[1], nb[1]),
            (dot[2], nb[2]),
            (dot[3], nb[3]),
        ]
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
        let dims = [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
        ];
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

    #[test]
    fn batch_cosine_matches_per_vector() {
        // Cover dims that straddle the SIMD width and batch sizes that exercise
        // the four-wide blocks plus a remainder.
        let dims = [1usize, 3, 8, 9, 15, 16, 17, 64, 65, 128];
        let counts = [0usize, 1, 2, 3, 4, 5, 7, 11];
        for &dim in &dims {
            for &n in &counts {
                let query: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
                let owned: Vec<Vec<f32>> = (0..n)
                    .map(|j| {
                        (0..dim)
                            .map(|i| ((i + j) as f32 * 0.21 + 0.5).cos())
                            .collect()
                    })
                    .collect();
                let refs: Vec<&[f32]> = owned.iter().map(|v| v.as_slice()).collect();
                let got = cosine_similarity_batch(&query, &refs);
                assert_eq!(got.len(), n);
                for (j, v) in owned.iter().enumerate() {
                    let expected = cosine_similarity(&query, v);
                    assert!(
                        (got[j] - expected).abs() < 1e-4,
                        "dim={dim} n={n} j={j} expected={expected} got={}",
                        got[j]
                    );
                }
            }
        }
    }

    #[test]
    fn batch_cosine_handles_zero_vector() {
        let query = vec![1.0f32, 2.0, 3.0, 4.0];
        let zero = vec![0.0f32; 4];
        let refs: Vec<&[f32]> = vec![&zero];
        let got = cosine_similarity_batch(&query, &refs);
        assert_eq!(got, vec![0.0]);
    }
}
