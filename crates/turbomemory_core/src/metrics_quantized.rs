//! Quantized distance kernels with SIMD dispatch.
//!
//! These kernels score quantized vectors directly without decoding them to
//! full `f32` first.  They are used by the Warm (scalar u8) and Cold (1-bit
//! sign) tiers.

/// Compute the dot product between a full-precision query and a scalar-quantized
/// vector.  The quantized code `c` represents `min + c as f32 * scale`.
pub fn scalar_quantized_dot(query: &[f32], encoded: &[u8], min: f32, scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { scalar_quantized_dot_avx2(query, encoded, min, scale) };
        }
        if is_x86_feature_detected!("sse4.1") {
            return scalar_quantized_dot_sse(query, encoded, min, scale);
        }
    }
    scalar_quantized_dot_scalar(query, encoded, min, scale)
}

fn scalar_quantized_dot_scalar(query: &[f32], encoded: &[u8], min: f32, scale: f32) -> f32 {
    let mut sum = 0.0f32;
    for (i, &code) in encoded.iter().enumerate() {
        let value = min + (code as f32) * scale;
        sum += query[i] * value;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scalar_quantized_dot_avx2(query: &[f32], encoded: &[u8], min: f32, scale: f32) -> f32 {
    use std::arch::x86_64::*;

    let len = query.len().min(encoded.len());
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();
    let v_min = _mm256_set1_ps(min);
    let v_scale = _mm256_set1_ps(scale);

    while i + 8 <= len {
        // Load 8 u8 codes and zero-extend to 8 i32, then to 8 f32.
        let codes = _mm_loadl_epi64(encoded.as_ptr().add(i) as *const _);
        let codes_i32 = _mm256_cvtepu8_epi32(codes);
        let codes_f32 = _mm256_cvtepi32_ps(codes_i32);
        let values = _mm256_fmadd_ps(codes_f32, v_scale, v_min);

        let q = _mm256_loadu_ps(query.as_ptr().add(i));
        acc = _mm256_fmadd_ps(q, values, acc);

        i += 8;
    }

    let mut sum = hsum256_ps(acc);
    while i < len {
        let value = min + (encoded[i] as f32) * scale;
        sum += query[i] * value;
        i += 1;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
fn scalar_quantized_dot_sse(query: &[f32], encoded: &[u8], min: f32, scale: f32) -> f32 {
    // SSE path without AVX2: fall back to scalar for simplicity.  A dedicated
    // SSE4.1 implementation can be added later if profiling shows it matters.
    scalar_quantized_dot_scalar(query, encoded, min, scale)
}

#[cfg(target_arch = "x86_64")]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum = _mm_add_ps(lo, hi);
    let sum = _mm_hadd_ps(sum, sum);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

/// Compute the dot product between a full-precision query and a 1-bit
/// sign-quantized vector.  Each set bit represents +1, each clear bit -1.
pub fn sign_quantized_dot(query: &[f32], encoded: &[u8]) -> f32 {
    let dim = query.len();
    debug_assert_eq!(encoded.len(), dim.div_ceil(8));

    // Build a byte mask of query signs so we can use popcount.
    let mut query_mask = vec![0u8; encoded.len()];
    let mut sum_pos = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        if q >= 0.0 {
            query_mask[i / 8] |= 1 << (i % 8);
            sum_pos += q;
        } else {
            sum_pos -= q;
        }
    }

    // Where query and encoded signs differ, the contribution flips from +|q|
    // to -|q|, i.e. a loss of 2|q|.  So score = sum_pos - 2 * sum_differing.
    let mut sum_diff_abs = 0.0f32;
    for i in 0..dim {
        let byte = encoded[i / 8];
        let encoded_bit = byte & (1 << (i % 8)) != 0;
        let query_bit = query_mask[i / 8] & (1 << (i % 8)) != 0;
        if encoded_bit != query_bit {
            sum_diff_abs += query[i].abs();
        }
    }

    sum_pos - 2.0 * sum_diff_abs
}

/// Score a contiguous block of `N` scalar-quantized vectors against `query`.
///
/// `encoded` must contain `N * query.len()` bytes. The per-vector SIMD kernel
/// is reused internally, but the batch interface removes iterator/closure
/// overhead from the caller.
pub fn scalar_quantized_dot_batch(query: &[f32], encoded: &[u8], min: f32, scale: f32) -> Vec<f32> {
    let dim = query.len();
    assert_eq!(
        encoded.len() % dim,
        0,
        "encoded length must be a multiple of query dimension"
    );
    let n = encoded.len() / dim;
    let mut scores = Vec::with_capacity(n);
    for vector in encoded.chunks_exact(dim) {
        scores.push(scalar_quantized_dot(query, vector, min, scale));
    }
    scores
}

/// Score a contiguous block of `N` 1-bit sign-quantized vectors using a
/// precomputed query sign mask and per-byte weight table.
///
/// `encoded` must contain `N * mask.len()` bytes. This is the fast path used by
/// [`SignEncodedQuery`](crate::quantized_search::SignEncodedQuery).
pub fn sign_quantized_score_batch(
    mask: &[u8],
    weights: &[[f32; 256]],
    encoded: &[u8],
    sum_abs: f32,
) -> Vec<f32> {
    let bytes_per_vec = mask.len();
    assert_eq!(
        encoded.len() % bytes_per_vec,
        0,
        "encoded length must be a multiple of bytes-per-vector"
    );
    let n = encoded.len() / bytes_per_vec;
    let mut scores = Vec::with_capacity(n);
    for vector in encoded.chunks_exact(bytes_per_vec) {
        let mut sum_diff = 0.0f32;
        for (b, &encoded_byte) in vector.iter().enumerate() {
            let xor = mask[b] ^ encoded_byte;
            sum_diff += weights[b][xor as usize];
        }
        scores.push(sum_abs - 2.0 * sum_diff);
    }
    scores
}

/// Score a contiguous block of `N` 1-bit sign-quantized vectors against `query`.
///
/// `encoded` must contain `N * query.len().div_ceil(8)` bytes. This is faster
/// than calling [`sign_quantized_dot`] in a loop because the query sign mask
/// and per-byte absolute weights are computed once.
pub fn sign_quantized_dot_batch(query: &[f32], encoded: &[u8]) -> Vec<f32> {
    let dim = query.len();
    let bytes_per_vec = dim.div_ceil(8);

    // Build query sign mask and per-byte absolute weights.
    let mut query_mask = vec![0u8; bytes_per_vec];
    let mut abs_weights = vec![[0.0f32; 256]; bytes_per_vec];
    let mut sum_abs = 0.0f32;
    for (i, &q) in query.iter().enumerate() {
        let abs_q = q.abs();
        sum_abs += abs_q;
        let byte = i / 8;
        let bit = i % 8;
        if q >= 0.0 {
            query_mask[byte] |= 1 << bit;
        }
        for (pattern, cell) in abs_weights[byte].iter_mut().enumerate() {
            if (pattern & (1 << bit)) != 0 {
                *cell += abs_q;
            }
        }
    }

    sign_quantized_score_batch(&query_mask, &abs_weights, encoded, sum_abs)
}

/// Batched cosine similarity between a query and a slice of full-precision
/// vectors.  All vectors must be the same length as `query`.
pub fn batched_cosine_similarity(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    vectors
        .iter()
        .map(|v| crate::cosine_similarity(query, v))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_quantized_dot_matches_reference() {
        let dim = 17;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let encoded: Vec<u8> = (0..dim).map(|i| ((i * 13) % 256) as u8).collect();
        let min = -1.0f32;
        let scale = 2.0f32 / 255.0;

        let expected = scalar_quantized_dot_scalar(&query, &encoded, min, scale);
        let actual = scalar_quantized_dot(&query, &encoded, min, scale);
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }

    #[test]
    fn sign_quantized_dot_matches_reference() {
        let dim = 10;
        let query = vec![0.5f32, -0.3, 0.2, 0.9, -0.1, 0.4, -0.6, 0.7, 0.1, -0.8];
        let encoded = vec![0b1010_0101u8, 0b0000_0001];

        // Decode manually and compute dot product.
        let mut decoded = Vec::with_capacity(dim);
        for i in 0..dim {
            let byte = encoded[i / 8];
            let bit = byte & (1 << (i % 8)) != 0;
            decoded.push(if bit { 1.0 } else { -1.0 });
        }
        let expected: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();
        let actual = sign_quantized_dot(&query, &encoded);
        assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
    }

    #[test]
    fn scalar_quantized_dot_batch_matches_per_vector() {
        let dim = 17;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let encoded: Vec<u8> = (0..(dim * 5)).map(|i| ((i * 13) % 256) as u8).collect();
        let min = -1.0f32;
        let scale = 2.0f32 / 255.0;

        let actual = scalar_quantized_dot_batch(&query, &encoded, min, scale);
        let expected: Vec<f32> = encoded
            .chunks_exact(dim)
            .map(|v| scalar_quantized_dot_scalar(&query, v, min, scale))
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-4, "{a} vs {e}");
        }
    }

    #[test]
    fn sign_quantized_dot_batch_matches_per_vector() {
        let dim: usize = 13;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.2 - 1.2).collect();
        let bytes_per_vec = dim.div_ceil(8);
        let encoded: Vec<u8> = (0..(bytes_per_vec * 7)).map(|i| (i % 256) as u8).collect();

        let actual = sign_quantized_dot_batch(&query, &encoded);
        let expected: Vec<f32> = encoded
            .chunks_exact(bytes_per_vec)
            .map(|v| sign_quantized_dot(&query, v))
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(&expected) {
            assert!((a - e).abs() < 1e-4, "{a} vs {e}");
        }
    }
}
