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
/// `encoded` must contain `N * query.len()` bytes. Detects SIMD once and
/// processes vectors in an AVX2/FMA blocked kernel (four vectors at a time).
pub fn scalar_quantized_dot_batch(query: &[f32], encoded: &[u8], min: f32, scale: f32) -> Vec<f32> {
    let dim = query.len();
    assert_eq!(
        encoded.len() % dim,
        0,
        "encoded length must be a multiple of query dimension"
    );
    let n = encoded.len() / dim;
    let mut scores = vec![0.0f32; n];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                scalar_quantized_dot_batch_avx2(query, encoded, min, scale, &mut scores);
            }
            return scores;
        }
    }

    for (i, vector) in encoded.chunks_exact(dim).enumerate() {
        scores[i] = scalar_quantized_dot_scalar(query, vector, min, scale);
    }
    scores
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn scalar_quantized_dot_batch_avx2(
    query: &[f32],
    encoded: &[u8],
    min: f32,
    scale: f32,
    scores: &mut [f32],
) {
    use std::arch::x86_64::*;

    let dim = query.len();
    let n = scores.len();
    let v_min = _mm256_set1_ps(min);
    let v_scale = _mm256_set1_ps(scale);

    let mut vec_idx = 0usize;
    // Process 4 vectors at a time to amortize query loads and hide latency.
    while vec_idx + 4 <= n {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        let mut acc2 = _mm256_setzero_ps();
        let mut acc3 = _mm256_setzero_ps();

        let base0 = encoded.as_ptr().add(vec_idx * dim);
        let base1 = base0.add(dim);
        let base2 = base0.add(2 * dim);
        let base3 = base0.add(3 * dim);

        let mut j = 0usize;
        while j + 8 <= dim {
            let q = _mm256_loadu_ps(query.as_ptr().add(j));

            let codes0 = _mm_loadl_epi64(base0.add(j) as *const _);
            let codes1 = _mm_loadl_epi64(base1.add(j) as *const _);
            let codes2 = _mm_loadl_epi64(base2.add(j) as *const _);
            let codes3 = _mm_loadl_epi64(base3.add(j) as *const _);

            let vals0 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes0)),
                v_scale,
                v_min,
            );
            let vals1 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes1)),
                v_scale,
                v_min,
            );
            let vals2 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes2)),
                v_scale,
                v_min,
            );
            let vals3 = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes3)),
                v_scale,
                v_min,
            );

            acc0 = _mm256_fmadd_ps(q, vals0, acc0);
            acc1 = _mm256_fmadd_ps(q, vals1, acc1);
            acc2 = _mm256_fmadd_ps(q, vals2, acc2);
            acc3 = _mm256_fmadd_ps(q, vals3, acc3);

            j += 8;
        }

        scores[vec_idx] = hsum256_ps(acc0);
        scores[vec_idx + 1] = hsum256_ps(acc1);
        scores[vec_idx + 2] = hsum256_ps(acc2);
        scores[vec_idx + 3] = hsum256_ps(acc3);

        // Scalar tail for the four vectors.
        for (k, qk) in query.iter().enumerate().take(dim).skip(j) {
            scores[vec_idx] += qk * (min + (*base0.add(k) as f32) * scale);
            scores[vec_idx + 1] += qk * (min + (*base1.add(k) as f32) * scale);
            scores[vec_idx + 2] += qk * (min + (*base2.add(k) as f32) * scale);
            scores[vec_idx + 3] += qk * (min + (*base3.add(k) as f32) * scale);
        }

        vec_idx += 4;
    }

    // Remaining vectors one at a time.
    while vec_idx < n {
        let mut acc = _mm256_setzero_ps();
        let base = encoded.as_ptr().add(vec_idx * dim);
        let mut j = 0usize;
        while j + 8 <= dim {
            let q = _mm256_loadu_ps(query.as_ptr().add(j));
            let codes = _mm_loadl_epi64(base.add(j) as *const _);
            let vals = _mm256_fmadd_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes)),
                v_scale,
                v_min,
            );
            acc = _mm256_fmadd_ps(q, vals, acc);
            j += 8;
        }
        scores[vec_idx] = hsum256_ps(acc);
        for (k, qk) in query.iter().enumerate().take(dim).skip(j) {
            scores[vec_idx] += qk * (min + (*base.add(k) as f32) * scale);
        }
        vec_idx += 1;
    }
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

    // Fast path: if every byte position has the same weight table (which
    // happens when all |query_i| are equal, e.g. after L2 normalization and
    // certain symmetric queries), we can reduce the inner loop to a popcount.
    // Detect this by comparing table pointers/contents for the first byte.
    let uniform_weight = weights.first().filter(|&first| {
        weights
            .iter()
            .all(|w| std::ptr::eq(w.as_ptr(), first.as_ptr()) || w == first)
    });

    if let Some(uniform) = uniform_weight {
        if uniform.iter().enumerate().skip(1).all(|(p, w)| {
            let popcnt = p.count_ones() as f32;
            (*w - popcnt * uniform[1]).abs() < f32::EPSILON
        }) {
            // weight[pattern] = popcount(pattern) * unit_weight, unit_weight = uniform[1].
            let unit_weight = uniform[1];
            return sign_quantized_score_batch_popcount(mask, encoded, sum_abs, unit_weight);
        }
    }

    // General path: 8-vector unroll with independent accumulators to hide
    // LUT load latency.
    let mut i = 0usize;
    while i + 8 <= n {
        let v: [&[u8]; 8] =
            std::array::from_fn(|k| &encoded[(i + k) * bytes_per_vec..(i + k + 1) * bytes_per_vec]);
        let mut diff = [0.0f32; 8];
        for b in 0..bytes_per_vec {
            let m = mask[b];
            let w = &weights[b];
            for (vk, diffk) in v.iter().zip(diff.iter_mut()) {
                let xor = vk[b] ^ m;
                *diffk += w[xor as usize];
            }
        }
        for diffk in diff {
            scores.push(sum_abs - 2.0 * diffk);
        }
        i += 8;
    }

    for vector in encoded[i * bytes_per_vec..].chunks_exact(bytes_per_vec) {
        let mut sum_diff = 0.0f32;
        for (b, &encoded_byte) in vector.iter().enumerate() {
            let xor = mask[b] ^ encoded_byte;
            sum_diff += weights[b][xor as usize];
        }
        scores.push(sum_abs - 2.0 * sum_diff);
    }
    scores
}

/// Popcount-only sign scoring used when every bit contributes the same weight.
/// This is the fastest possible path for 1-bit quantized data.
fn sign_quantized_score_batch_popcount(
    mask: &[u8],
    encoded: &[u8],
    sum_abs: f32,
    unit_weight: f32,
) -> Vec<f32> {
    let bytes_per_vec = mask.len();
    let n = encoded.len() / bytes_per_vec;
    let mut scores = Vec::with_capacity(n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("popcnt") {
            return unsafe {
                sign_quantized_score_batch_popcount_x86(mask, encoded, sum_abs, unit_weight)
            };
        }
    }

    for vector in encoded.chunks_exact(bytes_per_vec) {
        let mut differing_bits: u32 = 0;
        for b in 0..bytes_per_vec {
            differing_bits += (mask[b] ^ vector[b]).count_ones();
        }
        scores.push(sum_abs - 2.0 * (differing_bits as f32) * unit_weight);
    }
    scores
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "popcnt")]
unsafe fn sign_quantized_score_batch_popcount_x86(
    mask: &[u8],
    encoded: &[u8],
    sum_abs: f32,
    unit_weight: f32,
) -> Vec<f32> {
    let bytes_per_vec = mask.len();
    let n = encoded.len() / bytes_per_vec;
    let mut scores = Vec::with_capacity(n);
    let two_unit = -2.0 * unit_weight;

    for vector in encoded.chunks_exact(bytes_per_vec) {
        let mut differing_bits: u32 = 0;
        // Process 8 bytes at a time when possible.
        let mut b = 0usize;
        while b + 8 <= bytes_per_vec {
            let m = u64::from_le_bytes([
                *mask.get_unchecked(b),
                *mask.get_unchecked(b + 1),
                *mask.get_unchecked(b + 2),
                *mask.get_unchecked(b + 3),
                *mask.get_unchecked(b + 4),
                *mask.get_unchecked(b + 5),
                *mask.get_unchecked(b + 6),
                *mask.get_unchecked(b + 7),
            ]);
            let e = u64::from_le_bytes([
                *vector.get_unchecked(b),
                *vector.get_unchecked(b + 1),
                *vector.get_unchecked(b + 2),
                *vector.get_unchecked(b + 3),
                *vector.get_unchecked(b + 4),
                *vector.get_unchecked(b + 5),
                *vector.get_unchecked(b + 6),
                *vector.get_unchecked(b + 7),
            ]);
            differing_bits += (m ^ e).count_ones();
            b += 8;
        }
        while b < bytes_per_vec {
            differing_bits += (mask.get_unchecked(b) ^ vector.get_unchecked(b)).count_ones();
            b += 1;
        }
        scores.push(sum_abs + (differing_bits as f32) * two_unit);
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

/// Score a contiguous block of `TurboQuantMse` encoded vectors.
///
/// This fallback handles bit widths that do not divide a byte. Common bit
/// widths should use [`turbo_quant_mse_score_batch_byte_weights`] instead.
pub fn turbo_quant_mse_score_batch(
    dim: usize,
    bits: u8,
    bytes_per_vec: usize,
    levels: usize,
    lut: &[f32],
    encoded: &[u8],
) -> Vec<f32> {
    if bytes_per_vec == 0 || !encoded.len().is_multiple_of(bytes_per_vec) {
        return Vec::new();
    }
    let n = encoded.len() / bytes_per_vec;
    let mut scores = Vec::with_capacity(n);
    let mut codes = vec![0u8; dim];

    for vector in encoded.chunks_exact(bytes_per_vec) {
        crate::turbo_quant::unpack_codes(vector, bits, &mut codes);
        let mut score = 0.0f32;
        for (j, &code) in codes.iter().enumerate() {
            score += lut[j * levels + code as usize];
        }
        scores.push(score);
    }

    scores
}

/// Fast `TurboQuantMse` batch scoring using precomputed per-byte LUTs.
///
/// Used when the bit width divides 8, so each byte contains a fixed number of
/// complete coordinate codes. Scoring becomes a compact sum of byte-level LUT
/// entries with no per-vector unpack allocation.
pub fn turbo_quant_mse_score_batch_byte_weights(
    bytes_per_vec: usize,
    byte_weights: &[[f32; 256]],
    encoded: &[u8],
) -> Vec<f32> {
    if bytes_per_vec == 0 || !encoded.len().is_multiple_of(bytes_per_vec) {
        return Vec::new();
    }
    debug_assert_eq!(bytes_per_vec, byte_weights.len());

    let n = encoded.len() / bytes_per_vec;
    let mut scores = Vec::with_capacity(n);

    let mut i = 0usize;
    while i + 8 <= n {
        let v: [&[u8]; 8] =
            std::array::from_fn(|k| &encoded[(i + k) * bytes_per_vec..(i + k + 1) * bytes_per_vec]);
        let mut s = [0.0f32; 8];

        for b in 0..bytes_per_vec {
            let w = &byte_weights[b];
            for k in 0..8 {
                s[k] += w[v[k][b] as usize];
            }
        }

        scores.extend_from_slice(&s);
        i += 8;
    }

    while i < n {
        let vector = &encoded[i * bytes_per_vec..(i + 1) * bytes_per_vec];
        let mut score = 0.0f32;
        for (b, &byte) in vector.iter().enumerate() {
            score += byte_weights[b][byte as usize];
        }
        scores.push(score);
        i += 1;
    }

    scores
}

/// Score a contiguous block of `TurboQuantProd` encoded vectors.
///
/// `encoded` must contain `N * bytes_per_vec` bytes. The caller precomputes
/// `mse_lut`, `qjl_weights`, and `qjl_mask` from the query (see
/// `TurboQuantProdEncodedQuery`).
///
/// This function no longer unpacks the whole batch up-front.  Codes are
/// unpacked per-vector into a small reusable buffer, removing the `N * dim`
/// allocation that became the dominant cost for large batches.
#[allow(clippy::too_many_arguments)]
pub fn turbo_quant_prod_score_batch(
    dim: usize,
    bits: u8,
    bytes_per_vec: usize,
    mse_levels: usize,
    mse_lut: &[f32],
    qjl_scale: f32,
    qjl_sum_query: f32,
    qjl_weights: &[[f32; 256]],
    qjl_mask: &[u8],
    encoded: &[u8],
) -> Vec<f32> {
    #[allow(clippy::too_many_arguments)]
    if bytes_per_vec == 0 || !encoded.len().is_multiple_of(bytes_per_vec) {
        return Vec::new();
    }
    let n = encoded.len() / bytes_per_vec;
    let mse_bits = bits - 1;
    let mse_packed_bytes = crate::turbo_quant::packed_bytes(dim, mse_bits);
    let qjl_packed_bytes = crate::turbo_quant::packed_bytes(dim, 1);
    debug_assert_eq!(bytes_per_vec, mse_packed_bytes + qjl_packed_bytes + 4);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe {
                turbo_quant_prod_score_batch_avx2(
                    dim,
                    n,
                    mse_levels,
                    mse_lut,
                    qjl_scale,
                    qjl_sum_query,
                    qjl_weights,
                    qjl_mask,
                    encoded,
                    mse_packed_bytes,
                    qjl_packed_bytes,
                    bytes_per_vec,
                    mse_bits,
                )
            };
        }
    }

    turbo_quant_prod_score_batch_scalar(
        dim,
        n,
        mse_levels,
        mse_lut,
        qjl_scale,
        qjl_sum_query,
        qjl_weights,
        qjl_mask,
        encoded,
        mse_packed_bytes,
        qjl_packed_bytes,
        bytes_per_vec,
        mse_bits,
    )
}

#[allow(clippy::too_many_arguments)]
fn turbo_quant_prod_score_batch_scalar(
    dim: usize,
    n: usize,
    mse_levels: usize,
    mse_lut: &[f32],
    qjl_scale: f32,
    qjl_sum_query: f32,
    qjl_weights: &[[f32; 256]],
    qjl_mask: &[u8],
    encoded: &[u8],
    mse_packed_bytes: usize,
    qjl_packed_bytes: usize,
    bytes_per_vec: usize,
    mse_bits: u8,
) -> Vec<f32> {
    let mut scores = Vec::with_capacity(n);
    let mut codes = vec![0u8; dim];
    for vector in encoded.chunks_exact(bytes_per_vec) {
        crate::turbo_quant::unpack_codes(&vector[..mse_packed_bytes], mse_bits, &mut codes);
        let mut score = 0.0f32;
        for (j, &code) in codes.iter().enumerate() {
            score += mse_lut[j * mse_levels + code as usize];
        }

        let qjl_start = mse_packed_bytes;
        let mut qjl_dot = qjl_sum_query;
        for (b, &qjl_byte) in vector[qjl_start..qjl_start + qjl_packed_bytes]
            .iter()
            .enumerate()
        {
            let xor = qjl_byte ^ qjl_mask[b];
            qjl_dot -= 2.0 * qjl_weights[b][xor as usize];
        }

        let alpha = f32::from_le_bytes([
            vector[bytes_per_vec - 4],
            vector[bytes_per_vec - 3],
            vector[bytes_per_vec - 2],
            vector[bytes_per_vec - 1],
        ]);
        score += alpha * qjl_scale * qjl_dot;
        scores.push(score);
    }
    scores
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn turbo_quant_prod_score_batch_avx2(
    dim: usize,
    n: usize,
    mse_levels: usize,
    mse_lut: &[f32],
    qjl_scale: f32,
    qjl_sum_query: f32,
    qjl_weights: &[[f32; 256]],
    qjl_mask: &[u8],
    encoded: &[u8],
    mse_packed_bytes: usize,
    qjl_packed_bytes: usize,
    bytes_per_vec: usize,
    mse_bits: u8,
) -> Vec<f32> {
    use std::arch::x86_64::*;

    let mut scores = Vec::with_capacity(n);
    let base_ptr = mse_lut.as_ptr();
    let mut codes = vec![0u8; dim];

    for vector in encoded.chunks_exact(bytes_per_vec) {
        crate::turbo_quant::unpack_codes(&vector[..mse_packed_bytes], mse_bits, &mut codes);
        let mut acc = _mm256_setzero_ps();
        let mut j = 0usize;

        while j + 8 <= dim {
            // Load 8 codes as 32-bit indices.
            let c0 = *codes.get_unchecked(j) as i32;
            let c1 = *codes.get_unchecked(j + 1) as i32;
            let c2 = *codes.get_unchecked(j + 2) as i32;
            let c3 = *codes.get_unchecked(j + 3) as i32;
            let c4 = *codes.get_unchecked(j + 4) as i32;
            let c5 = *codes.get_unchecked(j + 5) as i32;
            let c6 = *codes.get_unchecked(j + 6) as i32;
            let c7 = *codes.get_unchecked(j + 7) as i32;
            let codes_i32 = _mm256_set_epi32(c7, c6, c5, c4, c3, c2, c1, c0);

            // Base dimension offsets: [j*levels, (j+1)*levels, ..., (j+7)*levels].
            let base_offset = (j * mse_levels) as i32;
            let base_indices = _mm256_set_epi32(
                base_offset + 7 * mse_levels as i32,
                base_offset + 6 * mse_levels as i32,
                base_offset + 5 * mse_levels as i32,
                base_offset + 4 * mse_levels as i32,
                base_offset + 3 * mse_levels as i32,
                base_offset + 2 * mse_levels as i32,
                base_offset + mse_levels as i32,
                base_offset,
            );
            let indices = _mm256_add_epi32(base_indices, codes_i32);

            // Gather 8 f32 LUT entries.
            let vals = _mm256_i32gather_ps(base_ptr, indices, 4);
            acc = _mm256_add_ps(acc, vals);

            j += 8;
        }

        let mut score = hsum256_ps(acc);
        while j < dim {
            let code = *codes.get_unchecked(j);
            score += mse_lut.get_unchecked(j * mse_levels + code as usize);
            j += 1;
        }

        let qjl_start = mse_packed_bytes;
        let mut qjl_dot = qjl_sum_query;
        for (b, &qjl_byte) in vector[qjl_start..qjl_start + qjl_packed_bytes]
            .iter()
            .enumerate()
        {
            let xor = qjl_byte ^ qjl_mask[b];
            qjl_dot -= 2.0 * qjl_weights[b][xor as usize];
        }

        let alpha = f32::from_le_bytes([
            *vector.get_unchecked(bytes_per_vec - 4),
            *vector.get_unchecked(bytes_per_vec - 3),
            *vector.get_unchecked(bytes_per_vec - 2),
            *vector.get_unchecked(bytes_per_vec - 1),
        ]);
        score += alpha * qjl_scale * qjl_dot;
        scores.push(score);
    }

    scores
}

/// Fast TurboQuantProd batch scoring using precomputed per-byte MSE LUTs.
///
/// This avoids the whole-batch code unpack and per-dimension gather.  It is
/// used when `bits - 1` divides 8 so each MSE byte encodes a fixed number of
/// coordinates.
#[allow(clippy::too_many_arguments)]
pub fn turbo_quant_prod_score_batch_byte_weights(
    bytes_per_vec: usize,
    mse_byte_weights: &[[f32; 256]],
    qjl_scale: f32,
    qjl_sum_query: f32,
    qjl_weights: &[[f32; 256]],
    qjl_mask: &[u8],
    encoded: &[u8],
) -> Vec<f32> {
    if bytes_per_vec == 0 || !encoded.len().is_multiple_of(bytes_per_vec) {
        return Vec::new();
    }
    let n = encoded.len() / bytes_per_vec;
    let mse_packed_bytes = mse_byte_weights.len();
    let qjl_packed_bytes = qjl_weights.len();
    debug_assert_eq!(bytes_per_vec, mse_packed_bytes + qjl_packed_bytes + 4);

    let mut scores = Vec::with_capacity(n);

    // Pre-compute the constant part of the QJL contribution so the per-vector
    // work is just a fused multiply-add with alpha.
    let qjl_const = qjl_scale * qjl_sum_query;

    // Unroll by 8 vectors to maximize instruction-level parallelism and hide
    // LUT load latency.  Keep the MSE and QJL passes separate because their
    // packed byte lengths differ, but use enough independent accumulators that
    // the CPU can overlap the dependent loads.
    let mut i = 0usize;
    while i + 8 <= n {
        let v: [&[u8]; 8] =
            std::array::from_fn(|k| &encoded[(i + k) * bytes_per_vec..(i + k + 1) * bytes_per_vec]);
        let mut s = [0.0f32; 8];
        let mut q = [0.0f32; 8];

        for b in 0..mse_packed_bytes {
            let w = &mse_byte_weights[b];
            for k in 0..8 {
                s[k] += w[v[k][b] as usize];
            }
        }

        for b in 0..qjl_packed_bytes {
            let w = &qjl_weights[b];
            let m = qjl_mask[b];
            let off = mse_packed_bytes + b;
            for k in 0..8 {
                let xor = v[k][off] ^ m;
                q[k] -= 2.0 * w[xor as usize];
            }
        }

        for k in 0..8 {
            let alpha = f32::from_le_bytes([
                v[k][bytes_per_vec - 4],
                v[k][bytes_per_vec - 3],
                v[k][bytes_per_vec - 2],
                v[k][bytes_per_vec - 1],
            ]);
            scores.push(s[k] + alpha * (qjl_const + qjl_scale * q[k]));
        }

        i += 8;
    }

    // Remaining vectors one at a time.
    while i < n {
        let vector = &encoded[i * bytes_per_vec..(i + 1) * bytes_per_vec];
        let mut score = 0.0f32;
        for (b, &byte) in vector[..mse_packed_bytes].iter().enumerate() {
            score += mse_byte_weights[b][byte as usize];
        }

        let mut qjl_dot = qjl_sum_query;
        for (b, &qjl_byte) in vector[mse_packed_bytes..mse_packed_bytes + qjl_packed_bytes]
            .iter()
            .enumerate()
        {
            let xor = qjl_byte ^ qjl_mask[b];
            qjl_dot -= 2.0 * qjl_weights[b][xor as usize];
        }

        let alpha = f32::from_le_bytes([
            vector[bytes_per_vec - 4],
            vector[bytes_per_vec - 3],
            vector[bytes_per_vec - 2],
            vector[bytes_per_vec - 1],
        ]);
        score += alpha * qjl_scale * qjl_dot;
        scores.push(score);
        i += 1;
    }
    scores
}

/// Batched cosine similarity between a query and a slice of full-precision
/// vectors.  All vectors must be the same length as `query`.
pub fn batched_cosine_similarity(query: &[f32], vectors: &[&[f32]]) -> Vec<f32> {
    crate::cosine_similarity_batch(query, vectors)
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
