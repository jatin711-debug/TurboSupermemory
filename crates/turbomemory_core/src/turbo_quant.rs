//! TurboQuant quantizers.
//!
//! Implements the MSE-optimal and inner-product-optimal quantizers from
//! "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate".
//!
//! The pipeline is:
//! 1. L2-normalize the input vector and remember its original norm.
//! 2. Apply a fast approximate random rotation (random diagonal precondition +
//!    FWHT, normalized to preserve L2 norm).
//! 3. Quantize each rotated coordinate with a Lloyd-Max codebook scaled to the
//!    rotated coordinate distribution (`N(0, 1/d)`).
//!
//! `TurboQuantProdQuantizer` adds a second stage: a 1-bit Quantized
//! Johnson-Lindenstrauss (QJL) transform applied to the residual after the MSE
//! reconstruction, yielding an unbiased inner-product estimator.

use crate::quantization::{LloydMaxTable, Quantizer};
use crate::quantized_search::{EncodedQuery, QuantizedStore};
use crate::{fwht, random_diagonal_precondition, validate_dimension, Result, TurboError};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::Arc;

/// Number of bytes required to store `dim` coordinates of `bits` each.
pub fn packed_bytes(dim: usize, bits: u8) -> usize {
    (dim * bits as usize).div_ceil(8)
}

/// Pack `dim` codes (each in `0..2^bits`) into `out` contiguously.
///
/// Codes are stored little-endian within the bit stream: the lowest bit of
/// coordinate `i` is at bit position `i * bits`.
fn pack_codes(codes: &[u8], bits: u8, out: &mut [u8]) {
    assert_eq!(out.len(), packed_bytes(codes.len(), bits));
    out.fill(0);
    let bits = bits as usize;
    for (i, &code) in codes.iter().enumerate() {
        let bit_pos = i * bits;
        let byte_pos = bit_pos / 8;
        let shift = bit_pos % 8;
        out[byte_pos] |= code << shift;
        if shift + bits > 8 {
            out[byte_pos + 1] |= code >> (8 - shift);
        }
    }
}

/// Unpack `dim` codes (each `bits` wide) from `encoded`.
pub fn unpack_codes(encoded: &[u8], bits: u8, out: &mut [u8]) {
    let bits = bits as usize;
    let mask = ((1u16 << bits) - 1) as u8;
    for (i, code) in out.iter_mut().enumerate() {
        let bit_pos = i * bits;
        let byte_pos = bit_pos / 8;
        let shift = bit_pos % 8;
        let low = encoded[byte_pos] >> shift;
        let high = if shift + bits > 8 {
            (encoded[byte_pos + 1] as u16) << (8 - shift)
        } else {
            0u16
        };
        *code = ((low as u16 | high) & mask as u16) as u8;
    }
}

/// Generate a `dim x dim` Gaussian matrix with i.i.d. `N(0,1)` entries.
///
/// Stored row-major: `s[i * dim + j]` is the `j`-th column of the `i`-th row.
fn generate_gaussian_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = vec![0.0f32; dim * dim];
    for x in out.iter_mut() {
        let sample: f64 = StandardNormal.sample(&mut rng);
        *x = sample as f32;
    }
    out
}

/// MSE-optimal TurboQuant quantizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantMseQuantizer {
    dim: usize,
    bits: u8,
    /// Seed for the fast approximate random rotation.
    rotation_seed: u64,
    /// Lloyd-Max centroids for a `N(0, 1)` distribution.  At encode/decode
    /// time they are scaled by `1/sqrt(dim)` to match the rotated coordinate
    /// distribution `N(0, 1/dim)`.
    centroids: Vec<f32>,
}

impl TurboQuantMseQuantizer {
    pub fn new(dim: usize, bits: u8, rotation_seed: u64) -> Result<Self> {
        if dim == 0 || !dim.is_power_of_two() {
            return Err(TurboError::QuantizationError(
                "TurboQuant requires power-of-two dimension".into(),
            ));
        }
        if bits == 0 || bits > 8 {
            return Err(TurboError::QuantizationError(
                "bits must be in 1..=8".into(),
            ));
        }
        let table = LloydMaxTable::new(bits)?;
        Ok(Self {
            dim,
            bits,
            rotation_seed,
            centroids: table.centroids,
        })
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn rotation_seed(&self) -> u64 {
        self.rotation_seed
    }

    /// Rotate `v` into the TurboQuant coordinate system in-place and return
    /// its original L2 norm.  After rotation `v` is unit-norm and each
    /// coordinate is approximately `N(0, 1/dim)`.
    fn rotate_forward(&self, v: &mut [f32]) -> f32 {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        random_diagonal_precondition(v, self.rotation_seed);
        fwht(v);
        let inv = 1.0 / (self.dim as f32).sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
        norm
    }

    /// Inverse of [`rotate_forward`].  `v` must already be in the rotated
    /// coordinate system (coordinates scaled by `1/sqrt(dim)`); it is returned
    /// to the original basis.
    fn rotate_backward(&self, v: &mut [f32]) {
        // Forward precondition is P(v) = H * D * v / sqrt(d).
        // Its inverse is P^{-1}(y) = D * H * y / sqrt(d).
        // In function-composition order that is H first, then D.
        fwht(v);
        random_diagonal_precondition(v, self.rotation_seed);
        let inv = 1.0 / (self.dim as f32).sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
    }

    /// Scale factor applied to the stored `N(0,1)` centroids to match the
    /// rotated coordinate distribution.
    fn centroid_scale(&self) -> f32 {
        1.0 / (self.dim as f32).sqrt()
    }

    /// Quantize a rotated coordinate to the nearest scaled centroid.
    fn quantize_coordinate(&self, y: f32) -> u8 {
        let scale = self.centroid_scale();
        let mut best_idx = 0u8;
        let mut best_dist = (y - self.centroids[0] * scale).abs();
        for (idx, &c) in self.centroids.iter().enumerate().skip(1) {
            let dist = (y - c * scale).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx as u8;
            }
        }
        best_idx
    }

    /// Encode a full-precision vector.  The original norm is discarded; the
    /// quantized representation encodes the direction.  This matches the
    /// behavior of cosine-similarity search where only direction matters for
    /// candidate ranking.
    pub fn encode_direction(&self, v: &[f32]) -> Result<Vec<u8>> {
        validate_dimension(v, self.dim)?;
        let mut rotated = v.to_vec();
        self.rotate_forward(&mut rotated);
        let mut codes = vec![0u8; self.dim];
        for (i, &y) in rotated.iter().enumerate() {
            codes[i] = self.quantize_coordinate(y);
        }
        let mut out = vec![0u8; self.encoded_bytes_per_vector()];
        pack_codes(&codes, self.bits, &mut out);
        Ok(out)
    }

    /// Decode a quantized vector back to full precision (unit-norm direction).
    pub fn decode_direction(&self, q: &[u8]) -> Result<Vec<f32>> {
        if q.len() != self.encoded_bytes_per_vector() {
            return Err(TurboError::DimensionMismatch {
                expected: self.encoded_bytes_per_vector(),
                got: q.len(),
            });
        }
        let mut codes = vec![0u8; self.dim];
        unpack_codes(q, self.bits, &mut codes);
        let scale = self.centroid_scale();
        let mut rotated: Vec<f32> = codes
            .iter()
            .map(|&c| self.centroids[c as usize] * scale)
            .collect();
        self.rotate_backward(&mut rotated);
        Ok(rotated)
    }
}

impl Quantizer for TurboQuantMseQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        packed_bytes(self.dim, self.bits)
    }

    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        self.encode_direction(v)
    }

    fn decode(&self, q: &[u8]) -> Result<Vec<f32>> {
        self.decode_direction(q)
    }
}

/// Inner-product-optimal TurboQuant quantizer.
///
/// Uses `bits - 1` for an MSE stage and one additional bit per coordinate for
/// a QJL transform on the residual, giving an unbiased inner-product estimator.
#[derive(Debug, Clone)]
pub struct TurboQuantProdQuantizer {
    dim: usize,
    bits: u8,
    mse: TurboQuantMseQuantizer,
    qjl_seed: u64,
    /// Row-major `dim x dim` Gaussian matrix.  Shared via `Arc` because the
    /// matrix is large and the quantizer must be cheap to clone.
    gaussian_matrix: Arc<Vec<f32>>,
    /// Precomputed `sqrt(pi/2) / dim` used during decode/score.
    qjl_scale: f32,
}

/// Serializable surrogate for [`TurboQuantProdQuantizer`].
///
/// The Gaussian matrix is regenerated from `qjl_seed` on deserialization so we
/// do not store `d²` floats in the manifest.
#[derive(Serialize, Deserialize)]
struct TurboQuantProdQuantizerSerialized {
    dim: usize,
    bits: u8,
    rotation_seed: u64,
    qjl_seed: u64,
}

impl Serialize for TurboQuantProdQuantizer {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        TurboQuantProdQuantizerSerialized {
            dim: self.dim,
            bits: self.bits,
            rotation_seed: self.mse.rotation_seed,
            qjl_seed: self.qjl_seed,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TurboQuantProdQuantizer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let s = TurboQuantProdQuantizerSerialized::deserialize(deserializer)?;
        Self::new(s.dim, s.bits, s.rotation_seed, s.qjl_seed)
            .map_err(|e| serde::de::Error::custom(format!("invalid TurboQuantProd: {e}")))
    }
}

impl TurboQuantProdQuantizer {
    pub fn new(dim: usize, bits: u8, rotation_seed: u64, qjl_seed: u64) -> Result<Self> {
        if !(2..=8).contains(&bits) {
            return Err(TurboError::QuantizationError(
                "TurboQuantProd requires bits in 2..=8".into(),
            ));
        }
        let mse = TurboQuantMseQuantizer::new(dim, bits - 1, rotation_seed)?;
        let gaussian_matrix = Arc::new(generate_gaussian_matrix(dim, qjl_seed));
        let qjl_scale = (std::f32::consts::PI / 2.0).sqrt() / dim as f32;
        Ok(Self {
            dim,
            bits,
            mse,
            qjl_seed,
            gaussian_matrix,
            qjl_scale,
        })
    }

    pub fn bits(&self) -> u8 {
        self.bits
    }

    pub fn qjl_seed(&self) -> u64 {
        self.qjl_seed
    }

    fn qjl_bytes(&self) -> usize {
        packed_bytes(self.dim, 1)
    }

    /// Encode a vector.  Layout: `[mse_codes][qjl_sign_bits][alpha: f32 LE]`.
    pub fn encode_direction(&self, v: &[f32]) -> Result<Vec<u8>> {
        validate_dimension(v, self.dim)?;
        // First MSE stage on the normalized direction.
        let mut rotated = v.to_vec();
        let norm = self.mse.rotate_forward(&mut rotated);
        let mut mse_codes = vec![0u8; self.dim];
        for (i, &y) in rotated.iter().enumerate() {
            mse_codes[i] = self.mse.quantize_coordinate(y);
        }

        // Reconstruct the MSE approximation in the rotated domain.
        let mse_scale = self.mse.centroid_scale();
        let mut mse_recon_rot: Vec<f32> = mse_codes
            .iter()
            .map(|&c| self.mse.centroids[c as usize] * mse_scale)
            .collect();
        // Rotate back to original basis to obtain x~_mse (still unit-norm).
        self.mse.rotate_backward(&mut mse_recon_rot);

        // Residual in original basis.
        let mut residual = vec![0.0f32; self.dim];
        let alpha = if norm > 0.0 {
            for i in 0..self.dim {
                residual[i] = v[i] / norm - mse_recon_rot[i];
            }
            residual.iter().map(|x| x * x).sum::<f32>().sqrt()
        } else {
            0.0
        };

        // QJL on the residual: qjl = sign(S * r).
        let s = &self.gaussian_matrix;
        let mut qjl_codes = vec![0u8; self.dim];
        for i in 0..self.dim {
            let row = &s[i * self.dim..(i + 1) * self.dim];
            let dot: f32 = row.iter().zip(&residual).map(|(a, b)| a * b).sum();
            qjl_codes[i] = if dot >= 0.0 { 1 } else { 0 };
        }

        // Pack everything together.
        let mse_packed_bytes = packed_bytes(self.dim, self.bits - 1);
        let qjl_packed_bytes = self.qjl_bytes();
        let mut out = vec![0u8; mse_packed_bytes + qjl_packed_bytes + 4];
        pack_codes(&mse_codes, self.bits - 1, &mut out[0..mse_packed_bytes]);
        pack_codes(
            &qjl_codes,
            1,
            &mut out[mse_packed_bytes..mse_packed_bytes + qjl_packed_bytes],
        );
        out[mse_packed_bytes + qjl_packed_bytes..].copy_from_slice(&alpha.to_le_bytes());
        Ok(out)
    }

    /// Decode a vector.  Splits the layout and returns the unit-norm
    /// reconstruction `x~_mse + x~_qjl`.
    pub fn decode_direction(&self, q: &[u8]) -> Result<Vec<f32>> {
        let expected = self.encoded_bytes_per_vector();
        if q.len() != expected {
            return Err(TurboError::DimensionMismatch {
                expected,
                got: q.len(),
            });
        }
        let mse_packed_bytes = packed_bytes(self.dim, self.bits - 1);
        let qjl_packed_bytes = self.qjl_bytes();

        // MSE part.
        let mut mse_codes = vec![0u8; self.dim];
        unpack_codes(&q[0..mse_packed_bytes], self.bits - 1, &mut mse_codes);
        let mse_scale = self.mse.centroid_scale();
        let mut mse_recon_rot: Vec<f32> = mse_codes
            .iter()
            .map(|&c| self.mse.centroids[c as usize] * mse_scale)
            .collect();
        self.mse.rotate_backward(&mut mse_recon_rot);

        // QJL part.
        let mut qjl_codes = vec![0u8; self.dim];
        unpack_codes(
            &q[mse_packed_bytes..mse_packed_bytes + qjl_packed_bytes],
            1,
            &mut qjl_codes,
        );
        let alpha = f32::from_le_bytes([
            q[expected - 4],
            q[expected - 3],
            q[expected - 2],
            q[expected - 1],
        ]);

        let s = &self.gaussian_matrix;
        let mut x_qjl = vec![0.0f32; self.dim];
        for j in 0..self.dim {
            let mut sum = 0.0f32;
            for i in 0..self.dim {
                let sign = if qjl_codes[i] == 1 { 1.0f32 } else { -1.0f32 };
                sum += s[i * self.dim + j] * sign;
            }
            x_qjl[j] = alpha * self.qjl_scale * sum;
        }

        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            out[i] = mse_recon_rot[i] + x_qjl[i];
        }
        Ok(out)
    }

    /// Project a vector through the QJL Gaussian matrix: `S * v`.
    fn project(&self, v: &[f32]) -> Vec<f32> {
        let s = &self.gaussian_matrix;
        let mut out = vec![0.0f32; self.dim];
        for i in 0..self.dim {
            let row = &s[i * self.dim..(i + 1) * self.dim];
            out[i] = row.iter().zip(v).map(|(a, b)| a * b).sum();
        }
        out
    }

    /// Score an encoded vector against a pre-projected query without full
    /// decode.  `query_rotated` is `D * H * query`; `query_projected` is
    /// `S * query`.  Both are `dim`-length.
    pub fn score_with_query_buffers(
        &self,
        encoded: &[u8],
        query_rotated: &[f32],
        query_projected: &[f32],
    ) -> f32 {
        let levels = 1usize << (self.bits - 1);
        let centroid_scale = self.mse.centroid_scale();
        let mut mse_lut = vec![0.0f32; self.dim * levels];
        for j in 0..self.dim {
            let q = query_rotated[j];
            for c in 0..levels {
                mse_lut[j * levels + c] = q * self.mse.centroids[c] * centroid_scale;
            }
        }

        let mut qjl_sum_abs = 0.0f32;
        let qjl_bytes = packed_bytes(self.dim, 1);
        let mut qjl_mask = vec![0u8; qjl_bytes];
        let mut qjl_weights = vec![[0.0f32; 256]; qjl_bytes];
        for (i, &q) in query_projected.iter().enumerate() {
            let abs_q = q.abs();
            qjl_sum_abs += abs_q;
            let byte = i / 8;
            let bit = i % 8;
            if q >= 0.0 {
                qjl_mask[byte] |= 1 << bit;
            }
            for (pattern, cell) in qjl_weights[byte].iter_mut().enumerate() {
                if (pattern & (1 << bit)) != 0 {
                    *cell += abs_q;
                }
            }
        }

        score_turbo_quant_prod(
            encoded,
            self.dim,
            self.bits,
            levels,
            &mse_lut,
            None,
            self.qjl_scale,
            qjl_sum_abs,
            &qjl_weights,
            &qjl_mask,
        )
    }
}

/// Shared scoring kernel for [`TurboQuantProdQuantizer`].
#[allow(clippy::too_many_arguments)]
fn score_turbo_quant_prod(
    encoded: &[u8],
    dim: usize,
    bits: u8,
    mse_levels: usize,
    mse_lut: &[f32],
    mse_byte_weights: Option<&[[f32; 256]]>,
    qjl_scale: f32,
    qjl_sum_query: f32,
    qjl_weights: &[[f32; 256]],
    qjl_mask: &[u8],
) -> f32 {
    let mse_bits = bits - 1;
    let mse_packed_bytes = packed_bytes(dim, mse_bits);
    let qjl_packed_bytes = packed_bytes(dim, 1);
    let expected = mse_packed_bytes + qjl_packed_bytes + 4;
    if encoded.len() != expected {
        return f32::NEG_INFINITY;
    }

    // MSE contribution: sum_j mse_lut[j * levels + code_j].
    let mut score = 0.0f32;
    if let Some(weights) = mse_byte_weights {
        for (b, &byte) in encoded[..mse_packed_bytes].iter().enumerate() {
            score += weights[b][byte as usize];
        }
    } else {
        let mut mse_codes = vec![0u8; dim];
        unpack_codes(&encoded[0..mse_packed_bytes], mse_bits, &mut mse_codes);
        for (j, &code) in mse_codes.iter().enumerate() {
            score += mse_lut[j * mse_levels + code as usize];
        }
    }

    // QJL contribution: alpha * qjl_scale * dot(qjl, sign).
    // Using the sign-quantizer LUT identity:
    //   dot = qjl_sum_query - 2 * Σ_byte weights[byte][qjl_byte ^ mask_byte]
    let mut qjl_dot = qjl_sum_query;
    for (b, &qjl_byte) in encoded[mse_packed_bytes..mse_packed_bytes + qjl_packed_bytes]
        .iter()
        .enumerate()
    {
        let xor = qjl_byte ^ qjl_mask[b];
        qjl_dot -= 2.0 * qjl_weights[b][xor as usize];
    }
    let alpha = f32::from_le_bytes([
        encoded[expected - 4],
        encoded[expected - 3],
        encoded[expected - 2],
        encoded[expected - 1],
    ]);
    score += alpha * qjl_scale * qjl_dot;
    score
}

impl Quantizer for TurboQuantProdQuantizer {
    fn dim(&self) -> usize {
        self.dim
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        packed_bytes(self.dim, self.bits - 1) + self.qjl_bytes() + 4
    }

    fn encode(&self, v: &[f32]) -> Result<Vec<u8>> {
        self.encode_direction(v)
    }

    fn decode(&self, q: &[u8]) -> Result<Vec<f32>> {
        self.decode_direction(q)
    }
}

/// Precomputed query buffers for fast [`TurboQuantProdQuantizer`] scoring.
///
/// Keeps the rotated query `D*H*query / sqrt(d)` and the QJL-projected query
/// `S*query` so each encoded vector can be scored in O(dim) without decoding.
pub struct TurboQuantProdEncodedQuery {
    dim: usize,
    bits: u8,
    qjl_scale: f32,
    bytes_per_vec: usize,
    /// Flattened MSE lookup table: `mse_lut[j * levels + c]`.
    mse_lut: Vec<f32>,
    mse_levels: usize,
    /// Per-byte 256-entry MSE lookup table.  Only populated when `bits-1`
    /// divides 8, so scoring becomes a simple byte LUT sum instead of bit
    /// unpacking + per-dimension gather.
    mse_byte_weights: Vec<[f32; 256]>,
    /// `Σ_i |query_projected[i]|`; used in the QJL dot-product shortcut.
    qjl_sum_query: f32,
    /// Per-byte 256-entry weight table for the QJL contribution.
    qjl_weights: Vec<[f32; 256]>,
    /// Query sign mask per QJL byte.
    qjl_mask: Vec<u8>,
}

impl TurboQuantProdEncodedQuery {
    fn new(quantizer: &TurboQuantProdQuantizer, query: &[f32]) -> Result<Self> {
        validate_dimension(query, quantizer.dim)?;
        let mut query_rotated = query.to_vec();
        random_diagonal_precondition(&mut query_rotated, quantizer.mse.rotation_seed);
        fwht(&mut query_rotated);
        let inv = 1.0 / (quantizer.dim as f32).sqrt();
        for x in query_rotated.iter_mut() {
            *x *= inv;
        }
        let query_projected = quantizer.project(query);

        let dim = quantizer.dim;
        let bits = quantizer.bits;
        let levels = 1usize << (bits - 1);
        let centroid_scale = quantizer.mse.centroid_scale();
        let mut mse_lut = vec![0.0f32; dim * levels];
        for j in 0..dim {
            let q = query_rotated[j];
            for c in 0..levels {
                mse_lut[j * levels + c] = q * quantizer.mse.centroids[c] * centroid_scale;
            }
        }

        // Build per-byte MSE lookup tables when the code width evenly divides a
        // byte.  This turns scoring into a fast sum over byte LUTs.
        let mse_bits = bits - 1;
        let mse_byte_weights = if mse_bits > 0 && (8 % mse_bits == 0) {
            let codes_per_byte = (8 / mse_bits) as usize;
            let code_mask = ((1u16 << mse_bits) - 1) as u8;
            let mse_packed_bytes = packed_bytes(dim, mse_bits);
            let mut weights = vec![[0.0f32; 256]; mse_packed_bytes];
            for (byte_idx, row) in weights.iter_mut().enumerate() {
                let start_dim = byte_idx * codes_per_byte;
                let valid_codes = codes_per_byte.min(dim.saturating_sub(start_dim));
                for (pattern, cell) in row.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for k in 0..valid_codes {
                        let code =
                            ((pattern >> (k * mse_bits as usize)) & code_mask as usize) as u8;
                        let j = start_dim + k;
                        sum += mse_lut[j * levels + code as usize];
                    }
                    *cell = sum;
                }
            }
            weights
        } else {
            Vec::new()
        };

        let mut qjl_sum_abs = 0.0f32;
        let qjl_bytes = packed_bytes(dim, 1);
        let mut qjl_mask = vec![0u8; qjl_bytes];
        let mut qjl_weights = vec![[0.0f32; 256]; qjl_bytes];
        for (i, &q) in query_projected.iter().enumerate() {
            let abs_q = q.abs();
            qjl_sum_abs += abs_q;
            let byte = i / 8;
            let bit = i % 8;
            if q >= 0.0 {
                qjl_mask[byte] |= 1 << bit;
            }
            for (pattern, cell) in qjl_weights[byte].iter_mut().enumerate() {
                if (pattern & (1 << bit)) != 0 {
                    *cell += abs_q;
                }
            }
        }

        Ok(Self {
            dim,
            bits,
            qjl_scale: quantizer.qjl_scale,
            bytes_per_vec: quantizer.encoded_bytes_per_vector(),
            mse_lut,
            mse_levels: levels,
            mse_byte_weights,
            qjl_sum_query: qjl_sum_abs,
            qjl_weights,
            qjl_mask,
        })
    }
}

impl EncodedQuery for TurboQuantProdEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        score_turbo_quant_prod(
            encoded,
            self.dim,
            self.bits,
            self.mse_levels,
            &self.mse_lut,
            if self.mse_byte_weights.is_empty() {
                None
            } else {
                Some(&self.mse_byte_weights)
            },
            self.qjl_scale,
            self.qjl_sum_query,
            &self.qjl_weights,
            &self.qjl_mask,
        )
    }

    fn score_batch(&self, encoded: &[u8]) -> Vec<f32> {
        if self.mse_byte_weights.is_empty() {
            crate::metrics_quantized::turbo_quant_prod_score_batch(
                self.dim,
                self.bits,
                self.bytes_per_vec,
                self.mse_levels,
                &self.mse_lut,
                self.qjl_scale,
                self.qjl_sum_query,
                &self.qjl_weights,
                &self.qjl_mask,
                encoded,
            )
        } else {
            crate::metrics_quantized::turbo_quant_prod_score_batch_byte_weights(
                self.bytes_per_vec,
                &self.mse_byte_weights,
                self.qjl_scale,
                self.qjl_sum_query,
                &self.qjl_weights,
                &self.qjl_mask,
                encoded,
            )
        }
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        self.bytes_per_vec
    }
}

impl QuantizedStore for TurboQuantProdQuantizer {
    type Query = TurboQuantProdEncodedQuery;

    fn encode_query(&self, query: &[f32]) -> Result<Self::Query> {
        TurboQuantProdEncodedQuery::new(self, query)
    }
}

/// Precomputed query buffer for fast [`TurboQuantMseQuantizer`] scoring.
pub struct TurboQuantMseEncodedQuery {
    dim: usize,
    bits: u8,
    bytes_per_vec: usize,
    /// Flattened lookup table: `lut[j * levels + code]`.
    lut: Vec<f32>,
    levels: usize,
    /// Per-byte 256-entry lookup table. Populated when `bits` divides 8.
    byte_weights: Vec<[f32; 256]>,
}

impl TurboQuantMseEncodedQuery {
    fn new(quantizer: &TurboQuantMseQuantizer, query: &[f32]) -> Result<Self> {
        validate_dimension(query, quantizer.dim)?;
        let mut query_rotated = query.to_vec();
        random_diagonal_precondition(&mut query_rotated, quantizer.rotation_seed);
        fwht(&mut query_rotated);
        let inv = 1.0 / (quantizer.dim as f32).sqrt();
        for x in query_rotated.iter_mut() {
            *x *= inv;
        }

        let dim = quantizer.dim;
        let bits = quantizer.bits;
        let levels = 1usize << bits;
        let centroid_scale = quantizer.centroid_scale();
        let mut lut = vec![0.0f32; dim * levels];
        for j in 0..dim {
            let q = query_rotated[j];
            for c in 0..levels {
                lut[j * levels + c] = q * quantizer.centroids[c] * centroid_scale;
            }
        }

        let bytes_per_vec = quantizer.encoded_bytes_per_vector();
        let byte_weights = if 8 % bits == 0 {
            let bit_width = bits as usize;
            let codes_per_byte = 8 / bit_width;
            let code_mask = ((1u16 << bit_width) - 1) as usize;
            let mut weights = vec![[0.0f32; 256]; bytes_per_vec];
            for (byte_idx, row) in weights.iter_mut().enumerate() {
                let start_dim = byte_idx * codes_per_byte;
                let valid_codes = codes_per_byte.min(dim.saturating_sub(start_dim));
                for (pattern, cell) in row.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for k in 0..valid_codes {
                        let code = (pattern >> (k * bit_width)) & code_mask;
                        let j = start_dim + k;
                        sum += lut[j * levels + code];
                    }
                    *cell = sum;
                }
            }
            weights
        } else {
            Vec::new()
        };

        Ok(Self {
            dim,
            bits,
            bytes_per_vec,
            lut,
            levels,
            byte_weights,
        })
    }
}

impl EncodedQuery for TurboQuantMseEncodedQuery {
    fn score(&self, encoded: &[u8]) -> f32 {
        if encoded.len() != self.bytes_per_vec {
            return f32::NEG_INFINITY;
        }
        if !self.byte_weights.is_empty() {
            return encoded
                .iter()
                .enumerate()
                .map(|(b, &byte)| self.byte_weights[b][byte as usize])
                .sum();
        }
        let mut codes = vec![0u8; self.dim];
        unpack_codes(encoded, self.bits, &mut codes);
        let mut score = 0.0f32;
        for (j, &code) in codes.iter().enumerate() {
            score += self.lut[j * self.levels + code as usize];
        }
        score
    }

    fn score_batch(&self, encoded: &[u8]) -> Vec<f32> {
        if self.byte_weights.is_empty() {
            crate::metrics_quantized::turbo_quant_mse_score_batch(
                self.dim,
                self.bits,
                self.bytes_per_vec,
                self.levels,
                &self.lut,
                encoded,
            )
        } else {
            crate::metrics_quantized::turbo_quant_mse_score_batch_byte_weights(
                self.bytes_per_vec,
                &self.byte_weights,
                encoded,
            )
        }
    }

    fn encoded_bytes_per_vector(&self) -> usize {
        self.bytes_per_vec
    }
}

impl QuantizedStore for TurboQuantMseQuantizer {
    type Query = TurboQuantMseEncodedQuery;

    fn encode_query(&self, query: &[f32]) -> Result<Self::Query> {
        TurboQuantMseEncodedQuery::new(self, query)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() {
            *x /= norm;
        }
        v
    }

    #[test]
    fn mse_roundtrip_distortion_decreases_with_bits() {
        let dim = 128;
        let v = random_unit_vector(dim, 123);
        let mut prev_mse = f32::INFINITY;
        for bits in [1, 2, 3, 4] {
            let q = TurboQuantMseQuantizer::new(dim, bits, 42).unwrap();
            let encoded = q.encode(&v).unwrap();
            let decoded = q.decode(&encoded).unwrap();
            let mse: f32 = v
                .iter()
                .zip(&decoded)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / dim as f32;
            assert!(mse < 0.5, "bits={bits} mse={mse}");
            assert!(
                mse <= prev_mse * 1.5,
                "bits={bits} mse={mse} not decreasing vs prev={prev_mse}"
            );
            prev_mse = mse;
        }
    }

    #[test]
    fn mse_against_true_gaussian() {
        for dim in [128, 256, 512, 1024] {
            // Sample from standard normal and normalize -> approx uniform on sphere.
            let mut rng = StdRng::seed_from_u64(777);
            let mut v: Vec<f32> = (0..dim)
                .map(|_| {
                    let sample: f64 = StandardNormal.sample(&mut rng);
                    sample as f32
                })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in v.iter_mut() {
                *x /= norm;
            }

            let q2 = TurboQuantMseQuantizer::new(dim, 2, 42).unwrap();
            let q4 = TurboQuantMseQuantizer::new(dim, 4, 42).unwrap();
            let mut rotated = v.clone();
            let _ = q2.rotate_forward(&mut rotated);
            let var = rotated.iter().map(|x| x * x).sum::<f32>() / dim as f32;

            let (mse2, norm2) = {
                let e = q2.encode(&v).unwrap();
                let d = q2.decode(&e).unwrap();
                let mse = v.iter().zip(&d).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / dim as f32;
                let norm = d.iter().map(|x| x * x).sum::<f32>().sqrt();
                (mse, norm)
            };
            let (mse4, norm4) = {
                let e = q4.encode(&v).unwrap();
                let d = q4.decode(&e).unwrap();
                let mse = v.iter().zip(&d).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / dim as f32;
                let norm = d.iter().map(|x| x * x).sum::<f32>().sqrt();
                (mse, norm)
            };
            println!("dim={dim} var={var} mse2={mse2} mse4={mse4} paper2~={} norm2={norm2} norm4={norm4}", 0.117 / dim as f32);
            assert!(mse4 < mse2, "mse4={mse4} should be less than mse2={mse2}");
        }
    }

    #[test]
    fn mse_higher_bits_lower_distortion() {
        let dim = 128;
        let q2 = TurboQuantMseQuantizer::new(dim, 2, 42).unwrap();
        let q4 = TurboQuantMseQuantizer::new(dim, 4, 42).unwrap();
        let mut mse2 = 0.0f32;
        let mut mse4 = 0.0f32;
        let n = 20;
        for i in 0..n {
            let v = random_unit_vector(dim, 1000 + i as u64);
            let e2 = q2.encode(&v).unwrap();
            let e4 = q4.encode(&v).unwrap();
            let d2 = q2.decode(&e2).unwrap();
            let d4 = q4.decode(&e4).unwrap();
            mse2 += v.iter().zip(&d2).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / dim as f32;
            mse4 += v.iter().zip(&d4).map(|(a, b)| (a - b).powi(2)).sum::<f32>() / dim as f32;
        }
        mse2 /= n as f32;
        mse4 /= n as f32;
        assert!(
            mse4 < mse2,
            "avg mse4={mse4} should be less than avg mse2={mse2}"
        );
    }

    #[test]
    fn prod_inner_product_unbiased() {
        let dim = 128;
        let q = TurboQuantProdQuantizer::new(dim, 3, 42, 7).unwrap();
        let query = random_unit_vector(dim, 999);

        let mut sum_err = 0.0f32;
        let n = 100;
        for i in 0..n {
            let v = random_unit_vector(dim, i as u64 + 1000);
            let encoded = q.encode(&v).unwrap();
            let decoded = q.decode(&encoded).unwrap();
            let true_ip: f32 = query.iter().zip(&v).map(|(a, b)| a * b).sum();
            let est_ip: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();
            sum_err += true_ip - est_ip;
        }
        let mean_err = sum_err / n as f32;
        assert!(
            mean_err.abs() < 0.05,
            "mean error={mean_err} should be near zero"
        );
    }

    #[test]
    fn prod_score_matches_decode_dot() {
        let dim = 64;
        let q = TurboQuantProdQuantizer::new(dim, 3, 42, 7).unwrap();
        let v = random_unit_vector(dim, 123);
        let query = random_unit_vector(dim, 456);
        let encoded = q.encode(&v).unwrap();
        let decoded = q.decode(&encoded).unwrap();
        let expected: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();

        // Build query buffers.
        let mut query_rotated = query.clone();
        random_diagonal_precondition(&mut query_rotated, q.mse.rotation_seed);
        fwht(&mut query_rotated);
        let inv = 1.0 / (dim as f32).sqrt();
        for x in query_rotated.iter_mut() {
            *x *= inv;
        }
        let query_projected = q.project(&query);

        let actual = q.score_with_query_buffers(&encoded, &query_rotated, &query_projected);
        assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
    }

    #[test]
    fn prod_quantized_store_score_matches_decode_dot() {
        use crate::quantized_search::QuantizedStore;
        let dim = 64;
        let q = TurboQuantProdQuantizer::new(dim, 3, 42, 7).unwrap();
        let v = random_unit_vector(dim, 123);
        let query = random_unit_vector(dim, 456);
        let encoded = q.encode(&v).unwrap();
        let decoded = q.decode(&encoded).unwrap();
        let expected: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();

        let eq = q.encode_query(&query).unwrap();
        let actual = eq.score(&encoded);
        assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
    }

    #[test]
    fn mse_quantized_store_score_matches_decode_dot() {
        use crate::quantized_search::QuantizedStore;
        let dim = 64;
        let q = TurboQuantMseQuantizer::new(dim, 3, 42).unwrap();
        let v = random_unit_vector(dim, 123);
        let query = random_unit_vector(dim, 456);
        let encoded = q.encode(&v).unwrap();
        let decoded = q.decode(&encoded).unwrap();
        let expected: f32 = query.iter().zip(&decoded).map(|(a, b)| a * b).sum();

        let eq = q.encode_query(&query).unwrap();
        let actual = eq.score(&encoded);
        assert!((actual - expected).abs() < 1e-3, "{actual} vs {expected}");
    }

    #[test]
    fn mse_batch_score_matches_per_vector() {
        use crate::quantized_search::QuantizedStore;
        let dim = 64;
        for bits in [1, 2, 3, 4, 8] {
            let q = TurboQuantMseQuantizer::new(dim, bits, 42).unwrap();
            let query = random_unit_vector(dim, 456);
            let n = 50;
            let mut encoded = Vec::new();
            for i in 0..n {
                let v = random_unit_vector(dim, i as u64 + 1000);
                encoded.extend_from_slice(&q.encode(&v).unwrap());
            }

            let eq = q.encode_query(&query).unwrap();
            let batch_scores = eq.score_batch(&encoded);
            assert_eq!(batch_scores.len(), n, "bits={bits}");

            let bytes_per_vec = eq.encoded_bytes_per_vector();
            for (i, chunk) in encoded.chunks_exact(bytes_per_vec).enumerate() {
                let per_vec = eq.score(chunk);
                assert!(
                    (batch_scores[i] - per_vec).abs() < 1e-4,
                    "bits={bits}, i={i}, batch={} per_vec={}",
                    batch_scores[i],
                    per_vec
                );
            }
        }
    }

    #[test]
    fn prod_serialization_roundtrip() {
        let dim = 64;
        let q = TurboQuantProdQuantizer::new(dim, 3, 42, 7).unwrap();
        let json = serde_json::to_string(&q).unwrap();
        let q2: TurboQuantProdQuantizer = serde_json::from_str(&json).unwrap();
        assert_eq!(q.dim, q2.dim);
        assert_eq!(q.bits, q2.bits);
        assert_eq!(q.qjl_seed, q2.qjl_seed);
        assert_eq!(q.mse.rotation_seed, q2.mse.rotation_seed);

        let v = random_unit_vector(dim, 123);
        let e1 = q.encode(&v).unwrap();
        let e2 = q2.encode(&v).unwrap();
        assert_eq!(e1, e2);
    }

    #[test]
    fn prod_batch_score_matches_per_vector() {
        use crate::quantized_search::QuantizedStore;
        let dim = 64;
        let q = TurboQuantProdQuantizer::new(dim, 3, 42, 7).unwrap();
        let query = random_unit_vector(dim, 456);
        let n = 50;
        let mut encoded = Vec::new();
        for i in 0..n {
            let v = random_unit_vector(dim, i as u64 + 1000);
            encoded.extend_from_slice(&q.encode(&v).unwrap());
        }

        let eq = q.encode_query(&query).unwrap();
        let batch_scores = eq.score_batch(&encoded);
        assert_eq!(batch_scores.len(), n);

        let bytes_per_vec = eq.encoded_bytes_per_vector();
        for (i, chunk) in encoded.chunks_exact(bytes_per_vec).enumerate() {
            let per_vec = eq.score(chunk);
            assert!(
                (batch_scores[i] - per_vec).abs() < 1e-4,
                "i={i} batch={} per_vec={}",
                batch_scores[i],
                per_vec
            );
        }
    }

    #[test]
    fn prod_batch_score_various_bit_widths() {
        use crate::quantized_search::QuantizedStore;
        for bits in [2, 3, 4] {
            let dim = 64;
            let q = TurboQuantProdQuantizer::new(dim, bits, 42, 7).unwrap();
            let query = random_unit_vector(dim, 456);
            let v = random_unit_vector(dim, 123);
            let encoded = q.encode(&v).unwrap();
            let eq = q.encode_query(&query).unwrap();
            let per_vec = eq.score(&encoded);
            let batch = eq.score_batch(&encoded);
            assert_eq!(batch.len(), 1);
            assert!(
                (batch[0] - per_vec).abs() < 1e-4,
                "bits={bits} batch={} per_vec={}",
                batch[0],
                per_vec
            );
        }
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let dim = 17;
        for bits in [1, 2, 3, 4, 5, 6, 7, 8] {
            let levels = 1u16 << bits;
            let codes: Vec<u8> = (0..dim).map(|i| (i as u16 % levels) as u8).collect();
            let mut packed = vec![0u8; packed_bytes(dim, bits)];
            pack_codes(&codes, bits, &mut packed);
            let mut unpacked = vec![0u8; dim];
            unpack_codes(&packed, bits, &mut unpacked);
            assert_eq!(codes, unpacked, "bits={bits}");
        }
    }
}
