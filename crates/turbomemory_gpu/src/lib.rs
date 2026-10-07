//! GPU acceleration crate for TurboSuperMemory.
//!
//! Provides optional CUDA-backed kernels:
//! - **Resident exact search** ([`ResidentMatrix`]): a store's full-precision
//!   vectors are kept on the device, so a query is one small upload, one
//!   cuBLAS `gemv` over every vector, and one download of the scores (a
//!   batch of queries is one `gemm`). Nothing is re-uploaded per query,
//!   which is what made the earlier per-query rerank slower than the CPU.
//!   This is the path the storage engine uses for search.
//! - Batched distance computation over a one-off upload (cuBLAS `gemv` /
//!   `gemm`), used to rerank a batch's candidates when the resident index is
//!   not active.
//! - An 8-bit scalar quantized scan and a CSR spreading-activation step
//!   (NVRTC kernels). These are tested here but not wired into the engine.
//!
//! There is no GPU graph index: the resident search is an exact scan, and
//! HNSW construction and traversal stay on the CPU (usearch) as the path for
//! stores that do not fit on the device. (An earlier brute-force "GPU HNSW
//! build" produced a graph nothing searched, at about twice the build time;
//! it was removed.)
//!
//! Every GPU path falls back to the CPU if CUDA is not available, GPU memory
//! is insufficient, or any CUDA error occurs.
//!
//! The design is trait-based so future backends (Vulkan, ROCm, Metal) can be added.

use std::sync::Arc;

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, GpuError>;

/// Errors that can occur in GPU operations.
#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("CUDA not available: {0}")]
    CudaNotAvailable(String),
    #[error("GPU out of memory: need {need_mb} MiB, have {have_mb} MiB")]
    OutOfMemory { need_mb: usize, have_mb: usize },
    #[error("CUDA kernel error: {0}")]
    KernelError(String),
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("Dimension not supported: {0} (must be power of 2 for some GPU kernels)")]
    UnsupportedDimension(usize),
    #[error("Backend not compiled: {0}")]
    BackendNotCompiled(String),
    #[error("Operation timed out: {0}")]
    Timeout(String),
}

/// A GPU backend capable of vector operations.
///
/// Implementations: `CudaBackend` (CUDA), `CpuFallback` (no GPU).
pub trait GpuBackend: Send + Sync {
    /// Human-readable backend name.
    fn name(&self) -> &str;

    /// Total GPU memory in bytes.
    fn total_memory(&self) -> usize;

    /// Available GPU memory in bytes.
    fn available_memory(&self) -> usize;

    /// Upload vectors to GPU device memory.
    ///
    /// `vectors` is a flat slice of `n × dim` f32 values.
    fn upload_vectors(&self, vectors: &[f32], dim: usize) -> Result<DeviceBuffer>;

    /// Upload 8-bit quantized vectors to GPU device memory.
    fn upload_quantized(&self, quantized: &[u8], n: usize, dim: usize) -> Result<DeviceBuffer>;

    /// Compute batched cosine similarity between one query and many vectors.
    ///
    /// Returns `n` scores in `[-1, 1]`.
    /// Batched cosine similarity between one query and all uploaded
    /// vectors.
    ///
    /// Contract: `query.len() == device_vectors.dim` and the uploaded
    /// vectors are unit-normalized. The CUDA implementation computes a
    /// raw dot product (correct for normalized vectors); the CPU fallback
    /// computes true cosine. Inputs violating either precondition are
    /// rejected with [`GpuError::InvalidArgument`], not mis-scored.
    fn batch_cosine_similarity(
        &self,
        query: &[f32],
        device_vectors: &DeviceBuffer,
    ) -> Result<Vec<f32>>;

    /// Compute batched dot product between one query and many vectors.
    fn batch_dot_product(&self, query: &[f32], device_vectors: &DeviceBuffer) -> Result<Vec<f32>>;

    /// Compute batched cosine similarity between M queries and N vectors in a
    /// single matrix multiply (cuBLAS `gemm` on CUDA). This is the GPU-native
    /// batch path: M×N dot products in one kernel call, which is where the GPU
    /// actually wins over CPU (per-query `gemv` loses to CPU SIMD due to
    /// launch/upload overhead, but one `gemm` saturates the GPU).
    ///
    /// - `queries` is a flat `m × dim` row-major slice of `m` query vectors.
    /// - `device_vectors` holds `n` pre-normalized vectors of `dim` each
    ///   (uploaded via [`upload_vectors`]).
    /// - Returns `m * n` scores, row-major (query-major): `scores[i*n + j]` is
    ///   the similarity between query `i` and vector `j`. For pre-normalized
    ///   vectors (as in TSM), dot product == cosine similarity.
    fn batch_cosine_similarity_matrix(
        &self,
        queries: &[f32],
        m: usize,
        device_vectors: &DeviceBuffer,
    ) -> Result<Vec<f32>>;

    /// Scan quantized vectors (Warm/Cold tier) on GPU.
    ///
    /// `quantized` is backend-specific (e.g., CUDA uint8 array).
    /// `query_lut` is a precomputed lookup table for the query.
    fn quantized_scan(
        &self,
        quantized: &DeviceBuffer,
        query_lut: &DeviceBuffer,
        n: usize,
        dim: usize,
        bits_per_dim: u8,
    ) -> Result<Vec<f32>>;

    /// Compute Spreading Activation on GPU via sparse matrix-vector multiplication (SpMV).
    fn spreading_activation_spmv(
        &self,
        row_ptrs: &[i32],
        col_indices: &[i32],
        weights: &[f32],
        seed_energies: &[f32],
        decay: f32,
        hops: usize,
    ) -> Result<Vec<f32>>;

    /// Allocate a resident matrix with room for `capacity_rows` vectors of
    /// `dim` components. It starts with zero rows.
    fn resident_create(&self, dim: usize, capacity_rows: usize) -> Result<ResidentMatrix>;

    /// Append rows (flat, row-major, `rows.len()` a multiple of the matrix
    /// dimension) after the rows already present. Fails with
    /// [`GpuError::InvalidArgument`] if they do not fit the capacity.
    fn resident_append(&self, matrix: &mut ResidentMatrix, rows: &[f32]) -> Result<()>;

    /// Dot product of `query` with every row: one score per row, in row
    /// order. For unit-length rows and a unit-length query this is the
    /// cosine similarity.
    fn resident_scores(&self, matrix: &mut ResidentMatrix, query: &[f32]) -> Result<Vec<f32>>;

    /// Dot products of `count` queries (flat, row-major) with every row.
    /// Returns `count * rows` scores, one contiguous block per query:
    /// `scores[q * rows + r]` is query `q` against row `r`.
    fn resident_scores_batch(
        &self,
        matrix: &mut ResidentMatrix,
        queries: &[f32],
        count: usize,
    ) -> Result<Vec<f32>>;

    /// Free a resident matrix and give its memory back to the system.
    /// Dropping the matrix frees it as well, but a device allocator may keep
    /// freed memory reserved for this process until its next operation, which
    /// never comes once the last engine is closed.
    fn resident_release(&self, matrix: ResidentMatrix) {
        drop(matrix);
    }
}

/// A matrix of row vectors that lives on the backend's device for as long as
/// the value does (on the host for [`CpuFallback`](init_backend)).
///
/// Rows are append-only and addressed by their position, so a caller that
/// appends its vectors in storage order can use the row index as the record
/// offset.
pub struct ResidentMatrix {
    dim: usize,
    rows: usize,
    capacity: usize,
    inner: Box<dyn std::any::Any + Send + Sync>,
}

impl ResidentMatrix {
    /// Rows currently stored.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Rows that fit without reallocating.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Device memory held by the matrix itself, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.capacity * self.dim * std::mem::size_of::<f32>()
    }

    /// Check an append against the matrix shape; returns the row count added.
    fn check_append(&self, rows: &[f32]) -> Result<usize> {
        if !rows.len().is_multiple_of(self.dim) {
            return Err(GpuError::InvalidArgument(format!(
                "appended data length {} is not a multiple of dim {}",
                rows.len(),
                self.dim
            )));
        }
        let added = rows.len() / self.dim;
        if self.rows + added > self.capacity {
            return Err(GpuError::InvalidArgument(format!(
                "resident matrix is full: {} rows + {added} exceeds capacity {}",
                self.rows, self.capacity
            )));
        }
        Ok(added)
    }
}

fn validate_resident_shape(dim: usize, capacity_rows: usize) -> Result<usize> {
    if dim == 0 || capacity_rows == 0 {
        return Err(GpuError::InvalidArgument(
            "resident matrix needs dim > 0 and capacity > 0".into(),
        ));
    }
    // cuBLAS takes 32-bit dimensions.
    if dim > i32::MAX as usize || capacity_rows > i32::MAX as usize {
        return Err(GpuError::InvalidArgument(
            "resident matrix dimensions exceed 32 bits".into(),
        ));
    }
    dim.checked_mul(capacity_rows)
        .ok_or_else(|| GpuError::InvalidArgument("resident matrix size overflows".into()))
}

/// Handle to a GPU device buffer (opaque — backend-specific).
pub struct DeviceBuffer {
    pub(crate) n: usize,
    pub(crate) dim: usize,
    pub(crate) bytes: usize,
    // Backend-specific handle stored as type-erased Arc
    pub(crate) inner: Arc<dyn std::any::Any + Send + Sync>,
}

impl DeviceBuffer {
    pub fn len(&self) -> usize {
        self.n
    }
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn memory_bytes(&self) -> usize {
        self.bytes
    }
}

/// Initialize the best available GPU backend.
///
/// Priority: CUDA → CPU fallback.
/// This is called once at engine startup and cached.
pub fn init_backend() -> Arc<dyn GpuBackend> {
    #[cfg(feature = "cuda")]
    {
        match cuda::CudaBackend::init() {
            Ok(backend) => {
                log::info!("GPU: CUDA backend initialized — {}", backend.name());
                return Arc::new(backend);
            }
            Err(e) => {
                log::warn!(
                    "GPU: CUDA initialization failed ({}), using CPU fallback",
                    e
                );
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        log::info!("GPU: CUDA feature not compiled, using CPU fallback");
    }
    Arc::new(cpu::CpuFallback::new())
}

/// Check if a GPU backend is actually GPU-accelerated (not CPU fallback).
pub fn is_gpu_accelerated(backend: &Arc<dyn GpuBackend>) -> bool {
    backend.name() != "CPU Fallback"
}

/// Validate an upload buffer: `dim` must be non-zero (division-by-zero guard)
/// and must divide the data evenly.
fn validate_upload(vectors: &[f32], dim: usize) -> Result<()> {
    if dim == 0 {
        return Err(GpuError::InvalidArgument("dim must be > 0".into()));
    }
    if !vectors.len().is_multiple_of(dim) {
        return Err(GpuError::InvalidArgument(format!(
            "vectors length {} is not a multiple of dim {dim}",
            vectors.len()
        )));
    }
    Ok(())
}

/// Validate a flat query buffer against the expected `count * dim` shape.
fn validate_queries(queries: &[f32], count: usize, dim: usize) -> Result<()> {
    if queries.len() != count * dim {
        return Err(GpuError::InvalidArgument(format!(
            "queries length {} != {count} * {dim}",
            queries.len()
        )));
    }
    Ok(())
}

// =============================================================================
// CPU Fallback Implementation (always available)
// =============================================================================
mod cpu {
    use super::*;
    use turbomemory_core::{cosine_similarity_batch, dot_product};

    pub struct CpuFallback;

    impl CpuFallback {
        pub fn new() -> Self {
            Self
        }
    }

    impl GpuBackend for CpuFallback {
        fn name(&self) -> &str {
            "CPU Fallback"
        }

        fn total_memory(&self) -> usize {
            0
        }

        fn available_memory(&self) -> usize {
            0
        }

        fn upload_vectors(&self, vectors: &[f32], dim: usize) -> Result<DeviceBuffer> {
            validate_upload(vectors, dim)?;
            // CPU fallback: just wrap the data, no actual GPU upload
            let n = vectors.len() / dim;
            let bytes = std::mem::size_of_val(vectors);
            Ok(DeviceBuffer {
                n,
                dim,
                bytes,
                inner: Arc::new(Vec::from(vectors)),
            })
        }

        fn upload_quantized(&self, quantized: &[u8], n: usize, dim: usize) -> Result<DeviceBuffer> {
            if quantized.len() != n * dim {
                return Err(GpuError::InvalidArgument(
                    "quantized slice size mismatch".into(),
                ));
            }
            Ok(DeviceBuffer {
                n,
                dim,
                bytes: quantized.len(),
                inner: Arc::new(quantized.to_vec()),
            })
        }

        fn batch_cosine_similarity(
            &self,
            query: &[f32],
            device_vectors: &DeviceBuffer,
        ) -> Result<Vec<f32>> {
            let data = device_vectors
                .inner
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| GpuError::InvalidArgument("CPU fallback buffer mismatch".into()))?;
            let n = device_vectors.n;
            let dim = device_vectors.dim;
            validate_queries(query, 1, dim)?;
            let mut refs: Vec<&[f32]> = Vec::with_capacity(n);
            for i in 0..n {
                refs.push(&data[i * dim..(i + 1) * dim]);
            }
            Ok(cosine_similarity_batch(query, &refs))
        }

        fn batch_dot_product(
            &self,
            query: &[f32],
            device_vectors: &DeviceBuffer,
        ) -> Result<Vec<f32>> {
            let data = device_vectors
                .inner
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| GpuError::InvalidArgument("CPU fallback buffer mismatch".into()))?;
            let n = device_vectors.n;
            let dim = device_vectors.dim;
            validate_queries(query, 1, dim)?;
            let mut scores = Vec::with_capacity(n);
            for i in 0..n {
                scores.push(dot_product(query, &data[i * dim..(i + 1) * dim]));
            }
            Ok(scores)
        }

        fn batch_cosine_similarity_matrix(
            &self,
            queries: &[f32],
            m: usize,
            device_vectors: &DeviceBuffer,
        ) -> Result<Vec<f32>> {
            // CPU fallback: loop queries × vectors using the batched SIMD kernel.
            // This keeps the batch API correct without a GPU, just slower than gemm.
            let data = device_vectors
                .inner
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| GpuError::InvalidArgument("CPU fallback buffer mismatch".into()))?;
            let n = device_vectors.n;
            let dim = device_vectors.dim;
            validate_queries(queries, m, dim)?;
            let mut scores = vec![0.0f32; m * n];
            let vec_refs: Vec<&[f32]> = (0..n)
                .map(|j| &data[j * dim..(j + 1) * dim] as &[f32])
                .collect();
            for i in 0..m {
                let q = &queries[i * dim..(i + 1) * dim];
                let row = cosine_similarity_batch(q, &vec_refs);
                scores[i * n..(i + 1) * n].copy_from_slice(&row);
            }
            Ok(scores)
        }

        fn quantized_scan(
            &self,
            quantized: &DeviceBuffer,
            query: &DeviceBuffer,
            n: usize,
            dim: usize,
            bits_per_dim: u8,
        ) -> Result<Vec<f32>> {
            if bits_per_dim != 8 {
                return Err(GpuError::InvalidArgument(
                    "Only 8-bit quantized scan supported".into(),
                ));
            }
            let q_data = quantized.inner.downcast_ref::<Vec<u8>>().ok_or_else(|| {
                GpuError::InvalidArgument("CPU buffer mismatch for quantized data".into())
            })?;
            let query_data = query
                .inner
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| GpuError::InvalidArgument("CPU buffer mismatch for query".into()))?;

            let min_val = -1.0f32;
            let step = 2.0f32 / 255.0f32;
            let mut scores = Vec::with_capacity(n);
            for i in 0..n {
                let vec_slice = &q_data[i * dim..(i + 1) * dim];
                let mut sum = 0.0f32;
                for d in 0..dim {
                    let val = min_val + (vec_slice[d] as f32) * step;
                    sum += val * query_data[d];
                }
                scores.push(sum);
            }
            Ok(scores)
        }

        fn spreading_activation_spmv(
            &self,
            row_ptrs: &[i32],
            col_indices: &[i32],
            weights: &[f32],
            seed_energies: &[f32],
            decay: f32,
            hops: usize,
        ) -> Result<Vec<f32>> {
            let n = seed_energies.len();
            if row_ptrs.len() != n + 1 {
                return Err(GpuError::InvalidArgument("row_ptrs len mismatch".into()));
            }
            let mut current = seed_energies.to_vec();
            for _ in 0..hops {
                let mut next = vec![0.0f32; n];
                for i in 0..n {
                    let start = row_ptrs[i] as usize;
                    let end = row_ptrs[i + 1] as usize;
                    let mut sum = 0.0f32;
                    for edge_idx in start..end {
                        let col = col_indices[edge_idx] as usize;
                        if col < n {
                            sum += current[col] * weights[edge_idx];
                        }
                    }
                    next[i] = (current[i] + sum * decay).clamp(0.0, 10.0);
                }
                current = next;
            }
            Ok(current)
        }

        fn resident_create(&self, dim: usize, capacity_rows: usize) -> Result<ResidentMatrix> {
            let len = validate_resident_shape(dim, capacity_rows)?;
            Ok(ResidentMatrix {
                dim,
                rows: 0,
                capacity: capacity_rows,
                inner: Box::new(Vec::<f32>::with_capacity(len)),
            })
        }

        fn resident_append(&self, matrix: &mut ResidentMatrix, rows: &[f32]) -> Result<()> {
            let added = matrix.check_append(rows)?;
            let data = matrix
                .inner
                .downcast_mut::<Vec<f32>>()
                .ok_or_else(|| GpuError::InvalidArgument("CPU resident matrix mismatch".into()))?;
            data.extend_from_slice(rows);
            matrix.rows += added;
            Ok(())
        }

        fn resident_scores(&self, matrix: &mut ResidentMatrix, query: &[f32]) -> Result<Vec<f32>> {
            validate_queries(query, 1, matrix.dim)?;
            let data = matrix
                .inner
                .downcast_ref::<Vec<f32>>()
                .ok_or_else(|| GpuError::InvalidArgument("CPU resident matrix mismatch".into()))?;
            Ok(data
                .chunks_exact(matrix.dim)
                .map(|row| dot_product(query, row))
                .collect())
        }

        fn resident_scores_batch(
            &self,
            matrix: &mut ResidentMatrix,
            queries: &[f32],
            count: usize,
        ) -> Result<Vec<f32>> {
            validate_queries(queries, count, matrix.dim)?;
            let mut out = Vec::with_capacity(count * matrix.rows);
            for query in queries.chunks_exact(matrix.dim) {
                out.extend(self.resident_scores(matrix, query)?);
            }
            Ok(out)
        }
    }
}

// =============================================================================
// CUDA Backend (only compiled with "cuda" feature)
// =============================================================================
#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, Gemv, GemvConfig};
    use cudarc::driver::{
        CudaContext, CudaModule, CudaSlice, DriverError, LaunchConfig, PushKernelArg,
    };
    use cudarc::nvrtc::compile_ptx;
    use std::sync::Mutex;

    const CUDA_KERNELS: &str = r#"
extern "C" __global__ void quantized_scan_u8_kernel(
    const unsigned char* __restrict__ quantized_vectors,
    const float* __restrict__ query,
    float* __restrict__ scores_out,
    int n,
    int dim,
    float min_val,
    float step
) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_idx >= n) return;

    const unsigned char* vec_ptr = quantized_vectors + (size_t)vec_idx * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        float val = min_val + ((float)vec_ptr[d]) * step;
        sum += val * query[d];
    }
    scores_out[vec_idx] = sum;
}

extern "C" __global__ void spreading_activation_csr_kernel(
    const int* __restrict__ row_ptrs,
    const int* __restrict__ col_indices,
    const float* __restrict__ weights,
    const float* __restrict__ current_energy,
    float* __restrict__ next_energy,
    int n,
    float decay
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int row_start = row_ptrs[i];
    int row_end = row_ptrs[i + 1];
    float sum = 0.0f;

    for (int edge = row_start; edge < row_end; ++edge) {
        int col = col_indices[edge];
        if (col < n) {
            sum += current_energy[col] * weights[edge];
        }
    }

    float total = current_energy[i] + sum * decay;
    if (total < 0.0f) total = 0.0f;
    if (total > 10.0f) total = 10.0f;
    next_energy[i] = total;
}
"#;

    /// CUDA GPU backend using cudarc.
    pub struct CudaBackend {
        ctx: Arc<CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
        total_mem: usize,
        // cuBLAS handle (initialized lazily)
        cublas: Mutex<Option<CudaBlas>>,
        // Compiled CUDA kernels module (initialized lazily)
        module: Mutex<Option<Arc<CudaModule>>>,
    }

    impl CudaBackend {
        pub fn init() -> Result<Self> {
            let ctx = CudaContext::new(0).map_err(|e: DriverError| {
                GpuError::CudaNotAvailable(format!("Failed to create CUDA context: {e}"))
            })?;

            let stream = ctx.default_stream();

            let total_mem = ctx.total_mem().map_err(|e: DriverError| {
                GpuError::CudaNotAvailable(format!("Failed to query GPU memory: {e}"))
            })?;

            log::info!(
                "CUDA device: {}, {} MiB total memory",
                ctx.name().unwrap_or_else(|_| "unknown".into()),
                total_mem / (1024 * 1024)
            );

            Ok(Self {
                ctx,
                stream: stream.clone(),
                total_mem,
                cublas: Mutex::new(None),
                module: Mutex::new(None),
            })
        }

        fn module(&self) -> Result<Arc<CudaModule>> {
            let mut guard = self
                .module
                .lock()
                .map_err(|_| GpuError::KernelError("module mutex poisoned".into()))?;
            if guard.is_none() {
                let ptx = compile_ptx(CUDA_KERNELS).map_err(|e| {
                    GpuError::KernelError(format!("NVRTC PTX compilation failed: {e}"))
                })?;
                let module = self.ctx.load_module(ptx).map_err(|e| {
                    GpuError::KernelError(format!("Failed to load CUDA module: {e}"))
                })?;
                *guard = Some(module);
            }
            Ok(guard.as_ref().unwrap().clone())
        }

        fn cublas(&self) -> Result<std::sync::MutexGuard<'_, Option<CudaBlas>>> {
            let mut guard = self
                .cublas
                .lock()
                .map_err(|_| GpuError::KernelError("cublas mutex poisoned".into()))?;
            if guard.is_none() {
                *guard = Some(CudaBlas::new(self.stream.clone()).map_err(|e| {
                    GpuError::CudaNotAvailable(format!("Failed to create cuBLAS: {e}"))
                })?);
            }
            Ok(guard)
        }

        fn check_memory(&self, need_bytes: usize) -> Result<()> {
            let available = self.available_memory();
            if need_bytes > available {
                return Err(GpuError::OutOfMemory {
                    need_mb: need_bytes / (1024 * 1024),
                    have_mb: available / (1024 * 1024),
                });
            }
            Ok(())
        }
    }

    impl GpuBackend for CudaBackend {
        fn name(&self) -> &str {
            "CUDA"
        }

        fn total_memory(&self) -> usize {
            self.total_mem
        }

        fn available_memory(&self) -> usize {
            // Free device memory right now; other processes share the card.
            self.ctx
                .mem_get_info()
                .map(|(free, _total)| free)
                .unwrap_or(self.total_mem)
        }

        fn upload_vectors(&self, vectors: &[f32], dim: usize) -> Result<DeviceBuffer> {
            validate_upload(vectors, dim)?;
            let n = vectors.len() / dim;
            let bytes = std::mem::size_of_val(vectors);
            self.check_memory(bytes)?;

            let slice: CudaSlice<f32> =
                self.stream
                    .clone_htod(vectors)
                    .map_err(|_e| GpuError::OutOfMemory {
                        need_mb: bytes / (1024 * 1024),
                        have_mb: self.total_mem / (1024 * 1024),
                    })?;

            Ok(DeviceBuffer {
                n,
                dim,
                bytes,
                inner: Arc::new(CudaBufferWrapper { slice }),
            })
        }

        fn upload_quantized(&self, quantized: &[u8], n: usize, dim: usize) -> Result<DeviceBuffer> {
            if quantized.len() != n * dim {
                return Err(GpuError::InvalidArgument(
                    "quantized slice size mismatch".into(),
                ));
            }
            let bytes = quantized.len();
            self.check_memory(bytes)?;

            let slice: CudaSlice<u8> =
                self.stream
                    .clone_htod(quantized)
                    .map_err(|_e| GpuError::OutOfMemory {
                        need_mb: bytes / (1024 * 1024),
                        have_mb: self.total_mem / (1024 * 1024),
                    })?;

            Ok(DeviceBuffer {
                n,
                dim,
                bytes,
                inner: Arc::new(CudaU8BufferWrapper { slice }),
            })
        }

        fn batch_cosine_similarity(
            &self,
            query: &[f32],
            device_vectors: &DeviceBuffer,
        ) -> Result<Vec<f32>> {
            let wrapper = device_vectors
                .inner
                .downcast_ref::<CudaBufferWrapper>()
                .ok_or_else(|| GpuError::InvalidArgument("CUDA buffer mismatch".into()))?;

            let n = device_vectors.n;
            let dim = device_vectors.dim;
            // cuBLAS would read `dim` elements from the device buffer
            // regardless of the query's real length.
            validate_queries(query, 1, dim)?;

            // Upload query to device
            let query_dev: CudaSlice<f32> = self
                .stream
                .clone_htod(query)
                .map_err(|e| GpuError::KernelError(format!("Failed to upload query: {e}")))?;

            // Allocate output buffer
            let mut scores_dev: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(|e| {
                GpuError::KernelError(format!("Failed to allocate scores buffer: {e}"))
            })?;

            // Use cuBLAS for batched dot product: scores = vectors^T × query
            // vectors is n×dim stored row-major, query is dim×1
            // We need gemv: y = alpha * A^T * x + beta * y
            // A is dim×n (vectors transposed), x is dim, y is n
            {
                let cublas_guard = self.cublas()?;
                let cublas = cublas_guard
                    .as_ref()
                    .ok_or_else(|| GpuError::CudaNotAvailable("cuBLAS not initialized".into()))?;

                // For cosine similarity, we need normalized vectors
                // Simplified: assume vectors are pre-normalized (as in TSM)
                // Then cosine similarity = dot product
                unsafe {
                    cublas
                        .gemv(
                            GemvConfig {
                                trans: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                                m: dim as i32,
                                n: n as i32,
                                alpha: 1.0f32,
                                lda: dim as i32,
                                incx: 1,
                                beta: 0.0f32,
                                incy: 1,
                            },
                            &wrapper.slice,
                            &query_dev,
                            &mut scores_dev,
                        )
                        .map_err(|e| GpuError::KernelError(format!("cuBLAS gemv failed: {e}")))?;
                }
            }

            // Download scores
            let scores = self
                .stream
                .clone_dtoh(&scores_dev)
                .map_err(|e| GpuError::KernelError(format!("Failed to download scores: {e}")))?;

            Ok(scores)
        }

        fn batch_dot_product(
            &self,
            query: &[f32],
            device_vectors: &DeviceBuffer,
        ) -> Result<Vec<f32>> {
            // Same as cosine similarity for pre-normalized vectors
            self.batch_cosine_similarity(query, device_vectors)
        }

        fn batch_cosine_similarity_matrix(
            &self,
            queries: &[f32],
            m: usize,
            device_vectors: &DeviceBuffer,
        ) -> Result<Vec<f32>> {
            // cuBLAS gemm: C = Q · V^T, where Q is M×dim and V is N×dim
            // (both pre-normalized, so dot == cosine).
            //
            // cuBLAS is column-major. Our row-major M×dim query buffer is a
            // dim×M column-major matrix Q_cb (lda = dim); transpose it (OP_T)
            // to get the M×dim A operand. Our row-major N×dim device buffer is
            // a dim×N column-major matrix V_cb (ldb = dim); use OP_N so B is
            // dim×N. Then C = A·B is M×N, stored column-major as ldc = M.
            let wrapper = device_vectors
                .inner
                .downcast_ref::<CudaBufferWrapper>()
                .ok_or_else(|| GpuError::InvalidArgument("CUDA buffer mismatch".into()))?;

            let n = device_vectors.n;
            let dim = device_vectors.dim;
            if m == 0 || n == 0 {
                return Ok(Vec::new());
            }
            // Same contract as gemv: cuBLAS reads m*dim elements regardless.
            validate_queries(queries, m, dim)?;

            // Upload the M×dim query matrix (one host->device copy for all queries).
            let queries_dev: CudaSlice<f32> = self.stream.clone_htod(queries).map_err(|e| {
                GpuError::KernelError(format!("Failed to upload query matrix: {e}"))
            })?;

            // Output M×N scores (column-major: column j has all M query scores
            // against vector j; ldc = M rows).
            let mut scores_dev: CudaSlice<f32> = self.stream.alloc_zeros(m * n).map_err(|e| {
                GpuError::KernelError(format!("Failed to allocate scores matrix: {e}"))
            })?;

            {
                let cublas_guard = self.cublas()?;
                let cublas = cublas_guard
                    .as_ref()
                    .ok_or_else(|| GpuError::CudaNotAvailable("cuBLAS not initialized".into()))?;

                // C(m×n) = Q(m×k) · V^T(k×n), with k = dim.
                unsafe {
                    cublas
                        .gemm(
                            GemmConfig {
                                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T, // Q_cb^T -> M×dim
                                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N, // V_cb -> dim×N
                                m: m as i32,
                                n: n as i32,
                                k: dim as i32,
                                alpha: 1.0f32,
                                lda: dim as i32, // leading dim of Q_cb (dim×M storage)
                                ldb: dim as i32, // leading dim of V_cb (dim×N storage)
                                beta: 0.0f32,
                                ldc: m as i32, // leading dim of C (M×N storage)
                            },
                            &queries_dev,
                            &wrapper.slice,
                            &mut scores_dev,
                        )
                        .map_err(|e| GpuError::KernelError(format!("cuBLAS gemm failed: {e}")))?;
                }
            }

            // Download M×N column-major scores, then transpose to row-major
            // (query-major: scores[i*n + j] = query i vs vector j).
            let col_major = self.stream.clone_dtoh(&scores_dev).map_err(|e| {
                GpuError::KernelError(format!("Failed to download scores matrix: {e}"))
            })?;
            let mut row_major = vec![0.0f32; m * n];
            for i in 0..m {
                for j in 0..n {
                    row_major[i * n + j] = col_major[j * m + i];
                }
            }
            Ok(row_major)
        }

        fn quantized_scan(
            &self,
            quantized: &DeviceBuffer,
            query: &DeviceBuffer,
            n: usize,
            dim: usize,
            bits_per_dim: u8,
        ) -> Result<Vec<f32>> {
            if bits_per_dim != 8 {
                return Err(GpuError::InvalidArgument(
                    "Only 8-bit quantized scan supported on GPU".into(),
                ));
            }
            if n == 0 || dim == 0 {
                return Ok(Vec::new());
            }
            let q_wrapper = quantized
                .inner
                .downcast_ref::<CudaU8BufferWrapper>()
                .ok_or_else(|| {
                    GpuError::InvalidArgument("CUDA buffer mismatch for quantized vectors".into())
                })?;
            let query_wrapper =
                query
                    .inner
                    .downcast_ref::<CudaBufferWrapper>()
                    .ok_or_else(|| {
                        GpuError::InvalidArgument("CUDA buffer mismatch for query".into())
                    })?;

            let module = self.module()?;
            let kernel = module
                .load_function("quantized_scan_u8_kernel")
                .map_err(|e| {
                    GpuError::KernelError(format!("Failed to load quantized scan kernel: {e}"))
                })?;

            let mut scores_dev: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(|e| {
                GpuError::KernelError(format!("Failed to allocate scores buffer: {e}"))
            })?;

            let min_val = -1.0f32;
            let step = 2.0f32 / 255.0f32;
            let n_i32 = n as i32;
            let dim_i32 = dim as i32;

            let mut builder = self.stream.launch_builder(&kernel);
            builder.arg(&q_wrapper.slice);
            builder.arg(&query_wrapper.slice);
            builder.arg(&mut scores_dev);
            builder.arg(&n_i32);
            builder.arg(&dim_i32);
            builder.arg(&min_val);
            builder.arg(&step);

            unsafe {
                builder
                    .launch(LaunchConfig::for_num_elems(n as u32))
                    .map_err(|e| {
                        GpuError::KernelError(format!("CUDA quantized scan launch failed: {e}"))
                    })?;
            }

            self.stream
                .clone_dtoh(&scores_dev)
                .map_err(|e| GpuError::KernelError(format!("Failed to download scores: {e}")))
        }

        fn spreading_activation_spmv(
            &self,
            row_ptrs: &[i32],
            col_indices: &[i32],
            weights: &[f32],
            seed_energies: &[f32],
            decay: f32,
            hops: usize,
        ) -> Result<Vec<f32>> {
            let n = seed_energies.len();
            if row_ptrs.len() != n + 1 {
                return Err(GpuError::InvalidArgument("row_ptrs len mismatch".into()));
            }
            if hops == 0 || n == 0 {
                return Ok(seed_energies.to_vec());
            }

            let module = self.module()?;
            let kernel = module
                .load_function("spreading_activation_csr_kernel")
                .map_err(|e| {
                    GpuError::KernelError(format!(
                        "Failed to load spreading activation kernel: {e}"
                    ))
                })?;

            let row_ptrs_dev = self
                .stream
                .clone_htod(row_ptrs)
                .map_err(|e| GpuError::KernelError(format!("Failed to upload row_ptrs: {e}")))?;
            let col_indices_dev = self
                .stream
                .clone_htod(col_indices)
                .map_err(|e| GpuError::KernelError(format!("Failed to upload col_indices: {e}")))?;
            let weights_dev = self
                .stream
                .clone_htod(weights)
                .map_err(|e| GpuError::KernelError(format!("Failed to upload weights: {e}")))?;

            let mut curr_dev = self.stream.clone_htod(seed_energies).map_err(|e| {
                GpuError::KernelError(format!("Failed to upload seed_energies: {e}"))
            })?;
            let mut next_dev: CudaSlice<f32> = self.stream.alloc_zeros(n).map_err(|e| {
                GpuError::KernelError(format!("Failed to allocate next_energy: {e}"))
            })?;

            let n_i32 = n as i32;

            for _ in 0..hops {
                let mut builder = self.stream.launch_builder(&kernel);
                builder.arg(&row_ptrs_dev);
                builder.arg(&col_indices_dev);
                builder.arg(&weights_dev);
                builder.arg(&curr_dev);
                builder.arg(&mut next_dev);
                builder.arg(&n_i32);
                builder.arg(&decay);

                unsafe {
                    builder
                        .launch(LaunchConfig::for_num_elems(n as u32))
                        .map_err(|e| {
                            GpuError::KernelError(format!("CUDA SpMV launch failed: {e}"))
                        })?;
                }
                std::mem::swap(&mut curr_dev, &mut next_dev);
            }

            self.stream.clone_dtoh(&curr_dev).map_err(|e| {
                GpuError::KernelError(format!("Failed to download activation energies: {e}"))
            })
        }

        fn resident_create(&self, dim: usize, capacity_rows: usize) -> Result<ResidentMatrix> {
            let len = validate_resident_shape(dim, capacity_rows)?;
            let bytes = (len + dim + capacity_rows) * std::mem::size_of::<f32>();
            self.check_memory(bytes)?;
            let oom = |_e: DriverError| GpuError::OutOfMemory {
                need_mb: bytes / (1024 * 1024),
                have_mb: self.available_memory() / (1024 * 1024),
            };
            // The matrix is left uninitialised: only rows that were appended
            // are ever read (`gemv`/`gemm` are told the row count).
            let data: CudaSlice<f32> = unsafe { self.stream.alloc(len) }.map_err(oom)?;
            let query: CudaSlice<f32> = self.stream.alloc_zeros(dim).map_err(oom)?;
            let scores: CudaSlice<f32> = self.stream.alloc_zeros(capacity_rows).map_err(oom)?;
            Ok(ResidentMatrix {
                dim,
                rows: 0,
                capacity: capacity_rows,
                inner: Box::new(CudaResident {
                    data,
                    query,
                    scores,
                }),
            })
        }

        fn resident_release(&self, matrix: ResidentMatrix) {
            drop(matrix);
            // Frees are ordered on the stream and land in the device's
            // memory pool. The pool returns memory to the system when the
            // stream is synchronized; trim it as well in case it holds more.
            let released = self.stream.synchronize().and_then(|()| unsafe {
                let pool =
                    cudarc::driver::result::device::get_default_mem_pool(self.ctx.cu_device())?;
                cudarc::driver::result::mem_pool::trim_to(pool, 0)
            });
            if let Err(e) = released {
                log::debug!("device memory not returned to the system yet: {e}");
            }
        }

        fn resident_append(&self, matrix: &mut ResidentMatrix, rows: &[f32]) -> Result<()> {
            let added = matrix.check_append(rows)?;
            if added == 0 {
                return Ok(());
            }
            let start = matrix.rows * matrix.dim;
            let resident = cuda_resident(&mut matrix.inner)?;
            let mut tail = resident.data.slice_mut(start..start + rows.len());
            self.stream
                .memcpy_htod(rows, &mut tail)
                .map_err(|e| GpuError::KernelError(format!("Failed to upload rows: {e}")))?;
            matrix.rows += added;
            Ok(())
        }

        fn resident_scores(&self, matrix: &mut ResidentMatrix, query: &[f32]) -> Result<Vec<f32>> {
            validate_queries(query, 1, matrix.dim)?;
            let (rows, dim) = (matrix.rows, matrix.dim);
            if rows == 0 {
                return Ok(Vec::new());
            }
            let resident = cuda_resident(&mut matrix.inner)?;
            self.stream
                .memcpy_htod(query, &mut resident.query)
                .map_err(|e| GpuError::KernelError(format!("Failed to upload query: {e}")))?;
            {
                let cublas_guard = self.cublas()?;
                let cublas = cublas_guard
                    .as_ref()
                    .ok_or_else(|| GpuError::CudaNotAvailable("cuBLAS not initialized".into()))?;
                // The row-major rows x dim buffer is a column-major dim x rows
                // matrix A; scores = A^T * query has one entry per row.
                let mut out = resident.scores.slice_mut(0..rows);
                unsafe {
                    cublas
                        .gemv(
                            GemvConfig {
                                trans: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                                m: dim as i32,
                                n: rows as i32,
                                alpha: 1.0f32,
                                lda: dim as i32,
                                incx: 1,
                                beta: 0.0f32,
                                incy: 1,
                            },
                            &resident.data,
                            &resident.query,
                            &mut out,
                        )
                        .map_err(|e| GpuError::KernelError(format!("cuBLAS gemv failed: {e}")))?;
                }
            }
            let mut scores = vec![0.0f32; rows];
            self.stream
                .memcpy_dtoh(&resident.scores.slice(0..rows), &mut scores)
                .map_err(|e| GpuError::KernelError(format!("Failed to download scores: {e}")))?;
            self.stream
                .synchronize()
                .map_err(|e| GpuError::KernelError(format!("stream synchronize failed: {e}")))?;
            Ok(scores)
        }

        fn resident_scores_batch(
            &self,
            matrix: &mut ResidentMatrix,
            queries: &[f32],
            count: usize,
        ) -> Result<Vec<f32>> {
            validate_queries(queries, count, matrix.dim)?;
            let (rows, dim) = (matrix.rows, matrix.dim);
            if rows == 0 || count == 0 {
                return Ok(Vec::new());
            }
            if count > i32::MAX as usize {
                return Err(GpuError::InvalidArgument("too many queries".into()));
            }
            let resident = cuda_resident(&mut matrix.inner)?;
            let queries_dev: CudaSlice<f32> = self.stream.clone_htod(queries).map_err(|e| {
                GpuError::KernelError(format!("Failed to upload query matrix: {e}"))
            })?;
            let mut scores_dev: CudaSlice<f32> =
                self.stream.alloc_zeros(rows * count).map_err(|e| {
                    GpuError::KernelError(format!("Failed to allocate scores matrix: {e}"))
                })?;
            {
                let cublas_guard = self.cublas()?;
                let cublas = cublas_guard
                    .as_ref()
                    .ok_or_else(|| GpuError::CudaNotAvailable("cuBLAS not initialized".into()))?;
                // C (rows x count) = V^T (rows x dim) * Q (dim x count), with V
                // the column-major dim x rows resident matrix and Q the
                // column-major dim x count query matrix. Column q of C is
                // query q against every row, i.e. one contiguous block each.
                unsafe {
                    cublas
                        .gemm(
                            GemmConfig {
                                transa: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_T,
                                transb: cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                                m: rows as i32,
                                n: count as i32,
                                k: dim as i32,
                                alpha: 1.0f32,
                                lda: dim as i32,
                                ldb: dim as i32,
                                beta: 0.0f32,
                                ldc: rows as i32,
                            },
                            &resident.data,
                            &queries_dev,
                            &mut scores_dev,
                        )
                        .map_err(|e| GpuError::KernelError(format!("cuBLAS gemm failed: {e}")))?;
                }
            }
            let scores = self.stream.clone_dtoh(&scores_dev).map_err(|e| {
                GpuError::KernelError(format!("Failed to download scores matrix: {e}"))
            })?;
            self.stream
                .synchronize()
                .map_err(|e| GpuError::KernelError(format!("stream synchronize failed: {e}")))?;
            Ok(scores)
        }
    }

    /// Device side of a [`ResidentMatrix`]: the vectors plus two scratch
    /// buffers reused by every single-query search, so a search allocates
    /// nothing on the device.
    struct CudaResident {
        data: CudaSlice<f32>,
        query: CudaSlice<f32>,
        scores: CudaSlice<f32>,
    }

    // SAFETY: the buffers are only touched through `&mut ResidentMatrix`, and
    // every device call binds the context to the calling thread first.
    unsafe impl Send for CudaResident {}
    unsafe impl Sync for CudaResident {}

    fn cuda_resident(
        inner: &mut Box<dyn std::any::Any + Send + Sync>,
    ) -> Result<&mut CudaResident> {
        inner
            .downcast_mut::<CudaResident>()
            .ok_or_else(|| GpuError::InvalidArgument("CUDA resident matrix mismatch".into()))
    }

    /// Wrapper to make CudaSlice<f32> Send + Sync for Arc storage.
    struct CudaBufferWrapper {
        slice: CudaSlice<f32>,
    }

    unsafe impl Send for CudaBufferWrapper {}
    unsafe impl Sync for CudaBufferWrapper {}

    /// Wrapper to make CudaSlice<u8> Send + Sync for Arc storage.
    struct CudaU8BufferWrapper {
        slice: CudaSlice<u8>,
    }

    unsafe impl Send for CudaU8BufferWrapper {}
    unsafe impl Sync for CudaU8BufferWrapper {}
}

// Stub module when CUDA is not compiled
#[cfg(not(feature = "cuda"))]
mod cuda {
    use super::*;
    pub struct CudaBackend;
    impl CudaBackend {
        pub fn init() -> Result<Self> {
            Err(GpuError::BackendNotCompiled(
                "CUDA feature not enabled. Rebuild with --features cuda".into(),
            ))
        }
    }
}

pub use cuda::CudaBackend;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_fallback_quantized_scan() {
        let backend = init_backend();
        let dim = 8;
        let n = 4;
        let quantized_data = vec![128u8; n * dim];
        let query_data = vec![1.0f32; dim];

        let q_buf = backend.upload_quantized(&quantized_data, n, dim).unwrap();
        let query_buf = backend.upload_vectors(&query_data, dim).unwrap();

        let scores = backend
            .quantized_scan(&q_buf, &query_buf, n, dim, 8)
            .unwrap();
        assert_eq!(scores.len(), n);
        for &s in &scores {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn test_cpu_fallback_spreading_activation_spmv() {
        let backend = init_backend();
        let n = 3;
        // Node 1 receives energy from Node 0 (weight 0.8)
        // Node 2 receives energy from Node 1 (weight 0.5)
        let row_ptrs = vec![0, 0, 1, 2];
        let col_indices = vec![0, 1];
        let weights = vec![0.8f32, 0.5f32];
        let seed_energies = vec![1.0f32, 0.0f32, 0.0f32];

        let result = backend
            .spreading_activation_spmv(&row_ptrs, &col_indices, &weights, &seed_energies, 0.5, 2)
            .unwrap();

        assert_eq!(result.len(), n);
        assert!(result[0] >= 1.0);
        assert!(result[1] > 0.0);
        assert!(result[2] > 0.0);
    }

    /// The resident matrix must return the dot product of the query with each
    /// row, in row order, for single queries and for batches, across
    /// appends. Runs on the CPU fallback, and on the device with `cuda`.
    fn check_resident(backend: &dyn GpuBackend) {
        let dim = 6;
        let row = |i: usize| -> Vec<f32> {
            (0..dim)
                .map(|d| ((i * 7 + d * 3) % 11) as f32 - 5.0)
                .collect()
        };
        let mut matrix = backend.resident_create(dim, 10).unwrap();
        assert_eq!(
            (matrix.rows(), matrix.capacity(), matrix.dim()),
            (0, 10, dim)
        );
        assert!(backend
            .resident_scores(&mut matrix, &row(0))
            .unwrap()
            .is_empty());

        let first: Vec<f32> = (0..4).flat_map(row).collect();
        backend.resident_append(&mut matrix, &first).unwrap();
        let second: Vec<f32> = (4..7).flat_map(row).collect();
        backend.resident_append(&mut matrix, &second).unwrap();
        assert_eq!(matrix.rows(), 7);

        let expect = |q: &[f32]| -> Vec<f32> {
            (0..7)
                .map(|i| row(i).iter().zip(q).map(|(a, b)| a * b).sum())
                .collect()
        };
        for q in [row(2), row(9), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] {
            let got = backend.resident_scores(&mut matrix, &q).unwrap();
            assert_eq!(got.len(), 7);
            for (g, e) in got.iter().zip(expect(&q)) {
                assert!((g - e).abs() < 1e-3, "{g} vs {e}");
            }
        }

        let queries: Vec<f32> = [row(1), row(5), row(8)].concat();
        let batch = backend
            .resident_scores_batch(&mut matrix, &queries, 3)
            .unwrap();
        assert_eq!(batch.len(), 3 * 7);
        for (qi, q) in [row(1), row(5), row(8)].iter().enumerate() {
            for (r, e) in expect(q).iter().enumerate() {
                assert!((batch[qi * 7 + r] - e).abs() < 1e-3, "query {qi} row {r}");
            }
        }

        // Shape errors are reported, never read out of bounds.
        assert!(backend.resident_append(&mut matrix, &[1.0; 5]).is_err());
        assert!(backend
            .resident_append(&mut matrix, &vec![0.0; dim * 4])
            .is_err());
        assert!(backend.resident_scores(&mut matrix, &[1.0; 5]).is_err());
        assert!(backend
            .resident_scores_batch(&mut matrix, &queries, 2)
            .is_err());
        assert!(backend.resident_create(0, 4).is_err());
        assert_eq!(matrix.rows(), 7, "failed appends change nothing");
    }

    #[test]
    fn test_cpu_fallback_resident_matrix() {
        check_resident(&cpu::CpuFallback::new());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_resident_matrix() {
        // No device: nothing to check.
        if let Ok(backend) = cuda::CudaBackend::init() {
            check_resident(&backend);
            assert!(backend.available_memory() <= backend.total_memory());
        }
    }

    /// Releasing a matrix returns its memory to the device, not only to this
    /// process's allocator: a closed engine must not keep holding it.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_resident_release_frees_device_memory() {
        let Ok(backend) = cuda::CudaBackend::init() else {
            return; // no device
        };
        const MIB: usize = 1024 * 1024;
        let (dim, rows) = (256usize, 200_000usize); // 195 MiB
        let mut matrix = backend.resident_create(dim, rows).unwrap();
        backend
            .resident_append(&mut matrix, &vec![0.5f32; dim * rows])
            .unwrap();
        backend
            .resident_scores(&mut matrix, &vec![1.0f32; dim])
            .unwrap();
        let held = backend.available_memory();
        backend.resident_release(matrix);
        let after = backend.available_memory();
        assert!(
            after >= held + 150 * MIB,
            "free device memory went from {} to {} MiB after releasing 195 MiB",
            held / MIB,
            after / MIB
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_quantized_scan_and_spmv() {
        let backend = match cuda::CudaBackend::init() {
            Ok(b) => b,
            Err(_) => return, // Skip if CUDA device cannot be initialized
        };

        let dim = 8;
        let n = 4;
        let quantized_data = vec![128u8; n * dim];
        let query_data = vec![1.0f32; dim];

        let q_buf = backend.upload_quantized(&quantized_data, n, dim).unwrap();
        let query_buf = backend.upload_vectors(&query_data, dim).unwrap();

        let scores = backend
            .quantized_scan(&q_buf, &query_buf, n, dim, 8)
            .unwrap();
        assert_eq!(scores.len(), n);
        for &s in &scores {
            assert!(s.is_finite());
        }

        // Test CUDA SpMV
        // Node 1 receives energy from Node 0 (weight 0.8)
        // Node 2 receives energy from Node 1 (weight 0.5)
        let row_ptrs = vec![0, 0, 1, 2];
        let col_indices = vec![0, 1];
        let weights = vec![0.8f32, 0.5f32];
        let seed_energies = vec![1.0f32, 0.0f32, 0.0f32];

        let spmv_res = backend
            .spreading_activation_spmv(&row_ptrs, &col_indices, &weights, &seed_energies, 0.5, 2)
            .unwrap();

        assert_eq!(spmv_res.len(), 3);
        assert!(spmv_res[0] >= 1.0);
        assert!(spmv_res[1] > 0.0);
        assert!(spmv_res[2] > 0.0);
    }
}
