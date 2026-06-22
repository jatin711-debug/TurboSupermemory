//! GPU acceleration crate for TurboSuperMemory.
//!
//! Provides optional CUDA-backed paths for:
//! - HNSW index construction (cuVS/RAFT CAGRA or custom CUDA HNSW)
//! - Batched distance computation (cuBLAS)
//! - Quantized tier scanning (CUDA kernels for scalar/sign/TurboQuant)
//!
//! All GPU paths silently fall back to CPU if:
//! - CUDA is not available (no GPU, no drivers)
//! - GPU memory is insufficient
//! - Any CUDA error occurs
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

    /// Compute batched cosine similarity between one query and many vectors.
    ///
    /// Returns `n` scores in `[-1, 1]`.
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

    /// Build an approximate nearest neighbor index on GPU.
    ///
    /// Returns a GPU-native index handle. The index format is backend-specific.
    fn build_ann_index(
        &self,
        device_vectors: &DeviceBuffer,
        dim: usize,
        config: &AnnBuildConfig,
    ) -> Result<Box<dyn GpuAnnIndex>>;

    /// Search the GPU ANN index.
    fn ann_search(
        &self,
        index: &dyn GpuAnnIndex,
        query: &[f32],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>>;

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

/// GPU-native approximate nearest neighbor index (opaque).
pub trait GpuAnnIndex: Send + Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn dim(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    /// Downcast to concrete type for backend-specific operations.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Configuration for GPU ANN index building.
#[derive(Debug, Clone)]
pub struct AnnBuildConfig {
    /// HNSW M (connectivity) or CAGRA graph degree.
    pub max_edges: usize,
    /// Construction beam width (HNSW ef_construction or CAGRA build params).
    pub ef_construction: usize,
    /// Search beam width.
    pub ef_search: usize,
    /// Target recall (0.0-1.0). Higher = slower build, better quality.
    pub target_recall: f32,
    /// Maximum GPU memory to use for build (bytes). 0 = unlimited.
    pub max_build_memory: usize,
}

impl Default for AnnBuildConfig {
    fn default() -> Self {
        Self {
            max_edges: 64,
            ef_construction: 800,
            ef_search: 256,
            target_recall: 0.95,
            max_build_memory: 0,
        }
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

        fn build_ann_index(
            &self,
            _device_vectors: &DeviceBuffer,
            _dim: usize,
            _config: &AnnBuildConfig,
        ) -> Result<Box<dyn GpuAnnIndex>> {
            Err(GpuError::BackendNotCompiled(
                "CPU fallback cannot build ANN index — use turbomemory_storage::UsearchIndex"
                    .into(),
            ))
        }

        fn ann_search(
            &self,
            _index: &dyn GpuAnnIndex,
            _query: &[f32],
            _top_k: usize,
        ) -> Result<Vec<(usize, f32)>> {
            Err(GpuError::BackendNotCompiled(
                "CPU fallback cannot search ANN index".into(),
            ))
        }

        fn quantized_scan(
            &self,
            _quantized: &DeviceBuffer,
            _query_lut: &DeviceBuffer,
            _n: usize,
            _dim: usize,
            _bits_per_dim: u8,
        ) -> Result<Vec<f32>> {
            Err(GpuError::BackendNotCompiled(
                "CPU fallback cannot run quantized scan — use turbomemory_core::quantized_search"
                    .into(),
            ))
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
    use cudarc::driver::{CudaContext, CudaSlice, DriverError};
    use std::sync::Mutex;

    /// CUDA GPU backend using cudarc.
    pub struct CudaBackend {
        ctx: Arc<CudaContext>,
        stream: Arc<cudarc::driver::CudaStream>,
        total_mem: usize,
        // cuBLAS handle (initialized lazily)
        cublas: Mutex<Option<CudaBlas>>,
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
            })
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
            // cudarc doesn't expose free memory directly; use total as conservative estimate
            self.total_mem
        }

        fn upload_vectors(&self, vectors: &[f32], dim: usize) -> Result<DeviceBuffer> {
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

        fn build_ann_index(
            &self,
            device_vectors: &DeviceBuffer,
            dim: usize,
            config: &AnnBuildConfig,
        ) -> Result<Box<dyn GpuAnnIndex>> {
            // Use GPU-accelerated HNSW construction
            let wrapper = device_vectors
                .inner
                .downcast_ref::<CudaBufferWrapper>()
                .ok_or_else(|| GpuError::InvalidArgument("CUDA buffer mismatch".into()))?;

            let n = device_vectors.n;
            log::info!("GPU HNSW: building index for {} vectors of dim {}", n, dim);

            let index = gpu_hnsw_build::build_hnsw_on_gpu(self, &wrapper.slice, n, dim, config)?;

            Ok(Box::new(index))
        }

        fn ann_search(
            &self,
            _index: &dyn GpuAnnIndex,
            _query: &[f32],
            _top_k: usize,
        ) -> Result<Vec<(usize, f32)>> {
            // GPU-native HNSW search is intentionally not implemented. The
            // previous `search_hnsw_on_gpu` ran on the CPU (over the CPU-
            // resident `CudaAnnIndex.vectors`) with a greedy hill-climbing
            // algorithm that has poor recall, so it was neither GPU-
            // accelerated nor correct. Search is delegated to the usearch
            // fallback index persisted alongside the GPU-built index; the GPU
            // accelerates search via the batched `gemm` rerank path
            // (`batch_cosine_similarity_matrix`) instead, which is the one
            // workload where the GPU actually beats CPU.
            Err(GpuError::BackendNotCompiled(
                "GPU HNSW search is not implemented — use the usearch fallback \
                 or the batched gemm rerank path"
                    .into(),
            ))
        }

        fn quantized_scan(
            &self,
            _quantized: &DeviceBuffer,
            _query_lut: &DeviceBuffer,
            _n: usize,
            _dim: usize,
            _bits_per_dim: u8,
        ) -> Result<Vec<f32>> {
            // TODO: Implement CUDA kernels for quantized scoring
            Err(GpuError::BackendNotCompiled(
                "CUDA quantized scan not yet implemented — falling back to CPU".into(),
            ))
        }
    }

    /// Wrapper to make CudaSlice Send + Sync for Arc storage.
    struct CudaBufferWrapper {
        slice: CudaSlice<f32>,
    }

    unsafe impl Send for CudaBufferWrapper {}
    unsafe impl Sync for CudaBufferWrapper {}

    /// GPU-native approximate nearest neighbor index using HNSW.
    pub struct CudaAnnIndex {
        pub n: usize,
        pub dim: usize,
        pub memory_bytes: usize,
        /// Hierarchical layers: layer[i] contains edges for nodes at level i
        /// Level 0 is the base layer (most edges), higher levels have fewer nodes
        pub layers: Vec<Vec<Vec<usize>>>, // layers[level][node] = list of neighbor indices
        /// All vectors stored on host for search (GPU memory is limited)
        pub vectors: Vec<f32>, // flat n×dim
    }

    impl GpuAnnIndex for CudaAnnIndex {
        fn len(&self) -> usize {
            self.n
        }
        fn dim(&self) -> usize {
            self.dim
        }
        fn memory_bytes(&self) -> usize {
            self.memory_bytes
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    /// GPU HNSW construction using batched distance computation and parallel edge selection.
    mod gpu_hnsw_build {
        use super::*;
        use cudarc::driver::CudaSlice;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        /// Build an HNSW index on GPU.
        ///
        /// Algorithm: Simplified GPU-friendly HNSW
        /// 1. Compute all pairwise distances in batches on GPU (for small N) or
        ///    use random projection for candidate generation (for large N)
        /// 2. For each node, select top-M neighbors using GPU-accelerated search
        /// 3. Build hierarchical layers by probabilistic assignment
        pub fn build_hnsw_on_gpu(
            backend: &CudaBackend,
            device_vectors: &CudaSlice<f32>,
            n: usize,
            dim: usize,
            config: &AnnBuildConfig,
        ) -> Result<CudaAnnIndex> {
            let start = std::time::Instant::now();
            let max_edges = config.max_edges;
            let ef_construction = config.ef_construction;

            // For small collections, use brute-force all-pairs on GPU
            // For large collections, use batched approach
            let vectors = backend
                .stream
                .clone_dtoh(device_vectors)
                .map_err(|e| GpuError::KernelError(format!("Failed to download vectors: {e}")))?;

            // Build base layer (level 0) using GPU-accelerated neighbor selection
            // Brute-force all-pairs on GPU is fast and exact for collections up to ~20K
            // vectors. Beyond that, we fall back to usearch (which has a proven HNSW
            // implementation) rather than using a buggy GPU approximation.
            let base_layer = if n <= 20000 {
                // Small/medium: brute force all-pairs on GPU — fast and correct
                build_base_layer_brute_force(backend, &vectors, n, dim, max_edges)?
            } else {
                // Large: usearch fallback is more reliable than GPU approximations
                return Err(GpuError::InvalidArgument(format!(
                    "GPU HNSW build for {} vectors exceeds 20K brute-force threshold; \
                             use usearch fallback instead",
                    n
                )));
            };

            // Build upper layers by probabilistic decay
            let mut layers = vec![base_layer];
            let mut current_level_nodes: Vec<usize> = (0..n).collect();
            let mut rng = fastrand::Rng::new();

            while current_level_nodes.len() > 1 {
                // Each node has probability 1/2 of being promoted to next level
                let next_level: Vec<usize> = current_level_nodes
                    .iter()
                    .copied()
                    .filter(|_| rng.bool())
                    .collect();

                if next_level.len() <= 1 {
                    break;
                }

                // Build edges for next level using brute force on subset
                let level_vectors: Vec<f32> = next_level
                    .iter()
                    .flat_map(|&idx| vectors[idx * dim..(idx + 1) * dim].iter().copied())
                    .collect();

                let level_layer = build_level_brute_force(
                    backend,
                    &level_vectors,
                    &next_level,
                    dim,
                    max_edges.max(8) / 2, // Fewer edges at higher levels
                )?;

                layers.push(level_layer);
                current_level_nodes = next_level;
            }

            let elapsed = start.elapsed();
            log::info!(
                "GPU HNSW: built {} levels for {} vectors in {:.2}s",
                layers.len(),
                n,
                elapsed.as_secs_f64()
            );

            let memory_bytes = layers
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|v| v.capacity() * std::mem::size_of::<usize>())
                        .sum::<usize>()
                })
                .sum::<usize>()
                + vectors.len() * std::mem::size_of::<f32>();

            Ok(CudaAnnIndex {
                n,
                dim,
                memory_bytes,
                layers,
                vectors,
            })
        }

        /// Build base layer using brute-force all-pairs distance computation on GPU.
        fn build_base_layer_brute_force(
            backend: &CudaBackend,
            vectors: &[f32],
            n: usize,
            dim: usize,
            max_edges: usize,
        ) -> Result<Vec<Vec<usize>>> {
            let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

            // Process in batches to avoid excessive GPU memory usage
            const BATCH_SIZE: usize = 1024;
            for batch_start in (0..n).step_by(BATCH_SIZE) {
                let batch_end = (batch_start + BATCH_SIZE).min(n);
                let batch_n = batch_end - batch_start;

                // Upload batch vectors to GPU
                let batch_vectors: Vec<f32> = vectors[batch_start * dim..batch_end * dim].to_vec();
                let device_buf = backend.upload_vectors(&batch_vectors, dim)?;

                // For each node in the collection, compute distance to batch
                for i in 0..n {
                    let query = &vectors[i * dim..(i + 1) * dim];
                    let scores = backend.batch_cosine_similarity(query, &device_buf)?;

                    // Collect top max_edges neighbors from this batch
                    let mut heap: BinaryHeap<Reverse<(u32, usize)>> = BinaryHeap::new();
                    for (j, score) in scores.iter().enumerate().take(batch_n) {
                        let global_j = batch_start + j;
                        if global_j == i {
                            continue; // Skip self
                        }
                        let dist_bits = (1.0 - score).to_bits();
                        if heap.len() < max_edges {
                            heap.push(Reverse((dist_bits, global_j)));
                        } else if let Some(Reverse((min_bits, _))) = heap.peek() {
                            if dist_bits < *min_bits {
                                heap.pop();
                                heap.push(Reverse((dist_bits, global_j)));
                            }
                        }
                    }

                    // Merge into existing neighbors
                    for Reverse((_, idx)) in heap {
                        if !neighbors[i].contains(&idx) {
                            neighbors[i].push(idx);
                        }
                    }
                }
            }

            // Trim to max_edges and make symmetric
            for i in 0..n {
                neighbors[i].sort_by_key(|&j| {
                    let j_vec = &vectors[j * dim..(j + 1) * dim];
                    let i_vec = &vectors[i * dim..(i + 1) * dim];
                    let score = turbomemory_core::cosine_similarity(i_vec, j_vec);
                    Reverse((1.0 - score).to_bits())
                });
                neighbors[i].truncate(max_edges);

                // Make edges symmetric
                let edges: Vec<usize> = neighbors[i].clone();
                for &j in &edges {
                    if !neighbors[j].contains(&i) {
                        neighbors[j].push(i);
                    }
                }
            }

            // Final trim after symmetry
            for i in 0..n {
                neighbors[i].sort_by_key(|&j| {
                    let j_vec = &vectors[j * dim..(j + 1) * dim];
                    let i_vec = &vectors[i * dim..(i + 1) * dim];
                    let score = turbomemory_core::cosine_similarity(i_vec, j_vec);
                    Reverse((1.0 - score).to_bits())
                });
                neighbors[i].truncate(max_edges);
            }

            Ok(neighbors)
        }

        /// Build an upper level using brute force on a subset of nodes.
        fn build_level_brute_force(
            backend: &CudaBackend,
            level_vectors: &[f32],
            node_map: &[usize],
            dim: usize,
            max_edges: usize,
        ) -> Result<Vec<Vec<usize>>> {
            let n = node_map.len();
            if n <= 1 {
                return Ok(vec![Vec::new(); n]);
            }

            let device_buf = backend.upload_vectors(level_vectors, dim)?;
            let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];

            for i in 0..n {
                let query = &level_vectors[i * dim..(i + 1) * dim];
                let scores = backend.batch_cosine_similarity(query, &device_buf)?;

                let mut top: Vec<(usize, f32)> = scores
                    .iter()
                    .enumerate()
                    .take(n)
                    .map(|(j, &s)| (j, s))
                    .filter(|&(j, _)| j != i)
                    .collect();

                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                top.truncate(max_edges);

                for (j, _) in top {
                    if !neighbors[i].contains(&j) {
                        neighbors[i].push(j);
                    }
                    // Make symmetric
                    if !neighbors[j].contains(&i) {
                        neighbors[j].push(i);
                    }
                }
            }

            Ok(neighbors)
        }
    }
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

    /// Stub GPU ANN index used when the `cuda` feature is disabled.
    /// Cannot be constructed; exists only so downstream crates can name the
    /// type unconditionally (e.g. `Option<CudaAnnIndex>`) without a cfg gate.
    pub struct CudaAnnIndex;
}

// Re-export unconditionally so downstream code can name both types
// regardless of feature flags. Without `cuda` they are non-constructible
// stubs; with `cuda` they are the real implementations.
pub use cuda::{CudaAnnIndex, CudaBackend};
