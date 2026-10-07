//! Exact search over a store's vectors kept resident on the GPU.
//!
//! The full-precision vectors of a store are mirrored once into device
//! memory, row `i` holding the vector at offset `i`, and kept there. A search
//! is then one small upload (the query), one matrix-vector product over every
//! row, and one download of the scores; a batch of queries is one
//! matrix-matrix product. Nothing is re-uploaded per query, which is what
//! made the earlier per-query GPU rerank slower than the CPU.
//!
//! The mirror is append-only and brought up to date lazily, at search time:
//! offsets are allocated in order and a vector is never rewritten in place,
//! so the rows already on the device stay valid and only the tail written
//! since the last search has to be uploaded. Deleted and filtered records are
//! masked on the host when the top results are picked.
//!
//! The result is exact (the same cosines an exact CPU scan returns), so when
//! the mirror is active it replaces the approximate tiered search for that
//! query. Any device error, or a store that outgrows the memory budget,
//! switches the mirror off for the life of the engine and searches fall back
//! to the CPU path.
//!
//! One product at a time runs on the device. Searches that arrive while it is
//! busy queue up, and the next thread to get the device answers all of them
//! with a single matrix-matrix product, which reads the vectors once instead
//! of once per query. A lone search pays nothing for this; concurrent
//! searches share the cost of the pass over device memory.

use crate::vector_store::VectorStore;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use turbomemory_gpu::{GpuBackend, GpuError, ResidentMatrix};

/// Bytes uploaded per device copy. The vector store's read view is held for
/// one chunk at a time so a large first upload does not stall writers.
const UPLOAD_CHUNK_BYTES: usize = 32 << 20;
/// Host memory one batch's score matrix may take; larger batches are split.
const BATCH_SCORE_BYTES: usize = 64 << 20;
/// Smallest mirror allocated, in rows.
const MIN_CAPACITY_ROWS: usize = 8_192;
/// Most queued searches answered by one product: bounds how long the first
/// of them waits for the rest.
const MAX_GROUP: usize = 64;

/// What the mirror currently holds, for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSearchStats {
    /// Backend name (`"CUDA"`, or `"CPU Fallback"` when emulated in tests).
    pub backend: String,
    /// False once the mirror has been switched off.
    pub active: bool,
    /// Vectors resident on the device.
    pub rows: usize,
    /// Rows the current allocation has room for.
    pub capacity_rows: usize,
    /// Device memory the mirror may use.
    pub budget_bytes: usize,
    /// Device memory the mirror holds.
    pub memory_bytes: usize,
    /// Queries answered from the mirror.
    pub queries: u64,
    /// Device products that answered them (fewer than `queries` when
    /// concurrent searches or batches shared a product).
    pub device_calls: u64,
}

/// A single-query search waiting for the device.
struct Request {
    query: Vec<f32>,
    reply: Arc<Mutex<Reply>>,
}

#[derive(Default)]
struct Reply {
    served: bool,
    /// `None` once served means the device failed: use the CPU path.
    scores: Option<Vec<f32>>,
}

pub(crate) struct GpuExactIndex {
    backend: Arc<dyn GpuBackend>,
    dim: usize,
    budget_bytes: usize,
    /// The mirror. Holding this lock is holding the device.
    matrix: Mutex<Option<ResidentMatrix>>,
    /// Single-query searches not yet answered. Never locked while waiting
    /// for `matrix`.
    waiting: Mutex<Vec<Request>>,
    disabled: AtomicBool,
    rows: AtomicUsize,
    capacity_rows: AtomicUsize,
    queries: AtomicU64,
    device_calls: AtomicU64,
}

impl GpuExactIndex {
    /// `budget_mb == 0` means "half of the device memory that is free now".
    pub(crate) fn new(backend: Arc<dyn GpuBackend>, dim: usize, budget_mb: usize) -> Self {
        let budget_bytes = if budget_mb > 0 {
            budget_mb.saturating_mul(1 << 20)
        } else if turbomemory_gpu::is_gpu_accelerated(&backend) {
            backend.available_memory() / 2
        } else {
            // Host emulation (tests): no device to run out of.
            usize::MAX
        };
        Self {
            backend,
            dim,
            budget_bytes,
            matrix: Mutex::new(None),
            waiting: Mutex::new(Vec::new()),
            disabled: AtomicBool::new(false),
            rows: AtomicUsize::new(0),
            capacity_rows: AtomicUsize::new(0),
            queries: AtomicU64::new(0),
            device_calls: AtomicU64::new(0),
        }
    }

    pub(crate) fn stats(&self) -> GpuSearchStats {
        let capacity_rows = self.capacity_rows.load(Ordering::Relaxed);
        GpuSearchStats {
            backend: self.backend.name().to_string(),
            active: !self.disabled.load(Ordering::Relaxed),
            rows: self.rows.load(Ordering::Relaxed),
            capacity_rows,
            budget_bytes: self.budget_bytes,
            memory_bytes: capacity_rows * self.dim * std::mem::size_of::<f32>(),
            queries: self.queries.load(Ordering::Relaxed),
            device_calls: self.device_calls.load(Ordering::Relaxed),
        }
    }

    /// Scores of `query` (unit length) against every vector at offsets
    /// `0..scores.len()`. `None` when the mirror is off or just failed; the
    /// caller then uses the CPU path.
    pub(crate) fn scores(&self, vectors: &VectorStore, query: &[f32]) -> Option<Vec<f32>> {
        if self.disabled.load(Ordering::Relaxed) {
            return None;
        }
        let reply = Arc::new(Mutex::new(Reply::default()));
        self.waiting.lock().push(Request {
            query: query.to_vec(),
            reply: reply.clone(),
        });
        // Whoever holds the device answers every search queued behind it, so
        // by the time this thread gets the device its own search has usually
        // been answered already.
        let mut slot = self.matrix.lock();
        loop {
            {
                let mut mine = reply.lock();
                if mine.served {
                    return mine.scores.take();
                }
            }
            // Switched off while this search was getting in line: it must not
            // bring the mirror back.
            if self.disabled.load(Ordering::Relaxed) {
                self.waiting
                    .lock()
                    .retain(|request| !Arc::ptr_eq(&request.reply, &reply));
                return None;
            }
            let group: Vec<Request> = {
                let mut waiting = self.waiting.lock();
                let take = waiting.len().min(MAX_GROUP);
                waiting.drain(..take).collect()
            };
            if group.is_empty() {
                return None; // unreachable: an unanswered search is queued
            }
            let result = self.answer(&mut slot, vectors, &group);
            let mut rows = self.settle(&mut slot, result).map(Vec::into_iter);
            for request in group {
                let mut theirs = request.reply.lock();
                theirs.served = true;
                theirs.scores = rows.as_mut().and_then(Iterator::next);
            }
        }
    }

    /// One product for a group of queued searches: a score row for each.
    fn answer(
        &self,
        slot: &mut Option<ResidentMatrix>,
        vectors: &VectorStore,
        group: &[Request],
    ) -> Result<Vec<Vec<f32>>, GpuError> {
        let matrix = self.sync(slot, vectors)?;
        let rows = matrix.rows();
        if rows == 0 {
            return Ok(vec![Vec::new(); group.len()]);
        }
        self.queries
            .fetch_add(group.len() as u64, Ordering::Relaxed);
        self.device_calls.fetch_add(1, Ordering::Relaxed);
        if let [only] = group {
            return Ok(vec![self.backend.resident_scores(matrix, &only.query)?]);
        }
        let mut flat = Vec::with_capacity(group.len() * self.dim);
        for request in group {
            flat.extend_from_slice(&request.query);
        }
        let scores = self
            .backend
            .resident_scores_batch(matrix, &flat, group.len())?;
        if scores.len() != rows * group.len() {
            return Err(GpuError::KernelError(format!(
                "batch returned {} scores for {} queries x {rows} rows",
                scores.len(),
                group.len()
            )));
        }
        Ok(scores.chunks_exact(rows).map(<[f32]>::to_vec).collect())
    }

    /// Run `finish` on the score row of each query (unit length), in order.
    /// One device matrix product per group of queries. `None` when the
    /// mirror is off or failed part-way; nothing is returned in that case so
    /// the caller can redo the whole batch on the CPU.
    pub(crate) fn scores_batch<R>(
        &self,
        vectors: &VectorStore,
        queries: &[Vec<f32>],
        mut finish: impl FnMut(&[f32]) -> R,
    ) -> Option<Vec<R>> {
        if self.disabled.load(Ordering::Relaxed) {
            return None;
        }
        let mut slot = self.matrix.lock();
        if self.disabled.load(Ordering::Relaxed) {
            return None; // switched off while waiting for the device
        }
        let mut out = Vec::with_capacity(queries.len());
        let result = (|| {
            let matrix = self.sync(&mut slot, vectors)?;
            let rows = matrix.rows();
            if rows == 0 {
                out.extend(queries.iter().map(|_| finish(&[])));
                return Ok(());
            }
            let per_group = (BATCH_SCORE_BYTES / (rows * std::mem::size_of::<f32>())).max(1);
            for group in queries.chunks(per_group) {
                self.queries
                    .fetch_add(group.len() as u64, Ordering::Relaxed);
                self.device_calls.fetch_add(1, Ordering::Relaxed);
                let mut flat = Vec::with_capacity(group.len() * self.dim);
                for query in group {
                    flat.extend_from_slice(query);
                }
                let scores = self
                    .backend
                    .resident_scores_batch(matrix, &flat, group.len())?;
                if scores.len() != rows * group.len() {
                    return Err(GpuError::KernelError(format!(
                        "batch returned {} scores for {} queries x {rows} rows",
                        scores.len(),
                        group.len()
                    )));
                }
                out.extend(scores.chunks_exact(rows).map(&mut finish));
            }
            Ok(())
        })();
        self.settle(&mut slot, result).map(|()| out)
    }

    /// Turn a device result into an `Option`, switching the mirror off (and
    /// freeing its memory) on any error.
    fn settle<T>(
        &self,
        slot: &mut Option<ResidentMatrix>,
        result: Result<T, GpuError>,
    ) -> Option<T> {
        match result {
            Ok(value) => Some(value),
            Err(e) => {
                log::warn!("GPU exact search switched off ({e}); searches continue on the CPU");
                if let Some(matrix) = slot.take() {
                    self.backend.resident_release(matrix);
                }
                self.rows.store(0, Ordering::Relaxed);
                self.capacity_rows.store(0, Ordering::Relaxed);
                self.disabled.store(true, Ordering::Relaxed);
                // Searches still queued are sent to the CPU path as well.
                for request in self.waiting.lock().drain(..) {
                    request.reply.lock().served = true;
                }
                None
            }
        }
    }

    /// Make the mirror hold every vector the store has, allocating or
    /// growing it as needed.
    fn sync<'a>(
        &self,
        slot: &'a mut Option<ResidentMatrix>,
        vectors: &VectorStore,
    ) -> Result<&'a mut ResidentMatrix, GpuError> {
        let count = vectors.count();
        let row_bytes = self.dim * std::mem::size_of::<f32>();
        let needs_allocation = slot.as_ref().is_none_or(|m| count > m.capacity());
        if needs_allocation {
            // Room to grow by half before the next reallocation, unless the
            // budget only allows an exact fit.
            let mut capacity = (count + count / 2).max(MIN_CAPACITY_ROWS);
            if capacity.saturating_mul(row_bytes) > self.budget_bytes {
                capacity = count.max(1);
            }
            let need = capacity.saturating_mul(row_bytes);
            if need > self.budget_bytes {
                return Err(GpuError::OutOfMemory {
                    need_mb: need >> 20,
                    have_mb: self.budget_bytes >> 20,
                });
            }
            // Free the old allocation first: device memory is the scarce
            // resource, and every row is re-read from the store anyway.
            *slot = None;
            self.rows.store(0, Ordering::Relaxed);
            *slot = Some(self.backend.resident_create(self.dim, capacity)?);
            self.capacity_rows.store(capacity, Ordering::Relaxed);
        }
        let matrix = slot.as_mut().expect("allocated above");
        let rows_per_chunk = (UPLOAD_CHUNK_BYTES / row_bytes).max(1);
        while matrix.rows() < count {
            let start = matrix.rows();
            let end = (start + rows_per_chunk).min(count);
            let view = vectors.read_view();
            let Some(rows) = view.rows(start, end) else {
                return Err(GpuError::InvalidArgument(format!(
                    "vector store no longer holds rows {start}..{end}"
                )));
            };
            self.backend.resident_append(matrix, rows)?;
        }
        self.rows.store(matrix.rows(), Ordering::Relaxed);
        Ok(matrix)
    }
}

/// Closing the engine gives the mirror's device memory back at once.
impl Drop for GpuExactIndex {
    fn drop(&mut self) {
        if let Some(matrix) = self.matrix.get_mut().take() {
            self.backend.resident_release(matrix);
        }
    }
}
