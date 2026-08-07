//! Background optimizer for sealing, compaction, and periodic flushing.
//!
//! The optimizer owns a background thread that:
//!   1. Performs fast Hot-seal swaps when the Hot segment is full.
//!   2. Builds HNSW / quantized segments from `sealing_plain` segments off the
//!      caller thread, then atomically installs them.
//!   3. Runs periodic consolidation + flush.
//!
//! CPU-heavy HNSW builds are gated by a `ResourceBudget` so the optimizer does
//! not allocate unbounded memory or run more than one expensive build at a time.

use crate::config::{OptimizerBudget, Tier};
use crate::engine::StorageEngine;
use crate::record::{PointOffset, Record};
use crate::segments::hot::HotSegment;
use crate::segments::{SealedHotSegment, VectorSegment, WarmSegment};
use parking_lot::{Mutex, RwLock};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Weak};
use std::thread::{Builder, JoinHandle};
use std::time::{Duration, Instant};

pub enum OptimizerMsg {
    /// The Hot segment is full; perform a fast seal swap.
    Seal,
    /// Run one consolidation + flush cycle immediately.
    Consolidate,
    /// Stop the worker thread.
    Shutdown,
}

/// Lightweight guard returned when the optimizer acquires a build slot.
/// The slot is released when the guard is dropped.
pub(crate) struct BuildGuard {
    budget: Arc<ResourceBudget>,
}

impl Drop for BuildGuard {
    fn drop(&mut self) {
        self.budget.inflight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Resource budget for CPU/memory-heavy optimizer work.
pub struct ResourceBudget {
    inflight: AtomicUsize,
    max_concurrent: usize,
    max_memory_bytes: Option<usize>,
}

impl ResourceBudget {
    pub fn new(budget: OptimizerBudget) -> Self {
        Self {
            inflight: AtomicUsize::new(0),
            max_concurrent: budget.max_concurrent_builds.max(1),
            max_memory_bytes: budget.max_build_memory_bytes,
        }
    }

    /// Try to acquire a build slot for an HNSW segment of `n` vectors.
    ///
    /// Returns `Some(BuildGuard)` if the concurrency and memory budgets allow
    /// the build, otherwise `None`. The caller should keep the guard alive for
    /// the duration of the build.
    pub(crate) fn try_acquire(
        budget: &Arc<Self>,
        n: usize,
        dim: usize,
        max_edges: usize,
    ) -> Option<BuildGuard> {
        let current = budget.inflight.load(Ordering::Relaxed);
        if current >= budget.max_concurrent {
            return None;
        }

        if let Some(max_mem) = budget.max_memory_bytes {
            // Realistic HNSW footprint: vector data + graph edges + 30% overhead.
            // The previous estimate multiplied vector data by max_edges, which
            // massively over-counted and rejected large but valid merges.
            let vector_bytes = n.saturating_mul(dim).saturating_mul(4);
            let graph_bytes = n.saturating_mul(max_edges).saturating_mul(8);
            let est = vector_bytes
                .saturating_add(graph_bytes)
                .saturating_mul(13)
                .div_euclid(10);
            if est > max_mem {
                return None;
            }
        }

        if budget
            .inflight
            .compare_exchange(current, current + 1, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            Some(BuildGuard {
                budget: Arc::clone(budget),
            })
        } else {
            None
        }
    }
}

/// Background optimizer that drives sealing, compaction, and flush.
pub struct BackgroundOptimizer {
    tx: Sender<OptimizerMsg>,
    handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
    /// Set while the foreground `drain` is running so the background loop
    /// yields and does not race for the same segments.
    draining: Arc<AtomicBool>,
    budget: Arc<ResourceBudget>,
    /// Directories of merged-away segments that are waiting to be deleted. They
    /// may still be mmap'd by in-flight searches, so deletion is retried.
    pending_deletion: Arc<Mutex<Vec<PathBuf>>>,
}

impl BackgroundOptimizer {
    /// Spawn the background optimizer.
    ///
    /// `weak_engine` is used so the worker does not keep the engine alive by
    /// itself. The thread exits when `Shutdown` is received or `running` is
    /// cleared (both happen in `stop`, called from `Drop`). If the engine is
    /// gone the loop simply idles until then.
    pub fn new(
        weak_engine: Weak<StorageEngine>,
        interval: Option<Duration>,
        budget: Arc<ResourceBudget>,
    ) -> Self {
        let (tx, rx) = channel::<OptimizerMsg>();
        let running = Arc::new(AtomicBool::new(interval.is_some()));
        let running_clone = running.clone();
        let draining = Arc::new(AtomicBool::new(false));
        let pending_deletion: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
        let handle = interval.map(|interval| {
            let draining_clone = draining.clone();
            let budget_clone = Arc::clone(&budget);
            let pending_deletion_clone = Arc::clone(&pending_deletion);
            Builder::new()
                .name("turbo-optimizer".into())
                .spawn(move || {
                    let mut last_run = Instant::now();
                    while running_clone.load(Ordering::Relaxed) {
                        // Yield to the foreground `drain` so it can complete merges
                        // without racing the background loop for the same segments.
                        if draining_clone.load(Ordering::Relaxed) {
                            std::thread::sleep(Duration::from_millis(10));
                            continue;
                        }
                        Self::try_cleanup_pending(&pending_deletion_clone);
                        let timeout = Duration::from_millis(100);
                        match rx.recv_timeout(timeout) {
                            Ok(OptimizerMsg::Shutdown) => break,
                            Ok(OptimizerMsg::Seal) => {
                                if let Some(engine) = weak_engine.upgrade() {
                                    Self::run_seal(&engine);
                                    Self::process_pending_seals(&engine, &budget_clone);
                                    Self::process_pending_merges(
                                        &engine,
                                        &budget_clone,
                                        &pending_deletion_clone,
                                        &draining_clone,
                                    );
                                }
                            }
                            Ok(OptimizerMsg::Consolidate) => {
                                if let Some(engine) = weak_engine.upgrade() {
                                    Self::run_consolidation(
                                        &engine,
                                        &budget_clone,
                                        &pending_deletion_clone,
                                        &draining_clone,
                                    );
                                }
                            }
                            Err(_) => {
                                if let Some(engine) = weak_engine.upgrade() {
                                    if last_run.elapsed() >= interval {
                                        Self::run_consolidation(
                                            &engine,
                                            &budget_clone,
                                            &pending_deletion_clone,
                                            &draining_clone,
                                        );
                                        last_run = Instant::now();
                                    }
                                    Self::process_pending_seals(&engine, &budget_clone);
                                    Self::process_pending_merges(
                                        &engine,
                                        &budget_clone,
                                        &pending_deletion_clone,
                                        &draining_clone,
                                    );
                                }
                            }
                        }
                    }
                    Self::try_cleanup_pending(&pending_deletion_clone);
                })
                .expect("failed to spawn turbo-optimizer")
        });
        Self {
            tx,
            handle,
            running,
            draining,
            budget,
            pending_deletion,
        }
    }

    fn run_seal(engine: &StorageEngine) {
        let segments = engine.segments.read();
        let _ = segments.seal_hot(&engine.vectors);
    }

    fn process_pending_seals(engine: &StorageEngine, budget: &Arc<ResourceBudget>) {
        while let Ok(true) = Self::process_one_seal_with_budget(engine, budget) {}
    }

    /// Pop one `sealing_plain` segment, build its persisted replacement, and
    /// install it using the optimizer's own budget.
    pub(crate) fn process_one_seal(&self, engine: &StorageEngine) -> crate::Result<bool> {
        Self::process_one_seal_with_budget(engine, &self.budget)
    }

    /// Pop one `sealing_plain` segment, build its persisted replacement, and
    /// install it. Returns `Ok(true)` if a segment was processed, `Ok(false)` if
    /// the queue is empty.
    fn process_one_seal_with_budget(
        engine: &StorageEngine,
        budget: &Arc<ResourceBudget>,
    ) -> crate::Result<bool> {
        // Grab the next plain segment and the metadata needed to build it.
        let job = {
            let segments = engine.segments.read();
            let plain = match segments.pop_sealing_plain() {
                Some(p) => p,
                None => return Ok(false),
            };
            let offsets: Vec<PointOffset> = plain.read().offsets().to_vec();
            if offsets.is_empty() {
                segments.remove_sealing_plain(&plain);
                segments.publish_snapshot();
                return Ok(true);
            }
            let threshold_met = {
                let cfg = engine.config();
                let count_met = offsets.len() >= cfg.tier.hnsw_threshold;
                let bytes = offsets
                    .len()
                    .saturating_mul(cfg.dimension)
                    .saturating_mul(4);
                let kb = bytes / 1024;
                let byte_met =
                    cfg.tier.full_scan_threshold_kb > 0 && kb >= cfg.tier.full_scan_threshold_kb;
                count_met || byte_met
            };
            let use_hnsw = threshold_met
                && ResourceBudget::try_acquire(
                    budget,
                    offsets.len(),
                    engine.config().dimension,
                    engine.config().max_edges,
                )
                .is_some();
            let path = if use_hnsw {
                segments.sealed_hot_path()
            } else {
                segments.segment_path(Tier::Warm)
            };
            Job {
                plain,
                offsets,
                use_hnsw,
                path,
            }
        };

        // Build the segment without holding the holder lock.
        let built = build_segment(engine, &job);

        // Install the new segment and drop the old plain one.
        let segments = engine.segments.read();
        segments.remove_sealing_plain(&job.plain);
        match built {
            Ok(BuiltSegment::Sealed(sealed)) => {
                segments.push_sealed_hot(sealed);
            }
            Ok(BuiltSegment::Warm(warm)) => {
                segments.push_warm(warm);
                let _ = segments.compact_warm(&engine.vectors);
            }
            Err(e) => {
                // On failure, put the plain segment back so records stay
                // searchable and we can retry later.
                eprintln!("turbo-optimizer: failed to build segment: {e}");
                segments.push_sealing_plain(job.plain);
                return Err(e);
            }
        }
        Ok(true)
    }

    fn run_consolidation(
        engine: &StorageEngine,
        budget: &Arc<ResourceBudget>,
        pending_deletion: &Arc<Mutex<Vec<PathBuf>>>,
        draining: &Arc<AtomicBool>,
    ) {
        if draining.load(Ordering::Acquire) {
            return;
        }
        let _ = engine.trigger_consolidation();
        // Drain any pending seal builds within the resource budget. The remaining
        // work will be picked up by the next optimizer tick.
        Self::process_pending_seals(engine, budget);
        Self::process_pending_merges(engine, budget, pending_deletion, draining);
        let _ = engine.flush();
    }

    fn process_pending_merges(
        engine: &StorageEngine,
        budget: &Arc<ResourceBudget>,
        pending_deletion: &Arc<Mutex<Vec<PathBuf>>>,
        draining: &Arc<AtomicBool>,
    ) {
        while !draining.load(Ordering::Acquire) {
            match Self::process_one_merge_with_budget(engine, budget, pending_deletion) {
                Ok(true) => {}
                Ok(false) | Err(_) => break,
            }
        }
    }

    /// Try to merge a group of sealed HNSW segments into a single larger segment.
    ///
    /// Returns `Ok(true)` if a merge was performed, `Ok(false)` if no merge was
    /// needed or the budget was exhausted.
    fn process_one_merge_with_budget(
        engine: &StorageEngine,
        budget: &Arc<ResourceBudget>,
        pending_deletion: &Arc<Mutex<Vec<PathBuf>>>,
    ) -> crate::Result<bool> {
        let config = engine.config();
        let threshold = config.tier.merge_threshold_segments();
        let max_records = config.tier.merge_max_records();

        // 1. Choose candidate segments under a read lock.
        let (candidates, old_paths): (Vec<Arc<RwLock<dyn VectorSegment>>>, Vec<PathBuf>) = {
            let segments = engine.segments.read();
            let cands = match segments.sealed_hot_merge_candidates(threshold, max_records) {
                Some(c) => c,
                None => return Ok(false),
            };
            let paths: Vec<PathBuf> = cands
                .iter()
                .filter_map(|seg| seg.read().segment_path().map(|p| p.to_path_buf()))
                .collect();
            (cands, paths)
        };

        // 2. Collect the union of offsets (deduplicated) and read full vectors.
        let mut offset_set = HashSet::new();
        let mut offsets: Vec<PointOffset> = Vec::new();
        for seg in &candidates {
            for &offset in seg.read().offsets() {
                if offset_set.insert(offset) {
                    offsets.push(offset);
                }
            }
        }
        if offsets.is_empty() {
            let segments = engine.segments.read();
            segments.remove_sealed_hot_segments(&candidates);
            segments.publish_snapshot();
            return Ok(true);
        }

        let vectors_for_build = {
            let segments = engine.segments.read();
            segments.read_vectors_for_offsets(&offsets, &engine.vectors)
        };
        if vectors_for_build.is_empty() {
            return Ok(false);
        }

        // 3. Acquire a build slot. If the budget rejects the merge, try again later.
        let _build_guard = match ResourceBudget::try_acquire(
            budget,
            vectors_for_build.len(),
            config.dimension,
            config.max_edges,
        ) {
            Some(guard) => guard,
            None => return Ok(false),
        };

        let borrowed: Vec<(PointOffset, &[f32])> = vectors_for_build
            .iter()
            .map(|(offset, v)| (*offset, v.as_slice()))
            .collect();

        // 4. Build the new segment outside the holder lock.
        let new_path = engine.segments.read().sealed_hot_path();
        let sealed = match SealedHotSegment::from_vectors(&new_path, config, &borrowed) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("turbo-optimizer: failed to build merged segment: {e}");
                return Err(e);
            }
        };

        // 5. Atomically install the merged segment and remove the old ones.
        let segments = engine.segments.read();
        let removed = segments.remove_sealed_hot_segments(&candidates);
        if removed == 0 {
            // The candidates disappeared; the new segment is orphaned. Keep it
            // anyway so the build work is not wasted, but schedule it for cleanup
            // on next flush.
            eprintln!("turbo-optimizer: merge candidates disappeared before install");
        }
        segments.push_sealed_hot(sealed);
        drop(segments);

        // 6. Schedule old segment directories for deletion.
        pending_deletion.lock().extend(old_paths);
        Ok(true)
    }

    fn try_cleanup_pending(pending: &Arc<Mutex<Vec<PathBuf>>>) {
        let mut guard = pending.lock();
        let mut still_pending = Vec::with_capacity(guard.len());
        for path in guard.drain(..) {
            if std::fs::remove_dir_all(&path).is_err() {
                still_pending.push(path);
            }
        }
        *guard = still_pending;
    }

    /// Request a fast Hot-seal swap.
    pub fn request_seal(&self) {
        let _ = self.tx.send(OptimizerMsg::Seal);
    }

    /// Request an immediate consolidation + flush cycle.
    pub fn request_consolidation(&self) {
        let _ = self.tx.send(OptimizerMsg::Consolidate);
    }

    /// Stop the optimizer thread and wait for it to finish.
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.tx.send(OptimizerMsg::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        Self::try_cleanup_pending(&self.pending_deletion);
    }

    /// Synchronously process every pending seal and merge until the work queue
    /// is empty. This is useful when the caller wants to search immediately
    /// after consolidation and does not want to wait for the background thread.
    ///
    /// While `drain` is running the background loop yields so it does not race
    /// for the same segments and cause spurious "candidates disappeared" errors.
    pub fn drain(&self, engine: &StorageEngine) {
        self.draining.store(true, Ordering::Release);
        // Wait for any in-progress background builds to finish so we don't race
        // for the same merge candidates mid-build.
        let deadline = Instant::now() + Duration::from_secs(30);
        while self.budget.inflight.load(Ordering::Acquire) > 0 {
            if Instant::now() >= deadline {
                break;
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        while self.process_one_seal(engine).unwrap_or(false) {}
        while Self::process_one_merge_with_budget(engine, &self.budget, &self.pending_deletion)
            .unwrap_or(false)
        {}
        Self::try_cleanup_pending(&self.pending_deletion);
        self.draining.store(false, Ordering::Release);
    }
}

impl Drop for BackgroundOptimizer {
    fn drop(&mut self) {
        self.stop();
    }
}

struct Job {
    plain: Arc<RwLock<HotSegment>>,
    offsets: Vec<PointOffset>,
    use_hnsw: bool,
    path: PathBuf,
}

enum BuiltSegment {
    Sealed(SealedHotSegment),
    Warm(WarmSegment),
}

fn build_segment(engine: &StorageEngine, job: &Job) -> crate::Result<BuiltSegment> {
    let vectors_for_build = {
        let segments = engine.segments.read();
        segments.read_vectors_for_offsets(&job.offsets, &engine.vectors)
    };

    if vectors_for_build.is_empty() {
        return Err(crate::StorageError::InvalidArgument(
            "no vectors found for segment build".into(),
        ));
    }

    let borrowed: Vec<(PointOffset, &[f32])> = vectors_for_build
        .iter()
        .map(|(offset, v)| (*offset, v.as_slice()))
        .collect();

    if job.use_hnsw {
        let sealed = SealedHotSegment::from_vectors(&job.path, engine.config(), &borrowed)?;
        Ok(BuiltSegment::Sealed(sealed))
    } else {
        let records: Vec<(PointOffset, Record)> = vectors_for_build
            .iter()
            .map(|(offset, v)| {
                (
                    *offset,
                    Record {
                        id: String::new(),
                        text: String::new(),
                        embedding: Arc::from(v.clone()),
                        importance: 0.0,
                        concepts: Vec::new(),
                        created_at: 0,
                        insert_seq: 0,
                        access_count: 0,
                        last_accessed: 0,
                        tier: Tier::Warm,
                        payload: None,
                        scope: None,
                        source_role: None,
                    },
                )
            })
            .collect();
        let warm =
            WarmSegment::from_records(&job.path, &records, engine.config().tier.warm_quantizer)?;
        Ok(BuiltSegment::Warm(warm))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resource_budget_allows_first_build() {
        let budget = Arc::new(ResourceBudget::new(OptimizerBudget::default()));
        assert!(
            ResourceBudget::try_acquire(&budget, 1000, 128, 16).is_some(),
            "default budget should allow a modest HNSW build"
        );
    }

    #[test]
    fn resource_budget_rejects_over_memory_build() {
        let budget = Arc::new(ResourceBudget::new(OptimizerBudget {
            max_concurrent_builds: 1,
            max_build_memory_bytes: Some(1024), // 1 KiB
        }));
        assert!(
            ResourceBudget::try_acquire(&budget, 10_000, 768, 16).is_none(),
            "a 10k×768 HNSW build should exceed a 1 KiB memory budget"
        );
    }

    #[test]
    fn resource_budget_rejects_concurrent_builds() {
        let budget = Arc::new(ResourceBudget::new(OptimizerBudget {
            max_concurrent_builds: 1,
            max_build_memory_bytes: None,
        }));
        let _guard = ResourceBudget::try_acquire(&budget, 100, 8, 8);
        assert!(
            ResourceBudget::try_acquire(&budget, 100, 8, 8).is_none(),
            "only one build should be allowed when max_concurrent_builds == 1"
        );
    }
}
