//! Background optimizer for sealing, compaction, and periodic flushing.
//!
//! The optimizer owns a background thread that:
//!   1. Performs fast Hot-seal swaps when the Hot segment is full.
//!   2. Builds HNSW / quantized segments from `sealing_plain` segments off the
//!      caller thread, then atomically installs them.
//!   3. Runs periodic consolidation + flush.

use crate::config::Tier;
use crate::engine::StorageEngine;
use crate::record::{PointOffset, Record};
use crate::segments::hot::HotSegment;
use crate::segments::{SealedHotSegment, VectorSegment, WarmSegment};
use parking_lot::RwLock;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// Background optimizer that drives sealing, compaction, and flush.
pub struct BackgroundOptimizer {
    tx: Sender<OptimizerMsg>,
    handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl BackgroundOptimizer {
    /// Spawn the background optimizer.
    ///
    /// `weak_engine` is used so the worker does not keep the engine alive by
    /// itself. The thread exits when it can no longer upgrade the weak reference
    /// or when `Shutdown` is received.
    pub fn new(weak_engine: Weak<StorageEngine>, interval: Duration) -> Self {
        let (tx, rx) = channel::<OptimizerMsg>();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let handle = Builder::new()
            .name("turbo-optimizer".into())
            .spawn(move || {
                let mut last_run = Instant::now();
                while running_clone.load(Ordering::Relaxed) {
                    let timeout = Duration::from_millis(100);
                    match rx.recv_timeout(timeout) {
                        Ok(OptimizerMsg::Shutdown) => break,
                        Ok(OptimizerMsg::Seal) => {
                            if let Some(engine) = weak_engine.upgrade() {
                                Self::run_seal(&engine);
                                Self::process_pending_seals(&engine);
                            }
                        }
                        Ok(OptimizerMsg::Consolidate) => {
                            if let Some(engine) = weak_engine.upgrade() {
                                Self::run_consolidation(&engine);
                            }
                        }
                        Err(_) => {
                            if let Some(engine) = weak_engine.upgrade() {
                                if last_run.elapsed() >= interval {
                                    Self::run_consolidation(&engine);
                                    last_run = Instant::now();
                                }
                                Self::process_pending_seals(&engine);
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn turbo-optimizer");
        Self {
            tx,
            handle: Some(handle),
            running,
        }
    }

    fn run_seal(engine: &StorageEngine) {
        let mut segments = engine.segments.write();
        let _ = segments.seal_hot(&engine.vectors);
    }

    fn process_pending_seals(engine: &StorageEngine) {
        while let Ok(true) = Self::process_one_seal(engine) {}
    }

    /// Pop one `sealing_plain` segment, build its persisted replacement, and
    /// install it. Returns `Ok(true)` if a segment was processed, `Ok(false)` if
    /// the queue is empty.
    pub(crate) fn process_one_seal(engine: &StorageEngine) -> crate::Result<bool> {
        // Grab the next plain segment and the metadata needed to build it.
        let job = {
            let mut segments = engine.segments.write();
            let plain = match segments.pop_sealing_plain() {
                Some(p) => p,
                None => return Ok(false),
            };
            let offsets: Vec<PointOffset> = plain.read().offsets().to_vec();
            if offsets.is_empty() {
                return Ok(true);
            }
            let use_hnsw = offsets.len() >= engine.config().tier.hnsw_threshold;
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
        let mut segments = engine.segments.write();
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

    fn run_consolidation(engine: &StorageEngine) {
        let _ = engine.trigger_consolidation();
        let _ = engine.flush();
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
                (*offset, Record {
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
                })
            })
            .collect();
        let warm = WarmSegment::from_records(&job.path, &records, engine.config().tier.warm_bits)?;
        Ok(BuiltSegment::Warm(warm))
    }
}
