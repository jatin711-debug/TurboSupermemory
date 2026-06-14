//! Background update / optimize / flush worker.
//!
//! This is a simplified Qdrant-style pipeline:
//!   - A background thread periodically seals the Hot segment and compacts
//!     Warm segments into Cold segments.
//!   - After optimization it flushes durable metadata.

use crate::engine::StorageEngine;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

enum WorkerMsg {
    TriggerConsolidation,
    Shutdown,
}

/// Background worker that drives consolidation and flush.
pub struct UpdateHandler {
    tx: Sender<WorkerMsg>,
    handle: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl UpdateHandler {
    /// Start the worker.  `interval` controls the periodic consolidation cadence.
    pub fn new(engine: Arc<StorageEngine>, interval: Duration) -> Self {
        let (tx, rx) = channel::<WorkerMsg>();
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let handle = std::thread::Builder::new()
            .name("turbo-consolidation".into())
            .spawn(move || {
                let mut last_run = Instant::now();
                while running_clone.load(Ordering::Relaxed) {
                    let timeout = Duration::from_millis(100);
                    match rx.recv_timeout(timeout) {
                        Ok(WorkerMsg::Shutdown) => break,
                        Ok(WorkerMsg::TriggerConsolidation) => {
                            last_run = Self::run_cycle(&engine);
                        }
                        Err(_) => {
                            if last_run.elapsed() >= interval {
                                last_run = Self::run_cycle(&engine);
                            }
                        }
                    }
                }
            })
            .expect("failed to spawn consolidation worker");
        Self {
            tx,
            handle: Some(handle),
            running,
        }
    }

    fn run_cycle(engine: &Arc<StorageEngine>) -> Instant {
        let _ = engine.trigger_consolidation();
        let _ = engine.flush();
        Instant::now()
    }

    /// Request an immediate consolidation cycle.
    pub fn trigger(&self) {
        let _ = self.tx.send(WorkerMsg::TriggerConsolidation);
    }

    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.tx.send(WorkerMsg::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for UpdateHandler {
    fn drop(&mut self) {
        self.stop();
    }
}
