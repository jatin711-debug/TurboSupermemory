//! Single-threaded update worker for the storage engine.
//!
//! All index mutations (metadata cache, id/payload/text indexes, cognitive graph,
//! and Hot segment insertion) are serialized through this worker.  Callers push
//! batches of already-durable records into a channel and wait for an
//! acknowledgement so reads remain strongly consistent.
//!
//! The worker owns an `IndexApplier` (a bundle of the Arcs it needs) rather than
//! a strong reference to the `StorageEngine`, so dropping the engine blocks until
//! the worker thread has released every reference.

use crate::metadata_store::MetadataStore;
use crate::payload_index::PayloadIndex;
use crate::record::{PointOffset, Record};
use crate::segment_holder::SegmentHolder;
use crate::text_index::TextIndex;
use crate::vector_store::VectorStore;
use ahash::HashMap as AHashMap;
use crossbeam_channel::{bounded, Receiver, Sender};
use parking_lot::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{Builder, JoinHandle};
use turbomemory_graph::SpreadingActivation;

/// Applies index mutations.  Owned by the update worker so it does not keep the
/// `StorageEngine` alive.
pub(crate) struct IndexApplier {
    pub meta: Arc<MetadataStore>,
    pub vectors: Arc<VectorStore>,
    pub segments: Arc<RwLock<SegmentHolder>>,
    pub graph: Arc<RwLock<SpreadingActivation>>,
    pub id_index: Arc<RwLock<AHashMap<Arc<str>, PointOffset>>>,
    pub payload_index: Arc<RwLock<PayloadIndex>>,
    pub text_index: Arc<TextIndex>,
}

impl IndexApplier {
    /// Apply a batch of already-durable records to the in-memory indexes.
    ///
    /// This is the single serialized write path for metadata, id/payload/text
    /// indexes, the cognitive graph, and the Hot segment.
    pub fn apply_batch(&self, records: &[(PointOffset, Record)]) -> crate::Result<()> {
        // 1. Metadata cache.
        self.meta.put_batch(records)?;

        // 2. Id, payload, and text indexes.
        {
            let mut idx = self.id_index.write();
            let mut pidx = self.payload_index.write();
            let text_docs: Vec<(PointOffset, &str)> = records
                .iter()
                .map(|(offset, rec)| (*offset, rec.text.as_str()))
                .collect();
            for (offset, rec) in records {
                idx.insert(Arc::from(rec.id.as_str()), *offset);
                pidx.add(*offset, rec.payload.as_deref())?;
            }
            self.text_index.add_batch(&text_docs)?;
        }

        // 3. Cognitive graph.
        {
            let mut graph = self.graph.write();
            for (_, rec) in records {
                graph.add_memory(&rec.id, &rec.text, &rec.concepts);
            }
        }

        // 4. Hot segment insertion and optional seal.
        let mut needs_seal = false;
        {
            let segments = self.segments.read();
            for (offset, record) in records {
                if segments.insert(*offset, record, &self.vectors)? {
                    needs_seal = true;
                }
            }
            if needs_seal {
                segments.seal_hot(&self.vectors)?;
            }
        }
        Ok(())
    }
}

pub(crate) enum UpdateMsg {
    /// Apply a batch of records to the in-memory indexes.
    Batch {
        records: Vec<(PointOffset, Record)>,
        ack: Sender<crate::Result<()>>,
    },
    /// Stop the worker thread.
    Shutdown,
}

/// Background worker that owns the write path for indexes.
pub(crate) struct UpdateWorker {
    tx: Sender<UpdateMsg>,
    handle: Mutex<Option<JoinHandle<()>>>,
    running: Arc<AtomicBool>,
}

impl UpdateWorker {
    /// Spawn the update worker.
    ///
    /// `applier` is owned by the worker thread so that dropping the engine
    /// blocks until the thread has released every reference.
    pub(crate) fn new(applier: Arc<IndexApplier>, bound: usize) -> Self {
        let (tx, rx) = bounded::<UpdateMsg>(bound);
        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();
        let handle = Builder::new()
            .name("turbo-update".into())
            .spawn(move || worker_loop(applier, rx, running_clone))
            .expect("failed to spawn turbo-update worker");
        Self {
            tx,
            handle: Mutex::new(Some(handle)),
            running,
        }
    }

    /// Submit a batch of records and wait until the worker has applied it.
    pub fn submit_and_wait(&self, records: Vec<(PointOffset, Record)>) -> crate::Result<()> {
        let (ack_tx, ack_rx) = bounded::<crate::Result<()>>(1);
        self.tx
            .send(UpdateMsg::Batch {
                records,
                ack: ack_tx,
            })
            .map_err(|_| {
                crate::StorageError::InvalidArgument("update worker disconnected".into())
            })?;
        ack_rx.recv().map_err(|_| {
            crate::StorageError::InvalidArgument("update worker acknowledgement lost".into())
        })?
    }

    /// Stop the worker thread.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        let _ = self.tx.send(UpdateMsg::Shutdown);
        if let Some(handle) = self.handle.lock().take() {
            let _ = handle.join();
        }
    }
}

impl Drop for UpdateWorker {
    fn drop(&mut self) {
        self.stop();
    }
}

fn worker_loop(applier: Arc<IndexApplier>, rx: Receiver<UpdateMsg>, running: Arc<AtomicBool>) {
    while running.load(Ordering::Relaxed) {
        let msg = match rx.recv() {
            Ok(msg) => msg,
            Err(_) => break,
        };
        match msg {
            UpdateMsg::Batch { records, ack } => {
                let result = applier.apply_batch(&records);
                let _ = ack.send(result);
            }
            UpdateMsg::Shutdown => break,
        }
    }
}
