//! Lock-free-ish access counters that replace per-query metadata writes.
//!
//! Access counts are accumulated in a small in-memory table and drained into the
//! `MetadataStore` on flush / consolidation. Counts since the last flush are lost
//! on crash, which is acceptable because access scoring is a heuristic for
//! promotion/demotion.

use crate::metadata_store::MetadataStore;
use crate::record::PointOffset;
use ahash::HashMap as AHashMap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[derive(Default, Debug)]
struct AccessEntry {
    count: AtomicU64,
    last_accessed: AtomicU64,
}

/// Fast per-offset access counters.
#[derive(Default, Debug)]
pub struct AccessCounters {
    counters: Mutex<AHashMap<PointOffset, Arc<AccessEntry>>>,
}

impl AccessCounters {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one access for `offset` at the current time.
    pub fn bump(&self, offset: PointOffset, now: u64) {
        let entry = {
            let mut map = self.counters.lock();
            match map.get(&offset) {
                Some(entry) => Arc::clone(entry),
                None => {
                    let entry = Arc::new(AccessEntry::default());
                    map.insert(offset, Arc::clone(&entry));
                    entry
                }
            }
        };
        entry.count.fetch_add(1, Ordering::Relaxed);
        // `last_accessed` only increases; store unconditionally is fine because
        // `now` comes from a monotonic clock.
        entry.last_accessed.store(now, Ordering::Relaxed);
    }

    /// Drain all accumulated counters into the metadata store.
    ///
    /// This is called on flush and before promotion so that the metadata cache
    /// sees the latest access scores without paying a metadata write on every
    /// search result.
    pub fn drain_into(&self, meta: &MetadataStore) -> crate::Result<()> {
        let counters: AHashMap<PointOffset, Arc<AccessEntry>> = {
            let mut map = self.counters.lock();
            std::mem::take(&mut *map)
        };
        if counters.is_empty() {
            return Ok(());
        }
        for (offset, entry) in counters {
            let count = entry.count.load(Ordering::Relaxed);
            let last = entry.last_accessed.load(Ordering::Relaxed);
            if count == 0 {
                continue;
            }
            if let Some(mut meta_rec) = meta.get(offset)? {
                meta_rec.access_count += count;
                meta_rec.last_accessed = meta_rec.last_accessed.max(last);
                meta.put_meta(offset, &meta_rec)?;
            }
        }
        Ok(())
    }
}
