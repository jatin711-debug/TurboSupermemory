//! Lock-free-ish access counters that replace per-query metadata writes.
//!
//! Access counts are accumulated in a small in-memory table and drained into the
//! `MetadataStore` on flush / consolidation. Counts since the last flush are lost
//! on crash, which is acceptable because access scoring is a heuristic for
//! promotion/demotion. Each bump also pushes the access timestamp into a small
//! per-record ring buffer (the last K accesses) that feeds the ACT-R base-level
//! activation model; the ring is drained and persisted alongside the counts and
//! shares the same crash-loss contract.

use crate::metadata_store::MetadataStore;
use crate::record::PointOffset;
use ahash::HashMap as AHashMap;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Default number of access timestamps kept per record for ACT-R activation
/// (roadmap K = 8 — the "history" half of the base-level equation).
pub const DEFAULT_ACCESS_HISTORY_LEN: usize = 8;

/// Upper clamp for the access-history ring length; bounds per-record memory
/// (32 × `u64` = 256 B worst case).
pub const MAX_ACCESS_HISTORY_LEN: usize = 32;

/// Clamp a configured history length to the supported range.
pub fn clamp_history_len(len: usize) -> usize {
    len.clamp(1, MAX_ACCESS_HISTORY_LEN)
}

/// ACT-R base-level activation (Anderson & Schooler 1991):
/// `A = ln(Σ_j age_j^-d)`, where `age_j = now - t_j` is the age of the j-th
/// access in seconds and `d` is the decay exponent (~0.5 fits human
/// forgetting data).
///
/// Ages are clamped to at least 1 second (ε) so a just-now access yields a
/// finite value instead of `+∞`. An empty history returns
/// `f64::NEG_INFINITY`: a never-accessed record sorts strictly below every
/// accessed one, mirroring the role the legacy score's 0 floor plays.
pub fn actr_activation(timestamps: &[u64], now: u64, decay: f64) -> f64 {
    if timestamps.is_empty() {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = timestamps
        .iter()
        .map(|&ts| now.saturating_sub(ts).max(1) as f64)
        .map(|age| age.powf(-decay))
        .sum();
    sum.ln()
}

#[derive(Debug)]
struct AccessEntry {
    count: AtomicU64,
    last_accessed: AtomicU64,
    /// Ring of the last K access timestamps (unix seconds). Slot
    /// `cursor % K` is overwritten on every bump, so once full the oldest
    /// surviving timestamp is evicted first. Zero slots were never written —
    /// epoch 0 doubles as the "never" sentinel, matching `last_accessed`.
    ring: Box<[AtomicU64]>,
    ring_cursor: AtomicU64,
}

impl AccessEntry {
    fn new(history_len: usize) -> Self {
        Self {
            count: AtomicU64::new(0),
            last_accessed: AtomicU64::new(0),
            ring: (0..history_len).map(|_| AtomicU64::new(0)).collect(),
            ring_cursor: AtomicU64::new(0),
        }
    }
}

/// Fast per-offset access counters.
#[derive(Debug)]
pub struct AccessCounters {
    counters: Mutex<AHashMap<PointOffset, Arc<AccessEntry>>>,
    /// Length of each entry's timestamp ring (already clamped to 1..=32).
    history_len: usize,
}

impl Default for AccessCounters {
    fn default() -> Self {
        Self::new(DEFAULT_ACCESS_HISTORY_LEN)
    }
}

impl AccessCounters {
    pub fn new(history_len: usize) -> Self {
        Self {
            counters: Mutex::new(AHashMap::default()),
            history_len: clamp_history_len(history_len),
        }
    }

    /// Record one access for `offset` at the current time.
    pub fn bump(&self, offset: PointOffset, now: u64) {
        let entry = {
            let mut map = self.counters.lock();
            match map.get(&offset) {
                Some(entry) => Arc::clone(entry),
                None => {
                    let entry = Arc::new(AccessEntry::new(self.history_len));
                    map.insert(offset, Arc::clone(&entry));
                    entry
                }
            }
        };
        entry.count.fetch_add(1, Ordering::Relaxed);
        // `last_accessed` only increases; store unconditionally is fine because
        // `now` comes from a monotonic clock.
        entry.last_accessed.store(now, Ordering::Relaxed);
        // Push the timestamp into the ring (overwrites the oldest slot once
        // full). One atomic store per bump; chronological order is recovered
        // by sorting at drain time, which is all ACT-R's order-free sum needs.
        let slot = (entry.ring_cursor.fetch_add(1, Ordering::Relaxed) as usize) % entry.ring.len();
        entry.ring[slot].store(now, Ordering::Relaxed);
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
                // Merge the surviving ring timestamps into the record's
                // persisted access history — same drain contract as the
                // counters above (undrained bumps may be lost on crash).
                let mut stamps: Vec<u64> = entry
                    .ring
                    .iter()
                    .map(|slot| slot.load(Ordering::Relaxed))
                    .filter(|&ts| ts != 0)
                    .collect();
                stamps.sort_unstable();
                meta.append_access_history(offset, &stamps, self.history_len);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Tier;
    use crate::record::MetaRecord;

    fn seed_meta(meta: &MetadataStore, offset: PointOffset) {
        meta.put_meta(
            offset,
            &MetaRecord {
                id: format!("id-{offset}"),
                text: "text".to_string(),
                importance: 1.0,
                concepts: vec![],
                created_at: 0,
                insert_seq: 0,
                access_count: 0,
                last_accessed: 0,
                tier: Tier::Hot,
                payload: None,
                scope: None,
                source_role: None,
            },
        )
        .unwrap();
    }

    #[test]
    fn empty_history_is_negative_infinity() {
        assert_eq!(actr_activation(&[], 1000, 0.5), f64::NEG_INFINITY);
    }

    #[test]
    fn just_now_access_uses_epsilon_clamp() {
        // Age 0 (same second) is clamped to ε = 1s: ln(1^-0.5) = 0.
        assert_eq!(actr_activation(&[1000], 1000, 0.5), 0.0);
        // Age 1 yields the identical value; future timestamps (clock skew)
        // saturate to the same clamp instead of producing negative ages.
        assert_eq!(actr_activation(&[999], 1000, 0.5), 0.0);
        assert_eq!(actr_activation(&[2000], 1000, 0.5), 0.0);
    }

    #[test]
    fn spaced_rehearsal_beats_single_recent_burst() {
        let now = 10_000u64;
        // One burst: three accesses crammed into a single moment 60s ago.
        let burst = [now - 60, now - 60, now - 60];
        // Spaced: three accesses distributed over the last ~17 minutes.
        let spaced = [now - 10, now - 100, now - 1000];
        let burst_score = actr_activation(&burst, now, 0.5);
        let spaced_score = actr_activation(&spaced, now, 0.5);
        assert!(
            spaced_score > burst_score,
            "spaced {spaced_score} should beat burst {burst_score}"
        );
        // Known values: burst = ln(3 × 60^-0.5) ≈ -0.949; spaced =
        // ln(10^-0.5 + 100^-0.5 + 1000^-0.5) ≈ -0.803.
        let expected_burst = (3.0f64 * 60f64.powf(-0.5)).ln();
        let expected_spaced = (10f64.powf(-0.5) + 100f64.powf(-0.5) + 1000f64.powf(-0.5)).ln();
        assert!((burst_score - expected_burst).abs() < 1e-9);
        assert!((spaced_score - expected_spaced).abs() < 1e-9);
    }

    #[test]
    fn several_spaced_beats_one_single_access() {
        let now = 10_000u64;
        let single = actr_activation(&[now - 10], now, 0.5);
        let several = actr_activation(&[now - 10, now - 100, now - 1000], now, 0.5);
        assert!(several > single, "frequency must add activation");
    }

    #[test]
    fn higher_decay_forgets_faster() {
        let now = 10_000u64;
        let history = [now - 100, now - 1000];
        assert!(actr_activation(&history, now, 1.0) < actr_activation(&history, now, 0.5));
    }

    #[test]
    fn ring_keeps_only_the_most_recent_k_timestamps() {
        let counters = AccessCounters::new(4);
        for ts in 1..=10u64 {
            counters.bump(7, ts);
        }
        let tmp = tempfile::tempdir().unwrap();
        let meta = MetadataStore::open(tmp.path()).unwrap();
        seed_meta(&meta, 7);
        counters.drain_into(&meta).unwrap();
        // 10 bumps through a K=4 ring leave the last 4 timestamps.
        assert_eq!(meta.access_history(7), vec![7, 8, 9, 10]);
    }

    #[test]
    fn drain_merges_with_persisted_history_capped_at_k() {
        let counters = AccessCounters::new(4);
        let tmp = tempfile::tempdir().unwrap();
        let meta = MetadataStore::open(tmp.path()).unwrap();
        seed_meta(&meta, 7);

        counters.bump(7, 100);
        counters.drain_into(&meta).unwrap();
        assert_eq!(meta.access_history(7), vec![100]);

        // A later drain merges with (not overwrites) the persisted ring.
        for ts in [200u64, 300, 400] {
            counters.bump(7, ts);
        }
        counters.drain_into(&meta).unwrap();
        assert_eq!(meta.access_history(7), vec![100, 200, 300, 400]);

        // Overflow drops the oldest timestamps first.
        counters.bump(7, 500);
        counters.drain_into(&meta).unwrap();
        assert_eq!(meta.access_history(7), vec![200, 300, 400, 500]);
    }

    #[test]
    fn history_len_is_clamped() {
        let counters = AccessCounters::new(0);
        assert_eq!(counters.history_len, 1);
        let counters = AccessCounters::new(10_000);
        assert_eq!(counters.history_len, MAX_ACCESS_HISTORY_LEN);
    }
}
