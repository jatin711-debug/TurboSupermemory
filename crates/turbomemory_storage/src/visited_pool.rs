//! Reusable visited-set pool for graph traversal.
//!
//! Graph searches (HNSW, ACORN, etc.) need a "visited" bitmap to avoid
//! revisiting nodes.  Allocating a fresh `Vec<u8>` per query is expensive at
//! high query rates.  This pool hands out token-based bitmaps: each slot is a
//! `Vec<u8>` sized to the current maximum point offset, and queries mark nodes
//! by writing a per-query generation token instead of clearing the array.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A checked-out visited set.  Marking a node writes the current generation
/// token; checking a node compares its stored token to the live token.
pub struct VisitedSet {
    tokens: Vec<u8>,
    generation: u8,
}

impl VisitedSet {
    /// Create a new visited set sized for `capacity` nodes.
    pub fn new(capacity: usize) -> Self {
        Self {
            tokens: vec![0u8; capacity],
            generation: 1,
        }
    }

    /// Ensure the set can hold at least `capacity` nodes.
    pub fn resize(&mut self, capacity: usize) {
        if self.tokens.len() < capacity {
            self.tokens.resize(capacity, 0);
        }
    }

    /// Mark `node` as visited in the current generation.
    #[inline]
    pub fn mark(&mut self, node: usize) {
        if node < self.tokens.len() {
            self.tokens[node] = self.generation;
        }
    }

    /// Returns `true` if `node` was marked in the current generation.
    #[inline]
    pub fn is_visited(&self, node: usize) -> bool {
        node < self.tokens.len() && self.tokens[node] == self.generation
    }

    /// Reset all visitation state for the next search.
    ///
    /// Instead of zeroing the array we bump the generation token.  If the
    /// token wraps to zero we must clear the array to avoid falsely reporting
    /// nodes as visited.
    pub fn clear(&mut self) {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == 0 {
            self.tokens.fill(0);
            self.generation = 1;
        }
    }
}

/// Pool of reusable visited sets.
pub struct VisitedPool {
    slots: Mutex<Vec<VisitedSet>>,
    capacity: AtomicUsize,
}

impl VisitedPool {
    /// Create a pool with `size` slots, each sized for `capacity` nodes.
    pub fn new(size: usize, capacity: usize) -> Self {
        let slots: Vec<VisitedSet> = (0..size).map(|_| VisitedSet::new(capacity)).collect();
        Self {
            slots: Mutex::new(slots),
            capacity: AtomicUsize::new(capacity),
        }
    }

    /// Ensure every slot in the pool can hold at least `capacity` nodes.
    pub fn resize(&self, capacity: usize) {
        let current = self.capacity.load(Ordering::Relaxed);
        if capacity <= current {
            return;
        }
        self.capacity.store(capacity, Ordering::Relaxed);
        let mut slots = self.slots.lock();
        for slot in slots.iter_mut() {
            slot.resize(capacity);
        }
    }

    /// Check out a visited set.  The set is already reset for a new search.
    pub fn acquire(&self) -> VisitedSetHandle<'_> {
        let mut slot = {
            let mut slots = self.slots.lock();
            slots
                .pop()
                .unwrap_or_else(|| VisitedSet::new(self.capacity.load(Ordering::Relaxed)))
        };
        slot.clear();
        VisitedSetHandle { slot, pool: self }
    }

    fn release(&self, slot: VisitedSet) {
        let mut slots = self.slots.lock();
        slots.push(slot);
    }
}

/// RAII handle for a checked-out visited set.
pub struct VisitedSetHandle<'a> {
    slot: VisitedSet,
    pool: &'a VisitedPool,
}

impl<'a> std::ops::Deref for VisitedSetHandle<'a> {
    type Target = VisitedSet;

    fn deref(&self) -> &Self::Target {
        &self.slot
    }
}

impl<'a> std::ops::DerefMut for VisitedSetHandle<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.slot
    }
}

impl<'a> Drop for VisitedSetHandle<'a> {
    fn drop(&mut self) {
        let slot = std::mem::replace(&mut self.slot, VisitedSet::new(0));
        self.pool.release(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visited_set_generations() {
        let mut v = VisitedSet::new(10);
        v.mark(3);
        assert!(v.is_visited(3));
        assert!(!v.is_visited(4));
        v.clear();
        assert!(!v.is_visited(3));
        v.mark(4);
        assert!(v.is_visited(4));
    }

    #[test]
    fn visited_pool_reuses_slots() {
        let pool = VisitedPool::new(2, 16);
        {
            let mut h = pool.acquire();
            h.mark(5);
        }
        {
            let h = pool.acquire();
            assert!(!h.is_visited(5));
        }
    }

    #[test]
    fn visited_pool_resizes() {
        let pool = VisitedPool::new(1, 8);
        pool.resize(32);
        let mut h = pool.acquire();
        h.mark(20);
        assert!(h.is_visited(20));
    }
}
