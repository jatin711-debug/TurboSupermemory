//! In-memory index for per-agent memory scoping.
//!
//! A record's `scope` is an optional namespace tag. Records with `scope = None`
//! are global/shared; records with `scope = Some(agent_id)` are private to
//! that agent. Scoped searches return records matching the requested scope
//! **plus** all global records.
//!
//! The index maps each scope name to a Roaring bitmap of offsets. Global
//! offsets are tracked separately so `query(None)` is cheap (no restriction)
//! and `query(Some(scope))` is a single bitmap union.

use crate::record::PointOffset;
use ahash::AHashMap;
use roaring::RoaringBitmap;

/// Fast lookup of offsets by memory scope.
#[derive(Debug, Default)]
pub struct ScopeIndex {
    /// scope name -> offsets with that exact scope.
    by_scope: AHashMap<String, RoaringBitmap>,
    /// Offsets whose scope is `None` (global/shared).
    global: RoaringBitmap,
}

impl ScopeIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Index a single offset under the given scope.
    pub fn add(&mut self, offset: PointOffset, scope: Option<&str>) {
        let bo = offset as u32;
        match scope {
            Some(s) => {
                self.by_scope.entry(s.to_string()).or_default().insert(bo);
            }
            None => {
                self.global.insert(bo);
            }
        }
    }

    /// Remove an offset from whichever scope it was indexed under.
    pub fn remove(&mut self, offset: PointOffset, scope: Option<&str>) {
        let bo = offset as u32;
        match scope {
            Some(s) => {
                if let Some(bm) = self.by_scope.get_mut(s) {
                    bm.remove(bo);
                    if bm.is_empty() {
                        self.by_scope.remove(s);
                    }
                }
            }
            None => {
                self.global.remove(bo);
            }
        }
    }

    /// Return the bitmap of offsets visible to a search with the given scope.
    ///
    /// - `None` -> no restriction (return empty bitmap; caller treats as
    ///   "all offsets allowed").
    /// - `Some(scope)` -> offsets in `scope` plus global offsets.
    pub fn query(&self, scope: Option<&str>) -> RoaringBitmap {
        match scope {
            None => RoaringBitmap::new(),
            Some(s) => {
                let mut result = self.global.clone();
                if let Some(bm) = self.by_scope.get(s) {
                    result |= bm;
                }
                result
            }
        }
    }

    /// All offsets currently indexed (global + every named scope).
    pub fn all_offsets(&self) -> RoaringBitmap {
        let mut result = self.global.clone();
        for bm in self.by_scope.values() {
            result |= bm;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_none_is_empty() {
        let mut idx = ScopeIndex::new();
        idx.add(0, None);
        idx.add(1, Some("a"));
        assert!(idx.query(None).is_empty());
    }

    #[test]
    fn query_scope_includes_global() {
        let mut idx = ScopeIndex::new();
        idx.add(0, None);
        idx.add(1, Some("agent1"));
        idx.add(2, Some("agent2"));
        let result: Vec<u32> = idx.query(Some("agent1")).iter().collect();
        assert_eq!(result, vec![0, 1]);
    }

    #[test]
    fn remove_updates_index() {
        let mut idx = ScopeIndex::new();
        idx.add(0, None);
        idx.add(1, Some("agent1"));
        idx.remove(0, None);
        idx.remove(1, Some("agent1"));
        assert!(idx.query(Some("agent1")).is_empty());
        assert!(idx.all_offsets().is_empty());
    }
}
