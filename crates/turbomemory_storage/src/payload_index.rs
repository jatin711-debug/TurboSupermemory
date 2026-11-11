//! In-memory payload index for filtered vector search.
//!
//! Indexes top-level fields of JSON payloads using Roaring bitmaps:
//! - keyword equality (strings, booleans, numbers as strings)
//! - numeric range queries
//!
//! Arrays are flattened: each element is indexed under the same field key.
//!
//! `PointOffset` is `u64` everywhere else in the engine; RoaringBitmap stores
//! `u32`, so offsets are cast when entering/leaving the index. This is safe for
//! collections with fewer than 2^32 points, which covers all realistic uses.

use crate::record::PointOffset;
use crate::StorageError;
use ahash::AHashMap;
use ordered_float::NotNan;
use roaring::RoaringBitmap;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound;

/// A filter predicate that can be evaluated against the payload index.
#[derive(Debug, Clone)]
pub enum Filter {
    /// Equality match on a top-level payload field.
    Eq { field: String, value: Value },
    /// Numeric range match on a top-level payload field.
    Range {
        field: String,
        low: Bound<f64>,
        high: Bound<f64>,
    },
    /// Full-text contains query (handled by the engine's text index).
    FullText { field: String, query: String },
    /// All sub-filters must match.
    And(Vec<Filter>),
    /// Any sub-filter must match.
    Or(Vec<Filter>),
    /// Negation.
    Not(Box<Filter>),
}

/// How deeply `Filter::from_json` lets filters nest. A filter is caller input
/// (REST, Python) and is parsed recursively; the cap keeps a hostile one from
/// exhausting the stack.
pub const MAX_FILTER_DEPTH: usize = 32;

impl Filter {
    /// Parse the JSON filter DSL shared by the REST API and the Python
    /// binding:
    ///
    /// - `{"op": "eq", "field": "<name>", "value": <json>}`
    /// - `{"op": "range", "field": "<name>", "low": <number>, "high": <number>,
    ///   "low_inclusive": bool, "high_inclusive": bool}` (either bound may be
    ///   left out; both are inclusive unless said otherwise)
    /// - `{"op": "full_text", "field": "<name>", "query": "<text>"}`
    /// - `{"op": "and" | "or", "filters": [ ... ]}`
    /// - `{"op": "not", "filter": { ... }}`
    ///
    /// Nesting deeper than [`MAX_FILTER_DEPTH`] is rejected. The error is a
    /// message for the caller.
    pub fn from_json(value: &Value) -> Result<Filter, String> {
        Self::from_json_at(value, 0)
    }

    fn from_json_at(value: &Value, depth: usize) -> Result<Filter, String> {
        if depth >= MAX_FILTER_DEPTH {
            return Err(format!(
                "filter nesting exceeds the maximum depth of {MAX_FILTER_DEPTH}"
            ));
        }
        let obj = value
            .as_object()
            .ok_or_else(|| "filter must be a JSON object".to_string())?;
        let op = obj
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "filter missing op".to_string())?;
        let field = |what: &str| -> Result<String, String> {
            obj.get("field")
                .and_then(|v| v.as_str())
                .map(str::to_string)
                .ok_or_else(|| format!("{what} missing field"))
        };
        let list = |what: &str| -> Result<Vec<Filter>, String> {
            obj.get("filters")
                .and_then(|v| v.as_array())
                .ok_or_else(|| format!("{what} missing filters"))?
                .iter()
                .map(|v| Self::from_json_at(v, depth + 1))
                .collect()
        };
        let bound = |value: &str, inclusive: &str| -> Bound<f64> {
            match obj.get(value).and_then(|v| v.as_f64()) {
                None => Bound::Unbounded,
                Some(v) if obj.get(inclusive).and_then(|b| b.as_bool()).unwrap_or(true) => {
                    Bound::Included(v)
                }
                Some(v) => Bound::Excluded(v),
            }
        };
        Ok(match op {
            "eq" => Filter::Eq {
                field: field("eq")?,
                value: obj.get("value").cloned().unwrap_or(Value::Null),
            },
            "range" => Filter::Range {
                field: field("range")?,
                low: bound("low", "low_inclusive"),
                high: bound("high", "high_inclusive"),
            },
            "full_text" => Filter::FullText {
                field: field("full_text")?,
                query: obj
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "full_text missing query".to_string())?
                    .to_string(),
            },
            "and" => Filter::And(list("and")?),
            "or" => Filter::Or(list("or")?),
            "not" => Filter::Not(Box::new(Self::from_json_at(
                obj.get("filter")
                    .ok_or_else(|| "not missing filter".to_string())?,
                depth + 1,
            )?)),
            other => return Err(format!("unknown op: {other}")),
        })
    }

    /// Returns true if this filter (recursively) contains a full-text query.
    pub fn uses_full_text(&self) -> bool {
        match self {
            Filter::FullText { .. } => true,
            Filter::And(parts) | Filter::Or(parts) => parts.iter().any(|p| p.uses_full_text()),
            Filter::Not(inner) => inner.uses_full_text(),
            _ => false,
        }
    }
}

fn bm_offset(offset: PointOffset) -> u32 {
    offset as u32
}

/// Inverted index over JSON payloads.
#[derive(Debug, Default)]
pub struct PayloadIndex {
    /// field -> value_str -> offsets (equality / keyword index)
    keyword: AHashMap<String, AHashMap<String, RoaringBitmap>>,
    /// field -> numeric value -> offsets (range index)
    numeric: AHashMap<String, BTreeMap<NotNan<f64>, RoaringBitmap>>,
    /// All offsets currently indexed.
    all_offsets: RoaringBitmap,
}

impl PayloadIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build an index from a snapshot of metadata records.
    pub fn from_meta_records(records: &HashMap<PointOffset, crate::record::MetaRecord>) -> Self {
        Self::from_meta_records_iter(records.iter().map(|(o, m)| (*o, m)))
    }

    /// Build an index from an iterator of `(offset, record)` pairs.
    ///
    /// Avoids cloning the whole metadata map during engine open.
    pub fn from_meta_records_iter<'a>(
        records: impl Iterator<Item = (PointOffset, &'a crate::record::MetaRecord)>,
    ) -> Self {
        let mut idx = Self::new();
        for (offset, meta) in records {
            // Best-effort indexing; malformed payloads are ignored.
            let _ = idx.add(offset, meta.payload.as_deref());
        }
        idx
    }

    /// Index the payload attached to `offset`.
    pub fn add(&mut self, offset: PointOffset, payload: Option<&str>) -> crate::Result<()> {
        let bo = bm_offset(offset);
        self.all_offsets.insert(bo);
        let Some(payload) = payload else {
            return Ok(());
        };
        let value: Value = serde_json::from_str(payload)
            .map_err(|e| StorageError::InvalidArgument(format!("invalid payload JSON: {e}")))?;
        let Value::Object(map) = value else {
            return Ok(());
        };
        for (field, v) in map {
            self.index_field_value(&field, bo, &v);
        }
        Ok(())
    }

    /// Remove an offset from all indexes.
    pub fn remove(&mut self, offset: PointOffset, payload: Option<&str>) {
        let bo = bm_offset(offset);
        self.all_offsets.remove(bo);
        let Some(payload) = payload else { return };
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        let Value::Object(map) = value else {
            return;
        };
        for (field, v) in map {
            self.remove_field_value(&field, bo, &v);
        }
    }

    /// Evaluate `filter` and return the matching offsets.
    pub fn query(&self, filter: &Filter) -> RoaringBitmap {
        self.query_inner(filter)
    }

    /// All offsets currently tracked by the index.
    pub fn all_offsets(&self) -> &RoaringBitmap {
        &self.all_offsets
    }

    fn query_inner(&self, filter: &Filter) -> RoaringBitmap {
        match filter {
            Filter::Eq { field, value } => self.query_eq(field, value),
            Filter::Range { field, low, high } => self.query_range(field, low, high),
            Filter::FullText { .. } => RoaringBitmap::new(),
            Filter::And(parts) => {
                let mut iter = parts.iter();
                let Some(first) = iter.next() else {
                    return RoaringBitmap::new();
                };
                let mut acc = self.query_inner(first);
                for part in iter {
                    if acc.is_empty() {
                        return acc;
                    }
                    acc &= self.query_inner(part);
                }
                acc
            }
            Filter::Or(parts) => {
                let mut acc = RoaringBitmap::new();
                for part in parts {
                    acc |= self.query_inner(part);
                }
                acc
            }
            Filter::Not(inner) => {
                let positives = self.query_inner(inner);
                &self.all_offsets - &positives
            }
        }
    }

    fn query_eq(&self, field: &str, value: &Value) -> RoaringBitmap {
        let key = value_to_keyword(value);
        self.keyword
            .get(field)
            .and_then(|m| m.get(&key))
            .cloned()
            .unwrap_or_default()
    }

    fn query_range(&self, field: &str, low: &Bound<f64>, high: &Bound<f64>) -> RoaringBitmap {
        let Some(tree) = self.numeric.get(field) else {
            return RoaringBitmap::new();
        };
        let mut result = RoaringBitmap::new();
        for (_, bitmap) in tree.range((map_bound(low), map_bound(high))) {
            result |= bitmap;
        }
        result
    }

    fn index_field_value(&mut self, field: &str, offset: u32, value: &Value) {
        match value {
            Value::Array(arr) => {
                for item in arr {
                    self.index_scalar(field, offset, item);
                }
            }
            _ => self.index_scalar(field, offset, value),
        }
    }

    fn index_scalar(&mut self, field: &str, offset: u32, value: &Value) {
        match value {
            Value::String(s) => {
                self.keyword
                    .entry(field.to_string())
                    .or_default()
                    .entry(s.clone())
                    .or_default()
                    .insert(offset);
            }
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if let Ok(nn) = NotNan::new(f) {
                        self.numeric
                            .entry(field.to_string())
                            .or_default()
                            .entry(nn)
                            .or_default()
                            .insert(offset);
                    }
                }
                // Also index the canonical string for equality queries.
                self.keyword
                    .entry(field.to_string())
                    .or_default()
                    .entry(n.to_string())
                    .or_default()
                    .insert(offset);
            }
            Value::Bool(b) => {
                self.keyword
                    .entry(field.to_string())
                    .or_default()
                    .entry(b.to_string())
                    .or_default()
                    .insert(offset);
            }
            _ => {}
        }
    }

    fn remove_field_value(&mut self, field: &str, offset: u32, value: &Value) {
        match value {
            Value::Array(arr) => {
                for item in arr {
                    self.remove_scalar(field, offset, item);
                }
            }
            _ => self.remove_scalar(field, offset, value),
        }
    }

    fn remove_scalar(&mut self, field: &str, offset: u32, value: &Value) {
        match value {
            Value::String(s) => {
                if let Some(m) = self.keyword.get_mut(field) {
                    if let Some(bm) = m.get_mut(s) {
                        bm.remove(offset);
                        if bm.is_empty() {
                            m.remove(s);
                        }
                    }
                }
            }
            Value::Number(n) => {
                if let Some(f) = n.as_f64() {
                    if let Ok(nn) = NotNan::new(f) {
                        if let Some(tree) = self.numeric.get_mut(field) {
                            if let Some(bm) = tree.get_mut(&nn) {
                                bm.remove(offset);
                                if bm.is_empty() {
                                    tree.remove(&nn);
                                }
                            }
                        }
                    }
                }
                if let Some(m) = self.keyword.get_mut(field) {
                    if let Some(bm) = m.get_mut(&n.to_string()) {
                        bm.remove(offset);
                        if bm.is_empty() {
                            m.remove(&n.to_string());
                        }
                    }
                }
            }
            Value::Bool(b) => {
                if let Some(m) = self.keyword.get_mut(field) {
                    if let Some(bm) = m.get_mut(&b.to_string()) {
                        bm.remove(offset);
                        if bm.is_empty() {
                            m.remove(&b.to_string());
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn value_to_keyword(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => "null".to_string(),
        other => other.to_string(),
    }
}

fn map_bound(b: &Bound<f64>) -> Bound<NotNan<f64>> {
    match *b {
        Bound::Included(f) => match NotNan::new(f) {
            Ok(nn) => Bound::Included(nn),
            Err(_) => Bound::Unbounded,
        },
        Bound::Excluded(f) => match NotNan::new(f) {
            Ok(nn) => Bound::Excluded(nn),
            Err(_) => Bound::Unbounded,
        },
        Bound::Unbounded => Bound::Unbounded,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(json: &str) -> Option<&str> {
        Some(json)
    }

    #[test]
    fn json_filters_parse_and_match() {
        let mut idx = PayloadIndex::new();
        idx.add(1, payload(r#"{"status":"pending","due":5}"#))
            .unwrap();
        idx.add(2, payload(r#"{"status":"pending","due":50}"#))
            .unwrap();
        idx.add(3, payload(r#"{"status":"done","due":1}"#)).unwrap();
        let parse = |s: &str| Filter::from_json(&serde_json::from_str(s).unwrap());
        let due_pending = parse(
            r#"{"op":"and","filters":[
                {"op":"eq","field":"status","value":"pending"},
                {"op":"range","field":"due","high":10}]}"#,
        )
        .unwrap();
        assert_eq!(idx.query(&due_pending).iter().collect::<Vec<_>>(), vec![1]);
        let open_low =
            parse(r#"{"op":"range","field":"due","low":5,"low_inclusive":false}"#).unwrap();
        assert_eq!(idx.query(&open_low).iter().collect::<Vec<_>>(), vec![2]);
        let not_done =
            parse(r#"{"op":"not","filter":{"op":"eq","field":"status","value":"done"}}"#).unwrap();
        assert_eq!(idx.query(&not_done).iter().collect::<Vec<_>>(), vec![1, 2]);

        assert!(parse(r#"{"field":"x"}"#).unwrap_err().contains("op"));
        assert!(parse(r#"{"op":"eq"}"#).unwrap_err().contains("field"));
        assert!(parse(r#"{"op":"nope"}"#)
            .unwrap_err()
            .contains("unknown op"));
        assert!(parse("[1]").is_err());
        let mut deep = serde_json::json!({"op": "eq", "field": "f", "value": 1});
        for _ in 0..MAX_FILTER_DEPTH {
            deep = serde_json::json!({"op": "not", "filter": deep});
        }
        assert!(Filter::from_json(&deep).unwrap_err().contains("depth"));
    }

    #[test]
    fn equality_and_range() {
        let mut idx = PayloadIndex::new();
        idx.add(0, payload(r#"{"tags":["rust","ai"],"count":42}"#))
            .unwrap();
        idx.add(1, payload(r#"{"tags":["python","ai"],"count":7}"#))
            .unwrap();
        idx.add(2, payload(r#"{"tags":["rust"],"count":100}"#))
            .unwrap();

        let f = Filter::Eq {
            field: "tags".into(),
            value: Value::String("rust".into()),
        };
        assert_eq!(idx.query(&f).iter().collect::<Vec<_>>(), vec![0, 2]);

        let f = Filter::Range {
            field: "count".into(),
            low: Bound::Included(10.0),
            high: Bound::Included(50.0),
        };
        assert_eq!(idx.query(&f).iter().collect::<Vec<_>>(), vec![0]);

        let f = Filter::And(vec![
            Filter::Eq {
                field: "tags".into(),
                value: Value::String("ai".into()),
            },
            Filter::Range {
                field: "count".into(),
                low: Bound::Included(0.0),
                high: Bound::Included(10.0),
            },
        ]);
        assert_eq!(idx.query(&f).iter().collect::<Vec<_>>(), vec![1]);
    }

    #[test]
    fn remove_keeps_index_consistent() {
        let mut idx = PayloadIndex::new();
        idx.add(0, payload(r#"{"category":"a","n":1}"#)).unwrap();
        idx.add(1, payload(r#"{"category":"a","n":2}"#)).unwrap();
        idx.remove(0, payload(r#"{"category":"a","n":1}"#));

        let f = Filter::Eq {
            field: "category".into(),
            value: Value::String("a".into()),
        };
        assert_eq!(idx.query(&f).iter().collect::<Vec<_>>(), vec![1]);

        let f = Filter::Range {
            field: "n".into(),
            low: Bound::Included(0.0),
            high: Bound::Included(1.5),
        };
        assert!(idx.query(&f).is_empty());
    }
}
