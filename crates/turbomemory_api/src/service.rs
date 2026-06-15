//! Shared service layer used by both the gRPC and REST frontends.

use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use turbomemory_storage::config::StoreConfig;
use turbomemory_storage::engine::StorageEngine;
use turbomemory_storage::payload_index::Filter as StorageFilter;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("storage error: {0}")]
    Storage(#[from] turbomemory_storage::StorageError),
    #[error("missing field: {0}")]
    MissingField(String),
    #[error("invalid filter: {0}")]
    InvalidFilter(String),
}

impl From<ApiError> for tonic::Status {
    fn from(e: ApiError) -> Self {
        match e {
            ApiError::MissingField(_) | ApiError::InvalidFilter(_) => {
                tonic::Status::invalid_argument(e.to_string())
            }
            ApiError::Storage(ref se) => match se {
                turbomemory_storage::StorageError::DuplicateId(_)
                | turbomemory_storage::StorageError::DimensionMismatch
                | turbomemory_storage::StorageError::InvalidArgument(_) => {
                    tonic::Status::invalid_argument(e.to_string())
                }
                turbomemory_storage::StorageError::NotFound(_) => {
                    tonic::Status::not_found(e.to_string())
                }
                _ => tonic::Status::internal(e.to_string()),
            },
        }
    }
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let status = match &self {
            ApiError::MissingField(_) | ApiError::InvalidFilter(_) => {
                axum::http::StatusCode::BAD_REQUEST
            }
            ApiError::Storage(se) => match se {
                turbomemory_storage::StorageError::DuplicateId(_)
                | turbomemory_storage::StorageError::DimensionMismatch
                | turbomemory_storage::StorageError::InvalidArgument(_) => {
                    axum::http::StatusCode::BAD_REQUEST
                }
                turbomemory_storage::StorageError::NotFound(_) => axum::http::StatusCode::NOT_FOUND,
                _ => axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            },
        };
        (status, self.to_string()).into_response()
    }
}

/// Shared memory service.
#[derive(Clone)]
pub struct MemoryService {
    engine: Arc<StorageEngine>,
}

impl MemoryService {
    pub fn open(db_path: impl AsRef<Path>, dimension: usize) -> Result<Self, ApiError> {
        let mut config = StoreConfig::default_for_dimension(dimension);
        config.max_edges = 16;
        config.search_list_size = 100;
        config.auto_consolidation_interval = None;
        let engine = StorageEngine::open(db_path, config)?;
        Ok(Self { engine })
    }

    pub fn engine(&self) -> &Arc<StorageEngine> {
        &self.engine
    }
}

/// Generic scored memory used by both REST and gRPC frontends.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ScoredMemory {
    pub id: String,
    pub score: f32,
}

pub fn map_results(results: Vec<(String, f32)>) -> Vec<ScoredMemory> {
    results
        .into_iter()
        .map(|(id, score)| ScoredMemory { id, score })
        .collect()
}

pub fn to_pb_results(results: Vec<ScoredMemory>) -> Vec<super::pb::ScoredMemory> {
    results
        .into_iter()
        .map(|sm| super::pb::ScoredMemory {
            id: sm.id,
            score: sm.score,
            payload: None,
        })
        .collect()
}

/// Convert a gRPC `Filter` into the storage-engine filter AST.
pub fn pb_filter_to_storage(
    filter: Option<&super::pb::Filter>,
) -> Result<Option<StorageFilter>, ApiError> {
    let Some(f) = filter else { return Ok(None) };
    Ok(Some(convert_pb_filter(f)?))
}

fn convert_pb_filter(filter: &super::pb::Filter) -> Result<StorageFilter, ApiError> {
    use super::pb::filter::Kind;
    let kind = filter
        .kind
        .as_ref()
        .ok_or_else(|| ApiError::InvalidFilter("empty filter".into()))?;
    Ok(match kind {
        Kind::Eq(eq) => {
            let value: serde_json::Value = serde_json::from_str(&eq.value_json)
                .map_err(|e| ApiError::InvalidFilter(format!("eq value_json: {e}")))?;
            StorageFilter::Eq {
                field: eq.field.clone(),
                value,
            }
        }
        Kind::Range(r) => {
            let low = r.low.map(|v| {
                if r.low_inclusive {
                    Bound::Included(v)
                } else {
                    Bound::Excluded(v)
                }
            });
            let high = r.high.map(|v| {
                if r.high_inclusive {
                    Bound::Included(v)
                } else {
                    Bound::Excluded(v)
                }
            });
            StorageFilter::Range {
                field: r.field.clone(),
                low: low.unwrap_or(Bound::Unbounded),
                high: high.unwrap_or(Bound::Unbounded),
            }
        }
        Kind::FullText(ft) => StorageFilter::FullText {
            field: ft.field.clone(),
            query: ft.query.clone(),
        },
        Kind::And(list) => StorageFilter::And(
            list.filters
                .iter()
                .map(convert_pb_filter)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Kind::Or(list) => StorageFilter::Or(
            list.filters
                .iter()
                .map(convert_pb_filter)
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Kind::Not(inner) => StorageFilter::Not(Box::new(convert_pb_filter(inner)?)),
    })
}

/// Convert a JSON filter object into the storage-engine filter AST.
///
/// Supported shapes:
/// - `{ "field": "<name>", "op": "eq", "value": <json> }`
/// - `{ "field": "<name>", "op": "range", "low": <number>, "high": <number>, "low_inclusive": bool, "high_inclusive": bool }`
/// - `{ "op": "full_text", "field": "<name>", "query": "<text>" }`
/// - `{ "op": "and" | "or", "filters": [ ... ] }`
/// - `{ "op": "not", "filter": { ... } }`
pub fn json_filter_to_storage(value: serde_json::Value) -> Result<Option<StorageFilter>, ApiError> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(convert_json_filter(value)?))
}

fn convert_json_filter(value: serde_json::Value) -> Result<StorageFilter, ApiError> {
    let obj = value
        .as_object()
        .ok_or_else(|| ApiError::InvalidFilter("filter must be a JSON object".into()))?;
    let op = obj
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or_else(|| ApiError::InvalidFilter("filter missing op".into()))?;
    Ok(match op {
        "eq" => {
            let field = obj
                .get("field")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ApiError::InvalidFilter("eq missing field".into()))?;
            let val = obj.get("value").cloned().unwrap_or(serde_json::Value::Null);
            StorageFilter::Eq {
                field: field.into(),
                value: val,
            }
        }
        "range" => {
            let field = obj
                .get("field")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ApiError::InvalidFilter("range missing field".into()))?;
            let low = obj.get("low").and_then(|v| v.as_f64());
            let high = obj.get("high").and_then(|v| v.as_f64());
            let low_inclusive = obj
                .get("low_inclusive")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let high_inclusive = obj
                .get("high_inclusive")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            StorageFilter::Range {
                field: field.into(),
                low: low
                    .map(|v| {
                        if low_inclusive {
                            Bound::Included(v)
                        } else {
                            Bound::Excluded(v)
                        }
                    })
                    .unwrap_or(Bound::Unbounded),
                high: high
                    .map(|v| {
                        if high_inclusive {
                            Bound::Included(v)
                        } else {
                            Bound::Excluded(v)
                        }
                    })
                    .unwrap_or(Bound::Unbounded),
            }
        }
        "full_text" => {
            let field = obj
                .get("field")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ApiError::InvalidFilter("full_text missing field".into()))?;
            let query = obj
                .get("query")
                .and_then(|v| v.as_str())
                .ok_or_else(|| ApiError::InvalidFilter("full_text missing query".into()))?;
            StorageFilter::FullText {
                field: field.into(),
                query: query.into(),
            }
        }
        "and" => {
            let arr = obj
                .get("filters")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ApiError::InvalidFilter("and missing filters".into()))?;
            StorageFilter::And(
                arr.iter()
                    .cloned()
                    .map(convert_json_filter)
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        "or" => {
            let arr = obj
                .get("filters")
                .and_then(|v| v.as_array())
                .ok_or_else(|| ApiError::InvalidFilter("or missing filters".into()))?;
            StorageFilter::Or(
                arr.iter()
                    .cloned()
                    .map(convert_json_filter)
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        "not" => {
            let inner = obj
                .get("filter")
                .cloned()
                .ok_or_else(|| ApiError::InvalidFilter("not missing filter".into()))?;
            StorageFilter::Not(Box::new(convert_json_filter(inner)?))
        }
        other => return Err(ApiError::InvalidFilter(format!("unknown op: {other}"))),
    })
}
