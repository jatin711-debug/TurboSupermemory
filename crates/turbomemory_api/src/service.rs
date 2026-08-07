//! Shared service layer used by both the gRPC and REST frontends.

use axum::response::IntoResponse;
use std::ops::Bound;
use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use turbomemory_storage::config::StoreConfig;
use turbomemory_storage::engine::StorageEngine;
use turbomemory_storage::payload_index::Filter as StorageFilter;

/// Maximum nesting depth accepted by the JSON filter DSL (`and`/`or`/`not`).
pub const MAX_FILTER_DEPTH: usize = 32;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("storage error: {0}")]
    Storage(#[from] turbomemory_storage::StorageError),
    #[error("missing field: {0}")]
    MissingField(String),
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("invalid filter: {0}")]
    InvalidFilter(String),
}

impl ApiError {
    /// HTTP status code and machine-readable error code for the REST frontend.
    pub fn status_and_code(&self) -> (axum::http::StatusCode, &'static str) {
        match self {
            ApiError::MissingField(_)
            | ApiError::InvalidArgument(_)
            | ApiError::InvalidFilter(_) => {
                (axum::http::StatusCode::BAD_REQUEST, "invalid_argument")
            }
            ApiError::Storage(se) if is_invalid_argument(se) => {
                (axum::http::StatusCode::BAD_REQUEST, "invalid_argument")
            }
            ApiError::Storage(turbomemory_storage::StorageError::NotFound(_)) => {
                (axum::http::StatusCode::NOT_FOUND, "not_found")
            }
            ApiError::Storage(_) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, "internal"),
        }
    }
}

/// True for storage errors caused by the caller, mapping to HTTP 400 /
/// gRPC InvalidArgument. This includes dimension mismatches and invalid
/// arguments reported by the core crate (`StorageError::Core`).
fn is_invalid_argument(se: &turbomemory_storage::StorageError) -> bool {
    use turbomemory_storage::StorageError as SE;
    match se {
        SE::DuplicateId(_) | SE::DimensionMismatch | SE::InvalidArgument(_) => true,
        SE::Core(ce) => matches!(
            ce,
            turbomemory_core::TurboError::DimensionMismatch { .. }
                | turbomemory_core::TurboError::InvalidArgument(_)
        ),
        _ => false,
    }
}

impl From<ApiError> for tonic::Status {
    fn from(e: ApiError) -> Self {
        match e {
            ApiError::MissingField(_)
            | ApiError::InvalidArgument(_)
            | ApiError::InvalidFilter(_) => tonic::Status::invalid_argument(e.to_string()),
            ApiError::Storage(ref se) if is_invalid_argument(se) => {
                tonic::Status::invalid_argument(e.to_string())
            }
            ApiError::Storage(turbomemory_storage::StorageError::NotFound(_)) => {
                tonic::Status::not_found(e.to_string())
            }
            ApiError::Storage(_) => tonic::Status::internal(e.to_string()),
        }
    }
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, code) = self.status_and_code();
        error_response(status, code, self.to_string())
    }
}

/// JSON error body returned by the REST frontend: `{"error":{"code","message"}}`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorBody {
    pub error: ErrorDetail,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ErrorDetail {
    pub code: String,
    pub message: String,
}

/// Build a JSON error response for the REST frontend.
pub fn error_response(
    status: axum::http::StatusCode,
    code: &str,
    message: String,
) -> axum::response::Response {
    let body = ErrorBody {
        error: ErrorDetail {
            code: code.into(),
            message,
        },
    };
    (status, axum::Json(body)).into_response()
}

/// 401 response used by the REST auth middleware.
pub fn unauthenticated_response() -> axum::response::Response {
    let mut response = error_response(
        axum::http::StatusCode::UNAUTHORIZED,
        "unauthenticated",
        "missing or invalid bearer token".into(),
    );
    response.headers_mut().insert(
        axum::http::header::WWW_AUTHENTICATE,
        axum::http::HeaderValue::from_static("Bearer"),
    );
    response
}

/// Optional bearer-token authentication shared by both transports.
///
/// Enabled by setting the `TURBO_API_KEY` environment variable; when disabled
/// the server accepts every request without a token.
#[derive(Clone, Debug)]
pub enum ApiAuth {
    Disabled,
    Required(Arc<str>),
}

impl ApiAuth {
    pub fn new(api_key: Option<String>) -> Self {
        match api_key {
            Some(key) if !key.is_empty() => Self::Required(key.into()),
            _ => Self::Disabled,
        }
    }

    /// Read the key from the `TURBO_API_KEY` environment variable.
    pub fn from_env() -> Self {
        Self::new(std::env::var("TURBO_API_KEY").ok())
    }

    pub fn is_required(&self) -> bool {
        matches!(self, Self::Required(_))
    }

    /// Check an `Authorization` header/metadata value against the configured
    /// key. Always returns `true` when auth is disabled.
    pub fn is_authorized(&self, header: Option<&str>) -> bool {
        match self {
            Self::Disabled => true,
            Self::Required(key) => header
                .and_then(|h| h.strip_prefix("Bearer "))
                .is_some_and(|token| token == key.as_ref()),
        }
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

/// Validate the parallel arrays of a batch insert.
///
/// `texts`, `embeddings`, `scores`, and `concepts` are required parallel
/// arrays and must match the number of `ids` exactly; `payloads`, `scopes`,
/// and `source_roles` are optional and must be either empty (absent) or match
/// the batch length.
#[allow(clippy::too_many_arguments)]
pub fn validate_batch_lengths(
    ids: usize,
    texts: usize,
    embeddings: usize,
    scores: usize,
    concepts: usize,
    payloads: usize,
    scopes: usize,
    source_roles: usize,
) -> Result<(), ApiError> {
    let mut mismatched = Vec::new();
    for (name, len) in [
        ("texts", texts),
        ("embeddings", embeddings),
        ("scores", scores),
        ("concepts", concepts),
    ] {
        if len != ids {
            mismatched.push(format!("{name} has {len}"));
        }
    }
    for (name, len) in [
        ("payloads", payloads),
        ("scopes", scopes),
        ("source_roles", source_roles),
    ] {
        if len != 0 && len != ids {
            mismatched.push(format!("{name} has {len}"));
        }
    }
    if mismatched.is_empty() {
        Ok(())
    } else {
        Err(ApiError::InvalidArgument(format!(
            "batch arrays must match the number of ids ({ids}): {}",
            mismatched.join(", ")
        )))
    }
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
///
/// Nested filters are rejected beyond [`MAX_FILTER_DEPTH`] levels.
pub fn json_filter_to_storage(value: serde_json::Value) -> Result<Option<StorageFilter>, ApiError> {
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(convert_json_filter(value, 0)?))
}

fn convert_json_filter(value: serde_json::Value, depth: usize) -> Result<StorageFilter, ApiError> {
    if depth >= MAX_FILTER_DEPTH {
        return Err(ApiError::InvalidFilter(format!(
            "filter nesting exceeds the maximum depth of {MAX_FILTER_DEPTH}"
        )));
    }
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
                    .map(|v| convert_json_filter(v, depth + 1))
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
                    .map(|v| convert_json_filter(v, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        "not" => {
            let inner = obj
                .get("filter")
                .cloned()
                .ok_or_else(|| ApiError::InvalidFilter("not missing filter".into()))?;
            StorageFilter::Not(Box::new(convert_json_filter(inner, depth + 1)?))
        }
        other => return Err(ApiError::InvalidFilter(format!("unknown op: {other}"))),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use turbomemory_storage::StorageError;

    fn nested_not_filter(depth: usize) -> serde_json::Value {
        let mut value = serde_json::json!({"op": "eq", "field": "f", "value": 1});
        for _ in 0..depth {
            value = serde_json::json!({"op": "not", "filter": value});
        }
        value
    }

    #[test]
    fn json_filter_at_depth_limit_parses() {
        let value = nested_not_filter(MAX_FILTER_DEPTH - 1);
        let filter = json_filter_to_storage(value).unwrap();
        assert!(filter.is_some());
    }

    #[test]
    fn json_filter_beyond_depth_limit_rejected() {
        let value = nested_not_filter(MAX_FILTER_DEPTH);
        let err = json_filter_to_storage(value).unwrap_err();
        assert!(matches!(err, ApiError::InvalidFilter(_)));
        assert!(err.to_string().contains("depth"));
    }

    #[test]
    fn json_filter_null_is_none() {
        assert!(json_filter_to_storage(serde_json::Value::Null)
            .unwrap()
            .is_none());
    }

    #[test]
    fn batch_lengths_valid() {
        assert!(validate_batch_lengths(2, 2, 2, 2, 2, 0, 0, 0).is_ok());
        assert!(validate_batch_lengths(2, 2, 2, 2, 2, 2, 2, 2).is_ok());
        assert!(validate_batch_lengths(0, 0, 0, 0, 0, 0, 0, 0).is_ok());
    }

    #[test]
    fn batch_lengths_mismatch_rejected() {
        for args in [
            (2, 3, 2, 2, 2, 0, 0, 0), // texts too long
            (2, 2, 1, 2, 2, 0, 0, 0), // embeddings too short
            (2, 2, 2, 0, 2, 0, 0, 0), // scores missing
            (2, 2, 2, 2, 3, 0, 0, 0), // concepts too long
            (2, 2, 2, 2, 2, 1, 0, 0), // optional payloads wrong length
            (2, 2, 2, 2, 2, 0, 3, 0), // optional scopes wrong length
            (2, 2, 2, 2, 2, 0, 0, 1), // optional source_roles wrong length
        ] {
            let err = validate_batch_lengths(
                args.0, args.1, args.2, args.3, args.4, args.5, args.6, args.7,
            )
            .unwrap_err();
            assert!(
                matches!(err, ApiError::InvalidArgument(_)),
                "expected InvalidArgument for {args:?}"
            );
        }
    }

    #[test]
    fn api_error_http_status_mapping() {
        let cases: Vec<(ApiError, StatusCode, &str)> = vec![
            (
                ApiError::MissingField("id".into()),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::InvalidArgument("bad batch".into()),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::InvalidFilter("bad filter".into()),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::Storage(StorageError::DuplicateId("a".into())),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::Storage(StorageError::DimensionMismatch),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::Storage(StorageError::InvalidArgument("x".into())),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::Storage(StorageError::Core(
                    turbomemory_core::TurboError::DimensionMismatch {
                        expected: 4,
                        got: 3,
                    },
                )),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::Storage(StorageError::Core(
                    turbomemory_core::TurboError::InvalidArgument("x".into()),
                )),
                StatusCode::BAD_REQUEST,
                "invalid_argument",
            ),
            (
                ApiError::Storage(StorageError::Core(turbomemory_core::TurboError::ZeroNorm)),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
            ),
            (
                ApiError::Storage(StorageError::NotFound("a".into())),
                StatusCode::NOT_FOUND,
                "not_found",
            ),
            (
                ApiError::Storage(StorageError::IndexError("x".into())),
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal",
            ),
        ];
        for (err, status, code) in cases {
            assert_eq!(err.status_and_code(), (status, code), "for {err}");
        }
    }

    #[test]
    fn api_error_grpc_status_mapping() {
        let status: tonic::Status = ApiError::InvalidFilter("bad".into()).into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        let status: tonic::Status = ApiError::Storage(StorageError::NotFound("a".into())).into();
        assert_eq!(status.code(), tonic::Code::NotFound);
        let status: tonic::Status = ApiError::Storage(StorageError::DimensionMismatch).into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        let status: tonic::Status = ApiError::Storage(StorageError::Core(
            turbomemory_core::TurboError::DimensionMismatch {
                expected: 4,
                got: 3,
            },
        ))
        .into();
        assert_eq!(status.code(), tonic::Code::InvalidArgument);
        let status: tonic::Status = ApiError::Storage(StorageError::IndexError("x".into())).into();
        assert_eq!(status.code(), tonic::Code::Internal);
    }

    #[tokio::test]
    async fn api_error_response_is_json() {
        let response = ApiError::Storage(StorageError::NotFound("abc".into())).into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let content_type = response
            .headers()
            .get(axum::http::header::CONTENT_TYPE)
            .unwrap();
        assert_eq!(content_type, "application/json");
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: ErrorBody = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body.error.code, "not_found");
        assert!(body.error.message.contains("abc"));
    }

    #[tokio::test]
    async fn unauthenticated_response_is_401_json() {
        let response = unauthenticated_response();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get(axum::http::header::WWW_AUTHENTICATE),
            Some(&axum::http::HeaderValue::from_static("Bearer"))
        );
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let body: ErrorBody = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body.error.code, "unauthenticated");
    }

    #[test]
    fn api_auth_disabled_accepts_everything() {
        let auth = ApiAuth::new(None);
        assert!(!auth.is_required());
        assert!(auth.is_authorized(None));
        assert!(auth.is_authorized(Some("Bearer anything")));
    }

    #[test]
    fn api_auth_empty_key_is_disabled() {
        assert!(!ApiAuth::new(Some(String::new())).is_required());
    }

    #[test]
    fn api_auth_required_checks_bearer_token() {
        let auth = ApiAuth::new(Some("s3cret".into()));
        assert!(auth.is_required());
        assert!(auth.is_authorized(Some("Bearer s3cret")));
        assert!(!auth.is_authorized(None));
        assert!(!auth.is_authorized(Some("Bearer wrong")));
        assert!(!auth.is_authorized(Some("Bearer s3cret extra")));
        assert!(!auth.is_authorized(Some("s3cret")));
        assert!(!auth.is_authorized(Some("bearer s3cret")));
    }
}
