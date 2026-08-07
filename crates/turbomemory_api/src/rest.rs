//! REST service implementation (Axum).

use crate::service::{
    json_filter_to_storage, map_results, unauthenticated_response, validate_batch_lengths, ApiAuth,
    ApiError, MemoryService,
};
use axum::{
    extract::{Request, State},
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

/// Middleware enforcing `Authorization: Bearer <key>` on every route when auth
/// is enabled; passes every request when it is not.
async fn require_auth(State(auth): State<ApiAuth>, request: Request, next: Next) -> Response {
    let header = request
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok());
    if auth.is_authorized(header) {
        next.run(request).await
    } else {
        unauthenticated_response()
    }
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
}

#[derive(Deserialize)]
struct InsertReq {
    id: String,
    text: String,
    embedding: Vec<f32>,
    #[serde(default)]
    importance: f32,
    #[serde(default)]
    concepts: Vec<String>,
    #[serde(default)]
    payload: Option<String>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    source_role: Option<String>,
}

#[derive(Serialize)]
struct InsertResp {
    success: bool,
}

#[derive(Deserialize)]
struct InsertBatchReq {
    ids: Vec<String>,
    texts: Vec<String>,
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    scores: Vec<f32>,
    #[serde(default)]
    concepts: Vec<Vec<String>>,
    #[serde(default)]
    payloads: Vec<String>,
    #[serde(default)]
    scopes: Vec<String>,
    #[serde(default)]
    source_roles: Vec<String>,
}

#[derive(Serialize)]
struct InsertBatchResp {
    count: usize,
}

#[derive(Deserialize)]
struct DeleteReq {
    id: String,
}

#[derive(Serialize)]
struct DeleteResp {
    success: bool,
}

#[derive(Deserialize)]
struct UpdateReq {
    id: String,
    text: String,
    embedding: Vec<f32>,
    #[serde(default)]
    importance: f32,
    #[serde(default)]
    concepts: Vec<String>,
    #[serde(default)]
    payload: Option<String>,
    #[serde(default)]
    scope: Option<String>,
    #[serde(default)]
    source_role: Option<String>,
}

#[derive(Serialize)]
struct UpdateResp {
    success: bool,
}

#[derive(Deserialize)]
struct GetPayloadReq {
    id: String,
}

#[derive(Serialize)]
struct GetPayloadResp {
    found: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    payload: Option<String>,
}

#[derive(Deserialize)]
struct SearchReq {
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: usize,
    #[serde(default)]
    filter: serde_json::Value,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Deserialize)]
struct SearchAnnReq {
    query_embedding: Vec<f32>,
    top_k: usize,
    #[serde(default)]
    filter: serde_json::Value,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Serialize)]
struct SearchResp {
    results: Vec<crate::service::ScoredMemory>,
    gated: bool,
}

#[derive(Deserialize)]
struct StepSessionReq {
    user_input: String,
    assistant_response: String,
}

#[derive(Serialize)]
struct StepSessionResp {
    ccs_json: String,
}

#[derive(Serialize)]
struct ConsolidationResp {
    sealed: usize,
    compacted: usize,
}

async fn health(State(_service): State<MemoryService>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".into(),
    })
}

async fn insert(
    State(service): State<MemoryService>,
    Json(req): Json<InsertReq>,
) -> Result<Json<InsertResp>, ApiError> {
    let success = service.engine().insert_with_payload_role(
        &req.id,
        &req.text,
        &req.embedding,
        req.importance,
        &req.concepts,
        req.payload,
        req.scope,
        req.source_role,
    )?;
    Ok(Json(InsertResp { success }))
}

async fn insert_batch(
    State(service): State<MemoryService>,
    Json(req): Json<InsertBatchReq>,
) -> Result<Json<InsertBatchResp>, ApiError> {
    validate_batch_lengths(
        req.ids.len(),
        req.texts.len(),
        req.embeddings.len(),
        req.scores.len(),
        req.concepts.len(),
        req.payloads.len(),
        req.scopes.len(),
        req.source_roles.len(),
    )?;
    let payloads: Vec<Option<String>> = if req.payloads.is_empty() {
        Vec::new()
    } else {
        req.payloads.into_iter().map(Some).collect()
    };
    let scopes: Vec<Option<String>> = if req.scopes.is_empty() {
        Vec::new()
    } else {
        req.scopes.into_iter().map(Some).collect()
    };
    let source_roles: Vec<Option<String>> = if req.source_roles.is_empty() {
        Vec::new()
    } else {
        req.source_roles.into_iter().map(Some).collect()
    };
    let emb_refs: Vec<&[f32]> = req.embeddings.iter().map(|v| v.as_slice()).collect();
    let count = service.engine().insert_batch_with_payload_role(
        &req.ids,
        &req.texts,
        &emb_refs,
        &req.scores,
        &req.concepts,
        &payloads,
        &scopes,
        &source_roles,
    )?;
    Ok(Json(InsertBatchResp { count }))
}

async fn delete(
    State(service): State<MemoryService>,
    Json(req): Json<DeleteReq>,
) -> Result<Json<DeleteResp>, ApiError> {
    let success = service.engine().delete_by_id(&req.id)?;
    Ok(Json(DeleteResp { success }))
}

async fn update(
    State(service): State<MemoryService>,
    Json(req): Json<UpdateReq>,
) -> Result<Json<UpdateResp>, ApiError> {
    let success = service.engine().update_with_payload_role(
        &req.id,
        &req.text,
        &req.embedding,
        req.importance,
        &req.concepts,
        req.payload,
        req.scope,
        req.source_role,
    )?;
    Ok(Json(UpdateResp { success }))
}

async fn get_payload(
    State(service): State<MemoryService>,
    Json(req): Json<GetPayloadReq>,
) -> Result<Json<GetPayloadResp>, ApiError> {
    match service.engine().get_payload(&req.id)? {
        Some(payload) => Ok(Json(GetPayloadResp {
            found: true,
            payload: Some(payload),
        })),
        None => Ok(Json(GetPayloadResp {
            found: false,
            payload: None,
        })),
    }
}

async fn search(
    State(service): State<MemoryService>,
    Json(req): Json<SearchReq>,
) -> Result<Json<SearchResp>, ApiError> {
    let filter = json_filter_to_storage(req.filter)?;
    let scope = req.scope.as_deref();
    let maybe = match filter {
        Some(f) => service.engine().search_filtered_with_scope(
            &req.query_text,
            &req.query_embedding,
            req.top_k,
            &f,
            None,
            scope,
        )?,
        None => service.engine().search_scoped(
            &req.query_text,
            &req.query_embedding,
            req.top_k,
            scope,
        )?,
    };
    if let Some(results) = maybe {
        Ok(Json(SearchResp {
            results: map_results(results),
            gated: false,
        }))
    } else {
        Ok(Json(SearchResp {
            results: Vec::new(),
            gated: true,
        }))
    }
}

async fn search_ann(
    State(service): State<MemoryService>,
    Json(req): Json<SearchAnnReq>,
) -> Result<Json<SearchResp>, ApiError> {
    let filter = json_filter_to_storage(req.filter)?;
    let scope = req.scope.as_deref();
    let results = match filter {
        Some(f) => service.engine().search_ann_candidates_filtered_with_ef(
            &req.query_embedding,
            req.top_k,
            Some(&f),
            None,
            scope,
        )?,
        None => service
            .engine()
            .search_ann_scoped(&req.query_embedding, req.top_k, None, scope)?,
    };
    Ok(Json(SearchResp {
        results: map_results(results),
        gated: false,
    }))
}

async fn step_session(
    State(service): State<MemoryService>,
    Json(req): Json<StepSessionReq>,
) -> Result<Json<StepSessionResp>, ApiError> {
    let ccs_json = service
        .engine()
        .step_session(&req.user_input, &req.assistant_response)?;
    Ok(Json(StepSessionResp { ccs_json }))
}

async fn trigger_consolidation(
    State(service): State<MemoryService>,
) -> Result<Json<ConsolidationResp>, ApiError> {
    let (sealed, compacted, _promoted) = service.engine().trigger_consolidation()?;
    Ok(Json(ConsolidationResp { sealed, compacted }))
}

async fn flush(State(service): State<MemoryService>) -> Result<Json<()>, ApiError> {
    service.engine().flush()?;
    Ok(Json(()))
}

pub fn router(service: MemoryService, auth: ApiAuth) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/insert", post(insert))
        .route("/insert_batch", post(insert_batch))
        .route("/delete", post(delete))
        .route("/update", post(update))
        .route("/get_payload", post(get_payload))
        .route("/search", post(search))
        .route("/search_ann", post(search_ann))
        .route("/step_session", post(step_session))
        .route("/trigger_consolidation", post(trigger_consolidation))
        .route("/flush", post(flush))
        .layer(middleware::from_fn_with_state(auth, require_auth))
        .with_state(service)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{header, StatusCode};
    use tower::ServiceExt;

    fn test_service() -> (tempfile::TempDir, MemoryService) {
        let dir = tempfile::tempdir().unwrap();
        let service = MemoryService::open(dir.path(), 4).unwrap();
        (dir, service)
    }

    fn get_health() -> Request<Body> {
        Request::builder()
            .uri("/health")
            .body(Body::empty())
            .unwrap()
    }

    #[tokio::test]
    async fn rejects_missing_token_when_key_set() {
        let (_dir, service) = test_service();
        let app = router(service, ApiAuth::new(Some("key".into())));
        let response = app.oneshot(get_health()).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers().get(header::WWW_AUTHENTICATE).unwrap(),
            "Bearer"
        );
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "unauthenticated");
    }

    #[tokio::test]
    async fn rejects_wrong_token_when_key_set() {
        let (_dir, service) = test_service();
        let app = router(service, ApiAuth::new(Some("key".into())));
        let request = Request::builder()
            .uri("/health")
            .header(header::AUTHORIZATION, "Bearer nope")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn accepts_valid_token() {
        let (_dir, service) = test_service();
        let app = router(service, ApiAuth::new(Some("key".into())));
        let request = Request::builder()
            .uri("/health")
            .header(header::AUTHORIZATION, "Bearer key")
            .body(Body::empty())
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn open_when_no_key_set() {
        let (_dir, service) = test_service();
        let app = router(service, ApiAuth::new(None));
        let response = app.oneshot(get_health()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn storage_error_maps_to_json_body() {
        // Wrong embedding dimension -> 400 with a JSON error body.
        let (_dir, service) = test_service();
        let app = router(service, ApiAuth::new(None));
        let request = Request::builder()
            .method("POST")
            .uri("/insert")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"id":"a","text":"t","embedding":[1.0,2.0,3.0]}"#,
            ))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "invalid_argument");
    }

    #[tokio::test]
    async fn batch_length_mismatch_maps_to_json_400() {
        let (_dir, service) = test_service();
        let app = router(service, ApiAuth::new(None));
        let request = Request::builder()
            .method("POST")
            .uri("/insert_batch")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                r#"{"ids":["a","b"],"texts":["t"],"embeddings":[[1.0,2.0,3.0,4.0],[1.0,2.0,3.0,4.0]],"scores":[0.5,0.5],"concepts":[[],[]]}"#,
            ))
            .unwrap();
        let response = app.oneshot(request).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["code"], "invalid_argument");
        assert!(body["error"]["message"].as_str().unwrap().contains("texts"));
    }
}
