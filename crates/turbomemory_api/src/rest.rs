//! REST service implementation (Axum).

use crate::service::{map_results, ApiError, MemoryService};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};

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
}

#[derive(Serialize)]
struct InsertBatchResp {
    count: usize,
}

#[derive(Deserialize)]
struct SearchReq {
    query_text: String,
    query_embedding: Vec<f32>,
    top_k: usize,
}

#[derive(Deserialize)]
struct SearchAnnReq {
    query_embedding: Vec<f32>,
    top_k: usize,
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
    let success = service.engine().insert(
        &req.id,
        &req.text,
        &req.embedding,
        req.importance,
        &req.concepts,
    )?;
    Ok(Json(InsertResp { success }))
}

async fn insert_batch(
    State(service): State<MemoryService>,
    Json(req): Json<InsertBatchReq>,
) -> Result<Json<InsertBatchResp>, ApiError> {
    let count = service.engine().insert_batch(
        &req.ids,
        &req.texts,
        &req.embeddings,
        &req.scores,
        &req.concepts,
    )?;
    Ok(Json(InsertBatchResp { count }))
}

async fn search(
    State(service): State<MemoryService>,
    Json(req): Json<SearchReq>,
) -> Result<Json<SearchResp>, ApiError> {
    let maybe = service
        .engine()
        .search(&req.query_text, &req.query_embedding, req.top_k)?;
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
    let results = service
        .engine()
        .search_ann(&req.query_embedding, req.top_k)?;
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
    let (sealed, compacted) = service.engine().trigger_consolidation()?;
    Ok(Json(ConsolidationResp { sealed, compacted }))
}

async fn flush(State(service): State<MemoryService>) -> Result<Json<()>, ApiError> {
    service.engine().flush()?;
    Ok(Json(()))
}

pub fn router(service: MemoryService) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/insert", post(insert))
        .route("/insert_batch", post(insert_batch))
        .route("/search", post(search))
        .route("/search_ann", post(search_ann))
        .route("/step_session", post(step_session))
        .route("/trigger_consolidation", post(trigger_consolidation))
        .route("/flush", post(flush))
        .with_state(service)
}
