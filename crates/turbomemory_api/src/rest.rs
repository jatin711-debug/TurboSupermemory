//! REST service implementation (Axum).

use crate::service::{json_filter_to_storage, map_results, ApiError, MemoryService};
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
    #[serde(default)]
    payload: Option<String>,
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
}

#[derive(Deserialize)]
struct SearchAnnReq {
    query_embedding: Vec<f32>,
    top_k: usize,
    #[serde(default)]
    filter: serde_json::Value,
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
    let success = service.engine().insert_with_payload(
        &req.id,
        &req.text,
        &req.embedding,
        req.importance,
        &req.concepts,
        req.payload,
    )?;
    Ok(Json(InsertResp { success }))
}

async fn insert_batch(
    State(service): State<MemoryService>,
    Json(req): Json<InsertBatchReq>,
) -> Result<Json<InsertBatchResp>, ApiError> {
    let payloads: Vec<Option<String>> = if req.payloads.is_empty() {
        Vec::new()
    } else {
        req.payloads.into_iter().map(Some).collect()
    };
    let count = service.engine().insert_batch_with_payload(
        &req.ids,
        &req.texts,
        &req.embeddings,
        &req.scores,
        &req.concepts,
        &payloads,
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
    let success = service.engine().update_with_payload(
        &req.id,
        &req.text,
        &req.embedding,
        req.importance,
        &req.concepts,
        req.payload,
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
    let maybe = match filter {
        Some(f) => service
            .engine()
            .search_filtered(&req.query_text, &req.query_embedding, req.top_k, &f)?,
        None => service
            .engine()
            .search(&req.query_text, &req.query_embedding, req.top_k)?,
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
    let results = match filter {
        Some(f) => service
            .engine()
            .search_ann_filtered(&req.query_embedding, req.top_k, &f)?,
        None => service.engine().search_ann(&req.query_embedding, req.top_k)?,
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

pub fn router(service: MemoryService) -> Router {
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
        .with_state(service)
}
