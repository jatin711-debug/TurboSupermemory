//! Shared service layer used by both the gRPC and REST frontends.

use std::path::Path;
use std::sync::Arc;
use thiserror::Error;
use turbomemory_storage::config::{StoreConfig, TierConfig};
use turbomemory_storage::engine::StorageEngine;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("storage error: {0}")]
    Storage(#[from] turbomemory_storage::StorageError),
    #[error("missing field: {0}")]
    MissingField(String),
}

impl From<ApiError> for tonic::Status {
    fn from(e: ApiError) -> Self {
        tonic::Status::internal(e.to_string())
    }
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            self.to_string(),
        )
            .into_response()
    }
}

/// Shared memory service.
#[derive(Clone)]
pub struct MemoryService {
    engine: Arc<StorageEngine>,
}

impl MemoryService {
    pub fn open(db_path: impl AsRef<Path>, dimension: usize) -> Result<Self, ApiError> {
        let config = StoreConfig {
            dimension,
            max_edges: 16,
            search_list_size: 100,
            outlier_count: 0,
            initial_capacity: 1024,
            tier: TierConfig::default(),
            auto_consolidation_interval: None,
        };
        let engine = StorageEngine::open(db_path, config)?;
        Ok(Self {
            engine: Arc::new(engine),
        })
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
        })
        .collect()
}
