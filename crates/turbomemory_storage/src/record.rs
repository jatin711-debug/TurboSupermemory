//! Durable memory record and point-offset types.

use crate::config::Tier;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Stable dense offset used inside vector segments.
pub type PointOffset = u64;

/// A durable memory record.
///
/// The embedding is stored as an `Arc<[f32]>` so it can be shared between the
/// metadata store and the Hot HNSW index without copying.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Record {
    pub id: String,
    pub text: String,
    pub embedding: Arc<[f32]>,
    pub importance: f32,
    pub concepts: Vec<String>,
    pub created_at: u64,
    pub insert_seq: u64,
    pub access_count: u64,
    pub tier: Tier,
}

impl Record {
    pub fn embedding_f32(&self) -> &[f32] {
        &self.embedding
    }

    pub fn with_tier(self, tier: Tier) -> Self {
        Self { tier, ..self }
    }
}
