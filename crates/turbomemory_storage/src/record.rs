//! Durable memory record and point-offset types.

use crate::config::Tier;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Stable dense offset used inside vector segments.
pub type PointOffset = u64;

/// A durable memory record as seen by the public API.
///
/// The embedding is stored as an `Arc<[f32]>` so it can be shared between the
/// WAL, the Hot HNSW index, and callers without copying.
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
    pub last_accessed: u64,
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

/// Metadata-only view of a record.
///
/// The embedding lives in the separate `VectorStore`; keeping it out of the
/// metadata cache removes the biggest in-memory duplicate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaRecord {
    pub id: String,
    pub text: String,
    pub importance: f32,
    pub concepts: Vec<String>,
    pub created_at: u64,
    pub insert_seq: u64,
    pub access_count: u64,
    pub last_accessed: u64,
    pub tier: Tier,
}

impl MetaRecord {
    /// Hydrate a full `Record` by attaching an embedding.
    pub fn with_embedding(self, embedding: Arc<[f32]>) -> Record {
        Record {
            id: self.id,
            text: self.text,
            embedding,
            importance: self.importance,
            concepts: self.concepts,
            created_at: self.created_at,
            insert_seq: self.insert_seq,
            access_count: self.access_count,
            last_accessed: self.last_accessed,
            tier: self.tier,
        }
    }
}

impl From<&Record> for MetaRecord {
    fn from(rec: &Record) -> Self {
        Self {
            id: rec.id.clone(),
            text: rec.text.clone(),
            importance: rec.importance,
            concepts: rec.concepts.clone(),
            created_at: rec.created_at,
            insert_seq: rec.insert_seq,
            access_count: rec.access_count,
            last_accessed: rec.last_accessed,
            tier: rec.tier,
        }
    }
}
