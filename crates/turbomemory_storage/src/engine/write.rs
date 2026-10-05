//! The write path: insert, batch insert, delete, and update.
//!
//! Every write holds the flush barrier across `seq allocate -> vectors.put ->
//! wal.append -> index apply`, preserving the durability order documented on
//! the engine module.

use super::{now_secs, StorageEngine};
use crate::record::{MetaRecord, PointOffset, Record};
use crate::wal::WalOp;
use crate::StorageError;
use std::collections::HashSet;
use std::sync::Arc;
use turbomemory_core::{normalize, validate_dimension};
use turbomemory_graph::merge_concepts_with_config;

impl StorageEngine {
    pub fn insert(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
    ) -> crate::Result<bool> {
        self.insert_with_payload(id, text, embedding, importance, concepts, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_with_payload(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
        scope: Option<String>,
    ) -> crate::Result<bool> {
        self.insert_with_payload_role(
            id, text, embedding, importance, concepts, payload, scope, None,
        )
    }

    /// Insert with an explicit provenance `source_role` (`"user"`,
    /// `"assistant"`, ...). The role never affects retrieval; it only gates
    /// belief-revision detection when `TierConfig::belief_source_roles` is set.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_with_payload_role(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
        scope: Option<String>,
        source_role: Option<String>,
    ) -> crate::Result<bool> {
        // Held across seq allocation → vectors.put → wal.append → meta apply
        // so a concurrent flush cannot truncate this record's WAL entry
        // before it reaches the redb snapshot.
        let _flush_guard = self.flush_barrier.read();
        validate_dimension(embedding, self.config.dimension)?;
        if !importance.is_finite() {
            return Err(StorageError::InvalidArgument(
                "importance_score must be finite".into(),
            ));
        }
        if self.id_index.read().contains_key(id) {
            return Err(StorageError::DuplicateId(id.to_string()));
        }
        let mut emb = embedding.to_vec();
        normalize(&mut emb)?;
        // Augment caller-supplied concepts with auto-extracted ones from the
        // text. If the caller already provided >= max_concepts, their tags
        // are used as-is. If max_concepts is 0, auto-extraction is disabled.
        let extractor_config = self.config.tier.extractor_config();
        let vocab = self.graph.read().vocab().clone();
        let concepts = merge_concepts_with_config(concepts, text, &extractor_config, Some(&vocab));
        let offset = self.meta.allocate_offset();
        let seq = self.meta.allocate_seq();
        let record = Record {
            id: id.to_string(),
            text: text.to_string(),
            embedding: Arc::from(emb),
            importance,
            concepts,
            created_at: now_secs(),
            insert_seq: seq,
            access_count: 0,
            last_accessed: 0,
            tier: crate::config::Tier::Hot,
            payload,
            scope,
            source_role,
        };

        // 1. Persist the embedding to the mmap-backed vector store first.
        //    The vector store is the durable physical source of truth for
        //    embeddings; the WAL only records the metadata operation.
        self.vectors.put(offset, record.embedding_f32())?;

        // 2. WAL metadata entry.
        {
            let meta = MetaRecord::from(&record);
            let mut wal = self.wal.lock();
            wal.append(&WalOp::Insert { offset, seq, meta })?;
        }

        // 3. Submit the index update to the serialized worker.
        self.update_worker.submit_and_wait(vec![(offset, record)])?;
        Ok(true)
    }

    pub fn insert_batch(
        &self,
        ids: &[String],
        texts: &[String],
        embeddings: &[Vec<f32>],
        importances: &[f32],
        concepts: &[Vec<String>],
    ) -> crate::Result<usize> {
        let refs: Vec<&[f32]> = embeddings.iter().map(|v| v.as_slice()).collect();
        self.insert_batch_with_payload(ids, texts, &refs, importances, concepts, &[], &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_batch_with_payload(
        &self,
        ids: &[String],
        texts: &[String],
        embeddings: &[&[f32]],
        importances: &[f32],
        concepts: &[Vec<String>],
        payloads: &[Option<String>],
        scopes: &[Option<String>],
    ) -> crate::Result<usize> {
        self.insert_batch_with_payload_role(
            ids,
            texts,
            embeddings,
            importances,
            concepts,
            payloads,
            scopes,
            &[],
        )
    }

    /// Batch insert with per-record provenance `source_roles` (parallel to
    /// `ids`). An empty slice means every record is unattributed. See
    /// `insert_with_payload_role`.
    #[allow(clippy::too_many_arguments)]
    pub fn insert_batch_with_payload_role(
        &self,
        ids: &[String],
        texts: &[String],
        embeddings: &[&[f32]],
        importances: &[f32],
        concepts: &[Vec<String>],
        payloads: &[Option<String>],
        scopes: &[Option<String>],
        source_roles: &[Option<String>],
    ) -> crate::Result<usize> {
        // See `insert_with_payload_role`: hold off flush for the whole
        // allocate → put → append → apply sequence.
        let _flush_guard = self.flush_barrier.read();
        let n = ids.len();
        if n == 0 {
            return Ok(0);
        }
        if texts.len() < n
            || embeddings.len() < n
            || importances.len() < n
            || concepts.len() < n
            || (!payloads.is_empty() && payloads.len() < n)
            || (!scopes.is_empty() && scopes.len() < n)
            || (!source_roles.is_empty() && source_roles.len() < n)
        {
            return Err(StorageError::InvalidArgument(
                "batch arrays have mismatched lengths".into(),
            ));
        }
        for &emb in embeddings {
            validate_dimension(emb, self.config.dimension)?;
        }
        if importances.iter().any(|i| !i.is_finite()) {
            return Err(StorageError::InvalidArgument(
                "importance scores must be finite".into(),
            ));
        }

        // Idempotent batch insert: skip existing ids and duplicate ids within the
        // batch.  This makes the operation safe to replay after a partial write.
        let idx = self.id_index.read();
        let mut seen = HashSet::with_capacity(n);
        let mut indices: Vec<usize> = Vec::with_capacity(n);
        for (i, raw_id) in ids.iter().enumerate().take(n) {
            let id = raw_id.as_str();
            if idx.contains_key(id) || !seen.insert(id) {
                continue;
            }
            indices.push(i);
        }
        drop(idx);

        // Snapshot the current vocabulary so all records in the batch are
        // canonicalized consistently (even if another thread evolves the
        // vocabulary while this batch is being prepared).
        let vocab = self.graph.read().vocab().clone();

        let mut records: Vec<(PointOffset, Record)> = Vec::with_capacity(indices.len());
        for &i in &indices {
            let mut emb = embeddings[i].to_vec();
            normalize(&mut emb)?;
            let offset = self.meta.allocate_offset();
            let seq = self.meta.allocate_seq();
            let payload = if payloads.is_empty() {
                None
            } else {
                payloads[i].clone()
            };
            let scope = if scopes.is_empty() {
                None
            } else {
                scopes[i].clone()
            };
            let source_role = if source_roles.is_empty() {
                None
            } else {
                source_roles[i].clone()
            };
            // Augment caller-supplied concepts with auto-extracted ones.
            let extractor_config = self.config.tier.extractor_config();
            let concepts = merge_concepts_with_config(
                &concepts[i],
                &texts[i],
                &extractor_config,
                Some(&vocab),
            );
            let record = Record {
                id: ids[i].clone(),
                text: texts[i].clone(),
                embedding: Arc::from(emb),
                importance: importances[i],
                concepts,
                created_at: now_secs(),
                insert_seq: seq,
                access_count: 0,
                last_accessed: 0,
                tier: crate::config::Tier::Hot,
                payload,
                scope,
                source_role,
            };
            records.push((offset, record));
        }

        // 1. Persist embeddings to the mmap-backed vector store first.
        for (offset, record) in &records {
            self.vectors.put(*offset, record.embedding_f32())?;
        }

        // 2. WAL metadata entries (batched under a single lock).
        {
            let mut wal = self.wal.lock();
            let ops: Vec<WalOp> = records
                .iter()
                .map(|(offset, record)| {
                    let meta = MetaRecord::from(record);
                    WalOp::Insert {
                        offset: *offset,
                        seq: record.insert_seq,
                        meta,
                    }
                })
                .collect();
            wal.append_batch(&ops)?;
        }

        // 3. Submit the index updates to the serialized worker.
        self.update_worker.submit_and_wait(records)?;
        Ok(indices.len())
    }

    /// Delete the record with the given id.
    ///
    /// The embedding is left in place in the vector store; the offset becomes
    /// unreachable because the metadata entry and id index are removed.  Segment
    /// searches already filter out offsets with no metadata, so deleted points
    /// disappear from results immediately.  Physical reclamation is deferred to
    /// the vacuum optimizer.
    pub fn delete_by_id(&self, id: &str) -> crate::Result<bool> {
        // Same flush barrier as inserts: the WAL delete entry and the
        // metadata removal must not straddle a flush's snapshot + truncate.
        let _flush_guard = self.flush_barrier.read();
        let offset = {
            let idx = self.id_index.read();
            match idx.get(id).copied() {
                Some(o) => o,
                None => return Ok(false),
            }
        };

        // 1. WAL delete entry.
        {
            let mut wal = self.wal.lock();
            wal.append(&WalOp::Delete { offset })?;
        }

        // 2. Remove from payload, text, and scope indexes while we still know
        //    the old values.
        if let Ok(Some(meta_rec)) = self.meta.get(offset) {
            self.payload_index
                .write()
                .remove(offset, meta_rec.payload.as_deref());
            self.scope_index
                .write()
                .remove(offset, meta_rec.scope.as_deref());
            self.text_index.remove(offset)?;
        }

        // 3. Remove from in-memory metadata and id index.
        self.meta.remove(offset)?;
        self.id_index.write().remove(id);

        // 4. Remove from cognitive graph.
        {
            let mut graph = self.graph.write();
            graph.remove_memory(id);
        }

        Ok(true)
    }

    /// Replace an existing record, preserving its id.
    ///
    /// Implemented as an atomic-in-metadata delete + insert: the old offset is
    /// tombstoned and a new offset is allocated.  This keeps HNSW segments
    /// correct without requiring in-place vector updates inside immutable
    /// indexes.
    pub fn update(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
    ) -> crate::Result<bool> {
        self.update_with_payload(id, text, embedding, importance, concepts, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn update_with_payload(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
        scope: Option<String>,
    ) -> crate::Result<bool> {
        self.update_with_payload_role(
            id, text, embedding, importance, concepts, payload, scope, None,
        )
    }

    /// Update with an explicit provenance `source_role`. See
    /// `insert_with_payload_role`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_with_payload_role(
        &self,
        id: &str,
        text: &str,
        embedding: &[f32],
        importance: f32,
        concepts: &[String],
        payload: Option<String>,
        scope: Option<String>,
        source_role: Option<String>,
    ) -> crate::Result<bool> {
        if !self.id_index.read().contains_key(id) {
            return Ok(false);
        }
        self.delete_by_id(id)?;
        self.insert_with_payload_role(
            id,
            text,
            embedding,
            importance,
            concepts,
            payload,
            scope,
            source_role,
        )?;
        Ok(true)
    }
}
