//! The write path: insert, batch insert, delete, and update.
//!
//! Every write validates its whole input first, so a rejected write changes
//! nothing. The commit section then runs under the engine's write lock and
//! holds the flush barrier across `id check -> seq allocate -> vectors.put ->
//! wal.append -> index apply`, preserving the durability order documented on
//! the engine module.

use super::{now_secs, StorageEngine};
use crate::record::{MetaRecord, PointOffset, Record};
use crate::vector_store::vector_crc;
use crate::wal::WalOp;
use crate::StorageError;
use std::collections::HashSet;
use std::sync::Arc;
use turbomemory_core::{normalize, validate_dimension};
use turbomemory_graph::merge_concepts_with_config;

/// One record's caller input, borrowed for validation.
struct WriteInput<'a> {
    id: &'a str,
    text: &'a str,
    embedding: &'a [f32],
    importance: f32,
    concepts: &'a [String],
    payload: Option<String>,
    scope: Option<String>,
    source_role: Option<String>,
}

/// A validated record that only lacks its offset and sequence number.
struct PreparedRecord {
    id: String,
    text: String,
    embedding: Arc<[f32]>,
    importance: f32,
    concepts: Vec<String>,
    payload: Option<String>,
    scope: Option<String>,
    source_role: Option<String>,
}

impl PreparedRecord {
    fn into_record(self, seq: u64) -> Record {
        Record {
            id: self.id,
            text: self.text,
            embedding: self.embedding,
            importance: self.importance,
            concepts: self.concepts,
            created_at: now_secs(),
            insert_seq: seq,
            access_count: 0,
            last_accessed: 0,
            tier: crate::config::Tier::Hot,
            payload: self.payload,
            scope: self.scope,
            source_role: self.source_role,
        }
    }
}

impl StorageEngine {
    /// Check everything about a record that can be wrong before any state is
    /// touched: id, dimension, finite non-zero embedding, finite importance,
    /// and payload JSON. Returns the record with its embedding normalized and
    /// its concepts merged.
    fn prepare(
        &self,
        input: WriteInput<'_>,
        vocab: &turbomemory_graph::ConceptVocabulary,
    ) -> crate::Result<PreparedRecord> {
        if input.id.is_empty() {
            return Err(StorageError::InvalidArgument("id must not be empty".into()));
        }
        validate_dimension(input.embedding, self.config.dimension)?;
        if !input.importance.is_finite() {
            return Err(StorageError::InvalidArgument(
                "importance_score must be finite".into(),
            ));
        }
        if let Some(payload) = input.payload.as_deref() {
            serde_json::from_str::<serde::de::IgnoredAny>(payload)
                .map_err(|e| StorageError::InvalidArgument(format!("invalid payload JSON: {e}")))?;
        }
        let mut emb = input.embedding.to_vec();
        normalize(&mut emb)?;
        // Augment caller-supplied concepts with auto-extracted ones from the
        // text. If the caller already provided >= max_concepts, their tags
        // are used as-is. If max_concepts is 0, auto-extraction is disabled.
        let extractor_config = self.config.tier.extractor_config();
        let concepts =
            merge_concepts_with_config(input.concepts, input.text, &extractor_config, Some(vocab));
        Ok(PreparedRecord {
            id: input.id.to_string(),
            text: input.text.to_string(),
            embedding: Arc::from(emb),
            importance: input.importance,
            concepts,
            payload: input.payload,
            scope: input.scope,
            source_role: input.source_role,
        })
    }

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
        let vocab = self.graph.read().vocab().clone();
        let prepared = self.prepare(
            WriteInput {
                id,
                text,
                embedding,
                importance,
                concepts,
                payload,
                scope,
                source_role,
            },
            &vocab,
        )?;

        // Held across seq allocation → vectors.put → wal.append → meta apply
        // so a concurrent flush cannot truncate this record's WAL entry
        // before it reaches the redb snapshot.
        let _flush_guard = self.flush_barrier.read();
        let _write_guard = self.write_lock.lock();
        if self.id_index.read().contains_key(id) {
            return Err(StorageError::DuplicateId(id.to_string()));
        }
        let offset = self.meta.allocate_offset();
        let record = prepared.into_record(self.meta.allocate_seq());

        // 1. Persist the embedding to the mmap-backed vector store first.
        //    The vector store is the durable physical source of truth for
        //    embeddings; the WAL only records the metadata operation.
        self.vectors.put(offset, record.embedding_f32())?;

        // 2. WAL metadata entry, with the vector's checksum so recovery can
        //    confirm the vector reached the file.
        self.wal.lock().append(&WalOp::Insert {
            offset,
            seq: record.insert_seq,
            meta: MetaRecord::from(&record),
            vector_crc: Some(vector_crc(record.embedding_f32())),
        })?;

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
    ///
    /// Idempotent: ids that already exist, and repeats of an id inside the
    /// batch, are skipped, so a batch can be replayed after a partial failure.
    /// Returns the number of records actually inserted. The whole batch is
    /// validated first; if any record is invalid nothing is inserted.
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
        let n = ids.len();
        if n == 0 {
            return Ok(0);
        }
        if texts.len() != n
            || embeddings.len() != n
            || importances.len() != n
            || concepts.len() != n
            || (!payloads.is_empty() && payloads.len() != n)
            || (!scopes.is_empty() && scopes.len() != n)
            || (!source_roles.is_empty() && source_roles.len() != n)
        {
            return Err(StorageError::InvalidArgument(
                "batch arrays have mismatched lengths".into(),
            ));
        }

        // Snapshot the current vocabulary so all records in the batch are
        // canonicalized consistently (even if another thread evolves the
        // vocabulary while this batch is being prepared).
        let vocab = self.graph.read().vocab().clone();
        let mut prepared: Vec<PreparedRecord> = Vec::with_capacity(n);
        let mut seen = HashSet::with_capacity(n);
        for i in 0..n {
            let input = WriteInput {
                id: &ids[i],
                text: &texts[i],
                embedding: embeddings[i],
                importance: importances[i],
                concepts: &concepts[i],
                payload: payloads.get(i).cloned().flatten(),
                scope: scopes.get(i).cloned().flatten(),
                source_role: source_roles.get(i).cloned().flatten(),
            };
            // Validate every record, including ones that will be skipped as
            // duplicates, so a bad batch is rejected the same way every time.
            let record = self.prepare(input, &vocab)?;
            if seen.insert(ids[i].as_str()) {
                prepared.push(record);
            }
        }

        // See `insert_with_payload_role`: hold off flush for the whole
        // allocate → put → append → apply sequence.
        let _flush_guard = self.flush_barrier.read();
        let _write_guard = self.write_lock.lock();
        {
            let idx = self.id_index.read();
            prepared.retain(|rec| !idx.contains_key(rec.id.as_str()));
        }
        if prepared.is_empty() {
            return Ok(0);
        }
        let records: Vec<(PointOffset, Record)> = prepared
            .into_iter()
            .map(|rec| {
                let offset = self.meta.allocate_offset();
                (offset, rec.into_record(self.meta.allocate_seq()))
            })
            .collect();

        // 1. Persist embeddings to the mmap-backed vector store first.
        for (offset, record) in &records {
            self.vectors.put(*offset, record.embedding_f32())?;
        }

        // 2. WAL metadata entries (batched under a single lock).
        {
            let ops: Vec<WalOp> = records
                .iter()
                .map(|(offset, record)| WalOp::Insert {
                    offset: *offset,
                    seq: record.insert_seq,
                    meta: MetaRecord::from(record),
                    vector_crc: Some(vector_crc(record.embedding_f32())),
                })
                .collect();
            self.wal.lock().append_batch(&ops)?;
        }

        // 3. Submit the index updates to the serialized worker.
        let inserted = records.len();
        self.update_worker.submit_and_wait(records)?;
        Ok(inserted)
    }

    /// Delete the record with the given id.
    ///
    /// The embedding is left in place in the vector store; the offset becomes
    /// unreachable because the metadata entry and id index are removed.  Segment
    /// searches filter out offsets with no metadata, so deleted points
    /// disappear from results immediately.  Physical reclamation is deferred to
    /// the vacuum optimizer.
    pub fn delete_by_id(&self, id: &str) -> crate::Result<bool> {
        self.delete_where(id, |_| true)
    }

    /// Delete `id` only if it still refers to the record at `offset`.
    ///
    /// Maintenance (eviction, deduplication) picks its victims by offset and
    /// deletes them later, sometimes after a slow summarizer call. If the
    /// caller updated the id in between, it now names a newer record that
    /// must not be deleted in the old one's place.
    pub(crate) fn delete_by_id_at(&self, id: &str, offset: PointOffset) -> crate::Result<bool> {
        self.delete_where(id, |current| current == offset)
    }

    fn delete_where(&self, id: &str, matches: impl Fn(PointOffset) -> bool) -> crate::Result<bool> {
        // Same flush barrier as inserts: the WAL delete entry and the
        // metadata removal must not straddle a flush's snapshot + truncate.
        let _flush_guard = self.flush_barrier.read();
        let _write_guard = self.write_lock.lock();
        let offset = match self.id_index.read().get(id).copied() {
            Some(o) if matches(o) => o,
            _ => return Ok(false),
        };

        // 1. WAL delete entry.
        self.wal.lock().append(&WalOp::Delete { offset })?;

        // 2. Remove from the in-memory indexes and the cognitive graph.
        self.remove_from_indexes(id, offset)?;
        Ok(true)
    }

    /// Drop a record from every in-memory index (payload, scope, text,
    /// metadata, id) and from the cognitive graph. The caller has already
    /// logged the removal.
    fn remove_from_indexes(&self, id: &str, offset: PointOffset) -> crate::Result<()> {
        // Payload, text, and scope indexes first, while the old values are
        // still known.
        if let Ok(Some(meta_rec)) = self.meta.get(offset) {
            self.payload_index
                .write()
                .remove(offset, meta_rec.payload.as_deref());
            self.scope_index
                .write()
                .remove(offset, meta_rec.scope.as_deref());
            self.text_index.remove(offset)?;
        }
        self.meta.remove(offset)?;
        self.id_index.write().remove(id);
        self.graph.write().remove_memory(id);
        Ok(())
    }

    /// Replace an existing record, preserving its id.
    ///
    /// The new version is written at a fresh offset and the old offset is
    /// tombstoned, which keeps HNSW segments correct without in-place vector
    /// updates inside immutable indexes. The swap is one WAL record, so after
    /// a crash the id refers to either the old version or the new one, never
    /// to nothing. An update that fails validation leaves the old record
    /// untouched. Returns `false` when the id does not exist.
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
        let vocab = self.graph.read().vocab().clone();
        let prepared = self.prepare(
            WriteInput {
                id,
                text,
                embedding,
                importance,
                concepts,
                payload,
                scope,
                source_role,
            },
            &vocab,
        )?;

        let _flush_guard = self.flush_barrier.read();
        let _write_guard = self.write_lock.lock();
        let Some(old_offset) = self.id_index.read().get(id).copied() else {
            // Deleted by another writer since the check above.
            return Ok(false);
        };
        let offset = self.meta.allocate_offset();
        let record = prepared.into_record(self.meta.allocate_seq());

        // Same order as an insert: vector, then one WAL record for the swap,
        // then the in-memory indexes.
        self.vectors.put(offset, record.embedding_f32())?;
        self.wal.lock().append(&WalOp::Replace {
            old_offset,
            offset,
            seq: record.insert_seq,
            meta: MetaRecord::from(&record),
            vector_crc: vector_crc(record.embedding_f32()),
        })?;
        self.remove_from_indexes(id, old_offset)?;
        self.update_worker.submit_and_wait(vec![(offset, record)])?;
        Ok(true)
    }
}
