//! Durable metadata store backed by redb with an in-memory cache.
//!
//! `redb` is now a lazy snapshot, not the primary durability log.  Writes land
//! in the WAL first; `MetadataStore` caches metadata records in memory and
//! flushes dirty entries to `redb` when asked.  Full embeddings live in the
//! separate `VectorStore`, so the metadata cache no longer duplicates vectors.

use crate::config::Flusher;
use crate::record::{MetaRecord, PointOffset, Record};
use crate::StorageError;
use parking_lot::{Mutex, RwLock};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const RECORDS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
const META_TABLE: TableDefinition<&str, &str> = TableDefinition::new("meta");

/// Owns the durable metadata: records (without embeddings), cognitive graph,
/// CCS, and sequence counters.
pub struct MetadataStore {
    db: Database,
    db_path: PathBuf,
    records: RwLock<HashMap<PointOffset, MetaRecord>>,
    dirty: Mutex<HashSet<PointOffset>>,
    next_offset: AtomicU64,
    next_seq: AtomicU64,
}

impl MetadataStore {
    pub fn open(db_path: impl AsRef<Path>) -> crate::Result<Self> {
        let db_path = db_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&db_path)?;
        let db_file = db_path.join("memory.redb");
        let db = redb(Database::create(db_file))?;
        let records = Self::load_records(&db)?;
        let next_offset = Self::load_meta(&db, "next_offset")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| records.keys().copied().max().map(|m| m + 1).unwrap_or(0));
        let next_seq = Self::load_meta(&db, "next_seq")
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                records
                    .values()
                    .map(|r| r.insert_seq + 1)
                    .max()
                    .unwrap_or(0)
            });
        Ok(Self {
            db,
            db_path,
            records: RwLock::new(records),
            dirty: Mutex::new(HashSet::new()),
            next_offset: AtomicU64::new(next_offset),
            next_seq: AtomicU64::new(next_seq),
        })
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }

    fn load_records(db: &Database) -> crate::Result<HashMap<PointOffset, MetaRecord>> {
        let txn = redb(db.begin_read())?;
        let mut map = HashMap::new();
        if let Ok(table) = txn.open_table(RECORDS_TABLE) {
            for item in table.iter()? {
                let (k, v) = item?;
                let offset: u64 = k.value();
                let rec: MetaRecord = bincode::deserialize(v.value())?;
                map.insert(offset, rec);
            }
        }
        Ok(map)
    }

    fn load_meta(db: &Database, key: &str) -> Option<String> {
        let txn = redb(db.begin_read()).ok()?;
        let table = txn.open_table(META_TABLE).ok()?;
        table.get(key).ok()??.value().to_string().into()
    }

    /// Return a snapshot of all cached metadata records.
    pub fn records(&self) -> crate::Result<HashMap<PointOffset, MetaRecord>> {
        Ok(self.records.read().clone())
    }

    /// Look up a metadata record from the in-memory cache.
    pub fn get(&self, offset: PointOffset) -> crate::Result<Option<MetaRecord>> {
        Ok(self.records.read().get(&offset).cloned())
    }

    /// Insert or update metadata from a full `Record`.  The embedding is *not*
    /// kept here; callers must write it to the `VectorStore` separately.
    pub fn put(&self, offset: PointOffset, record: &Record) -> crate::Result<()> {
        self.put_meta(offset, &MetaRecord::from(record))
    }

    /// Insert or update a metadata record directly.
    pub fn put_meta(&self, offset: PointOffset, meta: &MetaRecord) -> crate::Result<()> {
        let mut records = self.records.write();
        records.insert(offset, meta.clone());
        drop(records);
        self.dirty.lock().insert(offset);
        Ok(())
    }

    /// Batch insert/update metadata records.
    pub fn put_batch(&self, records: &[(PointOffset, Record)]) -> crate::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let mut cache = self.records.write();
        let mut dirty = self.dirty.lock();
        for (offset, rec) in records {
            cache.insert(*offset, MetaRecord::from(rec));
            dirty.insert(*offset);
        }
        Ok(())
    }

    /// Remove a record from the cache and mark the slot dirty.
    pub fn remove(&self, offset: PointOffset) -> crate::Result<()> {
        let mut records = self.records.write();
        records.remove(&offset);
        drop(records);
        self.dirty.lock().insert(offset);
        Ok(())
    }

    /// Persist dirty records, sequence counters, and the last applied WAL seq
    /// to `redb` in a single transaction.
    pub fn flush(&self, last_applied_seq: u64) -> crate::Result<()> {
        let dirty: HashSet<PointOffset> = std::mem::take(&mut *self.dirty.lock());
        if dirty.is_empty()
            && self.last_applied_seq() == Some(last_applied_seq)
            && self.records.read().is_empty()
        {
            // Nothing to do and snapshot already matches.
            return Ok(());
        }

        let records = self.records.read();
        let txn = redb(self.db.begin_write())?;
        {
            let mut table = redb(txn.open_table(RECORDS_TABLE))?;
            for offset in &dirty {
                if let Some(rec) = records.get(offset) {
                    let bytes = bincode::serialize(rec)?;
                    redb(table.insert(*offset, bytes.as_slice()))?;
                } else {
                    redb(table.remove(*offset))?;
                }
            }
        }
        {
            let mut meta = redb(txn.open_table(META_TABLE))?;
            let next_offset_str = self.next_offset.load(Ordering::SeqCst).to_string();
            redb(meta.insert("next_offset", next_offset_str.as_str()))?;
            let next_seq_str = self.next_seq.load(Ordering::SeqCst).to_string();
            redb(meta.insert("next_seq", next_seq_str.as_str()))?;
            let last_applied_str = last_applied_seq.to_string();
            redb(meta.insert("last_applied_seq", last_applied_str.as_str()))?;
        }
        redb(txn.commit())?;
        Ok(())
    }

    pub fn save_meta(&self, key: &str, value: &str) -> crate::Result<()> {
        let txn = redb(self.db.begin_write())?;
        {
            let mut table = redb(txn.open_table(META_TABLE))?;
            redb(table.insert(key, value))?;
        }
        redb(txn.commit())?;
        Ok(())
    }

    pub fn load_meta_str(&self, key: &str) -> Option<String> {
        Self::load_meta(&self.db, key)
    }

    pub fn allocate_offset(&self) -> PointOffset {
        self.next_offset.fetch_add(1, Ordering::SeqCst)
    }

    pub fn allocate_seq(&self) -> u64 {
        self.next_seq.fetch_add(1, Ordering::SeqCst)
    }

    pub fn next_offset(&self) -> PointOffset {
        self.next_offset.load(Ordering::SeqCst)
    }

    pub fn next_seq(&self) -> u64 {
        self.next_seq.load(Ordering::SeqCst)
    }

    /// Ensure the next offset counter is at least `offset + 1`.
    pub fn advance_offset_past(&self, offset: PointOffset) {
        self.next_offset.fetch_max(offset + 1, Ordering::SeqCst);
    }

    /// Ensure the next sequence counter is at least `seq + 1`.
    pub fn advance_seq_past(&self, seq: u64) {
        self.next_seq.fetch_max(seq + 1, Ordering::SeqCst);
    }

    pub fn last_applied_seq(&self) -> Option<u64> {
        Self::load_meta(&self.db, "last_applied_seq").and_then(|s| s.parse().ok())
    }

    pub fn save_sequences(&self) -> crate::Result<()> {
        self.save_meta(
            "next_offset",
            &self.next_offset.load(Ordering::SeqCst).to_string(),
        )?;
        self.save_meta(
            "next_seq",
            &self.next_seq.load(Ordering::SeqCst).to_string(),
        )?;
        Ok(())
    }

    /// Returns a flusher closure that persists sequence counters.
    pub fn flusher(&self) -> Flusher {
        let next_offset = self.next_offset();
        let next_seq = self.next_seq();
        let db_path = self.db_path.clone();
        Box::new(move || {
            let store = Self::open(&db_path)?;
            store.save_meta("next_offset", &next_offset.to_string())?;
            store.save_meta("next_seq", &next_seq.to_string())?;
            Ok(())
        })
    }
}

fn redb<T, E: Into<redb::Error>>(r: std::result::Result<T, E>) -> crate::Result<T> {
    r.map_err(|e| StorageError::Redb(e.into()))
}
