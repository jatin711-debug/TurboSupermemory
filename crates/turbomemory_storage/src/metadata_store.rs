//! Durable metadata store backed by redb.

use crate::config::Flusher;
use crate::record::{PointOffset, Record};
use crate::StorageError;
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const RECORDS_TABLE: TableDefinition<u64, &[u8]> = TableDefinition::new("records");
const META_TABLE: TableDefinition<&str, &str> = TableDefinition::new("meta");

/// Owns the durable metadata: records, cognitive graph, CCS, and sequence counters.
pub struct MetadataStore {
    db: Database,
    db_path: PathBuf,
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
            next_offset: AtomicU64::new(next_offset),
            next_seq: AtomicU64::new(next_seq),
        })
    }

    pub fn path(&self) -> &Path {
        &self.db_path
    }

    fn load_records(db: &Database) -> crate::Result<HashMap<PointOffset, Record>> {
        let txn = redb(db.begin_read())?;
        let mut map = HashMap::new();
        if let Ok(table) = txn.open_table(RECORDS_TABLE) {
            for item in table.iter()? {
                let (k, v) = item?;
                let offset: u64 = k.value();
                let rec: Record = bincode::deserialize(v.value())?;
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

    pub fn records(&self) -> crate::Result<HashMap<PointOffset, Record>> {
        Self::load_records(&self.db)
    }

    pub fn get(&self, offset: PointOffset) -> crate::Result<Option<Record>> {
        let txn = redb(self.db.begin_read())?;
        if let Ok(table) = txn.open_table(RECORDS_TABLE) {
            if let Some(v) = redb(table.get(offset))? {
                return Ok(Some(bincode::deserialize(v.value())?));
            }
        }
        Ok(None)
    }

    pub fn put(&self, offset: PointOffset, record: &Record) -> crate::Result<()> {
        let txn = redb(self.db.begin_write())?;
        {
            let mut table = redb(txn.open_table(RECORDS_TABLE))?;
            let bytes = bincode::serialize(record)?;
            redb(table.insert(offset, bytes.as_slice()))?;
        }
        redb(txn.commit())?;
        Ok(())
    }

    pub fn put_batch(&self, records: &[(PointOffset, Record)]) -> crate::Result<()> {
        if records.is_empty() {
            return Ok(());
        }
        let txn = redb(self.db.begin_write())?;
        {
            let mut table = redb(txn.open_table(RECORDS_TABLE))?;
            for (offset, rec) in records {
                let bytes = bincode::serialize(rec)?;
                redb(table.insert(*offset, bytes.as_slice()))?;
            }
        }
        redb(txn.commit())?;
        Ok(())
    }

    pub fn remove(&self, offset: PointOffset) -> crate::Result<()> {
        let txn = redb(self.db.begin_write())?;
        {
            let mut table = redb(txn.open_table(RECORDS_TABLE))?;
            redb(table.remove(offset))?;
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

    pub fn flush(&self) -> crate::Result<()> {
        self.save_sequences()?;
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
