//! Full-text index for the `text` field, backed by Tantivy.
//!
//! Each document stores the record's `PointOffset` in a u64 field so that
//! searches can be translated directly into a `RoaringBitmap` of offsets for
//! filtered ANN.

use crate::record::PointOffset;
use crate::StorageError;
use parking_lot::Mutex;
use roaring::RoaringBitmap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value as TantivyValue, FAST, INDEXED, STORED, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};

const TEXT_FIELD: &str = "text";
const OFFSET_FIELD: &str = "offset";

/// Tantivy-backed full-text index over memory text.
pub struct TextIndex {
    index: Index,
    writer: Mutex<IndexWriter>,
    reader: IndexReader,
    text_field: Field,
    offset_field: Field,
    path: PathBuf,
    /// Number of documents added/removed since the last commit. Used to avoid
    /// paying a Tantivy commit on every full-text query.
    pending: AtomicU64,
}

impl TextIndex {
    pub fn open(path: impl AsRef<Path>) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        std::fs::create_dir_all(&path)?;

        let mut schema_builder = Schema::builder();
        let text_field = schema_builder.add_text_field(TEXT_FIELD, TEXT);
        let offset_field = schema_builder.add_u64_field(OFFSET_FIELD, INDEXED | FAST | STORED);
        let schema = schema_builder.build();

        let index = if path.join("meta.json").exists() {
            Index::open_in_dir(&path)
                .map_err(|e| StorageError::IndexError(format!("tantivy open: {e}")))?
        } else {
            Index::create_in_dir(&path, schema)
                .map_err(|e| StorageError::IndexError(format!("tantivy create: {e}")))?
        };

        let writer = index
            .writer(50_000_000)
            .map_err(|e| StorageError::IndexError(format!("tantivy writer: {e}")))?;
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e| StorageError::IndexError(format!("tantivy reader: {e}")))?;

        Ok(Self {
            index,
            writer: Mutex::new(writer),
            reader,
            text_field,
            offset_field,
            path,
            pending: AtomicU64::new(0),
        })
    }

    /// Add a record's text to the index.
    pub fn add(&self, offset: PointOffset, text: &str) -> crate::Result<()> {
        self.add_batch(&[(offset, text)])
    }

    /// Add a batch of texts under a single writer lock acquisition.
    ///
    /// This amortizes Tantivy lock contention and small-write overhead for
    /// batch ingestion.
    pub fn add_batch(&self, docs: &[(PointOffset, &str)]) -> crate::Result<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let writer = self.writer.lock();
        for (offset, text) in docs {
            writer
                .add_document(doc!(
                    self.text_field => *text,
                    self.offset_field => *offset,
                ))
                .map_err(|e| StorageError::IndexError(format!("tantivy add: {e}")))?;
        }
        self.pending.fetch_add(docs.len() as u64, Ordering::Release);
        Ok(())
    }

    /// Remove a record from the index.
    pub fn remove(&self, offset: PointOffset) -> crate::Result<()> {
        let writer = self.writer.lock();
        writer.delete_term(Term::from_field_u64(self.offset_field, offset));
        self.pending.fetch_add(1, Ordering::Release);
        Ok(())
    }

    /// Commit pending additions/deletions and reload the reader.
    pub fn commit(&self) -> crate::Result<()> {
        let mut writer = self.writer.lock();
        writer
            .commit()
            .map_err(|e| StorageError::IndexError(format!("tantivy commit: {e}")))?;
        drop(writer);
        self.reader
            .reload()
            .map_err(|e| StorageError::IndexError(format!("tantivy reload: {e}")))?;
        self.pending.store(0, Ordering::Release);
        Ok(())
    }

    /// Commit only if there are pending writes. This removes the per-query
    /// commit stall when the text index is up to date.
    pub fn commit_if_pending(&self) -> crate::Result<()> {
        if self.pending.load(Ordering::Acquire) > 0 {
            self.commit()
        } else {
            Ok(())
        }
    }

    /// Search the full-text index and return all matching offsets.
    pub fn search(&self, query: &str) -> crate::Result<RoaringBitmap> {
        let searcher = self.reader.searcher();
        let parser = {
            let mut p = QueryParser::for_index(&self.index, vec![self.text_field]);
            p.set_conjunction_by_default();
            p
        };
        let q = parser
            .parse_query(query)
            .map_err(|e| StorageError::InvalidArgument(format!("text query: {e}")))?;
        let limit = (searcher.num_docs() as usize).max(1);
        let top_docs = searcher
            .search(&q, &TopDocs::with_limit(limit))
            .map_err(|e| StorageError::IndexError(format!("tantivy search: {e}")))?;

        let mut bitmap = RoaringBitmap::new();
        for (_score, doc_address) in top_docs {
            let doc: tantivy::TantivyDocument = searcher
                .doc(doc_address)
                .map_err(|e| StorageError::IndexError(format!("tantivy fetch doc: {e}")))?;
            if let Some(value) = doc
                .get_first(self.offset_field)
                .and_then(|v| TantivyValue::as_u64(&v))
            {
                bitmap.insert(value as u32);
            }
        }
        Ok(bitmap)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn flush(&self) -> crate::Result<()> {
        self.commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_search_and_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let idx = TextIndex::open(tmp.path()).unwrap();
        idx.add(0, "Rust is fast").unwrap();
        idx.add(1, "Python is easy").unwrap();
        idx.add(2, "Rust and Python").unwrap();
        idx.commit().unwrap();

        let bm = idx.search("Rust").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![0, 2]);

        let bm = idx.search("Rust Python").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![2]);

        idx.remove(0).unwrap();
        idx.commit().unwrap();
        let bm = idx.search("Rust").unwrap();
        assert_eq!(bm.iter().collect::<Vec<_>>(), vec![2]);
    }
}
