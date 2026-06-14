//! Tiered storage engine for TurboSuperMemory.

pub mod config;
pub mod engine;
pub mod metadata_store;
pub mod record;
pub mod segment_holder;
pub mod segments;
pub mod update_handler;
pub mod vector_store;
pub mod wal;

pub type Result<T> = std::result::Result<T, StorageError>;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("core error: {0}")]
    Core(#[from] turbomemory_core::TurboError),
    #[error("redb error: {0}")]
    Redb(#[from] redb::Error),
    #[error("redb storage error: {0}")]
    RedbStorage(#[from] redb::StorageError),
    #[error("redb commit error: {0}")]
    RedbCommit(#[from] redb::CommitError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("serialization error: {0}")]
    Serialize(#[from] bincode::Error),
    #[error("id already exists: {0}")]
    DuplicateId(String),
    #[error("id not found: {0}")]
    NotFound(String),
    #[error("dimension mismatch")]
    DimensionMismatch,
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    #[error("index error: {0}")]
    IndexError(String),
}
