//! Tiered storage engine for TurboSuperMemory.

pub mod access_counters;
pub mod config;
pub mod engine;
pub mod gpu_exact;
pub mod metadata_store;
pub mod optimizer;
pub mod payload_index;
pub mod record;
pub mod scope_index;
pub mod segment_holder;
pub mod segments;
pub mod text_index;
pub mod update_worker;
pub mod vector_store;
pub mod visited_pool;
pub mod wal;

pub use engine::GistCompressor;
pub use engine::RecoveryReport;
pub use engine::StorageEngine;
pub use gpu_exact::GpuSearchStats;

pub type Result<T> = std::result::Result<T, StorageError>;

/// Sync a directory so that a file just created in it, or renamed into it,
/// survives a power loss. POSIX keeps a file's name in its directory, and that
/// entry is only durable once the directory itself is synced. Windows has no
/// equivalent (and no way to open a directory for it), so this is a no-op
/// there. Best effort: a failure here must not fail the operation that
/// already succeeded.
pub(crate) fn sync_dir(dir: &std::path::Path) {
    #[cfg(unix)]
    {
        if let Ok(handle) = std::fs::File::open(dir) {
            let _ = handle.sync_all();
        }
    }
    #[cfg(not(unix))]
    {
        let _ = dir;
    }
}

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
    /// On-disk state that is damaged or inconsistent and cannot be repaired
    /// automatically (derived files that can be rebuilt are rebuilt instead).
    #[error("corrupted store: {0}")]
    Corrupted(String),
}
