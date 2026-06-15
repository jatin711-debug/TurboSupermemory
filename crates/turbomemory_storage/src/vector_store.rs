//! mmap-backed dense vector storage keyed by `PointOffset`.
//!
//! Vectors are stored as a contiguous `f32` array in a single file.  This keeps
//! the metadata cache small (no embeddings) and lets sealed segments rerank
//! directly from disk.

use crate::record::PointOffset;
use crate::StorageError;
use bytemuck::{Pod, Zeroable};
use memmap2::MmapMut;
use parking_lot::{MappedRwLockReadGuard, RwLock, RwLockReadGuard};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"TMDV";
const VERSION: u32 = 2;
const HEADER_SIZE: usize = std::mem::size_of::<Header>();

#[repr(C)]
#[derive(Clone, Copy)]
struct Header {
    magic: [u8; 4],
    version: u32,
    dimension: u32,
    count: u64,
    header_crc: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

unsafe impl Zeroable for Header {}
unsafe impl Pod for Header {}

fn compute_header_crc(header: &Header) -> u32 {
    let mut copy = *header;
    copy.header_crc = 0;
    crc32fast::hash(bytemuck::bytes_of(&copy))
}

fn validate_header(header: &Header, expected_dim: usize) -> crate::Result<()> {
    if header.magic != *MAGIC {
        return Err(StorageError::InvalidArgument(
            "vector store has invalid magic".into(),
        ));
    }
    if header.version != VERSION {
        return Err(StorageError::InvalidArgument(format!(
            "vector store version {} not supported",
            header.version
        )));
    }
    if header.dimension as usize != expected_dim {
        return Err(StorageError::DimensionMismatch);
    }
    let computed = compute_header_crc(header);
    if computed != header.header_crc {
        return Err(StorageError::InvalidArgument(
            "vector store header CRC mismatch".into(),
        ));
    }
    Ok(())
}

struct Inner {
    file: File,
    mmap: Option<MmapMut>,
    dim: usize,
    /// Number of populated vector slots (max offset + 1).
    count: usize,
    /// Number of vector slots currently allocated in the file.
    slots: usize,
}

/// Thread-safe mmap-backed store for dense f32 vectors.
pub struct VectorStore {
    inner: RwLock<Inner>,
    path: PathBuf,
    dim: usize,
}

impl VectorStore {
    /// Create a new vector store at `path` for vectors of dimension `dim`.
    pub fn new(path: impl AsRef<Path>, dim: usize) -> crate::Result<Self> {
        Self::new_with_capacity(path, dim, 1024)
    }

    /// Create a new vector store with a requested initial slot count.
    ///
    /// The file is sized to hold at least `max(1024, initial_slots)` vectors,
    /// avoiding repeated remaps for workloads where the expected record count is
    /// known up front.
    pub fn new_with_capacity(
        path: impl AsRef<Path>,
        dim: usize,
        initial_slots: usize,
    ) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if dim == 0 {
            return Err(StorageError::InvalidArgument(
                "vector dimension must be > 0".into(),
            ));
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        let initial_slots = initial_slots.max(1024);
        let data_bytes = initial_slots
            .checked_mul(dim)
            .and_then(|n| n.checked_mul(4));
        let data_bytes = data_bytes
            .ok_or_else(|| StorageError::InvalidArgument("vector store size overflow".into()))?;
        let len = HEADER_SIZE
            .checked_add(data_bytes)
            .ok_or_else(|| StorageError::InvalidArgument("vector store size overflow".into()))?;
        file.set_len(len as u64)?;

        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        let header = Header {
            magic: *MAGIC,
            version: VERSION,
            dimension: dim as u32,
            count: 0,
            header_crc: 0,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let crc = compute_header_crc(&header);
        let header = Header {
            header_crc: crc,
            ..header
        };
        write_header(&mut mmap, header);

        Ok(Self {
            inner: RwLock::new(Inner {
                file,
                mmap: Some(mmap),
                dim,
                count: 0,
                slots: initial_slots,
            }),
            path,
            dim,
        })
    }

    /// Open an existing vector store and validate its header.
    ///
    /// If the file does not exist, it is created with room for at least
    /// `max(1024, initial_slots)` vectors.
    pub fn open(path: impl AsRef<Path>, dim: usize, initial_slots: usize) -> crate::Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            return Self::new_with_capacity(&path, dim, initial_slots);
        }
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        if mmap.len() < HEADER_SIZE {
            return Err(StorageError::InvalidArgument(
                "vector store file is too small".into(),
            ));
        }
        let header: Header = *bytemuck::from_bytes(&mmap[..HEADER_SIZE]);
        validate_header(&header, dim)?;
        let file_len = mmap.len();
        let slots = file_len.saturating_sub(HEADER_SIZE).saturating_div(dim * 4);
        let count = (header.count as usize).min(slots);
        Ok(Self {
            inner: RwLock::new(Inner {
                file,
                mmap: Some(mmap),
                dim,
                count,
                slots,
            }),
            path,
            dim,
        })
    }

    /// Store a vector at the given global offset, growing the file if needed.
    pub fn put(&self, offset: PointOffset, vector: &[f32]) -> crate::Result<()> {
        if vector.len() != self.dim {
            return Err(StorageError::DimensionMismatch);
        }
        let mut inner = self.inner.write();
        let idx = offset as usize;
        if idx >= inner.slots {
            grow_inner(&mut inner, idx)?;
        }
        let dim = inner.dim;
        write_vector(inner.mmap.as_mut().unwrap(), dim, idx, vector);
        inner.count = inner.count.max(idx + 1);
        Ok(())
    }

    /// Return a read guard for the vector at `offset`, or `None` if the slot
    /// has never been written.
    pub fn get(&self, offset: PointOffset) -> Option<MappedRwLockReadGuard<'_, [f32]>> {
        let inner = self.inner.read();
        let idx = offset as usize;
        if idx >= inner.count {
            return None;
        }
        Some(RwLockReadGuard::map(inner, |inner| {
            let mmap = inner.mmap.as_ref().unwrap();
            let start = HEADER_SIZE + idx * inner.dim * 4;
            let end = start + inner.dim * 4;
            bytemuck::cast_slice(&mmap[start..end])
        }))
    }

    /// Return a stable read view of the vector store.
    ///
    /// The view holds a single read lock for its lifetime, so all reads through
    /// it are lock-free and see a consistent mmap snapshot.  This is the
    /// preferred API for search and reranking.
    pub fn read_view(&self) -> VectorReadView<'_> {
        VectorReadView {
            inner: self.inner.read(),
        }
    }

    /// Return the number of populated slots (max offset + 1).
    pub fn count(&self) -> usize {
        self.inner.read().count
    }

    /// Return the vector dimension.
    pub fn dimension(&self) -> usize {
        self.dim
    }

    /// Return the file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Sync the mmap to disk and update the persisted header count.
    pub fn flush(&self) -> crate::Result<()> {
        let mut inner = self.inner.write();
        let count = inner.count;
        let mmap = inner.mmap.as_mut().unwrap();
        let mut header: Header = *bytemuck::from_bytes(&mmap[..HEADER_SIZE]);
        header.count = count as u64;
        let crc = compute_header_crc(&header);
        header.header_crc = crc;
        write_header(mmap, header);
        mmap.flush()?;
        Ok(())
    }
}

/// Stable read view into a `VectorStore`.
pub struct VectorReadView<'a> {
    inner: RwLockReadGuard<'a, Inner>,
}

impl VectorReadView<'_> {
    /// Read the vector at `offset` without taking an additional lock.
    pub fn get(&self, offset: PointOffset) -> Option<&[f32]> {
        let idx = offset as usize;
        if idx >= self.inner.count {
            return None;
        }
        let mmap = self.inner.mmap.as_ref().unwrap();
        let start = HEADER_SIZE + idx * self.inner.dim * 4;
        let end = start + self.inner.dim * 4;
        Some(bytemuck::cast_slice(&mmap[start..end]))
    }

    /// Return the number of populated slots in this view.
    pub fn count(&self) -> usize {
        self.inner.count
    }

    /// Return the vector dimension.
    pub fn dimension(&self) -> usize {
        self.inner.dim
    }
}

fn write_header(mmap: &mut MmapMut, header: Header) {
    let bytes = bytemuck::bytes_of(&header);
    mmap[..HEADER_SIZE].copy_from_slice(bytes);
}

fn write_vector(mmap: &mut MmapMut, dim: usize, idx: usize, vector: &[f32]) {
    let start = HEADER_SIZE + idx * dim * 4;
    let end = start + dim * 4;
    let bytes = &mut mmap[start..end];
    let floats = bytemuck::cast_slice_mut::<u8, f32>(bytes);
    floats.copy_from_slice(vector);
}

fn grow_inner(inner: &mut Inner, min_idx: usize) -> crate::Result<()> {
    let new_slots = (min_idx + 1).max(inner.slots.saturating_mul(2)).max(1024);
    let data_bytes = new_slots
        .checked_mul(inner.dim)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| StorageError::InvalidArgument("vector store grow overflow".into()))?;
    let len = HEADER_SIZE
        .checked_add(data_bytes)
        .ok_or_else(|| StorageError::InvalidArgument("vector store grow overflow".into()))?;

    // Unmap before resizing the file, then remap.
    let _ = inner.mmap.take();
    inner.file.set_len(len as u64)?;
    let mut mmap = unsafe { MmapMut::map_mut(&inner.file)? };
    let header = Header {
        magic: *MAGIC,
        version: VERSION,
        dimension: inner.dim as u32,
        count: inner.count as u64,
        header_crc: 0,
        reserved0: 0,
        reserved1: 0,
        reserved2: 0,
    };
    let crc = compute_header_crc(&header);
    let header = Header {
        header_crc: crc,
        ..header
    };
    write_header(&mut mmap, header);
    inner.mmap = Some(mmap);
    inner.slots = new_slots;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_vectors() {
        let tmp = tempfile::tempdir().unwrap();
        let store = VectorStore::new(tmp.path().join("vectors.bin"), 4).unwrap();
        store.put(0, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        store.put(1, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        store.flush().unwrap();

        let got = store.get(0).unwrap();
        assert_eq!(&*got, &[1.0f32, 0.0, 0.0, 0.0]);

        let got = store.get(1).unwrap();
        assert_eq!(&*got, &[0.0f32, 1.0, 0.0, 0.0]);

        assert!(store.get(2).is_none());
    }

    #[test]
    fn reopen_keeps_data() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("vectors.bin");
        {
            let store = VectorStore::new(&path, 3).unwrap();
            store.put(0, &[1.0, 2.0, 3.0]).unwrap();
            store.put(1, &[4.0, 5.0, 6.0]).unwrap();
            store.flush().unwrap();
        }
        let store = VectorStore::open(&path, 3, 0).unwrap();
        assert_eq!(store.count(), 2);
        let got = store.get(0).unwrap();
        assert_eq!(&*got, &[1.0f32, 2.0, 3.0]);
        let got = store.get(1).unwrap();
        assert_eq!(&*got, &[4.0f32, 5.0, 6.0]);
        assert!(store.get(2).is_none());
    }

    #[test]
    fn pre_sized_open_allocates_expected_slots() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("vectors.bin");
        let dim = 128;
        let expected = 50_000;
        let store = VectorStore::open(&path, dim, expected).unwrap();
        assert!(path.exists());
        let min_len = HEADER_SIZE + expected * dim * 4;
        let meta = std::fs::metadata(&path).unwrap();
        assert!(
            meta.len() as usize >= min_len,
            "expected file size at least {}, got {}",
            min_len,
            meta.len()
        );
        // The store should still work normally.
        let vec: Vec<f32> = (0..dim).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
        let last_offset = (expected - 1) as u64;
        store.put(last_offset, &vec).unwrap();
        assert!(store.get(last_offset).is_some());
    }
}
