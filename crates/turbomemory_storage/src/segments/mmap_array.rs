//! A small owned-or-mmap byte buffer used by quantized segments.

use crate::segments::Result;
use bytemuck::Pod;
use memmap2::Mmap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Byte buffer that is either a Vec in memory or a read-only mmap.
pub enum MmapBuffer {
    Owned(Vec<u8>),
    Mmap(Mmap),
}

impl MmapBuffer {
    pub fn owned(len: usize) -> Self {
        Self::Owned(vec![0u8; len])
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self::Owned(bytes)
    }

    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(Self::Mmap(mmap))
    }

    pub fn as_bytes(&self) -> &[u8] {
        match self {
            Self::Owned(v) => v.as_slice(),
            Self::Mmap(m) => m.as_ref(),
        }
    }

    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Owned(v) => Some(v.as_mut_slice()),
            Self::Mmap(_) => None,
        }
    }

    pub fn len(&self) -> usize {
        self.as_bytes().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// View the buffer as a typed slice, if alignment and length allow.
    pub fn as_typed_slice<T: Pod>(&self) -> Option<&[T]> {
        let bytes = self.as_bytes();
        let align = bytes.as_ptr().align_offset(std::mem::align_of::<T>());
        if align != 0 || !bytes.len().is_multiple_of(std::mem::size_of::<T>()) {
            return None;
        }
        Some(bytemuck::cast_slice(bytes))
    }

    /// Mutable typed view for owned buffers.
    pub fn as_typed_slice_mut<T: Pod>(&mut self) -> Option<&mut [T]> {
        let bytes = self.as_bytes_mut()?;
        let align = bytes.as_ptr().align_offset(std::mem::align_of::<T>());
        if align != 0 || !bytes.len().is_multiple_of(std::mem::size_of::<T>()) {
            return None;
        }
        Some(bytemuck::cast_slice_mut(bytes))
    }

    /// Write the owned buffer to disk and reopen as a read-only mmap.
    pub fn flush_to_disk(&mut self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Self::Owned(bytes) = self {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)?;
            std::io::Write::write_all(&mut file, bytes)?;
            file.sync_all()?;
            let file = File::open(path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            *self = Self::Mmap(mmap);
        }
        Ok(())
    }

    pub fn path(&self) -> Option<&Path> {
        None
    }
}

/// Append-only byte builder that flushes to a file path.
pub struct MmapFileWriter {
    path: PathBuf,
    data: Vec<u8>,
}

impl MmapFileWriter {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            data: Vec::new(),
        }
    }

    pub fn write(&mut self, bytes: &[u8]) {
        self.data.extend_from_slice(bytes);
    }

    pub fn finish(self) -> Result<MmapBuffer> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&self.path)?;
        std::io::Write::write_all(&mut file, &self.data)?;
        file.sync_all()?;
        let file = File::open(&self.path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(MmapBuffer::Mmap(mmap))
    }
}
