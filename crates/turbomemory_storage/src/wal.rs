//! A simple append-only write-ahead log with CRC32-C framed records.
//!
//! Format:
//!   [magic: 4 bytes "TMSW"] [version: u32 BE]
//!   [length: u32 BE] [payload: length bytes] [crc: u32 BE]
//!
//! The WAL stores metadata operations only; full embeddings live in the
//! `VectorStore` mmap.  This removes the largest in-memory duplicate.

use crate::record::{MetaRecord, PointOffset};
use bytemuck::{Pod, Zeroable};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const WAL_FILE: &str = "wal_meta.bin";
const WAL_MAGIC: &[u8; 4] = b"TMSW";
const WAL_VERSION: u32 = 1;
pub const WAL_HEADER_SIZE: usize = std::mem::size_of::<WalHeader>();

#[repr(C)]
#[derive(Clone, Copy)]
struct WalHeader {
    magic: [u8; 4],
    version: u32,
}

unsafe impl Zeroable for WalHeader {}
unsafe impl Pod for WalHeader {}

/// A record in the WAL.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WalOp {
    Insert {
        offset: PointOffset,
        seq: u64,
        meta: MetaRecord,
    },
    Delete {
        offset: PointOffset,
    },
    Flush {
        offset: PointOffset,
    },
}

/// Append-only WAL.
pub struct Wal {
    path: PathBuf,
    file: File,
}

impl Wal {
    pub fn open(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join(WAL_FILE);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        let metadata = file.metadata()?;
        if metadata.len() == 0 {
            // New file: write header.
            let header = WalHeader {
                magic: *WAL_MAGIC,
                version: WAL_VERSION,
            };
            file.write_all(bytemuck::bytes_of(&header))?;
            file.sync_data()?;
        } else {
            // Verify existing header.
            let mut header_bytes = [0u8; WAL_HEADER_SIZE];
            file.read_exact(&mut header_bytes)?;
            let header: WalHeader = *bytemuck::from_bytes(&header_bytes);
            if header.magic != *WAL_MAGIC {
                return Err(crate::StorageError::InvalidArgument(
                    "WAL has invalid magic".into(),
                ));
            }
            if header.version != WAL_VERSION {
                return Err(crate::StorageError::InvalidArgument(format!(
                    "WAL version {} not supported",
                    header.version
                )));
            }
        }

        Ok(Self { path, file })
    }

    /// Append an operation to the WAL.
    ///
    /// The write is buffered; call [`Wal::flush`] to durably sync it to disk.
    pub fn append(&mut self, op: &WalOp) -> crate::Result<()> {
        let payload = bincode::serialize(op)?;
        let crc = crc32fast::hash(&payload);
        self.file.write_u32::<BigEndian>(payload.len() as u32)?;
        self.file.write_all(&payload)?;
        self.file.write_u32::<BigEndian>(crc)?;
        Ok(())
    }

    /// Durably sync the WAL to disk.
    pub fn flush(&mut self) -> crate::Result<()> {
        self.file.sync_data()?;
        Ok(())
    }

    pub fn iter(&self) -> crate::Result<WalIter> {
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;
        Ok(WalIter {
            reader: BufReader::new(file),
        })
    }

    pub fn clear(&mut self) -> crate::Result<()> {
        drop(std::fs::remove_file(&self.path));
        self.file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&self.path)?;
        let header = WalHeader {
            magic: *WAL_MAGIC,
            version: WAL_VERSION,
        };
        self.file.write_all(bytemuck::bytes_of(&header))?;
        self.file.sync_data()?;
        Ok(())
    }
}

pub struct WalIter {
    reader: BufReader<File>,
}

impl Iterator for WalIter {
    type Item = crate::Result<WalOp>;

    fn next(&mut self) -> Option<Self::Item> {
        let len = match self.reader.read_u32::<BigEndian>() {
            Ok(0) => return None,
            Ok(n) => n as usize,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e.into())),
        };
        let mut payload = vec![0u8; len];
        if let Err(e) = self.reader.read_exact(&mut payload) {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                return None;
            }
            return Some(Err(e.into()));
        }
        let stored_crc = match self.reader.read_u32::<BigEndian>() {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return None,
            Err(e) => return Some(Err(e.into())),
        };
        let computed_crc = crc32fast::hash(&payload);
        if stored_crc != computed_crc {
            return Some(Err(crate::StorageError::InvalidArgument(
                "WAL CRC mismatch".into(),
            )));
        }
        Some(bincode::deserialize(&payload).map_err(|e| e.into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Tier;

    fn dummy_meta(id: &str) -> MetaRecord {
        MetaRecord {
            id: id.to_string(),
            text: "text".to_string(),
            importance: 1.0,
            concepts: vec![],
            created_at: 0,
            insert_seq: 0,
            access_count: 0,
            last_accessed: 0,
            tier: Tier::Hot,
        }
    }

    #[test]
    fn wal_append_and_iter() {
        let tmp = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(tmp.path()).unwrap();
        wal.append(&WalOp::Insert {
            offset: 1,
            seq: 10,
            meta: dummy_meta("a"),
        })
        .unwrap();
        wal.append(&WalOp::Insert {
            offset: 2,
            seq: 11,
            meta: dummy_meta("b"),
        })
        .unwrap();
        wal.flush().unwrap();
        let ops: Vec<_> = wal.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(ops.len(), 2);
    }

    #[test]
    fn wal_clear() {
        let tmp = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(tmp.path()).unwrap();
        wal.append(&WalOp::Flush { offset: 0 }).unwrap();
        wal.flush().unwrap();
        wal.clear().unwrap();
        let ops: Vec<_> = wal.iter().unwrap().collect::<Result<Vec<_>, _>>().unwrap();
        assert!(ops.is_empty());
    }
}
