//! A simple append-only write-ahead log with CRC32 framed records.
//!
//! Format:
//!   [magic: 4 bytes "TMSW"] [version: u32, native endian]
//!   [length: u32 BE] [payload: length bytes] [crc: u32 BE]
//!
//! The WAL stores metadata operations only; full embeddings live in the
//! `VectorStore` mmap.  This removes the largest in-memory duplicate. Each
//! insert carries a checksum of its vector so recovery can tell a vector that
//! reached the file from one that did not.
//!
//! Recovery is point-in-time: the log is read up to the first frame that is
//! torn, fails its checksum, or cannot be decoded, and everything from that
//! point on is discarded. A bad tail never makes the store unopenable.

use crate::record::{MetaRecord, PointOffset};
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const WAL_FILE: &str = "wal_meta.bin";
const WAL_MAGIC: &[u8; 4] = b"TMSW";
/// Version 1 logged inserts without a vector checksum and had no `Replace`.
const WAL_VERSION_1: u32 = 1;
const WAL_VERSION: u32 = 2;
pub const WAL_HEADER_SIZE: usize = 8;
/// Bytes of framing around each payload (length prefix + checksum).
const FRAME_OVERHEAD: u64 = 8;

/// A record in the WAL.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum WalOp {
    Insert {
        offset: PointOffset,
        seq: u64,
        meta: MetaRecord,
        /// Checksum of the vector stored at `offset` (see
        /// `vector_store::vector_crc`). `None` only for entries read from a
        /// version-1 log.
        vector_crc: Option<u32>,
    },
    Delete {
        offset: PointOffset,
    },
    Flush {
        offset: PointOffset,
    },
    /// Atomically replace the record at `old_offset` with a new record at
    /// `offset` (an update). One frame, so a crash can never leave the old
    /// record deleted and the new one missing.
    Replace {
        old_offset: PointOffset,
        offset: PointOffset,
        seq: u64,
        meta: MetaRecord,
        vector_crc: u32,
    },
}

/// The version-1 operation set, kept only to decode logs left behind by
/// older builds.
#[derive(serde::Deserialize)]
enum WalOpV1 {
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

impl From<WalOpV1> for WalOp {
    fn from(op: WalOpV1) -> Self {
        match op {
            WalOpV1::Insert { offset, seq, meta } => WalOp::Insert {
                offset,
                seq,
                meta,
                vector_crc: None,
            },
            WalOpV1::Delete { offset } => WalOp::Delete { offset },
            WalOpV1::Flush { offset } => WalOp::Flush { offset },
        }
    }
}

/// Append-only WAL.
pub struct Wal {
    path: PathBuf,
    file: File,
    /// Format version of the frames currently in the file.
    version: u32,
}

fn header_bytes(version: u32) -> [u8; WAL_HEADER_SIZE] {
    let mut bytes = [0u8; WAL_HEADER_SIZE];
    bytes[..4].copy_from_slice(WAL_MAGIC);
    bytes[4..].copy_from_slice(&version.to_ne_bytes());
    bytes
}

impl Wal {
    pub fn open(dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let path = dir.join(WAL_FILE);
        // Read + write (not append) so the file can be truncated in place;
        // the cursor is kept at the end and this is the only writer.
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)?;

        let len = file.metadata()?.len();
        let version = if len < WAL_HEADER_SIZE as u64 {
            // New file, or a header that never finished being written.
            file.set_len(0)?;
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header_bytes(WAL_VERSION))?;
            file.sync_data()?;
            WAL_VERSION
        } else {
            let mut header = [0u8; WAL_HEADER_SIZE];
            file.seek(SeekFrom::Start(0))?;
            file.read_exact(&mut header)?;
            if header[..4] != *WAL_MAGIC {
                return Err(crate::StorageError::InvalidArgument(
                    "WAL has invalid magic".into(),
                ));
            }
            let version = u32::from_ne_bytes([header[4], header[5], header[6], header[7]]);
            if version != WAL_VERSION && version != WAL_VERSION_1 {
                return Err(crate::StorageError::InvalidArgument(format!(
                    "WAL version {version} not supported"
                )));
            }
            version
        };
        file.seek(SeekFrom::End(0))?;

        Ok(Self {
            path,
            file,
            version,
        })
    }

    /// True when the file holds frames in an older format. Such a log must be
    /// replayed and then [`clear`](Self::clear)ed before anything is appended.
    pub fn needs_upgrade(&self) -> bool {
        self.version != WAL_VERSION
    }

    /// Append an operation to the WAL.
    ///
    /// The write is buffered; call [`Wal::flush`] to durably sync it to disk.
    pub fn append(&mut self, op: &WalOp) -> crate::Result<()> {
        self.append_batch(std::slice::from_ref(op))
    }

    /// Append a batch of operations under a single lock.
    ///
    /// All frames are serialized into one buffer and written with a single
    /// call, so a frame is never split across writes by this process.
    pub fn append_batch(&mut self, ops: &[WalOp]) -> crate::Result<()> {
        if ops.is_empty() {
            return Ok(());
        }
        if self.needs_upgrade() {
            return Err(crate::StorageError::InvalidArgument(
                "WAL holds frames from an older format; replay and clear it first".into(),
            ));
        }
        // Pre-serialize to size the buffer reasonably; each op is small metadata.
        let mut buf = Vec::with_capacity(ops.len() * 256);
        for op in ops {
            Self::write_op_to_buffer(&mut buf, op)?;
        }
        let start = self.file.stream_position()?;
        if let Err(e) = self.file.write_all(&buf) {
            // Roll back a partial write (for example disk full) so the next
            // append does not land behind half a frame.
            let _ = self.file.set_len(start);
            let _ = self.file.seek(SeekFrom::Start(start));
            return Err(e.into());
        }
        Ok(())
    }

    fn write_op_to_buffer(buf: &mut Vec<u8>, op: &WalOp) -> crate::Result<()> {
        let payload = bincode::serialize(op)?;
        let len = u32::try_from(payload.len())
            .map_err(|_| crate::StorageError::InvalidArgument("WAL record exceeds 4 GiB".into()))?;
        let crc = crc32fast::hash(&payload);
        buf.write_u32::<BigEndian>(len)?;
        buf.write_all(&payload)?;
        buf.write_u32::<BigEndian>(crc)?;
        Ok(())
    }

    /// Durably sync the WAL to disk.
    pub fn flush(&mut self) -> crate::Result<()> {
        self.file.sync_data()?;
        Ok(())
    }

    pub fn iter(&self) -> crate::Result<WalIter> {
        let mut file = File::open(&self.path)?;
        let file_len = file.metadata()?.len();
        file.seek(SeekFrom::Start(WAL_HEADER_SIZE as u64))?;
        Ok(WalIter {
            reader: BufReader::new(file),
            version: self.version,
            valid_end: WAL_HEADER_SIZE as u64,
            file_len,
            done: false,
        })
    }

    /// Drop everything after `valid_end` (a byte offset returned by
    /// [`WalIter::valid_end`]) so later appends follow the last good frame.
    pub fn truncate_to(&mut self, valid_end: u64) -> crate::Result<()> {
        let valid_end = valid_end.max(WAL_HEADER_SIZE as u64);
        self.file.set_len(valid_end)?;
        self.file.seek(SeekFrom::Start(valid_end))?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Empty the log, leaving only a current-version header.
    pub fn clear(&mut self) -> crate::Result<()> {
        self.file.set_len(0)?;
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&header_bytes(WAL_VERSION))?;
        self.file.sync_data()?;
        self.version = WAL_VERSION;
        Ok(())
    }
}

/// Reads frames until the end of the log or the first unreadable frame.
///
/// Yields `Err` only for real I/O failures. A torn, corrupt, or undecodable
/// frame ends iteration; [`valid_end`](Self::valid_end) then says where the
/// good prefix stops and [`discarded_bytes`](Self::discarded_bytes) how much
/// follows it.
pub struct WalIter {
    reader: BufReader<File>,
    version: u32,
    /// File offset just past the last frame that was read successfully.
    valid_end: u64,
    file_len: u64,
    done: bool,
}

impl WalIter {
    pub fn valid_end(&self) -> u64 {
        self.valid_end
    }

    /// Bytes after the last good frame (zero for a clean log).
    pub fn discarded_bytes(&self) -> u64 {
        self.file_len.saturating_sub(self.valid_end)
    }

    fn read_frame(&mut self) -> std::io::Result<Option<WalOp>> {
        let remaining = self.file_len.saturating_sub(self.valid_end);
        if remaining < FRAME_OVERHEAD {
            return Ok(None);
        }
        let len = self.reader.read_u32::<BigEndian>()? as u64;
        // A length of zero, or one that runs past the end of the file, is a
        // torn or corrupt tail (also guards the allocation below).
        if len == 0 || len > remaining - FRAME_OVERHEAD {
            return Ok(None);
        }
        let mut payload = vec![0u8; len as usize];
        self.reader.read_exact(&mut payload)?;
        let stored_crc = self.reader.read_u32::<BigEndian>()?;
        if stored_crc != crc32fast::hash(&payload) {
            return Ok(None);
        }
        let op = if self.version == WAL_VERSION_1 {
            bincode::deserialize::<WalOpV1>(&payload).map(WalOp::from)
        } else {
            bincode::deserialize::<WalOp>(&payload)
        };
        match op {
            Ok(op) => {
                self.valid_end += FRAME_OVERHEAD + len;
                Ok(Some(op))
            }
            Err(_) => Ok(None),
        }
    }
}

impl Iterator for WalIter {
    type Item = crate::Result<WalOp>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.read_frame() {
            Ok(Some(op)) => Some(Ok(op)),
            Ok(None) => {
                self.done = true;
                None
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.done = true;
                None
            }
            Err(e) => {
                self.done = true;
                Some(Err(e.into()))
            }
        }
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
            payload: None,
            scope: None,
            source_role: None,
        }
    }

    fn insert(offset: PointOffset, id: &str) -> WalOp {
        WalOp::Insert {
            offset,
            seq: offset + 10,
            meta: dummy_meta(id),
            vector_crc: Some(7),
        }
    }

    fn read_all(wal: &Wal) -> (Vec<WalOp>, u64, u64) {
        let mut it = wal.iter().unwrap();
        let mut ops = Vec::new();
        for op in it.by_ref() {
            ops.push(op.unwrap());
        }
        (ops, it.valid_end(), it.discarded_bytes())
    }

    #[test]
    fn wal_append_and_iter() {
        let tmp = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(tmp.path()).unwrap();
        wal.append(&insert(1, "a")).unwrap();
        wal.append(&insert(2, "b")).unwrap();
        wal.flush().unwrap();
        let (ops, _, discarded) = read_all(&wal);
        assert_eq!(ops.len(), 2);
        assert_eq!(discarded, 0);
    }

    #[test]
    fn wal_clear() {
        let tmp = tempfile::tempdir().unwrap();
        let mut wal = Wal::open(tmp.path()).unwrap();
        wal.append(&WalOp::Flush { offset: 0 }).unwrap();
        wal.flush().unwrap();
        wal.clear().unwrap();
        assert!(read_all(&wal).0.is_empty());
        // The cleared log is still appendable and re-openable.
        wal.append(&insert(3, "c")).unwrap();
        drop(wal);
        let wal = Wal::open(tmp.path()).unwrap();
        assert_eq!(read_all(&wal).0.len(), 1);
    }

    /// A torn, garbage, or bit-flipped tail yields the good prefix and can be
    /// truncated away; frames appended afterwards are readable.
    #[test]
    fn bad_tail_keeps_prefix_and_is_truncatable() {
        type Damage = fn(&Path);
        let cases: [(&str, Damage); 4] = [
            ("torn", |p| {
                let len = std::fs::metadata(p).unwrap().len();
                OpenOptions::new()
                    .write(true)
                    .open(p)
                    .unwrap()
                    .set_len(len - 5)
                    .unwrap();
            }),
            ("garbage", |p| {
                let mut f = OpenOptions::new().append(true).open(p).unwrap();
                f.write_all(&[0xAB; 64]).unwrap();
            }),
            ("zeros", |p| {
                let mut f = OpenOptions::new().append(true).open(p).unwrap();
                f.write_all(&[0u8; 4096]).unwrap();
            }),
            ("bitflip", |p| {
                let mut bytes = std::fs::read(p).unwrap();
                let last = bytes.len() - 6;
                bytes[last] ^= 0xFF;
                std::fs::write(p, bytes).unwrap();
            }),
        ];
        for (name, damage) in cases {
            let tmp = tempfile::tempdir().unwrap();
            let path = tmp.path().join(WAL_FILE);
            {
                let mut wal = Wal::open(tmp.path()).unwrap();
                wal.append(&insert(1, "a")).unwrap();
                wal.append(&insert(2, "b")).unwrap();
                wal.append(&insert(3, "c")).unwrap();
                wal.flush().unwrap();
            }
            damage(&path);

            let mut wal = Wal::open(tmp.path()).unwrap();
            let (ops, valid_end, discarded) = read_all(&wal);
            let expected = if name == "garbage" || name == "zeros" {
                3
            } else {
                2
            };
            assert_eq!(ops.len(), expected, "{name}: good prefix");
            assert!(discarded > 0, "{name}: tail reported");

            wal.truncate_to(valid_end).unwrap();
            wal.append(&insert(9, "z")).unwrap();
            let (ops, _, discarded) = read_all(&wal);
            assert_eq!(ops.len(), expected + 1, "{name}: append after truncate");
            assert_eq!(discarded, 0, "{name}: clean after truncate");
        }
    }

    /// A version-1 log (no vector checksum) is still readable, and is
    /// rewritten in the current format by `clear`.
    #[test]
    fn version_1_log_is_readable_and_upgraded_on_clear() {
        #[derive(serde::Serialize)]
        enum V1 {
            Insert {
                offset: PointOffset,
                seq: u64,
                meta: MetaRecord,
            },
        }
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join(WAL_FILE);
        let payload = bincode::serialize(&V1::Insert {
            offset: 4,
            seq: 9,
            meta: dummy_meta("old"),
        })
        .unwrap();
        let mut bytes = header_bytes(WAL_VERSION_1).to_vec();
        bytes.write_u32::<BigEndian>(payload.len() as u32).unwrap();
        bytes.extend_from_slice(&payload);
        bytes
            .write_u32::<BigEndian>(crc32fast::hash(&payload))
            .unwrap();
        std::fs::write(&path, bytes).unwrap();

        let mut wal = Wal::open(tmp.path()).unwrap();
        assert!(wal.needs_upgrade());
        let (ops, _, _) = read_all(&wal);
        assert!(matches!(
            &ops[..],
            [WalOp::Insert {
                offset: 4,
                seq: 9,
                vector_crc: None,
                ..
            }]
        ));
        assert!(wal.append(&insert(1, "a")).is_err(), "no mixed formats");
        wal.clear().unwrap();
        assert!(!wal.needs_upgrade());
        wal.append(&insert(1, "a")).unwrap();
        assert_eq!(read_all(&wal).0.len(), 1);
    }
}
