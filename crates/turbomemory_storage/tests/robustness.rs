//! Robustness regressions, exercised through the public `StorageEngine` API.
//!
//! Each test pins one failure that was reproduced against the engine: writes
//! lost on an unclean stop, a store that stopped working in a later session,
//! records leaking between scopes, inputs that took the process down, and
//! concurrent use that hung or failed. Several need a store above the
//! 4,096-record exact-scan threshold, because the tiered path is where most
//! of them lived.

use std::collections::HashSet;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use turbomemory_storage::config::StoreConfig;
use turbomemory_storage::engine::SupersessionKind;
use turbomemory_storage::payload_index::Filter;
use turbomemory_storage::{GistCompressor, StorageEngine, StorageError};

const DIM: usize = 16;
/// Records in a "tiered" test store: above the 4,096 exact-scan threshold.
const TIERED: usize = 4_400;

fn config(hot_capacity: usize, hnsw_threshold: usize) -> StoreConfig {
    let mut config = StoreConfig::default_for_dimension(DIM);
    config.auto_consolidation_interval = None;
    config.cognitive_alpha = 0.7;
    config.tier.hot_capacity = hot_capacity;
    config.tier.hnsw_threshold = hnsw_threshold;
    config
}

/// Sealed Hot (HNSW) segments of 1,000 records each.
fn tiered_config() -> StoreConfig {
    config(1_000, 100)
}

/// A deterministic pseudo-random unit vector.
fn unit_vec(seed: u64) -> Vec<f32> {
    let mut state = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0xD1B5_4A32_D192_ED03)
        | 1;
    let mut v: Vec<f32> = (0..DIM)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn id(i: usize) -> String {
    format!("r{i}")
}

/// Insert records `range` in batches; record `i` gets `unit_vec(i)`.
fn fill(engine: &StorageEngine, range: std::ops::Range<usize>) {
    let all: Vec<usize> = range.collect();
    for chunk in all.chunks(500) {
        let ids: Vec<String> = chunk.iter().map(|i| id(*i)).collect();
        let texts: Vec<String> = chunk.iter().map(|i| format!("note number {i}")).collect();
        let vecs: Vec<Vec<f32>> = chunk.iter().map(|i| unit_vec(*i as u64)).collect();
        let n = chunk.len();
        let inserted = engine
            .insert_batch(&ids, &texts, &vecs, &vec![0.5; n], &vec![Vec::new(); n])
            .unwrap();
        assert_eq!(inserted, n);
    }
}

/// Every record in `range` is its own nearest neighbour.
fn assert_all_findable(engine: &StorageEngine, range: std::ops::Range<usize>) {
    for i in range.step_by(37) {
        let hits = engine
            .search_ann_with_ef(&unit_vec(i as u64), 1, Some(256))
            .unwrap();
        assert_eq!(hits.first().map(|h| h.0.as_str()), Some(id(i).as_str()));
        assert!(hits[0].1 > 0.999, "record {i} scored {}", hits[0].1);
    }
}

fn subdirs(path: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(path) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

// --------------------------------------------------------------- durability

/// Dropping the engine without `flush`/`shutdown` leaves the same files a
/// killed process does: vectors and WAL written, nothing snapshotted, and the
/// vector file's header count never stamped. Everything acknowledged must be
/// there on the next open.
#[test]
fn unflushed_writes_survive_an_unclean_stop() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        fill(&engine, 0..300);
        engine
            .insert(
                "single",
                "inserted one at a time",
                &unit_vec(9_001),
                1.0,
                &[],
            )
            .unwrap();
        // no flush, no shutdown
    }
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert_eq!(engine.record_count(), 301);
    assert!(engine.contains_id("single"));
    assert_all_findable(&engine, 0..300);
    let report = engine.recovery_report();
    assert_eq!(report.wal_ops_replayed, 301);
    assert_eq!(report.wal_inserts_without_vector, 0);
    drop(engine);

    // The recovery was persisted: the next open has nothing to repair.
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert!(engine.recovery_report().is_clean());
    assert_eq!(engine.record_count(), 301);
}

/// The same after an earlier clean flush: the new writes sit beyond the
/// header count the flush stamped.
#[test]
fn writes_after_a_flush_survive_an_unclean_stop() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        fill(&engine, 0..200);
        engine.flush().unwrap();
        fill(&engine, 200..350);
    }
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert_eq!(engine.record_count(), 350);
    assert_all_findable(&engine, 0..350);
}

/// `sync_writes` syncs each vector and log record before the write returns.
/// Whether the bytes reached the platter cannot be tested without cutting
/// power; what can be tested is that every write path works with it on and
/// that recovery sees exactly what was acknowledged.
#[test]
fn synced_writes_take_every_path_and_recover() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config(10_000, 1_000);
    cfg.tier.sync_writes = true;
    {
        let engine = StorageEngine::open(tmp.path(), cfg.clone()).unwrap();
        fill(&engine, 0..2_500); // grows the vector file past its initial size
        engine
            .insert("single", "one at a time", &unit_vec(9_001), 1.0, &[])
            .unwrap();
        assert!(engine
            .update(&id(0), "second version", &unit_vec(9_002), 1.0, &[])
            .unwrap());
        assert!(engine.delete_by_id(&id(1)).unwrap());
        // unclean stop
    }
    let engine = StorageEngine::open(tmp.path(), cfg).unwrap();
    assert_eq!(engine.record_count(), 2_500);
    assert_eq!(engine.recovery_report().wal_inserts_without_vector, 0);
    assert_eq!(
        engine.find_meta_by_id(&id(0)).unwrap().text,
        "second version"
    );
    assert!(!engine.contains_id(&id(1)));
    assert!(engine.contains_id("single"));
    assert_all_findable(&engine, 2..2_500);
}

/// An update is one WAL record: after an unclean stop the id names the new
/// version, and a delete stays deleted.
#[test]
fn unflushed_update_and_delete_survive_an_unclean_stop() {
    let tmp = tempfile::tempdir().unwrap();
    let new_vec = unit_vec(7_777);
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        fill(&engine, 0..3);
        engine.flush().unwrap();
        assert!(engine
            .update(&id(0), "second version", &new_vec, 1.0, &[])
            .unwrap());
        assert!(engine.delete_by_id(&id(1)).unwrap());
    }
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert_eq!(engine.record_count(), 2);
    assert_eq!(
        engine.find_meta_by_id(&id(0)).unwrap().text,
        "second version"
    );
    assert!(!engine.contains_id(&id(1)));
    assert!(engine.contains_id(&id(2)));
    let hits = engine.search_ann(&new_vec, 1).unwrap();
    assert_eq!(hits[0].0, id(0));
    assert!(hits[0].1 > 0.999);
    // The old vector no longer answers for the id.
    let old_hits = engine.search_ann(&unit_vec(0), 3).unwrap();
    assert!(old_hits.iter().all(|(_, score)| *score < 0.99));
}

/// An update that is rejected must leave the existing record exactly as it
/// was (it used to delete the record first and validate afterwards).
#[test]
fn rejected_update_keeps_the_existing_record() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    engine
        .insert("a", "original", &unit_vec(1), 1.0, &[])
        .unwrap();

    let mut nan = unit_vec(2);
    nan[3] = f32::NAN;
    let bad_inputs: Vec<(&str, Vec<f32>, f32, Option<String>)> = vec![
        ("zero vector", vec![0.0; DIM], 1.0, None),
        ("wrong dimension", vec![0.5; DIM + 1], 1.0, None),
        ("NaN component", nan, 1.0, None),
        ("NaN importance", unit_vec(2), f32::NAN, None),
        (
            "payload that is not JSON",
            unit_vec(2),
            1.0,
            Some("{not json".into()),
        ),
    ];
    for (what, embedding, importance, payload) in bad_inputs {
        let result = engine.update_with_payload(
            "a",
            "replacement",
            &embedding,
            importance,
            &[],
            payload,
            None,
        );
        assert!(result.is_err(), "{what} must be rejected");
        assert_eq!(
            engine.find_meta_by_id("a").map(|m| m.text),
            Some("original".to_string()),
            "{what} must not touch the record"
        );
    }
    assert_eq!(engine.record_count(), 1);
    assert_eq!(engine.search_ann(&unit_vec(1), 1).unwrap()[0].0, "a");
    // Updating an id that does not exist is not an error and inserts nothing.
    assert!(!engine.update("nope", "t", &unit_vec(3), 1.0, &[]).unwrap());
    assert_eq!(engine.record_count(), 1);
}

/// Segment directories are numbered from a counter. It restarted at zero on
/// every open, so the first seal of a later session wrote its new segment
/// into the directory of one that was loaded and mapped.
#[test]
fn sealing_keeps_working_in_later_sessions() {
    let tmp = tempfile::tempdir().unwrap();
    let sealed_dir = tmp.path().join("segments").join("sealed_hot");
    let mut seen_dirs: HashSet<String> = HashSet::new();
    for session in 0..4 {
        let engine = StorageEngine::open(tmp.path(), config(50, 10)).unwrap();
        fill(&engine, session * 50..(session + 1) * 50);
        engine.flush().expect("flush must succeed in every session");
        assert_eq!(engine.record_count(), (session + 1) * 50);
        seen_dirs.extend(subdirs(&sealed_dir));
        engine.shutdown().unwrap();
    }
    assert!(
        seen_dirs.len() >= 4,
        "each session must seal into a directory of its own, saw {seen_dirs:?}"
    );
    let engine = StorageEngine::open(tmp.path(), config(50, 10)).unwrap();
    assert!(engine.recovery_report().is_clean());
    assert_eq!(engine.record_count(), 200);
    assert_all_findable(&engine, 0..200);
}

/// A torn or garbage tail on the WAL costs at most the damaged records; it
/// must not stop the store from opening, and later writes must stay readable.
#[test]
fn damaged_wal_tail_is_cut_off() {
    use std::io::Write;
    let tmp = tempfile::tempdir().unwrap();
    let wal = tmp.path().join("wal").join("wal_meta.bin");
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        fill(&engine, 0..20);
    }
    std::fs::OpenOptions::new()
        .append(true)
        .open(&wal)
        .unwrap()
        .write_all(&[0xAB; 97])
        .unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        assert_eq!(engine.record_count(), 20);
        assert_eq!(engine.recovery_report().wal_bytes_discarded, 97);
        fill(&engine, 20..25);
    }
    // The records written behind the (now removed) garbage are intact.
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert_eq!(engine.record_count(), 25);
    assert_eq!(engine.recovery_report().wal_bytes_discarded, 0);
    assert_all_findable(&engine, 0..25);
}

/// Corruption in the middle of the log keeps everything before it.
#[test]
fn corrupt_wal_record_keeps_the_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    let wal = tmp.path().join("wal").join("wal_meta.bin");
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        for i in 0..20 {
            engine
                .insert(&id(i), "text", &unit_vec(i as u64), 1.0, &[])
                .unwrap();
        }
    }
    let mut bytes = std::fs::read(&wal).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xFF;
    std::fs::write(&wal, bytes).unwrap();

    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    let count = engine.record_count();
    assert!(
        (1..20).contains(&count),
        "kept a strict prefix, got {count}"
    );
    assert!(engine.recovery_report().wal_bytes_discarded > 0);
    for i in 0..count {
        assert!(engine.contains_id(&id(i)), "prefix record {i} survives");
    }
}

/// Segment files are derived data. A damaged one is discarded and its
/// records indexed again; it must never fail the open, and a truncated index
/// file must never reach the native index library (which reads it out of
/// bounds).
#[test]
fn damaged_segment_files_are_rebuilt() {
    type Damage = fn(&Path);
    let cases: [(&str, Damage); 5] = [
        ("index file cut in half", |seg| {
            let path = seg.join("index.usearch");
            let len = std::fs::metadata(&path).unwrap().len();
            let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            file.set_len(len / 2).unwrap();
        }),
        ("index file tail overwritten", |seg| {
            let path = seg.join("index.usearch");
            let mut bytes = std::fs::read(&path).unwrap();
            let n = bytes.len();
            bytes[n - 256..].iter_mut().for_each(|b| *b = 0x5A);
            std::fs::write(&path, bytes).unwrap();
        }),
        ("index file deleted", |seg| {
            std::fs::remove_file(seg.join("index.usearch")).unwrap();
        }),
        ("manifest emptied", |seg| {
            std::fs::write(seg.join("manifest.json"), b"").unwrap();
        }),
        ("manifest is not JSON", |seg| {
            std::fs::write(seg.join("manifest.json"), b"{]").unwrap();
        }),
    ];
    for (what, damage) in cases {
        let tmp = tempfile::tempdir().unwrap();
        let sealed_dir = tmp.path().join("segments").join("sealed_hot");
        {
            let engine = StorageEngine::open(tmp.path(), config(50, 10)).unwrap();
            // One batch per segment: a batch seals the Hot segment once.
            fill(&engine, 0..50);
            fill(&engine, 50..100);
            fill(&engine, 100..120);
            engine.shutdown().unwrap();
        }
        let segments = subdirs(&sealed_dir);
        assert_eq!(
            segments.len(),
            2,
            "{what}: two sealed segments to start with"
        );
        damage(&sealed_dir.join(&segments[0]));

        let engine = StorageEngine::open(tmp.path(), config(50, 10))
            .unwrap_or_else(|e| panic!("{what}: open failed: {e}"));
        assert_eq!(engine.recovery_report().segments_discarded, 1, "{what}");
        assert_eq!(engine.record_count(), 120, "{what}");
        assert_all_findable(&engine, 0..120);
        // The next flush indexes the affected records again.
        engine.shutdown().unwrap();
        drop(engine);
        let engine = StorageEngine::open(tmp.path(), config(50, 10)).unwrap();
        assert!(
            engine.recovery_report().is_clean(),
            "{what}: repaired for good"
        );
        assert_all_findable(&engine, 0..120);
    }
}

/// A directory left by a segment build that never finished has no manifest.
/// It is removed, and its number is not reused while it exists.
#[test]
fn abandoned_segment_build_is_cleaned_up() {
    let tmp = tempfile::tempdir().unwrap();
    let sealed_dir = tmp.path().join("segments").join("sealed_hot");
    {
        let engine = StorageEngine::open(tmp.path(), config(50, 10)).unwrap();
        fill(&engine, 0..50);
        engine.shutdown().unwrap();
    }
    let orphan = sealed_dir.join("segment_41");
    std::fs::create_dir_all(&orphan).unwrap();
    std::fs::write(orphan.join("index.usearch"), b"half written").unwrap();

    let engine = StorageEngine::open(tmp.path(), config(50, 10)).unwrap();
    assert_eq!(engine.recovery_report().segment_dirs_removed, 1);
    assert!(!orphan.exists());
    assert_eq!(engine.record_count(), 50);
}

/// The vector file is the only copy of the embeddings. If it is missing or
/// cut short, opening must fail loudly instead of presenting a store whose
/// record count is right and whose searches return nothing.
#[test]
fn missing_or_truncated_vector_file_is_reported() {
    for truncate in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
            fill(&engine, 0..2_000);
            engine.shutdown().unwrap();
        }
        let vectors = tmp.path().join("vectors.bin");
        if truncate {
            let file = std::fs::OpenOptions::new()
                .write(true)
                .open(&vectors)
                .unwrap();
            file.set_len(4_096).unwrap();
        } else {
            std::fs::remove_file(&vectors).unwrap();
        }
        match StorageEngine::open(tmp.path(), config(10_000, 1_000)) {
            Err(StorageError::Corrupted(message)) => {
                assert!(message.contains("vectors.bin"), "{message}")
            }
            Err(other) => panic!("expected a corruption error, got: {other}"),
            Ok(_) => panic!("a store without its vectors must not open"),
        }
    }
}

/// The metadata file holds the record texts. Damaged or missing, it must be
/// reported: not crash the caller, and not open as an empty store on top of
/// the old vectors.
#[test]
fn damaged_or_missing_metadata_file_is_reported() {
    for truncate in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        {
            let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
            fill(&engine, 0..500);
            engine.shutdown().unwrap();
        }
        let redb = tmp.path().join("memory.redb");
        if truncate {
            let len = std::fs::metadata(&redb).unwrap().len();
            let file = std::fs::OpenOptions::new().write(true).open(&redb).unwrap();
            file.set_len(len / 2).unwrap();
        } else {
            std::fs::remove_file(&redb).unwrap();
        }
        match StorageEngine::open(tmp.path(), config(10_000, 1_000)) {
            Err(StorageError::Corrupted(message)) => {
                assert!(message.contains("memory.redb"), "{message}")
            }
            Err(other) => panic!("truncate={truncate}: expected a corruption error, got: {other}"),
            Ok(_) => panic!("truncate={truncate}: a store without its metadata must not open"),
        }
    }
    // A store that was simply never flushed has fresh metadata too, but its
    // log restores it: that is not corruption.
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        fill(&engine, 0..50);
        engine.flush_vectors().unwrap(); // header count stamped, snapshot not
    }
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert_eq!(engine.record_count(), 50);
}

/// Warm segments that were compacted into a Cold segment are removed, not
/// loaded again as live segments on the next open.
#[test]
fn compacted_warm_segments_do_not_come_back() {
    let tmp = tempfile::tempdir().unwrap();
    // No HNSW (threshold out of reach): every seal becomes a Warm segment,
    // and Warm compacts to Cold above 40 records.
    let mut cfg = config(30, 1_000_000);
    cfg.tier.full_scan_threshold_kb = 0;
    cfg.tier.warm_capacity = 40;
    let warm_dir = tmp.path().join("segments").join("warm");
    let cold_dir = tmp.path().join("segments").join("cold");
    {
        let engine = StorageEngine::open(tmp.path(), cfg.clone()).unwrap();
        for start in (0..150).step_by(30) {
            fill(&engine, start..start + 30);
            engine.flush().unwrap();
        }
        engine.shutdown().unwrap();
    }
    assert!(!subdirs(&cold_dir).is_empty(), "compaction happened");
    let warm_before = subdirs(&warm_dir);
    assert!(
        warm_before.len() <= 1,
        "compacted warm directories are removed, found {warm_before:?}"
    );

    let engine = StorageEngine::open(tmp.path(), cfg).unwrap();
    assert!(engine.recovery_report().is_clean());
    assert_eq!(engine.record_count(), 150);
    assert_all_findable(&engine, 0..150);
    engine.shutdown().unwrap();
    assert_eq!(subdirs(&warm_dir), warm_before, "nothing reappears");
}

/// A delete that was never flushed is replayed on open; the saved graph
/// snapshot must not keep the deleted memory (its text, or an edge through
/// which it would go on hiding a live record).
#[test]
fn deleted_memory_is_pruned_from_the_graph_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
        fill(&engine, 0..4);
        engine.trigger_consolidation().unwrap();
        engine.flush().unwrap(); // graph snapshot now holds all four
        engine.delete_by_id(&id(3)).unwrap();
        // unclean stop: the snapshot is not rewritten
    }
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    assert_eq!(engine.record_count(), 3);
    assert_eq!(engine.recovery_report().graph_nodes_pruned, 1);
    let graph = engine.read_graph();
    let memory_nodes: Vec<String> = graph.graph().iter_memory_nodes().map(|(k, _)| k).collect();
    assert!(
        !memory_nodes.iter().any(|k| k == "mem:r3"),
        "{memory_nodes:?}"
    );
    assert_eq!(memory_nodes.len(), 3);
}

// ---------------------------------------------------------------- isolation

fn insert_scoped(engine: &StorageEngine, id: &str, text: &str, seed: u64, scope: &str) {
    engine
        .insert_with_payload(
            id,
            text,
            &unit_vec(seed),
            0.5,
            &[],
            Some(format!("{{\"owner\":\"{scope}\"}}")),
            Some(scope.to_string()),
        )
        .unwrap();
}

/// The concept graph and the lexical index are shared by every scope. A
/// scoped cognitive search used to return whatever they reached: 900 of
/// 1,000 results belonged to the other user when the query word was theirs.
#[test]
fn scoped_cognitive_search_never_returns_another_scope() {
    for tiered in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
        // Alice only ever talks about lisbon, bob only about pasta; both
        // share the word "note" and the concept graph.
        let n = if tiered { TIERED } else { 400 };
        for start in (0..n).step_by(500) {
            let end = (start + 500).min(n);
            let ids: Vec<String> = (start..end).map(id).collect();
            let texts: Vec<String> = (start..end)
                .map(|i| {
                    if i % 2 == 0 {
                        format!("alice note about lisbon number {i}")
                    } else {
                        format!("bob note about pasta number {i}")
                    }
                })
                .collect();
            let vecs: Vec<Vec<f32>> = (start..end).map(|i| unit_vec(i as u64)).collect();
            let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();
            let scopes: Vec<Option<String>> = (start..end)
                .map(|i| Some(if i % 2 == 0 { "alice" } else { "bob" }.to_string()))
                .collect();
            let m = end - start;
            engine
                .insert_batch_with_payload(
                    &ids,
                    &texts,
                    &refs,
                    &vec![0.5; m],
                    &vec![Vec::new(); m],
                    &[],
                    &scopes,
                )
                .unwrap();
        }
        engine.flush().unwrap();

        let mut returned = 0;
        for i in (0..200).step_by(2) {
            // Alice searches with her own vector and bob's word.
            let hits = engine
                .search_scoped("pasta note", &unit_vec(i as u64), 10, Some("alice"))
                .unwrap()
                .unwrap_or_default();
            for (hit, _) in &hits {
                let owner: usize = hit[1..].parse().unwrap();
                assert_eq!(owner % 2, 0, "tiered={tiered}: alice was given {hit}");
            }
            returned += hits.len();
        }
        assert_eq!(
            returned, 1_000,
            "tiered={tiered}: every query is still filled"
        );

        // An unknown scope sees nothing, and an unscoped search sees both.
        assert!(engine
            .search_scoped("pasta", &unit_vec(1), 10, Some("carol"))
            .unwrap()
            .is_none());
        let everyone = engine.search("pasta", &unit_vec(1), 10).unwrap().unwrap();
        assert!(everyone.iter().any(|(hit, _)| hit == "r1"));
    }
}

/// The payload filter bounds a cognitive search the same way.
#[test]
fn filtered_cognitive_search_only_returns_matching_records() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    for i in 0..60 {
        let owner = if i % 3 == 0 { "alice" } else { "bob" };
        insert_scoped(&engine, &id(i), "salary review meeting", i as u64, owner);
    }
    let filter = Filter::Eq {
        field: "owner".into(),
        value: serde_json::json!("alice"),
    };
    let hits = engine
        .search_filtered("salary review", &unit_vec(1), 10, &filter)
        .unwrap()
        .unwrap();
    assert_eq!(hits.len(), 10);
    for (hit, _) in &hits {
        let i: usize = hit[1..].parse().unwrap();
        assert_eq!(i % 3, 0, "{hit} does not match the filter");
    }
}

/// With the leak closed at the source, another user's better-matching records
/// no longer use up the caller's result slots.
#[test]
fn scoped_search_is_not_starved_by_other_scopes() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    for i in 0..40 {
        insert_scoped(
            &engine,
            &id(i),
            "salary payroll bonus detail",
            i as u64,
            "bob",
        );
    }
    insert_scoped(&engine, "a_salary", "my salary is 185000", 500, "alice");
    insert_scoped(&engine, "a_home", "I live in Lisbon", 501, "alice");
    insert_scoped(&engine, "a_hobby", "I play guitar", 502, "alice");

    let hits = engine
        .search_scoped("salary payroll", &unit_vec(7), 3, Some("alice"))
        .unwrap()
        .unwrap();
    let ids: HashSet<&str> = hits.iter().map(|(hit, _)| hit.as_str()).collect();
    assert_eq!(ids, HashSet::from(["a_salary", "a_home", "a_hobby"]));
}

/// Two users storing the same sentence each keep their copy; duplicates
/// inside one scope are still merged.
#[test]
fn deduplication_stays_inside_a_scope() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config(10_000, 1_000);
    cfg.tier.dedup_cosine_threshold = Some(0.97);
    let engine = StorageEngine::open(tmp.path(), cfg).unwrap();
    insert_scoped(&engine, "alice_veg", "I am vegetarian", 1, "alice");
    insert_scoped(&engine, "bob_veg", "I am vegetarian", 1, "bob");
    insert_scoped(&engine, "bob_veg_again", "I am vegetarian", 1, "bob");
    insert_scoped(&engine, "bob_other", "I like hiking", 2, "bob");

    assert_eq!(engine.deduplicate().unwrap(), 1, "only bob's own duplicate");
    assert!(engine.contains_id("alice_veg"));
    assert!(engine.contains_id("bob_other"));
    assert_eq!(
        engine.contains_id("bob_veg") as usize + engine.contains_id("bob_veg_again") as usize,
        1
    );
    let alice = engine
        .search_ann_scoped(&unit_vec(1), 5, None, Some("alice"))
        .unwrap();
    assert_eq!(alice.len(), 1);
    assert_eq!(alice[0].0, "alice_veg");
}

/// A supersession names two records of one scope. A pair that is a record
/// with itself, or that crosses scopes, is not committed.
#[test]
fn supersession_commit_rejects_impossible_pairs() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config(10_000, 1_000);
    cfg.tier.exclude_superseded = true;
    let engine = StorageEngine::open(tmp.path(), cfg).unwrap();
    insert_scoped(&engine, "a_old", "alice lives in Lisbon", 1, "alice");
    insert_scoped(&engine, "a_new", "alice lives in Berlin", 2, "alice");
    insert_scoped(&engine, "b_fact", "bob lives in Lisbon", 3, "bob");

    let rejected = engine
        .commit_supersessions_by_id(&[
            ("a_old".into(), "a_old".into(), SupersessionKind::Refinement),
            (
                "a_old".into(),
                "b_fact".into(),
                SupersessionKind::Refinement,
            ),
            (
                "b_fact".into(),
                "a_new".into(),
                SupersessionKind::Contradiction,
            ),
            (
                "a_old".into(),
                "missing".into(),
                SupersessionKind::Refinement,
            ),
        ])
        .unwrap();
    assert_eq!(rejected, 0);
    // Nothing is hidden yet.
    assert_eq!(engine.search_ann(&unit_vec(1), 5).unwrap().len(), 3);

    let committed = engine
        .commit_supersessions_by_id(&[(
            "a_old".into(),
            "a_new".into(),
            SupersessionKind::Refinement,
        )])
        .unwrap();
    assert_eq!(committed, 1);
    let visible: Vec<String> = engine
        .search_ann(&unit_vec(1), 5)
        .unwrap()
        .into_iter()
        .map(|(hit, _)| hit)
        .collect();
    assert!(!visible.contains(&"a_old".to_string()));
    assert!(visible.contains(&"b_fact".to_string()));
}

// ------------------------------------------------------------ bad input

/// One NaN embedding used to be stored, after which a search could abort the
/// process (a sort over NaN scores).
#[test]
fn non_finite_vectors_and_queries_are_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
    fill(&engine, 0..300);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut v = unit_vec(1);
        v[5] = bad;
        assert!(engine.insert("bad", "t", &v, 1.0, &[]).is_err());
        let refs: Vec<&[f32]> = vec![&v];
        assert!(engine
            .insert_batch_with_payload(
                &["bad".into()],
                &["t".into()],
                &refs,
                &[1.0],
                &[vec![]],
                &[],
                &[]
            )
            .is_err());
        assert!(engine.search_ann(&v, 5).is_err());
        assert!(engine.search("note", &v, 5).is_err());
        assert!(engine.search_ann_batch(&[&v], 5, None, None, None).is_err());
    }
    assert!(engine
        .insert("", "empty id", &unit_vec(1), 1.0, &[])
        .is_err());
    assert_eq!(engine.record_count(), 300);
    // A batch is validated as a whole: one bad record inserts nothing.
    let good = unit_vec(900);
    let zero = vec![0.0f32; DIM];
    let refs: Vec<&[f32]> = vec![&good, &zero];
    assert!(engine
        .insert_batch_with_payload(
            &["g".into(), "z".into()],
            &["t".into(), "t".into()],
            &refs,
            &[1.0, 1.0],
            &[vec![], vec![]],
            &[],
            &[]
        )
        .is_err());
    assert!(!engine.contains_id("g"));
    assert_eq!(engine.search_ann(&unit_vec(4), 400).unwrap().len(), 300);
}

/// `top_k` is a caller-supplied number. It must never size an allocation:
/// 2^40 used to abort the process on a store above the exact-scan threshold.
#[test]
fn enormous_top_k_is_clamped_to_the_store() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
    fill(&engine, 0..300);
    for top_k in [1usize << 40, usize::MAX] {
        assert_eq!(engine.search_ann(&unit_vec(3), top_k).unwrap().len(), 300);
    }
    fill(&engine, 300..TIERED);
    engine.flush().unwrap();
    let q = unit_vec(3);
    for top_k in [1usize << 40, usize::MAX] {
        let hits = engine.search_ann(&q, top_k).unwrap();
        assert!(
            hits.len() <= TIERED && hits.len() > TIERED / 2,
            "{}",
            hits.len()
        );
        let cognitive = engine.search("note", &q, top_k).unwrap().unwrap();
        assert!(cognitive.len() <= TIERED);
        let batch = engine
            .search_ann_batch(&[&q, &q], top_k, None, None, None)
            .unwrap();
        assert_eq!(batch.len(), 2);
        assert!(batch[0].len() <= TIERED);
    }
    assert!(engine.search_ann(&q, 0).unwrap().is_empty());
}

/// Immutable segments keep the offsets of deleted records. They were dropped
/// only after the result list had been cut to `top_k`, so deleting the ten
/// best matches made the same query return nothing.
#[test]
fn deleted_records_do_not_take_result_slots() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
    fill(&engine, 0..TIERED);
    engine.flush().unwrap();
    let q = unit_vec(7);
    let first = engine.search_ann(&q, 10).unwrap();
    assert_eq!(first.len(), 10);
    let mut deleted: HashSet<String> = HashSet::new();
    for round in 0..3 {
        for (hit, _) in engine.search_ann(&q, 10).unwrap() {
            assert!(engine.delete_by_id(&hit).unwrap());
            deleted.insert(hit);
        }
        let next = engine.search_ann(&q, 10).unwrap();
        assert_eq!(
            next.len(),
            10,
            "round {round}: the next ten take their place"
        );
        assert!(next.iter().all(|(hit, _)| !deleted.contains(hit)));
        let batch = engine
            .search_ann_batch(&[&q, &q], 10, None, None, None)
            .unwrap();
        // The same records in the same order. Scores are compared loosely:
        // on a GPU a batch is one matrix product and a single query another
        // kernel, and the two round differently in the last bit.
        let ids = |hits: &[(String, f32)]| -> Vec<String> {
            hits.iter().map(|(hit, _)| hit.clone()).collect()
        };
        assert_eq!(ids(&batch[0]), ids(&next));
        assert_eq!(ids(&batch[1]), ids(&next));
        for ((_, a), (_, b)) in batch[0].iter().zip(&next) {
            assert!((a - b).abs() < 1e-5, "batch scored {a}, single {b}");
        }
    }
    assert_eq!(engine.record_count(), TIERED - 30);
}

// ---------------------------------------------------------- concurrency

/// Run `work` on its own thread and fail if it has not finished in time
/// (a deadlock would otherwise hang the whole test run).
fn finishes_within(seconds: u64, work: impl FnOnce() + Send + 'static) {
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        work();
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(seconds))
        .unwrap_or_else(|_| panic!("did not finish within {seconds}s: deadlock"));
}

/// Searching while inserting hung on a tiered store: the rerank held the
/// vector-store read lock while waiting on the thread pool, an insert queued
/// for the write lock, and the pool's workers queued behind the insert.
#[test]
fn concurrent_search_and_insert_do_not_deadlock() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
    fill(&engine, 0..TIERED);
    engine.flush().unwrap();

    let failures = Arc::new(AtomicUsize::new(0));
    let failed = failures.clone();
    finishes_within(180, move || {
        std::thread::scope(|scope| {
            for t in 0..6 {
                let engine = &engine;
                let failed = &failed;
                scope.spawn(move || {
                    for i in 0..400u64 {
                        // top_k 64 with ef 512: a rerank pool of several hundred.
                        let ok = engine
                            .search_ann_with_ef(&unit_vec(t * 1_000 + i), 64, Some(512))
                            .is_ok_and(|hits| !hits.is_empty());
                        if !ok {
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
            for w in 0..2usize {
                let engine = &engine;
                let failed = &failed;
                scope.spawn(move || {
                    for i in 0..600usize {
                        let n = 100_000 + w * 10_000 + i;
                        if engine
                            .insert(
                                &id(n),
                                "written during search",
                                &unit_vec(n as u64),
                                0.5,
                                &[],
                            )
                            .is_err()
                        {
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        assert_eq!(engine.record_count(), TIERED + 1_200);
    });
    assert_eq!(failures.load(Ordering::Relaxed), 0);
}

/// More searching threads than the index was built with: in the process
/// that built it, most of them used to fail with "Reserve capacity ahead of
/// searches!".
#[test]
fn many_threads_can_search_a_freshly_built_index() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
    fill(&engine, 0..TIERED);
    engine.flush().unwrap();

    let failures = AtomicUsize::new(0);
    std::thread::scope(|scope| {
        for t in 0..48u64 {
            let engine = &engine;
            let failures = &failures;
            scope.spawn(move || {
                for i in 0..40u64 {
                    if let Err(e) = engine.search_ann(&unit_vec(t * 100 + i), 10) {
                        eprintln!("search failed: {e}");
                        failures.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
        }
    });
    assert_eq!(failures.load(Ordering::Relaxed), 0);
}

/// Two writers racing on one id: exactly one insert wins.
#[test]
fn concurrent_inserts_of_one_id_admit_exactly_one() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), config(10_000, 1_000)).unwrap();
    for round in 0..20u64 {
        let winners = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for t in 0..8u64 {
                let engine = &engine;
                let winners = &winners;
                scope.spawn(move || {
                    match engine.insert(
                        &format!("same{round}"),
                        "t",
                        &unit_vec(round * 10 + t),
                        1.0,
                        &[],
                    ) {
                        Ok(_) => {
                            winners.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(StorageError::DuplicateId(_)) => {}
                        Err(e) => panic!("unexpected error: {e}"),
                    }
                });
            }
        });
        assert_eq!(winners.load(Ordering::Relaxed), 1, "round {round}");
    }
    assert_eq!(engine.record_count(), 20);
}

/// Batch search must agree with single-query search above the exact-scan
/// threshold, including for queries that share candidates. (The CUDA rerank
/// mis-indexed shared candidates; with the `cuda` feature this test runs on
/// the GPU.)
#[test]
fn batch_search_matches_single_search_on_a_tiered_store() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), tiered_config()).unwrap();
    fill(&engine, 0..TIERED);
    engine.flush().unwrap();

    // Neighbouring queries (and one exact repeat) so candidate lists overlap.
    let mut queries: Vec<Vec<f32>> = Vec::new();
    for i in 0..12u64 {
        let base = unit_vec(i);
        let mut near = base.clone();
        near[0] += 0.05;
        queries.push(base);
        queries.push(near);
    }
    queries.push(unit_vec(0));
    // A query that is not unit length scores as cosine all the same.
    queries.push(unit_vec(5).iter().map(|x| x * 3.0).collect());
    let refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();

    for ef in [None, Some(400)] {
        let batch = engine.search_ann_batch(&refs, 10, ef, None, None).unwrap();
        assert_eq!(batch.len(), refs.len());
        for (q, from_batch) in refs.iter().zip(&batch) {
            let single = engine.search_ann_with_ef(q, 10, ef).unwrap();
            let single_ids: Vec<&str> = single.iter().map(|(hit, _)| hit.as_str()).collect();
            let batch_ids: Vec<&str> = from_batch.iter().map(|(hit, _)| hit.as_str()).collect();
            assert_eq!(batch_ids, single_ids, "ef={ef:?}");
            for ((_, a), (_, b)) in single.iter().zip(from_batch) {
                assert!((a - b).abs() < 1e-4, "ef={ef:?}: scores {a} vs {b}");
                assert!(*b <= 1.0 + 1e-4, "score {b} is not a cosine");
            }
        }
    }
}

// ------------------------------------------------------- gist-before-evict

struct Flaky {
    broken: std::sync::atomic::AtomicBool,
}

impl GistCompressor for Flaky {
    fn compress(&self, texts: &[String]) -> Result<Option<(String, Vec<f32>)>, String> {
        if self.broken.load(Ordering::Relaxed) {
            Err("summarizer unreachable".into())
        } else {
            Ok(Some((
                format!("gist of {} facts", texts.len()),
                unit_vec(4_242),
            )))
        }
    }
}

/// With gist-before-evict on, a memory leaves the store only once its gist
/// is in. When the summarizer fails the victims stay and are tried again;
/// they used to be deleted with nothing in their place and no error.
#[test]
fn failed_gist_keeps_the_memories_for_the_next_attempt() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = config(10_000, 1_000);
    cfg.tier.max_records = Some(20);
    cfg.tier.access_aware_eviction = false;
    cfg.tier.gist_before_evict = true;
    let engine = StorageEngine::open(tmp.path(), cfg).unwrap();
    let compressor = Arc::new(Flaky {
        broken: std::sync::atomic::AtomicBool::new(true),
    });
    engine.set_gist_compressor(Some(compressor.clone()));
    fill(&engine, 0..60);

    assert_eq!(
        engine.evict().unwrap(),
        0,
        "nothing is deleted without its gist"
    );
    assert_eq!(engine.record_count(), 60);

    compressor.broken.store(false, Ordering::Relaxed);
    assert_eq!(engine.evict().unwrap(), 40);
    let gists = engine.record_count() - 20;
    assert!(gists >= 1, "the evicted memories live on as gists");
    for i in 40..60 {
        assert!(engine.contains_id(&id(i)), "the newest 20 are kept");
    }

    // A gist that cannot be stored (wrong dimension) counts as a failure too.
    struct WrongDim;
    impl GistCompressor for WrongDim {
        fn compress(&self, _texts: &[String]) -> Result<Option<(String, Vec<f32>)>, String> {
            Ok(Some(("gist".into(), vec![1.0; DIM + 1])))
        }
    }
    engine.set_gist_compressor(Some(Arc::new(WrongDim)));
    fill(&engine, 60..100);
    let before = engine.record_count();
    assert_eq!(engine.evict().unwrap(), 0);
    assert_eq!(engine.record_count(), before);
}

// ------------------------------------------------------ resident GPU search

/// The resident-search path for a store of any size, on whatever backend this
/// build has: the real device with the `cuda` feature, host emulation of the
/// same calls without it. Everything stays in the Hot segment so the tests
/// do not spend their time building indexes the path never reads.
fn resident_config() -> StoreConfig {
    let mut cfg = config(1_000_000, 1_000_000);
    cfg.tier.gpu_exact_on_cpu_backend = true;
    cfg.tier.gpu_exact_min_records = 0;
    cfg
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

/// `hits` are the `top_k` best of `live` (seed, id) for `query`: the scores
/// match a brute-force ranking position by position, and every hit is a live
/// record scored as itself. Compared by score so that two records a rounding
/// error apart may come back in either order.
fn assert_exact(hits: &[(String, f32)], query: &[f32], live: &[(u64, String)], top_k: usize) {
    let mut truth: Vec<f32> = live
        .iter()
        .map(|(seed, _)| cosine(query, &unit_vec(*seed)))
        .collect();
    truth.sort_by(|a, b| b.total_cmp(a));
    truth.truncate(top_k);
    assert_eq!(hits.len(), truth.len(), "result count");
    let mut seen = HashSet::new();
    for ((hit, score), want) in hits.iter().zip(&truth) {
        assert!(
            (score - want).abs() < 1e-4,
            "{hit} scored {score}, the exact ranking has {want} here"
        );
        let (seed, _) = live
            .iter()
            .find(|(_, name)| name == hit)
            .unwrap_or_else(|| panic!("{hit} is not a live record"));
        assert!((cosine(query, &unit_vec(*seed)) - score).abs() < 1e-4);
        assert!(seen.insert(hit.as_str()), "{hit} returned twice");
    }
}

/// With the vectors resident on the device a search is one product over
/// every row, so the answer has to be the exact one, and it has to keep being
/// exact while the store is written to: new records, a reallocation, deletes
/// and updates (whose old rows stay on the device and are masked).
#[test]
fn resident_search_is_exact_and_follows_writes() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), resident_config()).unwrap();
    let mut live: Vec<(u64, String)> = Vec::new();
    let queries: Vec<Vec<f32>> = (0..8u64)
        .map(|i| unit_vec(50_000 + i))
        // not unit length: scores are cosines all the same
        .chain([unit_vec(3).iter().map(|x| x * 4.0).collect()])
        .collect();
    let check = |live: &[(u64, String)], what: &str| {
        for q in &queries {
            let hits = engine.search_ann(q, 10).unwrap();
            assert_exact(&hits, q, live, 10);
        }
        let refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
        let batch = engine
            .search_ann_batch(&refs, 10, None, None, None)
            .unwrap();
        for (q, hits) in queries.iter().zip(&batch) {
            assert_exact(hits, q, live, 10);
        }
        let stats = engine.gpu_search_stats().expect(what);
        assert!(stats.active, "{what}: {stats:?}");
        assert!(stats.rows >= live.len(), "{what}: {stats:?}");
    };

    // The first search uploads everything.
    fill(&engine, 0..1_500);
    live.extend((0..1_500).map(|i| (i as u64, id(i))));
    check(&live, "first upload");

    // Later searches upload only what was added; 9,000 records outgrow the
    // first allocation.
    fill(&engine, 1_500..9_000);
    live.extend((1_500..9_000).map(|i| (i as u64, id(i))));
    check(&live, "after growing");

    // Delete the ten best matches of the first query: the next ten take
    // their place.
    for (hit, _) in engine.search_ann(&queries[0], 10).unwrap() {
        assert!(engine.delete_by_id(&hit).unwrap());
        live.retain(|(_, name)| *name != hit);
    }
    check(&live, "after deletes");

    // An update moves the record to a new row; the old one must not answer.
    assert!(engine
        .update(&id(7), "moved", &unit_vec(70_000), 0.5, &[])
        .unwrap());
    live.retain(|(_, name)| *name != id(7));
    live.push((70_000, id(7)));
    check(&live, "after an update");
    let moved = engine.search_ann(&unit_vec(70_000), 1).unwrap();
    assert_eq!(moved[0].0, id(7));
    let old = engine.search_ann(&unit_vec(7), 1).unwrap();
    assert!(old[0].1 < 0.999, "the replaced vector still answers");
}

/// Scope and payload filters bound the resident search exactly as they bound
/// the CPU paths: the device scores every row, the host keeps only what the
/// caller may see.
#[test]
fn resident_search_honours_scope_and_filter() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), resident_config()).unwrap();
    for i in 0..600usize {
        let owner = if i % 3 == 0 { "alice" } else { "bob" };
        insert_scoped(&engine, &id(i), "note", i as u64, owner);
    }
    let alice: Vec<(u64, String)> = (0..600)
        .filter(|i| i % 3 == 0)
        .map(|i| (i as u64, id(i)))
        .collect();
    let filter = Filter::Eq {
        field: "owner".into(),
        value: serde_json::json!("alice"),
    };
    // Queries that sit exactly on records of bob: the best match is his.
    let queries: Vec<Vec<f32>> = [1u64, 2, 4, 5, 7].iter().map(|i| unit_vec(*i)).collect();
    let refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
    for q in &queries {
        let scoped = engine
            .search_ann_scoped(q, 10, None, Some("alice"))
            .unwrap();
        assert_exact(&scoped, q, &alice, 10);
        let filtered = engine
            .search_ann_candidates_filtered(q, 10, Some(&filter))
            .unwrap();
        assert_exact(&filtered, q, &alice, 10);
    }
    let batch = engine
        .search_ann_batch(&refs, 10, None, Some(&filter), Some("alice"))
        .unwrap();
    assert_eq!(batch.len(), queries.len());
    for (hits, q) in batch.iter().zip(&queries) {
        assert_exact(hits, q, &alice, 10);
    }
    // A scope nobody wrote to matches nothing, and asking for more than the
    // scope holds returns all of it.
    assert!(engine
        .search_ann_scoped(&queries[0], 10, None, Some("carol"))
        .unwrap()
        .is_empty());
    let all = engine
        .search_ann_scoped(&queries[0], 10_000, None, Some("alice"))
        .unwrap();
    assert_eq!(all.len(), alice.len());
    assert!(engine.gpu_search_stats().unwrap().active);
}

/// A store that outgrows the device memory it was given stops using the
/// device and keeps answering from the CPU.
#[test]
fn resident_search_falls_back_when_the_store_outgrows_its_budget() {
    let tmp = tempfile::tempdir().unwrap();
    let mut cfg = resident_config();
    cfg.tier.gpu_memory_budget_mb = 1; // 16,384 rows of 16 floats
    let engine = StorageEngine::open(tmp.path(), cfg).unwrap();
    fill(&engine, 0..5_000);
    assert_all_findable(&engine, 0..5_000);
    let stats = engine.gpu_search_stats().unwrap();
    assert!(stats.active && stats.rows == 5_000, "{stats:?}");
    assert!(stats.memory_bytes <= stats.budget_bytes, "{stats:?}");

    fill(&engine, 5_000..16_500);
    assert_all_findable(&engine, 0..16_500);
    let stats = engine.gpu_search_stats().unwrap();
    assert!(!stats.active, "over budget but still active: {stats:?}");
    assert_eq!(stats.memory_bytes, 0, "the device memory was not released");
    // ... and stays off; searches are simply CPU searches now.
    assert_all_findable(&engine, 16_000..16_500);
    assert!(!engine.gpu_search_stats().unwrap().active);
}

/// Searches share one device mirror while writers keep extending the store.
#[test]
fn resident_search_runs_alongside_writers() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), resident_config()).unwrap();
    fill(&engine, 0..3_000);

    let failures = Arc::new(AtomicUsize::new(0));
    let failed = failures.clone();
    finishes_within(180, move || {
        std::thread::scope(|scope| {
            for t in 0..6u64 {
                let engine = &engine;
                let failed = &failed;
                scope.spawn(move || {
                    for i in 0..300u64 {
                        // A record written before the threads started is
                        // always its own best match.
                        let target = (t * 499 + i * 7) % 3_000;
                        let q = unit_vec(target);
                        let ok = if i % 5 == 0 {
                            engine
                                .search_ann_batch(
                                    &[q.as_slice(), q.as_slice()],
                                    5,
                                    None,
                                    None,
                                    None,
                                )
                                .is_ok_and(|both| {
                                    both.iter().all(|hits| hits[0].0 == id(target as usize))
                                })
                        } else {
                            engine
                                .search_ann(&q, 5)
                                .is_ok_and(|hits| hits[0].0 == id(target as usize))
                        };
                        if !ok {
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
            for w in 0..2usize {
                let engine = &engine;
                let failed = &failed;
                scope.spawn(move || {
                    for i in 0..3_000usize {
                        let n = 100_000 + w * 10_000 + i;
                        if engine
                            .insert(
                                &id(n),
                                "written during search",
                                &unit_vec(n as u64),
                                0.5,
                                &[],
                            )
                            .is_err()
                        {
                            failed.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        assert_eq!(engine.record_count(), 9_000);
        // Everything written while the searches ran is on the device now.
        assert_all_findable(&engine, 100_000..103_000);
        assert_all_findable(&engine, 110_000..113_000);
        let stats = engine.gpu_search_stats().unwrap();
        assert!(stats.active && stats.rows == 9_000, "{stats:?}");
        // Every search above went through the mirror: 240 single queries and
        // 60 two-query batches per thread. Searches that queued behind one
        // another shared a device call.
        assert!(stats.queries >= 6 * (240 + 120), "{stats:?}");
        assert!(stats.device_calls <= stats.queries, "{stats:?}");
    });
    assert_eq!(failures.load(Ordering::Relaxed), 0);
}
