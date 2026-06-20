//! Crash-recovery integration tests.
//!
//! These tests exercise the public `StorageEngine` API across a process
//! restart.  They simulate the durable state that would survive a `kill -9`
//! after the OS has flushed the mmap/WAL files to disk.

use std::fs;
use turbomemory_storage::config::{OptimizerBudget, StoreConfig, TierConfig};
use turbomemory_storage::StorageEngine;

fn test_config(dim: usize, hot_capacity: usize, warm_capacity: usize) -> StoreConfig {
    StoreConfig {
        dimension: dim,
        max_edges: 8,
        level0_factor: 2,
        ef_construction: 100,
        search_list_size: 32,
        cognitive_alpha: 1.0,
        outlier_count: 0,
        initial_capacity: 1024,
        tier: TierConfig {
            hot_capacity,
            warm_capacity,
            warm_quantizer: turbomemory_storage::config::QuantizerKind::Scalar { bits: 8 },
            warm_chunk_bytes: 16 * 1024 * 1024,
            hnsw_threshold: 1000,
            full_scan_threshold_kb: 10_000,
            merge_threshold_segments: 4,
            merge_max_records: 200_000,
            cold_quantizer: turbomemory_storage::config::QuantizerKind::Sign,
            hot_promote_threshold: 2.0,
            warm_demote_threshold: 0.5,
            recency_half_life_secs: 3600,
            max_records: None,
            evict_score_floor: None,
            dedup_cosine_threshold: None,
            dedup_max_pairs_per_cycle: 1024,
            abstraction_co_occurrence_threshold: 0,
            edge_decay_half_life_secs: 0,
            max_concepts: 5,
            refinement_cosine_threshold: None,
            refinement_max_pairs_per_cycle: 1024,
            contradiction_cosine_threshold: None,
            contradiction_text_threshold: 0.3,
            contradiction_weaken_factor: 0.5,
            contradiction_max_pairs_per_cycle: 1024,
            importance_auto_scoring: false,
            importance_learning_rate: 0.3,
            importance_access_weight: 0.6,
            importance_floor: 0.1,
            importance_ceiling: 4.0,
        },
        optimizer_budget: OptimizerBudget::default(),
        auto_consolidation_interval: None,
        spreading: turbomemory_graph::SpreadingConfig::default(),
    }
}

fn random_vec(dim: usize, idx: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut v: Vec<f32> = (0..dim)
        .map(|i| rng.gen::<f32>() + ((idx + i) % 7) as f32)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in &mut v {
        *x /= norm;
    }
    v
}

fn insert_records(engine: &StorageEngine, dim: usize, n: usize) {
    for i in 0..n {
        let v = random_vec(dim, i);
        engine
            .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
            .unwrap();
    }
}

#[test]
fn reopen_replays_unflushed_wal() {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 64;
    let n = 100;

    {
        let engine = StorageEngine::open(tmp.path(), test_config(dim, 10_000, 100_000)).unwrap();
        insert_records(&engine, dim, n);
        // Persist only the WAL and vector store, not the metadata snapshot.
        // On reopen the engine must replay the WAL to recover the records.
        engine.flush_wal().unwrap();
        engine.flush_vectors().unwrap();
    }

    let engine = StorageEngine::open(tmp.path(), test_config(dim, 10_000, 100_000)).unwrap();
    assert_eq!(engine.record_count(), n);

    let query = random_vec(dim, n + 1);
    let results = engine.search_ann(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
    let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
    for id in &ids {
        assert!(id.starts_with("mem_"));
    }
}

#[test]
fn reopen_reloads_hot_warm_cold_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 32;
    // Small tiers so a Cold segment appears quickly while keeping the test fast.
    let n = 1_500;
    // Small tiers so a Cold segment appears quickly while keeping the test fast.
    // Each sealed Hot segment is below the HNSW threshold, so it becomes a Warm
    // segment; Warm compaction then merges them into a persisted Cold segment.
    let config = test_config(dim, 400, 400);

    {
        let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
        insert_records(&engine, dim, n);
        engine.flush().unwrap();
    }

    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    assert_eq!(engine.record_count(), n);

    let segments_dir = tmp.path().join("segments");
    assert!(segments_dir.join("cold").read_dir().unwrap().count() > 0);

    let query = random_vec(dim, n + 1);
    let results = engine.search_ann(&query, 10).unwrap();
    assert_eq!(results.len(), 10);
}

#[test]
fn reopen_tolerates_truncated_wal_record() {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 64;
    let n = 50;

    {
        let engine = StorageEngine::open(tmp.path(), test_config(dim, 10_000, 100_000)).unwrap();
        insert_records(&engine, dim, n);
        // Persist WAL and vectors but do not snapshot metadata or clear WAL.
        engine.flush_wal().unwrap();
        engine.flush_vectors().unwrap();
    }

    // Simulate a crash that truncated the trailing CRC of the last WAL record.
    let wal_path = tmp
        .path()
        .join("wal")
        .join(turbomemory_storage::wal::WAL_FILE);
    let mut wal_bytes = fs::read(&wal_path).unwrap();
    wal_bytes.truncate(wal_bytes.len().saturating_sub(4));
    fs::write(&wal_path, &wal_bytes).unwrap();

    let engine = StorageEngine::open(tmp.path(), test_config(dim, 10_000, 100_000)).unwrap();
    // The last, partially-written record should be skipped.
    assert_eq!(engine.record_count(), n - 1);
}
