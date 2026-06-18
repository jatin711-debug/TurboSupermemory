use criterion::{criterion_group, criterion_main, Criterion};
use std::time::Duration;
use turbomemory_storage::config::{OptimizerBudget, StoreConfig, TierConfig};
use turbomemory_storage::StorageEngine;

fn bench_config(dim: usize, hot_capacity: usize, warm_capacity: usize) -> StoreConfig {
    StoreConfig {
        dimension: dim,
        max_edges: 16,
        level0_factor: 2,
        ef_construction: 100,
        search_list_size: 100,
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
        },
        optimizer_budget: OptimizerBudget::default(),
        auto_consolidation_interval: None,
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

fn bench_exact_search(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 128;
    let n = 2_048;
    let engine = StorageEngine::open(tmp.path(), bench_config(dim, 10_000, 100_000)).unwrap();
    insert_records(&engine, dim, n);

    let query = random_vec(dim, n + 1);
    c.bench_function("exact_search_2k_d128", |b| {
        b.iter(|| engine.search_ann(&query, 10).unwrap())
    });
}

fn bench_hnsw_search(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 128;
    let n = 12_000;
    let engine = StorageEngine::open(tmp.path(), bench_config(dim, 10_000, 100_000)).unwrap();
    insert_records(&engine, dim, n);

    let query = random_vec(dim, n + 1);
    c.bench_function("hnsw_search_12k_d128", |b| {
        b.iter(|| engine.search_ann(&query, 10).unwrap())
    });
}

fn bench_warm_search(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 128;
    let n = 5_000;
    // hot_capacity 2k -> first seal stays Hot (HNSW), second seal becomes Warm.
    let engine = StorageEngine::open(tmp.path(), bench_config(dim, 2_000, 10_000)).unwrap();
    insert_records(&engine, dim, n);

    let query = random_vec(dim, n + 1);
    c.bench_function("warm_search_5k_d128", |b| {
        b.iter(|| engine.search_ann(&query, 10).unwrap())
    });
}

fn bench_cold_search(c: &mut Criterion) {
    let tmp = tempfile::tempdir().unwrap();
    let dim = 128;
    let n = 8_000;
    // hot 2k, warm capacity 3k -> after two warm seals warm total exceeds 3k and
    // gets compacted into a Cold segment.
    let engine = StorageEngine::open(tmp.path(), bench_config(dim, 2_000, 3_000)).unwrap();
    insert_records(&engine, dim, n);

    let query = random_vec(dim, n + 1);
    c.bench_function("cold_search_8k_d128", |b| {
        b.iter(|| engine.search_ann(&query, 10).unwrap())
    });
}

fn configure() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = configure();
    targets = bench_exact_search, bench_hnsw_search, bench_warm_search, bench_cold_search
}
criterion_main!(benches);
