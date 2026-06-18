use criterion::{criterion_group, criterion_main, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use turbomemory_core::quantization::{Quantizer, VectorQuantizer};
use turbomemory_core::quantized_search::{EncodedQuery, QuantizedStore};
use turbomemory_core::turbo_quant::{TurboQuantMseQuantizer, TurboQuantProdQuantizer};

fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

fn bench_turbo_quant(c: &mut Criterion) {
    let dim = 256;
    let n = 1024;
    let bits = 3;

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| random_unit_vector(dim, i)).collect();
    let query = random_unit_vector(dim, 999_999);

    let mse = TurboQuantMseQuantizer::new(dim, bits, 42).unwrap();
    let prod = TurboQuantProdQuantizer::new(dim, bits + 1, 42, 7).unwrap();

    let mse_encoded: Vec<Vec<u8>> = vectors.iter().map(|v| mse.encode(v).unwrap()).collect();
    let prod_encoded: Vec<Vec<u8>> = vectors.iter().map(|v| prod.encode(v).unwrap()).collect();

    let mse_eq = mse.encode_query(&query).unwrap();
    let prod_eq = prod.encode_query(&query).unwrap();

    let mut group = c.benchmark_group("turbo_quant");

    group.bench_function("mse_encode", |b| {
        b.iter(|| {
            for v in &vectors {
                let _ = mse.encode(v).unwrap();
            }
        })
    });

    group.bench_function("prod_encode", |b| {
        b.iter(|| {
            for v in &vectors {
                let _ = prod.encode(v).unwrap();
            }
        })
    });

    group.bench_function("mse_score", |b| {
        b.iter(|| {
            for e in &mse_encoded {
                let _ = mse_eq.score(e);
            }
        })
    });

    group.bench_function("prod_score", |b| {
        b.iter(|| {
            for e in &prod_encoded {
                let _ = prod_eq.score(e);
            }
        })
    });

    group.bench_function("vector_quantizer_enum_dispatch", |b| {
        let q = VectorQuantizer::TurboQuantProd(prod.clone());
        let eq = q.encode_query(&query).unwrap();
        b.iter(|| {
            for e in &prod_encoded {
                let _ = eq.score(e);
            }
        })
    });

    group.finish();
}

criterion_group!(benches, bench_turbo_quant);
criterion_main!(benches);
