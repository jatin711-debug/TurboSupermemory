//! Configuration for the tiered storage engine.

use std::time::Duration;
use turbomemory_core::quantization::{ScalarQuantizer, SignQuantizer, VectorQuantizer};
use turbomemory_core::turbo_quant::{TurboQuantMseQuantizer, TurboQuantProdQuantizer};

/// Which quantizer a tier uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum QuantizerKind {
    /// Per-calibration min/max scalar quantizer.
    Scalar { bits: u8 },
    /// 1-bit sign quantization.
    Sign,
    /// TurboQuant MSE-optimal quantizer (direction only; store norms separately).
    TurboQuantMse { bits: u8 },
    /// TurboQuant inner-product-optimal quantizer (direction only).
    TurboQuantProd { bits: u8 },
}

impl QuantizerKind {
    /// Default fixed-seed rotation/projection seeds.  These are deterministic
    /// across restarts so serialized segments remain readable.
    pub(crate) const ROTATION_SEED: u64 = 0x1234_5678_9ABC_DEF0;
    pub(crate) const QJL_SEED: u64 = 0xFEDC_BA98_7654_3210;

    /// Build a concrete quantizer for the given dimension.
    ///
    /// For [`QuantizerKind::Scalar`] the caller must supply calibration vectors
    /// because the min/max are data-dependent; this method returns a zero-range
    /// placeholder that must be replaced by [`ScalarQuantizer::calibrate`].
    pub fn build(self, dim: usize) -> VectorQuantizer {
        match self {
            Self::Scalar { bits } => VectorQuantizer::Scalar(ScalarQuantizer {
                bits,
                dim,
                min: 0.0,
                max: 0.0,
            }),
            Self::Sign => VectorQuantizer::Sign(SignQuantizer::new(dim)),
            Self::TurboQuantMse { bits } => VectorQuantizer::TurboQuantMse(
                TurboQuantMseQuantizer::new(dim, bits, Self::ROTATION_SEED)
                    .expect("valid TurboQuantMse bits"),
            ),
            Self::TurboQuantProd { bits } => VectorQuantizer::TurboQuantProd(
                TurboQuantProdQuantizer::new(dim, bits, Self::ROTATION_SEED, Self::QJL_SEED)
                    .expect("valid TurboQuantProd bits"),
            ),
        }
    }
}

/// Which tier a vector segment occupies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum Tier {
    Hot,
    Warm,
    Cold,
}

impl Tier {
    pub fn name(self) -> &'static str {
        match self {
            Tier::Hot => "hot",
            Tier::Warm => "warm",
            Tier::Cold => "cold",
        }
    }
}

/// Resource limits for CPU-heavy background optimizer work.
#[derive(Debug, Clone)]
pub struct OptimizerBudget {
    /// Maximum HNSW segment builds that may run at the same time.
    pub max_concurrent_builds: usize,
    /// Optional memory ceiling for a single HNSW build (bytes).
    /// If an estimated build would exceed it, the optimizer falls back to a
    /// Warm segment instead.
    pub max_build_memory_bytes: Option<usize>,
}

impl Default for OptimizerBudget {
    fn default() -> Self {
        // Allow the foreground `drain` and the background optimizer to merge
        // concurrently. Qdrant uses `clamp(num_cpus, 1, 16)` indexing threads
        // (TODO 2.11). We allow up to half the CPUs so we don't oversubscribe
        // the machine.
        let max_concurrent = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(2))
            .unwrap_or(2);
        Self {
            max_concurrent_builds: max_concurrent,
            max_build_memory_bytes: Some(512 * 1024 * 1024), // 512 MiB
        }
    }
}

/// Per-tier sizing and quantization policy.
#[derive(Debug, Clone, PartialEq)]
pub struct TierConfig {
    /// Number of records before the Hot segment is sealed.
    pub hot_capacity: usize,
    /// Number of records before the Warm segment is demoted to Cold.
    pub warm_capacity: usize,
    /// Quantizer used by the Warm tier.
    pub warm_quantizer: QuantizerKind,
    /// Byte size of a single Warm tier mmap chunk.
    pub warm_chunk_bytes: usize,
    /// Minimum number of records in a sealed segment before building an HNSW
    /// index. Smaller segments stay quantized/plain for efficiency.
    pub hnsw_threshold: usize,
    /// Minimum size (in KB of FP32 vector data) of a sealed segment before
    /// building an HNSW index.  If non-zero this overrides `hnsw_threshold`.
    pub full_scan_threshold_kb: usize,
    /// Maximum number of sealed HNSW segments before the merge optimizer combines
    /// them. Keeping this small improves recall and search latency.
    pub merge_threshold_segments: usize,
    /// Maximum number of records a single merge operation may combine. This
    /// limits peak memory during HNSW rebuilds.
    pub merge_max_records: usize,
    /// Quantizer used by the Cold tier.
    pub cold_quantizer: QuantizerKind,
    /// Access-score threshold above which a record is promoted back to Hot.
    pub hot_promote_threshold: f64,
    /// Access-score threshold below which a record in Hot is demoted.
    pub warm_demote_threshold: f64,
    /// Recency half-life in seconds for access scoring.
    pub recency_half_life_secs: u64,
    /// Hard cap on the number of live records. When `Some(k)`, consolidation
    /// evicts the lowest-`access_score` records until the live count is back
    /// at or under `k`. `None` (default) means unbounded storage — existing
    /// behavior, no eviction.
    pub max_records: Option<usize>,
    /// Eviction floor on `access_score`. When `Some(f)`, any record whose
    /// recency-weighted access score drops below `f` is evicted during
    /// consolidation, independent of `max_records`. `None` (default) disables
    /// score-floor eviction.
    pub evict_score_floor: Option<f64>,
    /// Cosine-similarity threshold for near-duplicate merge. When `Some(t)`
    /// (e.g. 0.97), consolidation merges record pairs whose cosine similarity
    /// is `>= t`, keeping the higher-salience record. `None` (default) disables
    /// semantic deduplication.
    pub dedup_cosine_threshold: Option<f32>,
    /// Upper bound on the number of duplicate pairs merged in a single
    /// consolidation cycle, to bound per-cycle work. Ignored when
    /// `dedup_cosine_threshold` is `None`.
    pub dedup_max_pairs_per_cycle: usize,
}

impl TierConfig {
    /// Default fixed thresholds used when dimension-aware scaling is not desired.
    pub const FIXED: Self = Self {
        hot_capacity: 10_000,
        warm_capacity: 100_000,
        warm_quantizer: QuantizerKind::Scalar { bits: 8 },
        warm_chunk_bytes: 16 * 1024 * 1024, // 16 MiB
        hnsw_threshold: 1000,
        full_scan_threshold_kb: 10_000,
        merge_threshold_segments: 2,
        merge_max_records: 20_000,
        // 8-bit scalar, not 1-bit sign: sign quantization cannot resolve the
        // tiny cosine gaps between near-orthogonal high-dimensional vectors, so
        // candidate selection from a sign-quantized Cold tier silently drops the
        // true top-k before the full-f32 rerank ever sees them.
        cold_quantizer: QuantizerKind::Scalar { bits: 8 },
        hot_promote_threshold: 2.0,
        warm_demote_threshold: 0.5,
        recency_half_life_secs: 3600,
        // Eviction and dedup are opt-in; default to current unbounded behavior.
        max_records: None,
        evict_score_floor: None,
        dedup_cosine_threshold: None,
        dedup_max_pairs_per_cycle: 1024,
    };

    /// Recommended thresholds for a given vector dimension.
    ///
    /// Higher dimensions get smaller record-count thresholds because each vector
    /// is more expensive to scan and to build into an HNSW graph. Lower
    /// dimensions can keep larger plain and warm segments before promoting to
    /// indexed or quantized tiers.
    pub fn scaled_for_dimension(dim: usize) -> Self {
        let dim = dim.max(1);
        // Target ~4 MiB of FP32 vectors in the appendable Hot segment.
        let hot_capacity = (4usize * 1024 * 1024)
            .saturating_div(dim.saturating_mul(4))
            .clamp(512, 8192);
        // HNSW pays off earlier for high dimensions; plain scan is cheap for
        // low dimensions so we require more records there.
        let hnsw_threshold = (20_000.0 / (dim as f64).sqrt()) as usize;
        let hnsw_threshold = hnsw_threshold.max(512);
        let full_scan_threshold_kb = 10_000usize;
        // Target ~64 MiB of 8-bit quantized vectors in the Warm tier. The Warm
        // tier keeps full 8-bit precision and reranks with f32, so it preserves
        // recall on near-orthogonal high-dim data; the Cold tier exists for
        // capacity, not recall. Sizing Warm generously keeps realistic
        // collections (tens of thousands of vectors) out of Cold.
        let warm_capacity = (64usize * 1024 * 1024)
            .saturating_div(dim)
            .clamp(2_048, 500_000);
        // Cap merged HNSW segments at a size that one indexing thread can
        // build quickly and that parallel multi-segment search can keep in
        // cache. This is the Qdrant "one segment per indexing thread" model
        // (lib/collection/src/optimizers_builder.rs:155): target ~10k vectors
        // at D=1024, fewer at higher dimensions.
        let merge_max_records = (40usize * 1024 * 1024)
            .saturating_div(dim.saturating_mul(4))
            .clamp(4_000, 25_000);

        Self {
            hot_capacity,
            warm_capacity,
            hnsw_threshold,
            full_scan_threshold_kb,
            merge_max_records,
            ..Self::FIXED
        }
    }

    pub fn merge_threshold_segments(&self) -> usize {
        self.merge_threshold_segments.max(2)
    }

    pub fn merge_max_records(&self) -> usize {
        // Always at least one hot segment worth of records, and never less
        // than 2 to avoid degenerate single-vector merges.
        self.merge_max_records.max(self.hot_capacity).max(2)
    }
}

impl Default for TierConfig {
    fn default() -> Self {
        Self::FIXED
    }
}

/// Top-level storage configuration.
#[derive(Debug, Clone)]
pub struct StoreConfig {
    pub dimension: usize,
    pub max_edges: usize,
    /// HNSW level-0 connectivity factor.  `M0 = max_edges * level0_factor`.
    pub level0_factor: usize,
    /// HNSW construction beam width.  If zero, falls back to
    /// `search_list_size.max(max_edges * 2)` for backward compatibility.
    pub ef_construction: usize,
    /// Search-time beam width (`ef`).  Used as the floor for the per-segment
    /// candidate pool.
    pub search_list_size: usize,
    pub outlier_count: usize,
    pub initial_capacity: usize,
    pub tier: TierConfig,
    /// Resource limits for background indexing work.
    pub optimizer_budget: OptimizerBudget,
    /// How often the background consolidation worker runs.  `None` disables it.
    pub auto_consolidation_interval: Option<Duration>,
}

impl StoreConfig {
    /// Effective HNSW construction beam width.
    pub fn ef_construction(&self) -> usize {
        let ef = if self.ef_construction == 0 {
            self.search_list_size.max(self.max_edges * 2)
        } else {
            self.ef_construction
        };
        ef.max(self.max_edges * 2)
    }

    /// Effective HNSW level-0 connectivity.
    pub fn m0(&self) -> usize {
        if self.level0_factor == 0 {
            self.max_edges * 2
        } else {
            self.max_edges * self.level0_factor
        }
    }

    /// Sensible defaults scaled for the given vector dimension.
    pub fn default_for_dimension(dimension: usize) -> Self {
        Self {
            dimension,
            tier: TierConfig::scaled_for_dimension(dimension),
            ..Self::default()
        }
    }
}

impl Default for StoreConfig {
    fn default() -> Self {
        Self {
            dimension: 768,
            max_edges: 32,
            level0_factor: 2,
            ef_construction: 200,
            search_list_size: 100,
            outlier_count: 0,
            initial_capacity: 1024,
            tier: TierConfig::default(),
            optimizer_budget: OptimizerBudget::default(),
            auto_consolidation_interval: Some(Duration::from_secs(60)),
        }
    }
}

/// A flush closure returned by segments.
pub type Flusher = Box<dyn FnOnce() -> crate::Result<()> + Send>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaled_thresholds_decrease_with_dimension() {
        let low_dim = TierConfig::scaled_for_dimension(8);
        let mid_dim = TierConfig::scaled_for_dimension(128);
        let high_dim = TierConfig::scaled_for_dimension(768);

        // Higher dimensions keep smaller appendable/plain segments because each
        // vector is more expensive to scan.
        assert!(low_dim.hot_capacity >= high_dim.hot_capacity);
        assert!(low_dim.hnsw_threshold >= high_dim.hnsw_threshold);

        // Mid-dimension values should sit between the extremes.
        assert!(mid_dim.hot_capacity <= low_dim.hot_capacity);
        assert!(mid_dim.hot_capacity >= high_dim.hot_capacity);
        assert!(mid_dim.hnsw_threshold <= low_dim.hnsw_threshold);
        assert!(mid_dim.hnsw_threshold >= high_dim.hnsw_threshold);
    }

    #[test]
    fn scaled_thresholds_are_clamped() {
        let tiny = TierConfig::scaled_for_dimension(1);
        assert!(tiny.hot_capacity <= 8192);
        assert!(tiny.hot_capacity >= 512);
        assert!(tiny.hnsw_threshold >= 512);
        assert!(tiny.warm_capacity <= 500_000);
        assert!(tiny.warm_capacity >= 2_048);

        let huge = TierConfig::scaled_for_dimension(8192);
        assert!(huge.hot_capacity >= 512);
        assert!(huge.hot_capacity <= 8192);
        assert!(huge.warm_capacity >= 2_048);
        assert!(huge.warm_capacity <= 500_000);
    }

    #[test]
    fn default_for_dimension_preserves_overrides() {
        let mut config = StoreConfig::default_for_dimension(256);
        config.max_edges = 32;
        config.search_list_size = 200;
        assert_eq!(config.dimension, 256);
        assert_eq!(config.max_edges, 32);
        assert_eq!(config.search_list_size, 200);
        assert_eq!(config.tier, TierConfig::scaled_for_dimension(256));
    }
}
