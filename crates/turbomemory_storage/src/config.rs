//! Configuration for the tiered storage engine.

use std::time::Duration;

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
        Self {
            max_concurrent_builds: 1,
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
    /// Bits used by the Warm tier scalar quantizer (1-8).
    pub warm_bits: u8,
    /// Byte size of a single Warm tier mmap chunk.
    pub warm_chunk_bytes: usize,
    /// Minimum number of records in a sealed segment before building an HNSW
    /// index. Smaller segments stay quantized/plain for efficiency.
    pub hnsw_threshold: usize,
    /// Maximum number of sealed HNSW segments before the merge optimizer combines
    /// them. Keeping this small improves recall and search latency.
    pub merge_threshold_segments: usize,
    /// Maximum number of records a single merge operation may combine. This
    /// limits peak memory during HNSW rebuilds.
    pub merge_max_records: usize,
    /// True if the Cold tier uses 1-bit sign quantization.
    pub cold_sign: bool,
    /// Access-score threshold above which a record is promoted back to Hot.
    pub hot_promote_threshold: f64,
    /// Access-score threshold below which a record in Hot is demoted.
    pub warm_demote_threshold: f64,
    /// Recency half-life in seconds for access scoring.
    pub recency_half_life_secs: u64,
}

impl TierConfig {
    /// Default fixed thresholds used when dimension-aware scaling is not desired.
    pub const FIXED: Self = Self {
        hot_capacity: 10_000,
        warm_capacity: 100_000,
        warm_bits: 8,
        warm_chunk_bytes: 16 * 1024 * 1024, // 16 MiB
        hnsw_threshold: 1000,
        merge_threshold_segments: 4,
        merge_max_records: 200_000,
        cold_sign: true,
        hot_promote_threshold: 2.0,
        warm_demote_threshold: 0.5,
        recency_half_life_secs: 3600,
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
        // Target ~16 MiB of 8-bit quantized vectors in the Warm tier.
        let warm_capacity = (16usize * 1024 * 1024)
            .saturating_div(dim)
            .clamp(2_048, 200_000);

        Self {
            hot_capacity,
            warm_capacity,
            hnsw_threshold,
            ..Self::FIXED
        }
    }

    pub fn merge_threshold_segments(&self) -> usize {
        self.merge_threshold_segments.max(2)
    }

    pub fn merge_max_records(&self) -> usize {
        self.merge_max_records.max(self.hot_capacity)
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
    pub fn ef_construction(&self) -> usize {
        self.search_list_size.max(self.max_edges * 2)
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
            max_edges: 16,
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
        assert!(tiny.warm_capacity <= 200_000);
        assert!(tiny.warm_capacity >= 2_048);

        let huge = TierConfig::scaled_for_dimension(8192);
        assert!(huge.hot_capacity >= 512);
        assert!(huge.hot_capacity <= 8192);
        assert!(huge.warm_capacity >= 2_048);
        assert!(huge.warm_capacity <= 200_000);
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
