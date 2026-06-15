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

/// Per-tier sizing and quantization policy.
#[derive(Debug, Clone)]
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
    /// True if the Cold tier uses 1-bit sign quantization.
    pub cold_sign: bool,
    /// Access-score threshold above which a record is promoted back to Hot.
    pub hot_promote_threshold: f64,
    /// Access-score threshold below which a record in Hot is demoted.
    pub warm_demote_threshold: f64,
    /// Recency half-life in seconds for access scoring.
    pub recency_half_life_secs: u64,
}

impl Default for TierConfig {
    fn default() -> Self {
        Self {
            hot_capacity: 10_000,
            warm_capacity: 100_000,
            warm_bits: 8,
            warm_chunk_bytes: 16 * 1024 * 1024, // 16 MiB
            hnsw_threshold: 1000,
            cold_sign: true,
            hot_promote_threshold: 2.0,
            warm_demote_threshold: 0.5,
            recency_half_life_secs: 3600,
        }
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
    /// How often the background consolidation worker runs.  `None` disables it.
    pub auto_consolidation_interval: Option<Duration>,
}

impl StoreConfig {
    pub fn ef_construction(&self) -> usize {
        self.search_list_size.max(self.max_edges * 2)
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
            auto_consolidation_interval: Some(Duration::from_secs(60)),
        }
    }
}

/// A flush closure returned by segments.
pub type Flusher = Box<dyn FnOnce() -> crate::Result<()> + Send>;
