//! Configuration for the tiered storage engine.

use std::time::Duration;
use turbomemory_core::quantization::{ScalarQuantizer, SignQuantizer, VectorQuantizer};
use turbomemory_core::turbo_quant::{TurboQuantMseQuantizer, TurboQuantProdQuantizer};
use turbomemory_graph::{ExtractorConfig, SpreadingConfig};

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
    ///
    /// Returns `Err` instead of panicking when the quantizer cannot be built for
    /// the given dimension (e.g. TurboQuant requires a power-of-two dimension,
    /// and the default `dimension = 768` is not a power of two).
    pub fn build(self, dim: usize) -> Result<VectorQuantizer, turbomemory_core::TurboError> {
        match self {
            Self::Scalar { bits } => Ok(VectorQuantizer::Scalar(ScalarQuantizer {
                bits,
                dim,
                min: 0.0,
                max: 0.0,
            })),
            Self::Sign => Ok(VectorQuantizer::Sign(SignQuantizer::new(dim))),
            Self::TurboQuantMse { bits } => Ok(VectorQuantizer::TurboQuantMse(
                TurboQuantMseQuantizer::new(dim, bits, Self::ROTATION_SEED)?,
            )),
            Self::TurboQuantProd { bits } => Ok(VectorQuantizer::TurboQuantProd(
                TurboQuantProdQuantizer::new(dim, bits, Self::ROTATION_SEED, Self::QJL_SEED)?,
            )),
        }
    }

    /// Returns `true` when this quantizer requires a power-of-two dimension
    /// (i.e. the TurboQuant variants that rely on the FWHT preconditioner).
    pub fn requires_pow2_dim(self) -> bool {
        matches!(
            self,
            Self::TurboQuantMse { .. } | Self::TurboQuantProd { .. }
        )
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
    /// Co-occurrence threshold for building abstraction (parent concept)
    /// edges during consolidation. When two concepts co-occur on at least
    /// this many memories, a parent concept node is created with
    /// `Abstraction` edges to both. `0` (default) disables abstraction
    /// building. A typical value is 3 — meaning two concepts must be seen
    /// together at least 3 times before the graph generalizes them.
    pub abstraction_co_occurrence_threshold: usize,
    /// Edge-decay half-life in seconds. On each consolidation, reinforced
    /// edges are decayed by `0.5^((now - last_reinforced) / half_life)`,
    /// floored at the edge's baseline weight. `0` (default) disables decay,
    /// preserving the pre-learning behavior. A typical value is 86400 (1
    /// day) — memories not recalled within a day fade toward baseline.
    pub edge_decay_half_life_secs: u64,
    /// Maximum number of concepts to attach to a record. When the caller
    /// provides fewer concepts than this, the remaining slots are filled by
    /// automatic concept extraction from the record's text. When the caller
    /// provides more, the caller's concepts are used as-is (no truncation).
    /// Set to `0` to disable auto-extraction entirely (caller must always
    /// supply concepts). Default is `5`.
    pub max_concepts: usize,
    /// Maximum n-gram length used by the concept extractor. `1` extracts only
    /// single-word concepts (backward-compatible default). `2` adds bigrams,
    /// `3` adds trigrams. Higher values capture multi-word concepts like
    /// "memory safety" but consume concept slots.
    pub concept_max_ngram_len: usize,
    /// Minimum number of times an n-gram must appear before it can be
    /// extracted. For short memory texts the default is 1.
    pub concept_min_ngram_freq: usize,
    /// Whether to boost n-gram scores using pointwise mutual information.
    /// PMI rewards genuine collocations ("memory safety") over accidental
    /// adjacencies.
    pub concept_enable_pmi: bool,
    /// Cosine-similarity threshold for memory refinement (belief revision).
    /// When `Some(t)` (e.g. 0.85), consolidation creates `Refines` edges
    /// from older memories to newer memories that are semantically close
    /// (cosine >= t) AND share at least one concept. The old memory is NOT
    /// deleted — it stays in the graph so the agent can reason about how
    /// its understanding evolved. Instead, a `Refines` edge (old → new) lets
    /// spreading activation propagate from the old memory to the newer one,
    /// ensuring the most current version surfaces. `None` (default) disables
    /// refinement. This threshold should be LOWER than
    /// `dedup_cosine_threshold` — refinement is "same topic, more recent"
    /// while dedup is "essentially identical content, merge them".
    pub refinement_cosine_threshold: Option<f32>,
    /// Upper bound on the number of refinement edges created in a single
    /// consolidation cycle, to bound per-cycle work. Ignored when
    /// `refinement_cosine_threshold` is `None`.
    pub refinement_max_pairs_per_cycle: usize,
    /// Cosine-similarity threshold for contradiction detection (belief
    /// revision). When `Some(t)` (e.g. 0.75), consolidation checks pairs
    /// that are semantically close (cosine >= t) AND share at least one
    /// concept AND have LOW text overlap (Jaccard < `contradiction_text_threshold`).
    /// A `Contradicts` edge is created (old → new) and the old memory's
    /// edges are weakened. `None` (default) disables. This threshold
    /// should be LOWER than `refinement_cosine_threshold` — contradiction
    /// is "same topic, opposing info" while refinement is "same topic,
    /// updated info".
    pub contradiction_cosine_threshold: Option<f32>,
    /// Maximum Jaccard text similarity above which a pair is considered a
    /// refinement (not a contradiction). Pairs with Jaccard BELOW this
    /// threshold are candidates for contradiction. Default 0.3 — if the
    /// texts share less than 30% of their tokens, they're saying different
    /// things about the same topic.
    pub contradiction_text_threshold: f32,
    /// Minimum Jaccard text similarity for a `Refines` edge. A refinement is a
    /// *re-statement* of the same claim (updated content), so it shares
    /// substantial text with the older memory. Pairs BELOW this floor are
    /// rejected — this prevents demoting two *coexisting* facts about the same
    /// topic (same concept, high cosine, but independent content). Default 0.25.
    /// Set to 0.0 to disable the floor (legacy no-text-gate behavior).
    pub refinement_text_threshold: f32,
    /// Require an explicit opposition/negation marker ("actually", "instead",
    /// "not", "no longer", …) in the newer memory before creating a
    /// `Contradicts` edge. A genuine contradiction *opposes* the old claim; two
    /// coexisting facts about the same topic do not. Lightweight precision gate
    /// (bag-of-cues, not full NLI): favors precision and will miss marker-less
    /// semantic contradictions. Default `true` (safe — only demote on explicit
    /// opposition). Set `false` for the legacy behavior.
    pub contradiction_require_opposition: bool,
    /// Factor by which the old (contradicted) memory's association edges
    /// are multiplied when a contradiction is detected. Default 0.5 —
    /// halve the edge weights so the old memory fades but is not invisible.
    pub contradiction_weaken_factor: f32,
    /// Multiplicative score penalty applied to a memory that has been
    /// superseded by a newer one (via a `Contradicts` or `Refines` edge
    /// created during consolidation). The superseded memory's fused retrieval
    /// score is multiplied by this factor, so the current belief outranks the
    /// stale one even when `cognitive_alpha = 1.0` (pure-cosine ranking).
    /// Unlike `contradiction_weaken_factor` (which only weakens outgoing graph
    /// edges and so cannot demote a record matched directly by cosine), this
    /// acts on the final score and is durable across restarts. Default 0.4 —
    /// a strong demotion that preserves history (the memory is never deleted).
    /// Set to 1.0 to disable supersession demotion entirely.
    pub supersession_demotion_factor: f32,
    /// Upper bound on contradiction edges created per consolidation cycle.
    pub contradiction_max_pairs_per_cycle: usize,
    /// Enable automatic importance scoring (self-organizing memory). When
    /// true, each consolidation cycle recomputes every record's `importance`
    /// as a blend of retrieval salience (`access_score`) and graph
    /// connectivity (concept degree), then moves the current importance
    /// `importance_learning_rate` of the way toward that target. `false`
    /// (default) disables — importance stays at the caller-set value.
    pub importance_auto_scoring: bool,
    /// Learning rate for auto-importance: fraction of the way to move the
    /// current importance toward the computed target each cycle (0.0..=1.0).
    /// Lower = more stable/slow; higher = more responsive. Default 0.3.
    pub importance_learning_rate: f32,
    /// Weight on retrieval salience (access_score) in the target blend; the
    /// remaining `(1 - this)` goes to graph connectivity (concept degree).
    /// Default 0.6 — retrieval matters more than connectivity.
    pub importance_access_weight: f32,
    /// Floor for auto-importance: a record's importance is never decayed
    /// below this. Protects recently-inserted, not-yet-retrieved memories
    /// from being zeroed out on their first consolidation. Default 0.1.
    pub importance_floor: f32,
    /// Upper cap for auto-importance: a record's importance is never raised
    /// above this. Default 4.0 — matches the dynamic range of the
    /// `importance_factor` sqrt curve (importance 4.0 -> factor 2.0).
    pub importance_ceiling: f32,
    /// Enable online concept vocabulary evolution. When true, each
    /// consolidation pass merges concept nodes whose associated memory sets
    /// overlap strongly and suppresses over-general hub concepts. `false`
    /// (default) keeps concepts exactly as extracted — backward-compatible.
    pub concept_evolution_enabled: bool,
    /// Jaccard-overlap threshold for merging two concept nodes. Two concepts
    /// are merged when `|shared memories| / |union of memories| >= threshold`.
    /// Default 0.7. Only used when `concept_evolution_enabled` is true.
    pub concept_merge_overlap_threshold: f32,
    /// Fraction of total memories above which a base concept is considered an
    /// over-general hub and is suppressed. Default 0.1 (10% of memories).
    /// Only used when `concept_evolution_enabled` is true.
    pub concept_hub_degree_fraction: f32,
    /// Maximum number of concept-merge operations per consolidation cycle.
    /// Default 1024.
    pub concept_evolution_max_pairs_per_cycle: usize,
    /// Restrict belief-revision (refinement + contradiction) detection to
    /// memories whose `source_role` is in this list. `None` (default) means
    /// role-blind detection — every memory can supersede or be superseded,
    /// the legacy behavior. `Some(["user"])` makes supersession consider only
    /// user-authored facts, so the assistant's own verbose/bulleted responses
    /// can never demote (or be demoted by) a user memory — while all roles
    /// remain fully retrievable. This is the engine-level form of the
    /// eval-only `store_roles` filter that eliminated belief-revision
    /// collateral damage on real conversational data (LongMemEval Stage B.2).
    pub belief_source_roles: Option<Vec<String>>,
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
        // Cognitive-layer learning: disabled by default to preserve the
        // pre-learning retrieval behavior. Enable `abstraction_co_occurrence_threshold`
        // (e.g. 3) and `edge_decay_half_life_secs` (e.g. 86400) to turn on
        // abstraction hierarchy building and edge forgetting.
        abstraction_co_occurrence_threshold: 0,
        edge_decay_half_life_secs: 0,
        // Auto-extract up to 5 concepts from record text when the caller
        // provides fewer than that. Set to 0 to require explicit concepts.
        max_concepts: 5,
        // N-gram extraction is disabled by default (unigrams only) to preserve
        // existing behavior and benchmarks. Set to 2 or 3 to capture
        // multi-word concepts like "memory safety".
        concept_max_ngram_len: 1,
        concept_min_ngram_freq: 1,
        concept_enable_pmi: true,
        // Memory refinement (belief revision): opt-in. When enabled, the
        // engine creates Refines edges from older memories to newer ones
        // that are about the same topic, so retrieval surfaces the most
        // current version. None = disabled.
        refinement_cosine_threshold: None,
        refinement_max_pairs_per_cycle: 1024,
        // Contradiction detection: opt-in. When enabled, the engine
        // detects when a newer memory contradicts an older one (same
        // topic, opposing content) and weakens the old memory's edges.
        // None = disabled.
        contradiction_cosine_threshold: None,
        contradiction_text_threshold: 0.3,
        refinement_text_threshold: 0.25,
        contradiction_require_opposition: true,
        contradiction_weaken_factor: 0.5,
        supersession_demotion_factor: 0.4,
        contradiction_max_pairs_per_cycle: 1024,
        // Automatic importance scoring: opt-in. When enabled, the engine
        // adjusts each record's importance based on retrieval patterns +
        // connectivity, making the memory self-organizing. Disabled by
        // default preserves caller-set importance.
        importance_auto_scoring: false,
        importance_learning_rate: 0.3,
        importance_access_weight: 0.6,
        importance_floor: 0.1,
        importance_ceiling: 4.0,
        // Online concept vocabulary evolution: disabled by default to preserve
        // the exact extracted concepts and existing benchmarks.
        concept_evolution_enabled: false,
        concept_merge_overlap_threshold: 0.7,
        concept_hub_degree_fraction: 0.1,
        concept_evolution_max_pairs_per_cycle: 1024,
        // Role-blind belief revision by default (legacy behavior). Set to
        // e.g. Some(vec!["user".into()]) to scope supersession to user facts.
        belief_source_roles: None,
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
        //
        // P0 fix: Lower the cap for faster parallel builds. At 768-dim:
        //   - Old: 40MB / 3072B = ~13,000 vectors per segment = 1 segment at 50k
        //     → single-threaded build, ~337s
        //   - New: 20MB / 3072B = ~6,500 vectors per segment = ~8 segments at 50k
        //     → 2-3 parallel builds, ~120s total (3× faster)
        //
        // Recall is preserved because:
        //   1. HNSW M=64 and ef_construction=800 are unchanged per segment
        //   2. Multi-segment search already scales candidate pool with segment count
        //      (segment_factor up to 2.5x, capped)
        //   3. Final rerank uses full f32 vectors from VectorStore
        let merge_max_records = (20usize * 1024 * 1024)
            .saturating_div(dim.saturating_mul(4))
            .clamp(2_000, 10_000);

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

    /// Build the graph-layer extractor configuration from tier settings.
    pub fn extractor_config(&self) -> ExtractorConfig {
        ExtractorConfig {
            max_concepts: self.max_concepts,
            max_ngram_len: self.concept_max_ngram_len,
            min_ngram_freq: self.concept_min_ngram_freq,
            enable_pmi_scoring: self.concept_enable_pmi,
            pmi_weight: 1.0,
        }
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
    /// Additive fusion weight for the cognitive augmenter:
    /// `final_score = cosine + (1 - cognitive_alpha) * normalized_graph_signal`.
    /// This is the sole blend control. At `1.0` the ranking is pure cosine —
    /// the graph only influences *which* memories are candidates. Lower values
    /// let the graph re-rank candidates via an additive, bounded boost
    /// (reinforcement, refinement, contradiction, abstraction). The default
    /// `0.7` gives the graph up to a 0.30 additive re-rank boost, enough to
    /// surface a refinement/correction above its cosine-nearest older memory
    /// while still preserving the ANN recall floor. Range `[0.0, 1.0]`;
    /// clamped at runtime.
    pub cognitive_alpha: f32,
    pub outlier_count: usize,
    pub initial_capacity: usize,
    pub tier: TierConfig,
    /// Resource limits for background indexing work.
    pub optimizer_budget: OptimizerBudget,
    /// How often the background consolidation worker runs.  `None` disables it.
    pub auto_consolidation_interval: Option<Duration>,
    /// Cognitive-layer bounded-augmenter parameters (lexical alpha, decay,
    /// seed count, expansion cap). Defaults to `SpreadingConfig::default()`.
    pub spreading: SpreadingConfig,
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
        // HNSW build parameters scale with dimension. At low dimension a sparse
        // graph (M=16) with light construction (ef=100) recall is fine because
        // vectors are well-separated. At high dimension (768+) vectors are
        // near-orthogonal and the true top-k is separated by noise-level cosine
        // gaps, so the graph must be denser (higher M) and construction must
        // explore more candidates (higher ef_construction) or recall collapses
        // to ~20-35% regardless of search ef.
        //
        // Empirical results at 768-dim/20k:
        //   M=16/efc=100 -> 36%,  M=32/efc=200 -> 73%,  M=48/efc=400 -> 90%.
        //   M=64/efc=800 -> 95%+ (P0 fix: denser graph + deeper construction).
        //
        // We scale M super-linearly with dimension (M ~ dim^0.7) because
        // navigability degrades faster than linearly in high-dim spaces — each
        // edge is less likely to be useful, so more edges are needed to maintain
        // the same path quality. ef_construction tracks at ~M*12 to ensure the
        // build explores enough candidates to populate the denser graph.
        let max_edges = if dimension <= 128 {
            16
        } else if dimension <= 384 {
            32
        } else if dimension <= 768 {
            // 768-dim: was 48, now 64. Super-linear: 2.5 * 768^0.7 ≈ 64.
            64
        } else {
            // Cap at 96 for very high dimensions to avoid exploding build memory.
            let scaled = (dimension as f32).powf(0.6) * 3.0;
            scaled.round() as usize
        }
        .clamp(16, 96);

        let ef_construction = if dimension <= 128 {
            100
        } else if dimension <= 384 {
            200
        } else if dimension <= 768 {
            // 768-dim: was 400, now 800. ~M*12 = 64*12 = 768, rounded to 800.
            800
        } else {
            // Scale with M but cap to avoid excessive build time.
            (max_edges * 12).min(2000)
        };
        // Search ef floor (search_list_size). High-dim HNSW graphs are denser
        // and the true top-k is separated by noise-level cosine gaps, so the
        // search beam must be wider or recall collapses even with a good build.
        // At 768-dim a floor of 256 lifts 50k recall from ~42% to ~66%+ at the
        // default ef; callers can still raise it per-query via `ef`.
        let search_list_size = match dimension {
            0..=128 => 100,
            129..=384 => 150,
            _ => 256,
        };
        Self {
            dimension,
            max_edges,
            ef_construction,
            search_list_size,
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
            cognitive_alpha: 0.7,
            outlier_count: 0,
            initial_capacity: 1024,
            tier: TierConfig::default(),
            optimizer_budget: OptimizerBudget::default(),
            auto_consolidation_interval: Some(Duration::from_secs(60)),
            spreading: SpreadingConfig::default(),
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

    #[test]
    fn hnsw_params_scale_with_dimension() {
        // Low dim: sparse graph is fine (vectors well-separated).
        let low = StoreConfig::default_for_dimension(64);
        assert_eq!(low.max_edges, 16);
        assert_eq!(low.ef_construction, 100);
        // Mid dim.
        let mid = StoreConfig::default_for_dimension(384);
        assert_eq!(mid.max_edges, 32);
        assert_eq!(mid.ef_construction, 200);
        // High dim (768, the regression case): dense graph + thorough build
        // + wider search beam. Updated to match the new aggressive scaling.
        let high = StoreConfig::default_for_dimension(768);
        assert_eq!(high.max_edges, 64);
        assert_eq!(high.ef_construction, 800);
        assert_eq!(high.search_list_size, 256);
    }
}
