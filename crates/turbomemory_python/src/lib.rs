//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use turbomemory_graph::{CognitiveCompressor, CompressedCognitiveState, DeterministicCompressor};
use turbomemory_storage::config::{QuantizerKind, StoreConfig};
use turbomemory_storage::engine::{GistCompressor, StorageEngine, SupersessionKind};

/// Map storage errors to specific Python exception types.
fn storage_err(e: turbomemory_storage::StorageError) -> PyErr {
    use turbomemory_storage::StorageError as E;
    match e {
        E::DuplicateId(_) | E::DimensionMismatch | E::InvalidArgument(_) => {
            PyValueError::new_err(e.to_string())
        }
        // Caller errors that arrive wrapped in the core error type get the
        // same ValueError treatment as their storage-level equivalents — the
        // API crate already classifies these as 400 / InvalidArgument.
        E::Core(
            turbomemory_core::TurboError::DimensionMismatch { .. }
            | turbomemory_core::TurboError::InvalidArgument(_)
            | turbomemory_core::TurboError::ZeroNorm
            | turbomemory_core::TurboError::NonFinite,
        ) => PyValueError::new_err(e.to_string()),
        E::NotFound(_) => PyKeyError::new_err(e.to_string()),
        _ => PyRuntimeError::new_err(e.to_string()),
    }
}

/// A 1-D f32 input that borrows a contiguous numpy array when possible and
/// only allocates for lists, non-contiguous arrays, or non-f32 dtypes.
enum F32Input<'py> {
    View(numpy::PyReadonlyArray1<'py, f32>),
    Owned(Vec<f32>),
}

impl F32Input<'_> {
    fn as_slice(&self) -> &[f32] {
        match self {
            // Constructed only when `as_slice` already succeeded, so this is
            // guaranteed contiguous.
            F32Input::View(arr) => arr.as_slice().expect("contiguous view"),
            F32Input::Owned(v) => v.as_slice(),
        }
    }
}

/// Borrow a 1-D f32 vector from a Python object (list, tuple, or numpy array).
///
/// Zero-copy for a contiguous `float32` ndarray; copies otherwise.
fn extract_f32_input<'py>(obj: &Bound<'py, PyAny>) -> PyResult<F32Input<'py>> {
    if let Ok(arr) = numpy::PyReadonlyArray1::<f32>::extract_bound(obj) {
        if arr.as_slice().is_ok() {
            return Ok(F32Input::View(arr));
        }
        // Non-contiguous f32 array: materialize a contiguous copy.
        return Ok(F32Input::Owned(arr.as_array().to_vec()));
    }
    if let Ok(v) = obj.extract::<Vec<f32>>() {
        return Ok(F32Input::Owned(v));
    }
    if obj.hasattr("tolist")? {
        let list_obj = obj.call_method0("tolist")?;
        return Ok(F32Input::Owned(list_obj.extract::<Vec<f32>>()?));
    }
    Err(PyValueError::new_err(
        "embedding must be a sequence or numpy array of f32",
    ))
}

/// A 2-D f32 input that borrows a contiguous numpy array when possible.
enum F32Matrix<'py> {
    View {
        arr: numpy::PyReadonlyArray2<'py, f32>,
        cols: usize,
    },
    Owned(Vec<Vec<f32>>),
}

impl F32Matrix<'_> {
    /// Per-row slices suitable for the engine's `&[&[f32]]` batch API. Borrows
    /// directly from the numpy buffer for the contiguous fast path.
    fn rows(&self) -> Vec<&[f32]> {
        match self {
            F32Matrix::View { arr, cols } => {
                let flat = arr.as_slice().expect("contiguous view");
                if *cols == 0 {
                    Vec::new()
                } else {
                    flat.chunks_exact(*cols).collect()
                }
            }
            F32Matrix::Owned(rows) => rows.iter().map(|r| r.as_slice()).collect(),
        }
    }
}

/// Borrow a 2-D f32 matrix from a Python object (list-of-lists or 2-D numpy array).
///
/// Zero-copy for a C-contiguous `float32` ndarray; copies otherwise.
fn extract_f32_matrix<'py>(obj: &Bound<'py, PyAny>) -> PyResult<F32Matrix<'py>> {
    if let Ok(arr) = numpy::PyReadonlyArray2::<f32>::extract_bound(obj) {
        let shape = arr.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("embeddings must be 2-D"));
        }
        let cols = shape[1];
        if arr.as_slice().is_ok() {
            return Ok(F32Matrix::View { arr, cols });
        }
        // Non-contiguous: materialize row-major copies.
        let owned: Vec<Vec<f32>> = arr
            .as_array()
            .rows()
            .into_iter()
            .map(|r| r.to_vec())
            .collect();
        return Ok(F32Matrix::Owned(owned));
    }
    if let Ok(m) = obj.extract::<Vec<Vec<f32>>>() {
        return Ok(F32Matrix::Owned(m));
    }
    if obj.hasattr("tolist")? {
        let list_obj = obj.call_method0("tolist")?;
        return Ok(F32Matrix::Owned(list_obj.extract::<Vec<Vec<f32>>>()?));
    }
    Err(PyValueError::new_err(
        "embeddings must be a 2-D sequence or numpy array of f32",
    ))
}

/// Parse a Python quantizer specifier into a [`QuantizerKind`].
///
/// Accepted forms:
/// - `"scalar"` or `"scalar<N>"` -> `QuantizerKind::Scalar { bits: N }`
/// - `"sign"` -> `QuantizerKind::Sign`
/// - `"turbo_mse"` or `"turbo_mse<N>"` -> `QuantizerKind::TurboQuantMse { bits: N }`
/// - `"turbo_prod"` or `"turbo_prod<N>"` -> `QuantizerKind::TurboQuantProd { bits: N }`
fn parse_quantizer_kind(spec: Option<String>, default: QuantizerKind) -> PyResult<QuantizerKind> {
    let spec = match spec {
        Some(s) => s,
        None => return Ok(default),
    };
    let spec = spec.trim().to_lowercase();
    if spec.is_empty() {
        return Ok(default);
    }

    fn extract_bits(prefix: &str, spec: &str) -> PyResult<u8> {
        if spec == prefix {
            return Err(PyValueError::new_err(format!(
                "{prefix} quantizer requires a bit width, e.g. {prefix}2"
            )));
        }
        if let Some(rest) = spec.strip_prefix(prefix) {
            rest.parse::<u8>()
                .map_err(|_| PyValueError::new_err(format!("invalid bit width in '{spec}'")))
        } else {
            Err(PyValueError::new_err(format!("unknown quantizer '{spec}'")))
        }
    }

    if spec.starts_with("scalar") {
        Ok(QuantizerKind::Scalar {
            bits: extract_bits("scalar", &spec)?,
        })
    } else if spec == "sign" {
        Ok(QuantizerKind::Sign)
    } else if spec == "rabitq" || spec == "rabitq1" {
        Ok(QuantizerKind::RaBitQ { bits: 1 })
    } else if spec == "rabitq2" {
        Ok(QuantizerKind::RaBitQ { bits: 2 })
    } else if spec.starts_with("rabitq") {
        Ok(QuantizerKind::RaBitQ {
            bits: extract_bits("rabitq", &spec)?,
        })
    } else if spec.starts_with("turbo_prod") {
        Ok(QuantizerKind::TurboQuantProd {
            bits: extract_bits("turbo_prod", &spec)?,
        })
    } else if spec.starts_with("turbo_mse") {
        Ok(QuantizerKind::TurboQuantMse {
            bits: extract_bits("turbo_mse", &spec)?,
        })
    } else {
        Err(PyValueError::new_err(format!(
            "unknown quantizer '{spec}'; expected scalar<N>, sign, rabitq<N>, turbo_mse<N>, or turbo_prod<N>"
        )))
    }
}

/// Validate an optional JSON payload string and return it as-is.
fn parse_payload(payload: Option<String>) -> PyResult<Option<String>> {
    match payload {
        Some(s) if !s.trim().is_empty() => {
            serde_json::from_str::<serde_json::Value>(&s)
                .map_err(|e| PyValueError::new_err(format!("invalid payload JSON: {e}")))?;
            Ok(Some(s))
        }
        _ => Ok(None),
    }
}

#[pyclass(name = "MemoryEngine")]
pub struct PyMemoryEngine {
    /// `None` once `close()` has run. Dropping the `Arc` is what releases the
    /// database lock, mmaps, and worker threads, so `close()` takes it out
    /// rather than leaving release to Python's garbage collector.
    ///
    /// Behind a mutex so `close()` works through a shared reference: every
    /// method clones the `Arc` and releases the mutex before doing any work,
    /// so `close()` from one thread cannot collide with a call that is still
    /// running on another (which simply finishes on its own clone).
    inner: Mutex<Option<Arc<StorageEngine>>>,
}

impl PyMemoryEngine {
    /// The live engine, or `RuntimeError` after `close()`.
    fn engine(&self) -> PyResult<Arc<StorageEngine>> {
        self.handle()
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("engine is closed"))
    }

    fn handle(&self) -> std::sync::MutexGuard<'_, Option<Arc<StorageEngine>>> {
        // Nothing panics while this is held, but a poisoned handle must
        // still be closable.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[pymethods]
impl PyMemoryEngine {
    #[new]
    #[pyo3(signature = (
        db_path,
        dimension,
        max_edges=None,
        search_list_size=None,
        outlier_count=0,
        initial_capacity=None,
        warm_quantizer=None,
        warm_bits=None,
        cold_quantizer=None,
        hot_capacity=None,
        warm_capacity=None,
        hnsw_threshold=None,
        ef_construction=None,
        level0_factor=None,
        full_scan_threshold_kb=None,
        max_records=None,
        evict_score_floor=None,
        dedup_cosine_threshold=None,
        dedup_max_pairs_per_cycle=None,
        auto_consolidation_secs=60,
        spreading_decay=None,
        spreading_iterations=None,
        abstraction_co_occurrence_threshold=None,
        edge_decay_half_life_secs=None,
        max_concepts=None,
        concept_max_ngram_len=None,
        concept_min_ngram_freq=None,
        concept_enable_pmi=None,
        refinement_cosine_threshold=None,
        refinement_max_pairs_per_cycle=None,
        cognitive_alpha=None,
        contradiction_cosine_threshold=None,
        contradiction_text_threshold=None,
        refinement_text_threshold=None,
        contradiction_require_opposition=None,
        contradiction_weaken_factor=None,
        supersession_demotion_factor=None,
        contradiction_max_pairs_per_cycle=None,
        importance_auto_scoring=None,
        importance_learning_rate=None,
        importance_access_weight=None,
        importance_floor=None,
        importance_ceiling=None,
        concept_evolution_enabled=None,
        concept_merge_overlap_threshold=None,
        concept_hub_degree_fraction=None,
        concept_evolution_max_pairs_per_cycle=None,
        belief_source_roles=None,
        defer_supersession_commit=None,
        exclude_superseded=None,
        access_aware_eviction=None,
        actr_activation=None,
        actr_decay=None,
        actr_history=None,
        incremental_supersession_detection=None,
        gist_before_evict=None,
        gist_chunk_facts=None,
        seed_hops_from=None,
        expansion_max_candidates=None,
        concept_expansion=None,
        temporal_recency_weight=None,
        sync_writes=None,
        gpu_exact_search=None,
        gpu_exact_min_records=None,
        gpu_memory_budget_mb=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        db_path: &str,
        dimension: usize,
        max_edges: Option<usize>,
        search_list_size: Option<usize>,
        outlier_count: usize,
        initial_capacity: Option<usize>,
        warm_quantizer: Option<String>,
        warm_bits: Option<u8>,
        cold_quantizer: Option<String>,
        hot_capacity: Option<usize>,
        warm_capacity: Option<usize>,
        hnsw_threshold: Option<usize>,
        ef_construction: Option<usize>,
        level0_factor: Option<usize>,
        full_scan_threshold_kb: Option<usize>,
        max_records: Option<usize>,
        evict_score_floor: Option<f64>,
        dedup_cosine_threshold: Option<f32>,
        dedup_max_pairs_per_cycle: Option<usize>,
        auto_consolidation_secs: u64,
        spreading_decay: Option<f32>,
        spreading_iterations: Option<usize>,
        abstraction_co_occurrence_threshold: Option<usize>,
        edge_decay_half_life_secs: Option<u64>,
        max_concepts: Option<usize>,
        concept_max_ngram_len: Option<usize>,
        concept_min_ngram_freq: Option<usize>,
        concept_enable_pmi: Option<bool>,
        refinement_cosine_threshold: Option<f32>,
        refinement_max_pairs_per_cycle: Option<usize>,
        cognitive_alpha: Option<f32>,
        contradiction_cosine_threshold: Option<f32>,
        contradiction_text_threshold: Option<f32>,
        refinement_text_threshold: Option<f32>,
        contradiction_require_opposition: Option<bool>,
        contradiction_weaken_factor: Option<f32>,
        supersession_demotion_factor: Option<f32>,
        contradiction_max_pairs_per_cycle: Option<usize>,
        importance_auto_scoring: Option<bool>,
        importance_learning_rate: Option<f32>,
        importance_access_weight: Option<f32>,
        importance_floor: Option<f32>,
        importance_ceiling: Option<f32>,
        concept_evolution_enabled: Option<bool>,
        concept_merge_overlap_threshold: Option<f32>,
        concept_hub_degree_fraction: Option<f32>,
        concept_evolution_max_pairs_per_cycle: Option<usize>,
        belief_source_roles: Option<Vec<String>>,
        defer_supersession_commit: Option<bool>,
        exclude_superseded: Option<bool>,
        access_aware_eviction: Option<bool>,
        actr_activation: Option<bool>,
        actr_decay: Option<f64>,
        actr_history: Option<usize>,
        incremental_supersession_detection: Option<bool>,
        gist_before_evict: Option<bool>,
        gist_chunk_facts: Option<usize>,
        seed_hops_from: Option<usize>,
        expansion_max_candidates: Option<usize>,
        concept_expansion: Option<bool>,
        temporal_recency_weight: Option<f32>,
        sync_writes: Option<bool>,
        gpu_exact_search: Option<bool>,
        gpu_exact_min_records: Option<usize>,
        gpu_memory_budget_mb: Option<usize>,
    ) -> PyResult<Self> {
        let mut config = StoreConfig::default_for_dimension(dimension);
        if let Some(me) = max_edges {
            config.max_edges = me;
        }
        if let Some(sls) = search_list_size {
            config.search_list_size = sls;
        }
        config.outlier_count = outlier_count;
        if let Some(cap) = initial_capacity {
            config.initial_capacity = cap.max(1024);
        }

        // Resolve warm quantizer.  An explicit warm_quantizer string wins over
        // warm_bits; when neither is given the default scalar quantizer is kept.
        if warm_quantizer.is_some() {
            config.tier.warm_quantizer =
                parse_quantizer_kind(warm_quantizer, config.tier.warm_quantizer)?;
        } else if let Some(bits) = warm_bits {
            config.tier.warm_quantizer = QuantizerKind::Scalar { bits };
        }

        config.tier.cold_quantizer =
            parse_quantizer_kind(cold_quantizer, config.tier.cold_quantizer)?;

        if let Some(cap) = hot_capacity {
            config.tier.hot_capacity = cap;
        }
        if let Some(cap) = warm_capacity {
            config.tier.warm_capacity = cap;
        }
        if let Some(th) = hnsw_threshold {
            config.tier.hnsw_threshold = th;
        }
        if let Some(ef) = ef_construction {
            config.ef_construction = ef;
        }
        if let Some(lf) = level0_factor {
            config.level0_factor = lf;
        }
        if let Some(fs) = full_scan_threshold_kb {
            config.tier.full_scan_threshold_kb = fs;
        }

        // Bounded-storage eviction and semantic dedup are opt-in; leaving these
        // unset preserves the default unbounded, no-dedup behavior.
        config.tier.max_records = max_records;
        config.tier.evict_score_floor = evict_score_floor;
        config.tier.dedup_cosine_threshold = dedup_cosine_threshold;
        if let Some(mp) = dedup_max_pairs_per_cycle {
            config.tier.dedup_max_pairs_per_cycle = mp;
        }

        // 0 disables background consolidation entirely; otherwise it runs on
        // the given interval. Disabling is useful for benchmarks and for
        // workloads that drive consolidation manually via trigger_consolidation.
        config.auto_consolidation_interval = if auto_consolidation_secs == 0 {
            None
        } else {
            Some(Duration::from_secs(auto_consolidation_secs))
        };

        // Cognitive-layer tuning (all optional, defaults preserved when None).
        // The cognitive layer is a bounded augmenter: ANN candidates form a
        // recall floor and a single 1-hop graph expansion can only add
        // candidates / apply a small additive boost.
        // - spreading_decay: energy decay applied to graph-discovered (1-hop)
        //   candidates. Default 0.5.
        // - spreading_iterations: number of expansion hops. 0 disables graph
        //   expansion (pure ANN + BM25); 1 (default) is the balanced setting.
        // - seed_hops_from: how many top ANN seeds to expand from. Default 10.
        // - expansion_max_candidates: cap on candidates added by expansion.
        //   Default 50.
        // - abstraction_co_occurrence_threshold: enable abstraction hierarchy
        //   building. 0 (default) disables. A value of 3 means two concepts
        //   must co-occur on >= 3 memories before a parent concept is created.
        // - edge_decay_half_life_secs: enable edge forgetting. 0 (default)
        //   disables. A value of 86400 (1 day) means unrehearsed reinforced
        //   edges fade toward baseline with a 1-day half-life.
        if let Some(decay) = spreading_decay {
            config.spreading.decay = decay;
        }
        if let Some(iters) = spreading_iterations {
            config.spreading.iterations = iters;
        }
        if let Some(shf) = seed_hops_from {
            config.spreading.seed_hops_from = shf;
        }
        if let Some(emc) = expansion_max_candidates {
            config.spreading.expansion_max_candidates = emc;
        }
        if let Some(ce) = concept_expansion {
            config.spreading.concept_expansion = ce;
        }
        if let Some(th) = abstraction_co_occurrence_threshold {
            config.tier.abstraction_co_occurrence_threshold = th;
        }
        if let Some(hl) = edge_decay_half_life_secs {
            config.tier.edge_decay_half_life_secs = hl;
        }
        // - max_concepts: how many concepts to attach per record. Caller
        //   concepts are used first; remaining slots filled by auto-extraction
        //   from text. Set to 0 to disable extraction. Default 5.
        // - refinement_cosine_threshold: enable memory evolution. When two
        //   memories are about the same topic (cosine >= threshold AND share
        //   a concept), a Refines edge lets retrieval surface the newer one.
        //   None (default) disables. Should be LOWER than
        //   dedup_cosine_threshold — refinement is "same topic, more recent"
        //   while dedup is "essentially identical, merge".
        // - refinement_max_pairs_per_cycle: cap on Refines edges per
        //   consolidation. Default 1024.
        if let Some(mc) = max_concepts {
            config.tier.max_concepts = mc;
        }
        if let Some(n) = concept_max_ngram_len {
            config.tier.concept_max_ngram_len = n.max(1);
        }
        if let Some(n) = concept_min_ngram_freq {
            config.tier.concept_min_ngram_freq = n.max(1);
        }
        if let Some(on) = concept_enable_pmi {
            config.tier.concept_enable_pmi = on;
        }
        config.tier.refinement_cosine_threshold = refinement_cosine_threshold;
        if let Some(rm) = refinement_max_pairs_per_cycle {
            config.tier.refinement_max_pairs_per_cycle = rm;
        }
        // - cognitive_alpha: additive fusion weight for cognitive search.
        //   final_score = cosine + (1 - cognitive_alpha) * normalized_graph_delta.
        //   1.0 = pure cosine (graph only chooses candidates). Lower values
        //   allow reinforcement/refinement/abstraction to add a bounded boost.
        if let Some(ca) = cognitive_alpha {
            config.cognitive_alpha = ca;
        }
        // - contradiction_cosine_threshold: enable belief revision. When a
        //   newer memory contradicts an older one (cosine >= threshold AND
        //   share a concept AND low text overlap), a Contradicts edge is
        //   created (old -> new) and the old memory's edges are weakened.
        //   None (default) disables. Should be LOWER than
        //   refinement_cosine_threshold — contradiction is "same topic,
        //   opposing content" (low text overlap) while refinement is
        //   "same topic, updated content" (high text overlap).
        // - contradiction_text_threshold: Jaccard similarity floor. Pairs
        //   with text overlap BELOW this are contradiction candidates;
        //   pairs at/above it are treated as refinements. Default 0.3.
        // - contradiction_weaken_factor: the old (contradicted) memory's
        //   association edges are multiplied by this factor. Default 0.5.
        // - supersession_demotion_factor: final-score multiplier applied to
        //   old memories superseded by refinement/contradiction. Default 0.4;
        //   set to 1.0 to disable final-score demotion.
        // - contradiction_max_pairs_per_cycle: cap on Contradicts edges
        //   per consolidation. Default 1024.
        config.tier.contradiction_cosine_threshold = contradiction_cosine_threshold;
        if let Some(tt) = contradiction_text_threshold {
            config.tier.contradiction_text_threshold = tt;
        }
        if let Some(rt) = refinement_text_threshold {
            config.tier.refinement_text_threshold = rt.clamp(0.0, 1.0);
        }
        if let Some(ro) = contradiction_require_opposition {
            config.tier.contradiction_require_opposition = ro;
        }
        if let Some(wf) = contradiction_weaken_factor {
            config.tier.contradiction_weaken_factor = wf;
        }
        if let Some(df) = supersession_demotion_factor {
            config.tier.supersession_demotion_factor = df.clamp(0.0, 1.0);
        }
        if let Some(cp) = contradiction_max_pairs_per_cycle {
            config.tier.contradiction_max_pairs_per_cycle = cp;
        }
        // - importance_auto_scoring: enable self-organizing memory. When true,
        //   each consolidation recomputes every record's importance as a blend
        //   of retrieval salience (access_score) and graph connectivity (concept
        //   degree), moving toward a computed target. Frequently retrieved +
        //   well-connected memories rise; never-retrieved memories decay toward
        //   the floor. None/false (default) keeps the caller-set importance.
        // - importance_learning_rate: fraction of the way to move toward the
        //   target each cycle (0.0..=1.0). Default 0.3.
        // - importance_access_weight: weight on retrieval salience in the target
        //   blend; the rest goes to connectivity. Default 0.6.
        // - importance_floor / importance_ceiling: clamp range. Defaults 0.1/4.0.
        if let Some(on) = importance_auto_scoring {
            config.tier.importance_auto_scoring = on;
        }
        if let Some(lr) = importance_learning_rate {
            config.tier.importance_learning_rate = lr;
        }
        if let Some(aw) = importance_access_weight {
            config.tier.importance_access_weight = aw;
        }
        if let Some(fl) = importance_floor {
            config.tier.importance_floor = fl;
        }
        if let Some(ce) = importance_ceiling {
            config.tier.importance_ceiling = ce;
        }
        // - concept_evolution_enabled: enable online vocabulary evolution.
        //   When true, consolidation merges similar concept nodes and
        //   suppresses over-general hub concepts. false (default) preserves
        //   exact extracted concepts.
        // - concept_merge_overlap_threshold: Jaccard overlap of associated
        //   memory sets required to merge two concepts. Default 0.7.
        // - concept_hub_degree_fraction: fraction of total memories above
        //   which a base concept is suppressed as a hub. Default 0.1.
        // - concept_evolution_max_pairs_per_cycle: max merge ops per pass.
        if let Some(on) = concept_evolution_enabled {
            config.tier.concept_evolution_enabled = on;
        }
        if let Some(th) = concept_merge_overlap_threshold {
            config.tier.concept_merge_overlap_threshold = th.clamp(0.0, 1.0);
        }
        if let Some(f) = concept_hub_degree_fraction {
            config.tier.concept_hub_degree_fraction = f.max(0.0);
        }
        if let Some(mp) = concept_evolution_max_pairs_per_cycle {
            config.tier.concept_evolution_max_pairs_per_cycle = mp;
        }
        // Belief-revision role scoping: an empty list is treated as "no filter"
        // (role-blind) so callers can pass [] to mean the default explicitly.
        if let Some(roles) = belief_source_roles {
            config.tier.belief_source_roles = if roles.is_empty() { None } else { Some(roles) };
        }
        if let Some(defer) = defer_supersession_commit {
            config.tier.defer_supersession_commit = defer;
        }
        // - exclude_superseded: when true, memories superseded by a newer
        //   belief (old side of a Refines/Contradicts edge) are EXCLUDED from
        //   search/search_ann results entirely, not merely rank-demoted — the
        //   A-TMA "ghost memory" fix proven in the B1 eval. Requires belief
        //   revision to be active to have any effect: no supersession edges
        //   means an empty superseded set and zero behavior change. Default
        //   false.
        if let Some(ex) = exclude_superseded {
            config.tier.exclude_superseded = ex;
        }
        if let Some(aae) = access_aware_eviction {
            config.tier.access_aware_eviction = aae;
        }
        // - actr_activation: rank eviction candidates by ACT-R base-level
        //   activation — ln(Σ age^-actr_decay) over the last actr_history
        //   access timestamps — instead of the legacy access_count × recency
        //   heuristic, so spaced/repeated rehearsal outweighs a single recent
        //   burst (power-law forgetting). Opt-in (default false); promotion
        //   scoring stays legacy. Only meaningful with access-aware eviction
        //   and max_records / evict_score_floor set.
        // - actr_decay: decay exponent d. Default 0.5; higher forgets faster.
        // - actr_history: ring length of retained access timestamps. Default
        //   8, clamped to 1..=32.
        if let Some(on) = actr_activation {
            config.tier.actr_activation = on;
        }
        if let Some(d) = actr_decay {
            config.tier.actr_decay = d;
        }
        if let Some(k) = actr_history {
            config.tier.actr_history = k;
        }
        if let Some(inc) = incremental_supersession_detection {
            config.tier.incremental_supersession_detection = inc;
        }
        // - gist_before_evict: compress eviction victims into searchable gist
        //   records (source_role "gist", scope-isolated) instead of dropping
        //   them outright (B4). Opt-in (default false); requires a gist
        //   compressor installed via `set_gist_compressor`, otherwise a no-op.
        // - gist_chunk_facts: evicted texts per compressor call. Default 24.
        if let Some(on) = gist_before_evict {
            config.tier.gist_before_evict = on;
        }
        if let Some(n) = gist_chunk_facts {
            config.tier.gist_chunk_facts = n;
        }
        if let Some(trw) = temporal_recency_weight {
            config.tier.temporal_recency_weight = trw.clamp(0.0, 2.0);
        }
        // Power-loss durability: sync every write to disk before returning.
        if let Some(sync) = sync_writes {
            config.tier.sync_writes = sync;
        }
        // GPU exact search (CUDA builds only): keep the vectors resident on
        // the device and answer a search with one product over all of them.
        if let Some(on) = gpu_exact_search {
            config.tier.gpu_exact_search = on;
        }
        if let Some(n) = gpu_exact_min_records {
            config.tier.gpu_exact_min_records = n;
        }
        if let Some(mb) = gpu_memory_budget_mb {
            config.tier.gpu_memory_budget_mb = mb;
        }

        // Opening replays the write-ahead log and rebuilds the in-memory
        // indexes, which can take a while on a large store: do not hold the
        // interpreter lock for it.
        let inner = py
            .allow_threads(|| StorageEngine::open(db_path, config))
            .map_err(storage_err)?;
        Ok(Self {
            inner: Mutex::new(Some(inner)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        id,
        text,
        embedding,
        importance_score,
        concepts,
        payload=None,
        scope=None,
        source_role=None
    ))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        text: &str,
        embedding: &Bound<'_, PyAny>,
        importance_score: f32,
        concepts: Vec<String>,
        payload: Option<String>,
        scope: Option<String>,
        source_role: Option<String>,
    ) -> PyResult<bool> {
        let engine = self.engine()?;
        let emb_input = extract_f32_input(embedding)?;
        let emb = emb_input.as_slice();
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            engine
                .insert_with_payload_role(
                    id,
                    text,
                    emb,
                    importance_score,
                    &concepts,
                    payload,
                    scope,
                    source_role,
                )
                .map_err(storage_err)
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        ids,
        texts,
        embeddings,
        scores,
        concepts,
        payloads=None,
        scopes=None,
        source_roles=None
    ))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        texts: Vec<String>,
        embeddings: &Bound<'_, PyAny>,
        scores: Vec<f32>,
        concepts: Vec<Vec<String>>,
        payloads: Option<Vec<String>>,
        scopes: Option<Vec<String>>,
        source_roles: Option<Vec<String>>,
    ) -> PyResult<usize> {
        let engine = self.engine()?;
        let matrix = extract_f32_matrix(embeddings)?;
        let rows = matrix.rows();
        let payloads: Vec<Option<String>> = match payloads {
            Some(list) => list
                .into_iter()
                .map(|s| parse_payload(Some(s)))
                .collect::<PyResult<_>>()?,
            None => Vec::new(),
        };
        let scopes: Vec<Option<String>> = match scopes {
            Some(list) => list.into_iter().map(Some).collect(),
            None => Vec::new(),
        };
        let source_roles: Vec<Option<String>> = match source_roles {
            Some(list) => list.into_iter().map(Some).collect(),
            None => Vec::new(),
        };
        py.allow_threads(|| {
            engine
                .insert_batch_with_payload_role(
                    &ids,
                    &texts,
                    &rows,
                    &scores,
                    &concepts,
                    &payloads,
                    &scopes,
                    &source_roles,
                )
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (query_embedding, top_k, search_list_size=None, scope=None))]
    fn search_ann(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
        scope: Option<String>,
    ) -> PyResult<Vec<(String, f32)>> {
        let engine = self.engine()?;
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            engine
                .search_ann_scoped(q, top_k, search_list_size, scope_ref)
                .map_err(storage_err)
        })
    }

    /// Batched ANN search for many queries at once. Accepts a 2-D `float32`
    /// array of shape `(num_queries, dimension)` (or a list of 1-D arrays) and
    /// returns `list[list[(id, score)]]` — one result list per query.
    ///
    /// In a CUDA build with a GPU present the store's vectors are kept on the
    /// device and the whole batch is one cuBLAS `gemm` over all of them: the
    /// exact answer for every query (see `gpu_search_stats`). If that path is
    /// off or the store does not fit in device memory, the per-query HNSW
    /// traversal runs on the CPU and only the candidate rerank is one `gemm`.
    #[pyo3(signature = (queries, top_k, search_list_size=None, scope=None))]
    fn search_ann_batch(
        &self,
        py: Python<'_>,
        queries: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
        scope: Option<String>,
    ) -> PyResult<Vec<Vec<(String, f32)>>> {
        let engine = self.engine()?;
        let matrix = extract_f32_matrix(queries)?;
        let rows = matrix.rows();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            engine
                .search_ann_batch(&rows, top_k, search_list_size, None, scope_ref)
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (query_embedding, top_k, search_list_size=None, scope=None))]
    fn search_ann_candidates(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
        scope: Option<String>,
    ) -> PyResult<Vec<(String, f32)>> {
        let engine = self.engine()?;
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            engine
                .search_ann_scoped(q, top_k, search_list_size, scope_ref)
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (query_text, query_embedding, top_k, search_list_size=None, scope=None))]
    fn search(
        &self,
        py: Python<'_>,
        query_text: &str,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
        scope: Option<String>,
    ) -> PyResult<Option<Vec<(String, f32)>>> {
        let engine = self.engine()?;
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            engine
                .search_scoped_with_ef(query_text, q, top_k, search_list_size, scope_ref)
                .map_err(storage_err)
        })
    }

    fn step_session(
        &self,
        py: Python<'_>,
        user_input: &str,
        assistant_response: &str,
    ) -> PyResult<String> {
        let engine = self.engine()?;
        py.allow_threads(|| {
            engine
                .step_session(user_input, assistant_response)
                .map_err(storage_err)
        })
    }

    /// Install a Python callable as the cognitive compressor for
    /// `step_session`. The callable receives three positional string arguments:
    /// `(current_ccs_json, user_input, assistant_response)` and must return a
    /// JSON string representing a `CompressedCognitiveState` (the same schema
    /// `step_session` emits). If the callable raises or returns invalid JSON,
    /// the engine falls back to the deterministic compressor for that turn so
    /// the working memory is never corrupted.
    ///
    /// Example:
    /// ```python
    /// def my_compressor(ccs_json, user_input, assistant_response):
    ///     return json.dumps({
    ///         "turn_count": json.loads(ccs_json).get("turn_count", 0) + 1,
    ///         "last_user_input": user_input,
    ///         "last_assistant_response": assistant_response,
    ///         "facts": [f"User asked: {user_input}"],
    ///         "topics": ["ai"],
    ///     })
    ///
    /// engine.set_llm_compressor(my_compressor)
    /// ```
    fn set_llm_compressor(&self, callable: Py<PyAny>) -> PyResult<()> {
        let engine = self.engine()?;
        let compressor = Arc::new(PythonCompressor { callable });
        engine.set_compressor(compressor);
        Ok(())
    }

    /// Install a Python callable as the gist compressor for B4
    /// gist-before-evict. Only consulted when the engine was constructed with
    /// `gist_before_evict=True`. The callable receives one argument — a list
    /// of evicted fact texts (one chunk, same scope, chronological order) —
    /// and must return either `None` (nothing worth keeping: the chunk is
    /// deleted without a gist) or a `(gist_text, embedding)` tuple: the
    /// compressed gist plus the vector to store it under (typically from the
    /// same embedder used for facts). If the callable raises or returns a
    /// malformed value, that chunk's records are NOT deleted; they stay in
    /// the store and are tried again on the next eviction, so a summarizer
    /// outage cannot silently erase memories.
    ///
    /// Example:
    /// ```python
    /// def my_gister(texts):
    ///     gist = llm_summarize(texts)                 # your LLM call
    ///     return gist, embedder.encode([gist])[0]     # (text, embedding)
    ///
    /// engine.set_gist_compressor(my_gister)
    /// ```
    fn set_gist_compressor(&self, callable: Py<PyAny>) -> PyResult<()> {
        let engine = self.engine()?;
        let compressor = Arc::new(PythonGistCompressor { callable });
        engine.set_gist_compressor(Some(compressor));
        Ok(())
    }

    fn trigger_consolidation(&self, py: Python<'_>) -> PyResult<(usize, usize, usize)> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.trigger_consolidation().map_err(storage_err))
    }

    /// Run bounded-storage eviction directly, returning the number of records
    /// dropped. No-op (returns 0) unless `max_records` or `evict_score_floor`
    /// was configured.
    fn evict(&self, py: Python<'_>) -> PyResult<usize> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.evict().map_err(storage_err))
    }

    /// Run semantic near-duplicate consolidation directly, returning the number
    /// of duplicate records merged away. No-op (returns 0) unless
    /// `dedup_cosine_threshold` was configured.
    fn deduplicate(&self, py: Python<'_>) -> PyResult<usize> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.deduplicate().map_err(storage_err))
    }

    /// Run automatic importance scoring directly, returning the number of
    /// records whose importance changed. No-op (returns 0) unless
    /// `importance_auto_scoring` was enabled. Runs automatically on each
    /// `trigger_consolidation` when enabled; this method lets callers run it
    /// independently.
    fn recompute_importance(&self, py: Python<'_>) -> PyResult<usize> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.recompute_importance().map_err(storage_err))
    }

    /// Run one pass of online concept vocabulary evolution, returning
    /// `(merged, newly_suppressed, examined_pairs)`. No-op `(0, 0, 0)` unless
    /// `concept_evolution_enabled` is true.
    fn evolve_concept_vocabulary(&self, py: Python<'_>) -> PyResult<(usize, usize, usize)> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.evolve_concept_vocabulary().map_err(storage_err))
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.flush().map_err(storage_err))
    }

    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<bool> {
        let engine = self.engine()?;
        py.allow_threads(|| engine.delete_by_id(id).map_err(storage_err))
    }

    /// Number of live (non-tombstoned) records. Lets callers assert that
    /// bounded-storage eviction is keeping the collection under `max_records`.
    fn record_count(&self, py: Python<'_>) -> PyResult<usize> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| engine.record_count()))
    }

    /// True if a record with this id is still live (not evicted/deleted). Used
    /// by the retention eval (W5) to measure gold-fact survival after eviction.
    fn contains_id(&self, py: Python<'_>, id: String) -> PyResult<bool> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| engine.contains_id(&id)))
    }

    /// Return the stored text for `id`, or None when the id is unknown. Lets
    /// callers render engine-generated records (e.g. gist-before-evict gist
    /// records) whose ids they did not mint themselves.
    fn get_text(&self, py: Python<'_>, id: String) -> PyResult<Option<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| engine.find_meta_by_id(&id).map(|r| r.text)))
    }

    /// Stored metadata for each id, parallel to `ids`: a dict with keys
    /// `id` / `text` / `payload` / `scope` / `source_role` / `importance` /
    /// `created_at` / `insert_seq`, or `None` for an id that is not live.
    /// `payload` is the raw JSON string given at insert time (or `None`).
    /// Lets callers hydrate search hits from the engine instead of mirroring
    /// record metadata in process memory, where it is lost on restart.
    fn get_records(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<Vec<Option<Py<PyDict>>>> {
        let engine = self.engine()?;
        let metas = py.allow_threads(|| {
            ids.iter()
                .map(|id| engine.find_meta_by_id(id))
                .collect::<Vec<_>>()
        });
        metas
            .into_iter()
            .map(|meta| {
                meta.map(|m| {
                    let d = PyDict::new(py);
                    d.set_item("id", m.id)?;
                    d.set_item("text", m.text)?;
                    d.set_item("payload", m.payload)?;
                    d.set_item("scope", m.scope)?;
                    d.set_item("source_role", m.source_role)?;
                    d.set_item("importance", m.importance)?;
                    d.set_item("created_at", m.created_at)?;
                    d.set_item("insert_seq", m.insert_seq)?;
                    Ok(d.unbind())
                })
                .transpose()
            })
            .collect()
    }

    /// Ids of the live records stored under exactly `scope`, oldest first
    /// (unscoped records belong to no scope's list). With `get_records` this
    /// gives a caller one scope's whole store, for maintenance such as
    /// compaction.
    fn scope_ids(&self, py: Python<'_>, scope: String) -> PyResult<Vec<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| engine.scope_ids(&scope)))
    }

    /// The scopes that hold at least one record, sorted.
    fn scopes(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| engine.scopes()))
    }

    /// The `insert_seq` the next inserted record will receive. Durable and
    /// monotonically increasing — never reused across restarts, deletes, or
    /// eviction — so callers can derive collision-free ids from it.
    fn next_insert_seq(&self, py: Python<'_>) -> PyResult<u64> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| engine.next_insert_seq()))
    }

    /// Returns True if the engine is using GPU acceleration for distance
    /// computation. This is determined at runtime based on CUDA availability.
    #[getter]
    fn gpu_accelerated(&self) -> PyResult<bool> {
        Ok(self.engine()?.is_gpu_accelerated())
    }

    /// What the GPU exact-search mirror holds, or `None` when this engine has
    /// not used one (not a CUDA build, no device, `gpu_exact_search=False`,
    /// or the store is still below `gpu_exact_min_records`).
    ///
    /// Keys: `backend`, `active` (False once the mirror was switched off: a
    /// device error or a store larger than the budget; searches then run on
    /// the CPU), `rows`, `capacity_rows`, `budget_bytes`, `memory_bytes`,
    /// `queries` (answered from the mirror) and `device_calls` (the products
    /// that answered them; fewer when concurrent searches shared one).
    fn gpu_search_stats(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        let Some(stats) = self.engine()?.gpu_search_stats() else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("backend", stats.backend)?;
        dict.set_item("active", stats.active)?;
        dict.set_item("rows", stats.rows)?;
        dict.set_item("capacity_rows", stats.capacity_rows)?;
        dict.set_item("budget_bytes", stats.budget_bytes)?;
        dict.set_item("memory_bytes", stats.memory_bytes)?;
        dict.set_item("queries", stats.queries)?;
        dict.set_item("device_calls", stats.device_calls)?;
        Ok(Some(dict.into()))
    }

    /// True once `close()` has run; every other method then raises
    /// `RuntimeError("engine is closed")`.
    #[getter]
    fn closed(&self) -> bool {
        self.handle().is_none()
    }

    /// What opening this store had to repair, as a dict of counters. All
    /// zeros when the store was closed cleanly last time.
    ///
    /// - `wal_ops_replayed`: writes recovered from the write-ahead log.
    /// - `wal_inserts_without_vector`: logged inserts dropped because their
    ///   vector never reached disk.
    /// - `wal_bytes_discarded`: bytes cut from a torn or corrupt log tail.
    /// - `segments_discarded`: unreadable index segments (their records are
    ///   indexed again).
    /// - `segment_dirs_removed`: incomplete or superseded segment directories.
    /// - `graph_nodes_pruned`: graph nodes of records that no longer exist.
    fn recovery_report(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let engine = self.engine()?;
        let report = engine.recovery_report();
        let dict = PyDict::new(py);
        dict.set_item("wal_ops_replayed", report.wal_ops_replayed)?;
        dict.set_item(
            "wal_inserts_without_vector",
            report.wal_inserts_without_vector,
        )?;
        dict.set_item("wal_bytes_discarded", report.wal_bytes_discarded)?;
        dict.set_item("segments_discarded", report.segments_discarded)?;
        dict.set_item("segment_dirs_removed", report.segment_dirs_removed)?;
        dict.set_item("graph_nodes_pruned", report.graph_nodes_pruned)?;
        Ok(dict.unbind())
    }

    // ---- Graph introspection API (C7) -------------------------------------
    // Read-only views over the learned cognitive graph. Each method acquires
    // a read lock on the graph for the minimum work needed, collects into
    // owned tuples/vecs (graph borrows cannot escape the lock guard), and
    // returns. Unknown ids return empty lists (no KeyError), matching the
    // underlying `refined_by` / `contradicted_by` semantics.

    /// Structural snapshot of the cognitive graph.
    /// Returns (node_count, edge_count, memory_count, concept_count,
    /// refinement_count, contradiction_count, abstraction_count).
    fn graph_stats(
        &self,
        py: Python<'_>,
    ) -> PyResult<(usize, usize, usize, usize, usize, usize, usize)> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| {
            let guard = engine.read_graph();
            let s = guard.graph().stats();
            (
                s.node_count,
                s.edge_count,
                s.memory_count,
                s.concept_count,
                s.refinement_count,
                s.contradiction_count,
                s.abstraction_count,
            )
        }))
    }

    /// All concepts in the graph with their degree (number of memories
    /// attached). Returns list[(concept, degree)] sorted by degree desc.
    /// Abstraction parent nodes (containing '+') are excluded.
    fn get_concepts(&self, py: Python<'_>) -> PyResult<Vec<(String, usize)>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| {
            let guard = engine.read_graph();
            let graph = guard.graph();
            let mut concepts: Vec<(String, usize)> = graph
                .nodes()
                .values()
                .filter_map(|n| match &n.id {
                    turbomemory_graph::NodeId::Concept(c) if !c.contains('+') => {
                        Some((c.clone(), graph.concept_degree(c)))
                    }
                    _ => None,
                })
                .collect();
            // Sort by degree desc, then concept asc for determinism.
            concepts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            concepts
        }))
    }

    /// Concepts attached to memory `id`. Returns list[concept]. Empty if the
    /// memory is unknown.
    fn get_memory_concepts(&self, py: Python<'_>, id: String) -> PyResult<Vec<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| {
            let guard = engine.read_graph();
            guard.graph().memory_concepts(&id)
        }))
    }

    /// Memories that `id` refines (the older memories `id` supersedes).
    /// Returns list[id]. Empty if `id` has no Refines edges or is unknown.
    fn get_refinements(&self, py: Python<'_>, id: String) -> PyResult<Vec<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| {
            let guard = engine.read_graph();
            guard.graph().refined_by(&id)
        }))
    }

    /// Memories that contradict `id` (the newer memories that correct it).
    /// Returns list[id]. Empty if `id` has no Contradicts edges or is unknown.
    fn get_contradictions(&self, py: Python<'_>, id: String) -> PyResult<Vec<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| {
            let guard = engine.read_graph();
            guard.graph().contradicted_by(&id)
        }))
    }

    /// All SUPERSEDED memory ids — the older side of every Refines/Contradicts
    /// edge to a live newer memory (B1). Exclude these from the answer context
    /// (or tag them outdated) instead of merely rank-demoting them.
    fn superseded_ids(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let engine = self.engine()?;
        Ok(py.allow_threads(|| {
            let guard = engine.read_graph();
            guard.graph().superseded_ids()
        }))
    }

    /// Resolve each id against the supersession graph — "what is true NOW" in
    /// place of that memory, with lineage. Returns a list of dicts with keys
    /// `id` / `current_id` / `superseded` / `chain` (chain = the full
    /// supersession chain containing `id`, oldest first, head last) /
    /// `demotion` (the factor a superseded memory's search score is
    /// multiplied by; 1.0 for a current one). Works
    /// without any cognitive config flags: with no supersession edges every id
    /// resolves to itself (`superseded=False`, `chain=[id]`).
    fn resolve_beliefs(&self, py: Python<'_>, ids: Vec<String>) -> PyResult<Vec<Py<PyDict>>> {
        let engine = self.engine()?;
        let resolutions = py.allow_threads(|| engine.resolve_beliefs(&ids));
        resolutions
            .into_iter()
            .map(|r| {
                let d = PyDict::new(py);
                d.set_item("id", r.id)?;
                d.set_item("current_id", r.current_id)?;
                d.set_item("superseded", r.superseded)?;
                d.set_item("chain", r.chain)?;
                d.set_item("demotion", r.demotion)?;
                Ok(d.unbind())
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, text, embedding, importance_score, concepts, payload=None, scope=None, source_role=None))]
    fn update(
        &self,
        py: Python<'_>,
        id: &str,
        text: &str,
        embedding: &Bound<'_, PyAny>,
        importance_score: f32,
        concepts: Vec<String>,
        payload: Option<String>,
        scope: Option<String>,
        source_role: Option<String>,
    ) -> PyResult<bool> {
        let engine = self.engine()?;
        let emb_input = extract_f32_input(embedding)?;
        let emb = emb_input.as_slice();
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            engine
                .update_with_payload_role(
                    id,
                    text,
                    emb,
                    importance_score,
                    &concepts,
                    payload,
                    scope,
                    source_role,
                )
                .map_err(storage_err)
        })
    }

    /// Detect belief-revision supersessions WITHOUT committing them (W3). Returns
    /// a list of `(old_id, new_id, kind, cosine)` where `kind` is `"refinement"`
    /// or `"contradiction"`. Feed the pairs that survive an external verifier
    /// (e.g. an NLI cross-encoder) back to `commit_supersessions`. Requires the
    /// engine to be built with `defer_supersession_commit=True` so consolidation
    /// does not auto-commit them first.
    fn propose_supersessions(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<(String, String, String, f32)>> {
        let engine = self.engine()?;
        let props = py.allow_threads(|| engine.propose_supersessions().map_err(storage_err))?;
        Ok(props
            .into_iter()
            .map(|p| (p.old_id, p.new_id, p.kind.as_str().to_string(), p.cosine))
            .collect())
    }

    /// Wide candidates for a verifier that can judge meaning (a language
    /// model): each memory with its nearest older memories of the same scope
    /// (at most `per_record`, cosine >= `min_cosine`, and no more than
    /// `margin` below the closest of them; `margin=None` drops that last
    /// rule), without the lexical gates `propose_supersessions` applies. Same
    /// tuples as `propose_supersessions`. Most candidates are NOT
    /// supersessions: vet every one before `commit_supersessions`.
    #[pyo3(signature = (min_cosine=0.45, per_record=2, margin=Some(0.1)))]
    fn propose_supersession_candidates(
        &self,
        py: Python<'_>,
        min_cosine: f32,
        per_record: usize,
        margin: Option<f32>,
    ) -> PyResult<Vec<(String, String, String, f32)>> {
        let engine = self.engine()?;
        let margin = margin.unwrap_or(f32::INFINITY);
        let props = py.allow_threads(|| {
            engine
                .propose_supersession_candidates(min_cosine, per_record, margin)
                .map_err(storage_err)
        })?;
        Ok(props
            .into_iter()
            .map(|p| (p.old_id, p.new_id, p.kind.as_str().to_string(), p.cosine))
            .collect())
    }

    /// Commit verified supersessions (W3). `pairs` is a list of
    /// `(old_id, new_id, kind)` with `kind` in `{"refinement","contradiction"}`.
    /// Creates the Refines/Contradicts edge and applies bounded demotion for
    /// each surviving pair. Unknown-kind or dead-id pairs are skipped. Returns
    /// the number of edges created.
    #[pyo3(signature = (pairs))]
    fn commit_supersessions(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, String, String)>,
    ) -> PyResult<usize> {
        let engine = self.engine()?;
        let parsed: Vec<(String, String, SupersessionKind)> = pairs
            .into_iter()
            .filter_map(|(o, n, k)| SupersessionKind::from_label(&k).map(|kind| (o, n, kind)))
            .collect();
        py.allow_threads(|| {
            engine
                .commit_supersessions_by_id(&parsed)
                .map_err(storage_err)
        })
    }

    /// Flush all durable state and release the engine: the background workers
    /// are stopped and the database lock, mmaps, and index files are let go, so
    /// the same `db_path` can be reopened immediately. Idempotent. The engine
    /// is released even when the final flush fails; that error is still raised.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let Some(engine) = self.handle().take() else {
            return Ok(());
        };
        // Drop the Python callbacks while the GIL is held. A gist/LLM callback
        // is usually a bound method of the object that owns this engine, a
        // reference cycle Python's collector cannot see through the Rust side.
        engine.set_compressor(Arc::new(DeterministicCompressor));
        engine.set_gist_compressor(None);
        // Shutdown joins the worker threads, which may be waiting for the GIL
        // inside a callback, so it must run with the GIL released.
        py.allow_threads(move || {
            let result = engine.shutdown().map_err(storage_err);
            drop(engine);
            result
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(
        &self,
        py: Python<'py>,
        _exc_type: &Bound<'py, PyAny>,
        _exc_value: &Bound<'py, PyAny>,
        _traceback: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
}

/// A `CognitiveCompressor` backed by a Python callable. The callable is
/// invoked with the GIL re-acquired for each compression call. Errors from
/// Python or from parsing the returned JSON fall back to the deterministic
/// compressor so a misbehaving callback cannot corrupt the working-memory
/// state.
///
/// The callable is held without a lock (`Py<PyAny>` is `Send + Sync`, and
/// the GIL already serializes the call). A lock taken after the GIL and held
/// across the call deadlocked two threads as soon as the callback released
/// the GIL: one held the lock and waited for the GIL, the other held the GIL
/// and waited for the lock.
struct PythonCompressor {
    callable: Py<PyAny>,
}

impl CognitiveCompressor for PythonCompressor {
    fn compress(
        &self,
        ccs: &CompressedCognitiveState,
        user_input: &str,
        assistant_response: &str,
    ) -> CompressedCognitiveState {
        let ccs_json = ccs.to_json();
        let result = Python::with_gil(|py| {
            let args = (ccs_json, user_input, assistant_response);
            let output = self.callable.call1(py, args)?;
            let json_str: String = output.extract(py)?;
            Ok::<_, PyErr>(json_str)
        });

        let json_str = match result {
            Ok(s) => s,
            Err(_) => {
                return DeterministicCompressor.compress(ccs, user_input, assistant_response);
            }
        };

        match serde_json::from_str::<CompressedCognitiveState>(&json_str) {
            Ok(parsed) => parsed,
            Err(_) => DeterministicCompressor.compress(ccs, user_input, assistant_response),
        }
    }
}

/// Gist compressor backed by a Python callable (B4 gist-before-evict). The
/// callable maps a chunk of evicted texts to `(gist_text, embedding)` or
/// `None`. `None` is an abstention (the chunk is deleted without a gist); an
/// exception or a malformed return is a failure, reported to the engine so
/// it keeps the chunk's records instead of deleting them unsummarized.
struct PythonGistCompressor {
    callable: Py<PyAny>,
}

impl GistCompressor for PythonGistCompressor {
    fn compress(&self, texts: &[String]) -> Result<Option<(String, Vec<f32>)>, String> {
        Python::with_gil(|py| {
            let output = self
                .callable
                .call1(py, (texts.to_vec(),))
                .map_err(|e| format!("gist callback raised: {e}"))?;
            if output.is_none(py) {
                return Ok(None);
            }
            output
                .extract::<(String, Vec<f32>)>(py)
                .map(Some)
                .map_err(|e| format!("gist callback must return None or (text, embedding): {e}"))
        })
    }
}

#[pymodule]
fn turbomemory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryEngine>()?;
    Ok(())
}
