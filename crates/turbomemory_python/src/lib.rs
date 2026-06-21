//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use turbomemory_graph::{CognitiveCompressor, CompressedCognitiveState, DeterministicCompressor};
use turbomemory_storage::config::{QuantizerKind, StoreConfig};
use turbomemory_storage::engine::StorageEngine;

/// Map storage errors to specific Python exception types.
fn storage_err(e: turbomemory_storage::StorageError) -> PyErr {
    use turbomemory_storage::StorageError as E;
    match e {
        E::DuplicateId(_) | E::DimensionMismatch | E::InvalidArgument(_) => {
            PyValueError::new_err(e.to_string())
        }
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
            "unknown quantizer '{spec}'; expected scalar<N>, sign, turbo_mse<N>, or turbo_prod<N>"
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
    inner: Arc<StorageEngine>,
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
        fok_threshold=None,
        spreading_decay=None,
        spreading_iterations=None,
        spreading_beta=None,
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
        contradiction_weaken_factor=None,
        contradiction_max_pairs_per_cycle=None,
        importance_auto_scoring=None,
        importance_learning_rate=None,
        importance_access_weight=None,
        importance_floor=None,
        importance_ceiling=None,
        concept_evolution_enabled=None,
        concept_merge_overlap_threshold=None,
        concept_hub_degree_fraction=None,
        concept_evolution_max_pairs_per_cycle=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
        fok_threshold: Option<f32>,
        spreading_decay: Option<f32>,
        spreading_iterations: Option<usize>,
        spreading_beta: Option<f32>,
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
        contradiction_weaken_factor: Option<f32>,
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
        // - fok_threshold: Feeling-of-Knowing gate. Lower = more permissive
        //   retrieval (returns more results); higher = stricter (rejects weak
        //   matches). Default 0.58.
        // - spreading_decay / spreading_iterations: control how far activation
        //   propagates through the memory graph. Defaults 0.5 / 4.
        // - abstraction_co_occurrence_threshold: enable abstraction hierarchy
        //   building. 0 (default) disables. A value of 3 means two concepts
        //   must co-occur on >= 3 memories before a parent concept is created.
        // - edge_decay_half_life_secs: enable edge forgetting. 0 (default)
        //   disables. A value of 86400 (1 day) means unrehearsed reinforced
        //   edges fade toward baseline with a 1-day half-life.
        if let Some(fok) = fok_threshold {
            config.spreading.fok_threshold = fok;
        }
        if let Some(decay) = spreading_decay {
            config.spreading.decay = decay;
        }
        if let Some(iters) = spreading_iterations {
            config.spreading.iterations = iters;
        }
        if let Some(beta) = spreading_beta {
            config.spreading.beta = beta;
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
        // - cognitive_alpha: fusion weight for cognitive search.
        //   final_score = cognitive_alpha * cosine + (1 - cognitive_alpha) * graph_activation
        //   1.0 (default) = pure cosine (graph only chooses candidates).
        //   0.5 = graph activation has equal vote with cosine (enables
        //   reinforcement/refinement/abstraction to re-rank memories).
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
        // - contradiction_max_pairs_per_cycle: cap on Contradicts edges
        //   per consolidation. Default 1024.
        config.tier.contradiction_cosine_threshold = contradiction_cosine_threshold;
        if let Some(tt) = contradiction_text_threshold {
            config.tier.contradiction_text_threshold = tt;
        }
        if let Some(wf) = contradiction_weaken_factor {
            config.tier.contradiction_weaken_factor = wf;
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

        let inner = StorageEngine::open(db_path, config).map_err(storage_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        id,
        text,
        embedding,
        importance_score,
        concepts,
        payload=None,
        scope=None
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
    ) -> PyResult<bool> {
        let emb_input = extract_f32_input(embedding)?;
        let emb = emb_input.as_slice();
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            self.inner
                .insert_with_payload(id, text, emb, importance_score, &concepts, payload, scope)
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
        scopes=None
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
    ) -> PyResult<usize> {
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
        py.allow_threads(|| {
            self.inner
                .insert_batch_with_payload(
                    &ids, &texts, &rows, &scores, &concepts, &payloads, &scopes,
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
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            self.inner
                .search_ann_scoped(q, top_k, search_list_size, scope_ref)
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
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            self.inner
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
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        let scope_ref = scope.as_deref();
        py.allow_threads(|| {
            self.inner
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
        py.allow_threads(|| {
            self.inner
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
        let compressor = Arc::new(PythonCompressor {
            callable: Mutex::new(callable),
        });
        self.inner.set_compressor(compressor);
        Ok(())
    }

    fn trigger_consolidation(&self, py: Python<'_>) -> PyResult<(usize, usize, usize)> {
        py.allow_threads(|| self.inner.trigger_consolidation().map_err(storage_err))
    }

    /// Run bounded-storage eviction directly, returning the number of records
    /// dropped. No-op (returns 0) unless `max_records` or `evict_score_floor`
    /// was configured.
    fn evict(&self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.inner.evict().map_err(storage_err))
    }

    /// Run semantic near-duplicate consolidation directly, returning the number
    /// of duplicate records merged away. No-op (returns 0) unless
    /// `dedup_cosine_threshold` was configured.
    fn deduplicate(&self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.inner.deduplicate().map_err(storage_err))
    }

    /// Run automatic importance scoring directly, returning the number of
    /// records whose importance changed. No-op (returns 0) unless
    /// `importance_auto_scoring` was enabled. Runs automatically on each
    /// `trigger_consolidation` when enabled; this method lets callers run it
    /// independently.
    fn recompute_importance(&self, py: Python<'_>) -> PyResult<usize> {
        py.allow_threads(|| self.inner.recompute_importance().map_err(storage_err))
    }

    /// Run one pass of online concept vocabulary evolution, returning
    /// `(merged, newly_suppressed, examined_pairs)`. No-op `(0, 0, 0)` unless
    /// `concept_evolution_enabled` is true.
    fn evolve_concept_vocabulary(&self, py: Python<'_>) -> PyResult<(usize, usize, usize)> {
        py.allow_threads(|| self.inner.evolve_concept_vocabulary().map_err(storage_err))
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.flush().map_err(storage_err))
    }

    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<bool> {
        py.allow_threads(|| self.inner.delete_by_id(id).map_err(storage_err))
    }

    /// Number of live (non-tombstoned) records. Lets callers assert that
    /// bounded-storage eviction is keeping the collection under `max_records`.
    fn record_count(&self, py: Python<'_>) -> PyResult<usize> {
        Ok(py.allow_threads(|| self.inner.record_count()))
    }

    /// Returns True if the engine is using GPU acceleration for distance
    /// computation. This is determined at runtime based on CUDA availability.
    #[getter]
    fn gpu_accelerated(&self) -> bool {
        self.inner.is_gpu_accelerated()
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
        Ok(py.allow_threads(|| {
            let guard = self.inner.read_graph();
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
        Ok(py.allow_threads(|| {
            let guard = self.inner.read_graph();
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
        Ok(py.allow_threads(|| {
            let guard = self.inner.read_graph();
            guard.graph().memory_concepts(&id)
        }))
    }

    /// Memories that `id` refines (the older memories `id` supersedes).
    /// Returns list[id]. Empty if `id` has no Refines edges or is unknown.
    fn get_refinements(&self, py: Python<'_>, id: String) -> PyResult<Vec<String>> {
        Ok(py.allow_threads(|| {
            let guard = self.inner.read_graph();
            guard.graph().refined_by(&id)
        }))
    }

    /// Memories that contradict `id` (the newer memories that correct it).
    /// Returns list[id]. Empty if `id` has no Contradicts edges or is unknown.
    fn get_contradictions(&self, py: Python<'_>, id: String) -> PyResult<Vec<String>> {
        Ok(py.allow_threads(|| {
            let guard = self.inner.read_graph();
            guard.graph().contradicted_by(&id)
        }))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, text, embedding, importance_score, concepts, payload=None, scope=None))]
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
    ) -> PyResult<bool> {
        let emb_input = extract_f32_input(embedding)?;
        let emb = emb_input.as_slice();
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            self.inner
                .update_with_payload(id, text, emb, importance_score, &concepts, payload, scope)
                .map_err(storage_err)
        })
    }

    /// Flush all durable state. The engine's built-in background optimizer is
    /// stopped automatically when the engine is dropped.
    fn close(&mut self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.shutdown().map_err(storage_err))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(
        &mut self,
        py: Python<'py>,
        _exc_type: &Bound<'py, PyAny>,
        _exc_value: &Bound<'py, PyAny>,
        _traceback: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        self.close(py)
    }
}

/// A `CognitiveCompressor` backed by a Python callable. The callable is
/// invoked with the GIL re-acquired for each compression call; a Mutex makes
/// the wrapper `Sync` as required by the trait. Errors from Python or from
/// parsing the returned JSON fall back to the deterministic compressor so a
/// misbehaving callback cannot corrupt the working-memory state.
struct PythonCompressor {
    callable: Mutex<Py<PyAny>>,
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
            let callable = self
                .callable
                .lock()
                .map_err(|e| PyRuntimeError::new_err(format!("compressor lock poisoned: {e}")))?;
            let args = (ccs_json, user_input, assistant_response);
            let output = callable.call1(py, args)?;
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

#[pymodule]
fn turbomemory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryEngine>()?;
    Ok(())
}
