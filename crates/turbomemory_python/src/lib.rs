//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

use numpy::PyUntypedArrayMethods;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Duration;
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
        abstraction_co_occurrence_threshold=None,
        edge_decay_half_life_secs=None
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
        abstraction_co_occurrence_threshold: Option<usize>,
        edge_decay_half_life_secs: Option<u64>,
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
        if let Some(th) = abstraction_co_occurrence_threshold {
            config.tier.abstraction_co_occurrence_threshold = th;
        }
        if let Some(hl) = edge_decay_half_life_secs {
            config.tier.edge_decay_half_life_secs = hl;
        }

        let inner = StorageEngine::open(db_path, config).map_err(storage_err)?;
        Ok(Self { inner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, text, embedding, importance_score, concepts, payload=None))]
    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        text: &str,
        embedding: &Bound<'_, PyAny>,
        importance_score: f32,
        concepts: Vec<String>,
        payload: Option<String>,
    ) -> PyResult<bool> {
        let emb_input = extract_f32_input(embedding)?;
        let emb = emb_input.as_slice();
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            self.inner
                .insert_with_payload(id, text, emb, importance_score, &concepts, payload)
                .map_err(storage_err)
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (ids, texts, embeddings, scores, concepts, payloads=None))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        texts: Vec<String>,
        embeddings: &Bound<'_, PyAny>,
        scores: Vec<f32>,
        concepts: Vec<Vec<String>>,
        payloads: Option<Vec<String>>,
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
        py.allow_threads(|| {
            self.inner
                .insert_batch_with_payload(&ids, &texts, &rows, &scores, &concepts, &payloads)
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (query_embedding, top_k, search_list_size=None))]
    fn search_ann(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
    ) -> PyResult<Vec<(String, f32)>> {
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        py.allow_threads(|| {
            self.inner
                .search_ann_with_ef(q, top_k, search_list_size)
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (query_embedding, top_k, search_list_size=None))]
    fn search_ann_candidates(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
    ) -> PyResult<Vec<(String, f32)>> {
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        py.allow_threads(|| {
            self.inner
                .search_ann_candidates_with_ef(q, top_k, search_list_size)
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (query_text, query_embedding, top_k, search_list_size=None))]
    fn search(
        &self,
        py: Python<'_>,
        query_text: &str,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
        search_list_size: Option<usize>,
    ) -> PyResult<Option<Vec<(String, f32)>>> {
        let q_input = extract_f32_input(query_embedding)?;
        let q = q_input.as_slice();
        py.allow_threads(|| {
            self.inner
                .search_with_ef(query_text, q, top_k, search_list_size)
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

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (id, text, embedding, importance_score, concepts, payload=None))]
    fn update(
        &self,
        py: Python<'_>,
        id: &str,
        text: &str,
        embedding: &Bound<'_, PyAny>,
        importance_score: f32,
        concepts: Vec<String>,
        payload: Option<String>,
    ) -> PyResult<bool> {
        let emb_input = extract_f32_input(embedding)?;
        let emb = emb_input.as_slice();
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            self.inner
                .update_with_payload(id, text, emb, importance_score, &concepts, payload)
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

#[pymodule]
fn turbomemory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryEngine>()?;
    Ok(())
}
