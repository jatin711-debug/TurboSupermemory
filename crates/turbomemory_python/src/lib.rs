//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

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

/// Extract a 1-D f32 vector from a Python object (list, tuple, or numpy array).
///
/// Uses a zero-copy view when the input is a numpy `ndarray` of `float32`.
fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(v) = obj.extract::<Vec<f32>>() {
        return Ok(v);
    }
    if let Ok(arr) = numpy::PyReadonlyArray1::<f32>::extract_bound(obj) {
        if let Ok(slice) = arr.as_slice() {
            return Ok(slice.to_vec());
        }
    }
    if obj.hasattr("tolist")? {
        let list_obj = obj.call_method0("tolist")?;
        return list_obj.extract::<Vec<f32>>();
    }
    Err(PyValueError::new_err(
        "embedding must be a sequence or numpy array of f32",
    ))
}

/// Extract a 2-D f32 matrix from a Python object (list-of-lists or 2-D numpy array).
///
/// Uses a zero-copy view when the input is a 2-D numpy `ndarray` of `float32`.
fn extract_f32_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(m) = obj.extract::<Vec<Vec<f32>>>() {
        return Ok(m);
    }
    if let Ok(arr) = numpy::PyReadonlyArray2::<f32>::extract_bound(obj) {
        let arr = arr.as_array();
        if arr.shape().len() != 2 {
            return Err(PyValueError::new_err("embeddings must be 2-D"));
        }
        let mut out = Vec::with_capacity(arr.shape()[0]);
        for row in arr.rows() {
            out.push(row.to_vec());
        }
        return Ok(out);
    }
    if obj.hasattr("tolist")? {
        let list_obj = obj.call_method0("tolist")?;
        return list_obj.extract::<Vec<Vec<f32>>>();
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
        auto_consolidation_secs=60
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
        auto_consolidation_secs: u64,
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

        // 0 disables background consolidation entirely; otherwise it runs on
        // the given interval. Disabling is useful for benchmarks and for
        // workloads that drive consolidation manually via trigger_consolidation.
        config.auto_consolidation_interval = if auto_consolidation_secs == 0 {
            None
        } else {
            Some(Duration::from_secs(auto_consolidation_secs))
        };
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
        let emb = extract_f32_vec(embedding)?;
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            self.inner
                .insert_with_payload(id, text, &emb, importance_score, &concepts, payload)
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
        let payloads: Vec<Option<String>> = match payloads {
            Some(list) => list
                .into_iter()
                .map(|s| parse_payload(Some(s)))
                .collect::<PyResult<_>>()?,
            None => Vec::new(),
        };
        py.allow_threads(|| {
            self.inner
                .insert_batch_with_payload(&ids, &texts, &matrix, &scores, &concepts, &payloads)
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
        let q = extract_f32_vec(query_embedding)?;
        py.allow_threads(|| {
            self.inner
                .search_ann_with_ef(&q, top_k, search_list_size)
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
        let q = extract_f32_vec(query_embedding)?;
        py.allow_threads(|| {
            self.inner
                .search_ann_candidates_with_ef(&q, top_k, search_list_size)
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
        let q = extract_f32_vec(query_embedding)?;
        py.allow_threads(|| {
            self.inner
                .search_with_ef(query_text, &q, top_k, search_list_size)
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

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| self.inner.flush().map_err(storage_err))
    }

    fn delete(&self, py: Python<'_>, id: &str) -> PyResult<bool> {
        py.allow_threads(|| self.inner.delete_by_id(id).map_err(storage_err))
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
        let emb = extract_f32_vec(embedding)?;
        let payload = parse_payload(payload)?;
        py.allow_threads(|| {
            self.inner
                .update_with_payload(id, text, &emb, importance_score, &concepts, payload)
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
