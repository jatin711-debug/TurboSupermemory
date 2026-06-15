//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use turbomemory_storage::config::StoreConfig;
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
    #[pyo3(signature = (db_path, dimension, max_edges, search_list_size, outlier_count, initial_capacity=None))]
    fn new(
        db_path: &str,
        dimension: usize,
        max_edges: usize,
        search_list_size: usize,
        outlier_count: usize,
        initial_capacity: Option<usize>,
    ) -> PyResult<Self> {
        let mut config = StoreConfig::default_for_dimension(dimension);
        config.max_edges = max_edges;
        config.search_list_size = search_list_size;
        config.outlier_count = outlier_count;
        if let Some(cap) = initial_capacity {
            config.initial_capacity = cap.max(1024);
        }
        config.auto_consolidation_interval = Some(Duration::from_secs(60));
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
