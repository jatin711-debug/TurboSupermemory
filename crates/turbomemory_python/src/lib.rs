//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use std::sync::Arc;
use std::time::Duration;
use turbomemory_storage::config::StoreConfig;
use turbomemory_storage::engine::StorageEngine;
use turbomemory_storage::update_handler::UpdateHandler;

fn storage_err(e: turbomemory_storage::StorageError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Extract a 1-D f32 vector from a Python object (list, tuple, or numpy array).
fn extract_f32_vec(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(v) = obj.extract::<Vec<f32>>() {
        return Ok(v);
    }
    if obj.hasattr("tolist")? {
        let list_obj = obj.call_method0("tolist")?;
        return list_obj.extract::<Vec<f32>>();
    }
    Err(PyRuntimeError::new_err(
        "embedding must be a sequence or numpy array of f32",
    ))
}

/// Extract a 2-D f32 matrix from a Python object (list-of-lists or 2-D numpy array).
fn extract_f32_matrix(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(m) = obj.extract::<Vec<Vec<f32>>>() {
        return Ok(m);
    }
    if obj.hasattr("tolist")? {
        let list_obj = obj.call_method0("tolist")?;
        return list_obj.extract::<Vec<Vec<f32>>>();
    }
    Err(PyRuntimeError::new_err(
        "embeddings must be a 2-D sequence or numpy array of f32",
    ))
}

#[pyclass(name = "MemoryEngine")]
pub struct PyMemoryEngine {
    inner: Arc<StorageEngine>,
    #[allow(dead_code)]
    handler: Option<UpdateHandler>,
}

#[pymethods]
impl PyMemoryEngine {
    #[new]
    #[pyo3(signature = (db_path, dimension, max_edges, search_list_size, outlier_count))]
    fn new(
        db_path: &str,
        dimension: usize,
        max_edges: usize,
        search_list_size: usize,
        outlier_count: usize,
    ) -> PyResult<Self> {
        let config = StoreConfig {
            dimension,
            max_edges,
            search_list_size,
            outlier_count,
            initial_capacity: 1024,
            tier: turbomemory_storage::config::TierConfig::default(),
            auto_consolidation_interval: Some(Duration::from_secs(60)),
        };
        let inner = Arc::new(StorageEngine::open(db_path, config).map_err(storage_err)?);
        let handler = inner
            .config()
            .auto_consolidation_interval
            .map(|interval| UpdateHandler::new(inner.clone(), interval));
        Ok(Self { inner, handler })
    }

    fn insert(
        &self,
        py: Python<'_>,
        id: &str,
        text: &str,
        embedding: &Bound<'_, PyAny>,
        importance_score: f32,
        concepts: Vec<String>,
    ) -> PyResult<bool> {
        let emb = extract_f32_vec(embedding)?;
        py.allow_threads(|| {
            self.inner
                .insert(id, text, &emb, importance_score, &concepts)
                .map_err(storage_err)
        })
    }

    #[pyo3(signature = (ids, texts, embeddings, scores, concepts))]
    fn insert_batch(
        &self,
        py: Python<'_>,
        ids: Vec<String>,
        texts: Vec<String>,
        embeddings: &Bound<'_, PyAny>,
        scores: Vec<f32>,
        concepts: Vec<Vec<String>>,
    ) -> PyResult<usize> {
        let matrix = extract_f32_matrix(embeddings)?;
        py.allow_threads(|| {
            self.inner
                .insert_batch(&ids, &texts, &matrix, &scores, &concepts)
                .map_err(storage_err)
        })
    }

    fn search_ann(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let q = extract_f32_vec(query_embedding)?;
        py.allow_threads(|| self.inner.search_ann(&q, top_k).map_err(storage_err))
    }

    fn search_ann_candidates(
        &self,
        py: Python<'_>,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let q = extract_f32_vec(query_embedding)?;
        py.allow_threads(|| self.inner.search_ann_candidates(&q, top_k).map_err(storage_err))
    }

    #[pyo3(signature = (query_text, query_embedding, top_k))]
    fn search(
        &self,
        py: Python<'_>,
        query_text: &str,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
    ) -> PyResult<Option<Vec<(String, f32)>>> {
        let q = extract_f32_vec(query_embedding)?;
        py.allow_threads(|| self.inner.search(query_text, &q, top_k).map_err(storage_err))
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
}

#[pymodule]
fn turbomemory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryEngine>()?;
    Ok(())
}
