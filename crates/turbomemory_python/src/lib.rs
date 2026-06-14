//! PyO3 bindings for TurboSuperMemory.
//!
//! Exposes a single `MemoryEngine` class with the exact API expected by
//! `verify.py` and `benchmark.py`.

use parking_lot::Mutex;
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
    inner: Arc<Mutex<StorageEngine>>,
    #[allow(dead_code)]
    handler: Mutex<Option<UpdateHandler>>,
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
        let engine = Arc::new(StorageEngine::open(db_path, config).map_err(storage_err)?);
        let handler = engine
            .config()
            .auto_consolidation_interval
            .map(|interval| UpdateHandler::new(engine.clone(), interval));
        let inner = Arc::new(Mutex::new((*engine).clone()));
        Ok(Self {
            inner,
            handler: Mutex::new(handler),
        })
    }

    fn insert(
        &mut self,
        id: &str,
        text: &str,
        embedding: &Bound<'_, PyAny>,
        importance_score: f32,
        concepts: Vec<String>,
    ) -> PyResult<bool> {
        let emb = extract_f32_vec(embedding)?;
        self.inner
            .lock()
            .insert(id, text, &emb, importance_score, &concepts)
            .map_err(storage_err)
    }

    #[pyo3(signature = (ids, texts, embeddings, scores, concepts))]
    fn insert_batch(
        &mut self,
        ids: Vec<String>,
        texts: Vec<String>,
        embeddings: &Bound<'_, PyAny>,
        scores: Vec<f32>,
        concepts: Vec<Vec<String>>,
    ) -> PyResult<usize> {
        let matrix = extract_f32_matrix(embeddings)?;
        self.inner
            .lock()
            .insert_batch(&ids, &texts, &matrix, &scores, &concepts)
            .map_err(storage_err)
    }

    fn search_ann(
        &self,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let q = extract_f32_vec(query_embedding)?;
        self.inner.lock().search_ann(&q, top_k).map_err(storage_err)
    }

    fn search_ann_candidates(
        &self,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
    ) -> PyResult<Vec<(String, f32)>> {
        let q = extract_f32_vec(query_embedding)?;
        self.inner
            .lock()
            .search_ann_candidates(&q, top_k)
            .map_err(storage_err)
    }

    #[pyo3(signature = (query_text, query_embedding, top_k))]
    fn search(
        &mut self,
        query_text: &str,
        query_embedding: &Bound<'_, PyAny>,
        top_k: usize,
    ) -> PyResult<Option<Vec<(String, f32)>>> {
        let q = extract_f32_vec(query_embedding)?;
        self.inner
            .lock()
            .search(query_text, &q, top_k)
            .map_err(storage_err)
    }

    fn step_session(&mut self, user_input: &str, assistant_response: &str) -> PyResult<String> {
        self.inner
            .lock()
            .step_session(user_input, assistant_response)
            .map_err(storage_err)
    }

    fn trigger_consolidation(&mut self) -> PyResult<(usize, usize)> {
        self.inner
            .lock()
            .trigger_consolidation()
            .map_err(storage_err)
    }

    fn flush(&self) -> PyResult<()> {
        self.inner.lock().flush().map_err(storage_err)
    }
}

#[pymodule]
fn turbomemory(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMemoryEngine>()?;
    Ok(())
}
