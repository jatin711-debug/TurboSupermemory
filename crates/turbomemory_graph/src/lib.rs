//! Cognitive graph layer for TurboSuperMemory.
//!
//! Implements an in-memory episodic-semantic graph, BM25 lexical triggering,
//! a bounded one-pass graph expansion that augments ANN candidates
//! ([`SpreadingActivation::search`]), a deterministic Compressed Cognitive
//! State (CCS) stub, and lightweight concept extraction from text.
//!
//! The expansion is deliberately a single bounded pass. The earlier
//! multi-iteration spreading activation with lateral inhibition and a
//! Feeling-of-Knowing gate was removed from the query path (see
//! `docs/cognitive_graph.md`); nothing here gates a query.

pub mod activation;
pub mod bm25;
pub mod ccs;
pub mod extract;
pub mod graph;

pub use activation::{SpreadingActivation, SpreadingConfig};
pub use bm25::{tokenize, Bm25Index};
pub use ccs::{
    step_session, step_session_with_compressor, CognitiveCompressor, CompressedCognitiveState,
    DeterministicCompressor, LlmCompressor,
};
pub use extract::{
    extract_concepts, extract_concepts_with_config, has_opposition_marker, merge_concepts,
    merge_concepts_with_config, text_jaccard_similarity, ConceptVocabulary, ExtractorConfig,
};
pub use graph::{
    ConceptKind, Edge, EdgeKind, GraphStats, MemoryGraph, Node, NodeId, VocabularyEvolutionStats,
};
