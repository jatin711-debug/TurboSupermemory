//! Cognitive graph layer for TurboSuperMemory.
//!
//! Implements an in-memory episodic-semantic graph, BM25 lexical triggering,
//! spreading activation with lateral inhibition, Feeling-of-Knowing gating,
//! and a deterministic Compressed Cognitive State (CCS) stub.

pub mod activation;
pub mod bm25;
pub mod ccs;
pub mod graph;

pub use activation::{SpreadingActivation, SpreadingConfig};
pub use bm25::{tokenize, Bm25Index};
pub use ccs::{step_session, CompressedCognitiveState};
pub use graph::{ConceptKind, Edge, EdgeKind, MemoryGraph, Node, NodeId};
