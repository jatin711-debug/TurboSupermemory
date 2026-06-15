//! In-memory episodic-semantic graph.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeId {
    Memory(String),
    Concept(String),
}

impl NodeId {
    pub fn memory(id: impl Into<String>) -> Self {
        NodeId::Memory(id.into())
    }
    pub fn concept(id: impl Into<String>) -> Self {
        NodeId::Concept(id.into())
    }
    pub fn as_str(&self) -> String {
        match self {
            NodeId::Memory(s) => format!("mem:{s}"),
            NodeId::Concept(s) => format!("concept:{s}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeKind {
    Association,
    Temporal,
    Abstraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub weight: f32,
}

#[derive(Debug, Clone, Default)]
pub enum ConceptKind {
    #[default]
    Generic,
}

/// A property graph over memories and extracted concepts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryGraph {
    nodes: BTreeMap<String, Node>,
    edges: Vec<Edge>,
    adjacency: BTreeMap<String, Vec<usize>>,
    last_memory_id: Option<String>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_memory(&mut self, id: &str, text: &str, concepts: &[String]) {
        let mem_key = NodeId::memory(id).as_str();
        self.nodes.insert(
            mem_key.clone(),
            Node {
                id: NodeId::memory(id),
                text: text.to_string(),
            },
        );

        // Concept association edges
        for concept in concepts {
            let concept_key = NodeId::concept(concept).as_str();
            self.nodes
                .entry(concept_key.clone())
                .or_insert_with(|| Node {
                    id: NodeId::concept(concept.clone()),
                    text: concept.clone(),
                });
            self.add_edge_internal(
                NodeId::memory(id),
                NodeId::concept(concept),
                EdgeKind::Association,
                1.0,
            );
            // Bidirectional association
            self.add_edge_internal(
                NodeId::concept(concept),
                NodeId::memory(id),
                EdgeKind::Association,
                1.0,
            );
        }

        // Temporal chaining between consecutive memories
        if let Some(prev) = self.last_memory_id.take() {
            self.add_edge_internal(
                NodeId::memory(&prev),
                NodeId::memory(id),
                EdgeKind::Temporal,
                0.5,
            );
        }
        self.last_memory_id = Some(id.to_string());
    }

    /// Remove a memory node and all edges connected to it.
    pub fn remove_memory(&mut self, id: &str) {
        let mem_key = NodeId::memory(id).as_str();
        if self.nodes.remove(&mem_key).is_none() {
            return;
        }
        self.edges
            .retain(|e| e.source.as_str() != mem_key && e.target.as_str() != mem_key);
        self.rebuild_adjacency();
    }

    fn add_edge_internal(&mut self, source: NodeId, target: NodeId, kind: EdgeKind, weight: f32) {
        let idx = self.edges.len();
        self.edges.push(Edge {
            source: source.clone(),
            target,
            kind,
            weight,
        });
        self.adjacency.entry(source.as_str()).or_default().push(idx);
    }

    /// Rebuild the adjacency map from the current edge list without sorting.
    ///
    /// Used after operations that change edge indices (e.g. removal).  Edges are
    /// kept in insertion order during normal use so that add_memory stays O(1).
    fn rebuild_adjacency(&mut self) {
        self.adjacency.clear();
        for (idx, edge) in self.edges.iter().enumerate() {
            self.adjacency
                .entry(edge.source.as_str())
                .or_default()
                .push(idx);
        }
    }

    /// Sort edges and rebuild adjacency so iteration is deterministic across
    /// incremental builds and reloads.
    ///
    /// This is only called on explicit compaction or serialization, not on the
    /// insert hot path.
    pub fn compact(&mut self) {
        self.edges.sort_by(|a, b| {
            a.source
                .as_str()
                .cmp(&b.source.as_str())
                .then(a.target.as_str().cmp(&b.target.as_str()))
                .then(a.kind.cmp(&b.kind))
                .then(
                    a.weight
                        .partial_cmp(&b.weight)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        self.rebuild_adjacency();
    }

    fn normalize_edges(&mut self) {
        self.compact();
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn nodes(&self) -> &BTreeMap<String, Node> {
        &self.nodes
    }

    pub fn edges(&self) -> &[Edge] {
        &self.edges
    }

    pub fn neighbors(&self, node_key: &str) -> Vec<&Edge> {
        self.adjacency
            .get(node_key)
            .map(|idxs| idxs.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    pub fn iter_memory_nodes(&self) -> impl Iterator<Item = (&String, &Node)> {
        self.nodes.iter().filter(|(k, _)| k.starts_with("mem:"))
    }

    pub fn to_json(&self) -> String {
        // Return a deterministic serialization so reloads reproduce the same graph.
        let mut sorted = self.clone();
        sorted.compact();
        serde_json::to_string_pretty(&sorted).unwrap_or_default()
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let mut graph: Self = serde_json::from_str(s)?;
        graph.normalize_edges();
        Ok(graph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_adds_nodes_and_edges() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "Rust is safe", &["rust".into(), "safety".into()]);
        g.add_memory("m2", "Python is easy", &["python".into()]);
        assert_eq!(g.node_count(), 5); // 2 memories + 3 concepts
        assert!(g.edge_count() >= 6);
    }
}
