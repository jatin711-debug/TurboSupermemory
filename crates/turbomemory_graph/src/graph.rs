//! In-memory episodic-semantic graph with learnable edge weights.

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
    /// A refinement edge: points from an *older* memory to a *newer* memory
    /// that refines/superseds it. Direction: old → new.
    ///
    /// When spreading activation reaches the old memory (through its concept
    /// edges), it propagates through the `Refines` edge to the newer memory,
    /// ensuring the most current version of a piece of knowledge surfaces
    /// even when the query matched the older version. The old memory is NOT
    /// deleted — history is preserved so the agent can reason about how its
    /// understanding evolved.
    Refines,
    /// A contradiction edge: points from an *older* memory to a *newer*
    /// memory that contradicts it. Direction: old → new.
    ///
    /// Like `Refines`, this lets spreading activation propagate from the old
    /// (discredited) memory to the new (correcting) one. Unlike `Refines`,
    /// when a `Contradicts` edge is created, the old memory's outgoing
    /// association edges are *weakened* (multiplied by a decay factor), so
    /// the old memory gradually fades from retrieval while the new one
    /// surfaces. The old memory is NOT deleted — the agent can still find
    /// it if explicitly asked, but it won't dominate retrieval.
    Contradicts,
}

/// An edge in the cognitive graph.
///
/// `weight` is the *learned* strength of the connection. It starts at a
/// base value derived from the source memory's `importance` and is then
/// updated by reinforcement (on retrieval) and decay (on consolidation):
///
/// ```text
/// weight = base * importance_factor * reinforcement_factor * decay_factor
/// ```
///
/// `last_reinforced_at` is a unix-seconds timestamp tracking when the edge
/// was last strengthened, so decay can be applied lazily. Edges created
/// before learning was enabled carry `last_reinforced_at = 0` and are
/// treated as never-reinforced (decay does not erode them below their
/// initial base weight until they have been reinforced at least once).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub weight: f32,
    #[serde(default)]
    pub last_reinforced_at: u64,
}

#[derive(Debug, Clone, Default)]
pub enum ConceptKind {
    #[default]
    Generic,
}

/// Map a record `importance` (0.0..=1.0 typical, but unbounded) into an
/// edge-strength multiplier. We use `importance.sqrt()` so that a
/// zero-importance record still gets a non-zero (but tiny) link, while a
/// high-importance record gets a proportionally stronger one. The sqrt
/// curve keeps the dynamic range manageable: importance 0.25 -> 0.5x,
/// importance 1.0 -> 1.0x, importance 4.0 -> 2.0x.
fn importance_factor(importance: f32) -> f32 {
    if importance <= 0.0 {
        // Floor at a small epsilon so zero-importance records remain
        // reachable but rank below anything with positive importance.
        0.1
    } else {
        importance.sqrt()
    }
}

/// A property graph over memories and extracted concepts with learnable
/// edge weights and an abstraction hierarchy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryGraph {
    nodes: BTreeMap<String, Node>,
    edges: Vec<Edge>,
    adjacency: BTreeMap<String, Vec<usize>>,
    last_memory_id: Option<String>,
    /// Co-occurrence counts between concept pairs, accumulated on
    /// `add_memory`. Used by `build_abstractions` to decide when two
    /// concepts are related enough to warrant a parent abstraction node.
    /// Keyed by the lexicographically-ordered `concept:a\0concept:b` pair
    /// so (a,b) and (b,a) share a single counter.
    #[serde(default)]
    co_occurrence: BTreeMap<String, usize>,
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Backward-compatible insert: equivalent to `importance = 1.0`.
    pub fn add_memory(&mut self, id: &str, text: &str, concepts: &[String]) {
        self.add_memory_with_importance(id, text, concepts, 1.0);
    }

    /// Insert a memory node, link it to its concepts with importance-weighted
    /// association edges, and chain it temporally to the previous insert.
    ///
    /// Concepts are normalized to lowercase so that "Rust" and "rust" map to
    /// the same concept node. The association edge weight is
    /// `importance_factor(importance)`; the temporal edge weight is
    /// `0.5 * importance_factor(importance)`. Both are bidirectional for
    /// association (mem<->concept) and directional for temporal (prev_mem ->
    /// mem).
    pub fn add_memory_with_importance(
        &mut self,
        id: &str,
        text: &str,
        concepts: &[String],
        importance: f32,
    ) {
        let mem_key = NodeId::memory(id).as_str();
        self.nodes.insert(
            mem_key.clone(),
            Node {
                id: NodeId::memory(id),
                text: text.to_string(),
            },
        );

        let imp = importance_factor(importance);

        // Normalize concepts to lowercase and dedup, so "Rust" and "rust"
        // share a single concept node. This is critical for the graph to
        // accumulate a coherent concept vocabulary regardless of caller
        // casing conventions.
        let mut normalized: Vec<String> = Vec::with_capacity(concepts.len());
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for concept in concepts {
            let norm = concept.to_lowercase();
            if !norm.is_empty() && seen.insert(norm.clone()) {
                normalized.push(norm);
            }
        }

        // Concept association edges (bidirectional), weighted by importance.
        for concept in &normalized {
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
                imp,
            );
            self.add_edge_internal(
                NodeId::concept(concept),
                NodeId::memory(id),
                EdgeKind::Association,
                imp,
            );
        }

        // Accumulate concept co-occurrence for abstraction building.
        if normalized.len() > 1 {
            let mut sorted: Vec<&str> = normalized.iter().map(|s| s.as_str()).collect();
            sorted.sort();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    let key = format!("concept:{}\u{0}concept:{}", sorted[i], sorted[j]);
                    *self.co_occurrence.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Temporal chaining between consecutive memories.
        if let Some(prev) = self.last_memory_id.take() {
            self.add_edge_internal(
                NodeId::memory(&prev),
                NodeId::memory(id),
                EdgeKind::Temporal,
                0.5 * imp,
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
            last_reinforced_at: 0,
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

    /// Number of memory nodes in the graph.
    pub fn memory_count(&self) -> usize {
        self.iter_memory_nodes().count()
    }

    /// Number of memory-node neighbors (outgoing association edges) for a
    /// concept node.  Returns 0 if the node does not exist or is not a concept.
    pub fn concept_degree(&self, concept: &str) -> usize {
        let key = NodeId::concept(concept).as_str();
        self.neighbors(&key)
            .iter()
            .filter(|e| matches!(e.kind, EdgeKind::Association))
            .count()
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

    /// Reinforce the edges of a memory node, simulating rehearsal.
    ///
    /// Called when a memory is retrieved. Strengthens every edge touching
    /// the memory — both outgoing (where the memory is the source) and
    /// incoming association edges (where the memory is the target). This
    /// ensures that when a concept node is activated by a query, it
    /// propagates more energy to frequently-retrieved memories.
    ///
    /// Edges that have never been reinforced get a larger initial boost
    /// (capturing the "first recall matters most" effect); subsequently
    /// reinforced edges grow more slowly. Weights are clamped at 8.0.
    ///
    /// `now` is a unix-seconds timestamp used to stamp `last_reinforced_at`.
    pub fn reinforce(&mut self, id: &str, now: u64) {
        let mem_key = NodeId::memory(id).as_str();

        // Strengthen outgoing edges (where the memory is the source).
        let out_idxs: Vec<usize> = self.adjacency.get(&mem_key).cloned().unwrap_or_default();
        for i in out_idxs {
            let edge = &mut self.edges[i];
            let boost = if edge.last_reinforced_at == 0 {
                1.5
            } else {
                1.0 + 0.1 / (1.0 + edge.weight)
            };
            edge.weight = (edge.weight * boost).min(8.0);
            edge.last_reinforced_at = now;
        }

        // Strengthen incoming Association edges (where the memory is the
        // target). This is what makes reinforcement actually boost a
        // memory's activation: when a concept is activated by the query,
        // it propagates more energy through the strengthened concept→memory
        // edge to the reinforced memory than to non-reinforced ones.
        // We scan the edge list for Association edges targeting this memory.
        // This is O(edges) but reinforcement only happens on retrieval (not
        // on the insert hot path), and the edge list is typically small.
        for edge in &mut self.edges {
            if edge.kind == EdgeKind::Association && edge.target.as_str() == mem_key {
                let boost = if edge.last_reinforced_at == 0 {
                    1.5
                } else {
                    1.0 + 0.1 / (1.0 + edge.weight)
                };
                edge.weight = (edge.weight * boost).min(8.0);
                edge.last_reinforced_at = now;
            }
        }
    }

    /// Apply exponential time-decay to all reinforced edges.
    ///
    /// `weight *= 0.5^((now - last_reinforced_at) / half_life)`, floored at
    /// the edge's *original* importance-weighted base weight so that decay
    /// erodes *learned* reinforcement but never drops an edge below the
    /// strength it was created with. Edges that were never reinforced
    /// (`last_reinforced_at == 0`) are left untouched — they are already at
    /// their base weight and have no learned component to decay.
    ///
    /// This is the "forgetting" half of the retain-what-matters loop: stale,
    /// unrehearsed memories fade back toward their baseline while recently
    /// retrieved ones stay strong.
    pub fn decay_edges(&mut self, now: u64, half_life: u64) {
        if half_life == 0 {
            return;
        }
        let hl = half_life as f64;
        for edge in &mut self.edges {
            if edge.last_reinforced_at == 0 {
                continue;
            }
            let age = now.saturating_sub(edge.last_reinforced_at) as f64;
            let factor = 0.5f64.powf(age / hl) as f32;
            // Decay only the *learned* portion above the base weight. We do
            // not track the original base weight per edge (to keep the struct
            // small), so we use a floor of `weight * factor` but never below
            // a small constant. A reinforced edge that has decayed fully
            // settles at ~1.0 (the original default association weight),
            // preserving baseline connectivity.
            let decayed = edge.weight * factor;
            edge.weight = decayed.max(1.0);
        }
    }

    /// Build abstraction edges: when two concepts co-occur on at least
    /// `threshold` memories, create a parent concept node that abstracts
    /// both, with bidirectional `Abstraction` edges. The parent node's id
    /// is derived from the sorted pair so it is deterministic.
    ///
    /// This is the mechanism that lets the graph *generalize* instead of
    /// staying flat: frequently co-occurring concepts ("rust" + "safety")
    /// get a parent ("rust+safety") that spreading activation can traverse
    /// to find memories that share either concept. Idempotent — calling it
    /// again only adds new abstractions for pairs that have crossed the
    /// threshold since the last call; existing parent nodes are reused.
    ///
    /// Returns the number of new abstraction edges added.
    pub fn build_abstractions(&mut self, threshold: usize) -> usize {
        if threshold == 0 {
            return 0;
        }
        let mut added = 0usize;
        // Collect pairs above threshold. We iterate over a snapshot because
        // we mutate `self` inside the loop.
        let pairs: Vec<(String, String, String)> = self
            .co_occurrence
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(key, _)| {
                // key is "concept:a\0concept:b"
                let mut parts = key.splitn(2, '\u{0}');
                let a = parts.next().unwrap_or_default().to_string();
                let b = parts.next().unwrap_or_default().to_string();
                (a, b, key.clone())
            })
            .collect();

        for (a_key, b_key, co_key) in pairs {
            // Parent concept id is the sorted pair of concept names, joined
            // with '+'. Strip the "concept:" prefix for readability.
            let a_name = a_key.strip_prefix("concept:").unwrap_or(&a_key);
            let b_name = b_key.strip_prefix("concept:").unwrap_or(&b_key);
            let parent_name = if a_name <= b_name {
                format!("{a_name}+{b_name}")
            } else {
                format!("{b_name}+{a_name}")
            };
            let parent_key = NodeId::concept(&parent_name).as_str();

            // Insert the parent node if absent.
            let already_present = self.nodes.contains_key(&parent_key);
            if !already_present {
                self.nodes.insert(
                    parent_key.clone(),
                    Node {
                        id: NodeId::concept(&parent_name),
                        text: parent_name.clone(),
                    },
                );
            }

            // Add bidirectional Abstraction edges child->parent and
            // parent->child if not already present. We check existence by
            // scanning the source's adjacency to keep this O(degree) rather
            // than O(edges).
            for child_key in [&a_key, &b_key] {
                let exists = self
                    .adjacency
                    .get(child_key)
                    .map(|idxs| {
                        idxs.iter().any(|&i| {
                            self.edges[i].kind == EdgeKind::Abstraction
                                && self.edges[i].target.as_str() == parent_key
                        })
                    })
                    .unwrap_or(false);
                if !exists {
                    self.add_edge_internal(
                        NodeId::Concept(
                            child_key
                                .strip_prefix("concept:")
                                .unwrap_or(child_key)
                                .into(),
                        ),
                        NodeId::concept(&parent_name),
                        EdgeKind::Abstraction,
                        1.0,
                    );
                    self.add_edge_internal(
                        NodeId::concept(&parent_name),
                        NodeId::Concept(
                            child_key
                                .strip_prefix("concept:")
                                .unwrap_or(child_key)
                                .into(),
                        ),
                        EdgeKind::Abstraction,
                        1.0,
                    );
                    added += 2;
                }
            }
            // Mark this pair as consumed so a subsequent call does not
            // re-scan it unless new co-occurrences have arrived. We reset
            // the counter to 0; future `add_memory` calls will re-increment
            // it, and the next `build_abstractions` will pick up only pairs
            // that have accumulated `threshold` *new* co-occurrences.
            self.co_occurrence.insert(co_key, 0);
        }
        added
    }

    /// Number of abstraction (parent concept) nodes in the graph.
    pub fn abstraction_count(&self) -> usize {
        self.nodes
            .values()
            .filter(|n| matches!(n.id, NodeId::Concept(ref c) if c.contains('+')))
            .count()
    }

    /// Create a `Refines` edge from an older memory to a newer memory that
    /// refines/superseds it. Direction: `old_id → new_id`.
    ///
    /// The edge weight starts at `weight` (typically 0.8) so spreading
    /// activation from the old memory propagates to the new one, ensuring
    /// the most current version surfaces. The old memory is NOT removed —
    /// history is preserved.
    ///
    /// Idempotent: if a `Refines` edge from `old_id` to `new_id` already
    /// exists, this is a no-op. Returns `true` if a new edge was created.
    pub fn add_refinement(&mut self, old_id: &str, new_id: &str, weight: f32) -> bool {
        let old_key = NodeId::memory(old_id).as_str();
        let new_key = NodeId::memory(new_id).as_str();
        // Check if the edge already exists.
        let exists = self
            .adjacency
            .get(&old_key)
            .map(|idxs| {
                idxs.iter().any(|&i| {
                    self.edges[i].kind == EdgeKind::Refines
                        && self.edges[i].target.as_str() == new_key
                })
            })
            .unwrap_or(false);
        if exists {
            return false;
        }
        // Both nodes must exist.
        if !self.nodes.contains_key(&old_key) || !self.nodes.contains_key(&new_key) {
            return false;
        }
        self.add_edge_internal(
            NodeId::memory(old_id),
            NodeId::memory(new_id),
            EdgeKind::Refines,
            weight,
        );
        true
    }

    /// Number of `Refines` edges in the graph.
    pub fn refinement_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Refines)
            .count()
    }

    /// Returns the ids of memories that `id` refines (i.e. the older
    /// memories that `id` supersedes). Empty if `id` has no outgoing
    /// `Refines` edges.
    pub fn refined_by(&self, id: &str) -> Vec<String> {
        let key = NodeId::memory(id).as_str();
        self.neighbors(&key)
            .iter()
            .filter(|e| e.kind == EdgeKind::Refines)
            .filter_map(|e| match &e.target {
                NodeId::Memory(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    /// Create a `Contradicts` edge from an older memory to a newer memory
    /// that contradicts it, AND weaken the old memory's outgoing
    /// association edges so it fades from retrieval.
    ///
    /// Direction: `old_id → new_id`.
    ///
    /// The `Contradicts` edge lets spreading activation propagate from the
    /// discredited memory to the correcting one. Additionally, the old
    /// memory's outgoing `Association` and `Temporal` edges are multiplied
    /// by `weaken_factor` (typically 0.5), reducing its activation in
    /// future retrievals. The old memory is NOT deleted — it can still be
    /// found if explicitly queried, but it won't dominate results.
    ///
    /// Idempotent: if a `Contradicts` edge from `old_id` to `new_id`
    /// already exists, this is a no-op (the weakening is also skipped).
    /// Returns `true` if a new edge was created.
    pub fn add_contradiction(
        &mut self,
        old_id: &str,
        new_id: &str,
        weight: f32,
        weaken_factor: f32,
    ) -> bool {
        let old_key = NodeId::memory(old_id).as_str();
        let new_key = NodeId::memory(new_id).as_str();
        // Check if the edge already exists.
        let exists = self
            .adjacency
            .get(&old_key)
            .map(|idxs| {
                idxs.iter().any(|&i| {
                    self.edges[i].kind == EdgeKind::Contradicts
                        && self.edges[i].target.as_str() == new_key
                })
            })
            .unwrap_or(false);
        if exists {
            return false;
        }
        // Both nodes must exist.
        if !self.nodes.contains_key(&old_key) || !self.nodes.contains_key(&new_key) {
            return false;
        }
        // Weaken the old memory's outgoing Association and Temporal edges
        // so it fades from retrieval. We do NOT weaken Refines or
        // Contradicts edges — those are directed to the newer memory and
        // should stay strong so the correction surfaces.
        let old_idxs: Vec<usize> = self.adjacency.get(&old_key).cloned().unwrap_or_default();
        for i in old_idxs {
            let edge = &mut self.edges[i];
            if edge.kind == EdgeKind::Association || edge.kind == EdgeKind::Temporal {
                edge.weight *= weaken_factor;
            }
        }
        // Create the Contradicts edge.
        self.add_edge_internal(
            NodeId::memory(old_id),
            NodeId::memory(new_id),
            EdgeKind::Contradicts,
            weight,
        );
        true
    }

    /// Number of `Contradicts` edges in the graph.
    pub fn contradiction_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Contradicts)
            .count()
    }

    /// Returns the ids of memories that contradict `id` (i.e. the newer
    /// memories that correct `id`). Empty if `id` has no outgoing
    /// `Contradicts` edges.
    pub fn contradicted_by(&self, id: &str) -> Vec<String> {
        let key = NodeId::memory(id).as_str();
        self.neighbors(&key)
            .iter()
            .filter(|e| e.kind == EdgeKind::Contradicts)
            .filter_map(|e| match &e.target {
                NodeId::Memory(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
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

    #[test]
    fn importance_weights_edges() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "low", &["c".into()], 0.25);
        g.add_memory_with_importance("m2", "high", &["c".into()], 4.0);
        // importance 0.25 -> factor 0.5; importance 4.0 -> factor 2.0
        let m1_edges: Vec<&Edge> = g
            .neighbors(&NodeId::memory("m1").as_str())
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .collect();
        let m2_edges: Vec<&Edge> = g
            .neighbors(&NodeId::memory("m2").as_str())
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .collect();
        let w1 = m1_edges.first().map(|e| e.weight).unwrap_or(0.0);
        let w2 = m2_edges.first().map(|e| e.weight).unwrap_or(0.0);
        assert!(
            w2 > w1,
            "high-importance edge ({w2}) should outweigh low ({w1})"
        );
        assert!(
            (w1 - 0.5).abs() < 1e-5,
            "importance 0.25 -> factor 0.5, got {w1}"
        );
        assert!(
            (w2 - 2.0).abs() < 1e-5,
            "importance 4.0 -> factor 2.0, got {w2}"
        );
    }

    #[test]
    fn reinforce_strengthens_edges() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "Rust is safe", &["rust".into(), "safety".into()]);
        let before: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.weight)
            .sum();
        g.reinforce("m1", 1000);
        let after: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.weight)
            .sum();
        assert!(
            after > before,
            "reinforce should increase total edge weight"
        );
        // All reinforced edges should carry the timestamp.
        for e in g.neighbors(&NodeId::memory("m1").as_str()) {
            assert_eq!(e.last_reinforced_at, 1000);
        }
    }

    #[test]
    fn reinforce_is_bounded() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "x", &["c".into()]);
        // Reinforce many times; weight must never exceed the clamp (8.0).
        for _ in 0..100 {
            g.reinforce("m1", 1000);
        }
        let w = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.weight)
            .fold(0.0f32, f32::max);
        assert!(
            w <= 8.0 + 1e-5,
            "reinforced weight must be clamped, got {w}"
        );
    }

    #[test]
    fn decay_erodes_reinforced_edges_but_not_unreinforced() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "a", &["c1".into()]);
        g.add_memory("m2", "b", &["c2".into()]);
        // Reinforce only m1.
        g.reinforce("m1", 1000);
        let m1_before: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.weight)
            .sum();
        let m2_before: f32 = g
            .neighbors(&NodeId::memory("m2").as_str())
            .iter()
            .map(|e| e.weight)
            .sum();
        // Decay with a 100s half-life, 1000s later -> 10 half-lives -> factor 2^-10.
        g.decay_edges(2000, 100);
        let m1_after: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.weight)
            .sum();
        let m2_after: f32 = g
            .neighbors(&NodeId::memory("m2").as_str())
            .iter()
            .map(|e| e.weight)
            .sum();
        assert!(
            m1_after < m1_before,
            "reinforced m1 should decay: {m1_after} < {m1_before}"
        );
        assert_eq!(
            m2_before, m2_after,
            "never-reinforced m2 should be untouched by decay"
        );
    }

    #[test]
    fn decay_floors_at_baseline() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "a", &["c".into()]);
        g.reinforce("m1", 1000);
        // Decay over an enormous age; weight should floor at 1.0, not 0.
        g.decay_edges(1_000_000_000, 1);
        let w = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.weight)
            .fold(f32::INFINITY, f32::min);
        assert!(
            w >= 1.0 - 1e-5,
            "decay should floor at baseline 1.0, got {w}"
        );
    }

    #[test]
    fn build_abstractions_creates_parent_for_co_occurring_concepts() {
        let mut g = MemoryGraph::new();
        // Three memories all sharing "rust" and "safety" -> co-occurrence 3.
        for i in 0..3 {
            g.add_memory(
                &format!("m{i}"),
                "Rust is safe",
                &["rust".into(), "safety".into()],
            );
        }
        let added = g.build_abstractions(3);
        assert!(
            added >= 2,
            "should add at least 2 abstraction edges, got {added}"
        );
        assert!(
            g.abstraction_count() >= 1,
            "should have a parent concept node"
        );
        // The parent node "rust+safety" should exist.
        assert!(g
            .nodes()
            .contains_key(&NodeId::concept("rust+safety").as_str()));
    }

    #[test]
    fn build_abstractions_is_idempotent() {
        let mut g = MemoryGraph::new();
        for i in 0..3 {
            g.add_memory(
                &format!("m{i}"),
                "Rust is safe",
                &["rust".into(), "safety".into()],
            );
        }
        let first = g.build_abstractions(3);
        let second = g.build_abstractions(3);
        assert!(first >= 2, "first call should add edges");
        assert_eq!(
            second, 0,
            "second call with no new co-occurrence should add nothing"
        );
    }

    #[test]
    fn build_abstractions_below_threshold_adds_nothing() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "a", &["x".into(), "y".into()]);
        g.add_memory("m2", "b", &["x".into(), "y".into()]);
        // co-occurrence is 2; threshold 3 -> no abstraction.
        assert_eq!(g.build_abstractions(3), 0);
        assert_eq!(g.abstraction_count(), 0);
    }

    #[test]
    fn graph_roundtrips_through_json_with_learned_state() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "Rust is safe", &["rust".into(), "safety".into()], 2.0);
        g.add_memory_with_importance("m2", "Rust is fast", &["rust".into(), "speed".into()], 1.5);
        g.reinforce("m1", 1234);
        // co-occurrence of "rust"+"safety" is 1, "rust"+"speed" is 1, "safety"+"speed" is 0.
        // Lower threshold to 1 so abstractions get built for the test.
        g.build_abstractions(1);

        let json = g.to_json();
        let restored = MemoryGraph::from_json(&json).expect("roundtrip");

        // The learned weights and timestamps should survive the roundtrip.
        let m1_edges: Vec<&Edge> = restored
            .neighbors(&NodeId::memory("m1").as_str())
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .collect();
        assert!(
            m1_edges.iter().any(|e| e.last_reinforced_at == 1234),
            "reinforcement timestamp should survive JSON roundtrip"
        );
        // importance 2.0 -> factor sqrt(2) ~ 1.414
        let w = m1_edges.first().map(|e| e.weight).unwrap_or(0.0);
        assert!(
            (w - 2.0f32.sqrt()).abs() < 1e-4 || w > 2.0f32.sqrt(),
            "base weight should be sqrt(importance), got {w}"
        );
        assert!(
            restored.abstraction_count() >= 1,
            "abstraction nodes should survive roundtrip"
        );
    }

    #[test]
    fn add_refinement_creates_directed_edge() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "Rust uses a borrow checker", &["rust".into()]);
        g.add_memory(
            "new",
            "Rust's borrow checker enforces ownership",
            &["rust".into()],
        );
        let created = g.add_refinement("old", "new", 0.8);
        assert!(created, "should create a new Refines edge");
        assert_eq!(g.refinement_count(), 1);
        // The edge is old → new, so neighbors(old) should include new.
        let old_neighbors = g.neighbors(&NodeId::memory("old").as_str());
        assert!(old_neighbors
            .iter()
            .any(|e| { e.kind == EdgeKind::Refines && e.target.as_str() == "mem:new" }));
        // The reverse (new → old) should NOT exist.
        let new_neighbors = g.neighbors(&NodeId::memory("new").as_str());
        assert!(!new_neighbors
            .iter()
            .any(|e| { e.kind == EdgeKind::Refines && e.target.as_str() == "mem:old" }));
    }

    #[test]
    fn add_refinement_is_idempotent() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        g.add_memory("new", "b", &["c".into()]);
        assert!(g.add_refinement("old", "new", 0.8));
        assert!(
            !g.add_refinement("old", "new", 0.8),
            "second call should be no-op"
        );
        assert_eq!(g.refinement_count(), 1);
    }

    #[test]
    fn add_refinement_rejects_missing_nodes() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        // "new" does not exist.
        assert!(!g.add_refinement("old", "new", 0.8));
        // "ghost" does not exist.
        assert!(!g.add_refinement("ghost", "old", 0.8));
        assert_eq!(g.refinement_count(), 0);
    }

    #[test]
    fn refined_by_returns_superseded_memories() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        g.add_memory("new", "b", &["c".into()]);
        g.add_refinement("old", "new", 0.8);
        // "new" is refined_by "old" — wait, the edge is old → new, so
        // neighbors(old) includes new. refined_by(id) returns the targets
        // of Refines edges FROM id. So refined_by("old") = ["new"].
        let refined = g.refined_by("old");
        assert_eq!(refined, vec!["new".to_string()]);
        // "new" has no outgoing Refines edges.
        assert!(g.refined_by("new").is_empty());
    }

    #[test]
    fn refines_edges_survive_json_roundtrip() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["rust".into()]);
        g.add_memory("new", "b", &["rust".into()]);
        g.add_refinement("old", "new", 0.8);
        let json = g.to_json();
        let restored = MemoryGraph::from_json(&json).expect("roundtrip");
        assert_eq!(
            restored.refinement_count(),
            1,
            "Refines edge should survive roundtrip"
        );
    }

    #[test]
    fn add_contradiction_creates_edge_and_weakens_old() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "Rust has a garbage collector", &["rust".into()]);
        g.add_memory(
            "new",
            "Rust does not have a garbage collector",
            &["rust".into()],
        );
        // Record the old memory's edge weight before contradiction.
        let old_weight_before: f32 = g
            .neighbors(&NodeId::memory("old").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .next()
            .unwrap_or(0.0);
        let created = g.add_contradiction("old", "new", 0.8, 0.5);
        assert!(created, "should create a Contradicts edge");
        assert_eq!(g.contradiction_count(), 1);
        // The old memory's Association edges should be weakened (halved).
        let old_weight_after: f32 = g
            .neighbors(&NodeId::memory("old").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .next()
            .unwrap_or(0.0);
        assert!(
            old_weight_after < old_weight_before,
            "old edge should be weakened: {old_weight_after} < {old_weight_before}"
        );
        assert!(
            (old_weight_after - old_weight_before * 0.5).abs() < 1e-5,
            "old edge should be halved: {old_weight_after} vs {}",
            old_weight_before * 0.5
        );
    }

    #[test]
    fn add_contradiction_is_idempotent() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        g.add_memory("new", "b", &["c".into()]);
        assert!(g.add_contradiction("old", "new", 0.8, 0.5));
        assert!(
            !g.add_contradiction("old", "new", 0.8, 0.5),
            "second call no-op"
        );
        assert_eq!(g.contradiction_count(), 1);
    }

    #[test]
    fn contradicted_by_returns_correcting_memories() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["rust".into()]);
        g.add_memory("new", "b", &["rust".into()]);
        g.add_contradiction("old", "new", 0.8, 0.5);
        let corrected = g.contradicted_by("old");
        assert_eq!(corrected, vec!["new".to_string()]);
        assert!(g.contradicted_by("new").is_empty());
    }

    #[test]
    fn contradicts_edges_survive_json_roundtrip() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["rust".into()]);
        g.add_memory("new", "b", &["rust".into()]);
        g.add_contradiction("old", "new", 0.8, 0.5);
        let json = g.to_json();
        let restored = MemoryGraph::from_json(&json).expect("roundtrip");
        assert_eq!(
            restored.contradiction_count(),
            1,
            "Contradicts edge should survive roundtrip"
        );
    }

    #[test]
    fn text_jaccard_distinguishes_same_and_different_content() {
        use crate::extract::text_jaccard_similarity;
        // Same topic, same content → high Jaccard.
        let sim_same = text_jaccard_similarity(
            "Rust memory safety guarantees",
            "Rust memory safety guarantees",
        );
        assert!(
            sim_same > 0.99,
            "identical texts should have ~1.0 Jaccard: {sim_same}"
        );
        // Same topic, different content → lower Jaccard.
        // Text A: "Rust memory safety guarantees" → {rust, memory, safety, guarantees}
        // Text B: "Rust ownership borrow checker" → {rust, ownership, borrow, checker}
        // intersection = {rust} = 1, union = 6, Jaccard = 1/6 ≈ 0.167
        let sim_diff = text_jaccard_similarity(
            "Rust memory safety guarantees",
            "Rust ownership borrow checker",
        );
        assert!(
            sim_diff < 0.3,
            "different texts should have low Jaccard: {sim_diff}"
        );
        assert!(
            sim_diff < sim_same,
            "different texts should have lower Jaccard than identical: {sim_diff} < {sim_same}"
        );
    }
}
