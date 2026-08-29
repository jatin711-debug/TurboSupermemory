//! In-memory episodic-semantic graph with learnable edge weights.
//!
//! Internal implementation uses integer-interned node IDs (u32) for O(1)
//! array indexing. The public API remains string-based for backward
//! compatibility.

use crate::extract::ConceptVocabulary;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Stable node identifier exposed to callers.
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
    /// Returns the canonical string key used for serialization and public APIs.
    pub fn as_str(&self) -> String {
        match self {
            NodeId::Memory(s) => format!("mem:{s}"),
            NodeId::Concept(s) => format!("concept:{s}"),
        }
    }
}

/// Internal node type tag (replaces string prefix for fast branching).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    Memory,
    Concept,
}

/// Internal node representation using integer ID.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalNode {
    pub kind: NodeKind,
    /// The original external ID (memory ID or concept name).
    pub external_id: String,
    pub text: String,
    /// For memory nodes: the `importance_factor` baseline.
    #[serde(default)]
    pub base_importance_factor: f32,
}

/// Internal edge representation using integer node IDs.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct InternalEdge {
    pub source: u32,
    pub target: u32,
    pub kind: EdgeKind,
    pub weight: f32,
    #[serde(default)]
    pub last_reinforced_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub text: String,
    #[serde(default)]
    pub base_importance_factor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EdgeKind {
    Association,
    Temporal,
    Abstraction,
    Refines,
    Contradicts,
}

/// An edge in the cognitive graph (public view).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub source: NodeId,
    pub target: NodeId,
    pub kind: EdgeKind,
    pub weight: f32,
    #[serde(default)]
    pub last_reinforced_at: u64,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub memory_count: usize,
    pub concept_count: usize,
    pub refinement_count: usize,
    pub contradiction_count: usize,
    pub abstraction_count: usize,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct VocabularyEvolutionStats {
    pub merged: usize,
    pub suppressed: usize,
    pub examined_pairs: usize,
}

#[derive(Debug, Clone, Default)]
pub enum ConceptKind {
    #[default]
    Generic,
}

fn importance_factor(importance: f32) -> f32 {
    // Non-finite importance (NaN/±inf) would poison edge weights and, via
    // partial_cmp in sort comparators, make edge ordering non-total — which
    // sort_by may panic on. Clamp it to the low-importance floor instead.
    if !importance.is_finite() || importance <= 0.0 {
        0.1
    } else {
        importance.sqrt()
    }
}

/// Maps external string keys to dense u32 node IDs. IDs are assigned in
/// insertion order and never reused, so iteration order is stable across
/// snapshots.
#[derive(Debug, Clone, Default)]
pub struct NodeInterner {
    key_to_id: HashMap<String, u32>,
    id_to_key: Vec<String>,
    id_to_node: Vec<InternalNode>,
}

impl NodeInterner {
    fn intern(&mut self, key: String, node: InternalNode) -> u32 {
        if let Some(&id) = self.key_to_id.get(&key) {
            return id;
        }
        let id = self.id_to_key.len() as u32;
        self.key_to_id.insert(key.clone(), id);
        self.id_to_key.push(key);
        self.id_to_node.push(node);
        id
    }

    fn resolve(&self, key: &str) -> Option<u32> {
        self.key_to_id.get(key).copied()
    }

    fn get(&self, id: u32) -> Option<&InternalNode> {
        self.id_to_node.get(id as usize)
    }

    fn get_mut(&mut self, id: u32) -> Option<&mut InternalNode> {
        self.id_to_node.get_mut(id as usize)
    }

    fn iter(&self) -> impl Iterator<Item = (u32, &String, &InternalNode)> {
        self.id_to_key
            .iter()
            .enumerate()
            .filter(|(i, k)| self.key_to_id.get(*k) == Some(&(*i as u32)))
            .map(|(i, k)| (i as u32, k, &self.id_to_node[i]))
    }

    fn memory_nodes(&self) -> impl Iterator<Item = (u32, &String, &InternalNode)> {
        self.iter()
            .filter(|(_, _, n)| matches!(n.kind, NodeKind::Memory))
    }
}

/// Adjacency stored as a HashMap<u32, Vec<usize>> for fast O(1) neighbor
/// lookup by integer node ID. This replaces the old String-keyed BTreeMap.
#[derive(Debug, Clone, Default)]
pub struct IntAdjacency {
    map: HashMap<u32, Vec<usize>>,
}

impl IntAdjacency {
    fn add(&mut self, source: u32, edge_idx: usize) {
        self.map.entry(source).or_default().push(edge_idx);
    }

    fn get(&self, source: u32) -> Option<&[usize]> {
        self.map.get(&source).map(|v| v.as_slice())
    }

    fn clear(&mut self) {
        self.map.clear();
    }

    fn rebuild(&mut self, edges: &[InternalEdge]) {
        self.clear();
        for (idx, edge) in edges.iter().enumerate() {
            self.add(edge.source, idx);
        }
    }
}

/// A property graph over memories and extracted concepts with learnable
/// edge weights and an abstraction hierarchy.
///
/// Internally uses integer node IDs (u32) for fast array-based access.
/// The public API remains string-based for backward compatibility.
#[derive(Debug, Clone, Default)]
pub struct MemoryGraph {
    interner: NodeInterner,
    int_edges: Vec<InternalEdge>,
    int_adjacency: IntAdjacency,
    last_memory_id: Option<String>,
    last_memory_by_scope: HashMap<String, String>,
    co_occurrence: BTreeMap<String, usize>,
    vocab: ConceptVocabulary,
    suppressed_concepts: BTreeSet<String>,
}

/// Magic prefix identifying a binary graph snapshot. Legacy snapshots are
/// JSON (they start with `{`), so the magic doubles as the format
/// discriminator on the load path.
const SNAPSHOT_MAGIC: [u8; 4] = *b"TMGR";
/// Current binary snapshot format version. Bump when the payload layout
/// changes; unknown versions are rejected on load.
const SNAPSHOT_VERSION: u32 = 1;

/// Compact binary snapshot of the full graph state (bincode payload behind a
/// magic+version header). Serialized directly from the interned integer
/// structures — no string-keyed `LegacyMemoryGraph` clone. Adjacency is not
/// persisted; it is rebuilt from the edge list on load, matching the JSON path.
#[derive(Debug, Serialize, Deserialize)]
struct BinaryGraphSnapshot {
    /// Live interner entries `(key, node)` in u32-id order. Tombstones left by
    /// `remove_memory` are dropped, so on load each node's dense id is its
    /// index here (the same compaction the JSON round-trip performs).
    nodes: Vec<(String, InternalNode)>,
    /// Edges remapped to the dense node ids in `nodes`, sorted with the
    /// `compact()` comparator so snapshot bytes are deterministic.
    edges: Vec<InternalEdge>,
    last_memory_id: Option<String>,
    co_occurrence: BTreeMap<String, usize>,
    /// Vocabulary alias pairs sorted by alias (`HashMap` iteration order would
    /// make snapshot bytes nondeterministic).
    vocab_aliases: Vec<(String, String)>,
    suppressed_concepts: BTreeSet<String>,
}

/// Legacy serialization format for backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LegacyMemoryGraph {
    nodes: BTreeMap<String, Node>,
    edges: Vec<Edge>,
    adjacency: BTreeMap<String, Vec<usize>>,
    last_memory_id: Option<String>,
    #[serde(default)]
    co_occurrence: BTreeMap<String, usize>,
    #[serde(default)]
    vocab: ConceptVocabulary,
    #[serde(default)]
    suppressed_concepts: BTreeSet<String>,
}

impl From<LegacyMemoryGraph> for MemoryGraph {
    fn from(legacy: LegacyMemoryGraph) -> Self {
        let mut graph = MemoryGraph {
            interner: NodeInterner::default(),
            int_edges: Vec::new(),
            int_adjacency: IntAdjacency::default(),
            last_memory_id: legacy.last_memory_id,
            last_memory_by_scope: HashMap::new(),
            co_occurrence: legacy.co_occurrence,
            vocab: legacy.vocab,
            suppressed_concepts: legacy.suppressed_concepts,
        };

        // Intern all nodes first
        for (key, node) in &legacy.nodes {
            let kind = match &node.id {
                NodeId::Memory(_) => NodeKind::Memory,
                NodeId::Concept(_) => NodeKind::Concept,
            };
            let external_id = match &node.id {
                NodeId::Memory(s) => s.clone(),
                NodeId::Concept(s) => s.clone(),
            };
            graph.intern_node(
                key.clone(),
                kind,
                external_id,
                node.text.clone(),
                node.base_importance_factor,
            );
        }

        // Convert edges to integer format
        for edge in &legacy.edges {
            let src_key = edge.source.as_str();
            let tgt_key = edge.target.as_str();
            if let (Some(src_id), Some(tgt_id)) =
                (graph.get_node_id(&src_key), graph.get_node_id(&tgt_key))
            {
                graph.int_edges.push(InternalEdge {
                    source: src_id,
                    target: tgt_id,
                    kind: edge.kind,
                    weight: edge.weight,
                    last_reinforced_at: edge.last_reinforced_at,
                });
            }
        }
        graph.int_adjacency.rebuild(&graph.int_edges);
        graph
    }
}

impl From<MemoryGraph> for LegacyMemoryGraph {
    fn from(graph: MemoryGraph) -> Self {
        let mut nodes = BTreeMap::new();
        for (_id, key, node) in graph.interner.iter() {
            let node_id = match node.kind {
                NodeKind::Memory => NodeId::memory(&node.external_id),
                NodeKind::Concept => NodeId::concept(&node.external_id),
            };
            nodes.insert(
                key.clone(),
                Node {
                    id: node_id,
                    text: node.text.clone(),
                    base_importance_factor: node.base_importance_factor,
                },
            );
        }

        let edges = graph.edges();
        let mut adjacency: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (idx, edge) in graph.int_edges.iter().enumerate() {
            let source_key = graph.interner.id_to_key[edge.source as usize].clone();
            adjacency.entry(source_key).or_default().push(idx);
        }

        LegacyMemoryGraph {
            nodes,
            edges,
            adjacency,
            last_memory_id: graph.last_memory_id,
            co_occurrence: graph.co_occurrence,
            vocab: graph.vocab,
            suppressed_concepts: graph.suppressed_concepts,
        }
    }
}

impl Serialize for MemoryGraph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let legacy: LegacyMemoryGraph = self.clone().into();
        legacy.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MemoryGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let legacy = LegacyMemoryGraph::deserialize(deserializer)?;
        Ok(legacy.into())
    }
}

impl MemoryGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Intern a node and return its u32 ID. If already present, returns existing.
    fn intern_node(
        &mut self,
        key: String,
        kind: NodeKind,
        external_id: String,
        text: String,
        base_importance_factor: f32,
    ) -> u32 {
        if let Some(&id) = self.interner.key_to_id.get(&key) {
            return id;
        }
        let node = InternalNode {
            kind,
            external_id,
            text,
            base_importance_factor,
        };
        self.interner.intern(key, node)
    }

    pub(crate) fn get_node_id(&self, key: &str) -> Option<u32> {
        self.interner.resolve(key)
    }

    fn add_int_edge(&mut self, source: u32, target: u32, kind: EdgeKind, weight: f32) {
        let idx = self.int_edges.len();
        self.int_edges.push(InternalEdge {
            source,
            target,
            kind,
            weight,
            last_reinforced_at: 0,
        });
        self.int_adjacency.add(source, idx);
    }

    pub fn add_memory(&mut self, id: &str, text: &str, concepts: &[String]) {
        self.add_memory_scoped(id, text, concepts, 1.0, None);
    }

    pub fn add_memory_with_importance(
        &mut self,
        id: &str,
        text: &str,
        concepts: &[String],
        importance: f32,
    ) {
        self.add_memory_scoped(id, text, concepts, importance, None);
    }

    pub fn add_memory_scoped(
        &mut self,
        id: &str,
        text: &str,
        concepts: &[String],
        importance: f32,
        scope: Option<&str>,
    ) {
        let mem_key = NodeId::memory(id).as_str();
        let imp = importance_factor(importance);
        let mem_id = self.intern_node(
            mem_key.clone(),
            NodeKind::Memory,
            id.to_string(),
            text.to_string(),
            imp,
        );

        // Normalize concepts to lowercase, canonicalize, dedup.
        let mut normalized: Vec<String> = Vec::with_capacity(concepts.len());
        let mut seen: HashSet<String> = HashSet::new();
        for concept in concepts {
            let norm = self.vocab.resolve(concept);
            if !norm.is_empty() && seen.insert(norm.clone()) {
                normalized.push(norm);
            }
        }

        // Concept association edges (bidirectional), weighted by importance.
        for concept in &normalized {
            let concept_key = NodeId::concept(concept).as_str();
            let concept_id = self.intern_node(
                concept_key.clone(),
                NodeKind::Concept,
                concept.clone(),
                concept.clone(),
                0.0,
            );
            self.add_int_edge(mem_id, concept_id, EdgeKind::Association, imp);
            self.add_int_edge(concept_id, mem_id, EdgeKind::Association, imp);
        }

        // Accumulate concept co-occurrence for abstraction building.
        if normalized.len() > 1 {
            let mut sorted: Vec<&str> = normalized.iter().map(|s| s.as_str()).collect();
            sorted.sort();
            for i in 0..sorted.len() {
                for j in (i + 1)..sorted.len() {
                    let key = format!("concept:{}{}concept:{}", sorted[i], '\0', sorted[j]);
                    *self.co_occurrence.entry(key).or_insert(0) += 1;
                }
            }
        }

        // Temporal chaining between consecutive memories in the same scope.
        let scope_key = scope.unwrap_or("").to_string();
        let prev_opt = if let Some(prev) = self.last_memory_by_scope.remove(&scope_key) {
            Some(prev)
        } else if scope.is_none() {
            self.last_memory_id.take()
        } else {
            None
        };

        if let Some(prev) = prev_opt {
            let prev_key = NodeId::memory(&prev).as_str();
            if let Some(prev_id) = self.get_node_id(&prev_key) {
                // Forward temporal chaining: prev -> mem
                self.add_int_edge(prev_id, mem_id, EdgeKind::Temporal, 0.5 * imp);
            }
        }
        self.last_memory_by_scope.insert(scope_key, id.to_string());
        if scope.is_none() {
            self.last_memory_id = Some(id.to_string());
        }
    }

    /// Add concept Association edges to an existing memory — and nothing else.
    ///
    /// Belief revision uses this to merge an older memory's concepts into a
    /// newer one *after both were inserted*. Calling `add_memory_scoped` for
    /// that re-adds duplicate edges for already-linked concepts, double-counts
    /// co-occurrence, and re-points the temporal chain head at an old memory.
    /// Existing edges keep their weights (`ensure_int_edge` merges by max).
    /// Returns false if the memory node does not exist.
    pub fn add_concepts_to_memory(
        &mut self,
        id: &str,
        concepts: &[String],
        importance: f32,
    ) -> bool {
        let mem_key = NodeId::memory(id).as_str();
        let Some(mem_id) = self.get_node_id(&mem_key) else {
            return false;
        };
        let imp = importance_factor(importance);
        let mut seen: HashSet<String> = HashSet::new();
        for concept in concepts {
            let norm = self.vocab.resolve(concept);
            if norm.is_empty() || !seen.insert(norm.clone()) {
                continue;
            }
            let concept_key = NodeId::concept(&norm).as_str();
            let concept_id = self.intern_node(
                concept_key,
                NodeKind::Concept,
                norm.clone(),
                norm.clone(),
                0.0,
            );
            self.ensure_int_edge(mem_id, concept_id, EdgeKind::Association, imp, 0);
            self.ensure_int_edge(concept_id, mem_id, EdgeKind::Association, imp, 0);
        }
        true
    }

    pub fn remove_memory(&mut self, id: &str) {
        let mem_key = NodeId::memory(id).as_str();
        let Some(mem_id) = self.get_node_id(&mem_key) else {
            return;
        };
        // Remove the node from interner (mark as removed by clearing key map)
        self.interner.key_to_id.remove(&mem_key);
        // Remove edges connected to this node
        self.int_edges
            .retain(|e| e.source != mem_id && e.target != mem_id);
        self.int_adjacency.rebuild(&self.int_edges);
    }

    pub fn node_count(&self) -> usize {
        self.interner.iter().count()
    }

    pub fn edge_count(&self) -> usize {
        self.int_edges.len()
    }

    pub fn memory_count(&self) -> usize {
        self.interner
            .iter()
            .filter(|(_, _, n)| matches!(n.kind, NodeKind::Memory))
            .count()
    }

    pub fn concept_degree(&self, concept: &str) -> usize {
        let key = NodeId::concept(concept).as_str();
        let Some(id) = self.get_node_id(&key) else {
            return 0;
        };
        self.int_adjacency
            .get(id)
            .map(|idxs| {
                idxs.iter()
                    .filter(|&&i| self.int_edges[i].kind == EdgeKind::Association)
                    .count()
            })
            .unwrap_or(0)
    }

    /// Public API: returns nodes as BTreeMap for backward compatibility.
    pub fn nodes(&self) -> BTreeMap<String, Node> {
        let mut map = BTreeMap::new();
        for (_id, key, node) in self.interner.iter() {
            let node_id = match node.kind {
                NodeKind::Memory => NodeId::memory(&node.external_id),
                NodeKind::Concept => NodeId::concept(&node.external_id),
            };
            map.insert(
                key.clone(),
                Node {
                    id: node_id,
                    text: node.text.clone(),
                    base_importance_factor: node.base_importance_factor,
                },
            );
        }
        map
    }

    /// Public API: returns edges with NodeId for backward compatibility.
    pub fn edges(&self) -> Vec<Edge> {
        self.int_edges
            .iter()
            .map(|e| {
                let source_node = self.interner.get(e.source).expect("valid source");
                let target_node = self.interner.get(e.target).expect("valid target");
                Edge {
                    source: match source_node.kind {
                        NodeKind::Memory => NodeId::memory(&source_node.external_id),
                        NodeKind::Concept => NodeId::concept(&source_node.external_id),
                    },
                    target: match target_node.kind {
                        NodeKind::Memory => NodeId::memory(&target_node.external_id),
                        NodeKind::Concept => NodeId::concept(&target_node.external_id),
                    },
                    kind: e.kind,
                    weight: e.weight,
                    last_reinforced_at: e.last_reinforced_at,
                }
            })
            .collect()
    }

    /// Public API: returns neighbor edges (allocates Vec for compat).
    pub fn neighbors(&self, node_key: &str) -> Vec<Edge> {
        let Some(id) = self.get_node_id(node_key) else {
            return Vec::new();
        };
        self.int_adjacency
            .get(id)
            .map(|idxs| {
                idxs.iter()
                    .map(|&i| {
                        let e = &self.int_edges[i];
                        let source_node = self.interner.get(e.source).expect("valid source");
                        let target_node = self.interner.get(e.target).expect("valid target");
                        Edge {
                            source: match source_node.kind {
                                NodeKind::Memory => NodeId::memory(&source_node.external_id),
                                NodeKind::Concept => NodeId::concept(&source_node.external_id),
                            },
                            target: match target_node.kind {
                                NodeKind::Memory => NodeId::memory(&target_node.external_id),
                                NodeKind::Concept => NodeId::concept(&target_node.external_id),
                            },
                            kind: e.kind,
                            weight: e.weight,
                            last_reinforced_at: e.last_reinforced_at,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Public API: iterate memory nodes as (key, Node) for backward compat.
    pub fn iter_memory_nodes(&self) -> impl Iterator<Item = (String, Node)> + '_ {
        self.interner.memory_nodes().map(|(_, key, node)| {
            let node_id = match node.kind {
                NodeKind::Memory => NodeId::memory(&node.external_id),
                NodeKind::Concept => NodeId::concept(&node.external_id),
            };
            (
                key.clone(),
                Node {
                    id: node_id,
                    text: node.text.clone(),
                    base_importance_factor: node.base_importance_factor,
                },
            )
        })
    }

    pub fn reinforce(&mut self, id: &str, now: u64) {
        let mem_key = NodeId::memory(id).as_str();
        let Some(mem_id) = self.get_node_id(&mem_key) else {
            return;
        };
        // 0 is the "never reinforced" sentinel; a real t=0 timestamp would
        // leave edges permanently decay-exempt and forever eligible for the
        // first-touch boost.
        let now = now.max(1);

        // Strengthen outgoing edges where memory is source.
        if let Some(out_idxs) = self.int_adjacency.get(mem_id) {
            let out_idxs: Vec<usize> = out_idxs.to_vec();
            for i in out_idxs {
                let edge = &mut self.int_edges[i];
                let boost = if edge.last_reinforced_at == 0 {
                    1.5
                } else {
                    1.0 + 0.1 / (1.0 + edge.weight)
                };
                edge.weight = (edge.weight * boost).min(8.0);
                edge.last_reinforced_at = now;
            }
        }

        // Strengthen incoming Association edges where memory is target.
        for edge in &mut self.int_edges {
            if edge.kind == EdgeKind::Association && edge.target == mem_id {
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

    pub fn decay_edges(&mut self, now: u64, half_life: u64) {
        if half_life == 0 {
            return;
        }
        let hl = half_life as f64;
        for edge in &mut self.int_edges {
            if edge.last_reinforced_at == 0 {
                continue;
            }
            let age = now.saturating_sub(edge.last_reinforced_at) as f64;
            let factor = 0.5f64.powf(age / hl) as f32;
            let decayed = edge.weight * factor;
            // Floor at 1.0, but never *raise* a weight: an edge created below
            // the floor (low-importance baseline) must not decay upward.
            edge.weight = decayed.max(edge.weight.min(1.0));
        }
    }

    pub fn reweight_memory(&mut self, id: &str, new_importance: f32) -> bool {
        let mem_key = NodeId::memory(id).as_str();
        let Some(mem_id) = self.get_node_id(&mem_key) else {
            return false;
        };
        let new_factor = importance_factor(new_importance);

        let mut own_idxs: Vec<usize> = self
            .int_adjacency
            .get(mem_id)
            .map(|idxs| {
                idxs.iter()
                    .copied()
                    .filter(|&i| {
                        let k = self.int_edges[i].kind;
                        k == EdgeKind::Association || k == EdgeKind::Temporal
                    })
                    .collect()
            })
            .unwrap_or_default();
        for (i, edge) in self.int_edges.iter().enumerate() {
            if edge.target == mem_id
                && (edge.kind == EdgeKind::Association || edge.kind == EdgeKind::Temporal)
                && !own_idxs.contains(&i)
            {
                own_idxs.push(i);
            }
        }
        if own_idxs.is_empty() {
            return false;
        }

        let stored = self
            .interner
            .get(mem_id)
            .map(|n| n.base_importance_factor)
            .unwrap_or(0.0);
        let old_factor = if stored > 0.0 {
            stored
        } else {
            let unreinforced_min = own_idxs
                .iter()
                .filter(|&&i| self.int_edges[i].last_reinforced_at == 0)
                .map(|&i| self.int_edges[i].weight)
                .fold(f32::INFINITY, f32::min);
            if unreinforced_min.is_finite() {
                unreinforced_min
            } else {
                own_idxs
                    .iter()
                    .map(|&i| self.int_edges[i].weight)
                    .fold(f32::INFINITY, f32::min)
            }
        }
        .max(1e-6);
        let ratio = new_factor / old_factor;

        for &i in &own_idxs {
            self.int_edges[i].weight = (self.int_edges[i].weight * ratio).clamp(0.01, 8.0);
        }
        if let Some(node) = self.interner.get_mut(mem_id) {
            node.base_importance_factor = new_factor;
        }
        true
    }

    pub fn build_abstractions(&mut self, threshold: usize) -> usize {
        if threshold == 0 {
            return 0;
        }
        let mut added = 0usize;
        let pairs: Vec<(String, String, String)> = self
            .co_occurrence
            .iter()
            .filter(|(_, &count)| count >= threshold)
            .map(|(key, _)| {
                let mut parts = key.splitn(2, '\0');
                let a = parts.next().unwrap_or_default().to_string();
                let b = parts.next().unwrap_or_default().to_string();
                (a, b, key.clone())
            })
            .collect();

        for (a_key, b_key, co_key) in pairs {
            let a_name = a_key.strip_prefix("concept:").unwrap_or(&a_key);
            let b_name = b_key.strip_prefix("concept:").unwrap_or(&b_key);
            let parent_name = if a_name <= b_name {
                format!("{a_name}+{b_name}")
            } else {
                format!("{b_name}+{a_name}")
            };
            let parent_key = NodeId::concept(&parent_name).as_str();

            let parent_id = if let Some(&id) = self.interner.key_to_id.get(&parent_key) {
                id
            } else {
                self.intern_node(
                    parent_key.clone(),
                    NodeKind::Concept,
                    parent_name.clone(),
                    parent_name.clone(),
                    0.0,
                )
            };

            for child_key in [&a_key, &b_key] {
                let child_id = self.get_node_id(child_key);
                let exists = child_id
                    .and_then(|cid| self.int_adjacency.get(cid))
                    .map(|idxs| {
                        idxs.iter().any(|&i| {
                            self.int_edges[i].kind == EdgeKind::Abstraction
                                && self.int_edges[i].target == parent_id
                        })
                    })
                    .unwrap_or(false);
                if !exists {
                    let child_id = child_id.unwrap_or_else(|| {
                        let name = child_key.strip_prefix("concept:").unwrap_or(child_key);
                        self.intern_node(
                            child_key.to_string(),
                            NodeKind::Concept,
                            name.to_string(),
                            name.to_string(),
                            0.0,
                        )
                    });
                    self.add_int_edge(child_id, parent_id, EdgeKind::Abstraction, 1.0);
                    self.add_int_edge(parent_id, child_id, EdgeKind::Abstraction, 1.0);
                    added += 2;
                }
            }
            self.co_occurrence.insert(co_key, 0);
        }
        added
    }

    pub fn abstraction_count(&self) -> usize {
        self.interner
            .iter()
            .filter(|(_, _, n)| matches!(n.kind, NodeKind::Concept) && n.external_id.contains('+'))
            .count()
    }

    pub fn concept_count(&self) -> usize {
        self.interner
            .iter()
            .filter(|(_, _, n)| matches!(n.kind, NodeKind::Concept) && !n.external_id.contains('+'))
            .count()
    }

    pub fn memory_concepts(&self, id: &str) -> Vec<String> {
        let key = NodeId::memory(id).as_str();
        let Some(mem_id) = self.get_node_id(&key) else {
            return Vec::new();
        };
        self.int_adjacency
            .get(mem_id)
            .map(|idxs| {
                idxs.iter()
                    .filter(|&&i| self.int_edges[i].kind == EdgeKind::Association)
                    .filter_map(|&i| {
                        let target = self.interner.get(self.int_edges[i].target)?;
                        if matches!(target.kind, NodeKind::Concept) {
                            Some(target.external_id.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            memory_count: self.memory_count(),
            concept_count: self.concept_count(),
            refinement_count: self.refinement_count(),
            contradiction_count: self.contradiction_count(),
            abstraction_count: self.abstraction_count(),
        }
    }

    pub fn vocab(&self) -> &ConceptVocabulary {
        &self.vocab
    }

    pub fn vocab_mut(&mut self) -> &mut ConceptVocabulary {
        &mut self.vocab
    }

    pub fn is_concept_suppressed(&self, concept: &str) -> bool {
        self.suppressed_concepts.contains(&concept.to_lowercase())
    }

    pub fn suppressed_concept_count(&self) -> usize {
        self.suppressed_concepts.len()
    }

    pub fn evolve_vocabulary(
        &mut self,
        overlap_threshold: f32,
        hub_fraction: f32,
        max_pairs: usize,
    ) -> VocabularyEvolutionStats {
        if max_pairs == 0 {
            return VocabularyEvolutionStats::default();
        }
        let threshold = overlap_threshold.clamp(0.0, 1.0);

        let mut concept_memories: BTreeMap<String, HashSet<String>> = BTreeMap::new();
        let mut co_occurrence: BTreeMap<String, usize> = BTreeMap::new();

        let memory_ids: Vec<String> = self
            .iter_memory_nodes()
            .map(|(k, _)| k.strip_prefix("mem:").unwrap_or(&k).to_string())
            .collect();
        for id in &memory_ids {
            let key = NodeId::memory(id).as_str();
            let Some(mem_id) = self.get_node_id(&key) else {
                continue;
            };
            let mut concepts: Vec<String> = self
                .int_adjacency
                .get(mem_id)
                .map(|idxs| {
                    idxs.iter()
                        .filter(|&&i| self.int_edges[i].kind == EdgeKind::Association)
                        .filter_map(|&i| {
                            let target = self.interner.get(self.int_edges[i].target)?;
                            if matches!(target.kind, NodeKind::Concept)
                                && !target.external_id.contains('+')
                            {
                                Some(target.external_id.clone())
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            concepts.sort();
            for c in &concepts {
                concept_memories
                    .entry(c.clone())
                    .or_default()
                    .insert(id.clone());
            }
            for i in 0..concepts.len() {
                for j in (i + 1)..concepts.len() {
                    let a = &concepts[i];
                    let b = &concepts[j];
                    let pair_key = if a <= b {
                        format!("concept:{a}\0concept:{b}")
                    } else {
                        format!("concept:{b}\0concept:{a}")
                    };
                    *co_occurrence.entry(pair_key).or_insert(0) += 1;
                }
            }
        }

        let mut candidates: Vec<(String, String, f32)> = Vec::new();
        for (pair_key, &count) in &co_occurrence {
            let mut parts = pair_key.splitn(2, '\0');
            let a_key = parts.next().unwrap_or_default().to_string();
            let b_key = parts.next().unwrap_or_default().to_string();
            let a = a_key.strip_prefix("concept:").unwrap_or(&a_key).to_string();
            let b = b_key.strip_prefix("concept:").unwrap_or(&b_key).to_string();
            let a_set = concept_memories.get(&a).map(|s| s.len()).unwrap_or(0);
            let b_set = concept_memories.get(&b).map(|s| s.len()).unwrap_or(0);
            let union = a_set + b_set - count;
            if union == 0 {
                continue;
            }
            let overlap = count as f32 / union as f32;
            if overlap >= threshold {
                candidates.push((a, b, overlap));
            }
        }
        candidates.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
                .then(a.1.cmp(&b.1))
        });

        let mut merged = 0usize;
        for (a, b, _) in &candidates {
            if merged >= max_pairs {
                break;
            }
            let a_canon = self.vocab.resolve(a);
            let b_canon = self.vocab.resolve(b);
            if a_canon == b_canon {
                continue;
            }
            let a_deg = self.concept_degree(&a_canon);
            let b_deg = self.concept_degree(&b_canon);
            let (canonical, alias) = if a_deg > b_deg || (a_deg == b_deg && a_canon < b_canon) {
                (a_canon, b_canon)
            } else {
                (b_canon, a_canon)
            };
            if self.merge_concept_node(&alias, &canonical) {
                merged += 1;
            }
        }

        let suppressed_before = self.suppressed_concepts.len();
        if hub_fraction > 0.0 {
            let memory_count = self.memory_count().max(1);
            let hub_threshold = (memory_count as f32 * hub_fraction).max(2.0) as usize;
            let base_concepts: Vec<String> = self
                .interner
                .iter()
                .filter_map(|(_, _, n)| {
                    if matches!(n.kind, NodeKind::Concept) && !n.external_id.contains('+') {
                        Some(n.external_id.clone())
                    } else {
                        None
                    }
                })
                .collect();
            for c in base_concepts {
                if self.concept_degree(&c) > hub_threshold {
                    self.suppressed_concepts.insert(c);
                }
            }
        }
        let suppressed = self.suppressed_concepts.len() - suppressed_before;

        self.rebuild_co_occurrence();

        VocabularyEvolutionStats {
            merged,
            suppressed,
            examined_pairs: candidates.len(),
        }
    }

    fn merge_concept_node(&mut self, alias: &str, canonical: &str) -> bool {
        let alias_key = NodeId::concept(alias).as_str();
        let canon_key = NodeId::concept(canonical).as_str();
        let Some(alias_id) = self.get_node_id(&alias_key) else {
            return false;
        };
        let canon_id = if let Some(&id) = self.interner.key_to_id.get(&canon_key) {
            id
        } else {
            // Canonical does not exist; rename alias in place. The interner's
            // reverse map must move with the key, or the node fails iter()'s
            // key_to_id filter and is silently dropped by snapshots.
            if let Some(node) = self.interner.get_mut(alias_id) {
                node.external_id = canonical.to_string();
                node.text = canonical.to_string();
            }
            self.interner.key_to_id.remove(&alias_key);
            self.interner.key_to_id.insert(canon_key.clone(), alias_id);
            self.interner.id_to_key[alias_id as usize] = canon_key.clone();
            self.vocab.add_alias(alias, canonical);
            self.suppressed_concepts.remove(alias);
            return true;
        };

        let outgoing: Vec<InternalEdge> = self
            .int_adjacency
            .get(alias_id)
            .map(|idxs| idxs.to_vec())
            .unwrap_or_default()
            .iter()
            .map(|&i| self.int_edges[i])
            .collect();
        let incoming: Vec<InternalEdge> = self
            .int_edges
            .iter()
            .filter(|e| e.target == alias_id)
            .copied()
            .collect();

        for edge in outgoing {
            match (
                self.interner.get(edge.source),
                self.interner.get(edge.target),
                edge.kind,
            ) {
                (Some(src), Some(_), EdgeKind::Association)
                    if matches!(src.kind, NodeKind::Concept) =>
                {
                    let target_id = if self.interner.get(edge.target).map(|n| n.kind)
                        == Some(NodeKind::Memory)
                    {
                        edge.target
                    } else {
                        continue;
                    };
                    // Migrate alias -> mem to canon -> mem. Sourcing the new
                    // edge at the alias was dead work: the retain below deletes
                    // every edge that still touches the alias, which used to
                    // take the concept -> memory direction with it.
                    self.ensure_int_edge(
                        canon_id,
                        target_id,
                        EdgeKind::Association,
                        edge.weight,
                        edge.last_reinforced_at,
                    );
                }
                (Some(src), Some(_), EdgeKind::Abstraction)
                    if matches!(src.kind, NodeKind::Concept) =>
                {
                    let target_id = if self
                        .interner
                        .get(edge.target)
                        .map(|n| matches!(n.kind, NodeKind::Concept))
                        .unwrap_or(false)
                    {
                        edge.target
                    } else {
                        continue;
                    };
                    self.ensure_int_edge(
                        canon_id,
                        target_id,
                        EdgeKind::Abstraction,
                        edge.weight,
                        edge.last_reinforced_at,
                    );
                }
                _ => {}
            }
        }
        for edge in incoming {
            match (
                self.interner.get(edge.source),
                self.interner.get(edge.target),
                edge.kind,
            ) {
                (Some(src), Some(_), EdgeKind::Association)
                    if matches!(src.kind, NodeKind::Memory) =>
                {
                    self.ensure_int_edge(
                        edge.source,
                        canon_id,
                        EdgeKind::Association,
                        edge.weight,
                        edge.last_reinforced_at,
                    );
                }
                (Some(src), Some(_), EdgeKind::Abstraction)
                    if matches!(src.kind, NodeKind::Concept) =>
                {
                    self.ensure_int_edge(
                        edge.source,
                        canon_id,
                        EdgeKind::Abstraction,
                        edge.weight,
                        edge.last_reinforced_at,
                    );
                }
                _ => {}
            }
        }

        self.interner.key_to_id.remove(&alias_key);
        self.int_edges
            .retain(|e| e.source != alias_id && e.target != alias_id);
        self.int_adjacency.rebuild(&self.int_edges);
        self.vocab.add_alias(alias, canonical);
        self.suppressed_concepts.remove(alias);
        true
    }

    fn ensure_int_edge(
        &mut self,
        source: u32,
        target: u32,
        kind: EdgeKind,
        weight: f32,
        last_reinforced_at: u64,
    ) {
        let existing = self.int_adjacency.get(source).and_then(|idxs| {
            idxs.iter()
                .copied()
                .find(|&i| self.int_edges[i].kind == kind && self.int_edges[i].target == target)
        });
        if let Some(i) = existing {
            let e = &mut self.int_edges[i];
            e.weight = e.weight.max(weight);
            e.last_reinforced_at = e.last_reinforced_at.max(last_reinforced_at);
        } else {
            let idx = self.int_edges.len();
            self.int_edges.push(InternalEdge {
                source,
                target,
                kind,
                weight,
                last_reinforced_at,
            });
            self.int_adjacency.add(source, idx);
        }
    }

    fn rebuild_co_occurrence(&mut self) {
        self.co_occurrence.clear();
        let memory_ids: Vec<String> = self
            .iter_memory_nodes()
            .map(|(k, _)| k.strip_prefix("mem:").unwrap_or(&k).to_string())
            .collect();
        for id in &memory_ids {
            let key = NodeId::memory(id).as_str();
            let Some(mem_id) = self.get_node_id(&key) else {
                continue;
            };
            let mut concepts: Vec<String> = self
                .int_adjacency
                .get(mem_id)
                .map(|idxs| {
                    idxs.iter()
                        .filter(|&&i| self.int_edges[i].kind == EdgeKind::Association)
                        .filter_map(|&i| {
                            let target = self.interner.get(self.int_edges[i].target)?;
                            if matches!(target.kind, NodeKind::Concept)
                                && !target.external_id.contains('+')
                            {
                                Some(target.external_id.clone())
                            } else {
                                None
                            }
                        })
                        .collect()
                })
                .unwrap_or_default();
            concepts.sort();
            for i in 0..concepts.len() {
                for j in (i + 1)..concepts.len() {
                    let a = &concepts[i];
                    let b = &concepts[j];
                    let pair_key = if a <= b {
                        format!("concept:{a}\0concept:{b}")
                    } else {
                        format!("concept:{b}\0concept:{a}")
                    };
                    *self.co_occurrence.entry(pair_key).or_insert(0) += 1;
                }
            }
        }
    }

    pub fn add_refinement(&mut self, old_id: &str, new_id: &str, weight: f32) -> bool {
        let old_key = NodeId::memory(old_id).as_str();
        let new_key = NodeId::memory(new_id).as_str();
        let Some(old_nid) = self.get_node_id(&old_key) else {
            return false;
        };
        let Some(new_nid) = self.get_node_id(&new_key) else {
            return false;
        };

        let exists = self
            .int_adjacency
            .get(old_nid)
            .map(|idxs| {
                idxs.iter().any(|&i| {
                    self.int_edges[i].kind == EdgeKind::Refines
                        && self.int_edges[i].target == new_nid
                })
            })
            .unwrap_or(false);
        if exists {
            return false;
        }
        self.add_int_edge(old_nid, new_nid, EdgeKind::Refines, weight);
        true
    }

    pub fn refinement_count(&self) -> usize {
        self.int_edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Refines)
            .count()
    }

    pub fn refined_by(&self, id: &str) -> Vec<String> {
        let key = NodeId::memory(id).as_str();
        let Some(nid) = self.get_node_id(&key) else {
            return Vec::new();
        };
        self.int_adjacency
            .get(nid)
            .map(|idxs| {
                idxs.iter()
                    .filter(|&&i| self.int_edges[i].kind == EdgeKind::Refines)
                    .filter_map(|&i| {
                        let target = self.interner.get(self.int_edges[i].target)?;
                        if matches!(target.kind, NodeKind::Memory) {
                            Some(target.external_id.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn add_contradiction(
        &mut self,
        old_id: &str,
        new_id: &str,
        weight: f32,
        weaken_factor: f32,
    ) -> bool {
        let old_key = NodeId::memory(old_id).as_str();
        let new_key = NodeId::memory(new_id).as_str();
        let Some(old_nid) = self.get_node_id(&old_key) else {
            return false;
        };
        let Some(new_nid) = self.get_node_id(&new_key) else {
            return false;
        };

        let exists = self
            .int_adjacency
            .get(old_nid)
            .map(|idxs| {
                idxs.iter().any(|&i| {
                    self.int_edges[i].kind == EdgeKind::Contradicts
                        && self.int_edges[i].target == new_nid
                })
            })
            .unwrap_or(false);
        if exists {
            return false;
        }

        let old_idxs: Vec<usize> = self
            .int_adjacency
            .get(old_nid)
            .map(|v| v.to_vec())
            .unwrap_or_default();
        for i in old_idxs {
            let edge = &mut self.int_edges[i];
            if edge.kind == EdgeKind::Association || edge.kind == EdgeKind::Temporal {
                edge.weight *= weaken_factor;
            }
        }
        self.add_int_edge(old_nid, new_nid, EdgeKind::Contradicts, weight);
        true
    }

    pub fn contradiction_count(&self) -> usize {
        self.int_edges
            .iter()
            .filter(|e| e.kind == EdgeKind::Contradicts)
            .count()
    }

    /// Every memory id that is SUPERSEDED — i.e. the source (older) side of a
    /// `Refines` or `Contradicts` edge whose newer target memory is still live.
    /// This is the set to EXCLUDE from (or tag as outdated in) the answer
    /// context: unlike rank-demotion, removing a stale fact from the context is
    /// what actually stops it misleading the answer model ("ghost memory",
    /// A-TMA 2026). One graph pass; deduplicated.
    pub fn superseded_ids(&self) -> Vec<String> {
        use std::collections::HashSet;
        let mut set: HashSet<String> = HashSet::new();
        for e in &self.int_edges {
            if !matches!(e.kind, EdgeKind::Refines | EdgeKind::Contradicts) {
                continue;
            }
            let (Some(src), Some(tgt)) = (self.interner.get(e.source), self.interner.get(e.target))
            else {
                continue;
            };
            if matches!(src.kind, NodeKind::Memory) && matches!(tgt.kind, NodeKind::Memory) {
                set.insert(src.external_id.clone());
            }
        }
        set.into_iter().collect()
    }

    /// Outgoing supersession edges (`Refines`/`Contradicts`, memory -> memory)
    /// of node `nid` as (edge index, target node) pairs. The edge index is an
    /// insertion-order stamp: a higher index was inserted later.
    fn supersession_successors(&self, nid: u32) -> Vec<(usize, u32)> {
        self.int_adjacency
            .get(nid)
            .map(|idxs| {
                idxs.iter()
                    .filter(|&&i| {
                        matches!(
                            self.int_edges[i].kind,
                            EdgeKind::Refines | EdgeKind::Contradicts
                        )
                    })
                    .filter_map(|&i| {
                        let target = self.int_edges[i].target;
                        match self.interner.get(target) {
                            Some(n) if matches!(n.kind, NodeKind::Memory) => Some((i, target)),
                            _ => None,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The next hop in a supersession walk: the target of the LATEST-INSERTED
    /// outgoing `Refines`/`Contradicts` edge. This is the branching rule: when
    /// two newer memories both supersede the same older one, the walk follows
    /// the edge committed most recently (deterministic, no string compares).
    fn supersession_next(&self, nid: u32) -> Option<u32> {
        self.supersession_successors(nid)
            .into_iter()
            .max_by_key(|(edge_idx, _)| *edge_idx)
            .map(|(_, target)| target)
    }

    /// The NEWEST memory in `id`'s supersession chain — the head of the chain,
    /// the memory nothing supersedes. Follows `Refines`/`Contradicts` edges
    /// (old -> new) from `id`. Returns `id` unchanged when the id is unknown
    /// or has no supersession edges. Cycle-safe: a visited-set stops the walk
    /// and returns the node where the cycle is detected, so corrupted data
    /// cannot hang a recall.
    pub fn belief_head(&self, id: &str) -> String {
        let key = NodeId::memory(id).as_str();
        let Some(mut cur) = self.get_node_id(&key) else {
            return id.to_string();
        };
        let mut visited: HashSet<u32> = HashSet::with_capacity(8);
        visited.insert(cur);
        while let Some(next) = self.supersession_next(cur) {
            if !visited.insert(next) {
                break; // cycle: stop at the node where it is detected
            }
            cur = next;
        }
        self.interner
            .get(cur)
            .map(|n| n.external_id.clone())
            .unwrap_or_else(|| id.to_string())
    }

    /// The full supersession chain containing `id`, oldest first, head last.
    /// Walks backward to the oldest ancestor (incoming `Refines`/`Contradicts`
    /// edges, same latest-inserted-edge branching rule) then forward to the
    /// head. Returns `[id]` when the id is unknown or has no supersession
    /// edges. Cycle-safe via a visited-set shared by both walks.
    pub fn belief_lineage(&self, id: &str) -> Vec<String> {
        let key = NodeId::memory(id).as_str();
        let Some(start) = self.get_node_id(&key) else {
            return vec![id.to_string()];
        };
        let mut visited: HashSet<u32> = HashSet::with_capacity(8);
        visited.insert(start);

        // Forward: from `id` toward the chain head.
        let mut chain: Vec<u32> = vec![start];
        while let Some(next) = self.supersession_next(*chain.last().expect("non-empty")) {
            if !visited.insert(next) {
                break; // cycle
            }
            chain.push(next);
        }

        // Backward: incoming supersession edges, toward the oldest ancestor.
        // The adjacency index is outgoing-only, so build the reverse view in
        // one pass (memory -> memory edges only).
        let mut predecessors: HashMap<u32, Vec<(usize, u32)>> = HashMap::new();
        for (i, e) in self.int_edges.iter().enumerate() {
            if !matches!(e.kind, EdgeKind::Refines | EdgeKind::Contradicts) {
                continue;
            }
            let (Some(src), Some(tgt)) = (self.interner.get(e.source), self.interner.get(e.target))
            else {
                continue;
            };
            if matches!(src.kind, NodeKind::Memory) && matches!(tgt.kind, NodeKind::Memory) {
                predecessors
                    .entry(e.target)
                    .or_default()
                    .push((i, e.source));
            }
        }
        let mut cur = start;
        while let Some(prev) = predecessors
            .get(&cur)
            .and_then(|cands| cands.iter().max_by_key(|(edge_idx, _)| *edge_idx))
            .map(|(_, source)| *source)
        {
            if !visited.insert(prev) {
                break; // cycle
            }
            chain.insert(0, prev);
            cur = prev;
        }

        chain
            .iter()
            .filter_map(|&nid| self.interner.get(nid).map(|n| n.external_id.clone()))
            .collect()
    }

    pub fn contradicted_by(&self, id: &str) -> Vec<String> {
        let key = NodeId::memory(id).as_str();
        let Some(nid) = self.get_node_id(&key) else {
            return Vec::new();
        };
        self.int_adjacency
            .get(nid)
            .map(|idxs| {
                idxs.iter()
                    .filter(|&&i| self.int_edges[i].kind == EdgeKind::Contradicts)
                    .filter_map(|&i| {
                        let target = self.interner.get(self.int_edges[i].target)?;
                        if matches!(target.kind, NodeKind::Memory) {
                            Some(target.external_id.clone())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    pub fn to_json(&self) -> String {
        let mut sorted = self.clone();
        sorted.compact();
        serde_json::to_string_pretty(&sorted).unwrap_or_default()
    }

    pub fn from_json(s: &str) -> Result<Self, serde_json::Error> {
        let graph: Self = serde_json::from_str(s)?;
        Ok(graph)
    }

    /// Serialize the graph to a compact, versioned binary snapshot.
    ///
    /// Unlike [`to_json`](Self::to_json) this writes the interned integer
    /// structures directly (no string-keyed legacy clone), so snapshots are
    /// far smaller. Bytes are deterministic for an unchanged graph: nodes are
    /// written in u32-id order, edges are sorted with the
    /// [`compact`](Self::compact) comparator, and vocabulary aliases are
    /// sorted (`HashMap` iteration order is not stable).
    ///
    /// Returns the serialization error rather than a header-only byte blob:
    /// silently saving an empty payload would destroy the graph on next load.
    pub fn to_snapshot_bytes(&self) -> Result<Vec<u8>, bincode::Error> {
        // Live nodes in id order; ids are reassigned densely on load.
        let mut remap: HashMap<u32, u32> = HashMap::with_capacity(self.interner.id_to_key.len());
        let mut nodes: Vec<(String, InternalNode)> = Vec::new();
        for (old_id, key, node) in self.interner.iter() {
            remap.insert(old_id, nodes.len() as u32);
            nodes.push((key.clone(), node.clone()));
        }
        // Remap edges to the dense ids, skipping any that reference removed
        // nodes (mirrors the legacy path, which drops edges with missing
        // endpoints).
        let mut edges: Vec<InternalEdge> = self
            .int_edges
            .iter()
            .filter_map(|e| {
                let source = *remap.get(&e.source)?;
                let target = *remap.get(&e.target)?;
                Some(InternalEdge {
                    source,
                    target,
                    ..*e
                })
            })
            .collect();
        // Sort like compact() so bytes are stable regardless of insertion
        // history, and a binary-restored graph iterates neighbors in the same
        // order as a JSON-restored one.
        edges.sort_by(|a, b| {
            let a_src = &nodes[a.source as usize].1.external_id;
            let b_src = &nodes[b.source as usize].1.external_id;
            let a_tgt = &nodes[a.target as usize].1.external_id;
            let b_tgt = &nodes[b.target as usize].1.external_id;
            a_src
                .cmp(b_src)
                .then(a_tgt.cmp(b_tgt))
                .then(a.kind.cmp(&b.kind))
                .then(
                    a.weight
                        .partial_cmp(&b.weight)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        let mut vocab_aliases: Vec<(String, String)> = self
            .vocab
            .aliases()
            .iter()
            .map(|(alias, canonical)| (alias.clone(), canonical.clone()))
            .collect();
        vocab_aliases.sort();
        let snapshot = BinaryGraphSnapshot {
            nodes,
            edges,
            last_memory_id: self.last_memory_id.clone(),
            co_occurrence: self.co_occurrence.clone(),
            vocab_aliases,
            suppressed_concepts: self.suppressed_concepts.clone(),
        };
        let payload = bincode::serialize(&snapshot)?;
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&SNAPSHOT_MAGIC);
        out.extend_from_slice(&SNAPSHOT_VERSION.to_le_bytes());
        out.extend_from_slice(&payload);
        Ok(out)
    }

    /// Returns true when `bytes` carries the binary snapshot magic. Anything
    /// else (notably a legacy JSON snapshot) returns false so callers can fall
    /// back to [`from_json`](Self::from_json).
    pub fn is_snapshot_bytes(bytes: &[u8]) -> bool {
        bytes.len() >= SNAPSHOT_MAGIC.len() && bytes[..SNAPSHOT_MAGIC.len()] == SNAPSHOT_MAGIC
    }

    /// Load a graph from a binary snapshot written by
    /// [`to_snapshot_bytes`](Self::to_snapshot_bytes).
    pub fn from_snapshot_bytes(bytes: &[u8]) -> Result<Self, bincode::Error> {
        let invalid =
            |msg: &str| -> bincode::Error { Box::new(bincode::ErrorKind::Custom(msg.to_string())) };
        if !Self::is_snapshot_bytes(bytes) {
            return Err(invalid("missing graph snapshot magic"));
        }
        let header_len = SNAPSHOT_MAGIC.len() + 4;
        if bytes.len() < header_len {
            return Err(invalid("truncated graph snapshot header"));
        }
        let version =
            u32::from_le_bytes(bytes[SNAPSHOT_MAGIC.len()..header_len].try_into().unwrap());
        if version != SNAPSHOT_VERSION {
            return Err(invalid("unsupported graph snapshot version"));
        }
        let snapshot: BinaryGraphSnapshot = bincode::deserialize(&bytes[header_len..])?;
        let mut graph = MemoryGraph {
            interner: NodeInterner::default(),
            int_edges: Vec::new(),
            int_adjacency: IntAdjacency::default(),
            last_memory_id: snapshot.last_memory_id,
            last_memory_by_scope: HashMap::new(),
            co_occurrence: snapshot.co_occurrence,
            vocab: ConceptVocabulary::from_alias_pairs(snapshot.vocab_aliases),
            suppressed_concepts: snapshot.suppressed_concepts,
        };
        for (key, node) in snapshot.nodes {
            graph.interner.intern(key, node);
        }
        // Drop edges referencing out-of-range nodes (corrupt snapshot) rather
        // than panicking later in `edges()`/`neighbors()`.
        let node_bound = graph.interner.id_to_node.len() as u32;
        graph.int_edges = snapshot
            .edges
            .into_iter()
            .filter(|e| e.source < node_bound && e.target < node_bound)
            .collect();
        graph.int_adjacency.rebuild(&graph.int_edges);
        Ok(graph)
    }

    pub fn compact(&mut self) {
        self.int_edges.sort_by(|a, b| {
            let a_src = self.interner.get(a.source).map(|n| &n.external_id);
            let b_src = self.interner.get(b.source).map(|n| &n.external_id);
            let a_tgt = self.interner.get(a.target).map(|n| &n.external_id);
            let b_tgt = self.interner.get(b.target).map(|n| &n.external_id);
            a_src
                .cmp(&b_src)
                .then(a_tgt.cmp(&b_tgt))
                .then(a.kind.cmp(&b.kind))
                .then(
                    a.weight
                        .partial_cmp(&b.weight)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        self.int_adjacency.rebuild(&self.int_edges);
    }

    // ------------------------------------------------------------------
    // Internal accessors for activation.rs (integer-based, fast)
    // ------------------------------------------------------------------

    /// Get the integer node ID for a memory ID string (fast path for activation).
    pub(crate) fn memory_node_id(&self, id: &str) -> Option<u32> {
        let key = NodeId::memory(id).as_str();
        self.get_node_id(&key)
    }

    /// Get neighbor edge indices for a node ID (zero-allocation).
    pub(crate) fn neighbor_indices(&self, node_id: u32) -> Option<&[usize]> {
        self.int_adjacency.get(node_id)
    }

    /// Get an internal edge by index.
    pub(crate) fn get_edge(&self, idx: usize) -> Option<&InternalEdge> {
        self.int_edges.get(idx)
    }

    /// Get the external memory ID for a node ID (for result hydration).
    pub(crate) fn node_external_id(&self, node_id: u32) -> Option<&str> {
        self.interner.get(node_id).map(|n| n.external_id.as_str())
    }

    /// Get the kind of a node.
    pub(crate) fn node_kind(&self, node_id: u32) -> Option<NodeKind> {
        self.interner.get(node_id).map(|n| n.kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{SpreadingActivation, SpreadingConfig};

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
        let m1_edges: Vec<Edge> = g
            .neighbors(&NodeId::memory("m1").as_str())
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .collect();
        let m2_edges: Vec<Edge> = g
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
        for e in g.neighbors(&NodeId::memory("m1").as_str()) {
            assert_eq!(e.last_reinforced_at, 1000);
        }
    }

    #[test]
    fn reinforce_is_bounded() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "x", &["c".into()]);
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
        assert_eq!(g.build_abstractions(3), 0);
        assert_eq!(g.abstraction_count(), 0);
    }

    #[test]
    fn graph_roundtrips_through_json_with_learned_state() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "Rust is safe", &["rust".into(), "safety".into()], 2.0);
        g.add_memory_with_importance("m2", "Rust is fast", &["rust".into(), "speed".into()], 1.5);
        g.reinforce("m1", 1234);
        g.build_abstractions(1);

        let json = g.to_json();
        let restored = MemoryGraph::from_json(&json).expect("roundtrip");

        let m1_edges: Vec<Edge> = restored
            .neighbors(&NodeId::memory("m1").as_str())
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .collect();
        assert!(
            m1_edges.iter().any(|e| e.last_reinforced_at == 1234),
            "reinforcement timestamp should survive JSON roundtrip"
        );
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
    fn graph_roundtrips_through_binary_with_learned_state() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "Rust is safe", &["rust".into(), "safety".into()], 2.0);
        g.add_memory_with_importance("m2", "Rust is fast", &["rust".into(), "speed".into()], 1.5);
        g.add_memory("m3", "Rust ownership", &["rust".into(), "ownership".into()]);
        g.reinforce("m1", 1234);
        g.build_abstractions(1);
        g.add_refinement("m1", "m2", 0.8);
        g.vocab_mut().add_alias("coding", "programming");
        g.suppressed_concepts.insert("hub".into());
        g.remove_memory("m3"); // leave an interner tombstone to compact over
        g.compact();

        let bytes = g.to_snapshot_bytes().expect("binary save");
        assert!(MemoryGraph::is_snapshot_bytes(&bytes));
        let restored = MemoryGraph::from_snapshot_bytes(&bytes).expect("binary roundtrip");

        // Reinforcement timestamps survive.
        let m1_edges: Vec<Edge> = restored
            .neighbors(&NodeId::memory("m1").as_str())
            .into_iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .collect();
        assert!(
            m1_edges.iter().any(|e| e.last_reinforced_at == 1234),
            "reinforcement timestamp should survive binary roundtrip"
        );
        // Node/edge counts match; the removed memory stays removed.
        assert_eq!(restored.node_count(), g.node_count());
        assert_eq!(restored.edge_count(), g.edge_count());
        assert!(restored.memory_node_id("m3").is_none());
        // Abstractions and belief edges survive.
        assert_eq!(restored.abstraction_count(), g.abstraction_count());
        assert_eq!(restored.refinement_count(), 1);
        assert_eq!(restored.refined_by("m1"), vec!["m2".to_string()]);
        // Co-occurrence counters, vocab aliases, suppression survive.
        assert_eq!(restored.co_occurrence, g.co_occurrence);
        assert_eq!(restored.vocab().aliases(), g.vocab().aliases());
        assert!(restored.is_concept_suppressed("hub"));
        assert_eq!(restored.last_memory_id, g.last_memory_id);
        // The full edge list is identical (both sides compact-sorted).
        assert_eq!(
            serde_json::to_string(&g.edges()).unwrap(),
            serde_json::to_string(&restored.edges()).unwrap()
        );
        // Determinism: re-saving the restored graph yields identical bytes.
        assert_eq!(restored.to_snapshot_bytes().unwrap(), bytes);
    }

    #[test]
    fn json_snapshot_loads_then_resaves_as_binary() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "Rust is safe", &["rust".into(), "safety".into()], 2.0);
        g.add_memory_with_importance("m2", "Rust is fast", &["rust".into(), "speed".into()], 1.5);
        g.reinforce("m1", 1234);
        g.build_abstractions(1);

        // Legacy on-disk format: JSON produced by the old save path.
        let json = g.to_json();
        assert!(!MemoryGraph::is_snapshot_bytes(json.as_bytes()));
        let from_json = MemoryGraph::from_json(&json).expect("legacy json loads");

        // Saving going forward produces a binary snapshot that reloads to the
        // same state.
        let bytes = from_json.to_snapshot_bytes().expect("binary save");
        let restored = MemoryGraph::from_snapshot_bytes(&bytes).expect("binary reload");
        assert_eq!(restored.node_count(), from_json.node_count());
        assert_eq!(restored.edge_count(), from_json.edge_count());
        assert_eq!(restored.abstraction_count(), from_json.abstraction_count());
        assert_eq!(restored.co_occurrence, from_json.co_occurrence);
        assert_eq!(
            serde_json::to_string(&from_json.edges()).unwrap(),
            serde_json::to_string(&restored.edges()).unwrap()
        );
        // The binary form is stable across further save/load cycles.
        assert_eq!(restored.to_snapshot_bytes().unwrap(), bytes);
    }

    #[test]
    fn binary_snapshot_is_far_smaller_than_json() {
        // Representative graph: 1k memories, a handful of concepts each drawn
        // from a shared pool so concept nodes are reused.
        let mut g = MemoryGraph::new();
        for i in 0..1000usize {
            let concepts: Vec<String> = (0..4)
                .map(|j| format!("concept-{}", (i * 7 + j * 13) % 200))
                .collect();
            g.add_memory(
                &format!("mem-{i}"),
                &format!("representative text for memory number {i}"),
                &concepts,
            );
        }
        let json_len = g.to_json().len();
        let bin_len = g.to_snapshot_bytes().unwrap().len();
        let json_per_record = json_len as f64 / 1000.0;
        let bin_per_record = bin_len as f64 / 1000.0;
        assert!(
            bin_len * 5 < json_len,
            "binary ({bin_len} B) should be at least 5x smaller than json ({json_len} B)"
        );
        assert!(
            bin_per_record < 2048.0,
            "binary should be well under 2 KB/record, got {bin_per_record:.0} B (json {json_per_record:.0} B)"
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
        let old_neighbors = g.neighbors(&NodeId::memory("old").as_str());
        assert!(old_neighbors
            .iter()
            .any(|e| { e.kind == EdgeKind::Refines && e.target.as_str() == "mem:new" }));
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
        assert!(!g.add_refinement("old", "new", 0.8));
        assert!(!g.add_refinement("ghost", "old", 0.8));
        assert_eq!(g.refinement_count(), 0);
    }

    #[test]
    fn refined_by_returns_superseded_memories() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        g.add_memory("new", "b", &["c".into()]);
        g.add_refinement("old", "new", 0.8);
        let refined = g.refined_by("old");
        assert_eq!(refined, vec!["new".to_string()]);
        assert!(g.refined_by("new").is_empty());
    }

    #[test]
    fn superseded_ids_returns_old_side_of_belief_edges() {
        let mut g = MemoryGraph::new();
        g.add_memory("old_r", "a", &["c".into()]);
        g.add_memory("new_r", "b", &["c".into()]);
        g.add_memory("old_c", "d", &["c".into()]);
        g.add_memory("new_c", "e", &["c".into()]);
        g.add_memory("plain", "f", &["c".into()]); // no belief edge
        g.add_refinement("old_r", "new_r", 0.8);
        g.add_contradiction("old_c", "new_c", 0.8, 0.5);
        let mut ids = g.superseded_ids();
        ids.sort();
        assert_eq!(ids, vec!["old_c".to_string(), "old_r".to_string()]);
        // Newer memories and unrelated memories are NOT superseded.
        assert!(!ids.contains(&"new_r".to_string()));
        assert!(!ids.contains(&"plain".to_string()));
    }

    /// Build the linear A -> B -> C supersession chain exactly as detection
    /// commits it (Refines edges, old -> new).
    fn chain_graph() -> MemoryGraph {
        let mut g = MemoryGraph::new();
        g.add_memory("a", "fact v1", &["c".into()]);
        g.add_memory("b", "fact v2", &["c".into()]);
        g.add_memory("c", "fact v3", &["c".into()]);
        g.add_refinement("a", "b", 0.8);
        g.add_refinement("b", "c", 0.8);
        g
    }

    #[test]
    fn belief_head_walks_linear_chain_to_newest() {
        let g = chain_graph();
        assert_eq!(g.belief_head("a"), "c");
        assert_eq!(g.belief_head("b"), "c");
        assert_eq!(g.belief_head("c"), "c", "head resolves to itself");
    }

    #[test]
    fn belief_lineage_returns_full_chain_oldest_first() {
        let g = chain_graph();
        let want = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert_eq!(g.belief_lineage("a"), want, "from the oldest");
        assert_eq!(g.belief_lineage("b"), want, "from the middle");
        assert_eq!(g.belief_lineage("c"), want, "from the head");
    }

    #[test]
    fn belief_chain_follows_contradiction_edges_too() {
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        g.add_memory("new", "b", &["c".into()]);
        g.add_contradiction("old", "new", 0.8, 0.5);
        assert_eq!(g.belief_head("old"), "new");
        assert_eq!(
            g.belief_lineage("old"),
            vec!["old".to_string(), "new".to_string()]
        );
    }

    #[test]
    fn belief_head_branching_follows_latest_inserted_edge() {
        // Two newer memories both refine the same old one: the walk follows
        // the edge committed most recently (deterministic branching rule).
        let mut g = MemoryGraph::new();
        g.add_memory("old", "a", &["c".into()]);
        g.add_memory("first", "b", &["c".into()]);
        g.add_memory("second", "c", &["c".into()]);
        g.add_refinement("old", "first", 0.8);
        g.add_refinement("old", "second", 0.8);
        assert_eq!(g.belief_head("old"), "second");
        assert_eq!(
            g.belief_lineage("old"),
            vec!["old".to_string(), "second".to_string()]
        );
    }

    #[test]
    fn belief_walk_survives_cycles() {
        // Corrupted data (a -> b -> a) must not hang recall: the walk stops
        // at the node where the cycle is detected.
        let mut g = MemoryGraph::new();
        g.add_memory("a", "x", &["c".into()]);
        g.add_memory("b", "y", &["c".into()]);
        g.add_refinement("a", "b", 0.8);
        g.add_refinement("b", "a", 0.8);
        assert_eq!(
            g.belief_head("a"),
            "b",
            "b's successor a is already visited"
        );
        assert_eq!(g.belief_head("b"), "a");
        assert_eq!(
            g.belief_lineage("a"),
            vec!["a".to_string(), "b".to_string()],
            "lineage terminates instead of looping"
        );
    }

    #[test]
    fn belief_resolution_of_unknown_id_is_identity() {
        let g = MemoryGraph::new();
        assert_eq!(g.belief_head("ghost"), "ghost");
        assert_eq!(g.belief_lineage("ghost"), vec!["ghost".to_string()]);
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
    fn memory_concepts_returns_attached_concepts() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "Rust safety", &["rust".into(), "safety".into()]);
        g.add_memory(
            "m2",
            "Python concurrency",
            &["python".into(), "concurrency".into()],
        );
        let mut c1 = g.memory_concepts("m1");
        c1.sort();
        assert_eq!(c1, vec!["rust".to_string(), "safety".to_string()]);
        let mut c2 = g.memory_concepts("m2");
        c2.sort();
        assert_eq!(c2, vec!["concurrency".to_string(), "python".to_string()]);
    }

    #[test]
    fn memory_concepts_unknown_id_returns_empty() {
        let g = MemoryGraph::new();
        assert!(g.memory_concepts("does_not_exist").is_empty());
    }

    #[test]
    fn stats_reports_counts() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "Rust safety", &["rust".into(), "safety".into()]);
        g.add_memory(
            "m2",
            "Rust concurrency",
            &["rust".into(), "concurrency".into()],
        );
        g.add_refinement("m1", "m2", 0.8);
        let s = g.stats();
        assert_eq!(s.memory_count, 2);
        assert_eq!(s.concept_count, 3);
        assert_eq!(s.refinement_count, 1);
        assert_eq!(s.contradiction_count, 0);
        assert_eq!(s.abstraction_count, 0);
        assert!(s.node_count >= 5);
        assert!(s.edge_count >= 6);
    }

    #[test]
    fn concept_count_excludes_abstraction_parents() {
        let mut g = MemoryGraph::new();
        for i in 0..3 {
            g.add_memory(&format!("m{i}"), "text", &["rust".into(), "safety".into()]);
        }
        g.build_abstractions(3);
        assert_eq!(g.concept_count(), 2);
        assert_eq!(g.abstraction_count(), 1);
    }

    #[test]
    fn reweight_memory_scales_edges_to_new_importance() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "Rust safety", &["rust".into(), "safety".into()], 1.0);
        let before: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (before - 1.0).abs() < 1e-5,
            "baseline min assoc weight {before}"
        );

        let changed = g.reweight_memory("m1", 4.0);
        assert!(changed, "should reweight existing memory");
        let after: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .fold(f32::INFINITY, f32::min);
        assert!(
            (after - 2.0).abs() < 1e-4,
            "min assoc weight should be importance_factor(4.0)=2.0, got {after}"
        );
    }

    #[test]
    fn reweight_memory_preserves_relative_reinforcement() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "Rust", &["rust".into()], 1.0);
        g.reinforce("m1", 1000);
        let reinforced_before: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .fold(0.0, f32::max);
        assert!(
            reinforced_before > 1.0,
            "reinforced edge should be above baseline: {reinforced_before}"
        );

        g.reweight_memory("m1", 1.0);
        let reinforced_after: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .fold(0.0, f32::max);
        assert!(
            (reinforced_after - reinforced_before).abs() < 1e-5,
            "reweight to same importance should preserve reinforcement: {reinforced_after} vs {reinforced_before}"
        );

        g.reweight_memory("m1", 4.0);
        let reinforced_scaled: f32 = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .filter(|e| e.kind == EdgeKind::Association)
            .map(|e| e.weight)
            .fold(0.0, f32::max);
        assert!(
            (reinforced_scaled - reinforced_before * 2.0).abs() < 1e-4,
            "reinforced edge should scale by 2.0: {reinforced_scaled} vs {}",
            reinforced_before * 2.0
        );
    }

    #[test]
    fn reweight_memory_unknown_id_returns_false() {
        let mut g = MemoryGraph::new();
        assert!(!g.reweight_memory("nope", 2.0));
    }

    #[test]
    fn evolve_vocabulary_merges_overlapping_concepts() {
        let mut g = MemoryGraph::new();
        g.add_memory(
            "m1",
            "coding programming",
            &["coding".into(), "programming".into()],
        );
        g.add_memory(
            "m2",
            "coding programming rust",
            &["coding".into(), "programming".into(), "rust".into()],
        );
        let stats = g.evolve_vocabulary(0.9, 0.0, 1024);
        assert_eq!(stats.merged, 1, "expected one merge, got {stats:?}");
        let survivor = if "coding" < "programming" {
            "coding"
        } else {
            "programming"
        };
        let loser = if survivor == "coding" {
            "programming"
        } else {
            "coding"
        };
        assert!(g.nodes().contains_key(&NodeId::concept(survivor).as_str()));
        assert!(g.nodes().contains_key(&NodeId::concept("rust").as_str()));
        assert!(!g.nodes().contains_key(&NodeId::concept(loser).as_str()));
        assert_eq!(g.vocab().resolve(loser), survivor);
    }

    #[test]
    fn merge_concept_node_migrates_both_edge_directions() {
        // Regression: the outgoing-edge arm re-created concept -> memory edges
        // sourced at the *alias*, which the cleanup retain then deleted — the
        // canonical concept ended up with memory -> concept edges but no
        // concept -> memory edges (asymmetric adjacency, undercounted degree,
        // unreachable sibling hop in spreading activation).
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "rust coding", &["coding".into()]);
        g.add_memory("m2", "programming", &["programming".into()]);

        assert!(g.merge_concept_node("programming", "coding"));

        // Incoming migration: m2's association now targets the canonical concept.
        let m2_out = g.neighbors(&NodeId::memory("m2").as_str());
        assert!(
            m2_out
                .iter()
                .any(|e| e.kind == EdgeKind::Association && e.target == NodeId::concept("coding")),
            "m2 -> coding edge missing after merge"
        );

        // Outgoing migration: the canonical concept reaches BOTH memories.
        assert_eq!(g.concept_degree("coding"), 2);
        let coding_out = g.neighbors(&NodeId::concept("coding").as_str());
        assert!(
            coding_out
                .iter()
                .any(|e| e.kind == EdgeKind::Association && e.target == NodeId::memory("m2")),
            "coding -> m2 edge missing after merge"
        );

        // No edges reference the alias anymore.
        assert!(g
            .neighbors(&NodeId::concept("programming").as_str())
            .is_empty());
        assert!(!g
            .nodes()
            .contains_key(&NodeId::concept("programming").as_str()));
        assert!(g
            .edges()
            .iter()
            .all(|e| e.source != NodeId::concept("programming")
                && e.target != NodeId::concept("programming")));
    }

    #[test]
    fn merge_concept_node_rename_survives_snapshot() {
        // Regression: the rename-in-place path (canonical node absent) updated
        // key_to_id but left id_to_key pointing at the alias, so iter()
        // filtered the renamed node out and snapshots silently dropped it.
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "rust coding", &["coding".into()]);
        let before = g.node_count();

        assert!(g.merge_concept_node("coding", "programming"));

        // Renamed in place: same node count, new key visible, edges intact.
        assert_eq!(g.node_count(), before);
        assert!(g
            .nodes()
            .contains_key(&NodeId::concept("programming").as_str()));
        assert!(!g.nodes().contains_key(&NodeId::concept("coding").as_str()));
        assert_eq!(g.concept_degree("programming"), 1);

        // The renamed node and its edges survive a snapshot round-trip.
        let bytes = g.to_snapshot_bytes().unwrap();
        let restored = MemoryGraph::from_snapshot_bytes(&bytes).unwrap();
        assert_eq!(restored.node_count(), before);
        assert!(restored
            .nodes()
            .contains_key(&NodeId::concept("programming").as_str()));
        assert_eq!(restored.concept_degree("programming"), 1);
        assert!(restored
            .neighbors(&NodeId::concept("programming").as_str())
            .iter()
            .any(|e| e.kind == EdgeKind::Association && e.target == NodeId::memory("m1")));
    }

    #[test]
    fn add_concepts_to_memory_is_side_effect_free_beyond_new_edges() {
        // Belief revision merges an older memory's concepts into a newer one
        // after both were inserted. The full insert path used for that
        // duplicated existing edges, double-counted co-occurrence, and
        // re-pointed the temporal chain head at an old memory.
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "alpha beta", &["alpha".into(), "beta".into()]);
        g.add_memory("m2", "gamma", &["gamma".into()]);
        let edges_before = g.edge_count();
        let pair_key = "concept:alpha\0concept:beta".to_string();
        let co_before = g.co_occurrence.get(&pair_key).copied();

        // "beta" already linked (dedup), "delta" is new.
        assert!(g.add_concepts_to_memory("m1", &["beta".into(), "delta".into()], 1.0));

        // Exactly the two new edges (m1 -> delta, delta -> m1).
        assert_eq!(g.edge_count(), edges_before + 2);
        assert!(g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .any(|e| e.kind == EdgeKind::Association && e.target == NodeId::concept("delta")));

        // Co-occurrence not recounted, temporal chain head not moved.
        assert_eq!(g.co_occurrence.get(&pair_key).copied(), co_before);
        assert_eq!(g.last_memory_id.as_deref(), Some("m2"));

        // A second identical call adds nothing.
        assert!(g.add_concepts_to_memory("m1", &["delta".into()], 1.0));
        assert_eq!(g.edge_count(), edges_before + 2);

        assert!(!g.add_concepts_to_memory("nope", &["x".into()], 1.0));
    }

    #[test]
    fn non_finite_importance_does_not_poison_edge_weights() {
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "nan", &["c".into()], f32::NAN);
        g.add_memory_with_importance("m2", "inf", &["c".into()], f32::INFINITY);
        for e in g.edges() {
            assert!(e.weight.is_finite(), "non-finite edge weight: {e:?}");
        }
    }

    #[test]
    fn decay_never_raises_low_baseline_edge_weight() {
        // importance 0.04 -> factor 0.2: association edges start below the
        // 1.0 decay floor. Decay must not push them back up toward it.
        let mut g = MemoryGraph::new();
        g.add_memory_with_importance("m1", "low importance", &["low".into()], 0.04);
        g.reinforce("m1", 1);
        let assoc = |g: &MemoryGraph| {
            g.neighbors(&NodeId::memory("m1").as_str())
                .iter()
                .filter(|e| e.kind == EdgeKind::Association)
                .map(|e| e.weight)
                .fold(0.0, f32::max)
        };
        let before = assoc(&g);
        assert!(before < 1.0, "setup: edge should start below the floor");

        g.decay_edges(1_000, 10);
        let after = assoc(&g);
        assert!(
            after <= before,
            "decay raised a low-baseline edge: {before} -> {after}"
        );
    }

    #[test]
    fn reinforce_at_time_zero_does_not_collide_with_never_reinforced() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "text", &["c".into()]);
        g.reinforce("m1", 0);
        // t=0 is the "never reinforced" sentinel; the call must record a
        // non-zero timestamp so the edge is not permanently decay-exempt.
        let ts = g
            .neighbors(&NodeId::memory("m1").as_str())
            .iter()
            .map(|e| e.last_reinforced_at)
            .max()
            .unwrap();
        assert!(ts > 0, "t=0 reinforce must be remapped above the sentinel");
    }

    #[test]
    fn evolve_vocabulary_suppresses_hub_concepts() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "a", &["system".into(), "alpha".into()]);
        g.add_memory("m2", "b", &["system".into(), "beta".into()]);
        g.add_memory("m3", "c", &["system".into(), "gamma".into()]);
        g.add_memory("m4", "d", &["delta".into()]);
        let stats = g.evolve_vocabulary(1.0, 0.1, 1024);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.suppressed, 1, "expected one hub suppression");
        assert!(g.is_concept_suppressed("system"));
        assert!(!g.is_concept_suppressed("alpha"));
    }

    #[test]
    fn evolve_vocabulary_does_nothing_when_disabled() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "a", &["coding".into()]);
        g.add_memory("m2", "b", &["programming".into()]);
        let stats = g.evolve_vocabulary(0.5, 0.1, 0);
        assert_eq!(stats.merged, 0);
        assert_eq!(stats.suppressed, 0);
    }

    #[test]
    fn suppressed_concept_does_not_expand() {
        let mut g = MemoryGraph::new();
        g.add_memory(
            "m2",
            "programming python",
            &["programming".into(), "python".into()],
        );
        g.add_memory(
            "m1",
            "rust programming",
            &["rust".into(), "programming".into()],
        );
        g.evolve_vocabulary(1.0, 0.0, 1024);
        g.suppressed_concepts.insert("programming".into());

        let sa = SpreadingActivation::new(g, SpreadingConfig::default());
        let results = sa.search("rust", &[("m1".into(), 0.9)], 5);
        assert!(results.is_some());
        let ids: Vec<String> = results.unwrap().into_iter().map(|(id, _)| id).collect();
        assert!(ids.contains(&"m1".to_string()));
        assert!(!ids.contains(&"m2".to_string()));
    }

    #[test]
    fn vocab_and_suppressed_survive_json_roundtrip() {
        let mut g = MemoryGraph::new();
        g.add_memory("m1", "coding", &["coding".into()]);
        g.add_memory("m2", "programming", &["programming".into()]);
        g.vocab_mut().add_alias("programming", "coding");
        g.suppressed_concepts.insert("coding".into());

        let json = g.to_json();
        let restored = MemoryGraph::from_json(&json).expect("roundtrip");
        assert_eq!(restored.vocab().resolve("programming"), "coding");
        assert!(restored.is_concept_suppressed("coding"));
    }
}
