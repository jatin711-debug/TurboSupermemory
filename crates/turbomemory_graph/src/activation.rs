//! Spreading activation retrieval over the cognitive graph.

use crate::bm25::Bm25Index;
use crate::graph::{EdgeKind, MemoryGraph};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct SpreadingConfig {
    /// Scaling factor for the initial semantic trigger.
    pub semantic_alpha: f32,
    /// Scaling factor for the initial lexical trigger.
    pub lexical_alpha: f32,
    /// Energy decay per hop.
    pub decay: f32,
    /// Number of propagation iterations.
    pub iterations: usize,
    /// Lateral inhibition strength.
    pub beta: f32,
    /// Feeling-of-Knowing gate threshold. If the peak memory activation is below this, return None.
    pub fok_threshold: f32,
    /// Maximum number of activated nodes kept after each propagation iteration.
    /// Limits memory and time for queries that would otherwise activate the
    /// whole graph (e.g. empty text or very common terms).
    pub max_frontier: usize,
    /// When `query_text` is empty, concept nodes that connect to more than this
    /// fraction of all memories are treated as hubs and are not expanded.
    pub hub_fraction_threshold: f32,
}

impl Default for SpreadingConfig {
    fn default() -> Self {
        Self {
            semantic_alpha: 1.0,
            lexical_alpha: 0.6,
            decay: 0.5,
            iterations: 4,
            beta: 0.3,
            fok_threshold: 0.58,
            max_frontier: 1_000,
            hub_fraction_threshold: 0.05,
        }
    }
}

/// Combines BM25 lexical triggers and dense semantic seeds, then propagates
/// activation through the memory graph.
pub struct SpreadingActivation {
    graph: MemoryGraph,
    bm25: Bm25Index,
    config: SpreadingConfig,
}

impl SpreadingActivation {
    pub fn new(graph: MemoryGraph, config: SpreadingConfig) -> Self {
        let mut bm25 = Bm25Index::new();
        for (key, node) in graph.iter_memory_nodes() {
            let id = key.strip_prefix("mem:").unwrap_or(key);
            bm25.add(id, &node.text);
        }
        Self {
            graph,
            bm25,
            config,
        }
    }

    pub fn add_memory(&mut self, id: &str, text: &str, concepts: &[String]) {
        self.graph.add_memory(id, text, concepts);
        self.bm25.add(id, text);
    }

    /// Insert a memory with an importance-weighted edge contribution. See
    /// [`MemoryGraph::add_memory_with_importance`].
    pub fn add_memory_with_importance(
        &mut self,
        id: &str,
        text: &str,
        concepts: &[String],
        importance: f32,
    ) {
        self.graph
            .add_memory_with_importance(id, text, concepts, importance);
        self.bm25.add(id, text);
    }

    pub fn remove_memory(&mut self, id: &str) {
        self.graph.remove_memory(id);
        self.bm25.remove(id);
    }

    /// Reinforce the edges of a retrieved memory (rehearsal). See
    /// [`MemoryGraph::reinforce`].
    pub fn reinforce(&mut self, id: &str, now: u64) {
        self.graph.reinforce(id, now);
    }

    /// Apply time-decay to all reinforced edges. See [`MemoryGraph::decay_edges`].
    pub fn decay_edges(&mut self, now: u64, half_life: u64) {
        self.graph.decay_edges(now, half_life);
    }

    /// Build abstraction hierarchy from accumulated concept co-occurrence.
    /// See [`MemoryGraph::build_abstractions`]. Returns the number of new
    /// abstraction edges added.
    pub fn build_abstractions(&mut self, threshold: usize) -> usize {
        self.graph.build_abstractions(threshold)
    }

    /// Create a `Refines` edge from an older memory to a newer one. See
    /// [`MemoryGraph::add_refinement`]. Returns `true` if a new edge was
    /// created.
    pub fn add_refinement(&mut self, old_id: &str, new_id: &str, weight: f32) -> bool {
        self.graph.add_refinement(old_id, new_id, weight)
    }

    /// Create a `Contradicts` edge and weaken the old memory's edges. See
    /// [`MemoryGraph::add_contradiction`]. Returns `true` if a new edge
    /// was created.
    pub fn add_contradiction(
        &mut self,
        old_id: &str,
        new_id: &str,
        weight: f32,
        weaken_factor: f32,
    ) -> bool {
        self.graph
            .add_contradiction(old_id, new_id, weight, weaken_factor)
    }

    /// Semantic seeds are `(memory_id, normalized_similarity)` pairs from dense ANN search.
    /// Returns `None` when the Feeling-of-Knowing gate rejects the query.
    pub fn search(
        &self,
        query_text: &str,
        semantic_seeds: &[(String, f32)],
        top_k: usize,
    ) -> Option<Vec<(String, f32)>> {
        let mut activation: BTreeMap<String, f32> = BTreeMap::new();

        // Semantic trigger
        for (id, score) in semantic_seeds {
            *activation.entry(format!("mem:{id}")).or_insert(0.0) +=
                self.config.semantic_alpha * score.max(0.0);
        }

        // Lexical trigger
        let lexical = self.bm25.score(query_text);
        let max_lex = lexical.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let lex_norm = if max_lex > 0.0 { 1.0 / max_lex } else { 0.0 };
        for (id, score) in lexical {
            let key = format!("mem:{id}");
            let a = self.config.lexical_alpha * score * lex_norm;
            *activation.entry(key).or_insert(0.0) += a;
        }

        if activation.is_empty() {
            return None;
        }

        // Early-exit FOK gate: if even the best seed is below threshold,
        // propagation cannot rescue the query.
        let peak_seed = activation.values().cloned().fold(0.0f32, f32::max);
        if peak_seed < self.config.fok_threshold {
            return None;
        }

        let total_memories = self.graph.memory_count().max(1);
        let is_empty_query = query_text.trim().is_empty();
        let hub_threshold =
            (total_memories as f32 * self.config.hub_fraction_threshold).max(2.0) as usize;
        let max_frontier = self.config.max_frontier.max(top_k * 4);

        // Spreading activation (BTreeMap iteration is sorted/deterministic)
        for _ in 0..self.config.iterations {
            let mut next = activation.clone();
            for (key, energy) in &activation {
                if *energy <= 0.0 {
                    continue;
                }
                for edge in self.graph.neighbors(key) {
                    if edge.kind == EdgeKind::Temporal && edge.weight < 0.0 {
                        continue;
                    }
                    // All edge kinds are traversed: Association (mem<->concept),
                    // Temporal (prev_mem -> mem), Abstraction (concept <-> parent),
                    // and Refines (old_mem -> new_mem). The Refines traversal is
                    // what makes memory evolution work: when activation reaches
                    // an older, superseded memory, it propagates to the newer
                    // refinement, ensuring the most current knowledge surfaces.
                    // Hub suppression for empty queries: do not expand concept
                    // nodes that are connected to a large fraction of memories.
                    // Abstraction (parent concept) edges are exempt from hub
                    // suppression because they connect concepts to concepts,
                    // not to memories, and are the graph's generalization path.
                    if is_empty_query
                        && edge.kind == EdgeKind::Association
                        && edge.source.as_str().starts_with("concept:")
                    {
                        let source_key = edge.source.as_str();
                        let concept = source_key.strip_prefix("concept:").unwrap_or("");
                        if self.graph.concept_degree(concept) > hub_threshold {
                            continue;
                        }
                    }
                    let target = edge.target.as_str();
                    *next.entry(target).or_insert(0.0) += energy * edge.weight * self.config.decay;
                }
            }
            activation = next;

            // Frontier cap: keep only the top-max_frontier activated nodes.
            // Always keep at least top_k memory nodes so we do not discard
            // genuine results prematurely.
            if activation.len() > max_frontier {
                let mut ranked: Vec<(String, f32)> = activation.into_iter().collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let mut kept_memory = 0usize;
                let mut kept = BTreeMap::new();
                for (k, v) in ranked {
                    let is_mem = k.starts_with("mem:");
                    if kept.len() < max_frontier || (is_mem && kept_memory < top_k) {
                        if is_mem {
                            kept_memory += 1;
                        }
                        kept.insert(k, v);
                    }
                    if kept.len() >= max_frontier && kept_memory >= top_k {
                        break;
                    }
                }
                activation = kept;
            }

            // Lateral inhibition among memory nodes in the current frontier.
            let memory_keys: Vec<String> = activation
                .keys()
                .filter(|k| k.starts_with("mem:"))
                .cloned()
                .collect();
            let mut inhibited = activation.clone();
            for key in &memory_keys {
                let u = *activation.get(key).unwrap_or(&0.0);
                let penalty: f32 = memory_keys
                    .iter()
                    .filter(|k| *k != key)
                    .map(|k| {
                        let uk = *activation.get(k).unwrap_or(&0.0);
                        if uk > u {
                            uk - u
                        } else {
                            0.0
                        }
                    })
                    .sum::<f32>()
                    * self.config.beta;
                let new_u = (u - penalty).max(0.0);
                inhibited.insert(key.clone(), new_u);
            }
            activation = inhibited;
        }

        // Extract memory-node activations
        let mut memories: Vec<(String, f32)> = activation
            .into_iter()
            .filter(|(k, _)| k.starts_with("mem:"))
            .map(|(k, v)| (k.strip_prefix("mem:").unwrap_or(&k).to_string(), v))
            .collect();
        memories.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let peak = memories.first().map(|(_, s)| *s).unwrap_or(0.0);
        if peak < self.config.fok_threshold {
            return None;
        }

        memories.truncate(top_k);
        Some(memories)
    }

    pub fn graph(&self) -> &MemoryGraph {
        &self.graph
    }

    pub fn graph_mut(&mut self) -> &mut MemoryGraph {
        &mut self.graph
    }

    pub fn into_graph(self) -> MemoryGraph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spreading_retrieves_related_memory() {
        let mut graph = MemoryGraph::new();
        graph.add_memory(
            "m1",
            "Rust is fast and safe",
            &["rust".into(), "safety".into()],
        );
        graph.add_memory("m2", "Python is easy", &["python".into()]);
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());
        let results = sa.search("Rust safety", &[("m1".into(), 0.9)], 2);
        assert!(results.is_some());
        let r = results.unwrap();
        assert_eq!(r[0].0, "m1");
    }

    #[test]
    fn fok_rejects_unrelated_query() {
        let mut graph = MemoryGraph::new();
        graph.add_memory("m1", "Rust is fast", &["rust".into()]);
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());
        let results = sa.search("banana chocolate cookie recipe", &[], 2);
        assert!(results.is_none());
    }
}
