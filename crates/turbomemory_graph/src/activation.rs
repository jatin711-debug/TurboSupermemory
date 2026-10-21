//! Spreading activation retrieval over the cognitive graph.

use crate::bm25::Bm25Index;
use crate::graph::{EdgeKind, MemoryGraph, NodeKind};

/// Configuration for the bounded augmenter search.
///
/// The augmenter takes the ANN candidate set as a non-negotiable recall floor
/// and performs a single bounded 1-hop graph expansion that can only *add*
/// candidates and apply a small additive re-rank boost — it never drops or
/// reorders away an ANN hit.
#[derive(Debug, Clone)]
pub struct SpreadingConfig {
    /// Scaling factor for the initial semantic (ANN) trigger.
    pub semantic_alpha: f32,
    /// Scaling factor for the BM25 lexical boost.
    pub lexical_alpha: f32,
    /// Energy decay applied to graph-discovered candidates (1 hop).
    pub decay: f32,
    /// Number of expansion hops. `0` disables graph expansion (pure ANN+BM25).
    /// `1` is the intended balanced setting.
    pub iterations: usize,
    /// Number of top-scoring ANN seeds to expand from (M). Clamped to [1, 20].
    pub seed_hops_from: usize,
    /// Maximum number of new candidates added by graph expansion. Clamped to
    /// [10, 200]. Bounds the cost and blast radius of the augmenter.
    pub expansion_max_candidates: usize,
    /// Enable concept-mediated expansion: the 2-hop `mem → concept → sibling
    /// memory` and 4-hop `mem → concept → abstraction parent → sibling concept →
    /// memory` paths. This is what lets the concept graph + abstraction
    /// hierarchy influence ranking (a query can reach a memory it shares a
    /// concept/parent with, despite low cosine). `true` (default) is the current
    /// behavior; `false` isolates the belief/temporal edges from concept +
    /// abstraction expansion (the W4 OFF arm). Belief (`Refines`/`Contradicts`)
    /// and `Temporal` traversal are unaffected either way.
    pub concept_expansion: bool,
    /// Weight of the direct query-concept seeding: a memory tagged with a
    /// concept that is also a word of the query gets `query_concept_boost *
    /// edge weight` for each such word. `0.75` is the original behaviour, in
    /// which three shared words outweigh any cosine difference between two
    /// candidates; `0.0` switches the seeding off (BM25 already scores word
    /// overlap, with IDF and length normalization).
    pub query_concept_boost: f32,
    /// Fan normalization. A signal that reaches a memory through a concept
    /// is divided by `fan ^ fan_exponent`, where `fan` is the number of
    /// memories the concept is attached to. `0.0` (the original behaviour)
    /// does not normalize, so a word half the store shares links as strongly
    /// as a name two memories share. `1.0` divides by the fan: what a
    /// concept passes on is split among everything it points at.
    pub fan_exponent: f32,
}

impl Default for SpreadingConfig {
    fn default() -> Self {
        Self {
            semantic_alpha: 1.0,          // Full weight for ANN candidates (floor)
            lexical_alpha: 0.3,           // Moderate BM25 boost
            decay: 0.5,                   // Decay for 1-hop graph candidates
            iterations: 1,                // 1-hop bounded expansion
            seed_hops_from: 10,           // Expand from top-10 ANN seeds
            expansion_max_candidates: 50, // Cap added candidates at 50
            concept_expansion: true,      // Concept + abstraction graph expansion on
            query_concept_boost: 0.75,    // Direct query-concept seeding at full strength
            fan_exponent: 0.0,            // No fan normalization
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
            let id = key.strip_prefix("mem:").unwrap_or(&key);
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
        self.add_memory_scoped(id, text, concepts, importance, None);
    }

    /// Insert a memory with scope-aware temporal chaining.
    pub fn add_memory_scoped(
        &mut self,
        id: &str,
        text: &str,
        concepts: &[String],
        importance: f32,
        scope: Option<&str>,
    ) {
        self.graph
            .add_memory_scoped(id, text, concepts, importance, scope);
        self.bm25.add(id, text);
    }

    /// Add concept edges to an existing memory — and nothing else. Belief
    /// revision uses this instead of the full insert path, which would
    /// duplicate edges, recount co-occurrence, and move the temporal head.
    /// See [`MemoryGraph::add_concepts_to_memory`].
    pub fn add_concepts_to_memory(
        &mut self,
        id: &str,
        concepts: &[String],
        importance: f32,
    ) -> bool {
        self.graph.add_concepts_to_memory(id, concepts, importance)
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

    /// What a signal passing through a concept is divided by: the number of
    /// memories the concept is attached to, raised to `fan_exponent`. `1.0`
    /// when fan normalization is off.
    fn fan_divisor(&self, concept_nid: u32) -> f32 {
        if self.config.fan_exponent <= 0.0 {
            return 1.0;
        }
        let fan = self.graph.neighbor_indices(concept_nid).map_or(0, |edges| {
            edges
                .iter()
                .filter(|&&idx| {
                    self.graph.get_edge(idx).is_some_and(|edge| {
                        edge.kind == EdgeKind::Association
                            && matches!(self.graph.node_kind(edge.target), Some(NodeKind::Memory))
                    })
                })
                .count()
        });
        (fan.max(1) as f32).powf(self.config.fan_exponent)
    }

    /// Bounded augmenter search: ANN candidates + 1-hop graph expansion.
    ///
    /// Core principle: the augmenter can only ADD candidates, never remove ANN
    /// hits. It returns a **pure, non-negative graph delta** per candidate — it
    /// does NOT fold cosine into the returned score. The engine owns the
    /// authoritative cosine (computed from exact f32 embeddings) and fuses it
    /// with this delta as `final = cosine + (1 - alpha) * normalized_delta`.
    /// Keeping cosine out of this delta is what lets the graph term actually
    /// re-rank — folding cosine in here (as a previous version did) made the
    /// delta a monotone function of cosine and washed the graph signal out.
    ///
    /// The cosine seed scores ARE used internally to (a) select which seeds to
    /// expand from and (b) weight the magnitude of the propagated graph signal,
    /// but they are never added to the returned value.
    ///
    /// Algorithm:
    /// 1. ANN candidates form the floor (delta seeded at 0 so they always
    ///    appear; the engine ranks them by exact cosine).
    /// 2. BM25 lexical boost contributes to the delta.
    /// 3. Single 1-hop expansion from top-M seeds (M ≈ 10), split into pools:
    ///    - Strong (Refines/Contradicts): always traversed, including to
    ///      targets already in the ANN floor. This is what lets a refinement
    ///      or correction surface above its cosine-nearest older memory.
    ///      Boosted at full strength (factor 1.0).
    ///    - Temporal: traversed to nearby-turn memories. Boosted at 0.5.
    ///    - Normal (high-weight Association): only adds candidates NOT already
    ///      in the ANN floor. Boosted at factor 0.3.
    /// 4. Union expanded candidates into the pool — never replace, never drop
    ///    ANN hits.
    ///
    /// Returns `(id, graph_delta)` pairs (delta >= 0). Returns None only if
    /// there are no ANN seeds.
    pub fn search(
        &self,
        query_text: &str,
        semantic_seeds: &[(String, f32)],
        top_k: usize,
    ) -> Option<Vec<(String, f32)>> {
        self.search_restricted(query_text, semantic_seeds, top_k, None)
    }

    /// [`search`](Self::search) restricted to the memories `allow` accepts.
    ///
    /// The graph, the concept layer and the BM25 index are shared by every
    /// scope in the store, so an unrestricted expansion reaches memories the
    /// caller's scope or filter excludes. With `allow` set, a memory it
    /// rejects is never added as a candidate, never counts toward the
    /// expansion cap, and never influences the lexical normalization. The
    /// seeds are assumed to be allowed already (they come from a scoped ANN
    /// search).
    pub fn search_restricted(
        &self,
        query_text: &str,
        semantic_seeds: &[(String, f32)],
        top_k: usize,
        allow: Option<&dyn Fn(&str) -> bool>,
    ) -> Option<Vec<(String, f32)>> {
        use std::collections::HashMap;

        let allowed = |id: &str| allow.is_none_or(|f| f(id));

        // Step 1: ANN candidates are the floor — we never drop them. `delta`
        // holds the pure graph signal (cosine is NOT folded in); `seed_scores`
        // holds the seed activation (semantic_alpha * cosine) used only to
        // select and weight graph expansion.
        let mut delta: HashMap<String, f32> = HashMap::with_capacity(semantic_seeds.len() * 2);
        let mut seed_scores: HashMap<String, f32> = HashMap::with_capacity(semantic_seeds.len());
        let mut ann_candidates: std::collections::HashSet<String> =
            std::collections::HashSet::with_capacity(semantic_seeds.len());

        for (id, score) in semantic_seeds {
            let key = format!("mem:{id}");
            // Floor candidate: delta starts at 0 so the engine ranks it by
            // exact cosine unless the graph adds a boost.
            delta.insert(key.clone(), 0.0);
            seed_scores.insert(key.clone(), self.config.semantic_alpha * score.max(0.0));
            ann_candidates.insert(key);
        }

        if delta.is_empty() {
            return None;
        }

        // Step 2: BM25 lexical boost contributes to the graph delta.
        let mut lexical = self.bm25.score(query_text);
        lexical.retain(|(id, _)| allowed(id));
        let max_lex = lexical.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let lex_norm = if max_lex > 0.0 { 1.0 / max_lex } else { 0.0 };
        for (id, score) in lexical {
            let key = format!("mem:{id}");
            let lex_boost = self.config.lexical_alpha * score * lex_norm;
            *delta.entry(key).or_insert(0.0) += lex_boost;
        }

        // Step 3: 1-hop graph expansion from top-M seeds
        // Only expand if we have graph edges and iterations > 0
        if self.config.iterations > 0 {
            // Pick top-M seeds by cosine activation (M = seed_hops_from,
            // default 10). Seeds are chosen by `seed_scores` (semantic_alpha *
            // cosine), NOT by the graph delta — the strongest ANN matches are
            // the right anchors to expand from.
            let seed_hops_from = self.config.seed_hops_from.clamp(1, 20);
            let mut seed_list: Vec<(String, f32)> = seed_scores
                .iter()
                .filter(|(k, _)| k.starts_with("mem:"))
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            // Ties broken by id so the same query always expands from the
            // same seeds (the map this list came from has no stable order).
            seed_list.sort_by(|a, b| {
                turbomemory_core::cmp_score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0))
            });
            seed_list.truncate(seed_hops_from);

            // Collect all memory nodes reachable in 1 hop from seeds.
            //
            // Three expansion pools prevent semantically strong edges from
            // being drowned out by the much larger pool of Association edges:
            // - `strong`: Refines/Contradicts targets. These are NOT skipped
            //   even when they are already ANN candidates, because the whole
            //   point of a Refines/Contradicts edge is to re-rank a memory
            //   that ANN found but ranked too low. They receive a full-strength
            //   boost (factor 1.0).
            // - `temporal`: Temporal (sequential) edges to nearby memories.
            //   These help with temporal-reasoning queries where the answer
            //   is in a conversation turn near the seed. Boosted at 0.5.
            // - `normal`: high-weight Association targets that are NOT already
            //   ANN candidates. These add recall; they receive a capped boost
            //   (factor 0.3) so the graph cannot flood the result set.
            let mut strong: HashMap<String, f32> = HashMap::new();
            let mut temporal: HashMap<String, f32> = HashMap::new();
            let mut normal: HashMap<String, f32> = HashMap::new();
            let mut abstraction: HashMap<String, f32> = HashMap::new();
            let max_candidates = self.config.expansion_max_candidates.clamp(10, 200);
            let decay = self.config.decay;

            // Direct query concept seeding: bridge abstract queries directly to concept hubs
            let q_tokens: Vec<&str> = query_text
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .filter(|w| w.len() >= 3)
                .collect();
            for &token in &q_tokens {
                let token_lower = token.to_lowercase();
                let ckey = format!("concept:{token_lower}");
                if let Some(cnid) = self.graph.get_node_id(&ckey) {
                    if !self.graph.is_concept_suppressed(&token_lower)
                        && self.config.query_concept_boost > 0.0
                    {
                        let seed_weight = self.config.query_concept_boost / self.fan_divisor(cnid);
                        if let Some(cedges) = self.graph.neighbor_indices(cnid) {
                            for &cidx in cedges {
                                if let Some(cedge) = self.graph.get_edge(cidx) {
                                    if cedge.kind == EdgeKind::Association
                                        && matches!(
                                            self.graph.node_kind(cedge.target),
                                            Some(NodeKind::Memory)
                                        )
                                    {
                                        if let Some(mid) = self
                                            .graph
                                            .node_external_id(cedge.target)
                                            .filter(|mid| allowed(mid))
                                        {
                                            let key = format!("mem:{mid}");
                                            *normal.entry(key).or_insert(0.0) +=
                                                seed_weight * cedge.weight;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            'seeds: for (seed_key, seed_score) in seed_list {
                let seed_id = seed_key.strip_prefix("mem:").unwrap_or(&seed_key);
                let Some(seed_nid) = self.graph.memory_node_id(seed_id) else {
                    continue;
                };
                let Some(seed_edges) = self.graph.neighbor_indices(seed_nid) else {
                    continue;
                };

                for &edge_idx in seed_edges {
                    let Some(edge) = self.graph.get_edge(edge_idx) else {
                        continue;
                    };
                    let target_kind = self.graph.node_kind(edge.target);

                    match edge.kind {
                        // mem -> mem belief edge: always traverse, full boost.
                        EdgeKind::Refines | EdgeKind::Contradicts
                            if matches!(target_kind, Some(NodeKind::Memory)) =>
                        {
                            if let Some(tid) = self
                                .graph
                                .node_external_id(edge.target)
                                .filter(|tid| allowed(tid))
                            {
                                let signal = seed_score * edge.weight * decay;
                                *strong.entry(format!("mem:{tid}")).or_insert(0.0) += signal;
                            }
                        }
                        // mem -> mem sequential edge.
                        EdgeKind::Temporal if matches!(target_kind, Some(NodeKind::Memory)) => {
                            if let Some(tid) = self
                                .graph
                                .node_external_id(edge.target)
                                .filter(|tid| allowed(tid))
                            {
                                let signal = seed_score * edge.weight * decay;
                                *temporal.entry(format!("mem:{tid}")).or_insert(0.0) += signal;
                            }
                        }
                        // mem -> concept: spread through the concept layer so a
                        // query can reach memories that share a concept (2 hops)
                        // or a sibling concept via the abstraction parent (4 hops)
                        // despite low cosine to the query. Without this the concept
                        // graph and abstraction hierarchy never influence ranking.
                        EdgeKind::Association
                            if self.config.concept_expansion
                                && matches!(target_kind, Some(NodeKind::Concept)) =>
                        {
                            let concept_nid = edge.target;
                            let concept_name =
                                self.graph.node_external_id(concept_nid).unwrap_or("");
                            if self.graph.is_concept_suppressed(concept_name) {
                                continue;
                            }
                            // What this concept passes on, split by its fan.
                            let concept_w = edge.weight / self.fan_divisor(concept_nid);
                            let Some(concept_edges) = self.graph.neighbor_indices(concept_nid)
                            else {
                                continue;
                            };
                            for &cidx in concept_edges {
                                let Some(cedge) = self.graph.get_edge(cidx) else {
                                    continue;
                                };
                                let ckind = self.graph.node_kind(cedge.target);
                                if cedge.kind == EdgeKind::Association
                                    && matches!(ckind, Some(NodeKind::Memory))
                                    && cedge.target != seed_nid
                                {
                                    // concept -> sibling memory (2 hops).
                                    if let Some(mid) = self
                                        .graph
                                        .node_external_id(cedge.target)
                                        .filter(|mid| allowed(mid))
                                    {
                                        let key = format!("mem:{mid}");
                                        let signal = seed_score * concept_w * cedge.weight * decay;
                                        *normal.entry(key).or_insert(0.0) += signal;
                                    }
                                } else if cedge.kind == EdgeKind::Abstraction
                                    && matches!(ckind, Some(NodeKind::Concept))
                                {
                                    // concept -> parent -> sibling concept -> memory.
                                    let parent_nid = cedge.target;
                                    let Some(parent_edges) =
                                        self.graph.neighbor_indices(parent_nid)
                                    else {
                                        continue;
                                    };
                                    for &pidx in parent_edges {
                                        let Some(pedge) = self.graph.get_edge(pidx) else {
                                            continue;
                                        };
                                        if pedge.kind != EdgeKind::Abstraction
                                            || pedge.target == concept_nid
                                            || !matches!(
                                                self.graph.node_kind(pedge.target),
                                                Some(NodeKind::Concept)
                                            )
                                        {
                                            continue;
                                        }
                                        let sib_nid = pedge.target;
                                        let sib_name =
                                            self.graph.node_external_id(sib_nid).unwrap_or("");
                                        if self.graph.is_concept_suppressed(sib_name) {
                                            continue;
                                        }
                                        let Some(sib_edges) = self.graph.neighbor_indices(sib_nid)
                                        else {
                                            continue;
                                        };
                                        for &sidx in sib_edges {
                                            let Some(sedge) = self.graph.get_edge(sidx) else {
                                                continue;
                                            };
                                            if sedge.kind != EdgeKind::Association
                                                || !matches!(
                                                    self.graph.node_kind(sedge.target),
                                                    Some(NodeKind::Memory)
                                                )
                                            {
                                                continue;
                                            }
                                            if let Some(mid) = self
                                                .graph
                                                .node_external_id(sedge.target)
                                                .filter(|mid| allowed(mid))
                                            {
                                                let key = format!("mem:{mid}");
                                                let signal = seed_score
                                                    * concept_w
                                                    * cedge.weight
                                                    * pedge.weight
                                                    * sedge.weight
                                                    * decay
                                                    / self.fan_divisor(sib_nid);
                                                *abstraction.entry(key).or_insert(0.0) += signal;
                                            }
                                        }
                                    }
                                }
                                if strong.len() + temporal.len() + normal.len() + abstraction.len()
                                    >= max_candidates
                                {
                                    break 'seeds;
                                }
                            }
                        }
                        _ => {}
                    }
                    if strong.len() + temporal.len() + normal.len() + abstraction.len()
                        >= max_candidates
                    {
                        break 'seeds;
                    }
                }
            }

            // Merge the pools into the graph delta with edge-kind-aware boost
            // factors. Strong (Refines/Contradicts) full strength; Temporal 0.5;
            // Association-mediated 2-hop concept spread 0.75; Abstraction (4-hop,
            // through the parent) 0.75 so a sibling-concept memory the query cannot
            // reach by cosine can still surface.
            for (key, signal) in strong {
                if signal > 0.01 {
                    *delta.entry(key).or_insert(0.0) += signal;
                }
            }
            for (key, signal) in temporal {
                let boosted = signal * 0.5;
                if boosted > 0.01 {
                    *delta.entry(key).or_insert(0.0) += boosted;
                }
            }
            for (key, signal) in normal {
                let boosted = signal * 0.75;
                if boosted > 0.01 {
                    *delta.entry(key).or_insert(0.0) += boosted;
                }
            }
            for (key, signal) in abstraction {
                let boosted = signal * 0.75;
                if boosted > 0.01 {
                    *delta.entry(key).or_insert(0.0) += boosted;
                }
            }
        }

        // Step 4: Build the candidate list. The returned value is the PURE
        // graph delta (>= 0); the engine fuses it with exact cosine. Every ANN
        // seed is returned whatever its delta: that is the floor. What is left
        // of `top_k` goes to the graph-discovered candidates with the largest
        // delta. (Ranking both kinds together by cosine + delta, as this once
        // did, let lexical matches crowd seeds out: in a store of a few
        // thousand memories hundreds share a word with the query, each with a
        // delta larger than a seed's cosine.)
        let mut floor: Vec<(String, f32)> = Vec::with_capacity(seed_scores.len());
        let mut discovered: Vec<(String, f32)> = Vec::new();
        for (key, d) in delta {
            let Some(id) = key.strip_prefix("mem:") else {
                continue;
            };
            if seed_scores.contains_key(&key) {
                floor.push((id.to_string(), d));
            } else {
                discovered.push((id.to_string(), d));
            }
        }
        // Ties broken by id: `delta` is a hash map, so without this the
        // candidates that survive the cut would differ from call to call.
        let by_delta = |a: &(String, f32), b: &(String, f32)| {
            turbomemory_core::cmp_score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0))
        };
        discovered.sort_by(by_delta);
        discovered.truncate(top_k.saturating_sub(floor.len()));
        floor.sort_by(by_delta);
        floor.extend(discovered);

        if floor.is_empty() {
            None
        } else {
            Some(floor)
        }
    }

    pub fn graph(&self) -> &MemoryGraph {
        &self.graph
    }

    /// Access the learned concept vocabulary.
    pub fn vocab(&self) -> &crate::extract::ConceptVocabulary {
        self.graph.vocab()
    }

    /// Mutable access to the learned concept vocabulary.
    pub fn vocab_mut(&mut self) -> &mut crate::extract::ConceptVocabulary {
        self.graph.vocab_mut()
    }

    /// Evolve the concept vocabulary: merge synonyms and suppress over-general
    /// hubs. See [`MemoryGraph::evolve_vocabulary`].
    pub fn evolve_vocabulary(
        &mut self,
        overlap_threshold: f32,
        hub_fraction: f32,
        max_pairs: usize,
    ) -> crate::graph::VocabularyEvolutionStats {
        self.graph
            .evolve_vocabulary(overlap_threshold, hub_fraction, max_pairs)
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

    /// Six memories share the word "garden"; two of them also share the name
    /// "rex". Unnormalized, the common word links as strongly as the name;
    /// with fan normalization the name links more strongly.
    fn hub_and_name() -> MemoryGraph {
        let mut graph = MemoryGraph::new();
        // Only the memory added right after a seed is its temporal neighbour;
        // the tests compare memories that are not.
        graph.add_memory(
            "seed",
            "rex digs in the garden",
            &["rex".into(), "garden".into()],
        );
        for id in ["hub0", "hub1", "named", "hub2", "hub3", "hub4"] {
            if id == "named" {
                graph.add_memory(id, "rex was at the vet", &["rex".into()]);
            } else {
                graph.add_memory(id, "the garden needs water", &["garden".into()]);
            }
        }
        graph
    }

    fn delta_of(results: &[(String, f32)], id: &str) -> f32 {
        results
            .iter()
            .find(|(r, _)| r == id)
            .map_or(0.0, |(_, d)| *d)
    }

    #[test]
    fn fan_normalization_favours_the_rare_concept() {
        let seeds = [("seed".to_string(), 0.9f32)];
        // "walk" is in no memory: no lexical or query-concept signal.
        let plain = SpreadingActivation::new(hub_and_name(), SpreadingConfig::default());
        let results = plain.search("walk", &seeds, 10).unwrap();
        assert!(delta_of(&results, "named") > 0.0);
        assert!((delta_of(&results, "named") - delta_of(&results, "hub3")).abs() < 1e-6);

        let config = SpreadingConfig {
            fan_exponent: 1.0,
            ..SpreadingConfig::default()
        };
        let normalized = SpreadingActivation::new(hub_and_name(), config);
        let results = normalized.search("walk", &seeds, 10).unwrap();
        // "rex" has a fan of 2, "garden" of 6.
        assert!(delta_of(&results, "named") > 2.5 * delta_of(&results, "hub3"));
        let unnormalized = plain.search("walk", &seeds, 10).unwrap();
        assert!(delta_of(&results, "named") < delta_of(&unnormalized, "named"));
    }

    #[test]
    fn lexical_matches_never_crowd_out_an_ann_seed() {
        // Forty memories repeat the query's word; the two seeds do not.
        let mut graph = MemoryGraph::new();
        graph.add_memory("seed_a", "the cat sat on the mat", &["cat".into()]);
        graph.add_memory("seed_b", "a dog barked at night", &["dog".into()]);
        for i in 0..40 {
            graph.add_memory(
                &format!("word{i:02}"),
                "garden garden garden",
                &["garden".into()],
            );
        }
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());
        let seeds = [
            ("seed_a".to_string(), 0.4f32),
            ("seed_b".to_string(), 0.35f32),
        ];
        let results = sa.search("garden", &seeds, 10).unwrap();
        assert_eq!(results.len(), 10);
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(
            ids.contains(&"seed_a") && ids.contains(&"seed_b"),
            "seeds are the floor"
        );
        assert_eq!(delta_of(&results, "seed_a"), 0.0);
        // The other eight places go to the strongest graph candidates.
        assert_eq!(ids.iter().filter(|id| id.starts_with("word")).count(), 8);
        // More seeds than places: every seed is still returned.
        let results = sa.search("garden", &seeds, 1).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn query_concept_seeding_can_be_switched_off() {
        let seeds = [("named".to_string(), 0.9f32)];
        let on = SpreadingActivation::new(hub_and_name(), SpreadingConfig::default());
        let config = SpreadingConfig {
            query_concept_boost: 0.0,
            ..SpreadingConfig::default()
        };
        let off = SpreadingActivation::new(hub_and_name(), config);
        // "garden" is a concept of hub0 and a word of the query. hub0 shares
        // no concept with the seed, so only the lexical boost is left.
        let with_seeding = delta_of(&on.search("garden", &seeds, 10).unwrap(), "hub0");
        let without = delta_of(&off.search("garden", &seeds, 10).unwrap(), "hub0");
        assert!(with_seeding > without + 0.5);
        assert!(without > 0.0 && without <= 0.3 + 1e-6);
    }

    #[test]
    fn returns_none_without_ann_seeds() {
        let mut graph = MemoryGraph::new();
        graph.add_memory("m1", "Rust is fast", &["rust".into()]);
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());
        // No ANN seeds means no recall floor, so the augmenter has nothing to
        // build on and returns None.
        let results = sa.search("banana chocolate cookie recipe", &[], 2);
        assert!(results.is_none());
    }

    #[test]
    fn augmenter_surfaces_refined_memory_without_dropping_ann_hit() {
        let mut graph = MemoryGraph::new();
        graph.add_memory("old", "Rust uses garbage collection", &["rust".into()]);
        graph.add_memory("new", "Rust uses ownership not GC", &["rust".into()]);
        // new refines old; the augmenter should reach `new` from the `old`
        // ANN seed via the Refines edge.
        graph.add_refinement("old", "new", 1.0);
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());

        // Only `old` is an ANN seed; `new` is reachable only through the graph.
        let results = sa.search("rust", &[("old".into(), 0.9)], 5).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"old"), "ANN hit must never be dropped");
        assert!(ids.contains(&"new"), "refined memory must be surfaced");
    }

    #[test]
    fn returned_score_is_pure_graph_delta_without_cosine() {
        // The augmenter must return ONLY the graph delta — cosine must NOT be
        // folded in. An ANN seed with a high cosine score but no lexical match
        // and no graph edges must come back with delta 0, while a memory reached
        // through a Refines edge must have a strictly positive delta.
        let mut graph = MemoryGraph::new();
        graph.add_memory("old", "alpha beta gamma", &["topic".into()]);
        graph.add_memory("new", "alpha beta gamma delta", &["topic".into()]);
        graph.add_memory("iso", "completely unrelated wording", &["other".into()]);
        graph.add_refinement("old", "new", 1.0);
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());

        // `iso` has a high cosine seed (0.95) but the query text shares no
        // tokens with it and it has no in-edges, so its delta must be 0.
        let results = sa
            .search(
                "alpha beta gamma",
                &[("old".into(), 0.9), ("iso".into(), 0.95)],
                5,
            )
            .unwrap();
        let delta = |id: &str| results.iter().find(|(k, _)| k == id).map(|(_, d)| *d);

        assert_eq!(
            delta("iso"),
            Some(0.0),
            "a high-cosine ANN seed with no graph/lexical signal must have delta 0 (cosine not folded in)"
        );
        let new_delta = delta("new").expect("refined memory must be present");
        assert!(
            new_delta > 0.0,
            "Refines target must receive a positive graph delta, got {new_delta}"
        );
    }

    #[test]
    fn augmenter_reaches_sibling_concept_via_abstraction() {
        let mut graph = MemoryGraph::new();
        // Co-occurring memories tagged [a, b] build the abstraction parent a+b.
        for i in 0..3 {
            graph.add_memory(&format!("co{i}"), "co occur", &["a".into(), "b".into()]);
        }
        assert!(
            graph.build_abstractions(3) >= 2,
            "abstraction parent should form"
        );
        // Anchor memory tagged only with the query concept `a` (the ANN seed).
        graph.add_memory("anchor", "anchor note", &["a".into()]);
        // Target tagged only with the SIBLING concept `b`; not an ANN seed and
        // shares no concept with the anchor directly. Reachable only through
        // a -> (a+b parent) -> b -> target.
        graph.add_memory("target", "target note", &["b".into()]);
        let sa = SpreadingActivation::new(graph, SpreadingConfig::default());

        let results = sa.search("a", &[("anchor".into(), 0.9)], 10).unwrap();
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"anchor"), "ANN seed must be preserved");
        assert!(
            ids.contains(&"target"),
            "sibling-concept memory must surface via abstraction, got {ids:?}"
        );
        let target_delta = results
            .iter()
            .find(|(k, _)| k == "target")
            .map(|(_, d)| *d)
            .unwrap();
        assert!(
            target_delta > 0.0,
            "abstraction-reached target must have a positive graph delta"
        );
    }

    #[test]
    fn concept_expansion_toggle_isolates_abstraction_reachability() {
        // A/B isolation of the concept + abstraction contribution (W4). A filler
        // memory sits between `anchor` and `target` so they are NOT temporally
        // adjacent — the ONLY path from the anchor's concept `a` to `target`
        // (concept `b`) is a -> (a+b parent) -> b -> target. So `target` surfaces
        // iff concept_expansion is on; with it off only belief/temporal remain.
        let build = || {
            let mut graph = MemoryGraph::new();
            for i in 0..3 {
                graph.add_memory(&format!("co{i}"), "co occur", &["a".into(), "b".into()]);
            }
            assert!(graph.build_abstractions(3) >= 2);
            graph.add_memory("anchor", "anchor note", &["a".into()]);
            graph.add_memory("filler", "filler note", &["z".into()]); // breaks temporal adjacency
            graph.add_memory("target", "target note", &["b".into()]);
            graph
        };

        // ON: the sibling-concept target surfaces via the abstraction bridge.
        let sa_on = SpreadingActivation::new(build(), SpreadingConfig::default());
        let on_ids: Vec<String> = sa_on
            .search("a", &[("anchor".into(), 0.9)], 10)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(
            on_ids.iter().any(|id| id == "target"),
            "concept_expansion ON: target must surface via abstraction, got {on_ids:?}"
        );

        // OFF: with concept expansion disabled, target is unreachable (no belief
        // edge, and the filler broke the temporal path).
        let cfg_off = SpreadingConfig {
            concept_expansion: false,
            ..SpreadingConfig::default()
        };
        let sa_off = SpreadingActivation::new(build(), cfg_off);
        let off_ids: Vec<String> = sa_off
            .search("a", &[("anchor".into(), 0.9)], 10)
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        assert!(
            off_ids.iter().any(|id| id == "anchor"),
            "ANN seed preserved"
        );
        assert!(
            !off_ids.iter().any(|id| id == "target"),
            "concept_expansion OFF: target must NOT surface, got {off_ids:?}"
        );
    }
}
