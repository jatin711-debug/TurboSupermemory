//! Belief revision: refinement / contradiction detection (propose), bounded
//! demotion (commit), and resolving a memory to its current belief.

use super::StorageEngine;
use crate::record::PointOffset;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use turbomemory_core::cosine_similarity;

/// A superseded memory's demotion never compounds below this floor, no matter
/// how many times it is flagged. Bounds the blast radius of any false-positive
/// supersession so belief revision can never bury a memory below 30% of score.
const SUPERSESSION_DEMOTION_FLOOR: f32 = 0.3;

impl StorageEngine {
    /// Memory evolution: detect when a newer memory refines an older one
    /// and create `Refines` edges (old → new) so retrieval surfaces the
    /// most current version.
    ///
    /// For each pair of memories where:
    /// - cosine similarity >= `refinement_cosine_threshold`
    /// - they share at least one concept
    /// - the newer one has a higher `insert_seq`
    ///
    /// a `Refines` edge is created from the older memory to the newer one.
    /// The older memory is NOT deleted — it stays in the graph so the agent
    /// can reason about how its understanding evolved. The newer memory
    /// also inherits the older one's unique concepts (so it's discoverable
    /// through the same concept paths).
    ///
    /// Opt-in: a no-op when `refinement_cosine_threshold` is `None`. Work
    /// is bounded by `refinement_max_pairs_per_cycle`. Candidate pairs are
    /// found via the existing ANN index (no O(n²) scan).
    ///
    /// Returns the number of new `Refines` edges created.
    pub fn check_refinements(&self) -> crate::Result<usize> {
        let props = self.propose_refinements()?;
        self.commit_supersessions(&props)
    }

    /// Detection half of refinement belief-revision: find newer memories that
    /// re-state older ones (mutual-nearest-neighbour + text-overlap + role/scope
    /// gates) and return them as [`ProposedSupersession`]s **without** mutating
    /// the graph or demoting anything. `commit_supersessions` applies the result.
    /// This split lets a verifier vet each demotion before it happens (W3).
    pub fn propose_refinements(&self) -> crate::Result<Vec<ProposedSupersession>> {
        let Some(threshold) = self.config.tier.refinement_cosine_threshold else {
            return Ok(Vec::new());
        };
        let max_pairs = self.config.tier.refinement_max_pairs_per_cycle;
        if max_pairs == 0 {
            return Ok(Vec::new());
        }
        let text_floor = self.config.tier.refinement_text_threshold;
        let allowed_roles = self.config.tier.belief_source_roles.as_deref();
        // Incremental (W7): only process records inserted since the last
        // consolidation. Cands are sorted newest-first, so we break once we
        // reach the watermark. `0` (full scan) when the flag is off.
        let watermark = if self.config.tier.incremental_supersession_detection {
            self.supersession_watermark.load(Ordering::Relaxed)
        } else {
            0
        };

        self.access_counters.drain_into(&self.meta)?;

        // Snapshot live records sorted by insert_seq (newest first, so we
        // process the most recent refinements first).
        struct Cand {
            id: String,
            offset: PointOffset,
            insert_seq: u64,
            concepts: Vec<String>,
            text: String,
            scope: Option<String>,
            source_role: Option<String>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                insert_seq: rec.insert_seq,
                concepts: rec.concepts.clone(),
                text: rec.text.clone(),
                scope: rec.scope.clone(),
                source_role: rec.source_role.clone(),
            });
        })?;
        cands.sort_by_key(|c| std::cmp::Reverse(c.insert_seq));

        let mut proposed: Vec<ProposedSupersession> = Vec::new();

        for cand in &cands {
            if proposed.len() >= max_pairs {
                break;
            }
            // Incremental: cands are newest-first, so once we reach an already-
            // processed record (below the watermark) every remaining one is too.
            if cand.insert_seq < watermark {
                break;
            }
            // Role gate: when belief_source_roles is set, only memories whose
            // provenance role is in the list can create supersessions.
            if !role_allowed(&cand.source_role, allowed_roles) {
                continue;
            }
            // Get this record's embedding for ANN search.
            let view = self.vectors.read_view();
            let Some(vec) = view.get(cand.offset) else {
                continue;
            };
            let embedding: Vec<f32> = vec.to_vec();
            drop(view);

            // Forward: cand's single nearest OLDER same-concept, same-scope
            // neighbour above the cosine floor. One supersession edge per new
            // memory (not one per qualifying neighbour, which over-fired).
            let Some(other_id) = self.nearest_superseding_neighbor(
                &embedding,
                &cand.concepts,
                &cand.scope,
                &cand.id,
                cand.insert_seq,
                false,
                threshold,
                allowed_roles,
            )?
            else {
                continue;
            };
            let Some(other) = cands.iter().find(|c| c.id == other_id) else {
                continue;
            };

            // A refinement is a *re-statement* of the same claim: require high
            // text overlap so two coexisting facts are not treated as one.
            if turbomemory_graph::text_jaccard_similarity(&cand.text, &other.text) < text_floor {
                continue;
            }

            // Mutual-nearest gate: `other`'s nearest NEWER same-concept,
            // same-scope neighbour must be `cand`. This scale-free check is what
            // stops the over-firing that made belief revision net-negative on
            // real conversational data.
            let view = self.vectors.read_view();
            let Some(ovec) = view.get(other.offset) else {
                continue;
            };
            let other_emb: Vec<f32> = ovec.to_vec();
            drop(view);
            let back = self.nearest_superseding_neighbor(
                &other_emb,
                &other.concepts,
                &other.scope,
                &other.id,
                other.insert_seq,
                true,
                threshold,
                allowed_roles,
            )?;
            if back.as_deref() != Some(cand.id.as_str()) {
                continue;
            }

            proposed.push(ProposedSupersession {
                old_id: other.id.clone(),
                new_id: cand.id.clone(),
                old_offset: other.offset,
                new_offset: cand.offset,
                kind: SupersessionKind::Refinement,
                cosine: cosine_similarity(&embedding, &other_emb),
            });
        }

        Ok(proposed)
    }

    /// Commit half of belief-revision: for each proposed supersession, create
    /// the `Refines`/`Contradicts` edge (old → new), apply bounded demotion to
    /// the older memory, and (for refinements) transfer the older memory's
    /// unique concepts to the newer one. Idempotent per pair (a duplicate edge
    /// is skipped and does not re-demote). Returns the number of edges created.
    ///
    /// Splitting this from `propose_*` is the verification seam (W3): a caller
    /// can filter the proposed list (e.g. through an NLI cross-encoder) before
    /// committing, so a demotion only happens once it is semantically confirmed.
    pub fn commit_supersessions(&self, proposed: &[ProposedSupersession]) -> crate::Result<usize> {
        let demotion_factor = self.config.tier.supersession_demotion_factor;
        let weaken_factor = self.config.tier.contradiction_weaken_factor;
        let mut created = 0usize;

        for p in proposed {
            let added = {
                let mut graph = self.graph.write();
                match p.kind {
                    SupersessionKind::Refinement => graph.add_refinement(&p.old_id, &p.new_id, 0.8),
                    SupersessionKind::Contradiction => {
                        graph.add_contradiction(&p.old_id, &p.new_id, 0.8, weaken_factor)
                    }
                }
            };
            if !added {
                continue;
            }
            created += 1;

            // Bounded, persisted demotion of the superseded memory.
            let cur = self.meta.demotion_factor(p.old_offset);
            self.meta.set_demotion_factor(
                p.old_offset,
                (cur * demotion_factor).max(SUPERSESSION_DEMOTION_FLOOR),
            );

            // Refinements also transfer the older memory's unique concepts to
            // the newer one so it is discoverable through the same concept paths.
            if p.kind == SupersessionKind::Refinement {
                self.transfer_unique_concepts(p.old_offset, p.new_offset, &p.new_id)?;
            }
        }

        if created > 0 {
            self.save_graph()?;
        }
        Ok(created)
    }

    /// Commit supersessions identified only by `(old_id, new_id, kind)` — the
    /// shape that survives a round-trip through an external (Python) verifier,
    /// which does not see internal offsets. Offsets are re-resolved from the id
    /// index at commit time (robust if the layout shifted since `propose_*`).
    /// Pairs whose ids are no longer live are silently skipped.
    pub fn commit_supersessions_by_id(
        &self,
        pairs: &[(String, String, SupersessionKind)],
    ) -> crate::Result<usize> {
        let props: Vec<ProposedSupersession> = {
            let id_index = self.id_index.read();
            pairs
                .iter()
                .filter_map(|(old_id, new_id, kind)| {
                    let old_offset = *id_index.get(old_id.as_str())?;
                    let new_offset = *id_index.get(new_id.as_str())?;
                    Some(ProposedSupersession {
                        old_id: old_id.clone(),
                        new_id: new_id.clone(),
                        old_offset,
                        new_offset,
                        kind: *kind,
                        cosine: 0.0,
                    })
                })
                .collect()
        };
        self.commit_supersessions(&props)
    }

    /// Resolve each id against the supersession graph: "what is true NOW" in
    /// place of that memory, plus its lineage. One graph read-lock for the
    /// whole batch. Usable without enabling any cognitive config flags — the
    /// graph exists regardless, so with no supersession edges every id
    /// resolves to itself (`superseded = false`, `chain = [id]`). Unknown ids
    /// also resolve to themselves.
    pub fn resolve_beliefs(&self, ids: &[String]) -> Vec<BeliefResolution> {
        let guard = self.graph.read();
        let graph = guard.graph();
        ids.iter()
            .map(|id| {
                let current_id = graph.belief_head(id);
                BeliefResolution {
                    id: id.clone(),
                    superseded: current_id != *id,
                    current_id,
                    chain: graph.belief_lineage(id),
                }
            })
            .collect()
    }

    /// Merge the older memory's concepts (that the newer one lacks) into the
    /// newer memory's concept set, updating both metadata and the graph.
    fn transfer_unique_concepts(
        &self,
        old_offset: PointOffset,
        new_offset: PointOffset,
        new_id: &str,
    ) -> crate::Result<()> {
        let (Some(old_rec), Some(new_rec)) =
            (self.meta.get(old_offset)?, self.meta.get(new_offset)?)
        else {
            return Ok(());
        };
        let unique: Vec<String> = old_rec
            .concepts
            .iter()
            .filter(|c| !new_rec.concepts.contains(c))
            .cloned()
            .collect();
        if unique.is_empty() {
            return Ok(());
        }
        let mut new_concepts = new_rec.concepts.clone();
        for c in unique {
            if !new_concepts.contains(&c) {
                new_concepts.push(c);
            }
        }
        let embedding: Vec<f32> = {
            let view = self.vectors.read_view();
            match view.get(new_offset) {
                Some(v) => v.to_vec(),
                None => return Ok(()),
            }
        };
        let mut rec = new_rec;
        rec.concepts = new_concepts;
        self.meta.put(
            new_offset,
            &rec.with_embedding(Arc::from(embedding.as_slice())),
        )?;
        let mut graph = self.graph.write();
        if let Some(r) = self.find_record_by_id(new_id) {
            // Not add_memory_scoped: the memory node already exists, and the
            // full insert path would duplicate its existing edges, recount
            // co-occurrence, and re-point the temporal chain head.
            graph.add_concepts_to_memory(new_id, &r.concepts, r.importance);
        }
        Ok(())
    }

    /// Nearest live neighbour of `embedding` (examined nearest-first via the ANN
    /// index) that shares a concept with `concepts`, is in the same `scope`, and
    /// is strictly newer (`newer=true`) or strictly older (`newer=false`) than
    /// `pivot_seq`, with cosine >= `floor`. `exclude` is skipped. Returns the
    /// neighbour's id.
    ///
    /// This is the core of **mutual-nearest-neighbour** supersession detection:
    /// a genuine "B supersedes A" pair are each other's nearest same-concept
    /// neighbour. Requiring mutuality is scale-free — independent of the absolute
    /// cosine threshold, which means different things across embedding models —
    /// and is what stops the detector from over-firing on real conversational
    /// text (where many same-topic sentences clear any fixed threshold, so the
    /// old "link every older neighbour above threshold" logic demoted ~20% of
    /// all fact pairs and made belief revision net-negative on LongMemEval).
    #[allow(clippy::too_many_arguments)]
    fn nearest_superseding_neighbor(
        &self,
        embedding: &[f32],
        concepts: &[String],
        scope: &Option<String>,
        exclude: &str,
        pivot_seq: u64,
        newer: bool,
        floor: f32,
        allowed_roles: Option<&[String]>,
    ) -> crate::Result<Option<String>> {
        let neighbors = self.search_ann_candidates(embedding, 10)?;
        let id_index = self.id_index.read();
        let view = self.vectors.read_view();
        for (nid, _) in neighbors {
            if nid == exclude {
                continue;
            }
            let Some(&off) = id_index.get(nid.as_str()) else {
                continue;
            };
            let Some(rec) = self.meta.get(off)? else {
                continue;
            };
            if newer {
                if rec.insert_seq <= pivot_seq {
                    continue;
                }
            } else if rec.insert_seq >= pivot_seq {
                continue;
            }
            if rec.scope != *scope {
                continue;
            }
            // Role gate: a neighbour can only participate in a supersession
            // (as the superseded memory) if its provenance role is allowed.
            if !role_allowed(&rec.source_role, allowed_roles) {
                continue;
            }
            if !concepts.iter().any(|c| rec.concepts.contains(c)) {
                continue;
            }
            let Some(ov) = view.get(off) else {
                continue;
            };
            if cosine_similarity(embedding, ov) < floor {
                continue;
            }
            return Ok(Some(nid));
        }
        Ok(None)
    }

    /// Contradiction detection: when a newer memory contradicts an older
    /// one (same topic, opposing content), create a `Contradicts` edge and
    /// weaken the old memory's edges so it fades from retrieval.
    ///
    /// For each pair where:
    /// - cosine similarity >= `contradiction_cosine_threshold`
    /// - they share at least one concept
    /// - text Jaccard similarity < `contradiction_text_threshold` (the
    ///   texts say different things about the same topic)
    /// - the newer one has a higher `insert_seq`
    ///
    /// a `Contradicts` edge is created (old → new) and the old memory's
    /// outgoing Association/Temporal edges are weakened by
    /// `contradiction_weaken_factor`. The old memory is NOT deleted.
    ///
    /// Opt-in: a no-op when `contradiction_cosine_threshold` is `None`.
    /// Runs after `check_refinements` so that pairs that are refinements
    /// (high text overlap) are not also flagged as contradictions.
    ///
    /// Returns the number of new `Contradicts` edges created.
    pub fn check_contradictions(&self) -> crate::Result<usize> {
        let props = self.propose_contradictions()?;
        self.commit_supersessions(&props)
    }

    /// Detection half of contradiction belief-revision: find newer memories
    /// that oppose older ones (same topic, low text overlap, opposition marker,
    /// mutual-nearest-neighbour) and return them as [`ProposedSupersession`]s
    /// **without** mutating the graph. `commit_supersessions` applies the result.
    pub fn propose_contradictions(&self) -> crate::Result<Vec<ProposedSupersession>> {
        let Some(threshold) = self.config.tier.contradiction_cosine_threshold else {
            return Ok(Vec::new());
        };
        let max_pairs = self.config.tier.contradiction_max_pairs_per_cycle;
        if max_pairs == 0 {
            return Ok(Vec::new());
        }
        let text_threshold = self.config.tier.contradiction_text_threshold;
        let require_opposition = self.config.tier.contradiction_require_opposition;
        let allowed_roles = self.config.tier.belief_source_roles.as_deref();
        let watermark = if self.config.tier.incremental_supersession_detection {
            self.supersession_watermark.load(Ordering::Relaxed)
        } else {
            0
        };

        self.access_counters.drain_into(&self.meta)?;

        // Snapshot live records sorted by insert_seq (newest first).
        struct Cand {
            id: String,
            offset: PointOffset,
            insert_seq: u64,
            text: String,
            concepts: Vec<String>,
            scope: Option<String>,
            source_role: Option<String>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                insert_seq: rec.insert_seq,
                text: rec.text.clone(),
                concepts: rec.concepts.clone(),
                scope: rec.scope.clone(),
                source_role: rec.source_role.clone(),
            });
        })?;
        cands.sort_by_key(|c| std::cmp::Reverse(c.insert_seq));

        let mut proposed: Vec<ProposedSupersession> = Vec::new();
        for cand in &cands {
            if proposed.len() >= max_pairs {
                break;
            }
            // Incremental: newest-first, so stop at the first already-checked one.
            if cand.insert_seq < watermark {
                break;
            }
            if !role_allowed(&cand.source_role, allowed_roles) {
                continue;
            }
            let view = self.vectors.read_view();
            let Some(vec) = view.get(cand.offset) else {
                continue;
            };
            let embedding: Vec<f32> = vec.to_vec();
            drop(view);

            // Forward: cand's single nearest OLDER same-concept, same-scope
            // neighbour above the cosine floor (one edge per new memory).
            let Some(other_id) = self.nearest_superseding_neighbor(
                &embedding,
                &cand.concepts,
                &cand.scope,
                &cand.id,
                cand.insert_seq,
                false,
                threshold,
                allowed_roles,
            )?
            else {
                continue;
            };
            let Some(other) = cands.iter().find(|c| c.id == other_id) else {
                continue;
            };

            // KEY DISTINGUISHER from refinement: a contradiction has LOW text
            // overlap (same topic, opposing content); high overlap is a
            // refinement (handled by check_refinements).
            let text_sim = turbomemory_graph::text_jaccard_similarity(&cand.text, &other.text);
            if text_sim >= text_threshold {
                continue;
            }
            // ...and an explicit opposition/negation marker in the newer text.
            if require_opposition && !turbomemory_graph::has_opposition_marker(&cand.text) {
                continue;
            }

            // Mutual-nearest gate (see propose_refinements): `other`'s nearest
            // NEWER same-concept, same-scope neighbour must be `cand`.
            let view = self.vectors.read_view();
            let Some(ovec) = view.get(other.offset) else {
                continue;
            };
            let other_emb: Vec<f32> = ovec.to_vec();
            drop(view);
            let back = self.nearest_superseding_neighbor(
                &other_emb,
                &other.concepts,
                &other.scope,
                &other.id,
                other.insert_seq,
                true,
                threshold,
                allowed_roles,
            )?;
            if back.as_deref() != Some(cand.id.as_str()) {
                continue;
            }

            proposed.push(ProposedSupersession {
                old_id: other.id.clone(),
                new_id: cand.id.clone(),
                old_offset: other.offset,
                new_offset: cand.offset,
                kind: SupersessionKind::Contradiction,
                cosine: cosine_similarity(&embedding, &other_emb),
            });
        }

        Ok(proposed)
    }

    /// Detection over BOTH refinement and contradiction, for the verified
    /// belief-revision path (W3): a caller runs this, vets each pair through an
    /// external verifier, then `commit_supersessions` on the survivors. When no
    /// verifier is installed, consolidation instead auto-commits via
    /// `check_refinements` + `check_contradictions` (identical detection logic).
    pub fn propose_supersessions(&self) -> crate::Result<Vec<ProposedSupersession>> {
        let mut proposed = self.propose_refinements()?;
        proposed.extend(self.propose_contradictions()?);
        Ok(proposed)
    }
}

/// The kind of supersession relationship a proposed pair represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupersessionKind {
    /// Newer memory re-states / updates the older one (high text overlap).
    Refinement,
    /// Newer memory opposes the older one (same topic, low overlap + marker).
    Contradiction,
}

impl SupersessionKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            SupersessionKind::Refinement => "refinement",
            SupersessionKind::Contradiction => "contradiction",
        }
    }

    /// Parse `"refinement"` / `"contradiction"` (case-insensitive). Any other
    /// value maps to `None`.
    pub fn from_label(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "refinement" | "refine" | "refines" => Some(SupersessionKind::Refinement),
            "contradiction" | "contra" | "contradicts" => Some(SupersessionKind::Contradiction),
            _ => None,
        }
    }
}

/// A supersession candidate produced by detection (`propose_*`) but NOT yet
/// committed to the graph. This is the seam that lets an external verifier
/// (e.g. an NLI cross-encoder) vet a demotion before it happens: detection is
/// pure (no graph mutation), and `commit_supersessions` applies the edge +
/// bounded demotion only for the pairs that survive verification.
#[derive(Debug, Clone)]
pub struct ProposedSupersession {
    pub old_id: String,
    pub new_id: String,
    pub old_offset: PointOffset,
    pub new_offset: PointOffset,
    pub kind: SupersessionKind,
    /// Cosine similarity between the two memories at detection time.
    pub cosine: f32,
}

/// The resolution of one memory id against the supersession graph — the
/// "revises beliefs" read contract: what is believed NOW in place of `id`,
/// and the lineage that led there. Produced by `StorageEngine::resolve_beliefs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeliefResolution {
    /// The queried id.
    pub id: String,
    /// Head of `id`'s supersession chain — the current belief. Equals `id`
    /// when nothing supersedes it (or the id is unknown to the graph).
    pub current_id: String,
    /// True when `id` has been superseded by a newer memory.
    pub superseded: bool,
    /// The full supersession chain containing `id`, oldest first, head last.
    /// `[id]` when `id` has no supersession edges.
    pub chain: Vec<String>,
}

/// Whether a memory's provenance `role` may participate in belief revision
/// under the `allowed` role filter (`TierConfig::belief_source_roles`).
///
/// - `allowed == None` → role-blind: every memory participates (legacy).
/// - `allowed == Some(list)` → only memories whose role is in `list`
///   participate. Unattributed memories (`role == None`) are excluded, so a
///   caller that opts into role-scoping without tagging roles simply gets no
///   supersessions rather than silently role-blind behavior.
fn role_allowed(role: &Option<String>, allowed: Option<&[String]>) -> bool {
    match allowed {
        None => true,
        Some(list) => match role {
            Some(r) => list.iter().any(|a| a == r),
            None => false,
        },
    }
}
