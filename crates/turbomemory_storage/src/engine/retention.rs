//! Bounded-storage maintenance: eviction (access-aware, ACT-R, or FIFO),
//! gist-before-evict, semantic dedup, and automatic importance scoring.

use super::{now_secs, GistCompressor, StorageEngine};
use crate::record::{MetaRecord, PointOffset};
use std::collections::HashSet;
use std::sync::atomic::Ordering;
use turbomemory_core::cosine_similarity;

impl StorageEngine {
    /// Bounded-storage eviction: drop the lowest-salience records when the
    /// collection exceeds `max_records` or when a record's `access_score`
    /// falls below `evict_score_floor`. Returns the number of records evicted.
    ///
    /// Both triggers are opt-in; when neither is configured this is a no-op.
    /// Salience is the same recency-weighted `access_score` used for promotion,
    /// or ACT-R base-level activation when `actr_activation` is enabled.
    /// A grace period protects freshly inserted records that simply have not
    /// been queried yet from being evicted on their first consolidation.
    pub fn evict(&self) -> crate::Result<usize> {
        let tier = &self.config.tier;
        let max_records = tier.max_records;
        let floor = tier.evict_score_floor;
        if max_records.is_none() && floor.is_none() {
            return Ok(0);
        }

        // Make access counts current before scoring.
        self.access_counters.drain_into(&self.meta)?;

        let now = now_secs();
        let half_life = tier.recency_half_life_secs.max(1);
        let access_aware = tier.access_aware_eviction;
        // ACT-R base-level activation (opt-in) replaces the legacy
        // access_count × recency heuristic for eviction ranking only;
        // promotion scoring (`segment_holder::promote_hot`) always stays on
        // the legacy path.
        let use_actr = tier.actr_activation;
        let actr_decay = tier.actr_decay.max(0.0);
        // Grace window: never evict a record whose last access is more recent
        // than this. Guards against evicting a just-rehearsed record before it
        // has been queried. Only meaningful for access-aware eviction; the naive
        // FIFO baseline ignores access entirely. Same semantics under ACT-R.
        let grace = half_life / 8;

        // Snapshot (offset, id, score) for every live record. Lower score =
        // evicted first. Access-aware: score = access_count × recency (a
        // rehearsed memory scores high and survives), or ACT-R
        // `ln(Σ age^-decay)` over the persisted access-history ring when
        // `actr_activation` is on (spaced rehearsal outweighs one recent
        // burst). FIFO baseline: score = insert_seq (oldest-inserted evicted
        // first), access ignored.
        let histories = if use_actr {
            self.meta.access_histories()
        } else {
            std::collections::HashMap::new()
        };
        let mut scored: Vec<(PointOffset, String, f64)> = Vec::new();
        let mut protected: Vec<(PointOffset, String)> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            if access_aware && now.saturating_sub(rec.last_accessed) < grace {
                protected.push((offset, rec.id.clone()));
            } else if access_aware {
                let score = if use_actr {
                    let history = histories.get(&offset).map(Vec::as_slice).unwrap_or(&[]);
                    crate::access_counters::actr_activation(history, now, actr_decay)
                } else {
                    access_score(rec, now, half_life)
                };
                scored.push((offset, rec.id.clone(), score));
            } else {
                scored.push((offset, rec.id.clone(), rec.insert_seq as f64));
            }
        })?;

        // Collect victims as (offset, id), deduplicated via a set of offsets.
        // The offset pins the exact record: an id can be updated to a newer
        // record while the (slow) gist step below runs.
        let mut victim_offsets: HashSet<PointOffset> = HashSet::new();
        let mut victims: Vec<(PointOffset, String)> = Vec::new();

        // Floor pass: anything below the score floor is a victim.
        if let Some(floor) = floor {
            for (offset, id, score) in &scored {
                if *score < floor && victim_offsets.insert(*offset) {
                    victims.push((*offset, id.clone()));
                }
            }
        }

        // Cap pass: if still over the cap, evict the lowest-scoring survivors.
        if let Some(max_records) = max_records {
            let live = self.meta.record_count();
            let mut over = live
                .saturating_sub(victims.len())
                .saturating_sub(max_records);
            if over > 0 {
                // Survivors not already marked, sorted by ascending score.
                // Equal scores (common: every never-queried record scores 0)
                // go oldest first; the metadata map has no stable order, so
                // without the tie-break the choice of victim would be random.
                let mut survivors: Vec<&(PointOffset, String, f64)> = scored
                    .iter()
                    .filter(|(offset, _, _)| !victim_offsets.contains(offset))
                    .collect();
                survivors.sort_by(|a, b| a.2.total_cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
                for (offset, id, _) in survivors {
                    if over == 0 {
                        break;
                    }
                    if victim_offsets.insert(*offset) {
                        victims.push((*offset, id.clone()));
                        over -= 1;
                    }
                }
            }
        }

        // B4 gist-before-evict: compress the victims into searchable gist
        // records BEFORE deleting them, so evicted content stays retrievable.
        // No-op unless the flag is on AND a GistCompressor is installed.
        // Victims whose gist could not be produced or stored are spared this
        // cycle: "compress instead of delete" must not quietly become
        // "delete" because the summarizer was unreachable.
        if tier.gist_before_evict && !victims.is_empty() {
            let compressor = self.gist_compressor.read().clone();
            if let Some(compressor) = compressor {
                let spared =
                    self.gist_victims(&victim_offsets, compressor.as_ref(), tier.gist_chunk_facts)?;
                if !spared.is_empty() {
                    log::warn!(
                        "gist-before-evict: kept {} records whose gist failed; \
                         they will be retried on the next eviction",
                        spared.len()
                    );
                    victims.retain(|(offset, _)| !spared.contains(offset));
                }
            }
        }

        let mut evicted = 0usize;
        for (offset, id) in &victims {
            if self.delete_by_id_at(id, *offset)? {
                evicted += 1;
            }
        }
        Ok(evicted)
    }

    /// Compress eviction victims into gist records (B4). Victims are grouped
    /// by scope (a gist never crosses scope boundaries), chunked to
    /// `chunk_facts` texts per compressor call, and each non-empty gist is
    /// inserted as an ordinary record under `source_role = "gist"` with a
    /// `{"gist": true, "victims": n}` payload. The role keeps gists out of
    /// belief-revision detection (`belief_source_roles` gating) while scope-
    /// based retrieval keeps them visible to the memories they summarize.
    ///
    /// Returns the offsets of victims that must NOT be deleted: those in a
    /// chunk whose compressor call failed or whose gist could not be stored.
    /// A chunk the compressor abstained on (`Ok(None)` or an empty gist) is
    /// not spared; abstaining is the compressor saying there is nothing to
    /// keep.
    fn gist_victims(
        &self,
        victim_offsets: &HashSet<PointOffset>,
        compressor: &dyn GistCompressor,
        chunk_facts: usize,
    ) -> crate::Result<HashSet<PointOffset>> {
        // Collect victim (scope, seq, offset, text), then compress per scope
        // in chronological (insert_seq) order — for_each_record does not
        // iterate in insertion order.
        type Victim = (u64, PointOffset, String);
        let mut by_scope: std::collections::BTreeMap<Option<String>, Vec<Victim>> =
            std::collections::BTreeMap::new();
        self.meta.for_each_record(|offset, rec| {
            if victim_offsets.contains(&offset) {
                by_scope.entry(rec.scope.clone()).or_default().push((
                    rec.insert_seq,
                    offset,
                    rec.text.clone(),
                ));
            }
        })?;

        let mut spared: HashSet<PointOffset> = HashSet::new();
        let chunk_facts = chunk_facts.max(1);
        let now = now_secs();
        for (scope, mut entries) in by_scope {
            entries.sort_by_key(|(seq, _, _)| *seq);
            for chunk in entries.chunks(chunk_facts) {
                let texts: Vec<String> = chunk.iter().map(|(_, _, text)| text.clone()).collect();
                let mut spare = |reason: &str| {
                    log::warn!(
                        "gist-before-evict: {reason}; keeping {} records",
                        chunk.len()
                    );
                    spared.extend(chunk.iter().map(|(_, offset, _)| *offset));
                };
                let (gist, embedding) = match compressor.compress(&texts) {
                    Ok(Some(out)) => out,
                    Ok(None) => continue,
                    Err(reason) => {
                        spare(&format!("compressor failed: {reason}"));
                        continue;
                    }
                };
                if gist.trim().is_empty() {
                    continue;
                }
                let n = self.gist_seq.fetch_add(1, Ordering::Relaxed) + 1;
                let gist_id = format!(
                    "gist:{}:{}:{}",
                    scope.as_deref().unwrap_or("global"),
                    now,
                    n
                );
                let payload = format!("{{\"gist\":true,\"victims\":{}}}", chunk.len());
                if let Err(e) = self.insert_with_payload_role(
                    &gist_id,
                    &gist,
                    &embedding,
                    1.0,
                    &[],
                    Some(payload),
                    scope.clone(),
                    Some("gist".to_string()),
                ) {
                    spare(&format!("could not store {gist_id}: {e}"));
                }
            }
        }
        Ok(spared)
    }

    /// Semantic consolidation: merge near-duplicate records.
    ///
    /// Two records whose cosine similarity is `>= dedup_cosine_threshold` are
    /// considered duplicates. For each duplicate pair the higher-salience
    /// record is kept (tiebreak: `importance`, then earlier `insert_seq`) and
    /// the other is deleted. The survivor inherits the victim's concept edges
    /// so graph relationships are not lost. Returns the number of records
    /// merged away.
    ///
    /// Only records in the same scope are ever merged: two users who each
    /// store the same sentence keep their own copy.
    ///
    /// Opt-in: a no-op when `dedup_cosine_threshold` is `None`. Work is bounded
    /// by `dedup_max_pairs_per_cycle`. Candidate neighbors are found via the
    /// existing ANN index (no O(n^2) scan).
    pub fn deduplicate(&self) -> crate::Result<usize> {
        let Some(threshold) = self.config.tier.dedup_cosine_threshold else {
            return Ok(0);
        };
        let max_pairs = self.config.tier.dedup_max_pairs_per_cycle;
        if max_pairs == 0 {
            return Ok(0);
        }

        self.access_counters.drain_into(&self.meta)?;
        let now = now_secs();
        let half_life = self.config.tier.recency_half_life_secs.max(1);

        // Snapshot live records: id -> (offset, score, importance, insert_seq,
        // concepts). Cloning concepts is acceptable; the set of live records is
        // bounded and this runs only on consolidation.
        struct Cand {
            offset: PointOffset,
            id: String,
            score: f64,
            importance: f32,
            insert_seq: u64,
            concepts: Vec<String>,
            scope: Option<String>,
        }
        let mut cands: Vec<Cand> = Vec::new();
        self.meta.for_each_record(|offset, rec| {
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                score: access_score(rec, now, half_life),
                importance: rec.importance,
                insert_seq: rec.insert_seq,
                concepts: rec.concepts.clone(),
                scope: rec.scope.clone(),
            });
        })?;
        // Deterministic pass order (the metadata map has none).
        cands.sort_by_key(|c| c.insert_seq);

        // Higher salience wins. Returns true if `a` should be kept over `b`.
        let keeps = |a: &Cand, b: &Cand| -> bool {
            a.score
                .total_cmp(&b.score)
                .then(a.importance.total_cmp(&b.importance))
                .then(b.insert_seq.cmp(&a.insert_seq))
                .is_ge()
        };

        let mut merged_offsets: HashSet<PointOffset> = HashSet::new();
        // (survivor_id, victim_id, victim_offset, victim_concepts)
        let mut merges: Vec<(String, String, PointOffset, Vec<String>)> = Vec::new();

        'outer: for cand in &cands {
            if merged_offsets.contains(&cand.offset) {
                continue;
            }
            let view = self.vectors.read_view();
            let Some(vec) = view.get(cand.offset) else {
                continue;
            };
            let embedding: Vec<f32> = vec.to_vec();
            drop(view);

            // Find near neighbors via ANN (within the candidate's own scope,
            // so other scopes cannot crowd the short list), then verify exact
            // cosine.
            let neighbors = self.search_ann_scoped(&embedding, 5, None, cand.scope.as_deref())?;
            for (nid, _) in neighbors {
                if nid == cand.id {
                    continue;
                }
                let Some(other) = cands.iter().find(|c| c.id == nid) else {
                    continue;
                };
                // Never merge across scopes: that would delete one user's
                // memory because another user said the same thing.
                if other.scope != cand.scope {
                    continue;
                }
                if merged_offsets.contains(&other.offset) {
                    continue;
                }
                let view = self.vectors.read_view();
                let Some(other_vec) = view.get(other.offset) else {
                    continue;
                };
                let sim = cosine_similarity(&embedding, other_vec);
                drop(view);
                if sim < threshold {
                    continue;
                }
                // Decide survivor vs victim.
                let (survivor, victim) = if keeps(cand, other) {
                    (cand, other)
                } else {
                    (other, cand)
                };
                merged_offsets.insert(victim.offset);
                merges.push((
                    survivor.id.clone(),
                    victim.id.clone(),
                    victim.offset,
                    victim.concepts.clone(),
                ));
                if merges.len() >= max_pairs {
                    break 'outer;
                }
                // `cand` may itself have become a victim; stop scanning its
                // neighbors and move to the next candidate.
                if victim.offset == cand.offset {
                    continue 'outer;
                }
            }
        }

        let mut count = 0usize;
        for (survivor_id, victim_id, victim_offset, victim_concepts) in &merges {
            // Transfer the victim's concept edges to the survivor before
            // deleting it, so relationships are preserved. Only the concepts
            // the survivor lacks are added: re-adding the whole memory would
            // duplicate every edge it already has.
            if let Some(survivor) = self.find_meta_by_id(survivor_id) {
                let missing: Vec<String> = victim_concepts
                    .iter()
                    .filter(|c| !survivor.concepts.contains(c))
                    .cloned()
                    .collect();
                if !missing.is_empty() {
                    self.graph.write().add_concepts_to_memory(
                        survivor_id,
                        &missing,
                        survivor.importance,
                    );
                }
            }
            if self.delete_by_id_at(victim_id, *victim_offset)? {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Automatic importance scoring (self-organizing memory). For each live
    /// record, compute a target importance as a blend of:
    ///   - retrieval salience: normalized `access_score` (recency-weighted
    ///     access count), and
    ///   - graph connectivity: normalized concept degree (how many distinct
    ///     concepts the memory is linked to).
    ///
    /// Then move the record's current importance `importance_learning_rate`
    /// of the way toward that target, clamped to `[floor, ceiling]`.
    ///
    /// Frequently retrieved + well-connected memories rise; never-retrieved
    /// memories decay toward the floor. The recomputed importance is written
    /// back to metadata and synced into the graph via `reweight_memory` so
    /// edge weights reflect the new importance.
    ///
    /// Opt-in: a no-op when `importance_auto_scoring` is false (the default).
    /// Returns the number of records whose importance changed by more than a
    /// small epsilon.
    pub fn recompute_importance(&self) -> crate::Result<usize> {
        if !self.config.tier.importance_auto_scoring {
            return Ok(0);
        }
        let rate = self.config.tier.importance_learning_rate.clamp(0.0, 1.0);
        let access_weight = self.config.tier.importance_access_weight.clamp(0.0, 1.0);
        let floor = self.config.tier.importance_floor;
        let ceiling = self.config.tier.importance_ceiling.max(floor);

        // Make access counts current before scoring.
        self.access_counters.drain_into(&self.meta)?;

        let now = now_secs();
        let half_life = self.config.tier.recency_half_life_secs.max(1);

        // Snapshot: offset, id, importance, salience (access_score), degree.
        struct Cand {
            offset: PointOffset,
            id: String,
            importance: f32,
            salience: f64,
            degree: usize,
        }
        let mut cands: Vec<Cand> = Vec::new();
        let mut max_salience: f64 = 0.0;
        let mut max_degree: usize = 0;
        self.meta.for_each_record(|offset, rec| {
            let salience = access_score(rec, now, half_life);
            let degree = rec.concepts.len();
            max_salience = max_salience.max(salience);
            max_degree = max_degree.max(degree);
            cands.push(Cand {
                offset,
                id: rec.id.clone(),
                importance: rec.importance,
                salience,
                degree,
            });
        })?;

        if cands.is_empty() {
            return Ok(0);
        }

        let mut changed = 0usize;
        for cand in &cands {
            // Normalize salience and degree to [0, 1].
            let sal = if max_salience > 0.0 {
                cand.salience / max_salience
            } else {
                0.0
            };
            let deg = if max_degree > 0 {
                cand.degree as f64 / max_degree as f64
            } else {
                0.0
            };
            // Target importance in [floor, ceiling]. Retrieval salience is the
            // primary driver; connectivity is a bounded boost that can lift a
            // retrieved memory but cannot, on its own, push a never-retrieved
            // memory to the ceiling. `access_weight` blends how much of the
            // band is salience-driven (the rest is a connectivity bonus scaled
            // down by salience so it only matters once a memory is being
            // retrieved). This keeps "never retrieved" decaying toward the
            // floor regardless of how many concepts it touches.
            let salience_band = sal;
            let connectivity_bonus = (1.0 - access_weight as f64) * deg * (0.5 + 0.5 * sal);
            let blend = (access_weight as f64 * salience_band + connectivity_bonus).clamp(0.0, 1.0);
            let target = floor + (ceiling - floor) * blend as f32;
            // Move a `rate` fraction of the way toward the target.
            let new_importance =
                (cand.importance + rate * (target - cand.importance)).clamp(floor, ceiling);

            if (new_importance - cand.importance).abs() <= 1e-4 {
                continue;
            }

            // Write back to metadata.
            if let Some(mut rec) = self.meta.get(cand.offset)? {
                rec.importance = new_importance;
                self.meta.put_meta(cand.offset, &rec)?;
            }
            // Sync the graph's edge weights to the new importance.
            {
                let mut graph = self.graph.write();
                graph.graph_mut().reweight_memory(&cand.id, new_importance);
            }
            changed += 1;
        }

        if changed > 0 {
            self.save_graph()?;
        }
        Ok(changed)
    }
}

/// Recency-weighted salience: `access_count * 2^(-age / half_life)`.
///
/// Mirrors `segment_holder::access_score` (kept private there); used by
/// eviction and dedup to rank records by importance.
fn access_score(meta: &MetaRecord, now: u64, half_life: u64) -> f64 {
    let age = now.saturating_sub(meta.last_accessed).max(1);
    let recency = 2.0f64.powf(-(age as f64) / half_life as f64);
    meta.access_count as f64 * recency
}
