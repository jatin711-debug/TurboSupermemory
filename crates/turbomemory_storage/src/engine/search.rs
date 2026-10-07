//! The read path: ANN, cognitive (graph-fused), filtered, and batch search,
//! plus superseded-memory exclusion.

use super::StorageEngine;
use crate::payload_index::Filter;
use crate::record::PointOffset;
use crate::segment_holder::SegmentSnapshot;
use crate::segments::ScoredPoint;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use turbomemory_core::{cosine_similarity, validate_query};

/// For small collections an exact scan is deterministic and higher-recall than
/// a lightly-configured HNSW index.
const EXACT_FALLBACK_THRESHOLD: usize = 4096;

/// `query` scaled to unit length (all zeros if it has no direction). The GPU
/// scan computes dot products against unit-length rows, so a unit query makes
/// those the cosines the CPU paths return.
fn unit_length(query: &[f32]) -> Vec<f32> {
    let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm.is_normal() {
        query.iter().map(|x| x / norm).collect()
    } else {
        vec![0.0; query.len()]
    }
}

/// Exact-search order: best score first, the older record first on a tie, so
/// equal scores always come back in the same order.
fn best_first(a: &(PointOffset, f32), b: &(PointOffset, f32)) -> std::cmp::Ordering {
    turbomemory_core::cmp_score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0))
}

/// The best `want` of `candidates` (which must arrive in offset order), best
/// first, in one pass and without holding more than `2 * want` of them.
///
/// Candidates are kept in a small buffer that is cut back to the best `want`
/// whenever it fills; the worst score kept becomes a floor, and since every
/// later candidate is a newer record, one that only ties the floor would
/// rank after everything kept and is skipped without being stored.
///
/// `admit` is asked only about candidates that beat the floor, so a check
/// that costs a lookup (is this row still a live record?) runs for a handful
/// of rows instead of for all of them.
fn top_of(
    candidates: impl Iterator<Item = (PointOffset, f32)>,
    want: usize,
    admit: impl Fn(PointOffset) -> bool,
) -> Vec<(PointOffset, f32)> {
    if want == 0 {
        return Vec::new();
    }
    let cap = want.saturating_mul(2).max(64);
    let mut kept: Vec<(PointOffset, f32)> = Vec::new();
    let mut floor = f32::NEG_INFINITY;
    for (offset, score) in candidates {
        // Written so that a NaN score is skipped as well.
        if score.partial_cmp(&floor) != Some(std::cmp::Ordering::Greater) || !admit(offset) {
            continue;
        }
        kept.push((offset, score));
        if kept.len() >= cap {
            kept.select_nth_unstable_by(want - 1, best_first);
            kept.truncate(want);
            floor = kept[want - 1].1;
        }
    }
    kept.sort_unstable_by(best_first);
    kept.truncate(want);
    kept
}

impl StorageEngine {
    pub fn search_ann(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_with_ef(query_embedding, top_k, None)
    }

    pub fn search_ann_with_ef(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_scoped(query_embedding, top_k, ef, None)
    }

    /// ANN search restricted to a single agent scope (plus global records).
    pub fn search_ann_scoped(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Vec<(String, f32)>> {
        let candidates =
            self.search_ann_candidates_filtered_with_ef(query_embedding, top_k, None, ef, scope)?;
        Ok(candidates)
    }

    pub fn search_ann_candidates(
        &self,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered(query_embedding, top_k, None)
    }

    pub fn search_ann_candidates_with_ef(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered_with_ef(query_embedding, top_k, None, ef, None)
    }

    /// Filtered ANN candidate search.
    ///
    /// `filter` is evaluated against the payload index; the resulting offset
    /// bitmap is intersected with tiered segment search.
    pub fn search_ann_candidates_filtered(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered_with_ef(query_embedding, top_k, filter, None, None)
    }

    pub fn search_ann_candidates_filtered_with_ef(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Vec<(String, f32)>> {
        validate_query(query_embedding, self.config.dimension)?;
        let allowed = self.allowed_offsets(filter, scope)?;
        self.ann_top_k(query_embedding, top_k, ef, allowed.as_ref())
    }

    /// The offsets a payload filter and/or scope allow. `None` means the
    /// search is unrestricted; `Some` always restricts, even when empty (a
    /// filter or scope that matches nothing must return nothing).
    fn allowed_offsets(
        &self,
        filter: Option<&Filter>,
        scope: Option<&str>,
    ) -> crate::Result<Option<RoaringBitmap>> {
        let mut allowed = match filter {
            Some(f) => Some(self.evaluate_filter(f)?),
            None => None,
        };
        if let Some(s) = scope {
            let scope_bitmap = self.scope_index.read().query(Some(s));
            allowed = Some(match allowed {
                Some(existing) => existing & scope_bitmap,
                None => scope_bitmap,
            });
        }
        Ok(allowed)
    }

    /// The `top_k` nearest live records, best first.
    ///
    /// `top_k` is clamped to the number of records, so a caller-supplied
    /// value can never size an allocation.
    fn ann_top_k(
        &self,
        query: &[f32],
        top_k: usize,
        ef: Option<usize>,
        allowed: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<(String, f32)>> {
        // B1: snapshot the superseded id set once per query. `None` (flag off
        // or no supersessions) skips the exclusion entirely.
        let exclusion = self.superseded_exclusion_set();
        self.ann_top_k_excluding(query, top_k, ef, allowed, exclusion.as_ref())
    }

    /// `ann_top_k` with the ids to leave out supplied by the caller (`None`:
    /// leave nothing out).
    fn ann_top_k_excluding(
        &self,
        query: &[f32],
        top_k: usize,
        ef: Option<usize>,
        allowed: Option<&RoaringBitmap>,
        exclusion: Option<&HashSet<String>>,
    ) -> crate::Result<Vec<(String, f32)>> {
        let records = self.record_count();
        let top_k = top_k.min(records);
        if top_k == 0 {
            return Ok(Vec::new());
        }
        // Vectors resident on the GPU: the exact answer from one device
        // product, in place of the approximate tiered search below.
        if records >= self.config.tier.gpu_exact_min_records {
            if let Some(gpu) = self.gpu_exact() {
                if let Some(scores) = gpu.scores(&self.vectors, &unit_length(query)) {
                    return Ok(self.finish_exact(
                        |want| self.top_of_scores(&scores, allowed, want),
                        top_k,
                        exclusion,
                    ));
                }
            }
        }
        if records <= EXACT_FALLBACK_THRESHOLD {
            let scored = self.exact_candidates(query, allowed);
            return Ok(self.finish_exact(
                |want| top_of(scored.iter().copied(), want, |_| true),
                top_k,
                exclusion,
            ));
        }
        let snapshot = self.segment_snapshot.load_full();
        let fetch = Self::first_fetch(exclusion, top_k);
        let scored = snapshot.search(query, fetch, ef, &self.vectors, allowed)?;
        self.live_top_k(
            &snapshot, scored, fetch, query, top_k, ef, allowed, exclusion,
        )
    }

    /// Reduce a segment search result to the `top_k` live, non-superseded
    /// records, searching again with a wider pool while that leaves a
    /// shortfall the index could still fill.
    ///
    /// Immutable segments keep the offsets of records that were deleted,
    /// updated, or evicted until they are rebuilt, and superseded records are
    /// still indexed. Both are dropped here, after the segment search has
    /// already cut its results to `fetch`; without the retry, deleting the
    /// ten best matches for a query made that query return nothing.
    #[allow(clippy::too_many_arguments)]
    fn live_top_k(
        &self,
        snapshot: &SegmentSnapshot,
        mut scored: Vec<ScoredPoint>,
        mut fetch: usize,
        query: &[f32],
        top_k: usize,
        ef: Option<usize>,
        allowed: Option<&RoaringBitmap>,
        exclusion: Option<&HashSet<String>>,
    ) -> crate::Result<Vec<(String, f32)>> {
        let total = snapshot.point_count();
        loop {
            let mut live: Vec<(PointOffset, String, f32)> = Vec::with_capacity(scored.len());
            for c in &scored {
                let Some(meta_rec) = self.meta.get(c.offset)? else {
                    continue;
                };
                if exclusion.is_some_and(|set| set.contains(&meta_rec.id)) {
                    continue;
                }
                live.push((c.offset, meta_rec.id, c.score));
            }
            let exhausted = scored.len() < fetch || fetch >= total;
            if live.len() >= top_k || exhausted {
                live.truncate(top_k);
                let mut results = Vec::with_capacity(live.len());
                for (offset, id, score) in live {
                    self.bump_access(offset);
                    results.push((id, score));
                }
                return Ok(results);
            }
            fetch = fetch.saturating_mul(2).min(total);
            scored = snapshot.search(query, fetch, ef, &self.vectors, allowed)?;
        }
    }

    /// Batched search for M queries. With the vectors resident on the GPU the
    /// whole batch is one device matrix product (exact). Otherwise each
    /// query's HNSW traversal runs on the CPU and, when CUDA is available,
    /// the candidates of all queries are reranked in one `gemm`
    /// (`search_gpu_batch`). Returns one result list per query, each sorted
    /// by score desc and truncated to `top_k`.
    ///
    /// Filter and scope apply identically to every query in the batch.
    pub fn search_ann_batch(
        &self,
        queries: &[&[f32]],
        top_k: usize,
        ef: Option<usize>,
        filter: Option<&Filter>,
        scope: Option<&str>,
    ) -> crate::Result<Vec<Vec<(String, f32)>>> {
        let m = queries.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        for q in queries {
            validate_query(q, self.config.dimension)?;
        }
        let allowed = self.allowed_offsets(filter, scope)?;
        let records = self.record_count();
        let top_k = top_k.min(records);
        if top_k == 0 {
            return Ok(vec![Vec::new(); m]);
        }

        if records >= self.config.tier.gpu_exact_min_records {
            if let Some(gpu) = self.gpu_exact() {
                let exclusion = self.superseded_exclusion_set();
                let units: Vec<Vec<f32>> = queries.iter().map(|q| unit_length(q)).collect();
                let results = gpu.scores_batch(&self.vectors, &units, |scores| {
                    self.finish_exact(
                        |want| self.top_of_scores(scores, allowed.as_ref(), want),
                        top_k,
                        exclusion.as_ref(),
                    )
                });
                if let Some(results) = results {
                    return Ok(results);
                }
            }
        }

        // Small collection: an exact scan per query (cheap), which is also
        // what the single-query path does.
        if records <= EXACT_FALLBACK_THRESHOLD {
            return queries
                .iter()
                .map(|q| self.ann_top_k(q, top_k, ef, allowed.as_ref()))
                .collect();
        }

        // Large collection: batched snapshot search (GPU gemm rerank). B1:
        // superseded exclusion applies identically to every query in the
        // batch, from one snapshot of the superseded set.
        let exclusion = self.superseded_exclusion_set();
        let fetch = Self::first_fetch(exclusion.as_ref(), top_k);
        let snapshot = self.segment_snapshot.load_full();
        let gpu = self.gpu_backend();
        let batch_scored = snapshot.search_gpu_batch(
            queries,
            fetch,
            ef,
            &self.vectors,
            allowed.as_ref(),
            Some(&gpu),
        )?;

        let mut out = Vec::with_capacity(m);
        for (q, scored) in queries.iter().zip(batch_scored) {
            out.push(self.live_top_k(
                &snapshot,
                scored,
                fetch,
                q,
                top_k,
                ef,
                allowed.as_ref(),
                exclusion.as_ref(),
            )?);
        }
        Ok(out)
    }

    /// Exact CPU scan: the cosine of `query` with every live record that
    /// `allowed` admits, as `(offset, score)` in offset order.
    fn exact_candidates(
        &self,
        query: &[f32],
        allowed: Option<&RoaringBitmap>,
    ) -> Vec<(PointOffset, f32)> {
        let view = self.vectors.read_view();
        let index = self.payload_index.read();
        let live = index.all_offsets();
        let mut scored: Vec<(PointOffset, f32)> = Vec::with_capacity(live.len() as usize);
        let mut score = |offset: u32| {
            if let Some(v) = view.get(offset as PointOffset) {
                scored.push((offset as PointOffset, cosine_similarity(query, v)));
            }
        };
        match allowed {
            // `Some` always filters, even when empty: a filter/scope that
            // resolves to zero offsets must match NOTHING.
            Some(bitmap) => bitmap
                .iter()
                .filter(|offset| live.contains(*offset))
                .for_each(&mut score),
            None => live.iter().for_each(&mut score),
        }
        scored
    }

    /// The best `want` live records out of the scores the GPU computed for
    /// every vector (`scores[i]` belongs to offset `i`). Deleted records and
    /// anything outside `allowed` are masked here, on the host.
    fn top_of_scores(
        &self,
        scores: &[f32],
        allowed: Option<&RoaringBitmap>,
        want: usize,
    ) -> Vec<(PointOffset, f32)> {
        let index = self.payload_index.read();
        let live = index.all_offsets();
        let is_live = |offset: PointOffset| live.contains(offset as u32);
        match allowed {
            // `Some` always filters, even when empty: a filter/scope that
            // resolves to zero offsets must match NOTHING. A record inserted
            // after the scores were computed has no score yet; it is simply
            // not a candidate for this query.
            Some(bitmap) => top_of(
                bitmap.iter().filter_map(|offset| {
                    scores
                        .get(offset as usize)
                        .map(|score| (offset as PointOffset, score.clamp(-1.0, 1.0)))
                }),
                want,
                is_live,
            ),
            // Unrestricted: a straight pass over the score row.
            None => top_of(
                scores
                    .iter()
                    .enumerate()
                    .map(|(offset, score)| (offset as PointOffset, score.clamp(-1.0, 1.0))),
                want,
                is_live,
            ),
        }
    }

    /// Turn the head of an exact ranking into the `top_k` results, as
    /// `(id, score)` with superseded ids dropped.
    ///
    /// `top(want)` returns the best `want` candidates, best first (fewer when
    /// there are no more). Ids are looked up only for those, and the head is
    /// asked for again, wider, when superseded or just-deleted records took
    /// slots that live results should have had.
    fn finish_exact(
        &self,
        top: impl Fn(usize) -> Vec<(PointOffset, f32)>,
        top_k: usize,
        exclusion: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        let mut want = Self::first_fetch(exclusion, top_k);
        loop {
            let head = top(want);
            let exhausted = head.len() < want;
            let mut results: Vec<(PointOffset, String, f32)> = Vec::with_capacity(top_k);
            for (offset, score) in head {
                // A record deleted since the candidates were collected has
                // no id any more.
                let Some(id) = self.meta.id_of(offset) else {
                    continue;
                };
                if exclusion.is_some_and(|set| set.contains(&id)) {
                    continue;
                }
                results.push((offset, id, score));
                if results.len() == top_k {
                    break;
                }
            }
            if results.len() == top_k || exhausted {
                return results
                    .into_iter()
                    .map(|(offset, id, score)| {
                        self.bump_access(offset);
                        (id, score)
                    })
                    .collect();
            }
            want = want.saturating_mul(2);
        }
    }

    /// Snapshot of the graph's superseded-id set for one query (B1: the A-TMA
    /// "ghost memory" fix — memories on the old side of a Refines/Contradicts
    /// edge are dropped from results instead of merely rank-demoted). Read-locks
    /// the graph exactly once. Returns `None` when exclusion is disabled or the
    /// graph has no supersessions, so the common case pays nothing and behaves
    /// exactly as before.
    fn superseded_exclusion_set(&self) -> Option<HashSet<String>> {
        if !self.config.tier.exclude_superseded {
            return None;
        }
        let set: HashSet<String> = self
            .graph
            .read()
            .graph()
            .superseded_ids()
            .into_iter()
            .collect();
        if set.is_empty() {
            None
        } else {
            Some(set)
        }
    }

    /// Size of the first segment search. With superseded exclusion active it
    /// over-fetches `top_k * 2 + 5` (the pool size proven in the B1 eval
    /// adapter), so dropping the stale ids usually still fills `top_k` in one
    /// pass; `live_top_k` widens it when that is not enough.
    fn first_fetch(exclusion: Option<&HashSet<String>>, top_k: usize) -> usize {
        if exclusion.is_some() {
            top_k.saturating_mul(2).saturating_add(5)
        } else {
            top_k
        }
    }

    pub fn search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_with_ef(query_text, query_embedding, top_k, None)
    }

    /// Cognitive search restricted to a single agent scope (plus global records).
    pub fn search_scoped(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_scoped_with_ef(query_text, query_embedding, top_k, None, scope)
    }

    /// Cognitive search with an explicit `ef` and optional agent scope.
    pub fn search_scoped_with_ef(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_with_ef_scoped(query_text, query_embedding, top_k, ef, scope)
    }

    /// Hydrate augmenter results with embeddings and additively fuse the graph
    /// boost with cosine similarity to produce the final ranking.
    ///
    /// `final_score = cosine + (1 - alpha) * saturating_graph_delta`, where
    /// `saturating_graph_delta = act / (1 + act)` — an **absolute** transform of
    /// the candidate's own graph signal, NOT a result-set-relative
    /// `act / max_act`. The old max normalization shrank a cosine-far but
    /// graph-reached memory's boost below the noise whenever some other
    /// candidate had a large delta (and inflated a weak incidental signal when
    /// every delta was small). The saturating form fixes both.
    ///
    /// `results` carries the augmenter's **pure graph delta** (>= 0, cosine is
    /// NOT folded in). The boost is additive, so it can re-order candidates and
    /// surface graph-discovered ones but never drops an ANN hit — preserving
    /// the recall floor. `alpha` controls how much the graph may nudge the
    /// ranking: `1.0` = pure cosine (graph only decides which candidates
    /// exist); lower values give the graph delta more of a vote.
    ///
    /// `exclusion` is the per-query superseded-id snapshot (B1): when `Some`,
    /// those ids are dropped before the top-k truncation (the callers already
    /// over-fetch the graph candidate pool to compensate).
    ///
    /// `allowed` is the scope/filter bitmap of the query. The graph is shared
    /// by every scope, so each candidate is checked against it here as well:
    /// nothing outside the caller's scope or filter can be returned, whatever
    /// the expansion produced.
    fn hydrate_and_fuse(
        &self,
        results: Vec<(String, f32)>,
        query_embedding: &[f32],
        top_k: usize,
        exclusion: Option<&HashSet<String>>,
        allowed: Option<&RoaringBitmap>,
    ) -> crate::Result<Vec<(String, f32)>> {
        if results.is_empty() {
            return Ok(Vec::new());
        }

        let alpha = self.config.cognitive_alpha.clamp(0.0, 1.0);
        let temporal_recency_weight = self.config.tier.temporal_recency_weight.clamp(0.0, 2.0);
        let max_seq = if temporal_recency_weight > 0.0 {
            results
                .iter()
                .filter_map(|(id, _)| self.find_meta_by_id(id).map(|m| m.insert_seq))
                .max()
                .unwrap_or(1)
        } else {
            1
        };

        let mut hydrated: Vec<(String, f32)> = results
            .into_iter()
            .filter_map(|(id, act)| {
                let offset = self.id_index.read().get(id.as_str()).copied()?;
                if allowed.is_some_and(|bitmap| !bitmap.contains(offset as u32)) {
                    return None;
                }
                self.get_record(offset).map(|rec| {
                    let cos = cosine_similarity(query_embedding, rec.embedding_f32());
                    // Absolute, saturating graph boost: `act / (1 + act)` depends
                    // on the candidate's OWN graph signal, not the result-set
                    // maximum.
                    let graph_boost = (1.0 - alpha) * (act / (1.0 + act));
                    let base_fused = cos + graph_boost;

                    let recency_boost = if temporal_recency_weight > 0.0 && max_seq > 0 {
                        let seq = self.find_meta_by_id(&id).map(|m| m.insert_seq).unwrap_or(0);
                        1.0 + temporal_recency_weight * (seq as f32 / max_seq as f32)
                    } else {
                        1.0
                    };
                    let fused = base_fused * recency_boost;

                    // Supersession demotion: a memory superseded by a newer one
                    // carries a persisted factor < 1.0.
                    let demotion = self.meta.demotion_factor(offset);
                    (id, fused * demotion)
                })
            })
            .collect();
        // B1: drop superseded memories entirely rather than merely demoting
        // them. No-op when `exclusion` is `None` (flag off / no supersessions).
        if let Some(set) = exclusion {
            hydrated.retain(|(id, _)| !set.contains(id));
        }
        // Ties broken by id so equal scores come back in a stable order.
        hydrated
            .sort_by(|a, b| turbomemory_core::cmp_score_desc(a.1, b.1).then_with(|| a.0.cmp(&b.0)));
        hydrated.truncate(top_k);
        Ok(hydrated)
    }

    pub fn search_with_ef(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_with_ef_scoped(query_text, query_embedding, top_k, ef, None)
    }

    fn search_with_ef_scoped(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.cognitive_search(query_text, query_embedding, top_k, None, ef, scope)
    }

    pub fn search_ann_filtered(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
    ) -> crate::Result<Vec<(String, f32)>> {
        self.search_ann_candidates_filtered(query_embedding, top_k, Some(filter))
    }

    /// Cognitive search with a payload filter.
    pub fn search_filtered(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_filtered_with_ef(query_text, query_embedding, top_k, filter, None)
    }

    pub fn search_filtered_with_ef(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
        ef: Option<usize>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.search_filtered_with_scope(query_text, query_embedding, top_k, filter, ef, None)
    }

    /// Cognitive search with both a payload filter and an agent scope.
    pub fn search_filtered_with_scope(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: &Filter,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        self.cognitive_search(query_text, query_embedding, top_k, Some(filter), ef, scope)
    }

    /// Cognitive search: ANN seeds, graph and lexical expansion, then fusion
    /// with exact cosine. The filter and scope bound every stage, not only
    /// the seeds: the expansion walks a graph and a lexical index shared by
    /// all scopes, so it is given the same restriction, and the fusion step
    /// checks each candidate once more.
    fn cognitive_search(
        &self,
        query_text: &str,
        query_embedding: &[f32],
        top_k: usize,
        filter: Option<&Filter>,
        ef: Option<usize>,
        scope: Option<&str>,
    ) -> crate::Result<Option<Vec<(String, f32)>>> {
        validate_query(query_embedding, self.config.dimension)?;
        let top_k = top_k.min(self.record_count());
        let allowed = self.allowed_offsets(filter, scope)?;
        let seeds = self.ann_top_k(query_embedding, top_k.max(10), ef, allowed.as_ref())?;

        let graph = self.graph.read();
        // B1: snapshot the superseded id set while the graph read lock is
        // already held (once per query). `None` when exclusion is disabled
        // or the graph has no supersessions — the zero-cost fast path.
        let exclusion: Option<HashSet<String>> = if self.config.tier.exclude_superseded {
            let set: HashSet<String> = graph.graph().superseded_ids().into_iter().collect();
            if set.is_empty() {
                None
            } else {
                Some(set)
            }
        } else {
            None
        };
        // Request more candidates from the graph than the final top_k so
        // that memories reached through multi-hop traversal (abstraction
        // edges, refinement edges) have a chance to be in the candidate
        // set even if their graph activation is lower than direct matches.
        // The fusion step (hydrate_and_fuse) will then re-rank using the
        // combination of cosine + graph activation and truncate to top_k.
        // With superseded exclusion active, over-fetch at least the
        // B1-proven `top_k * 2 + 5` pool so dropping stale ids still
        // leaves enough candidates to fill top_k.
        let mut graph_k = top_k.saturating_mul(3).max(top_k.saturating_add(5));
        if exclusion.is_some() {
            graph_k = graph_k.max(top_k.saturating_mul(2).saturating_add(5));
        }
        let activated = match &allowed {
            Some(bitmap) => {
                let in_scope = |id: &str| {
                    self.id_index
                        .read()
                        .get(id)
                        .is_some_and(|offset| bitmap.contains(*offset as u32))
                };
                graph.search_restricted(query_text, &seeds, graph_k, Some(&in_scope))
            }
            None => graph.search(query_text, &seeds, graph_k),
        };
        drop(graph);
        let Some(results) = activated else {
            return Ok(None);
        };
        let hydrated = self.hydrate_and_fuse(
            results,
            query_embedding,
            top_k,
            exclusion.as_ref(),
            allowed.as_ref(),
        )?;
        if hydrated.is_empty() {
            return Ok(None);
        }
        for (id, _) in &hydrated {
            self.bump_access_by_id(id);
        }
        // One graph write lock for the whole batch of hits (rehearsal),
        // not one per hit.
        let hit_ids: Vec<&str> = hydrated.iter().map(|(id, _)| id.as_str()).collect();
        self.reinforce_graph_ids(&hit_ids);
        Ok(Some(hydrated))
    }

    /// Evaluate a filter against the payload and full-text indexes.
    fn evaluate_filter(&self, filter: &Filter) -> crate::Result<RoaringBitmap> {
        if filter.uses_full_text() {
            // Tantivy writes are deferred; make them visible before querying,
            // but only if there are actually pending documents.
            self.text_index.commit_if_pending()?;
        }
        self.evaluate_filter_recursive(filter)
    }

    fn evaluate_filter_recursive(&self, filter: &Filter) -> crate::Result<RoaringBitmap> {
        use crate::payload_index::Filter as F;
        Ok(match filter {
            F::FullText { query, .. } => self.text_index.search(query)?,
            F::Eq { .. } | F::Range { .. } => self.payload_index.read().query(filter),
            F::And(parts) => {
                let mut iter = parts.iter();
                let Some(first) = iter.next() else {
                    return Ok(RoaringBitmap::new());
                };
                let mut acc = self.evaluate_filter_recursive(first)?;
                for part in iter {
                    if acc.is_empty() {
                        break;
                    }
                    acc &= self.evaluate_filter_recursive(part)?;
                }
                acc
            }
            F::Or(parts) => {
                let mut acc = RoaringBitmap::new();
                for part in parts {
                    acc |= self.evaluate_filter_recursive(part)?;
                }
                acc
            }
            F::Not(inner) => {
                let positives = self.evaluate_filter_recursive(inner)?;
                self.payload_index.read().all_offsets() - &positives
            }
        })
    }
}
