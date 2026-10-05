//! The read path: ANN, cognitive (graph-fused), filtered, and batch search,
//! plus superseded-memory exclusion.

use super::StorageEngine;
use crate::payload_index::Filter;
use roaring::RoaringBitmap;
use std::collections::HashSet;
use turbomemory_core::{cosine_similarity, validate_dimension};

/// For small collections an exact scan is deterministic and higher-recall than
/// a lightly-configured HNSW index.
const EXACT_FALLBACK_THRESHOLD: usize = 4096;

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
        validate_dimension(query_embedding, self.config.dimension)?;
        let mut allowed_offsets = match filter {
            Some(f) => Some(self.evaluate_filter(f)?),
            None => None,
        };
        if let Some(s) = scope {
            let scope_bitmap = self.scope_index.read().query(Some(s));
            allowed_offsets = Some(match allowed_offsets {
                Some(existing) => existing & scope_bitmap,
                None => scope_bitmap,
            });
        }
        // B1: with superseded exclusion enabled, snapshot the superseded id
        // set once and over-fetch so the stale ids can be dropped while still
        // filling top_k. `None` (flag off / no supersessions) keeps fetch_k
        // == top_k and skips the filter entirely.
        let exclusion = self.superseded_exclusion_set();
        let fetch_k = Self::exclusion_fetch_k(&exclusion, top_k);
        if self.record_count() <= EXACT_FALLBACK_THRESHOLD {
            let mut results = match &allowed_offsets {
                Some(bitmap) => self.exact_top_k_filtered(query_embedding, fetch_k, Some(bitmap)),
                None => self.exact_top_k(query_embedding, fetch_k),
            };
            Self::apply_superseded_exclusion(&mut results, top_k, exclusion.as_ref());
            for (id, _) in &results {
                self.bump_access_by_id(id);
            }
            return Ok(results);
        }
        let snapshot = self.segment_snapshot.load_full();
        let gpu = self.gpu_backend();
        let scored = snapshot.search_gpu(
            query_embedding,
            fetch_k,
            ef,
            &self.vectors,
            allowed_offsets.as_ref(),
            Some(&gpu),
        )?;
        let mut results = Vec::with_capacity(scored.len());
        for c in scored {
            if let Some(meta_rec) = self.meta.get(c.offset)? {
                if let Some(set) = &exclusion {
                    if set.contains(&meta_rec.id) {
                        continue;
                    }
                }
                self.bump_access(c.offset);
                results.push((meta_rec.id, c.score));
            }
        }
        results.truncate(top_k);
        Ok(results)
    }

    /// Batched ANN search for M queries. Runs each query's HNSW traversal on
    /// CPU, then reranks all queries' candidate lists in a single GPU `gemm`
    /// when CUDA is available (`search_gpu_batch`), which is the workload
    /// where GPU genuinely beats CPU. Returns one result list per query, each
    /// sorted by score desc and truncated to `top_k`.
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
            validate_dimension(q, self.config.dimension)?;
        }
        let mut allowed_offsets = match filter {
            Some(f) => Some(self.evaluate_filter(f)?),
            None => None,
        };
        if let Some(s) = scope {
            let scope_bitmap = self.scope_index.read().query(Some(s));
            allowed_offsets = Some(match allowed_offsets {
                Some(existing) => existing & scope_bitmap,
                None => scope_bitmap,
            });
        }
        // B1: superseded exclusion applies identically to every query in the
        // batch — one snapshot for the whole batch, same over-fetch pool.
        let exclusion = self.superseded_exclusion_set();
        let fetch_k = Self::exclusion_fetch_k(&exclusion, top_k);

        // Small collection: batch the exact scan (per-query, but cheap).
        if self.record_count() <= EXACT_FALLBACK_THRESHOLD {
            let mut out = Vec::with_capacity(m);
            for q in queries {
                let mut results = match &allowed_offsets {
                    Some(bitmap) => self.exact_top_k_filtered(q, fetch_k, Some(bitmap)),
                    None => self.exact_top_k(q, fetch_k),
                };
                Self::apply_superseded_exclusion(&mut results, top_k, exclusion.as_ref());
                for (id, _) in &results {
                    self.bump_access_by_id(id);
                }
                out.push(results);
            }
            return Ok(out);
        }

        // Large collection: batched snapshot search (GPU gemm rerank).
        let snapshot = self.segment_snapshot.load_full();
        let gpu = self.gpu_backend();
        let batch_scored = snapshot.search_gpu_batch(
            queries,
            fetch_k,
            ef,
            &self.vectors,
            allowed_offsets.as_ref(),
            Some(&gpu),
        )?;

        // Map offsets → ids per query and bump access counters.
        let mut out = Vec::with_capacity(m);
        for scored in batch_scored {
            let mut results = Vec::with_capacity(scored.len());
            for c in scored {
                if let Some(meta_rec) = self.meta.get(c.offset)? {
                    if let Some(set) = &exclusion {
                        if set.contains(&meta_rec.id) {
                            continue;
                        }
                    }
                    self.bump_access(c.offset);
                    results.push((meta_rec.id, c.score));
                }
            }
            results.truncate(top_k);
            out.push(results);
        }
        Ok(out)
    }

    fn exact_top_k(&self, query: &[f32], top_k: usize) -> Vec<(String, f32)> {
        self.exact_top_k_filtered(query, top_k, None)
    }

    fn exact_top_k_filtered(
        &self,
        query: &[f32],
        top_k: usize,
        allowed_offsets: Option<&RoaringBitmap>,
    ) -> Vec<(String, f32)> {
        let view = self.vectors.read_view();
        let mut all: Vec<(String, f32)> = Vec::new();
        let _ = self.meta.for_each_record(|offset, rec| {
            // `Some` always filters, even when empty: a filter/scope that
            // resolves to zero offsets must match NOTHING (previously an empty
            // bitmap was treated as "unfiltered", leaking other scopes).
            if let Some(bitmap) = allowed_offsets {
                if !bitmap.contains(offset as u32) {
                    return;
                }
            }
            if let Some(v) = view.get(offset) {
                let score = cosine_similarity(query, v);
                all.push((rec.id.clone(), score));
            }
        });
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(top_k);
        all
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

    /// Candidate-pool size used when superseded exclusion is active: over-fetch
    /// `top_k * 2 + 5` so that dropping the stale ids still leaves enough
    /// candidates to fill `top_k` (the pool size proven in the B1 eval adapter).
    fn exclusion_fetch_k(exclusion: &Option<HashSet<String>>, top_k: usize) -> usize {
        if exclusion.is_some() {
            top_k * 2 + 5
        } else {
            top_k
        }
    }

    /// Drop superseded ids from an over-fetched result list and truncate back
    /// to `top_k`. No-op when `exclusion` is `None` (flag off or no
    /// supersessions), preserving the exact prior behavior.
    fn apply_superseded_exclusion(
        results: &mut Vec<(String, f32)>,
        top_k: usize,
        exclusion: Option<&HashSet<String>>,
    ) {
        if let Some(set) = exclusion {
            results.retain(|(id, _)| !set.contains(id));
            results.truncate(top_k);
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
    fn hydrate_and_fuse(
        &self,
        results: Vec<(String, f32)>,
        query_embedding: &[f32],
        top_k: usize,
        exclusion: Option<&HashSet<String>>,
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
                self.find_record_by_id(&id).map(|rec| {
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
                    let demotion = self
                        .id_index
                        .read()
                        .get(id.as_str())
                        .map(|&offset| self.meta.demotion_factor(offset))
                        .unwrap_or(crate::metadata_store::NO_DEMOTION);
                    (id, fused * demotion)
                })
            })
            .collect();
        // B1: drop superseded memories entirely rather than merely demoting
        // them. No-op when `exclusion` is `None` (flag off / no supersessions).
        if let Some(set) = exclusion {
            hydrated.retain(|(id, _)| !set.contains(id));
        }
        hydrated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
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
        validate_dimension(query_embedding, self.config.dimension)?;

        let seeds = self.search_ann_candidates_filtered_with_ef(
            query_embedding,
            top_k.max(10),
            None,
            ef,
            scope,
        )?;

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
        let mut graph_k = (top_k * 3).max(top_k + 5);
        if exclusion.is_some() {
            graph_k = graph_k.max(top_k * 2 + 5);
        }
        let activated = graph.search(query_text, &seeds, graph_k);
        drop(graph);
        if let Some(results) = activated {
            let hydrated =
                self.hydrate_and_fuse(results, query_embedding, top_k, exclusion.as_ref())?;
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
        } else {
            Ok(None)
        }
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
        validate_dimension(query_embedding, self.config.dimension)?;
        let seeds = self.search_ann_candidates_filtered_with_ef(
            query_embedding,
            top_k.max(10),
            Some(filter),
            ef,
            scope,
        )?;
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
        let mut graph_k = (top_k * 3).max(top_k + 5);
        if exclusion.is_some() {
            graph_k = graph_k.max(top_k * 2 + 5);
        }
        let activated = graph.search(query_text, &seeds, graph_k);
        drop(graph);
        if let Some(results) = activated {
            let hydrated =
                self.hydrate_and_fuse(results, query_embedding, top_k, exclusion.as_ref())?;
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
        } else {
            Ok(None)
        }
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
