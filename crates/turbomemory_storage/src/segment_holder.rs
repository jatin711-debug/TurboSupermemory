//! Owns the tiered vector segments and decides when to roll them.

use crate::config::{StoreConfig, Tier};
use crate::metadata_store::MetadataStore;
use crate::record::{PointOffset, Record};
use crate::segments::cold::ColdSegment;
use crate::segments::hot::HotSegment;
use crate::segments::warm::WarmSegment;
use crate::segments::{merge_candidates, ScoredPoint, VectorSegment};
use std::path::{Path, PathBuf};

/// Owns Hot/Warm/Cold segments for a single collection.
pub struct SegmentHolder {
    config: StoreConfig,
    hot: HotSegment,
    warm: Vec<WarmSegment>,
    cold: Vec<ColdSegment>,
    next_segment_id: u64,
    base_path: PathBuf,
}

impl SegmentHolder {
    pub fn new(config: StoreConfig, base_path: impl AsRef<Path>) -> crate::Result<Self> {
        let base_path = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base_path)?;
        Ok(Self {
            hot: HotSegment::new(&config)?,
            warm: Vec::new(),
            cold: Vec::new(),
            next_segment_id: 0,
            base_path,
            config,
        })
    }

    pub fn from_records(
        config: StoreConfig,
        base_path: impl AsRef<Path>,
        records: &[(PointOffset, Record)],
        meta: &MetadataStore,
    ) -> crate::Result<Self> {
        let mut holder = Self::new(config, base_path)?;
        for (offset, rec) in records {
            holder.insert(*offset, rec, meta)?;
        }
        Ok(holder)
    }

    fn segment_path(&mut self, tier: Tier) -> PathBuf {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        self.base_path
            .join(tier.name())
            .join(format!("segment_{id}.bin"))
    }

    pub fn insert(
        &mut self,
        offset: PointOffset,
        record: &Record,
        meta: &MetadataStore,
    ) -> crate::Result<()> {
        self.hot.insert(offset, record)?;
        if self.hot.point_count() >= self.config.tier.hot_capacity {
            self.seal_hot(meta)?;
        }
        Ok(())
    }

    pub fn search(
        &self,
        query: &[f32],
        top_k: usize,
        meta: &MetadataStore,
    ) -> crate::Result<Vec<ScoredPoint>> {
        let pool_k = top_k.saturating_mul(4).max(top_k);
        let mut lists = Vec::with_capacity(2 + self.warm.len() + self.cold.len());
        lists.push(self.hot.search(query, pool_k, meta)?);
        for warm in &self.warm {
            lists.push(warm.search(query, pool_k, meta)?);
        }
        for cold in &self.cold {
            lists.push(cold.search(query, pool_k, meta)?);
        }
        let candidates = merge_candidates(lists, pool_k);

        // Final rerank with full f32 embeddings.
        let mut reranked: Vec<ScoredPoint> = candidates
            .into_iter()
            .filter_map(|c| {
                meta.get(c.offset).ok().flatten().map(|rec| ScoredPoint {
                    offset: c.offset,
                    score: turbomemory_core::cosine_similarity(query, rec.embedding_f32()),
                    tier: c.tier,
                })
            })
            .collect();
        reranked.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        reranked.truncate(top_k);
        Ok(reranked)
    }

    /// Move the current Hot segment into a Warm segment and create a fresh Hot.
    pub fn seal_hot(&mut self, meta: &MetadataStore) -> crate::Result<()> {
        // Collect the records currently in the Hot segment.
        let offsets: Vec<PointOffset> = self
            .hot
            .search(
                &vec![0.0f32; self.config.dimension],
                self.hot.point_count(),
                meta,
            )?
            .into_iter()
            .map(|c| c.offset)
            .collect();
        if offsets.is_empty() {
            return Ok(());
        }
        let records: Vec<(PointOffset, Record)> = offsets
            .into_iter()
            .filter_map(|off| meta.get(off).ok().flatten().map(|r| (off, r)))
            .collect();
        if records.is_empty() {
            return Ok(());
        }
        let path = self.segment_path(Tier::Warm);
        let warm = WarmSegment::from_records(&path, &records, self.config.tier.warm_bits)?;
        self.warm.push(warm);
        self.hot = HotSegment::new(&self.config)?;
        // Roll warm to cold if over capacity.
        self.compact_warm(meta)?;
        Ok(())
    }

    /// If total warm records exceed the warm capacity, merge all warm segments
    /// into a single Cold segment.
    fn compact_warm(&mut self, meta: &MetadataStore) -> crate::Result<()> {
        let total_warm: usize = self.warm.iter().map(|s| s.point_count()).sum();
        if total_warm <= self.config.tier.warm_capacity || self.warm.is_empty() {
            return Ok(());
        }
        let mut records: Vec<(PointOffset, Record)> = Vec::with_capacity(total_warm);
        for warm in &self.warm {
            for &offset in warm.offsets() {
                if let Some(rec) = meta.get(offset)? {
                    records.push((offset, rec));
                }
            }
        }
        if !records.is_empty() {
            let path = self.segment_path(Tier::Cold);
            let cold = ColdSegment::from_records(&path, &records)?;
            self.cold.push(cold);
        }
        self.warm.clear();
        Ok(())
    }

    pub fn point_count(&self) -> usize {
        self.hot.point_count()
            + self.warm.iter().map(|s| s.point_count()).sum::<usize>()
            + self.cold.iter().map(|s| s.point_count()).sum::<usize>()
    }

    pub fn memory_bytes(&self) -> usize {
        self.hot.memory_bytes()
            + self.warm.iter().map(|s| s.memory_bytes()).sum::<usize>()
            + self.cold.iter().map(|s| s.memory_bytes()).sum::<usize>()
    }

    pub fn flush(&self) -> crate::Result<()> {
        self.hot.flusher()()?;
        for warm in &self.warm {
            (warm.flusher())()?;
        }
        for cold in &self.cold {
            (cold.flusher())()?;
        }
        Ok(())
    }

    pub fn trigger_consolidation(&mut self, meta: &MetadataStore) -> crate::Result<(usize, usize)> {
        let mut sealed = 0usize;
        let mut compacted = 0usize;
        if self.hot.point_count() >= self.config.tier.hot_capacity {
            self.seal_hot(meta)?;
            sealed += 1;
        }
        let total_warm: usize = self.warm.iter().map(|s| s.point_count()).sum();
        if total_warm > self.config.tier.warm_capacity {
            // Placeholder: future work merges warm segments to cold.
            compacted += 1;
        }
        Ok((sealed, compacted))
    }
}
