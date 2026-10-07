use super::*;
use crate::config::TierConfig;
use crate::payload_index::Filter;

fn small_config(dim: usize) -> StoreConfig {
    StoreConfig {
        dimension: dim,
        max_edges: 3,
        level0_factor: 2,
        ef_construction: 100,
        search_list_size: 5,
        cognitive_alpha: 1.0,
        outlier_count: 0,
        initial_capacity: 16,
        tier: TierConfig {
            hot_capacity: 3,
            warm_capacity: 6,
            warm_quantizer: crate::config::QuantizerKind::Scalar { bits: 4 },
            warm_chunk_bytes: 4096,
            hnsw_threshold: 1000,
            full_scan_threshold_kb: 10_000,
            merge_threshold_segments: 2,
            merge_max_records: 200_000,
            cold_quantizer: crate::config::QuantizerKind::Sign,
            hot_promote_threshold: 2.0,
            warm_demote_threshold: 0.5,
            recency_half_life_secs: 60,
            max_records: None,
            evict_score_floor: None,
            dedup_cosine_threshold: None,
            dedup_max_pairs_per_cycle: 1024,
            abstraction_co_occurrence_threshold: 0,
            edge_decay_half_life_secs: 0,
            max_concepts: 5,
            refinement_cosine_threshold: None,
            refinement_max_pairs_per_cycle: 1024,
            contradiction_cosine_threshold: None,
            contradiction_text_threshold: 0.3,
            refinement_text_threshold: 0.25,
            contradiction_require_opposition: true,
            contradiction_weaken_factor: 0.5,
            contradiction_max_pairs_per_cycle: 1024,
            importance_auto_scoring: false,
            importance_learning_rate: 0.3,
            importance_access_weight: 0.6,
            importance_floor: 0.1,
            importance_ceiling: 4.0,
            concept_max_ngram_len: 1,
            concept_min_ngram_freq: 1,
            concept_enable_pmi: true,
            ..TierConfig::default()
        },
        optimizer_budget: crate::config::OptimizerBudget::default(),
        auto_consolidation_interval: None,
        spreading: turbomemory_graph::SpreadingConfig::default(),
    }
}

fn make_vec(dim: usize, idx: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    v[idx % dim] = 1.0;
    v
}

fn hnsw_test_config(dim: usize) -> StoreConfig {
    StoreConfig {
        dimension: dim,
        max_edges: 8,
        level0_factor: 2,
        ef_construction: 100,
        search_list_size: 10,
        cognitive_alpha: 1.0,
        outlier_count: 0,
        initial_capacity: 4096,
        tier: TierConfig {
            hot_capacity: 100,
            warm_capacity: 10_000,
            warm_quantizer: crate::config::QuantizerKind::Scalar { bits: 8 },
            warm_chunk_bytes: 4096,
            hnsw_threshold: 10,
            full_scan_threshold_kb: 10_000,
            merge_threshold_segments: 2,
            merge_max_records: 10_000,
            cold_quantizer: crate::config::QuantizerKind::Sign,
            hot_promote_threshold: 2.0,
            warm_demote_threshold: 0.5,
            recency_half_life_secs: 60,
            max_records: None,
            evict_score_floor: None,
            dedup_cosine_threshold: None,
            dedup_max_pairs_per_cycle: 1024,
            abstraction_co_occurrence_threshold: 0,
            edge_decay_half_life_secs: 0,
            max_concepts: 5,
            refinement_cosine_threshold: None,
            refinement_max_pairs_per_cycle: 1024,
            contradiction_cosine_threshold: None,
            contradiction_text_threshold: 0.3,
            refinement_text_threshold: 0.25,
            contradiction_require_opposition: true,
            contradiction_weaken_factor: 0.5,
            contradiction_max_pairs_per_cycle: 1024,
            importance_auto_scoring: false,
            importance_learning_rate: 0.3,
            importance_access_weight: 0.6,
            importance_floor: 0.1,
            importance_ceiling: 4.0,
            concept_max_ngram_len: 1,
            concept_min_ngram_freq: 1,
            concept_enable_pmi: true,
            ..TierConfig::default()
        },
        optimizer_budget: crate::config::OptimizerBudget::default(),
        auto_consolidation_interval: None,
        spreading: turbomemory_graph::SpreadingConfig::default(),
    }
}

#[test]
fn insert_and_search_ann() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    engine
        .insert(
            "m1",
            "Rust is safe",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();
    engine
        .insert(
            "m2",
            "Python is easy",
            &[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();
    let results = engine
        .search_ann(&[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
        .unwrap();
    assert_eq!(results[0].0, "m1");
}

#[test]
fn auto_extracts_concepts_when_none_provided() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    // Insert with NO caller concepts — the engine should auto-extract
    // concepts from the text.
    engine
        .insert(
            "m1",
            "Rust memory safety concurrency",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();
    // Verify the record has extracted concepts by checking the graph.
    let graph = engine.graph.read();
    let graph = graph.graph();
    // "rust", "memory", "safety", "concurrency" should all be concept nodes.
    assert!(
        graph.nodes().contains_key("concept:rust"),
        "auto-extracted concept 'rust' should be a graph node"
    );
    assert!(
        graph.nodes().contains_key("concept:safety"),
        "auto-extracted concept 'safety' should be a graph node"
    );
    assert!(
        graph.nodes().contains_key("concept:concurrency"),
        "auto-extracted concept 'concurrency' should be a graph node"
    );
}

#[test]
fn caller_concepts_are_preserved_and_augmented() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    // Insert with one caller concept — it should be preserved, and the
    // remaining slots filled by extraction.
    engine
        .insert(
            "m1",
            "Rust memory safety concurrency",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["my_tag".to_string()],
        )
        .unwrap();
    let graph = engine.graph.read();
    let graph = graph.graph();
    // Caller concept preserved.
    assert!(graph.nodes().contains_key("concept:my_tag"));
    // Extracted concepts augmented.
    assert!(graph.nodes().contains_key("concept:rust"));
}

#[test]
fn max_concepts_zero_disables_extraction() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_concepts = 0;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine
        .insert(
            "m1",
            "Rust memory safety concurrency",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();
    let graph = engine.graph.read();
    let graph = graph.graph();
    // With max_concepts=0 and no caller concepts, no concept nodes should exist.
    assert!(
        !graph.nodes().contains_key("concept:rust"),
        "extraction should be disabled when max_concepts=0"
    );
}

#[test]
fn check_refinements_creates_edge_for_related_memories() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    // Enable refinement with a low threshold so our test vectors trigger it.
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.refinement_max_pairs_per_cycle = 100;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();

    // Insert an "old" memory about rust safety.
    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    // Insert a "new" memory about the same topic (high cosine, shares concept).
    engine
        .insert(
            "new",
            "Rust borrow checker safety",
            &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();

    // Run check_refinements — should create a Refines edge old → new.
    let created = engine.check_refinements().unwrap();
    assert!(
        created >= 1,
        "should create at least one Refines edge, got {created}"
    );

    let graph = engine.graph.read();
    let graph = graph.graph();
    assert!(
        graph.refinement_count() >= 1,
        "graph should have Refines edges"
    );
    // The edge should be old → new.
    let refined = graph.refined_by("old");
    assert!(
        refined.contains(&"new".to_string()),
        "old should refine to new, got {refined:?}"
    );
}

/// With `belief_source_roles = ["user"]`, an assistant-authored memory can
/// neither create nor receive a supersession edge, even though the same
/// vectors DO create one when both memories are user-authored (control).
/// The assistant memory stays fully retrievable — role never filters search.
#[test]
fn role_filtered_detection_excludes_nonuser_memories() {
    let old_vec = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let new_vec = [0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Control: both user-authored → a Refines edge is created.
    {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.refinement_cosine_threshold = Some(0.5);
        config.tier.refinement_max_pairs_per_cycle = 100;
        config.tier.belief_source_roles = Some(vec!["user".to_string()]);
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        engine
            .insert_with_payload_role(
                "old",
                "Rust memory safety",
                &old_vec,
                1.0,
                &["rust".to_string()],
                None,
                None,
                Some("user".to_string()),
            )
            .unwrap();
        engine
            .insert_with_payload_role(
                "new",
                "Rust borrow checker safety",
                &new_vec,
                1.0,
                &["rust".to_string()],
                None,
                None,
                Some("user".to_string()),
            )
            .unwrap();
        assert_eq!(
            engine.check_refinements().unwrap(),
            1,
            "control: two user facts should create a Refines edge"
        );
    }

    // Role-filtered: the older memory is assistant-authored → excluded from
    // supersession, so no edge forms despite identical vectors/concepts.
    {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = small_config(8);
        config.tier.refinement_cosine_threshold = Some(0.5);
        config.tier.refinement_max_pairs_per_cycle = 100;
        config.tier.belief_source_roles = Some(vec!["user".to_string()]);
        let engine = StorageEngine::open(tmp.path(), config).unwrap();
        engine
            .insert_with_payload_role(
                "asst_old",
                "Rust memory safety",
                &old_vec,
                1.0,
                &["rust".to_string()],
                None,
                None,
                Some("assistant".to_string()),
            )
            .unwrap();
        engine
            .insert_with_payload_role(
                "user_new",
                "Rust borrow checker safety",
                &new_vec,
                1.0,
                &["rust".to_string()],
                None,
                None,
                Some("user".to_string()),
            )
            .unwrap();
        assert_eq!(
            engine.check_refinements().unwrap(),
            0,
            "assistant memory must be excluded from supersession"
        );

        // ...but the assistant memory is still fully retrievable.
        let results = engine
            .search("rust memory safety", &old_vec, 5)
            .unwrap()
            .expect("search should return candidates");
        assert!(
            results.iter().any(|(id, _)| id == "asst_old"),
            "role filter must not affect retrieval; got {results:?}"
        );
    }
}

/// Incremental supersession detection (W7): with the seq-cursor on, a
/// refinement detected in a LATER consolidation cycle (a newer memory
/// superseding one from an earlier cycle) is still found — the watermark
/// only skips records that were already checked, not new ones.
#[test]
fn incremental_detection_finds_cross_cycle_refinement() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.refinement_max_pairs_per_cycle = 100;
    config.tier.incremental_supersession_detection = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();

    // Cycle 1: two memories about the same topic → one refinement.
    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "mid",
            "Rust borrow checker safety",
            &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine.trigger_consolidation().unwrap();
    assert_eq!(
        engine.graph.read().graph().refinement_count(),
        1,
        "cycle 1 refinement"
    );

    // Cycle 2: a newer memory superseding the previous one — inserted AFTER
    // the watermark advanced, so it must still be detected.
    engine
        .insert(
            "new",
            "Rust borrow checker memory safety",
            &[0.85f32, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine.trigger_consolidation().unwrap();
    assert!(
        engine.graph.read().graph().refinement_count() >= 2,
        "cross-cycle refinement must be detected with the incremental cursor, got {}",
        engine.graph.read().graph().refinement_count()
    );
}

/// `source_role` is durable: it survives WAL replay across a restart.
#[test]
fn source_role_survives_flush_and_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let config = small_config(8);
    {
        let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
        engine
            .insert_with_payload_role(
                "u1",
                "a user fact",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["fact".to_string()],
                None,
                None,
                Some("user".to_string()),
            )
            .unwrap();
        engine.flush().unwrap();
    }
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    let offset = *engine.id_index.read().get("u1").unwrap();
    let rec = engine.meta.get(offset).unwrap().unwrap();
    assert_eq!(
        rec.source_role.as_deref(),
        Some("user"),
        "source_role should persist across restart"
    );
}

#[test]
fn graph_state_survives_restart_via_binary_snapshot() {
    let tmp = tempfile::tempdir().unwrap();
    let config = small_config(8);
    let (edge_count, refinement_count);
    {
        let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
        engine
            .insert(
                "old",
                "Rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Rust borrow checker safety",
                &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        // Learn state that only the persisted snapshot can preserve.
        engine.graph.write().add_refinement("old", "new", 0.8);
        engine.graph.write().reinforce("old", 1234);
        engine.flush().unwrap();
        {
            let graph = engine.graph.read();
            let graph = graph.graph();
            edge_count = graph.edge_count();
            refinement_count = graph.refinement_count();
        }
        // The persisted snapshot must be the binary format.
        let bytes = engine
            .meta
            .load_meta_bytes("graph")
            .expect("binary graph snapshot persisted");
        assert!(MemoryGraph::is_snapshot_bytes(&bytes));
    }
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    let graph = engine.graph.read();
    let graph = graph.graph();
    assert_eq!(graph.edge_count(), edge_count);
    assert_eq!(graph.refinement_count(), refinement_count);
    assert_eq!(graph.refined_by("old"), vec!["new".to_string()]);
    let reinforced = graph
        .neighbors(&turbomemory_graph::NodeId::memory("old").as_str())
        .iter()
        .any(|e| e.last_reinforced_at == 1234);
    assert!(reinforced, "reinforcement timestamp should survive restart");
}

#[test]
fn legacy_json_graph_snapshot_still_loads_on_open() {
    let tmp = tempfile::tempdir().unwrap();
    let config = small_config(8);
    {
        let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
        engine
            .insert(
                "m1",
                "Rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine.flush().unwrap();
        // Simulate a pre-binary database: keep only a legacy JSON snapshot
        // under the old string meta key.
        let json = engine.graph.read().graph().to_json();
        engine.meta.save_meta("graph", &json).unwrap();
        engine.meta.remove_meta_bytes("graph").unwrap();
    }
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    {
        let graph = engine.graph.read();
        let graph = graph.graph();
        assert_eq!(graph.memory_count(), 1);
        assert!(graph.nodes().contains_key("concept:rust"));
    }
    // The next flush rewrites the snapshot in binary form and reclaims the
    // legacy JSON entry.
    engine.flush().unwrap();
    let bytes = engine
        .meta
        .load_meta_bytes("graph")
        .expect("binary snapshot after flush");
    assert!(MemoryGraph::is_snapshot_bytes(&bytes));
    assert!(engine.meta.load_meta_str("graph").is_none());
}

/// Deferred commit + incremental detection: consolidation must leave the
/// cursor alone so the caller's `propose_supersessions` still sees the new
/// records (it used to advance first, and nothing was ever proposed), and a
/// second proposal pass does not repeat pairs already examined.
#[test]
fn deferred_commit_with_incremental_detection_still_proposes() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.refinement_max_pairs_per_cycle = 100;
    config.tier.incremental_supersession_detection = true;
    config.tier.defer_supersession_commit = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust borrow checker safety",
            &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();

    engine.trigger_consolidation().unwrap();
    assert_eq!(
        engine.graph.read().graph().refinement_count(),
        0,
        "deferred: consolidation commits nothing"
    );
    let props = engine.propose_supersessions().unwrap();
    assert_eq!(props.len(), 1, "the pair is still there to propose");
    assert_eq!(
        (props[0].old_id.as_str(), props[0].new_id.as_str()),
        ("old", "new")
    );
    assert_eq!(engine.commit_supersessions(&props).unwrap(), 1);

    // Already examined: the next cycle proposes nothing new.
    engine.trigger_consolidation().unwrap();
    assert!(engine.propose_supersessions().unwrap().is_empty());
}

/// The propose/commit split (W3): `propose_*` detects pairs WITHOUT
/// mutating the graph or demoting; `commit_supersessions` is what applies
/// the edge + demotion. A verifier can drop pairs between the two steps.
#[test]
fn propose_supersessions_is_pure_commit_mutates() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.refinement_max_pairs_per_cycle = 100;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust borrow checker safety",
            &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();

    // propose detects the pair but does NOT touch the graph or demotion.
    let props = engine.propose_supersessions().unwrap();
    assert_eq!(props.len(), 1, "one refinement should be proposed");
    assert_eq!(props[0].old_id, "old");
    assert_eq!(props[0].new_id, "new");
    assert_eq!(props[0].kind, SupersessionKind::Refinement);
    let old_off = *engine.id_index.read().get("old").unwrap();
    assert!(
        (engine.meta.demotion_factor(old_off) - 1.0).abs() < 1e-6,
        "propose must not demote"
    );
    assert_eq!(
        engine.graph.read().graph().refinement_count(),
        0,
        "propose must not create edges"
    );

    // A verifier that REJECTS the pair (empty commit) leaves everything untouched.
    assert_eq!(engine.commit_supersessions(&[]).unwrap(), 0);
    assert_eq!(engine.graph.read().graph().refinement_count(), 0);

    // Committing the accepted pair applies the edge + bounded demotion.
    assert_eq!(engine.commit_supersessions(&props).unwrap(), 1);
    assert_eq!(engine.graph.read().graph().refinement_count(), 1);
    assert!(
        engine.meta.demotion_factor(old_off) < 1.0,
        "commit must demote"
    );
}

/// Wide candidates: a memory is offered its closest older memory and those
/// about as close, not every older memory on the topic, and that still holds
/// in the cycle after the closest one was replaced (with superseded records
/// hidden from ordinary searches).
#[test]
fn supersession_candidates_keep_to_the_closest_older_memories() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.refinement_max_pairs_per_cycle = 100;
    config.tier.exclude_superseded = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // Cosine to "new": target 0.98, reworded 0.97, other 0.59.
    for (id, text, vector) in [
        (
            "target",
            "the recital starts at five",
            [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "reworded",
            "five is when the recital begins",
            [0.99f32, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "other",
            "my monday class starts at nine",
            [0.6f32, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "new",
            "actually the recital starts at six",
            [0.98f32, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ] {
        engine.insert(id, text, &vector, 1.0, &[]).unwrap();
    }
    let offered_to_new = |margin: f32| -> Vec<String> {
        engine
            .propose_supersession_candidates(0.45, 3, margin)
            .unwrap()
            .into_iter()
            .filter(|p| p.new_id == "new")
            .map(|p| p.old_id)
            .collect()
    };
    assert_eq!(
        offered_to_new(f32::INFINITY),
        ["target", "reworded", "other"]
    );
    assert_eq!(offered_to_new(0.1), ["target", "reworded"]);

    // The verifier accepted both wordings of the old time.
    let committed = engine
        .commit_supersessions_by_id(&[
            (
                "target".to_string(),
                "new".to_string(),
                SupersessionKind::Contradiction,
            ),
            (
                "reworded".to_string(),
                "new".to_string(),
                SupersessionKind::Contradiction,
            ),
        ])
        .unwrap();
    assert_eq!(committed, 2);

    // Next cycle: what "new" replaced still marks its closest neighbour, so
    // the unrelated fact is not offered in its place.
    assert!(offered_to_new(0.1).is_empty());
    assert_eq!(offered_to_new(f32::INFINITY), ["other"]);
    // A replaced memory replaces nothing itself any more.
    let all = engine
        .propose_supersession_candidates(0.45, 3, f32::INFINITY)
        .unwrap();
    assert!(all
        .iter()
        .all(|p| p.new_id != "reworded" && p.old_id != "target" && p.old_id != "reworded"));
}

/// A memory that some OTHER statement replaced is not the reference: the
/// current belief is offered even when it is worded far from the original.
#[test]
fn supersession_candidates_ignore_memories_replaced_by_others() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.refinement_max_pairs_per_cycle = 100;
    config.tier.exclude_superseded = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // Cosine to "madrid": lisbon 0.98, porto 0.49.
    for (id, text, vector) in [
        (
            "lisbon",
            "i live in lisbon",
            [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "porto",
            "we relocated to porto last spring",
            [0.5f32, 0.866, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ),
    ] {
        engine.insert(id, text, &vector, 1.0, &[]).unwrap();
    }
    engine
        .commit_supersessions_by_id(&[(
            "lisbon".to_string(),
            "porto".to_string(),
            SupersessionKind::Contradiction,
        )])
        .unwrap();
    engine
        .insert(
            "madrid",
            "i live in madrid",
            &[0.98f32, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();

    let offered: Vec<(String, String)> = engine
        .propose_supersession_candidates(0.45, 2, 0.1)
        .unwrap()
        .into_iter()
        .map(|p| (p.old_id, p.new_id))
        .collect();
    assert_eq!(offered, [("porto".to_string(), "madrid".to_string())]);
}

/// `scope_ids` is one scope's own records, oldest first: no other scope's,
/// no global ones, nothing deleted, and an updated record keeps its id but
/// moves to the end (it is a new version).
#[test]
fn scope_ids_lists_one_scope_in_insertion_order() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    let put = |id: &str, axis: usize, scope: Option<&str>| {
        let mut v = [0.0f32; 8];
        v[axis] = 1.0;
        engine
            .insert_with_payload(id, id, &v, 1.0, &[], None, scope.map(str::to_string))
            .unwrap();
    };
    put("a1", 0, Some("alice"));
    put("b1", 1, Some("bob"));
    put("g1", 2, None);
    put("a2", 3, Some("alice"));
    put("a3", 4, Some("alice"));
    assert_eq!(engine.scope_ids("alice"), ["a1", "a2", "a3"]);
    assert_eq!(engine.scope_ids("bob"), ["b1"]);
    assert!(engine.scope_ids("carol").is_empty());
    assert_eq!(engine.scopes(), ["alice", "bob"]);

    assert!(engine.delete_by_id("a2").unwrap());
    let mut v = [0.0f32; 8];
    v[5] = 1.0;
    // An update replaces the whole record, scope included.
    assert!(engine
        .update_with_payload(
            "a1",
            "a1 again",
            &v,
            1.0,
            &[],
            None,
            Some("alice".to_string())
        )
        .unwrap());
    assert_eq!(engine.scope_ids("alice"), ["a3", "a1"]);

    assert!(engine.delete_by_id("b1").unwrap());
    assert_eq!(engine.scopes(), ["alice"]);
}

/// Belief-state resolution: after committing the A <- B <- C supersession
/// chain, `resolve_beliefs` maps every chain element to the CURRENT head
/// with the full lineage — with `exclude_superseded` off (resolution is an
/// annotation contract, independent of the exclusion flag).
#[test]
fn resolve_beliefs_walks_supersession_chain_to_head() {
    let tmp = tempfile::tempdir().unwrap();
    let config = small_config(8);
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    for (id, text) in [
        ("a", "Rust memory safety v1"),
        ("b", "Rust memory safety v2"),
        ("c", "Rust memory safety v3"),
    ] {
        engine
            .insert(
                id,
                text,
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
    }
    let committed = engine
        .commit_supersessions_by_id(&[
            (
                "a".to_string(),
                "b".to_string(),
                SupersessionKind::Refinement,
            ),
            (
                "b".to_string(),
                "c".to_string(),
                SupersessionKind::Refinement,
            ),
        ])
        .unwrap();
    assert_eq!(committed, 2);

    let res = engine.resolve_beliefs(&[
        "a".to_string(),
        "b".to_string(),
        "c".to_string(),
        "ghost".to_string(),
    ]);
    assert_eq!(res.len(), 4);

    assert_eq!(res[0].id, "a");
    assert_eq!(res[0].current_id, "c", "a resolves to the chain head");
    assert!(res[0].superseded);
    assert_eq!(
        res[0].chain,
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        "lineage is oldest-first with the head last"
    );

    assert_eq!(res[1].current_id, "c");
    assert!(res[1].superseded);
    assert_eq!(
        res[1].chain,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );

    assert_eq!(res[2].current_id, "c", "the head resolves to itself");
    assert!(!res[2].superseded);
    assert_eq!(
        res[2].chain,
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
        "the lineage of the head is the full chain containing it"
    );

    assert_eq!(
        res[3].current_id, "ghost",
        "unknown ids resolve to themselves"
    );
    assert!(!res[3].superseded);
    assert_eq!(res[3].chain, vec!["ghost".to_string()]);

    // A superseded memory reports the factor its score is multiplied by, so
    // a caller can tell where it ranked before; a current one reports 1.
    let factor = engine.config.tier.supersession_demotion_factor;
    assert!(factor < 1.0);
    assert_eq!(res[0].demotion, factor);
    assert_eq!(res[1].demotion, factor);
    assert_eq!(res[2].demotion, 1.0);
    assert_eq!(res[3].demotion, 1.0);
}

/// With no supersession edges at all (no cognitive flags enabled),
/// every id resolves to itself.
#[test]
fn resolve_beliefs_without_edges_is_identity() {
    let tmp = tempfile::tempdir().unwrap();
    let config = small_config(8);
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine
        .insert(
            "plain",
            "an ordinary fact",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    let res = engine.resolve_beliefs(&["plain".to_string()]);
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].current_id, "plain");
    assert!(!res[0].superseded);
    assert_eq!(res[0].chain, vec!["plain".to_string()]);
}

#[test]
fn supersession_demotion_makes_new_refinement_outrank_old_at_alpha_one() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.cognitive_alpha = 1.0;
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.supersession_demotion_factor = 0.4;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();

    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust memory safety updated",
            &[0.95f32, 0.3122499, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();

    assert_eq!(engine.check_refinements().unwrap(), 1);
    let old_offset = *engine.id_index.read().get("old").unwrap();
    assert!((engine.meta.demotion_factor(old_offset) - 0.4).abs() < 1e-6);

    let results = engine
        .search(
            "rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        )
        .unwrap()
        .expect("search should return candidates");
    assert_eq!(results[0].0, "new", "demotion should lower stale memory");
    assert!(results[0].1 > results[1].1);
}

/// B1 OFF arm (default): superseded memories are rank-demoted but still
/// returned. With `exclude_superseded` at its default `false`, the engine
/// must behave exactly as before.
#[test]
fn exclude_superseded_off_keeps_superseded_memory_in_results() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.cognitive_alpha = 1.0;
    config.tier.refinement_cosine_threshold = Some(0.5);
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    assert!(!engine.config.tier.exclude_superseded);

    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust memory safety updated",
            &[0.95f32, 0.3122499, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    assert_eq!(engine.check_refinements().unwrap(), 1);

    let results = engine
        .search(
            "rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        )
        .unwrap()
        .expect("search should return candidates");
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ids.contains(&"old"), "OFF: stale memory is only demoted");
    assert!(ids.contains(&"new"));

    let ann = engine
        .search_ann(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
        .unwrap();
    let ann_ids: Vec<&str> = ann.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ann_ids.contains(&"old"), "OFF: ANN keeps the stale memory");
    assert!(ann_ids.contains(&"new"));
}

/// B1 ON arm: with `exclude_superseded = true`, the stale memory is
/// dropped from BOTH cognitive search and ANN search (single + batch),
/// while the current memory remains.
#[test]
fn exclude_superseded_on_removes_stale_memory_from_all_search_paths() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.cognitive_alpha = 1.0;
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.exclude_superseded = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();

    engine
        .insert(
            "old",
            "Rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust memory safety updated",
            &[0.95f32, 0.3122499, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    assert_eq!(engine.check_refinements().unwrap(), 1);
    assert_eq!(
        engine.graph.read().graph().superseded_ids(),
        vec!["old".to_string()],
        "old must be on the superseded side of the Refines edge"
    );

    let results = engine
        .search(
            "rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        )
        .unwrap()
        .expect("search should return candidates");
    let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ids.contains(&"new"), "current memory must remain");
    assert!(
        !ids.contains(&"old"),
        "ON: stale memory must be excluded from cognitive search"
    );

    let ann = engine
        .search_ann(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
        .unwrap();
    let ann_ids: Vec<&str> = ann.iter().map(|(id, _)| id.as_str()).collect();
    assert!(ann_ids.contains(&"new"));
    assert!(
        !ann_ids.contains(&"old"),
        "ON: stale memory must be excluded from search_ann"
    );

    let batch = engine
        .search_ann_batch(
            &[&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0][..]],
            2,
            None,
            None,
            None,
        )
        .unwrap();
    let batch_ids: Vec<&str> = batch[0].iter().map(|(id, _)| id.as_str()).collect();
    assert!(batch_ids.contains(&"new"));
    assert!(
        !batch_ids.contains(&"old"),
        "ON: stale memory must be excluded from search_ann_batch"
    );
}

/// B1 fast path: exclusion enabled but the graph has NO supersessions —
/// results must be byte-identical to the flag-OFF engine.
#[test]
fn exclude_superseded_on_without_supersessions_matches_off() {
    let insert_all = |engine: &StorageEngine| {
        engine
            .insert(
                "a",
                "alpha fact",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["misc".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "b",
                "beta fact",
                &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["misc".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "c",
                "gamma fact",
                &[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["misc".to_string()],
            )
            .unwrap();
    };

    let tmp_off = tempfile::tempdir().unwrap();
    let engine_off = StorageEngine::open(tmp_off.path(), small_config(8)).unwrap();
    insert_all(&engine_off);

    let tmp_on = tempfile::tempdir().unwrap();
    let mut config_on = small_config(8);
    config_on.tier.exclude_superseded = true;
    let engine_on = StorageEngine::open(tmp_on.path(), config_on).unwrap();
    insert_all(&engine_on);

    let query: &[f32] = &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(
        engine_on.search_ann(query, 2).unwrap(),
        engine_off.search_ann(query, 2).unwrap(),
        "no supersessions: ANN results must be identical"
    );
    assert_eq!(
        engine_on.search("misc fact", query, 2).unwrap(),
        engine_off.search("misc fact", query, 2).unwrap(),
        "no supersessions: cognitive results must be identical"
    );
}

#[test]
fn supersession_demotion_persists_after_flush_and_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.cognitive_alpha = 1.0;
    config.tier.refinement_cosine_threshold = Some(0.5);
    config.tier.supersession_demotion_factor = 0.4;

    {
        let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
        engine
            .insert(
                "old",
                "Rust memory safety",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        engine
            .insert(
                "new",
                "Rust memory safety updated",
                &[0.95f32, 0.3122499, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &["rust".to_string()],
            )
            .unwrap();
        assert_eq!(engine.check_refinements().unwrap(), 1);
        engine.flush().unwrap();
    }

    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    let old_offset = *engine.id_index.read().get("old").unwrap();
    assert!((engine.meta.demotion_factor(old_offset) - 0.4).abs() < 1e-6);

    let results = engine
        .search(
            "rust memory safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            2,
        )
        .unwrap()
        .expect("search should return candidates after restart");
    assert_eq!(results[0].0, "new");
}

#[test]
fn check_refinements_disabled_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    engine
        .insert(
            "old",
            "Rust",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    let created = engine.check_refinements().unwrap();
    assert_eq!(
        created, 0,
        "refinement should be disabled when threshold is None"
    );
}

#[test]
fn check_refinements_skips_unrelated_concepts() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.refinement_cosine_threshold = Some(0.5);
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // Two memories with high cosine (same vector) but NO shared concepts.
    engine
        .insert(
            "old",
            "Rust",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Python",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["python".to_string()],
        )
        .unwrap();
    let created = engine.check_refinements().unwrap();
    assert_eq!(created, 0, "should not refine when concepts don't match");
}

#[test]
fn check_contradictions_creates_edge_and_weakens_old() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    // Enable contradiction detection with a low cosine threshold.
    config.tier.contradiction_cosine_threshold = Some(0.5);
    config.tier.contradiction_text_threshold = 0.4; // texts must be < 40% similar
    config.tier.contradiction_weaken_factor = 0.5;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();

    // Old memory: a FALSE claim. Text mentions the topic word "rust" but
    // otherwise uses vocabulary disjoint from the correction so the Jaccard
    // similarity stays below the threshold — a contradiction, not a refinement.
    //   A tokens: {rust, requires, manual, compilation, execution}
    //   B tokens: {rust, compiled, actually, runs, interpretation}
    //   intersection = {rust} → Jaccard = 1/9 ≈ 0.11 < 0.4, AND the
    //   correction carries opposition markers ("not", "actually").
    engine
        .insert(
            "old_claim",
            "Rust requires manual compilation before execution",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    // New memory: the CORRECTION (same topic, opposing claim).
    engine
        .insert(
            "new_correction",
            "Rust is not compiled; it actually runs through interpretation",
            &[0.9f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.5,
            &["rust".to_string()],
        )
        .unwrap();

    let created = engine.check_contradictions().unwrap();
    assert!(
        created >= 1,
        "should create at least one Contradicts edge, got {created}"
    );

    let graph = engine.graph.read();
    let graph = graph.graph();
    assert!(
        graph.contradiction_count() >= 1,
        "graph should have Contradicts edges"
    );
    let corrected = graph.contradicted_by("old_claim");
    assert!(
        corrected.contains(&"new_correction".to_string()),
        "old_claim should be contradicted by new_correction, got {corrected:?}"
    );
    let old_offset = *engine.id_index.read().get("old_claim").unwrap();
    assert!(
        engine.meta.demotion_factor(old_offset) < crate::metadata_store::NO_DEMOTION,
        "contradicted memory should receive a final-score demotion"
    );
}

#[test]
fn check_contradictions_disabled_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    engine
        .insert(
            "old",
            "Rust gc",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust borrow",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    let created = engine.check_contradictions().unwrap();
    assert_eq!(
        created, 0,
        "contradiction should be disabled when threshold is None"
    );
}

#[test]
fn check_contradictions_skips_high_text_overlap() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.contradiction_cosine_threshold = Some(0.5);
    config.tier.contradiction_text_threshold = 0.3; // low threshold = harder to be a contradiction
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // Two memories with same text (high Jaccard) — should NOT be a contradiction.
    engine
        .insert(
            "old",
            "Rust is safe fast",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "new",
            "Rust is safe fast",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    let created = engine.check_contradictions().unwrap();
    assert_eq!(
        created, 0,
        "high text overlap should not be a contradiction"
    );
}

#[test]
fn recompute_importance_disabled_by_default() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    engine
        .insert(
            "a",
            "Rust safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    let changed = engine.recompute_importance().unwrap();
    assert_eq!(changed, 0, "should be a no-op when auto scoring is off");
}

#[test]
fn recompute_importance_raises_frequently_retrieved() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.importance_auto_scoring = true;
    config.tier.importance_learning_rate = 1.0; // jump straight to target
                                                // Both records start at importance 1.0.
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine
        .insert(
            "hot",
            "Rust safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["rust".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "cold",
            "Python threads",
            &[0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["python".to_string()],
        )
        .unwrap();

    // Retrieve "hot" many times so its access_count dominates. Each
    // search_ann bumps the matched record's access counter.
    let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..20 {
        let _ = engine.search_ann(&q, 1).unwrap();
    }
    // Drain counters into metadata so recompute sees them.
    engine.recompute_importance().unwrap();

    let hot_offset = *engine.id_index.read().get("hot").unwrap();
    let cold_offset = *engine.id_index.read().get("cold").unwrap();
    let hot_imp = engine.meta.get(hot_offset).unwrap().unwrap().importance;
    let cold_imp = engine.meta.get(cold_offset).unwrap().unwrap().importance;
    assert!(
        hot_imp > cold_imp,
        "frequently-retrieved record should have higher importance: hot={hot_imp} cold={cold_imp}"
    );
    // The never-retrieved record decays toward the floor.
    assert!(
        cold_imp < 1.0,
        "never-retrieved record should decay below its start: cold={cold_imp}"
    );
}

#[test]
fn recompute_importance_respects_floor_and_ceiling() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.importance_auto_scoring = true;
    config.tier.importance_learning_rate = 1.0;
    config.tier.importance_floor = 0.5;
    config.tier.importance_ceiling = 2.0;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // One record, never retrieved. Its target blends to the floor.
    engine
        .insert(
            "x",
            "Rust safety",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            5.0, // above ceiling
            &["rust".to_string()],
        )
        .unwrap();
    engine.recompute_importance().unwrap();
    let offset = *engine.id_index.read().get("x").unwrap();
    let imp = engine.meta.get(offset).unwrap().unwrap().importance;
    assert!(
        imp <= 2.0 + 1e-5,
        "importance should not exceed ceiling 2.0, got {imp}"
    );
    assert!(
        imp >= 0.5 - 1e-5,
        "importance should not drop below floor 0.5, got {imp}"
    );
}

#[test]
fn tier_seal_keeps_search_correct() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    for i in 0..8usize {
        let v = make_vec(8, i);
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &v,
                1.0,
                &[format!("c{}", i % 2)],
            )
            .unwrap();
    }
    engine.trigger_consolidation().unwrap();
    let q = make_vec(8, 3);
    let results = engine.search_ann(&q, 3).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].0, "mem_3");
}

#[test]
fn restart_reloads_records_and_tiers() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..6usize {
            let v = make_vec(8, i);
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }
        engine.trigger_consolidation().unwrap();
        engine.flush().unwrap();
    }
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(engine.record_count(), 6);
        let q = make_vec(8, 2);
        let results = engine.search_ann(&q, 1).unwrap();
        assert_eq!(results[0].0, "mem_2");
    }
}

#[test]
fn wal_truncation_recovery() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..5usize {
            let v = make_vec(8, i);
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }
        // Make embeddings and WAL durable but do not snapshot metadata / clear WAL.
        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
    }

    // Simulate a torn write by truncating the last 4 bytes (the CRC of the
    // final WAL record).  The iterator should detect the CRC mismatch and
    // stop replay, recovering all preceding records.
    let wal_path = tmp.path().join("wal").join(crate::wal::WAL_FILE);
    let bytes = std::fs::read(&wal_path).unwrap();
    // Walk the WAL to find the byte offset of the final record's CRC.
    let mut pos = crate::wal::WAL_HEADER_SIZE;
    let mut last_record_crc_end = pos;
    while pos + 4 <= bytes.len() {
        let len = u32::from_be_bytes([bytes[pos], bytes[pos + 1], bytes[pos + 2], bytes[pos + 3]])
            as usize;
        if pos + 4 + len + 4 > bytes.len() {
            break;
        }
        last_record_crc_end = pos + 4 + len + 4;
        pos = last_record_crc_end;
    }
    // Truncate only the CRC of the last complete record.
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&wal_path)
        .unwrap();
    file.set_len((last_record_crc_end - 4) as u64).unwrap();
    drop(file);

    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    assert_eq!(engine.record_count(), 4);
    for i in 0..4usize {
        let q = make_vec(8, i);
        let results = engine.search_ann(&q, 1).unwrap();
        assert_eq!(results[0].0, format!("mem_{i}"));
    }
}

#[test]
fn wal_replay_without_flush() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..5usize {
            let v = make_vec(8, i);
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }
        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
    }

    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    assert_eq!(engine.record_count(), 5);
}

#[test]
fn hot_warm_cold_persist_and_reload() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..8usize {
            let v = make_vec(8, i);
            engine
                .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
                .unwrap();
        }
        // First seal -> SealedHot, second seal -> Warm.
        engine.trigger_consolidation().unwrap();
        engine.flush().unwrap();
    }
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        assert_eq!(engine.record_count(), 8);
        // Each query should be correct regardless of which tier the data
        // currently lives in.
        for i in 0..8usize {
            let q = make_vec(8, i);
            let results = engine.search_ann(&q, 1).unwrap();
            assert_eq!(results[0].0, format!("mem_{i}"));
        }
    }
}

#[test]
fn promotion_brings_accessed_records_back_to_hot() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.hot_capacity = 2;
    config.tier.hot_promote_threshold = 0.5;
    config.tier.recency_half_life_secs = 3600;

    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    for i in 0..4usize {
        let v = make_vec(8, i);
        engine
            .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
            .unwrap();
    }

    // First consolidation seals the initial Hot segment (mem_0, mem_1).
    engine.trigger_consolidation().unwrap();

    // Repeatedly search for mem_0 to bump its access score.
    let q = make_vec(8, 0);
    for _ in 0..3 {
        let results = engine.search_ann(&q, 1).unwrap();
        assert_eq!(results[0].0, "mem_0");
    }

    // The next consolidation should promote mem_0 back into Hot.
    let (_sealed, _compacted, promoted) = engine.trigger_consolidation().unwrap();
    assert!(promoted > 0, "expected at least one promotion");

    // Search still works correctly after promotion.
    let results = engine.search_ann(&q, 1).unwrap();
    assert_eq!(results[0].0, "mem_0");
}

#[test]
fn eviction_disabled_by_default_keeps_all_records() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    for i in 0..6usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    // Default config has max_records = None and evict_score_floor = None.
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 0);
    engine.trigger_consolidation().unwrap();
    assert_eq!(engine.record_count(), 6);
}

#[test]
fn eviction_cap_bounds_record_count() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(2);
    config.tier.recency_half_life_secs = 3600;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    for i in 0..5usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    // Give mem_0 and mem_1 a high access score so they survive the cap.
    // Searching bumps access_count and refreshes last_accessed (also
    // protecting them via the grace window).
    for idx in [0usize, 1] {
        let q = make_vec(8, idx);
        for _ in 0..5 {
            engine.search_ann(&q, 1).unwrap();
        }
    }
    let evicted = engine.evict().unwrap();
    assert!(
        evicted >= 3,
        "expected to evict down to the cap, got {evicted}"
    );
    assert!(engine.record_count() <= 2);
    // The frequently-accessed records must survive.
    assert!(engine.find_record_by_id("mem_0").is_some());
    assert!(engine.find_record_by_id("mem_1").is_some());
}

/// W5 isolation: with `access_aware_eviction = false`, eviction is pure FIFO
/// (oldest-inserted evicted first) and IGNORES rehearsal — even a
/// heavily-accessed old record is dropped, while the naive access-aware
/// policy (previous test) would have kept it. This is the OFF baseline that
/// isolates the retain-what-is-used mechanism.
#[test]
fn fifo_eviction_ignores_rehearsal() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(2);
    config.tier.recency_half_life_secs = 3600;
    config.tier.access_aware_eviction = false; // naive FIFO baseline
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    for i in 0..5usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    // Rehearse the OLDEST records heavily — under FIFO this must NOT save them.
    for idx in [0usize, 1] {
        let q = make_vec(8, idx);
        for _ in 0..5 {
            engine.search_ann(&q, 1).unwrap();
        }
    }
    let evicted = engine.evict().unwrap();
    assert!(
        evicted >= 3,
        "expected eviction down to the cap, got {evicted}"
    );
    assert!(engine.record_count() <= 2);
    // FIFO keeps the NEWEST, drops the oldest — rehearsal is ignored.
    assert!(
        engine.find_record_by_id("mem_0").is_none(),
        "FIFO must evict the oldest despite rehearsal"
    );
    assert!(
        engine.find_record_by_id("mem_4").is_some(),
        "newest must survive under FIFO"
    );
}

#[test]
fn eviction_score_floor_drops_stale() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    // Floor above 0 so never-accessed records (score 0) are evicted, while
    // accessed records stay above it.
    config.tier.evict_score_floor = Some(0.5);
    config.tier.recency_half_life_secs = 3600;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    for i in 0..4usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    // Access mem_0 so its score climbs above the floor and the grace window
    // protects it.
    let q = make_vec(8, 0);
    for _ in 0..3 {
        engine.search_ann(&q, 1).unwrap();
    }
    let evicted = engine.evict().unwrap();
    // mem_1, mem_2, mem_3 are never accessed (score 0 < 0.5) and were
    // inserted with last_accessed = 0 (outside the grace window).
    assert_eq!(evicted, 3);
    assert!(engine.find_record_by_id("mem_0").is_some());
    assert!(engine.find_record_by_id("mem_1").is_none());
}

#[test]
fn eviction_respects_grace_period() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    // Aggressive floor that would evict everything by score alone.
    config.tier.evict_score_floor = Some(1000.0);
    config.tier.recency_half_life_secs = 3600; // grace = 450s
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // Touch each record so last_accessed = now, placing it inside the
    // grace window even though its score is below the floor.
    for i in 0..3usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
        engine.search_ann(&make_vec(8, i), 1).unwrap();
    }
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 0, "freshly accessed records must be protected");
    assert_eq!(engine.record_count(), 3);
}

/// ACT-R eviction (opt-in): a record whose accesses were spaced out over
/// time outranks a record with the same-ish count crammed into one recent
/// burst, because base-level activation sums the power-law decay of every
/// access instead of just count × last-access recency.
#[test]
fn actr_eviction_prefers_spaced_over_burst() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(1);
    config.tier.recency_half_life_secs = 3600; // grace = 450s
    config.tier.actr_activation = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine
        .insert("mem_burst", "text burst", &make_vec(8, 0), 1.0, &[])
        .unwrap();
    engine
        .insert("mem_spaced", "text spaced", &make_vec(8, 1), 1.0, &[])
        .unwrap();

    let now = now_secs();
    let burst_off = *engine.id_index.read().get("mem_burst").unwrap();
    let spaced_off = *engine.id_index.read().get("mem_spaced").unwrap();
    // One burst: three accesses crammed into a single moment 600s ago.
    for _ in 0..3 {
        engine.access_counters.bump(burst_off, now - 600);
    }
    // Spaced rehearsal: six accesses spread over the last hour.
    for age in [500u64, 600, 800, 1200, 2000, 3500] {
        engine.access_counters.bump(spaced_off, now - age);
    }
    // Both last accesses (600s / 500s) are outside the 450s grace window,
    // so ranking — not protection — decides. ACT-R scores:
    //   burst  = ln(3 × 600^-0.5)                 ≈ -2.10
    //   spaced = ln(500^-0.5 + … + 3500^-0.5)     ≈ -1.67
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 1, "exactly one record over the cap");
    assert!(
        engine.find_record_by_id("mem_spaced").is_some(),
        "spaced rehearsal must survive ACT-R eviction"
    );
    assert!(
        engine.find_record_by_id("mem_burst").is_none(),
        "single recent burst must be evicted first"
    );
    // The evicted record's access-history ring is cleaned up too.
    assert!(engine.meta.access_history(burst_off).is_empty());
}

/// B4 gist-before-evict: victims are compressed into a gist record before
/// deletion, so their content stays retrievable under the same scope.
struct JoinCompressor;
impl GistCompressor for JoinCompressor {
    fn compress(&self, texts: &[String]) -> Result<Option<(String, Vec<f32>)>, String> {
        Ok(Some((texts.join(" | "), make_vec(8, 7))))
    }
}

struct NullCompressor;
impl GistCompressor for NullCompressor {
    fn compress(&self, _texts: &[String]) -> Result<Option<(String, Vec<f32>)>, String> {
        Ok(None)
    }
}

/// (id, text, scope, role, payload) of every record with role "gist".
type GistRow = (
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

fn find_gist_records(engine: &StorageEngine) -> Vec<GistRow> {
    let mut out = Vec::new();
    engine
        .meta
        .for_each_record(|_offset, rec| {
            if rec.source_role.as_deref() == Some("gist") {
                out.push((
                    rec.id.clone(),
                    rec.text.clone(),
                    rec.scope.clone(),
                    rec.source_role.clone(),
                    rec.payload.clone(),
                ));
            }
        })
        .unwrap();
    out
}

#[test]
fn gist_before_evict_compresses_victims() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(1);
    config.tier.access_aware_eviction = false; // FIFO: oldest evicted first
    config.tier.gist_before_evict = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine.set_gist_compressor(Some(Arc::new(JoinCompressor)));
    for i in 0..3usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 2, "mem_0 and mem_1 are the FIFO victims");
    assert!(engine.contains_id("mem_2"));
    // One gist record carries the victims' content: 1 survivor + 1 gist.
    assert_eq!(engine.record_count(), 2);
    let gists = find_gist_records(&engine);
    assert_eq!(gists.len(), 1);
    let (id, text, scope, role, payload) = &gists[0];
    assert!(id.starts_with("gist:global:"));
    assert_eq!(text, "text 0 | text 1");
    assert_eq!(scope, &None);
    assert_eq!(role.as_deref(), Some("gist"));
    assert!(payload.as_deref().unwrap().contains("\"victims\":2"));
    // The gist is retrievable through the normal ANN path.
    let hits = engine.search_ann(&make_vec(8, 7), 1).unwrap();
    assert_eq!(hits[0].0, *id);
}

#[test]
fn gist_before_evict_respects_scope_isolation() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(0); // evict everything
    config.tier.access_aware_eviction = false;
    config.tier.gist_before_evict = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine.set_gist_compressor(Some(Arc::new(JoinCompressor)));
    for (scope, i) in [("a", 0usize), ("a", 1), ("b", 2), ("b", 3)] {
        engine
            .insert_with_payload_role(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
                None,
                Some(scope.to_string()),
                None,
            )
            .unwrap();
    }
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 4);
    // One gist per scope; no gist mixes texts across scopes.
    let gists = find_gist_records(&engine);
    assert_eq!(gists.len(), 2);
    assert_eq!(engine.record_count(), 2);
    let gist_a = gists.iter().find(|g| g.2.as_deref() == Some("a")).unwrap();
    let gist_b = gists.iter().find(|g| g.2.as_deref() == Some("b")).unwrap();
    assert_eq!(gist_a.1, "text 0 | text 1");
    assert_eq!(gist_b.1, "text 2 | text 3");
    assert!(gist_a.0.starts_with("gist:a:"));
    assert!(gist_b.0.starts_with("gist:b:"));
}

#[test]
fn gist_before_evict_off_by_default_drops_victims() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(1);
    config.tier.access_aware_eviction = false;
    // gist_before_evict defaults to false — no compressor consulted.
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    for i in 0..3usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 2);
    assert_eq!(engine.record_count(), 1);
    assert!(find_gist_records(&engine).is_empty());
}

#[test]
fn gist_before_evict_skips_chunk_when_compressor_abstains() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.max_records = Some(1);
    config.tier.access_aware_eviction = false;
    config.tier.gist_before_evict = true;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    engine.set_gist_compressor(Some(Arc::new(NullCompressor)));
    for i in 0..3usize {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    let evicted = engine.evict().unwrap();
    assert_eq!(evicted, 2, "compressor abstention must not block eviction");
    assert_eq!(engine.record_count(), 1);
    assert!(find_gist_records(&engine).is_empty());
}

/// The access-history ring drains through flush and reloads on open, so
/// ACT-R scoring keeps working across restarts.
#[test]
fn access_history_survives_flush_and_reopen() {
    let tmp = tempfile::tempdir().unwrap();
    let config = small_config(8);
    {
        let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
        engine
            .insert("mem_0", "text 0", &make_vec(8, 0), 1.0, &[])
            .unwrap();
        // Three real searches → three access bumps at real timestamps.
        let q = make_vec(8, 0);
        for _ in 0..3 {
            assert_eq!(engine.search_ann(&q, 1).unwrap()[0].0, "mem_0");
        }
        engine.flush().unwrap();
    }
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    let offset = *engine.id_index.read().get("mem_0").unwrap();
    let history = engine.meta.access_history(offset);
    assert_eq!(
        history.len(),
        3,
        "access history should persist across restart"
    );
    assert!(history.iter().all(|&ts| ts > 0));
    assert!(history.windows(2).all(|w| w[0] <= w[1]));
    // ACT-R activation over the reloaded history is finite.
    assert!(crate::access_counters::actr_activation(&history, now_secs(), 0.5).is_finite());
    // A record with no recorded accesses loads with an empty ring.
    engine
        .insert("mem_1", "text 1", &make_vec(8, 1), 1.0, &[])
        .unwrap();
    let offset1 = *engine.id_index.read().get("mem_1").unwrap();
    assert!(engine.meta.access_history(offset1).is_empty());
}

#[test]
fn dedup_merges_near_duplicates_keeping_salient() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    config.tier.dedup_cosine_threshold = Some(0.95);
    config.tier.recency_half_life_secs = 3600;
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    // Two near-identical vectors (same direction) plus a distinct one.
    engine
        .insert(
            "dup_keep",
            "keep me",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["a".to_string()],
        )
        .unwrap();
    engine
        .insert(
            "dup_drop",
            "drop me",
            &[0.999f32, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &["b".to_string()],
        )
        .unwrap();
    engine
        .insert("distinct", "different", &make_vec(8, 4), 1.0, &[])
        .unwrap();
    // Make dup_keep the more salient of the pair.
    let q = make_vec(8, 0);
    for _ in 0..3 {
        engine.search_ann(&q, 1).unwrap();
    }
    let merged = engine.deduplicate().unwrap();
    assert_eq!(
        merged, 1,
        "exactly one of the duplicate pair should be merged"
    );
    assert!(engine.find_record_by_id("dup_keep").is_some());
    assert!(engine.find_record_by_id("dup_drop").is_none());
    // The distinct record is untouched.
    assert!(engine.find_record_by_id("distinct").is_some());
    // Survivor inherited the victim's concept edge.
    let survivor = engine.find_record_by_id("dup_keep").unwrap();
    assert!(survivor.concepts.contains(&"a".to_string()));
    let graph = engine.graph.read();
    assert!(
        graph.graph().concept_degree("b") >= 1,
        "survivor should inherit victim concept 'b'"
    );
}

#[test]
fn dedup_disabled_leaves_duplicates() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    engine
        .insert(
            "a",
            "x",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();
    engine
        .insert(
            "b",
            "y",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
        )
        .unwrap();
    // dedup_cosine_threshold defaults to None.
    let merged = engine.deduplicate().unwrap();
    assert_eq!(merged, 0);
    assert_eq!(engine.record_count(), 2);
}

#[test]
fn delete_removes_record_from_search_and_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    for i in 0..4usize {
        let v = make_vec(8, i);
        engine
            .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
            .unwrap();
    }

    assert!(engine.delete_by_id("mem_2").unwrap());
    assert!(!engine.delete_by_id("mem_2").unwrap());
    assert_eq!(engine.record_count(), 3);

    // Searching for the deleted vector should return its nearest neighbor among
    // the remaining records, not mem_2.
    let q = make_vec(8, 2);
    let results = engine.search_ann(&q, 1).unwrap();
    assert_ne!(results[0].0, "mem_2");

    // Reopen and replay: the delete should survive.
    engine.flush_vectors().unwrap();
    engine.flush_wal().unwrap();
    drop(engine);
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    assert_eq!(engine.record_count(), 3);
    let results = engine.search_ann(&q, 1).unwrap();
    assert_ne!(results[0].0, "mem_2");
}

#[test]
fn shutdown_stops_optimizer_so_reopen_succeeds() {
    let tmp = tempfile::tempdir().unwrap();
    let mut config = small_config(8);
    // Background optimizer ON, cycling fast, so it is live at shutdown.
    config.auto_consolidation_interval = Some(std::time::Duration::from_millis(10));
    let engine = StorageEngine::open(tmp.path(), config.clone()).unwrap();
    engine
        .insert("mem_0", "text 0", &make_vec(8, 0), 1.0, &[])
        .unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    engine.shutdown().unwrap();
    // Idempotent, and foreground calls keep working afterwards.
    engine.shutdown().unwrap();
    assert_eq!(engine.record_count(), 1);
    drop(engine);

    // redb takes an exclusive lock on the database file, so this only
    // succeeds if the previous engine was really released.
    let engine = StorageEngine::open(tmp.path(), config).unwrap();
    assert_eq!(engine.record_count(), 1);
}

#[test]
fn next_insert_seq_is_never_reused_after_delete_and_restart() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    // Sequences start at 1; 0 is the "nothing applied yet" WAL sentinel.
    assert_eq!(engine.next_insert_seq(), 1);
    for i in 0..3usize {
        let v = make_vec(8, i);
        engine
            .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
            .unwrap();
    }
    assert_eq!(engine.next_insert_seq(), 4);
    // Deleting the newest record must not hand its sequence out again.
    assert!(engine.delete_by_id("mem_2").unwrap());

    // Restart via WAL replay (no metadata flush).
    engine.flush_vectors().unwrap();
    engine.flush_wal().unwrap();
    drop(engine);
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    assert_eq!(engine.record_count(), 2);
    assert_eq!(engine.next_insert_seq(), 4);

    // Restart via a clean shutdown (redb snapshot).
    engine.shutdown().unwrap();
    drop(engine);
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    assert_eq!(engine.next_insert_seq(), 4);
}

#[test]
fn update_replaces_record() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    engine
        .insert("mem_0", "original text", &make_vec(8, 0), 1.0, &[])
        .unwrap();

    assert!(engine
        .update("mem_0", "updated text", &make_vec(8, 1), 2.0, &[])
        .unwrap());
    assert!(!engine
        .update("missing", "x", &make_vec(8, 1), 1.0, &[])
        .unwrap());
    assert_eq!(engine.record_count(), 1);

    let q = make_vec(8, 1);
    let results = engine.search_ann(&q, 1).unwrap();
    assert_eq!(results[0].0, "mem_0");

    // The old vector should no longer be returned.
    let q_old = make_vec(8, 0);
    let results = engine.search_ann(&q_old, 1).unwrap();
    assert_eq!(results[0].0, "mem_0");
}

#[test]
fn batch_insert_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    let ids: Vec<String> = (0..4).map(|i| format!("mem_{i}")).collect();
    let texts: Vec<String> = (0..4).map(|i| format!("text {i}")).collect();
    let embeddings: Vec<Vec<f32>> = (0..4).map(|i| make_vec(8, i)).collect();
    let scores = vec![1.0f32; 4];
    let concepts: Vec<Vec<String>> = vec![vec![]; 4];

    let n1 = engine
        .insert_batch(&ids, &texts, &embeddings, &scores, &concepts)
        .unwrap();
    assert_eq!(n1, 4);

    // Replay the same batch: existing ids should be skipped.
    let n2 = engine
        .insert_batch(&ids, &texts, &embeddings, &scores, &concepts)
        .unwrap();
    assert_eq!(n2, 0);
    assert_eq!(engine.record_count(), 4);

    // Duplicate ids within a batch should be deduplicated.
    let dup_ids = vec![
        "new_1".to_string(),
        "new_1".to_string(),
        "new_2".to_string(),
    ];
    let dup_texts = vec!["t1".to_string(), "t1".to_string(), "t2".to_string()];
    let dup_embs = vec![make_vec(8, 5), make_vec(8, 5), make_vec(8, 6)];
    let dup_scores = vec![1.0f32; 3];
    let dup_concepts = vec![vec![]; 3];
    let n3 = engine
        .insert_batch(&dup_ids, &dup_texts, &dup_embs, &dup_scores, &dup_concepts)
        .unwrap();
    assert_eq!(n3, 2);
    assert_eq!(engine.record_count(), 6);
}

#[test]
fn payload_round_trip_and_replay() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    let payload = r#"{"tags":["rust","ai"],"count":42}"#.to_string();
    engine
        .insert_with_payload(
            "mem_0",
            "text",
            &make_vec(8, 0),
            1.0,
            &[],
            Some(payload.clone()),
            None,
        )
        .unwrap();

    assert_eq!(
        engine.get_payload("mem_0").unwrap().as_ref(),
        Some(&payload)
    );

    engine.flush_vectors().unwrap();
    engine.flush_wal().unwrap();
    drop(engine);

    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    assert_eq!(
        engine.get_payload("mem_0").unwrap().as_ref(),
        Some(&payload)
    );
}

#[test]
fn filtered_search_by_payload() {
    use serde_json::json;
    use std::ops::Bound;

    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    for i in 0..4usize {
        let payload = json!({"category": if i % 2 == 0 { "even" } else { "odd" }, "score": (i as f64) * 10.0 });
        engine
            .insert_with_payload(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(8, i),
                1.0,
                &[],
                Some(payload.to_string()),
                None,
            )
            .unwrap();
    }

    // Equality filter.
    let filter = Filter::Eq {
        field: "category".into(),
        value: json!("even"),
    };
    let results = engine
        .search_ann_filtered(&make_vec(8, 0), 10, &filter)
        .unwrap();
    let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec!["mem_0", "mem_2"]);

    // Range filter.
    let filter = Filter::Range {
        field: "score".into(),
        low: Bound::Included(15.0),
        high: Bound::Included(35.0),
    };
    let results = engine
        .search_ann_filtered(&make_vec(8, 0), 10, &filter)
        .unwrap();
    let mut ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
    ids.sort();
    assert_eq!(ids, vec!["mem_2", "mem_3"]);

    // Delete removes from index.
    engine.delete_by_id("mem_2").unwrap();
    let filter = Filter::Eq {
        field: "category".into(),
        value: json!("even"),
    };
    let results = engine
        .search_ann_filtered(&make_vec(8, 0), 10, &filter)
        .unwrap();
    let ids: Vec<_> = results.into_iter().map(|(id, _)| id).collect();
    assert_eq!(ids, vec!["mem_0"]);
}

#[test]
fn filtered_search_survives_replay() {
    use serde_json::json;

    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        for i in 0..4usize {
            let payload = json!({"group": "a", "idx": i});
            engine
                .insert_with_payload(
                    &format!("mem_{i}"),
                    &format!("text {i}"),
                    &make_vec(8, i),
                    1.0,
                    &[],
                    Some(payload.to_string()),
                    None,
                )
                .unwrap();
        }
        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
    }

    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    let filter = Filter::Eq {
        field: "group".into(),
        value: json!("a"),
    };
    let results = engine
        .search_ann_filtered(&make_vec(8, 0), 10, &filter)
        .unwrap();
    assert_eq!(results.len(), 4);
}

#[test]
fn merge_optimizer_combines_sealed_segments() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
    let n = 600usize;
    for i in 0..n {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(32, i),
                1.0,
                &[],
            )
            .unwrap();
    }
    engine.trigger_consolidation().unwrap();
    engine.flush().unwrap();
    // The merge optimizer runs in the background; wait for it to reduce the
    // sealed segment count.
    let mut sealed_count = engine.segments.read().sealed_hot_count();
    for _ in 0..60 {
        if sealed_count <= 2 {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        sealed_count = engine.segments.read().sealed_hot_count();
    }
    assert!(
        sealed_count <= 2,
        "expected at most 2 sealed segments, got {}",
        sealed_count
    );
    // Search should still return results.
    let results = engine.search_ann(&make_vec(32, 0), 5).unwrap();
    assert_eq!(results.len(), 5);
    // mem_0 and mem_32 are identical one-hot vectors; the approximate index may
    // return either of them. Just verify the top result is a perfect cosine match.
    assert!(results[0].1 > 0.9999, "top result should be an exact match");
    assert!(results[0].0.starts_with("mem_"));
}

/// Run searches on many threads while another thread drives consolidation
/// (seal + merge installs) on the same engine. The ArcSwap snapshot path
/// means searches must keep returning valid results and never deadlock with
/// the now-internally-synchronized SegmentHolder mutations.
#[test]
fn concurrent_search_during_consolidation() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;

    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
    let n = 600usize;
    for i in 0..n {
        engine
            .insert(
                &format!("mem_{i}"),
                &format!("text {i}"),
                &make_vec(32, i),
                1.0,
                &[],
            )
            .unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));

    // Searcher threads hammer the engine while consolidation runs.
    let searchers: Vec<_> = (0..4)
        .map(|t| {
            let engine = Arc::clone(&engine);
            let stop = Arc::clone(&stop);
            thread::spawn(move || {
                let mut iters = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let results = engine
                        .search_ann(&make_vec(32, t), 5)
                        .expect("search must not fail during consolidation");
                    for (id, _) in &results {
                        assert!(id.starts_with("mem_"), "unexpected id {id}");
                    }
                    iters += 1;
                }
                iters
            })
        })
        .collect();

    // Driver thread: repeatedly consolidate (seal + merge + flush).
    let driver = {
        let engine = Arc::clone(&engine);
        thread::spawn(move || {
            for _ in 0..5 {
                engine.trigger_consolidation().unwrap();
            }
        })
    };

    driver.join().unwrap();
    stop.store(true, Ordering::Relaxed);
    for s in searchers {
        let iters = s.join().unwrap();
        assert!(iters > 0, "searcher should have run at least once");
    }

    // Final search still returns a full result set with a perfect top match.
    let results = engine.search_ann(&make_vec(32, 0), 5).unwrap();
    assert_eq!(results.len(), 5);
    assert!(results[0].1 > 0.9999, "top result should be an exact match");
}

#[test]
fn search_ann_with_ef_can_improve_recall() {
    use rand::Rng;
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
    let dim = 32;
    let n = 600;
    let mut rng = rand::thread_rng();
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
        embeddings.push(v.clone());
        engine
            .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
            .unwrap();
    }
    engine.trigger_consolidation().unwrap();

    // Compute flat ground truth for a few queries.
    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
            v
        })
        .collect();

    let mut low_recall = 0.0f32;
    let mut high_recall = 0.0f32;
    for q in &queries {
        let gt = {
            let mut scored: Vec<(String, f32)> = embeddings
                .iter()
                .enumerate()
                .map(|(i, emb)| {
                    let score = turbomemory_core::cosine_similarity(q, emb);
                    (format!("mem_{i}"), score)
                })
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(5);
            scored
                .into_iter()
                .map(|(id, _)| id)
                .collect::<std::collections::HashSet<_>>()
        };
        let low = engine.search_ann_with_ef(q, 5, None).unwrap();
        let high = engine.search_ann_with_ef(q, 5, Some(200)).unwrap();
        let low_ids: std::collections::HashSet<_> = low.into_iter().map(|(id, _)| id).collect();
        let high_ids: std::collections::HashSet<_> = high.into_iter().map(|(id, _)| id).collect();
        low_recall += low_ids.intersection(&gt).count() as f32 / 5.0;
        high_recall += high_ids.intersection(&gt).count() as f32 / 5.0;
    }
    low_recall /= queries.len() as f32;
    high_recall /= queries.len() as f32;
    assert!(
        high_recall >= low_recall,
        "higher ef should not reduce recall: low={}, high={}",
        low_recall,
        high_recall
    );
}

#[test]
fn search_ann_batch_matches_single_query() {
    // Batch search must return the same results as running each query
    // individually. Exercises the batched rerank path (CPU fallback here;
    // the gemm path is validated on-GPU separately).
    use rand::Rng;
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), hnsw_test_config(32)).unwrap();
    let dim = 32;
    let n = 600;
    let mut rng = rand::thread_rng();
    for i in 0..n {
        let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
        engine
            .insert(&format!("mem_{i}"), &format!("text {i}"), &v, 1.0, &[])
            .unwrap();
    }
    engine.trigger_consolidation().unwrap();

    // 8 random queries.
    let queries: Vec<Vec<f32>> = (0..8)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() - 0.5).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.iter_mut().for_each(|x| *x /= norm.max(1e-8));
            v
        })
        .collect();

    // Per-query results.
    let single: Vec<Vec<(String, f32)>> = queries
        .iter()
        .map(|q| engine.search_ann_with_ef(q, 5, Some(200)).unwrap())
        .collect();

    // Batch results.
    let q_refs: Vec<&[f32]> = queries.iter().map(|q| q.as_slice()).collect();
    let batch = engine
        .search_ann_batch(&q_refs, 5, Some(200), None, None)
        .unwrap();

    assert_eq!(
        batch.len(),
        single.len(),
        "batch should return one list per query"
    );
    for (i, (b, s)) in batch.iter().zip(single.iter()).enumerate() {
        let b_ids: Vec<&str> = b.iter().map(|(id, _)| id.as_str()).collect();
        let s_ids: Vec<&str> = s.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(
            b_ids, s_ids,
            "query {i}: batch results must match single-query results"
        );
    }
}

#[test]
fn search_ann_batch_empty_returns_empty() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    let batch = engine.search_ann_batch(&[], 5, None, None, None).unwrap();
    assert!(batch.is_empty(), "empty batch should return empty");
}

#[test]
fn scoped_search_isolates_agent_memories() {
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();

    // Global memory: visible to all scopes.
    engine
        .insert_with_payload(
            "global_mem",
            "global knowledge",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
            None,
            None,
        )
        .unwrap();

    // Agent A private memory.
    engine
        .insert_with_payload(
            "agent_a_mem",
            "agent a secret",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
            None,
            Some("agent_a".into()),
        )
        .unwrap();

    // Agent B private memory.
    engine
        .insert_with_payload(
            "agent_b_mem",
            "agent b secret",
            &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1.0,
            &[],
            None,
            Some("agent_b".into()),
        )
        .unwrap();

    let q = &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Global search sees all three records.
    let global_results: Vec<String> = engine
        .search_ann_scoped(q, 10, None, None)
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(global_results.len(), 3);

    // Agent A search sees global + agent A, but not agent B.
    let a_results: Vec<String> = engine
        .search_ann_scoped(q, 10, None, Some("agent_a"))
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert!(a_results.contains(&"global_mem".to_string()));
    assert!(a_results.contains(&"agent_a_mem".to_string()));
    assert!(!a_results.contains(&"agent_b_mem".to_string()));

    // Agent B search sees global + agent B, but not agent A.
    let b_results: Vec<String> = engine
        .search_ann_scoped(q, 10, None, Some("agent_b"))
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert!(b_results.contains(&"global_mem".to_string()));
    assert!(!b_results.contains(&"agent_a_mem".to_string()));
    assert!(b_results.contains(&"agent_b_mem".to_string()));
}

#[test]
fn scoped_search_with_empty_scope_matches_nothing() {
    // Regression: an empty filter/scope bitmap was treated as "unfiltered"
    // by `exact_top_k_filtered`, leaking every scope's records to a scoped
    // query that legitimately matches zero records.
    let tmp = tempfile::tempdir().unwrap();
    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();

    for (id, scope) in [("a_mem", "agent_a"), ("b_mem", "agent_b")] {
        engine
            .insert_with_payload(
                id,
                "private note",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
                None,
                Some(scope.into()),
            )
            .unwrap();
    }

    let q = &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    // No global records exist, so a scope that has never been written must
    // return NOTHING (previously returned every record in the store).
    let results = engine
        .search_ann_scoped(q, 10, None, Some("agent_never_written"))
        .unwrap();
    assert!(results.is_empty(), "empty scope leaked: {results:?}");

    // Batch path shares the same choke point and must agree.
    let batch = engine
        .search_ann_batch(&[q.as_slice()], 10, None, None, Some("agent_never_written"))
        .unwrap();
    assert!(batch[0].is_empty(), "batch empty scope leaked: {batch:?}");
}

#[test]
fn scoped_search_survives_replay() {
    let tmp = tempfile::tempdir().unwrap();
    {
        let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
        engine
            .insert_with_payload(
                "scoped_mem",
                "scoped",
                &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                1.0,
                &[],
                None,
                Some("agent_x".into()),
            )
            .unwrap();
        engine.flush_vectors().unwrap();
        engine.flush_wal().unwrap();
    }

    let engine = StorageEngine::open(tmp.path(), small_config(8)).unwrap();
    let q = &[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results: Vec<String> = engine
        .search_ann_scoped(q, 10, None, Some("agent_x"))
        .unwrap()
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(results, vec!["scoped_mem".to_string()]);
}
