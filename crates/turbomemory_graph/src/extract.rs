//! Concept extraction from text.
//!
//! Provides a lightweight, deterministic, zero-dependency concept extractor
//! that runs on the insert path without adding latency. Quality is below a
//! full NER/LLM extractor but makes the cognitive graph work as a turnkey
//! layer — callers no longer *have* to supply `concepts` on every insert.
//!
//! The extractor:
//! 1. Lowercases and tokenizes the text on non-alphanumeric boundaries.
//! 2. Drops stopwords, tokens shorter than 3 characters, and pure-digit
//!    tokens.
//! 3. Scores surviving tokens by term-frequency within the text (with a
//!    small length bonus so longer, more specific tokens rank higher).
//! 4. Extracts n-grams (bigrams and trigrams by default when enabled) so
//!    multi-word concepts like "memory safety" and "borrow checker" are
//!    captured as single concepts instead of being split into unrelated
//!    unigrams.
//! 5. Optionally canonicalizes surface forms through a [`ConceptVocabulary`]
//!    so synonyms ("programming" / "coding") map to a single concept node.

use std::collections::HashMap;

/// A compact set of common English stopwords. Kept inline to avoid a
/// dependency on a stopword crate. Covers ~170 high-frequency function words
/// that carry no domain meaning.
const STOPWORDS: &[&str] = &[
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "am",
    "an",
    "and",
    "any",
    "are",
    "aren",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "cannot",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "don",
    "down",
    "during",
    "each",
    "few",
    "for",
    "from",
    "further",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "herself",
    "him",
    "himself",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "just",
    "me",
    "more",
    "most",
    "my",
    "myself",
    "no",
    "nor",
    "not",
    "now",
    "of",
    "off",
    "on",
    "once",
    "only",
    "or",
    "other",
    "our",
    "ours",
    "ourselves",
    "out",
    "over",
    "own",
    "same",
    "she",
    "should",
    "so",
    "some",
    "such",
    "than",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "themselves",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "to",
    "too",
    "under",
    "until",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "also",
    "been",
    "being",
    "but",
    "can",
    "may",
    "might",
    "must",
    "need",
    "shall",
    "well",
    "even",
    "still",
    "yet",
    "many",
    "much",
    "lot",
    "way",
    "thing",
    "things",
    "make",
    "makes",
    "made",
    "get",
    "got",
    "go",
    "going",
    "gone",
    "one",
    "two",
    "three",
    "first",
    "last",
    "new",
    "old",
    "good",
    "bad",
    "big",
    "small",
    "use",
    "used",
    "using",
    "like",
    "likely",
    "via",
    "per",
    "etc",
    "ie",
    "eg",
    "would",
    "could",
    "should",
    "might",
    "must",
    "shall",
    "will",
    "may",
];

/// Configuration for the concept extractor.
///
/// The default configuration is intentionally conservative (unigrams only) so
/// existing behavior and benchmarks are preserved. Enable `max_ngram_len > 1`
/// to capture multi-word concepts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtractorConfig {
    /// Maximum number of concepts to return. `0` disables extraction.
    pub max_concepts: usize,
    /// Maximum n-gram length. `1` = unigrams only, `2` = unigrams + bigrams,
    /// `3` = up to trigrams. Values above 3 are clamped to 3.
    pub max_ngram_len: usize,
    /// Minimum number of times an n-gram must appear to be considered. For
    /// short memory texts this is typically 1.
    pub min_ngram_freq: usize,
    /// Whether to boost n-gram scores by pointwise mutual information. This
    /// rewards collocations ("memory safety") over accidental adjacencies
    /// ("the rust").
    pub enable_pmi_scoring: bool,
    /// Weight given to PMI in the n-gram score. Higher values make PMI more
    /// influential relative to raw frequency.
    pub pmi_weight: f32,
}

impl Default for ExtractorConfig {
    fn default() -> Self {
        Self {
            max_concepts: 5,
            max_ngram_len: 1,
            min_ngram_freq: 1,
            enable_pmi_scoring: true,
            pmi_weight: 1.0,
        }
    }
}

impl ExtractorConfig {
    /// A config tuned for richer concept extraction: unigrams + bigrams +
    /// trigrams with PMI scoring.
    pub fn ngram() -> Self {
        Self {
            max_ngram_len: 3,
            ..Self::default()
        }
    }

    fn effective_max_ngram_len(&self) -> usize {
        self.max_ngram_len.clamp(1, 3)
    }
}

/// A shared concept vocabulary that canonicalizes surface forms.
///
/// This is the foundation for online concept vocabulary evolution (C3). It
/// currently supports explicit alias mappings (e.g. "coding" → "programming");
/// embedding-based matching can be layered on top by higher-level code that
/// has access to concept embeddings.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ConceptVocabulary {
    aliases: HashMap<String, String>,
}

impl ConceptVocabulary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Map a surface form `alias` to a canonical concept. Both are normalized
    /// to lowercase.
    pub fn add_alias(&mut self, alias: &str, canonical: &str) {
        let canonical_norm = canonical.to_lowercase();
        let alias_norm = alias.to_lowercase();
        // If the canonical form itself is an alias of something else, resolve
        // it so the chain collapses to the root.
        let root = self
            .aliases
            .get(&canonical_norm)
            .cloned()
            .unwrap_or(canonical_norm);
        self.aliases.insert(alias_norm, root);
    }

    /// Resolve a surface concept to its canonical form, if known.
    /// Resolve a surface concept to its canonical form, if known.
    ///
    /// Follows alias chains (e.g. "coding" -> "programming" -> "software
    /// engineering") and detects cycles.
    pub fn resolve(&self, concept: &str) -> String {
        let key = concept.to_lowercase();
        let mut current = key;
        let mut seen = std::collections::HashSet::new();
        while let Some(next) = self.aliases.get(&current) {
            if !seen.insert(next.clone()) {
                break; // cycle detected
            }
            current = next.clone();
        }
        current
    }

    /// Returns true if `concept` has a known canonical alias.
    pub fn is_alias(&self, concept: &str) -> bool {
        self.aliases.contains_key(&concept.to_lowercase())
    }

    /// Iterate over known aliases.
    pub fn aliases(&self) -> &HashMap<String, String> {
        &self.aliases
    }

    /// Rebuild a vocabulary from raw alias pairs without chain resolution.
    /// Used by snapshot restore, where the pairs were captured verbatim and
    /// must round-trip exactly.
    pub(crate) fn from_alias_pairs(pairs: Vec<(String, String)>) -> Self {
        Self {
            aliases: pairs.into_iter().collect(),
        }
    }
}

/// Tokenize text into clean, filtered tokens.
///
/// Returns lowercase tokens with stopwords, short tokens (<3 chars), and
/// pure-digit tokens removed.
fn tokenize_filtered(text: &str) -> Vec<String> {
    let stopwords: std::collections::HashSet<&str> = STOPWORDS.iter().copied().collect();
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter_map(|token| {
            let t = token.trim();
            if t.len() < 3 {
                return None;
            }
            if t.chars().all(|c| c.is_ascii_digit()) {
                return None;
            }
            if stopwords.contains(t) {
                return None;
            }
            Some(t.to_string())
        })
        .collect()
}

/// Compute the word-set of a concept string.
fn word_set(concept: &str) -> std::collections::HashSet<&str> {
    concept.split_whitespace().collect()
}

/// Returns true if `candidate` is a strict subset of any already-selected
/// concept. This prevents returning both "memory safety" and "memory".
fn is_subsumed(candidate: &str, selected: &[String]) -> bool {
    let cand_words = word_set(candidate);
    if cand_words.len() <= 1 {
        // A unigram is subsumed if it appears as a word in a selected n-gram.
        let unigram = candidate;
        for s in selected {
            let s_words: Vec<&str> = s.split_whitespace().collect();
            if s_words.len() > 1 && s_words.contains(&unigram) {
                return true;
            }
        }
        false
    } else {
        // An n-gram is subsumed only if all its words appear in a single
        // selected concept (e.g. "memory safety guarantees" inside
        // "memory safety guarantee"). We keep this conservative.
        for s in selected {
            let s_words = word_set(s);
            if s_words.len() > cand_words.len() && cand_words.is_subset(&s_words) {
                return true;
            }
        }
        false
    }
}

/// Extract up to `config.max_concepts` concept strings from `text`.
///
/// Concepts are ranked by a composite score that combines term frequency,
/// length specificity, and (for n-grams) pointwise mutual information. The
/// returned concepts are deduplicated, subsumed unigrams are suppressed, and
/// the list is ordered by descending score.
///
/// Returns an empty `Vec` if `text` is empty or `max` is 0.
pub fn extract_concepts_with_config(text: &str, config: &ExtractorConfig) -> Vec<String> {
    if config.max_concepts == 0 || text.trim().is_empty() {
        return Vec::new();
    }

    let tokens = tokenize_filtered(text);
    if tokens.is_empty() {
        return Vec::new();
    }

    let max_n = config.effective_max_ngram_len();
    let min_freq = config.min_ngram_freq.max(1);

    // Count unigrams and n-grams.
    let mut unigram_counts: HashMap<String, usize> = HashMap::new();
    let mut ngram_counts: Vec<HashMap<String, usize>> = (0..max_n.saturating_sub(1))
        .map(|_| HashMap::new())
        .collect();

    for (i, t) in tokens.iter().enumerate() {
        *unigram_counts.entry(t.clone()).or_insert(0) += 1;
        for n in 2..=max_n {
            if i + n > tokens.len() {
                break;
            }
            let ngram = tokens[i..i + n].join(" ");
            *ngram_counts[n - 2].entry(ngram).or_insert(0) += 1;
        }
    }

    let total_tokens = tokens.len() as f32;

    // Build scored candidates.
    let mut candidates: Vec<(String, f32)> = Vec::new();

    // Unigrams.
    for (term, count) in &unigram_counts {
        if *count < min_freq {
            continue;
        }
        let len_bonus = (term.len() as f32 - 3.0).max(0.0) * 0.1;
        let score = *count as f32 + len_bonus;
        candidates.push((term.clone(), score));
    }

    // N-grams.
    for (n_minus_2, counts) in ngram_counts.iter().enumerate() {
        let n = n_minus_2 + 2;
        for (ngram, count) in counts {
            if *count < min_freq {
                continue;
            }
            let words: Vec<&str> = ngram.split_whitespace().collect();
            if words.len() != n {
                continue;
            }

            let mut base_score = *count as f32 * n as f32;
            // Small length bonus for specificity.
            base_score += (ngram.len() as f32 - 3.0).max(0.0) * 0.03;

            let pmi_bonus = if config.enable_pmi_scoring {
                let joint_prob = *count as f32 / (tokens.len().saturating_sub(n - 1)) as f32;
                let mut independent_prob = 1.0f32;
                for w in &words {
                    let unigram_count = *unigram_counts.get(*w).unwrap_or(&0) as f32;
                    if unigram_count > 0.0 {
                        independent_prob *= unigram_count / total_tokens;
                    } else {
                        independent_prob = 0.0;
                        break;
                    }
                }
                if independent_prob > 0.0 && joint_prob > 0.0 {
                    let pmi = (joint_prob / independent_prob).ln();
                    (pmi * config.pmi_weight).max(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            candidates.push((ngram.clone(), base_score + pmi_bonus));
        }
    }

    // Sort by descending score, then alphabetically for determinism.
    candidates.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Greedy selection with subsumption suppression.
    let mut selected: Vec<String> = Vec::with_capacity(config.max_concepts);
    for (concept, _) in candidates {
        if selected.len() >= config.max_concepts {
            break;
        }
        if is_subsumed(&concept, &selected) {
            continue;
        }
        selected.push(concept);
    }

    selected
}

/// Extract up to `max` concept strings from `text` using the default
/// unigram-only configuration.
///
/// This is the backward-compatible API used by the rest of the engine.
pub fn extract_concepts(text: &str, max: usize) -> Vec<String> {
    let config = ExtractorConfig {
        max_concepts: max,
        ..ExtractorConfig::default()
    };
    extract_concepts_with_config(text, &config)
}

/// Merge caller-supplied concepts with extracted concepts using a configurable
/// extractor and optional vocabulary canonicalization.
///
/// Caller concepts are normalized to lowercase and kept first. Extracted
/// concepts fill the remaining slots up to `max`. Each extracted concept is
/// canonicalized through `vocab` if provided.
pub fn merge_concepts_with_config(
    caller_concepts: &[String],
    text: &str,
    config: &ExtractorConfig,
    vocab: Option<&ConceptVocabulary>,
) -> Vec<String> {
    let max = config.max_concepts;
    let mut result: Vec<String> = Vec::with_capacity(max);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    let resolve = |c: &str| -> String {
        match vocab {
            Some(v) => v.resolve(c),
            None => c.to_lowercase(),
        }
    };

    // Caller concepts first, normalized and canonicalized.
    for c in caller_concepts {
        let norm = resolve(c);
        if !norm.is_empty() && seen.insert(norm.clone()) {
            result.push(norm);
        }
        if result.len() >= max {
            return result;
        }
    }

    // Fill the remaining slots with extracted concepts.
    let mut extract_config = *config;
    extract_config.max_concepts = max.saturating_sub(result.len());
    if extract_config.max_concepts > 0 {
        // Over-fetch so canonicalization/dedup still leaves enough candidates.
        extract_config.max_concepts *= 2;
        for c in extract_concepts_with_config(text, &extract_config) {
            let canon = resolve(&c);
            if seen.insert(canon.clone()) {
                result.push(canon);
            }
            if result.len() >= max {
                break;
            }
        }
    }

    result
}

/// Merge caller-supplied concepts with extracted concepts.
///
/// If `caller_concepts` is non-empty, the caller has explicitly tagged the
/// memory and we trust their judgment — the extracted concepts are only used
/// to *augment* (fill in) up to `max` total, prioritizing the caller's tags.
/// If `caller_concepts` is empty, all concepts come from extraction.
///
/// All concepts are normalized to lowercase and deduplicated.
pub fn merge_concepts(caller_concepts: &[String], text: &str, max: usize) -> Vec<String> {
    let config = ExtractorConfig {
        max_concepts: max,
        ..ExtractorConfig::default()
    };
    merge_concepts_with_config(caller_concepts, text, &config, None)
}

/// Compute Jaccard similarity between the token sets of two texts.
///
/// `Jaccard(A, B) = |A ∩ B| / |A ∪ B|` where A and B are the sets of
/// lowercase alphanumeric tokens (length >= 3, stopwords removed). Returns
/// 0.0 if either text has no valid tokens.
///
/// Used by the contradiction detector: two memories about the same topic
/// (high vector cosine) but with *different content* (low Jaccard) are
/// likely contradictions rather than refinements.
pub fn text_jaccard_similarity(a: &str, b: &str) -> f32 {
    let tokens_a: std::collections::HashSet<String> =
        extract_concepts(a, 100).into_iter().collect();
    let tokens_b: std::collections::HashSet<String> =
        extract_concepts(b, 100).into_iter().collect();
    if tokens_a.is_empty() || tokens_b.is_empty() {
        return 0.0;
    }
    let intersection = tokens_a.intersection(&tokens_b).count();
    let union = tokens_a.union(&tokens_b).count();
    if union == 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

/// Single-word opposition/negation markers, matched as whole tokens so "not"
/// does not match inside "another". Kept lowercase.
const OPPOSITION_MARKERS_WORD: &[&str] = &[
    "not",
    "never",
    "cannot",
    "actually",
    "instead",
    "false",
    "incorrect",
    "wrong",
    "unlike",
    "opposite",
    "contrary",
    "mistaken",
];

/// Multi-word / contraction opposition markers, matched as substrings of the
/// lowercased text. `"n't"` catches the negation contractions (isn't, doesn't,
/// don't, won't, can't, aren't, wasn't, didn't) that the tokenizer would split.
const OPPOSITION_MARKERS_PHRASE: &[&str] = &["n't", "no longer", "rather than", "in fact"];

/// Returns true if `text` contains an explicit opposition / negation marker.
///
/// Used by the contradiction detector to distinguish a genuine contradiction
/// ("X actually uses Y instead") from two *coexisting* facts about the same
/// topic (same concept, low text overlap, but no opposition). This is a
/// lightweight bag-of-cues heuristic, not full NLI: it favors precision and
/// will miss marker-less semantic contradictions.
pub fn has_opposition_marker(text: &str) -> bool {
    let lower = text.to_lowercase();
    if OPPOSITION_MARKERS_PHRASE.iter().any(|m| lower.contains(m)) {
        return true;
    }
    let words: std::collections::HashSet<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();
    OPPOSITION_MARKERS_WORD.iter().any(|m| words.contains(m))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_concepts_from_simple_text() {
        let concepts = extract_concepts("Rust is a safe systems programming language", 5);
        // "rust", "safe", "systems", "programming", "language" — "is", "a" are stopwords.
        assert!(concepts.contains(&"rust".to_string()));
        assert!(concepts.contains(&"safe".to_string()));
        assert!(concepts.contains(&"systems".to_string()));
        assert!(concepts.contains(&"programming".to_string()));
        assert!(!concepts.contains(&"is".to_string()));
        assert!(!concepts.contains(&"a".to_string()));
    }

    #[test]
    fn respects_max_limit() {
        let concepts = extract_concepts("alpha beta gamma delta epsilon zeta", 3);
        assert_eq!(concepts.len(), 3);
    }

    #[test]
    fn empty_text_returns_empty() {
        assert!(extract_concepts("", 5).is_empty());
        assert!(extract_concepts("   ", 5).is_empty());
    }

    #[test]
    fn max_zero_returns_empty() {
        assert!(extract_concepts("rust safety", 0).is_empty());
    }

    #[test]
    fn filters_stopwords() {
        let concepts = extract_concepts("the quick brown fox jumps over the lazy dog", 10);
        // "the", "over" are stopwords; "dog" is 3 chars and passes.
        assert!(!concepts.contains(&"the".to_string()));
        assert!(!concepts.contains(&"over".to_string()));
        assert!(concepts.contains(&"quick".to_string()));
        assert!(concepts.contains(&"brown".to_string()));
    }

    #[test]
    fn filters_short_tokens_and_digits() {
        let concepts = extract_concepts("ai 42 code py", 10);
        // "ai" is 2 chars (filtered), "42" is pure digits (filtered),
        // "code" and "py" — "py" is 2 chars (filtered).
        assert!(!concepts.contains(&"ai".to_string()));
        assert!(!concepts.contains(&"42".to_string()));
        assert!(!concepts.contains(&"py".to_string()));
        assert!(concepts.contains(&"code".to_string()));
    }

    #[test]
    fn tf_ranking_prefers_repeated_tokens() {
        let text = "rust rust rust safety safety python";
        let concepts = extract_concepts(text, 3);
        // "rust" appears 3x -> highest TF, should be first.
        assert_eq!(concepts[0], "rust");
    }

    #[test]
    fn length_bonus_helps_longer_tokens() {
        // Both appear once, but "concurrency" is longer than "cpu".
        let text = "cpu concurrency";
        let concepts = extract_concepts(text, 2);
        // "concurrency" has a higher length bonus and should rank first.
        assert_eq!(concepts[0], "concurrency");
    }

    #[test]
    fn merge_prefers_caller_concepts() {
        let caller = vec!["rust".to_string(), "safety".to_string()];
        let merged = merge_concepts(&caller, "Rust is a safe systems language", 5);
        // Caller concepts come first.
        assert_eq!(merged[0], "rust");
        assert_eq!(merged[1], "safety");
        // Then extracted concepts fill the rest up to 5.
        assert!(merged.len() >= 2);
        assert!(merged.len() <= 5);
        // No duplicates.
        let mut sorted = merged.clone();
        sorted.sort();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(sorted.len(), deduped.len());
    }

    #[test]
    fn merge_extracts_when_caller_provides_none() {
        let merged = merge_concepts(&[], "Rust is a safe systems language", 5);
        assert!(!merged.is_empty());
        assert!(merged.contains(&"rust".to_string()));
        assert!(merged.contains(&"safe".to_string()));
    }

    #[test]
    fn merge_normalizes_to_lowercase() {
        let caller = vec!["Rust".to_string(), "SAFETY".to_string()];
        let merged = merge_concepts(&caller, "", 5);
        assert!(merged.contains(&"rust".to_string()));
        assert!(merged.contains(&"safety".to_string()));
        assert!(!merged.contains(&"Rust".to_string()));
        assert!(!merged.contains(&"SAFETY".to_string()));
    }

    #[test]
    fn merge_deduplicates_caller_and_extracted() {
        // Caller provides "rust"; extractor would also find "rust".
        let caller = vec!["rust".to_string()];
        let merged = merge_concepts(&caller, "rust is safe", 5);
        let rust_count = merged.iter().filter(|c| *c == "rust").count();
        assert_eq!(rust_count, 1, "rust should appear exactly once");
    }

    // ------------------------------------------------------------------
    // N-gram extraction tests
    // ------------------------------------------------------------------

    #[test]
    fn ngram_extracts_multi_word_concepts() {
        let text = "Rust is known for memory safety and concurrency safety";
        let config = ExtractorConfig {
            max_concepts: 5,
            max_ngram_len: 2,
            min_ngram_freq: 1,
            enable_pmi_scoring: true,
            pmi_weight: 1.0,
        };
        let concepts = extract_concepts_with_config(text, &config);
        assert!(
            concepts.contains(&"memory safety".to_string()),
            "expected bigram 'memory safety', got {:?}",
            concepts
        );
    }

    #[test]
    fn ngram_suppresses_subsumed_unigrams() {
        let text = "memory safety guarantees prevent bugs";
        let config = ExtractorConfig {
            max_concepts: 3,
            max_ngram_len: 2,
            min_ngram_freq: 1,
            enable_pmi_scoring: false,
            pmi_weight: 0.0,
        };
        let concepts = extract_concepts_with_config(text, &config);
        // "memory safety" should beat the separate "memory" and "safety".
        assert!(concepts.contains(&"memory safety".to_string()));
        assert!(
            !concepts.contains(&"memory".to_string()),
            "'memory' should be subsumed by 'memory safety'"
        );
        assert!(
            !concepts.contains(&"safety".to_string()),
            "'safety' should be subsumed by 'memory safety'"
        );
    }

    #[test]
    fn ngram_prefers_high_pmi_collocations() {
        // "deep learning" is a real collocation; "learning models" is weaker.
        let text = "deep learning models use deep learning layers";
        let config = ExtractorConfig {
            max_concepts: 2,
            max_ngram_len: 2,
            min_ngram_freq: 1,
            enable_pmi_scoring: true,
            pmi_weight: 2.0,
        };
        let concepts = extract_concepts_with_config(text, &config);
        assert_eq!(concepts[0], "deep learning");
    }

    #[test]
    fn trigram_extraction_works() {
        let text = "the rust borrow checker enforces memory safety rules";
        let config = ExtractorConfig {
            max_concepts: 3,
            max_ngram_len: 3,
            min_ngram_freq: 1,
            enable_pmi_scoring: false,
            pmi_weight: 0.0,
        };
        let concepts = extract_concepts_with_config(text, &config);
        // With max_ngram_len=3 the top concepts are trigrams because they
        // receive a length multiplier. Verify at least one expected trigram
        // is present.
        let expected = [
            "rust borrow checker",
            "borrow checker enforces",
            "checker enforces memory",
            "enforces memory safety",
        ];
        assert!(
            expected.iter().any(|e| concepts.contains(&e.to_string())),
            "expected a borrow-checker or memory-safety trigram, got {:?}",
            concepts
        );
    }

    #[test]
    fn ngram_disabled_by_default() {
        let text = "memory safety is important";
        let concepts = extract_concepts(text, 5);
        assert!(!concepts.contains(&"memory safety".to_string()));
        assert!(concepts.contains(&"memory".to_string()));
        assert!(concepts.contains(&"safety".to_string()));
    }

    #[test]
    fn min_ngram_freq_filters_rare_ngrams() {
        let text = "memory safety once and memory safety twice";
        let config = ExtractorConfig {
            max_concepts: 5,
            max_ngram_len: 2,
            min_ngram_freq: 2,
            enable_pmi_scoring: false,
            pmi_weight: 0.0,
        };
        let concepts = extract_concepts_with_config(text, &config);
        assert!(concepts.contains(&"memory safety".to_string()));
        // "safety once" appears once and should be filtered.
        assert!(!concepts.contains(&"safety once".to_string()));
    }

    // ------------------------------------------------------------------
    // Concept vocabulary tests
    // ------------------------------------------------------------------

    #[test]
    fn vocabulary_canonicalizes_aliases() {
        let mut vocab = ConceptVocabulary::new();
        vocab.add_alias("coding", "programming");
        vocab.add_alias("coding", "programming"); // idempotent
        assert_eq!(vocab.resolve("coding"), "programming");
        assert_eq!(vocab.resolve("programming"), "programming");
    }

    #[test]
    fn vocabulary_resolves_chains() {
        let mut vocab = ConceptVocabulary::new();
        vocab.add_alias("coding", "programming");
        vocab.add_alias("programming", "software engineering");
        assert_eq!(vocab.resolve("coding"), "software engineering");
    }

    #[test]
    fn merge_uses_vocabulary() {
        let mut vocab = ConceptVocabulary::new();
        vocab.add_alias("coding", "programming");
        let caller = vec!["coding".to_string()];
        let config = ExtractorConfig::default();
        let merged =
            merge_concepts_with_config(&caller, "Rust programming language", &config, Some(&vocab));
        assert_eq!(merged[0], "programming");
    }

    #[test]
    fn extracted_concepts_are_canonicalized() {
        let mut vocab = ConceptVocabulary::new();
        vocab.add_alias("safety", "security");
        let config = ExtractorConfig {
            max_concepts: 5,
            ..ExtractorConfig::default()
        };
        let merged =
            merge_concepts_with_config(&[], "memory safety guarantees", &config, Some(&vocab));
        assert!(merged.contains(&"security".to_string()));
        assert!(!merged.contains(&"safety".to_string()));
    }

    #[test]
    fn jaccard_distinguishes_same_and_different_content() {
        let sim_same = text_jaccard_similarity(
            "Rust memory safety guarantees",
            "Rust memory safety guarantees",
        );
        assert!(
            sim_same > 0.99,
            "identical texts should have ~1.0 Jaccard: {sim_same}"
        );

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

    #[test]
    fn opposition_marker_detects_negation_and_contrast() {
        // Genuine corrections carry an explicit opposition/negation cue.
        assert!(has_opposition_marker(
            "Rust is not compiled; it actually runs through interpretation"
        ));
        assert!(has_opposition_marker("Python no longer uses a global lock"));
        assert!(has_opposition_marker("it isn't interpreted"));
        assert!(has_opposition_marker("the model uses JIT instead"));
        // Two coexisting facts about the same topic carry NO opposition marker.
        assert!(!has_opposition_marker(
            "python ships with a large standard library"
        ));
        assert!(!has_opposition_marker(
            "the database exposes a command line client"
        ));
        // Whole-token matching: "not" must not fire inside "notation"/"another".
        assert!(!has_opposition_marker("another notation for annotation"));
    }
}
