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
//! 4. Returns the top-N tokens as concept strings.
//!
//! Multi-word concepts (e.g. "memory safety") are not extracted by this MVP;
//! a future n-gram or embedding-similarity pass can layer on top.

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

/// Extract up to `max` concept strings from `text`.
///
/// Concepts are lowercase single-word tokens ranked by term-frequency within
/// the text, with a small length bonus. Stopwords, tokens shorter than 3
/// characters, and pure-digit tokens are excluded. The returned concepts are
/// deduplicated and ordered by descending score.
///
/// Returns an empty `Vec` if `text` is empty or `max` is 0.
pub fn extract_concepts(text: &str, max: usize) -> Vec<String> {
    if max == 0 || text.trim().is_empty() {
        return Vec::new();
    }

    // Build a stopword lookup. This is small (~170 entries) so a linear
    // scan would be fine, but a HashSet is cleaner and handles the "is
    // this a stopword?" check in O(1).
    let stopwords: std::collections::HashSet<&str> = STOPWORDS.iter().copied().collect();

    // Tokenize: lowercase, split on non-alphanumeric, filter by length and
    // stopword status.
    let mut tf: HashMap<String, f32> = HashMap::new();
    for token in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
        let t = token.trim();
        if t.len() < 3 {
            continue;
        }
        // Skip pure-digit tokens (dates, numbers) — they are rarely useful
        // concept labels.
        if t.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if stopwords.contains(t) {
            continue;
        }
        // Score: term frequency + a small length bonus so that longer, more
        // specific tokens ("concurrency") rank above shorter generic ones
        // ("cpu") even at equal TF.
        let len_bonus = (t.len() as f32 - 3.0).max(0.0) * 0.1;
        *tf.entry(t.to_string()).or_insert(0.0) += 1.0 + len_bonus;
    }

    let mut scored: Vec<(String, f32)> = tf.into_iter().collect();
    // Sort by descending score, then alphabetically for determinism.
    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    scored.into_iter().take(max).map(|(s, _)| s).collect()
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
    let mut result: Vec<String> = Vec::with_capacity(max);
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Caller concepts first, normalized to lowercase.
    for c in caller_concepts {
        let norm = c.to_lowercase();
        if !norm.is_empty() && seen.insert(norm.clone()) {
            result.push(norm);
        }
        if result.len() >= max {
            return result;
        }
    }

    // Fill the remaining slots with extracted concepts.
    let remaining = max.saturating_sub(result.len());
    if remaining > 0 {
        for c in extract_concepts(text, remaining * 2) {
            if seen.insert(c.clone()) {
                result.push(c);
            }
            if result.len() >= max {
                break;
            }
        }
    }

    result
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
}
