//! A minimal BM25 implementation for lexical memory triggers.

use std::collections::HashMap;

/// Tokenize by lowercasing and splitting on non-alphanumeric characters.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|s| s.len() > 1)
        .map(|s| s.to_string())
        .collect()
}

/// In-memory BM25 index over a set of documents keyed by an arbitrary ID type.
#[derive(Debug, Clone, Default)]
pub struct Bm25Index {
    docs: HashMap<String, Vec<String>>,
    /// Document frequency for each term (number of documents containing the term).
    term_doc_freq: HashMap<String, usize>,
    /// Total number of tokens across all documents.
    total_doc_len: usize,
    k1: f32,
    b: f32,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self {
            k1: 1.2,
            b: 0.75,
            ..Default::default()
        }
    }

    fn avg_doc_len(&self) -> f32 {
        if self.docs.is_empty() {
            1.0
        } else {
            self.total_doc_len as f32 / self.docs.len() as f32
        }
    }

    fn idf(&self, term: &str) -> f32 {
        let n = self.docs.len().max(1) as f32;
        let f = *self.term_doc_freq.get(term).unwrap_or(&0) as f32;
        ((n - f + 0.5) / (f + 0.5) + 1.0).ln()
    }

    /// Increment the document frequency for each distinct term in `tokens`.
    fn add_term_freqs(&mut self, tokens: &[String]) {
        let mut seen = std::collections::HashSet::new();
        for t in tokens {
            if seen.insert(t.clone()) {
                *self.term_doc_freq.entry(t.clone()).or_insert(0) += 1;
            }
        }
    }

    /// Decrement the document frequency for each distinct term in `tokens`.
    fn remove_term_freqs(&mut self, tokens: &[String]) {
        let mut seen = std::collections::HashSet::new();
        for t in tokens {
            if !seen.insert(t.clone()) {
                continue;
            }
            if let Some(count) = self.term_doc_freq.get_mut(t) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.term_doc_freq.remove(t);
                }
            }
        }
    }

    /// Add or replace a document in the index.
    pub fn add(&mut self, id: &str, text: &str) {
        let tokens = tokenize(text);
        // If the document already exists, remove its old contribution first.
        if let Some(old) = self.docs.remove(id) {
            self.remove_term_freqs(&old);
            self.total_doc_len = self.total_doc_len.saturating_sub(old.len());
        }
        self.add_term_freqs(&tokens);
        self.total_doc_len += tokens.len();
        self.docs.insert(id.to_string(), tokens);
    }

    /// Remove a document.
    pub fn remove(&mut self, id: &str) {
        if let Some(old) = self.docs.remove(id) {
            self.remove_term_freqs(&old);
            self.total_doc_len = self.total_doc_len.saturating_sub(old.len());
        }
    }

    /// Force a full recomputation of term frequencies. Useful for tests and
    /// recovery; not used on the insert hot path.
    pub fn recompute(&mut self) {
        self.term_doc_freq.clear();
        self.total_doc_len = 0;
        let all_tokens: Vec<Vec<String>> = self.docs.values().cloned().collect();
        for tokens in all_tokens {
            self.add_term_freqs(&tokens);
            self.total_doc_len += tokens.len();
        }
    }

    /// Score a query against all indexed documents. Returns (id, score) pairs sorted descending.
    pub fn score(&self, query: &str) -> Vec<(String, f32)> {
        let qtokens = tokenize(query);
        if qtokens.is_empty() || self.docs.is_empty() {
            return Vec::new();
        }
        let avg_doc_len = self.avg_doc_len();
        let mut scores: HashMap<String, f32> = HashMap::new();
        for (id, tokens) in &self.docs {
            let doc_len = tokens.len() as f32;
            let mut tf_map: HashMap<&str, usize> = HashMap::new();
            for t in tokens {
                *tf_map.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut score = 0.0f32;
            for qt in &qtokens {
                let idf = self.idf(qt);
                let tf = *tf_map.get(qt.as_str()).unwrap_or(&0) as f32;
                let denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / avg_doc_len);
                score += idf * (tf * (self.k1 + 1.0)) / denom;
            }
            if score > 0.0 {
                scores.insert(id.clone(), score);
            }
        }
        let mut out: Vec<_> = scores.into_iter().collect();
        out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_basic() {
        let mut idx = Bm25Index::new();
        idx.add("m1", "The Rust programming language is fast");
        idx.add("m2", "Python is great for AI and data science");
        let results = idx.score("Rust language");
        assert_eq!(results[0].0, "m1");
        assert!(results[0].1 > results.get(1).map(|(_, s)| *s).unwrap_or(0.0));
    }
}
