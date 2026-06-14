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
    term_idf: HashMap<String, f32>,
    avg_doc_len: f32,
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

    /// Add or replace a document in the index.
    pub fn add(&mut self, id: &str, text: &str) {
        let tokens = tokenize(text);
        self.docs.insert(id.to_string(), tokens);
        self.recompute();
    }

    /// Remove a document.
    pub fn remove(&mut self, id: &str) {
        self.docs.remove(id);
        self.recompute();
    }

    fn recompute(&mut self) {
        let n = self.docs.len().max(1) as f32;
        let mut df: HashMap<String, usize> = HashMap::new();
        let mut total_len = 0usize;
        for tokens in self.docs.values() {
            total_len += tokens.len();
            let mut seen = std::collections::HashSet::new();
            for t in tokens {
                seen.insert(t.clone());
            }
            for t in seen {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        self.avg_doc_len = if self.docs.is_empty() {
            1.0
        } else {
            total_len as f32 / self.docs.len() as f32
        };
        self.term_idf.clear();
        for (term, f) in df {
            let idf = ((n - f as f32 + 0.5) / (f as f32 + 0.5) + 1.0).ln();
            self.term_idf.insert(term, idf);
        }
    }

    /// Score a query against all indexed documents. Returns (id, score) pairs sorted descending.
    pub fn score(&self, query: &str) -> Vec<(String, f32)> {
        let qtokens = tokenize(query);
        if qtokens.is_empty() || self.docs.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<String, f32> = HashMap::new();
        for (id, tokens) in &self.docs {
            let doc_len = tokens.len() as f32;
            let mut tf_map: HashMap<&str, usize> = HashMap::new();
            for t in tokens {
                *tf_map.entry(t.as_str()).or_insert(0) += 1;
            }
            let mut score = 0.0f32;
            for qt in &qtokens {
                if let Some(&idf) = self.term_idf.get(qt) {
                    let tf = *tf_map.get(qt.as_str()).unwrap_or(&0) as f32;
                    let denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / self.avg_doc_len);
                    score += idf * (tf * (self.k1 + 1.0)) / denom;
                }
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
