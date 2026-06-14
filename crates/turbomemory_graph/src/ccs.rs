//! Compressed Cognitive State (CCS) — deterministic MVP stub.
//!
//! The real ACC loop would call an LLM-based Cognitive Compressor Model (CCM)
//! under a schema constraint. For the MVP we maintain a bounded structured
//! state that evolves deterministically from the interaction stream.

use serde::{Deserialize, Serialize};

const MAX_FACTS: usize = 8;
const MAX_TOPICS: usize = 6;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompressedCognitiveState {
    pub turn_count: usize,
    pub last_user_input: String,
    pub last_assistant_response: String,
    pub facts: Vec<String>,
    pub topics: Vec<String>,
}

impl CompressedCognitiveState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn step(&mut self, user_input: &str, assistant_response: &str) {
        self.turn_count += 1;
        self.last_user_input = user_input.to_string();
        self.last_assistant_response = assistant_response.to_string();

        // Extract simple topic tokens from user input
        let new_topics: Vec<String> = user_input
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();
        for t in new_topics {
            if !self.topics.contains(&t) {
                self.topics.push(t);
            }
        }
        if self.topics.len() > MAX_TOPICS {
            self.topics.drain(0..self.topics.len() - MAX_TOPICS);
        }

        // Add a bounded fact summarizing the latest exchange
        let fact = format!(
            "Turn {}: user asked about '{}'; assistant responded about '{}'.",
            self.turn_count,
            truncate(user_input, 40),
            truncate(assistant_response, 40)
        );
        self.facts.push(fact);
        if self.facts.len() > MAX_FACTS {
            self.facts.remove(0);
        }
    }

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        s.chars().take(max_len).collect::<String>() + "..."
    }
}

/// Convenience wrapper used by the Python binding.
pub fn step_session(ccs_json: Option<&str>, user_input: &str, assistant_response: &str) -> String {
    let mut ccs = ccs_json
        .and_then(|s| serde_json::from_str::<CompressedCognitiveState>(s).ok())
        .unwrap_or_default();
    ccs.step(user_input, assistant_response);
    ccs.to_json()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccs_updates_and_serializes() {
        let json = step_session(None, "What is Rust?", "Rust is a safe systems language.");
        assert!(!json.is_empty());
        let ccs = serde_json::from_str::<CompressedCognitiveState>(&json).unwrap();
        assert_eq!(ccs.turn_count, 1);
        assert!(ccs.topics.iter().any(|t| t == "rust"));
    }
}
