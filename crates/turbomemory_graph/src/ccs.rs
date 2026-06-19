//! Compressed Cognitive State (CCS) — the agent's working-memory stub.
//!
//! The CCS is a bounded, schema-governed summary of the current interaction
//! that persists across turns. It is the "working memory" layer that sits
//! *above* the long-term memory graph: each turn, the compressor distills
//! the user input + assistant response into the CCS, which then guides
//! future retrieval and generation.
//!
//! ## Compressor trait
//!
//! Compression is pluggable via the [`CognitiveCompressor`] trait. Two
//! implementations ship:
//!
//! - [`DeterministicCompressor`]: the MVP — a fast, deterministic
//!   keyword/topic extractor with a FIFO fact buffer. No external calls.
//! - [`LlmCompressor`]: calls a user-supplied closure (typically an LLM
//!   API call) that receives the current CCS JSON + the turn's user input
//!   and the assistant response, and returns the new CCS JSON. This makes
//!   the README's "an LLM-based compressor can be plugged in" claim true.
//!
//! The engine defaults to [`DeterministicCompressor`]; callers can install
//! an [`LlmCompressor`] to get LLM-driven compression.

use serde::{Deserialize, Serialize};

const MAX_FACTS: usize = 8;
const MAX_TOPICS: usize = 6;

/// The bounded working-memory state carried across turns.
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

    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

/// A cognitive compressor: distills a turn (user input + assistant response)
/// into the working-memory state.
///
/// Implementations may be deterministic (fast, no I/O) or call an external
/// model (LLM, summarizer). The compressor receives the *current* CCS and
/// the turn's raw text, and returns the *new* CCS.
///
/// Implementations are called on the engine's `step_session` path. If the
/// compressor performs I/O (e.g. an LLM API call), the caller is responsible
/// for ensuring it does not block the engine's locks — the engine calls
/// `compress` without holding any internal lock, so async/wrapper compressors
/// are safe.
pub trait CognitiveCompressor: Send + Sync {
    fn compress(
        &self,
        ccs: &CompressedCognitiveState,
        user_input: &str,
        assistant_response: &str,
    ) -> CompressedCognitiveState;
}

/// The default deterministic compressor: no external calls, no LLM.
///
/// Extracts topic tokens from the user input (lowercase, split on
/// non-alphanumeric, length > 2) and appends a templated fact string
/// summarizing the turn. Facts follow a FIFO ring buffer (max
/// [`MAX_FACTS`]); topics follow a FIFO ring buffer (max [`MAX_TOPICS`]).
///
/// This is the compressor that was hardcoded in the original MVP; it is
/// now one of two pluggable implementations.
#[derive(Debug, Clone, Default)]
pub struct DeterministicCompressor;

impl CognitiveCompressor for DeterministicCompressor {
    fn compress(
        &self,
        ccs: &CompressedCognitiveState,
        user_input: &str,
        assistant_response: &str,
    ) -> CompressedCognitiveState {
        let mut new_ccs = ccs.clone();
        new_ccs.turn_count += 1;
        new_ccs.last_user_input = user_input.to_string();
        new_ccs.last_assistant_response = assistant_response.to_string();

        // Extract simple topic tokens from user input.
        let new_topics: Vec<String> = user_input
            .to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() > 2)
            .map(|s| s.to_string())
            .collect();
        for t in new_topics {
            if !new_ccs.topics.contains(&t) {
                new_ccs.topics.push(t);
            }
        }
        if new_ccs.topics.len() > MAX_TOPICS {
            new_ccs.topics.drain(0..new_ccs.topics.len() - MAX_TOPICS);
        }

        // Add a bounded fact summarizing the latest exchange.
        let fact = format!(
            "Turn {}: user asked about '{}'; assistant responded about '{}'.",
            new_ccs.turn_count,
            truncate(user_input, 40),
            truncate(assistant_response, 40)
        );
        new_ccs.facts.push(fact);
        if new_ccs.facts.len() > MAX_FACTS {
            new_ccs.facts.remove(0);
        }

        new_ccs
    }
}

/// An LLM-backed cognitive compressor.
///
/// Wraps a user-supplied closure that receives `(current_ccs_json,
/// user_input, assistant_response)` and returns the new CCS JSON. The
/// closure is expected to call an LLM (or any external summarizer) that
/// produces a `CompressedCognitiveState`-compatible JSON string.
///
/// If the closure returns invalid JSON, the compressor falls back to the
/// [`DeterministicCompressor`] so a malformed LLM response never corrupts
/// the working memory.
///
/// # Example
/// ```ignore
/// use turbomemory_graph::ccs::{LlmCompressor, CognitiveCompressor};
///
/// let compressor = LlmCompressor::new(|ccs_json, user, assistant| {
///     // Call your LLM here, return new CCS JSON.
///     call_my_llm(&format!(
///         "Current state: {ccs_json}\nUser: {user}\nAssistant: {assistant}\n\
///          Produce updated CCS JSON."
///     ))
/// });
/// ```
pub struct LlmCompressor<F>
where
    F: Fn(&str, &str, &str) -> String + Send + Sync,
{
    caller: F,
    fallback: DeterministicCompressor,
}

impl<F> LlmCompressor<F>
where
    F: Fn(&str, &str, &str) -> String + Send + Sync,
{
    pub fn new(caller: F) -> Self {
        Self {
            caller,
            fallback: DeterministicCompressor,
        }
    }
}

impl<F> CognitiveCompressor for LlmCompressor<F>
where
    F: Fn(&str, &str, &str) -> String + Send + Sync,
{
    fn compress(
        &self,
        ccs: &CompressedCognitiveState,
        user_input: &str,
        assistant_response: &str,
    ) -> CompressedCognitiveState {
        let ccs_json = ccs.to_json();
        let new_json = (self.caller)(&ccs_json, user_input, assistant_response);
        // Try to parse the LLM output as a CCS. Fall back to the deterministic
        // compressor if the output is not valid CCS JSON — a malformed LLM
        // response must never corrupt the working memory.
        match serde_json::from_str::<CompressedCognitiveState>(&new_json) {
            Ok(parsed) => parsed,
            Err(_) => {
                // Fall back: run the deterministic compressor so the turn is
                // still recorded, just without the LLM's insight.
                self.fallback.compress(ccs, user_input, assistant_response)
            }
        }
    }
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        s.to_string()
    } else {
        s.chars().take(max_len).collect::<String>() + "..."
    }
}

/// Convenience wrapper used by the Python binding and REST/gRPC servers.
///
/// Uses the [`DeterministicCompressor`] (the default). To use an
/// [`LlmCompressor`], call `step_session_with_compressor` instead.
pub fn step_session(ccs_json: Option<&str>, user_input: &str, assistant_response: &str) -> String {
    let compressor = DeterministicCompressor;
    step_session_with_compressor(&compressor, ccs_json, user_input, assistant_response)
}

/// Step the session with an explicit compressor. Used by the engine when a
/// custom (e.g. LLM) compressor is installed.
pub fn step_session_with_compressor(
    compressor: &dyn CognitiveCompressor,
    ccs_json: Option<&str>,
    user_input: &str,
    assistant_response: &str,
) -> String {
    let ccs = ccs_json
        .and_then(|s| serde_json::from_str::<CompressedCognitiveState>(s).ok())
        .unwrap_or_default();
    let new_ccs = compressor.compress(&ccs, user_input, assistant_response);
    new_ccs.to_json()
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

    #[test]
    fn deterministic_compressor_increments_turn() {
        let comp = DeterministicCompressor;
        let ccs = CompressedCognitiveState::default();
        let new = comp.compress(&ccs, "hello world", "hi there");
        assert_eq!(new.turn_count, 1);
        assert_eq!(new.last_user_input, "hello world");
        assert_eq!(new.last_assistant_response, "hi there");
    }

    #[test]
    fn deterministic_compressor_facts_are_fifo_bounded() {
        let comp = DeterministicCompressor;
        let mut ccs = CompressedCognitiveState::default();
        for i in 0..(MAX_FACTS + 3) {
            ccs = comp.compress(&ccs, &format!("q{i}"), &format!("a{i}"));
        }
        assert!(ccs.facts.len() <= MAX_FACTS);
        // The oldest facts should have been evicted; the latest should mention
        // the last turn number.
        let last_turn = MAX_FACTS + 3;
        assert!(ccs
            .facts
            .last()
            .unwrap()
            .contains(&format!("Turn {last_turn}")));
    }

    #[test]
    fn deterministic_compressor_topics_dedup_and_bound() {
        let comp = DeterministicCompressor;
        let mut ccs = CompressedCognitiveState::default();
        // Repeatedly mention "rust" — should only appear once.
        for _ in 0..5 {
            ccs = comp.compress(&ccs, "rust rust rust", "safe");
        }
        let rust_count = ccs.topics.iter().filter(|t| *t == "rust").count();
        assert_eq!(rust_count, 1);
        assert!(ccs.topics.len() <= MAX_TOPICS);
    }

    #[test]
    fn llm_compressor_uses_closure_output() {
        // A mock "LLM" that returns a fixed CCS JSON with a custom fact.
        let compressor = LlmCompressor::new(|_ccs, user, _assistant| {
            serde_json::to_string(&CompressedCognitiveState {
                turn_count: 42,
                last_user_input: user.to_string(),
                last_assistant_response: "mock response".into(),
                facts: vec!["LLM-generated fact".into()],
                topics: vec!["custom".into()],
            })
            .unwrap()
        });
        let result =
            step_session_with_compressor(&compressor, None, "what is rust", "rust is safe");
        let ccs = serde_json::from_str::<CompressedCognitiveState>(&result).unwrap();
        assert_eq!(ccs.turn_count, 42);
        assert!(ccs.facts.contains(&"LLM-generated fact".to_string()));
        assert!(ccs.topics.contains(&"custom".to_string()));
    }

    #[test]
    fn llm_compressor_falls_back_on_invalid_json() {
        // A mock "LLM" that returns garbage. The compressor should fall back
        // to the deterministic compressor so the turn is still recorded.
        let compressor = LlmCompressor::new(|_ccs, _u, _a| "not valid json {{{".to_string());
        let result =
            step_session_with_compressor(&compressor, None, "what is rust", "rust is safe");
        let ccs = serde_json::from_str::<CompressedCognitiveState>(&result).unwrap();
        // Fallback: turn_count should be 1 (first turn via deterministic).
        assert_eq!(ccs.turn_count, 1);
        // Fallback: topics should come from deterministic extraction.
        assert!(ccs.topics.iter().any(|t| t == "rust"));
    }

    #[test]
    fn llm_compressor_preserves_state_across_turns() {
        let compressor = LlmCompressor::new(|ccs_json, _u, _a| {
            // Parse the incoming CCS, increment turn_count, add a fact.
            let mut ccs: CompressedCognitiveState =
                serde_json::from_str(ccs_json).unwrap_or_default();
            ccs.turn_count += 1;
            ccs.facts.push(format!("LLM turn {}", ccs.turn_count));
            serde_json::to_string(&ccs).unwrap()
        });
        let json1 = step_session_with_compressor(&compressor, None, "q1", "a1");
        let json2 = step_session_with_compressor(&compressor, Some(&json1), "q2", "a2");
        let ccs2 = serde_json::from_str::<CompressedCognitiveState>(&json2).unwrap();
        assert_eq!(ccs2.turn_count, 2);
        assert!(ccs2.facts.contains(&"LLM turn 2".to_string()));
    }
}
