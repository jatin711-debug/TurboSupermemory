//! Choosing the best set of memories that fits a token budget.
//!
//! This is the selection `tsm.budget.select_under_budget` makes (greedy
//! maximal marginal relevance with a bonus for each turn not yet covered),
//! run on vectors the caller already has: the engine's stored embeddings,
//! not a second embedding of every candidate's text. Each step adds the
//! candidate that fits and maximizes
//!
//! ```text
//! lam * relevance - (1 - lam) * redundancy + new_turn_bonus
//! ```
//!
//! where redundancy is the highest cosine to anything already chosen (0 when
//! nothing chosen counts), a candidate more redundant than `max_redundancy`
//! is passed over, and the bonus goes to the first candidate of a turn. Two
//! candidates in `never_redundant` (a memory and the one that replaced it)
//! do not count against each other however alike they read.
//!
//! The Python version recomputes each candidate's redundancy against the
//! whole chosen set at every step. Here it is kept up to date as candidates
//! are chosen, so a step costs one pass over the remaining candidates.
//! Gains within [`GAIN_TIE`] of each other are a tie and go to the earlier
//! candidate, as there: the two round a cosine differently in its last bit.

use std::collections::HashSet;

use crate::{metrics::dot_product, Result, TurboError};

/// Gains closer than this are a tie (`tsm.budget.GAIN_TIE`).
pub const GAIN_TIE: f64 = 1e-6;

/// How a set is chosen. The defaults are `tsm.budget`'s.
#[derive(Debug, Clone, Copy)]
pub struct PackParams {
    /// Total tokens the chosen candidates may take.
    pub token_budget: u64,
    /// Weight of relevance against redundancy.
    pub lam: f64,
    /// At most this many candidates (`None`: only the budget limits).
    pub max_items: Option<usize>,
    /// A candidate more similar than this to something chosen is skipped.
    pub max_redundancy: f64,
    /// Added to the gain of the first candidate from a turn.
    pub new_turn_bonus: f64,
}

impl Default for PackParams {
    fn default() -> Self {
        Self {
            token_budget: 0,
            lam: 0.7,
            max_items: None,
            max_redundancy: 0.95,
            new_turn_bonus: 0.20,
        }
    }
}

/// The candidates to keep, in the order they were chosen.
///
/// `vectors` holds one row of `dim` values per candidate (any length; each
/// row is scaled to unit length here, and an all-zero row is similar to
/// nothing). `scores`, `tokens` and `turns` are parallel to the rows.
pub fn select_under_budget(
    vectors: &[f32],
    dim: usize,
    scores: &[f32],
    tokens: &[u32],
    turns: &[Option<i64>],
    never_redundant: &[(usize, usize)],
    params: &PackParams,
) -> Result<Vec<usize>> {
    let n = scores.len();
    if dim == 0 || vectors.len() != n * dim {
        return Err(TurboError::InvalidArgument(format!(
            "expected {n} vectors of {dim} values, got {} values",
            vectors.len()
        )));
    }
    if tokens.len() != n || turns.len() != n {
        return Err(TurboError::InvalidArgument(
            "scores, tokens and turns must be the same length".into(),
        ));
    }
    if let Some(&(a, b)) = never_redundant.iter().find(|(a, b)| *a >= n || *b >= n) {
        return Err(TurboError::InvalidArgument(format!(
            "never_redundant pair ({a}, {b}) is out of range for {n} candidates"
        )));
    }

    // Unit rows, as `embs / (norm + 1e-9)` does in Python.
    let mut unit = vectors.to_vec();
    for row in unit.chunks_exact_mut(dim) {
        let norm = dot_product(row, row).sqrt();
        let scale = 1.0 / (norm + 1e-9);
        for x in row.iter_mut() {
            *x *= scale;
        }
    }
    let row = |i: usize| &unit[i * dim..(i + 1) * dim];
    let paired: HashSet<(usize, usize)> = never_redundant
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .collect();

    let cap = params.max_items.filter(|&m| m > 0).unwrap_or(n);
    let mut selected: Vec<usize> = Vec::new();
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut covered: HashSet<i64> = HashSet::new();
    // Highest similarity to a chosen candidate that counts, per candidate.
    let mut redundancy: Vec<Option<f32>> = vec![None; n];
    let mut used: u64 = 0;

    while !remaining.is_empty() && selected.len() < cap {
        let mut best: Option<usize> = None;
        let mut best_gain = -1e9_f64;
        for &i in &remaining {
            if used + u64::from(tokens[i]) > params.token_budget {
                continue;
            }
            let red = redundancy[i].map_or(0.0, f64::from);
            if red > params.max_redundancy {
                continue;
            }
            let bonus = match turns[i] {
                Some(t) if !covered.contains(&t) => params.new_turn_bonus,
                _ => 0.0,
            };
            let gain = params.lam * f64::from(scores[i]) - (1.0 - params.lam) * red + bonus;
            if gain > best_gain + GAIN_TIE {
                best_gain = gain;
                best = Some(i);
            }
        }
        let Some(chosen) = best else {
            break; // nothing else fits the budget or the redundancy limit
        };
        selected.push(chosen);
        used += u64::from(tokens[chosen]);
        if let Some(t) = turns[chosen] {
            covered.insert(t);
        }
        remaining.retain(|&i| i != chosen);
        let chosen_row = row(chosen);
        for &i in &remaining {
            if paired.contains(&(i, chosen)) {
                continue;
            }
            let sim = dot_product(row(i), chosen_row);
            redundancy[i] = Some(redundancy[i].map_or(sim, |r| r.max(sim)));
        }
    }
    Ok(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(budget: u64) -> PackParams {
        PackParams {
            token_budget: budget,
            ..PackParams::default()
        }
    }

    #[test]
    fn takes_the_most_relevant_that_fit() {
        // Orthogonal rows: no redundancy, no turns; relevance order, budget-bound.
        let vectors = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let got = select_under_budget(
            &vectors,
            3,
            &[0.5, 0.9, 0.7],
            &[10, 10, 10],
            &[None, None, None],
            &[],
            &params(20),
        )
        .unwrap();
        assert_eq!(got, vec![1, 2]);
    }

    #[test]
    fn a_near_duplicate_is_skipped_and_a_different_turn_preferred() {
        // 0 and 1 say the same thing; 2 is less relevant but from a new turn.
        let vectors = [1.0, 0.0, 0.999, 0.0447, 0.0, 1.0];
        let got = select_under_budget(
            &vectors,
            2,
            &[0.9, 0.89, 0.6],
            &[5, 5, 5],
            &[Some(1), Some(1), Some(2)],
            &[],
            &params(100),
        )
        .unwrap();
        assert_eq!(got, vec![0, 2], "the duplicate of 0 never gets in");
    }

    #[test]
    fn a_replaced_memory_is_not_redundant_with_its_replacement() {
        let vectors = [1.0, 0.0, 1.0, 0.0];
        let without = select_under_budget(
            &vectors,
            2,
            &[0.9, 0.8],
            &[5, 5],
            &[None, None],
            &[],
            &params(100),
        )
        .unwrap();
        assert_eq!(without, vec![0]);
        let with = select_under_budget(
            &vectors,
            2,
            &[0.9, 0.8],
            &[5, 5],
            &[None, None],
            &[(1, 0)],
            &params(100),
        )
        .unwrap();
        assert_eq!(with, vec![0, 1]);
    }

    #[test]
    fn a_tie_within_rounding_goes_to_the_earlier_candidate() {
        // 1 is ahead of 0 by less than GAIN_TIE: still a tie, 0 is taken.
        let vectors = [1.0, 0.0, 1.0, 0.0];
        let scores = [0.5, 0.5 + 1e-7];
        let got = select_under_budget(
            &vectors,
            2,
            &scores,
            &[5, 5],
            &[None, None],
            &[],
            &params(5),
        )
        .unwrap();
        assert_eq!(got, vec![0]);
    }

    #[test]
    fn max_items_and_zero_rows() {
        let vectors = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let got = select_under_budget(
            &vectors,
            2,
            &[0.9, 0.8, 0.7],
            &[1, 1, 1],
            &[None, None, None],
            &[],
            &PackParams {
                token_budget: 100,
                max_items: Some(2),
                ..PackParams::default()
            },
        )
        .unwrap();
        assert_eq!(got, vec![0, 1]);
    }

    #[test]
    fn rejects_mismatched_input() {
        assert!(select_under_budget(&[1.0], 2, &[0.5], &[1], &[None], &[], &params(9)).is_err());
        assert!(
            select_under_budget(&[1.0, 0.0], 2, &[0.5], &[1], &[None], &[(0, 1)], &params(9))
                .is_err()
        );
    }
}
