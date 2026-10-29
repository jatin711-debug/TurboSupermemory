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
//! do not count against each other however alike they read. Two candidates
//! said on different occasions ([`Candidate::occasion`]) are never passed
//! over as duplicates of each other, though they still weigh as redundant.
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

/// What the packer knows about a candidate besides its vector and score.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Candidate {
    /// Tokens it takes in the context.
    pub tokens: u32,
    /// The turn it is from: the first candidate of a turn gets
    /// `new_turn_bonus`.
    pub turn: Option<i64>,
    /// When it was said (a day). The same words said on two days are two
    /// occurrences, and a count or a date question needs both: two
    /// candidates whose occasions are both known and differ are never
    /// skipped as near-duplicates of each other.
    pub occasion: Option<i64>,
}

impl Candidate {
    fn same_occasion(&self, other: &Candidate) -> bool {
        match (self.occasion, other.occasion) {
            (Some(a), Some(b)) => a == b,
            _ => true,
        }
    }
}

/// The candidates to keep, in the order they were chosen.
///
/// `vectors` holds one row of `dim` values per candidate (any length; each
/// row is scaled to unit length here, and an all-zero row is similar to
/// nothing). `scores` and `candidates` are parallel to the rows.
pub fn select_under_budget(
    vectors: &[f32],
    dim: usize,
    scores: &[f32],
    candidates: &[Candidate],
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
    if candidates.len() != n {
        return Err(TurboError::InvalidArgument(
            "scores and candidates must be the same length".into(),
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
    // Highest similarity to a chosen candidate that counts, per candidate,
    // and the same among the chosen said on the same occasion (what the
    // near-duplicate skip looks at).
    let mut redundancy: Vec<Option<f32>> = vec![None; n];
    let mut duplicate: Vec<Option<f32>> = vec![None; n];
    let mut used: u64 = 0;

    while !remaining.is_empty() && selected.len() < cap {
        let mut best: Option<usize> = None;
        let mut best_gain = -1e9_f64;
        for &i in &remaining {
            if used + u64::from(candidates[i].tokens) > params.token_budget {
                continue;
            }
            if duplicate[i].map_or(0.0, f64::from) > params.max_redundancy {
                continue;
            }
            let red = redundancy[i].map_or(0.0, f64::from);
            let bonus = match candidates[i].turn {
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
        used += u64::from(candidates[chosen].tokens);
        if let Some(t) = candidates[chosen].turn {
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
            if candidates[i].same_occasion(&candidates[chosen]) {
                duplicate[i] = Some(duplicate[i].map_or(sim, |r| r.max(sim)));
            }
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

    fn cands(tokens: &[u32], turns: &[Option<i64>]) -> Vec<Candidate> {
        tokens
            .iter()
            .zip(turns)
            .map(|(&tokens, &turn)| Candidate {
                tokens,
                turn,
                occasion: None,
            })
            .collect()
    }

    #[test]
    fn takes_the_most_relevant_that_fit() {
        // Orthogonal rows: no redundancy, no turns; relevance order, budget-bound.
        let vectors = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let c = cands(&[10, 10, 10], &[None, None, None]);
        let got = select_under_budget(&vectors, 3, &[0.5, 0.9, 0.7], &c, &[], &params(20));
        assert_eq!(got.unwrap(), vec![1, 2]);
    }

    #[test]
    fn a_near_duplicate_is_skipped_and_a_different_turn_preferred() {
        // 0 and 1 say the same thing; 2 is less relevant but from a new turn.
        let vectors = [1.0, 0.0, 0.999, 0.0447, 0.0, 1.0];
        let c = cands(&[5, 5, 5], &[Some(1), Some(1), Some(2)]);
        let got = select_under_budget(&vectors, 2, &[0.9, 0.89, 0.6], &c, &[], &params(100));
        assert_eq!(got.unwrap(), vec![0, 2], "the duplicate of 0 never gets in");
    }

    #[test]
    fn the_same_words_on_another_day_are_another_occurrence() {
        // 0 and 1 read the same. Said on one day, 1 is a duplicate; said on
        // two days, both are kept (after 2, as 1 still weighs as redundant).
        let vectors = [1.0, 0.0, 0.999, 0.0447, 0.0, 1.0];
        let scores = [0.9, 0.89, 0.6];
        let mut c = cands(&[5, 5, 5], &[Some(1), Some(2), Some(3)]);
        for (k, day) in [Some(10), Some(10), None].into_iter().enumerate() {
            c[k].occasion = day;
        }
        let one_day = select_under_budget(&vectors, 2, &scores, &c, &[], &params(100));
        assert_eq!(one_day.unwrap(), vec![0, 2]);
        c[1].occasion = Some(11);
        let two_days = select_under_budget(&vectors, 2, &scores, &c, &[], &params(100));
        assert_eq!(two_days.unwrap(), vec![0, 2, 1]);
    }

    #[test]
    fn a_replaced_memory_is_not_redundant_with_its_replacement() {
        let vectors = [1.0, 0.0, 1.0, 0.0];
        let c = cands(&[5, 5], &[None, None]);
        let without = select_under_budget(&vectors, 2, &[0.9, 0.8], &c, &[], &params(100));
        assert_eq!(without.unwrap(), vec![0]);
        let with = select_under_budget(&vectors, 2, &[0.9, 0.8], &c, &[(1, 0)], &params(100));
        assert_eq!(with.unwrap(), vec![0, 1]);
    }

    #[test]
    fn a_tie_within_rounding_goes_to_the_earlier_candidate() {
        // 1 is ahead of 0 by less than GAIN_TIE: still a tie, 0 is taken.
        let vectors = [1.0, 0.0, 1.0, 0.0];
        let c = cands(&[5, 5], &[None, None]);
        let got = select_under_budget(&vectors, 2, &[0.5, 0.5 + 1e-7], &c, &[], &params(5));
        assert_eq!(got.unwrap(), vec![0]);
    }

    #[test]
    fn max_items_and_zero_rows() {
        let vectors = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let c = cands(&[1, 1, 1], &[None, None, None]);
        let capped = PackParams {
            token_budget: 100,
            max_items: Some(2),
            ..PackParams::default()
        };
        let got = select_under_budget(&vectors, 2, &[0.9, 0.8, 0.7], &c, &[], &capped);
        assert_eq!(got.unwrap(), vec![0, 1]);
    }

    #[test]
    fn rejects_mismatched_input() {
        let one = cands(&[1], &[None]);
        assert!(select_under_budget(&[1.0], 2, &[0.5], &one, &[], &params(9)).is_err());
        assert!(select_under_budget(&[1.0, 0.0], 2, &[0.5], &one, &[(0, 1)], &params(9)).is_err());
        assert!(select_under_budget(&[1.0, 0.0], 2, &[0.5], &[], &[], &params(9)).is_err());
    }
}
