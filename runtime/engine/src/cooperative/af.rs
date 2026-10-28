use std::collections::{BTreeMap, BTreeSet};

use crate::link::{log_t_norm_for, BlockModel, LinkTable};
use crate::presence::{PresenceProfile, ResolvedPresence};
use crate::{BlockRangeProfile, CalibrationModel, LocalizationMode, PreparedProblem, RangeErrorProfile};

use crate::optimize::DifferentiableObjective;
use crate::Point2;

#[derive(Clone)]
pub struct Pair {
    pub left: usize,
    pub right: usize,
    pub observed: bool,
}

/// The range term of one agent pair, pooled over both directions and every repeated observation.
#[derive(Clone)]
pub struct Range {
    pub(crate) pair: usize,
    /// Median of the pooled packets, the measurement of the median residual density.
    pub(crate) measured: f64,
    /// Reported scale of the median packet, None for raw timing whose scale follows the candidate distance.
    fixed_scale: Option<f64>,
    /// The block likelihood table of the pooled packets when the profile selects the block range model.
    pub(crate) table: Option<LinkTable>,
}

pub struct Graph {
    pub agents: usize,
    pub pairs: Vec<Pair>,
    pub ranges: Vec<Range>,
    /// The range calibration of raw links, evaluated at the candidate distance by the median density.
    pub calibration: CalibrationModel,
    /// The presence law of the scenario, None when it has no successful link or the law is off.
    pub(crate) presence: Option<ResolvedPresence>,
    /// Per link median distance of every link with admissible packets, the input of the reach estimate.
    pub(crate) successful: Vec<f64>,
}

impl Graph {
    /// The graph with the law of `presence` resolved at the request's start positions, which only matter to the
    /// maximum likelihood reach.
    #[cfg(test)]
    pub fn new(problem: &PreparedProblem, block: Option<&BlockRangeProfile>, calibration: CalibrationModel, presence: &PresenceProfile) -> Self {
        let mut graph = Self::build(problem, block, calibration);
        graph.presence = graph.resolve_presence(presence, &problem.initial_positions);
        graph
    }

    /// The law of the scenario under the profile's reach rule, with the reach from the per link medians of the
    /// successful links. The maximum likelihood reach judges the outcomes of every opportunity at `layout`, from the
    /// counts of attempted and successful pairs.
    pub(crate) fn resolve_presence(&self, profile: &PresenceProfile, layout: &[Point2]) -> Option<ResolvedPresence> {
        let successes = self.pairs.iter().filter(|p| p.observed).count();
        profile.resolve_with(&self.successful, self.pairs.len(), successes, |law| self.outcome_log_likelihood(law, layout))
    }

    /// The log likelihood of the attempt outcomes of every opportunity under `law` at `layout`.
    pub(crate) fn outcome_log_likelihood(&self, law: &ResolvedPresence, layout: &[Point2]) -> f64 {
        self.pairs
            .iter()
            .map(|pair| law.log_probability(layout[pair.left].distance(layout[pair.right]), pair.observed).0)
            .sum()
    }

    /// The graph without a presence law.
    pub fn build(problem: &PreparedProblem, block: Option<&BlockRangeProfile>, calibration: CalibrationModel) -> Self {
        assert_eq!(problem.mode, LocalizationMode::AnchorFree);
        assert!(problem.anchors.is_empty());
        let block_model = block.map(BlockModel::from);
        let norm = log_t_norm_for(block_model.as_ref().map_or(4.0, |m| m.nu));
        let agents = problem.agent_ids.len();
        let indices: BTreeMap<_, _> = problem
            .agent_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        // Every observed pair with its packets pooled over both directions and repeated observations. An
        // observation is a responded exchange, so it marks the pair as present and joins the two agents in one
        // component even when none of its samples is admissible, in which case the pair has no range term.
        let pooled = super::pool_observations(problem, &indices, &BTreeMap::new(), &calibration);
        let components = super::FrameComponents::new(agents, pooled.keys().copied());
        // declared attempts restrict the opportunities to the attempted and the observed pairs
        let attempted: Option<BTreeSet<(usize, usize)>> = problem.attempts.as_ref().map(|attempts| {
            attempts
                .iter()
                .filter(|a| !a.to_is_anchor)
                .filter_map(|a| Some((*indices.get(&a.from)?, *indices.get(&a.to)?)))
                .map(|(l, r)| (l.min(r), l.max(r)))
                .collect()
        });
        let mut pairs = Vec::new();
        let mut pair_indices = BTreeMap::new();
        for left in 0..agents {
            for right in left + 1..agents {
                let observed = pooled.contains_key(&(left, right));
                let include = components.joined(left, right)
                    && attempted.as_ref().is_none_or(|set| observed || set.contains(&(left, right)));
                if include {
                    pair_indices.insert((left, right), pairs.len());
                    pairs.push(Pair { left, right, observed });
                }
            }
        }
        // One range per pooled link. The median of its packets serves the median residual density, taken in time
        // before the conversion for raw timing, and the table of every packet serves the block likelihood.
        let mut ranges = Vec::new();
        for (key, packets) in &pooled {
            let Some(packets) = packets else {
                continue;
            };
            let (measured, fixed_scale, distances, scales): (f64, Option<f64>, Vec<f64>, Vec<f64>) = match packets {
                super::Packets::Raw(tofs) => {
                    let distances: Vec<f64> = tofs.iter().map(|t| calibration.corrected_distance_cm(*t)).collect();
                    let scales = distances.iter().map(|d| calibration.scale_cm(*d)).collect();
                    (calibration.corrected_distance_cm(super::sorted_median(tofs)), None, distances, scales)
                }
                super::Packets::Calibrated(samples) => {
                    let mid = samples.len() / 2;
                    let (distance, scale) = if samples.len() % 2 == 1 {
                        samples[mid]
                    } else {
                        ((samples[mid - 1].0 + samples[mid].0) / 2.0, (samples[mid - 1].1 + samples[mid].1) / 2.0)
                    };
                    (distance, Some(scale), samples.iter().map(|s| s.0).collect(), samples.iter().map(|s| s.1).collect())
                }
            };
            let table = block_model.as_ref().map(|model| LinkTable::from_packets(&distances, &scales, model, norm));
            ranges.push(Range { pair: pair_indices[key], measured, fixed_scale, table });
        }
        let successful: Vec<f64> = ranges.iter().map(|r| r.measured).collect();
        Self {
            agents,
            pairs,
            ranges,
            calibration,
            presence: None,
            successful,
        }
    }

    pub fn point(&self, parameters: &[f64], index: usize) -> Point2 {
        Point2::new(parameters[2 * index], parameters[2 * index + 1])
    }

    pub fn typical_range(&self) -> f64 {
        if self.ranges.is_empty() {
            return 100.0;
        }
        let mut values: Vec<_> = self.ranges.iter().map(|r| r.measured).collect();
        values.sort_by(f64::total_cmp);
        let mid = values.len() / 2;
        (if values.len() % 2 == 1 {
            values[mid]
        } else {
            (values[mid - 1] + values[mid]) / 2.0
        })
        .max(1.0)
    }
}

pub struct Objective<'a> {
    pub graph: &'a Graph,
    pub range_error: &'a RangeErrorProfile,
}

impl Objective<'_> {
    /// Loss and its derivative in the predicted distance.
    fn range_loss(&self, factor: &Range, distance: f64) -> (f64, f64) {
        if let Some(table) = &factor.table {
            // block likelihood: the negative log likelihood of the packet block at the candidate distance
            let (log_likelihood, slope) = table.eval_smooth(distance);
            return (-log_likelihood, -slope);
        }
        let (scale, derivative) = if let Some(scale) = factor.fixed_scale {
            (scale, 0.0)
        } else {
            self.graph.calibration.scale_and_derivative(distance)
        };
        let residual = factor.measured - distance;
        let standardized = residual / scale;
        let multiplier = if residual >= 0.0 {
            self.range_error.positive_scale
        } else {
            self.range_error.negative_scale
        };
        let normalized = standardized / multiplier;
        let delta = 0.7;
        let root = (1.0 + (normalized / delta).powi(2)).sqrt();
        let mut value = delta * delta * (root - 1.0);
        let mut gradient =
            normalized / root * (-1.0 / scale - residual * derivative / scale.powi(2)) / multiplier;
        value += scale.ln();
        gradient += derivative / scale;
        value += ((self.range_error.positive_scale + self.range_error.negative_scale) / 2.0).ln();
        (value, gradient)
    }
}

impl DifferentiableObjective for Objective<'_> {
    fn value_gradient(&mut self, parameters: &[f64], gradient: &mut [f64]) -> f64 {
        gradient.fill(0.0);
        let graph = self.graph;
        let distances: Vec<_> = graph
            .pairs
            .iter()
            .map(|p| {
                let a = graph.point(parameters, p.left);
                let b = graph.point(parameters, p.right);
                ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt().max(1e-9)
            })
            .collect();
        let mut slopes = vec![0.0; graph.pairs.len()];
        let mut value = 0.0;
        for factor in &graph.ranges {
            let pair = &graph.pairs[factor.pair];
            let (loss, derivative) = self.range_loss(factor, distances[factor.pair]);
            value += loss;
            let left = graph.point(parameters, pair.left);
            let right = graph.point(parameters, pair.right);
            let gx = derivative * (left.x - right.x) / distances[factor.pair];
            let gy = derivative * (left.y - right.y) / distances[factor.pair];
            gradient[2 * pair.left] += gx;
            gradient[2 * pair.left + 1] += gy;
            gradient[2 * pair.right] -= gx;
            gradient[2 * pair.right + 1] -= gy;
        }
        if let Some(law) = graph.presence {
            // one presence law on every opportunity, a log likelihood term of unit weight
            for (i, pair) in graph.pairs.iter().enumerate() {
                let (log_probability, derivative) = law.log_probability(distances[i], pair.observed);
                value -= log_probability;
                slopes[i] -= derivative;
            }
        }
        for (i, pair) in graph.pairs.iter().enumerate() {
            let left = graph.point(parameters, pair.left);
            let right = graph.point(parameters, pair.right);
            let gx = slopes[i] * (left.x - right.x) / distances[i];
            let gy = slopes[i] * (left.y - right.y) / distances[i];
            gradient[2 * pair.left] += gx;
            gradient[2 * pair.left + 1] += gy;
            if pair.right < graph.agents {
                gradient[2 * pair.right] -= gx;
                gradient[2 * pair.right + 1] -= gy;
            }
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presence::softplus;

    fn graph() -> Graph {
        Graph {
            calibration: CalibrationModel::default(),
            presence: crate::PresenceProfile::default().resolve(&[300.0, 500.0]),
            successful: vec![570.0, 660.0],
            agents: 4,
            pairs: vec![
                Pair {
                    left: 0,
                    right: 1,
                    observed: true,
                },
                Pair {
                    left: 2,
                    right: 3,
                    observed: true,
                },
                Pair {
                    left: 0,
                    right: 2,
                    observed: false,
                },
                Pair {
                    left: 1,
                    right: 3,
                    observed: false,
                },
            ],
            ranges: vec![
                Range {
                    pair: 0,
                    measured: 570.0,
                    fixed_scale: None,
                    table: None,
                },
                Range {
                    pair: 1,
                    measured: 660.0,
                    fixed_scale: None,
                    table: None,
                },
            ],
        }
    }

    #[test]
    fn selected_objective_gradient_matches_finite_differences() {
        let graph = graph();
        let range_error = RangeErrorProfile {
            positive_scale: 2.0,
            negative_scale: 0.7,
        };
        let mut objective = Objective {
            graph: &graph,
            range_error: &range_error,
        };
        let mut x = vec![5.0, 20.0, 550.0, 15.0, 75.0, 745.0, 600.0, 800.0];
        let mut gradient = vec![0.0; x.len()];
        objective.value_gradient(&x, &mut gradient);
        for i in 0..x.len() {
            let h = 1e-3;
            x[i] += h;
            let high = objective.value_gradient(&x, &mut vec![0.0; x.len()]);
            x[i] -= 2.0 * h;
            let low = objective.value_gradient(&x, &mut vec![0.0; x.len()]);
            x[i] += h;
            assert!(((high - low) / (2.0 * h) - gradient[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn missing_link_penalty_decreases_with_distance_without_a_hard_radius() {
        assert!(softplus(6.0 - 0.5 * 3.0).is_finite());
        assert!(softplus(6.0 - 0.5 * 3.0) > softplus(6.0 - 0.5 * 15.0));
        assert!(softplus(1000.0).is_finite() && softplus(-1000.0).is_finite());
    }
}
