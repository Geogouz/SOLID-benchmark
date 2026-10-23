use std::collections::BTreeMap;

use crate::link::{log_t_norm_for, BlockModel, LinkTable};
use crate::presence::{PresenceProfile, ResolvedPresence};
use crate::{BlockRangeProfile, CalibrationModel, LocalizationMode, PreparedProblem, RangeErrorProfile};

use crate::optimize::DifferentiableObjective;

/// The range term of one node pair, pooled over both directions and every repeated observation.
pub struct Link {
    /// Median of the pooled distances, the measurement of the median residual density.
    pub measured: f64,
    /// Median of the reported packet scales, None for raw timing whose scale follows the candidate distance.
    pub scale: Option<f64>,
    /// The block likelihood table of the pooled packets when the profile selects the block range model.
    pub table: Option<LinkTable>,
    /// The pooled packets as calibrated distances and scales in the canonical order of the pooling, which orders the
    /// links to anchors at one place.
    pub packets: Vec<(f64, f64)>,
}

pub struct Pair {
    pub left: usize,
    pub right: usize,
    /// The pooled range of the pair, None when no observation of it carried an admissible sample.
    pub range: Option<Link>,
    /// Whether the pair has at least one observation, a responded exchange, with or without admissible samples.
    pub observed: bool,
    /// Whether the pair is a ranging opportunity of the presence law. The base set holds the attempted and the
    /// observed pairs under declared attempts, otherwise every agent pair and every link to an active anchor. A
    /// failed attempt stays in the set only when a path of observed pairs joins its two ends, with all anchors
    /// counted as one frame node.
    pub opportunity: bool,
}

pub struct Graph {
    pub agents: usize,
    pub anchors: Vec<[f64; 3]>,
    pub pairs: Vec<Pair>,
    /// For every agent, the indices of its anchor pairs with a range or a presence opportunity.
    pub anchor_links: Vec<Vec<usize>>,
    /// Height nodes and normalized log weights of the block range model. Every agent has one unknown height,
    /// shared by all its anchor links, and the objective marginalizes it over these nodes per agent.
    pub heights: Vec<(f64, f64)>,
    /// The range calibration of raw links, evaluated at the candidate distance by the median density.
    pub calibration: CalibrationModel,
    /// The presence law of the scenario, None when it has no successful link.
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
    /// counts of attempted and successful opportunities.
    pub(crate) fn resolve_presence(&self, profile: &PresenceProfile, layout: &[crate::Point2]) -> Option<ResolvedPresence> {
        let attempts = self.pairs.iter().filter(|p| p.opportunity).count();
        let successes = self.pairs.iter().filter(|p| p.opportunity && p.observed).count();
        profile.resolve_with(&self.successful, attempts, successes, |law| self.outcome_log_likelihood(law, layout))
    }

    /// The log likelihood of the attempt outcomes of every opportunity under `law` at `layout`. Agent pairs count at
    /// their horizontal distance, and each agent's anchor opportunities count with its height marginalized over the
    /// nodes, or at the horizontal distance without nodes, as in the objective.
    pub(crate) fn outcome_log_likelihood(&self, law: &ResolvedPresence, layout: &[crate::Point2]) -> f64 {
        let per_agent = !self.heights.is_empty();
        let horizontal = |pair: &Pair| {
            let a = layout[pair.left];
            let b = if pair.right < self.agents {
                [layout[pair.right].x, layout[pair.right].y]
            } else {
                let anchor = self.anchors[pair.right - self.agents];
                [anchor[0], anchor[1]]
            };
            ((a.x - b[0]).powi(2) + (a.y - b[1]).powi(2)).sqrt().max(1e-9)
        };
        let mut total: f64 = self
            .pairs
            .iter()
            .filter(|pair| pair.opportunity && !(per_agent && pair.right >= self.agents))
            .map(|pair| law.log_probability(horizontal(pair), pair.observed).0)
            .sum();
        if per_agent {
            let (mut terms, mut slopes, mut derivatives) = (Vec::new(), Vec::new(), Vec::new());
            for links in &self.anchor_links {
                if !links.iter().any(|k| self.pairs[*k].opportunity) {
                    continue;
                }
                let distances: Vec<f64> = links.iter().map(|k| horizontal(&self.pairs[*k])).collect();
                total += marginal_anchor_terms(self, links, &distances, Some(law), false, &mut terms, &mut slopes, &mut derivatives);
            }
        }
        total
    }

    /// The graph without a presence law.
    pub fn build(problem: &PreparedProblem, block: Option<&BlockRangeProfile>, calibration: CalibrationModel) -> Self {
        assert_eq!(problem.mode, LocalizationMode::AnchorAided);
        let block_model = block.map(BlockModel::from);
        let norm = log_t_norm_for(block_model.as_ref().map_or(4.0, |m| m.nu));
        let agents = problem.agent_ids.len();
        let ids: BTreeMap<_, _> = problem
            .agent_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i))
            .collect();
        let anchors: BTreeMap<_, _> = problem
            .anchors
            .iter()
            .enumerate()
            .map(|(i, a)| (a.id, agents + i))
            .collect();
        // Every observed pair with its packets pooled over both directions and repeated observations. An
        // observation is a responded exchange, whether or not any of its samples is admissible.
        let pooled = super::pool_observations(problem, &ids, &anchors, &calibration);
        let attempted: Option<std::collections::BTreeSet<(usize, usize)>> = problem.attempts.as_ref().map(|attempts| {
            attempts
                .iter()
                .filter_map(|a| {
                    let left = *ids.get(&a.from)?;
                    let right = if a.to_is_anchor { *anchors.get(&a.to)? } else { *ids.get(&a.to)? };
                    Some((left.min(right), left.max(right)))
                })
                .collect()
        });
        let mut pairs = Vec::new();
        for left in 0..agents {
            for right in left + 1..agents + anchors.len() {
                let packets = pooled.get(&(left, right));
                // the median of the pooled distances for the median residual density, where raw timing converts
                // first, and the table of every packet for the block likelihood
                let range = packets.and_then(Option::as_ref).map(|packets| {
                    let (distances, scales, scale) = match packets {
                        super::Packets::Raw(tofs) => {
                            let distances: Vec<f64> = tofs.iter().map(|t| calibration.corrected_distance_cm(*t)).collect();
                            let scales: Vec<f64> = distances.iter().map(|d| calibration.scale_cm(*d)).collect();
                            (distances, scales, None)
                        }
                        super::Packets::Calibrated(samples) => {
                            let scales: Vec<f64> = samples.iter().map(|s| s.1).collect();
                            let mut sorted = scales.clone();
                            sorted.sort_by(f64::total_cmp);
                            (samples.iter().map(|s| s.0).collect(), scales, Some(super::sorted_median(&sorted)))
                        }
                    };
                    Link {
                        measured: super::sorted_median(&distances),
                        scale,
                        table: block_model.as_ref().map(|model| LinkTable::from_packets(&distances, &scales, model, norm)),
                        packets: distances.iter().copied().zip(scales).collect(),
                    }
                });
                pairs.push(Pair { left, right, range, observed: packets.is_some(), opportunity: false });
            }
        }
        let successful: Vec<f64> = pairs.iter().filter_map(|p| p.range.as_ref().map(|link| link.measured)).collect();
        let heights = block
            .map(|model| {
                let total: f64 = model.height_weights.iter().sum();
                model
                    .height_nodes
                    .iter()
                    .zip(&model.height_weights)
                    .map(|(z, w)| (*z, (w / total).max(1e-300).ln()))
                    .collect()
            })
            .unwrap_or_default();
        // an anchor is active when some agent exchanged with it anywhere in the request
        let mut active_anchors = vec![false; problem.anchors.len()];
        for pair in &pairs {
            if pair.right >= agents && pair.observed {
                active_anchors[pair.right - agents] = true;
            }
        }
        let frame = super::FrameComponents::new(agents, pairs.iter().filter(|p| p.observed).map(|p| (p.left, p.right)));
        for pair in &mut pairs {
            let attempted = match &attempted {
                Some(set) => pair.observed || set.contains(&(pair.left, pair.right)),
                None => pair.right < agents || active_anchors[pair.right - agents],
            };
            // a failed attempt is evidence only between nodes that observed pairs join, the anchors as one frame
            pair.opportunity = attempted && frame.joined(pair.left, pair.right);
        }
        let mut anchor_links = vec![Vec::new(); agents];
        for (index, pair) in pairs.iter().enumerate() {
            if pair.right >= agents && (pair.range.is_some() || pair.opportunity) {
                anchor_links[pair.left].push(index);
            }
        }
        Self {
            agents,
            anchors: problem
                .anchors
                .iter()
                .map(|a| [a.position.x, a.position.y, a.height_cm])
                .collect(),
            pairs,
            anchor_links,
            heights,
            calibration,
            presence: None,
            successful,
        }
    }

    pub fn point(&self, parameters: &[f64], node: usize, dimensions: usize) -> [f64; 3] {
        if node >= self.agents {
            return self.anchors[node - self.agents];
        }
        let at = node * dimensions;
        [
            parameters[at],
            parameters[at + 1],
            if dimensions == 3 {
                parameters[at + 2]
            } else {
                0.0
            },
        ]
    }

    pub fn typical_range(&self) -> f64 {
        let mut values: Vec<_> = self
            .pairs
            .iter()
            .filter_map(|p| p.range.as_ref().map(|link| link.measured))
            .collect();
        values.sort_by(f64::total_cmp);
        values
            .get(values.len() / 2)
            .copied()
            .unwrap_or(100.0)
            .max(1.0)
    }
}

pub struct Objective<'a> {
    pub graph: &'a Graph,
    pub profile: &'a RangeErrorProfile,
}

/// The anchor terms of one agent with its height marginalized. The range and presence terms of every anchor link are
/// evaluated at the slant distance of each height node and summed per node, and the node sums are combined by the
/// log sum exp with their log weights. `distances` holds the horizontal distance of each link of `links`, and
/// `ranges` says whether the range terms count or only the presence terms. Returns the log marginal and fills
/// `derivatives` with its derivative in the horizontal distance of each link. `terms` and `slopes` are buffers
/// reused across agents.
#[allow(clippy::too_many_arguments)]
pub(crate) fn marginal_anchor_terms(
    graph: &Graph,
    links: &[usize],
    distances: &[f64],
    law: Option<&ResolvedPresence>,
    ranges: bool,
    terms: &mut Vec<f64>,
    slopes: &mut Vec<f64>,
    derivatives: &mut Vec<f64>,
) -> f64 {
    let width = links.len();
    terms.clear();
    slopes.clear();
    slopes.resize(graph.heights.len() * width, 0.0);
    for (node, &(z, log_weight)) in graph.heights.iter().enumerate() {
        let mut total = log_weight;
        for (j, (&k, &distance)) in links.iter().zip(distances).enumerate() {
            let pair = &graph.pairs[k];
            let dz = z - graph.anchors[pair.right - graph.agents][2];
            let slant = (distance * distance + dz * dz).sqrt().max(1e-9);
            let mut d_slant = 0.0;
            if ranges {
                if let Some(table) = pair.range.as_ref().and_then(|link| link.table.as_ref()) {
                    let (log_likelihood, slope) = table.eval_smooth(slant);
                    total += log_likelihood;
                    d_slant += slope;
                }
            }
            if let (Some(law), true) = (law, pair.opportunity) {
                let (log_probability, slope) = law.log_probability(slant, pair.observed);
                total += log_probability;
                d_slant += slope;
            }
            slopes[node * width + j] = d_slant * distance / slant;
        }
        terms.push(total);
    }
    let top = terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let marginal = top + terms.iter().map(|t| (t - top).exp()).sum::<f64>().ln();
    derivatives.clear();
    derivatives.resize(width, 0.0);
    for (node, term) in terms.iter().enumerate() {
        let weight = (term - marginal).exp();
        for (j, derivative) in derivatives.iter_mut().enumerate() {
            *derivative += weight * slopes[node * width + j];
        }
    }
    marginal
}

impl DifferentiableObjective for Objective<'_> {
    fn value_gradient(&mut self, parameters: &[f64], gradient: &mut [f64]) -> f64 {
        gradient.fill(0.0);
        let graph = self.graph;
        let dimensions = 2;
        let mut value = 0.0;
        // With height nodes every agent has one unknown height shared by all its anchor links, so their block and
        // presence terms are marginalized together per agent below. Without them, the median residual stage,
        // an anchor link is evaluated at its horizontal distance like an agent pair.
        let per_agent = !graph.heights.is_empty();
        for pair in &graph.pairs {
            let joint = per_agent && pair.right >= graph.agents;
            let a = graph.point(parameters, pair.left, dimensions);
            let b = graph.point(parameters, pair.right, dimensions);
            let distance = (0..dimensions)
                .map(|d| (a[d] - b[d]).powi(2))
                .sum::<f64>()
                .sqrt()
                .max(1e-9);
            let mut derivative = 0.0;
            if let Some(link) = &pair.range {
                match &link.table {
                    // the block likelihood is taken at the horizontal distance
                    Some(table) if !joint => {
                        let (log_likelihood, slope) = table.eval_smooth(distance);
                        value -= log_likelihood;
                        derivative -= slope;
                    }
                    Some(_) => {}
                    None => {
                        let (scale, slope) = match link.scale {
                            Some(scale) => (scale, 0.0),
                            None => graph.calibration.scale_and_derivative(distance),
                        };
                        let residual = link.measured - distance;
                        let multiplier = if residual >= 0.0 {
                            self.profile.positive_scale
                        } else {
                            self.profile.negative_scale
                        };
                        let z = residual / (scale * multiplier);
                        let root = (1.0 + (z / 0.7).powi(2)).sqrt();
                        value += 0.49 * (root - 1.0) + scale.ln();
                        let residual_derivative = z / root / (scale * multiplier);
                        derivative += -residual_derivative - z / root * residual * slope / scale.powi(2) / multiplier
                            + slope / scale;
                        value += ((self.profile.positive_scale + self.profile.negative_scale) / 2.0).ln();
                    }
                }
            }
            // one presence law on every opportunity at the horizontal distance, a log likelihood term of unit weight
            if let (Some(law), true, false) = (graph.presence, pair.opportunity, joint) {
                let (log_probability, d) = law.log_probability(distance, pair.observed);
                value -= log_probability;
                derivative -= d;
            }
            for d in 0..dimensions {
                let delta = derivative * (a[d] - b[d]) / distance;
                gradient[dimensions * pair.left + d] += delta;
                if pair.right < graph.agents {
                    gradient[dimensions * pair.right + d] -= delta;
                }
            }
        }
        if per_agent {
            let (mut terms, mut slopes, mut derivatives, mut distances) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            for (agent, links) in graph.anchor_links.iter().enumerate() {
                if links.is_empty() {
                    continue;
                }
                let a = graph.point(parameters, agent, dimensions);
                distances.clear();
                for &k in links {
                    let pair = &graph.pairs[k];
                    let b = graph.anchors[pair.right - graph.agents];
                    distances.push(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt().max(1e-9));
                }
                let presence = graph.presence.as_ref();
                value -= marginal_anchor_terms(graph, links, &distances, presence, true, &mut terms, &mut slopes, &mut derivatives);
                for ((&k, &distance), &d_distance) in links.iter().zip(&distances).zip(&derivatives) {
                    let pair = &graph.pairs[k];
                    let b = graph.anchors[pair.right - graph.agents];
                    for d in 0..dimensions {
                        gradient[dimensions * agent + d] -= d_distance * (a[d] - b[d]) / distance;
                    }
                }
            }
        }
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Anchor, Point2, RangeObservation};
    fn problem(mode: LocalizationMode) -> PreparedProblem {
        PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2],
            anchors: vec![
                Anchor {
                    id: 3,
                    position: Point2::new(250.0, 100.0),
                    height_cm: 120.0,
                },
                Anchor {
                    id: 4,
                    position: Point2::new(-80.0, 280.0),
                    height_cm: -20.0,
                },
            ],
            observations: vec![
                RangeObservation {
                    from: 1,
                    to: 2,
                    tof_ns: vec![8.0, 10.0],
                    calibrated_samples: vec![],
                    to_is_anchor: false,
                },
                RangeObservation {
                    from: 1,
                    to: 3,
                    tof_ns: vec![7.0],
                    calibrated_samples: vec![],
                    to_is_anchor: true,
                },
            ],
            initial_positions: vec![Point2::new(0.0, 0.0), Point2::new(10.0, 20.0)],
            mode,
        }
    }

    fn profile() -> RangeErrorProfile {
        RangeErrorProfile {
            positive_scale: 2.4,
            negative_scale: 0.6,
        }
    }

    #[test]
    fn selected_aac_gradient_matches_finite_differences() {
        let profile = profile();
        for law in [crate::PresenceProfile::default(), crate::PresenceProfile { factor: 0.5, ..crate::PresenceProfile::default() }] {
            let graph = Graph::new(&problem(LocalizationMode::AnchorAided), None, CalibrationModel::default(), &law);
            let mut objective = Objective {
                graph: &graph,
                profile: &profile,
            };
            let parameters = vec![30.0, 60.0, 170.0, -50.0];
            let mut gradient = vec![0.0; parameters.len()];
            objective.value_gradient(&parameters, &mut gradient);
            for i in 0..parameters.len() {
                let mut p = parameters.clone();
                p[i] += 1e-4;
                let plus = objective.value_gradient(&p, &mut vec![0.0; p.len()]);
                p[i] -= 2e-4;
                let minus = objective.value_gradient(&p, &mut vec![0.0; p.len()]);
                assert!((gradient[i] - (plus - minus) / 2e-4).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn aac_keeps_peer_and_anchor_medians() {
        let graph = Graph::new(&problem(LocalizationMode::AnchorAided), None, CalibrationModel::default(), &crate::PresenceProfile::default());
        assert_eq!(graph.pairs.len(), 5);
        assert_eq!(graph.pairs.iter().filter(|p| p.range.is_some()).count(), 2);
    }
}
