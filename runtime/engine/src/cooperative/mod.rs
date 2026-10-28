use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

use crate::model::ObservationAggregation;
use crate::optimize::{minimize, DifferentiableObjective};
use crate::{
    CalibrationModel, CooperativeProfile, EngineError, LocalizationMode, Point2, PreparedProblem, SolveConfig,
    SolveMetrics, SolveResult,
};

mod aac;
mod af;
mod peak;
#[cfg(test)]
mod tests;

pub(crate) fn solve(
    problem: &PreparedProblem,
    profile: CooperativeProfile,
) -> Result<SolveResult, EngineError> {
    let started = Instant::now();
    profile.validate(problem.mode)?;
    if let Some(block) = &profile.block_range {
        if !block.height_nodes.iter().all(|z| crate::model::sane_coordinate(*z)) {
            return Err(EngineError::InvalidProfile("height nodes must be finite and within 100 km"));
        }
    }
    validate(problem)?;
    validate_attempts(problem)?;
    let config = SolveConfig {
        max_time_micros: 0,
        aggregation: ObservationAggregation::Median,
        ..SolveConfig::production()
    };
    let calibration = profile.calibration.unwrap_or_default();
    let starts = profile.starts;
    let provided = profile.use_provided_start.unwrap_or(false);
    // The geometric initializer's layout with the agents' own laterations, or the request's positions without any
    // lateration in the ablation that starts from them.
    let geometric = || {
        if provided {
            Ok((problem.initial_positions.clone(), Vec::new()))
        } else {
            crate::initializer::layout_and_laterations(problem, &config, calibration)
        }
    };
    // With the two stage block option, the median residual objective runs first and its layout replaces the
    // start of the block objective's multistart.
    let refine = profile.block_range.as_ref().is_some_and(|b| b.refine_from_median);
    let stages: Vec<Option<crate::BlockRangeProfile>> = if refine {
        vec![None, profile.block_range.clone()]
    } else {
        vec![profile.block_range.clone()]
    };
    // One presence law serves every stage, resolved once with the model of the final stage.
    let mut solved = None;
    if problem.mode == LocalizationMode::AnchorFree {
        let (mut start, _) = geometric()?;
        let graphs: Vec<af::Graph> = stages.iter().map(|block| af::Graph::build(problem, block.as_ref(), calibration)).collect();
        // the maximum likelihood reach alone uses the start layout
        let law = graphs.last().and_then(|graph| graph.resolve_presence(&profile.presence, &start));
        for mut graph in graphs {
            graph.presence = law;
            let runs = multistart(&start, graph.typical_range(), starts);
            let mut objective = af::Objective { graph: &graph, range_error: &profile.range_error };
            let stage = optimize_layout(&mut objective, runs, graph.agents, graph.ranges.len() + graph.pairs.len(), &config)?;
            start = stage.0.clone();
            solved = Some(stage);
        }
    } else {
        let graphs: Vec<aac::Graph> = stages.iter().map(|block| aac::Graph::build(problem, block.as_ref(), calibration)).collect();
        // The block range model starts from the geometric initializer's layout in which every agent whose block range
        // links reach at least three horizontal anchor places, not all within a metre of one line, has the peak of its
        // own anchor posterior in place of its lateration and sits there (peak::start). The median residual density
        // keeps the geometric initializer, and the ablation starts from the request's positions. Both resolve the law
        // at their start layout, which only the maximum likelihood reach uses. After the peak start and after the
        // geometric initializer alike every stage runs one more start at the agents' own laterations, while the
        // ablation runs none.
        let (mut start, law, laterations) = match (&profile.block_range, graphs.last()) {
            (Some(_), Some(graph)) if !provided => {
                let (start, law) = peak::start(problem, &config, calibration, graph, &profile.presence)?;
                (start, law, crate::initializer::laterations(problem, &config, calibration)?)
            }
            _ => {
                let (initial, laterations) = geometric()?;
                let law = graphs.last().and_then(|graph| graph.resolve_presence(&profile.presence, &initial));
                (initial, law, laterations)
            }
        };
        for mut graph in graphs {
            graph.presence = law;
            let mut runs = multistart(&start, graph.typical_range(), starts);
            // One more start places every agent with at least three anchor links at its own lateration and keeps the
            // other agents where the stage starts. The initializer aligns each component rigidly to its laterated
            // agents, and the peak start to the peaks, which can move an agent away from a good lateration of its own.
            if let Some(layout) = lateration_start(&start, &laterations) {
                runs.push(flatten(&layout));
            }
            let mut objective = aac::Objective { graph: &graph, profile: &profile.range_error };
            let stage = optimize_layout(&mut objective, runs, graph.agents, graph.pairs.iter().filter(|p| p.range.is_some()).count(), &config)?;
            start = stage.0.clone();
            solved = Some(stage);
        }
    }
    let (positions, mut metrics) = solved.expect("at least one optimization stage");
    metrics.elapsed_micros = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
    Ok(SolveResult {
        agent_ids: problem.agent_ids.clone(),
        positions,
        metrics,
    })
}

/// Declared attempts must name known members: an agent ranging an agent or an anchor of the request.
pub(crate) fn validate_attempts(problem: &PreparedProblem) -> Result<(), EngineError> {
    if let Some(attempts) = &problem.attempts {
        for attempt in attempts {
            if !problem.agent_ids.contains(&attempt.from) {
                return Err(EngineError::UnknownAgent(attempt.from));
            }
            if attempt.to_is_anchor {
                if !problem.anchors.iter().any(|a| a.id == attempt.to) {
                    return Err(EngineError::UnknownAnchor(attempt.to));
                }
            } else if !problem.agent_ids.contains(&attempt.to) {
                return Err(EngineError::UnknownAgent(attempt.to));
            }
        }
    }
    Ok(())
}

fn validate(problem: &PreparedProblem) -> Result<(), EngineError> {
    if problem.agent_ids.len() != problem.initial_positions.len() {
        return Err(EngineError::InvalidInitialPositions);
    }
    let mut agents = BTreeSet::new();
    for &id in &problem.agent_ids {
        if !agents.insert(id) {
            return Err(EngineError::DuplicateAgent(id));
        }
    }
    let mut anchors = BTreeSet::new();
    for anchor in &problem.anchors {
        if !anchors.insert(anchor.id) {
            return Err(EngineError::DuplicateAnchor(anchor.id));
        }
    }
    crate::model::validate_geometry(problem)?;
    if problem.mode == LocalizationMode::AnchorFree && !anchors.is_empty() {
        return Err(EngineError::InvalidCooperativeProblem(
            "AF cannot contain anchors",
        ));
    }
    for link in &problem.observations {
        if !agents.contains(&link.from) {
            return Err(EngineError::UnknownAgent(link.from));
        }
        if link.to_is_anchor {
            if !anchors.contains(&link.to) {
                return Err(EngineError::UnknownAnchor(link.to));
            }
        } else {
            if !agents.contains(&link.to) {
                return Err(EngineError::UnknownAgent(link.to));
            }
            if link.from == link.to {
                return Err(EngineError::InvalidCooperativeProblem(
                    "self-ranging is not supported",
                ));
            }
        }
    }
    Ok(())
}

/// The admissible packets of one node pair, pooled over both directions and every repeated observation.
pub(crate) enum Packets {
    /// Raw times of flight in nanoseconds, sorted, when no observation of the pair carried calibrated samples.
    Raw(Vec<f64>),
    /// Calibrated distances and scales in centimetres, sorted by distance then scale. Raw timings of a pair that
    /// also carries calibrated samples join them, converted with the calibration and the scale of their distance.
    Calibrated(Vec<(f64, f64)>),
}

/// Every node pair with at least one observation, keyed by its two node indices with the lower one first. Agents
/// take the indices of `agents` and anchors those of `anchors`, which follow the agents. The value holds the pooled
/// admissible packets of the pair, None when no observation of the pair carried one. Pooling makes the two
/// directions of an agent pair one link with one persistent offset and excess, as they share the same path.
pub(crate) fn pool_observations(
    problem: &PreparedProblem,
    agents: &BTreeMap<u32, usize>,
    anchors: &BTreeMap<u32, usize>,
    calibration: &CalibrationModel,
) -> BTreeMap<(usize, usize), Option<Packets>> {
    // the raw timings and the calibrated distances and scales of every pair
    type Collected = (Vec<f64>, Vec<(f64, f64)>);
    let mut collected: BTreeMap<(usize, usize), Collected> = BTreeMap::new();
    for link in &problem.observations {
        let left = agents[&link.from];
        let right = if link.to_is_anchor { anchors[&link.to] } else { agents[&link.to] };
        let (raw, calibrated) = collected.entry((left.min(right), left.max(right))).or_default();
        if link.calibrated_samples.is_empty() {
            raw.extend(link.tof_ns.iter().copied().filter(|v| crate::model::admissible_raw(*v, crate::model::MAX_TOF_NS, calibration)));
        } else {
            calibrated.extend(link.calibrated_samples.iter().filter(|s| s.is_admissible()).map(|s| (s.distance_cm, s.scale_cm)));
        }
    }
    collected
        .into_iter()
        .map(|(key, (mut raw, mut calibrated))| {
            let packets = if raw.is_empty() && calibrated.is_empty() {
                None
            } else if calibrated.is_empty() {
                raw.sort_by(f64::total_cmp);
                Some(Packets::Raw(raw))
            } else {
                calibrated.extend(raw.iter().map(|tof| {
                    let distance = calibration.corrected_distance_cm(*tof);
                    (distance, calibration.scale_cm(distance))
                }));
                calibrated.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
                Some(Packets::Calibrated(calibrated))
            };
            (key, packets)
        })
        .collect()
}

/// The median of sorted values, the mean of the two middle values for an even count.
pub(crate) fn sorted_median(sorted: &[f64]) -> f64 {
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid]
    } else {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    }
}

/// Connected components of the observed pairs with every anchor merged into one frame node. Nodes below `agents`
/// are agents, every other node index is an anchor and maps to the frame. A failed attempt carries evidence about a
/// distance only when a path of observed pairs joins its two ends, because otherwise the distance is not identified
/// and the failure term would push one end away from the other without bound.
pub(crate) struct FrameComponents {
    agents: usize,
    parents: Vec<usize>,
}

impl FrameComponents {
    pub(crate) fn new(agents: usize, observed: impl IntoIterator<Item = (usize, usize)>) -> Self {
        let mut components = Self {
            agents,
            parents: (0..=agents).collect(),
        };
        for (left, right) in observed {
            let (a, b) = (components.find(left), components.find(right));
            if a != b {
                components.parents[b] = a;
            }
        }
        components
    }

    fn find(&self, node: usize) -> usize {
        let mut node = node.min(self.agents);
        while self.parents[node] != node {
            node = self.parents[node];
        }
        node
    }

    /// Whether a path of observed pairs joins the two nodes.
    pub(crate) fn joined(&self, left: usize, right: usize) -> bool {
        self.find(left) == self.find(right)
    }
}

fn jitter(run: usize, coordinate: usize) -> f64 {
    let mut seed = (run as u64)
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(coordinate as u64);
    seed = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
    seed = (seed ^ (seed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    seed = (seed ^ (seed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    seed ^= seed >> 31;
    2.0 * (seed >> 11) as f64 / (1_u64 << 53) as f64 - 1.0
}

/// The starts of one multistart stage as parameter vectors. The first is the start layout and the others jitter it
/// with fixed pseudo random offsets at six amplitudes relative to the typical range, one offset per coordinate of
/// the layout in both cooperative modes.
fn multistart(initial: &[Point2], scale: f64, starts: usize) -> Vec<Vec<f64>> {
    (0..starts)
        .map(|run| {
            let relative = if run == 0 { 0.0 } else { [0.03, 0.07, 0.15, 0.30, 0.55, 0.90][(run - 1) % 6] };
            let amplitude = scale * relative;
            initial
                .iter()
                .enumerate()
                .flat_map(|(i, p)| {
                    let point = [p.x, p.y];
                    (0..2).map(move |d| point[d] + amplitude * jitter(run, 2 * i + d))
                })
                .collect()
        })
        .collect()
}

/// The lateration start of a stage. Every agent with a lateration of its own sits there and the others keep their
/// place in `start`. None when no agent has a lateration or the layout would repeat the stage's start.
pub(crate) fn lateration_start(start: &[Point2], laterations: &[Option<Point2>]) -> Option<Vec<Point2>> {
    if laterations.len() != start.len() {
        return None;
    }
    let layout: Vec<Point2> = start.iter().zip(laterations).map(|(p, lateration)| lateration.unwrap_or(*p)).collect();
    (layout != start).then_some(layout)
}

/// A layout as the parameter vector of one start.
fn flatten(layout: &[Point2]) -> Vec<f64> {
    layout.iter().flat_map(|p| [p.x, p.y]).collect()
}

/// Minimizes one objective from every start and keeps the layout with the lowest final objective. The layout
/// takes the two coordinates of each of the `agents`.
fn optimize_layout(
    objective: &mut impl DifferentiableObjective,
    runs: Vec<Vec<f64>>,
    agents: usize,
    factors: usize,
    config: &SolveConfig,
) -> Result<(Vec<Point2>, SolveMetrics), EngineError> {
    let mut positions = Vec::new();
    let mut best = f64::INFINITY;
    let mut metrics = SolveMetrics {
        dimensions: 2,
        factors,
        converged: true,
        starts: runs.len(),
        ..SolveMetrics::default()
    };
    for start in runs {
        let optimized = minimize(objective, start, config);
        if !optimized.parameters.iter().all(|v| v.is_finite()) {
            return Err(EngineError::NonFiniteCooperativeResult);
        }
        let cost = objective.value_gradient(
            &optimized.parameters,
            &mut vec![0.0; optimized.parameters.len()],
        );
        if !cost.is_finite() {
            return Err(EngineError::NonFiniteCooperativeResult);
        }
        metrics.iterations += optimized.iterations;
        // Include the final ranking call as well as the optimizer's calls.
        metrics.objective_evaluations += optimized.evaluations + 1;
        metrics.converged &= optimized.converged;
        metrics.termination = metrics.termination.merge(optimized.termination);
        let layout: Vec<Point2> = optimized.parameters[..2 * agents]
            .chunks_exact(2)
            .map(|p| Point2::new(p[0], p[1]))
            .collect();
        if cost.total_cmp(&best).is_lt() {
            best = cost;
            positions = layout.clone();
        }
    }
    metrics.final_objective = best;
    metrics.gradient_evaluations = metrics.objective_evaluations;
    metrics.range_factor_evaluations = metrics.objective_evaluations.saturating_mul(factors);
    metrics.coordinate_factor_evaluations = metrics.range_factor_evaluations.saturating_mul(2);
    Ok((positions, metrics))
}
