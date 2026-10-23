//! Anchor only SOLID.
//!
//! One normalized range block likelihood (Student t core with a spike and slab persistent positive excess and
//! a persistent per link offset), one presence law on the attempted anchor links, one adaptive exact posterior per
//! agent on the grid with the latent height marginalized (posterior.rs), and one decision that minimizes the
//! empirical capped pairwise distance risk over joint posterior samples. No routes and no geometry thresholds.

use std::collections::BTreeMap;
use std::ops::Range;
use std::time::Instant;

use crate::link::{log_t_norm_for, BlockModel, LinkTable};
use crate::posterior::{
    adaptive, kept_mass, log_add_exp, outcome_log_likelihoods, CellEvaluator, Grid, Rng, Sampler, Support, GUARD_NODES, MIN_GUARD_NODES,
};
use crate::presence::{ml_choice, ml_null, ResolvedPresence};
use crate::{
    Anchor, AnchorOnlyProfile, CalibrationModel, DeploymentProfile, EngineError, LocalizationMode,
    Point2, PreparedProblem, ReachRule, SolveMetrics, SolveResult, TerminationReason,
};

/// One accepted range of a link in centimetres with its reported or calibrated scale.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sample {
    distance_cm: f64,
    scale_cm: f64,
}

/// Every accepted range between one agent and one anchor, sorted by distance.
#[derive(Clone, Debug, PartialEq)]
struct Link {
    agent_index: usize,
    agent_id: u32,
    anchor: Anchor,
    samples: Vec<Sample>,
}

/// Links sorted by agent identifier then anchor identifier, with each agent's contiguous range.
struct PreparedLinks {
    links: Vec<Link>,
    links_by_agent: Vec<Range<usize>>,
    /// The anchors of every responded exchange of each agent in request order, sorted by identifier. An
    /// observation is a responded exchange whether or not any of its samples is admissible.
    responded: Vec<Vec<u32>>,
}

fn positive_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn sample_order(left: &Sample, right: &Sample) -> std::cmp::Ordering {
    left.distance_cm
        .to_bits()
        .cmp(&right.distance_cm.to_bits())
        .then_with(|| left.scale_cm.to_bits().cmp(&right.scale_cm.to_bits()))
}

/// Groups the anchor ranges of an anchor only problem by link. Raw Bern timing is converted with the shared
/// calibration and its distance dependent scale, calibrated samples keep their reported scale. The deployment
/// profile carries no calibration, so raw timings always take the built-in TWR-CLOUD-Bern law, and a caller with a
/// calibration of its own, a held-scan fold's, passes calibrated samples.
fn prepare_links(problem: &PreparedProblem) -> Result<PreparedLinks, EngineError> {
    if problem.initial_positions.len() != problem.agent_ids.len() {
        return Err(EngineError::InvalidInitialPositions);
    }
    let mut agent_lookup = BTreeMap::new();
    for (index, &id) in problem.agent_ids.iter().enumerate() {
        if agent_lookup.insert(id, index).is_some() {
            return Err(EngineError::DuplicateAgent(id));
        }
    }
    let mut anchor_lookup = BTreeMap::new();
    for anchor in &problem.anchors {
        if !anchor.position.x.is_finite()
            || !anchor.position.y.is_finite()
            || !anchor.height_cm.is_finite()
        {
            return Err(EngineError::NonFiniteAnchor(anchor.id));
        }
        let canonical_anchor = Anchor {
            id: anchor.id,
            position: Point2::new(positive_zero(anchor.position.x), positive_zero(anchor.position.y)),
            height_cm: positive_zero(anchor.height_cm),
        };
        if anchor_lookup.insert(anchor.id, canonical_anchor).is_some() {
            return Err(EngineError::DuplicateAnchor(anchor.id));
        }
    }
    crate::model::validate_geometry(problem)?;
    let calibration = CalibrationModel::default();
    let mut grouped: BTreeMap<(u32, u32), (usize, Anchor, Vec<Sample>)> = BTreeMap::new();
    let mut responded = vec![std::collections::BTreeSet::new(); problem.agent_ids.len()];
    for observation in &problem.observations {
        if problem.mode != LocalizationMode::AnchorOnly || !observation.to_is_anchor {
            continue;
        }
        let &agent_index = agent_lookup
            .get(&observation.from)
            .ok_or(EngineError::UnknownAgent(observation.from))?;
        let &anchor = anchor_lookup
            .get(&observation.to)
            .ok_or(EngineError::UnknownAnchor(observation.to))?;
        responded[agent_index].insert(anchor.id);
        let samples: Vec<Sample> = if observation.calibrated_samples.is_empty() {
            observation
                .tof_ns
                .iter()
                .copied()
                .filter(|tof_ns| crate::model::admissible_raw(*tof_ns, crate::model::MAX_TOF_NS, &calibration))
                .map(|tof_ns| {
                    let distance_cm = calibration.corrected_distance_cm(tof_ns);
                    let scale_cm = calibration.scale_cm(distance_cm);
                    Sample {
                        distance_cm: positive_zero(distance_cm),
                        scale_cm: positive_zero(scale_cm),
                    }
                })
                .collect()
        } else {
            observation
                .calibrated_samples
                .iter()
                .filter(|sample| sample.is_admissible())
                .map(|sample| Sample {
                    distance_cm: positive_zero(sample.distance_cm),
                    scale_cm: positive_zero(sample.scale_cm),
                })
                .collect()
        };
        if samples.is_empty() {
            continue;
        }
        grouped
            .entry((observation.from, observation.to))
            .or_insert((agent_index, anchor, Vec::new()))
            .2
            .extend(samples);
    }
    let mut links = Vec::with_capacity(grouped.len());
    for ((agent_id, _), (agent_index, anchor, mut samples)) in grouped {
        samples.sort_unstable_by(sample_order);
        links.push(Link {
            agent_index,
            agent_id,
            anchor,
            samples,
        });
    }
    let mut links_by_agent = vec![0..0; problem.agent_ids.len()];
    for &agent_index in agent_lookup.values() {
        let start = links.partition_point(|link| link.agent_id < problem.agent_ids[agent_index]);
        let end = links.partition_point(|link| link.agent_id <= problem.agent_ids[agent_index]);
        links_by_agent[agent_index] = start..end;
    }
    Ok(PreparedLinks {
        links,
        links_by_agent,
        responded: responded.into_iter().map(|set| set.into_iter().collect()).collect(),
    })
}

impl LinkTable {
    fn build(link: &Link, profile: &AnchorOnlyProfile, norm: f64) -> Self {
        let y: Vec<f64> = link.samples.iter().map(|s| s.distance_cm).collect();
        let scales: Vec<f64> = link.samples.iter().map(|s| s.scale_cm).collect();
        Self::from_packets(&y, &scales, &BlockModel::from(profile), norm)
    }
}

#[derive(Clone, Copy)]
struct Anchor3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Anchor3 {
    /// The anchor as the posterior takes it, x, y and height.
    fn site(&self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// The normalized log weights of the height nodes.
fn height_log_weights(settings: &AnchorOnlyProfile) -> Vec<f64> {
    let total: f64 = settings.height_weights.iter().sum();
    settings.height_weights.iter().map(|w| (w / total).max(1e-300).ln()).collect()
}

/// The inputs of every agent's posterior in the canonical order of the agents, ascending identifier, so that the
/// order of the request cannot change the result. Links and anchors of an agent follow the canonical order of
/// the anchor geometry, so that neither the anchor identifiers nor the order of the observations can.
struct AgentInputs {
    /// The request index of the agent at every canonical rank.
    order: Vec<usize>,
    /// The range links with their block likelihood tables.
    links: Vec<Vec<(Anchor3, LinkTable)>>,
    /// Every anchor the agent exchanged with.
    present: Vec<Vec<Anchor3>>,
    /// The failed attempts that carry evidence, none for an agent without a responded exchange.
    missing: Vec<Vec<Anchor3>>,
    /// Per link median distance of every link with admissible samples, the input of the reach estimate.
    successful: Vec<f64>,
    /// The seed of the scenario from the profile's seed, the sorted agent identifiers and the sorted link data.
    seed: u64,
}

impl AgentInputs {
    /// The presence law of one agent, the scenario's law when the agent exchanged with some anchor and none
    /// otherwise, so that an agent without a path to the anchor frame takes the uninformative support.
    fn law(&self, agent: usize, scenario: Option<ResolvedPresence>) -> Option<ResolvedPresence> {
        if self.present[agent].is_empty() {
            None
        } else {
            scenario
        }
    }
}

/// The canonical key of an anchor, its geometry, free of its identifier.
fn geometry_key(anchor: &Anchor) -> [u64; 3] {
    [anchor.position.x.to_bits(), anchor.position.y.to_bits(), anchor.height_cm.to_bits()]
}

/// The canonical order of an agent's links, by anchor geometry and then by the sorted samples.
fn link_order(left: &Link, right: &Link) -> std::cmp::Ordering {
    geometry_key(&left.anchor)
        .cmp(&geometry_key(&right.anchor))
        .then_with(|| {
            left.samples
                .iter()
                .zip(&right.samples)
                .map(|(a, b)| sample_order(a, b))
                .find(|order| order.is_ne())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .then_with(|| left.samples.len().cmp(&right.samples.len()))
}

/// Builds the range links, the present anchors and the failed attempts of every agent in canonical order. The
/// opportunities of an agent are its declared attempts, or every active anchor without declared attempts, where an
/// active anchor answered at least one exchange anywhere in the request. An agent without any responded exchange
/// shares no path with the anchor frame, so none of its failed attempts is evidence about its position.
fn agent_inputs(problem: &PreparedProblem, prepared: &PreparedLinks, settings: &AnchorOnlyProfile, norm: f64) -> AgentInputs {
    let mut active_anchor_ids: Vec<u32> = prepared.responded.iter().flatten().copied().collect();
    active_anchor_ids.sort_unstable();
    active_anchor_ids.dedup();
    let anchor_by_id = |id: u32| problem.anchors.iter().find(|a| a.id == id);
    let anchor3 = |a: &Anchor| Anchor3 { x: a.position.x, y: a.position.y, z: a.height_cm };
    // anchors of one agent sorted by their geometry, identifiers only between anchors at the same place
    let anchors_of = |ids: &mut Vec<&Anchor>| -> Vec<Anchor3> {
        ids.sort_by(|a, b| geometry_key(a).cmp(&geometry_key(b)).then(a.id.cmp(&b.id)));
        ids.iter().map(|a| anchor3(a)).collect()
    };
    let attempted_by_agent: Option<Vec<Vec<u32>>> = problem.attempts.as_ref().map(|attempts| {
        problem
            .agent_ids
            .iter()
            .map(|agent| attempts.iter().filter(|a| a.from == *agent && a.to_is_anchor).map(|a| a.to).collect())
            .collect()
    });
    let n = problem.agent_ids.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| problem.agent_ids[i]);
    // the seed hashes the sorted agent identifiers, then every link's anchor geometry and samples in canonical order
    let mix = |seed: u64, value: u64| (seed ^ value).wrapping_mul(0x0100_0000_01b3);
    let mut seed = order.iter().fold(0xcbf2_9ce4_8422_2325_u64 ^ settings.seed, |seed, &i| mix(seed, problem.agent_ids[i] as u64));
    let mut inputs = AgentInputs {
        order: order.clone(),
        links: Vec::with_capacity(n),
        present: Vec::with_capacity(n),
        missing: Vec::with_capacity(n),
        successful: Vec::new(),
        seed: 0,
    };
    for &agent_index in &order {
        let mut agent_links: Vec<&Link> = prepared.links[prepared.links_by_agent[agent_index].clone()].iter().collect();
        agent_links.sort_by(|a, b| link_order(a, b));
        let mut links = Vec::with_capacity(agent_links.len());
        for link in agent_links {
            for bits in geometry_key(&link.anchor) {
                seed = mix(seed, bits);
            }
            for sample in &link.samples {
                seed = mix(seed, sample.distance_cm.to_bits());
            }
            let mut distances: Vec<f64> = link.samples.iter().map(|s| s.distance_cm).collect();
            inputs.successful.extend(crate::presence::median(&mut distances));
            links.push((anchor3(&link.anchor), LinkTable::build(link, settings, norm)));
        }
        let responded = &prepared.responded[agent_index];
        let present = anchors_of(&mut responded.iter().filter_map(|id| anchor_by_id(*id)).collect());
        let mut missing: Vec<Anchor3> = Vec::new();
        if !responded.is_empty() {
            let candidates: &[u32] = match &attempted_by_agent {
                Some(per_agent) => &per_agent[agent_index],
                None => &active_anchor_ids,
            };
            let mut missing_ids: Vec<u32> = candidates.iter().copied().filter(|id| !responded.contains(id)).collect();
            missing_ids.sort_unstable();
            missing_ids.dedup();
            missing = anchors_of(&mut missing_ids.iter().filter_map(|id| anchor_by_id(*id)).collect());
        }
        inputs.links.push(links);
        inputs.present.push(present);
        inputs.missing.push(missing);
    }
    inputs.seed = seed;
    inputs
}

/// The evaluator of the agent at canonical rank `rank` on `grid`, with its links, present anchors and failed
/// attempts in canonical order, the profile's height nodes with their log weights `weights` and the law `law`.
fn agent_evaluator<'a>(
    grid: &'a Grid,
    inputs: &'a AgentInputs,
    rank: usize,
    settings: &AnchorOnlyProfile,
    weights: &[f64],
    law: Option<ResolvedPresence>,
) -> CellEvaluator<'a> {
    let links: Vec<([f64; 3], &LinkTable)> = inputs.links[rank].iter().map(|(anchor, table)| (anchor.site(), table)).collect();
    let present: Vec<[f64; 3]> = inputs.present[rank].iter().map(Anchor3::site).collect();
    let missing: Vec<[f64; 3]> = inputs.missing[rank].iter().map(Anchor3::site).collect();
    CellEvaluator::new(grid, &links, &present, &missing, &settings.height_nodes, weights, law)
}

/// The work and the certificates of the posteriors of one solve: per canonical rank whether the node guard cut a
/// posterior of the agent and the largest certified share of its own mass one of them left out, and the nodes bounded
/// and the cells evaluated over all of them. The posteriors of an agent are the one its draws come from and, under the
/// maximum likelihood reach, its range only posterior and its posteriors under the candidate laws the reach refined.
struct Tally {
    guarded: Vec<bool>,
    dropped: Vec<f64>,
    nodes: usize,
    cells: usize,
}

impl Tally {
    fn new(agents: usize) -> Self {
        Self { guarded: vec![false; agents], dropped: vec![0.0; agents], nodes: 0, cells: 0 }
    }

    fn record(&mut self, rank: usize, support: &Support) {
        self.guarded[rank] |= support.guarded;
        self.dropped[rank] = self.dropped[rank].max(support.dropped_share());
        self.nodes += support.nodes;
        self.cells += support.cells;
    }
}

/// The presence law of the scenario, with the reach from the successful links: the quantile rule's, or the one the
/// maximum likelihood reach chooses (`judge_reach`) under the node guard `guard`.
fn scenario_law(grid: &Grid, inputs: &AgentInputs, settings: &AnchorOnlyProfile, guard: usize, tally: &mut Tally) -> Option<ResolvedPresence> {
    let presence = &settings.presence;
    if presence.reach == ReachRule::Quantile {
        return presence.resolve(&inputs.successful);
    }
    judge_reach(grid, inputs, settings, guard, tally).and_then(|judgement| judgement.chosen)
}

/// Relative allowance of a comparison of two summed log likelihoods of the maximum likelihood reach, 2^-30 of their
/// size and one. It lies far above the rounding of the log sums over the cells and of the sums over the agents, so a
/// comparison it certifies holds for the exact sums and not only for their floating point values.
const REACH_ROUNDING: f64 = 1.0 / 1_073_741_824.0;

/// Whether `a` exceeds `b` by more than REACH_ROUNDING of their size.
fn certainly_above(a: f64, b: f64) -> bool {
    a > b && (a == f64::INFINITY || b == f64::NEG_INFINITY || a - b > REACH_ROUNDING * (1.0 + a.abs() + b.abs()))
}

/// ln(e^a + e^b), plus infinity where either is.
fn log_sum(a: f64, b: f64) -> f64 {
    if a == f64::INFINITY || b == f64::INFINITY {
        f64::INFINITY
    } else {
        log_add_exp(a, b)
    }
}

/// The sum of the per agent terms `terms`, minus infinity where one of them is: an agent whose posterior has no mass
/// anywhere makes the scenario's likelihood nothing under every law.
fn agent_sum(terms: impl Iterator<Item = f64>) -> f64 {
    terms.fold(0.0, |sum, term| if sum == f64::NEG_INFINITY || term == f64::NEG_INFINITY { f64::NEG_INFINITY } else { sum + term })
}

/// What the maximum likelihood reach found for the candidate laws of a scenario.
///
/// For a law, the rule's value is the sum over the agents of ln(N / D), where N integrates the range terms, the height
/// weights and the law's presence terms, all at most zero, over every cell and height node, and D the range terms and
/// weights alone. The range only posterior of an agent keeps the mass K of D and certifies the rest at most e^R. With
/// F = ln(N over the kept cells) - ln K, today's value of the rule, the presence terms can only shrink what is left
/// out, so ln N - ln K lies between F and ln(e^F + e^(R - ln K)), while ln D - ln K lies between zero and
/// ln(1 + e^(R - ln K)), which is the same for every law. Summed over the agents, `lower` and `upper` bound the first
/// and `slack` the second: the rule's value lies between lower - slack and upper, and the difference between two laws
/// is at least the lower bound of one less the upper bound of the other. The solve reads the choice alone, and the tests
/// and the equivalence run read the rest.
#[cfg_attr(not(test), allow(dead_code))]
struct ReachJudgement {
    candidates: Vec<ResolvedPresence>,
    /// Per law, the sum over the agents of F.
    values: Vec<f64>,
    /// Per law, the bounds of the sum of ln N - ln K, after the refinement of the refined laws.
    lower: Vec<f64>,
    upper: Vec<f64>,
    slack: f64,
    /// The log likelihood of the outcomes independent of the distance, the value a law must beat.
    null: f64,
    /// The laws whose numerators were refined by the posteriors under them.
    refined: Vec<usize>,
    /// Whether the bounds settled the choice, or the choice is the likeliest by the lower bounds among the laws they
    /// could not tell apart.
    certified: bool,
    chosen: Option<ResolvedPresence>,
}

/// The choice the bounds certify among the laws `open`, or the laws that may still be chosen. A law cannot be chosen
/// when its upper bound certainly lies at or below the null's value, since it must be strictly likelier, or below the
/// lower bound of another law, since that law is certainly likelier. When no law remains the choice is none, and when
/// one remains whose lower bound less the slack certainly lies above the null's value, it is that law.
fn settle(lower: &[f64], upper: &[f64], slack: f64, null: f64, open: &[usize]) -> Result<Option<usize>, Vec<usize>> {
    let remaining: Vec<usize> = open
        .iter()
        .copied()
        .filter(|&j| !certainly_above(null, upper[j]) && !(0..lower.len()).any(|i| i != j && certainly_above(lower[i], upper[j])))
        .collect();
    match remaining.as_slice() {
        [] => Ok(None),
        [k] if certainly_above(lower[*k] - slack, null) => Ok(Some(*k)),
        _ => Err(remaining),
    }
}

/// The maximum likelihood reach of a scenario, certified. Every agent that exchanged with an anchor integrates its
/// position and height over its range only posterior, the adaptive posterior without the presence terms, under the
/// node guard `guard`, streaming the kept cells once for all nine candidate laws, so the chosen reach maximizes the
/// likelihood of all attempt outcomes given the ranges without a layout to condition on, deterministically and before
/// any presence term shapes the posteriors.
///
/// The kept cells hold all but a certified share of the range only mass, but the presence terms may favour the cells
/// left out, so the choice must be certified (`ReachJudgement`). When the bounds do not settle it, every law that may
/// still be chosen takes its numerator from the adaptive posterior under that law, whose kept cells hold all but a
/// certified share of that law's own mass, and the bounds narrow to it. When they still do not settle the choice, the
/// laws left differ by less than those certificates can tell, and the likeliest by the lower bounds wins, the first of
/// equal laws. Where the bounds settle the choice at once, it is the choice by the values over the kept cells alone.
fn judge_reach(grid: &Grid, inputs: &AgentInputs, settings: &AnchorOnlyProfile, guard: usize, tally: &mut Tally) -> Option<ReachJudgement> {
    let presence = &settings.presence;
    let attempts: usize = inputs.present.iter().zip(&inputs.missing).map(|(present, missing)| present.len() + missing.len()).sum();
    let successes: usize = inputs.present.iter().map(Vec::len).sum();
    let candidates = presence.ml_candidates(&inputs.successful, attempts, successes)?;
    let null = ml_null(attempts, successes);
    let weights = height_log_weights(settings);
    // every agent that exchanged with an anchor, over its range only posterior: its kept mass ln K and per law the
    // bounds of ln N - ln K
    let agents: Vec<usize> = (0..inputs.order.len()).filter(|rank| !inputs.present[*rank].is_empty()).collect();
    let mut kept = Vec::with_capacity(agents.len());
    let mut bounds: Vec<Vec<(f64, f64)>> = Vec::with_capacity(agents.len());
    let mut slacks = Vec::with_capacity(agents.len());
    for &rank in &agents {
        let evaluator = agent_evaluator(grid, inputs, rank, settings, &weights, None);
        let support = adaptive(&evaluator, guard);
        tally.record(rank, &support);
        let (mass, values) = outcome_log_likelihoods(&evaluator, &support, &candidates, &mut tally.cells);
        // what the range only posterior left out relative to what it kept
        let left = if support.dropped_log() == f64::NEG_INFINITY {
            f64::NEG_INFINITY
        } else if mass == f64::NEG_INFINITY {
            f64::INFINITY
        } else {
            support.dropped_log() - mass
        };
        slacks.push(log_sum(0.0, left));
        bounds.push(values.iter().map(|value| (*value, log_sum(*value, left))).collect());
        kept.push(mass);
    }
    let count = candidates.len();
    let values: Vec<f64> = (0..count).map(|j| agent_sum(bounds.iter().map(|agent| agent[j].0))).collect();
    let sums = |bounds: &[Vec<(f64, f64)>]| -> (Vec<f64>, Vec<f64>) {
        (0..count).map(|j| (agent_sum(bounds.iter().map(|agent| agent[j].0)), agent_sum(bounds.iter().map(|agent| agent[j].1)))).unzip()
    };
    let slack = agent_sum(slacks.into_iter());
    let (mut lower, mut upper) = sums(&bounds);
    let mut refined = Vec::new();
    let (chosen, certified) = match settle(&lower, &upper, slack, null, &(0..count).collect::<Vec<_>>()) {
        Ok(chosen) => (chosen.map(|k| candidates[k]), true),
        Err(open) => {
            // the numerator of every law that may still be chosen from the posterior under that law, whose kept cells
            // hold its mass N but a certified part, and whose bounds narrow those of the range only posterior
            for &j in &open {
                for (a, &rank) in agents.iter().enumerate() {
                    if kept[a] == f64::NEG_INFINITY {
                        continue;
                    }
                    let evaluator = agent_evaluator(grid, inputs, rank, settings, &weights, Some(candidates[j]));
                    let support = adaptive(&evaluator, guard);
                    tally.record(rank, &support);
                    let mass = kept_mass(&evaluator, &support, &mut tally.cells);
                    let (low, high) = bounds[a][j];
                    bounds[a][j] = (low.max(mass - kept[a]), high.min(log_sum(mass, support.dropped_log()) - kept[a]));
                }
            }
            (lower, upper) = sums(&bounds);
            refined.clone_from(&open);
            match settle(&lower, &upper, slack, null, &open) {
                Ok(chosen) => (chosen.map(|k| candidates[k]), true),
                Err(open) => {
                    let best: Vec<f64> = (0..count).map(|j| if open.contains(&j) { lower[j] } else { f64::NEG_INFINITY }).collect();
                    (ml_choice(&candidates, &best, attempts, successes), false)
                }
            }
        }
    };
    Some(ReachJudgement { candidates, values, lower, upper, slack, null, refined, certified, chosen })
}

/// The random stream of one agent, derived from the scenario seed and the agent's identifier alone.
fn agent_stream(seed: u64, agent_id: u32) -> Rng {
    let mut mixer = Rng(seed ^ (agent_id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
    Rng(mixer.next_u64())
}

pub(crate) fn solve(
    problem: &PreparedProblem,
    profile: &DeploymentProfile,
) -> Result<SolveResult, EngineError> {
    solve_guarded(problem, profile, GUARD_NODES)
}

/// `solve` with the node guard `guard` of every posterior, the range only posteriors of the maximum likelihood reach
/// among them, GUARD_NODES but in the tests. A guard below MIN_GUARD_NODES is refused.
fn solve_guarded(problem: &PreparedProblem, profile: &DeploymentProfile, guard: usize) -> Result<SolveResult, EngineError> {
    assert!(guard >= MIN_GUARD_NODES, "a node guard of {guard} cannot hold the four children of a node");
    let settings = &profile.anchor_only;
    if problem.mode != LocalizationMode::AnchorOnly {
        return Err(EngineError::AncPreparation("anchor only positioning requires anchor only mode"));
    }
    if !settings.height_nodes.iter().all(|z| crate::model::sane_coordinate(*z)) {
        return Err(EngineError::InvalidProfile("height nodes must be finite and within 100 km"));
    }
    let started = Instant::now();
    let prepared = prepare_links(problem)?;
    let norm = log_t_norm_for(settings.nu);
    let grid = Grid::new(&profile.area, settings.grid_step)?;
    // every agent in canonical rank order from here on, back in request order at the end
    let inputs = agent_inputs(problem, &prepared, settings, norm);
    let n = inputs.order.len();
    let total_links: usize = inputs.links.iter().map(Vec::len).sum();
    let mut tally = Tally::new(n);
    let detection = scenario_law(&grid, &inputs, settings, guard, &mut tally);
    let weights = height_log_weights(settings);
    let samples = settings.joint_samples;
    let mut layouts = vec![vec![[0.0f64; 2]; n]; samples];
    // one agent at a time: its posterior, its draws from its own stream, so neither the number nor the order of the
    // other agents changes them, and then the next agent
    for (rank, &index) in inputs.order.iter().enumerate() {
        let mut stream = agent_stream(inputs.seed, problem.agent_ids[index]);
        let law = inputs.law(rank, detection);
        // an agent without a range link or a presence term carries no evidence about its position
        let informative = !inputs.present[rank].is_empty() && !(inputs.links[rank].is_empty() && law.is_none());
        let evaluator = informative.then(|| agent_evaluator(&grid, &inputs, rank, settings, &weights, law));
        let support = match &evaluator {
            Some(evaluator) => adaptive(evaluator, guard),
            None => Support::uniform(&grid),
        };
        tally.record(rank, &support);
        let draws = agent_draws(&grid, &support, evaluator.as_ref(), &mut stream, samples, &mut tally.cells, problem.agent_ids[index])?;
        for (layout, point) in layouts.iter_mut().zip(draws) {
            layout[rank] = point;
        }
    }
    let (canonical, risk, evaluations, iterations) = decide(&layouts, n, settings.decision == crate::DecisionRule::Mean);
    let mut positions = vec![Point2::default(); n];
    for (rank, &index) in inputs.order.iter().enumerate() {
        positions[index] = Point2::new(canonical[rank][0], canonical[rank][1]);
    }
    let elapsed = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
    Ok(SolveResult {
        agent_ids: problem.agent_ids.clone(),
        positions,
        metrics: SolveMetrics {
            elapsed_micros: elapsed,
            dimensions: 3,
            iterations,
            objective_evaluations: evaluations,
            gradient_evaluations: iterations,
            factors: total_links,
            range_factor_evaluations: evaluations.saturating_mul(total_links),
            coordinate_factor_evaluations: 0,
            converged: true,
            termination: TerminationReason::Converged,
            final_objective: risk,
            starts: samples,
            guarded_supports: Some(tally.guarded.iter().filter(|g| **g).count()),
            largest_dropped_mass: Some(tally.dropped.iter().copied().fold(0.0, f64::max)),
            posterior_nodes: Some(tally.nodes),
            posterior_cells: Some(tally.cells),
        },
    })
}

/// Why a support gave no draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Undrawn {
    /// It keeps no cell.
    Empty,
    /// A drawn cell of a coarse leaf lay above the leaf's certified upper bound (`Sampler::draw`).
    AboveBound,
}

/// One agent's place in every joint sample, drawn from its own stream: a kept cell by posterior weight, then a uniform
/// point in the cell. Nothing where the posterior keeps no cell, which leaves nothing to draw from, or where a drawn
/// cell lies above its leaf's certified bound.
fn joint_draws(grid: &Grid, support: &Support, evaluator: Option<&CellEvaluator>, stream: &mut Rng, samples: usize, cells: &mut usize) -> Result<Vec<[f64; 2]>, Undrawn> {
    let sampler = Sampler::new(support).ok_or(Undrawn::Empty)?;
    (0..samples)
        .map(|_| {
            let (i, j) = sampler.draw(evaluator, stream, cells).ok_or(Undrawn::AboveBound)?;
            let (x, y) = (grid.xs[i], grid.ys[j]);
            Ok([x + (stream.uniform() - 0.5) * grid.step, y + (stream.uniform() - 0.5) * grid.step])
        })
        .collect()
}

/// The joint draws of agent `agent_id` from its support (`joint_draws`). Where the node guard cut the posterior and left
/// it no kept cell, the agent is drawn as one without evidence, every cell of the area alike, and the tally already
/// counts its posterior as guarded, with its whole mass left out. A support that keeps no cell although the guard did not
/// cut it is not finite on any cell, and a cell above its leaf's certified bound is a failure of the bounds.
fn agent_draws(
    grid: &Grid,
    support: &Support,
    evaluator: Option<&CellEvaluator>,
    stream: &mut Rng,
    samples: usize,
    cells: &mut usize,
    agent_id: u32,
) -> Result<Vec<[f64; 2]>, EngineError> {
    match joint_draws(grid, support, evaluator, stream, samples, cells) {
        Ok(draws) => Ok(draws),
        Err(Undrawn::Empty) if support.guarded => agent_draws(grid, &Support::uniform(grid), None, stream, samples, cells, agent_id),
        Err(Undrawn::Empty) => Err(EngineError::NonFinitePosterior(agent_id)),
        Err(Undrawn::AboveBound) => Err(EngineError::PosteriorAboveBound(agent_id)),
    }
}

fn pair_distances(layout: &[[f64; 2]]) -> Vec<f64> {
    let n = layout.len();
    let mut out = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in i + 1..n {
            out.push(((layout[i][0] - layout[j][0]).powi(2) + (layout[i][1] - layout[j][1]).powi(2)).sqrt());
        }
    }
    out
}

fn risk(action_d: &[f64], sample_d: &[Vec<f64>], caps: &[f64]) -> f64 {
    let pairs = action_d.len() as f64;
    let mut total = 0.0;
    for (sd, cap) in sample_d.iter().zip(caps) {
        let mut e = 0.0;
        for (a, d) in action_d.iter().zip(sd) {
            e += (a - d).abs().min(*cap);
        }
        total += e / pairs;
    }
    total / sample_d.len() as f64
}

/// Minimum empirical capped pairwise risk action over the joint samples.
fn decide(layouts: &[Vec<[f64; 2]>], n: usize, mean_only: bool) -> (Vec<[f64; 2]>, f64, usize, usize) {
    let s = layouts.len();
    let mut mean = vec![[0.0f64; 2]; n];
    for layout in layouts {
        for (i, p) in layout.iter().enumerate() {
            mean[i][0] += p[0] / s as f64;
            mean[i][1] += p[1] / s as f64;
        }
    }
    if n < 2 {
        return (mean, 0.0, 0, 0);
    }
    let sample_d: Vec<Vec<f64>> = layouts.iter().map(|l| pair_distances(l)).collect();
    let caps: Vec<f64> = sample_d.iter().map(|d| d.iter().copied().fold(0.0, f64::max).max(1e-9)).collect();
    if mean_only {
        // ablation: report the posterior coordinate mean instead of the pairwise risk minimizer
        let r = risk(&pair_distances(&mean), &sample_d, &caps);
        return (mean, r, 1, 0);
    }
    let mut evaluations = 0usize;
    let mut candidates = vec![mean.clone()];
    // medoid among a subsample of joint layouts
    let stride = (s / 48).max(1);
    let mut best_i = 0;
    let mut best_r = f64::INFINITY;
    for (i, layout) in layouts.iter().enumerate().step_by(stride) {
        let r = risk(&sample_d[i], &sample_d, &caps);
        evaluations += 1;
        if r < best_r {
            best_r = r;
            best_i = i;
        }
        let _ = layout;
    }
    candidates.push(layouts[best_i].clone());
    let mut caps_sorted = caps.clone();
    caps_sorted.sort_by(f64::total_cmp);
    let scale = caps_sorted[caps_sorted.len() / 2].max(1.0);
    let mut best: Option<(Vec<[f64; 2]>, f64)> = None;
    let mut iterations = 0usize;
    let pairs = n * (n - 1) / 2;
    for cand in candidates {
        let mut p = cand;
        let mut d = pair_distances(&p);
        let mut r = risk(&d, &sample_d, &caps);
        evaluations += 1;
        let mut step = 0.05 * scale;
        for _ in 0..120 {
            iterations += 1;
            // subgradient of the empirical capped risk
            let mut g = vec![[0.0f64; 2]; n];
            let mut k = 0;
            for i in 0..n {
                for j in i + 1..n {
                    let dij = d[k].max(1e-9);
                    let mut coef = 0.0;
                    for (sd, cap) in sample_d.iter().zip(&caps) {
                        let diff = dij - sd[k];
                        if diff.abs() < *cap {
                            coef += diff.signum();
                        }
                    }
                    coef /= (s * pairs) as f64;
                    let ux = (p[i][0] - p[j][0]) / dij;
                    let uy = (p[i][1] - p[j][1]) / dij;
                    g[i][0] += coef * ux;
                    g[i][1] += coef * uy;
                    g[j][0] -= coef * ux;
                    g[j][1] -= coef * uy;
                    k += 1;
                }
            }
            let gn = g.iter().map(|v| v[0] * v[0] + v[1] * v[1]).sum::<f64>().sqrt();
            if gn < 1e-12 {
                break;
            }
            let trial: Vec<[f64; 2]> = p
                .iter()
                .zip(&g)
                .map(|(pi, gi)| [pi[0] - step * gi[0] / gn, pi[1] - step * gi[1] / gn])
                .collect();
            let td = pair_distances(&trial);
            let tr = risk(&td, &sample_d, &caps);
            evaluations += 1;
            if tr < r {
                p = trial;
                d = td;
                r = tr;
            } else {
                step *= 0.6;
                if step < 0.01 {
                    break;
                }
            }
        }
        if best.as_ref().is_none_or(|(_, br)| r < *br) {
            best = Some((p, r));
        }
    }
    let (p, r) = best.expect("at least one candidate");
    (p, r, evaluations, iterations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::tests::{compare, exhaustive};
    use crate::posterior::{peak_of, LogSum, EPSILON};
    use crate::{CalibratedRangeSample, CalibrationModel, PlanarArea, RangeObservation};

    /// The grid, the agent inputs, the scenario's law and the height log weights of a problem, as the solve builds them.
    struct Setup {
        grid: Grid,
        inputs: AgentInputs,
        law: Option<ResolvedPresence>,
        weights: Vec<f64>,
    }

    fn setup(problem: &PreparedProblem, profile: &DeploymentProfile) -> Setup {
        let mut setup = lawless_setup(problem, profile);
        setup.law = scenario_law(&setup.grid, &setup.inputs, &profile.anchor_only, GUARD_NODES, &mut Tally::new(setup.inputs.order.len()));
        setup
    }

    /// The setup of a problem before its law, which the caller resolves.
    fn lawless_setup(problem: &PreparedProblem, profile: &DeploymentProfile) -> Setup {
        let settings = &profile.anchor_only;
        let prepared = prepare_links(problem).unwrap();
        let inputs = agent_inputs(problem, &prepared, settings, log_t_norm_for(settings.nu));
        let grid = Grid::new(&profile.area, settings.grid_step).unwrap();
        Setup { grid, inputs, law: None, weights: height_log_weights(settings) }
    }

    fn deployment() -> DeploymentProfile {
        DeploymentProfile {
            area: PlanarArea { minimum_x_cm: 0.0, maximum_x_cm: 1000.0, minimum_y_cm: 0.0, maximum_y_cm: 1000.0 },
            anchor_only: AnchorOnlyProfile {
                kappa: 1.0,
                excess_probability: 0.05,
                excess_mean_rel: 2.0,
                persistent_sd_rel: 0.5,
                height_nodes: vec![100.0],
                height_weights: vec![1.0],
                grid_step: 10.0,
                joint_samples: 48,
                ..AnchorOnlyProfile::default()
            },
        }
    }

    fn observation(from: u32, to: u32, distances: &[f64]) -> RangeObservation {
        RangeObservation {
            from,
            to,
            tof_ns: vec![],
            calibrated_samples: distances.iter().map(|d| CalibratedRangeSample::new(*d, 6.0)).collect(),
            to_is_anchor: true,
        }
    }

    /// One agent ranging one anchor 400 cm away with a wide persistent offset, over a 2000 cm square at 20 cm.
    fn ring_problem() -> (PreparedProblem, DeploymentProfile) {
        let problem = PreparedProblem {
            agent_ids: vec![1],
            anchors: vec![Anchor { id: 1, position: Point2::new(1000.0, 1000.0), height_cm: 200.0 }],
            observations: vec![observation(1, 1, &[446.0, 447.0, 448.0, 447.5, 446.5])],
            initial_positions: vec![Point2::default()],
            mode: LocalizationMode::AnchorOnly,
            attempts: None,
        };
        let mut profile = deployment();
        profile.area = PlanarArea { minimum_x_cm: 0.0, maximum_x_cm: 2000.0, minimum_y_cm: 0.0, maximum_y_cm: 2000.0 };
        profile.anchor_only.persistent_sd_rel = 8.0;
        profile.anchor_only.height_nodes = vec![0.0];
        profile.anchor_only.grid_step = 20.0;
        (problem, profile)
    }

    /// The ring posterior around the anchor covers
    /// more than a thousand cells of 20 cm. A cap of 200 cells once left three quarters of its mass outside; the
    /// adaptive posterior keeps every cell of the smallest set with all but EPSILON of the mass, and certifies what it
    /// leaves out.
    #[test]
    fn a_ring_posterior_keeps_its_mass_without_a_cap() {
        let (problem, profile) = ring_problem();
        let result = solve(&problem, &profile).unwrap();
        let settings = &profile.anchor_only;
        let setup = setup(&problem, &profile);
        let evaluator = agent_evaluator(&setup.grid, &setup.inputs, 0, settings, &setup.weights, setup.inputs.law(0, setup.law));
        let support = adaptive(&evaluator, GUARD_NODES);
        let values = exhaustive(&evaluator);
        let comparison = compare(&support, &values);
        assert!(comparison.leading > 1000 && comparison.kept >= comparison.leading, "{comparison:?}");
        // today's support under a cap of 200 cells left more than half of the ring's mass outside
        let range = today::range_support(&setup.grid, &setup.inputs.links[0], settings, true);
        let law = setup.inputs.law(0, setup.law);
        let capped = today::agent_support(&setup.grid, &range, &setup.inputs.present[0], &setup.inputs.missing[0], settings, law, 200);
        assert!(capped.cells.len() == 200 && capped.loglik.len() == 200 && capped.capped.unwrap() > 0.5, "{:?}", capped.capped);
        assert_eq!(result.metrics.guarded_supports, Some(0));
        let dropped = result.metrics.largest_dropped_mass.unwrap();
        assert!(dropped <= EPSILON && dropped == support.dropped_share(), "{dropped}");
        let serialized = serde_json::to_value(&result.metrics).unwrap();
        assert_eq!(serialized["guarded_supports"], 0);
        assert!(serialized.get("capped_supports").is_none() && serialized["posterior_cells"].as_u64().unwrap() > 0);
    }

    /// Ten packets of a Gaussian core without offset or excess, the reduced model without both, on three anchors whose
    /// third link reads 500 cm long for the first agent. Its likelihood falls below the table's linear range within a metre of the readings,
    /// where the log domain sum once took the kernel weight of a missing offset as zero, so every cell of the agent was
    /// not finite, its support empty, and the draw from it aborted the process.
    #[test]
    fn a_gaussian_core_without_offset_or_excess_keeps_a_finite_posterior() {
        let anchors = vec![
            Anchor { id: 1, position: Point2::new(0.0, 0.0), height_cm: 100.0 },
            Anchor { id: 2, position: Point2::new(1000.0, 0.0), height_cm: 100.0 },
            Anchor { id: 3, position: Point2::new(500.0, 1000.0), height_cm: 100.0 },
        ];
        let truth = [[400.0, 300.0], [650.0, 500.0]];
        let mut observations = Vec::new();
        for (i, agent) in truth.iter().enumerate() {
            for anchor in &anchors {
                let d = (agent[0] - anchor.position.x).hypot(agent[1] - anchor.position.y).hypot(100.0);
                let error = if i == 0 && anchor.id == 3 { 500.0 } else { 0.0 };
                let samples: Vec<f64> = (0..10).map(|k| d + error + 3.0 * (2.0 * uniform(10 * i as u64 + anchor.id as u64, k) - 1.0)).collect();
                observations.push(observation(i as u32 + 1, anchor.id, &samples));
            }
        }
        let problem = PreparedProblem {
            agent_ids: vec![1, 2],
            anchors,
            observations,
            initial_positions: vec![Point2::default(); 2],
            mode: LocalizationMode::AnchorOnly,
            attempts: None,
        };
        let mut profile = deployment();
        profile.anchor_only.excess_probability = 0.0;
        profile.anchor_only.persistent_sd_rel = 0.0;
        profile.anchor_only.nu = 1e6;
        profile.anchor_only.height_nodes = vec![0.0];
        let result = solve(&problem, &profile).unwrap();
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
        assert!(result.positions[1].distance(Point2::new(truth[1][0], truth[1][1])) < 10.0, "{:?}", result.positions);
        // an agent without a single finite cell has no support to draw from
        let grid = Grid::new(&profile.area, profile.anchor_only.grid_step).unwrap();
        assert_eq!(joint_draws(&grid, &Support::empty(&grid, false, f64::NEG_INFINITY), None, &mut Rng(2), 8, &mut 0), Err(Undrawn::Empty));
        assert!(joint_draws(&grid, &Support::uniform(&grid), None, &mut Rng(3), 8, &mut 0).is_ok());
    }

    /// Three anchors near the left edge of a 2000 by 1000 cm area. Agent 1 ranges anchors 1 and 2, agent 2 has no
    /// observation and agent 3 answered anchor 3 with an inadmissible sample only.
    fn one_sided_problem() -> (PreparedProblem, DeploymentProfile) {
        let anchors = vec![
            Anchor { id: 1, position: Point2::new(100.0, 100.0), height_cm: 100.0 },
            Anchor { id: 2, position: Point2::new(100.0, 900.0), height_cm: 100.0 },
            Anchor { id: 3, position: Point2::new(300.0, 500.0), height_cm: 100.0 },
        ];
        let problem = PreparedProblem {
            agent_ids: vec![1, 2, 3],
            anchors,
            observations: vec![
                observation(1, 1, &[330.0, 332.0]),
                observation(1, 2, &[610.0, 606.0]),
                RangeObservation { from: 3, to: 3, tof_ns: vec![], calibrated_samples: vec![CalibratedRangeSample::new(-4.0, 6.0)], to_is_anchor: true },
            ],
            initial_positions: vec![Point2::default(); 3],
            mode: LocalizationMode::AnchorOnly,
            attempts: None,
        };
        let mut profile = deployment();
        profile.area.maximum_x_cm = 2000.0;
        // the reach is 400 cm, since the quantile distance is the 608 cm median of agent 1's link to anchor 2
        profile.anchor_only.presence.factor = 400.0 / 608.0;
        (problem, profile)
    }



    /// A fixed uniform draw in [0, 1) for a pair of indices.
    fn uniform(a: u64, b: u64) -> f64 {
        let mut z = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_add(0x632b_e59b_d9b4_e019);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Four agents on the floor of a 20 by 15 m room with an anchor in every ceiling corner and ten packets per link
    /// within 2 cm of the slant distance, with the agents' true positions.
    fn ten_packet_problem() -> (PreparedProblem, [[f64; 2]; 4]) {
        let calibration = CalibrationModel::default();
        let anchors = vec![
            Anchor { id: 1, position: Point2::new(0.0, 0.0), height_cm: 250.0 },
            Anchor { id: 2, position: Point2::new(2000.0, 0.0), height_cm: 250.0 },
            Anchor { id: 3, position: Point2::new(2000.0, 1500.0), height_cm: 250.0 },
            Anchor { id: 4, position: Point2::new(0.0, 1500.0), height_cm: 250.0 },
        ];
        let agents = [[420.0, 380.0], [1310.0, 520.0], [760.0, 1130.0], [1650.0, 1240.0]];
        let mut observations = Vec::new();
        for (i, agent) in agents.iter().enumerate() {
            for anchor in &anchors {
                let d = (agent[0] - anchor.position.x).hypot(agent[1] - anchor.position.y).hypot(250.0);
                let samples = (0..10)
                    .map(|k| {
                        let y = d + 2.0 * (2.0 * uniform(100 * i as u64 + anchor.id as u64, k) - 1.0);
                        CalibratedRangeSample::new(y, calibration.scale_cm(y))
                    })
                    .collect();
                observations.push(RangeObservation { from: i as u32 + 1, to: anchor.id, tof_ns: vec![], calibrated_samples: samples, to_is_anchor: true });
            }
        }
        let problem = PreparedProblem {
            agent_ids: vec![1, 2, 3, 4],
            anchors,
            observations,
            initial_positions: vec![Point2::default(); 4],
            mode: LocalizationMode::AnchorOnly,
            attempts: None,
        };
        (problem, agents)
    }

    /// The corrected Bern link model over the room with a margin of two metres, under the profile's default seed for
    /// `seed` zero.
    fn ten_packet_profile(seed: u64) -> DeploymentProfile {
        DeploymentProfile {
            area: PlanarArea { minimum_x_cm: -200.0, maximum_x_cm: 2200.0, minimum_y_cm: -200.0, maximum_y_cm: 1700.0 },
            anchor_only: AnchorOnlyProfile {
                kappa: 0.28,
                excess_probability: 0.31,
                excess_mean_rel: 8.0,
                persistent_sd_rel: 2.5,
                height_nodes: vec![0.0],
                height_weights: vec![1.0],
                grid_step: 10.0,
                joint_samples: 96,
                seed: if seed == 0 { AnchorOnlyProfile::default().seed } else { seed },
                ..AnchorOnlyProfile::default()
            },
        }
    }

    /// The four agents of the room under the corrected Bern link model. Past the upper end of a ten packet table
    /// the continuation once rose to thousands of nats above the table's peak, where the mode term's three Newton steps
    /// had not reached a mode, and two agents went to that ring around an anchor, 640 and 1470 cm from their truth.
    ///
    /// The pairwise decision fixes the layout up to a rigid motion, which the last bits of the joint samples move as a
    /// whole. Over 200 seeds of the profile on the node, drawn from the adaptive posteriors, the largest distance of an
    /// agent from its truth reached 59.5 cm, and 87 seeds put an agent more than 15 cm off, while every distance between
    /// two agents stayed within 15.32 cm of its truth. Every one of those seeds keeps the distances between the agents
    /// within 30 cm.
    #[test]
    fn ten_packet_links_place_every_agent_at_its_truth() {
        let (problem, agents) = ten_packet_problem();
        // the largest error of a distance between two agents under one seed of the profile, the default seed first
        let worst_pair = |seed: u64| {
            let result = solve(&problem, &ten_packet_profile(seed)).unwrap();
            let mut worst = (0.0_f64, 0, 0);
            for i in 0..agents.len() {
                for j in i + 1..agents.len() {
                    let truth = (agents[i][0] - agents[j][0]).hypot(agents[i][1] - agents[j][1]);
                    let error = (result.positions[i].distance(result.positions[j]) - truth).abs();
                    if error > worst.0 {
                        worst = (error, i, j);
                    }
                }
            }
            worst
        };
        let seeds: Vec<u64> = (0..200).collect();
        let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
        let worst: Vec<(u64, (f64, usize, usize))> = std::thread::scope(|scope| {
            let handles: Vec<_> = seeds
                .chunks(seeds.len().div_ceil(workers))
                .map(|chunk| scope.spawn(|| chunk.iter().map(|&seed| (seed, worst_pair(seed))).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
        });
        for (seed, (error, i, j)) in worst {
            assert!(error < 30.0, "seed {seed}: the distance between agents {i} and {j} is {error:.1} cm off");
        }
    }

    #[test]
    fn invalid_heights_areas_and_grids_are_rejected() {
        let (problem, profile) = one_sided_problem();
        for height in [f64::NAN, f64::NEG_INFINITY, 2.0e7] {
            let mut invalid = profile.clone();
            invalid.anchor_only.height_nodes = vec![100.0, height];
            invalid.anchor_only.height_weights = vec![1.0, 1.0];
            assert!(matches!(solve(&problem, &invalid), Err(EngineError::InvalidProfile(_))), "{height}");
        }
        let mut far = profile.clone();
        far.area.maximum_x_cm = 1.0e9;
        assert!(matches!(solve(&problem, &far), Err(EngineError::InvalidProfile(_))));
        // five million centimetres at one centimetre is beyond the axis guard of 2^22 centres and fails before allocating
        let mut fine = profile.clone();
        fine.area.maximum_x_cm = 5.0e6;
        fine.anchor_only.grid_step = 1.0;
        assert!(matches!(solve(&problem, &fine), Err(EngineError::InvalidProfile(_))));
        let mut distant = problem.clone();
        distant.anchors[0].position.y = -3.0e7;
        assert!(matches!(solve(&distant, &profile), Err(EngineError::AnchorOutOfRange(1))));
        let mut start = problem.clone();
        start.initial_positions[0].x = f64::NAN;
        assert!(matches!(solve(&start, &profile), Err(EngineError::InvalidProblem(_))));
    }

    #[test]
    fn samples_beyond_the_range_bound_answer_without_a_range() {
        let (mut problem, profile) = one_sided_problem();
        problem.observations.push(observation(2, 2, &[3.0e8, 4.0e8]));
        let prepared = prepare_links(&problem).unwrap();
        assert_eq!(prepared.links.len(), 2);
        assert_eq!(prepared.responded[1], vec![2]);
        assert!(solve(&problem, &profile).unwrap().positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    /// Four corner anchors and three agents. Agent 1's link to the anchor at (50, 50) reads 300 cm short on ten
    /// tight packets, an error neither the persistent offset nor the positive excess can explain. `ids` labels the
    /// four anchors in corner order.
    fn outlier_problem(ids: [u32; 4]) -> (PreparedProblem, Vec<[f64; 2]>) {
        let corners = [(50.0, 50.0), (950.0, 50.0), (950.0, 950.0), (50.0, 950.0)];
        let agents: Vec<[f64; 2]> = vec![[300.0, 400.0], [650.0, 300.0], [480.0, 720.0]];
        let mut observations = Vec::new();
        for (i, agent) in agents.iter().enumerate() {
            for (k, (x, y)) in corners.iter().enumerate() {
                let truth = ((agent[0] - x).powi(2) + (agent[1] - y).powi(2) + 100.0_f64.powi(2)).sqrt();
                let error = if i == 0 && k == 0 { -300.0 } else { 0.0 };
                let samples: Vec<f64> = (0..10).map(|q| truth + error + ((q * 7 + i * 3 + k) % 5) as f64 - 2.0).collect();
                observations.push(observation(i as u32 + 1, ids[k], &samples));
            }
        }
        let problem = PreparedProblem {
            agent_ids: vec![1, 2, 3],
            anchors: corners.iter().zip(ids).map(|((x, y), id)| Anchor { id, position: Point2::new(*x, *y), height_cm: 200.0 }).collect(),
            observations,
            initial_positions: vec![Point2::new(500.0, 500.0); 3],
            mode: LocalizationMode::AnchorOnly,
            attempts: None,
        };
        (problem, agents)
    }

    #[test]
    fn the_posterior_of_every_link_keeps_the_true_position_whatever_the_anchor_order() {
        let profile = deployment();
        let settings = &profile.anchor_only;
        let (first, truth) = outlier_problem([1, 2, 3, 4]);
        let setup = setup(&first, &profile);
        let near_truth = |support: &Support| {
            let mut near = false;
            support.for_each_cell(|i, j| near |= (setup.grid.xs[i] - truth[0][0]).hypot(setup.grid.ys[j] - truth[0][1]) < 15.0);
            near
        };
        // the posterior of the outlier link alone, the first link in identifier order, loses the true position
        let links = &setup.inputs.links[0];
        let outlier = links.iter().position(|(anchor, _)| (anchor.x, anchor.y) == (50.0, 50.0)).unwrap();
        let alone = CellEvaluator::new(&setup.grid, &[(links[outlier].0.site(), &links[outlier].1)], &[], &[], &settings.height_nodes, &setup.weights, None);
        assert!(!near_truth(&adaptive(&alone, GUARD_NODES)));
        // the posterior of every link keeps it, and its mode lies next to it
        let every = agent_evaluator(&setup.grid, &setup.inputs, 0, settings, &setup.weights, None);
        assert!(near_truth(&adaptive(&every, GUARD_NODES)));
        let peak = peak_of(&every, GUARD_NODES).unwrap();
        assert!((peak.x - truth[0][0]).hypot(peak.y - truth[0][1]) < 20.0, "{peak:?}");
        // the result does not depend on the anchor identifiers, the anchor order or the observation order
        let reference = solve(&first, &profile).unwrap();
        let (mut relabeled, _) = outlier_problem([9, 7, 5, 3]);
        relabeled.observations.reverse();
        relabeled.anchors.reverse();
        assert_eq!(solve(&relabeled, &profile).unwrap().positions, reference.positions);
        assert!(reference.positions[0].distance(Point2::new(truth[0][0], truth[0][1])) < 30.0, "{:?}", reference.positions);
    }

    #[test]
    fn the_order_of_the_agents_does_not_change_the_result() {
        // three ranged agents and one without any exchange, whose uninformative support draws cells uniformly
        let (mut problem, _) = outlier_problem([1, 2, 3, 4]);
        problem.agent_ids.push(7);
        problem.initial_positions.push(Point2::new(500.0, 500.0));
        let profile = deployment();
        let reference = solve(&problem, &profile).unwrap();
        // the request lists the agents in reverse order with their start positions
        let mut reversed = problem.clone();
        reversed.agent_ids.reverse();
        reversed.initial_positions.reverse();
        let result = solve(&reversed, &profile).unwrap();
        let mut positions = result.positions.clone();
        positions.reverse();
        assert_eq!(positions, reference.positions);
        assert_eq!(result.metrics.final_objective, reference.metrics.final_objective);
    }

    #[test]
    fn a_posterior_keeps_a_superset_of_the_smallest_set_with_nearly_all_the_mass() {
        let profile = deployment();
        let settings = &profile.anchor_only;
        let (problem, _) = outlier_problem([1, 2, 3, 4]);
        let setup = setup(&problem, &profile);
        // agent 2 has four accurate links, a posterior far narrower than the grid
        let evaluator = agent_evaluator(&setup.grid, &setup.inputs, 1, settings, &setup.weights, None);
        let support = adaptive(&evaluator, GUARD_NODES);
        let values = exhaustive(&evaluator);
        let comparison = compare(&support, &values);
        assert!(comparison.kept >= comparison.leading && comparison.kept < values.len() / 10, "{comparison:?}");
        // a flat posterior over the whole area keeps every cell in one coarse leaf
        let flat = CellEvaluator::new(&setup.grid, &[], &[], &[], &settings.height_nodes, &setup.weights, None);
        let flat_support = adaptive(&flat, GUARD_NODES);
        assert_eq!(flat_support.leaves.len(), 1);
        assert_eq!(compare(&flat_support, &exhaustive(&flat)).kept, setup.grid.count());
    }

    /// Sixteen agents on a 250 cm grid at the height node and eight anchors around them. Every attempt succeeds with
    /// the Gaussian law of `reach` at its slant distance, or with probability one half whatever the distance when
    /// `reach` is None, and every success carries five exact packets.
    fn presence_scenario(reach: Option<f64>) -> PreparedProblem {
        let uniform = |a: u64, b: u64| {
            let mut z = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_add(0x632b_e59b_d9b4_e019);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let corners = [(0.0, 0.0), (500.0, -100.0), (1000.0, 0.0), (1100.0, 500.0), (1000.0, 1000.0), (500.0, 1100.0), (0.0, 1000.0), (-100.0, 500.0)];
        let anchors: Vec<Anchor> =
            corners.iter().enumerate().map(|(k, (x, y))| Anchor { id: k as u32 + 1, position: Point2::new(*x, *y), height_cm: 200.0 }).collect();
        let mut observations = Vec::new();
        for i in 0..16 {
            let (x, y) = (125.0 + 250.0 * (i % 4) as f64, 125.0 + 250.0 * (i / 4) as f64);
            for anchor in &anchors {
                let d = ((x - anchor.position.x).powi(2) + (y - anchor.position.y).powi(2) + 100.0_f64.powi(2)).sqrt();
                let p = reach.map_or(0.5, |r| (-d * d / (2.0 * r * r)).exp());
                if uniform(i as u64, anchor.id as u64) < p {
                    observations.push(observation(i as u32 + 1, anchor.id, &[d - 2.0, d - 1.0, d, d + 1.0, d + 2.0]));
                }
            }
        }
        PreparedProblem {
            agent_ids: (1..=16).collect(),
            anchors,
            observations,
            initial_positions: vec![Point2::new(500.0, 500.0); 16],
            mode: LocalizationMode::AnchorOnly,
            attempts: None,
        }
    }

    /// The maximum likelihood reach without the adaptive posterior: the candidate laws, the log likelihood of every
    /// agent's attempt outcomes under each integrated over every cell of the grid, summed over the agents, and the law
    /// the rule chooses by them. None where the rule judges no candidate.
    fn reach_over_every_cell(grid: &Grid, inputs: &AgentInputs, settings: &AnchorOnlyProfile) -> Option<(Vec<ResolvedPresence>, Vec<f64>, Option<ResolvedPresence>)> {
        let attempts: usize = inputs.present.iter().zip(&inputs.missing).map(|(present, missing)| present.len() + missing.len()).sum();
        let successes: usize = inputs.present.iter().map(Vec::len).sum();
        let candidates = settings.presence.ml_candidates(&inputs.successful, attempts, successes)?;
        let weights = height_log_weights(settings);
        let mut values = vec![0.0; candidates.len()];
        for rank in (0..inputs.order.len()).filter(|rank| !inputs.present[*rank].is_empty()) {
            let evaluator = agent_evaluator(grid, inputs, rank, settings, &weights, None);
            for (value, agent) in values.iter_mut().zip(outcome_log_likelihoods(&evaluator, &Support::uniform(grid), &candidates, &mut 0).1) {
                *value += agent;
            }
        }
        let chosen = ml_choice(&candidates, &values, attempts, successes);
        Some((candidates, values, chosen))
    }

    #[test]
    fn the_maximum_likelihood_reach_integrates_each_agent_over_its_range_posterior() {
        let mut profile = deployment();
        profile.anchor_only.presence = crate::PresenceProfile { reach: crate::ReachRule::Ml, ..crate::PresenceProfile::default() };
        let settings = &profile.anchor_only;
        let grid = Grid::new(&profile.area, settings.grid_step).unwrap();
        let law = |problem: &PreparedProblem| {
            let prepared = prepare_links(problem).unwrap();
            let inputs = agent_inputs(problem, &prepared, settings, log_t_norm_for(settings.nu));
            assert!(inputs.missing.iter().any(|missing| !missing.is_empty()));
            let chosen = scenario_law(&grid, &inputs, settings, GUARD_NODES, &mut Tally::new(inputs.order.len()));
            // the integral over every cell of the grid, without the adaptive posterior, chooses the same law
            let whole = reach_over_every_cell(&grid, &inputs, settings).and_then(|(_, _, chosen)| chosen);
            assert_eq!(chosen, whole);
            chosen
        };
        for reach in [500.0, 800.0] {
            let chosen = law(&presence_scenario(Some(reach))).expect("a law").reach;
            assert!(chosen >= reach / std::f64::consts::SQRT_2 && chosen <= reach * std::f64::consts::SQRT_2, "{chosen} for {reach}");
        }
        // outcomes that do not depend on the distance leave no law
        assert!(law(&presence_scenario(None)).is_none());
        let result = solve(&presence_scenario(Some(600.0)), &profile).unwrap();
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    /// `copies` copies of the presence scenario of reach 520 cm with every agent's attempts at its eight anchors
    /// declared, and 1500 cm to the right of it a mirror cluster whose anchors stand at 200 cm. Agent 101 ranges A and B
    /// with ten packets each, whose rings at the height node cross at M1 = (2500, 1384) and at its mirror image across
    /// the line AB, and C with two packets true at M1, which leave the mirror image some nats below M1. It failed twelve
    /// anchors 30 cm around M1, which agent 102 answered with every other anchor of the cluster. Agent 900 at (200, 1800)
    /// ranges three anchors and failed `failures` anchors 200 cm around it, which moves the margins between the
    /// candidate reaches.
    fn mirror_scenario(copies: u32, failures: usize) -> PreparedProblem {
        let packets = |d: f64, count: usize, scale: f64| -> Vec<CalibratedRangeSample> {
            (0..count).map(|k| CalibratedRangeSample::new(d + 0.3 * (k as f64 - (count as f64 - 1.0) / 2.0), scale)).collect()
        };
        let ranged = |from: u32, to: u32, samples: Vec<CalibratedRangeSample>| RangeObservation { from, to, tof_ns: vec![], calibrated_samples: samples, to_is_anchor: true };
        let attempt = |from: u32, to: u32| crate::LinkAttempt { from, to, to_is_anchor: true };
        let mut problem = presence_scenario(Some(520.0));
        let base = problem.clone();
        for copy in 1..copies {
            for o in &base.observations {
                problem.observations.push(RangeObservation { from: o.from + 1000 * copy, ..o.clone() });
            }
            problem.agent_ids.extend(base.agent_ids.iter().map(|a| a + 1000 * copy));
            problem.initial_positions.extend(base.initial_positions.iter().copied());
        }
        let mut attempts: Vec<crate::LinkAttempt> =
            (0..copies).flat_map(|copy| (1..=16u32).flat_map(move |agent| (1..=8u32).map(move |anchor| attempt(agent + 1000 * copy, anchor)))).collect();
        // the agents stand at the height node, 100 cm, and the anchors at 200 cm
        let slant = |a: &Anchor, p: (f64, f64)| (a.position.x - p.0).hypot(a.position.y - p.1).hypot(100.0);
        let m1 = (2500.0, 1384.0);
        let mut cluster = vec![
            Anchor { id: 201, position: Point2::new(2000.0, 1000.0), height_cm: 200.0 },
            Anchor { id: 202, position: Point2::new(3000.0, 1000.0), height_cm: 200.0 },
            Anchor { id: 203, position: Point2::new(2500.0, 1900.0), height_cm: 200.0 },
        ];
        for k in 0..12 {
            let angle = std::f64::consts::TAU * k as f64 / 12.0;
            cluster.push(Anchor { id: 210 + k as u32, position: Point2::new(m1.0 + 30.0 * angle.cos(), m1.1 + 30.0 * angle.sin()), height_cm: 200.0 });
        }
        problem.observations.push(ranged(101, 201, packets(slant(&cluster[0], m1), 10, 6.0)));
        problem.observations.push(ranged(101, 202, packets(slant(&cluster[1], m1), 10, 6.0)));
        problem.observations.push(ranged(101, 203, packets(slant(&cluster[2], m1), 2, 1.0)));
        attempts.extend(cluster.iter().map(|a| attempt(101, a.id)));
        for a in &cluster {
            problem.observations.push(ranged(102, a.id, packets(slant(a, (2500.0, 1600.0)), 10, 6.0)));
            attempts.push(attempt(102, a.id));
        }
        problem.anchors.extend(cluster);
        let p = (200.0, 1800.0);
        let mut own = vec![
            Anchor { id: 301, position: Point2::new(0.0, 1500.0), height_cm: 200.0 },
            Anchor { id: 302, position: Point2::new(400.0, 1500.0), height_cm: 200.0 },
            Anchor { id: 303, position: Point2::new(200.0, 2100.0), height_cm: 200.0 },
        ];
        for a in &own {
            problem.observations.push(ranged(900, a.id, packets(slant(a, p), 10, 6.0)));
            attempts.push(attempt(900, a.id));
        }
        for k in 0..failures {
            let angle = std::f64::consts::TAU * (k as f64 + 0.5) / failures as f64;
            own.push(Anchor { id: 310 + k as u32, position: Point2::new(p.0 + 200.0 * angle.cos(), p.1 + 200.0 * angle.sin()), height_cm: 200.0 });
            attempts.push(attempt(900, 310 + k as u32));
        }
        problem.anchors.extend(own);
        problem.agent_ids.extend([101, 102, 900]);
        problem.initial_positions.extend([Point2::default(); 3]);
        problem.attempts = Some(attempts);
        problem
    }

    /// The deployment of the mirror scenarios under the maximum likelihood reach, a 3600 by 2400 cm area at 20 cm.
    fn mirror_profile() -> DeploymentProfile {
        let mut profile = deployment();
        profile.area = PlanarArea { minimum_x_cm: -200.0, maximum_x_cm: 3400.0, minimum_y_cm: -200.0, maximum_y_cm: 2200.0 };
        profile.anchor_only.grid_step = 20.0;
        profile.anchor_only.presence = crate::PresenceProfile { reach: crate::ReachRule::Ml, ..crate::PresenceProfile::default() };
        profile
    }

    /// The maximum likelihood reach once integrated every agent over the kept cells of its range only posterior alone.
    /// Those cells hold all but a certified share of the range only mass, but the presence terms of agent 101, which
    /// failed twelve anchors around one of its two mirror places, favour the other, part of which lies outside them: the
    /// integral over the kept cells fell up to 6.6 nats short of the one over every cell, and on these two scenarios the
    /// choice turned the presence law off, and took 444.8 cm, where the integral over every cell takes 629 cm. The bounds
    /// of the certified choice hold the integral over every cell, they refine the laws they cannot tell apart, and the
    /// choice is that of every cell.
    #[test]
    fn the_maximum_likelihood_reach_is_certified_against_the_integral_over_every_cell() {
        let profile = mirror_profile();
        let settings = &profile.anchor_only;
        let grid = Grid::new(&profile.area, settings.grid_step).unwrap();
        for (copies, failures, kept_alone) in [(4, 22, None), (5, 30, Some(2))] {
            let problem = mirror_scenario(copies, failures);
            let prepared = prepare_links(&problem).unwrap();
            let inputs = agent_inputs(&problem, &prepared, settings, log_t_norm_for(settings.nu));
            let (candidates, every_cell, whole) = reach_over_every_cell(&grid, &inputs, settings).unwrap();
            let judgement = judge_reach(&grid, &inputs, settings, GUARD_NODES, &mut Tally::new(inputs.order.len())).unwrap();
            assert_eq!(judgement.candidates, candidates);
            assert_eq!(whole.map(|law| law.reach), Some(candidates[3].reach), "{copies} copies: 629 cm over every cell");
            assert_eq!(judgement.chosen, whole, "{copies} copies");
            assert!(judgement.certified && !judgement.refined.is_empty(), "{copies} copies: refined {:?}", judgement.refined);
            // the values over the kept cells alone chose another law
            let attempts: usize = inputs.present.iter().zip(&inputs.missing).map(|(present, missing)| present.len() + missing.len()).sum();
            let successes: usize = inputs.present.iter().map(Vec::len).sum();
            let alone = ml_choice(&candidates, &judgement.values, attempts, successes);
            assert_eq!(alone, kept_alone.map(|k: usize| candidates[k]), "{copies} copies");
            // the bounds hold the integral over every cell, whose values lie below the kept cells' by up to 6.6 nats
            for (j, exact) in every_cell.iter().enumerate() {
                let tolerance = 1e-9 * (1.0 + exact.abs());
                assert!(judgement.lower[j] - judgement.slack - tolerance <= *exact && *exact <= judgement.upper[j] + tolerance, "law {j}: {exact} outside [{} - {}, {}]", judgement.lower[j], judgement.slack, judgement.upper[j]);
            }
            let short = every_cell.iter().zip(&judgement.values).map(|(exact, kept)| exact - kept).fold(0.0, f64::max);
            println!("{copies} copies: chosen {:?} over every cell {:?} over the kept cells alone {:?}, refined {:?}, kept cells {short:.3} nats short", judgement.chosen.map(|law| law.reach), whole.map(|law| law.reach), alone.map(|law| law.reach), judgement.refined);
            assert!(short > 6.0, "{copies} copies: {short}");
            assert_eq!(judgement.null, ml_null(attempts, successes));
        }
    }

    /// The solve under the maximum likelihood reach of a mirror scenario positions every agent bitwise as the solve under
    /// the quantile rule with the law the integral over every cell chooses, 629 cm, the ninety percent distance over the
    /// square root of two.
    #[test]
    fn the_certified_reach_solves_as_the_law_over_every_cell() {
        let profile = mirror_profile();
        let mut fixed = profile.clone();
        fixed.anchor_only.presence = crate::PresenceProfile { factor: 2f64.powf(-0.5), ..crate::PresenceProfile::default() };
        for (copies, failures) in [(4, 22), (5, 30)] {
            let problem = mirror_scenario(copies, failures);
            let (chosen, quantile) = (solve(&problem, &profile).unwrap(), solve(&problem, &fixed).unwrap());
            assert_eq!(chosen.positions, quantile.positions, "{copies} copies");
        }
    }

    /// A node guard below the four children of one node is refused where a solve sets it, since a posterior that has kept
    /// no leaf yet could then split no node, drop every one and be named not finite.
    #[test]
    #[should_panic(expected = "cannot hold the four children of a node")]
    fn a_node_guard_below_four_nodes_is_refused() {
        let (problem, profile) = one_sided_problem();
        let _ = solve_guarded(&problem, &profile, 3);
    }

    /// The range only posteriors of the maximum likelihood reach take the node guard of their solve. At a guard of 96
    /// nodes the range only posteriors of three agents of the presence scenario are cut, and the posteriors under the
    /// chosen law of two, and the solve counts every one of the three. The reach once took its range only posteriors
    /// under the production guard whatever the solve's, and the solve counted two.
    #[test]
    fn the_maximum_likelihood_reach_takes_the_node_guard_of_its_solve() {
        let mut profile = deployment();
        profile.anchor_only.presence = crate::PresenceProfile { reach: crate::ReachRule::Ml, ..crate::PresenceProfile::default() };
        let settings = &profile.anchor_only;
        let problem = presence_scenario(Some(600.0));
        let prepared = prepare_links(&problem).unwrap();
        let inputs = agent_inputs(&problem, &prepared, settings, log_t_norm_for(settings.nu));
        let grid = Grid::new(&profile.area, settings.grid_step).unwrap();
        let weights = height_log_weights(settings);
        let cut = (0..inputs.order.len()).filter(|&rank| adaptive(&agent_evaluator(&grid, &inputs, rank, settings, &weights, None), 96).guarded).count();
        assert_eq!(cut, 3);
        let result = solve_guarded(&problem, &profile, 96).unwrap();
        assert!(result.metrics.guarded_supports.unwrap() >= cut, "{:?}", result.metrics);
        assert_eq!(solve(&problem, &profile).unwrap().metrics.guarded_supports, Some(0));
    }

    /// An agent whose posterior the node guard left without a kept cell is drawn uniformly over the area, and its
    /// posterior counts as guarded with its whole mass left out, never as not finite. An agent whose posterior keeps no
    /// cell without the guard is not finite on any cell.
    #[test]
    fn a_support_the_guard_left_empty_is_drawn_uniformly() {
        let (_, profile) = one_sided_problem();
        let grid = Grid::new(&profile.area, profile.anchor_only.grid_step).unwrap();
        let emptied = Support::empty(&grid, true, 3.5);
        assert_eq!(emptied.dropped_share(), 1.0);
        let mut tally = Tally::new(1);
        tally.record(0, &emptied);
        assert!(tally.guarded[0] && tally.dropped[0] == 1.0);
        let draws = agent_draws(&grid, &emptied, None, &mut Rng(5), 4000, &mut 0, 7).unwrap();
        let uniform = agent_draws(&grid, &Support::uniform(&grid), None, &mut Rng(5), 4000, &mut 0, 7).unwrap();
        assert_eq!(draws, uniform);
        let mean_x = draws.iter().map(|p| p[0]).sum::<f64>() / draws.len() as f64;
        assert!((mean_x - 1000.0).abs() < 50.0, "{mean_x}");
        assert!(matches!(
            agent_draws(&grid, &Support::empty(&grid, false, f64::NEG_INFINITY), None, &mut Rng(5), 8, &mut 0, 7),
            Err(EngineError::NonFinitePosterior(7))
        ));
    }

    #[test]
    fn agents_without_a_responded_exchange_carry_no_presence_term() {
        let (problem, profile) = one_sided_problem();
        let settings = &profile.anchor_only;
        let prepared = prepare_links(&problem).unwrap();
        let inputs = agent_inputs(&problem, &prepared, settings, log_t_norm_for(settings.nu));
        let counts = |lists: &[Vec<Anchor3>]| lists.iter().map(Vec::len).collect::<Vec<_>>();
        assert_eq!(inputs.links.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 0, 0]);
        // agent 3's inadmissible exchange is present and activates anchor 3 for the other agents
        assert_eq!(counts(&inputs.present), vec![2, 0, 1]);
        assert_eq!(counts(&inputs.missing), vec![1, 0, 2]);
        let law = settings.presence.resolve(&inputs.successful);
        assert!(law.is_some());
        assert!(inputs.law(0, law).is_some() && inputs.law(1, law).is_none() && inputs.law(2, law).is_some());
        let grid = Grid::new(&profile.area, settings.grid_step).unwrap();
        let weights = height_log_weights(settings);
        // the agent without a responded exchange keeps the uninformative support, every cell of the area alike
        assert!(inputs.links[1].is_empty() && inputs.law(1, law).is_none());
        let draws = joint_draws(&grid, &Support::uniform(&grid), None, &mut Rng(7), 4000, &mut 0).unwrap();
        let mean_x = draws.iter().map(|p| p[0]).sum::<f64>() / draws.len() as f64;
        assert!((mean_x - 1000.0).abs() < 50.0, "{mean_x}");
        // under the failure only rule it replaces, every anchor pushed the agent to the far edge of the area
        let every_anchor: Vec<[f64; 3]> = inputs.present[0].iter().chain(&inputs.missing[0]).map(Anchor3::site).collect();
        let pushed = CellEvaluator::new(&grid, &[], &[], &every_anchor, &settings.height_nodes, &weights, law);
        let values = exhaustive(&pushed);
        let mut ranked: Vec<usize> = (0..values.len()).collect();
        ranked.sort_by(|a, b| values[*b].total_cmp(&values[*a]));
        let mean_x = ranked[..300].iter().map(|c| grid.xs[c % grid.xs.len()]).sum::<f64>() / 300.0;
        assert!(mean_x > 1400.0, "the 300 likeliest cells at {mean_x}");
        // the agent with an inadmissible exchange only keeps its presence pattern without any range term
        let answered = agent_evaluator(&grid, &inputs, 2, settings, &weights, inputs.law(2, law));
        let best = peak_of(&answered, GUARD_NODES).unwrap();
        assert!((best.x - 300.0).hypot(best.y - 500.0) < 400.0, "{best:?}");
        let result = solve(&problem, &profile).unwrap();
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    /// The synthetic problems of these tests with their deployments: the ring, the outlier link, the one sided room,
    /// the presence scenario under the maximum likelihood reach and the ten packet room.
    fn synthetic_problems() -> Vec<(&'static str, PreparedProblem, DeploymentProfile)> {
        let mut ml = deployment();
        ml.anchor_only.presence = crate::PresenceProfile { reach: crate::ReachRule::Ml, ..crate::PresenceProfile::default() };
        let (ring, ring_profile) = ring_problem();
        let (one_sided, one_sided_profile) = one_sided_problem();
        vec![
            ("ring", ring, ring_profile),
            ("outlier", outlier_problem([1, 2, 3, 4]).0, deployment()),
            ("one sided", one_sided, one_sided_profile),
            ("presence", presence_scenario(Some(600.0)), ml),
            ("ten packets", ten_packet_problem().0, ten_packet_profile(0)),
        ]
    }

    /// Every informative agent of the synthetic problems under the law its solve resolves: the grid, the inputs, the
    /// agent's rank and law.
    fn informative_agents(problem: &PreparedProblem, profile: &DeploymentProfile) -> (Setup, Vec<(usize, Option<ResolvedPresence>)>) {
        let setup = setup(problem, profile);
        let agents = (0..setup.inputs.order.len())
            .map(|rank| (rank, setup.inputs.law(rank, setup.law)))
            .filter(|(rank, law)| !setup.inputs.present[*rank].is_empty() && !(setup.inputs.links[*rank].is_empty() && law.is_none()))
            .collect();
        (setup, agents)
    }

    /// On every agent of the synthetic problems the adaptive posterior against the exhaustive reference: the kept cells
    /// carry bitwise the values of the uniform grid, the smallest leading set with 1 - EPSILON of the mass is kept, the
    /// certified dropped mass is at least the exact one and the reported share at most EPSILON.
    #[test]
    fn every_agent_keeps_its_cells_with_the_values_of_the_uniform_grid() {
        for (name, problem, profile) in synthetic_problems() {
            let (setup, agents) = informative_agents(&problem, &profile);
            assert!(!agents.is_empty(), "{name}");
            for (rank, law) in agents {
                let evaluator = agent_evaluator(&setup.grid, &setup.inputs, rank, &profile.anchor_only, &setup.weights, law);
                let values = exhaustive(&evaluator);
                let support = adaptive(&evaluator, GUARD_NODES);
                let comparison = compare(&support, &values);
                assert!(!support.guarded && comparison.dropped < comparison.total && comparison.kept >= comparison.leading, "{name} agent {rank}: {comparison:?}");
                // the draws of a coarse leaf evaluate its cells with the same evaluator, bitwise the reference too
                let (nx, _) = evaluator.shape();
                support.for_each_cell(|i, j| assert_eq!(evaluator.value(i, j).to_bits(), values[j * nx + i].to_bits()));
            }
        }
    }

    /// Today's loops over the whole grid, without the prune, give every cell of every agent of the synthetic problems
    /// bitwise the evaluator's value, with the solve's law and without presence.
    #[test]
    fn todays_loops_give_every_cell_the_evaluators_value() {
        for (name, problem, profile) in synthetic_problems() {
            let settings = &profile.anchor_only;
            let (setup, agents) = informative_agents(&problem, &profile);
            for (rank, law) in agents {
                for law in [law, None] {
                    let evaluator = agent_evaluator(&setup.grid, &setup.inputs, rank, settings, &setup.weights, law);
                    let range = today::range_support(&setup.grid, &setup.inputs.links[rank], settings, false);
                    let frozen = today::agent_support(&setup.grid, &range, &setup.inputs.present[rank], &setup.inputs.missing[rank], settings, law, usize::MAX);
                    let values = exhaustive(&evaluator);
                    assert_eq!(frozen.marginals.len(), values.len());
                    for (c, (a, b)) in frozen.marginals.iter().zip(&values).enumerate() {
                        assert_eq!(a.to_bits(), b.to_bits(), "{name} agent {rank} cell {c}: {a} {b}");
                    }
                }
            }
        }
    }

    /// Neither the anchor identifiers, the order of the anchors and the observations nor the order of the agents changes
    /// an agent's adaptive posterior, its kept leaves bitwise alike, or the work the solve counts.
    #[test]
    fn the_posteriors_do_not_depend_on_the_order_of_the_request() {
        let profile = deployment();
        let settings = &profile.anchor_only;
        let (first, _) = outlier_problem([1, 2, 3, 4]);
        let (mut relabeled, _) = outlier_problem([9, 7, 5, 3]);
        relabeled.observations.reverse();
        relabeled.anchors.reverse();
        relabeled.agent_ids.reverse();
        relabeled.initial_positions.reverse();
        let (a, b) = (setup(&first, &profile), setup(&relabeled, &profile));
        let leaves = |s: &Setup, rank: usize| adaptive(&agent_evaluator(&s.grid, &s.inputs, rank, settings, &s.weights, s.inputs.law(rank, s.law)), GUARD_NODES).leaves;
        for rank in 0..3 {
            assert_eq!(leaves(&a, rank), leaves(&b, rank), "agent {rank}");
        }
        let (x, y) = (solve(&first, &profile).unwrap(), solve(&relabeled, &profile).unwrap());
        let mut reversed = y.positions.clone();
        reversed.reverse();
        assert_eq!(reversed, x.positions);
        assert_eq!((x.metrics.posterior_nodes, x.metrics.posterior_cells), (y.metrics.posterior_nodes, y.metrics.posterior_cells));
    }

    /// A node guard far below what the posteriors need cuts them and reports it with the certified share it leaves out,
    /// and the solve still answers.
    #[test]
    fn a_tiny_node_guard_is_reported_not_a_failure() {
        let (problem, profile) = one_sided_problem();
        let result = solve_guarded(&problem, &profile, 24).unwrap();
        assert!(result.metrics.guarded_supports.unwrap() >= 1, "{:?}", result.metrics);
        let dropped = result.metrics.largest_dropped_mass.unwrap();
        assert!(dropped > 0.0 && dropped <= 1.0, "{dropped}");
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
        let free = solve(&problem, &profile).unwrap();
        assert_eq!(free.metrics.guarded_supports, Some(0));
        assert!(free.metrics.largest_dropped_mass.unwrap() <= EPSILON);
    }

    /// Today's per cell arithmetic before the adaptive posterior, `range_support` and `agent_support` of a827125 copied
    /// verbatim but for the new shape of the grid, whose cells are spelled out and whose slant span takes the anchor as
    /// an array, and for two switches: the prune after all links runs where `prune` holds, and the cap on the kept cells,
    /// once `top_cells`, is `cap`. `agent_support` also returns the marginal of every range cell in the order of the
    /// range cells, before the ranking.
    pub(crate) mod today {
        use super::super::*;
        use crate::posterior::{detection_loglik, log_add_exp};

        const PRUNE_MARGIN: f64 = 60.0;
        const SUPPORT_MASS: f64 = 0.9999;

        pub(crate) struct RangeSupport {
            pub(crate) cells: Vec<usize>,
            pub(crate) terms: Vec<Vec<f64>>,
        }

        pub(crate) fn range_support(grid: &Grid, links: &[(Anchor3, LinkTable)], settings: &AnchorOnlyProfile, prune: bool) -> RangeSupport {
            let zs = &settings.height_nodes;
            let zw = height_log_weights(settings);
            let count = grid.count();
            let mut acc: Vec<Vec<f64>> = vec![vec![0.0; count]; zs.len()];
            for (anchor, table) in links {
                let (nearest, farthest) = grid.slant_span(&anchor.site(), zs);
                let table = table.far(nearest, farthest);
                let dz2: Vec<f64> = zs.iter().map(|z| (z - anchor.z) * (z - anchor.z)).collect();
                // the cells row by row, cell c at (xs[c % columns], ys[c / columns]), every height node of a cell in turn
                let columns = grid.xs.len();
                for (r, y) in grid.ys.iter().enumerate() {
                    let dy = y - anchor.y;
                    for (i, x) in grid.xs.iter().enumerate() {
                        let dx = x - anchor.x;
                        let horizontal = dx * dx + dy * dy;
                        for (row, dz2) in acc.iter_mut().zip(&dz2) {
                            row[r * columns + i] += table.eval((horizontal + dz2).sqrt());
                        }
                    }
                }
            }
            let marginal: Vec<f64> =
                (0..count).map(|c| (0..zs.len()).map(|zi| acc[zi][c] + zw[zi]).fold(f64::NEG_INFINITY, log_add_exp)).collect();
            let best = marginal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let cells: Vec<usize> = (0..count).filter(|&c| !prune || marginal[c] > best - PRUNE_MARGIN).collect();
            let terms = acc.iter().map(|row| cells.iter().map(|&c| row[c]).collect()).collect();
            RangeSupport { cells, terms }
        }

        pub(crate) struct AgentSupport {
            pub(crate) cells: Vec<usize>,
            pub(crate) loglik: Vec<f64>,
            pub(crate) capped: Option<f64>,
            pub(crate) marginals: Vec<f64>,
        }

        pub(crate) fn agent_support(
            grid: &Grid,
            range: &RangeSupport,
            present: &[Anchor3],
            missing: &[Anchor3],
            settings: &AnchorOnlyProfile,
            detection: Option<ResolvedPresence>,
            cap: usize,
        ) -> AgentSupport {
            let zs = &settings.height_nodes;
            let zw = height_log_weights(settings);
            let mut terms = range.terms.clone();
            if let Some(law) = detection {
                let opportunities: Vec<(&Anchor3, bool)> = present
                    .iter()
                    .map(|a| (a, true))
                    .chain(missing.iter().map(|a| (a, false)))
                    .collect();
                let outcomes: Vec<bool> = opportunities.iter().map(|(_, p)| *p).collect();
                let mut dists = vec![0.0; opportunities.len()];
                for (zi, z) in zs.iter().enumerate() {
                    for (j, &c) in range.cells.iter().enumerate() {
                        let (x, y) = (grid.xs[c % grid.xs.len()], grid.ys[c / grid.xs.len()]);
                        for (k, (a, _)) in opportunities.iter().enumerate() {
                            let dx = x - a.x;
                            let dy = y - a.y;
                            let dz = z - a.z;
                            dists[k] = (dx * dx + dy * dy + dz * dz).sqrt();
                        }
                        terms[zi][j] += detection_loglik(&dists, &outcomes, &law);
                    }
                }
            }
            let mut ranked: Vec<(usize, f64)> = (0..range.cells.len())
                .map(|j| (j, (0..zs.len()).map(|zi| terms[zi][j] + zw[zi]).fold(f64::NEG_INFINITY, log_add_exp)))
                .collect();
            let marginals = ranked.iter().map(|(_, m)| *m).collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            // the smallest leading set with SUPPORT_MASS of the mass, the running sum in rank order ends at the total
            let total = ranked.iter().map(|(_, m)| *m).fold(f64::NEG_INFINITY, log_add_exp);
            let threshold = total + SUPPORT_MASS.ln();
            let mut held = f64::NEG_INFINITY;
            let mut kept = ranked.len();
            for (rank, (_, m)) in ranked.iter().enumerate() {
                held = log_add_exp(held, *m);
                if held >= threshold {
                    kept = rank + 1;
                    break;
                }
            }
            ranked.truncate(kept.min(cap));
            let cells: Vec<usize> = ranked.iter().map(|(j, _)| range.cells[*j]).collect();
            let loglik = ranked.iter().map(|(_, m)| *m).collect::<Vec<f64>>();
            let capped = (kept > cap).then(|| -(loglik.iter().copied().fold(f64::NEG_INFINITY, log_add_exp) - total).exp_m1());
            AgentSupport { cells, loglik, capped, marginals }
        }

        /// Whether today's support, pruned after all links and capped at `cap` cells, cut the agent, with the cells of the
        /// smallest leading set with SUPPORT_MASS of the mass of the cells that survive the prune. `range` holds the range
        /// terms of every cell, from `range_support` without the prune, and `marginals` the marginal log posterior of every
        /// cell with its presence terms, the values `agent_support` computes for the cells that survive. The prune,
        /// the ranking, the total and the leading set follow `range_support` and `agent_support` step by step.
        pub(crate) fn cap_cuts(range: &RangeSupport, marginals: &[f64], settings: &AnchorOnlyProfile, cap: usize) -> (bool, usize) {
            let zs = &settings.height_nodes;
            let zw = height_log_weights(settings);
            let count = marginals.len();
            assert_eq!(range.cells.len(), count, "the range terms of every cell");
            let marginal: Vec<f64> =
                (0..count).map(|c| (0..zs.len()).map(|zi| range.terms[zi][c] + zw[zi]).fold(f64::NEG_INFINITY, log_add_exp)).collect();
            let best = marginal.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let cells: Vec<usize> = (0..count).filter(|&c| marginal[c] > best - PRUNE_MARGIN).collect();
            let mut ranked: Vec<(usize, f64)> = cells.iter().map(|&c| (c, marginals[c])).collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
            let total = ranked.iter().map(|(_, m)| *m).fold(f64::NEG_INFINITY, log_add_exp);
            let threshold = total + SUPPORT_MASS.ln();
            let mut held = f64::NEG_INFINITY;
            let mut kept = ranked.len();
            for (rank, (_, m)) in ranked.iter().enumerate() {
                held = log_add_exp(held, *m);
                if held >= threshold {
                    kept = rank + 1;
                    break;
                }
            }
            (kept > cap, kept)
        }
    }

    /// One run of a stored unit: the run's key, its budget factor, the top_cells of its stored profile and its deployment
    /// profile without them, and for a unit solved again under a candidate of the round, the candidate's name.
    #[derive(serde::Deserialize)]
    struct StoredRun {
        key: String,
        factor: f64,
        top_cells: usize,
        deployment_profile: DeploymentProfile,
        #[serde(default)]
        candidate: Option<String>,
    }

    /// What the equivalence run finds for the maximum likelihood reach of one run: the adaptive judgement against the
    /// integral over every cell.
    #[derive(Clone, Debug, Default, serde::Serialize)]
    struct ReachFinding {
        /// The reach each side chose, none for the law off, and the choice of the values over the kept cells alone,
        /// the rule before its certificate.
        adaptive: Option<f64>,
        exhaustive: Option<f64>,
        kept_alone: Option<f64>,
        /// Whether the bounds settled the choice, and the laws refined by the posteriors under them.
        certified: bool,
        refined: usize,
        /// Laws whose exact value over every cell lies outside the bounds, and the largest distance of the values over
        /// the kept cells alone from the exact ones.
        bound_violations: usize,
        kept_deviation: f64,
        /// The margin of the exact choice over the next law or the null's value, whichever is closer.
        margin: f64,
        judgement_seconds: f64,
        exhaustive_seconds: f64,
    }

    impl ReachFinding {
        fn anomalous(&self) -> bool {
            self.adaptive != self.exhaustive || self.bound_violations > 0 || !self.certified
        }
    }

    /// The maximum likelihood reach of a setup's scenario judged by the engine, `judge_reach`, against the integral over
    /// every cell of the grid, `reach_over_every_cell`, the law the solve takes, and what compares them. The law is the
    /// judgement's, which the solve and so both sides of the posterior's comparison take.
    fn judged_reach(setup: &Setup, settings: &AnchorOnlyProfile) -> Option<(ReachFinding, Option<ResolvedPresence>)> {
        let started = Instant::now();
        let judgement = judge_reach(&setup.grid, &setup.inputs, settings, GUARD_NODES, &mut Tally::new(setup.inputs.order.len()))?;
        let judgement_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let (candidates, exact, whole) = reach_over_every_cell(&setup.grid, &setup.inputs, settings)?;
        let exhaustive_seconds = started.elapsed().as_secs_f64();
        let attempts: usize = setup.inputs.present.iter().zip(&setup.inputs.missing).map(|(present, missing)| present.len() + missing.len()).sum();
        let successes: usize = setup.inputs.present.iter().map(Vec::len).sum();
        assert_eq!(candidates, judgement.candidates);
        let tolerance = |value: f64| 1e-9 * (1.0 + value.abs());
        let bound_violations = (0..candidates.len())
            .filter(|&j| !(judgement.lower[j] - judgement.slack - tolerance(exact[j]) <= exact[j] && exact[j] <= judgement.upper[j] + tolerance(exact[j])))
            .count();
        let kept_deviation = exact.iter().zip(&judgement.values).map(|(e, v)| (e - v).abs()).filter(|d| !d.is_nan()).fold(0.0, f64::max);
        let mut sorted: Vec<f64> = exact.iter().copied().chain([judgement.null]).collect();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let finding = ReachFinding {
            adaptive: judgement.chosen.map(|law| law.reach),
            exhaustive: whole.map(|law| law.reach),
            kept_alone: ml_choice(&candidates, &judgement.values, attempts, successes).map(|law| law.reach),
            certified: judgement.certified,
            refined: judgement.refined.len(),
            bound_violations,
            kept_deviation,
            margin: sorted[0] - sorted[1],
            judgement_seconds,
            exhaustive_seconds,
        };
        Some((finding, judgement.chosen))
    }

    /// One stored unit, a line of a file of research/solid/equivalence_units.py.
    #[derive(serde::Deserialize)]
    struct StoredUnit {
        dataset: String,
        id: usize,
        label: String,
        problem: PreparedProblem,
        runs: Vec<StoredRun>,
    }

    /// What the equivalence run finds for one informative agent of one run.
    #[derive(Clone, Debug, Default, serde::Serialize)]
    struct AgentFinding {
        /// The grid's cells, the nodes bounded, the cells evaluated exactly and the cells kept, single and in coarse leaves.
        grid_cells: usize,
        nodes: usize,
        cells: usize,
        kept: usize,
        finest: usize,
        coarse: usize,
        /// The cells of the smallest leading set with 1 - EPSILON of the exact mass, and those of them left out.
        leading: usize,
        leading_left_out: usize,
        /// Kept single cells whose value is not bitwise today's, cells of coarse leaves whose evaluation is not, and cells
        /// of coarse leaves outside their bounds.
        finest_mismatches: usize,
        coarse_mismatches: usize,
        bound_violations: usize,
        /// Whether the certified dropped mass or the reported share fell below the exact one.
        certified_below_exact: bool,
        share_below_exact: bool,
        guarded: bool,
        share: f64,
        exact_share: f64,
        /// Whether some failed attempt of the agent lies at the law's cutoff or beyond from some cell of the grid, where
        /// the posterior may leave its term to the count of those beyond reach, a unit the run reports.
        beyond_reach: bool,
        /// Whether today's support capped at the stored top_cells cut the agent, and its leading set's cells.
        capped_today: bool,
        today_kept: usize,
        reference_seconds: f64,
        adaptive_seconds: f64,
    }

    impl AgentFinding {
        fn anomalous(&self) -> bool {
            self.finest_mismatches + self.coarse_mismatches + self.bound_violations + self.leading_left_out > 0
                || self.certified_below_exact
                || self.share_below_exact
                || (!self.guarded && self.share > EPSILON)
                || self.guarded
                || self.beyond_reach
        }
    }

    /// One informative agent of a run against today's loops over the whole grid, the reference: the adaptive posterior's
    /// kept cells and their values, its certificate against the exact masses, its work, and today's cap.
    fn equivalence_of_agent(setup: &Setup, settings: &AnchorOnlyProfile, rank: usize, law: Option<ResolvedPresence>, top_cells: usize) -> AgentFinding {
        let started = Instant::now();
        let range = today::range_support(&setup.grid, &setup.inputs.links[rank], settings, false);
        let reference = today::agent_support(&setup.grid, &range, &setup.inputs.present[rank], &setup.inputs.missing[rank], settings, law, usize::MAX).marginals;
        let (capped_today, today_kept) = today::cap_cuts(&range, &reference, settings, top_cells);
        drop(range);
        let reference_seconds = started.elapsed().as_secs_f64();
        let started = Instant::now();
        let evaluator = agent_evaluator(&setup.grid, &setup.inputs, rank, settings, &setup.weights, law);
        let support = adaptive(&evaluator, GUARD_NODES);
        let adaptive_seconds = started.elapsed().as_secs_f64();
        let (nx, ny) = evaluator.shape();
        let mut finding = AgentFinding {
            grid_cells: nx * ny,
            nodes: support.nodes,
            cells: support.cells,
            guarded: support.guarded,
            share: support.dropped_share(),
            beyond_reach: evaluator.beyond_reach(),
            capped_today,
            today_kept,
            reference_seconds,
            adaptive_seconds,
            ..AgentFinding::default()
        };
        let mut kept = vec![false; nx * ny];
        for leaf in &support.leaves {
            let (i1, j1) = leaf.last(nx, ny);
            for j in leaf.j as usize..=j1 {
                for i in leaf.i as usize..=i1 {
                    let c = j * nx + i;
                    kept[c] = true;
                    if leaf.level == 0 {
                        finding.finest += 1;
                        finding.finest_mismatches += usize::from(leaf.upper.to_bits() != reference[c].to_bits());
                    } else {
                        finding.coarse += 1;
                        finding.coarse_mismatches += usize::from(evaluator.value(i, j).to_bits() != reference[c].to_bits());
                        finding.bound_violations += usize::from(!(leaf.lower <= reference[c] && reference[c] <= leaf.upper));
                    }
                }
            }
        }
        finding.kept = finding.finest + finding.coarse;
        let (mut total, mut dropped) = (LogSum::default(), LogSum::default());
        for (c, value) in reference.iter().enumerate() {
            total.add(*value);
            if !kept[c] {
                dropped.add(*value);
            }
        }
        let (total, exact_dropped) = (total.value(), dropped.value());
        let certified = support.dropped_log();
        finding.certified_below_exact = !(exact_dropped == f64::NEG_INFINITY || exact_dropped <= certified + 1e-9 * (1.0 + certified.abs()));
        finding.exact_share = if exact_dropped == f64::NEG_INFINITY { 0.0 } else { (exact_dropped - total).exp() };
        finding.share_below_exact = finding.exact_share > finding.share * (1.0 + 1e-9) + 1e-300;
        let mut order: Vec<usize> = (0..reference.len()).filter(|c| !reference[*c].is_nan()).collect();
        order.sort_by(|a, b| reference[*b].total_cmp(&reference[*a]));
        let mut held = LogSum::default();
        for &c in &order {
            held.add(reference[c]);
            finding.leading += 1;
            if held.value() >= total + (1.0 - EPSILON).ln() {
                break;
            }
        }
        if let Some(&last) = order.get(finding.leading.max(1) - 1) {
            let last = reference[last];
            finding.leading_left_out = reference.iter().enumerate().filter(|(c, v)| **v >= last && !kept[*c]).count();
        }
        finding
    }

    /// The totals of one run key over its units.
    #[derive(Default)]
    struct RunTotals {
        dataset: String,
        candidate: String,
        factor: f64,
        units: usize,
        agents: usize,
        uninformative: usize,
        findings: Vec<AgentFinding>,
        reach: Vec<ReachFinding>,
        solve_errors: Vec<String>,
        solve_guarded: usize,
        solve_largest_dropped: f64,
        solve_seconds: f64,
    }

    /// Quantiles of a list of counts, from the median to the largest.
    fn quantiles(mut values: Vec<f64>) -> serde_json::Value {
        values.sort_by(f64::total_cmp);
        let at = |q: f64| values.get(((values.len() as f64 * q).ceil() as usize).clamp(1, values.len().max(1)) - 1).copied().unwrap_or(0.0);
        serde_json::json!({ "p50": at(0.5), "p90": at(0.9), "p99": at(0.99), "max": at(1.0) })
    }

    impl RunTotals {
        fn summary(&self) -> serde_json::Value {
            let f = &self.findings;
            let sum = |g: fn(&AgentFinding) -> usize| f.iter().map(g).sum::<usize>();
            let count = |g: fn(&AgentFinding) -> bool| f.iter().filter(|a| g(a)).count();
            let largest = |g: fn(&AgentFinding) -> f64| f.iter().map(g).fold(0.0, f64::max);
            let seconds = |g: fn(&AgentFinding) -> f64| f.iter().map(g).sum::<f64>();
            serde_json::json!({
                "dataset": self.dataset,
                "factor": self.factor,
                "units": self.units,
                "agents": self.agents,
                "informative_agents": f.len(),
                "uninformative_agents": self.uninformative,
                "finest_mismatches": sum(|a| a.finest_mismatches),
                "coarse_mismatches": sum(|a| a.coarse_mismatches),
                "bound_violations": sum(|a| a.bound_violations),
                "leading_left_out": sum(|a| a.leading_left_out),
                "certified_below_exact": count(|a| a.certified_below_exact),
                "share_below_exact": count(|a| a.share_below_exact),
                "share_above_epsilon_unguarded": count(|a| !a.guarded && a.share > EPSILON),
                "guarded_agents": count(|a| a.guarded),
                "beyond_reach_agents": count(|a| a.beyond_reach),
                "largest_reported_share": largest(|a| a.share),
                "largest_exact_share": largest(|a| a.exact_share),
                "grid_cells": sum(|a| a.grid_cells),
                "nodes": sum(|a| a.nodes),
                "cells": sum(|a| a.cells),
                "kept_cells": sum(|a| a.kept),
                "kept_single_cells": sum(|a| a.finest),
                "kept_coarse_cells": sum(|a| a.coarse),
                "leading_cells": sum(|a| a.leading),
                "grid_cells_per_agent": quantiles(f.iter().map(|a| a.grid_cells as f64).collect()),
                "nodes_per_agent": quantiles(f.iter().map(|a| a.nodes as f64).collect()),
                "cells_per_agent": quantiles(f.iter().map(|a| a.cells as f64).collect()),
                "kept_per_agent": quantiles(f.iter().map(|a| a.kept as f64).collect()),
                "cell_share_per_agent": quantiles(f.iter().map(|a| a.cells as f64 / a.grid_cells as f64).collect()),
                "capped_today": count(|a| a.capped_today),
                "today_leading_per_agent": quantiles(f.iter().map(|a| a.today_kept as f64).collect()),
                "reference_seconds": seconds(|a| a.reference_seconds),
                "adaptive_seconds": seconds(|a| a.adaptive_seconds),
                "solve_seconds": self.solve_seconds,
                "solve_errors": self.solve_errors.len(),
                "solve_error_examples": self.solve_errors.iter().take(5).collect::<Vec<_>>(),
                "solve_guarded_agents": self.solve_guarded,
                "solve_largest_dropped_mass": self.solve_largest_dropped,
                "reach": self.reach_summary(),
            })
        }

        /// The maximum likelihood reach of the units whose profile takes it, adaptive against every cell.
        fn reach_summary(&self) -> serde_json::Value {
            let r = &self.reach;
            if r.is_empty() {
                return serde_json::Value::Null;
            }
            let count = |g: fn(&ReachFinding) -> bool| r.iter().filter(|a| g(a)).count();
            let mut choices: BTreeMap<String, usize> = BTreeMap::new();
            for finding in r {
                *choices.entry(format!("{:?}", finding.exhaustive.map(|reach| (reach * 10.0).round() / 10.0))).or_default() += 1;
            }
            serde_json::json!({
                "units": r.len(),
                "choice_mismatches": count(|a| a.adaptive != a.exhaustive),
                "uncertified": count(|a| !a.certified),
                "refined_units": count(|a| a.refined > 0),
                "refined_laws": r.iter().map(|a| a.refined).sum::<usize>(),
                "bound_violations": r.iter().map(|a| a.bound_violations).sum::<usize>(),
                "kept_alone_mismatches": count(|a| a.kept_alone != a.exhaustive),
                "largest_kept_deviation": r.iter().map(|a| a.kept_deviation).fold(0.0, f64::max),
                "kept_deviation": quantiles(r.iter().map(|a| a.kept_deviation).collect()),
                "smallest_margin": r.iter().map(|a| a.margin).fold(f64::INFINITY, f64::min),
                "exhaustive_choices": choices,
                "judgement_seconds": r.iter().map(|a| a.judgement_seconds).sum::<f64>(),
                "exhaustive_seconds": r.iter().map(|a| a.exhaustive_seconds).sum::<f64>(),
            })
        }

        /// The totals of several run keys together, under `dataset` and `candidate`.
        fn merged(dataset: &str, candidate: &str, runs: &[&RunTotals]) -> RunTotals {
            RunTotals {
                dataset: dataset.to_string(),
                candidate: candidate.to_string(),
                factor: f64::NAN,
                units: runs.iter().map(|r| r.units).sum(),
                agents: runs.iter().map(|r| r.agents).sum(),
                uninformative: runs.iter().map(|r| r.uninformative).sum(),
                findings: runs.iter().flat_map(|r| r.findings.iter().cloned()).collect(),
                reach: runs.iter().flat_map(|r| r.reach.iter().cloned()).collect(),
                solve_errors: runs.iter().flat_map(|r| r.solve_errors.iter().cloned()).collect(),
                solve_guarded: runs.iter().map(|r| r.solve_guarded).sum(),
                solve_largest_dropped: runs.iter().map(|r| r.solve_largest_dropped).fold(0.0, f64::max),
                solve_seconds: runs.iter().map(|r| r.solve_seconds).sum(),
            }
        }
    }

    /// The equivalence of the adaptive posterior with today's per cell arithmetic on the stored real units of a round,
    /// the ignored test the node runs:
    /// `EQUIVALENCE_UNITS=a.jsonl,b.jsonl EQUIVALENCE_OUT=summary.json [EQUIVALENCE_WORKERS=160] [EQUIVALENCE_LIMIT=n]
    /// cargo test --release equivalence_on_stored_units -- --ignored --nocapture`, the files written by
    /// research/solid/equivalence_units.py and the first `n` lines of each file alone under a limit. The lines of every
    /// file share one queue of workers.
    ///
    /// For every run of every unit, every informative agent under the law its solve resolves is compared with today's
    /// loops over the whole grid without the prune, the reference: every kept single cell's value and every value the
    /// draws take in a coarse leaf bitwise against the reference, every coarse cell within its leaf's bounds, the
    /// certified dropped mass and share against the exact ones, the smallest leading set with 1 - EPSILON of the exact
    /// mass kept, the nodes and cells the posterior took against the grid's cells, and whether today's cap, the stored
    /// profile's top_cells after the prune, cut the agent. Every unit is also solved once per run. An agent whose grid
    /// reaches the cutoff of one of its failed attempts, where the posterior may leave that term to the count of those
    /// beyond reach and so differ from the reference, is counted and listed.
    ///
    /// A run whose profile takes the maximum likelihood reach also judges its law apart: the engine's certified
    /// judgement against the integral over every cell of every agent's range only posterior under every candidate law,
    /// their choices, whether the judgement's bounds hold the exact values, whether it refined laws or settled without
    /// its certificate, how far the values over the kept cells alone lie from the exact ones and what they would have
    /// chosen. Both sides of the posterior's comparison take the law the solve takes, the judgement's.
    ///
    /// Writes the totals of every run key, of every dataset and, for units solved again under the candidates of the round,
    /// of every dataset and candidate, and every anomalous agent and reach, to the summary.
    #[test]
    #[ignore]
    fn equivalence_on_stored_units() {
        let (Ok(files), Ok(out)) = (std::env::var("EQUIVALENCE_UNITS"), std::env::var("EQUIVALENCE_OUT")) else {
            return;
        };
        let workers: usize = std::env::var("EQUIVALENCE_WORKERS").ok().and_then(|w| w.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
        let limit: usize = std::env::var("EQUIVALENCE_LIMIT").ok().and_then(|l| l.parse().ok()).unwrap_or(usize::MAX);
        let started = Instant::now();
        let totals = std::sync::Mutex::new(BTreeMap::<String, RunTotals>::new());
        let anomalies = std::sync::Mutex::new(Vec::<serde_json::Value>::new());
        // every line of every file in one queue, each a unit with the runs to solve it under
        let names: Vec<&str> = files.split(',').filter(|f| !f.is_empty()).collect();
        let texts: Vec<String> = names.iter().map(|file| std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"))).collect();
        let lines: Vec<(&str, usize, &str)> = names
            .iter()
            .zip(&texts)
            .flat_map(|(file, text)| text.lines().filter(|l| !l.trim().is_empty()).take(limit).enumerate().map(move |(index, line)| (*file, index, line)))
            .collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let done = std::sync::atomic::AtomicUsize::new(0);
        {
            std::thread::scope(|scope| {
                for _ in 0..workers {
                    scope.spawn(|| loop {
                        let task = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(&(file, index, line)) = lines.get(task) else { break };
                        let unit: StoredUnit = serde_json::from_str(line).unwrap_or_else(|e| panic!("{file} line {index}: {e}"));
                        for run in &unit.runs {
                            let profile = &run.deployment_profile;
                            let settings = &profile.anchor_only;
                            profile.validate().unwrap();
                            let mut setup = lawless_setup(&unit.problem, profile);
                            let reach = match settings.presence.reach {
                                ReachRule::Ml => judged_reach(&setup, settings).map(|(finding, law)| {
                                    setup.law = law;
                                    finding
                                }),
                                ReachRule::Quantile => {
                                    setup.law = settings.presence.resolve(&setup.inputs.successful);
                                    None
                                }
                            };
                            if let Some(finding) = reach.as_ref().filter(|finding| finding.anomalous()) {
                                anomalies.lock().unwrap().push(serde_json::json!({
                                    "dataset": unit.dataset, "key": run.key, "unit": unit.id, "label": unit.label, "reach": finding,
                                }));
                            }
                            let mut findings = Vec::new();
                            let mut uninformative = 0;
                            for rank in 0..setup.inputs.order.len() {
                                let law = setup.inputs.law(rank, setup.law);
                                if setup.inputs.present[rank].is_empty() || (setup.inputs.links[rank].is_empty() && law.is_none()) {
                                    uninformative += 1;
                                    continue;
                                }
                                let finding = equivalence_of_agent(&setup, settings, rank, law, run.top_cells);
                                if finding.anomalous() {
                                    anomalies.lock().unwrap().push(serde_json::json!({
                                        "dataset": unit.dataset, "key": run.key, "unit": unit.id, "label": unit.label, "rank": rank, "finding": finding,
                                    }));
                                }
                                findings.push(finding);
                            }
                            let solve_started = Instant::now();
                            let solved = solve(&unit.problem, profile);
                            let solve_seconds = solve_started.elapsed().as_secs_f64();
                            let mut totals = totals.lock().unwrap();
                            let entry = totals.entry(format!("{}/{}", unit.dataset, run.key)).or_default();
                            entry.dataset.clone_from(&unit.dataset);
                            entry.candidate = run.candidate.clone().unwrap_or_default();
                            entry.factor = run.factor;
                            entry.reach.extend(reach);
                            entry.units += 1;
                            entry.agents += setup.inputs.order.len();
                            entry.uninformative += uninformative;
                            entry.findings.extend(findings);
                            entry.solve_seconds += solve_seconds;
                            match solved {
                                Ok(result) => {
                                    entry.solve_guarded += result.metrics.guarded_supports.unwrap_or(0);
                                    entry.solve_largest_dropped = entry.solve_largest_dropped.max(result.metrics.largest_dropped_mass.unwrap_or(0.0));
                                }
                                Err(error) => entry.solve_errors.push(format!("{} {}: {error}", unit.id, unit.label)),
                            }
                        }
                        let finished = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                        if finished % 1000 == 0 || finished == lines.len() {
                            eprintln!("{finished} of {} lines, {:.0} s", lines.len(), started.elapsed().as_secs_f64());
                        }
                    });
                }
            });
        }
        let totals = totals.into_inner().unwrap();
        let mut datasets: BTreeMap<String, Vec<&RunTotals>> = BTreeMap::new();
        let mut candidates: BTreeMap<(String, String), Vec<&RunTotals>> = BTreeMap::new();
        for run in totals.values() {
            datasets.entry(run.dataset.clone()).or_default().push(run);
            if !run.candidate.is_empty() {
                candidates.entry((run.dataset.clone(), run.candidate.clone())).or_default().push(run);
            }
        }
        let dataset_totals: BTreeMap<String, serde_json::Value> =
            datasets.iter().map(|(name, runs)| (name.clone(), RunTotals::merged(name, "", runs).summary())).collect();
        let candidate_totals: BTreeMap<String, serde_json::Value> = candidates
            .iter()
            .map(|((dataset, candidate), runs)| (format!("{dataset}/{candidate}"), RunTotals::merged(dataset, candidate, runs).summary()))
            .collect();
        let anomalies = anomalies.into_inner().unwrap();
        let summary = serde_json::json!({
            "files": names,
            "lines": lines.len(),
            "limit": if limit == usize::MAX { None } else { Some(limit) },
            "workers": workers,
            "seconds": started.elapsed().as_secs_f64(),
            "constants": { "epsilon": EPSILON, "flat_gap": crate::posterior::FLAT_GAP, "guard_nodes": GUARD_NODES, "reach_rounding": REACH_ROUNDING },
            "datasets": dataset_totals,
            "candidates": candidate_totals,
            "runs": totals.iter().map(|(key, run)| (key.clone(), run.summary())).collect::<BTreeMap<_, _>>(),
            "anomalous_agents": anomalies.len(),
            "anomalies": anomalies.iter().take(200).collect::<Vec<_>>(),
        });
        std::fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&summary["datasets"]).unwrap());
        eprintln!("{}", serde_json::to_string_pretty(&summary["candidates"]).unwrap());
    }

    /// Every informative agent of the stored units whose grid reaches the cutoff of one of its failed attempts, where the
    /// posterior leaves that term to the count of those beyond reach, over every line of every file:
    /// `EQUIVALENCE_UNITS=a.jsonl,b.jsonl EQUIVALENCE_OUT=reach.json [EQUIVALENCE_WORKERS=160] cargo test --release
    /// beyond_reach_on_stored_units -- --ignored --nocapture`. The check takes the solve's law and each agent's failed
    /// attempts, height nodes and grid, without a posterior. Writes the agents and units per dataset and run key, and
    /// every such agent.
    #[test]
    #[ignore]
    fn beyond_reach_on_stored_units() {
        let (Ok(files), Ok(out)) = (std::env::var("EQUIVALENCE_UNITS"), std::env::var("EQUIVALENCE_OUT")) else {
            return;
        };
        let workers: usize = std::env::var("EQUIVALENCE_WORKERS").ok().and_then(|w| w.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
        let names: Vec<&str> = files.split(',').filter(|f| !f.is_empty()).collect();
        let texts: Vec<String> = names.iter().map(|file| std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"))).collect();
        let lines: Vec<(&str, usize, &str)> = names
            .iter()
            .zip(&texts)
            .flat_map(|(file, text)| text.lines().filter(|l| !l.trim().is_empty()).enumerate().map(move |(index, line)| (*file, index, line)))
            .collect();
        // per dataset and run key: units, informative agents, agents and units beyond reach
        let totals = std::sync::Mutex::new(BTreeMap::<String, [usize; 4]>::new());
        let beyond = std::sync::Mutex::new(Vec::<serde_json::Value>::new());
        let next = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..workers {
                scope.spawn(|| loop {
                    let task = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some(&(file, index, line)) = lines.get(task) else { break };
                    let unit: StoredUnit = serde_json::from_str(line).unwrap_or_else(|e| panic!("{file} line {index}: {e}"));
                    for run in &unit.runs {
                        let profile = &run.deployment_profile;
                        let setup = setup(&unit.problem, profile);
                        let (mut informative, mut reached) = (0, 0);
                        for rank in 0..setup.inputs.order.len() {
                            let law = setup.inputs.law(rank, setup.law);
                            if setup.inputs.present[rank].is_empty() || (setup.inputs.links[rank].is_empty() && law.is_none()) {
                                continue;
                            }
                            informative += 1;
                            let present: Vec<[f64; 3]> = setup.inputs.present[rank].iter().map(Anchor3::site).collect();
                            let missing: Vec<[f64; 3]> = setup.inputs.missing[rank].iter().map(Anchor3::site).collect();
                            let evaluator = CellEvaluator::new(&setup.grid, &[], &present, &missing, &profile.anchor_only.height_nodes, &setup.weights, law);
                            if evaluator.beyond_reach() {
                                reached += 1;
                                beyond.lock().unwrap().push(serde_json::json!({
                                    "dataset": unit.dataset, "key": run.key, "unit": unit.id, "label": unit.label, "rank": rank,
                                    "reach": law.map(|l| l.reach), "failed": missing.len(), "grid": [setup.grid.xs.len(), setup.grid.ys.len()],
                                }));
                            }
                        }
                        let mut totals = totals.lock().unwrap();
                        let entry = totals.entry(format!("{}/{}", unit.dataset, run.key)).or_default();
                        entry[0] += 1;
                        entry[1] += informative;
                        entry[2] += reached;
                        entry[3] += usize::from(reached > 0);
                    }
                });
            }
        });
        let totals = totals.into_inner().unwrap();
        let beyond = beyond.into_inner().unwrap();
        let mut datasets: BTreeMap<String, [usize; 4]> = BTreeMap::new();
        for (key, counts) in &totals {
            let entry = datasets.entry(key.split('/').next().unwrap().to_string()).or_default();
            for (sum, count) in entry.iter_mut().zip(counts) {
                *sum += count;
            }
        }
        let table = |map: &BTreeMap<String, [usize; 4]>| -> BTreeMap<String, serde_json::Value> {
            map.iter()
                .map(|(key, c)| (key.clone(), serde_json::json!({ "units": c[0], "informative_agents": c[1], "beyond_reach_agents": c[2], "beyond_reach_units": c[3] })))
                .collect()
        };
        let summary = serde_json::json!({
            "files": names, "lines": lines.len(), "workers": workers,
            "datasets": table(&datasets), "runs": table(&totals),
            "beyond_reach_agents": beyond.len(), "agents": beyond,
        });
        std::fs::write(&out, serde_json::to_string_pretty(&summary).unwrap()).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&summary["datasets"]).unwrap());
    }
}
