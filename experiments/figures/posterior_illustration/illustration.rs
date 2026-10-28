//! The record of the adaptive posteriors of one anchor only unit that the figure of experiments/figures/
//! posterior_illustration draws.
//!
//! The figure tool adds this file to a copy of the runtime workspace as engine/src/illustration.rs and declares it at the
//! end of that copy's lib.rs under cfg(test), so the engine compiles as it is and only its own tests see the module. The
//! ignored test reads the prepared problem and the deployment profile from the file ILLUSTRATION_INPUT names, which
//! dump_problem.rs writes, and writes the record to the file ILLUSTRATION_OUTPUT names. ILLUSTRATION_AGENT, when set,
//! keeps the decisions of that agent alone in the record, every agent being checked all the same. The node runs
//! `ILLUSTRATION_INPUT=input.json ILLUSTRATION_OUTPUT=record.json cargo test --release --locked -p solid-core --lib
//! illustration::posterior_illustration -- --ignored --exact --nocapture`.
//!
//! The helpers of the solve in anchor_only.rs and the node helpers in posterior.rs are private to their modules, so the
//! mirror below copies them line for line, each naming the lines it copies. Four assertions tie the record to the frozen
//! engine. The digest of the problem and the profile equals the one the dump computed before it wrote them, so both
//! crossed the JSON file bitwise. The frozen solve of the input reproduces the result the dump computed through the
//! research CLI's own path, bitwise but for its elapsed time. The mirror reproduces the frozen solve's result the same
//! way, its positions, risk, iterations, evaluations and posterior work among them. For every informative agent the
//! recording refinement keeps bitwise the leaves of the frozen `adaptive` on the same evaluator, with the same dropped
//! mass and share, guard flag, nodes and cells, so the recorded nodes are those the frozen engine decided about.

use std::collections::BTreeMap;
use std::ops::Range;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{json, Value};

use crate::link::{log_t_norm_for, BlockModel, LinkTable};
use crate::posterior::{
    adaptive, kept_mass, log_add_exp, outcome_log_likelihoods, CellEvaluator, Grid, LogSum, Node, Rng, Sampler, Support, EPSILON,
    FLAT_GAP, GUARD_NODES, MIN_GUARD_NODES,
};
use crate::presence::{ml_choice, ml_null, ResolvedPresence};
use crate::{
    Anchor, AnchorOnlyProfile, CalibratedRangeSample, CalibrationModel, DeploymentProfile, EngineError, LinkAttempt, LocalizationEngine,
    LocalizationMode, PlanarArea, Point2, PreparedProblem, PresenceProfile, RangeObservation, ReachRule, SolveMetrics, SolveResult,
    TerminationReason,
};

/// The input of the record as dump_problem.rs writes it. It holds the prepared problem and the deployment profile of one
/// anchor only unit, and optionally the unit's description, the true positions of its agents keyed by identifier, the
/// result of the research CLI's solve of the unit and the digest of the problem and the profile.
#[derive(Deserialize)]
struct Input {
    problem: PreparedProblem,
    profile: DeploymentProfile,
    #[serde(default)]
    unit: Option<Value>,
    #[serde(default)]
    truth: Option<Value>,
    #[serde(default)]
    reference: Option<SolveResult>,
    #[serde(default)]
    fingerprint: Option<String>,
}

/// The record of the posterior illustration, the ignored test the figure tool runs (see the module's documentation).
#[test]
#[ignore]
fn posterior_illustration() {
    let (Ok(input_path), Ok(output_path)) = (std::env::var("ILLUSTRATION_INPUT"), std::env::var("ILLUSTRATION_OUTPUT")) else {
        panic!("ILLUSTRATION_INPUT and ILLUSTRATION_OUTPUT must name the input and the record");
    };
    let only: Option<u32> =
        std::env::var("ILLUSTRATION_AGENT").ok().map(|agent| agent.parse::<u32>().unwrap_or_else(|e| panic!("ILLUSTRATION_AGENT {agent}: {e}")));
    let text = std::fs::read_to_string(&input_path).unwrap_or_else(|e| panic!("{input_path}: {e}"));
    let input: Input = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{input_path}: {e}"));
    let (problem, profile) = (&input.problem, &input.profile);
    let digest = fingerprint(problem, profile);
    let transfer = match &input.fingerprint {
        Some(expected) if *expected == digest => "equal",
        Some(expected) => panic!("the problem and the profile read {digest}, not the {expected} the dump wrote"),
        None => "absent",
    };
    // the frozen solve, as the research CLI calls it
    let frozen = LocalizationEngine::new().solve_with_profile(problem, profile.clone()).unwrap_or_else(|e| panic!("the frozen solve fails, {e}"));
    let reference = match &input.reference {
        Some(reference) => {
            if let Err(difference) = same_result(&frozen, reference) {
                panic!("the frozen solve of the input differs from the dump's solve through the research CLI's path, {difference}");
            }
            "equal"
        }
        None => "absent",
    };
    // the mirror with the recording refinements, after the validation that solve_with_profile runs first
    profile.validate().unwrap_or_else(|e| panic!("the deployment profile is refused, {e}"));
    let mut records = Vec::new();
    let mirrored = mirror_solve(problem, profile, GUARD_NODES, &mut records).unwrap_or_else(|e| panic!("the mirrored solve fails, {e}"));
    if let Err(difference) = same_result(&mirrored, &frozen) {
        panic!("the mirrored solve differs from the frozen solve, {difference}");
    }
    let informative = records.iter().filter(|record| record.refinement.is_some()).count();
    if let Some(id) = only {
        assert!(records.iter().any(|record| record.agent_id == id && record.refinement.is_some()), "ILLUSTRATION_AGENT {id} is no informative agent of the unit");
    }
    let grid = Grid::new(&profile.area, profile.anchor_only.grid_step).expect("the grid of the solve");
    let (nx, ny) = (grid.xs.len(), grid.ys.len());
    let truth_of = |agent: u32| input.truth.as_ref().and_then(|truth| truth.get(agent.to_string())).cloned().unwrap_or(Value::Null);
    let mut agents = Vec::with_capacity(records.len());
    for record in &records {
        let estimate = frozen.positions[record.index];
        let mut agent = json!({
            "agent_id": record.agent_id,
            "rank": record.rank,
            "informative": record.refinement.is_some(),
            "law": record.law.map(|law| json!({ "shape": law.shape, "reach": law.reach, "width": law.width })),
            "links": record.links,
            "present": record.present,
            "missing": record.missing,
            "estimate": [estimate.x, estimate.y],
            "truth": truth_of(record.agent_id),
            "grid_cells": nx * ny,
            "guarded": record.guarded,
            "nodes": record.nodes,
            "cells": record.cells,
            "dropped_share": number(record.share),
        });
        let mut summary = format!("agent {} uninformative, every cell alike", record.agent_id);
        if let Some(refinement) = &record.refinement {
            let kept: usize = refinement.leaves.iter().map(|leaf| leaf.cells(nx, ny)).sum();
            let single = refinement.leaves.iter().filter(|leaf| leaf.level == 0).count();
            let coarse = refinement.leaves.len() - single;
            let mut fates = [0usize; 6];
            for (_, fate) in &refinement.decisions {
                fates[*fate as usize] += 1;
            }
            agent["kept_cells"] = json!(kept);
            agent["single_cells"] = json!(single);
            agent["coarse_leaves"] = json!(coarse);
            agent["coarse_cells"] = json!(kept - single);
            agent["dropped_log"] = number(refinement.dropped);
            agent["kept_log"] = number(refinement.kept);
            agent["best"] = number(refinement.best);
            agent["delta"] = number(refinement.delta);
            agent["threshold"] = number(refinement.best - refinement.delta);
            agent["fate_counts"] = json!(fates);
            agent["leaves"] = node_columns(&refinement.leaves);
            if only.is_none_or(|id| id == record.agent_id) {
                agent["decisions"] = decision_columns(&refinement.decisions);
            }
            summary = format!(
                "agent {} {} grid cells, {} evaluated exactly, {} nodes bounded, {kept} kept cells ({single} single, {} in {coarse} coarse leaves), dropped share {:e}{}",
                record.agent_id,
                nx * ny,
                record.cells,
                record.nodes,
                kept - single,
                record.share,
                if record.guarded { ", guarded" } else { "" }
            );
        }
        eprintln!("{summary}");
        agents.push(agent);
    }
    let anchors: Vec<Value> = problem.anchors.iter().map(|a| json!([a.id, a.position.x, a.position.y, a.height_cm])).collect();
    let record = json!({
        "schema": "solid.posterior-illustration.v1",
        "unit": input.unit,
        "checks": {
            "fingerprint": transfer,
            "reference": reference,
            "mirror": "equal",
            "adaptive": format!("equal for {informative} informative agents"),
            "digest": digest,
        },
        "constants": { "epsilon": EPSILON, "flat_gap": FLAT_GAP, "guard_nodes": GUARD_NODES },
        "fates": Fate::NAMES,
        "area": profile.area,
        "grid": { "step": grid.step, "nx": nx, "ny": ny, "xs": grid.xs, "ys": grid.ys },
        "anchors": anchors,
        "metrics": frozen.metrics,
        "agents": agents,
    });
    std::fs::write(&output_path, serde_json::to_string(&record).expect("a serializable record")).unwrap_or_else(|e| panic!("{output_path}: {e}"));
    eprintln!("wrote {output_path}, checks fingerprint {transfer}, reference {reference}, mirror equal, adaptive equal for {informative} agents");
}

/// A number of the record, or for one that is not finite, which JSON lacks, the string "Infinity", "-Infinity" or "NaN".
fn number(value: f64) -> Value {
    if value.is_finite() {
        json!(value)
    } else if value.is_nan() {
        json!("NaN")
    } else if value > 0.0 {
        json!("Infinity")
    } else {
        json!("-Infinity")
    }
}

/// Nodes as columns of their level, their first cell (i, j) and their lower and upper bounds. A node of level k covers
/// the cells [i, i + 2^k) x [j, j + 2^k), clipped to the grid.
fn node_columns(nodes: &[Node]) -> Value {
    json!({
        "level": nodes.iter().map(|node| node.level).collect::<Vec<u8>>(),
        "i": nodes.iter().map(|node| node.i).collect::<Vec<u32>>(),
        "j": nodes.iter().map(|node| node.j).collect::<Vec<u32>>(),
        "lower": nodes.iter().map(|node| number(node.lower)).collect::<Vec<Value>>(),
        "upper": nodes.iter().map(|node| number(node.upper)).collect::<Vec<Value>>(),
    })
}

/// The decisions as the columns of their nodes and the column of their fates, the codes of `Fate`.
fn decision_columns(decisions: &[(Node, Fate)]) -> Value {
    let nodes: Vec<Node> = decisions.iter().map(|(node, _)| *node).collect();
    let mut columns = node_columns(&nodes);
    columns["fate"] = json!(decisions.iter().map(|(_, fate)| *fate as u8).collect::<Vec<u8>>());
    columns
}

/// Whether two results of a solve agree bitwise but for their elapsed time, in their agents, every coordinate, the risk,
/// the largest dropped share and every count of their metrics.
fn same_result(a: &SolveResult, b: &SolveResult) -> Result<(), String> {
    if a.agent_ids != b.agent_ids {
        return Err(format!("agents {:?} against {:?}", a.agent_ids, b.agent_ids));
    }
    let bits = |p: &Point2| (p.x.to_bits(), p.y.to_bits());
    if a.positions.len() != b.positions.len() || a.positions.iter().zip(&b.positions).any(|(p, q)| bits(p) != bits(q)) {
        return Err(format!("positions {:?} against {:?}", a.positions, b.positions));
    }
    let (m, n) = (&a.metrics, &b.metrics);
    let counts = |metrics: &SolveMetrics| {
        [
            metrics.dimensions,
            metrics.iterations,
            metrics.objective_evaluations,
            metrics.gradient_evaluations,
            metrics.factors,
            metrics.range_factor_evaluations,
            metrics.coordinate_factor_evaluations,
            metrics.starts,
        ]
    };
    if counts(m) != counts(n) {
        return Err(format!("counts {:?} against {:?}", counts(m), counts(n)));
    }
    let work = |metrics: &SolveMetrics| (metrics.converged, metrics.termination, metrics.guarded_supports, metrics.posterior_nodes, metrics.posterior_cells);
    if work(m) != work(n) {
        return Err(format!("termination and posterior work {:?} against {:?}", work(m), work(n)));
    }
    if m.final_objective.to_bits() != n.final_objective.to_bits() || m.largest_dropped_mass.map(f64::to_bits) != n.largest_dropped_mass.map(f64::to_bits) {
        return Err(format!(
            "risk and largest dropped share {} {:?} against {} {:?}",
            m.final_objective, m.largest_dropped_mass, n.final_objective, n.largest_dropped_mass
        ));
    }
    Ok(())
}

/// FNV-1a over 64 bit words.
struct Digest(u64);

impl Digest {
    fn word(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn numbers(&mut self, values: &[f64]) {
        self.word(values.len() as u64);
        for value in values {
            self.word(value.to_bits());
        }
    }

    fn name(&mut self, name: &str) {
        self.word(name.len() as u64);
        for byte in name.bytes() {
            self.word(u64::from(byte));
        }
    }
}

/// A digest of every number, flag and name of a problem and a profile in the order of their fields, the bits of every
/// number among them. dump_problem.rs computes it the same way before it writes the input, so equal digests show that
/// the problem and the profile crossed the JSON file bitwise. Every field is named, so a field this digest missed would
/// not compile.
fn fingerprint(problem: &PreparedProblem, profile: &DeploymentProfile) -> String {
    let mut digest = Digest(0xcbf2_9ce4_8422_2325);
    let PreparedProblem { agent_ids, anchors, observations, initial_positions, mode, attempts } = problem;
    digest.word(agent_ids.len() as u64);
    for id in agent_ids {
        digest.word(u64::from(*id));
    }
    digest.word(anchors.len() as u64);
    for Anchor { id, position, height_cm } in anchors {
        digest.word(u64::from(*id));
        digest.numbers(&[position.x, position.y, *height_cm]);
    }
    digest.word(observations.len() as u64);
    for RangeObservation { from, to, tof_ns, calibrated_samples, to_is_anchor } in observations {
        digest.word(u64::from(*from));
        digest.word(u64::from(*to));
        digest.numbers(tof_ns);
        digest.word(calibrated_samples.len() as u64);
        for CalibratedRangeSample { distance_cm, scale_cm } in calibrated_samples {
            digest.numbers(&[*distance_cm, *scale_cm]);
        }
        digest.word(u64::from(*to_is_anchor));
    }
    digest.word(initial_positions.len() as u64);
    for Point2 { x, y } in initial_positions {
        digest.numbers(&[*x, *y]);
    }
    digest.name(&format!("{mode:?}"));
    match attempts {
        None => digest.word(0),
        Some(attempts) => {
            digest.word(1 + attempts.len() as u64);
            for LinkAttempt { from, to, to_is_anchor } in attempts {
                digest.word(u64::from(*from));
                digest.word(u64::from(*to));
                digest.word(u64::from(*to_is_anchor));
            }
        }
    }
    let DeploymentProfile { area, anchor_only } = profile;
    let PlanarArea { minimum_x_cm, maximum_x_cm, minimum_y_cm, maximum_y_cm } = area;
    digest.numbers(&[*minimum_x_cm, *maximum_x_cm, *minimum_y_cm, *maximum_y_cm]);
    let AnchorOnlyProfile {
        kappa,
        excess_probability,
        excess_mean_rel,
        persistent_sd_rel,
        height_nodes,
        height_weights,
        grid_step,
        joint_samples,
        seed,
        nu,
        independent_packets,
        decision,
        presence,
        offset_shape,
        early_detection_per_cm,
    } = anchor_only;
    digest.numbers(&[*kappa, *excess_probability, *excess_mean_rel, *persistent_sd_rel]);
    digest.numbers(height_nodes);
    digest.numbers(height_weights);
    digest.numbers(&[*grid_step, *nu, *early_detection_per_cm]);
    digest.word(*joint_samples as u64);
    digest.word(*seed);
    digest.word(u64::from(*independent_packets));
    digest.name(&format!("{decision:?} {offset_shape:?}"));
    let PresenceProfile { shape, reach, quantile, factor, width, enabled } = presence;
    digest.name(&format!("{shape:?} {reach:?}"));
    digest.numbers(&[*quantile, *factor, *width]);
    digest.word(u64::from(*enabled));
    format!("{:016x}", digest.0)
}

/// What the record keeps of one agent. It holds its identifier, its canonical rank and its index in the request, the law
/// of its posterior, the anchors of its links, of its successful attempts and of its failed ones, the certificate and the
/// work of the frozen support its draws take, and for an informative agent the recording refinement of that support.
struct AgentRecord {
    agent_id: u32,
    rank: usize,
    index: usize,
    law: Option<ResolvedPresence>,
    links: Vec<[f64; 3]>,
    present: Vec<[f64; 3]>,
    missing: Vec<[f64; 3]>,
    guarded: bool,
    nodes: usize,
    cells: usize,
    share: f64,
    refinement: Option<Recorded>,
}

/// `solve_guarded` of anchor_only.rs:535 line for line, with the recording refinement of every informative agent's
/// posterior beside the frozen `adaptive`, whose support the draws take as ever. The recording evaluates cells with the
/// same evaluator and changes nothing the solve counts or draws, and its support must be the frozen one bitwise.
fn mirror_solve(problem: &PreparedProblem, profile: &DeploymentProfile, guard: usize, records: &mut Vec<AgentRecord>) -> Result<SolveResult, EngineError> {
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
        // the recording refinement of the same evaluator, which must keep the frozen support bitwise
        let refinement = evaluator.as_ref().map(|evaluator| {
            let recorded = recording_adaptive(evaluator, guard, FLAT_GAP);
            if let Err(difference) = same_support(&recorded, &support) {
                panic!("agent {}: the recording refinement differs from the frozen adaptive posterior, {difference}", problem.agent_ids[index]);
            }
            recorded
        });
        records.push(AgentRecord {
            agent_id: problem.agent_ids[index],
            rank,
            index,
            law,
            links: inputs.links[rank].iter().map(|(anchor, _)| anchor.site()).collect(),
            present: inputs.present[rank].iter().map(Anchor3::site).collect(),
            missing: inputs.missing[rank].iter().map(Anchor3::site).collect(),
            guarded: support.guarded,
            nodes: support.nodes,
            cells: support.cells,
            share: support.dropped_share(),
            refinement,
        });
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

/// What the refinement decided about a node, the code the record gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fate {
    /// Its children were bounded at the next level.
    Split = 0,
    /// It dropped at its level with its bound mass, its upper bound minus infinity or more than delta below the best
    /// value seen, or at level zero a value that is not a number.
    Dropped = 1,
    /// The node guard dropped it with its bound mass instead of splitting it.
    Guarded = 2,
    /// It is kept as a single cell.
    Cell = 3,
    /// It is kept whole as a coarse leaf.
    Coarse = 4,
    /// It was a leaf and the last pass dropped it, below the final threshold or with no finite best value.
    Final = 5,
}

impl Fate {
    /// The names of the codes, in the order of the codes.
    const NAMES: [&'static str; 6] = ["split", "dropped", "guarded", "cell", "coarse", "final"];
}

/// The refinement of one posterior as `adaptive_with` runs it, with every node it decided about.
struct Recorded {
    /// The fields of the frozen `Support` (posterior.rs:774), which are the kept leaves in canonical order, the natural logs
    /// of the certified dropped mass and of the kept mass, whether the node guard cut the refinement, the nodes bounded
    /// above level zero and the cells evaluated exactly.
    leaves: Vec<Node>,
    dropped: f64,
    kept: f64,
    guarded: bool,
    nodes: usize,
    cells: usize,
    /// The best value seen at the end and delta = ln(N / EPSILON), N the grid's cells, the final threshold being their
    /// difference.
    best: f64,
    delta: f64,
    /// Every node the refinement decided about with its fate, level by level in the order of the frontier.
    decisions: Vec<(Node, Fate)>,
}

/// The certified share of the mass left out, `Support::dropped_share` of posterior.rs:817.
fn dropped_share(dropped: f64, kept: f64) -> f64 {
    if dropped == f64::NEG_INFINITY {
        0.0
    } else {
        1.0 / (1.0 + (kept - dropped).exp())
    }
}

/// Whether a recording refinement keeps the frozen support bitwise, with the same leaves in the same order and with the
/// same bounds, the same dropped mass and share, guard flag, nodes and cells. The kept mass is private to posterior.rs
/// and enters the solve through the share alone, which is compared.
fn same_support(recorded: &Recorded, support: &Support) -> Result<(), String> {
    if recorded.leaves.len() != support.leaves.len() {
        return Err(format!("{} leaves against {}", recorded.leaves.len(), support.leaves.len()));
    }
    for (k, (a, b)) in recorded.leaves.iter().zip(&support.leaves).enumerate() {
        if (a.level, a.i, a.j, a.lower.to_bits(), a.upper.to_bits()) != (b.level, b.i, b.j, b.lower.to_bits(), b.upper.to_bits()) {
            return Err(format!("leaf {k} {a:?} against {b:?}"));
        }
    }
    if recorded.dropped.to_bits() != support.dropped_log().to_bits() {
        return Err(format!("dropped mass {} against {}", recorded.dropped, support.dropped_log()));
    }
    let share = dropped_share(recorded.dropped, recorded.kept);
    if share.to_bits() != support.dropped_share().to_bits() {
        return Err(format!("dropped share {share} against {}", support.dropped_share()));
    }
    let work = |guarded: bool, nodes: usize, cells: usize| (guarded, nodes, cells);
    if work(recorded.guarded, recorded.nodes, recorded.cells) != work(support.guarded, support.nodes, support.cells) {
        return Err(format!(
            "guard, nodes and cells {:?} against {:?}",
            work(recorded.guarded, recorded.nodes, recorded.cells),
            work(support.guarded, support.nodes, support.cells)
        ));
    }
    Ok(())
}

/// `adaptive_with` of posterior.rs:861 line for line, recording the fate of every node it decides about. Every bound,
/// value, comparison and sum is the frozen function's in the frozen order, so its support is the frozen support.
fn recording_adaptive(evaluator: &CellEvaluator, guard: usize, flat_gap: f64) -> Recorded {
    assert!(guard >= MIN_GUARD_NODES, "a node guard of {guard} cannot hold the four children of a node");
    let (nx, ny) = evaluator.shape();
    let delta = ((nx * ny) as f64 / EPSILON).ln();
    let (mut nodes, mut cells) = (0, 0);
    let mut frontier = vec![node_root(nx, ny)];
    let mut leaves: Vec<Node> = Vec::new();
    let mut dropped = LogSum::default();
    let mut best = f64::NEG_INFINITY;
    let mut guarded = false;
    // every decision in the order of the frontier, and the decision of every leaf, which the last pass may turn into a drop
    let mut decisions: Vec<(Node, Fate)> = Vec::new();
    let mut leaf_decisions: Vec<usize> = Vec::new();
    loop {
        let level = frontier[0].level;
        best = best.max(bound_level(evaluator, &mut frontier, &mut nodes, &mut cells));
        if level > 0 {
            let probed = probe(evaluator, &frontier, &mut cells);
            if probed > best {
                best = probed;
            }
        }
        let threshold = best - delta;
        let mut pending: Vec<Node> = Vec::new();
        let mut pending_decisions: Vec<usize> = Vec::new();
        for node in frontier.drain(..) {
            if node.upper == f64::NEG_INFINITY || (level == 0 && node.upper.is_nan()) || node.upper < threshold {
                dropped.add(node_mass(&node, nx, ny));
                decisions.push((node, Fate::Dropped));
            } else if level == 0 || node.upper - node.lower <= flat_gap {
                leaf_decisions.push(decisions.len());
                decisions.push((node, if level == 0 { Fate::Cell } else { Fate::Coarse }));
                leaves.push(node);
            } else {
                pending_decisions.push(decisions.len());
                decisions.push((node, Fate::Split));
                pending.push(node);
            }
        }
        if pending.is_empty() {
            break;
        }
        let children: usize = pending.iter().map(|node| node_child_count(node, nx, ny)).sum();
        let mut split = vec![true; pending.len()];
        if leaves.len() + children > guard {
            guarded = true;
            let mut order: Vec<usize> = (0..pending.len()).collect();
            order.sort_by(|a, b| node_mass(&pending[*b], nx, ny).total_cmp(&node_mass(&pending[*a], nx, ny)).then(a.cmp(b)));
            let mut held = leaves.len();
            let mut fits = true;
            for &k in &order {
                let count = node_child_count(&pending[k], nx, ny);
                fits &= held + count <= guard;
                if fits {
                    held += count;
                } else {
                    split[k] = false;
                    dropped.add(node_mass(&pending[k], nx, ny));
                    decisions[pending_decisions[k]].1 = Fate::Guarded;
                }
            }
            // the best value seen may lie in a dropped node, so it restarts from the lower bounds of what is kept, which
            // hold a cell at least that good, and the kept cells cannot all fall below the threshold later
            best = leaves.iter().chain(pending.iter().zip(&split).filter(|(_, s)| **s).map(|(node, _)| node)).map(|node| node.lower).fold(f64::NEG_INFINITY, f64::max);
        }
        for (node, split) in pending.iter().zip(&split) {
            if *split {
                node_children(node, nx, ny, &mut frontier);
            }
        }
        if frontier.is_empty() {
            break;
        }
    }
    if !best.is_finite() {
        for leaf in leaves.drain(..) {
            dropped.add(node_mass(&leaf, nx, ny));
        }
        for &decision in &leaf_decisions {
            decisions[decision].1 = Fate::Final;
        }
        leaf_decisions.clear();
    }
    let threshold = best - delta;
    let mut kept = LogSum::default();
    let mut position = 0;
    leaves.retain(|leaf| {
        let decision = leaf_decisions[position];
        position += 1;
        if leaf.upper < threshold {
            dropped.add(node_mass(leaf, nx, ny));
            decisions[decision].1 = Fate::Final;
            false
        } else {
            kept.add(if leaf.level == 0 { leaf.upper } else { (leaf.cells(nx, ny) as f64).ln() + leaf.lower });
            true
        }
    });
    let mut kept = kept.value();
    if !guarded && best.is_finite() {
        kept = kept.max(best);
    }
    Recorded { leaves, dropped: dropped.value(), kept, guarded, nodes, cells, best, delta, decisions }
}

/// `Node::root` of posterior.rs:663.
fn node_root(nx: usize, ny: usize) -> Node {
    let mut level = 0;
    while (1usize << level) < nx.max(ny) {
        level += 1;
    }
    Node { level, i: 0, j: 0, lower: f64::NEG_INFINITY, upper: f64::INFINITY }
}

/// `Node::mass` of posterior.rs:692, the natural log of the node's mass bound.
fn node_mass(node: &Node, nx: usize, ny: usize) -> f64 {
    (node.cells(nx, ny) as f64).ln() + node.upper
}

/// `Node::centre` of posterior.rs:697, the middle cell of the node.
fn node_centre(node: &Node, nx: usize, ny: usize) -> (usize, usize) {
    let (i1, j1) = node.last(nx, ny);
    let (i, j) = (node.i as usize, node.j as usize);
    (i + (i1 - i) / 2, j + (j1 - j) / 2)
}

/// `Node::children` of posterior.rs:705, the children within the grid in canonical order.
fn node_children(node: &Node, nx: usize, ny: usize, out: &mut Vec<Node>) {
    let level = node.level - 1;
    let half = 1u32 << level;
    let child = |i: u32, j: u32| Node { level, i, j, lower: f64::NEG_INFINITY, upper: f64::INFINITY };
    let right = ((node.i + half) as usize) < nx;
    let up = ((node.j + half) as usize) < ny;
    out.push(child(node.i, node.j));
    if right {
        out.push(child(node.i + half, node.j));
    }
    if up {
        out.push(child(node.i, node.j + half));
    }
    if right && up {
        out.push(child(node.i + half, node.j + half));
    }
}

/// `Node::child_count` of posterior.rs:723.
fn node_child_count(node: &Node, nx: usize, ny: usize) -> usize {
    let half = 1usize << (node.level - 1);
    let right = node.i as usize + half < nx;
    let up = node.j as usize + half < ny;
    1 + usize::from(right) + usize::from(up) + usize::from(right && up)
}

/// `bound_level` of posterior.rs:733.
fn bound_level(evaluator: &CellEvaluator, frontier: &mut [Node], nodes: &mut usize, cells: &mut usize) -> f64 {
    let (nx, ny) = evaluator.shape();
    let mut best = f64::NEG_INFINITY;
    for node in frontier.iter_mut() {
        if node.level == 0 {
            let value = evaluator.value(node.i as usize, node.j as usize);
            (node.lower, node.upper) = (value, value);
            *cells += 1;
        } else {
            let (i1, j1) = node.last(nx, ny);
            (node.lower, node.upper) = evaluator.bounds(node.i as usize, i1, node.j as usize, j1);
            *nodes += 1;
        }
        if node.lower > best {
            best = node.lower;
        }
    }
    best
}

/// `probe` of posterior.rs:755.
fn probe(evaluator: &CellEvaluator, frontier: &[Node], cells: &mut usize) -> f64 {
    let (nx, ny) = evaluator.shape();
    let mut chosen: Option<&Node> = None;
    for node in frontier {
        if node.upper > chosen.map_or(f64::NEG_INFINITY, |c| c.upper) {
            chosen = Some(node);
        }
    }
    match chosen {
        Some(node) if node.level > 0 => {
            let (i, j) = node_centre(node, nx, ny);
            *cells += 1;
            evaluator.value(i, j)
        }
        _ => f64::NEG_INFINITY,
    }
}

// The helpers of the solve, anchor_only.rs:22 to 775 but for `solve` and `solve_guarded`, line for line. The one change
// is `LinkTable::build`, an inherent method private to anchor_only.rs, which is the function `link_table` here.

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

/// `prepare_links` of anchor_only.rs:66.
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

/// `LinkTable::build` of anchor_only.rs:167.
fn link_table(link: &Link, profile: &AnchorOnlyProfile, norm: f64) -> LinkTable {
    let y: Vec<f64> = link.samples.iter().map(|s| s.distance_cm).collect();
    let scales: Vec<f64> = link.samples.iter().map(|s| s.scale_cm).collect();
    LinkTable::from_packets(&y, &scales, &BlockModel::from(profile), norm)
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

/// The inputs of every agent's posterior in the canonical order of the agents (anchor_only.rs:197).
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

/// `agent_inputs` of anchor_only.rs:248, with `link_table` for `LinkTable::build`.
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
            links.push((anchor3(&link.anchor), link_table(link, settings, norm)));
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

/// The work and the certificates of the posteriors of one solve (anchor_only.rs:336).
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
/// size and one.
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

/// The sum of the per agent terms `terms`, minus infinity where one of them is.
fn agent_sum(terms: impl Iterator<Item = f64>) -> f64 {
    terms.fold(0.0, |sum, term| if sum == f64::NEG_INFINITY || term == f64::NEG_INFINITY { f64::NEG_INFINITY } else { sum + term })
}

/// What the maximum likelihood reach found for the candidate laws of a scenario (anchor_only.rs:403). The solve reads
/// the choice alone.
#[allow(dead_code)]
struct ReachJudgement {
    candidates: Vec<ResolvedPresence>,
    values: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    slack: f64,
    null: f64,
    refined: Vec<usize>,
    certified: bool,
    chosen: Option<ResolvedPresence>,
}

/// `settle` of anchor_only.rs:425.
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

/// `judge_reach` of anchor_only.rs:450.
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

/// Why a support gave no draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Undrawn {
    /// It keeps no cell.
    Empty,
    /// A drawn cell of a coarse leaf lay above the leaf's certified upper bound (`Sampler::draw`).
    AboveBound,
}

/// `joint_draws` of anchor_only.rs:617.
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

/// `agent_draws` of anchor_only.rs:632.
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

/// Minimum empirical capped pairwise risk action over the joint samples (anchor_only.rs:674).
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
