//! The baseline adapter of the held-out anchor only benchmarks, HYMN and AAU Meshed UWB.
//!
//! The existing adapters check the populations of their own campaigns, eight TIERS nodes, eight UAV or Bregar anchors or
//! the fifteen IDLab nodes. This one accepts any anchor only unit with at least one anchor and two agents. Both
//! constellations have real height spread, so it follows the rule of the Bregar adapter. Jin 2021, FNL 2025 and
//! STRONG 2021 run with the planar or the spatial model, the one their selection on the training half chose, and
//! ADE-BP 2015 runs with the planar model of its paper, the one it implements, with the settings of its anchor only runs
//! and the operating area each unit carries (experiments/ade-heightspread/declaration.json). A run sends the three and a
//! selection of a method's options sends that method alone. It writes one JSON record per unit and method. Truth never
//! enters a request.
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use solid_core::{
    DimensionPolicy, LocalizationMode, Point2, PreparedProblem, SolveConfig, SolveResult,
};
use solid_methods::{stayed_at_start, AreaBounds, MethodConfig, MethodKind, MethodRunner};

/// The most units one request may hold. The runner splits a half into many small requests.
const MAX_UNITS: usize = 64;
/// The range of 200 ns of flight at the speed of light, the largest admissible reading of the held-out builders.
const MAXIMUM_RANGE_CM: f64 = 200.0 * 29.979_245_8;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Baseline {
    method: MethodKind,
    method_config: MethodConfig,
    solve_config: SolveConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Unit {
    id: u32,
    samples: usize,
    problem: PreparedProblem,
    /// The operating area of the unit by the deployment rules, the area SOLID receives. ADE-BP takes it, and the
    /// adapter checks that it holds every anchor.
    area: AreaBounds,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    baselines: Vec<Baseline>,
    units: Vec<Unit>,
}

fn validate(unit: &Unit) -> Result<()> {
    let p = &unit.problem;
    ensure!(
        p.mode == LocalizationMode::AnchorOnly,
        "anchor only units only"
    );
    ensure!(matches!(unit.samples, 1 | 10), "unexpected report count");
    let agents: BTreeSet<_> = p.agent_ids.iter().copied().collect();
    let anchors: BTreeSet<_> = p.anchors.iter().map(|a| a.id).collect();
    ensure!(
        agents.len() == p.agent_ids.len() && agents.len() >= 2,
        "a unit needs two distinct agents"
    );
    ensure!(
        anchors.len() == p.anchors.len() && !anchors.is_empty(),
        "a unit needs distinct anchors"
    );
    ensure!(agents.is_disjoint(&anchors), "overlapping node roles");
    ensure!(
        p.initial_positions.len() == agents.len()
            && p.initial_positions
                .iter()
                .all(|v| v.x.is_finite() && v.y.is_finite()),
        "invalid starting coordinates"
    );
    ensure!(
        p.anchors.iter().all(|a| a.position.x.is_finite()
            && a.position.y.is_finite()
            && a.height_cm.is_finite()),
        "invalid known anchor"
    );
    let area = unit.area;
    ensure!(
        [
            area.minimum_x_cm,
            area.maximum_x_cm,
            area.minimum_y_cm,
            area.maximum_y_cm
        ]
        .iter()
        .all(|v| v.is_finite())
            && area.minimum_x_cm < area.maximum_x_cm
            && area.minimum_y_cm < area.maximum_y_cm,
        "invalid operating area"
    );
    ensure!(
        p.anchors.iter().all(|a| area.minimum_x_cm <= a.position.x
            && a.position.x <= area.maximum_x_cm
            && area.minimum_y_cm <= a.position.y
            && a.position.y <= area.maximum_y_cm),
        "an anchor lies outside the operating area"
    );
    let mut links = BTreeSet::new();
    let mut scales = BTreeSet::new();
    for link in &p.observations {
        ensure!(
            link.to_is_anchor
                && agents.contains(&link.from)
                && anchors.contains(&link.to)
                && links.insert((link.from, link.to)),
            "invalid or repeated agent to anchor link"
        );
        // A link whose every reading was inadmissible after calibration is present without samples.
        ensure!(
            link.tof_ns.is_empty() && link.calibrated_samples.len() <= unit.samples,
            "invalid report count or raw timings"
        );
        ensure!(
            link.calibrated_samples
                .iter()
                .all(|s| s.is_valid() && s.distance_cm <= MAXIMUM_RANGE_CM),
            "invalid or inadmissible calibrated range"
        );
        scales.extend(link.calibrated_samples.iter().map(|s| s.scale_cm.to_bits()));
    }
    // Every method receives the one packet scale of the dataset's calibration.
    ensure!(scales.len() <= 1, "changed uncertainty within a unit");
    // Missing links and agents without any link remain solver inputs. No availability model enters.
    Ok(())
}

/// Jin, FNL, STRONG and ADE-BP, each at most once. A run sends the three and a selection of a method's options sends
/// that method alone, ADE-BP's runs and selections send ADE-BP alone. The isolation of a crashed solve resends a crashed
/// request method by method, so every single method of the four must pass.
fn roster_allowed(methods: &[MethodKind]) -> bool {
    let names: BTreeSet<_> = methods.iter().map(|m| m.name()).collect();
    !methods.is_empty()
        && names.len() == methods.len()
        && methods.iter().all(|m| {
            matches!(
                m,
                MethodKind::Jin2021 | MethodKind::Fnl | MethodKind::Strong | MethodKind::AdeBp2015
            )
        })
}

/// The rule of the Bregar adapter for scenes with real height spread. Jin, FNL and STRONG take the planar or the
/// spatial model that their selection on the training half chose (baseline_selection.py). ADE-BP implements the planar
/// model alone, the published form of its paper.
fn dimension_allowed(method: MethodKind, policy: DimensionPolicy) -> bool {
    if method == MethodKind::AdeBp2015 {
        return policy == DimensionPolicy::Planar;
    }
    matches!(
        method,
        MethodKind::Jin2021 | MethodKind::Fnl | MethodKind::Strong
    ) && matches!(policy, DimensionPolicy::Planar | DimensionPolicy::Spatial)
}

/// ADE-BP keeps the declared settings of its anchor only runs, the 500 particles and the four rounds of its paper and
/// the cap of 256 importance refinements, and takes each unit's area rather than one of its own, as the TIERS and IDLab
/// adapters give it.
fn budget_allowed(config: &MethodConfig) -> bool {
    match config {
        MethodConfig::AdeBp2015 {
            area,
            particles,
            iterations,
            importance_iterations,
            ..
        } => area.is_none() && (*particles, *iterations, *importance_iterations) == (500, 4, 256),
        _ => true,
    }
}

/// The runner of one baseline for one unit, ADE-BP with the unit's operating area and every other method as the
/// request configures it.
fn configured_for(baseline: &Baseline, unit: &Unit) -> Result<MethodRunner> {
    let mut config = baseline.method_config.clone();
    if let MethodConfig::AdeBp2015 { area, .. } = &mut config {
        *area = Some(unit.area);
    }
    Ok(MethodRunner::configured(baseline.method, config)?)
}

/// The estimate of a record, with whether the solve stayed where it started (stayed_at_start).
fn estimate((result, start): (SolveResult, Option<Vec<Point2>>)) -> Value {
    json!({"positions": result.positions.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
        "stayed_at_start": stayed_at_start(&result, start.as_deref()),
        "agent_ids": result.agent_ids, "metrics": result.metrics})
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--identity" {
        println!("{}", env!("SOLID_SOURCE_IDENTITY"));
        return Ok(());
    }
    ensure!(
        args.len() == 3,
        "usage: heldout-adapter request-json predictions-jsonl"
    );
    let request: Request = serde_json::from_reader(File::open(&args[1])?)?;
    ensure!(
        !request.units.is_empty() && request.units.len() <= MAX_UNITS,
        "a request holds one to {MAX_UNITS} units"
    );
    let methods: Vec<_> = request.baselines.iter().map(|b| b.method).collect();
    ensure!(roster_allowed(&methods), "unexpected baseline roster");
    for b in &request.baselines {
        ensure!(
            dimension_allowed(b.method, b.solve_config.dimension_policy),
            "unexpected inference dimension"
        );
        ensure!(
            b.solve_config.max_time_micros == 0,
            "no wall-clock truncation"
        );
        ensure!(
            budget_allowed(&b.method_config),
            "ADE-BP keeps its declared anchor only settings and the unit's area"
        );
        MethodRunner::configured(b.method, b.method_config.clone())?;
    }
    let mut keys = BTreeSet::new();
    for unit in &request.units {
        validate(unit)?;
        ensure!(keys.insert((unit.id, unit.samples)), "duplicate unit");
    }
    let mut writer = BufWriter::new(File::create_new(&args[2])?);
    for unit in request.units {
        for b in &request.baselines {
            let started = Instant::now();
            let runner = configured_for(b, &unit)?;
            let result: Result<Value> = runner
                .solve_with_start(&unit.problem, &b.solve_config)
                .map(estimate)
                .map_err(Into::into);
            let elapsed_ns = started.elapsed().as_nanos();
            let (estimate, failure) = match result {
                Ok(value) => (Some(value), None),
                Err(error) => (None, Some(format!("{error:#}"))),
            };
            serde_json::to_writer(
                &mut writer,
                &json!({"id": unit.id, "samples": unit.samples, "method": b.method.name(),
                "agent_ids": unit.problem.agent_ids, "estimate": estimate, "failure": failure,
                "elapsed_ns": elapsed_ns}),
            )?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        eprintln!("heldout unit {} samples {} complete", unit.id, unit.samples);
    }
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use solid_core::{Anchor, CalibratedRangeSample, RangeObservation};
    use solid_methods::{AdeRangingModel, LocalizationMethod};

    fn unit(anchors: u32, agents: u32) -> Unit {
        let anchors: Vec<_> = (0..anchors)
            .map(|i| Anchor {
                id: 101 + i,
                position: Point2::new(300.0 * i as f64, 100.0 * (i % 2) as f64),
                height_cm: 150.0 + 40.0 * i as f64,
            })
            .collect();
        let mut observations = Vec::new();
        for agent in 1..=agents {
            let position = Point2::new(80.0 * agent as f64, 50.0);
            for anchor in &anchors {
                observations.push(RangeObservation {
                    from: agent,
                    to: anchor.id,
                    to_is_anchor: true,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(
                        position.distance(anchor.position),
                        20.0,
                    )],
                });
            }
        }
        Unit {
            id: 7,
            samples: 1,
            area: AreaBounds {
                minimum_x_cm: -2000.0,
                maximum_x_cm: 4000.0,
                minimum_y_cm: -2000.0,
                maximum_y_cm: 2000.0,
            },
            problem: PreparedProblem {
                attempts: None,
                agent_ids: (1..=agents).collect(),
                anchors,
                observations,
                initial_positions: vec![Point2::new(0.0, 0.0); agents as usize],
                mode: LocalizationMode::AnchorOnly,
            },
        }
    }

    #[test]
    fn any_anchor_and_agent_count_is_accepted() {
        for anchors in 1..=10 {
            for agents in [2, 8] {
                validate(&unit(anchors, agents)).unwrap();
            }
        }
        let mut single = unit(3, 2);
        single.problem.agent_ids.pop();
        single.problem.initial_positions.pop();
        single.problem.observations.retain(|o| o.from == 1);
        assert!(validate(&single).is_err());
        let mut none = unit(1, 2);
        none.problem.anchors.clear();
        none.problem.observations.clear();
        assert!(validate(&none).is_err());
    }

    #[test]
    fn missing_links_unranged_agents_and_empty_answered_links_are_inputs() {
        let mut input = unit(4, 3);
        input.problem.observations.retain(|o| o.from != 3);
        validate(&input).unwrap();
        input.problem.observations[0].calibrated_samples.clear();
        validate(&input).unwrap();
        input.problem.observations.clear();
        validate(&input).unwrap();
    }

    #[test]
    fn peer_links_timings_repeats_and_extra_readings_are_refused() {
        let mut input = unit(2, 2);
        input.problem.observations[0].to_is_anchor = false;
        assert!(validate(&input).is_err());
        let mut input = unit(2, 2);
        input.problem.observations[0].tof_ns.push(12.0);
        assert!(validate(&input).is_err());
        let mut input = unit(2, 2);
        let repeated = input.problem.observations[0].clone();
        input.problem.observations.push(repeated);
        assert!(validate(&input).is_err());
        let mut input = unit(2, 2);
        input.problem.observations[0]
            .calibrated_samples
            .push(CalibratedRangeSample::new(10.0, 20.0));
        assert!(validate(&input).is_err());
        let mut input = unit(2, 2);
        input.problem.observations[0].calibrated_samples[0].distance_cm = -1.0;
        assert!(validate(&input).is_err());
        let mut input = unit(2, 2);
        input.problem.mode = LocalizationMode::AnchorAided;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn the_area_must_hold_every_anchor() {
        let mut input = unit(3, 2);
        input.area.maximum_x_cm = 500.0;
        assert!(validate(&input).is_err());
        let mut input = unit(3, 2);
        input.area.minimum_y_cm = f64::NAN;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn every_part_that_crash_isolation_sends_passes_the_roster() {
        use MethodKind::{Fnl, Jin2021, Strong};
        let run = [Jin2021, Fnl, Strong];
        assert!(roster_allowed(&run));
        assert!(roster_allowed(&[Strong, Fnl, Jin2021]));
        // A crashed request is solved again unit by unit with the same roster, then method by method.
        for method in run {
            assert!(roster_allowed(&[method]));
        }
        assert!(roster_allowed(&[Jin2021, Strong]));
        assert!(!roster_allowed(&[Jin2021, Fnl, Fnl]));
        assert!(!roster_allowed(&[Strong, Strong]));
        assert!(!roster_allowed(&[]));
    }

    #[test]
    fn ade_bp_runs_alone_or_beside_the_three_once() {
        use MethodKind::{AdeBp2015, Fnl, Jin2021, Strong};
        assert!(roster_allowed(&[AdeBp2015]));
        assert!(roster_allowed(&[Jin2021, AdeBp2015, Fnl, Strong]));
        assert!(roster_allowed(&[Strong, AdeBp2015]));
        assert!(!roster_allowed(&[AdeBp2015, AdeBp2015]));
    }

    #[test]
    fn one_packet_scale_per_unit_and_admissible_ranges_only() {
        let mut input = unit(3, 2);
        input.problem.observations[1].calibrated_samples[0].scale_cm = 21.0;
        assert!(validate(&input).is_err());
        let mut input = unit(3, 2);
        input.problem.observations[0].calibrated_samples[0].distance_cm = 6000.0;
        assert!(validate(&input).is_err());
        input.problem.observations[0].calibrated_samples[0].distance_cm = 5995.8;
        validate(&input).unwrap();
    }

    #[test]
    fn every_baseline_takes_its_selected_model() {
        use DimensionPolicy::{Automatic, Planar, Spatial};
        for method in [MethodKind::Jin2021, MethodKind::Fnl, MethodKind::Strong] {
            assert!(dimension_allowed(method, Planar));
            assert!(dimension_allowed(method, Spatial));
            assert!(!dimension_allowed(method, Automatic));
        }
        // ADE-BP implements the planar model alone
        assert!(dimension_allowed(MethodKind::AdeBp2015, Planar));
        assert!(!dimension_allowed(MethodKind::AdeBp2015, Spatial));
        assert!(!dimension_allowed(MethodKind::AdeBp2015, Automatic));
    }

    /// ADE-BP's baseline entry as the runner builds it from the declared base configuration, with a ranging model.
    fn ade(area: Option<AreaBounds>, particles: usize, iterations: usize, aggregation: &str) -> Baseline {
        serde_json::from_value(json!({"method": "ade-bp-2015",
            "method_config": {"method": "ade-bp-2015", "area": area, "particles": particles, "iterations": iterations,
                "importance_iterations": 256, "ranging_model": {"positive_slope": 0.02, "positive_intercept_cm": 14.0,
                "negative_slope": 0.0, "negative_intercept_cm": 9.0}},
            "solve_config": {"aggregation": aggregation, "dimension_policy": "planar", "gradient_tolerance": 1e-5,
                "history_size": 10, "max_iterations": 10000, "max_time_micros": 0, "max_tof_ns": 200.0,
                "robust_delta": 0.7}}))
        .unwrap()
    }

    #[test]
    fn ade_bp_keeps_its_budget_and_takes_the_unit_area() {
        assert!(budget_allowed(&ade(None, 500, 4, "Median").method_config));
        assert!(budget_allowed(&ade(None, 500, 4, "AllSamples").method_config));
        assert!(!budget_allowed(&ade(None, 500, 10, "Median").method_config));
        assert!(!budget_allowed(&ade(None, 2000, 4, "Median").method_config));
        let input = unit(4, 3);
        assert!(!budget_allowed(&ade(Some(input.area), 500, 4, "Median").method_config));
        // the runner of a unit carries the unit's area, and its solve is the solve with that area configured directly
        let baseline = ade(None, 500, 4, "Median");
        let runner = configured_for(&baseline, &input).unwrap();
        match runner.config() {
            Some(MethodConfig::AdeBp2015 { area, .. }) => assert_eq!(*area, Some(input.area)),
            _ => panic!("ADE-BP lost its configuration"),
        }
        let direct = MethodRunner::configured(
            MethodKind::AdeBp2015,
            MethodConfig::AdeBp2015 {
                area: Some(input.area),
                particles: 500,
                iterations: 4,
                importance_iterations: 256,
                ranging_model: AdeRangingModel {
                    positive_slope: 0.02,
                    positive_intercept_cm: 14.0,
                    negative_slope: 0.0,
                    negative_intercept_cm: 9.0,
                },
            },
        )
        .unwrap();
        let ours = runner.solve(&input.problem, &baseline.solve_config).unwrap();
        let theirs = direct.solve(&input.problem, &baseline.solve_config).unwrap();
        assert_eq!(ours.positions, theirs.positions);
        // the other methods keep the configuration of the request
        let strong: Baseline = serde_json::from_value(json!({"method": "strong",
            "method_config": {"method": "strong", "initialization": "released"},
            "solve_config": {"aggregation": "Median", "dimension_policy": "spatial", "gradient_tolerance": 1e-5,
                "history_size": 10, "max_iterations": 3, "max_time_micros": 0, "max_tof_ns": 200.0,
                "robust_delta": 40.0}}))
        .unwrap();
        assert_eq!(configured_for(&strong, &input).unwrap().config(), Some(&strong.method_config));
    }

    /// A baseline of a request as the runner builds it, with the given start and a short budget, so that the start
    /// shows in the estimate before the solve settles.
    fn started(method: &str, initialization: &str) -> Baseline {
        serde_json::from_value(json!({"method": method,
            "method_config": {"method": method, "initialization": initialization},
            "solve_config": {"aggregation": "Median", "dimension_policy": "spatial", "gradient_tolerance": 1e-5,
                "history_size": 10, "max_iterations": 3, "max_time_micros": 0, "max_tof_ns": 200.0,
                "robust_delta": 40.0}}))
        .unwrap()
    }

    #[test]
    fn the_provided_and_the_geometry_start_give_different_solves() {
        // every agent starts far from the constellation, where the geometric layout does not place it
        let mut input = unit(4, 3);
        input.problem.initial_positions = vec![Point2::new(-3000.0, 2500.0); 3];
        validate(&input).unwrap();
        for method in ["fnl", "strong"] {
            let solve = |initialization: &str| {
                let baseline = started(method, initialization);
                MethodRunner::configured(baseline.method, baseline.method_config)
                    .unwrap()
                    .solve(&input.problem, &baseline.solve_config)
                    .unwrap()
                    .positions
            };
            let provided = solve("released");
            let geometry = solve("solid-geometry");
            // a start gives the same solve again, and the two starts give different ones
            assert_eq!(provided, solve("released"), "{method}");
            assert_eq!(geometry, solve("solid-geometry"), "{method}");
            let largest = provided
                .iter()
                .zip(&geometry)
                .map(|(a, b)| a.distance(*b))
                .fold(0.0, f64::max);
            assert!(largest > 1.0, "{method}: the start moved no estimate");
        }
    }

    #[test]
    fn every_baseline_solves_a_single_anchor_unit_without_panic() {
        let input = unit(1, 3);
        for (kind, policy) in [
            (MethodKind::Jin2021, DimensionPolicy::Spatial),
            (MethodKind::Fnl, DimensionPolicy::Spatial),
            (MethodKind::Strong, DimensionPolicy::Spatial),
            (MethodKind::Strong, DimensionPolicy::Planar),
        ] {
            let config = SolveConfig {
                dimension_policy: policy,
                max_time_micros: 0,
                ..SolveConfig::default()
            };
            let result = MethodRunner::new(kind).solve(&input.problem, &config);
            if let Ok(estimate) = result {
                assert_eq!(estimate.agent_ids, input.problem.agent_ids);
            }
        }
        let baseline = ade(None, 500, 4, "Median");
        let estimate = configured_for(&baseline, &input)
            .unwrap()
            .solve(&input.problem, &baseline.solve_config)
            .unwrap();
        assert_eq!(estimate.agent_ids, input.problem.agent_ids);
    }

    #[test]
    fn truth_is_not_an_adapter_field() {
        let value = json!({"id": 1, "samples": 1, "problem": unit(2, 2).problem, "area": unit(2, 2).area,
            "reference": [[0.0, 0.0]]});
        assert!(serde_json::from_value::<Unit>(value).is_err());
        let value = json!({"baselines": [], "units": [], "profile": {}});
        assert!(serde_json::from_value::<Request>(value).is_err());
    }
}
