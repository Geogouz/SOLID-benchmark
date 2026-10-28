use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::Deserialize;
use serde_json::json;
use solid_core::{Anchor, DimensionPolicy, LocalizationMode, PreparedProblem, SolveConfig};
use solid_methods::{
    stayed_at_start, AreaBounds, LocalizationMethod, MethodConfig, MethodKind, MethodRunner,
};

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
    scale_cm: f64,
    strong_radius_cm: f64,
    problem: PreparedProblem,
    /// SOLID's operating area of the unit by the deployment rules (layout_inputs.py), which ADE-BP receives wherever it
    /// runs. The other baselines take no area. The runners of this addition send an area in ADE-BP's requests alone,
    /// which hold ADE-BP alone (select_ade.py, run_ade.py), and this adapter checks that an area holds every anchor.
    area: Option<AreaBounds>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    baselines: Vec<Baseline>,
    units: Vec<Unit>,
}

/// A finite box that holds every known anchor, the operating area ADE-BP receives (the rule of the TIERS adapter).
fn area_holds_anchors(area: &AreaBounds, anchors: &[Anchor]) -> bool {
    [
        area.minimum_x_cm,
        area.maximum_x_cm,
        area.minimum_y_cm,
        area.maximum_y_cm,
    ]
    .iter()
    .all(|v| v.is_finite())
        && area.minimum_x_cm < area.maximum_x_cm
        && area.minimum_y_cm < area.maximum_y_cm
        && anchors.iter().all(|a| {
            (area.minimum_x_cm..=area.maximum_x_cm).contains(&a.position.x)
                && (area.minimum_y_cm..=area.maximum_y_cm).contains(&a.position.y)
        })
}

fn validate(unit: &Unit) -> Result<()> {
    let p = &unit.problem;
    ensure!(
        p.mode == LocalizationMode::AnchorOnly,
        "Bregar has no peer ranges"
    );
    ensure!(matches!(unit.samples, 1 | 10), "unexpected report count");
    ensure!(
        unit.scale_cm.is_finite() && unit.scale_cm > 0.0,
        "invalid training scale"
    );
    ensure!(
        unit.strong_radius_cm.is_finite() && unit.strong_radius_cm > 0.0,
        "invalid STRONG radius"
    );
    ensure!(
        !p.agent_ids.is_empty() && p.agent_ids.len() == p.initial_positions.len(),
        "missing agents"
    );
    let agents: BTreeSet<_> = p.agent_ids.iter().copied().collect();
    let anchors: BTreeSet<_> = p.anchors.iter().map(|a| a.id).collect();
    ensure!(
        agents.len() == p.agent_ids.len() && anchors.len() == p.anchors.len(),
        "duplicate node"
    );
    ensure!(
        matches!(anchors.len(), 7 | 8),
        "unexpected surveyed anchors"
    );
    ensure!(
        p.initial_positions
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite()),
        "invalid initialization"
    );
    ensure!(
        p.anchors.iter().all(|a| a.position.x.is_finite()
            && a.position.y.is_finite()
            && a.height_cm.is_finite()),
        "invalid surveyed anchor"
    );
    if let Some(area) = &unit.area {
        ensure!(
            area_holds_anchors(area, &p.anchors),
            "the operating area must be a finite box that holds every known anchor"
        );
    }
    let mut links = BTreeSet::new();
    for link in &p.observations {
        ensure!(
            link.to_is_anchor
                && agents.contains(&link.from)
                && anchors.contains(&link.to)
                && links.insert((link.from, link.to)),
            "invalid or repeated link"
        );
        // a link the radio answered whose every reading was inadmissible after calibration carries no sample
        ensure!(
            link.tof_ns.is_empty() && link.calibrated_samples.len() <= unit.samples,
            "invalid accepted report count"
        );
        ensure!(
            link.calibrated_samples
                .iter()
                .all(|s| s.is_valid() && s.scale_cm == unit.scale_cm),
            "invalid calibrated range or changed uncertainty"
        );
    }
    // Missing links are permitted. No availability model or truth enters this adapter.
    Ok(())
}

/// Jin 2021, FNL, STRONG and ADE-BP 2015, each at most once. A run sends the three and the selection of a method's
/// options on the training block sends that method alone, ADE-BP's runs and selections send ADE-BP alone. A request
/// that crashes the adapter is sent again unit by unit and then method by method (isolation.adapter_parts), so every
/// nonempty subset of the four passes.
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

/// The planar or the spatial model, which the selection of each method's options on the training block of each
/// environment chooses for Jin 2021, FNL and STRONG alike (baseline_selection.py). ADE-BP implements the planar model
/// alone, the published form of its paper.
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

/// ADE-BP's runner for one unit, with the unit's operating area, which every unit of an ADE-BP request carries.
fn with_area(method: &MethodRunner, unit: &Unit) -> Result<MethodRunner> {
    let mut options = method
        .config()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("ADE-BP runs from its declared configuration"))?;
    if let MethodConfig::AdeBp2015 { area, .. } = &mut options {
        *area = unit.area;
    }
    Ok(MethodRunner::configured(method.kind(), options)?)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--identity" {
        println!("{}", env!("SOLID_SOURCE_IDENTITY"));
        return Ok(());
    }
    ensure!(
        args.len() == 3,
        "usage: bregar-adapter request-json predictions-jsonl"
    );
    let request: Request = serde_json::from_reader(File::open(&args[1])?)?;
    ensure!(
        !request.units.is_empty() && request.units.len() <= 8,
        "Bregar batches contain at most eight inputs"
    );
    let methods: Vec<_> = request.baselines.iter().map(|b| b.method).collect();
    ensure!(roster_allowed(&methods), "unexpected spatial roster");
    let mut keys = BTreeSet::new();
    for unit in &request.units {
        validate(unit)?;
        ensure!(keys.insert((unit.id, unit.samples)), "duplicate unit");
    }
    let mut baselines = Vec::new();
    for b in request.baselines {
        ensure!(
            dimension_allowed(b.method, b.solve_config.dimension_policy)
                && b.solve_config.max_time_micros == 0,
            "unexpected dimension or time limit"
        );
        ensure!(
            budget_allowed(&b.method_config),
            "ADE-BP keeps its declared anchor only settings and the unit's area"
        );
        ensure!(
            b.method != MethodKind::AdeBp2015 || request.units.iter().all(|u| u.area.is_some()),
            "ADE-BP takes the operating area every unit carries"
        );
        baselines.push((
            MethodRunner::configured(b.method, b.method_config)?,
            b.solve_config,
        ));
    }
    let mut writer = BufWriter::new(File::create_new(&args[2])?);
    for unit in request.units {
        for (method, config) in &baselines {
            let started = Instant::now();
            let mut config = *config;
            if method.kind().name() == "strong" {
                config.robust_delta = unit.strong_radius_cm;
            }
            let configured;
            let method = if method.kind() == MethodKind::AdeBp2015 {
                configured = with_area(method, &unit)?;
                &configured
            } else {
                method
            };
            let result = method
                .solve_with_start(&unit.problem, &config)
                .map(|(r, start)| {
                    let stayed = stayed_at_start(&r, start.as_deref());
                    let mut value = json!(r);
                    value["stayed_at_start"] = json!(stayed);
                    value
                })
                .map_err(|e| e.to_string());
            let method = method.kind().name();
            let config = Some(config);
            let elapsed_ns = started.elapsed().as_nanos();
            let (estimate, failure) = match result {
                Ok(value) => (Some(value), None),
                Err(error) => (None, Some(error.to_string())),
            };
            serde_json::to_writer(
                &mut writer,
                &json!({"id": unit.id, "samples": unit.samples,
                "method": method, "agent_ids": unit.problem.agent_ids, "estimate": estimate,
                "failure": failure, "elapsed_ns": elapsed_ns, "solve_config": config}),
            )?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        eprintln!("Bregar unit {} samples {} complete", unit.id, unit.samples);
    }
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use solid_core::{CalibratedRangeSample, Point2, RangeObservation};
    use solid_methods::AdeRangingModel;

    fn unit() -> Unit {
        Unit {
            id: 1,
            samples: 10,
            scale_cm: 20.0,
            strong_radius_cm: 50.0,
            area: None,
            problem: PreparedProblem {
                attempts: None,
                agent_ids: vec![1, 2],
                anchors: (101..109)
                    .map(|id| Anchor {
                        id,
                        position: Point2::new(id as f64, 0.0),
                        height_cm: 50.0,
                    })
                    .collect(),
                observations: vec![RangeObservation {
                    from: 1,
                    to: 101,
                    to_is_anchor: true,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(0.0, 20.0)],
                }],
                initial_positions: vec![Point2::new(0.0, 0.0); 2],
                mode: LocalizationMode::AnchorOnly,
            },
        }
    }

    #[test]
    fn incomplete_blocks_zero_ranges_and_agents_without_links_are_retained() {
        let mut input = unit();
        validate(&input).unwrap();
        input.problem.observations.clear();
        validate(&input).unwrap();
        assert_eq!(input.problem.agent_ids, vec![1, 2]);
    }

    #[test]
    fn rejected_readings_and_uncertainty_changes_cannot_reenter() {
        let mut input = unit();
        input.problem.observations[0].calibrated_samples[0].distance_cm = -0.1;
        assert!(validate(&input).is_err());
        input.problem.observations[0].calibrated_samples[0].distance_cm = 1.0;
        input.problem.observations[0].calibrated_samples[0].scale_cm = 21.0;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn answered_links_without_admissible_readings_are_accepted_and_skipped() {
        let mut input = unit();
        input.problem.observations.push(RangeObservation {
            from: 2,
            to: 102,
            to_is_anchor: true,
            tof_ns: vec![],
            calibrated_samples: vec![],
        });
        validate(&input).unwrap();
        input.problem.observations[1].tof_ns.push(10.0);
        assert!(validate(&input).is_err());
        input.problem.observations[1].tof_ns.clear();
        input.problem.observations[1].calibrated_samples =
            vec![CalibratedRangeSample::new(1.0, 20.0); 11];
        assert!(validate(&input).is_err());
        input.problem.observations[1].calibrated_samples.clear();
        // the empty link changes no baseline's estimate
        let config = SolveConfig {
            dimension_policy: DimensionPolicy::Spatial,
            max_time_micros: 0,
            ..SolveConfig::default()
        };
        let mut without = unit();
        without
            .problem
            .observations
            .extend((102..109).map(|anchor| RangeObservation {
                from: 1,
                to: anchor,
                to_is_anchor: true,
                tof_ns: vec![],
                calibrated_samples: vec![CalibratedRangeSample::new(100.0, 20.0)],
            }));
        let mut with = without.problem.clone();
        with.observations.push(RangeObservation {
            from: 2,
            to: 101,
            to_is_anchor: true,
            tof_ns: vec![],
            calibrated_samples: vec![],
        });
        for kind in [MethodKind::Jin2021, MethodKind::Fnl, MethodKind::Strong] {
            let method = MethodRunner::new(kind);
            let expected = method.solve(&without.problem, &config);
            let actual = method.solve(&with, &config);
            match (expected, actual) {
                (Ok(expected), Ok(actual)) => {
                    assert_eq!(expected.positions, actual.positions, "{kind:?}")
                }
                (Err(_), Err(_)) => {}
                (expected, actual) => panic!(
                    "{kind:?}: {:?} against {:?}",
                    expected.is_ok(),
                    actual.is_ok()
                ),
            }
        }
    }

    #[test]
    fn every_part_that_crash_isolation_sends_passes_the_roster() {
        use MethodKind::{AdeBp2015, Fnl, Jin2021, Strong};
        assert!(roster_allowed(&[Jin2021, Fnl, Strong]));
        assert!(roster_allowed(&[Strong, Jin2021, Fnl]));
        // a selection of one method's options, and a crashed request solved again method by method
        for method in [Jin2021, Fnl, Strong, AdeBp2015] {
            assert!(roster_allowed(&[method]));
        }
        assert!(roster_allowed(&[Jin2021, Strong]));
        assert!(roster_allowed(&[Fnl, Jin2021]));
        assert!(!roster_allowed(&[Strong, Strong]));
        assert!(!roster_allowed(&[Jin2021, Fnl, Fnl]));
        assert!(!roster_allowed(&[]));
        // ADE-BP runs with its planar model, alone or beside the three, once
        assert!(roster_allowed(&[Jin2021, Fnl, Strong, AdeBp2015]));
        assert!(!roster_allowed(&[AdeBp2015, AdeBp2015]));
    }

    #[test]
    fn every_baseline_takes_its_selected_model() {
        for method in [MethodKind::Jin2021, MethodKind::Fnl, MethodKind::Strong] {
            assert!(dimension_allowed(method, DimensionPolicy::Planar));
            assert!(dimension_allowed(method, DimensionPolicy::Spatial));
            assert!(!dimension_allowed(method, DimensionPolicy::Automatic));
        }
        // ADE-BP implements the planar model alone
        assert!(dimension_allowed(
            MethodKind::AdeBp2015,
            DimensionPolicy::Planar
        ));
        assert!(!dimension_allowed(
            MethodKind::AdeBp2015,
            DimensionPolicy::Spatial
        ));
        assert!(!dimension_allowed(
            MethodKind::AdeBp2015,
            DimensionPolicy::Automatic
        ));
    }

    #[test]
    fn frozen_radius_is_separate_from_shared_range_uncertainty() {
        let mut input = unit();
        validate(&input).unwrap();
        assert_eq!(input.scale_cm, 20.0);
        assert_eq!(input.strong_radius_cm, 50.0);
        input.strong_radius_cm = 0.0;
        assert!(validate(&input).is_err());
    }

    #[test]
    fn unknown_coordinates_and_profiles_are_not_predictor_inputs() {
        let input = json!({"id": 1, "samples": 10, "scale_cm": 20.0, "strong_radius_cm": 50.0,
            "problem": unit().problem, "reference": [[0.0, 0.0]]});
        assert!(serde_json::from_value::<Unit>(input).is_err());
        let input = json!({"baselines": [], "units": [], "deployment_profile": {}});
        assert!(serde_json::from_value::<Request>(input).is_err());
    }

    fn area(minimum_x_cm: f64, maximum_x_cm: f64) -> AreaBounds {
        AreaBounds {
            minimum_x_cm,
            maximum_x_cm,
            minimum_y_cm: -300.0,
            maximum_y_cm: 300.0,
        }
    }

    #[test]
    fn a_unit_without_an_area_reads_as_before_and_an_area_must_hold_every_anchor() {
        let without = json!({"id": 1, "samples": 10, "scale_cm": 20.0, "strong_radius_cm": 50.0, "problem": unit().problem});
        assert!(serde_json::from_value::<Unit>(without).unwrap().area.is_none());
        let mut input = unit();
        for accepted in [area(0.0, 200.0), area(-1000.0, 1000.0)] {
            input.area = Some(accepted);
            validate(&input).unwrap();
        }
        for refused in [area(102.0, 200.0), area(200.0, 0.0), area(f64::NAN, 200.0), area(0.0, f64::INFINITY)] {
            input.area = Some(refused);
            assert!(validate(&input).is_err());
        }
    }

    fn ranging_model() -> AdeRangingModel {
        AdeRangingModel {
            positive_slope: 0.02,
            positive_intercept_cm: 14.0,
            negative_slope: 0.0,
            negative_intercept_cm: 9.0,
        }
    }

    fn ade(area: Option<AreaBounds>, particles: usize, iterations: usize) -> MethodConfig {
        MethodConfig::AdeBp2015 {
            area,
            particles,
            iterations,
            importance_iterations: 256,
            ranging_model: ranging_model(),
        }
    }

    #[test]
    fn ade_bp_keeps_its_budget_and_takes_the_unit_area() {
        assert!(budget_allowed(&ade(None, 500, 4)));
        assert!(!budget_allowed(&ade(None, 500, 10)));
        assert!(!budget_allowed(&ade(None, 100, 4)));
        assert!(!budget_allowed(&ade(Some(area(0.0, 200.0)), 500, 4)));
        let strong: MethodConfig = serde_json::from_value(json!({"method": "strong", "initialization": "released"})).unwrap();
        assert!(budget_allowed(&strong));
        // the runner of a unit carries the unit's area, and solves as ADE-BP configured with that area directly
        let mut input = unit();
        input.problem.agent_ids = vec![1];
        input.problem.initial_positions = vec![Point2::new(0.0, 0.0)];
        input.problem.observations = (101..109)
            .map(|anchor| RangeObservation {
                from: 1,
                to: anchor,
                to_is_anchor: true,
                tof_ns: vec![],
                calibrated_samples: vec![CalibratedRangeSample::new(100.0 + anchor as f64, 20.0)],
            })
            .collect();
        input.area = Some(area(-500.0, 700.0));
        validate(&input).unwrap();
        let declared = MethodRunner::configured(MethodKind::AdeBp2015, ade(None, 500, 4)).unwrap();
        let runner = with_area(&declared, &input).unwrap();
        assert_eq!(runner.config(), Some(&ade(input.area, 500, 4)));
        let config = SolveConfig {
            dimension_policy: DimensionPolicy::Planar,
            max_time_micros: 0,
            ..SolveConfig::default()
        };
        let direct = MethodRunner::configured(MethodKind::AdeBp2015, ade(input.area, 500, 4)).unwrap();
        assert_eq!(
            runner.solve(&input.problem, &config).unwrap().positions,
            direct.solve(&input.problem, &config).unwrap().positions
        );
    }
}
