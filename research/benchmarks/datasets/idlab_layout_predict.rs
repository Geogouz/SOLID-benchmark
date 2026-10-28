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
    area: Option<AreaBounds>,
    /// SOLID's per unit profile in frozen requests. The adapter runs the baselines only and ignores it.
    #[allow(dead_code)]
    deployment_profile: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    /// SOLID's cooperative profile in frozen requests, ignored by the baseline adapter.
    #[allow(dead_code)]
    profile: Option<Value>,
    scale_cm: f64,
    baselines: Vec<Baseline>,
    units: Vec<Unit>,
}

fn validate(unit: &Unit, scale: f64) -> Result<()> {
    let p = &unit.problem;
    let agents: BTreeSet<_> = p.agent_ids.iter().copied().collect();
    let anchors: BTreeSet<_> = p.anchors.iter().map(|a| a.id).collect();
    ensure!(
        matches!(unit.samples, 1 | 10),
        "unexpected report allowance"
    );
    ensure!(scale.is_finite() && scale > 0.0, "invalid training scale");
    ensure!(
        agents.len() == p.agent_ids.len()
            && anchors.len() == p.anchors.len()
            && agents.len() >= 2
            && agents.is_disjoint(&anchors)
            && agents.union(&anchors).copied().collect::<BTreeSet<_>>() == (1..=15).collect(),
        "retain every IDLab node exactly once"
    );
    ensure!(
        p.initial_positions.len() == agents.len()
            && p.initial_positions
                .iter()
                .all(|v| v.x.is_finite() && v.y.is_finite()),
        "invalid initialization"
    );
    ensure!(
        p.anchors.iter().all(|a| a.position.x.is_finite()
            && a.position.y.is_finite()
            && a.height_cm.is_finite()),
        "invalid known anchor"
    );
    if p.mode == LocalizationMode::AnchorFree {
        ensure!(
            anchors.is_empty() && unit.area.is_none(),
            "AF has no known positions"
        );
    } else {
        ensure!(!anchors.is_empty(), "missing anchors");
        let area = unit
            .area
            .ok_or_else(|| anyhow::anyhow!("missing operating area"))?;
        // SOLID's operating area of the unit by the deployment rules of research/solid/deployment.py, which the runner sets
        // from the known anchors, the anchor ranges and the published floor alone (layout_inputs.py). ADE-BP receives it.
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
                && area.minimum_y_cm < area.maximum_y_cm
                && p.anchors.iter().all(|a| {
                    (area.minimum_x_cm..=area.maximum_x_cm).contains(&a.position.x)
                        && (area.minimum_y_cm..=area.maximum_y_cm).contains(&a.position.y)
                }),
            "the operating area must be a finite box that holds every known anchor"
        );
    }
    let mut pairs = BTreeSet::new();
    for link in &p.observations {
        ensure!(
            agents.contains(&link.from)
                && if link.to_is_anchor {
                    anchors.contains(&link.to)
                } else {
                    p.mode != LocalizationMode::AnchorOnly
                        && agents.contains(&link.to)
                        && link.from < link.to
                }
                && pairs.insert((link.from.min(link.to), link.from.max(link.to))),
            "invalid or duplicate physical link"
        );
        ensure!(
            link.tof_ns.is_empty()
                && !link.calibrated_samples.is_empty()
                && link.calibrated_samples.len() <= unit.samples
                && link
                    .calibrated_samples
                    .iter()
                    .all(|s| s.is_valid() && s.scale_cm == scale),
            "invalid calibrated reports or changed uncertainty"
        );
    }
    // Missing links and isolated agents remain solver inputs, including empty graphs.
    Ok(())
}

/// The Huber radii of STRONG's same-layout selection (strong_threshold.joint_candidates) are 10, 20, 40, 80 and 160 cm,
/// and 1e9 cm stands for no robust loss, a radius that no residual reaches.
const STRONG_RADII_CM: [f64; 6] = [10.0, 20.0, 40.0, 80.0, 160.0, 1.0e9];

/// Jin 2021, FNL and STRONG take the planar or the spatial model that their selection on the IDLab training half chose
/// (baseline_selection.py). ADE-BP implements the planar model alone.
fn dimension_allowed(method: MethodKind, policy: DimensionPolicy) -> bool {
    if method == MethodKind::AdeBp2015 {
        return policy == DimensionPolicy::Planar;
    }
    matches!(policy, DimensionPolicy::Planar | DimensionPolicy::Spatial)
}

/// STRONG's Huber radius is one of the selection's candidates. The other baselines read no radius, and theirs stays
/// unchecked.
fn radius_allowed(method: MethodKind, radius_cm: f64) -> bool {
    method != MethodKind::Strong || STRONG_RADII_CM.contains(&radius_cm)
}

fn check_baseline(b: &Baseline) -> Result<()> {
    ensure!(
        b.solve_config.max_time_micros == 0,
        "no wall-clock truncation"
    );
    ensure!(
        dimension_allowed(b.method, b.solve_config.dimension_policy),
        "unexpected inference dimension"
    );
    ensure!(
        radius_allowed(b.method, b.solve_config.robust_delta),
        "STRONG's radius is not a selection candidate"
    );
    MethodRunner::configured(b.method, b.method_config.clone())?;
    if let MethodConfig::AdeBp2015 { area, .. } = b.method_config {
        ensure!(area.is_none(), "ADE uses the shared per-input area");
    }
    Ok(())
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
        "usage: idlab-layout-adapter request-json predictions-jsonl"
    );
    let request: Request = serde_json::from_reader(File::open(&args[1])?)?;
    ensure!(
        !request.units.is_empty() && request.units.len() <= 32,
        "bounded IDLab check only"
    );
    let mode = request.units[0].problem.mode;
    let af = mode == LocalizationMode::AnchorFree;
    // The frozen campaign's adapter requires the full roster. This copy runs any subset of it, so that each
    // baseline can be run and rerun on the same-layout halves on its own and each selection solves its method alone. It
    // also lets Jin, FNL and STRONG take the planar or the spatial model of their selection on the training half and
    // STRONG any candidate radius of its selection. Everything else is unchanged.
    let roster: BTreeSet<_> = request.baselines.iter().map(|b| b.method.name()).collect();
    let permitted = if af {
        BTreeSet::from(["jin-2021", "fnl"])
    } else {
        BTreeSet::from(["jin-2021", "fnl", "strong", "ade-bp-2015"])
    };
    ensure!(
        !roster.is_empty() && roster.len() == request.baselines.len() && roster.is_subset(&permitted),
        "unexpected native baseline roster"
    );
    let mut keys = BTreeSet::new();
    for unit in &request.units {
        ensure!(
            unit.problem.mode == mode && keys.insert((unit.id, unit.samples)),
            "mixed modes or repeated input"
        );
        validate(unit, request.scale_cm)?;
    }
    for b in &request.baselines {
        check_baseline(b)?;
    }
    let mut writer = BufWriter::new(File::create_new(&args[2])?);
    for unit in request.units {
        for b in &request.baselines {
            let started = Instant::now();
            let mut config = b.method_config.clone();
            if let MethodConfig::AdeBp2015 { area, .. } = &mut config {
                *area = unit.area;
            }
            let runner = MethodRunner::configured(b.method, config)?;
            let method = b.method.name();
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
                &json!({"id": unit.id, "samples": unit.samples,
                "method": method, "agent_ids": unit.problem.agent_ids,
                "estimate": estimate, "failure": failure, "elapsed_ns": elapsed_ns}),
            )?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        eprintln!("IDLab unit {} samples {} complete", unit.id, unit.samples);
    }
    writer.get_ref().sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use solid_core::{CalibratedRangeSample, RangeObservation};

    fn unit() -> Unit {
        Unit {
            id: 1,
            samples: 10,
            area: None,
            deployment_profile: None,
            problem: PreparedProblem {
                attempts: None,
                agent_ids: (1..=15).collect(),
                anchors: vec![],
                initial_positions: vec![Point2::new(0.0, 0.0); 15],
                mode: LocalizationMode::AnchorFree,
                observations: vec![RangeObservation {
                    from: 1,
                    to: 2,
                    to_is_anchor: false,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(10.0, 20.0)],
                }],
            },
        }
    }

    #[test]
    fn short_pools_missing_links_and_isolated_nodes_are_retained() {
        let mut input = unit();
        validate(&input, 20.0).unwrap();
        input.problem.observations.clear();
        validate(&input, 20.0).unwrap();
        assert_eq!(input.problem.agent_ids.len(), 15);
    }

    #[test]
    fn duplicate_links_and_dropped_nodes_are_rejected() {
        let mut input = unit();
        input
            .problem
            .observations
            .push(input.problem.observations[0].clone());
        assert!(validate(&input, 20.0).is_err());
        input.problem.observations.pop();
        input.problem.agent_ids.pop();
        input.problem.initial_positions.pop();
        assert!(validate(&input, 20.0).is_err());
    }

    #[test]
    fn range_scales_and_report_allowance_are_checked() {
        let mut input = unit();
        assert!(validate(&input, 21.0).is_err());
        input.samples = 1;
        let sample = input.problem.observations[0].calibrated_samples[0].clone();
        input.problem.observations[0]
            .calibrated_samples
            .push(sample);
        assert!(validate(&input, 20.0).is_err());
        input.problem.observations[0].calibrated_samples.pop();
        input.problem.observations[0].calibrated_samples[0].distance_cm = -1.0;
        assert!(validate(&input, 20.0).is_err());
    }

    #[test]
    fn anchored_units_take_an_operating_area_that_holds_their_anchors() {
        let mut input = unit();
        input.problem.mode = LocalizationMode::AnchorAided;
        input.problem.agent_ids.retain(|&n| n != 15);
        input.problem.initial_positions.pop();
        input.problem.anchors = vec![solid_core::Anchor {
            id: 15,
            position: Point2::new(100.0, 200.0),
            height_cm: 250.0,
        }];
        input.problem.observations.push(RangeObservation {
            from: 3,
            to: 15,
            to_is_anchor: true,
            tof_ns: vec![],
            calibrated_samples: vec![CalibratedRangeSample::new(300.0, 20.0)],
        });
        assert!(validate(&input, 20.0).is_err());
        let area = |minimum_x_cm, maximum_x_cm| AreaBounds {
            minimum_x_cm,
            maximum_x_cm,
            minimum_y_cm: -300.0,
            maximum_y_cm: 700.0,
        };
        // SOLID's area of the unit, and the extent box of the frozen campaign, both hold the anchor
        let diameter = 4100.0_f64.hypot(2600.0);
        for accepted in [area(-400.0, 600.0), area(100.0 - diameter, 100.0 + diameter)] {
            input.area = Some(accepted);
            validate(&input, 20.0).unwrap();
        }
        for refused in [area(150.0, 600.0), area(600.0, -400.0), area(f64::NAN, 600.0), area(-400.0, f64::INFINITY)] {
            input.area = Some(refused);
            assert!(validate(&input, 20.0).is_err());
        }
    }

    #[test]
    fn truth_is_not_an_adapter_field() {
        let value = json!({"id": 1, "samples": 10, "problem": unit().problem,
            "area": null, "deployment_profile": null, "reference": [[0.0, 0.0]]});
        assert!(serde_json::from_value::<Unit>(value).is_err());
    }

    /// A baseline as the campaign requests carry it, with the given model and robust radius.
    fn baseline(method: &str, dimension: &str, radius_cm: f64) -> Baseline {
        let method_config = match method {
            "jin-2021" => json!({"method": "jin-2021", "residual_multiplier": 2.0}),
            "ade-bp-2015" => json!({"method": "ade-bp-2015", "area": null, "particles": 500,
                "iterations": 10, "importance_iterations": 256, "ranging_model": {"positive_slope": 0.02,
                "positive_intercept_cm": 14.0, "negative_slope": 0.03, "negative_intercept_cm": 9.0}}),
            _ => json!({"method": method, "initialization": "solid-geometry"}),
        };
        serde_json::from_value(json!({"method": method, "method_config": method_config,
            "solve_config": {"aggregation": "Median", "dimension_policy": dimension,
            "gradient_tolerance": 1e-5, "history_size": 10, "max_iterations": 10000,
            "max_time_micros": 0, "max_tof_ns": 200.0, "robust_delta": radius_cm}}))
        .unwrap()
    }

    #[test]
    fn strong_takes_either_model_with_every_candidate_radius() {
        for dimension in ["planar", "spatial"] {
            for radius in STRONG_RADII_CM {
                check_baseline(&baseline("strong", dimension, radius)).unwrap();
            }
            for radius in [0.7, 30.0, 40.5, 1.0e8] {
                assert!(check_baseline(&baseline("strong", dimension, radius)).is_err());
            }
        }
        assert!(check_baseline(&baseline("strong", "automatic", 40.0)).is_err());
        let mut timed = baseline("strong", "planar", 10.0);
        timed.solve_config.max_time_micros = 1;
        assert!(check_baseline(&timed).is_err());
        assert!(!radius_allowed(MethodKind::Strong, f64::NAN));
        assert!(!radius_allowed(MethodKind::Strong, f64::INFINITY));
    }

    #[test]
    fn jin_and_fnl_take_either_model_and_ade_bp_the_planar_one() {
        for method in ["jin-2021", "fnl"] {
            check_baseline(&baseline(method, "planar", 0.7)).unwrap();
            check_baseline(&baseline(method, "spatial", 0.7)).unwrap();
            assert!(check_baseline(&baseline(method, "automatic", 0.7)).is_err());
        }
        // the candidate radii bind STRONG alone
        check_baseline(&baseline("jin-2021", "planar", 33.0)).unwrap();
        check_baseline(&baseline("ade-bp-2015", "planar", 0.7)).unwrap();
        assert!(check_baseline(&baseline("ade-bp-2015", "spatial", 0.7)).is_err());
        let mut fixed_area = baseline("ade-bp-2015", "planar", 0.7);
        if let MethodConfig::AdeBp2015 { area, .. } = &mut fixed_area.method_config {
            *area = Some(AreaBounds {
                minimum_x_cm: 0.0,
                maximum_x_cm: 1.0,
                minimum_y_cm: 0.0,
                maximum_y_cm: 1.0,
            });
        }
        assert!(check_baseline(&fixed_area).is_err());
        let mut mismatched = baseline("strong", "planar", 10.0);
        mismatched.method = MethodKind::Fnl;
        assert!(check_baseline(&mismatched).is_err());
    }
}
