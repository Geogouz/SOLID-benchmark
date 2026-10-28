use std::collections::BTreeSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{ensure, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use solid_core::{Anchor, DimensionPolicy, LocalizationMode, PreparedProblem, SolveConfig};
use solid_methods::{stayed_at_start, AreaBounds, LocalizationMethod, MethodConfig, MethodKind, MethodRunner};

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
    samples: u32,
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
    baselines: Vec<Baseline>,
    units: Vec<Unit>,
}

/// The operating area of an anchored unit is SOLID's, by the deployment rules of research/solid/deployment.py, which the
/// runners set from the known anchors, the anchor ranges and the published arena alone (layout_inputs.py). ADE-BP
/// receives it. It is a finite box that holds every known anchor.
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
    let anchored = p.mode == LocalizationMode::AnchorOnly;
    let cooperative = p.mode == LocalizationMode::AnchorAided;
    ensure!(matches!(unit.samples, 1 | 10), "unexpected sample count");
    let anchors: BTreeSet<_> = p.anchors.iter().map(|a| a.id).collect();
    ensure!(
        if anchored || cooperative {
            (1..=6).contains(&anchors.len()) && anchors.len() == p.anchors.len()
        } else {
            p.anchors.is_empty()
        },
        "unexpected anchors"
    );
    ensure!(
        p.anchors.iter().all(|a| a.position.x.is_finite()
            && a.position.y.is_finite()
            && a.height_cm.is_finite()),
        "invalid anchors"
    );
    let nodes: BTreeSet<_> = p.agent_ids.iter().copied().collect();
    ensure!(
        nodes.len() == p.agent_ids.len() && nodes.len() >= 2,
        "invalid agents"
    );
    ensure!(nodes.is_disjoint(&anchors), "overlapping node roles");
    ensure!(
        !(cooperative || anchored) || nodes.len() + anchors.len() == 8,
        "eight-node TIERS check only"
    );
    if cooperative || anchored {
        let area = unit
            .area
            .ok_or_else(|| anyhow::anyhow!("TIERS anchored modes require the unit's operating area"))?;
        ensure!(
            area_holds_anchors(&area, &p.anchors),
            "the operating area must be a finite box that holds every known anchor"
        );
    } else {
        ensure!(unit.area.is_none(), "unexpected operating area");
    }
    ensure!(
        p.initial_positions.len() == nodes.len(),
        "missing initial positions"
    );
    ensure!(
        p.initial_positions
            .iter()
            .all(|v| v.x.is_finite() && v.y.is_finite()),
        "invalid initialization"
    );
    let mut pairs = BTreeSet::new();
    for link in &p.observations {
        ensure!(
            nodes.contains(&link.from)
                && if anchored {
                    anchors.contains(&link.to)
                } else {
                    nodes.contains(&link.to) || (cooperative && anchors.contains(&link.to))
                },
            "unknown endpoint"
        );
        ensure!(
            if anchors.contains(&link.to) {
                link.to_is_anchor
            } else {
                link.from < link.to && !link.to_is_anchor
            },
            "expected one ordered reporting stream"
        );
        ensure!(pairs.insert((link.from, link.to)), "duplicate link");
        ensure!(
            link.tof_ns.is_empty(),
            "metric ranges must not use Bern timing conversion"
        );
        ensure!(
            link.calibrated_samples.len() == unit.samples as usize,
            "wrong sample count"
        );
        ensure!(
            link.calibrated_samples.iter().all(|v| v.is_valid()),
            "invalid metric sample"
        );
    }
    ensure!(
        pairs.len()
            == if anchored {
                nodes.len() * anchors.len()
            } else {
                nodes.len() * (nodes.len() - 1) / 2 + nodes.len() * anchors.len()
            },
        "incomplete pilot graph"
    );
    Ok(())
}

/// The baselines of the mode, each at most once, Jin 2021 and FNL in the anchor free mode and ADE-BP and STRONG
/// beside them in the anchored modes. A run sends every baseline of its mode and the selection of a method's options on a
/// training half sends that method alone. A request that crashes the adapter is sent again unit by unit and then method
/// by method (isolation.adapter_parts), so every nonempty subset of the roster passes.
fn roster_allowed(methods: &[MethodKind], anchored: bool, cooperative: bool) -> bool {
    let names: BTreeSet<_> = methods.iter().map(|m| m.name()).collect();
    !methods.is_empty()
        && names.len() == methods.len()
        && methods
            .iter()
            .all(|m| matches!(m, MethodKind::Jin2021 | MethodKind::Fnl) || anchored || cooperative)
}

/// Jin, FNL and STRONG take the planar or the spatial model that their selection on the training half chose
/// (baseline_selection.py). ADE-BP implements the planar model alone.
fn dimension_allowed(method: MethodKind, policy: DimensionPolicy) -> bool {
    if method == MethodKind::AdeBp2015 {
        return policy == DimensionPolicy::Planar;
    }
    matches!(policy, DimensionPolicy::Planar | DimensionPolicy::Spatial)
}

/// ADE-BP's rounds of message passing, the four of its paper in the anchor only mode and in the anchor aided mode four or
/// ten, the candidates its selection on the training half chooses between (baseline_selection.py).
fn ade_rounds_allowed(iterations: usize, cooperative: bool) -> bool {
    iterations == 4 || (cooperative && iterations == 10)
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--identity" {
        println!("{}", env!("SOLID_SOURCE_IDENTITY"));
        return Ok(());
    }
    ensure!(
        args.len() == 3,
        "usage: external-pilot request output-jsonl"
    );
    let request: Request = serde_json::from_reader(File::open(&args[1])?)?;
    ensure!(!request.units.is_empty(), "bounded pilot only");
    let mode = request.units[0].problem.mode;
    let anchored = mode == LocalizationMode::AnchorOnly;
    let cooperative = mode == LocalizationMode::AnchorAided;
    ensure!(
        request.units.len()
            <= if anchored {
                600
            } else if cooperative {
                140
            } else {
                60
            },
        "bounded pilot only"
    );
    ensure!(
        request.units.iter().all(|u| u.problem.mode == mode),
        "mixed modes"
    );
    let methods: Vec<_> = request.baselines.iter().map(|b| b.method).collect();
    ensure!(
        roster_allowed(&methods, anchored, cooperative),
        "unexpected baseline roster"
    );
    let baselines = request
        .baselines
        .into_iter()
        .map(|b| {
            ensure!(
                b.solve_config.max_time_micros == 0,
                "no wall-clock truncation"
            );
            ensure!(
                dimension_allowed(b.method, b.solve_config.dimension_policy),
                "unexpected inference dimension"
            );
            if let MethodConfig::AdeBp2015 {
                area,
                particles,
                iterations,
                importance_iterations,
                ..
            } = &b.method_config
            {
                ensure!(
                    area.is_none()
                        && (*particles, *importance_iterations) == (500, 256)
                        && ade_rounds_allowed(*iterations, cooperative),
                    "use the published particles, the declared or selected rounds and the per-input area"
                );
            }
            Ok((
                MethodRunner::configured(b.method, b.method_config)?,
                b.solve_config,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut keys = BTreeSet::new();
    for unit in &request.units {
        ensure!(keys.insert((unit.id, unit.samples)), "duplicate unit");
        validate(unit)?;
    }
    let mut writer = BufWriter::new(File::create_new(&args[2])?);
    for unit in request.units {
        for (method, config) in &baselines {
            let started = Instant::now();
            let configured;
            let method = if let Some(MethodConfig::AdeBp2015 { .. }) = method.config() {
                let mut options = method.config().unwrap().clone();
                if let MethodConfig::AdeBp2015 { area, .. } = &mut options {
                    *area = unit.area;
                }
                configured = MethodRunner::configured(method.kind(), options)?;
                &configured
            } else {
                method
            };
            let result = method
                .solve_with_start(&unit.problem, config)
                .map(|(r, start)| {
                    json!({
                        "positions": r.positions.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                        "stayed_at_start": stayed_at_start(&r, start.as_deref()),
                        "agent_ids": r.agent_ids, "metrics": r.metrics
                    })
                })
                .map_err(Into::into);
            write_record(&mut writer, &unit, method.kind().name(), started, result)?;
        }
        writer.flush()?;
        eprintln!("unit {} samples {} complete", unit.id, unit.samples);
    }
    writer.get_ref().sync_all()?;
    Ok(())
}

fn write_record(
    writer: &mut impl Write,
    unit: &Unit,
    method: &str,
    started: Instant,
    result: Result<Value>,
) -> Result<()> {
    let elapsed_ns = started.elapsed().as_nanos();
    let (estimate, failure) = match result {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(format!("{error:#}"))),
    };
    serde_json::to_writer(
        &mut *writer,
        &json!({
            "id": unit.id, "samples": unit.samples, "method": method,
            "agent_ids": unit.problem.agent_ids, "estimate": estimate,
            "failure": failure, "elapsed_ns": elapsed_ns
        }),
    )?;
    writer.write_all(b"\n")?;
    Ok(())
}

#[cfg(test)]
#[path = "jin_controls.rs"]
mod jin_controls;

#[cfg(test)]
mod strong_training;

#[cfg(test)]
mod tests {
    use super::*;
    use solid_core::{Anchor, CalibratedRangeSample, Point2, RangeObservation};

    /// Seven agents and one anchor in the anchor aided mode, every agent ranging the anchor once.
    fn anchor_aided_unit() -> Unit {
        let anchor = Anchor {
            id: 101,
            position: Point2::new(0.0, 0.0),
            height_cm: 0.0,
        };
        let observations = (1..=7)
            .map(|id| {
                let position = Point2::new(200.0 + 50.0 * id as f64, 300.0 + 20.0 * id as f64);
                let height = 400.0 + 5.0 * id as f64;
                RangeObservation {
                    from: id,
                    to: anchor.id,
                    to_is_anchor: true,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(
                        position.distance(anchor.position).hypot(height - anchor.height_cm),
                        10.0,
                    )],
                }
            })
            .collect();
        Unit {
            id: 1,
            samples: 1,
            area: None,
            deployment_profile: None,
            problem: PreparedProblem {
                attempts: None,
                agent_ids: (1..=7).collect(),
                anchors: vec![anchor],
                observations,
                initial_positions: vec![Point2::new(-500.0, 800.0); 7],
                mode: LocalizationMode::AnchorAided,
            },
        }
    }

    #[test]
    fn every_part_that_crash_isolation_sends_passes_the_roster() {
        use MethodKind::{AdeBp2015, Fnl, Jin2021, Strong};
        // the two flags say whether the mode is anchor only and whether it is anchor aided
        for (anchored, cooperative, roster) in [
            (true, false, vec![Jin2021, AdeBp2015, Fnl, Strong]),
            (false, true, vec![Jin2021, AdeBp2015, Fnl, Strong]),
            (false, false, vec![Jin2021, Fnl]),
        ] {
            // a run, a selection of STRONG's settings, and every part of a crashed request, down to single methods
            for mask in 1..1usize << roster.len() {
                let part: Vec<_> = (0..roster.len())
                    .filter(|index| mask >> index & 1 == 1)
                    .map(|index| roster[index])
                    .collect();
                assert!(roster_allowed(&part, anchored, cooperative), "{part:?}");
                let reversed: Vec<_> = part.iter().rev().copied().collect();
                assert!(roster_allowed(&reversed, anchored, cooperative), "{part:?}");
            }
            assert!(!roster_allowed(&[], anchored, cooperative));
            assert!(!roster_allowed(&[Jin2021, Fnl, Fnl], anchored, cooperative));
        }
        // ADE-BP and STRONG have no anchor free estimator
        for method in [AdeBp2015, Strong] {
            assert!(!roster_allowed(&[method], false, false));
            assert!(!roster_allowed(&[Jin2021, Fnl, method], false, false));
        }
        assert!(!roster_allowed(&[Strong, Strong], true, false));
    }

    #[test]
    fn ade_bp_takes_the_rounds_of_its_paper_or_the_selected_anchor_aided_ones() {
        assert!(ade_rounds_allowed(4, false));
        assert!(!ade_rounds_allowed(10, false));
        assert!(ade_rounds_allowed(4, true));
        assert!(ade_rounds_allowed(10, true));
        for rounds in [0, 1, 5, 9, 11, 20] {
            assert!(!ade_rounds_allowed(rounds, true));
            assert!(!ade_rounds_allowed(rounds, false));
        }
    }

    #[test]
    fn jin_fnl_and_strong_take_their_selected_dimension() {
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

    #[test]
    fn pilot_rejects_timing_conversion_and_duplicate_links() {
        let mut unit = Unit {
            id: 1,
            samples: 1,
            area: None,
            deployment_profile: None,
            problem: PreparedProblem {
                attempts: None,
                agent_ids: vec![1, 2],
                anchors: vec![],
                mode: LocalizationMode::AnchorFree,
                initial_positions: vec![Point2::default(); 2],
                observations: vec![RangeObservation {
                    from: 1,
                    to: 2,
                    to_is_anchor: false,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(150.0, 10.9)],
                }],
            },
        };
        validate(&unit).unwrap();
        unit.problem.observations[0].tof_ns.push(5.0);
        assert!(validate(&unit).is_err());
        unit.problem.observations[0].tof_ns.clear();
        unit.problem
            .observations
            .push(unit.problem.observations[0].clone());
        assert!(validate(&unit).is_err());
    }

    #[test]
    fn aac_requires_all_peer_and_anchor_links_and_the_operating_area() {
        let mut unit = anchor_aided_unit();
        for a in 1..=7 {
            for b in a + 1..=7 {
                unit.problem.observations.push(RangeObservation {
                    from: a,
                    to: b,
                    to_is_anchor: false,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(100.0, 10.0)],
                });
            }
        }
        assert!(validate(&unit).is_err());
        // SOLID's operating area of the unit, a box around the anchor at the origin padded by its ranges
        unit.area = Some(AreaBounds {
            minimum_x_cm: -700.0,
            maximum_x_cm: 700.0,
            minimum_y_cm: -650.0,
            maximum_y_cm: 750.0,
        });
        validate(&unit).unwrap();
        // the extent box of the frozen campaign holds the anchor too
        let diagonal = 800.0_f64.hypot(900.0);
        unit.area = Some(AreaBounds {
            minimum_x_cm: -diagonal,
            maximum_x_cm: diagonal,
            minimum_y_cm: -diagonal,
            maximum_y_cm: diagonal,
        });
        validate(&unit).unwrap();
        // a box without the anchor, an empty box and a box without finite bounds are refused
        for (minimum_x_cm, maximum_x_cm) in [(1.0, 700.0), (10.0, -10.0), (f64::NAN, 700.0), (-700.0, f64::INFINITY)] {
            unit.area = Some(AreaBounds {
                minimum_x_cm,
                maximum_x_cm,
                minimum_y_cm: -650.0,
                maximum_y_cm: 750.0,
            });
            assert!(validate(&unit).is_err(), "{minimum_x_cm} {maximum_x_cm}");
        }
        unit.area = Some(AreaBounds {
            minimum_x_cm: -700.0,
            maximum_x_cm: 700.0,
            minimum_y_cm: -650.0,
            maximum_y_cm: 750.0,
        });
        unit.problem.observations.pop();
        assert!(validate(&unit).is_err());
    }

    #[test]
    fn jin_noiseless_anchor_geometry() {
        let method = MethodRunner::new(MethodKind::Jin2021);
        let config = SolveConfig {
            dimension_policy: DimensionPolicy::Planar,
            aggregation: solid_core::ObservationAggregation::Median,
            max_time_micros: 0,
            ..SolveConfig::default()
        };
        let mut cases = Vec::new();
        for control in jin_controls::cases() {
            let problem = &control.problem;
            let estimate = method.solve(problem, &config).unwrap();
            assert_eq!(estimate.agent_ids, problem.agent_ids);
            assert!(estimate
                .positions
                .iter()
                .all(|p| p.x.is_finite() && p.y.is_finite()));
            let pairs = jin_controls::pairs(&control.reference, &estimate.positions);
            let error = jin_controls::mean_error(&pairs);
            if problem.anchors.len() == 1 {
                assert!(estimate
                    .positions
                    .iter()
                    .all(|p| p.distance(problem.anchors[0].position) < 1e-4));
            } else if control.name == "three-noncollinear" {
                assert!(error < 1.0, "noncollinear control error {error}");
            }
            cases.push(json!({"case": control.name, "samples": control.samples, "problem": problem,
                "reference": control.reference, "estimate": estimate, "pairs": pairs, "mean_pair_error_cm": error}));
        }
        println!(
            "JIN_RECOVERY_DIAGNOSTIC {}",
            json!({"method_config": method.config(), "solve_config": config, "cases": cases})
        );
    }
}
