use crate::LocalizationEngine;
use super::*;
use crate::{Anchor, BlockRangeProfile, CalibratedRangeSample, CalibrationModel, RangeErrorProfile, RangeObservation};

fn profile() -> CooperativeProfile {
    CooperativeProfile {
        range_error: RangeErrorProfile {
            positive_scale: 1.2,
            negative_scale: 0.8,
        },
        block_range: None,
        calibration: None,
        starts: 32,
        use_provided_start: None,
        presence: crate::PresenceProfile::default(),
    }
}

fn block_profile() -> CooperativeProfile {
    let mut profile = profile();
    profile.block_range = Some(BlockRangeProfile {
        kappa: 0.4,
        excess_probability: 0.1,
        excess_mean_rel: 5.0,
        persistent_sd_rel: 1.5,
        height_nodes: vec![-50.0, 0.0, 50.0],
        height_weights: vec![1.0, 1.0, 1.0],
        refine_from_median: false,
        nu: 4.0,
        independent_packets: false,
        offset_shape: crate::OffsetShape::Gaussian,
        early_detection_per_cm: 0.0,
    });
    profile
}

fn problem(mode: LocalizationMode) -> PreparedProblem {
    let mut problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3, 4],
        anchors: Vec::new(),
        observations: vec![
            RangeObservation {
                from: 1,
                to: 2,
                to_is_anchor: false,
                tof_ns: vec![5.0, 8.0, f64::NAN, -1.0, 201.0],
                calibrated_samples: Vec::new(),
            },
            RangeObservation {
                from: 2,
                to: 3,
                to_is_anchor: false,
                tof_ns: Vec::new(),
                calibrated_samples: vec![CalibratedRangeSample::new(350.0, 10.0)],
            },
        ],
        initial_positions: vec![Point2::default(); 4],
        mode,
    };
    if mode == LocalizationMode::AnchorAided {
        problem.anchors.push(Anchor {
            id: 10,
            position: Point2::new(100.0, -50.0),
            height_cm: 200.0,
        });
        problem.observations.push(RangeObservation {
            from: 1,
            to: 10,
            to_is_anchor: true,
            tof_ns: vec![10.0, 11.0],
            calibrated_samples: Vec::new(),
        });
    }
    problem
}

#[test]
fn missingness_is_limited_to_observed_components() {
    let graph = af::Graph::new(&problem(LocalizationMode::AnchorFree), None, CalibrationModel::default(), &crate::PresenceProfile::default());
    assert_eq!(graph.ranges.len(), 2);
    assert_eq!(graph.pairs.len(), 3);
    assert!(graph.pairs.iter().all(|p| p.left != 3 && p.right != 3));
    assert_eq!(graph.pairs.iter().filter(|p| !p.observed).count(), 1);
}

#[test]
fn sparse_and_disconnected_inputs_keep_every_agent() {
    for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided] {
        let p = problem(mode);
        let engine = LocalizationEngine::new();
        let mut a = engine
            .solve_with_cooperative_profile(&p, profile())
            .unwrap();
        let mut b = engine
            .solve_with_cooperative_profile(&p, profile())
            .unwrap();
        a.metrics.elapsed_micros = 0;
        b.metrics.elapsed_micros = 0;
        assert_eq!(a, b);
        assert_eq!(a.agent_ids, p.agent_ids);
        assert_eq!(a.positions.len(), 4);
        assert!(a
            .positions
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite()));
        assert_eq!(a.metrics.starts, 32);
        assert_eq!(a.metrics.dimensions, 2);
    }
}

#[test]
fn presence_law_objectives_match_finite_differences() {
    let profile = block_profile();
    let block = profile.block_range.as_ref();
    let range_error = profile.range_error;
    for shape in [crate::PresenceShape::Gaussian, crate::PresenceShape::Logistic] {
        let law = crate::PresenceProfile { shape, quantile: 1.0, factor: 0.75, ..crate::PresenceProfile::default() };
        let af_problem = problem(LocalizationMode::AnchorFree);
        let af_graph = af::Graph::new(&af_problem, block, CalibrationModel::default(), &law);
        assert!(af_graph.presence.is_some());
        assert!(af_graph.ranges.iter().all(|r| r.table.is_some()));
        let mut af_objective = af::Objective { graph: &af_graph, range_error: &range_error };
        let aac_problem = problem(LocalizationMode::AnchorAided);
        let aac_graph = aac::Graph::new(&aac_problem, block, CalibrationModel::default(), &law);
        assert!(aac_graph.presence.is_some());
        assert_eq!(aac_graph.heights.len(), 3);
        assert!(aac_graph.pairs.iter().any(|p| p.opportunity && p.range.is_none()), "the test problem has a missing opportunity");
        let mut aac_objective = aac::Objective { graph: &aac_graph, profile: &range_error };
        let layouts: [(&mut dyn DifferentiableObjective, Vec<f64>); 2] = [
            (&mut af_objective, vec![5.0, 20.0, 150.0, 15.0, 75.0, 245.0, 300.0, 400.0]),
            (&mut aac_objective, vec![30.0, 60.0, 170.0, -50.0, 90.0, 210.0, -40.0, 5.0]),
        ];
        for (objective, mut x) in layouts {
            let mut gradient = vec![0.0; x.len()];
            let value = objective.value_gradient(&x, &mut gradient);
            assert!(value.is_finite());
            for i in 0..x.len() {
                let h = 1e-3;
                x[i] += h;
                let high = objective.value_gradient(&x, &mut vec![0.0; x.len()]);
                x[i] -= 2.0 * h;
                let low = objective.value_gradient(&x, &mut vec![0.0; x.len()]);
                x[i] += h;
                let numeric = (high - low) / (2.0 * h);
                assert!(
                    (numeric - gradient[i]).abs() < 1e-5 * (1.0 + gradient[i].abs()),
                    "{shape:?} coordinate {i}: analytic {} numeric {numeric}",
                    gradient[i]
                );
            }
        }
    }
}

#[test]
fn declared_attempts_define_the_opportunities() {
    let block = block_profile();
    let block = block.block_range.as_ref();
    let law = crate::PresenceProfile { quantile: 1.0, factor: 1.0, ..crate::PresenceProfile::default() };
    let open = problem(LocalizationMode::AnchorAided);
    let graph = aac::Graph::new(&open, block, CalibrationModel::default(), &law);
    let missing_open = graph.pairs.iter().filter(|p| p.opportunity && p.range.is_none()).count();
    assert!(missing_open > 0);
    // no declared attempt beyond the observed links: nothing is missing
    let mut closed = open.clone();
    closed.attempts = Some(Vec::new());
    let graph = aac::Graph::new(&closed, block, CalibrationModel::default(), &law);
    assert_eq!(graph.pairs.iter().filter(|p| p.opportunity && p.range.is_none()).count(), 0);
    assert!(graph.pairs.iter().filter(|p| p.range.is_some()).all(|p| p.opportunity));
    // one declared attempt of an agent to an anchor it never ranged makes exactly that pair missing
    let (agent, anchor) = closed
        .agent_ids
        .iter()
        .flat_map(|agent| closed.anchors.iter().map(move |anchor| (*agent, anchor.id)))
        .find(|(agent, anchor)| !closed.observations.iter().any(|o| o.from == *agent && o.to_is_anchor && o.to == *anchor))
        .expect("an agent without a link to some anchor");
    closed.attempts = Some(vec![crate::LinkAttempt { from: agent, to: anchor, to_is_anchor: true }]);
    let graph = aac::Graph::new(&closed, block, CalibrationModel::default(), &law);
    assert_eq!(graph.pairs.iter().filter(|p| p.opportunity && p.range.is_none()).count(), 1);
    // the agent only graph drops unattempted pairs, an unknown member is rejected
    let mut af = problem(LocalizationMode::AnchorFree);
    let all = af::Graph::new(&af, block, CalibrationModel::default(), &law).pairs.len();
    af.attempts = Some(Vec::new());
    let observed_only = af::Graph::new(&af, block, CalibrationModel::default(), &law).pairs.len();
    assert!(observed_only < all);
    af.attempts = Some(vec![crate::LinkAttempt { from: 999, to: af.agent_ids[0], to_is_anchor: false }]);
    assert!(LocalizationEngine::new().solve_with_cooperative_profile(&af, block_profile()).is_err());
}

#[test]
fn block_range_solves_every_cooperative_mode() {
    let engine = LocalizationEngine::new();
    for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided] {
        let p = problem(mode);
        let block = engine.solve_with_cooperative_profile(&p, block_profile()).unwrap();
        let median = engine.solve_with_cooperative_profile(&p, profile()).unwrap();
        assert_eq!(block.positions.len(), 4);
        assert!(block.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
        assert_eq!(block.metrics.starts, 32);
        assert_ne!(block.positions, median.positions);
    }
    let mut refined = block_profile();
    refined.block_range.as_mut().unwrap().refine_from_median = true;
    for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided] {
        let result = engine.solve_with_cooperative_profile(&problem(mode), refined.clone()).unwrap();
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }
    let mut variant = block_profile();
    variant.block_range.as_mut().unwrap().independent_packets = true;
    for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided] {
        let result = engine.solve_with_cooperative_profile(&problem(mode), variant.clone()).unwrap();
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
    }
    let mut invalid = block_profile();
    invalid.block_range.as_mut().unwrap().kappa = 0.0;
    assert!(engine
        .solve_with_cooperative_profile(&problem(LocalizationMode::AnchorFree), invalid)
        .is_err());
}

#[test]
fn invalid_profile_or_topology_returns_errors() {
    let engine = LocalizationEngine::new();
    let mut p = problem(LocalizationMode::AnchorFree);
    let mut missing = profile();
    missing.presence.quantile = 0.0;
    assert!(engine.solve_with_cooperative_profile(&p, missing).is_err());
    for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut invalid = profile();
        invalid.range_error.positive_scale = value;
        assert!(engine.solve_with_cooperative_profile(&p, invalid).is_err());
    }
    p.observations[0].to = 99;
    assert!(matches!(
        engine.solve_with_cooperative_profile(&p, profile()),
        Err(EngineError::UnknownAgent(99))
    ));
    p.observations[0].to = 1;
    assert!(matches!(
        engine.solve_with_cooperative_profile(&p, profile()),
        Err(EngineError::InvalidCooperativeProblem(_))
    ));
    p.agent_ids[1] = 1;
    assert!(matches!(
        engine.solve_with_cooperative_profile(&p, profile()),
        Err(EngineError::DuplicateAgent(1))
    ));
}


fn check_gradient(objective: &mut impl crate::optimize::DifferentiableObjective, parameters: &[f64]) {
    let mut gradient = vec![0.0; parameters.len()];
    objective.value_gradient(parameters, &mut gradient);
    for i in 0..parameters.len() {
        let mut p = parameters.to_vec();
        p[i] += 1e-4;
        let plus = objective.value_gradient(&p, &mut vec![0.0; p.len()]);
        p[i] -= 2e-4;
        let minus = objective.value_gradient(&p, &mut vec![0.0; p.len()]);
        assert!((gradient[i] - (plus - minus) / 2e-4).abs() < 1e-5, "parameter {i}: {} vs {}", gradient[i], (plus - minus) / 2e-4);
    }
}

fn calibrated(from: u32, to: u32, to_is_anchor: bool, distances: &[f64]) -> RangeObservation {
    RangeObservation {
        from,
        to,
        to_is_anchor,
        tof_ns: Vec::new(),
        calibrated_samples: distances.iter().map(|d| CalibratedRangeSample::new(*d, 8.0)).collect(),
    }
}

fn anchor(id: u32, x: f64, y: f64, height_cm: f64) -> Anchor {
    Anchor { id, position: Point2::new(x, y), height_cm }
}

fn pair_set(graph: &aac::Graph, keep: impl Fn(&aac::Pair) -> bool) -> std::collections::BTreeSet<(usize, usize)> {
    graph.pairs.iter().filter(|p| keep(p)).map(|p| (p.left, p.right)).collect()
}

/// Agents 1 and 2 reach the anchor frame through agent 1's anchor links, agents 3 and 4 range only each other and
/// agent 5 has no observation at all.
fn split_problem() -> PreparedProblem {
    PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3, 4, 5],
        anchors: vec![anchor(10, 0.0, 0.0, 100.0), anchor(11, 800.0, 0.0, 100.0), anchor(12, 0.0, 800.0, 100.0)],
        observations: vec![
            calibrated(1, 10, true, &[300.0]),
            calibrated(1, 11, true, &[600.0]),
            calibrated(1, 12, true, &[650.0]),
            calibrated(1, 2, false, &[250.0]),
            calibrated(3, 4, false, &[300.0]),
        ],
        initial_positions: vec![Point2::default(); 5],
        mode: LocalizationMode::AnchorAided,
    }
}

#[test]
fn failed_attempts_need_a_path_of_observed_pairs_between_their_ends() {
    let graph = aac::Graph::new(&split_problem(), None, CalibrationModel::default(), &crate::PresenceProfile::default());
    // agents 0 to 4, anchors 5 to 7
    let expected: std::collections::BTreeSet<(usize, usize)> =
        [(0, 1), (2, 3), (0, 5), (0, 6), (0, 7), (1, 5), (1, 6), (1, 7)].into_iter().collect();
    assert_eq!(pair_set(&graph, |p| p.opportunity), expected);
    assert_eq!(pair_set(&graph, |p| p.opportunity && !p.observed), [(1, 5), (1, 6), (1, 7)].into_iter().collect());
    // declared attempts across components stay out as well
    let mut declared = split_problem();
    declared.attempts = Some(vec![
        crate::LinkAttempt { from: 3, to: 10, to_is_anchor: true },
        crate::LinkAttempt { from: 5, to: 1, to_is_anchor: false },
        crate::LinkAttempt { from: 2, to: 11, to_is_anchor: true },
    ]);
    let graph = aac::Graph::new(&declared, None, CalibrationModel::default(), &crate::PresenceProfile::default());
    assert_eq!(pair_set(&graph, |p| p.opportunity && !p.observed), [(1, 6)].into_iter().collect());
}

#[test]
fn unanchored_agents_have_no_gradient_toward_the_frame() {
    let graph = aac::Graph::new(&split_problem(), None, CalibrationModel::default(), &crate::PresenceProfile::default());
    let range_error = profile().range_error;
    let mut objective = aac::Objective { graph: &graph, profile: &range_error };
    let layout = vec![200.0, 150.0, 330.0, 260.0, -900.0, 40.0, -700.0, 300.0, 1500.0, -800.0];
    let mut gradient = vec![0.0; layout.len()];
    let value = objective.value_gradient(&layout, &mut gradient);
    // the isolated agent has no term at all, the unanchored pair only its own range term
    assert_eq!((gradient[8], gradient[9]), (0.0, 0.0));
    assert!((gradient[4] + gradient[6]).abs() < 1e-12 && (gradient[5] + gradient[7]).abs() < 1e-12);
    // moving the unanchored pair and the isolated agent far away leaves the objective unchanged
    let mut moved = layout.clone();
    for (i, shift) in [(4, 5.0e4), (5, -3.0e4), (6, 5.0e4), (7, -3.0e4), (8, 9.0e4), (9, 7.0e4)] {
        moved[i] += shift;
    }
    let moved_value = objective.value_gradient(&moved, &mut gradient);
    assert!((moved_value - value).abs() < 1e-9 * value.abs().max(1.0), "{value} {moved_value}");
}

#[test]
fn isolated_and_unanchored_agents_do_not_run_away_in_aac() {
    let problem = split_problem();
    let config = SolveConfig { max_time_micros: 0, aggregation: ObservationAggregation::Median, ..SolveConfig::production() };
    let initial = crate::initializer::initial_layout_with(&problem, &config, CalibrationModel::default()).unwrap();
    let typical = aac::Graph::new(&problem, None, CalibrationModel::default(), &crate::PresenceProfile::default()).typical_range();
    for profile in [profile(), block_profile()] {
        let result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, profile).unwrap();
        // the jitter of the multistart moves a start by at most 0.9 typical ranges per coordinate
        let envelope = 0.9 * std::f64::consts::SQRT_2 * typical + 1e-6;
        assert!(result.positions[4].distance(initial[4]) <= envelope, "isolated agent moved to {:?}", result.positions[4]);
        let midpoint = |p: &[Point2]| Point2::new((p[2].x + p[3].x) / 2.0, (p[2].y + p[3].y) / 2.0);
        assert!(midpoint(&result.positions).distance(midpoint(&initial)) <= envelope, "unanchored pair moved away");
        assert!((result.positions[2].distance(result.positions[3]) - 300.0).abs() < 5.0);
    }
}

#[test]
fn responded_exchanges_without_admissible_samples_are_present_without_a_range() {
    // agent 3 answered agent 2 with inadmissible timings only, agent 4 answered agent 3 with no sample at all
    let mut af = problem(LocalizationMode::AnchorFree);
    af.observations = vec![
        calibrated(1, 2, false, &[300.0]),
        RangeObservation { from: 2, to: 3, to_is_anchor: false, tof_ns: vec![-1.0, f64::NAN, 250.0], calibrated_samples: Vec::new() },
        RangeObservation { from: 4, to: 3, to_is_anchor: false, tof_ns: Vec::new(), calibrated_samples: Vec::new() },
    ];
    let graph = af::Graph::new(&af, None, CalibrationModel::default(), &crate::PresenceProfile::default());
    assert_eq!(graph.ranges.len(), 1);
    // every pair is in one component, the three answered pairs are present and the other three failed
    assert_eq!(graph.pairs.len(), 6);
    assert_eq!(graph.pairs.iter().filter(|p| p.observed).count(), 3);
    // in anchor aided mode an answered anchor exchange joins the agent to the frame and activates the anchor
    let aac_problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2],
        anchors: vec![anchor(10, 0.0, 0.0, 0.0), anchor(11, 500.0, 0.0, 0.0)],
        observations: vec![
            calibrated(1, 10, true, &[300.0]),
            RangeObservation { from: 2, to: 11, to_is_anchor: true, tof_ns: Vec::new(), calibrated_samples: vec![CalibratedRangeSample::new(-5.0, 8.0)] },
        ],
        initial_positions: vec![Point2::default(); 2],
        mode: LocalizationMode::AnchorAided,
    };
    let graph = aac::Graph::new(&aac_problem, None, CalibrationModel::default(), &crate::PresenceProfile::default());
    let present_without_range = pair_set(&graph, |p| p.observed && p.range.is_none());
    assert_eq!(present_without_range, [(1, 3)].into_iter().collect());
    // agents 0 and 1, anchors 2 and 3
    assert_eq!(pair_set(&graph, |p| p.opportunity && !p.observed), [(0, 1), (0, 3), (1, 2)].into_iter().collect());
    assert!(graph.presence.is_some());
    let result = LocalizationEngine::new().solve_with_cooperative_profile(&aac_problem, profile()).unwrap();
    assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
}

#[test]
fn geometry_beyond_sane_magnitudes_is_rejected() {
    let engine = LocalizationEngine::new();
    let mut far = split_problem();
    far.anchors[1].position.x = 2.0e7;
    assert!(matches!(engine.solve_with_cooperative_profile(&far, profile()), Err(EngineError::AnchorOutOfRange(11))));
    let mut nan = split_problem();
    nan.anchors[0].height_cm = f64::NAN;
    assert!(matches!(engine.solve_with_cooperative_profile(&nan, profile()), Err(EngineError::NonFiniteAnchor(10))));
    let mut start = split_problem();
    start.initial_positions[2].y = f64::INFINITY;
    assert!(matches!(engine.solve_with_cooperative_profile(&start, profile()), Err(EngineError::InvalidProblem(_))));
    for height in [f64::NAN, f64::INFINITY, 5.0e7] {
        let mut heights = block_profile();
        heights.block_range.as_mut().unwrap().height_nodes[1] = height;
        assert!(matches!(engine.solve_with_cooperative_profile(&split_problem(), heights), Err(EngineError::InvalidProfile(_))), "{height}");
    }
    // samples beyond one kilometre, also after a steep calibration, answer the exchange without a range term
    let mut distant = split_problem();
    distant.observations.push(calibrated(2, 11, true, &[5.0e9]));
    distant.observations.push(RangeObservation { from: 2, to: 12, to_is_anchor: true, tof_ns: vec![30.0], calibrated_samples: Vec::new() });
    let mut steep = block_profile();
    steep.calibration = Some(CalibrationModel { range_slope_cm_per_ns: 1.0e12, ..CalibrationModel::default() });
    let graph = aac::Graph::new(&distant, steep.block_range.as_ref(), steep.calibration.unwrap(), &steep.presence);
    assert_eq!(pair_set(&graph, |p| p.observed && p.range.is_none()), [(1, 6), (1, 7)].into_iter().collect());
    let result = engine.solve_with_cooperative_profile(&distant, steep).unwrap();
    assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
}

fn raw(from: u32, to: u32, to_is_anchor: bool, tof_ns: &[f64]) -> RangeObservation {
    RangeObservation { from, to, to_is_anchor, tof_ns: tof_ns.to_vec(), calibrated_samples: Vec::new() }
}

/// Four agents and two anchors. Every link carries six raw packets, the two directions of the agent pairs and
/// two repeated observations of one anchor link included.
fn pooled_problem(split: bool) -> PreparedProblem {
    let packets: [(u32, u32, bool, [f64; 6]); 6] = [
        (1, 2, false, [9.1, 9.4, 8.8, 12.0, 9.0, 9.3]),
        (2, 3, false, [14.2, 14.0, 14.9, 13.8, 14.1, 18.0]),
        (3, 4, false, [11.0, 10.7, 11.2, 10.9, 11.4, 10.8]),
        (1, 4, false, [16.0, 15.6, 15.9, 16.3, 15.8, 16.1]),
        (1, 10, true, [7.0, 7.3, 6.9, 7.1, 9.5, 7.2]),
        (3, 11, true, [12.1, 12.4, 11.9, 12.0, 12.2, 12.3]),
    ];
    let mut observations = Vec::new();
    for (from, to, anchor, tof) in packets {
        if split {
            // agent pairs answer in both directions, anchor links in repeated observations, packets out of order
            let (first, second) = ([tof[5], tof[1], tof[3]], [tof[4], tof[0], tof[2]]);
            observations.push(raw(from, to, anchor, &first));
            if anchor {
                observations.push(raw(from, to, anchor, &second));
            } else {
                observations.push(raw(to, from, anchor, &second));
            }
        } else {
            observations.push(raw(from, to, anchor, &tof));
        }
    }
    if split {
        observations.reverse();
    }
    PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3, 4],
        anchors: vec![anchor(10, 0.0, 0.0, 150.0), anchor(11, 600.0, 400.0, 150.0)],
        observations,
        initial_positions: vec![Point2::default(); 4],
        mode: LocalizationMode::AnchorAided,
    }
}

#[test]
fn both_directions_and_repeated_observations_pool_into_one_link() {
    let block = block_profile();
    for split in [false, true] {
        let problem = pooled_problem(split);
        let graph = aac::Graph::new(&problem, block.block_range.as_ref(), CalibrationModel::default(), &block.presence);
        let links: Vec<&aac::Pair> = graph.pairs.iter().filter(|p| p.range.is_some()).collect();
        assert_eq!(links.len(), 6);
        // the median of the six pooled packets, converted first, and one table of all six
        let calibration = CalibrationModel::default();
        let pair = links.iter().find(|p| (p.left, p.right) == (0, 1)).unwrap();
        let mut distances: Vec<f64> = [9.1, 9.4, 8.8, 12.0, 9.0, 9.3].iter().map(|t| calibration.corrected_distance_cm(*t)).collect();
        distances.sort_by(f64::total_cmp);
        assert_eq!(pair.range.as_ref().unwrap().measured, (distances[2] + distances[3]) / 2.0);
        let scales: Vec<f64> = distances.iter().map(|d| calibration.scale_cm(*d)).collect();
        let model = crate::link::BlockModel::from(block.block_range.as_ref().unwrap());
        let expected = crate::link::LinkTable::from_packets(&distances, &scales, &model, crate::link::log_t_norm_for(model.nu));
        let table = pair.range.as_ref().unwrap().table.as_ref().unwrap();
        for d in [200.0, 270.0, 281.5, 300.0, 420.0] {
            assert_eq!(table.eval_smooth(d), expected.eval_smooth(d));
        }
        let mut af = pooled_problem(split);
        af.mode = LocalizationMode::AnchorFree;
        af.anchors.clear();
        af.observations.retain(|o| !o.to_is_anchor);
        let graph = af::Graph::new(&af, block.block_range.as_ref(), calibration, &block.presence);
        assert_eq!(graph.ranges.len(), 4);
        // agent only mode takes the median in time and converts it
        let first = graph.ranges.iter().find(|r| (graph.pairs[r.pair].left, graph.pairs[r.pair].right) == (0, 1)).unwrap();
        assert_eq!(first.measured, calibration.corrected_distance_cm((9.1 + 9.3) / 2.0));
        assert_eq!(first.table.as_ref().unwrap().eval_smooth(281.5), expected.eval_smooth(281.5));
    }
}

#[test]
fn pooled_links_do_not_depend_on_direction_or_observation_order() {
    // The geometric initializer keeps one median per observation, the disclosed start of the baselines too, so
    // the check starts every solve from the same layout and compares the likelihood and the optimizer alone.
    let engine = LocalizationEngine::new();
    let mut refined = block_profile();
    refined.block_range.as_mut().unwrap().refine_from_median = true;
    let start = |mut p: PreparedProblem| {
        p.initial_positions = vec![Point2::new(150.0, -100.0), Point2::new(350.0, 120.0), Point2::new(700.0, 90.0), Point2::new(600.0, -250.0)];
        p
    };
    let agent_only = |mut p: PreparedProblem| {
        p.mode = LocalizationMode::AnchorFree;
        p.anchors.clear();
        p.observations.retain(|o| !o.to_is_anchor);
        p
    };
    for mut profile in [profile(), block_profile(), refined] {
        profile.use_provided_start = Some(true);
        for convert in [|p| p, agent_only] {
            let mut whole = engine.solve_with_cooperative_profile(&convert(start(pooled_problem(false))), profile.clone()).unwrap();
            let mut split = engine.solve_with_cooperative_profile(&convert(start(pooled_problem(true))), profile.clone()).unwrap();
            whole.metrics.elapsed_micros = 0;
            split.metrics.elapsed_micros = 0;
            assert_eq!(whole, split);
        }
    }
}

#[test]
fn each_agent_marginalizes_one_height_over_all_its_anchor_terms() {
    // anchors at three heights, agent 1 ranges two of them, agent 2 the third, and three anchor attempts failed
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2],
        anchors: vec![anchor(10, 0.0, 0.0, 40.0), anchor(11, 500.0, 0.0, 220.0), anchor(12, 250.0, 450.0, 380.0)],
        observations: vec![
            calibrated(1, 10, true, &[230.0, 236.0, 229.0]),
            calibrated(1, 11, true, &[330.0, 341.0]),
            calibrated(2, 12, true, &[260.0, 255.0, 270.0, 262.0]),
            calibrated(1, 2, false, &[300.0, 310.0]),
        ],
        initial_positions: vec![Point2::default(); 2],
        mode: LocalizationMode::AnchorAided,
    };
    let mut profile = block_profile();
    let block = profile.block_range.as_mut().unwrap();
    block.height_nodes = vec![60.0, 120.0, 200.0];
    block.height_weights = vec![1.0, 2.0, 1.0];
    let law = crate::PresenceProfile::default();
    let graph = aac::Graph::new(&problem, profile.block_range.as_ref(), CalibrationModel::default(), &law);
    assert_eq!(graph.anchor_links, vec![vec![1, 2, 3], vec![4, 5, 6]]);
    assert!(graph.pairs.iter().all(|p| p.opportunity));
    let presence = graph.presence.unwrap();
    let mut objective = aac::Objective { graph: &graph, profile: &profile.range_error };
    let parameters = vec![120.0, 180.0, 330.0, 290.0];
    let value = objective.value_gradient(&parameters, &mut vec![0.0; parameters.len()]);
    // the explicit objective takes the agent pair on its horizontal distance and each agent's anchor terms summed per
    // height node before the weighted log sum exp
    let point = |agent: usize| (parameters[2 * agent], parameters[2 * agent + 1]);
    let (p, q) = (point(0), point(1));
    let separation = (p.0 - q.0).hypot(p.1 - q.1);
    let peer = &graph.pairs[0];
    let mut expected = -peer.range.as_ref().unwrap().table.as_ref().unwrap().eval_smooth(separation).0
        - presence.log_probability(separation, true).0;
    // the anchor terms marginalized per agent, and per link as the previous model did
    let (mut per_agent, mut per_link) = (0.0, 0.0);
    let weights = [0.25_f64, 0.5, 0.25];
    let marginal = |nodes: &[f64; 3]| nodes.iter().zip(weights).map(|(t, w)| w * t.exp()).sum::<f64>().ln();
    for agent in 0..2 {
        let (x, y) = point(agent);
        let mut agent_nodes = [0.0; 3];
        for &k in &graph.anchor_links[agent] {
            let pair = &graph.pairs[k];
            let anchor = graph.anchors[pair.right - 2];
            let horizontal = (x - anchor[0]).hypot(y - anchor[1]);
            let mut link_nodes = [0.0; 3];
            for (node, z) in [60.0, 120.0, 200.0].iter().enumerate() {
                let slant = horizontal.hypot(z - anchor[2]);
                let mut term = presence.log_probability(slant, pair.observed).0;
                if let Some(link) = &pair.range {
                    term += link.table.as_ref().unwrap().eval_smooth(slant).0;
                }
                agent_nodes[node] += term;
                link_nodes[node] = term;
            }
            per_link -= marginal(&link_nodes);
        }
        per_agent -= marginal(&agent_nodes);
    }
    expected += per_agent;
    assert!((value - expected).abs() < 1e-9 * expected.abs().max(1.0), "{value} {expected}");
    // the per link marginal of the previous model is a different likelihood
    assert!((per_agent - per_link).abs() > 1e-3, "{per_agent} {per_link}");
    check_gradient(&mut objective, &parameters);
    // the gradient also holds where one agent sits on an anchor and far outside the layout
    check_gradient(&mut objective, &[0.5, 0.5, 2400.0, -1800.0]);
}

/// Three agents on the floor of four ceiling anchors. Agents 1 and 2 range every anchor, agent 3 only two of them,
/// and the pair 2 and 3 carries a range 440 cm short that distorts the initializer's embedding.
fn lateration_problem() -> (PreparedProblem, [[f64; 2]; 3]) {
    let truth = [[200.0, 200.0], [800.0, 250.0], [500.0, 650.0]];
    let anchors = [anchor(10, 0.0, 0.0, 250.0), anchor(11, 1000.0, 0.0, 250.0), anchor(12, 0.0, 800.0, 250.0), anchor(13, 1000.0, 800.0, 250.0)];
    let mut observations = Vec::new();
    for (i, agent) in truth.iter().enumerate() {
        for anchor in anchors.iter().filter(|a| i < 2 || a.id < 12) {
            let d = (agent[0] - anchor.position.x).hypot(agent[1] - anchor.position.y).hypot(250.0);
            observations.push(calibrated(i as u32 + 1, anchor.id, true, &[d - 1.0, d, d + 1.5]));
        }
    }
    let separation = |i: usize, j: usize| (truth[i][0] - truth[j][0]).hypot(truth[i][1] - truth[j][1]);
    observations.push(calibrated(1, 2, false, &[separation(0, 1)]));
    observations.push(calibrated(2, 3, false, &[separation(1, 2) - 440.0]));
    observations.push(calibrated(1, 3, false, &[separation(0, 2)]));
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3],
        anchors: anchors.to_vec(),
        observations,
        initial_positions: vec![Point2::default(); 3],
        mode: LocalizationMode::AnchorAided,
    };
    (problem, truth)
}

#[test]
fn agents_with_three_anchor_links_get_a_lateration_start() {
    let (problem, truth) = lateration_problem();
    let config = SolveConfig { max_time_micros: 0, aggregation: ObservationAggregation::Median, ..SolveConfig::production() };
    let (layout, laterations) = crate::initializer::layout_and_laterations(&problem, &config, CalibrationModel::default()).unwrap();
    assert_eq!(layout, crate::initializer::initial_layout_with(&problem, &config, CalibrationModel::default()).unwrap());
    // the two agents with four anchor links are laterated within a centimetre, the agent with two is not
    for (lateration, truth) in laterations.iter().zip(&truth).take(2) {
        let lateration = lateration.unwrap();
        assert!((lateration.x - truth[0]).hypot(lateration.y - truth[1]) < 1.0, "{lateration:?}");
    }
    assert!(laterations[2].is_none());
    // the rigid alignment of the distorted embedding moves them away from their own laterations
    assert!(layout[1].distance(laterations[1].unwrap()) > 20.0, "{:?} {:?}", layout[1], laterations[1]);
    let start = lateration_start(&layout, &laterations).unwrap();
    assert_eq!((start[0], start[1], start[2]), (laterations[0].unwrap(), laterations[1].unwrap(), layout[2]));
    assert!(lateration_start(&start, &laterations).is_none());
    assert!(lateration_start(&layout, &[None, None, None]).is_none());
}

#[test]
fn the_multistart_counts_the_lateration_start_and_keeps_the_lowest_objective() {
    let few_anchors = problem(LocalizationMode::AnchorAided);
    let (problem, _) = lateration_problem();
    let engine = LocalizationEngine::new();
    // the geometric start of the median residual density and the peak start of the block range model, in one stage
    // and in both of two stages, get one more start at the laterations
    let mut refined = block_profile();
    refined.block_range.as_mut().unwrap().refine_from_median = true;
    for profile in [profile(), block_profile(), refined] {
        assert_eq!(engine.solve_with_cooperative_profile(&problem, profile.clone()).unwrap().metrics.starts, 33);
        // the ablation that starts from the request's positions has no initializer and no lateration start
        let mut provided = profile.clone();
        provided.use_provided_start = Some(true);
        assert_eq!(engine.solve_with_cooperative_profile(&problem, provided).unwrap().metrics.starts, 32);
        // without an agent of three anchor links there is nothing to add
        assert_eq!(engine.solve_with_cooperative_profile(&few_anchors, profile).unwrap().metrics.starts, 32);
    }
    // the extra start competes on the objective like every other start
    let config = SolveConfig { max_time_micros: 0, aggregation: ObservationAggregation::Median, ..SolveConfig::production() };
    let graph = aac::Graph::new(&problem, None, CalibrationModel::default(), &crate::PresenceProfile::default());
    let range_error = profile().range_error;
    let (layout, laterations) = crate::initializer::layout_and_laterations(&problem, &config, CalibrationModel::default()).unwrap();
    let mirrored: Vec<Point2> = layout.iter().map(|p| Point2::new(p.x, 800.0 - p.y)).collect();
    let lateration = lateration_start(&layout, &laterations).unwrap();
    let run = |starts: Vec<&[Point2]>| {
        let runs = starts.into_iter().map(flatten).collect();
        optimize_layout(&mut aac::Objective { graph: &graph, profile: &range_error }, runs, 3, 1, &config).unwrap()
    };
    let (alone, lateration_alone) = (run(vec![&mirrored]), run(vec![&lateration]));
    let both = run(vec![&mirrored, &lateration]);
    assert_eq!(both.1.starts, 2);
    let lowest = if lateration_alone.1.final_objective < alone.1.final_objective { &lateration_alone } else { &alone };
    assert_eq!((both.0.clone(), both.1.final_objective), (lowest.0.clone(), lowest.1.final_objective));
}

/// The settings of the cooperative solver's optimizer and initializer.
fn cooperative_config() -> SolveConfig {
    SolveConfig { max_time_micros: 0, aggregation: ObservationAggregation::Median, ..SolveConfig::production() }
}

/// The bits of every coordinate of a layout.
fn bits(layout: &[Point2]) -> Vec<[u64; 2]> {
    layout.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect()
}

/// Four ceiling anchors, a fifth at the place of the third, and three agents on the floor. Agent 1 ranges the first
/// three anchors and the fifth, whose packets read 6 cm longer than the third's, and its declared attempt at the
/// fourth failed. Agent 2 ranges the second and the third anchor, and agent 3 ranges only agents 1 and 2. `ids` labels
/// the five anchors in that order.
fn peak_problem(ids: [u32; 5]) -> (PreparedProblem, [Point2; 3]) {
    let places = [(0.0, 0.0), (400.0, 0.0), (0.0, 300.0), (1200.0, 900.0), (0.0, 300.0)];
    let truth = [Point2::new(150.0, 100.0), Point2::new(300.0, 200.0), Point2::new(250.0, 350.0)];
    let slant = |agent: usize, k: usize| (truth[agent].x - places[k].0).hypot(truth[agent].y - places[k].1).hypot(250.0);
    let mut observations = Vec::new();
    for (agent, ranged) in [(0, vec![0, 1, 2, 4]), (1, vec![1, 2])] {
        for k in ranged {
            let d = slant(agent, k) + if k == 4 { 6.0 } else { 0.0 };
            observations.push(calibrated(agent as u32 + 1, ids[k], true, &[d - 1.5, d, d + 2.0]));
        }
    }
    observations.push(calibrated(1, 3, false, &[truth[0].distance(truth[2])]));
    observations.push(calibrated(2, 3, false, &[truth[1].distance(truth[2])]));
    let problem = PreparedProblem {
        attempts: Some(vec![crate::LinkAttempt { from: 1, to: ids[3], to_is_anchor: true }]),
        agent_ids: vec![1, 2, 3],
        anchors: places.iter().zip(ids).map(|((x, y), id)| anchor(id, *x, *y, 250.0)).collect(),
        observations,
        initial_positions: vec![Point2::default(); 3],
        mode: LocalizationMode::AnchorAided,
    };
    (problem, truth)
}

/// The anchor only estimator's value of every cell of its grid over `area` at step `h`, row by row, with the cell
/// centres of both axes. A transcription of Grid::new, Grid::slant_span, range_support without its prune, and the
/// presence terms and the height marginal of agent_support in anchor_only.rs.
fn anchor_only_grid(
    links: &[peak::AnchorLink<'_>],
    present: &[[f64; 3]],
    missing: &[[f64; 3]],
    heights: &[(f64, f64)],
    law: Option<crate::presence::ResolvedPresence>,
    area: &crate::PlanarArea,
    h: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    #[derive(Clone, Copy)]
    struct Anchor3 {
        x: f64,
        y: f64,
        z: f64,
    }
    fn log_add_exp(a: f64, b: f64) -> f64 {
        if a == f64::NEG_INFINITY {
            return b;
        }
        if b == f64::NEG_INFINITY {
            return a;
        }
        let m = a.max(b);
        m + ((a - m).exp() + (b - m).exp()).ln()
    }
    fn detection_loglik(dists: &[f64], present: &[bool], law: &crate::presence::ResolvedPresence) -> f64 {
        dists.iter().zip(present).map(|(d, p)| law.log_probability(*d, *p).0).sum()
    }
    let anchor3 = |a: &[f64; 3]| Anchor3 { x: a[0], y: a[1], z: a[2] };
    let mut xs = Vec::new();
    let mut x = area.minimum_x_cm + h / 2.0;
    while x < area.maximum_x_cm {
        xs.push(x);
        x += h;
    }
    let mut ys = Vec::new();
    let mut y = area.minimum_y_cm + h / 2.0;
    while y < area.maximum_y_cm {
        ys.push(y);
        y += h;
    }
    if xs.is_empty() {
        xs.push((area.minimum_x_cm + area.maximum_x_cm) / 2.0);
    }
    if ys.is_empty() {
        ys.push((area.minimum_y_cm + area.maximum_y_cm) / 2.0);
    }
    let slant_span = |anchor: &Anchor3, heights: &[f64]| {
        let (x_low, x_high) = (xs[0], xs[xs.len() - 1]);
        let (y_low, y_high) = (ys[0], ys[ys.len() - 1]);
        let near_x = anchor.x.clamp(x_low, x_high) - anchor.x;
        let near_y = anchor.y.clamp(y_low, y_high) - anchor.y;
        let far_x = (x_low - anchor.x).abs().max((x_high - anchor.x).abs());
        let far_y = (y_low - anchor.y).abs().max((y_high - anchor.y).abs());
        let dz: Vec<f64> = heights.iter().map(|z| (z - anchor.z).abs()).collect();
        let near_z = dz.iter().copied().fold(f64::INFINITY, f64::min);
        let far_z = dz.iter().copied().fold(0.0, f64::max);
        let nearest = (near_x * near_x + near_y * near_y + near_z * near_z).sqrt();
        let farthest = (far_x * far_x + far_y * far_y + far_z * far_z).sqrt();
        ((nearest - 1.0).max(0.0), farthest + 1.0)
    };
    let zs: Vec<f64> = heights.iter().map(|(z, _)| *z).collect();
    let zw: Vec<f64> = heights.iter().map(|(_, w)| *w).collect();
    let count = xs.len() * ys.len();
    let mut acc: Vec<Vec<f64>> = vec![vec![0.0; count]; zs.len()];
    for (anchor, table) in links {
        let anchor = anchor3(anchor);
        let (nearest, farthest) = slant_span(&anchor, &zs);
        let table = table.far(nearest, farthest);
        let dz2: Vec<f64> = zs.iter().map(|z| (z - anchor.z) * (z - anchor.z)).collect();
        let columns = xs.len();
        for (r, y) in ys.iter().enumerate() {
            let dy = y - anchor.y;
            for (i, x) in xs.iter().enumerate() {
                let dx = x - anchor.x;
                let horizontal = dx * dx + dy * dy;
                for (row, dz2) in acc.iter_mut().zip(&dz2) {
                    row[r * columns + i] += table.eval((horizontal + dz2).sqrt());
                }
            }
        }
    }
    let mut terms = acc;
    if let Some(law) = law {
        let opportunities: Vec<(Anchor3, bool)> =
            present.iter().map(|a| (anchor3(a), true)).chain(missing.iter().map(|a| (anchor3(a), false))).collect();
        let outcomes: Vec<bool> = opportunities.iter().map(|(_, p)| *p).collect();
        let mut dists = vec![0.0; opportunities.len()];
        for (zi, z) in zs.iter().enumerate() {
            for c in 0..count {
                let (x, y) = (xs[c % xs.len()], ys[c / xs.len()]);
                for (k, (a, _)) in opportunities.iter().enumerate() {
                    let dx = x - a.x;
                    let dy = y - a.y;
                    let dz = z - a.z;
                    dists[k] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
                terms[zi][c] += detection_loglik(&dists, &outcomes, &law);
            }
        }
    }
    let values = (0..count).map(|j| (0..zs.len()).map(|zi| terms[zi][j] + zw[zi]).fold(f64::NEG_INFINITY, log_add_exp)).collect();
    (xs, ys, values)
}

/// The centre of the first cell of largest value in the grid's order, None without a value above minus infinity.
fn first_largest(xs: &[f64], ys: &[f64], values: &[f64]) -> Option<Point2> {
    let mut best: Option<(f64, usize)> = None;
    for (cell, value) in values.iter().enumerate() {
        if *value > best.map_or(f64::NEG_INFINITY, |(top, _)| top) {
            best = Some((*value, cell));
        }
    }
    best.map(|(_, cell)| Point2::new(xs[cell % xs.len()], ys[cell / xs.len()]))
}

#[test]
fn the_peak_search_keeps_the_anchor_only_arithmetic_on_every_cell() {
    let block = block_profile();
    let (problem, truth) = peak_problem([10, 11, 12, 13, 14]);
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), CalibrationModel::default());
    let (links, present, missing) = peak::agent_inputs(&graph, 0);
    assert_eq!((links.len(), present.len(), missing.len()), (4, 4, 1));
    let gaussian = block.presence.resolve(&graph.successful);
    let logistic = crate::PresenceProfile { shape: crate::PresenceShape::Logistic, ..block.presence }.resolve(&graph.successful);
    assert!(gaussian.is_some() && logistic.is_some());
    // a coarse grid over the whole area, and the search's own step around the agent
    let area = peak::peak_area(&links);
    let around = crate::PlanarArea {
        minimum_x_cm: truth[0].x - 60.0,
        maximum_x_cm: truth[0].x + 60.0,
        minimum_y_cm: truth[0].y - 50.0,
        maximum_y_cm: truth[0].y + 50.0,
    };
    for (region, step) in [(area, 4.0), (around, peak::PEAK_STEP_CM)] {
        for law in [None, gaussian, logistic] {
            let (xs, ys, expected) = anchor_only_grid(&links, &present, &missing, &graph.heights, law, &region, step);
            let mut posterior = peak::CellPosterior::new(&links, &present, &missing, &graph.heights, law.as_ref(), &region, step);
            let axis_bits = |axis: &[f64]| axis.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!((axis_bits(&posterior.xs), axis_bits(&posterior.ys)), (axis_bits(&xs), axis_bits(&ys)));
            let mut row = vec![0.0; xs.len()];
            for r in 0..ys.len() {
                posterior.row(r, &mut row);
                for (i, value) in row.iter().enumerate() {
                    let reference = expected[r * xs.len() + i];
                    assert_eq!(value.to_bits(), reference.to_bits(), "cell ({}, {}) under {law:?}: {value} {reference}", xs[i], ys[r]);
                }
            }
            let peak = peak::exhaustive_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &region, step);
            assert!(peak.is_some());
            assert_eq!(peak, first_largest(&xs, &ys, &expected));
            // the branch and bound the start calls finds the same cell
            assert_eq!(crate::posterior::posterior_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &region, step), peak);
        }
    }
}

#[test]
fn the_peak_is_the_first_cell_of_largest_value_over_the_linked_anchors() {
    let block = block_profile();
    let (problem, truth) = peak_problem([10, 11, 12, 13, 14]);
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), CalibrationModel::default());
    // agents 1 and 2 range anchors, agent 3 does not
    assert!(peak::agent_inputs(&graph, 2).0.is_empty());
    for law in [None, block.presence.resolve(&graph.successful)] {
        let mut peaks = Vec::new();
        for agent in 0..2 {
            let (links, present, missing) = peak::agent_inputs(&graph, agent);
            // the area is the smallest box that holds the square of half side the table's upper end around every
            // linked anchor
            let area = peak::peak_area(&links);
            let squares: Vec<[f64; 4]> = links
                .iter()
                .map(|(anchor, table)| {
                    let half = table.range().1;
                    [anchor[0] - half, anchor[0] + half, anchor[1] - half, anchor[1] + half]
                })
                .collect();
            let sides = [area.minimum_x_cm, area.maximum_x_cm, area.minimum_y_cm, area.maximum_y_cm];
            for (side, bound) in sides.iter().enumerate() {
                let reached = squares.iter().map(|square| square[side]);
                let extreme = if side % 2 == 0 { reached.fold(f64::INFINITY, f64::min) } else { reached.fold(f64::NEG_INFINITY, f64::max) };
                assert_eq!(*bound, extreme);
            }
            let peak = peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &area);
            let (xs, ys, values) = anchor_only_grid(&links, &present, &missing, &graph.heights, law, &area, peak::PEAK_STEP_CM);
            assert_eq!(peak, first_largest(&xs, &ys, &values));
            peaks.push(peak);
        }
        // agent 1, with four links to three places, peaks next to its truth and is the only agent whose peak replaces
        // its lateration, while agent 2 ranges two anchors and keeps its lateration
        assert!(peaks[0].unwrap().distance(truth[0]) < 5.0, "{:?}", peaks[0]);
        assert!(peaks[1].is_some());
        assert_eq!(peak::peak_overrides(&graph, law.as_ref()), vec![peaks[0], None, None]);
    }
}

#[test]
fn equal_cells_leave_the_peak_at_the_first_in_the_grid_order() {
    // one agent ranging one anchor, whose ring posterior repeats its values on the cells mirrored across the anchor
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1],
        anchors: vec![anchor(10, 0.0, 0.0, 250.0)],
        observations: vec![calibrated(1, 10, true, &[300.0, 301.0, 302.5])],
        initial_positions: vec![Point2::default()],
        mode: LocalizationMode::AnchorAided,
    };
    let block = block_profile();
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), CalibrationModel::default());
    let law = block.presence.resolve(&graph.successful);
    let (links, present, missing) = peak::agent_inputs(&graph, 0);
    let (xs, ys, values) = anchor_only_grid(&links, &present, &missing, &graph.heights, law, &peak::peak_area(&links), peak::PEAK_STEP_CM);
    let top = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let equal: Vec<usize> = (0..values.len()).filter(|cell| values[*cell] == top).collect();
    assert!(equal.len() > 1, "{} cells of the largest value", equal.len());
    let first = Point2::new(xs[equal[0] % xs.len()], ys[equal[0] / xs.len()]);
    let area = peak::peak_area(&links);
    assert_eq!(peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &area), Some(first));
    // a ring leaves the position unidentified, so the peak does not replace the agent's lateration
    assert_eq!(peak::peak_overrides(&graph, law.as_ref()), vec![None]);
}

#[test]
fn the_peaks_do_not_depend_on_the_anchor_order() {
    let block = block_profile();
    let calibration = CalibrationModel::default();
    let (first, _) = peak_problem([10, 11, 12, 13, 14]);
    // the anchors relabeled and listed in reverse, and the observations in reverse
    let (mut second, _) = peak_problem([24, 23, 22, 21, 20]);
    second.anchors.reverse();
    second.observations.reverse();
    let graphs = [&first, &second].map(|problem| aac::Graph::build(problem, block.block_range.as_ref(), calibration));
    let law = block.presence.resolve(&graphs[0].successful);
    assert_eq!(law, block.presence.resolve(&graphs[1].successful));
    // the pairs follow the anchor order of the request, the inputs of every agent do not
    let inputs = |graph: &aac::Graph, agent: usize| {
        let (links, present, missing) = peak::agent_inputs(graph, agent);
        let links: Vec<([f64; 3], (f64, f64), u64)> = links.iter().map(|(anchor, table)| (*anchor, table.range(), table.eval(320.0).to_bits())).collect();
        (links, present, missing)
    };
    for agent in 0..3 {
        assert_eq!(inputs(&graphs[0], agent), inputs(&graphs[1], agent));
    }
    // the two links at one place follow their packets, the shorter readings first
    let (links, ..) = inputs(&graphs[0], 0);
    assert!(links[1].0 == links[2].0 && links[1].1 .1 < links[2].1 .1, "{links:?}");
    // so every cell keeps its value and the peaks and the start stay the same
    let (agent_links, present, missing) = peak::agent_inputs(&graphs[1], 0);
    let (reference_links, ..) = peak::agent_inputs(&graphs[0], 0);
    let area = peak::peak_area(&agent_links);
    let values = |links: &[peak::AnchorLink<'_>]| {
        let mut posterior = peak::CellPosterior::new(links, &present, &missing, &graphs[0].heights, law.as_ref(), &area, 5.0);
        let mut row = vec![0.0; posterior.xs.len()];
        (0..posterior.ys.len())
            .flat_map(|r| {
                posterior.row(r, &mut row);
                row.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            })
            .collect::<Vec<u64>>()
    };
    assert_eq!(values(&agent_links), values(&reference_links));
    for law in [None, law] {
        for agent in 0..2 {
            let search = |graph: &aac::Graph| {
                let (links, present, missing) = peak::agent_inputs(graph, agent);
                peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &peak::peak_area(&links))
            };
            assert_eq!(search(&graphs[0]), search(&graphs[1]));
        }
        assert_eq!(peak::peak_overrides(&graphs[0], law.as_ref()), peak::peak_overrides(&graphs[1], law.as_ref()));
    }
    let config = cooperative_config();
    let starts = [(&first, &graphs[0]), (&second, &graphs[1])]
        .map(|(problem, graph)| peak::start(problem, &config, calibration, graph, &block.presence).unwrap());
    assert_eq!(starts[0], starts[1]);
}

#[test]
fn a_two_anchor_agent_keeps_its_lateration_as_a_target_of_the_alignment() {
    let config = cooperative_config();
    let calibration = CalibrationModel::default();
    let block = block_profile();
    let with = |problem: &PreparedProblem, overrides: &[Option<Point2>]| {
        crate::initializer::layout_with_overrides(problem, &config, calibration, overrides).unwrap()
    };
    // the lateration problem and a fourth agent that ranges the other three and no anchor
    let (mut problem, truth) = lateration_problem();
    let fourth = Point2::new(450.0, 380.0);
    for (i, t) in truth.iter().enumerate() {
        problem.observations.push(calibrated(4, i as u32 + 1, false, &[fourth.distance(Point2::new(t[0], t[1]))]));
    }
    problem.agent_ids.push(4);
    problem.initial_positions.push(Point2::default());
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), calibration);
    let law = block.presence.resolve(&graph.successful);
    // agents 1 and 2 range four anchors and their peaks replace their laterations, agent 3 ranges two and keeps its
    // lateration although its posterior has a peak, and agent 4 ranges no anchor
    let overrides = peak::peak_overrides(&graph, law.as_ref());
    assert!(overrides[..2].iter().all(Option::is_some) && overrides[2..].iter().all(Option::is_none), "{overrides:?}");
    let (links, present, missing) = peak::agent_inputs(&graph, 2);
    assert_eq!(links.len(), 2);
    let mirror = peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &peak::peak_area(&links)).unwrap();
    let start = with(&problem, &overrides);
    let peaks: Vec<Point2> = overrides[..2].iter().map(|p| p.unwrap()).collect();
    assert_eq!(bits(&start[..2]), bits(&peaks));
    // the peak of agent 3's two anchors lies at the mirror image of its truth across the anchors' line, while the
    // alignment, which keeps its lateration as a target, leaves the agent on the side of its truth
    let third = Point2::new(truth[2][0], truth[2][1]);
    assert!(mirror.y < 0.0 && mirror.distance(Point2::new(third.x, -third.y)) < 30.0, "{mirror:?}");
    assert!(start[2].y > 0.0 && start[2].distance(third) < start[2].distance(mirror), "{:?}", start[2]);
    // with one anchored agent, its peak replaces the one target of its component and moves its peer by the same
    // offset, while the component without an anchor and the isolated agent keep the initializer's places
    let problem = split_problem();
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), calibration);
    let overrides = peak::peak_overrides(&graph, block.presence.resolve(&graph.successful).as_ref());
    assert!(overrides[0].is_some() && overrides[1..].iter().all(Option::is_none), "{overrides:?}");
    let start = with(&problem, &overrides);
    let initial = with(&problem, &[None; 5]);
    assert_eq!(bits(&initial), bits(&crate::initializer::initial_layout_with(&problem, &config, calibration).unwrap()));
    assert_eq!(start[0], overrides[0].unwrap());
    let moved = Point2::new(initial[1].x - initial[0].x + start[0].x, initial[1].y - initial[0].y + start[0].y);
    assert!(start[1].distance(moved) < 1e-9, "{:?} {moved:?}", start[1]);
    assert_eq!(bits(&start[2..]), bits(&initial[2..]));
}

/// Agent 3 of the lateration problem ranges two anchors, and the peak of its posterior is the mirror image of its truth.
/// Started there, one stage of the block objective with 64 starts ended with the agent 12.8 m from its truth at an
/// objective of 139.79. Its peak does not replace its lateration, and the solve reaches the geometric start's optimum.
#[test]
fn a_two_anchor_mirror_does_not_start_the_single_stage_solve() {
    let (problem, truth) = lateration_problem();
    let single = CooperativeProfile { starts: 64, ..block_profile() };
    let result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, single.clone()).unwrap();
    for (position, truth) in result.positions.iter().zip(&truth) {
        assert!(position.distance(Point2::new(truth[0], truth[1])) < 1.0, "{:?}", result.positions);
    }
    let geometric = geometric_solve(&problem, &single).metrics.final_objective;
    assert!((result.metrics.final_objective - geometric).abs() < 1e-6 * geometric, "{} {geometric}", result.metrics.final_objective);
}

#[test]
fn only_three_anchor_places_let_a_peak_replace_a_lateration() {
    // anchors 12 and 13 share one place. Agent 1 ranges three places, agent 2 ranges three anchors at two places,
    // agent 3 one anchor and agent 4 two, all four joined by ranges between them, and agent 5 ranges two anchors alone
    let places: [(u32, f64, f64); 4] = [(10, 0.0, 0.0), (11, 600.0, 0.0), (12, 0.0, 500.0), (13, 0.0, 500.0)];
    let truth: [(f64, f64); 5] = [(200.0, 150.0), (350.0, 300.0), (450.0, 100.0), (150.0, 380.0), (500.0, 400.0)];
    let ranged: [&[u32]; 5] = [&[10, 11, 12], &[10, 12, 13], &[11], &[10, 11], &[11, 12]];
    let mut observations = Vec::new();
    for (agent, ids) in ranged.iter().enumerate() {
        for id in *ids {
            let (_, x, y) = places.iter().find(|p| p.0 == *id).unwrap();
            let d = (truth[agent].0 - x).hypot(truth[agent].1 - y).hypot(250.0);
            observations.push(calibrated(agent as u32 + 1, *id, true, &[d - 1.0, d, d + 1.5]));
        }
    }
    for (a, b) in [(0, 1), (1, 2), (0, 3), (2, 3)] {
        let d = (truth[a].0 - truth[b].0).hypot(truth[a].1 - truth[b].1);
        observations.push(calibrated(a as u32 + 1, b as u32 + 1, false, &[d]));
    }
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3, 4, 5],
        anchors: places.iter().map(|(id, x, y)| anchor(*id, *x, *y, 250.0)).collect(),
        observations,
        initial_positions: vec![Point2::default(); 5],
        mode: LocalizationMode::AnchorAided,
    };
    let block = block_profile();
    let calibration = CalibrationModel::default();
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), calibration);
    let law = block.presence.resolve(&graph.successful);
    // every anchored agent's posterior has a peak, and the count of anchor places alone keeps the peaks of the four
    // with fewer places out
    for (agent, links) in [3, 3, 1, 2, 2].into_iter().enumerate() {
        let (inputs, present, missing) = peak::agent_inputs(&graph, agent);
        assert_eq!(inputs.len(), links);
        assert!(peak::agent_peak(&inputs, &present, &missing, &graph.heights, law.as_ref(), &peak::peak_area(&inputs)).is_some());
    }
    let overrides = peak::peak_overrides(&graph, law.as_ref());
    assert!(overrides[0].is_some() && overrides[1..].iter().all(Option::is_none), "{overrides:?}");
    // the start is the initializer's layout with that one override, which places agent 1 at its peak, while agent 5,
    // alone with two anchors, stays where the initializer puts it
    let config = cooperative_config();
    let with = |overrides: &[Option<Point2>]| crate::initializer::layout_with_overrides(&problem, &config, calibration, overrides).unwrap();
    let (start, start_law) = peak::start(&problem, &config, calibration, &graph, &block.presence).unwrap();
    assert_eq!(start_law, law);
    assert_eq!(bits(&start), bits(&with(&overrides)));
    assert_eq!(start[0], overrides[0].unwrap());
    assert_eq!(bits(&start[4..]), bits(&with(&[None; 5])[4..]));
}

/// Agent 1 ranges the three anchors of a room at the edge of the supported coordinates, a few metres inside 100 km, and
/// agent 2 those of a room at the origin, and no range joins them. The squares around agent 1's anchors, of half side
/// the upper end of each link's table, reach beyond 100 km, where the peak grid refuses the area, so its posterior has
/// no peak. Its lateration then stays the target of its component's alignment, and it starts where the initializer
/// puts it, at its own lateration, while agent 2 starts at its peak.
#[test]
fn an_agent_whose_posterior_has_no_peak_starts_at_its_lateration() {
    let edge = 9_999_000.0;
    let places: [(u32, f64, f64); 6] = [(10, edge, 0.0), (11, edge + 600.0, 0.0), (12, edge, 500.0), (20, 0.0, 0.0), (21, 600.0, 0.0), (22, 0.0, 500.0)];
    let truth = [Point2::new(edge + 250.0, 200.0), Point2::new(200.0, 150.0)];
    let mut observations = Vec::new();
    for (agent, ids) in [(0usize, [10u32, 11, 12]), (1, [20, 21, 22])] {
        for id in ids {
            let (_, x, y) = places.iter().find(|p| p.0 == id).unwrap();
            let d = (truth[agent].x - x).hypot(truth[agent].y - y).hypot(250.0);
            observations.push(calibrated(agent as u32 + 1, id, true, &[d - 1.5, d, d + 2.0]));
        }
    }
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2],
        anchors: places.iter().map(|(id, x, y)| anchor(*id, *x, *y, 250.0)).collect(),
        observations,
        initial_positions: vec![Point2::default(); 2],
        mode: LocalizationMode::AnchorAided,
    };
    let block = block_profile();
    let calibration = CalibrationModel::default();
    let config = cooperative_config();
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), calibration);
    let law = block.presence.resolve(&graph.successful);
    let (links, present, missing) = peak::agent_inputs(&graph, 0);
    let area = peak::peak_area(&links);
    assert!(links.len() == 3 && area.maximum_x_cm > 1.0e7, "{area:?}");
    for law in [None, law] {
        assert_eq!(peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &area), None);
        let overrides = peak::peak_overrides(&graph, law.as_ref());
        assert!(overrides[0].is_none() && overrides[1].is_some(), "{overrides:?}");
    }
    let (start, start_law) = peak::start(&problem, &config, calibration, &graph, &block.presence).unwrap();
    assert_eq!(start_law, law);
    let overrides = peak::peak_overrides(&graph, law.as_ref());
    assert_eq!(bits(&start), bits(&crate::initializer::layout_with_overrides(&problem, &config, calibration, &overrides).unwrap()));
    // agent 1 sits at its own lateration, as in the initializer's layout, and agent 2 at its peak
    let (initial, laterations) = crate::initializer::layout_and_laterations(&problem, &config, calibration).unwrap();
    assert_eq!(bits(&start[..1]), bits(&initial[..1]));
    assert_eq!(bits(&start[..1]), bits(&[laterations[0].unwrap()]));
    assert!(start[0].distance(truth[0]) < 50.0, "{:?}", start[0]);
    assert_eq!(start[1], overrides[1].unwrap());
    assert!(start[1].distance(truth[1]) < 5.0, "{:?}", start[1]);
    let result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, block).unwrap();
    assert!(result.positions[0].distance(truth[0]) < 50.0 && result.positions[1].distance(truth[1]) < 5.0, "{:?}", result.positions);
}

#[test]
fn the_smallest_width_is_that_of_the_narrowest_strip_holding_the_places() {
    let width = |points: &[[f64; 2]]| {
        let mut places = points.to_vec();
        places.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
        places.dedup();
        peak::minimum_width(&places)
    };
    // no place, one place, two places and places on one line leave no width
    for points in [&[][..], &[[40.0, 70.0]], &[[0.0, 0.0], [600.0, 0.0]], &[[0.0, 0.0], [300.0, 150.0], [900.0, 450.0], [600.0, 300.0]]] {
        assert_eq!(width(points), 0.0, "{points:?}");
    }
    // a triangle is as wide as its least height, the one onto its longest side
    let triangle = [[0.0, 0.0], [600.0, 0.0], [0.0, 500.0]];
    assert!((width(&triangle) - 600.0 * 500.0 / 600.0_f64.hypot(500.0)).abs() < 1e-9, "{}", width(&triangle));
    // a square is as wide as its side, although its corners lie farther from its diagonal, the line through its two
    // places farthest apart
    assert!((width(&[[0.0, 0.0], [800.0, 0.0], [800.0, 800.0], [0.0, 800.0], [400.0, 400.0]]) - 800.0).abs() < 1e-9);
    // a strip 1000 cm long and 150 cm wide turned by 30 degrees, with places on its sides and inside, in any order
    let (sin, cos) = 30.0_f64.to_radians().sin_cos();
    let mut strip: Vec<[f64; 2]> = [[0.0, 0.0], [1000.0, 0.0], [1000.0, 150.0], [0.0, 150.0], [500.0, 150.0], [420.0, 60.0], [0.0, 75.0]]
        .iter()
        .map(|[x, y]| [200.0 + cos * x - sin * y, -300.0 + sin * x + cos * y])
        .collect();
    assert!((width(&strip) - 150.0).abs() < 1e-6, "{}", width(&strip));
    strip.reverse();
    strip.push(strip[2]);
    assert!((width(&strip) - 150.0).abs() < 1e-6, "{}", width(&strip));
}

#[test]
fn only_places_off_one_line_let_a_peak_replace_a_lateration() {
    // Anchors 10, 11 and 12 lie on the line y = 0, 13 and 14 lie 190 and 210 cm off it, and 15 hangs below anchor 10.
    // Agent 1 ranges three anchors on the line, agent 2 the two ends and the anchor 190 cm off it, agent 3 the two ends
    // and the anchor 210 cm off it, agent 4 anchor 10, the one below it and anchor 11, agent 5 the three on the line and
    // the one 190 cm off it, and agent 6 the three on the line and the one 210 cm off it, all joined by ranges.
    let places: [(u32, f64, f64, f64); 6] =
        [(10, 0.0, 0.0, 250.0), (11, 1000.0, 0.0, 250.0), (12, 500.0, 0.0, 250.0), (13, 500.0, 190.0, 250.0), (14, 500.0, 210.0, 250.0), (15, 0.0, 0.0, 100.0)];
    let truth: [(f64, f64); 6] = [(200.0, 300.0), (450.0, 420.0), (700.0, 350.0), (300.0, 520.0), (600.0, 600.0), (850.0, 480.0)];
    let ranged: [&[u32]; 6] = [&[10, 11, 12], &[10, 11, 13], &[10, 11, 14], &[10, 15, 11], &[10, 11, 12, 13], &[10, 11, 12, 14]];
    let mut observations = Vec::new();
    for (agent, ids) in ranged.iter().enumerate() {
        for id in *ids {
            let (_, x, y, z) = places.iter().find(|p| p.0 == *id).unwrap();
            let d = (truth[agent].0 - x).hypot(truth[agent].1 - y).hypot(*z);
            observations.push(calibrated(agent as u32 + 1, *id, true, &[d - 1.0, d, d + 1.5]));
        }
    }
    for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)] {
        let d = (truth[a].0 - truth[b].0).hypot(truth[a].1 - truth[b].1);
        observations.push(calibrated(a as u32 + 1, b as u32 + 1, false, &[d]));
    }
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3, 4, 5, 6],
        anchors: places.iter().map(|(id, x, y, z)| anchor(*id, *x, *y, *z)).collect(),
        observations,
        initial_positions: vec![Point2::default(); 6],
        mode: LocalizationMode::AnchorAided,
    };
    let block = block_profile();
    let calibration = CalibrationModel::default();
    let graph = aac::Graph::build(&problem, block.block_range.as_ref(), calibration);
    let law = block.presence.resolve(&graph.successful);
    // every agent ranges three anchor places or more, of which agent 4's lie at two horizontal places, and its
    // posterior has a peak, so the rule alone keeps the peaks of agents 1, 2, 4 and 5 out
    for (agent, (horizontal, identified)) in [(3, false), (3, false), (3, true), (2, false), (4, false), (4, true)].into_iter().enumerate() {
        let (links, present, missing) = peak::agent_inputs(&graph, agent);
        assert_eq!(links.len(), ranged[agent].len());
        assert_eq!((peak::horizontal_places(&links).len(), peak::identifies(&links)), (horizontal, identified), "agent {}", agent + 1);
        assert!(peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &peak::peak_area(&links)).is_some());
    }
    // the places of agents 2 and 5 lie within 95 cm of the line y = 95 and those of agents 3 and 6 are 210 cm wide
    for (agent, expected) in [(1, 190.0), (2, 210.0), (4, 190.0), (5, 210.0)] {
        let width = peak::minimum_width(&peak::horizontal_places(&peak::agent_inputs(&graph, agent).0));
        assert!((width - expected).abs() < 1e-9, "agent {}: {width}", agent + 1);
    }
    for law in [None, law] {
        let overrides = peak::peak_overrides(&graph, law.as_ref());
        assert_eq!(overrides.iter().map(Option::is_some).collect::<Vec<_>>(), [false, false, true, false, false, true], "{overrides:?}");
    }
}

/// The lateration problem with a fifth anchor, `extra`, that every agent ranges, so that agent 3 reaches three anchor
/// places.
fn lateration_problem_with(extra: (f64, f64, f64)) -> (PreparedProblem, [[f64; 2]; 3]) {
    let (mut problem, truth) = lateration_problem();
    problem.anchors.push(anchor(14, extra.0, extra.1, extra.2));
    for (i, agent) in truth.iter().enumerate() {
        let d = (agent[0] - extra.0).hypot(agent[1] - extra.1).hypot(extra.2);
        problem.observations.push(calibrated(i as u32 + 1, 14, true, &[d - 1.0, d, d + 1.5]));
    }
    (problem, truth)
}

/// Agent 3 of the lateration problem ranges a third anchor on the line of its two anchors, or below the first of them,
/// which leaves the mirror of its two anchors. While three anchor places sufficed, the peak of its posterior, the mirror
/// image of its truth, replaced its lateration, and one stage of the block objective with 64 starts ended with the agent
/// 12.8 m from its truth at an objective of 163.82 against the geometric start's 142.70 with the anchor on the line, and
/// 4.4 m from it at 150.88 against 143.22 with the anchor below. Its lateration stays, and the solve reaches the
/// geometric start's optimum.
#[test]
fn three_anchor_places_on_one_line_do_not_start_the_single_stage_solve() {
    let single = CooperativeProfile { starts: 64, ..block_profile() };
    for extra in [(500.0, 0.0, 250.0), (0.0, 0.0, 100.0)] {
        let (problem, truth) = lateration_problem_with(extra);
        let graph = aac::Graph::build(&problem, single.block_range.as_ref(), CalibrationModel::default());
        let law = single.presence.resolve(&graph.successful);
        let (links, present, missing) = peak::agent_inputs(&graph, 2);
        assert_eq!(links.len(), 3);
        let mirror = peak::agent_peak(&links, &present, &missing, &graph.heights, law.as_ref(), &peak::peak_area(&links)).unwrap();
        assert!(mirror.y < 0.0 && mirror.distance(Point2::new(truth[2][0], -truth[2][1])) < 30.0, "{extra:?}: {mirror:?}");
        let overrides = peak::peak_overrides(&graph, law.as_ref());
        assert!(overrides[..2].iter().all(Option::is_some) && overrides[2].is_none(), "{extra:?}: {overrides:?}");
        let result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, single.clone()).unwrap();
        for (position, truth) in result.positions.iter().zip(&truth) {
            assert!(position.distance(Point2::new(truth[0], truth[1])) < 1.0, "{extra:?}: {:?}", result.positions);
        }
        let geometric = geometric_solve(&problem, &single).metrics.final_objective;
        assert!((result.metrics.final_objective - geometric).abs() < 1e-6 * geometric, "{extra:?}: {} {geometric}", result.metrics.final_objective);
    }
}

#[test]
fn the_block_range_solve_starts_from_the_peak_layout_under_its_law() {
    let (lateration, _) = lateration_problem();
    let (peaked, _) = peak_problem([10, 11, 12, 13, 14]);
    let config = cooperative_config();
    let calibration = CalibrationModel::default();
    // In the lateration problem agents 1 and 2 range four anchors and agent 3 two, and the outcomes do not follow the
    // distance, so the maximum likelihood reach finds no law. In the peak problem agent 1 ranges three places and
    // failed its attempt at the far fourth anchor, agent 2 ranges two anchors, and that reach finds a law.
    let cases = [
        (&lateration, crate::ReachRule::Quantile, true, [true, true, false]),
        (&lateration, crate::ReachRule::Ml, false, [true, true, false]),
        (&peaked, crate::ReachRule::Ml, true, [true, false, false]),
    ];
    for (problem, reach, lawful, overridden) in cases {
        let mut profile = block_profile();
        profile.presence.reach = reach;
        profile.starts = 1;
        let mut graph = aac::Graph::build(problem, profile.block_range.as_ref(), calibration);
        let (start, law) = peak::start(problem, &config, calibration, &graph, &profile.presence).unwrap();
        let layout = |law: Option<&crate::presence::ResolvedPresence>| {
            crate::initializer::layout_with_overrides(problem, &config, calibration, &peak::peak_overrides(&graph, law)).unwrap()
        };
        // the quantile reach needs no layout, the maximum likelihood reach judges the outcomes at the layout with the
        // range only peaks
        let expected = match reach {
            crate::ReachRule::Quantile => profile.presence.resolve(&graph.successful),
            crate::ReachRule::Ml => graph.resolve_presence(&profile.presence, &layout(None)),
        };
        assert_eq!((law, law.is_some()), (expected, lawful), "{reach:?}");
        // the peaks are then taken again under the law, or the range only layout stays without one, and every agent
        // whose anchors reach three places starts at its own peak while the others keep their laterations
        assert_eq!(bits(&start), bits(&layout(law.as_ref())));
        let overrides = peak::peak_overrides(&graph, law.as_ref());
        assert_eq!(overrides.iter().map(Option::is_some).collect::<Vec<_>>(), overridden, "{overrides:?}");
        for (agent, peak) in overrides.iter().enumerate() {
            if let Some(peak) = peak {
                assert_eq!(start[agent], *peak);
            }
        }
        // the solve's one start is that layout, and its one more start places the agents with a lateration of their
        // own there, the initializer's laterations, both optimized under that law
        let laterations = crate::initializer::laterations(problem, &config, calibration).unwrap();
        assert_eq!(laterations, crate::initializer::layout_and_laterations(problem, &config, calibration).unwrap().1);
        let lateration = lateration_start(&start, &laterations).unwrap();
        let expected: Vec<Point2> = start.iter().zip(&laterations).map(|(p, own)| own.unwrap_or(*p)).collect();
        assert_eq!(bits(&lateration), bits(&expected));
        assert!(laterations[0].is_some() && laterations[2].is_none(), "{laterations:?}");
        graph.presence = law;
        let mut runs = multistart(&start, graph.typical_range(), 1);
        runs.push(flatten(&lateration));
        let factors = graph.pairs.iter().filter(|p| p.range.is_some()).count();
        let (positions, metrics) =
            optimize_layout(&mut aac::Objective { graph: &graph, profile: &profile.range_error }, runs, graph.agents, factors, &config).unwrap();
        assert_eq!(metrics.starts, 2);
        let mut result = LocalizationEngine::new().solve_with_cooperative_profile(problem, profile).unwrap();
        result.metrics.elapsed_micros = 0;
        assert_eq!((bits(&result.positions), result.metrics), (bits(&positions), metrics));
    }
}

/// With the median stage, the block range solve runs its peak start and one more start at the laterations in the
/// median stage, and the median stage's layout and one more start at the laterations in the block stage, every start
/// under the law of the peak start, as the geometric solve runs its lateration start in every stage.
#[test]
fn the_block_range_solve_adds_the_lateration_start_in_every_stage() {
    let (problem, _) = lateration_problem();
    let config = cooperative_config();
    let calibration = CalibrationModel::default();
    let mut refined = block_profile();
    refined.block_range.as_mut().unwrap().refine_from_median = true;
    refined.starts = 4;
    let graphs = [None, refined.block_range.clone()].map(|block| aac::Graph::build(&problem, block.as_ref(), calibration));
    let (mut start, law) = peak::start(&problem, &config, calibration, &graphs[1], &refined.presence).unwrap();
    let laterations = crate::initializer::laterations(&problem, &config, calibration).unwrap();
    let mut stages = Vec::new();
    for mut graph in graphs {
        graph.presence = law;
        let mut runs = multistart(&start, graph.typical_range(), refined.starts);
        runs.push(flatten(&lateration_start(&start, &laterations).unwrap()));
        let factors = graph.pairs.iter().filter(|p| p.range.is_some()).count();
        let stage = optimize_layout(&mut aac::Objective { graph: &graph, profile: &refined.range_error }, runs, graph.agents, factors, &config).unwrap();
        assert_eq!(stage.1.starts, 5);
        start = stage.0.clone();
        stages.push(stage);
    }
    let (positions, metrics) = stages.pop().unwrap();
    let mut result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, refined).unwrap();
    result.metrics.elapsed_micros = 0;
    assert_eq!((bits(&result.positions), result.metrics), (bits(&positions), metrics));
}

/// The cooperative solve as it ran before the peak start, composed of the solver's steps, for the starts that keep
/// the geometric initializer or the request's positions.
fn geometric_solve(problem: &PreparedProblem, profile: &CooperativeProfile) -> SolveResult {
    let config = cooperative_config();
    let calibration = profile.calibration.unwrap_or_default();
    let (mut start, laterations) = if profile.use_provided_start.unwrap_or(false) {
        (problem.initial_positions.clone(), Vec::new())
    } else {
        crate::initializer::layout_and_laterations(problem, &config, calibration).unwrap()
    };
    let refine = profile.block_range.as_ref().is_some_and(|b| b.refine_from_median);
    let stages = if refine { vec![None, profile.block_range.clone()] } else { vec![profile.block_range.clone()] };
    let mut solved = None;
    if problem.mode == LocalizationMode::AnchorFree {
        let graphs: Vec<af::Graph> = stages.iter().map(|block| af::Graph::build(problem, block.as_ref(), calibration)).collect();
        let law = graphs.last().and_then(|graph| graph.resolve_presence(&profile.presence, &start));
        for mut graph in graphs {
            graph.presence = law;
            let runs = multistart(&start, graph.typical_range(), profile.starts);
            let mut objective = af::Objective { graph: &graph, range_error: &profile.range_error };
            let stage = optimize_layout(&mut objective, runs, graph.agents, graph.ranges.len() + graph.pairs.len(), &config).unwrap();
            start = stage.0.clone();
            solved = Some(stage);
        }
    } else {
        let graphs: Vec<aac::Graph> = stages.iter().map(|block| aac::Graph::build(problem, block.as_ref(), calibration)).collect();
        let law = graphs.last().and_then(|graph| graph.resolve_presence(&profile.presence, &start));
        for mut graph in graphs {
            graph.presence = law;
            let mut runs = multistart(&start, graph.typical_range(), profile.starts);
            if let Some(layout) = lateration_start(&start, &laterations) {
                runs.push(flatten(&layout));
            }
            let mut objective = aac::Objective { graph: &graph, profile: &profile.range_error };
            let factors = graph.pairs.iter().filter(|p| p.range.is_some()).count();
            let stage = optimize_layout(&mut objective, runs, graph.agents, factors, &config).unwrap();
            start = stage.0.clone();
            solved = Some(stage);
        }
    }
    let (positions, metrics) = solved.unwrap();
    SolveResult { agent_ids: problem.agent_ids.clone(), positions, metrics }
}

#[test]
fn the_median_residual_density_the_ablation_and_agent_only_mode_solve_as_before() {
    let (mut aided, _) = lateration_problem();
    aided.initial_positions = vec![Point2::new(150.0, 120.0), Point2::new(700.0, 300.0), Point2::new(450.0, 500.0)];
    let mut free = problem(LocalizationMode::AnchorFree);
    free.initial_positions = vec![Point2::new(0.0, 0.0), Point2::new(280.0, 40.0), Point2::new(560.0, -60.0), Point2::new(90.0, 300.0)];
    let ml = |mut profile: CooperativeProfile| {
        profile.presence.reach = crate::ReachRule::Ml;
        profile
    };
    let provided = |mut profile: CooperativeProfile| {
        profile.use_provided_start = Some(true);
        profile
    };
    let mut refined = block_profile();
    refined.block_range.as_mut().unwrap().refine_from_median = true;
    let cases = [
        (&aided, profile()),
        (&aided, ml(profile())),
        (&aided, provided(profile())),
        (&aided, provided(block_profile())),
        (&aided, provided(ml(refined.clone()))),
        (&free, profile()),
        (&free, block_profile()),
        (&free, ml(refined.clone())),
        (&free, provided(refined)),
    ];
    for (problem, profile) in cases {
        let mut result = LocalizationEngine::new().solve_with_cooperative_profile(problem, profile.clone()).unwrap();
        result.metrics.elapsed_micros = 0;
        let expected = geometric_solve(problem, &profile);
        assert_eq!((bits(&result.positions), &result.metrics), (bits(&expected.positions), &expected.metrics), "{profile:?}");
    }
}

/// A fixed uniform draw in [0, 1) for a pair of indices.
fn uniform(a: u64, b: u64) -> f64 {
    let mut z = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_add(0x632b_e59b_d9b4_e019);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// Twelve agents on a 300 cm grid and `anchors` at zero height. Every attempt succeeds with the Gaussian law of
/// `reach` at its horizontal distance, or with probability one half whatever the distance when `reach` is None,
/// and every success carries exact ranges.
fn presence_scenario(reach: Option<f64>, anchors: &[(u32, f64, f64)]) -> (PreparedProblem, Vec<Point2>) {
    let truth: Vec<Point2> = (0..12).map(|i| Point2::new(300.0 * (i % 4) as f64, 300.0 * (i / 4) as f64)).collect();
    let succeeds = |d: f64, a: usize, b: usize| uniform(a as u64, b as u64) < reach.map_or(0.5, |r| (-d * d / (2.0 * r * r)).exp());
    let mut observations = Vec::new();
    for i in 0..12 {
        for j in i + 1..12 {
            let d = truth[i].distance(truth[j]);
            if succeeds(d, i, j) {
                observations.push(calibrated(i as u32 + 1, j as u32 + 1, false, &[d - 1.0, d, d + 1.0]));
            }
        }
        for (k, (id, x, y)) in anchors.iter().enumerate() {
            let d = truth[i].distance(Point2::new(*x, *y));
            if succeeds(d, i, 100 + k) {
                observations.push(calibrated(i as u32 + 1, *id, true, &[d - 1.0, d, d + 1.0]));
            }
        }
    }
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: (1..=12).collect(),
        anchors: anchors.iter().map(|(id, x, y)| anchor(*id, *x, *y, 0.0)).collect(),
        observations,
        initial_positions: truth.clone(),
        mode: if anchors.is_empty() { LocalizationMode::AnchorFree } else { LocalizationMode::AnchorAided },
    };
    (problem, truth)
}

#[test]
fn the_maximum_likelihood_reach_recovers_the_law_that_drew_the_outcomes() {
    let ml = crate::PresenceProfile { reach: crate::ReachRule::Ml, ..crate::PresenceProfile::default() };
    let within_a_step = |law: Option<crate::presence::ResolvedPresence>, truth: f64| {
        let reach = law.expect("a law").reach;
        assert!(reach >= truth / std::f64::consts::SQRT_2 && reach <= truth * std::f64::consts::SQRT_2, "{reach} for {truth}");
    };
    let config = SolveConfig { max_time_micros: 0, aggregation: ObservationAggregation::Median, ..SolveConfig::production() };
    let calibration = CalibrationModel::default();
    let corners = [(20, -300.0, -300.0), (21, 1200.0, -300.0), (22, 1200.0, 900.0), (23, -300.0, 900.0)];
    for reach in [450.0, 700.0] {
        // agent only, at the true layout and at the initializer's layout
        let (problem, truth) = presence_scenario(Some(reach), &[]);
        let graph = af::Graph::build(&problem, None, calibration);
        assert!(graph.pairs.iter().any(|p| !p.observed));
        within_a_step(graph.resolve_presence(&ml, &truth), reach);
        let layout = crate::initializer::initial_layout_with(&problem, &config, calibration).unwrap();
        within_a_step(graph.resolve_presence(&ml, &layout), reach);
        // anchor aided with the anchor outcomes, with and without height nodes at the anchors' height
        let (problem, truth) = presence_scenario(Some(reach), &corners);
        within_a_step(aac::Graph::build(&problem, None, calibration).resolve_presence(&ml, &truth), reach);
        let mut flat = block_profile();
        flat.block_range.as_mut().unwrap().height_nodes = vec![0.0];
        flat.block_range.as_mut().unwrap().height_weights = vec![1.0];
        within_a_step(aac::Graph::build(&problem, flat.block_range.as_ref(), calibration).resolve_presence(&ml, &truth), reach);
    }
    // outcomes that do not depend on the distance leave no law, and so does a scenario without a failure
    let (problem, truth) = presence_scenario(None, &[]);
    assert!(af::Graph::build(&problem, None, calibration).resolve_presence(&ml, &truth).is_none());
    let (problem, truth) = presence_scenario(None, &corners);
    assert!(aac::Graph::build(&problem, None, calibration).resolve_presence(&ml, &truth).is_none());
    let (complete, truth) = presence_scenario(Some(1.0e6), &[]);
    let graph = af::Graph::build(&complete, None, calibration);
    assert!(graph.pairs.iter().all(|p| p.observed));
    assert!(graph.resolve_presence(&ml, &truth).is_none());
    assert!(graph.resolve_presence(&crate::PresenceProfile::default(), &truth).is_some());
    // both cooperative modes solve with the rule, one stage or two
    let mut refined = block_profile();
    refined.block_range.as_mut().unwrap().refine_from_median = true;
    for mut profile in [profile(), refined] {
        profile.presence = ml;
        for anchors in [&[][..], &corners[..]] {
            let (problem, _) = presence_scenario(Some(600.0), anchors);
            let result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, profile.clone()).unwrap();
            assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
        }
    }
}

#[test]
fn complete_networks_keep_every_pair_as_an_opportunity() {
    let anchors = [anchor(10, 0.0, 0.0, 50.0), anchor(11, 900.0, 0.0, 50.0), anchor(12, 450.0, 800.0, 50.0)];
    let agents: [[f64; 2]; 3] = [[200.0, 200.0], [600.0, 300.0], [400.0, 550.0]];
    let mut observations = Vec::new();
    for (i, a) in agents.iter().enumerate() {
        for (j, b) in agents.iter().enumerate().skip(i + 1) {
            observations.push(calibrated(i as u32 + 1, j as u32 + 1, false, &[(a[0] - b[0]).hypot(a[1] - b[1])]));
        }
        for anchor in &anchors {
            let d = (a[0] - anchor.position.x).hypot(a[1] - anchor.position.y).hypot(50.0);
            observations.push(calibrated(i as u32 + 1, anchor.id, true, &[d]));
        }
    }
    let problem = PreparedProblem {
        attempts: None,
        agent_ids: vec![1, 2, 3],
        anchors: anchors.to_vec(),
        observations,
        initial_positions: vec![Point2::default(); 3],
        mode: LocalizationMode::AnchorAided,
    };
    let graph = aac::Graph::new(&problem, None, CalibrationModel::default(), &crate::PresenceProfile::default());
    assert!(graph.pairs.iter().all(|p| p.opportunity && p.observed));
    assert_eq!(graph.pairs.len(), 3 + 9);
}

/// A splitmix sequence of uniform draws in [0, 1), and standard normal draws from two of them.
struct Draw(u64);

impl Draw {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform().max(1e-300), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

/// Sixty agent only units of six to eight agents within 8 by 9 m, ten packets on every pair drawn with the packet
/// scale of the TIERS layout fit, solved with that fit's link model reduced to no persistent offset and false early
/// detections at 1e-6 per cm, a reduced model the confirmatory round refits and solves. The continuation below such a
/// table divided by the spread of an offset of zero and was not finite, so 7 units failed on a result that was not
/// finite and 50 of the other 53 ended unconverged, where all 60 converged with the smallest fitted offset. The table
/// now reaches down to zero distance whenever early detections are on.
#[test]
fn no_offset_with_early_detections_solves_every_agent_only_unit() {
    let (kappa, scale) = (0.1862761985520489, 11.168277015691466);
    let profile = CooperativeProfile {
        block_range: Some(BlockRangeProfile {
            kappa,
            excess_probability: 0.95,
            excess_mean_rel: 3.8316288315373983,
            persistent_sd_rel: 0.0,
            height_nodes: vec![6.6, 9.7, 12.7, 15.8, 18.8],
            height_weights: vec![1.0; 5],
            refine_from_median: true,
            nu: 3.0,
            independent_packets: false,
            offset_shape: crate::OffsetShape::StudentT,
            early_detection_per_cm: 1e-6,
        }),
        starts: 64,
        ..profile()
    };
    let mut rng = Draw(77);
    let units: Vec<PreparedProblem> = (0..60)
        .map(|unit| {
            let agents = 6 + unit % 3;
            let truth: Vec<[f64; 2]> = (0..agents).map(|_| [800.0 * rng.uniform(), 900.0 * rng.uniform()]).collect();
            let mut observations = Vec::new();
            for i in 0..agents {
                for j in i + 1..agents {
                    let d = (truth[i][0] - truth[j][0]).hypot(truth[i][1] - truth[j][1]);
                    let samples = (0..10).map(|_| CalibratedRangeSample::new((d + kappa * scale * rng.normal()).max(0.0), scale)).collect();
                    observations.push(RangeObservation { from: i as u32 + 1, to: j as u32 + 1, tof_ns: vec![], calibrated_samples: samples, to_is_anchor: false });
                }
            }
            PreparedProblem {
                attempts: None,
                agent_ids: (1..=agents as u32).collect(),
                anchors: vec![],
                observations,
                initial_positions: vec![Point2::default(); agents],
                mode: LocalizationMode::AnchorFree,
            }
        })
        .collect();
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get());
    let results: Vec<Result<crate::SolveResult, crate::EngineError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = units
            .chunks(units.len().div_ceil(workers))
            .map(|chunk| {
                let profile = &profile;
                scope.spawn(move || chunk.iter().map(|p| LocalizationEngine::new().solve_with_cooperative_profile(p, profile.clone())).collect::<Vec<_>>())
            })
            .collect();
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    let mut unconverged = 0;
    for (unit, result) in results.into_iter().enumerate() {
        let result = result.unwrap_or_else(|error| panic!("unit {unit}: {error}"));
        assert!(result.positions.iter().all(|p| p.x.is_finite() && p.y.is_finite()), "unit {unit}");
        unconverged += usize::from(!result.metrics.converged);
    }
    println!("{unconverged} of 60 units ended unconverged");
    assert!(unconverged <= 3, "{unconverged} of 60 units ended unconverged");
}
