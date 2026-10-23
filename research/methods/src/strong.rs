use std::collections::BTreeMap;
use std::time::Instant;

use solid_core::{
    LocalizationMode, ObservationAggregation, PreparedProblem, SolveConfig, SolveMetrics,
    SolveResult, TerminationReason,
};

use crate::geometry::{
    dimensions, initial_parameters, points, prepare_link_factors, Endpoint, RangeFactor,
};
use crate::{LocalizationMethod, MethodError, MethodKind};

pub struct Strong;

impl LocalizationMethod for Strong {
    fn kind(&self) -> MethodKind {
        MethodKind::Strong
    }

    fn solve(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError> {
        if problem.mode == LocalizationMode::AnchorFree {
            return Err(MethodError::UnsupportedMode(
                "STRONG's convex underestimator collapses without anchor terms".to_owned(),
            ));
        }
        let started = Instant::now();
        let dimensions = dimensions(problem, config);
        let directed = prepare_link_factors(problem, config)?;
        let factors = prepare_factors(&directed, config.aggregation, config.robust_delta);
        let mut current = Variables::initialize(
            initial_parameters(problem, config, dimensions)?,
            dimensions,
            &factors,
        );
        let mut previous = current.clone();
        let lipschitz = lipschitz_constant(problem.agent_ids.len(), &factors);
        let maximum_iterations = config.max_iterations.clamp(1, 100_000);
        let tolerance = config.gradient_tolerance.max(1e-12);
        // The loop stops once a step is small against the size of the state, with the declared relative tolerance.
        // Soares and Gomes leave the stopping criterion open, as Algorithm 1 of their paper runs "while some stopping
        // criterion is not met" (STRONG, Signal Processing 185, 2021, 108066), so the tolerance is this
        // implementation's, and it is relative so that it scales with the size of the network. The norm of absolute
        // coordinates scales instead with the distance of the network from the origin of the frame, which the method
        // never sees. In a frame whose origin lies far from the network, as TWR-CLOUD-Bern's does, it stopped the loop
        // early and left the estimate short of the minimizer. The positions are therefore measured from the centroid of
        // the known anchors, or from the centroid of the estimate itself in a problem without anchors. Every step of
        // Algorithm 1 reads the positions only through the differences between two nodes or between a node and an
        // anchor, so the centring moves no iterate, objective value or minimizer, and the method stays the published
        // one. It changes only where the loop of a frame far from its origin stops, which is now the iteration at which
        // the loop of any other frame stops.
        let centre = anchor_centroid(problem, dimensions);
        let mut iterations = 0usize;
        let mut termination = TerminationReason::IterationLimit;
        let mut converged = false;

        for iteration in 1..=maximum_iterations {
            if time_limit_reached(started, config) {
                termination = TerminationReason::TimeLimit;
                break;
            }
            let coefficient = (iteration as f64 - 2.0) / (iteration as f64 + 1.0);
            let extrapolated = Variables::extrapolate(&current, &previous, coefficient);
            let next = fista_step(extrapolated, dimensions, &factors, lipschitz);
            iterations = iteration;
            let step_norm = next.distance_squared(&current).sqrt();
            let state_norm = next
                .norm_squared_about(
                    &centre.unwrap_or_else(|| next.centroid(dimensions)),
                    dimensions,
                )
                .sqrt();
            previous = current;
            current = next;
            if step_norm <= tolerance * state_norm.max(1.0) {
                converged = true;
                termination = TerminationReason::Converged;
                break;
            }
        }

        let final_objective = objective(&current, dimensions, &factors);
        let factor_count = factors.len();

        Ok(SolveResult {
            agent_ids: problem.agent_ids.clone(),
            positions: points(&current.positions, dimensions),
            metrics: SolveMetrics {
                elapsed_micros: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                dimensions,
                iterations,
                objective_evaluations: 1,
                gradient_evaluations: iterations,
                factors: factor_count,
                range_factor_evaluations: iterations.saturating_add(1).saturating_mul(factor_count),
                coordinate_factor_evaluations: iterations
                    .saturating_mul(factor_count)
                    .saturating_mul(dimensions),
                converged,
                termination,
                final_objective,
                starts: 1,
                guarded_supports: None,
                largest_dropped_mass: None,
                posterior_nodes: None,
                posterior_cells: None,
            },
        })
    }
}

#[derive(Clone, Copy)]
struct StrongFactor {
    from: usize,
    to: Endpoint,
    distance_cm: f64,
    huber_radius_cm: f64,
}

#[derive(Clone)]
struct Variables {
    positions: Vec<f64>,
    auxiliaries: Vec<[f64; 3]>,
}

impl Variables {
    fn initialize(positions: Vec<f64>, dimensions: usize, factors: &[StrongFactor]) -> Self {
        let auxiliaries = factors
            .iter()
            .map(|factor| {
                project_ball(
                    difference(&positions, dimensions, factor.from, factor.to),
                    dimensions,
                    factor.distance_cm,
                )
            })
            .collect();
        Self {
            positions,
            auxiliaries,
        }
    }

    fn extrapolate(current: &Self, previous: &Self, coefficient: f64) -> Self {
        Self {
            positions: current
                .positions
                .iter()
                .zip(&previous.positions)
                .map(|(current, previous)| current + coefficient * (current - previous))
                .collect(),
            auxiliaries: current
                .auxiliaries
                .iter()
                .zip(&previous.auxiliaries)
                .map(|(current, previous)| {
                    std::array::from_fn(|index| {
                        current[index] + coefficient * (current[index] - previous[index])
                    })
                })
                .collect(),
        }
    }

    fn distance_squared(&self, other: &Self) -> f64 {
        self.positions
            .iter()
            .zip(&other.positions)
            .map(|(left, right)| (left - right).powi(2))
            .chain(
                self.auxiliaries
                    .iter()
                    .zip(&other.auxiliaries)
                    .flat_map(|(left, right)| {
                        left.iter()
                            .zip(right)
                            .map(|(left, right)| (left - right).powi(2))
                    }),
            )
            .sum()
    }

    /// The squared norm of the state with every position measured from `centre`. The auxiliaries are differences of
    /// positions already and enter as they are.
    fn norm_squared_about(&self, centre: &[f64; 3], dimensions: usize) -> f64 {
        self.positions
            .chunks_exact(dimensions)
            .flat_map(|position| {
                position
                    .iter()
                    .zip(centre)
                    .map(|(value, centre)| (value - centre).powi(2))
            })
            .chain(
                self.auxiliaries
                    .iter()
                    .flat_map(|value| value.iter().map(|value| value * value)),
            )
            .sum()
    }

    /// The centroid of the positions.
    fn centroid(&self, dimensions: usize) -> [f64; 3] {
        let count = (self.positions.len() / dimensions).max(1) as f64;
        std::array::from_fn(|dimension| {
            if dimension < dimensions {
                self.positions
                    .chunks_exact(dimensions)
                    .map(|position| position[dimension])
                    .sum::<f64>()
                    / count
            } else {
                0.0
            }
        })
    }
}

/// The centroid of the known anchors in the coordinates of the estimate, the planar or the spatial ones, and none for a
/// problem without anchors.
fn anchor_centroid(problem: &PreparedProblem, dimensions: usize) -> Option<[f64; 3]> {
    if problem.anchors.is_empty() {
        return None;
    }
    let count = problem.anchors.len() as f64;
    Some(std::array::from_fn(|dimension| {
        if dimension < dimensions {
            problem
                .anchors
                .iter()
                .map(|anchor| [anchor.position.x, anchor.position.y, anchor.height_cm][dimension])
                .sum::<f64>()
                / count
        } else {
            0.0
        }
    }))
}

fn prepare_factors(
    factors: &[RangeFactor],
    aggregation: ObservationAggregation,
    huber_radius_cm: f64,
) -> Vec<StrongFactor> {
    match aggregation {
        ObservationAggregation::Median => median_undirected_links(factors, huber_radius_cm),
        ObservationAggregation::AllSamples => factors
            .iter()
            .map(|factor| canonical_factor(*factor, huber_radius_cm))
            .collect(),
    }
}

fn median_undirected_links(factors: &[RangeFactor], huber_radius_cm: f64) -> Vec<StrongFactor> {
    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    enum Key {
        Agent(usize, usize),
        Anchor(usize, u32),
    }

    let mut grouped = BTreeMap::<Key, Vec<(f64, Endpoint)>>::new();
    for factor in factors {
        let key = match factor.to {
            Endpoint::Agent(to) => Key::Agent(factor.from.min(to), factor.from.max(to)),
            Endpoint::Anchor(anchor) => Key::Anchor(factor.from, anchor.id),
        };
        grouped
            .entry(key)
            .or_default()
            .push((factor.distance_cm, factor.to));
    }

    grouped
        .into_iter()
        .map(|(key, measurements)| {
            let distance_cm = median(measurements.iter().map(|value| value.0).collect());
            let (from, to) = match key {
                Key::Agent(from, to) => (from, Endpoint::Agent(to)),
                Key::Anchor(from, _) => (from, measurements[0].1),
            };
            StrongFactor {
                from,
                to,
                distance_cm,
                huber_radius_cm: huber_radius_cm.max(1e-3),
            }
        })
        .collect()
}

fn canonical_factor(factor: RangeFactor, huber_radius_cm: f64) -> StrongFactor {
    let (from, to) = match factor.to {
        Endpoint::Agent(to) => (factor.from.min(to), Endpoint::Agent(factor.from.max(to))),
        Endpoint::Anchor(anchor) => (factor.from, Endpoint::Anchor(anchor)),
    };
    StrongFactor {
        from,
        to,
        distance_cm: factor.distance_cm,
        huber_radius_cm: huber_radius_cm.max(1e-3),
    }
}

fn fista_step(
    extrapolated: Variables,
    dimensions: usize,
    factors: &[StrongFactor],
    lipschitz: f64,
) -> Variables {
    let mut position_gradient = vec![0.0; extrapolated.positions.len()];
    let mut auxiliary_gradient = vec![[0.0; 3]; factors.len()];

    for (index, factor) in factors.iter().enumerate() {
        let difference = difference(&extrapolated.positions, dimensions, factor.from, factor.to);
        let residual = std::array::from_fn(|dimension| {
            difference[dimension] - extrapolated.auxiliaries[index][dimension]
        });
        let projected = project_ball(residual, dimensions, factor.huber_radius_cm);
        let from = factor.from * dimensions;
        for dimension in 0..dimensions {
            position_gradient[from + dimension] += projected[dimension];
            auxiliary_gradient[index][dimension] -= projected[dimension];
            if let Endpoint::Agent(to) = factor.to {
                position_gradient[to * dimensions + dimension] -= projected[dimension];
            }
        }
    }

    let positions = extrapolated
        .positions
        .iter()
        .zip(position_gradient)
        .map(|(position, gradient)| position - gradient / lipschitz)
        .collect::<Vec<_>>();
    let auxiliaries = extrapolated
        .auxiliaries
        .iter()
        .zip(auxiliary_gradient)
        .zip(factors)
        .map(|((auxiliary, gradient), factor)| {
            let stepped = std::array::from_fn(|dimension| {
                auxiliary[dimension] - gradient[dimension] / lipschitz
            });
            project_ball(stepped, dimensions, factor.distance_cm)
        })
        .collect();

    Variables {
        positions,
        auxiliaries,
    }
}

fn lipschitz_constant(agents: usize, factors: &[StrongFactor]) -> f64 {
    let mut agent_degrees = vec![0usize; agents];
    let mut anchor_degrees = vec![0usize; agents];
    for factor in factors {
        match factor.to {
            Endpoint::Agent(to) => {
                agent_degrees[factor.from] += 1;
                agent_degrees[to] += 1;
            }
            Endpoint::Anchor(_) => anchor_degrees[factor.from] += 1,
        }
    }
    2.0 + 2.0 * maximum(agent_degrees) as f64 + maximum(anchor_degrees) as f64
}

fn maximum(values: Vec<usize>) -> usize {
    values.into_iter().max().unwrap_or(0)
}

fn difference(parameters: &[f64], dimensions: usize, from: usize, to: Endpoint) -> [f64; 3] {
    let from = from * dimensions;
    match to {
        Endpoint::Agent(to) => {
            let to = to * dimensions;
            std::array::from_fn(|dimension| {
                if dimension < dimensions {
                    parameters[from + dimension] - parameters[to + dimension]
                } else {
                    0.0
                }
            })
        }
        Endpoint::Anchor(anchor) => {
            let position = [anchor.position.x, anchor.position.y, anchor.height_cm];
            std::array::from_fn(|dimension| {
                if dimension < dimensions {
                    parameters[from + dimension] - position[dimension]
                } else {
                    0.0
                }
            })
        }
    }
}

fn project_ball(vector: [f64; 3], dimensions: usize, radius: f64) -> [f64; 3] {
    let norm = vector[..dimensions]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm <= radius || norm <= f64::EPSILON {
        vector
    } else {
        vector.map(|value| value * radius / norm)
    }
}

fn objective(state: &Variables, dimensions: usize, factors: &[StrongFactor]) -> f64 {
    factors
        .iter()
        .enumerate()
        .map(|(index, factor)| {
            let difference = difference(&state.positions, dimensions, factor.from, factor.to);
            let residual_norm = difference[..dimensions]
                .iter()
                .zip(&state.auxiliaries[index])
                .map(|(difference, auxiliary)| (difference - auxiliary).powi(2))
                .sum::<f64>()
                .sqrt();
            let radius = factor.huber_radius_cm;
            0.5 * if residual_norm <= radius {
                residual_norm * residual_norm
            } else {
                2.0 * radius * residual_norm - radius * radius
            }
        })
        .sum()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

fn time_limit_reached(started: Instant, config: &SolveConfig) -> bool {
    config.max_time_micros > 0 && started.elapsed().as_micros() >= config.max_time_micros as u128
}

#[cfg(test)]
mod tests {
    use solid_core::{
        Anchor, CalibratedRangeSample, DimensionPolicy, LocalizationEngine, LocalizationMode,
        Point2, PreparedProblem, RangeObservation, SolveConfig,
    };

    use super::*;
    use crate::{BaselineInitialization, MethodConfig, MethodKind, MethodRunner};

    fn exact_problem() -> PreparedProblem {
        let anchors = vec![
            anchor(10, 0.0, 0.0),
            anchor(11, 300.0, 0.0),
            anchor(12, 0.0, 300.0),
        ];
        let truth = [Point2::new(100.0, 100.0), Point2::new(200.0, 100.0)];
        let observations = truth
            .iter()
            .enumerate()
            .flat_map(|(index, point)| {
                anchors.iter().map(move |anchor| RangeObservation {
                    from: index as u32 + 1,
                    to: anchor.id,
                    tof_ns: vec![tof(point.distance(anchor.position))],
                    calibrated_samples: Vec::new(),
                    to_is_anchor: true,
                })
            })
            .collect();
        PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2],
            anchors,
            observations,
            initial_positions: vec![Point2::default(); 2],
            mode: LocalizationMode::AnchorOnly,
        }
    }

    #[test]
    fn solves_exact_anchor_geometry_without_truth() {
        let problem = exact_problem();
        let config = SolveConfig {
            max_iterations: 10_000,
            gradient_tolerance: 1e-8,
            ..SolveConfig::default()
        };

        let result = Strong.solve(&problem, &config).unwrap();

        assert!(result
            .positions
            .iter()
            .all(|point| point.x.is_finite() && point.y.is_finite()));
        assert!(result.metrics.final_objective < 1e-6);
    }

    /// Six agents and four anchors in a room of 18 m by 12 m, every agent ranging every anchor and in the anchor aided
    /// mode every other agent, with noise of up to 10 cm and every seventh range 250 cm too long, all agents starting
    /// near the middle of the room.
    fn noisy_problem(mode: LocalizationMode) -> PreparedProblem {
        let anchors = vec![
            Anchor {
                id: 101,
                position: Point2::new(0.0, 0.0),
                height_cm: 260.0,
            },
            Anchor {
                id: 102,
                position: Point2::new(1800.0, 0.0),
                height_cm: 240.0,
            },
            Anchor {
                id: 103,
                position: Point2::new(1800.0, 1200.0),
                height_cm: 250.0,
            },
            Anchor {
                id: 104,
                position: Point2::new(0.0, 1200.0),
                height_cm: 270.0,
            },
        ];
        let truth: Vec<[f64; 3]> = (0..6)
            .map(|index| {
                [
                    250.0 + 230.0 * index as f64,
                    200.0 + 150.0 * ((3 * index) % 5) as f64,
                    120.0,
                ]
            })
            .collect();
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut noise = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let mut ends: Vec<(u32, u32, bool, [f64; 3])> = Vec::new();
        for index in 0..truth.len() {
            for anchor in &anchors {
                ends.push((
                    index as u32 + 1,
                    anchor.id,
                    true,
                    [anchor.position.x, anchor.position.y, anchor.height_cm],
                ));
            }
            if mode == LocalizationMode::AnchorAided {
                for (other, target) in truth.iter().enumerate().skip(index + 1) {
                    ends.push((index as u32 + 1, other as u32 + 1, false, *target));
                }
            }
        }
        let observations = ends
            .into_iter()
            .enumerate()
            .map(|(link, (from, to, to_is_anchor, target))| {
                let source = truth[from as usize - 1];
                let distance = (0..3)
                    .map(|d| (source[d] - target[d]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let outlier = if link % 7 == 3 { 250.0 } else { 0.0 };
                RangeObservation {
                    from,
                    to,
                    to_is_anchor,
                    tof_ns: vec![],
                    calibrated_samples: vec![CalibratedRangeSample::new(
                        distance + 20.0 * noise() + outlier,
                        15.0,
                    )],
                }
            })
            .collect();
        PreparedProblem {
            attempts: None,
            agent_ids: (1..=6).collect(),
            anchors,
            observations,
            initial_positions: (0..6)
                .map(|index| Point2::new(850.0 + 37.0 * index as f64, 650.0 - 23.0 * index as f64))
                .collect(),
            mode,
        }
    }

    /// The problem in a frame whose origin lies at `offset` from the original one, heights included.
    fn shifted(problem: &PreparedProblem, offset: [f64; 3]) -> PreparedProblem {
        let mut moved = problem.clone();
        for anchor in &mut moved.anchors {
            anchor.position =
                Point2::new(anchor.position.x - offset[0], anchor.position.y - offset[1]);
            anchor.height_cm -= offset[2];
        }
        for point in &mut moved.initial_positions {
            *point = Point2::new(point.x - offset[0], point.y - offset[1]);
        }
        moved
    }

    /// The same network in a frame like TWR-CLOUD-Bern's, whose scans lie about 210 m and 180 m from its origin, and in
    /// a frame 10 km away that moves the heights too, stops at the same iteration with the same estimate up to floating
    /// point. The norm of absolute coordinates stopped the first at about half the iterations, up to 22 cm from the
    /// estimate, and the second after one iteration.
    #[test]
    fn a_shifted_frame_gives_the_same_estimate() {
        for offset in [[21_000.0, 18_000.0, 0.0], [-1.0e6, 3.0e5, 700.0]] {
            for mode in [LocalizationMode::AnchorAided, LocalizationMode::AnchorOnly] {
                for dimension_policy in [DimensionPolicy::Planar, DimensionPolicy::Spatial] {
                    let config = SolveConfig {
                        aggregation: ObservationAggregation::Median,
                        dimension_policy,
                        max_iterations: 10_000,
                        gradient_tolerance: 1e-5,
                        robust_delta: 40.0,
                        max_time_micros: 0,
                        ..SolveConfig::default()
                    };
                    let problem = noisy_problem(mode);
                    let here = Strong.solve(&problem, &config).unwrap();
                    let there = Strong.solve(&shifted(&problem, offset), &config).unwrap();
                    let case = format!("{mode:?} {dimension_policy:?} {offset:?}");
                    assert_eq!(
                        here.metrics.termination,
                        TerminationReason::Converged,
                        "{case}"
                    );
                    assert_eq!(there.metrics.iterations, here.metrics.iterations, "{case}");
                    for (a, b) in here.positions.iter().zip(&there.positions) {
                        assert!(
                            (a.x - (b.x + offset[0])).abs() < 1e-6
                                && (a.y - (b.y + offset[1])).abs() < 1e-6,
                            "{case}"
                        );
                    }
                    assert!(
                        (here.metrics.final_objective - there.metrics.final_objective).abs()
                            <= 1e-6 * here.metrics.final_objective.max(1.0)
                    );
                }
            }
        }
    }

    #[test]
    fn initialization_adapter_changes_only_the_seed() {
        let problem = exact_problem();
        let original = problem.clone();
        let config = SolveConfig::default();
        for initialization in [
            BaselineInitialization::Released,
            BaselineInitialization::SolidGeometry,
        ] {
            let mut seeded = problem.clone();
            if initialization == BaselineInitialization::SolidGeometry {
                seeded.initial_positions = LocalizationEngine::new()
                    .initial_layout(&problem, &config)
                    .unwrap();
            }
            let mut expected = Strong.solve(&seeded, &config).unwrap();
            let mut actual = MethodRunner::configured(
                MethodKind::Strong,
                MethodConfig::Strong { initialization },
            )
            .unwrap()
            .solve(&problem, &config)
            .unwrap();
            expected.metrics.elapsed_micros = 0;
            actual.metrics.elapsed_micros = 0;
            assert_eq!(actual, expected);
            assert_eq!(problem, original);
        }
    }

    /// A configured runner reports the positions FNL and STRONG started from, the problem's own for the released start
    /// and the engine's geometric layout for the geometry start, and whether the estimate stayed there.
    #[test]
    fn a_solve_reports_its_start_and_whether_it_stayed_there() {
        let problem = noisy_problem(LocalizationMode::AnchorOnly);
        let config = SolveConfig {
            max_iterations: 10_000,
            robust_delta: 40.0,
            ..SolveConfig::default()
        };
        let layout = LocalizationEngine::new()
            .initial_layout(&problem, &config)
            .unwrap();
        for kind in [MethodKind::Fnl, MethodKind::Strong] {
            for (initialization, expected) in [
                (
                    BaselineInitialization::Released,
                    problem.initial_positions.clone(),
                ),
                (BaselineInitialization::SolidGeometry, layout.clone()),
            ] {
                let method_config = if kind == MethodKind::Fnl {
                    MethodConfig::Fnl { initialization }
                } else {
                    MethodConfig::Strong { initialization }
                };
                let runner = MethodRunner::configured(kind, method_config).unwrap();
                let (result, start) = runner.solve_with_start(&problem, &config).unwrap();
                let start = start.unwrap();
                assert_eq!(start, expected, "{kind:?} {initialization:?}");
                // the estimate of the solve that does not report its start, and one that left the start
                assert_eq!(
                    result.positions,
                    runner.solve(&problem, &config).unwrap().positions
                );
                assert_eq!(crate::stayed_at_start(&result, Some(&start)), Some(false));
                assert_eq!(
                    crate::stayed_at_start(&result, Some(&result.positions)),
                    Some(true)
                );
            }
        }
        // a runner without a configuration starts from the problem's positions, and a solve without a start reports none
        let (result, start) = MethodRunner::new(MethodKind::Strong)
            .solve_with_start(&problem, &config)
            .unwrap();
        assert_eq!(start, Some(problem.initial_positions.clone()));
        assert_eq!(crate::stayed_at_start(&result, None), None);
    }

    #[test]
    fn combines_both_directions_into_one_edge() {
        let factors = vec![
            RangeFactor {
                from: 0,
                to: Endpoint::Agent(1),
                distance_cm: 100.0,
                inverse_scale: 0.1,
            },
            RangeFactor {
                from: 1,
                to: Endpoint::Agent(0),
                distance_cm: 102.0,
                inverse_scale: 0.1,
            },
        ];

        let combined = prepare_factors(&factors, ObservationAggregation::Median, 0.7);

        assert_eq!(combined.len(), 1);
        assert_eq!(combined[0].from, 0);
        assert!(matches!(combined[0].to, Endpoint::Agent(1)));
        assert_eq!(combined[0].distance_cm, 101.0);
        assert_eq!(combined[0].huber_radius_cm, 0.7);
    }

    #[test]
    fn uses_one_huber_radius_for_every_link() {
        let factors = vec![
            RangeFactor {
                from: 0,
                to: Endpoint::Agent(1),
                distance_cm: 100.0,
                inverse_scale: 0.1,
            },
            RangeFactor {
                from: 0,
                to: Endpoint::Anchor(anchor(10, 0.0, 0.0)),
                distance_cm: 120.0,
                inverse_scale: 0.01,
            },
        ];

        let combined = prepare_factors(&factors, ObservationAggregation::Median, 40.0);

        assert_eq!(combined.len(), 2);
        assert!(combined.iter().all(|factor| factor.huber_radius_cm == 40.0));
    }

    #[test]
    fn all_sample_adapter_retains_repeated_and_opposite_direction_measurements() {
        let factors = vec![
            RangeFactor {
                from: 0,
                to: Endpoint::Agent(1),
                distance_cm: 100.0,
                inverse_scale: 0.1,
            },
            RangeFactor {
                from: 1,
                to: Endpoint::Agent(0),
                distance_cm: 102.0,
                inverse_scale: 0.1,
            },
        ];

        let retained = prepare_factors(&factors, ObservationAggregation::AllSamples, 40.0);

        assert_eq!(retained.len(), 2);
        assert!(retained
            .iter()
            .all(|factor| factor.from == 0 && matches!(factor.to, Endpoint::Agent(1))));
        assert_eq!(retained[0].distance_cm, 100.0);
        assert_eq!(retained[1].distance_cm, 102.0);
    }

    #[test]
    fn rejects_anchor_free_instead_of_reporting_the_collapsed_relaxation() {
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2],
            anchors: Vec::new(),
            observations: vec![RangeObservation {
                from: 1,
                to: 2,
                tof_ns: vec![tof(100.0)],
                calibrated_samples: Vec::new(),
                to_is_anchor: false,
            }],
            initial_positions: vec![Point2::default(); 2],
            mode: LocalizationMode::AnchorFree,
        };

        let error = Strong.solve(&problem, &SolveConfig::default()).unwrap_err();

        assert_eq!(
            error,
            MethodError::UnsupportedMode(
                "STRONG's convex underestimator collapses without anchor terms".to_owned()
            )
        );
    }

    #[test]
    fn anchor_free_collapsed_state_has_zero_published_objective() {
        let factors = vec![StrongFactor {
            from: 0,
            to: Endpoint::Agent(1),
            distance_cm: 100.0,
            huber_radius_cm: 40.0,
        }];
        let state = Variables {
            positions: vec![0.0; 4],
            auxiliaries: vec![[0.0; 3]],
        };

        assert_eq!(objective(&state, 2, &factors), 0.0);
    }

    fn anchor(id: u32, x: f64, y: f64) -> Anchor {
        Anchor {
            id,
            position: Point2::new(x, y),
            height_cm: 0.0,
        }
    }

    fn tof(distance: f64) -> f64 {
        (distance - 19.207_257_79) / 28.509_090_85
    }
}
