use std::time::Instant;

use nalgebra::{DMatrix, SymmetricEigen};
use solid_core::{
    LocalizationMode, PreparedProblem, SolveConfig, SolveMetrics, SolveResult, TerminationReason,
};

use crate::geometry::{
    agent_components, center_components, dimensions, initial_parameters, points,
    prepare_link_factors, Endpoint, RangeFactor,
};
use crate::{LocalizationMethod, MethodError, MethodKind};

const INITIAL_INNER_STEPS: usize = 40;
const INNER_GROWTH_PERIOD: usize = 1_000;
const PAPER_ITERATION_BUDGET: usize = 10_000;

pub struct FastNetworkLocalization;

impl LocalizationMethod for FastNetworkLocalization {
    fn kind(&self) -> MethodKind {
        MethodKind::Fnl
    }

    fn solve(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError> {
        let started = Instant::now();
        let dimensions = dimensions(problem, config);
        let factors = prepare_link_factors(problem, config)?;
        let mut parameters = initial_parameters(problem, config, dimensions)?;
        let components = agent_components(problem.agent_ids.len(), &factors);
        if problem.mode == LocalizationMode::AnchorFree {
            seed_coincident_components(&mut parameters, dimensions, &components, &factors);
            center_components(&mut parameters, dimensions, &components);
        }
        let lipschitz = quadratic_lipschitz(problem.agent_ids.len(), &factors).max(1e-12);
        let iteration_budget = config.max_iterations.clamp(1, PAPER_ITERATION_BUDGET);
        let mut objective_evaluations = 0usize;
        let mut gradient_evaluations = 0usize;
        let mut outer_iterations = 0usize;
        let mut termination = TerminationReason::IterationLimit;

        while gradient_evaluations < iteration_budget {
            if time_limit_reached(started, config) {
                termination = TerminationReason::TimeLimit;
                break;
            }
            let targets = sphere_targets(&parameters, dimensions, &factors);
            let mut accelerated = parameters.clone();
            let mut momentum = 1.0_f64;
            let remaining = iteration_budget - gradient_evaluations;
            let inner_steps = scheduled_inner_steps(outer_iterations).min(remaining);

            for _ in 0..inner_steps {
                if time_limit_reached(started, config) {
                    termination = TerminationReason::TimeLimit;
                    break;
                }
                let gradient = surrogate_gradient(
                    &accelerated,
                    dimensions,
                    problem.agent_ids.len(),
                    &factors,
                    &targets,
                );
                gradient_evaluations += 1;
                let mut next = accelerated
                    .iter()
                    .zip(&gradient)
                    .map(|(value, gradient)| value - gradient / lipschitz)
                    .collect::<Vec<_>>();
                if problem.mode == LocalizationMode::AnchorFree {
                    center_components(&mut next, dimensions, &components);
                }
                let next_momentum = 0.5 * (1.0 + (1.0 + 4.0 * momentum * momentum).sqrt());
                let coefficient = (momentum - 1.0) / next_momentum;
                accelerated = next
                    .iter()
                    .zip(&parameters)
                    .map(|(next, current)| next + coefficient * (next - current))
                    .collect();
                parameters = next;
                momentum = next_momentum;
            }

            outer_iterations += 1;
            if termination == TerminationReason::TimeLimit {
                break;
            }
        }

        let final_objective = range_objective(&parameters, dimensions, &factors);
        objective_evaluations += 1;
        let factor_count = factors.len();

        Ok(SolveResult {
            agent_ids: problem.agent_ids.clone(),
            positions: points(&parameters, dimensions),
            metrics: SolveMetrics {
                elapsed_micros: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                dimensions,
                iterations: gradient_evaluations,
                objective_evaluations,
                gradient_evaluations,
                factors: factor_count,
                range_factor_evaluations: objective_evaluations
                    .saturating_add(gradient_evaluations)
                    .saturating_add(outer_iterations)
                    .saturating_mul(factor_count),
                coordinate_factor_evaluations: gradient_evaluations
                    .saturating_mul(factor_count)
                    .saturating_mul(dimensions),
                converged: false,
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

fn seed_coincident_components(
    parameters: &mut [f64],
    dimensions: usize,
    components: &[Vec<usize>],
    factors: &[RangeFactor],
) {
    for component in components {
        if component.len() < 2 || !component_is_coincident(parameters, dimensions, component) {
            continue;
        }
        let scale = factors
            .iter()
            .filter(|factor| {
                component.contains(&factor.from)
                    && matches!(factor.to, Endpoint::Agent(to) if component.contains(&to))
            })
            .map(|factor| factor.distance_cm)
            .filter(|distance| distance.is_finite() && *distance > 0.0)
            .fold(1.0_f64, f64::max)
            / 2.0;
        for (order, agent) in component.iter().copied().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * order as f64 / component.len() as f64;
            parameters[agent * dimensions] = scale * angle.cos();
            if dimensions > 1 {
                parameters[agent * dimensions + 1] = scale * angle.sin();
            }
        }
    }
}

fn component_is_coincident(parameters: &[f64], dimensions: usize, component: &[usize]) -> bool {
    let reference = component[0] * dimensions;
    component.iter().skip(1).all(|agent| {
        let offset = agent * dimensions;
        (0..dimensions).all(|dimension| {
            (parameters[offset + dimension] - parameters[reference + dimension]).abs() <= 1e-12
        })
    })
}

fn sphere_targets(parameters: &[f64], dimensions: usize, factors: &[RangeFactor]) -> Vec<[f64; 3]> {
    factors
        .iter()
        .map(|factor| {
            let difference = difference(parameters, dimensions, factor.from, factor.to);
            let norm = difference[..dimensions]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let direction = if norm > 0.0 {
                difference.map(|value| value / norm)
            } else {
                [1.0, 0.0, 0.0]
            };
            direction.map(|value| value * factor.distance_cm)
        })
        .collect()
}

fn surrogate_gradient(
    parameters: &[f64],
    dimensions: usize,
    agents: usize,
    factors: &[RangeFactor],
    targets: &[[f64; 3]],
) -> Vec<f64> {
    let mut gradient = vec![0.0; agents * dimensions];
    for (factor, target) in factors.iter().zip(targets) {
        let difference = difference(parameters, dimensions, factor.from, factor.to);
        let weight = factor.inverse_scale * factor.inverse_scale;
        let from = factor.from * dimensions;
        for dimension in 0..dimensions {
            let value = weight * (difference[dimension] - target[dimension]);
            gradient[from + dimension] += value;
            if let Endpoint::Agent(to) = factor.to {
                gradient[to * dimensions + dimension] -= value;
            }
        }
    }
    gradient
}

fn difference(parameters: &[f64], dimensions: usize, from: usize, to: Endpoint) -> [f64; 3] {
    let from = from * dimensions;
    let mut difference = [0.0; 3];
    match to {
        Endpoint::Agent(to) => {
            let to = to * dimensions;
            for dimension in 0..dimensions {
                difference[dimension] = parameters[from + dimension] - parameters[to + dimension];
            }
        }
        Endpoint::Anchor(anchor) => {
            let anchor = [anchor.position.x, anchor.position.y, anchor.height_cm];
            for dimension in 0..dimensions {
                difference[dimension] = parameters[from + dimension] - anchor[dimension];
            }
        }
    }
    difference
}

fn quadratic_lipschitz(agents: usize, factors: &[RangeFactor]) -> f64 {
    let mut matrix = DMatrix::<f64>::zeros(agents, agents);
    for factor in factors {
        let weight = factor.inverse_scale * factor.inverse_scale;
        matrix[(factor.from, factor.from)] += weight;
        if let Endpoint::Agent(to) = factor.to {
            matrix[(to, to)] += weight;
            matrix[(factor.from, to)] -= weight;
            matrix[(to, factor.from)] -= weight;
        }
    }
    SymmetricEigen::new(matrix)
        .eigenvalues
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
}

fn range_objective(parameters: &[f64], dimensions: usize, factors: &[RangeFactor]) -> f64 {
    factors
        .iter()
        .map(|factor| {
            let difference = difference(parameters, dimensions, factor.from, factor.to);
            let distance = difference[..dimensions]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            let residual = (distance - factor.distance_cm) * factor.inverse_scale;
            0.5 * residual * residual
        })
        .sum()
}

fn scheduled_inner_steps(outer_iteration: usize) -> usize {
    let exponent = outer_iteration / INNER_GROWTH_PERIOD;
    INITIAL_INNER_STEPS
        .saturating_add(1usize.checked_shl(exponent as u32).unwrap_or(usize::MAX))
        .saturating_sub(1)
}

fn time_limit_reached(started: Instant, config: &SolveConfig) -> bool {
    config.max_time_micros > 0 && started.elapsed().as_micros() >= config.max_time_micros as u128
}

#[cfg(test)]
mod tests {
    use solid_core::{
        Anchor, CalibratedRangeSample, LocalizationMode, Point2, PreparedProblem, RangeObservation,
        SolveConfig,
    };

    use super::*;
    use crate::{BaselineInitialization, MethodConfig, MethodRunner};
    use solid_core::LocalizationEngine;

    #[test]
    fn solves_exact_anchor_geometry() {
        let problem = anchored_problem();
        let config = SolveConfig {
            max_iterations: 2_000,
            gradient_tolerance: 1e-7,
            ..SolveConfig::default()
        };

        let result = FastNetworkLocalization.solve(&problem, &config).unwrap();

        assert!((result.positions[0].distance(result.positions[1]) - 100.0).abs() < 0.1);
        assert_eq!(result.metrics.iterations, 2_000);
        assert_eq!(
            result.metrics.termination,
            TerminationReason::IterationLimit
        );
    }

    #[test]
    fn released_initialization_preserves_default_results() {
        let problem = anchored_problem();
        let config = SolveConfig {
            max_iterations: 200,
            ..SolveConfig::default()
        };
        let expected = FastNetworkLocalization.solve(&problem, &config).unwrap();
        for runner in [
            MethodRunner::new(MethodKind::Fnl),
            MethodRunner::configured(
                MethodKind::Fnl,
                MethodConfig::Fnl {
                    initialization: BaselineInitialization::Released,
                },
            )
            .unwrap(),
        ] {
            let mut actual = runner.solve(&problem, &config).unwrap();
            actual.metrics.elapsed_micros = expected.metrics.elapsed_micros;
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn solid_geometry_initialization_only_replaces_the_fnl_seed() {
        let problem = anchored_problem();
        let original = problem.clone();
        let config = SolveConfig {
            max_iterations: 200,
            ..SolveConfig::default()
        };
        let mut seeded = problem.clone();
        seeded.initial_positions = LocalizationEngine::new()
            .initial_layout(&problem, &config)
            .unwrap();
        assert_ne!(seeded.initial_positions, problem.initial_positions);
        let expected = FastNetworkLocalization.solve(&seeded, &config).unwrap();
        let runner = MethodRunner::configured(
            MethodKind::Fnl,
            MethodConfig::Fnl {
                initialization: BaselineInitialization::SolidGeometry,
            },
        )
        .unwrap();
        let mut actual = runner.solve(&problem, &config).unwrap();

        actual.metrics.elapsed_micros = expected.metrics.elapsed_micros;
        assert_eq!(actual, expected);
        assert_eq!(problem, original);
    }

    fn anchored_problem() -> PreparedProblem {
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
    fn solves_anchor_free_geometry_from_a_coincident_start() {
        let truth = [
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 200.0),
            Point2::new(160.0, 80.0),
        ];
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2, 3],
            anchors: Vec::new(),
            observations: [(0, 1), (0, 2), (1, 2)]
                .into_iter()
                .map(|(left, right)| RangeObservation {
                    from: left as u32 + 1,
                    to: right as u32 + 1,
                    tof_ns: Vec::new(),
                    calibrated_samples: vec![CalibratedRangeSample::new(
                        truth[left].distance(truth[right]),
                        5.0,
                    )],
                    to_is_anchor: false,
                })
                .collect(),
            initial_positions: vec![Point2::default(); 3],
            mode: LocalizationMode::AnchorFree,
        };
        let config = SolveConfig {
            max_iterations: 10_000,
            gradient_tolerance: 1e-8,
            ..SolveConfig::default()
        };

        let result = FastNetworkLocalization.solve(&problem, &config).unwrap();
        let mut translated_start = problem.clone();
        translated_start.initial_positions = vec![Point2::new(7_000.0, -3_000.0); 3];
        let repeated = FastNetworkLocalization
            .solve(&translated_start, &config)
            .unwrap();

        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            let expected = truth[left].distance(truth[right]);
            let actual = result.positions[left].distance(result.positions[right]);
            let repeated = repeated.positions[left].distance(repeated.positions[right]);
            assert!(
                (actual - expected).abs() < 0.1,
                "{left}-{right}: {actual} vs {expected}"
            );
            assert!((actual - repeated).abs() < 1e-9);
        }
    }

    #[test]
    fn uses_the_published_inner_iteration_schedule() {
        assert_eq!(scheduled_inner_steps(0), 40);
        assert_eq!(scheduled_inner_steps(999), 40);
        assert_eq!(scheduled_inner_steps(1_000), 41);
    }

    #[test]
    fn central_lipschitz_constant_is_the_largest_eigenvalue() {
        let factors = vec![
            RangeFactor {
                from: 0,
                to: Endpoint::Agent(1),
                distance_cm: 1.0,
                inverse_scale: 1.0,
            },
            RangeFactor {
                from: 0,
                to: Endpoint::Anchor(anchor(10, 0.0, 0.0)),
                distance_cm: 1.0,
                inverse_scale: 1.0,
            },
        ];

        let value = quadratic_lipschitz(2, &factors);

        assert!((value - (3.0 + 5.0_f64.sqrt()) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn coincident_positions_use_the_published_first_axis_direction() {
        let factors = vec![RangeFactor {
            from: 0,
            to: Endpoint::Agent(1),
            distance_cm: 5.0,
            inverse_scale: 1.0,
        }];

        let targets = sphere_targets(&[0.0, 0.0, 0.0, 0.0], 2, &factors);

        assert_eq!(targets, vec![[5.0, 0.0, 0.0]]);
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
