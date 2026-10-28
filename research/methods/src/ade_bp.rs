use std::time::Instant;

use solid_core::{
    LocalizationMode, PreparedProblem, SolveConfig, SolveMetrics, SolveResult, TerminationReason,
};

use crate::bayesian::{
    log_ade_gaussian_message, normalized_log_weights, problem_seed, resample_gaussian,
    resolve_area, GaussianBelief, Network, Node, ParticleBelief, Rng,
};
use crate::{
    AdeRangingModel, AreaBounds, LocalizationMethod, MethodConfig, MethodError, MethodKind,
};

const METHOD_SALT: u64 = 0x4144_452d_4250_2015;
const MEAN_CONVERGENCE_FRACTION: f64 = 1e-3;
const COVARIANCE_CONVERGENCE_FRACTION: f64 = 1e-4;
const REQUIRED_STABLE_REFINEMENTS: usize = 2;

pub struct EmpiricalAdeBp {
    area: Option<AreaBounds>,
    particles: usize,
    iterations: usize,
    maximum_importance_iterations: usize,
    ranging_model: AdeRangingModel,
}

impl EmpiricalAdeBp {
    pub fn from_config(config: &MethodConfig) -> Self {
        let MethodConfig::AdeBp2015 {
            area,
            particles,
            iterations,
            importance_iterations,
            ranging_model,
        } = config
        else {
            unreachable!("ADE-BP received another method's configuration")
        };
        Self {
            area: *area,
            particles: *particles,
            iterations: *iterations,
            maximum_importance_iterations: *importance_iterations,
            ranging_model: *ranging_model,
        }
    }

    fn update_belief(
        &self,
        agent: usize,
        state: &AdeState,
        network: &Network,
        problem: &PreparedProblem,
        neighbor_beliefs: &[crate::bayesian::GaussianBelief],
    ) -> ParticleBelief {
        let log_weights = state
            .points
            .iter()
            .zip(&state.log_importance)
            .map(|(point, importance)| {
                let mut value = *importance;
                for link_index in &network.incident[agent] {
                    let link = &network.links[*link_index];
                    match network.other(*link_index, agent) {
                        Node::Anchor(anchor) => {
                            let anchor = problem.anchors[anchor].position;
                            let distance = point.distance(anchor);
                            value += link
                                .samples
                                .iter()
                                .map(|sample| {
                                    self.ranging_model.log_density(
                                        sample.distance_cm - distance,
                                        sample.distance_cm,
                                    )
                                })
                                .sum::<f64>();
                        }
                        Node::Agent(neighbor) => {
                            value += link
                                .samples
                                .iter()
                                .map(|sample| {
                                    log_ade_gaussian_message(
                                        *point,
                                        neighbor_beliefs[neighbor],
                                        sample.distance_cm,
                                        self.ranging_model,
                                    )
                                })
                                .sum::<f64>();
                        }
                    }
                }
                value
            })
            .collect::<Vec<_>>();
        ParticleBelief {
            points: state.points.clone(),
            weights: normalized_log_weights(&log_weights),
        }
    }
}

fn importance_converged(
    previous: GaussianBelief,
    current: GaussianBelief,
    area: AreaBounds,
) -> bool {
    let position_scale = area.diameter().max(1.0);
    let covariance_scale = position_scale * position_scale;
    let mean_change = previous.mean.distance(current.mean) / position_scale;
    let xx = current.xx - previous.xx;
    let xy = current.xy - previous.xy;
    let yy = current.yy - previous.yy;
    let covariance_change = (xx * xx + 2.0 * xy * xy + yy * yy).sqrt() / covariance_scale;

    mean_change <= MEAN_CONVERGENCE_FRACTION && covariance_change <= COVARIANCE_CONVERGENCE_FRACTION
}

impl LocalizationMethod for EmpiricalAdeBp {
    fn kind(&self) -> MethodKind {
        MethodKind::AdeBp2015
    }

    fn solve(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError> {
        if problem.mode == LocalizationMode::AnchorFree {
            return Err(MethodError::UnsupportedMode(
                "Empirical ADE-BP implements the paper's anchored localization procedure"
                    .to_owned(),
            ));
        }
        let started = Instant::now();
        let network = Network::prepare(problem, config)?;
        let area = resolve_area(problem, self.area, &network)?;
        let mut rng = Rng::new(problem_seed(problem, METHOD_SALT));
        let mut states = (0..problem.agent_ids.len())
            .map(|_| AdeState {
                points: (0..self.particles).map(|_| area.sample(&mut rng)).collect(),
                log_importance: vec![0.0; self.particles],
            })
            .collect::<Vec<_>>();
        let mut beliefs = states
            .iter()
            .map(|state| state.uniform_belief())
            .collect::<Vec<_>>();
        let mut numerical_iterations = 0usize;
        let mut objective_evaluations = 0usize;
        let mut all_importance_updates_converged = true;

        for iteration in 0..self.iterations {
            let neighbor_beliefs = beliefs
                .iter()
                .map(ParticleBelief::importance_gaussian)
                .collect::<Vec<_>>();
            let prior_states = std::mem::take(&mut states);
            let mut next_states = Vec::with_capacity(prior_states.len());
            let mut next_beliefs = Vec::with_capacity(prior_states.len());
            let mut round_iterations = 0usize;
            for (agent, mut state) in prior_states.into_iter().enumerate() {
                let mut belief =
                    self.update_belief(agent, &state, &network, problem, &neighbor_beliefs);
                let mut refinements = 1usize;
                let mut stable_refinements = 0usize;
                let mut previous_gaussian = belief.importance_gaussian();
                for _ in 1..self.maximum_importance_iterations {
                    state = AdeState::resampled(&belief, area, self.particles, &mut rng);
                    belief =
                        self.update_belief(agent, &state, &network, problem, &neighbor_beliefs);
                    refinements += 1;
                    let current_gaussian = belief.importance_gaussian();
                    if importance_converged(previous_gaussian, current_gaussian, area) {
                        stable_refinements += 1;
                    } else {
                        stable_refinements = 0;
                    }
                    previous_gaussian = current_gaussian;
                    if stable_refinements == REQUIRED_STABLE_REFINEMENTS {
                        break;
                    }
                }
                round_iterations = round_iterations.max(refinements);
                objective_evaluations = objective_evaluations
                    .saturating_add(refinements.saturating_mul(self.particles));
                all_importance_updates_converged &=
                    stable_refinements == REQUIRED_STABLE_REFINEMENTS;
                if iteration + 1 < self.iterations {
                    state = AdeState::resampled(&belief, area, self.particles, &mut rng);
                }
                next_states.push(state);
                next_beliefs.push(belief);
            }
            numerical_iterations = numerical_iterations.saturating_add(round_iterations);
            states = next_states;
            beliefs = next_beliefs;
        }

        let termination = if all_importance_updates_converged {
            TerminationReason::Converged
        } else {
            TerminationReason::IterationLimit
        };

        Ok(SolveResult {
            agent_ids: problem.agent_ids.clone(),
            positions: beliefs.iter().map(ParticleBelief::mean).collect(),
            metrics: SolveMetrics {
                elapsed_micros: started.elapsed().as_micros().min(u64::MAX as u128) as u64,
                dimensions: 2,
                iterations: numerical_iterations,
                objective_evaluations,
                gradient_evaluations: 0,
                factors: network.links.len(),
                range_factor_evaluations: numerical_iterations
                    .saturating_mul(self.particles)
                    .saturating_mul(network.links.len()),
                coordinate_factor_evaluations: 0,
                converged: all_importance_updates_converged,
                termination,
                final_objective: 0.0,
                starts: 1,
                guarded_supports: None,
                largest_dropped_mass: None,
                posterior_nodes: None,
                posterior_cells: None,
            },
        })
    }
}

struct AdeState {
    points: Vec<solid_core::Point2>,
    log_importance: Vec<f64>,
}

impl AdeState {
    fn uniform_belief(&self) -> ParticleBelief {
        ParticleBelief {
            points: self.points.clone(),
            weights: vec![1.0 / self.points.len() as f64; self.points.len()],
        }
    }

    fn resampled(
        belief: &ParticleBelief,
        area: AreaBounds,
        particles: usize,
        rng: &mut Rng,
    ) -> Self {
        let gaussian = belief.importance_gaussian();
        let (points, log_importance) = resample_gaussian(gaussian, area, particles, rng);
        Self {
            points,
            log_importance,
        }
    }
}

#[cfg(test)]
mod tests {
    use solid_core::{Anchor, CalibratedRangeSample, Point2, RangeObservation, SolveConfig};

    use super::*;

    #[test]
    fn locates_an_anchor_only_node_without_evaluation_truth() {
        let truth = Point2::new(120.0, 90.0);
        let anchors = [
            Point2::new(0.0, 0.0),
            Point2::new(300.0, 0.0),
            Point2::new(0.0, 300.0),
            Point2::new(300.0, 300.0),
        ];
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1],
            anchors: anchors
                .iter()
                .enumerate()
                .map(|(index, position)| Anchor {
                    id: 10 + index as u32,
                    position: *position,
                    height_cm: 0.0,
                })
                .collect(),
            observations: anchors
                .iter()
                .enumerate()
                .map(|(index, anchor)| RangeObservation {
                    from: 1,
                    to: 10 + index as u32,
                    tof_ns: Vec::new(),
                    calibrated_samples: vec![CalibratedRangeSample::new(
                        truth.distance(*anchor),
                        5.0,
                    )],
                    to_is_anchor: true,
                })
                .collect(),
            initial_positions: vec![Point2::default()],
            mode: LocalizationMode::AnchorOnly,
        };
        let config = MethodConfig::AdeBp2015 {
            area: Some(AreaBounds {
                minimum_x_cm: -50.0,
                maximum_x_cm: 350.0,
                minimum_y_cm: -50.0,
                maximum_y_cm: 350.0,
            }),
            particles: 2_000,
            iterations: 4,
            importance_iterations: 64,
            ranging_model: AdeRangingModel {
                positive_slope: 0.0,
                positive_intercept_cm: 5.0,
                negative_slope: 0.0,
                negative_intercept_cm: 5.0,
            },
        };

        let method = EmpiricalAdeBp::from_config(&config);
        let result = method.solve(&problem, &SolveConfig::default()).unwrap();
        let repeated = method.solve(&problem, &SolveConfig::default()).unwrap();

        assert!(result.positions[0].distance(truth) < 15.0);
        assert_eq!(result.positions, repeated.positions);
        assert_eq!(result.metrics.iterations, repeated.metrics.iterations);
        assert_eq!(
            result.metrics.objective_evaluations,
            repeated.metrics.objective_evaluations
        );
        assert!((12..=256).contains(&result.metrics.iterations));
        assert_eq!(
            result.metrics.objective_evaluations,
            result.metrics.iterations * 2_000
        );

        let mut capped_config = config;
        let MethodConfig::AdeBp2015 {
            importance_iterations,
            ..
        } = &mut capped_config
        else {
            unreachable!()
        };
        *importance_iterations = 1;
        let capped = EmpiricalAdeBp::from_config(&capped_config)
            .solve(&problem, &SolveConfig::default())
            .unwrap();
        assert!(!capped.metrics.converged);
        assert_eq!(
            capped.metrics.termination,
            TerminationReason::IterationLimit
        );
        assert_eq!(capped.metrics.iterations, 4);
        assert_eq!(capped.metrics.objective_evaluations, 8_000);
    }

    #[test]
    fn rejects_unsupported_anchor_free_mode() {
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2],
            anchors: Vec::new(),
            observations: vec![RangeObservation {
                from: 1,
                to: 2,
                tof_ns: Vec::new(),
                calibrated_samples: vec![CalibratedRangeSample::new(100.0, 5.0)],
                to_is_anchor: false,
            }],
            initial_positions: vec![Point2::default(); 2],
            mode: LocalizationMode::AnchorFree,
        };
        let config = MethodConfig::default_for(MethodKind::AdeBp2015).unwrap();
        let method = EmpiricalAdeBp::from_config(&config);
        assert!(matches!(
            method.solve(&problem, &SolveConfig::default()),
            Err(MethodError::UnsupportedMode(_))
        ));
    }

    #[test]
    fn convergence_is_normalized_to_the_search_area() {
        let area = AreaBounds {
            minimum_x_cm: 0.0,
            maximum_x_cm: 300.0,
            minimum_y_cm: 0.0,
            maximum_y_cm: 400.0,
        };
        let reference = GaussianBelief {
            mean: Point2::new(100.0, 100.0),
            xx: 400.0,
            xy: 25.0,
            yy: 625.0,
        };
        let stable = GaussianBelief {
            mean: Point2::new(100.2, 100.1),
            xx: 405.0,
            xy: 27.0,
            yy: 619.0,
        };
        let shifted_mean = GaussianBelief {
            mean: Point2::new(101.0, 100.0),
            ..reference
        };
        let shifted_covariance = GaussianBelief {
            xx: 450.0,
            ..reference
        };

        assert!(importance_converged(reference, stable, area));
        assert!(!importance_converged(reference, shifted_mean, area));
        assert!(!importance_converged(reference, shifted_covariance, area));
    }
}
