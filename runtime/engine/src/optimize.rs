use std::time::Instant;

use crate::{SolveConfig, TerminationReason};

pub(crate) trait DifferentiableObjective {
    fn value_gradient(&mut self, parameters: &[f64], gradient: &mut [f64]) -> f64;
}

pub(crate) struct OptimizationReport {
    pub parameters: Vec<f64>,
    pub iterations: usize,
    pub evaluations: usize,
    pub converged: bool,
    pub termination: TerminationReason,
}

pub(crate) fn minimize<O: DifferentiableObjective>(
    objective: &mut O,
    initial: Vec<f64>,
    config: &SolveConfig,
) -> OptimizationReport {
    let started = Instant::now();
    let dimension = initial.len();
    let mut parameters = initial;
    let mut gradient = vec![0.0; dimension];
    let mut value = objective.value_gradient(&parameters, &mut gradient);
    let mut evaluations = 1;
    let mut s_history: Vec<Vec<f64>> = Vec::with_capacity(config.history_size);
    let mut y_history: Vec<Vec<f64>> = Vec::with_capacity(config.history_size);
    let mut rho_history = Vec::with_capacity(config.history_size);

    for iteration in 0..config.max_iterations {
        if infinity_norm(&gradient) <= config.gradient_tolerance {
            return OptimizationReport {
                parameters,
                iterations: iteration,
                evaluations,
                converged: true,
                termination: TerminationReason::Converged,
            };
        }
        if time_limit_reached(started, config) {
            return OptimizationReport {
                parameters,
                iterations: iteration,
                evaluations,
                converged: false,
                termination: TerminationReason::TimeLimit,
            };
        }

        let mut direction = lbfgs_direction(&gradient, &s_history, &y_history, &rho_history);
        let mut directional_derivative = dot(&gradient, &direction);
        if !directional_derivative.is_finite() || directional_derivative >= 0.0 {
            direction
                .iter_mut()
                .zip(&gradient)
                .for_each(|(direction, gradient)| *direction = -*gradient);
            directional_derivative = -dot(&gradient, &gradient);
            s_history.clear();
            y_history.clear();
            rho_history.clear();
        }

        let mut step = 1.0;
        let mut candidate = vec![0.0; dimension];
        let mut candidate_gradient = vec![0.0; dimension];
        let mut accepted = false;
        let mut candidate_value = value;

        for _ in 0..30 {
            if time_limit_reached(started, config) {
                return OptimizationReport {
                    parameters,
                    iterations: iteration,
                    evaluations,
                    converged: false,
                    termination: TerminationReason::TimeLimit,
                };
            }
            candidate
                .iter_mut()
                .zip(parameters.iter().zip(&direction))
                .for_each(|(candidate, (parameter, direction))| {
                    *candidate = *parameter + step * *direction
                });
            candidate_value = objective.value_gradient(&candidate, &mut candidate_gradient);
            evaluations += 1;

            if candidate_value.is_finite()
                && candidate_value <= value + 1e-4 * step * directional_derivative
            {
                accepted = true;
                break;
            }
            step *= 0.5;
        }

        if !accepted {
            return OptimizationReport {
                parameters,
                iterations: iteration + 1,
                evaluations,
                converged: false,
                termination: TerminationReason::LineSearchFailed,
            };
        }

        let s: Vec<f64> = candidate
            .iter()
            .zip(&parameters)
            .map(|(next, current)| next - current)
            .collect();
        let y: Vec<f64> = candidate_gradient
            .iter()
            .zip(&gradient)
            .map(|(next, current)| next - current)
            .collect();
        let curvature = dot(&s, &y);
        if curvature > 1e-10 * dot(&s, &s).sqrt() * dot(&y, &y).sqrt() {
            if s_history.len() == config.history_size {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            rho_history.push(1.0 / curvature);
            s_history.push(s);
            y_history.push(y);
        }

        parameters = candidate;
        gradient = candidate_gradient;
        value = candidate_value;
    }

    let converged = infinity_norm(&gradient) <= config.gradient_tolerance;
    OptimizationReport {
        parameters,
        iterations: config.max_iterations,
        evaluations,
        converged,
        termination: if converged {
            TerminationReason::Converged
        } else {
            TerminationReason::IterationLimit
        },
    }
}

fn time_limit_reached(started: Instant, config: &SolveConfig) -> bool {
    config.max_time_micros > 0 && started.elapsed().as_micros() >= config.max_time_micros as u128
}

fn lbfgs_direction(
    gradient: &[f64],
    s_history: &[Vec<f64>],
    y_history: &[Vec<f64>],
    rho_history: &[f64],
) -> Vec<f64> {
    let mut q = gradient.to_vec();
    let mut alpha = vec![0.0; s_history.len()];

    for index in (0..s_history.len()).rev() {
        alpha[index] = rho_history[index] * dot(&s_history[index], &q);
        add_scaled(&mut q, &y_history[index], -alpha[index]);
    }

    if let (Some(s), Some(y)) = (s_history.last(), y_history.last()) {
        let scaling = dot(s, y) / dot(y, y).max(1e-20);
        q.iter_mut().for_each(|value| *value *= scaling);
    }

    for index in 0..s_history.len() {
        let beta = rho_history[index] * dot(&y_history[index], &q);
        add_scaled(&mut q, &s_history[index], alpha[index] - beta);
    }

    q.iter_mut().for_each(|value| *value = -*value);
    q
}

fn add_scaled(target: &mut [f64], source: &[f64], scale: f64) {
    target
        .iter_mut()
        .zip(source)
        .for_each(|(target, source)| *target += scale * source);
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn infinity_norm(values: &[f64]) -> f64 {
    values
        .iter()
        .fold(0.0_f64, |norm, value| norm.max(value.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Quadratic;

    impl DifferentiableObjective for Quadratic {
        fn value_gradient(&mut self, parameters: &[f64], gradient: &mut [f64]) -> f64 {
            gradient[0] = 2.0 * (parameters[0] - 3.0);
            gradient[1] = 4.0 * (parameters[1] + 2.0);
            (parameters[0] - 3.0).powi(2) + 2.0 * (parameters[1] + 2.0).powi(2)
        }
    }

    #[test]
    fn minimizes_quadratic() {
        let report = minimize(&mut Quadratic, vec![20.0, 20.0], &SolveConfig::default());
        assert!(report.converged);
        assert!((report.parameters[0] - 3.0).abs() < 1e-6);
        assert!((report.parameters[1] + 2.0).abs() < 1e-6);
    }

    #[test]
    fn retains_small_positive_curvature() {
        struct ScaledQuadratic(f64);

        impl DifferentiableObjective for ScaledQuadratic {
            fn value_gradient(&mut self, parameters: &[f64], gradient: &mut [f64]) -> f64 {
                for (x, g) in parameters.iter().zip(gradient) {
                    *g = self.0 * x;
                }
                0.5 * self.0 * dot(parameters, parameters)
            }
        }

        let config = SolveConfig {
            max_iterations: 10,
            max_time_micros: 0,
            ..SolveConfig::production()
        };
        for scale in [1e-4, 4e-4, 1.0] {
            let report = minimize(&mut ScaledQuadratic(scale), vec![0.5, 0.3, 0.1], &config);
            assert!(report.converged, "scale {scale}");
            assert!(report.iterations <= 2);
            assert!(infinity_norm(&report.parameters) < 1e-8);
        }
    }
}
