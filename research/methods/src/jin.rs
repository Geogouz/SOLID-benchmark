use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use clarabel::algebra::CscMatrix;
use clarabel::solver::{
    DefaultSettingsBuilder, DefaultSolver, IPSolver, NonnegativeConeT, PSDTriangleConeT,
    SolverStatus, ZeroConeT,
};
use nalgebra::{DMatrix, SymmetricEigen};
use serde::{Deserialize, Serialize};
use solid_core::{
    LocalizationMode, PreparedProblem, SolveConfig, SolveMetrics, SolveResult, TerminationReason,
};

use crate::geometry::{
    agent_components, dimensions, prepare_undirected_means, prepare_undirected_medians, Endpoint,
    ProblemScale, RangeFactor,
};
use crate::{LocalizationMethod, MethodConfig, MethodError, MethodKind};

pub struct JinSparseBias {
    residual_multiplier: f64,
    range_aggregation: JinRangeAggregation,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JinRangeAggregation {
    #[default]
    Median,
    Mean,
}

impl JinRangeAggregation {
    pub(crate) fn is_median(&self) -> bool {
        *self == Self::Median
    }
}

impl JinSparseBias {
    pub(crate) fn from_config(config: &MethodConfig) -> Self {
        let MethodConfig::Jin2021 {
            residual_multiplier,
            range_aggregation,
        } = config
        else {
            unreachable!("Jin receives its own method configuration")
        };
        Self {
            residual_multiplier: *residual_multiplier,
            range_aggregation: *range_aggregation,
        }
    }
}

impl LocalizationMethod for JinSparseBias {
    fn kind(&self) -> MethodKind {
        MethodKind::Jin2021
    }

    fn solve(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError> {
        let started = Instant::now();
        let dimensions = dimensions(problem, config);
        let factors = match self.range_aggregation {
            JinRangeAggregation::Median => prepare_undirected_medians(problem, config)?,
            JinRangeAggregation::Mean => prepare_undirected_means(problem, config)?,
        };
        let scale = if problem.mode == LocalizationMode::AnchorFree {
            ProblemScale::from_relative_factors(&factors)?
        } else {
            ProblemScale::from_problem(problem, &factors, dimensions)?
        };
        let normalized = normalize_factors(&factors, &scale, dimensions);
        let minimum_residual = solve_sdp(
            problem.agent_ids.len(),
            dimensions,
            &normalized,
            None,
            remaining_seconds(started, config),
        )?;
        let residual_bound = self.residual_multiplier * minimum_residual.objective.max(1e-10);
        let sparse_bias = solve_sdp(
            problem.agent_ids.len(),
            dimensions,
            &normalized,
            Some(residual_bound),
            remaining_seconds(started, config),
        )?;
        let positions = if problem.mode == LocalizationMode::AnchorFree {
            decode_relative_positions(
                &sparse_bias.psd,
                problem.agent_ids.len(),
                dimensions,
                &factors,
                &scale,
            )
        } else {
            (0..problem.agent_ids.len())
                .map(|agent| {
                    let point = (0..dimensions)
                        .map(|dimension| {
                            matrix_entry(&sparse_bias.psd, dimension, dimensions + agent)
                        })
                        .collect::<Vec<_>>();
                    scale.decode_point(&point)
                })
                .collect()
        };
        let iterations = minimum_residual.iterations + sparse_bias.iterations;
        let converged = minimum_residual.solved && sparse_bias.solved;
        let termination = if converged {
            TerminationReason::Converged
        } else if time_limit_reached(started, config) {
            TerminationReason::TimeLimit
        } else {
            TerminationReason::IterationLimit
        };
        let factor_count = factors.len();

        Ok(SolveResult {
            agent_ids: problem.agent_ids.clone(),
            positions,
            metrics: SolveMetrics {
                elapsed_micros: elapsed_micros(started),
                dimensions,
                iterations,
                objective_evaluations: 2,
                gradient_evaluations: 0,
                factors: factor_count,
                range_factor_evaluations: factor_count.saturating_mul(2),
                coordinate_factor_evaluations: iterations
                    .saturating_mul(factor_count)
                    .saturating_mul(dimensions),
                converged,
                termination,
                final_objective: sparse_bias.objective,
                starts: 1,
                guarded_supports: None,
                largest_dropped_mass: None,
                posterior_nodes: None,
                posterior_cells: None,
            },
        })
    }
}

fn decode_relative_positions(
    psd: &[f64],
    agents: usize,
    dimensions: usize,
    factors: &[RangeFactor],
    scale: &ProblemScale,
) -> Vec<solid_core::Point2> {
    let mut positions = vec![solid_core::Point2::default(); agents];
    for component in agent_components(agents, factors) {
        if component.len() < 2 {
            continue;
        }
        let size = component.len();
        let mut gram = DMatrix::from_fn(size, size, |row, column| {
            matrix_entry(
                psd,
                dimensions + component[row],
                dimensions + component[column],
            )
        });
        let centering =
            DMatrix::identity(size, size) - DMatrix::from_element(size, size, 1.0 / size as f64);
        gram = &centering * gram * &centering;
        let eigen = SymmetricEigen::new(0.5 * (&gram + gram.transpose()));
        let mut order = (0..size).collect::<Vec<_>>();
        order.sort_by(|left, right| eigen.eigenvalues[*right].total_cmp(&eigen.eigenvalues[*left]));
        let mut coordinates = DMatrix::zeros(size, dimensions);
        for (target, source) in order.into_iter().take(dimensions).enumerate() {
            let magnitude = eigen.eigenvalues[source].max(0.0).sqrt();
            coordinates.set_column(target, &(eigen.eigenvectors.column(source) * magnitude));
        }
        for (row, agent) in component.into_iter().enumerate() {
            let point = (0..dimensions)
                .map(|dimension| coordinates[(row, dimension)])
                .collect::<Vec<_>>();
            positions[agent] = scale.decode_relative_point(&point);
        }
    }
    positions
}

#[derive(Clone)]
struct NormalizedFactor {
    from: usize,
    to: NormalizedEndpoint,
    distance: f64,
}

#[derive(Clone)]
enum NormalizedEndpoint {
    Agent(usize),
    Anchor([f64; 3]),
}

struct SdpSolution {
    psd: Vec<f64>,
    objective: f64,
    iterations: usize,
    solved: bool,
}

fn normalize_factors(
    factors: &[RangeFactor],
    scale: &ProblemScale,
    dimensions: usize,
) -> Vec<NormalizedFactor> {
    factors
        .iter()
        .map(|factor| NormalizedFactor {
            from: factor.from,
            to: match factor.to {
                Endpoint::Agent(agent) => NormalizedEndpoint::Agent(agent),
                Endpoint::Anchor(anchor) => {
                    let mut position = scale.encode_anchor(anchor);
                    for value in position.iter_mut().skip(dimensions) {
                        *value = 0.0;
                    }
                    NormalizedEndpoint::Anchor(position)
                }
            },
            distance: scale.encode_distance(factor.distance_cm),
        })
        .collect()
}

/// One Clarabel solve at a time in the process. On Linux Clarabel's PSD cone calls OpenBLAS, which the build compiles
/// from source as a single-threaded library. Built without locking, such a library shares its work buffers between the
/// threads that call it, so two solves on two threads of one process overwrite each other's buffers and a solve can
/// return another result or fail. The repository's cargo configuration builds it with locking (.cargo/config.toml), but
/// a build directory keeps the library it compiled first, so every Clarabel solve of the process, its setup included,
/// holds this lock whatever the library. The benchmark runners solve in single-threaded processes and never wait for
/// it. The threads of one process take turns, the tests and a CLI with more than one worker among them, and a solve's
/// time budget, which every benchmark run leaves unlimited, counts the wait.
static CLARABEL: Mutex<()> = Mutex::new(());

fn solve_sdp(
    agents: usize,
    dimensions: usize,
    factors: &[NormalizedFactor],
    residual_bound: Option<f64>,
    time_limit: f64,
) -> Result<SdpSolution, MethodError> {
    let links = factors.len();
    let matrix_dimension = dimensions + agents;
    let psd_variables = triangular_number(matrix_dimension);
    let nonnegative_variables = 3 * links;
    let variables = nonnegative_variables + psd_variables;
    let top_block_equalities = triangular_number(dimensions);
    let rows = nonnegative_variables
        + psd_variables
        + links
        + top_block_equalities
        + usize::from(residual_bound.is_some());
    let epsilon_offset = 0;
    let positive_offset = links;
    let negative_offset = 2 * links;
    let psd_offset = 3 * links;
    let mut row_indexes = Vec::new();
    let mut column_indexes = Vec::new();
    let mut values = Vec::new();
    let mut b = vec![0.0; rows];

    for variable in 0..nonnegative_variables {
        push_triplet(
            &mut row_indexes,
            &mut column_indexes,
            &mut values,
            variable,
            variable,
            -1.0,
        );
    }
    let psd_row = nonnegative_variables;
    for variable in 0..psd_variables {
        push_triplet(
            &mut row_indexes,
            &mut column_indexes,
            &mut values,
            psd_row + variable,
            psd_offset + variable,
            -1.0,
        );
    }
    let equality_row = psd_row + psd_variables;
    for (index, factor) in factors.iter().enumerate() {
        let row = equality_row + index;
        push_triplet(
            &mut row_indexes,
            &mut column_indexes,
            &mut values,
            row,
            epsilon_offset + index,
            1.0,
        );
        push_triplet(
            &mut row_indexes,
            &mut column_indexes,
            &mut values,
            row,
            positive_offset + index,
            1.0,
        );
        push_triplet(
            &mut row_indexes,
            &mut column_indexes,
            &mut values,
            row,
            negative_offset + index,
            -1.0,
        );
        let vector = factor_vector(factor, agents, dimensions);
        append_trace_coefficients(
            &mut row_indexes,
            &mut column_indexes,
            &mut values,
            row,
            psd_offset,
            &vector,
        );
        b[row] = factor.distance * factor.distance;
    }
    let top_block_row = equality_row + links;
    let mut equality = 0;
    for column in 0..dimensions {
        for row_in_matrix in 0..=column {
            let row = top_block_row + equality;
            push_triplet(
                &mut row_indexes,
                &mut column_indexes,
                &mut values,
                row,
                psd_offset + svec_index(row_in_matrix, column),
                1.0,
            );
            b[row] = f64::from(row_in_matrix == column);
            equality += 1;
        }
    }
    if let Some(bound) = residual_bound {
        let row = top_block_row + top_block_equalities;
        for index in 0..links {
            push_triplet(
                &mut row_indexes,
                &mut column_indexes,
                &mut values,
                row,
                positive_offset + index,
                1.0,
            );
            push_triplet(
                &mut row_indexes,
                &mut column_indexes,
                &mut values,
                row,
                negative_offset + index,
                1.0,
            );
        }
        b[row] = bound;
    }

    let mut q = vec![0.0; variables];
    if residual_bound.is_some() {
        q[epsilon_offset..positive_offset].fill(1.0);
    } else {
        q[positive_offset..psd_offset].fill(1.0);
    }
    let p = CscMatrix::zeros((variables, variables));
    let a = CscMatrix::new_from_triplets(rows, variables, row_indexes, column_indexes, values);
    let mut cones = vec![
        NonnegativeConeT(nonnegative_variables),
        PSDTriangleConeT(matrix_dimension),
        ZeroConeT(links + top_block_equalities),
    ];
    if residual_bound.is_some() {
        cones.push(NonnegativeConeT(1));
    }
    let settings = DefaultSettingsBuilder::default()
        .verbose(false)
        .max_iter(200)
        .time_limit(time_limit)
        .tol_gap_abs(1e-7)
        .tol_gap_rel(1e-7)
        .tol_feas(1e-7)
        .build()
        .map_err(|error| MethodError::Solver(error.to_string()))?;
    let solver = {
        // the lock guards no data, so a solve that panicked while holding it leaves nothing to repair
        let _turn = CLARABEL.lock().unwrap_or_else(PoisonError::into_inner);
        let mut solver = DefaultSolver::new(&p, &q, &a, &b, &cones, settings)
            .map_err(|error| MethodError::Solver(error.to_string()))?;
        solver.solve();
        solver
    };
    let status = solver.solution.status;
    let usable = matches!(
        status,
        SolverStatus::Solved
            | SolverStatus::AlmostSolved
            | SolverStatus::MaxIterations
            | SolverStatus::MaxTime
    ) && solver.solution.x.iter().all(|value| value.is_finite());
    if !usable {
        return Err(MethodError::Solver(format!(
            "Jin SDP terminated with {status}"
        )));
    }
    Ok(SdpSolution {
        psd: solver.solution.x[psd_offset..].to_vec(),
        objective: solver.solution.obj_val,
        iterations: solver.solution.iterations as usize,
        solved: matches!(status, SolverStatus::Solved | SolverStatus::AlmostSolved),
    })
}

fn factor_vector(factor: &NormalizedFactor, agents: usize, dimensions: usize) -> Vec<f64> {
    let mut vector = vec![0.0; dimensions + agents];
    match &factor.to {
        NormalizedEndpoint::Agent(to) => {
            vector[dimensions + factor.from] = 1.0;
            vector[dimensions + *to] = -1.0;
        }
        NormalizedEndpoint::Anchor(anchor) => {
            vector[..dimensions].copy_from_slice(&anchor[..dimensions]);
            vector[dimensions + factor.from] = -1.0;
        }
    }
    vector
}

fn append_trace_coefficients(
    rows: &mut Vec<usize>,
    columns: &mut Vec<usize>,
    values: &mut Vec<f64>,
    constraint_row: usize,
    psd_offset: usize,
    vector: &[f64],
) {
    for column in 0..vector.len() {
        for row in 0..=column {
            let coefficient = if row == column {
                vector[row] * vector[column]
            } else {
                2.0_f64.sqrt() * vector[row] * vector[column]
            };
            if coefficient != 0.0 {
                push_triplet(
                    rows,
                    columns,
                    values,
                    constraint_row,
                    psd_offset + svec_index(row, column),
                    coefficient,
                );
            }
        }
    }
}

fn push_triplet(
    rows: &mut Vec<usize>,
    columns: &mut Vec<usize>,
    values: &mut Vec<f64>,
    row: usize,
    column: usize,
    value: f64,
) {
    rows.push(row);
    columns.push(column);
    values.push(value);
}

const fn triangular_number(size: usize) -> usize {
    size * (size + 1) / 2
}

const fn svec_index(row: usize, column: usize) -> usize {
    column * (column + 1) / 2 + row
}

fn matrix_entry(values: &[f64], row: usize, column: usize) -> f64 {
    let (row, column) = if row <= column {
        (row, column)
    } else {
        (column, row)
    };
    let value = values[svec_index(row, column)];
    if row == column {
        value
    } else {
        value / 2.0_f64.sqrt()
    }
}

fn remaining_seconds(started: Instant, config: &SolveConfig) -> f64 {
    if config.max_time_micros == 0 {
        f64::INFINITY
    } else {
        (config.max_time_micros as f64 / 1_000_000.0 - started.elapsed().as_secs_f64()).max(1e-6)
    }
}

fn time_limit_reached(started: Instant, config: &SolveConfig) -> bool {
    config.max_time_micros > 0 && started.elapsed().as_micros() >= config.max_time_micros as u128
}

fn elapsed_micros(started: Instant) -> u64 {
    started.elapsed().as_micros().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use solid_core::{Anchor, Point2, RangeObservation};

    use super::*;

    #[test]
    fn solves_an_exact_anchor_only_problem() {
        let result = method().solve(&anchor_only_problem(), &SolveConfig::default()).unwrap();

        assert!((result.positions[0].distance(result.positions[1]) - 100.0).abs() < 1.0);
    }

    #[test]
    fn decodes_an_exact_anchor_free_triangle_from_the_gram_block() {
        let (problem, truth) = anchor_free_triangle();
        let method = method();
        let result = method.solve(&problem, &SolveConfig::default()).unwrap();
        let mut changed_initialization = problem.clone();
        changed_initialization.initial_positions = vec![
            Point2::new(10_000.0, -4_000.0),
            Point2::new(-7_000.0, 3_000.0),
            Point2::new(900.0, 12_000.0),
        ];
        let repeated = method
            .solve(&changed_initialization, &SolveConfig::default())
            .unwrap();

        assert_eq!(result.positions, repeated.positions);
        for (left, right) in [(0, 1), (0, 2), (1, 2)] {
            let expected = truth[left].distance(truth[right]);
            let actual = result.positions[left].distance(result.positions[right]);
            assert!(
                (actual - expected).abs() < 1.0,
                "{left}-{right}: {actual} vs {expected}"
            );
        }
    }

    /// Solves on eight threads of one process at once take turns at Clarabel (CLARABEL), so every one of them returns
    /// the positions of the same problem solved alone, bit for bit. Without the lock the OpenBLAS of a Linux build
    /// without locking changed or failed about one solve in thirty on two threads.
    #[test]
    fn solves_on_many_threads_equal_the_solve_alone_bit_for_bit() {
        let method = method();
        let config = SolveConfig::default();
        let problems = [anchor_free_triangle().0, anchor_only_problem()];
        let bits = |result: &SolveResult| {
            result
                .positions
                .iter()
                .map(|p| [p.x.to_bits(), p.y.to_bits()])
                .collect::<Vec<_>>()
        };
        let alone: Vec<Vec<[u64; 2]>> = problems
            .iter()
            .map(|problem| bits(&method.solve(problem, &config).unwrap()))
            .collect();
        let differing = std::thread::scope(|scope| {
            let threads: Vec<_> = (0..8)
                .map(|thread| {
                    let (method, config, problems, alone) = (&method, &config, &problems, &alone);
                    scope.spawn(move || {
                        (0..24)
                            .filter(|repeat| {
                                let which = (thread + repeat) % problems.len();
                                method
                                    .solve(&problems[which], config)
                                    .map_or(true, |result| bits(&result) != alone[which])
                            })
                            .count()
                    })
                })
                .collect();
            threads
                .into_iter()
                .map(|thread| thread.join().unwrap())
                .sum::<usize>()
        });

        assert_eq!(differing, 0, "{differing} of 192 solves failed or differed");
    }

    fn method() -> JinSparseBias {
        JinSparseBias {
            residual_multiplier: 2.0,
            range_aggregation: JinRangeAggregation::Median,
        }
    }

    /// Two agents inside a square of four anchors, each ranging every anchor exactly.
    fn anchor_only_problem() -> PreparedProblem {
        let anchors = vec![
            anchor(10, 0.0, 0.0),
            anchor(11, 300.0, 0.0),
            anchor(12, 0.0, 300.0),
            anchor(13, 300.0, 300.0),
        ];
        let truth = [Point2::new(100.0, 100.0), Point2::new(200.0, 100.0)];
        let observations = truth
            .iter()
            .enumerate()
            .flat_map(|(agent, point)| {
                anchors.iter().map(move |anchor| RangeObservation {
                    from: agent as u32 + 1,
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

    /// Three agents of a triangle ranging each other exactly, with their true positions.
    fn anchor_free_triangle() -> (PreparedProblem, [Point2; 3]) {
        let truth = [
            Point2::new(0.0, 0.0),
            Point2::new(300.0, 0.0),
            Point2::new(80.0, 240.0),
        ];
        let observations = [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .map(|(left, right)| RangeObservation {
                from: left as u32 + 1,
                to: right as u32 + 1,
                tof_ns: vec![tof(truth[left].distance(truth[right]))],
                calibrated_samples: Vec::new(),
                to_is_anchor: false,
            })
            .collect();
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2, 3],
            anchors: Vec::new(),
            observations,
            initial_positions: vec![Point2::default(); 3],
            mode: LocalizationMode::AnchorFree,
        };
        (problem, truth)
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
