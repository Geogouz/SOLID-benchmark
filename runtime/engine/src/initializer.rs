//! Geometry initializer shared by the cooperative solver and, as a disclosed start, by the FNL and STRONG
//! baselines.
//!
//! Every agent that ranges to anchors is laterated from its median anchor distances. Agents that only range to
//! peers are placed by classical multidimensional scaling of the shortest path distances inside their
//! connected component, and each component is then rigidly aligned to its laterated members.
//!
//! The anchor aided solver with the block range model starts from the same layout with overrides in place of some
//! laterations, the anchor posterior peaks of the agents whose anchors identify their position, and places those
//! agents at their peaks (`layout_with_overrides`). Every stage of the anchor aided solver also starts once from the
//! agents' own laterations (`layout_and_laterations`, `laterations`).

use std::collections::{BTreeMap, HashMap};

use nalgebra::{DMatrix, SymmetricEigen};

use crate::likelihood::{PreparedMeasurement, RangeModel};
use crate::model::ObservationAggregation;
use crate::{
    CalibratedRangeSample, CalibrationModel, EngineError, LocalizationMode, Point2, PreparedProblem,
    SolveConfig,
};

#[derive(Clone, Copy, Debug)]
enum Target {
    Agent(usize),
    Anchor(AnchorGeometry),
}

#[derive(Clone, Copy, Debug)]
struct AnchorGeometry {
    position: Point2,
    vertical_cm: f64,
}

#[derive(Clone, Copy, Debug)]
struct Factor {
    from: usize,
    to: Target,
    measurement: PreparedMeasurement,
}


/// The initial layout of a prepared problem under the given aggregation and robust width.
pub(crate) fn initial_layout(
    problem: &PreparedProblem,
    config: &SolveConfig,
) -> Result<Vec<Point2>, EngineError> {
    initial_layout_with(problem, config, CalibrationModel::default())
}

/// The initializer's layout with an explicit range calibration for raw timings.
pub(crate) fn initial_layout_with(
    problem: &PreparedProblem,
    config: &SolveConfig,
    calibration: CalibrationModel,
) -> Result<Vec<Point2>, EngineError> {
    layout_and_laterations(problem, config, calibration).map(|(layout, _)| layout)
}

/// The initializer's layout together with the own lateration of every agent that ranges at least three distinct
/// anchors, None for the others. The layout aligns each component rigidly to its laterated agents, which can move
/// an agent away from a good lateration of its own, so the anchor aided solver adds a start at them in every stage.
pub(crate) fn layout_and_laterations(
    problem: &PreparedProblem,
    config: &SolveConfig,
    calibration: CalibrationModel,
) -> Result<(Vec<Point2>, Vec<Option<Point2>>), EngineError> {
    if problem.initial_positions.len() != problem.agent_ids.len() {
        return Err(EngineError::InvalidInitialPositions);
    }
    let model = RangeModel::for_mode(config.robust_delta, problem.mode).with_calibration(calibration);
    let factors = prepare_factors(problem, config, calibration)?;
    let layout = initialize_positions(problem, &factors, &model, &vec![None; problem.agent_ids.len()]);
    Ok((layout, own_laterations(problem, &factors, &model)))
}

/// The laterations of `layout_and_laterations` without the layout, for the anchor aided solver with the block range
/// model, which starts from the layout with the anchor posterior peaks (`layout_with_overrides`) and adds a start at
/// the laterations in every stage.
pub(crate) fn laterations(
    problem: &PreparedProblem,
    config: &SolveConfig,
    calibration: CalibrationModel,
) -> Result<Vec<Option<Point2>>, EngineError> {
    if problem.initial_positions.len() != problem.agent_ids.len() {
        return Err(EngineError::InvalidInitialPositions);
    }
    let model = RangeModel::for_mode(config.robust_delta, problem.mode).with_calibration(calibration);
    let factors = prepare_factors(problem, config, calibration)?;
    Ok(own_laterations(problem, &factors, &model))
}

/// The own lateration of every agent that ranges at least three distinct anchors and laterates to a finite position,
/// None for the others.
fn own_laterations(problem: &PreparedProblem, factors: &[Factor], model: &RangeModel) -> Vec<Option<Point2>> {
    (0..problem.agent_ids.len())
        .map(|agent| {
            let position = (anchor_ranges(agent, factors).len() >= 3).then(|| laterate(agent, factors, model)).flatten()?;
            (position.x.is_finite() && position.y.is_finite()).then_some(position)
        })
        .collect()
}

/// The initializer's layout with `overrides`, one entry per agent, in place of laterations. An agent with an override
/// has it as the target of its component's alignment instead of its own lateration and then sits at it. The other
/// laterations stay targets of the alignment alone, as in the initializer, so without any override this is the
/// initializer's layout.
pub(crate) fn layout_with_overrides(
    problem: &PreparedProblem,
    config: &SolveConfig,
    calibration: CalibrationModel,
    overrides: &[Option<Point2>],
) -> Result<Vec<Point2>, EngineError> {
    if problem.initial_positions.len() != problem.agent_ids.len() {
        return Err(EngineError::InvalidInitialPositions);
    }
    assert_eq!(overrides.len(), problem.agent_ids.len(), "one override entry per agent");
    let model = RangeModel::for_mode(config.robust_delta, problem.mode).with_calibration(calibration);
    let factors = prepare_factors(problem, config, calibration)?;
    Ok(initialize_positions(problem, &factors, &model, overrides))
}

fn prepare_factors(
    problem: &PreparedProblem,
    config: &SolveConfig,
    calibration: CalibrationModel,
) -> Result<Vec<Factor>, EngineError> {
    let agents: HashMap<u32, usize> = problem
        .agent_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();
    let mut anchor_heights: Vec<f64> = problem
        .anchors
        .iter()
        .map(|anchor| anchor.height_cm)
        .collect();
    let agent_plane_height = if anchor_heights.is_empty() {
        0.0
    } else {
        median(&mut anchor_heights)
    };
    let anchors: HashMap<u32, AnchorGeometry> = problem
        .anchors
        .iter()
        .map(|anchor| {
            (
                anchor.id,
                AnchorGeometry {
                    position: anchor.position,
                    vertical_cm: anchor.height_cm - agent_plane_height,
                },
            )
        })
        .collect();
    let measurement_model = RangeModel::new(config.robust_delta).with_calibration(calibration);
    let mut factors = Vec::new();

    for observation in &problem.observations {
        let include = match problem.mode {
            LocalizationMode::AnchorFree => !observation.to_is_anchor,
            LocalizationMode::AnchorAided => true,
            LocalizationMode::AnchorOnly => observation.to_is_anchor,
        };
        if !include {
            continue;
        }

        let from = *agents
            .get(&observation.from)
            .ok_or(EngineError::UnknownAgent(observation.from))?;
        let to = if observation.to_is_anchor {
            Target::Anchor(
                *anchors
                    .get(&observation.to)
                    .ok_or(EngineError::UnknownAnchor(observation.to))?,
            )
        } else {
            Target::Agent(
                *agents
                    .get(&observation.to)
                    .ok_or(EngineError::UnknownAgent(observation.to))?,
            )
        };

        if !observation.calibrated_samples.is_empty() {
            let mut samples: Vec<CalibratedRangeSample> = observation
                .calibrated_samples
                .iter()
                .filter(|sample| sample.is_admissible())
                .cloned()
                .collect();
            if samples.is_empty() {
                continue;
            }
            match config.aggregation {
                ObservationAggregation::Median => factors.push(Factor {
                    from,
                    to,
                    measurement: measurement_model
                        .prepare_calibrated_measurement(median_calibrated(&mut samples)),
                }),
                ObservationAggregation::AllSamples => {
                    factors.extend(samples.into_iter().map(|sample| Factor {
                        from,
                        to,
                        measurement: measurement_model.prepare_calibrated_measurement(sample),
                    }));
                }
            }
            continue;
        }

        let mut samples: Vec<f64> = observation
            .tof_ns
            .iter()
            .copied()
            .filter(|sample| crate::model::admissible_raw(*sample, config.max_tof_ns, &calibration))
            .collect();
        if samples.is_empty() {
            continue;
        }

        match config.aggregation {
            ObservationAggregation::Median => factors.push(Factor {
                from,
                to,
                measurement: measurement_model.prepare_measurement(median(&mut samples)),
            }),
            ObservationAggregation::AllSamples => {
                factors.extend(samples.into_iter().map(|tof_ns| Factor {
                    from,
                    to,
                    measurement: measurement_model.prepare_measurement(tof_ns),
                }))
            }
        }
    }

    Ok(factors)
}

fn median_calibrated(samples: &mut [CalibratedRangeSample]) -> CalibratedRangeSample {
    samples.sort_by(|left, right| left.distance_cm.total_cmp(&right.distance_cm));
    let middle = samples.len() / 2;
    if samples.len() % 2 == 0 {
        CalibratedRangeSample::new(
            0.5 * (samples[middle - 1].distance_cm + samples[middle].distance_cm),
            0.5 * (samples[middle - 1].scale_cm + samples[middle].scale_cm),
        )
    } else {
        samples[middle].clone()
    }
}

/// The initializer's layout with `overrides` in place of laterations. In anchor only positioning every agent sits at
/// its own lateration. Otherwise every component is embedded and aligned rigidly to the laterations of its members,
/// which the alignment leaves where it puts them, while an agent with an override is a target by its override and
/// then sits at it.
fn initialize_positions(
    problem: &PreparedProblem,
    factors: &[Factor],
    model: &RangeModel,
    overrides: &[Option<Point2>],
) -> Vec<Point2> {
    let count = problem.agent_ids.len();
    let mut positions = mds_initialization(count, factors, model);
    let laterated: Vec<Option<Point2>> = (0..count)
        .map(|agent| overrides[agent].or_else(|| laterate(agent, factors, model)))
        .collect();

    if problem.mode == LocalizationMode::AnchorOnly {
        for index in 0..count {
            positions[index] = laterated[index].unwrap_or(problem.initial_positions[index]);
        }
        return positions;
    }

    let components = connected_components(count, factors);
    for component in components {
        align_component_to_anchors(&component, &mut positions, &laterated);
    }

    for index in 0..count {
        if let Some(target) = overrides[index] {
            positions[index] = target;
        } else if !positions[index].x.is_finite() || !positions[index].y.is_finite() {
            positions[index] = laterated[index].unwrap_or(problem.initial_positions[index]);
        }
    }
    positions
}

fn laterate(agent: usize, factors: &[Factor], model: &RangeModel) -> Option<Point2> {
    let ranges = anchor_ranges(agent, factors);
    let summarized: Vec<(Point2, f64)> = ranges
        .iter()
        .map(|(anchor, samples)| {
            let mut distances: Vec<f64> = samples
                .iter()
                .map(|measurement| {
                    let range = model.estimated_distance(measurement);
                    (range * range - anchor.vertical_cm * anchor.vertical_cm)
                        .max(0.0)
                        .sqrt()
                })
                .collect();
            (anchor.position, median(&mut distances))
        })
        .collect();

    match summarized.as_slice() {
        [] => None,
        [(anchor, distance)] => Some(Point2::new(anchor.x + distance, anchor.y)),
        [first, second] => Some(two_anchor_initialization(*first, *second)),
        _ => robust_lateration(&summarized, &ranges, model),
    }
}

fn anchor_ranges(
    agent: usize,
    factors: &[Factor],
) -> Vec<(AnchorGeometry, Vec<PreparedMeasurement>)> {
    let mut ranges: BTreeMap<(u64, u64, u64), (AnchorGeometry, Vec<PreparedMeasurement>)> =
        BTreeMap::new();
    for factor in factors.iter().filter(|factor| factor.from == agent) {
        if let Target::Anchor(anchor) = factor.to {
            ranges
                .entry((
                    anchor.position.x.to_bits(),
                    anchor.position.y.to_bits(),
                    anchor.vertical_cm.to_bits(),
                ))
                .or_insert_with(|| (anchor, Vec::new()))
                .1
                .push(factor.measurement);
        }
    }
    ranges.into_values().collect()
}

fn robust_lateration(
    summarized: &[(Point2, f64)],
    ranges: &[(AnchorGeometry, Vec<PreparedMeasurement>)],
    model: &RangeModel,
) -> Option<Point2> {
    let mut best = linear_lateration(summarized)?;
    let mut best_score = lateration_score(best, ranges, model);
    for first in 0..summarized.len() - 2 {
        for second in first + 1..summarized.len() - 1 {
            for third in second + 1..summarized.len() {
                let subset = [summarized[first], summarized[second], summarized[third]];
                let Some(candidate) = linear_lateration(&subset) else {
                    continue;
                };
                let score = lateration_score(candidate, ranges, model);
                if score < best_score {
                    best = candidate;
                    best_score = score;
                }
            }
        }
    }
    Some(best)
}

fn lateration_score(
    candidate: Point2,
    ranges: &[(AnchorGeometry, Vec<PreparedMeasurement>)],
    model: &RangeModel,
) -> f64 {
    ranges
        .iter()
        .flat_map(|(anchor, samples)| {
            samples.iter().map(|measurement| {
                let horizontal = candidate.distance(anchor.position);
                let distance = (horizontal * horizontal + anchor.vertical_cm.powi(2)).sqrt();
                model
                    .loss_and_distance_gradient_prepared(distance, measurement)
                    .0
            })
        })
        .sum()
}

fn two_anchor_initialization(first: (Point2, f64), second: (Point2, f64)) -> Point2 {
    let (a, ra) = first;
    let (b, rb) = second;
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let separation = dx.hypot(dy).max(1e-9);
    let along = (ra * ra - rb * rb + separation * separation) / (2.0 * separation);
    let height = (ra * ra - along * along).max(0.0).sqrt();
    Point2::new(
        a.x + along * dx / separation - height * dy / separation,
        a.y + along * dy / separation + height * dx / separation,
    )
}

fn linear_lateration(ranges: &[(Point2, f64)]) -> Option<Point2> {
    let (reference, reference_range) = ranges[0];
    let mut ata00 = 0.0;
    let mut ata01 = 0.0;
    let mut ata11 = 0.0;
    let mut atb0 = 0.0;
    let mut atb1 = 0.0;

    for (anchor, range) in &ranges[1..] {
        let ax = 2.0 * (anchor.x - reference.x);
        let ay = 2.0 * (anchor.y - reference.y);
        let b = reference_range.powi(2) - range.powi(2) + anchor.x.powi(2) - reference.x.powi(2)
            + anchor.y.powi(2)
            - reference.y.powi(2);
        ata00 += ax * ax;
        ata01 += ax * ay;
        ata11 += ay * ay;
        atb0 += ax * b;
        atb1 += ay * b;
    }

    let determinant = ata00 * ata11 - ata01 * ata01;
    if determinant.abs() < 1e-9 {
        return Some(two_anchor_initialization(ranges[0], ranges[1]));
    }
    Some(Point2::new(
        (ata11 * atb0 - ata01 * atb1) / determinant,
        (ata00 * atb1 - ata01 * atb0) / determinant,
    ))
}

fn mds_initialization(count: usize, factors: &[Factor], model: &RangeModel) -> Vec<Point2> {
    let components = connected_components(count, factors);
    let mut positions = vec![Point2::default(); count];
    let mut component_offset = 0.0;

    for component in components {
        if component.len() == 1 {
            positions[component[0]] = Point2::new(component_offset, 0.0);
            component_offset += 100.0;
            continue;
        }

        let local_index: HashMap<usize, usize> = component
            .iter()
            .enumerate()
            .map(|(local, global)| (*global, local))
            .collect();
        let size = component.len();
        let mut edge_samples: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
        for factor in factors {
            if let Target::Agent(to) = factor.to {
                if let (Some(&from_local), Some(&to_local)) =
                    (local_index.get(&factor.from), local_index.get(&to))
                {
                    let key = if from_local < to_local {
                        (from_local, to_local)
                    } else {
                        (to_local, from_local)
                    };
                    edge_samples
                        .entry(key)
                        .or_default()
                        .push(model.estimated_distance(&factor.measurement));
                }
            }
        }

        let mut distances = DMatrix::from_element(size, size, f64::INFINITY);
        for index in 0..size {
            distances[(index, index)] = 0.0;
        }
        for ((from, to), mut samples) in edge_samples {
            let distance = median(&mut samples);
            distances[(from, to)] = distance;
            distances[(to, from)] = distance;
        }
        for middle in 0..size {
            for from in 0..size {
                for to in 0..size {
                    let through = distances[(from, middle)] + distances[(middle, to)];
                    if through < distances[(from, to)] {
                        distances[(from, to)] = through;
                    }
                }
            }
        }

        let centering =
            DMatrix::identity(size, size) - DMatrix::from_element(size, size, 1.0 / size as f64);
        let squared = distances.map(|distance| distance * distance);
        let gram = -0.5 * &centering * squared * &centering;
        let eigen = SymmetricEigen::new(gram);
        let mut axes: Vec<usize> = (0..size).collect();
        axes.sort_by(|left, right| eigen.eigenvalues[*right].total_cmp(&eigen.eigenvalues[*left]));

        for (local, global) in component.iter().enumerate() {
            let x_axis = axes[0];
            let y_axis = axes[1];
            let x = eigen.eigenvectors[(local, x_axis)] * eigen.eigenvalues[x_axis].max(0.0).sqrt();
            let y = eigen.eigenvectors[(local, y_axis)] * eigen.eigenvalues[y_axis].max(0.0).sqrt();
            positions[*global] = Point2::new(x + component_offset, y);
        }
        component_offset += 100.0;
    }

    positions
}

fn connected_components(count: usize, factors: &[Factor]) -> Vec<Vec<usize>> {
    let mut parents: Vec<usize> = (0..count).collect();
    for factor in factors {
        if let Target::Agent(to) = factor.to {
            union(&mut parents, factor.from, to);
        }
    }
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for index in 0..count {
        let root = find(&mut parents, index);
        groups.entry(root).or_default().push(index);
    }
    let mut components: Vec<Vec<usize>> = groups.into_values().collect();
    components.sort_by_key(|component| component[0]);
    components
}

fn find(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = find(parents, parents[index]);
    }
    parents[index]
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left = find(parents, left);
    let right = find(parents, right);
    if left != right {
        parents[right] = left;
    }
}

fn align_component_to_anchors(
    component: &[usize],
    positions: &mut [Point2],
    laterated: &[Option<Point2>],
) {
    let correspondences: Vec<(Point2, Point2)> = component
        .iter()
        .filter_map(|index| laterated[*index].map(|target| (positions[*index], target)))
        .collect();
    if correspondences.is_empty() {
        return;
    }

    let source_center = mean_point(correspondences.iter().map(|(source, _)| *source));
    let target_center = mean_point(correspondences.iter().map(|(_, target)| *target));
    let direct = rigid_alignment(&correspondences, source_center, target_center, false);
    let reflected = rigid_alignment(&correspondences, source_center, target_center, true);
    let (cosine, sine, reflect) = if direct.0 <= reflected.0 {
        (direct.1, direct.2, false)
    } else {
        (reflected.1, reflected.2, true)
    };

    for index in component {
        let mut centered = Point2::new(
            positions[*index].x - source_center.x,
            positions[*index].y - source_center.y,
        );
        if reflect {
            centered.y = -centered.y;
        }
        positions[*index] = Point2::new(
            target_center.x + cosine * centered.x - sine * centered.y,
            target_center.y + sine * centered.x + cosine * centered.y,
        );
    }
}

fn rigid_alignment(
    correspondences: &[(Point2, Point2)],
    source_center: Point2,
    target_center: Point2,
    reflect: bool,
) -> (f64, f64, f64) {
    let (sin_sum, cos_sum) = correspondences.iter().fold((0.0, 0.0), |sum, pair| {
        let source_x = pair.0.x - source_center.x;
        let source_y = if reflect {
            source_center.y - pair.0.y
        } else {
            pair.0.y - source_center.y
        };
        let target_x = pair.1.x - target_center.x;
        let target_y = pair.1.y - target_center.y;
        (
            sum.0 + source_x * target_y - source_y * target_x,
            sum.1 + source_x * target_x + source_y * target_y,
        )
    });
    let angle = sin_sum.atan2(cos_sum);
    let cosine = angle.cos();
    let sine = angle.sin();
    let error = correspondences
        .iter()
        .map(|(source, target)| {
            let source_x = source.x - source_center.x;
            let source_y = if reflect {
                source_center.y - source.y
            } else {
                source.y - source_center.y
            };
            let aligned = Point2::new(
                target_center.x + cosine * source_x - sine * source_y,
                target_center.y + sine * source_x + cosine * source_y,
            );
            aligned.distance(*target).powi(2)
        })
        .sum();
    (error, cosine, sine)
}

fn mean_point(points: impl Iterator<Item = Point2>) -> Point2 {
    let (sum_x, sum_y, count) = points.fold((0.0, 0.0, 0usize), |sum, point| {
        (sum.0 + point.x, sum.1 + point.y, sum.2 + 1)
    });
    Point2::new(sum_x / count as f64, sum_y / count as f64)
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Anchor, LocalizationEngine, RangeObservation};

    /// The initializer's layout as it stood before the aligned layout shared its embedding, kept verbatim so that the
    /// start of the FNL and STRONG baselines cannot change unnoticed.
    fn frozen_initialize_positions(problem: &PreparedProblem, factors: &[Factor], model: &RangeModel) -> Vec<Point2> {
        let count = problem.agent_ids.len();
        let mut positions = mds_initialization(count, factors, model);
        let laterated: Vec<Option<Point2>> = (0..count)
            .map(|agent| laterate(agent, factors, model))
            .collect();

        if problem.mode == LocalizationMode::AnchorOnly {
            for index in 0..count {
                positions[index] = laterated[index].unwrap_or(problem.initial_positions[index]);
            }
            return positions;
        }

        let components = connected_components(count, factors);
        for component in components {
            align_component_to_anchors(&component, &mut positions, &laterated);
        }

        for index in 0..count {
            if !positions[index].x.is_finite() || !positions[index].y.is_finite() {
                positions[index] = laterated[index].unwrap_or(problem.initial_positions[index]);
            }
        }
        positions
    }

    /// Six agents and four anchors in `mode`. Agents 1 to 4 range the first one, two, three and four anchors, raw
    /// timings for agents 1 and 3 and calibrated samples for agents 2 and 4. The anchor free and anchor aided problems
    /// also chain agents 1 to 5 by ranges between them, one of them 180 cm short, and agent 6 has no observation.
    fn problem(mode: LocalizationMode) -> PreparedProblem {
        let truth: [(f64, f64); 5] = [(150.0, 120.0), (620.0, 180.0), (400.0, 520.0), (820.0, 640.0), (300.0, 760.0)];
        let anchors: [(u32, f64, f64, f64); 4] = [(10, 0.0, 0.0, 240.0), (11, 1000.0, 0.0, 250.0), (12, 0.0, 900.0, 245.0), (13, 1000.0, 900.0, 260.0)];
        let mut observations = Vec::new();
        if mode != LocalizationMode::AnchorFree {
            for agent in 0..4 {
                for (id, x, y, z) in &anchors[..=agent] {
                    let d = (truth[agent].0 - x).hypot(truth[agent].1 - y).hypot(*z);
                    let (tof_ns, calibrated_samples) = if agent % 2 == 0 {
                        (vec![(d - 19.6) / 28.5, (d - 11.0) / 28.5, (d - 25.0) / 28.5], Vec::new())
                    } else {
                        (Vec::new(), vec![CalibratedRangeSample::new(d + 3.0, 9.0), CalibratedRangeSample::new(d - 2.0, 11.0)])
                    };
                    observations.push(RangeObservation { from: agent as u32 + 1, to: *id, to_is_anchor: true, tof_ns, calibrated_samples });
                }
            }
        }
        if mode != LocalizationMode::AnchorOnly {
            for (a, b, error) in [(0, 1, 0.0), (1, 2, -180.0), (2, 3, 0.0), (3, 4, 25.0), (0, 2, 0.0)] {
                let d = (truth[a].0 - truth[b].0).hypot(truth[a].1 - truth[b].1) + error;
                let samples = vec![CalibratedRangeSample::new(d, 10.0), CalibratedRangeSample::new(d + 4.0, 12.0)];
                observations.push(RangeObservation { from: a as u32 + 1, to: b as u32 + 1, to_is_anchor: false, tof_ns: Vec::new(), calibrated_samples: samples });
            }
        }
        let anchors = anchors.iter().map(|(id, x, y, z)| Anchor { id: *id, position: Point2::new(*x, *y), height_cm: *z });
        PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2, 3, 4, 5, 6],
            anchors: if mode == LocalizationMode::AnchorFree { Vec::new() } else { anchors.collect() },
            observations,
            initial_positions: (0..6).map(|i| Point2::new(40.0 * i as f64, -30.0 * i as f64)).collect(),
            mode,
        }
    }

    fn bits(layout: &[Point2]) -> Vec<[u64; 2]> {
        layout.iter().map(|p| [p.x.to_bits(), p.y.to_bits()]).collect()
    }

    fn configs() -> [SolveConfig; 2] {
        [SolveConfig::default(), SolveConfig { max_time_micros: 0, aggregation: ObservationAggregation::Median, ..SolveConfig::production() }]
    }

    #[test]
    fn the_baselines_start_keeps_the_frozen_initializer_layout() {
        for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided, LocalizationMode::AnchorOnly] {
            for config in configs() {
                let problem = problem(mode);
                let calibration = CalibrationModel::default();
                let model = RangeModel::for_mode(config.robust_delta, problem.mode).with_calibration(calibration);
                let factors = prepare_factors(&problem, &config, calibration).unwrap();
                let frozen = frozen_initialize_positions(&problem, &factors, &model);
                assert!(frozen.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
                assert_eq!(bits(&initial_layout(&problem, &config).unwrap()), bits(&frozen), "{mode:?} {:?}", config.aggregation);
                // the engine's interface that the FNL and STRONG baselines call
                assert_eq!(bits(&LocalizationEngine::new().initial_layout(&problem, &config).unwrap()), bits(&frozen));
            }
        }
    }

    #[test]
    fn overrides_replace_laterations_as_targets_and_place_their_agents() {
        let calibration = CalibrationModel::default();
        for config in configs() {
            // without any override the layout is the initializer's
            for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided] {
                let problem = problem(mode);
                let unchanged = layout_with_overrides(&problem, &config, calibration, &[None; 6]).unwrap();
                assert_eq!(bits(&unchanged), bits(&initial_layout(&problem, &config).unwrap()), "{mode:?}");
            }
            let problem = problem(LocalizationMode::AnchorAided);
            let model = RangeModel::for_mode(config.robust_delta, problem.mode).with_calibration(calibration);
            let factors = prepare_factors(&problem, &config, calibration).unwrap();
            let laterated: Vec<Option<Point2>> = (0..6).map(|agent| laterate(agent, &factors, &model)).collect();
            assert_eq!(laterated.iter().filter(|lateration| lateration.is_some()).count(), 4);
            let initial = initial_layout_with(&problem, &config, calibration).unwrap();
            // agents 3 and 4 range three and four anchors. Their own laterations as overrides leave every target of the
            // alignment as it was, so only these two agents move, onto their laterations
            let own: Vec<Option<Point2>> = (0..6).map(|agent| if agent == 2 || agent == 3 { laterated[agent] } else { None }).collect();
            let placed = layout_with_overrides(&problem, &config, calibration, &own).unwrap();
            let mut moved = 0;
            for agent in 0..6 {
                match own[agent] {
                    Some(target) => {
                        assert_eq!(bits(&[placed[agent]]), bits(&[target]));
                        moved += usize::from(initial[agent].distance(target) > 1.0);
                    }
                    None => assert_eq!(bits(&[placed[agent]]), bits(&[initial[agent]]), "agent {agent}"),
                }
            }
            // the distorted range leaves some laterated agent away from its lateration in the initializer's layout
            assert!(moved > 0);
            // another override of agent 4 moves the alignment of the whole component, while the laterations of agents 1
            // and 2, with one and two anchors, stay targets of the alignment alone
            let mut shifted = own.clone();
            shifted[3] = laterated[3].map(|p| Point2::new(p.x + 300.0, p.y - 200.0));
            let moved_layout = layout_with_overrides(&problem, &config, calibration, &shifted).unwrap();
            assert_eq!((moved_layout[2], moved_layout[3]), (own[2].unwrap(), shifted[3].unwrap()));
            for agent in [0, 1, 4] {
                assert!(moved_layout[agent].distance(placed[agent]) > 1.0, "agent {agent}");
            }
            for agent in [0, 1] {
                assert!(moved_layout[agent].distance(laterated[agent].unwrap()) > 1e-6, "agent {agent}");
            }
            // agent 6 has no observation and keeps its place
            assert_eq!(bits(&moved_layout[5..]), bits(&initial[5..]));
        }
        // a request whose start positions do not match its agents is refused
        let mut short = problem(LocalizationMode::AnchorAided);
        short.initial_positions.pop();
        let refused = layout_with_overrides(&short, &SolveConfig::default(), calibration, &[None; 6]);
        assert!(matches!(refused, Err(EngineError::InvalidInitialPositions)));
    }
}
