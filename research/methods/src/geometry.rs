use std::collections::{BTreeMap, HashMap};

use solid_core::{
    Anchor, CalibrationModel, DimensionPolicy, ObservationAggregation, Point2, PreparedProblem,
    SolveConfig,
};

use crate::MethodError;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Endpoint {
    Agent(usize),
    Anchor(Anchor),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RangeFactor {
    pub from: usize,
    pub to: Endpoint,
    pub distance_cm: f64,
    pub inverse_scale: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct ProblemScale {
    origin: [f64; 3],
    scale_cm: f64,
}

impl ProblemScale {
    pub(crate) fn from_problem(
        problem: &PreparedProblem,
        factors: &[RangeFactor],
        dimensions: usize,
    ) -> Result<Self, MethodError> {
        if problem.anchors.is_empty() {
            return Err(MethodError::UnsupportedMode(
                "the method requires at least one anchor".to_owned(),
            ));
        }
        let mut origin = [0.0; 3];
        for anchor in &problem.anchors {
            let position = anchor_coordinates(*anchor);
            for dimension in 0..dimensions {
                origin[dimension] += position[dimension];
            }
        }
        for value in origin.iter_mut().take(dimensions) {
            *value /= problem.anchors.len() as f64;
        }
        let maximum_range = factors
            .iter()
            .map(|factor| factor.distance_cm)
            .fold(0.0_f64, f64::max);
        let maximum_anchor_radius = problem
            .anchors
            .iter()
            .map(|anchor| {
                let position = anchor_coordinates(*anchor);
                position[..dimensions]
                    .iter()
                    .zip(&origin)
                    .map(|(coordinate, origin)| (coordinate - origin).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .fold(0.0_f64, f64::max);
        Ok(Self {
            origin,
            scale_cm: maximum_range.max(maximum_anchor_radius).max(1.0),
        })
    }

    pub(crate) fn from_relative_factors(factors: &[RangeFactor]) -> Result<Self, MethodError> {
        let scale_cm = factors
            .iter()
            .map(|factor| factor.distance_cm)
            .fold(0.0_f64, f64::max);
        if !scale_cm.is_finite() || scale_cm <= 0.0 {
            return Err(MethodError::InvalidInput(
                "relative localization requires a positive range".to_owned(),
            ));
        }
        Ok(Self {
            origin: [0.0; 3],
            scale_cm,
        })
    }

    pub(crate) fn encode_point(&self, point: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|dimension| (point[dimension] - self.origin[dimension]) / self.scale_cm)
    }

    pub(crate) fn encode_anchor(&self, anchor: Anchor) -> [f64; 3] {
        self.encode_point(anchor_coordinates(anchor))
    }

    pub(crate) fn encode_distance(&self, distance_cm: f64) -> f64 {
        distance_cm / self.scale_cm
    }

    pub(crate) fn decode_point(&self, point: &[f64]) -> Point2 {
        Point2::new(
            point[0] * self.scale_cm + self.origin[0],
            point[1] * self.scale_cm + self.origin[1],
        )
    }

    pub(crate) fn decode_relative_point(&self, point: &[f64]) -> Point2 {
        Point2::new(point[0] * self.scale_cm, point[1] * self.scale_cm)
    }
}

pub(crate) fn dimensions(problem: &PreparedProblem, config: &SolveConfig) -> usize {
    let _ = problem;
    match config.dimension_policy {
        DimensionPolicy::Spatial => 3,
        DimensionPolicy::Automatic | DimensionPolicy::Planar => 2,
    }
}

pub(crate) fn initial_parameters(
    problem: &PreparedProblem,
    _config: &SolveConfig,
    dimensions: usize,
) -> Result<Vec<f64>, MethodError> {
    if problem.initial_positions.len() != problem.agent_ids.len()
        || problem
            .initial_positions
            .iter()
            .any(|point| !point.x.is_finite() || !point.y.is_finite())
    {
        return Err(MethodError::InvalidInput(
            "released initial coordinates are missing or nonfinite".to_owned(),
        ));
    }
    let initial_height = median(
        problem
            .anchors
            .iter()
            .map(|anchor| anchor.height_cm)
            .collect(),
    )
    .unwrap_or(0.0);
    Ok(problem
        .initial_positions
        .iter()
        .flat_map(|point| {
            [point.x, point.y, initial_height]
                .into_iter()
                .take(dimensions)
        })
        .collect())
}

pub(crate) fn prepare_link_factors(
    problem: &PreparedProblem,
    config: &SolveConfig,
) -> Result<Vec<RangeFactor>, MethodError> {
    match config.aggregation {
        ObservationAggregation::Median => prepare_links(problem, config, Some(median)),
        ObservationAggregation::AllSamples => prepare_links(problem, config, None),
    }
}

fn prepare_links(
    problem: &PreparedProblem,
    config: &SolveConfig,
    summary: Option<fn(Vec<f64>) -> Option<f64>>,
) -> Result<Vec<RangeFactor>, MethodError> {
    let agents: HashMap<u32, usize> = problem
        .agent_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();
    let anchors: HashMap<u32, Anchor> = problem
        .anchors
        .iter()
        .map(|anchor| (anchor.id, *anchor))
        .collect();
    let calibration = CalibrationModel::default();
    let mut factors = Vec::with_capacity(problem.observations.len());

    for observation in &problem.observations {
        let Some(&from) = agents.get(&observation.from) else {
            return Err(MethodError::InvalidInput(format!(
                "unknown source agent {}",
                observation.from
            )));
        };
        let calibrated = if observation.calibrated_samples.is_empty() {
            let samples = observation
                .tof_ns
                .iter()
                .copied()
                .filter(|sample| {
                    sample.is_finite() && *sample >= 0.0 && *sample <= config.max_tof_ns
                })
                .collect::<Vec<_>>();
            match summary {
                Some(summarize) => {
                    let Some(tof_ns) = summarize(samples) else {
                        continue;
                    };
                    let distance_cm = calibration.corrected_distance_cm(tof_ns);
                    vec![(distance_cm, calibration.scale_cm(distance_cm))]
                }
                None => samples
                    .into_iter()
                    .map(|tof_ns| {
                        let distance_cm = calibration.corrected_distance_cm(tof_ns);
                        (distance_cm, calibration.scale_cm(distance_cm))
                    })
                    .collect(),
            }
        } else {
            let samples = observation
                .calibrated_samples
                .iter()
                .filter(|sample| sample.is_valid())
                .collect::<Vec<_>>();
            if samples.is_empty() {
                continue;
            }
            match summary {
                Some(summarize) => vec![(
                    summarize(samples.iter().map(|sample| sample.distance_cm).collect())
                        .expect("calibrated samples are nonempty"),
                    summarize(samples.iter().map(|sample| sample.scale_cm).collect())
                        .expect("calibrated samples are nonempty"),
                )],
                None => samples
                    .into_iter()
                    .map(|sample| (sample.distance_cm, sample.scale_cm))
                    .collect(),
            }
        };
        let to = if observation.to_is_anchor {
            let Some(&anchor) = anchors.get(&observation.to) else {
                return Err(MethodError::InvalidInput(format!(
                    "unknown anchor {}",
                    observation.to
                )));
            };
            Endpoint::Anchor(anchor)
        } else {
            let Some(&to) = agents.get(&observation.to) else {
                return Err(MethodError::InvalidInput(format!(
                    "unknown target agent {}",
                    observation.to
                )));
            };
            Endpoint::Agent(to)
        };
        factors.extend(
            calibrated
                .into_iter()
                .map(|(distance_cm, scale_cm)| RangeFactor {
                    from,
                    to,
                    distance_cm,
                    inverse_scale: scale_cm.recip(),
                }),
        );
    }

    Ok(factors)
}

pub(crate) fn prepare_undirected_medians(
    problem: &PreparedProblem,
    config: &SolveConfig,
) -> Result<Vec<RangeFactor>, MethodError> {
    prepare_undirected(problem, config, median)
}

pub(crate) fn prepare_undirected_means(
    problem: &PreparedProblem,
    config: &SolveConfig,
) -> Result<Vec<RangeFactor>, MethodError> {
    prepare_undirected(problem, config, mean)
}

fn prepare_undirected(
    problem: &PreparedProblem,
    config: &SolveConfig,
    summarize: fn(Vec<f64>) -> Option<f64>,
) -> Result<Vec<RangeFactor>, MethodError> {
    #[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
    enum Key {
        Agent(usize, usize),
        Anchor(usize, u32),
    }

    let directed = prepare_links(problem, config, Some(summarize))?;
    let mut grouped = BTreeMap::<Key, Vec<RangeFactor>>::new();
    for factor in directed {
        let key = match factor.to {
            Endpoint::Agent(to) => Key::Agent(factor.from.min(to), factor.from.max(to)),
            Endpoint::Anchor(anchor) => Key::Anchor(factor.from, anchor.id),
        };
        grouped.entry(key).or_default().push(factor);
    }
    Ok(grouped
        .into_iter()
        .map(|(key, factors)| {
            let distance_cm = summarize(factors.iter().map(|factor| factor.distance_cm).collect())
                .expect("an undirected link has at least one measurement");
            let scale_cm = summarize(
                factors
                    .iter()
                    .map(|factor| factor.inverse_scale.recip())
                    .collect(),
            )
            .expect("an undirected link has at least one measurement");
            let (from, to) = match key {
                Key::Agent(from, to) => (from, Endpoint::Agent(to)),
                Key::Anchor(from, _) => (from, factors[0].to),
            };
            RangeFactor {
                from,
                to,
                distance_cm,
                inverse_scale: scale_cm.recip(),
            }
        })
        .collect())
}

pub(crate) fn points(parameters: &[f64], dimensions: usize) -> Vec<Point2> {
    parameters
        .chunks_exact(dimensions)
        .map(|position| Point2::new(position[0], position[1]))
        .collect()
}

pub(crate) fn agent_components(agents: usize, factors: &[RangeFactor]) -> Vec<Vec<usize>> {
    let mut parents = (0..agents).collect::<Vec<_>>();
    for factor in factors {
        if let Endpoint::Agent(to) = factor.to {
            union(&mut parents, factor.from, to);
        }
    }
    let mut components = Vec::<Vec<usize>>::new();
    for agent in 0..agents {
        let root = find(&mut parents, agent);
        if let Some(component) = components
            .iter_mut()
            .find(|component| find(&mut parents, component[0]) == root)
        {
            component.push(agent);
        } else {
            components.push(vec![agent]);
        }
    }
    components
}

pub(crate) fn center_components(
    parameters: &mut [f64],
    dimensions: usize,
    components: &[Vec<usize>],
) {
    for component in components {
        for dimension in 0..dimensions {
            let mean = component
                .iter()
                .map(|agent| parameters[agent * dimensions + dimension])
                .sum::<f64>()
                / component.len() as f64;
            for agent in component {
                parameters[agent * dimensions + dimension] -= mean;
            }
        }
    }
}

fn find(parents: &mut [usize], node: usize) -> usize {
    if parents[node] != node {
        parents[node] = find(parents, parents[node]);
    }
    parents[node]
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let left = find(parents, left);
    let right = find(parents, right);
    if left != right {
        parents[right] = left;
    }
}

fn mean(values: Vec<f64>) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    })
}

fn anchor_coordinates(anchor: Anchor) -> [f64; 3] {
    [anchor.position.x, anchor.position.y, anchor.height_cm]
}

#[cfg(test)]
mod tests {
    use solid_core::{CalibratedRangeSample, LocalizationMode, RangeObservation};

    use super::*;

    #[test]
    fn undirected_mean_weights_directions_equally_not_raw_sample_counts() {
        let mut problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1, 2],
            anchors: Vec::new(),
            observations: [(1, 2, vec![100.0, 100.0, 400.0]), (2, 1, vec![600.0])]
                .into_iter()
                .map(|(from, to, distances)| RangeObservation {
                    from,
                    to,
                    tof_ns: distances
                        .iter()
                        .map(|distance| (distance - 19.579_395_666_091_16) / 28.512_859_035_713_586)
                        .collect(),
                    calibrated_samples: distances
                        .into_iter()
                        .map(|distance| CalibratedRangeSample::new(distance, 10.0))
                        .collect(),
                    to_is_anchor: false,
                })
                .collect(),
            initial_positions: vec![Point2::default(); 2],
            mode: LocalizationMode::AnchorFree,
        };
        for raw_samples in [false, true] {
            if raw_samples {
                for observation in &mut problem.observations {
                    observation.calibrated_samples.clear();
                }
            }
            let mean = prepare_undirected_means(&problem, &SolveConfig::default()).unwrap();
            let median = prepare_undirected_medians(&problem, &SolveConfig::default()).unwrap();
            assert_eq!(mean.len(), 1);
            assert_eq!(median.len(), 1);
            assert!((mean[0].distance_cm - 400.0).abs() < 1e-10);
            assert!((median[0].distance_cm - 350.0).abs() < 1e-10);
        }
    }

    #[test]
    fn shared_geometry_uses_supplied_calibration_without_bern_conversion() {
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1],
            anchors: vec![Anchor {
                id: 10,
                position: Point2::new(0.0, 0.0),
                height_cm: 0.0,
            }],
            observations: vec![RangeObservation {
                from: 1,
                to: 10,
                tof_ns: vec![1.0],
                calibrated_samples: vec![CalibratedRangeSample::new(321.0, 12.5)],
                to_is_anchor: true,
            }],
            initial_positions: vec![Point2::default()],
            mode: LocalizationMode::AnchorOnly,
        };

        let factors = prepare_link_factors(&problem, &SolveConfig {
            aggregation: ObservationAggregation::Median,
            ..SolveConfig::default()
        }).unwrap();

        assert_eq!(factors.len(), 1);
        assert_eq!(factors[0].distance_cm, 321.0);
        assert_eq!(factors[0].inverse_scale, 1.0 / 12.5);
    }

    #[test]
    fn all_sample_geometry_preserves_every_calibrated_measurement() {
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1],
            anchors: vec![Anchor {
                id: 10,
                position: Point2::new(0.0, 0.0),
                height_cm: 0.0,
            }],
            observations: vec![RangeObservation {
                from: 1,
                to: 10,
                tof_ns: Vec::new(),
                calibrated_samples: vec![
                    CalibratedRangeSample::new(300.0, 10.0),
                    CalibratedRangeSample::new(320.0, 20.0),
                ],
                to_is_anchor: true,
            }],
            initial_positions: vec![Point2::default()],
            mode: LocalizationMode::AnchorOnly,
        };

        let factors = prepare_link_factors(&problem, &SolveConfig::default()).unwrap();

        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0].distance_cm, 300.0);
        assert_eq!(factors[0].inverse_scale, 0.1);
        assert_eq!(factors[1].distance_cm, 320.0);
        assert_eq!(factors[1].inverse_scale, 0.05);
    }
}
