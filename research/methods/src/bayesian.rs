use std::collections::{BTreeMap, HashMap};

use libm::erfc;
use solid_core::{
    CalibrationModel, ObservationAggregation, Point2, PreparedProblem, RangeObservation,
    SolveConfig,
};

use crate::{AdeRangingModel, AreaBounds, MethodError};

const MIN_DENSITY: f64 = 1e-300;
const SQRT_TWO: f64 = std::f64::consts::SQRT_2;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum Node {
    Agent(usize),
    Anchor(usize),
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RangeSample {
    pub distance_cm: f64,
    pub scale_cm: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct Link {
    pub left: usize,
    pub right: Node,
    pub samples: Vec<RangeSample>,
}

#[derive(Clone, Debug)]
pub(crate) struct Network {
    pub links: Vec<Link>,
    pub incident: Vec<Vec<usize>>,
}

impl Network {
    pub(crate) fn prepare(
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<Self, MethodError> {
        let agents = problem
            .agent_ids
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect::<HashMap<_, _>>();
        let anchors = problem
            .anchors
            .iter()
            .enumerate()
            .map(|(index, anchor)| (anchor.id, index))
            .collect::<HashMap<_, _>>();
        let mut grouped = BTreeMap::<(usize, Node), Vec<RangeSample>>::new();
        for observation in &problem.observations {
            let Some(&from) = agents.get(&observation.from) else {
                return Err(MethodError::InvalidInput(format!(
                    "unknown source agent {}",
                    observation.from
                )));
            };
            let right = if observation.to_is_anchor {
                let Some(&anchor) = anchors.get(&observation.to) else {
                    return Err(MethodError::InvalidInput(format!(
                        "unknown anchor {}",
                        observation.to
                    )));
                };
                Node::Anchor(anchor)
            } else {
                let Some(&agent) = agents.get(&observation.to) else {
                    return Err(MethodError::InvalidInput(format!(
                        "unknown target agent {}",
                        observation.to
                    )));
                };
                Node::Agent(agent)
            };
            if right == Node::Agent(from) {
                continue;
            }
            let (left, right) = match right {
                Node::Agent(to) if to < from => (to, Node::Agent(from)),
                _ => (from, right),
            };
            let samples = grouped.entry((left, right)).or_default();
            append_samples(samples, observation, config);
        }
        let links = grouped
            .into_iter()
            .filter_map(|((left, right), mut samples)| {
                if samples.is_empty() {
                    return None;
                }
                samples.sort_by(|left, right| {
                    left.distance_cm
                        .total_cmp(&right.distance_cm)
                        .then_with(|| left.scale_cm.total_cmp(&right.scale_cm))
                });
                Some(Link {
                    left,
                    right,
                    samples,
                })
            })
            .collect::<Vec<_>>();
        let mut incident = vec![Vec::new(); problem.agent_ids.len()];
        for (index, link) in links.iter().enumerate() {
            incident[link.left].push(index);
            if let Node::Agent(right) = link.right {
                incident[right].push(index);
            }
        }
        Ok(Self { links, incident })
    }

    pub(crate) fn other(&self, link: usize, agent: usize) -> Node {
        let link = &self.links[link];
        if link.left == agent {
            link.right
        } else {
            Node::Agent(link.left)
        }
    }
}

fn append_samples(
    target: &mut Vec<RangeSample>,
    observation: &RangeObservation,
    config: &SolveConfig,
) {
    let mut samples = if observation.calibrated_samples.is_empty() {
        let calibration = CalibrationModel::default();
        observation
            .tof_ns
            .iter()
            .copied()
            .filter(|tof| tof.is_finite() && *tof >= 0.0 && *tof <= config.max_tof_ns)
            .map(|tof| {
                let distance_cm = calibration.corrected_distance_cm(tof);
                RangeSample {
                    distance_cm,
                    scale_cm: calibration.scale_cm(distance_cm),
                }
            })
            .collect::<Vec<_>>()
    } else {
        observation
            .calibrated_samples
            .iter()
            .filter(|sample| sample.is_valid())
            .map(|sample| RangeSample {
                distance_cm: sample.distance_cm,
                scale_cm: sample.scale_cm,
            })
            .collect::<Vec<_>>()
    };
    match config.aggregation {
        ObservationAggregation::Median if !samples.is_empty() => {
            samples.sort_by(|left, right| left.distance_cm.total_cmp(&right.distance_cm));
            let middle = samples.len() / 2;
            if samples.len() % 2 == 0 {
                target.push(RangeSample {
                    distance_cm: 0.5
                        * (samples[middle - 1].distance_cm + samples[middle].distance_cm),
                    scale_cm: 0.5 * (samples[middle - 1].scale_cm + samples[middle].scale_cm),
                });
            } else {
                target.push(samples[middle]);
            }
        }
        ObservationAggregation::Median => {}
        ObservationAggregation::AllSamples => target.extend(samples),
    }
}

pub(crate) fn resolve_area(
    problem: &PreparedProblem,
    configured: Option<AreaBounds>,
    network: &Network,
) -> Result<AreaBounds, MethodError> {
    if let Some(area) = configured {
        area.validate()?;
        return Ok(area);
    }
    let maximum_range = network
        .links
        .iter()
        .flat_map(|link| &link.samples)
        .map(|sample| sample.distance_cm)
        .fold(100.0_f64, f64::max);
    if problem.anchors.is_empty() {
        return Ok(AreaBounds {
            minimum_x_cm: -maximum_range,
            maximum_x_cm: maximum_range,
            minimum_y_cm: -maximum_range,
            maximum_y_cm: maximum_range,
        });
    }
    let minimum_x = problem
        .anchors
        .iter()
        .map(|anchor| anchor.position.x)
        .fold(f64::INFINITY, f64::min);
    let maximum_x = problem
        .anchors
        .iter()
        .map(|anchor| anchor.position.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let minimum_y = problem
        .anchors
        .iter()
        .map(|anchor| anchor.position.y)
        .fold(f64::INFINITY, f64::min);
    let maximum_y = problem
        .anchors
        .iter()
        .map(|anchor| anchor.position.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let span = (maximum_x - minimum_x)
        .max(maximum_y - minimum_y)
        .max(maximum_range);
    let margin = 0.5 * span;
    Ok(AreaBounds {
        minimum_x_cm: minimum_x - margin,
        maximum_x_cm: maximum_x + margin,
        minimum_y_cm: minimum_y - margin,
        maximum_y_cm: maximum_y + margin,
    })
}

impl AreaBounds {
    pub(crate) fn validate(self) -> Result<(), MethodError> {
        if [
            self.minimum_x_cm,
            self.maximum_x_cm,
            self.minimum_y_cm,
            self.maximum_y_cm,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.minimum_x_cm < self.maximum_x_cm
            && self.minimum_y_cm < self.maximum_y_cm
        {
            Ok(())
        } else {
            Err(MethodError::InvalidInput(
                "invalid Bayesian area boundary".to_owned(),
            ))
        }
    }

    pub(crate) fn contains(self, point: Point2) -> bool {
        point.x >= self.minimum_x_cm
            && point.x <= self.maximum_x_cm
            && point.y >= self.minimum_y_cm
            && point.y <= self.maximum_y_cm
    }

    pub(crate) fn diameter(self) -> f64 {
        (self.maximum_x_cm - self.minimum_x_cm).hypot(self.maximum_y_cm - self.minimum_y_cm)
    }

    pub(crate) fn sample(self, rng: &mut Rng) -> Point2 {
        Point2::new(
            self.minimum_x_cm + rng.uniform() * (self.maximum_x_cm - self.minimum_x_cm),
            self.minimum_y_cm + rng.uniform() * (self.maximum_y_cm - self.minimum_y_cm),
        )
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ParticleBelief {
    pub points: Vec<Point2>,
    pub weights: Vec<f64>,
}

impl ParticleBelief {
    pub(crate) fn mean(&self) -> Point2 {
        let mut mean = Point2::default();
        for (point, weight) in self.points.iter().zip(&self.weights) {
            mean.x += point.x * weight;
            mean.y += point.y * weight;
        }
        mean
    }

    pub(crate) fn importance_gaussian(&self) -> GaussianBelief {
        GaussianBelief::from_importance_weights(&self.points, &self.weights)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GaussianBelief {
    pub mean: Point2,
    pub xx: f64,
    pub xy: f64,
    pub yy: f64,
}

impl GaussianBelief {
    fn from_importance_weights(points: &[Point2], weights: &[f64]) -> Self {
        let denominator = 1.0 - weights.iter().map(|weight| weight * weight).sum::<f64>();
        Self::from_weighted_with_denominator(points, weights, denominator.max(1e-12))
    }

    fn from_weighted_with_denominator(
        points: &[Point2],
        weights: &[f64],
        denominator: f64,
    ) -> Self {
        let mut mean = Point2::default();
        for (point, weight) in points.iter().zip(weights) {
            mean.x += point.x * weight;
            mean.y += point.y * weight;
        }
        let mut xx = 0.0;
        let mut xy = 0.0;
        let mut yy = 0.0;
        for (point, weight) in points.iter().zip(weights) {
            let x = point.x - mean.x;
            let y = point.y - mean.y;
            xx += weight * x * x;
            xy += weight * x * y;
            yy += weight * y * y;
        }
        Self {
            mean,
            xx: (xx / denominator).max(1.0),
            xy: xy / denominator,
            yy: (yy / denominator).max(1.0),
        }
        .regularized()
    }

    fn regularized(mut self) -> Self {
        let maximum_xy = (self.xx * self.yy).sqrt() * 0.999_999;
        self.xy = self.xy.clamp(-maximum_xy, maximum_xy);
        self
    }

    pub(crate) fn sample(self, rng: &mut Rng) -> Point2 {
        let l11 = self.xx.sqrt();
        let l21 = self.xy / l11;
        let l22 = (self.yy - l21 * l21).max(1.0).sqrt();
        let first = rng.normal();
        let second = rng.normal();
        Point2::new(
            self.mean.x + l11 * first,
            self.mean.y + l21 * first + l22 * second,
        )
    }

    pub(crate) fn log_density(self, point: Point2) -> f64 {
        let determinant = (self.xx * self.yy - self.xy * self.xy).max(1e-12);
        let x = point.x - self.mean.x;
        let y = point.y - self.mean.y;
        let quadratic = (self.yy * x * x - 2.0 * self.xy * x * y + self.xx * y * y) / determinant;
        -0.5 * (quadratic + determinant.ln() + 2.0 * (2.0 * std::f64::consts::PI).ln())
    }

    pub(crate) fn radial_variance(self, point: Point2) -> f64 {
        let x = point.x - self.mean.x;
        let y = point.y - self.mean.y;
        let norm = x.hypot(y);
        if norm <= 1e-9 {
            return 0.5 * (self.xx + self.yy);
        }
        let ux = x / norm;
        let uy = y / norm;
        (ux * ux * self.xx + 2.0 * ux * uy * self.xy + uy * uy * self.yy).max(1e-12)
    }
}

pub(crate) fn resample_gaussian(
    belief: GaussianBelief,
    area: AreaBounds,
    count: usize,
    rng: &mut Rng,
) -> (Vec<Point2>, Vec<f64>) {
    let mut points = Vec::with_capacity(count);
    let mut attempts = 0usize;
    while points.len() < count && attempts < 100 * count {
        let point = belief.sample(rng);
        if area.contains(point) {
            points.push(point);
        }
        attempts += 1;
    }
    while points.len() < count {
        let point = area.sample(rng);
        if area.contains(point) {
            points.push(point);
        }
    }
    let log_importance = points
        .iter()
        .map(|point| -belief.log_density(*point))
        .collect();
    (points, log_importance)
}

pub(crate) fn normalized_log_weights(log_weights: &[f64]) -> Vec<f64> {
    let maximum = log_weights
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return vec![1.0 / log_weights.len().max(1) as f64; log_weights.len()];
    }
    let mut weights = log_weights
        .iter()
        .map(|value| {
            if value.is_finite() {
                (value - maximum).exp()
            } else {
                0.0
            }
        })
        .collect::<Vec<_>>();
    let sum = weights.iter().sum::<f64>();
    if !sum.is_finite() || sum <= 0.0 {
        return vec![1.0 / weights.len().max(1) as f64; weights.len()];
    }
    for weight in &mut weights {
        *weight /= sum;
    }
    weights
}

pub(crate) fn log_ade_gaussian_message(
    point: Point2,
    neighbor: GaussianBelief,
    measured_range_cm: f64,
    model: AdeRangingModel,
) -> f64 {
    let distance = point.distance(neighbor.mean);
    let residual = measured_range_cm - distance;
    let variance = neighbor.radial_variance(point);
    if variance <= 1e-8 {
        return model.log_density(residual, measured_range_cm);
    }
    let sigma = variance.sqrt();
    let (positive, negative) = model.scales(measured_range_cm);
    let first = log_normal_survival(-residual / sigma + sigma / positive) - residual / positive
        + variance / (2.0 * positive * positive);
    let second = log_normal_survival(residual / sigma + sigma / negative)
        + residual / negative
        + variance / (2.0 * negative * negative);
    log_add(first, second) - (positive + negative).ln()
}

fn log_normal_survival(value: f64) -> f64 {
    if value < -8.0 {
        return 0.0;
    }
    if value > 8.0 {
        let inverse_square = 1.0 / (value * value);
        return -0.5 * value * value - value.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            + (-inverse_square + 2.5 * inverse_square * inverse_square);
    }
    (0.5 * erfc(value / SQRT_TWO)).max(MIN_DENSITY).ln()
}

fn log_add(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    if maximum.is_infinite() {
        maximum
    } else {
        maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
    }
}

pub(crate) fn problem_seed(problem: &PreparedProblem, salt: u64) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64 ^ salt;
    for id in &problem.agent_ids {
        hash = fnv(hash, *id as u64);
    }
    for anchor in &problem.anchors {
        hash = fnv(hash, anchor.id as u64);
        hash = fnv(hash, anchor.position.x.to_bits());
        hash = fnv(hash, anchor.position.y.to_bits());
    }
    for observation in &problem.observations {
        hash = fnv(hash, observation.from as u64);
        hash = fnv(hash, observation.to as u64);
        for value in &observation.tof_ns {
            hash = fnv(hash, value.to_bits());
        }
        for sample in &observation.calibrated_samples {
            hash = fnv(hash, sample.distance_cm.to_bits());
            hash = fnv(hash, sample.scale_cm.to_bits());
        }
    }
    hash
}

fn fnv(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[derive(Clone, Debug)]
pub(crate) struct Rng {
    state: u64,
    spare_normal: Option<f64>,
}

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / (1_u64 << 53) as f64)
    }

    pub(crate) fn normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        let radius = (-2.0 * self.uniform().max(f64::MIN_POSITIVE).ln()).sqrt();
        let angle = 2.0 * std::f64::consts::PI * self.uniform();
        self.spare_normal = Some(radius * angle.sin());
        radius * angle.cos()
    }
}

#[cfg(test)]
mod tests {
    use solid_core::{Anchor, CalibratedRangeSample, LocalizationMode};

    use super::*;

    #[test]
    fn network_applies_the_requested_repeated_sample_aggregation() {
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1],
            anchors: vec![Anchor {
                id: 10,
                position: Point2::default(),
                height_cm: 0.0,
            }],
            observations: vec![RangeObservation {
                from: 1,
                to: 10,
                tof_ns: Vec::new(),
                calibrated_samples: vec![
                    CalibratedRangeSample::new(10.0, 1.0),
                    CalibratedRangeSample::new(40.0, 4.0),
                    CalibratedRangeSample::new(20.0, 2.0),
                    CalibratedRangeSample::new(30.0, 3.0),
                ],
                to_is_anchor: true,
            }],
            initial_positions: vec![Point2::default()],
            mode: LocalizationMode::AnchorOnly,
        };
        let mut median_config = SolveConfig::default();
        median_config.aggregation = ObservationAggregation::Median;

        let median = Network::prepare(&problem, &median_config).unwrap();
        let all = Network::prepare(&problem, &SolveConfig::default()).unwrap();

        assert_eq!(median.links[0].samples.len(), 1);
        assert_eq!(median.links[0].samples[0].distance_cm, 25.0);
        assert_eq!(median.links[0].samples[0].scale_cm, 2.5);
        assert_eq!(all.links[0].samples.len(), 4);
    }

    #[test]
    fn importance_gaussian_uses_the_published_covariance_correction() {
        let belief = ParticleBelief {
            points: vec![Point2::new(-10.0, 0.0), Point2::new(10.0, 0.0)],
            weights: vec![0.5, 0.5],
        }
        .importance_gaussian();

        assert_eq!(belief.mean, Point2::default());
        assert_eq!(belief.xx, 200.0);
        assert_eq!(belief.yy, 1.0);
    }

    #[test]
    fn asymmetric_gaussian_message_matches_a_concentrated_neighbor() {
        let model = AdeRangingModel {
            positive_slope: 0.02,
            positive_intercept_cm: 10.0,
            negative_slope: 0.0,
            negative_intercept_cm: 20.0,
        };
        let point = Point2::new(100.0, 0.0);
        let neighbor = GaussianBelief {
            mean: Point2::default(),
            xx: 1e-10,
            xy: 0.0,
            yy: 1e-10,
        };

        let actual = log_ade_gaussian_message(point, neighbor, 130.0, model);
        let expected = model.log_density(30.0, 130.0);

        assert!((actual - expected).abs() < 1e-12);
    }

    #[test]
    fn asymmetric_gaussian_message_matches_numerical_convolution() {
        let model = AdeRangingModel {
            positive_slope: 0.02,
            positive_intercept_cm: 10.0,
            negative_slope: 0.0,
            negative_intercept_cm: 20.0,
        };
        let point = Point2::new(100.0, 0.0);
        let variance = 400.0;
        let neighbor = GaussianBelief {
            mean: Point2::default(),
            xx: variance,
            xy: 0.0,
            yy: 100.0,
        };
        let measured_range = 130.0;
        let residual = measured_range - point.distance(neighbor.mean);
        let sigma = variance.sqrt();
        let steps = 20_000;
        let lower = -8.0 * sigma;
        let step = 16.0 * sigma / steps as f64;
        let normalizer = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
        let expected = (0..steps)
            .map(|index| {
                let offset = lower + (index as f64 + 0.5) * step;
                normalizer
                    * (-0.5 * (offset / sigma).powi(2)).exp()
                    * model.log_density(residual + offset, measured_range).exp()
                    * step
            })
            .sum::<f64>()
            .ln();

        let actual = log_ade_gaussian_message(point, neighbor, measured_range, model);

        assert!((actual - expected).abs() < 1e-6);
    }

    #[test]
    fn deterministic_generator_repeats_uniform_and_normal_draws() {
        let mut first = Rng::new(91);
        let mut second = Rng::new(91);
        for _ in 0..20 {
            assert_eq!(first.uniform().to_bits(), second.uniform().to_bits());
            assert_eq!(first.normal().to_bits(), second.normal().to_bits());
        }
    }
}
