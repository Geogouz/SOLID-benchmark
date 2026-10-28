use serde::{Deserialize, Serialize};

use crate::EngineError;

/// Highest admissible raw time of flight in nanoseconds.
pub(crate) const MAX_TOF_NS: f64 = 200.0;

/// Largest admissible calibrated distance and scale in centimetres, one kilometre, far beyond any UWB range.
pub(crate) const MAX_RANGE_CM: f64 = 1.0e5;

/// Largest accepted magnitude in centimetres of an anchor or start coordinate, an anchor height, a height node or
/// a bound of the operating area, one hundred kilometres. The scorers of the research workspace read a predicted
/// coordinate beyond it as missing.
pub const MAX_COORDINATE_CM: f64 = 1.0e7;

/// Whether a raw time of flight in nanoseconds is admissible under the given limit.
pub(crate) fn admissible_tof(tof_ns: f64, max_tof_ns: f64) -> bool {
    tof_ns.is_finite() && tof_ns >= 0.0 && tof_ns <= max_tof_ns
}

/// Whether a raw time of flight is admissible and the calibration turns it into a distance and a scale within the
/// sane range bound, which a profile's own calibration need not guarantee.
pub(crate) fn admissible_raw(tof_ns: f64, max_tof_ns: f64, calibration: &crate::CalibrationModel) -> bool {
    if !admissible_tof(tof_ns, max_tof_ns) {
        return false;
    }
    let distance = calibration.corrected_distance_cm(tof_ns);
    distance <= MAX_RANGE_CM && calibration.scale_cm(distance) <= MAX_RANGE_CM
}

/// Whether a coordinate in centimetres is finite and within the accepted magnitude.
pub(crate) fn sane_coordinate(value: f64) -> bool {
    value.is_finite() && value.abs() <= MAX_COORDINATE_CM
}

/// Rejects anchors and start positions that are not finite or lie beyond the accepted magnitude.
pub(crate) fn validate_geometry(problem: &PreparedProblem) -> Result<(), EngineError> {
    for anchor in &problem.anchors {
        let coordinates = [anchor.position.x, anchor.position.y, anchor.height_cm];
        if !coordinates.iter().all(|v| v.is_finite()) {
            return Err(EngineError::NonFiniteAnchor(anchor.id));
        }
        if !coordinates.iter().all(|v| sane_coordinate(*v)) {
            return Err(EngineError::AnchorOutOfRange(anchor.id));
        }
    }
    if !problem.initial_positions.iter().all(|p| sane_coordinate(p.x) && sane_coordinate(p.y)) {
        return Err(EngineError::InvalidProblem("initial positions must be finite and within 100 km"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point2 {
    pub x: f64,
    pub y: f64,
}

impl Point2 {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance(self, other: Self) -> f64 {
        (self.x - other.x).hypot(self.y - other.y)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum LocalizationMode {
    AnchorFree,
    AnchorAided,
    AnchorOnly,
}

impl LocalizationMode {
    pub const fn scorer_name(self) -> &'static str {
        match self {
            Self::AnchorFree => "AF",
            Self::AnchorAided => "AAC",
            Self::AnchorOnly => "ANC",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Anchor {
    pub id: u32,
    pub position: Point2,
    pub height_cm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibratedRangeSample {
    pub distance_cm: f64,
    pub scale_cm: f64,
}

impl CalibratedRangeSample {
    pub const fn new(distance_cm: f64, scale_cm: f64) -> Self {
        Self {
            distance_cm,
            scale_cm,
        }
    }

    pub fn is_valid(&self) -> bool {
        self.distance_cm.is_finite()
            && self.distance_cm >= 0.0
            && self.scale_cm.is_finite()
            && self.scale_cm > 0.0
    }

    /// Whether the engine accepts the sample into a range likelihood, valid and with its distance and scale within
    /// the sane range bound.
    pub(crate) fn is_admissible(&self) -> bool {
        self.is_valid() && self.distance_cm <= MAX_RANGE_CM && self.scale_cm <= MAX_RANGE_CM
    }
}

/// One observation of a ranging exchange in which agent `from` ranged agent or anchor `to`.
///
/// An observation is a responded exchange. Its existence counts as a successful exchange of the pair for the
/// presence law, whatever its samples. The samples are the raw times of flight in `tof_ns`, converted with the range
/// calibration, or the calibrated samples when `calibrated_samples` is not empty. A raw sample is admissible when it
/// is finite, not negative and at most the time of flight limit of 200 ns, and the calibration turns it into a
/// distance and a scale of at most one kilometre. A calibrated sample is admissible when its distance is finite, not
/// negative and at most one kilometre, and its scale is finite, positive and at most one kilometre. Only admissible
/// samples enter the range likelihood, so an observation without samples, or whose every sample is inadmissible,
/// adds a success to the presence law and no range term. Repeated observations of one pair, including both
/// directions of an agent pair, count as one exchange of the pair for the presence law, and their admissible
/// samples pool into one link with one persistent offset and excess. A ranging opportunity without any observation
/// between its two nodes is a failed attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeObservation {
    pub from: u32,
    pub to: u32,
    #[serde(default)]
    pub tof_ns: Vec<f64>,
    #[serde(default)]
    pub calibrated_samples: Vec<CalibratedRangeSample>,
    pub to_is_anchor: bool,
}

impl RangeObservation {
    pub fn sample_count(&self) -> usize {
        if self.calibrated_samples.is_empty() {
            self.tof_ns.len()
        } else {
            self.calibrated_samples.len()
        }
    }
}

/// A ranging attempt of the system: `from` tried to range `to`. Attempts carry no samples; an attempt that
/// produced ranges is an observation. Declared attempts define the ranging opportunities of the missing-link
/// factor, so a pair the system never tried is not treated as missing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LinkAttempt {
    pub from: u32,
    pub to: u32,
    pub to_is_anchor: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PreparedProblem {
    pub agent_ids: Vec<u32>,
    pub anchors: Vec<Anchor>,
    pub observations: Vec<RangeObservation>,
    pub initial_positions: Vec<Point2>,
    pub mode: LocalizationMode,
    /// The ranging attempts of the request. None means the system ranged all to all among its members, so every
    /// agent pair and every anchor with at least one observation anywhere in the request counts as an opportunity.
    /// Some lists the attempts explicitly, and the opportunities are these attempts plus the observed links.
    /// In every mode a failed attempt enters the presence law only when a path of observed pairs joins its two
    /// ends, with all anchors merged into one frame node, because only then is the distance it describes
    /// identified. An agent without any observed pair to the frame keeps no anchor failure in anchor aided and
    /// anchor only positioning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempts: Option<Vec<LinkAttempt>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ObservationAggregation {
    Median,
    AllSamples,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DimensionPolicy {
    #[default]
    Automatic,
    Planar,
    Spatial,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
/// Optimizer and range input settings shared by the cooperative solver, the initializer and the baselines.
pub struct SolveConfig {
    pub aggregation: ObservationAggregation,
    #[serde(default)]
    pub dimension_policy: DimensionPolicy,
    pub max_iterations: usize,
    pub gradient_tolerance: f64,
    pub history_size: usize,
    pub robust_delta: f64,
    pub max_tof_ns: f64,
    pub max_time_micros: u64,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            aggregation: ObservationAggregation::AllSamples,
            dimension_policy: DimensionPolicy::Automatic,
            max_iterations: 100,
            gradient_tolerance: 1e-5,
            history_size: 10,
            robust_delta: 0.7,
            max_tof_ns: MAX_TOF_NS,
            max_time_micros: 0,
        }
    }
}

impl SolveConfig {
    pub fn production() -> Self {
        Self {
            dimension_policy: DimensionPolicy::Planar,
            max_iterations: 10_000,
            max_time_micros: 1_000_000,
            ..Self::default()
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub enum TerminationReason {
    #[default]
    Converged,
    IterationLimit,
    TimeLimit,
    LineSearchFailed,
}

impl TerminationReason {
    pub(crate) fn merge(self, other: Self) -> Self {
        let priority = |reason| match reason {
            Self::Converged => 0,
            Self::IterationLimit => 1,
            Self::LineSearchFailed => 2,
            Self::TimeLimit => 3,
        };
        if priority(other) > priority(self) {
            other
        } else {
            self
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Converged => "converged",
            Self::IterationLimit => "iteration_limit",
            Self::TimeLimit => "time_limit",
            Self::LineSearchFailed => "line_search_failed",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SolveMetrics {
    pub elapsed_micros: u64,
    pub dimensions: usize,
    pub iterations: usize,
    pub objective_evaluations: usize,
    pub gradient_evaluations: usize,
    pub factors: usize,
    pub range_factor_evaluations: usize,
    pub coordinate_factor_evaluations: usize,
    pub converged: bool,
    pub termination: TerminationReason,
    #[serde(deserialize_with = "objective_or_unavailable")]
    pub final_objective: f64,
    pub starts: usize,
    /// The anchor-only count of agents one of whose adaptive posteriors the node guard cut, the refinement holding at most
    /// 2^23 nodes at once: the posterior its draws come from and, under the maximum likelihood reach, its range only
    /// posterior and its posteriors under the candidate laws the reach refined. An agent whose posterior the guard left
    /// without a kept cell is drawn uniformly over the area and counts here. None for the other solvers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guarded_supports: Option<usize>,
    /// The anchor-only largest certified share of its own mass that one of the adaptive posteriors of an agent left out
    /// of its kept cells, over the posteriors `guarded_supports` names: at most 1e-4 unless the node guard cut that
    /// posterior, one where the guard left it no kept cell, zero where nothing was left out. It bounds each posterior
    /// alone and not the likelihood the maximum likelihood reach integrates over the range only posteriors, whose
    /// presence terms may favour the cells left out; the reach certifies its own choice. None for the other solvers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub largest_dropped_mass: Option<f64>,
    /// The anchor-only nodes of the adaptive posteriors bounded above single cells, summed over the agents. None for
    /// the other solvers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub posterior_nodes: Option<usize>,
    /// The anchor-only cells whose log posterior was evaluated exactly, summed over the agents' posteriors, their
    /// draws and the maximum likelihood reach. None for the other solvers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub posterior_cells: Option<usize>,
}

fn objective_or_unavailable<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<f64, D::Error> {
    Option::<f64>::deserialize(deserializer).map(|value| value.unwrap_or(f64::INFINITY))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveResult {
    pub agent_ids: Vec<u32>,
    pub positions: Vec<Point2>,
    pub metrics: SolveMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn production_profile_is_frozen() {
        let config = SolveConfig::production();
        assert_eq!(config.dimension_policy, DimensionPolicy::Planar);
        assert_eq!(config.max_iterations, 10_000);
        assert_eq!(config.max_time_micros, 1_000_000);
    }

    #[test]
    fn problems_and_observations_reject_unknown_fields() {
        let problem = serde_json::json!({
            "agent_ids": [1, 2],
            "anchors": [{"id": 9, "position": {"x": 0.0, "y": 0.0}, "height_cm": 100.0}],
            "observations": [{"from": 1, "to": 9, "to_is_anchor": true, "calibrated_samples": [{"distance_cm": 300.0, "scale_cm": 8.0}]}],
            "initial_positions": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
            "mode": "AnchorAided",
        });
        let parsed: PreparedProblem = serde_json::from_value(problem.clone()).unwrap();
        assert_eq!(serde_json::from_value::<PreparedProblem>(serde_json::to_value(&parsed).unwrap()).unwrap(), parsed);
        let mut extra = problem.clone();
        extra["config"] = serde_json::json!({});
        assert!(serde_json::from_value::<PreparedProblem>(extra).unwrap_err().to_string().contains("unknown field `config`"));
        let mut extra = problem;
        extra["observations"][0]["ranges_cm"] = serde_json::json!([300.0]);
        assert!(serde_json::from_value::<PreparedProblem>(extra).unwrap_err().to_string().contains("unknown field `ranges_cm`"));
    }

    #[test]
    fn admissibility_bounds_samples_and_coordinates() {
        assert!(CalibratedRangeSample::new(MAX_RANGE_CM, 8.0).is_admissible());
        assert!(!CalibratedRangeSample::new(2.0 * MAX_RANGE_CM, 8.0).is_admissible());
        assert!(!CalibratedRangeSample::new(300.0, 2.0 * MAX_RANGE_CM).is_admissible());
        assert!(CalibratedRangeSample::new(2.0 * MAX_RANGE_CM, 8.0).is_valid());
        let calibration = crate::CalibrationModel::default();
        assert!(admissible_raw(20.0, MAX_TOF_NS, &calibration) && !admissible_raw(201.0, MAX_TOF_NS, &calibration));
        let steep = crate::CalibrationModel { range_slope_cm_per_ns: 1.0e9, ..calibration };
        assert!(!admissible_raw(20.0, MAX_TOF_NS, &steep));
        assert!(sane_coordinate(-MAX_COORDINATE_CM) && !sane_coordinate(1.5 * MAX_COORDINATE_CM) && !sane_coordinate(f64::NAN));
    }

    #[test]
    fn unavailable_objective_roundtrips_as_json_null() {
        let metrics = SolveMetrics {
            final_objective: f64::INFINITY,
            ..SolveMetrics::default()
        };
        let json = serde_json::to_value(&metrics).unwrap();
        assert!(json["final_objective"].is_null());
        let recovered: SolveMetrics = serde_json::from_value(json).unwrap();
        assert_eq!(recovered, metrics);
    }
}
