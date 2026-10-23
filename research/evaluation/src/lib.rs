use std::collections::{BTreeMap, BTreeSet};

use solid_core::Point2;
use thiserror::Error;

mod g11;

pub use g11::{assess_g11, G11Case, G11Cell, G11Comparison, G11Dataset, G11Error};

const ONE_CM: f64 = 1.0;
const RATIO_ONE: f64 = 1.0;
const RATIO_ONE_PERCENT: f64 = 1.01;
const RATIO_TWO_PERCENT: f64 = 1.02;
const RATIO_FIVE_PERCENT: f64 = 1.05;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BalancedPercentile {
    P50,
    P75,
    P90,
    P95,
}

impl BalancedPercentile {
    const fn probability(self) -> f64 {
        match self {
            Self::P50 => 0.50,
            Self::P75 => 0.75,
            Self::P90 => 0.90,
            Self::P95 => 0.95,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Ratio {
    Finite(u64),
    PositiveInfinity,
}

impl Ratio {
    pub fn finite_value(self) -> Option<f64> {
        match self {
            Self::Finite(bits) => Some(f64::from_bits(bits)),
            Self::PositiveInfinity => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InclusiveRatioLimit {
    OnePercent,
    TwoPercent,
    FivePercent,
}

impl InclusiveRatioLimit {
    const fn value(self) -> f64 {
        match self {
            Self::OnePercent => RATIO_ONE_PERCENT,
            Self::TwoPercent => RATIO_TWO_PERCENT,
            Self::FivePercent => RATIO_FIVE_PERCENT,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaseScore {
    pub error_cm: f64,
    pub penalized_agent_count: u64,
}

#[derive(Clone, Debug)]
pub struct PreparedAnchoredScorer {
    agents: Vec<(u32, Point2)>,
    expected_identifiers: BTreeSet<u32>,
    failure_penalty_cm: f64,
}

impl PreparedAnchoredScorer {
    pub fn new(
        agent_truth: &BTreeMap<u32, Point2>,
        declared_anchor_ids: &[u32],
        known_anchor_positions: &BTreeMap<u32, Point2>,
    ) -> Result<Self, ScoringError> {
        if agent_truth.is_empty() {
            return Err(ScoringError::EmptyDomain);
        }
        let failure_penalty_cm =
            failure_penalty_cm(agent_truth, declared_anchor_ids, known_anchor_positions)?;
        Ok(Self {
            agents: agent_truth
                .iter()
                .map(|(&identifier, &point)| (identifier, point))
                .collect(),
            expected_identifiers: agent_truth.keys().copied().collect(),
            failure_penalty_cm,
        })
    }

    pub fn score(&self, predictions: &[(u32, Point2)]) -> Result<CaseScore, ScoringError> {
        let predictions = validate_prepared_predictions(predictions, &self.expected_identifiers)?;
        let mut sum = Neumaier::default();
        let mut penalized_agent_count = 0_u64;
        for &(identifier, truth) in &self.agents {
            let error = match predictions.get(&identifier).copied() {
                Some(prediction) if finite_point(prediction) => {
                    horizontal_distance_cm(prediction, truth)?
                }
                _ => {
                    penalized_agent_count = penalized_agent_count
                        .checked_add(1)
                        .ok_or(ScoringError::IntegerOverflow)?;
                    self.failure_penalty_cm
                }
            };
            sum.add(error)?;
        }
        Ok(CaseScore {
            error_cm: mean(sum, self.agents.len())?,
            penalized_agent_count,
        })
    }

    pub const fn failure_penalty_cm(&self) -> f64 {
        self.failure_penalty_cm
    }

    pub fn expected_agent_count(&self) -> Result<u64, ScoringError> {
        u64::try_from(self.agents.len()).map_err(|_| ScoringError::IntegerOverflow)
    }
}

#[derive(Clone, Debug)]
pub struct PreparedBalancedDistribution {
    ordered_cells: Vec<Vec<f64>>,
    candidates: Vec<f64>,
}

impl PreparedBalancedDistribution {
    pub fn new<K: Ord>(cells: &BTreeMap<K, Vec<f64>>) -> Result<Self, ScoringError> {
        if cells.is_empty() || cells.values().any(Vec::is_empty) {
            return Err(ScoringError::EmptyDomain);
        }
        let mut ordered_cells = Vec::with_capacity(cells.len());
        let mut candidates = Vec::new();
        for errors in cells.values() {
            let mut errors = errors
                .iter()
                .copied()
                .map(nonnegative_finite)
                .collect::<Result<Vec<_>, _>>()?;
            errors.sort_unstable_by(f64::total_cmp);
            candidates.extend_from_slice(&errors);
            ordered_cells.push(errors);
        }
        candidates.sort_unstable_by(f64::total_cmp);
        candidates.dedup_by(|left, right| left.to_bits() == right.to_bits());
        Ok(Self {
            ordered_cells,
            candidates,
        })
    }

    pub fn percentile(&self, percentile: BalancedPercentile) -> Result<f64, ScoringError> {
        let probability = percentile.probability();
        for &threshold in &self.candidates {
            if self.cdf(threshold)? >= probability {
                return Ok(positive_zero(threshold));
            }
        }
        Err(ScoringError::InvalidArithmetic)
    }

    pub fn cdf(&self, threshold: f64) -> Result<f64, ScoringError> {
        let threshold = nonnegative_finite(threshold)?;
        let mut cdf = Neumaier::default();
        for errors in &self.ordered_cells {
            let count = errors.partition_point(|error| *error <= threshold);
            cdf.add(finite(exact_count(count)? / exact_count(errors.len())?)?)?;
        }
        mean(cdf, self.ordered_cells.len())
    }

    pub fn upper_tail(&self, threshold: f64) -> Result<f64, ScoringError> {
        let threshold = nonnegative_finite(threshold)?;
        let mut tail = Neumaier::default();
        for errors in &self.ordered_cells {
            let count = errors.len() - errors.partition_point(|error| *error <= threshold);
            tail.add(finite(exact_count(count)? / exact_count(errors.len())?)?)?;
        }
        mean(tail, self.ordered_cells.len())
    }

    pub fn cell_count(&self) -> Result<u64, ScoringError> {
        u64::try_from(self.ordered_cells.len()).map_err(|_| ScoringError::IntegerOverflow)
    }

    pub fn observation_count(&self) -> Result<u64, ScoringError> {
        self.ordered_cells.iter().try_fold(0_u64, |total, cell| {
            total
                .checked_add(u64::try_from(cell.len()).map_err(|_| ScoringError::IntegerOverflow)?)
                .ok_or(ScoringError::IntegerOverflow)
        })
    }
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum ScoringError {
    #[error("scoring requires finite canonical truth geometry")]
    InvalidTruthGeometry,
    #[error("declared anchors and known anchor positions differ")]
    AnchorIdentityMismatch,
    #[error("prediction identifiers repeat or are not expected")]
    InvalidPredictionIdentifiers,
    #[error("the requested score has no expected agents, pairs, cells, or sample tracks")]
    EmptyDomain,
    #[error("a score, probability, or arithmetic intermediate is invalid")]
    InvalidArithmetic,
    #[error("an integer count cannot round trip exactly through binary64")]
    InexactCount,
    #[error("sample-specific endpoint keys differ from the expected sample tracks")]
    SampleTrackMismatch,
    #[error("integer gate arithmetic overflowed")]
    IntegerOverflow,
}

pub fn horizontal_distance_cm(left: Point2, right: Point2) -> Result<f64, ScoringError> {
    require_finite_point(left)?;
    require_finite_point(right)?;
    let dx = finite(left.x - right.x)?;
    let dy = finite(left.y - right.y)?;
    let abs_dx = dx.abs();
    let abs_dy = dy.abs();
    let hi = abs_dx.max(abs_dy);
    let lo = abs_dx.min(abs_dy);
    if hi == 0.0 {
        return Ok(0.0);
    }
    let ratio = finite(lo / hi)?;
    let square = finite(ratio * ratio)?;
    let radicand = finite(1.0 + square)?;
    let root = finite(libm::sqrt(radicand))?;
    positive_finite(hi * root)
}

pub fn failure_penalty_cm(
    agent_truth: &BTreeMap<u32, Point2>,
    declared_anchor_ids: &[u32],
    known_anchor_positions: &BTreeMap<u32, Point2>,
) -> Result<f64, ScoringError> {
    validate_truth_geometry(agent_truth, declared_anchor_ids, known_anchor_positions)?;
    let positions = agent_truth
        .values()
        .chain(known_anchor_positions.values())
        .copied()
        .collect::<Vec<_>>();
    let mut diameter = 0.0_f64;
    for left in 0..positions.len() {
        for right in left + 1..positions.len() {
            diameter = diameter.max(horizontal_distance_cm(positions[left], positions[right])?);
        }
    }
    Ok(positive_zero(diameter.max(ONE_CM)))
}

pub fn anchored_case_error_cm(
    predictions: &[(u32, Point2)],
    agent_truth: &BTreeMap<u32, Point2>,
    declared_anchor_ids: &[u32],
    known_anchor_positions: &BTreeMap<u32, Point2>,
) -> Result<CaseScore, ScoringError> {
    PreparedAnchoredScorer::new(agent_truth, declared_anchor_ids, known_anchor_positions)?
        .score(predictions)
}

pub fn pair_distance_case_error_cm(
    predictions: &[(u32, Point2)],
    agent_truth: &BTreeMap<u32, Point2>,
    declared_anchor_ids: &[u32],
    known_anchor_positions: &BTreeMap<u32, Point2>,
) -> Result<CaseScore, ScoringError> {
    if agent_truth.len() < 2 {
        return Err(ScoringError::EmptyDomain);
    }
    let penalty = failure_penalty_cm(agent_truth, declared_anchor_ids, known_anchor_positions)?;
    let predictions = validate_predictions(predictions, agent_truth)?;
    let agents = agent_truth.iter().collect::<Vec<_>>();
    let mut sum = Neumaier::default();
    let mut pair_count = 0_usize;
    let mut penalized_agent_count = BTreeSet::new();
    for left in 0..agents.len() {
        for right in left + 1..agents.len() {
            let (&left_id, &left_truth) = agents[left];
            let (&right_id, &right_truth) = agents[right];
            let predicted = predictions
                .get(&left_id)
                .copied()
                .zip(predictions.get(&right_id).copied());
            let pair_error = match predicted {
                Some((left_prediction, right_prediction))
                    if finite_point(left_prediction) && finite_point(right_prediction) =>
                {
                    let predicted_distance =
                        horizontal_distance_cm(left_prediction, right_prediction)?;
                    let truth_distance = horizontal_distance_cm(left_truth, right_truth)?;
                    positive_finite((predicted_distance - truth_distance).abs())?
                }
                _ => {
                    if !predictions
                        .get(&left_id)
                        .is_some_and(|point| finite_point(*point))
                    {
                        penalized_agent_count.insert(left_id);
                    }
                    if !predictions
                        .get(&right_id)
                        .is_some_and(|point| finite_point(*point))
                    {
                        penalized_agent_count.insert(right_id);
                    }
                    penalty
                }
            };
            sum.add(pair_error)?;
            pair_count = pair_count
                .checked_add(1)
                .ok_or(ScoringError::IntegerOverflow)?;
        }
    }
    Ok(CaseScore {
        error_cm: mean(sum, pair_count)?,
        penalized_agent_count: u64::try_from(penalized_agent_count.len())
            .map_err(|_| ScoringError::IntegerOverflow)?,
    })
}

pub fn balanced_percentile<K: Ord>(
    cells: &BTreeMap<K, Vec<f64>>,
    percentile: BalancedPercentile,
) -> Result<f64, ScoringError> {
    PreparedBalancedDistribution::new(cells)?.percentile(percentile)
}

pub fn dataset_endpoint(
    expected_sample_counts: &[u32],
    sample_percentiles: &BTreeMap<u32, f64>,
) -> Result<f64, ScoringError> {
    if expected_sample_counts.is_empty()
        || !expected_sample_counts
            .windows(2)
            .all(|pair| pair[0] < pair[1])
        || expected_sample_counts
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            != sample_percentiles.keys().copied().collect()
    {
        return Err(ScoringError::SampleTrackMismatch);
    }
    let mut sum = Neumaier::default();
    for sample_count in expected_sample_counts {
        sum.add(nonnegative_finite(sample_percentiles[sample_count])?)?;
    }
    mean(sum, expected_sample_counts.len())
}

pub fn nonnegative_ratio(left: f64, right: f64) -> Result<Ratio, ScoringError> {
    let left = nonnegative_finite(left)?;
    let right = nonnegative_finite(right)?;
    if right == 0.0 {
        return if left == 0.0 {
            Ok(Ratio::Finite(RATIO_ONE.to_bits()))
        } else {
            Ok(Ratio::PositiveInfinity)
        };
    }
    Ok(Ratio::Finite(positive_finite(left / right)?.to_bits()))
}

pub fn is_strict_ratio_win(left: f64, right: f64) -> Result<bool, ScoringError> {
    Ok(matches!(
        nonnegative_ratio(left, right)?,
        Ratio::Finite(bits) if f64::from_bits(bits) < RATIO_ONE
    ))
}

pub fn passes_inclusive_ratio_limit(
    left: f64,
    right: f64,
    limit: InclusiveRatioLimit,
) -> Result<bool, ScoringError> {
    Ok(matches!(
        nonnegative_ratio(left, right)?,
        Ratio::Finite(bits) if f64::from_bits(bits) <= limit.value()
    ))
}

pub fn cluster_favors_candidate(
    candidate_errors: &[f64],
    jin_errors: &[f64],
) -> Result<bool, ScoringError> {
    if candidate_errors.is_empty() || candidate_errors.len() != jin_errors.len() {
        return Err(ScoringError::EmptyDomain);
    }
    let mut differences = Neumaier::default();
    for (&candidate, &jin) in candidate_errors.iter().zip(jin_errors) {
        let candidate = nonnegative_finite(candidate)?;
        let jin = nonnegative_finite(jin)?;
        differences.add(finite(candidate - jin)?)?;
    }
    Ok(positive_zero(mean(differences, candidate_errors.len())?) < 0.0)
}

pub fn required_favorable_clusters(cluster_count: u64) -> Result<u64, ScoringError> {
    let quotient = cluster_count / 5;
    let remainder = cluster_count % 5;
    quotient
        .checked_mul(3)
        .and_then(|base| base.checked_add((3 * remainder).div_ceil(5)))
        .ok_or(ScoringError::IntegerOverflow)
}

pub fn decision_bits(value: f64) -> Result<String, ScoringError> {
    Ok(format!("{:016x}", nonnegative_finite(value)?.to_bits()))
}

fn validate_truth_geometry(
    agent_truth: &BTreeMap<u32, Point2>,
    declared_anchor_ids: &[u32],
    known_anchor_positions: &BTreeMap<u32, Point2>,
) -> Result<(), ScoringError> {
    if declared_anchor_ids
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
        || declared_anchor_ids.iter().copied().collect::<BTreeSet<_>>()
            != known_anchor_positions.keys().copied().collect()
        || declared_anchor_ids
            .iter()
            .any(|identifier| agent_truth.contains_key(identifier))
    {
        return Err(ScoringError::AnchorIdentityMismatch);
    }
    if agent_truth
        .values()
        .chain(known_anchor_positions.values())
        .any(|point| !finite_point(*point))
    {
        return Err(ScoringError::InvalidTruthGeometry);
    }
    Ok(())
}

fn validate_predictions(
    predictions: &[(u32, Point2)],
    expected: &BTreeMap<u32, Point2>,
) -> Result<BTreeMap<u32, Point2>, ScoringError> {
    let mut parsed = BTreeMap::new();
    for &(identifier, point) in predictions {
        if !expected.contains_key(&identifier) || parsed.insert(identifier, point).is_some() {
            return Err(ScoringError::InvalidPredictionIdentifiers);
        }
    }
    Ok(parsed)
}

fn validate_prepared_predictions(
    predictions: &[(u32, Point2)],
    expected: &BTreeSet<u32>,
) -> Result<BTreeMap<u32, Point2>, ScoringError> {
    let mut parsed = BTreeMap::new();
    for &(identifier, point) in predictions {
        if !expected.contains(&identifier) || parsed.insert(identifier, point).is_some() {
            return Err(ScoringError::InvalidPredictionIdentifiers);
        }
    }
    Ok(parsed)
}

fn mean(sum: Neumaier, count: usize) -> Result<f64, ScoringError> {
    if count == 0 {
        return Err(ScoringError::EmptyDomain);
    }
    positive_finite(sum.finish()? / exact_count(count)?)
}

fn exact_count(count: usize) -> Result<f64, ScoringError> {
    const MAX_CONSECUTIVE_BINARY64_INTEGER: u64 = 1_u64 << 53;

    if u64::try_from(count).map_or(true, |value| value > MAX_CONSECUTIVE_BINARY64_INTEGER) {
        return Err(ScoringError::InexactCount);
    }
    let value = count as f64;
    if value as usize != count {
        return Err(ScoringError::InexactCount);
    }
    Ok(value)
}

fn finite_point(point: Point2) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

fn require_finite_point(point: Point2) -> Result<(), ScoringError> {
    finite_point(point)
        .then_some(())
        .ok_or(ScoringError::InvalidTruthGeometry)
}

fn nonnegative_finite(value: f64) -> Result<f64, ScoringError> {
    if value.is_finite() && value >= 0.0 {
        Ok(positive_zero(value))
    } else {
        Err(ScoringError::InvalidArithmetic)
    }
}

fn positive_finite(value: f64) -> Result<f64, ScoringError> {
    if value.is_finite() {
        Ok(positive_zero(value))
    } else {
        Err(ScoringError::InvalidArithmetic)
    }
}

fn finite(value: f64) -> Result<f64, ScoringError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ScoringError::InvalidArithmetic)
    }
}

fn positive_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Neumaier {
    sum: f64,
    correction: f64,
}

impl Neumaier {
    fn add(&mut self, value: f64) -> Result<(), ScoringError> {
        let value = finite(value)?;
        let trial = finite(self.sum + value)?;
        let correction = if self.sum.abs() >= value.abs() {
            let difference = finite(self.sum - trial)?;
            finite(difference + value)?
        } else {
            let difference = finite(value - trial)?;
            finite(difference + self.sum)?
        };
        self.correction = finite(self.correction + correction)?;
        self.sum = trial;
        Ok(())
    }

    fn finish(self) -> Result<f64, ScoringError> {
        finite(self.sum + self.correction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn truth() -> BTreeMap<u32, Point2> {
        BTreeMap::from([
            (1, Point2::new(0.0, 0.0)),
            (2, Point2::new(3.0, 4.0)),
            (3, Point2::new(6.0, 8.0)),
        ])
    }

    fn anchors() -> BTreeMap<u32, Point2> {
        BTreeMap::from([(10, Point2::new(100.0, 0.0))])
    }

    #[test]
    fn normative_constants_have_frozen_bits() {
        assert_eq!(
            BalancedPercentile::P50.probability().to_bits(),
            0x3fe0_0000_0000_0000
        );
        assert_eq!(
            BalancedPercentile::P75.probability().to_bits(),
            0x3fe8_0000_0000_0000
        );
        assert_eq!(
            BalancedPercentile::P90.probability().to_bits(),
            0x3fec_cccc_cccc_cccd
        );
        assert_eq!(
            BalancedPercentile::P95.probability().to_bits(),
            0x3fee_6666_6666_6666
        );
        assert_eq!(RATIO_ONE_PERCENT.to_bits(), 0x3ff0_28f5_c28f_5c29);
        assert_eq!(RATIO_TWO_PERCENT.to_bits(), 0x3ff0_51eb_851e_b852);
        assert_eq!(RATIO_FIVE_PERCENT.to_bits(), 0x3ff0_cccc_cccc_cccd);
    }

    #[test]
    fn scaling_safe_norm_uses_libm_and_normalizes_zero() {
        assert_eq!(
            horizontal_distance_cm(Point2::new(0.0, 0.0), Point2::new(3.0, 4.0)).unwrap(),
            5.0
        );
        assert_eq!(
            horizontal_distance_cm(Point2::new(0.0, 0.0), Point2::new(3.0, 7.0))
                .unwrap()
                .to_bits(),
            0x401e_768d_399d_c470
        );
        assert_eq!(
            horizontal_distance_cm(Point2::new(-0.0, 0.0), Point2::new(0.0, -0.0))
                .unwrap()
                .to_bits(),
            0
        );
        assert!(horizontal_distance_cm(
            Point2::new(f64::MAX, f64::MAX),
            Point2::new(-f64::MAX, 0.0)
        )
        .is_err());
    }

    #[test]
    fn geometry_diameter_penalty_uses_agents_then_anchors_and_one_cm_floor() {
        assert_eq!(
            failure_penalty_cm(&truth(), &[10], &anchors()).unwrap(),
            100.0
        );
        let singleton = BTreeMap::from([(1, Point2::new(0.0, 0.0))]);
        assert_eq!(
            failure_penalty_cm(&singleton, &[], &BTreeMap::new()).unwrap(),
            1.0
        );
        assert_eq!(
            failure_penalty_cm(&truth(), &[11], &anchors()),
            Err(ScoringError::AnchorIdentityMismatch)
        );
    }

    #[test]
    fn anchored_score_penalizes_each_missing_or_nonfinite_expected_agent() {
        let predictions = [(1, Point2::new(0.0, 0.0)), (2, Point2::new(f64::NAN, 4.0))];
        let score = anchored_case_error_cm(&predictions, &truth(), &[10], &anchors()).unwrap();
        assert_eq!(score.error_cm, 200.0 / 3.0);
        assert_eq!(score.penalized_agent_count, 2);

        let duplicate = [(1, Point2::new(0.0, 0.0)), (1, Point2::new(1.0, 1.0))];
        assert_eq!(
            anchored_case_error_cm(&duplicate, &truth(), &[10], &anchors()),
            Err(ScoringError::InvalidPredictionIdentifiers)
        );
    }

    #[test]
    fn prepared_anchored_scorer_reuses_geometry_without_changing_results() {
        let scorer = PreparedAnchoredScorer::new(&truth(), &[10], &anchors()).unwrap();
        let complete = [
            (1, Point2::new(0.0, 0.0)),
            (2, Point2::new(3.0, 4.0)),
            (3, Point2::new(6.0, 8.0)),
        ];
        let incomplete = [(1, Point2::new(0.0, 0.0))];
        assert_eq!(scorer.failure_penalty_cm(), 100.0);
        assert_eq!(scorer.expected_agent_count().unwrap(), 3);
        assert_eq!(scorer.score(&complete).unwrap().error_cm.to_bits(), 0);
        assert_eq!(
            scorer.score(&incomplete).unwrap(),
            anchored_case_error_cm(&incomplete, &truth(), &[10], &anchors()).unwrap()
        );
        assert_eq!(
            scorer.score(&[(99, Point2::new(0.0, 0.0))]),
            Err(ScoringError::InvalidPredictionIdentifiers)
        );
    }

    #[test]
    fn pair_score_uses_lexicographic_pairs_and_divides_once() {
        let predictions = [
            (1, Point2::new(0.0, 0.0)),
            (2, Point2::new(0.0, 5.0)),
            (3, Point2::new(0.0, 10.0)),
        ];
        let score = pair_distance_case_error_cm(&predictions, &truth(), &[10], &anchors()).unwrap();
        assert_eq!(score.error_cm.to_bits(), 0);
        assert_eq!(score.penalized_agent_count, 0);
    }

    #[test]
    fn pair_score_does_not_cap_a_finite_prediction_error() {
        let truth = BTreeMap::from([(1, Point2::new(0.0, 0.0)), (2, Point2::new(10.0, 0.0))]);
        let anchors = BTreeMap::from([(10, Point2::new(0.0, 0.0))]);
        let predictions = [(1, Point2::new(0.0, 0.0)), (2, Point2::new(100.0, 0.0))];

        let score = pair_distance_case_error_cm(&predictions, &truth, &[10], &anchors).unwrap();

        assert_eq!(score.error_cm, 90.0);
        assert_eq!(score.penalized_agent_count, 0);
    }

    #[test]
    fn neumaier_recovers_ordered_cancellation() {
        let mut sum = Neumaier::default();
        for value in [1.0e16, 1.0, -1.0e16] {
            sum.add(value).unwrap();
        }
        assert_eq!(sum.finish().unwrap(), 1.0);
    }

    #[test]
    fn balanced_percentile_weights_cells_equally_not_rows() {
        let cells = BTreeMap::from([("large", vec![0.0; 100]), ("small", vec![10.0])]);
        assert_eq!(
            balanced_percentile(&cells, BalancedPercentile::P50).unwrap(),
            0.0
        );
        assert_eq!(
            balanced_percentile(&cells, BalancedPercentile::P75).unwrap(),
            10.0
        );
        assert!(balanced_percentile(
            &BTreeMap::<u8, Vec<f64>>::from([(0, Vec::new())]),
            BalancedPercentile::P50
        )
        .is_err());
    }

    #[test]
    fn balanced_percentile_cell_order_is_deterministic_under_cancellation() {
        let cells = BTreeMap::from([
            (1_u8, vec![0.0, 1.0]),
            (2_u8, vec![1.0, 1.0]),
            (3_u8, vec![0.0, 0.0]),
        ]);
        assert_eq!(
            balanced_percentile(&cells, BalancedPercentile::P50).unwrap(),
            0.0
        );
        assert_eq!(
            balanced_percentile(&cells, BalancedPercentile::P75).unwrap(),
            1.0
        );
    }

    #[test]
    fn prepared_distribution_sorts_once_for_percentiles_and_cdfs() {
        let cells = BTreeMap::from([(1_u8, vec![-0.0, 1.0]), (2_u8, vec![1.0, 2.0])]);
        let prepared = PreparedBalancedDistribution::new(&cells).unwrap();
        assert_eq!(prepared.cell_count().unwrap(), 2);
        assert_eq!(
            prepared.percentile(BalancedPercentile::P50).unwrap(),
            balanced_percentile(&cells, BalancedPercentile::P50).unwrap()
        );
        assert_eq!(prepared.cdf(1.0).unwrap(), 0.75);
        assert_eq!(prepared.cdf(-0.0).unwrap(), 0.25);
        assert_eq!(prepared.upper_tail(1.0).unwrap(), 0.25);
        assert_eq!(prepared.observation_count().unwrap(), 4);
        assert_eq!(
            prepared.cdf(f64::INFINITY),
            Err(ScoringError::InvalidArithmetic)
        );
    }

    #[test]
    fn sample_endpoint_requires_the_exact_ascending_track_set() {
        let values = BTreeMap::from([(1, 1.0), (10, 3.0)]);
        assert_eq!(dataset_endpoint(&[1, 10], &values).unwrap(), 2.0);
        assert_eq!(
            dataset_endpoint(&[10, 1], &values),
            Err(ScoringError::SampleTrackMismatch)
        );
    }

    #[test]
    fn ratio_rules_and_ulp_boundaries_are_exact() {
        assert_eq!(
            nonnegative_ratio(0.0, 0.0).unwrap(),
            Ratio::Finite(RATIO_ONE.to_bits())
        );
        assert_eq!(
            nonnegative_ratio(1.0, 0.0).unwrap(),
            Ratio::PositiveInfinity
        );
        assert!(is_strict_ratio_win(0.0, 1.0).unwrap());
        assert!(!is_strict_ratio_win(1.0, 1.0).unwrap());

        for (threshold, limit) in [
            (RATIO_ONE_PERCENT, InclusiveRatioLimit::OnePercent),
            (RATIO_TWO_PERCENT, InclusiveRatioLimit::TwoPercent),
            (RATIO_FIVE_PERCENT, InclusiveRatioLimit::FivePercent),
        ] {
            assert!(passes_inclusive_ratio_limit(threshold, 1.0, limit).unwrap());
            let next = f64::from_bits(threshold.to_bits() + 1);
            assert!(!passes_inclusive_ratio_limit(next, 1.0, limit).unwrap());
        }
    }

    #[test]
    fn cluster_direction_is_strict_and_negative_zero_is_a_tie() {
        assert!(cluster_favors_candidate(&[1.0, 2.0], &[2.0, 3.0]).unwrap());
        assert!(!cluster_favors_candidate(&[1.0, 2.0], &[1.0, 2.0]).unwrap());
        assert!(!cluster_favors_candidate(&[-0.0], &[0.0]).unwrap());
    }

    #[test]
    fn exact_three_fifths_ceiling_handles_boundaries_and_large_counts() {
        assert_eq!(required_favorable_clusters(0).unwrap(), 0);
        assert_eq!(required_favorable_clusters(1).unwrap(), 1);
        assert_eq!(required_favorable_clusters(5).unwrap(), 3);
        assert_eq!(required_favorable_clusters(6).unwrap(), 4);
        assert_eq!(
            required_favorable_clusters(u64::MAX).unwrap(),
            11_068_046_444_225_730_969
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn exact_count_rejects_values_above_the_binary64_consecutive_integer_limit() {
        assert_eq!(exact_count(1_usize << 53).unwrap(), (1_u64 << 53) as f64);
        assert_eq!(
            exact_count((1_usize << 53) + 1),
            Err(ScoringError::InexactCount)
        );
        assert_eq!(exact_count(usize::MAX), Err(ScoringError::InexactCount));
    }

    #[test]
    fn decision_values_serialize_as_lowercase_binary64_bits() {
        assert_eq!(decision_bits(-0.0).unwrap(), "0000000000000000");
        assert_eq!(decision_bits(1.01).unwrap(), "3ff028f5c28f5c29");
    }
}
