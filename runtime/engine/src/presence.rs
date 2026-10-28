//! One presence law for every link type. An attempted link succeeds with a probability that falls with the
//! distance on the scale of a single reach parameter. A successful link contributes the log probability of
//! success at the candidate distance, a failed attempt the log probability of failure. The reach is estimated
//! from the scenario itself. The method's rule takes a quantile of the successful link distances times a factor.
//! The maximum likelihood rule chooses the reach on a small log spaced grid, or no presence term at all, by the
//! likelihood of the scenario's attempt outcomes.

use serde::{Deserialize, Serialize};

use crate::EngineError;

/// The shape of the success probability as a function of the distance over the reach.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresenceShape {
    /// exp(-d^2 / (2 reach^2)), the detection law of the anchor only estimator.
    Gaussian,
    /// sigmoid((reach - d) / (width reach)), a symmetric transition centred on the reach.
    Logistic,
}

/// How the reach is estimated from the scenario.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReachRule {
    /// `factor` times the `quantile` of the scenario's successful link distances, the method's rule.
    #[default]
    Quantile,
    /// The reach that maximizes the likelihood of the scenario's attempt outcomes among nine values from a quarter
    /// to four times the `quantile` distance, spaced by a factor of the square root of two, or no presence term at
    /// all when the outcomes are likelier independent of the distance. Without a failed attempt the law is off.
    Ml,
}

impl ReachRule {
    fn is_quantile(&self) -> bool {
        *self == Self::Quantile
    }
}

/// The presence law of a profile, its shape, reach rule and logistic width. The default is the law of the
/// method, selected on the Bern training scans: Gaussian, reach at three quarters of the 90th percentile of the
/// scenario's successful link distances. A profile names the shape, the quantile, the factor and the width. The
/// switches of a variant or an ablation, `reach` and `enabled`, default to the method's quantile rule with the
/// law on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PresenceProfile {
    pub shape: PresenceShape,
    /// The rule that estimates the reach from the scenario, the quantile rule when absent.
    #[serde(default, skip_serializing_if = "ReachRule::is_quantile")]
    pub reach: ReachRule,
    /// Quantile of the scenario's successful link distances that anchors the estimate, 1 is the longest link.
    pub quantile: f64,
    /// The estimated reach is `factor` times the quantile distance.
    pub factor: f64,
    /// Relative width of the logistic transition, the scale is `width` times the reach. Unused by the Gaussian.
    pub width: f64,
    /// Whether the law enters the likelihood at all. Off, no attempt carries evidence about distance, the
    /// ablation of the presence component. On when absent.
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

fn default_enabled() -> bool {
    true
}

impl Default for PresenceProfile {
    fn default() -> Self {
        Self {
            shape: PresenceShape::Gaussian,
            reach: ReachRule::Quantile,
            quantile: 0.9,
            factor: 0.75,
            width: 0.15,
            enabled: true,
        }
    }
}

impl PresenceProfile {
    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        let ok = self.quantile.is_finite()
            && self.quantile > 0.0
            && self.quantile <= 1.0
            && self.factor.is_finite()
            && self.factor > 0.0
            && self.width.is_finite()
            && self.width > 0.0;
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidProfile("invalid presence law settings"))
        }
    }

    /// The law of one scenario under the quantile rule, with the reach estimated from the successful link distances.
    /// None when the law is off or the scenario has no successful link, in which case no attempt failed for a reason
    /// the law could describe.
    pub(crate) fn resolve(&self, successful_distances_cm: &[f64]) -> Option<ResolvedPresence> {
        if !self.enabled {
            return None;
        }
        let reach = (self.quantile_distance(successful_distances_cm)? * self.factor).max(1.0);
        Some(ResolvedPresence {
            shape: self.shape,
            reach,
            width: self.width,
        })
    }

    /// The `quantile` of the scenario's successful link distances, None without a positive one.
    fn quantile_distance(&self, successful_distances_cm: &[f64]) -> Option<f64> {
        let mut sorted: Vec<f64> = successful_distances_cm.iter().copied().filter(|d| d.is_finite() && *d > 0.0).collect();
        if sorted.is_empty() {
            return None;
        }
        sorted.sort_by(f64::total_cmp);
        Some(sorted[quantile_index(self.quantile, sorted.len())])
    }

    /// The law of one scenario under the profile's reach rule. `outcomes` gives the log likelihood of the
    /// scenario's attempt outcomes under a candidate law and is only called by the maximum likelihood rule, once
    /// per candidate. The counts are those of the opportunities that enter the likelihood.
    pub(crate) fn resolve_with(
        &self,
        successful_distances_cm: &[f64],
        attempts: usize,
        successes: usize,
        outcomes: impl FnMut(&ResolvedPresence) -> f64,
    ) -> Option<ResolvedPresence> {
        match self.reach {
            ReachRule::Ml => self.resolve_ml(successful_distances_cm, attempts, successes, outcomes),
            ReachRule::Quantile => self.resolve(successful_distances_cm),
        }
    }

    /// The maximum likelihood law. Without the law the outcomes are independent of the distance, with the
    /// scenario's success ratio as their probability, the likeliest law that carries no distance information. A
    /// candidate reach replaces it only when its outcomes are strictly likelier, the first of equal candidates wins.
    fn resolve_ml(
        &self,
        successful_distances_cm: &[f64],
        attempts: usize,
        successes: usize,
        mut outcomes: impl FnMut(&ResolvedPresence) -> f64,
    ) -> Option<ResolvedPresence> {
        let candidates = self.ml_candidates(successful_distances_cm, attempts, successes)?;
        let values: Vec<f64> = candidates.iter().map(&mut outcomes).collect();
        ml_choice(&candidates, &values, attempts, successes)
    }

    /// The candidate laws of the maximum likelihood reach in the order they are judged, nine reaches from a quarter to
    /// four times the `quantile` distance. None where the rule settles without judging any: the law off, no failed
    /// attempt or no successful distance. A caller that judges every candidate in one pass takes them from here and
    /// chooses with `ml_choice`, as `resolve_with` does.
    pub(crate) fn ml_candidates(&self, successful_distances_cm: &[f64], attempts: usize, successes: usize) -> Option<Vec<ResolvedPresence>> {
        if !self.enabled || successes >= attempts {
            return None;
        }
        let base = self.quantile_distance(successful_distances_cm)?;
        Some(
            (-4..=4)
                .map(|step| ResolvedPresence { shape: self.shape, reach: (base * 2f64.powf(f64::from(step) / 2.0)).max(1.0), width: self.width })
                .collect(),
        )
    }
}

/// The maximum likelihood law among `candidates`, `values` the log likelihood of the attempt outcomes under each. The
/// outcomes independent of the distance, with the scenario's success ratio as their probability, are the law to beat,
/// a candidate replaces it only when strictly likelier, and the first of equal candidates wins.
pub(crate) fn ml_choice(candidates: &[ResolvedPresence], values: &[f64], attempts: usize, successes: usize) -> Option<ResolvedPresence> {
    let mut best = ml_null(attempts, successes);
    let mut chosen = None;
    for (law, &value) in candidates.iter().zip(values) {
        if value > best {
            best = value;
            chosen = Some(*law);
        }
    }
    chosen
}

/// The log likelihood of the attempt outcomes independent of the distance, with the scenario's success ratio as their
/// probability, the law the maximum likelihood reach must beat.
pub(crate) fn ml_null(attempts: usize, successes: usize) -> f64 {
    let log_term = |events: usize| if events == 0 { 0.0 } else { events as f64 * (events as f64 / attempts as f64).ln() };
    log_term(successes) + log_term(attempts - successes)
}

/// A presence law with its reach fixed for one scenario.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResolvedPresence {
    pub shape: PresenceShape,
    pub reach: f64,
    pub width: f64,
}

impl ResolvedPresence {
    /// The Gaussian probability of success at distance `d` clamped to [1e-12, 1 - 1e-12], whether the clamp acted, and
    /// the square of the reach.
    #[inline]
    fn gaussian_probability(&self, d: f64) -> (f64, bool, f64) {
        let r2 = self.reach * self.reach;
        let raw = (-d * d / (2.0 * r2)).exp();
        let p = raw.clamp(1e-12, 1.0 - 1e-12);
        (p, p != raw, r2)
    }

    /// Log probability that an attempt at distance `d` succeeds (`present`) or fails, and its derivative in `d`.
    /// Where the Gaussian probability is clamped to [1e-12, 1 - 1e-12] the value is flat, and so is its derivative.
    pub(crate) fn log_probability(&self, d: f64, present: bool) -> (f64, f64) {
        match self.shape {
            PresenceShape::Gaussian => {
                let (p, clamped, r2) = self.gaussian_probability(d);
                if present {
                    (p.ln(), if clamped { 0.0 } else { -d / r2 })
                } else {
                    ((-p).ln_1p(), if clamped { 0.0 } else { p / (1.0 - p) * d / r2 })
                }
            }
            PresenceShape::Logistic => {
                let scale = self.width * self.reach;
                let z = (self.reach - d) / scale;
                if present {
                    (-softplus(-z), -sigmoid(-z) / scale)
                } else {
                    (-softplus(z), sigmoid(z) / scale)
                }
            }
        }
    }

    /// Certified bounds of `log_probability(d, present).0` over the slant distances d in [a, b], 0 <= a <= b. Both
    /// shapes are monotone in the distance, a success's term falling and a failure's rising, so the bounds are the
    /// terms at the two ends, widened by BOUND_MARGIN of their size since the library's exponential and logarithm
    /// need not be monotone in their last bits. Minus and plus infinity where a term is not a number.
    ///
    /// A failure's Gaussian term ln(1 - p) moves by p / (1 - p) times a relative error of p, which grows without bound
    /// as p nears one, so a last bit of the exponential that is not monotone can move it far beyond that margin: an
    /// exponential within one unit in the last place of the exact one can give a farther distance a probability two
    /// units above the nearer one's, 2 epsilon p at most. Both bounds of a failure's Gaussian term therefore widen by
    /// 4 epsilon p / (1 - p) more, twice that shift at the near end's p, the largest of the interval: at most 8.9e-4 nats
    /// at the clamp, and far less below it. The terms themselves are those of `log_probability`, unchanged.
    pub(crate) fn bounds(&self, a: f64, b: f64, present: bool) -> (f64, f64) {
        let (near, far) = (self.log_probability(a, present).0, self.log_probability(b, present).0);
        if near.is_nan() || far.is_nan() {
            return (f64::NEG_INFINITY, f64::INFINITY);
        }
        let (low, high) = if present { (far, near) } else { (near, far) };
        let (low, high) = crate::link::widened(low, high, low.abs().max(high.abs()));
        if present || self.shape != PresenceShape::Gaussian {
            return (low, high);
        }
        let p = self.gaussian_probability(a).0;
        let shift = 4.0 * f64::EPSILON * p / (1.0 - p);
        (low - shift, high + shift)
    }
}

/// The index of the `quantile` among `len` sorted values, the smallest value with at least that share of the values
/// at or below it. The rank is clamped in floating point before the conversion, so no quantile leaves the range.
fn quantile_index(quantile: f64, len: usize) -> usize {
    let rank = (quantile * len as f64).ceil();
    if rank.is_finite() {
        rank.clamp(1.0, len.max(1) as f64) as usize - 1
    } else {
        len.saturating_sub(1)
    }
}

pub(crate) fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let e = value.exp();
        e / (1.0 + e)
    }
}

pub(crate) fn softplus(value: f64) -> f64 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

/// The per link median distance of a scenario's successful links, the input of the reach estimate.
pub(crate) fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const METHOD: &str = r#""shape": "gaussian", "quantile": 0.9, "factor": 0.75, "width": 0.15"#;

    #[test]
    fn a_profile_names_the_law_and_only_the_switches_take_the_method() {
        let method = PresenceProfile::default();
        assert_eq!(serde_json::from_str::<PresenceProfile>(&format!("{{{METHOD}}}")).unwrap(), method);
        let off: PresenceProfile = serde_json::from_str(&format!(r#"{{{METHOD}, "enabled": false}}"#)).unwrap();
        assert_eq!(off, PresenceProfile { enabled: false, ..method });
        assert!(off.resolve(&[300.0, 500.0]).is_none());
        // a law that leaves out one of its parameters is refused instead of taking the method's value
        for missing in ["shape", "quantile", "factor", "width"] {
            let partial: serde_json::Value = serde_json::from_str(&format!("{{{METHOD}}}")).unwrap();
            let mut partial = partial.as_object().unwrap().clone();
            partial.remove(missing);
            assert!(serde_json::from_value::<PresenceProfile>(serde_json::Value::Object(partial)).is_err(), "{missing}");
        }
        assert!(serde_json::from_str::<PresenceProfile>("{}").is_err());
        assert!(serde_json::from_str::<PresenceProfile>(&format!(r#"{{{METHOD}, "reach": 3}}"#)).is_err());
        assert!(serde_json::from_str::<PresenceProfile>(&format!(r#"{{{METHOD}, "reach_m": 3}}"#)).is_err());
    }

    #[test]
    fn the_reach_rule_is_the_quantile_unless_named() {
        let ml: PresenceProfile = serde_json::from_str(&format!(r#"{{{METHOD}, "reach": "ml"}}"#)).unwrap();
        assert_eq!(ml, PresenceProfile { reach: ReachRule::Ml, ..PresenceProfile::default() });
        assert!(serde_json::from_str::<PresenceProfile>(&format!(r#"{{{METHOD}, "reach": "median"}}"#)).is_err());
        // the default rule stays out of serialized profiles, so existing profiles serialize as before
        assert!(serde_json::to_value(PresenceProfile::default()).unwrap().get("reach").is_none());
        assert_eq!(serde_json::to_value(ml).unwrap()["reach"], "ml");
        assert!(ml.validate().is_ok());
    }

    #[test]
    fn the_maximum_likelihood_reach_searches_nine_reaches_against_no_law() {
        let ml = PresenceProfile { reach: ReachRule::Ml, ..PresenceProfile::default() };
        let distances: Vec<f64> = (1..=10).map(|i| i as f64 * 100.0).collect();
        // the candidates run from a quarter to four times the 90th percentile distance by factors of the root of two
        let mut reaches = Vec::new();
        let chosen = ml.resolve_with(&distances, 20, 10, |law| {
            reaches.push(law.reach);
            -1.0e9
        });
        assert!(chosen.is_none(), "no candidate beats outcomes independent of the distance");
        let expected: Vec<f64> = (-4..=4).map(|k| 900.0 * 2f64.powf(f64::from(k) / 2.0)).collect();
        assert_eq!(reaches, expected);
        // the likeliest candidate wins over the null law, the first of equal candidates stays
        let null = 20.0 * 0.5_f64.ln();
        let peak = |law: &ResolvedPresence| null + 1.0 - (law.reach / 1273.0).ln().abs();
        assert_eq!(ml.resolve_with(&distances, 20, 10, peak).unwrap().reach, expected[5]);
        assert_eq!(ml.resolve_with(&distances, 20, 10, |_| null + 1.0).unwrap().reach, expected[0]);
        // without a failure, without a successful distance or with the law disabled there is no law
        assert!(ml.resolve_with(&distances, 10, 10, |_| 0.0).is_none());
        assert!(ml.resolve_with(&[], 10, 0, |_| 0.0).is_none());
        assert!(PresenceProfile { enabled: false, ..ml }.resolve_with(&distances, 20, 10, |_| 0.0).is_none());
        // the quantile rule never asks for the outcomes
        let quantile = PresenceProfile::default().resolve_with(&distances, 20, 10, |_| unreachable!());
        assert_eq!(quantile, PresenceProfile::default().resolve(&distances));
    }

    #[test]
    fn derivatives_match_finite_differences_for_both_shapes() {
        for shape in [PresenceShape::Gaussian, PresenceShape::Logistic] {
            let law = ResolvedPresence {
                shape,
                reach: 1000.0,
                width: 0.15,
            };
            for present in [true, false] {
                for d in [100.0, 700.0, 1000.0, 1300.0, 2500.0] {
                    let (_, derivative) = law.log_probability(d, present);
                    let h = 1e-3;
                    let numeric = (law.log_probability(d + h, present).0 - law.log_probability(d - h, present).0) / (2.0 * h);
                    assert!((derivative - numeric).abs() < 1e-6 * (1.0 + numeric.abs()), "{shape:?} {present} {d}");
                }
            }
        }
    }

    /// Both shapes, successes and failures, short and long reaches: the bounds over random intervals from nothing to
    /// several reaches hold the term at 65 distances of the interval, the clamps of the Gaussian included, and an
    /// interval of one distance is bounded within 1e-9 of the term.
    #[test]
    fn bounds_hold_the_term_over_their_interval() {
        let mut state = 7_u64;
        let mut uniform = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1_u64 << 53) as f64
        };
        for shape in [PresenceShape::Gaussian, PresenceShape::Logistic] {
            for reach in [50.0, 400.0, 1500.0] {
                let law = ResolvedPresence { shape, reach, width: 0.15 };
                for present in [true, false] {
                    for trial in 0..2000 {
                        let width = if trial % 3 == 0 { 0.0 } else { 8.0 * reach * uniform() };
                        let a = 10.0 * reach * uniform();
                        let (low, high) = law.bounds(a, a + width, present);
                        for k in 0..=64 {
                            let d = a + width * k as f64 / 64.0;
                            let value = law.log_probability(d, present).0;
                            assert!(low <= value && value <= high, "{shape:?} {present} reach {reach}: {value} at {d} outside [{low}, {high}]");
                        }
                        if width == 0.0 {
                            assert!(high - low <= 1e-9 * (1.0 + low.abs()), "{low} {high}");
                        }
                    }
                }
            }
        }
        let law = ResolvedPresence { shape: PresenceShape::Gaussian, reach: 400.0, width: 0.15 };
        assert_eq!(law.bounds(f64::NAN, 3.0, true), (f64::NEG_INFINITY, f64::INFINITY));
    }

    /// An exponential within one unit in the last place of the exact one may give a farther distance a Gaussian
    /// probability two units above a nearer one's, and a failure's term ln(1 - p) then moves by p / (1 - p) times that
    /// shift: 2.2e-7 nats at p = 1 - 1e-9, ten thousand times the margin of its size. The bounds of a failure's term hold
    /// the terms of probabilities two units above the near end's and two below the far end's, at the clamp too, while the
    /// terms themselves stay those of `log_probability`.
    #[test]
    fn a_failures_gaussian_bounds_hold_an_exponential_off_by_its_last_bits() {
        let law = ResolvedPresence { shape: PresenceShape::Gaussian, reach: 400.0, width: 0.15 };
        // the probability as the law computes it, and one shifted by k units in the last place
        let r2 = 400.0_f64 * 400.0;
        let probability = |d: f64| (-d * d / (2.0 * r2)).exp().clamp(1e-12, 1.0 - 1e-12);
        let units = |p: f64, k: i64| f64::from_bits((p.to_bits() as i64 + k) as u64).clamp(1e-12, 1.0 - 1e-12);
        for a in [0.018, 0.05, 0.5, 5.0, 50.0, 500.0, 1e-4, 0.0] {
            for width in [0.0, 1e-3, 1.0, 100.0] {
                let b = a + width;
                let (low, high) = law.bounds(a, b, false);
                for (p, k) in [(probability(a), 2), (probability(b), -2)] {
                    let term = (-units(p, k)).ln_1p();
                    assert!(low <= term && term <= high, "{term} for p {p} shifted by {k} units, outside [{low}, {high}] over [{a}, {b}]");
                }
                let spread = law.log_probability(b, false).0 - law.log_probability(a, false).0;
                assert!(high - low <= spread + 2e-3, "[{low}, {high}] over [{a}, {b}]");
            }
        }
        // the terms are the sums the evaluator adds, unchanged by the bounds
        for d in [0.0, 0.018, 7.0, 399.0, 4000.0] {
            let p = probability(d);
            assert_eq!(law.log_probability(d, false).0.to_bits(), (-p).ln_1p().to_bits());
            assert_eq!(law.log_probability(d, true).0.to_bits(), p.ln().to_bits());
        }
    }

    #[test]
    fn clamped_probabilities_have_a_flat_value_and_no_slope() {
        let law = ResolvedPresence { shape: PresenceShape::Gaussian, reach: 400.0, width: 0.15 };
        // a success far beyond the reach and a failure at almost no distance sit on the clamp
        for (d, present) in [(3200.0, true), (4000.0, false), (1e-7, false), (1e-7, true)] {
            let (value, slope) = law.log_probability(d, present);
            let h = 1e-8_f64.max(d * 1e-6);
            let numeric = (law.log_probability(d + h, present).0 - law.log_probability((d - h).max(0.0), present).0)
                / (d + h - (d - h).max(0.0));
            assert!(value.is_finite() && (slope - numeric).abs() < 1e-6, "{d} {present}: {slope} {numeric}");
        }
    }

    #[test]
    fn the_reach_estimate_takes_the_quantile_times_the_factor() {
        let law = PresenceProfile { quantile: 1.0, factor: 0.5, ..PresenceProfile::default() };
        assert_eq!(law.resolve(&[300.0, 1200.0, 900.0]).unwrap().reach, 600.0);
        let ninety = PresenceProfile { quantile: 0.9, factor: 1.0, ..law };
        let distances: Vec<f64> = (1..=10).map(|i| i as f64 * 100.0).collect();
        assert_eq!(ninety.resolve(&distances).unwrap().reach, 900.0);
        // only positive finite distances count, and a scenario without one has no law
        assert_eq!(ninety.resolve(&[0.0, f64::NAN, 400.0]).unwrap().reach, 400.0);
        assert!(law.resolve(&[]).is_none());
    }
}
