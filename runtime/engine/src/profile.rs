use serde::{Deserialize, Serialize};

use crate::{CalibrationModel, EngineError};

/// Multipliers of the distance-dependent range uncertainty, fitted for the
/// sampling policy used by the caller. Shared by AF and AAC.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RangeErrorProfile {
    pub positive_scale: f64,
    pub negative_scale: f64,
}


/// Optional block likelihood of the cooperative solver. Every link keeps its packets and contributes the
/// anchor only link model instead of the median residual density: Student t packet errors of scale `kappa`
/// times the packet scale, a spike and slab persistent excess and a persistent Gaussian offset, all relative
/// to the link's reference scale. Anchor links of AAC evaluate the slant
/// distance marginalized over the height nodes. With this model `range_error` is not used by the objective.
///
/// Every field is required except the switch of an ablation, `independent_packets`, which defaults to the method's
/// dependent packets. The offset shape and the rate of false early detections are learned values of the link model,
/// so a profile that left them out would run another model than the one fitted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlockRangeProfile {
    pub kappa: f64,
    pub excess_probability: f64,
    pub excess_mean_rel: f64,
    pub persistent_sd_rel: f64,
    pub height_nodes: Vec<f64>,
    pub height_weights: Vec<f64>,
    /// Two stage optimization: the median residual objective is minimized first and its layout starts the
    /// multistart of the block objective. The block likelihood is redescending, so a start in the right
    /// basin matters more than for the pseudo-Huber density.
    pub refine_from_median: bool,
    /// Degrees of freedom of the Student t packet error, 3 in the method; a large value approaches a Gaussian.
    pub nu: f64,
    /// Ablation: packets are modelled independently, each with its own offset and excess. False when absent.
    #[serde(default)]
    pub independent_packets: bool,
    /// Distribution of the persistent per link offset: Gaussian or Student t with 3 degrees of freedom whatever
    /// `nu`, both of scale `persistent_sd_rel` times the link's reference scale.
    pub offset_shape: OffsetShape,
    /// Rate per centimetre of a false early detection: each packet independently triggers on noise before the
    /// first path with probability `1 - exp(-rate s)` at detection distance `s`, and then reads uniformly in the
    /// exponential law of the trigger point below `s`. Zero disables it.
    pub early_detection_per_cm: f64,
}

/// Distribution of the zero mean persistent offset shared by the packets of one link.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffsetShape {
    #[default]
    Gaussian,
    StudentT,
}

impl Default for BlockRangeProfile {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            excess_probability: 0.25,
            excess_mean_rel: 1.0,
            persistent_sd_rel: 0.0,
            height_nodes: vec![0.0],
            height_weights: vec![1.0],
            refine_from_median: false,
            nu: 3.0,
            independent_packets: false,
            offset_shape: OffsetShape::Gaussian,
            early_detection_per_cm: 0.0,
        }
    }
}

impl BlockRangeProfile {
    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        let ok = self.kappa.is_finite()
            && self.kappa > 0.0
            && (0.0..=1.0).contains(&self.excess_probability)
            && self.excess_mean_rel.is_finite()
            && self.excess_mean_rel > 0.0
            && self.persistent_sd_rel.is_finite()
            && self.persistent_sd_rel >= 0.0
            && !self.height_nodes.is_empty()
            && self.height_nodes.len() == self.height_weights.len()
            && self.height_nodes.iter().all(|z| z.is_finite())
            && self.height_weights.iter().all(|w| w.is_finite() && *w > 0.0)
            && self.nu.is_finite()
            && self.nu > 0.0
            && self.early_detection_per_cm.is_finite()
            && self.early_detection_per_cm >= 0.0;
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidProfile("invalid block range settings"))
        }
    }
}


/// Fitted settings of the cooperative solver. `range_error` holds the scales of the median residual objective of
/// the first stage, `presence` the law of the attempted links. Sample-count selection belongs to the caller.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CooperativeProfile {
    pub range_error: RangeErrorProfile,
    /// The block likelihood range model, None keeps the median residual density.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_range: Option<BlockRangeProfile>,
    /// Range calibration applied to raw timings by the cooperative solver and its initializer, the built-in
    /// TWR-CLOUD-Bern law when absent. Lets a held-scan fold supply its own coefficients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationModel>,
    /// Starts of every multistart stage of the cooperative solver, 64 in the method.
    pub starts: usize,
    /// Ablation: the multistart starts from the request's initial positions instead of the geometric
    /// initializer's layout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_provided_start: Option<bool>,
    /// One presence law for every attempted link, agent pairs and anchor links alike, a log likelihood term of
    /// unit weight, with the reach estimated from the scenario.
    pub presence: crate::PresenceProfile,
}

impl CooperativeProfile {
    pub fn validate(&self, mode: crate::LocalizationMode) -> Result<(), EngineError> {
        if let Some(block) = &self.block_range {
            block.validate()?;
        }
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        if self.starts == 0 {
            return Err(EngineError::InvalidProfile("the multistart needs at least one start"));
        }
        self.presence.validate()?;
        let range = self.range_error;
        if ![range.positive_scale, range.negative_scale]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
        {
            return Err(EngineError::InvalidProfile(
                "invalid cooperative range scales",
            ));
        }
        match mode {
            crate::LocalizationMode::AnchorOnly => Err(EngineError::InvalidProfile(
                "ANC requires its own deployment profile",
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanarArea {
    pub minimum_x_cm: f64,
    pub maximum_x_cm: f64,
    pub minimum_y_cm: f64,
    pub maximum_y_cm: f64,
}

impl PlanarArea {
    pub(crate) fn validate(self) -> Result<(), EngineError> {
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
            Err(EngineError::InvalidProfile("invalid planar area"))
        }
    }
}

/// Deployment evidence of the anchor only estimator: the operating area, the fitted detection range and
/// the fitted link model settings.
///
/// Supplying this profile asserts that every declared agent-anchor pair was a ranging opportunity, so a
/// missing link is a censored detection rather than an unscheduled measurement.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeploymentProfile {
    pub area: PlanarArea,
    pub anchor_only: AnchorOnlyProfile,
}

/// Fitted and declared settings of the anchor only estimator.
///
/// Every field is required except the switches of a variant or an ablation, `independent_packets` and `decision`,
/// which default to the method's dependent packets and pairwise risk decision. The offset shape and the rate of false
/// early detections are learned values of the link model, required as in `BlockRangeProfile`. Raw timings are
/// converted with the built-in TWR-CLOUD-Bern law, never with a fold's calibration, so a caller with its own
/// calibration passes calibrated samples.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorOnlyProfile {
    pub kappa: f64,
    pub excess_probability: f64,
    pub excess_mean_rel: f64,
    /// Standard deviation, relative to the reported sample scale, of a zero mean offset that persists over all
    /// samples of one link (multipath bias that repeated ranging does not average out). Zero disables it.
    pub persistent_sd_rel: f64,
    pub height_nodes: Vec<f64>,
    pub height_weights: Vec<f64>,
    /// Side of a grid cell in centimetres, at least one. Every agent's adaptive posterior keeps the cells that hold all
    /// but a certified 1e-4 of its mass, without a cap on their number.
    pub grid_step: f64,
    /// Joint samples of the layout, at least eight.
    pub joint_samples: usize,
    pub seed: u64,
    /// Degrees of freedom of the Student t packet error, 3 in the method.
    pub nu: f64,
    /// Ablation: packets are modelled independently, each with its own offset and excess. False when absent.
    #[serde(default)]
    pub independent_packets: bool,
    /// The reported layout: the pairwise risk minimizer of the method, or the posterior coordinate mean. The risk
    /// minimizer when absent.
    #[serde(default)]
    pub decision: DecisionRule,
    /// One presence law for the anchor links, a log likelihood term of unit weight.
    pub presence: crate::PresenceProfile,
    /// Distribution of the persistent per link offset: Gaussian or Student t with 3 degrees of freedom whatever
    /// `nu`, both of scale `persistent_sd_rel` times the link's reference scale.
    pub offset_shape: OffsetShape,
    /// Rate per centimetre of a false early detection of every packet, as in `BlockRangeProfile`. Zero disables it.
    pub early_detection_per_cm: f64,
}

/// The decision rule of the anchor only estimator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionRule {
    #[default]
    Risk,
    Mean,
}

impl Default for AnchorOnlyProfile {
    fn default() -> Self {
        Self {
            kappa: 1.0,
            excess_probability: 0.25,
            excess_mean_rel: 1.0,
            persistent_sd_rel: 0.0,
            height_nodes: vec![0.0],
            height_weights: vec![1.0],
            grid_step: 10.0,
            joint_samples: 96,
            seed: 20260918,
            nu: 3.0,
            independent_packets: false,
            decision: DecisionRule::Risk,
            presence: crate::PresenceProfile::default(),
            offset_shape: OffsetShape::Gaussian,
            early_detection_per_cm: 0.0,
        }
    }
}

impl AnchorOnlyProfile {
    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        self.presence.validate()?;
        let ok = self.kappa.is_finite()
            && self.kappa > 0.0
            && (0.0..=1.0).contains(&self.excess_probability)
            && self.excess_mean_rel.is_finite()
            && self.excess_mean_rel > 0.0
            && self.persistent_sd_rel.is_finite()
            && self.persistent_sd_rel >= 0.0
            && !self.height_nodes.is_empty()
            && self.height_nodes.len() == self.height_weights.len()
            && self.height_weights.iter().all(|w| w.is_finite() && *w > 0.0)
            && self.grid_step.is_finite()
            && self.grid_step >= 1.0
            && self.joint_samples >= 8
            && self.nu.is_finite()
            && self.nu > 0.0
            && self.early_detection_per_cm.is_finite()
            && self.early_detection_per_cm >= 0.0;
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidProfile("invalid anchor only settings"))
        }
    }
}

impl DeploymentProfile {
    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        self.area.validate()?;
        self.anchor_only.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> DeploymentProfile {
        DeploymentProfile {
            area: PlanarArea {
                minimum_x_cm: -100.0,
                maximum_x_cm: 100.0,
                minimum_y_cm: -100.0,
                maximum_y_cm: 100.0,
            },
            anchor_only: AnchorOnlyProfile::default(),
        }
    }

    #[test]
    fn profile_rejects_invalid_learned_values() {
        let mut candidate = profile();
        candidate.anchor_only.presence.factor = 0.0;
        assert!(candidate.validate().is_err());

        let mut candidate = profile();
        candidate.area.maximum_x_cm = candidate.area.minimum_x_cm;
        assert!(candidate.validate().is_err());

        let mut candidate = profile();
        candidate.anchor_only.kappa = 0.0;
        assert!(candidate.validate().is_err());

        let block = BlockRangeProfile { early_detection_per_cm: -1.0e-4, ..BlockRangeProfile::default() };
        assert!(block.validate().is_err());

        // a grid step below a centimetre and fewer than eight joint samples are refused, not raised
        let mut candidate = profile();
        candidate.anchor_only.grid_step = 0.5;
        assert!(candidate.validate().is_err());
        let mut candidate = profile();
        candidate.anchor_only.joint_samples = 7;
        assert!(candidate.validate().is_err());
    }

    /// Whether `T` parses from `full` with each of its keys left out, keyed by the key left out.
    fn without_each_key<T: serde::de::DeserializeOwned>(full: &serde_json::Value) -> Vec<(String, bool)> {
        let full = full.as_object().unwrap();
        full.keys()
            .map(|key| {
                let mut partial = full.clone();
                partial.remove(key);
                (key.clone(), serde_json::from_value::<T>(serde_json::Value::Object(partial)).is_ok())
            })
            .collect()
    }

    #[test]
    fn a_profile_names_every_learned_value_and_only_the_switches_take_the_method() {
        // a profile that leaves out a learned value or a budget is refused instead of running with an engine default,
        // the offset shape and the rate of false early detections among them, which once defaulted to the model
        // without either
        let switches = ["independent_packets", "decision"];
        let anchor_only = serde_json::to_value(AnchorOnlyProfile::default()).unwrap();
        for (key, parsed) in without_each_key::<AnchorOnlyProfile>(&anchor_only) {
            assert_eq!(parsed, switches.contains(&key.as_str()), "{key}");
        }
        // the cap on the kept cells is gone with the adaptive posterior, and a profile that still names it is refused
        let mut capped = anchor_only.clone();
        capped["top_cells"] = serde_json::json!(4096);
        assert!(serde_json::from_value::<AnchorOnlyProfile>(capped).unwrap_err().to_string().contains("unknown field `top_cells`"));
        let block = serde_json::to_value(BlockRangeProfile::default()).unwrap();
        for (key, parsed) in without_each_key::<BlockRangeProfile>(&block) {
            assert_eq!(parsed, switches.contains(&key.as_str()), "{key}");
        }
        let cooperative = serde_json::json!({
            "range_error": {"positive_scale": 1.2, "negative_scale": 0.8},
            "block_range": block,
            "calibration": crate::CalibrationModel::default(),
            "starts": 64,
            "use_provided_start": false,
            "presence": crate::PresenceProfile::default(),
        });
        let optional = ["block_range", "calibration", "use_provided_start"];
        for (key, parsed) in without_each_key::<CooperativeProfile>(&cooperative) {
            assert_eq!(parsed, optional.contains(&key.as_str()), "{key}");
        }
    }
}
