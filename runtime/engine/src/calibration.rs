/// The shared TWR-CLOUD-Bern range calibration, fitted on the corrected training table of the eleven
/// training scans. The benchmark's calibrated archives were produced with exactly this law, and the engine
/// applies it to raw time of flight inputs. The anchor only estimator gives every packet the scale of its
/// measured distance, so raw and pre calibrated inputs are identical there. The cooperative solver evaluates
/// the scale law at the candidate distance for ranges that report no scale of their own.
use serde::{Deserialize, Serialize};

use crate::EngineError;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationModel {
    pub range_slope_cm_per_ns: f64,
    pub range_offset_cm: f64,
    pub scale_slope: f64,
    pub scale_intercept_cm: f64,
    pub scale_min_cm: f64,
    pub scale_max_cm: f64,
}

/// The production law, fitted on the corrected training table of the eleven training scans.
impl Default for CalibrationModel {
    fn default() -> Self {
        Self {
            range_slope_cm_per_ns: 28.512_859_035_713_586,
            range_offset_cm: 19.579_395_666_091_16,
            scale_slope: 0.026_356_211_752_864_23,
            scale_intercept_cm: 4.649_664_346_296_255,
            scale_min_cm: 8.0,
            scale_max_cm: 55.0,
        }
    }
}

impl CalibrationModel {
    /// Corrected distance of a time of flight in nanoseconds, never negative.
    pub fn corrected_distance_cm(&self, tof_ns: f64) -> f64 {
        (self.range_slope_cm_per_ns * tof_ns + self.range_offset_cm).max(0.0)
    }

    /// Uncertainty scale of a corrected distance.
    pub fn scale_cm(&self, distance_cm: f64) -> f64 {
        self.scale_and_derivative(distance_cm).0
    }

    /// Uncertainty scale of a distance and its derivative, zero on the clamped ends.
    pub fn scale_and_derivative(&self, distance_cm: f64) -> (f64, f64) {
        let scale = self.scale_slope * distance_cm + self.scale_intercept_cm;
        if scale <= self.scale_min_cm {
            (self.scale_min_cm, 0.0)
        } else if scale >= self.scale_max_cm {
            (self.scale_max_cm, 0.0)
        } else {
            (scale, self.scale_slope)
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EngineError> {
        let ok = [self.range_slope_cm_per_ns, self.range_offset_cm, self.scale_slope, self.scale_intercept_cm, self.scale_min_cm, self.scale_max_cm]
            .into_iter()
            .all(f64::is_finite)
            && self.range_slope_cm_per_ns > 0.0
            && self.scale_min_cm > 0.0
            && self.scale_min_cm <= self.scale_max_cm;
        if ok {
            Ok(())
        } else {
            Err(EngineError::InvalidProfile("invalid range calibration"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CalibrationModel;

    #[test]
    fn reproduces_the_calibrated_evaluation_archive_bit_for_bit() {
        // Scenario 18-9 of the evaluation corpus, one sample, as stored in the calibrated archive.
        let distance = CalibrationModel::default().corrected_distance_cm(5.070_612_938_266_574);
        assert_eq!(distance.to_bits(), 164.157_067_599_551_45_f64.to_bits());
        assert_eq!(
            CalibrationModel::default().scale_cm(distance).to_bits(),
            8.976_222_780_679_281_f64.to_bits()
        );
    }

    #[test]
    fn scale_is_clamped_with_a_flat_derivative() {
        assert_eq!(CalibrationModel::default().scale_and_derivative(0.0), (8.0, 0.0));
        assert_eq!(CalibrationModel::default().scale_and_derivative(10_000.0), (55.0, 0.0));
        let (scale, derivative) = CalibrationModel::default().scale_and_derivative(1_000.0);
        assert!((8.0..55.0).contains(&scale));
        assert!(derivative > 0.0);
    }
}
