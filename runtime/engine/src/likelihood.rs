use crate::{CalibratedRangeSample, CalibrationModel, LocalizationMode};

/// The pseudo Huber range model of the geometry initializer.
///
/// A raw Bern time of flight is converted with the shared calibration and carries the calibration's distance
/// dependent scale. A calibrated sample keeps its reported scale. In the agent only mode the scale of a raw
/// measurement follows the predicted distance instead of the measured one.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RangeModel {
    calibration: CalibrationModel,
    robust_delta: f64,
    predicted_scale: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PreparedMeasurement {
    calibrated_distance_cm: f64,
    calibrated_scale_cm: f64,
    fixed_scale: bool,
}

impl RangeModel {
    pub(crate) fn new(robust_delta: f64) -> Self {
        Self {
            calibration: CalibrationModel::default(),
            robust_delta,
            predicted_scale: false,
        }
    }

    pub(crate) fn with_calibration(mut self, calibration: CalibrationModel) -> Self {
        self.calibration = calibration;
        self
    }

    pub(crate) fn for_mode(robust_delta: f64, mode: LocalizationMode) -> Self {
        Self {
            predicted_scale: mode == LocalizationMode::AnchorFree,
            ..Self::new(robust_delta)
        }
    }

    pub(crate) fn prepare_measurement(&self, tof_ns: f64) -> PreparedMeasurement {
        let calibrated_distance_cm = self.calibration.corrected_distance_cm(tof_ns);
        PreparedMeasurement {
            calibrated_distance_cm,
            calibrated_scale_cm: self.calibration.scale_cm(calibrated_distance_cm),
            fixed_scale: false,
        }
    }

    pub(crate) fn prepare_calibrated_measurement(
        &self,
        sample: CalibratedRangeSample,
    ) -> PreparedMeasurement {
        PreparedMeasurement {
            calibrated_distance_cm: sample.distance_cm,
            calibrated_scale_cm: sample.scale_cm,
            fixed_scale: true,
        }
    }

    pub(crate) fn estimated_distance(&self, measurement: &PreparedMeasurement) -> f64 {
        measurement.calibrated_distance_cm
    }

    /// Pseudo Huber loss of a predicted distance and its derivative with respect to that distance.
    pub(crate) fn loss_and_distance_gradient_prepared(
        &self,
        distance_cm: f64,
        measurement: &PreparedMeasurement,
    ) -> (f64, f64) {
        let measured = measurement.calibrated_distance_cm;
        let (scale, scale_derivative) = if self.predicted_scale && !measurement.fixed_scale {
            self.calibration.scale_and_derivative(distance_cm)
        } else {
            (measurement.calibrated_scale_cm, 0.0)
        };
        let residual = measured - distance_cm;
        let normalized = residual / scale;
        let delta = self.robust_delta;
        let root = (1.0 + (normalized / delta).powi(2)).sqrt();
        let loss = delta * delta * (root - 1.0);
        let psi = normalized / root;

        let normalized_derivative = -1.0 / scale - residual * scale_derivative / scale.powi(2);
        (loss, psi * normalized_derivative)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_difference(model: &RangeModel, distance: f64, tof_ns: f64) {
        let measurement = model.prepare_measurement(tof_ns);
        let epsilon = 1e-4;
        let (_, gradient) = model.loss_and_distance_gradient_prepared(distance, &measurement);
        let left = model
            .loss_and_distance_gradient_prepared(distance - epsilon, &measurement)
            .0;
        let right = model
            .loss_and_distance_gradient_prepared(distance + epsilon, &measurement)
            .0;
        assert!((gradient - (right - left) / (2.0 * epsilon)).abs() < 1e-7);
    }

    #[test]
    fn robust_gradient_matches_finite_difference() {
        finite_difference(&RangeModel::new(0.7), 650.0, 21.0);
    }

    #[test]
    fn anchor_free_gradient_matches_finite_difference() {
        finite_difference(
            &RangeModel::for_mode(0.7, LocalizationMode::AnchorFree),
            650.0,
            21.0,
        );
    }

    #[test]
    fn calibrated_sample_matches_the_raw_calibration_path() {
        let model = RangeModel::new(0.7);
        let raw = model.prepare_measurement(21.0);
        let calibrated = model.prepare_calibrated_measurement(CalibratedRangeSample::new(
            raw.calibrated_distance_cm,
            raw.calibrated_scale_cm,
        ));
        assert_eq!(
            model.loss_and_distance_gradient_prepared(650.0, &raw),
            model.loss_and_distance_gradient_prepared(650.0, &calibrated)
        );
    }

    #[test]
    fn calibrated_sample_keeps_its_external_scale_in_anchor_free_mode() {
        let sample = CalibratedRangeSample::new(700.0, 12.0);
        let anchored = RangeModel::new(0.7);
        let anchor_free = RangeModel::for_mode(0.7, LocalizationMode::AnchorFree);
        let measurement = anchored.prepare_calibrated_measurement(sample);
        assert_eq!(
            anchored.loss_and_distance_gradient_prepared(650.0, &measurement),
            anchor_free.loss_and_distance_gradient_prepared(650.0, &measurement)
        );
    }
}
