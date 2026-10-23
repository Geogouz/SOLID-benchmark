mod ade_bp;
mod bayesian;
mod fnl;
mod geometry;
mod jin;
mod strong;

use std::time::Instant;

use serde::{Deserialize, Deserializer, Serialize};
use solid_core::{LocalizationEngine, Point2, PreparedProblem, SolveConfig, SolveResult};
use thiserror::Error;

pub use ade_bp::EmpiricalAdeBp;
pub use fnl::FastNetworkLocalization;
pub use jin::{JinRangeAggregation, JinSparseBias};
pub use strong::Strong;

/// The four published baselines. SOLID itself runs through the engine's fitted profiles.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MethodKind {
    Fnl,
    Strong,
    #[serde(rename = "jin-2021")]
    Jin2021,
    #[serde(rename = "ade-bp-2015")]
    AdeBp2015,
}

impl MethodKind {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Fnl => "fnl",
            Self::Strong => "strong",
            Self::Jin2021 => "jin-2021",
            Self::AdeBp2015 => "ade-bp-2015",
        }
    }

    pub const fn supports(self, mode: solid_core::LocalizationMode) -> bool {
        match self {
            Self::Strong | Self::AdeBp2015 => {
                !matches!(mode, solid_core::LocalizationMode::AnchorFree)
            }
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AreaBounds {
    pub minimum_x_cm: f64,
    pub maximum_x_cm: f64,
    pub minimum_y_cm: f64,
    pub maximum_y_cm: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdeRangingModel {
    pub positive_slope: f64,
    pub positive_intercept_cm: f64,
    pub negative_slope: f64,
    pub negative_intercept_cm: f64,
}

impl AdeRangingModel {
    pub(crate) fn scales(self, measured_range_cm: f64) -> (f64, f64) {
        (
            (self.positive_slope * measured_range_cm + self.positive_intercept_cm).max(1e-6),
            (self.negative_slope * measured_range_cm + self.negative_intercept_cm).max(1e-6),
        )
    }

    pub(crate) fn log_density(self, residual_cm: f64, measured_range_cm: f64) -> f64 {
        let (positive, negative) = self.scales(measured_range_cm);
        let tail = if residual_cm >= 0.0 {
            -residual_cm / positive
        } else {
            residual_cm / negative
        };
        tail - (positive + negative).ln()
    }

    fn validate(self) -> bool {
        [
            self.positive_slope,
            self.positive_intercept_cm,
            self.negative_slope,
            self.negative_intercept_cm,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.positive_slope >= 0.0
            && self.negative_slope >= 0.0
            && self.positive_intercept_cm > 0.0
            && self.negative_intercept_cm > 0.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BaselineInitialization {
    Released,
    SolidGeometry,
}

impl BaselineInitialization {
    fn solve(
        self,
        method: &impl LocalizationMethod,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError> {
        self.solve_from(method, problem, config)
            .map(|(result, _)| result)
    }

    /// A solve from this start and the positions it started from, the problem's own for the released start and the
    /// engine's geometric layout of the problem for the geometry start.
    fn solve_from(
        self,
        method: &impl LocalizationMethod,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<(SolveResult, Vec<Point2>), MethodError> {
        if self == Self::Released {
            return Ok((method.solve(problem, config)?, problem.initial_positions.clone()));
        }
        let started = Instant::now();
        let mut initialized = problem.clone();
        initialized.initial_positions = LocalizationEngine::new()
            .initial_layout(problem, config)
            .map_err(|error| MethodError::Initializer(error.to_string()))?;
        let mut result = method.solve(&initialized, config)?;
        result.metrics.elapsed_micros = started.elapsed().as_micros().min(u64::MAX as u128) as u64;
        Ok((result, initialized.initial_positions))
    }
}

/// Whether a solve's estimate equals the positions it started from, every agent exactly where it started, None for a
/// solve without a start.
pub fn stayed_at_start(result: &SolveResult, start: Option<&[Point2]>) -> Option<bool> {
    start.map(|start| result.positions == start)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "method", rename_all = "kebab-case", deny_unknown_fields)]
pub enum MethodConfig {
    Fnl {
        initialization: BaselineInitialization,
    },
    Strong {
        initialization: BaselineInitialization,
    },
    #[serde(rename = "jin-2021")]
    Jin2021 {
        residual_multiplier: f64,
        #[serde(skip_serializing_if = "JinRangeAggregation::is_median")]
        range_aggregation: JinRangeAggregation,
    },
    #[serde(rename = "ade-bp-2015")]
    AdeBp2015 {
        area: Option<AreaBounds>,
        particles: usize,
        iterations: usize,
        importance_iterations: usize,
        ranging_model: AdeRangingModel,
    },
}

impl<'de> Deserialize<'de> for MethodConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(tag = "method", rename_all = "kebab-case", deny_unknown_fields)]
        enum WireConfig {
            Fnl {
                initialization: BaselineInitialization,
            },
            Strong {
                initialization: BaselineInitialization,
            },
            #[serde(rename = "jin-2021")]
            Jin2021 {
                residual_multiplier: f64,
                #[serde(default)]
                range_aggregation: JinRangeAggregation,
            },
            #[serde(rename = "ade-bp-2015")]
            AdeBp2015 {
                area: Option<AreaBounds>,
                particles: usize,
                iterations: usize,
                importance_iterations: usize,
                ranging_model: AdeRangingModel,
            },
        }

        Ok(match WireConfig::deserialize(deserializer)? {
            WireConfig::Fnl { initialization } => Self::Fnl { initialization },
            WireConfig::Strong { initialization } => Self::Strong { initialization },
            WireConfig::Jin2021 {
                residual_multiplier,
                range_aggregation,
            } => Self::Jin2021 {
                residual_multiplier,
                range_aggregation,
            },
            WireConfig::AdeBp2015 {
                area,
                particles,
                iterations,
                importance_iterations,
                ranging_model,
            } => Self::AdeBp2015 {
                area,
                particles,
                iterations,
                importance_iterations,
                ranging_model,
            },
        })
    }
}

impl MethodConfig {
    pub fn default_for(kind: MethodKind) -> Option<Self> {
        match kind {
            MethodKind::Jin2021 => Some(Self::Jin2021 {
                residual_multiplier: 2.0,
                range_aggregation: JinRangeAggregation::Median,
            }),
            MethodKind::AdeBp2015 => Some(Self::AdeBp2015 {
                area: None,
                particles: 500,
                iterations: 4,
                importance_iterations: 64,
                ranging_model: AdeRangingModel {
                    positive_slope: 0.0,
                    positive_intercept_cm: 25.0,
                    negative_slope: 0.0,
                    negative_intercept_cm: 25.0,
                },
            }),
            _ => None,
        }
    }

    const fn kind(&self) -> MethodKind {
        match self {
            Self::Fnl { .. } => MethodKind::Fnl,
            Self::Strong { .. } => MethodKind::Strong,
            Self::Jin2021 { .. } => MethodKind::Jin2021,
            Self::AdeBp2015 { .. } => MethodKind::AdeBp2015,
        }
    }

    fn validate(&self) -> Result<(), MethodError> {
        let valid = match self {
            Self::Fnl { .. } | Self::Strong { .. } => true,
            Self::Jin2021 {
                residual_multiplier,
                ..
            } => residual_multiplier.is_finite() && *residual_multiplier >= 1.0,
            Self::AdeBp2015 {
                area,
                particles,
                iterations,
                importance_iterations,
                ranging_model,
            } => {
                area.is_none_or(|area| area.validate().is_ok())
                    && *particles > 0
                    && *iterations > 0
                    && *importance_iterations > 0
                    && ranging_model.validate()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(MethodError::InvalidInput(
                "invalid external method configuration".to_owned(),
            ))
        }
    }
}

#[derive(Debug, Error, PartialEq)]
pub enum MethodError {
    #[error("geometry initializer failed: {0}")]
    Initializer(String),
    #[error("invalid method input: {0}")]
    InvalidInput(String),
    #[error("method is not applicable: {0}")]
    UnsupportedMode(String),
    #[error("external solver failed: {0}")]
    Solver(String),
    #[error("telemetry overflow")]
    TelemetryOverflow,
}

pub trait LocalizationMethod: Send + Sync {
    fn kind(&self) -> MethodKind;

    fn solve(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError>;
}

#[derive(Clone)]
pub struct MethodRunner {
    kind: MethodKind,
    config: Option<MethodConfig>,
}

impl MethodRunner {
    pub fn new(kind: MethodKind) -> Self {
        Self {
            kind,
            config: MethodConfig::default_for(kind),
        }
    }

    pub fn configured(kind: MethodKind, config: MethodConfig) -> Result<Self, MethodError> {
        if config.kind() != kind {
            return Err(MethodError::InvalidInput(format!(
                "configuration for {} cannot be used with {}",
                config.kind().name(),
                kind.name()
            )));
        }
        config.validate()?;
        Ok(Self {
            kind,
            config: Some(config),
        })
    }

    pub fn config(&self) -> Option<&MethodConfig> {
        self.config.as_ref()
    }

    /// A solve and the positions it started from, None for Jin 2021, whose relaxation takes no start, and ADE-BP,
    /// whose particles do not start from the problem's positions. FNL and STRONG start from the problem's positions or
    /// from the engine's geometric layout, as their configuration asks (stayed_at_start).
    pub fn solve_with_start(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<(SolveResult, Option<Vec<Point2>>), MethodError> {
        let initialization = match (&self.config, self.kind) {
            (Some(MethodConfig::Fnl { initialization }), MethodKind::Fnl)
            | (Some(MethodConfig::Strong { initialization }), MethodKind::Strong) => *initialization,
            (_, MethodKind::Fnl | MethodKind::Strong) => BaselineInitialization::Released,
            _ => return Ok((self.solve(problem, config)?, None)),
        };
        let (result, start) = if self.kind == MethodKind::Fnl {
            initialization.solve_from(&FastNetworkLocalization, problem, config)?
        } else {
            initialization.solve_from(&Strong, problem, config)?
        };
        Ok((result, Some(start)))
    }

    pub const fn supports(&self, mode: solid_core::LocalizationMode) -> bool {
        self.kind.supports(mode)
    }
}

impl LocalizationMethod for MethodRunner {
    fn kind(&self) -> MethodKind {
        self.kind
    }

    fn solve(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<SolveResult, MethodError> {
        match self.kind {
            MethodKind::Fnl => match &self.config {
                Some(MethodConfig::Fnl { initialization }) => {
                    initialization.solve(&FastNetworkLocalization, problem, config)
                }
                _ => FastNetworkLocalization.solve(problem, config),
            },
            MethodKind::Strong => match &self.config {
                Some(MethodConfig::Strong { initialization }) => {
                    initialization.solve(&Strong, problem, config)
                }
                _ => Strong.solve(problem, config),
            },
            MethodKind::Jin2021 => JinSparseBias::from_config(
                self.config
                    .as_ref()
                    .expect("Jin has a default method configuration"),
            )
            .solve(problem, config),
            MethodKind::AdeBp2015 => EmpiricalAdeBp::from_config(
                self.config
                    .as_ref()
                    .expect("ADE-BP has a default method configuration"),
            )
            .solve(problem, config),
        }
    }
}

#[cfg(test)]
mod tests {
    use solid_core::LocalizationMode;

    use super::*;

    #[test]
    fn methods_report_their_audited_mode_applicability() {
        for method in [MethodKind::Strong, MethodKind::AdeBp2015] {
            assert!(!method.supports(LocalizationMode::AnchorFree));
            assert!(method.supports(LocalizationMode::AnchorAided));
            assert!(method.supports(LocalizationMode::AnchorOnly));
        }
        for method in [MethodKind::Jin2021, MethodKind::Fnl] {
            assert!(method.supports(LocalizationMode::AnchorFree));
            assert!(method.supports(LocalizationMode::AnchorAided));
            assert!(method.supports(LocalizationMode::AnchorOnly));
        }
    }

    #[test]
    fn method_configuration_cannot_be_assigned_to_another_method() {
        let config = MethodConfig::default_for(MethodKind::Jin2021).unwrap();

        assert!(MethodRunner::configured(MethodKind::Fnl, config).is_err());
    }

    #[test]
    fn baseline_initialization_is_explicit_and_roundtrips() {
        for method in [MethodKind::Fnl, MethodKind::Strong] {
            assert_eq!(MethodConfig::default_for(method), None);
            assert_eq!(MethodRunner::new(method).config(), None);
            for initialization in ["released", "solid-geometry"] {
                let json = format!(
                    r#"{{"method":"{}","initialization":"{initialization}"}}"#,
                    method.name()
                );
                let config: MethodConfig = serde_json::from_str(&json).unwrap();
                assert_eq!(serde_json::to_string(&config).unwrap(), json);
                assert!(MethodRunner::configured(method, config.clone()).is_ok());
                assert!(MethodRunner::configured(MethodKind::Jin2021, config).is_err());
            }
        }
        for json in [
            r#"{"method":"fnl"}"#,
            r#"{"method":"fnl","initialization":"geometry"}"#,
            r#"{"method":"fnl","initialization":"released","iterations":100}"#,
        ] {
            assert!(serde_json::from_str::<MethodConfig>(json).is_err());
        }
    }

    #[test]
    fn ade_bp_accepts_a_convergence_safety_cap() {
        let json = r#"{
            "method":"ade-bp-2015",
            "area":null,
            "particles":500,
            "iterations":4,
            "importance_iterations":64,
            "ranging_model":{
                "positive_slope":0.0,
                "positive_intercept_cm":25.0,
                "negative_slope":0.0,
                "negative_intercept_cm":25.0
            }
        }"#;

        assert!(serde_json::from_str::<MethodConfig>(json).is_ok());
    }

    #[test]
    fn jin_uses_its_published_name_in_json() {
        assert_eq!(
            serde_json::to_string(&MethodKind::Jin2021).unwrap(),
            "\"jin-2021\""
        );
        let config: MethodConfig =
            serde_json::from_str(r#"{"method":"jin-2021","residual_multiplier":2.0}"#).unwrap();
        assert_eq!(
            config,
            MethodConfig::default_for(MethodKind::Jin2021).unwrap()
        );
        assert_eq!(
            serde_json::to_string(&config).unwrap(),
            r#"{"method":"jin-2021","residual_multiplier":2.0}"#
        );
        let mean = r#"{"method":"jin-2021","residual_multiplier":2.0,"range_aggregation":"mean"}"#;
        let config: MethodConfig = serde_json::from_str(mean).unwrap();
        assert_eq!(serde_json::to_string(&config).unwrap(), mean);
        assert!(serde_json::from_str::<MethodConfig>(
            r#"{"method":"jin-2021","residual_multiplier":2.0,"range_aggregation":"all"}"#
        )
        .is_err());
    }
}
