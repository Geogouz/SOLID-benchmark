use thiserror::Error;

use crate::{LocalizationMode, Point2, PreparedProblem, SolveConfig, SolveResult};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("initial position count does not match agent count")]
    InvalidInitialPositions,
    #[error("range observation references unknown agent {0}")]
    UnknownAgent(u32),
    #[error("range observation references unknown anchor {0}")]
    UnknownAnchor(u32),
    #[error("duplicate agent identifier {0}")]
    DuplicateAgent(u32),
    #[error("duplicate anchor identifier {0}")]
    DuplicateAnchor(u32),
    #[error("anchor {0} has nonfinite geometry")]
    NonFiniteAnchor(u32),
    #[error("anchor {0} lies beyond the supported coordinate magnitude of 100 km")]
    AnchorOutOfRange(u32),
    #[error("invalid problem: {0}")]
    InvalidProblem(&'static str),
    #[error("ANC preparation failed: {0}")]
    AncPreparation(&'static str),
    #[error("invalid deployment profile: {0}")]
    InvalidProfile(&'static str),
    #[error("AF and AAC require an explicit cooperative profile")]
    CooperativeProfileRequired,
    #[error("invalid cooperative problem: {0}")]
    InvalidCooperativeProblem(&'static str),
    #[error("cooperative optimization returned a nonfinite result")]
    NonFiniteCooperativeResult,
    #[error("the anchor only posterior of agent {0} is not finite on any cell")]
    NonFinitePosterior(u32),
    #[error("a cell of the anchor only posterior of agent {0} lies above the certified upper bound of its block")]
    PosteriorAboveBound(u32),
}

/// The SOLID engine. One estimator per mode, every parameter comes from a fitted profile.
pub struct LocalizationEngine;

impl Default for LocalizationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalizationEngine {
    pub const fn new() -> Self {
        Self
    }

    /// Agent only and anchor aided positioning with the fitted cooperative profile of the sample count.
    pub fn solve_with_cooperative_profile(
        &self,
        problem: &PreparedProblem,
        profile: crate::CooperativeProfile,
    ) -> Result<SolveResult, EngineError> {
        crate::cooperative::solve(problem, profile)
    }

    /// Anchor only positioning with the fitted deployment profile.
    pub fn solve_with_profile(
        &self,
        problem: &PreparedProblem,
        profile: crate::DeploymentProfile,
    ) -> Result<SolveResult, EngineError> {
        if problem.mode != LocalizationMode::AnchorOnly {
            return Err(EngineError::InvalidProfile(
                "the ANC profile cannot be used for AF or AAC",
            ));
        }
        profile.validate()?;
        crate::anchor_only::solve(problem, &profile)
    }

    /// The geometry initializer's layout, the disclosed start that FNL and STRONG use in the benchmarks.
    pub fn initial_layout(
        &self,
        problem: &PreparedProblem,
        config: &SolveConfig,
    ) -> Result<Vec<Point2>, EngineError> {
        crate::initializer::initial_layout(problem, config)
    }
}
