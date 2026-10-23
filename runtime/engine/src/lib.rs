//! SOLID positioning engine: agent only, anchor aided and anchor only estimation from UWB ranges.
//!
//! Every estimator takes its parameters from a fitted profile. The cooperative solver serves the agent only
//! and anchor aided modes, the anchor only estimator serves the anchor only mode, and the geometry
//! initializer supplies the starting layouts of the cooperative solver and of two baselines.

mod anchor_only;
mod calibration;
mod cooperative;
mod initializer;
mod likelihood;
mod link;
mod model;
mod optimize;
mod posterior;
mod presence;
mod profile;
mod solver;

pub use calibration::CalibrationModel;
pub use model::{
    Anchor, CalibratedRangeSample, DimensionPolicy, LocalizationMode, ObservationAggregation,
    Point2, PreparedProblem, RangeObservation, SolveConfig, SolveMetrics, SolveResult,
    TerminationReason,
};
pub use profile::{
    AnchorOnlyProfile, BlockRangeProfile, CooperativeProfile, DecisionRule, DeploymentProfile, OffsetShape,
    PlanarArea, RangeErrorProfile,
};
pub use model::{LinkAttempt, MAX_COORDINATE_CM};
pub use presence::{PresenceProfile, PresenceShape, ReachRule};
pub use solver::{EngineError, LocalizationEngine};
