use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::{self, File};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use serde::Serialize;
use solid_core::{
    CooperativeProfile, DeploymentProfile, DimensionPolicy, LocalizationEngine, LocalizationMode,
    ObservationAggregation, Point2, SolveConfig, SolveMetrics, SolveResult, TerminationReason,
    MAX_COORDINATE_CM,
};
use solid_io::EvaluationScenario;
use solid_methods::{LocalizationMethod, MethodConfig, MethodKind, MethodRunner};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

const BENCHMARK_BATCHES_PER_WORKER: usize = 16;

#[derive(Parser)]
#[command(
    name = "solid",
    version,
    about = "SOLID positioning benchmark on TWR-CLOUD-Bern scenario archives"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Solves every applicable unit of a scenario archive with one method and scores it.
    Benchmark {
        #[arg(long)]
        scenarios: PathBuf,
        #[arg(long, value_enum, default_value_t = MethodArgument::Solid)]
        method: MethodArgument,
        /// Baseline configuration JSON, the published defaults apply without it.
        #[arg(long)]
        method_config: Option<PathBuf>,
        /// SOLID's fitted anchor only profile, required for the ANC mode.
        #[arg(long)]
        deployment_profile: Option<PathBuf>,
        /// SOLID's fitted cooperative profiles, one per sample count, required for the AF and AAC modes.
        #[arg(long)]
        cooperative_profiles: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        /// A file of scenario identifiers separated by white space. The run keeps these scenarios of the corpus
        /// and refuses an identifier the corpus lacks. The Bern runners split a corpus into such lists to solve
        /// it in single-threaded processes.
        #[arg(long, conflicts_with = "limit")]
        scenario_ids: Option<PathBuf>,
        #[arg(long, value_delimiter = ',', default_value = "1,10")]
        sample_counts: Vec<u32>,
        #[arg(long, value_delimiter = ',', default_value = "1,2")]
        iterations: Vec<u32>,
        #[arg(long)]
        include_cases: bool,
        #[arg(long)]
        scores_output: Option<PathBuf>,
        #[arg(long)]
        predictions_output: Option<PathBuf>,
        #[arg(long, value_enum)]
        mode: Option<ModeArgument>,
        /// Baseline range aggregation.
        #[arg(long, value_enum, default_value_t = AggregationArgument::All)]
        aggregation: AggregationArgument,
        /// Baseline internal dimension.
        #[arg(long, value_enum, default_value_t = DimensionArgument::Automatic)]
        dimensions: DimensionArgument,
        #[arg(long, default_value_t = 200.0)]
        max_tof: f64,
        #[arg(long, default_value_t = 10_000)]
        solver_iterations: usize,
        #[arg(long, default_value_t = 0.7)]
        robust_delta: f64,
        #[arg(long, default_value_t = 1_000)]
        time_budget_ms: u64,
        #[arg(long, default_value_t = 1)]
        selection_seed: u64,
        #[arg(long, default_value_t = 0)]
        workers: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum MethodArgument {
    #[default]
    Solid,
    Fnl,
    Strong,
    #[value(name = "jin-2021")]
    Jin2021,
    #[value(name = "ade-bp-2015")]
    AdeBp2015,
}

impl MethodArgument {
    /// The baseline kind, None for SOLID itself.
    fn baseline(self) -> Option<MethodKind> {
        match self {
            Self::Solid => None,
            Self::Fnl => Some(MethodKind::Fnl),
            Self::Strong => Some(MethodKind::Strong),
            Self::Jin2021 => Some(MethodKind::Jin2021),
            Self::AdeBp2015 => Some(MethodKind::AdeBp2015),
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ModeArgument {
    Af,
    Aac,
    Anc,
}

impl From<ModeArgument> for LocalizationMode {
    fn from(value: ModeArgument) -> Self {
        match value {
            ModeArgument::Af => Self::AnchorFree,
            ModeArgument::Aac => Self::AnchorAided,
            ModeArgument::Anc => Self::AnchorOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum AggregationArgument {
    Median,
    #[default]
    All,
}

impl From<AggregationArgument> for ObservationAggregation {
    fn from(value: AggregationArgument) -> Self {
        match value {
            AggregationArgument::Median => Self::Median,
            AggregationArgument::All => Self::AllSamples,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
enum DimensionArgument {
    #[default]
    Automatic,
    TwoD,
    ThreeD,
}

impl From<DimensionArgument> for DimensionPolicy {
    fn from(value: DimensionArgument) -> Self {
        match value {
            DimensionArgument::Automatic => Self::Automatic,
            DimensionArgument::TwoD => Self::Planar,
            DimensionArgument::ThreeD => Self::Spatial,
        }
    }
}

/// The method under evaluation, SOLID through its fitted profiles or one configured baseline.
enum Solver {
    Solid {
        deployment: Option<DeploymentProfile>,
        cooperative: Option<BTreeMap<u32, CooperativeProfile>>,
    },
    Baseline(MethodRunner),
}

impl Solver {
    fn name(&self) -> String {
        match self {
            Self::Solid { .. } => "solid".to_owned(),
            Self::Baseline(runner) => runner.kind().name().to_owned(),
        }
    }

    fn method_config(&self) -> Option<MethodConfig> {
        match self {
            Self::Solid { .. } => None,
            Self::Baseline(runner) => runner.config().cloned(),
        }
    }

    /// SOLID serves a mode when the matching fitted profile was supplied, a baseline when it defines the mode.
    fn supports(&self, mode: LocalizationMode) -> bool {
        match self {
            Self::Solid { deployment, cooperative } => match mode {
                LocalizationMode::AnchorOnly => deployment.is_some(),
                _ => cooperative.is_some(),
            },
            Self::Baseline(runner) => runner.supports(mode),
        }
    }

    /// A solve and whether its estimate stayed where it started, None for SOLID and for a baseline without a start
    /// (solid_methods::stayed_at_start).
    fn solve(
        &self,
        problem: &solid_core::PreparedProblem,
        samples: u32,
        config: &SolveConfig,
    ) -> Result<(SolveResult, Option<bool>)> {
        if let Self::Baseline(runner) = self {
            let (result, start) = runner.solve_with_start(problem, config)?;
            let stayed = solid_methods::stayed_at_start(&result, start.as_deref());
            return Ok((result, stayed));
        }
        Ok((self.solve_solid(problem, samples)?, None))
    }

    fn solve_solid(
        &self,
        problem: &solid_core::PreparedProblem,
        samples: u32,
    ) -> Result<SolveResult> {
        match self {
            Self::Solid {
                deployment,
                cooperative,
            } => {
                let engine = LocalizationEngine::new();
                if problem.mode == LocalizationMode::AnchorOnly {
                    let profile = deployment
                        .clone()
                        .context("SOLID ANC needs --deployment-profile")?;
                    Ok(engine.solve_with_profile(problem, profile)?)
                } else {
                    let profile = cooperative
                        .as_ref()
                        .and_then(|profiles| profiles.get(&samples))
                        .cloned()
                        .with_context(|| {
                            format!("SOLID has no cooperative profile for {samples} samples")
                        })?;
                    Ok(engine.solve_with_cooperative_profile(problem, profile)?)
                }
            }
            Self::Baseline(_) => unreachable!("a baseline solves through solve"),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Benchmark {
            scenarios,
            method,
            method_config,
            deployment_profile,
            cooperative_profiles,
            output,
            limit,
            scenario_ids,
            sample_counts,
            iterations,
            include_cases,
            scores_output,
            predictions_output,
            mode,
            aggregation,
            dimensions,
            max_tof,
            solver_iterations,
            robust_delta,
            time_budget_ms,
            selection_seed,
            workers,
        } => {
            let scenario_ids = scenario_ids.as_deref().map(read_scenario_ids).transpose()?;
            let solver = match method.baseline() {
                None => {
                    if method_config.is_some() {
                        bail!("--method-config configures a baseline, SOLID takes its fitted profiles");
                    }
                    Solver::Solid {
                        deployment: deployment_profile
                            .as_deref()
                            .map(load_deployment_profile)
                            .transpose()?,
                        cooperative: cooperative_profiles
                            .as_deref()
                            .map(load_cooperative_profiles)
                            .transpose()?,
                    }
                }
                Some(kind) => {
                    if deployment_profile.is_some() || cooperative_profiles.is_some() {
                        bail!("fitted SOLID profiles cannot be given to a baseline");
                    }
                    Solver::Baseline(method_runner(kind, method_config.as_deref())?)
                }
            };
            benchmark(
                &scenarios,
                &output,
                solver,
                BenchmarkOptions {
                    limit,
                    scenario_ids: scenario_ids.as_ref(),
                    selection_seed,
                    workers,
                    sample_counts: &sample_counts,
                    iterations: &iterations,
                    include_cases,
                    scores_output: scores_output.as_deref(),
                    predictions_output: predictions_output.as_deref(),
                    mode: mode.map(Into::into),
                },
                solve_config(
                    aggregation,
                    dimensions,
                    max_tof,
                    solver_iterations,
                    robust_delta,
                    time_budget_ms,
                ),
            )
        }
    }
}

/// The scenario identifiers of a list file, separated by white space, each listed once.
fn read_scenario_ids(path: &Path) -> Result<BTreeSet<u32>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read scenario identifiers {}", path.display()))?;
    let mut ids = BTreeSet::new();
    for token in text.split_whitespace() {
        let id: u32 = token
            .parse()
            .with_context(|| format!("invalid scenario identifier {token}"))?;
        if !ids.insert(id) {
            bail!("scenario identifier {id} is listed twice");
        }
    }
    if ids.is_empty() {
        bail!("the scenario identifier list {} is empty", path.display());
    }
    Ok(ids)
}

fn load_deployment_profile(path: &Path) -> Result<DeploymentProfile> {
    let file = File::open(path)
        .with_context(|| format!("failed to open deployment profile {}", path.display()))?;
    serde_json::from_reader(file)
        .with_context(|| format!("invalid deployment profile {}", path.display()))
}

fn load_cooperative_profiles(path: &Path) -> Result<BTreeMap<u32, CooperativeProfile>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open cooperative profiles {}", path.display()))?;
    let profiles: BTreeMap<String, CooperativeProfile> = serde_json::from_reader(file)
        .with_context(|| format!("invalid cooperative profiles {}", path.display()))?;
    profiles
        .into_iter()
        .map(|(samples, profile)| {
            Ok((
                samples
                    .parse()
                    .with_context(|| format!("sample count key {samples} is not a number"))?,
                profile,
            ))
        })
        .collect()
}

fn method_runner(kind: MethodKind, config: Option<&Path>) -> Result<MethodRunner> {
    let Some(path) = config else {
        return Ok(MethodRunner::new(kind));
    };
    let file = File::open(path)
        .with_context(|| format!("failed to open method configuration {}", path.display()))?;
    let config: MethodConfig = serde_json::from_reader(file)
        .with_context(|| format!("invalid method configuration {}", path.display()))?;
    MethodRunner::configured(kind, config).map_err(Into::into)
}

fn solve_config(
    aggregation: AggregationArgument,
    dimensions: DimensionArgument,
    max_tof: f64,
    solver_iterations: usize,
    robust_delta: f64,
    time_budget_ms: u64,
) -> SolveConfig {
    SolveConfig {
        aggregation: aggregation.into(),
        dimension_policy: dimensions.into(),
        max_tof_ns: max_tof,
        max_iterations: solver_iterations,
        robust_delta,
        max_time_micros: time_budget_ms.saturating_mul(1000),
        ..SolveConfig::default()
    }
}

enum ScenarioSource {
    Directory(Vec<(u32, PathBuf)>),
    Zip {
        archive: ZipArchive<File>,
        entries: Vec<(u32, usize)>,
    },
}

impl ScenarioSource {
    fn open(path: &Path, limit: usize) -> Result<Self> {
        let mut source = if path.is_dir() {
            let mut entries: Vec<(u32, PathBuf)> = fs::read_dir(path)?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| {
                    path.extension()
                        .is_some_and(|extension| extension == "json")
                })
                .map(|path| Ok((scenario_id(&path)?, path)))
                .collect::<Result<_>>()?;
            entries.sort_by_key(|entry| entry.0);
            Self::Directory(entries)
        } else if path
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("zip"))
        {
            let file = File::open(path)
                .with_context(|| format!("failed to open scenario archive {}", path.display()))?;
            let mut archive = ZipArchive::new(file)
                .with_context(|| format!("invalid scenario archive {}", path.display()))?;
            let mut entries = Vec::new();
            for index in 0..archive.len() {
                let file = archive.by_index(index)?;
                if !file.is_dir() {
                    if let Some(id) = archive_scenario_id(file.name()) {
                        entries.push((id, index));
                    }
                }
            }
            entries.sort_by_key(|entry| entry.0);
            Self::Zip { archive, entries }
        } else {
            bail!("scenario input must be a directory or zip archive");
        };
        source.truncate(limit);
        Ok(source)
    }

    fn len(&self) -> usize {
        match self {
            Self::Directory(entries) => entries.len(),
            Self::Zip { entries, .. } => entries.len(),
        }
    }

    fn open_sampled(path: &Path, limit: usize, seed: u64) -> Result<Self> {
        let mut source = Self::open(path, 0)?;
        if limit > 0 && limit < source.len() {
            match &mut source {
                Self::Directory(entries) => select_entries(entries, limit, seed),
                Self::Zip { entries, .. } => select_entries(entries, limit, seed),
            }
        }
        Ok(source)
    }

    fn truncate(&mut self, limit: usize) {
        if limit == 0 {
            return;
        }
        match self {
            Self::Directory(entries) => entries.truncate(limit),
            Self::Zip { entries, .. } => entries.truncate(limit),
        }
    }

    /// Keeps the listed scenarios and refuses an identifier the corpus lacks.
    fn retain_ids(&mut self, ids: &BTreeSet<u32>) -> Result<()> {
        match self {
            Self::Directory(entries) => retain_entries(entries, ids),
            Self::Zip { entries, .. } => retain_entries(entries, ids),
        }
    }

    fn for_each_batch(
        mut self,
        batch_size: usize,
        mut operation: impl FnMut(Vec<(u32, EvaluationScenario)>) -> Result<()>,
    ) -> Result<()> {
        let batch_size = batch_size.max(1);
        match &mut self {
            Self::Directory(entries) => {
                for entries in entries.chunks(batch_size) {
                    let batch = entries
                        .iter()
                        .map(|(id, path)| Ok((*id, EvaluationScenario::from_path(path)?)))
                        .collect::<Result<_>>()?;
                    operation(batch)?;
                }
            }
            Self::Zip { archive, entries } => {
                let mut contents = Vec::new();
                for entries in entries.chunks(batch_size) {
                    let mut batch = Vec::with_capacity(entries.len());
                    for (id, index) in entries {
                        contents.clear();
                        archive.by_index(*index)?.read_to_end(&mut contents)?;
                        batch.push((*id, EvaluationScenario::from_slice(&contents)?));
                    }
                    operation(batch)?;
                }
            }
        }
        Ok(())
    }
}

fn retain_entries<T>(entries: &mut Vec<(u32, T)>, ids: &BTreeSet<u32>) -> Result<()> {
    let present = entries.iter().map(|entry| entry.0).collect::<BTreeSet<_>>();
    if let Some(missing) = ids.difference(&present).next() {
        bail!("scenario {missing} of the identifier list is not in the corpus");
    }
    entries.retain(|entry| ids.contains(&entry.0));
    Ok(())
}

fn select_entries<T>(entries: &mut Vec<(u32, T)>, limit: usize, seed: u64) {
    entries.sort_by_key(|entry| splitmix64(seed ^ entry.0 as u64));
    entries.truncate(limit);
    entries.sort_by_key(|entry| entry.0);
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn archive_scenario_id(name: &str) -> Option<u32> {
    Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.parse().ok())
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CellKey {
    mode: String,
    iteration: u32,
    q: usize,
    samples: u32,
    anchors: usize,
    agents: usize,
}

#[derive(Default)]
struct CellAccumulator {
    scenario_units: usize,
    possible_pairs: usize,
    eligible_pairs: usize,
    penalized_pairs: usize,
    errors: Vec<f64>,
}

#[derive(Serialize)]
struct CellSummary {
    mode: String,
    q: usize,
    samples: u32,
    anchors: usize,
    agents: usize,
    scenario_units: usize,
    scored_scenario_units: usize,
    non_scorable_units: usize,
    possible_pairs: usize,
    eligible_pairs: usize,
    scored_pair_coverage: f64,
    penalized_pairs: usize,
    penalized_pair_rate: f64,
    mean: Option<f64>,
    q50: Option<f64>,
    q75: Option<f64>,
    q90: Option<f64>,
}

#[derive(Serialize)]
struct TrackSummary {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    iteration: Option<u32>,
    q: usize,
    samples: u32,
    cells: usize,
    scored_cells: usize,
    scenario_units: usize,
    scored_scenario_units: usize,
    non_scorable_units: usize,
    possible_pairs: usize,
    eligible_pairs: usize,
    scored_pair_coverage: f64,
    penalized_pairs: usize,
    penalized_pair_rate: f64,
    balanced_s50: Option<f64>,
    balanced_s75: Option<f64>,
    balanced_s90: Option<f64>,
}

#[derive(Serialize)]
struct BenchmarkReport {
    schema_version: &'static str,
    /// The identity of the sources the CLI is built from, the digest research/cli/build.rs embeds.
    implementation_revision: Option<String>,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    method_config: Option<MethodConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    deployment_profile: Option<DeploymentProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cooperative_profiles: Option<BTreeMap<u32, CooperativeProfile>>,
    applicable_modes: Vec<String>,
    not_applicable_modes: Vec<String>,
    scenarios: String,
    scenario_count: usize,
    applicable_scenario_count: usize,
    not_applicable_scenario_count: usize,
    selection_seed: u64,
    /// The number of listed scenarios when an identifier list restricted the run.
    #[serde(skip_serializing_if = "Option::is_none")]
    listed_scenarios: Option<usize>,
    sample_counts: Vec<u32>,
    iterations: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mode_filter: Option<String>,
    cohort_fingerprint: String,
    workers: usize,
    solves: usize,
    wall_seconds: f64,
    solver_mean_micros: f64,
    solver_p50_micros: u64,
    solver_p95_micros: u64,
    solver_p99_micros: u64,
    solver_max_micros: u64,
    optimizer_mean_iterations: f64,
    optimizer_max_iterations: usize,
    optimizer_mean_starts: f64,
    optimizer_max_starts: usize,
    objective_evaluations: u128,
    gradient_evaluations: u128,
    range_factor_evaluations: u128,
    coordinate_factor_evaluations: u128,
    mean_objective_evaluations: f64,
    mean_gradient_evaluations: f64,
    mean_range_factor_evaluations: f64,
    mean_coordinate_factor_evaluations: f64,
    track_performance: Vec<PerformanceSummary>,
    iteration_performance: Vec<PerformanceSummary>,
    termination_counts: BTreeMap<String, usize>,
    failure_counts: BTreeMap<String, usize>,
    baseline_config: SolveConfig,
    tracks: Vec<TrackSummary>,
    iteration_tracks: Vec<TrackSummary>,
    cells: Vec<CellSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score_archive: Option<ScoreArchiveSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prediction_archive: Option<PredictionArchiveSummary>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    cases: Vec<BenchmarkCase>,
}

#[derive(Default)]
struct PerformanceAccumulator {
    dimensions: BTreeSet<usize>,
    elapsed_micros: Vec<u64>,
    iterations: usize,
    max_iterations: usize,
    starts: usize,
    max_starts: usize,
    objective_evaluations: u128,
    gradient_evaluations: u128,
    range_factor_evaluations: u128,
    coordinate_factor_evaluations: u128,
}

impl PerformanceAccumulator {
    fn record(&mut self, result: &SolveResult) {
        self.dimensions.insert(result.metrics.dimensions);
        self.elapsed_micros.push(result.metrics.elapsed_micros);
        self.iterations += result.metrics.iterations;
        self.max_iterations = self.max_iterations.max(result.metrics.iterations);
        self.starts += result.metrics.starts;
        self.max_starts = self.max_starts.max(result.metrics.starts);
        self.objective_evaluations += result.metrics.objective_evaluations as u128;
        self.gradient_evaluations += result.metrics.gradient_evaluations as u128;
        self.range_factor_evaluations += result.metrics.range_factor_evaluations as u128;
        self.coordinate_factor_evaluations += result.metrics.coordinate_factor_evaluations as u128;
    }
}

#[derive(Serialize)]
struct PerformanceSummary {
    mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    iteration: Option<u32>,
    samples: u32,
    dimensions: Vec<usize>,
    solves: usize,
    solver_mean_micros: f64,
    solver_p50_micros: u64,
    solver_p95_micros: u64,
    solver_p99_micros: u64,
    solver_max_micros: u64,
    optimizer_mean_iterations: f64,
    optimizer_max_iterations: usize,
    optimizer_mean_starts: f64,
    optimizer_max_starts: usize,
    mean_objective_evaluations: f64,
    mean_gradient_evaluations: f64,
    mean_range_factor_evaluations: f64,
    mean_coordinate_factor_evaluations: f64,
}

struct ScenarioScore {
    mean_error: Option<f64>,
    possible_pairs: usize,
    eligible_pairs: usize,
    penalized_pairs: usize,
}

struct EvaluatedSolve {
    key: CellKey,
    score: ScenarioScore,
    result: SolveResult,
    /// Whether the estimate equals the positions the solve started from, None for SOLID, for a baseline without a start
    /// and for a failed solve.
    stayed_at_start: Option<bool>,
    expected_agent_ids: Vec<u32>,
    scan_id: Option<u32>,
    generated_scenario_id: Option<u32>,
    group_index: Option<usize>,
    data_role: Option<String>,
    method_error: Option<String>,
}

#[derive(Serialize)]
struct BenchmarkCase {
    archive_scenario_id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    generated_scenario_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    scan_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    group_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    data_role: Option<String>,
    mode: String,
    iteration: u32,
    q: usize,
    samples: u32,
    anchors: usize,
    agents: usize,
    mean_pair_error_cm: Option<f64>,
    possible_pairs: usize,
    eligible_pairs: usize,
    penalized_pairs: usize,
    elapsed_micros: u64,
    dimensions: usize,
    optimizer_iterations: usize,
    final_objective: Option<f64>,
    starts: usize,
    factors: usize,
    range_factor_evaluations: usize,
    coordinate_factor_evaluations: usize,
    termination: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    method_error: Option<String>,
    /// Whether a baseline's estimate equals the positions its solve started from, absent without a start.
    #[serde(skip_serializing_if = "Option::is_none")]
    stayed_at_start: Option<bool>,
    /// The engine's metrics of the solve whole, so that a case carries every metric the engine reports, the anchor
    /// only supports its node guard bound and the largest share of posterior mass it dropped among them.
    metrics: SolveMetrics,
}

struct ScoreRecord {
    archive_scenario_id: u32,
    mode: String,
    iteration: u32,
    q: usize,
    samples: u32,
    anchors: usize,
    agents: usize,
    mean_pair_error_cm: Option<f64>,
    possible_pairs: usize,
    eligible_pairs: usize,
    penalized_pairs: usize,
}

#[derive(Serialize)]
struct ScoreArchiveSummary {
    path: String,
    member: &'static str,
    format: &'static str,
    records: usize,
}

#[derive(Serialize)]
struct PredictionArchiveSummary {
    path: String,
    member: &'static str,
    format: &'static str,
    records: usize,
    missing_coordinates: usize,
    nonfinite_coordinates: usize,
}

struct PredictionArchiveWriter {
    path: PathBuf,
    temporary_path: PathBuf,
    archive: ZipWriter<BufWriter<File>>,
    records: usize,
    missing_coordinates: usize,
    nonfinite_coordinates: usize,
}

struct BenchmarkOptions<'a> {
    limit: usize,
    scenario_ids: Option<&'a BTreeSet<u32>>,
    selection_seed: u64,
    workers: usize,
    sample_counts: &'a [u32],
    iterations: &'a [u32],
    include_cases: bool,
    scores_output: Option<&'a Path>,
    predictions_output: Option<&'a Path>,
    mode: Option<LocalizationMode>,
}

fn benchmark(
    scenarios: &Path,
    output: &Path,
    solver: Solver,
    options: BenchmarkOptions<'_>,
    config: SolveConfig,
) -> Result<()> {
    let BenchmarkOptions {
        limit,
        scenario_ids,
        selection_seed,
        workers,
        sample_counts,
        iterations,
        include_cases,
        scores_output,
        predictions_output,
        mode,
    } = options;
    validate_requested_cases(sample_counts, iterations)?;
    let started = Instant::now();
    let mut source = ScenarioSource::open_sampled(scenarios, limit, selection_seed)?;
    if let Some(ids) = scenario_ids {
        source.retain_ids(ids)?;
    }
    let scenario_count = source.len();
    let workers = if workers == 0 {
        std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1)
    } else {
        workers
    };
    let worker_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(workers)
        .thread_name(|index| format!("solid-dimensions-{index}"))
        .build()?;
    let mut cells: BTreeMap<CellKey, CellAccumulator> = BTreeMap::new();
    let mut performance = PerformanceAccumulator::default();
    let mut track_performance: BTreeMap<(String, u32), PerformanceAccumulator> = BTreeMap::new();
    let mut iteration_performance: BTreeMap<(String, u32, u32), PerformanceAccumulator> =
        BTreeMap::new();
    let mut termination_counts = BTreeMap::new();
    let mut failure_counts = BTreeMap::new();
    let mut cases = Vec::new();
    let mut score_records = Vec::new();
    let mut prediction_archive = predictions_output
        .map(PredictionArchiveWriter::create)
        .transpose()?;
    let mut processed_scenarios = 0usize;
    let mut applicable_scenarios = 0usize;
    let mut next_progress = 1_000usize;
    let mut cohort_fingerprint = Fnv1a64::new();

    source.for_each_batch(
        workers.saturating_mul(BENCHMARK_BATCHES_PER_WORKER),
        |batch| {
            let evaluated = worker_pool.install(|| {
                batch
                    .into_par_iter()
                    .map(|(id, scenario)| {
                        (
                            id,
                            evaluate_benchmark_scenario(
                                scenario,
                                &solver,
                                config,
                                mode,
                                sample_counts,
                                iterations,
                            ),
                        )
                    })
                    .collect::<Vec<_>>()
            });
            for (id, scenario) in evaluated {
                cohort_fingerprint.write_u64(id as u64);
                let scenario = scenario?;
                applicable_scenarios += usize::from(!scenario.is_empty());
                for evaluated in scenario {
                    if let Some(archive) = prediction_archive.as_mut() {
                        archive.write_solve(id, &evaluated)?;
                    }
                    if include_cases {
                        cases.push(benchmark_case(id, &evaluated));
                    }
                    if scores_output.is_some() {
                        score_records.push(score_record(id, &evaluated));
                    }
                    let cell = cells.entry(evaluated.key.clone()).or_default();
                    cell.scenario_units += 1;
                    cell.possible_pairs += evaluated.score.possible_pairs;
                    cell.eligible_pairs += evaluated.score.eligible_pairs;
                    cell.penalized_pairs += evaluated.score.penalized_pairs;
                    if let Some(error) = evaluated.score.mean_error {
                        cell.errors.push(error);
                    }
                    performance.record(&evaluated.result);
                    track_performance
                        .entry((evaluated.key.mode.clone(), evaluated.key.samples))
                        .or_default()
                        .record(&evaluated.result);
                    iteration_performance
                        .entry((
                            evaluated.key.mode.clone(),
                            evaluated.key.iteration,
                            evaluated.key.samples,
                        ))
                        .or_default()
                        .record(&evaluated.result);
                    *termination_counts
                        .entry(evaluated.result.metrics.termination.name().to_owned())
                        .or_insert(0) += 1;
                    if let Some(error) = &evaluated.method_error {
                        *failure_counts.entry(error.clone()).or_insert(0) += 1;
                    }
                }
                processed_scenarios += 1;
            }
            if processed_scenarios >= next_progress || processed_scenarios == scenario_count {
                eprintln!(
                    "processed {processed_scenarios}/{scenario_count} scenarios in {:.1}s",
                    started.elapsed().as_secs_f64()
                );
                next_progress = processed_scenarios / 1_000 * 1_000 + 1_000;
            }
            Ok(())
        },
    )?;

    for cell in cells.values_mut() {
        cell.errors.sort_by(f64::total_cmp);
    }
    let official_cells = collapse_initializations(&cells);
    let cell_summaries = official_cells
        .iter()
        .map(|(key, cell)| summarize_cell(key, cell))
        .collect();
    let tracks = summarize_tracks(&official_cells, false);
    let iteration_tracks = summarize_tracks(&cells, true);
    performance.elapsed_micros.sort_unstable();
    let track_performance = track_performance
        .into_iter()
        .map(|((mode, samples), mut performance)| {
            performance.elapsed_micros.sort_unstable();
            summarize_performance(mode, None, samples, &performance)
        })
        .collect();
    let iteration_performance = iteration_performance
        .into_iter()
        .map(|((mode, iteration, samples), mut performance)| {
            performance.elapsed_micros.sort_unstable();
            summarize_performance(mode, Some(iteration), samples, &performance)
        })
        .collect();
    let solves = performance.elapsed_micros.len();
    let solver_micros = performance
        .elapsed_micros
        .iter()
        .map(|elapsed| *elapsed as u128)
        .sum::<u128>();
    let modes = [
        LocalizationMode::AnchorFree,
        LocalizationMode::AnchorAided,
        LocalizationMode::AnchorOnly,
    ];
    let applicable_modes = modes
        .iter()
        .filter(|mode| solver.supports(**mode))
        .map(|mode| mode.scorer_name().to_owned())
        .collect();
    let not_applicable_modes = modes
        .iter()
        .filter(|mode| !solver.supports(**mode))
        .map(|mode| mode.scorer_name().to_owned())
        .collect();
    let (deployment_profile, cooperative_profiles) = match &solver {
        Solver::Solid { deployment, cooperative } => (deployment.clone(), cooperative.clone()),
        Solver::Baseline(_) => (None, None),
    };
    let score_archive = scores_output
        .map(|path| write_score_archive(path, &score_records))
        .transpose()?;
    let prediction_archive = prediction_archive
        .map(PredictionArchiveWriter::finish)
        .transpose()?;
    let report = BenchmarkReport {
        schema_version: "solid.benchmark.v6",
        implementation_revision: source_identity(),
        method: solver.name(),
        method_config: solver.method_config(),
        deployment_profile,
        cooperative_profiles,
        applicable_modes,
        not_applicable_modes,
        scenarios: scenarios.display().to_string(),
        scenario_count,
        applicable_scenario_count: applicable_scenarios,
        not_applicable_scenario_count: scenario_count - applicable_scenarios,
        selection_seed,
        listed_scenarios: scenario_ids.map(BTreeSet::len),
        sample_counts: sample_counts.to_vec(),
        iterations: iterations.to_vec(),
        mode_filter: mode.map(|mode| mode.scorer_name().to_owned()),
        cohort_fingerprint: cohort_fingerprint.hex(),
        workers,
        solves,
        wall_seconds: started.elapsed().as_secs_f64(),
        solver_mean_micros: solver_micros as f64 / solves.max(1) as f64,
        solver_p50_micros: percentile_u64(&performance.elapsed_micros, 0.50),
        solver_p95_micros: percentile_u64(&performance.elapsed_micros, 0.95),
        solver_p99_micros: percentile_u64(&performance.elapsed_micros, 0.99),
        solver_max_micros: performance
            .elapsed_micros
            .last()
            .copied()
            .unwrap_or_default(),
        optimizer_mean_iterations: performance.iterations as f64 / solves.max(1) as f64,
        optimizer_max_iterations: performance.max_iterations,
        optimizer_mean_starts: performance.starts as f64 / solves.max(1) as f64,
        optimizer_max_starts: performance.max_starts,
        objective_evaluations: performance.objective_evaluations,
        gradient_evaluations: performance.gradient_evaluations,
        range_factor_evaluations: performance.range_factor_evaluations,
        coordinate_factor_evaluations: performance.coordinate_factor_evaluations,
        mean_objective_evaluations: ratio_u128(performance.objective_evaluations, solves),
        mean_gradient_evaluations: ratio_u128(performance.gradient_evaluations, solves),
        mean_range_factor_evaluations: ratio_u128(performance.range_factor_evaluations, solves),
        mean_coordinate_factor_evaluations: ratio_u128(
            performance.coordinate_factor_evaluations,
            solves,
        ),
        track_performance,
        iteration_performance,
        termination_counts,
        failure_counts,
        baseline_config: config,
        tracks,
        iteration_tracks,
        cells: cell_summaries,
        score_archive,
        prediction_archive,
        cases,
    };
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(
        File::create(output).with_context(|| format!("failed to create {}", output.display()))?,
    );
    serde_json::to_writer_pretty(&mut writer, &report)?;
    writeln!(writer)?;
    eprintln!(
        "benchmarked {} scenarios and {} solves in {:.3}s, solver mean {:.3}ms",
        scenario_count,
        solves,
        report.wall_seconds,
        report.solver_mean_micros / 1000.0
    );
    Ok(())
}

fn source_identity() -> Option<String> {
    option_env!("SOLID_SOURCE_IDENTITY").map(str::to_owned)
}

struct Fnv1a64(u64);

impl Fnv1a64 {
    const fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write_u64(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn hex(&self) -> String {
        format!("{:016x}", self.0)
    }
}

fn evaluate_benchmark_scenario(
    scenario: EvaluationScenario,
    solver: &Solver,
    config: SolveConfig,
    mode_filter: Option<LocalizationMode>,
    sample_counts: &[u32],
    iterations: &[u32],
) -> Result<Vec<EvaluatedSolve>> {
    let scan_id = scenario
        .research_metadata
        .as_ref()
        .map(|metadata| metadata.scan_id);
    let generated_scenario_id = scenario
        .research_metadata
        .as_ref()
        .map(|metadata| metadata.scenario_id);
    let group_index = scenario
        .research_metadata
        .as_ref()
        .map(|metadata| metadata.group_index);
    let data_role = scenario
        .research_metadata
        .as_ref()
        .map(|metadata| metadata.data_role.clone());
    let declared_anchor_count = scenario.anchors.len();
    let modes: &[LocalizationMode] = if scenario.anchors.is_empty() {
        &[LocalizationMode::AnchorFree]
    } else {
        &[LocalizationMode::AnchorAided, LocalizationMode::AnchorOnly]
    };
    let mut evaluated = Vec::new();
    for &iteration in iterations {
        for &samples in sample_counts {
            for mode in modes {
                if !solver.supports(*mode) || mode_filter.is_some_and(|filter| filter != *mode) {
                    continue;
                }
                let problem = scenario.prepare_problem(iteration, samples, *mode)?;
                let solve_started = Instant::now();
                let (result, stayed_at_start, method_error) =
                    match solver.solve(&problem, samples, &config) {
                        Ok((result, stayed)) => (result, stayed, None),
                        Err(error) => {
                            let mut result = failed_solve(&problem, config.dimension_policy);
                            result.metrics.elapsed_micros =
                                solve_started.elapsed().as_micros().min(u64::MAX as u128) as u64;
                            (result, None, Some(error.to_string()))
                        }
                    };
                evaluated.push(EvaluatedSolve {
                    key: CellKey {
                        mode: mode.scorer_name().to_owned(),
                        iteration,
                        // The public result contains x and y coordinates only. Internal
                        // spatial inference does not turn it into a 3D submission.
                        q: 2,
                        samples,
                        anchors: declared_anchor_count,
                        agents: problem.agent_ids.len(),
                    },
                    score: score_scenario(&scenario, &problem, &result)?,
                    result,
                    stayed_at_start,
                    expected_agent_ids: problem.agent_ids.clone(),
                    scan_id,
                    generated_scenario_id,
                    group_index,
                    data_role: data_role.clone(),
                    method_error,
                });
            }
        }
    }
    Ok(evaluated)
}

fn validate_requested_cases(sample_counts: &[u32], iterations: &[u32]) -> Result<()> {
    if sample_counts.is_empty() || iterations.is_empty() {
        bail!("sample counts and iterations must not be empty");
    }
    if sample_counts.contains(&0) || iterations.contains(&0) {
        bail!("sample counts and iterations must be positive");
    }
    if sample_counts.iter().copied().collect::<BTreeSet<_>>().len() != sample_counts.len()
        || iterations.iter().copied().collect::<BTreeSet<_>>().len() != iterations.len()
    {
        bail!("sample counts and iterations must not contain duplicates");
    }
    Ok(())
}

fn benchmark_case(archive_scenario_id: u32, evaluated: &EvaluatedSolve) -> BenchmarkCase {
    BenchmarkCase {
        archive_scenario_id,
        generated_scenario_id: evaluated.generated_scenario_id,
        scan_id: evaluated.scan_id,
        group_index: evaluated.group_index,
        data_role: evaluated.data_role.clone(),
        mode: evaluated.key.mode.clone(),
        iteration: evaluated.key.iteration,
        q: evaluated.key.q,
        samples: evaluated.key.samples,
        anchors: evaluated.key.anchors,
        agents: evaluated.key.agents,
        mean_pair_error_cm: evaluated.score.mean_error,
        possible_pairs: evaluated.score.possible_pairs,
        eligible_pairs: evaluated.score.eligible_pairs,
        penalized_pairs: evaluated.score.penalized_pairs,
        elapsed_micros: evaluated.result.metrics.elapsed_micros,
        dimensions: evaluated.result.metrics.dimensions,
        optimizer_iterations: evaluated.result.metrics.iterations,
        final_objective: evaluated
            .result
            .metrics
            .final_objective
            .is_finite()
            .then_some(evaluated.result.metrics.final_objective),
        starts: evaluated.result.metrics.starts,
        factors: evaluated.result.metrics.factors,
        range_factor_evaluations: evaluated.result.metrics.range_factor_evaluations,
        coordinate_factor_evaluations: evaluated.result.metrics.coordinate_factor_evaluations,
        termination: evaluated.result.metrics.termination.name(),
        method_error: evaluated.method_error.clone(),
        stayed_at_start: evaluated.stayed_at_start,
        metrics: evaluated.result.metrics.clone(),
    }
}

fn score_record(archive_scenario_id: u32, evaluated: &EvaluatedSolve) -> ScoreRecord {
    ScoreRecord {
        archive_scenario_id,
        mode: evaluated.key.mode.clone(),
        iteration: evaluated.key.iteration,
        q: evaluated.key.q,
        samples: evaluated.key.samples,
        anchors: evaluated.key.anchors,
        agents: evaluated.key.agents,
        mean_pair_error_cm: evaluated.score.mean_error,
        possible_pairs: evaluated.score.possible_pairs,
        eligible_pairs: evaluated.score.eligible_pairs,
        penalized_pairs: evaluated.score.penalized_pairs,
    }
}

fn write_score_csv(mut output: impl Write, records: &[ScoreRecord]) -> Result<()> {
    writeln!(
        output,
        "archive_scenario_id,mode,iteration,q,samples,anchors,agents,mean_pair_error_cm,possible_pairs,eligible_pairs,penalized_pairs"
    )?;
    for record in records {
        write!(
            output,
            "{},{},{},{},{},{},{},",
            record.archive_scenario_id,
            record.mode,
            record.iteration,
            record.q,
            record.samples,
            record.anchors,
            record.agents
        )?;
        if let Some(error) = record.mean_pair_error_cm {
            write!(output, "{error}")?;
        }
        writeln!(
            output,
            ",{},{},{}",
            record.possible_pairs, record.eligible_pairs, record.penalized_pairs
        )?;
    }
    Ok(())
}

fn write_score_archive(path: &Path, records: &[ScoreRecord]) -> Result<ScoreArchiveSummary> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let file = BufWriter::new(
        File::create(path)
            .with_context(|| format!("failed to create score archive {}", path.display()))?,
    );
    let mut archive = ZipWriter::new(file);
    archive.start_file(
        "scores.csv",
        SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
    )?;
    write_score_csv(&mut archive, records)?;
    archive.finish()?.flush()?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("scores.zip")
        .to_owned();
    Ok(ScoreArchiveSummary {
        path: file_name,
        member: "scores.csv",
        format: "zip-compressed CSV",
        records: records.len(),
    })
}

impl PredictionArchiveWriter {
    fn create(path: &Path) -> Result<Self> {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("predictions.zip");
        let temporary_path = path.with_file_name(format!(".{file_name}.partial"));
        let file = BufWriter::new(File::create(&temporary_path).with_context(|| {
            format!(
                "failed to create prediction archive {}",
                temporary_path.display()
            )
        })?);
        let mut archive = ZipWriter::new(file);
        archive.start_file(
            "predictions.csv",
            SimpleFileOptions::default().compression_method(CompressionMethod::Deflated),
        )?;
        writeln!(
            archive,
            "scenario_id,iteration,samples,mode,node_id,q,x,y,z"
        )?;
        Ok(Self {
            path: path.to_owned(),
            temporary_path,
            archive,
            records: 0,
            missing_coordinates: 0,
            nonfinite_coordinates: 0,
        })
    }

    fn write_solve(&mut self, scenario_id: u32, evaluated: &EvaluatedSolve) -> Result<()> {
        if evaluated.result.agent_ids.len() != evaluated.result.positions.len() {
            bail!("prediction identifiers and coordinates have different lengths");
        }
        let unique = evaluated
            .result
            .agent_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if unique.len() != evaluated.result.agent_ids.len() {
            bail!("prediction output contains duplicate agent identifiers");
        }
        let expected = evaluated
            .expected_agent_ids
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if unique.difference(&expected).next().is_some() {
            bail!("prediction output contains an unknown agent identifier");
        }
        let positions = evaluated
            .result
            .agent_ids
            .iter()
            .zip(&evaluated.result.positions)
            .map(|(node_id, position)| (*node_id, position))
            .collect::<BTreeMap<_, _>>();
        for node_id in expected {
            write!(
                self.archive,
                "{scenario_id},{},{},{},{node_id},2,",
                evaluated.key.iteration, evaluated.key.samples, evaluated.key.mode,
            )?;
            if let Some(position) = positions.get(&node_id) {
                write!(self.archive, "{:.12},{:.12},", position.x, position.y)?;
                self.nonfinite_coordinates +=
                    usize::from(!position.x.is_finite() || !position.y.is_finite());
            } else {
                write!(self.archive, ",,")?;
                self.missing_coordinates += 1;
            }
            writeln!(self.archive)?;
            self.records += 1;
        }
        Ok(())
    }

    fn finish(self) -> Result<PredictionArchiveSummary> {
        self.archive.finish()?.flush()?;
        fs::rename(&self.temporary_path, &self.path).with_context(|| {
            format!(
                "failed to publish prediction archive {}",
                self.path.display()
            )
        })?;
        let path = self
            .path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("predictions.zip")
            .to_owned();
        Ok(PredictionArchiveSummary {
            path,
            member: "predictions.csv",
            format: "zip-compressed official prediction CSV",
            records: self.records,
            missing_coordinates: self.missing_coordinates,
            nonfinite_coordinates: self.nonfinite_coordinates,
        })
    }
}

fn failed_solve(
    problem: &solid_core::PreparedProblem,
    dimension_policy: DimensionPolicy,
) -> SolveResult {
    let dimensions = usize::from(dimension_policy == DimensionPolicy::Spatial) + 2;
    SolveResult {
        agent_ids: problem.agent_ids.clone(),
        positions: vec![Point2::new(f64::NAN, f64::NAN); problem.agent_ids.len()],
        metrics: SolveMetrics {
            dimensions,
            termination: TerminationReason::LineSearchFailed,
            factors: problem.observations.len(),
            ..SolveMetrics::default()
        },
    }
}

fn summarize_performance(
    mode: String,
    iteration: Option<u32>,
    samples: u32,
    performance: &PerformanceAccumulator,
) -> PerformanceSummary {
    let solves = performance.elapsed_micros.len();
    let elapsed_total: u128 = performance
        .elapsed_micros
        .iter()
        .map(|elapsed| *elapsed as u128)
        .sum();
    PerformanceSummary {
        mode,
        iteration,
        samples,
        dimensions: performance.dimensions.iter().copied().collect(),
        solves,
        solver_mean_micros: elapsed_total as f64 / solves.max(1) as f64,
        solver_p50_micros: percentile_u64(&performance.elapsed_micros, 0.50),
        solver_p95_micros: percentile_u64(&performance.elapsed_micros, 0.95),
        solver_p99_micros: percentile_u64(&performance.elapsed_micros, 0.99),
        solver_max_micros: performance
            .elapsed_micros
            .last()
            .copied()
            .unwrap_or_default(),
        optimizer_mean_iterations: performance.iterations as f64 / solves.max(1) as f64,
        optimizer_max_iterations: performance.max_iterations,
        optimizer_mean_starts: performance.starts as f64 / solves.max(1) as f64,
        optimizer_max_starts: performance.max_starts,
        mean_objective_evaluations: ratio_u128(performance.objective_evaluations, solves),
        mean_gradient_evaluations: ratio_u128(performance.gradient_evaluations, solves),
        mean_range_factor_evaluations: ratio_u128(performance.range_factor_evaluations, solves),
        mean_coordinate_factor_evaluations: ratio_u128(
            performance.coordinate_factor_evaluations,
            solves,
        ),
    }
}

fn ratio_u128(total: u128, count: usize) -> f64 {
    total as f64 / count.max(1) as f64
}

fn percentile_u64(sorted: &[u64], percentile: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

/// Whether the scorers read a predicted position. Before any scorer reads the predictions, a coordinate that is not
/// finite or whose magnitude exceeds MAX_COORDINATE_CM, the engine's bound on a sane coordinate of 1e7 cm, is treated as
/// missing, for every method alike, as research/benchmarks/datasets/scoring.py counts it (scored_point).
fn scored_position(position: &Point2) -> bool {
    [position.x, position.y]
        .iter()
        .all(|value| value.is_finite() && value.abs() <= MAX_COORDINATE_CM)
}

/// A unit's score under the released rule, its mean pair error over the eligible pairs, each error capped at the
/// diameter of the true agent positions. A pair with an agent the estimate lacks, or whose position the coordinate rule
/// makes missing (scored_position), is charged that cap and counted as penalized.
fn score_scenario(
    scenario: &EvaluationScenario,
    problem: &solid_core::PreparedProblem,
    result: &SolveResult,
) -> Result<ScenarioScore> {
    let truth = scenario.true_agent_positions()?;
    let positions: HashMap<u32, Point2> = result
        .agent_ids
        .iter()
        .copied()
        .zip(result.positions.iter().copied())
        .collect();
    let count = problem.agent_ids.len();
    let possible_pairs = count.saturating_mul(count.saturating_sub(1)) / 2;
    let mut parents: Vec<usize> = (0..count).collect();
    let indexes: HashMap<u32, usize> = problem
        .agent_ids
        .iter()
        .enumerate()
        .map(|(index, id)| (*id, index))
        .collect();
    if problem.mode == LocalizationMode::AnchorFree {
        for observation in &problem.observations {
            if !observation.to_is_anchor {
                if let (Some(&left), Some(&right)) =
                    (indexes.get(&observation.from), indexes.get(&observation.to))
                {
                    union_components(&mut parents, left, right);
                }
            }
        }
    }
    let failure_penalty = problem
        .agent_ids
        .iter()
        .enumerate()
        .flat_map(|(left, left_id)| {
            problem.agent_ids[left + 1..]
                .iter()
                .filter_map(|right_id| Some(truth.get(left_id)?.distance(*truth.get(right_id)?)))
        })
        .fold(0.0_f64, f64::max);
    let mut total_error = 0.0;
    let mut eligible_pairs = 0usize;
    let mut penalized_pairs = 0usize;

    for left in 0..count {
        for right in left + 1..count {
            if problem.mode == LocalizationMode::AnchorFree
                && component_root(&mut parents, left) != component_root(&mut parents, right)
            {
                continue;
            }
            eligible_pairs += 1;
            let left_id = problem.agent_ids[left];
            let right_id = problem.agent_ids[right];
            let error = match (
                positions.get(&left_id),
                positions.get(&right_id),
                truth.get(&left_id),
                truth.get(&right_id),
            ) {
                (
                    Some(left_position),
                    Some(right_position),
                    Some(left_truth),
                    Some(right_truth),
                ) if scored_position(left_position) && scored_position(right_position) => {
                    (left_position.distance(*right_position) - left_truth.distance(*right_truth))
                        .abs()
                        .min(failure_penalty)
                }
                _ => {
                    penalized_pairs += 1;
                    failure_penalty
                }
            };
            total_error += error;
        }
    }

    Ok(ScenarioScore {
        mean_error: (eligible_pairs > 0).then_some(total_error / eligible_pairs.max(1) as f64),
        possible_pairs,
        eligible_pairs,
        penalized_pairs,
    })
}

fn summarize_cell(key: &CellKey, cell: &CellAccumulator) -> CellSummary {
    CellSummary {
        mode: key.mode.clone(),
        q: key.q,
        samples: key.samples,
        anchors: key.anchors,
        agents: key.agents,
        scenario_units: cell.scenario_units,
        scored_scenario_units: cell.errors.len(),
        non_scorable_units: cell.scenario_units - cell.errors.len(),
        possible_pairs: cell.possible_pairs,
        eligible_pairs: cell.eligible_pairs,
        scored_pair_coverage: ratio(cell.eligible_pairs, cell.possible_pairs),
        penalized_pairs: cell.penalized_pairs,
        penalized_pair_rate: ratio(cell.penalized_pairs, cell.eligible_pairs),
        mean: (!cell.errors.is_empty())
            .then(|| cell.errors.iter().sum::<f64>() / cell.errors.len() as f64),
        q50: empirical_quantile(&cell.errors, 0.50),
        q75: empirical_quantile(&cell.errors, 0.75),
        q90: empirical_quantile(&cell.errors, 0.90),
    }
}

fn summarize_tracks(
    cells: &BTreeMap<CellKey, CellAccumulator>,
    by_iteration: bool,
) -> Vec<TrackSummary> {
    let mut groups: BTreeMap<(String, Option<u32>, usize, u32), Vec<&CellAccumulator>> =
        BTreeMap::new();
    for (key, cell) in cells {
        groups
            .entry((
                key.mode.clone(),
                by_iteration.then_some(key.iteration),
                key.q,
                key.samples,
            ))
            .or_default()
            .push(cell);
    }
    groups
        .into_iter()
        .map(|((mode, iteration, q, samples), track_cells)| {
            let scored_cells: Vec<&CellAccumulator> = track_cells
                .iter()
                .copied()
                .filter(|cell| !cell.errors.is_empty())
                .collect();
            let scenario_units = track_cells.iter().map(|cell| cell.scenario_units).sum();
            let scored_scenario_units = track_cells.iter().map(|cell| cell.errors.len()).sum();
            let possible_pairs = track_cells.iter().map(|cell| cell.possible_pairs).sum();
            let eligible_pairs = track_cells.iter().map(|cell| cell.eligible_pairs).sum();
            let penalized_pairs = track_cells.iter().map(|cell| cell.penalized_pairs).sum();
            TrackSummary {
                mode,
                iteration,
                q,
                samples,
                cells: track_cells.len(),
                scored_cells: scored_cells.len(),
                scenario_units,
                scored_scenario_units,
                non_scorable_units: scenario_units - scored_scenario_units,
                possible_pairs,
                eligible_pairs,
                scored_pair_coverage: ratio(eligible_pairs, possible_pairs),
                penalized_pairs,
                penalized_pair_rate: ratio(penalized_pairs, eligible_pairs),
                balanced_s50: balanced_quantile(&scored_cells, 0.50),
                balanced_s75: balanced_quantile(&scored_cells, 0.75),
                balanced_s90: balanced_quantile(&scored_cells, 0.90),
            }
        })
        .collect()
}

fn collapse_initializations(
    cells: &BTreeMap<CellKey, CellAccumulator>,
) -> BTreeMap<CellKey, CellAccumulator> {
    let mut collapsed: BTreeMap<CellKey, CellAccumulator> = BTreeMap::new();
    for (key, cell) in cells {
        let official_key = CellKey {
            mode: key.mode.clone(),
            iteration: 0,
            q: key.q,
            samples: key.samples,
            anchors: key.anchors,
            agents: key.agents,
        };
        let target = collapsed.entry(official_key).or_default();
        target.scenario_units += cell.scenario_units;
        target.possible_pairs += cell.possible_pairs;
        target.eligible_pairs += cell.eligible_pairs;
        target.penalized_pairs += cell.penalized_pairs;
        target.errors.extend_from_slice(&cell.errors);
    }
    for cell in collapsed.values_mut() {
        cell.errors.sort_by(f64::total_cmp);
    }
    collapsed
}

fn empirical_quantile(sorted: &[f64], probability: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let index = ((probability * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    Some(sorted[index])
}

fn balanced_quantile(cells: &[&CellAccumulator], probability: f64) -> Option<f64> {
    let mut candidates: Vec<f64> = cells
        .iter()
        .flat_map(|cell| cell.errors.iter().copied())
        .collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by(f64::total_cmp);
    let mut left = 0usize;
    let mut right = candidates.len();
    while left < right {
        let middle = left + (right - left) / 2;
        let threshold = candidates[middle];
        let cdf = cells
            .iter()
            .map(|cell| {
                cell.errors.partition_point(|error| *error <= threshold) as f64
                    / cell.errors.len() as f64
            })
            .sum::<f64>()
            / cells.len() as f64;
        if cdf >= probability {
            right = middle;
        } else {
            left = middle + 1;
        }
    }
    Some(candidates[left.min(candidates.len() - 1)])
}

fn ratio(numerator: usize, denominator: usize) -> f64 {
    numerator as f64 / denominator.max(1) as f64
}

fn component_root(parents: &mut [usize], index: usize) -> usize {
    if parents[index] != index {
        parents[index] = component_root(parents, parents[index]);
    }
    parents[index]
}

fn union_components(parents: &mut [usize], left: usize, right: usize) {
    let left = component_root(parents, left);
    let right = component_root(parents, right);
    if left != right {
        parents[right] = left;
    }
}

fn scenario_id(path: &Path) -> Result<u32> {
    path.file_stem()
        .and_then(|name| name.to_str())
        .and_then(|name| name.parse().ok())
        .with_context(|| format!("scenario filename must be a numeric ID: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scenario(contents: &[u8]) -> EvaluationScenario {
        EvaluationScenario::from_slice(contents).unwrap()
    }

    #[test]
    fn identifier_lists_keep_listed_scenarios_and_refuse_unknown_ones() {
        let directory = std::env::temp_dir().join(format!("solid-ids-{}", std::process::id()));
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("ids.txt");
        fs::write(&path, "7 3\n12\n").unwrap();
        let ids = read_scenario_ids(&path).unwrap();
        assert_eq!(ids.iter().copied().collect::<Vec<_>>(), [3, 7, 12]);

        let mut entries = (1..=12).map(|id| (id, id * 10)).collect::<Vec<_>>();
        retain_entries(&mut entries, &ids).unwrap();
        assert_eq!(entries, [(3, 30), (7, 70), (12, 120)]);
        let mut short = (1..=10).map(|id| (id, id)).collect::<Vec<_>>();
        assert!(retain_entries(&mut short, &ids)
            .unwrap_err()
            .to_string()
            .contains("scenario 12"));

        fs::write(&path, "4 4").unwrap();
        assert!(read_scenario_ids(&path)
            .unwrap_err()
            .to_string()
            .contains("twice"));
        fs::write(&path, " \n").unwrap();
        assert!(read_scenario_ids(&path).is_err());
        fs::write(&path, "4 x").unwrap();
        assert!(read_scenario_ids(&path).is_err());
        fs::remove_dir_all(&directory).unwrap();
    }

    #[test]
    fn identifier_lists_exclude_sampling() {
        let listed = [
            "solid",
            "benchmark",
            "--scenarios",
            "a.zip",
            "--output",
            "b.json",
            "--scenario-ids",
            "ids.txt",
        ];
        assert!(Cli::try_parse_from(listed).is_ok());
        assert!(Cli::try_parse_from(listed.iter().chain(&["--limit", "3"])).is_err());
    }

    #[test]
    fn sampled_scenario_selection_is_deterministic_and_sorted() {
        let entries = (1..=100).map(|id| (id, id * 10)).collect::<Vec<_>>();
        let mut first = entries.clone();
        let mut second = entries;
        select_entries(&mut first, 12, 7);
        select_entries(&mut second, 12, 7);

        assert_eq!(first, second);
        assert!(first.windows(2).all(|pair| pair[0].0 < pair[1].0));
    }

    #[test]
    fn official_cells_pool_initializations_and_separate_dimensions() {
        let mut cells = BTreeMap::new();
        for (iteration, q, error) in [(1, 2, 10.0), (2, 2, 20.0), (1, 3, 30.0)] {
            cells.insert(
                CellKey {
                    mode: "ANC".to_owned(),
                    iteration,
                    q,
                    samples: 1,
                    anchors: 4,
                    agents: 3,
                },
                CellAccumulator {
                    scenario_units: 1,
                    possible_pairs: 3,
                    eligible_pairs: 3,
                    penalized_pairs: 0,
                    errors: vec![error],
                },
            );
        }

        let collapsed = collapse_initializations(&cells);

        assert_eq!(collapsed.len(), 2);
        let planar = collapsed
            .iter()
            .find(|(key, _)| key.q == 2)
            .map(|(_, cell)| cell)
            .unwrap();
        assert_eq!(planar.scenario_units, 2);
        assert_eq!(planar.errors, [10.0, 20.0]);
    }

    #[test]
    fn score_csv_retains_the_exact_official_score_key() {
        let records = [ScoreRecord {
            archive_scenario_id: 17,
            mode: "AAC".to_owned(),
            iteration: 2,
            q: 2,
            samples: 10,
            anchors: 8,
            agents: 4,
            mean_pair_error_cm: Some(12.25),
            possible_pairs: 6,
            eligible_pairs: 6,
            penalized_pairs: 1,
        }];
        let mut output = Vec::new();

        write_score_csv(&mut output, &records).unwrap();

        assert_eq!(
            String::from_utf8(output).unwrap(),
            concat!(
                "archive_scenario_id,mode,iteration,q,samples,anchors,agents,mean_pair_error_cm,possible_pairs,eligible_pairs,penalized_pairs\n",
                "17,AAC,2,2,10,8,4,12.25,6,6,1\n"
            )
        );
    }

    #[test]
    fn missing_agent_penalizes_every_incident_anchored_pair() {
        let scenario = test_scenario(
            br#"{
              "Agents":[1,2,3], "Anchors":[10],
              "True Positions of Agents":{
                "1":{"x":0,"y":0,"z":0},
                "2":{"x":3,"y":0,"z":0},
                "3":{"x":0,"y":4,"z":0}},
              "True Positions of Anchors":{"10":{"x":0,"y":0,"z":0}},
              "Iterations":[{"ID":1,"Initial positions of Agents":{},
              "Ranging samples between Agents":{},"Ranging samples to Anchors":{}}]
            }"#,
        );
        let problem = scenario
            .prepare_problem(1, 1, LocalizationMode::AnchorOnly)
            .unwrap();
        let result = SolveResult {
            agent_ids: vec![1, 3],
            positions: vec![Point2::new(0.0, 0.0), Point2::new(0.0, 1.0)],
            metrics: SolveMetrics::default(),
        };

        let score = score_scenario(&scenario, &problem, &result).unwrap();

        assert_eq!(score.possible_pairs, 3);
        assert_eq!(score.eligible_pairs, 3);
        assert_eq!(score.penalized_pairs, 2);
        assert_eq!(score.mean_error, Some(13.0 / 3.0));
    }

    #[test]
    fn a_coordinate_beyond_the_engine_bound_is_scored_as_missing() {
        let scenario = test_scenario(
            br#"{
              "Agents":[1,2,3], "Anchors":[10],
              "True Positions of Agents":{
                "1":{"x":0,"y":0,"z":0},
                "2":{"x":3,"y":0,"z":0},
                "3":{"x":0,"y":4,"z":0}},
              "True Positions of Anchors":{"10":{"x":0,"y":0,"z":0}},
              "Iterations":[{"ID":1,"Initial positions of Agents":{},
              "Ranging samples between Agents":{},"Ranging samples to Anchors":{}}]
            }"#,
        );
        let problem = scenario
            .prepare_problem(1, 1, LocalizationMode::AnchorOnly)
            .unwrap();
        let score = |agent_ids: Vec<u32>, positions: Vec<Point2>| {
            let result = SolveResult {
                agent_ids,
                positions,
                metrics: SolveMetrics::default(),
            };
            let score = score_scenario(&scenario, &problem, &result).unwrap();
            (score.eligible_pairs, score.penalized_pairs, score.mean_error)
        };
        let with_second = |second: Point2| {
            score(
                vec![1, 2, 3],
                vec![Point2::new(0.0, 0.0), second, Point2::new(0.0, 1.0)],
            )
        };
        // agent 2 left out, as the missing agent test has it
        let missing = score(
            vec![1, 3],
            vec![Point2::new(0.0, 0.0), Point2::new(0.0, 1.0)],
        );
        assert_eq!(missing, (3, 2, Some(13.0 / 3.0)));
        // a coordinate that is not finite, or finite and beyond the engine's bound, makes agent 2 missing, both its
        // pairs charged the diameter of 5 cm and counted as penalized
        let beyond = MAX_COORDINATE_CM * (1.0 + f64::EPSILON);
        assert!(beyond > MAX_COORDINATE_CM);
        for second in [
            Point2::new(f64::NAN, 0.0),
            Point2::new(0.0, f64::NEG_INFINITY),
            Point2::new(1e200, -1e200),
            Point2::new(beyond, 0.0),
            Point2::new(0.0, -beyond),
        ] {
            assert_eq!(with_second(second), missing, "{second:?}");
        }
        // a coordinate at the bound is read, and its pairs take the capped error without a penalty
        assert_eq!(
            with_second(Point2::new(-MAX_COORDINATE_CM, 0.0)),
            (3, 0, Some(13.0 / 3.0))
        );
    }

    #[test]
    fn anchor_free_scoring_excludes_cross_component_pairs() {
        let scenario = test_scenario(
            br#"{
              "Agents":[1,2,3,4], "Anchors":[],
              "True Positions of Agents":{
                "1":{"x":0,"y":0,"z":0},
                "2":{"x":1,"y":0,"z":0},
                "3":{"x":10,"y":0,"z":0},
                "4":{"x":11,"y":0,"z":0}},
              "True Positions of Anchors":{},
              "Iterations":[{"ID":1,"Initial positions of Agents":{},
              "Ranging samples between Agents":{
                "1-2":{"1":1.0},"3-4":{"1":1.0}},
              "Ranging samples to Anchors":{}}]
            }"#,
        );
        let problem = scenario
            .prepare_problem(1, 1, LocalizationMode::AnchorFree)
            .unwrap();
        let result = SolveResult {
            agent_ids: vec![1, 2, 3, 4],
            positions: vec![
                Point2::new(0.0, 0.0),
                Point2::new(1.0, 0.0),
                Point2::new(10.0, 0.0),
                Point2::new(11.0, 0.0),
            ],
            metrics: SolveMetrics::default(),
        };

        let score = score_scenario(&scenario, &problem, &result).unwrap();

        assert_eq!(score.possible_pairs, 6);
        assert_eq!(score.eligible_pairs, 2);
        assert_eq!(score.penalized_pairs, 0);
        assert_eq!(score.mean_error, Some(0.0));
    }
}
