//! The input of the posterior illustration for one anchor only unit of a scenario archive, and the list of the anchor
//! only scenarios of an archive.
//!
//! The input holds the prepared problem and the deployment profile of the unit exactly as the research CLI builds them for
//! `solid benchmark --method solid --mode anc`, the true positions of the agents, the digest of the problem and the
//! profile, and the result of the CLI's own solve of the unit. The engine test of illustration.rs recomputes the digest
//! from what it reads and solves the unit again, and both must agree bitwise with what this tool wrote.
//!
//! The figure tool builds this file as an example of the solid-cli package in a copy of the research
//! workspace, research/cli/examples/illustration_dump.rs, found by cargo without any change to a manifest. So the lock
//! file stays as it is, `cargo build --locked` holds, and the tool links the crates of the CLI with the features of the
//! CLI's own build, serde_json's float parsing among them.
//!
//! The steps of the CLI it repeats, from research/cli/src/main.rs, are these. The archive's members are every member
//! that is not a folder and whose file stem is a number, the scenario identifier (`ScenarioSource::open`, line 427, and
//! `archive_scenario_id`, line 544). A member is read whole and parsed with `EvaluationScenario::from_slice`
//! (`for_each_batch`, line 505). A scenario with anchors has an anchor only unit, prepared by
//! `prepare_problem(iteration, samples, LocalizationMode::AnchorOnly)` (`evaluate_benchmark_scenario`, lines 1157 and
//! 1169). The deployment profile is one file read by `serde_json::from_reader` (`load_deployment_profile`, line 351),
//! the same for every unit, and the solve is `LocalizationEngine::solve_with_profile` with a clone of it
//! (`Solver::solve_solid`, line 220). Sample counts and iterations are positive (`validate_requested_cases`, line 1208).
//!
//! Usage
//!
//! ```text
//! illustration_dump ARCHIVE SCENARIO SAMPLES ITERATION PROFILE OUTPUT
//! illustration_dump --list ARCHIVE SAMPLES ITERATION
//! ```
//!
//! The list prints one tab separated line per anchor only scenario, its identifier, data role and scan, its agents, its
//! declared anchors, the anchors any agent ranged and the anchors each agent ranged in the order of the agents. A
//! scenario whose unit cannot be prepared, one without the iteration for example, keeps its identifier, role, scan and
//! declared anchors, and the reason stands in the last column.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use serde_json::json;
use solid_core::{
    Anchor, AnchorOnlyProfile, CalibratedRangeSample, DeploymentProfile, LinkAttempt, LocalizationEngine, LocalizationMode, PlanarArea,
    Point2, PreparedProblem, PresenceProfile, RangeObservation,
};
use solid_io::EvaluationScenario;
use zip::ZipArchive;

const USAGE: &str = "usage: illustration_dump ARCHIVE SCENARIO SAMPLES ITERATION PROFILE OUTPUT | --list ARCHIVE SAMPLES ITERATION";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, archive, samples, iteration] if flag.as_str() == "--list" => {
            let (samples, iteration) = case(samples, iteration)?;
            list(Path::new(archive), samples, iteration)
        }
        [archive, scenario, samples, iteration, profile, output] => {
            let scenario: u32 = scenario.parse().with_context(|| format!("invalid scenario identifier {scenario}"))?;
            let (samples, iteration) = case(samples, iteration)?;
            dump(Path::new(archive), scenario, samples, iteration, Path::new(profile), Path::new(output))
        }
        _ => bail!(USAGE),
    }
}

/// The sample count and the iteration of a unit, each positive as the CLI requires.
fn case(samples: &str, iteration: &str) -> Result<(u32, u32)> {
    let count: u32 = samples.parse().with_context(|| format!("invalid sample count {samples}"))?;
    let number: u32 = iteration.parse().with_context(|| format!("invalid iteration {iteration}"))?;
    ensure!(count > 0 && number > 0, "sample counts and iterations must be positive");
    Ok((count, number))
}

/// The scenario identifier of an archive member, its file stem as a number (`archive_scenario_id`).
fn archive_scenario_id(name: &str) -> Option<u32> {
    Path::new(name).file_stem().and_then(|stem| stem.to_str()).and_then(|stem| stem.parse().ok())
}

/// The archive with its members and their scenario identifiers, sorted by identifier, as `ScenarioSource::open` finds
/// them.
fn open_archive(path: &Path) -> Result<(ZipArchive<File>, Vec<(u32, usize)>)> {
    let file = File::open(path).with_context(|| format!("failed to open scenario archive {}", path.display()))?;
    let mut archive = ZipArchive::new(file).with_context(|| format!("invalid scenario archive {}", path.display()))?;
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
    Ok((archive, entries))
}

/// One member of the archive read whole and parsed as the CLI parses it.
fn read_scenario(archive: &mut ZipArchive<File>, index: usize) -> Result<EvaluationScenario> {
    let mut contents = Vec::new();
    archive.by_index(index)?.read_to_end(&mut contents)?;
    Ok(EvaluationScenario::from_slice(&contents)?)
}

/// The deployment profile as `load_deployment_profile` reads it.
fn load_deployment_profile(path: &Path) -> Result<DeploymentProfile> {
    let file = File::open(path).with_context(|| format!("failed to open deployment profile {}", path.display()))?;
    serde_json::from_reader(file).with_context(|| format!("invalid deployment profile {}", path.display()))
}

/// Writes the input of the illustration for scenario `scenario_id` of the archive at `samples` samples per link and
/// iteration `iteration` under the profile at `profile_path`.
fn dump(archive_path: &Path, scenario_id: u32, samples: u32, iteration: u32, profile_path: &Path, output: &Path) -> Result<()> {
    let (mut archive, entries) = open_archive(archive_path)?;
    let indices: Vec<usize> = entries.iter().filter(|(id, _)| *id == scenario_id).map(|(_, index)| *index).collect();
    let index = match indices.as_slice() {
        [index] => *index,
        [] => bail!("scenario {scenario_id} is not in {}", archive_path.display()),
        _ => bail!("scenario {scenario_id} names {} members of {}", indices.len(), archive_path.display()),
    };
    let member = archive.by_index(index)?.name().to_owned();
    let scenario = read_scenario(&mut archive, index)?;
    ensure!(!scenario.anchors.is_empty(), "scenario {scenario_id} declares no anchor, so the CLI solves no anchor only unit of it");
    let problem = scenario.prepare_problem(iteration, samples, LocalizationMode::AnchorOnly)?;
    let profile = load_deployment_profile(profile_path)?;
    let digest = fingerprint(&problem, &profile);
    // the CLI's solve of the unit, the reference the engine test checks its own solve against
    let started = Instant::now();
    let reference = LocalizationEngine::new()
        .solve_with_profile(&problem, profile.clone())
        .with_context(|| format!("the CLI's solve of scenario {scenario_id} fails"))?;
    let seconds = started.elapsed().as_secs_f64();
    let truth: BTreeMap<String, [f64; 2]> =
        scenario.true_agent_positions()?.into_iter().map(|(id, position)| (id.to_string(), [position.x, position.y])).collect();
    let metadata = scenario.research_metadata.as_ref().map(|m| {
        json!({
            "schema_version": m.schema_version,
            "data_role": m.data_role,
            "scan_id": m.scan_id,
            "scenario_id": m.scenario_id,
            "group_index": m.group_index,
            "sample_counts": m.sample_counts,
            "seed": m.seed,
        })
    });
    let record = json!({
        "unit": {
            "archive": archive_path.display().to_string(),
            "member": member,
            "scenario": scenario_id,
            "samples": samples,
            "iteration": iteration,
            "mode": "ANC",
            "profile": profile_path.display().to_string(),
            "agents": problem.agent_ids.len(),
            "anchors": problem.anchors.len(),
            "observations": problem.observations.len(),
            "research_metadata": metadata,
            "reference_seconds": seconds,
        },
        "problem": problem,
        "profile": profile,
        "truth": truth,
        "fingerprint": digest,
        "reference": reference,
    });
    let file = File::create(output).with_context(|| format!("failed to create {}", output.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, &record)?;
    writeln!(writer)?;
    writer.flush()?;
    eprintln!(
        "wrote {} for scenario {scenario_id}, {samples} samples, iteration {iteration}, {} agents and {} anchors, digest {digest}, the CLI's solve took {seconds:.2} s",
        output.display(),
        problem.agent_ids.len(),
        problem.anchors.len()
    );
    Ok(())
}

/// Prints every anchor only scenario of the archive with the anchors its agents ranged at `samples` samples per link and
/// iteration `iteration`.
fn list(archive_path: &Path, samples: u32, iteration: u32) -> Result<()> {
    let (mut archive, entries) = open_archive(archive_path)?;
    let mut out = BufWriter::new(std::io::stdout().lock());
    writeln!(out, "scenario\trole\tscan\tagents\tanchors\tranged\tranged_per_agent")?;
    for (id, index) in entries {
        let scenario = read_scenario(&mut archive, index)?;
        if scenario.anchors.is_empty() {
            continue;
        }
        let (role, scan) = scenario.research_metadata.as_ref().map_or(("-".to_owned(), "-".to_owned()), |m| (m.data_role.clone(), m.scan_id.to_string()));
        let problem = match scenario.prepare_problem(iteration, samples, LocalizationMode::AnchorOnly) {
            Ok(problem) => problem,
            Err(error) => {
                writeln!(out, "{id}\t{role}\t{scan}\t-\t{}\t-\t{error}", scenario.anchors.len())?;
                continue;
            }
        };
        let mut ranged: BTreeMap<u32, BTreeSet<u32>> = BTreeMap::new();
        for observation in problem.observations.iter().filter(|observation| observation.to_is_anchor) {
            ranged.entry(observation.from).or_default().insert(observation.to);
        }
        let every: BTreeSet<u32> = ranged.values().flatten().copied().collect();
        let per_agent: Vec<String> = problem.agent_ids.iter().map(|agent| ranged.get(agent).map_or(0, BTreeSet::len).to_string()).collect();
        writeln!(out, "{id}\t{role}\t{scan}\t{}\t{}\t{}\t{}", problem.agent_ids.len(), scenario.anchors.len(), every.len(), per_agent.join(","))?;
    }
    out.flush()?;
    Ok(())
}

/// FNV-1a over 64 bit words.
struct Digest(u64);

impl Digest {
    fn word(&mut self, value: u64) {
        for byte in value.to_le_bytes() {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    fn numbers(&mut self, values: &[f64]) {
        self.word(values.len() as u64);
        for value in values {
            self.word(value.to_bits());
        }
    }

    fn name(&mut self, name: &str) {
        self.word(name.len() as u64);
        for byte in name.bytes() {
            self.word(u64::from(byte));
        }
    }
}

/// A digest of every number, flag and name of a problem and a profile in the order of their fields, the bits of every
/// number among them, as illustration.rs computes it from the input it reads. Every field is named, so a field this
/// digest missed would not compile.
fn fingerprint(problem: &PreparedProblem, profile: &DeploymentProfile) -> String {
    let mut digest = Digest(0xcbf2_9ce4_8422_2325);
    let PreparedProblem { agent_ids, anchors, observations, initial_positions, mode, attempts } = problem;
    digest.word(agent_ids.len() as u64);
    for id in agent_ids {
        digest.word(u64::from(*id));
    }
    digest.word(anchors.len() as u64);
    for Anchor { id, position, height_cm } in anchors {
        digest.word(u64::from(*id));
        digest.numbers(&[position.x, position.y, *height_cm]);
    }
    digest.word(observations.len() as u64);
    for RangeObservation { from, to, tof_ns, calibrated_samples, to_is_anchor } in observations {
        digest.word(u64::from(*from));
        digest.word(u64::from(*to));
        digest.numbers(tof_ns);
        digest.word(calibrated_samples.len() as u64);
        for CalibratedRangeSample { distance_cm, scale_cm } in calibrated_samples {
            digest.numbers(&[*distance_cm, *scale_cm]);
        }
        digest.word(u64::from(*to_is_anchor));
    }
    digest.word(initial_positions.len() as u64);
    for Point2 { x, y } in initial_positions {
        digest.numbers(&[*x, *y]);
    }
    digest.name(&format!("{mode:?}"));
    match attempts {
        None => digest.word(0),
        Some(attempts) => {
            digest.word(1 + attempts.len() as u64);
            for LinkAttempt { from, to, to_is_anchor } in attempts {
                digest.word(u64::from(*from));
                digest.word(u64::from(*to));
                digest.word(u64::from(*to_is_anchor));
            }
        }
    }
    let DeploymentProfile { area, anchor_only } = profile;
    let PlanarArea { minimum_x_cm, maximum_x_cm, minimum_y_cm, maximum_y_cm } = area;
    digest.numbers(&[*minimum_x_cm, *maximum_x_cm, *minimum_y_cm, *maximum_y_cm]);
    let AnchorOnlyProfile {
        kappa,
        excess_probability,
        excess_mean_rel,
        persistent_sd_rel,
        height_nodes,
        height_weights,
        grid_step,
        joint_samples,
        seed,
        nu,
        independent_packets,
        decision,
        presence,
        offset_shape,
        early_detection_per_cm,
    } = anchor_only;
    digest.numbers(&[*kappa, *excess_probability, *excess_mean_rel, *persistent_sd_rel]);
    digest.numbers(height_nodes);
    digest.numbers(height_weights);
    digest.numbers(&[*grid_step, *nu, *early_detection_per_cm]);
    digest.word(*joint_samples as u64);
    digest.word(*seed);
    digest.word(u64::from(*independent_packets));
    digest.name(&format!("{decision:?} {offset_shape:?}"));
    let PresenceProfile { shape, reach, quantile, factor, width, enabled } = presence;
    digest.name(&format!("{shape:?} {reach:?}"));
    digest.numbers(&[*quantile, *factor, *width]);
    digest.word(u64::from(*enabled));
    format!("{:016x}", digest.0)
}
