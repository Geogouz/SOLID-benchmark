//! SOLID runner for prepared problems.
//!
//! Reads a request with prepared problems and per unit deployment or cooperative profiles, solves every
//! unit with the production interface, and writes one JSON record per unit. No dataset specific validation.
//! With --identity it prints the identity of the sources it is built from, as the adapters do.
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::time::Instant;

use anyhow::{ensure, Result};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use solid_core::{CooperativeProfile, DeploymentProfile, LocalizationEngine, PreparedProblem};

#[derive(Deserialize)]
struct Unit {
    id: u64,
    samples: u32,
    problem: PreparedProblem,
    #[serde(default)]
    deployment_profile: Option<DeploymentProfile>,
    #[serde(default)]
    cooperative_profile: Option<CooperativeProfile>,
    #[serde(default)]
    label: Option<String>,
}

#[derive(Deserialize)]
struct Request {
    units: Vec<Unit>,
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--identity" {
        println!("{}", env!("SOLID_SOURCE_IDENTITY"));
        return Ok(());
    }
    ensure!(args.len() >= 3, "usage: solid-units request.json output.jsonl [workers] | --identity");
    let workers: usize = args.get(3).map(|w| w.parse()).transpose()?.unwrap_or(4);
    rayon::ThreadPoolBuilder::new().num_threads(workers).build_global()?;
    let request: Request = serde_json::from_reader(BufReader::new(File::open(&args[1])?))?;
    let started = Instant::now();
    let records: Vec<String> = request
        .units
        .par_iter()
        .map(|unit| {
            let t0 = Instant::now();
            let engine = LocalizationEngine::new();
            let result = match (&unit.deployment_profile, &unit.cooperative_profile) {
                (Some(profile), _) => engine.solve_with_profile(&unit.problem, profile.clone()),
                (None, Some(profile)) => engine.solve_with_cooperative_profile(&unit.problem, profile.clone()),
                (None, None) => Err(solid_core::EngineError::CooperativeProfileRequired),
            };
            let elapsed = t0.elapsed().as_nanos() as u64;
            let record = match result {
                Ok(r) => json!({
                    "id": unit.id, "samples": unit.samples, "label": unit.label, "method": "solid",
                    "agent_ids": r.agent_ids,
                    "estimate": {"agent_ids": r.agent_ids, "positions": r.positions.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>()},
                    "metrics": r.metrics, "elapsed_ns": elapsed, "failure": null,
                }),
                Err(e) => json!({
                    "id": unit.id, "samples": unit.samples, "label": unit.label, "method": "solid",
                    "agent_ids": unit.problem.agent_ids, "estimate": null, "elapsed_ns": elapsed, "failure": e.to_string(),
                }),
            };
            serde_json::to_string(&record).expect("serializable record")
        })
        .collect();
    let mut out = BufWriter::new(File::create(&args[2])?);
    for line in &records {
        writeln!(out, "{line}")?;
    }
    out.flush()?;
    eprintln!("solved {} units in {:.1}s", records.len(), started.elapsed().as_secs_f64());
    Ok(())
}
