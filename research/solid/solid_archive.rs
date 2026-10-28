//! SOLID runner for scenario archives in the agent only and anchor aided modes.
//!
//! Solves the agent only or anchor aided units of a scenario archive with the fitted cooperative profile of
//! each sample count, writes a prediction archive in the official column layout and one score row per unit. It exists because the production interface needs an externally fitted cooperative profile for these
//! modes, so the benchmark command cannot exercise them with the production flag.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::time::Instant;

use anyhow::{ensure, Result};
use rayon::prelude::*;
use solid_core::{CooperativeProfile, LocalizationEngine, LocalizationMode, Point2};
use solid_evaluation::pair_distance_case_error_cm;
use solid_io::EvaluationScenario;

struct Row {
    scenario: u32,
    iteration: u32,
    samples: u32,
    anchors: usize,
    agents: usize,
    error_cm: f64,
    positions: Vec<(u32, Point2)>,
}

fn shuffle(ids: &mut Vec<(u32, String)>, seed: u64) {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    for i in (1..ids.len()).rev() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let j = (z % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        args.len() == 10,
        "usage: solid-archive scenarios.zip af|aac profiles.json samples iterations limit seed workers out-prefix"
    );
    let mode = match args[2].as_str() {
        "af" => LocalizationMode::AnchorFree,
        "aac" => LocalizationMode::AnchorAided,
        other => anyhow::bail!("unknown mode {other}"),
    };
    // One cooperative profile per sample count, keyed by the sample count as a string.
    let profiles: BTreeMap<String, CooperativeProfile> =
        serde_json::from_reader(BufReader::new(File::open(&args[3])?))?;
    let samples: Vec<u32> = args[4].split(',').map(|v| v.parse()).collect::<Result<_, _>>()?;
    // every requested sample count needs its profile, or its units would drop out of the scores unseen
    let missing: Vec<u32> = samples.iter().copied().filter(|count| !profiles.contains_key(&count.to_string())).collect();
    ensure!(missing.is_empty(), "no cooperative profile for the sample counts {missing:?}");
    let iterations: Vec<u32> = args[5].split(',').map(|v| v.parse()).collect::<Result<_, _>>()?;
    let limit: usize = args[6].parse()?;
    let seed: u64 = args[7].parse()?;
    let workers: usize = args[8].parse()?;
    let prefix = &args[9];
    rayon::ThreadPoolBuilder::new().num_threads(workers.max(1)).build_global()?;

    let mut archive = zip::ZipArchive::new(BufReader::new(File::open(&args[1])?))?;
    let mut members: Vec<(u32, String)> = (0..archive.len())
        .filter_map(|i| {
            let file = archive.by_index(i).ok()?;
            let name = file.name().to_owned();
            if !name.ends_with(".json") || name.starts_with("__MACOSX") {
                return None;
            }
            let stem = name.rsplit('/').next()?.trim_end_matches(".json");
            stem.parse::<u32>().ok().map(|id| (id, name))
        })
        .collect();
    members.sort();
    if limit > 0 && limit < members.len() {
        shuffle(&mut members, seed);
        members.truncate(limit);
        members.sort();
    }
    let mut contents: Vec<(u32, Vec<u8>)> = Vec::with_capacity(members.len());
    for (id, name) in &members {
        let mut file = archive.by_name(name)?;
        let mut bytes = Vec::with_capacity(file.size() as usize);
        file.read_to_end(&mut bytes)?;
        contents.push((*id, bytes));
    }
    let started = Instant::now();
    // parse and check every scenario in parallel, then solve every unit in parallel: units are the work items, so
    // a few slow scenarios no longer leave the other threads idle; the row order stays the scenario order
    let prepared: Vec<_> = contents
        .par_iter()
        .filter_map(|(id, bytes)| {
            let scenario = EvaluationScenario::from_slice(bytes).ok()?;
            let applicable = match mode {
                LocalizationMode::AnchorFree => scenario.anchors.is_empty(),
                _ => !scenario.anchors.is_empty(),
            };
            if !applicable {
                return None;
            }
            let truth = scenario.true_agent_positions().ok()?;
            let anchor_truth = scenario.true_anchor_positions().unwrap_or_default();
            let mut declared: Vec<u32> = scenario.anchors.clone();
            declared.sort_unstable();
            Some((*id, scenario, truth, anchor_truth, declared))
        })
        .collect();
    let mut units = Vec::with_capacity(prepared.len() * iterations.len() * samples.len());
    for index in 0..prepared.len() {
        for &iteration in &iterations {
            for &count in &samples {
                units.push((index, iteration, count));
            }
        }
    }
    let rows: Vec<Row> = units
        .par_iter()
        .filter_map(|&(index, iteration, count)| {
            let (id, scenario, truth, anchor_truth, declared) = &prepared[index];
            let problem = scenario.prepare_problem(iteration, count, mode).ok()?;
            let profile = profiles[&count.to_string()].clone();
            let result = LocalizationEngine::new().solve_with_cooperative_profile(&problem, profile);
            let positions: Vec<(u32, Point2)> = match &result {
                Ok(r) => r.agent_ids.iter().copied().zip(r.positions.iter().copied()).collect(),
                Err(_) => Vec::new(),
            };
            let error_cm = pair_distance_case_error_cm(&positions, truth, declared, anchor_truth)
                .map(|s| s.error_cm)
                .unwrap_or(f64::NAN);
            Some(Row { scenario: *id, iteration, samples: count, anchors: scenario.anchors.len(), agents: scenario.agents.len(), error_cm, positions })
        })
        .collect();
    let mut scores = std::io::BufWriter::new(File::create(format!("{prefix}-scores.csv"))?);
    writeln!(scores, "scenario_id,iteration,samples,anchors,agents,mean_pair_error_cm")?;
    let mut csv = String::from("scenario_id,iteration,samples,mode,node_id,q,x,y,z\n");
    let mode_name = match mode {
        LocalizationMode::AnchorFree => "AF",
        LocalizationMode::AnchorAided => "AAC",
        _ => "ANC",
    };
    for row in &rows {
        writeln!(scores, "{},{},{},{},{},{}", row.scenario, row.iteration, row.samples, row.anchors, row.agents, row.error_cm)?;
        for (node, p) in &row.positions {
            csv.push_str(&format!("{},{},{},{},{},2,{},{},\n", row.scenario, row.iteration, row.samples, mode_name, node, p.x, p.y));
        }
    }
    scores.flush()?;
    let file = File::create(format!("{prefix}-predictions.zip"))?;
    let mut writer = zip::ZipWriter::new(file);
    writer.start_file("predictions.csv", zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated))?;
    writer.write_all(csv.as_bytes())?;
    writer.finish()?;
    eprintln!("solved {} units from {} scenarios in {:.1}s", rows.len(), contents.len(), started.elapsed().as_secs_f64());
    Ok(())
}
