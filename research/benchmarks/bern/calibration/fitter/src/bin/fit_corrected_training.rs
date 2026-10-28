use std::{collections::BTreeSet, path::PathBuf};

use clap::Parser;

use solid_bern_calibration::{fit_calibration, read_row_manifest, TRAINING_SCANS};

#[derive(Parser)]
struct Arguments {
    rows: PathBuf,
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=11))]
    held_scan: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Arguments::parse();
    let mut input = read_row_manifest(&args.rows)?;
    let scans = input
        .rows
        .iter()
        .map(|row| row.scan_id)
        .collect::<BTreeSet<_>>();
    if scans != BTreeSet::from(TRAINING_SCANS) || input.rows.len() != 1_947_886 {
        return Err("expected all eleven training scans and no evaluation rows".into());
    }
    if let Some(scan) = args.held_scan {
        input.rows.retain(|row| row.scan_id != i64::from(scan));
    }
    let fit = fit_calibration(&input.rows)?;
    println!(
        "{{\"range_slope_cm_per_ns\":{},\"range_offset_cm\":{},\"scale_slope\":{},\"scale_intercept_cm\":{},\"scale_min_cm\":8.0,\"scale_max_cm\":55.0,\"runtime_tof_floor_ns\":1.0,\"runtime_tof_max_ns\":200.0}}",
        fit.range_slope_cm_per_ns, fit.range_offset_cm, fit.scale_slope, fit.scale_intercept_cm
    );
    Ok(())
}
