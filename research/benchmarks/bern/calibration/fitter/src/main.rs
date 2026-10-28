use std::path::PathBuf;

use clap::Parser;
use solid_bern_calibration::fit_all;

#[derive(Debug, Parser)]
struct Arguments {
    #[arg(long)]
    calibration: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = Arguments::parse();
    let manifest = fit_all(&arguments.calibration)?;
    println!("{}", manifest.sha256);
    Ok(())
}
