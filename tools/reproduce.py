#!/usr/bin/env python3
"""Run the commands that reproduce the results of the SOLID paper, stage by stage, in reader mode.

Every command is one of the repository's own, run from the repository root with SOLID_READER=1 and with
OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and RAYON_NUM_THREADS at 1, as in the paper's runs. The driver prints each command
before it runs it and stops at the first that fails. It computes nothing itself. The data folder is SOLID_DATA, or
research/data of the repository when it is unset, and the runs go to out/. In a printed command, python stands for the
interpreter that runs this script.

Stages, in their order:
  build     the executables of SOLID, of the baselines' adapters and of the TWR-CLOUD calibration fitter
  data      checks the downloads against settings/data/pins.json and builds the training and evaluation parts
  settings  refits the shipped settings, optional, and compares them with the shipped files where a script can
  runs      SOLID and the four baselines on the evaluation parts
  tables    the final tables of each dataset, then the paper's tables, figures and counts
  ablation  Table S-XVII, from refits and solves on the training parts
  timing    the controlled timing of Fig. S5

Without --stage the driver runs build, data, runs and tables, and without --dataset all six datasets. The commands that
read several datasets run only when all of them are selected: the checks of the deployment settings and profiles of HYMN
and AAU need both datasets, the ablation the four datasets of its table, and the check of every profile and the paper's
tables, figures and counts all six. The timing and Fig. 2 need Linux.

Usage:
    python tools/reproduce.py [--dataset NAME ...] [--stage NAME ...] [--workers N] [--dry-run]
"""

import argparse
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data")
DATASETS = ("twr-cloud", "tiers", "bregar", "idlab", "hymn", "aau")
DESIGN = ("twr-cloud", "tiers", "bregar", "idlab")
STAGES = ("build", "data", "settings", "runs", "tables", "ablation", "timing")
DEFAULT = ("build", "data", "runs", "tables")
MODES = ("af", "aac", "anc")
PROFILE = dict(af="agent-only", aac="anchor-aided", anc="anchor-only")
PROFILES = "research/artifacts/solid/profiles"
FIGURES = "research/artifacts/solid/figures"
CURVES = f"{FIGURES}/data"
TABLES = "research/artifacts/solid/tables"
TEX = "out/tables"
BUILDERS = "research/benchmarks/datasets"
BERN = "research/benchmarks/bern"
PYTHON = "python"
# the scripts that need GNU tools: the idle checks of the timing and the path checks of the posterior illustration
LINUX = ("experiments/timing/timing_bern.sh", "experiments/figures/posterior_illustration/run_illustration.sh")


def py(script, *args):
    return [PYTHON, script, *map(str, args)]


def fetched(*names):
    return py("tools/fetch_data.py", "--check", "--strict", "--dataset", *names, "scorer")


def build(datasets, workers):
    cargo = ["cargo", "build", "--release", "--locked", "--manifest-path", "research/Cargo.toml"]
    return [[*cargo, "-p", "solid-cli"], [*cargo, "-p", "solid-bern-calibration"]]


def data(datasets, workers):
    rows, inputs = DATA / "corrected-training/rows", DATA / "inputs"
    steps = {
        "twr-cloud": [
            fetched("twr-cloud-bern"),
            py(f"{BERN}/prepare_scenarios.py",
               "--source", inputs / "8617b54db218e099/TWR-CLOUD Bern Eval Scenarios.zip",
               "--output", inputs / "a4f213a271d308d9/TWR-CLOUD-Bern-v5-scorer-input.zip"),
            py(f"{BERN}/calibration/extract_rows.py", "--input", inputs / "65c077bb1aa9b879/Rangings at Responders.pkl",
               "--output", rows),
            py(f"{BERN}/calibration/write_calibration.py", "--rows", rows / "full-training.rows.bin"),
            py(f"{BERN}/calibration/write_calibrated_archive.py"),
            py(f"{BERN}/training/generate_heldscan_corpora.py", "cooperative", "--workers", workers),
            py(f"{BERN}/training/generate_heldscan_corpora.py", "anchor-only", "--workers", workers)],
        # the campaigns of TIERS and IDLab freeze from the check inputs that the prepare stages write, without the
        # adapter checks, with the settings of the paper's campaigns in settings/data
        "tiers": [
            fetched("tiers", "idlab"),
            py(f"{BUILDERS}/audit.py"),
            py(f"{BUILDERS}/static_preflight.py"),
            py(f"{BUILDERS}/pilot.py", "prepare"),
            py(f"{BUILDERS}/calibration.py", "prepare"),
            py(f"{BUILDERS}/aac_preflight.py"),
            py(f"{BUILDERS}/aac_training.py", "prepare"),
            py(f"{BUILDERS}/tiers_static_check.py", "prepare"),
            py(f"{BUILDERS}/tiers_campaign.py", "freeze")],
        "bregar": [
            fetched("bregar"),
            py(f"{BUILDERS}/bregar.py"),
            py(f"{BUILDERS}/bregar.py", "--freeze-split"),
            py(f"{BUILDERS}/bregar.py", "--audit-raw"),
            py(f"{BUILDERS}/bregar_layout.py", "build")],
        "idlab": [
            fetched("idlab"),
            py(f"{BUILDERS}/idlab_admission.py"),
            py(f"{BUILDERS}/idlab_split.py"),
            py(f"{BUILDERS}/idlab_training.py"),
            py(f"{BUILDERS}/idlab_population.py"),
            py(f"{BUILDERS}/idlab_check.py", "prepare"),
            py(f"{BUILDERS}/idlab_campaign.py"),
            py(f"{BUILDERS}/idlab_training_population.py")],
    }
    for dataset in ("hymn", "aau"):
        builder = f"{BUILDERS}/{dataset}_layout.py"
        steps[dataset] = [fetched(dataset), py(builder, "build", "--split", "training"),
                          py(builder, "build", "--split", "evaluation")]
    return [command for dataset in datasets for command in steps[dataset]]


def settings(datasets, workers):
    # SOLID's link fits of the scripts without --check go beside the shipped files, never over them
    steps = {
        "twr-cloud": [
            py("research/solid/fit_bern_mixed_blocks.py", "--held", 0, "--workers", workers),
            py("research/solid/collect_bern_mixed_fits.py", "--all-only", "--all", "bern_mixed_fit_nu3_refit.json"),
            py(f"{BERN}/fit_cooperative.py", "out/settings/bern-cooperative"),
            py("research/solid/deployment.py", "--datasets", "bern", "--check")],
        "tiers": [
            py("research/solid/fit_layout.py", "--datasets", "tiers", "--workers", workers),
            py("research/solid/deployment.py", "--datasets", "tiers", "--check")],
        "bregar": [
            py("research/solid/fit_layout.py", "--datasets", "bregar", "--workers", workers),
            py("research/solid/deployment.py", "--datasets", "bregar", "--check")],
        "idlab": [
            py("research/solid/fit_idlab.py", "--workers", workers),
            py("research/solid/deployment.py", "--datasets", "idlab", "--check")],
    }
    for dataset in ("hymn", "aau"):
        steps[dataset] = [py("research/solid/fit_heldout.py", "links", "--datasets", dataset, "--workers", workers,
                             "--check")]
    commands = [command for dataset in datasets for command in steps[dataset]]
    if "hymn" in datasets and "aau" in datasets:
        commands += [py("research/solid/fit_heldout.py", "deployment", "--check"),
                     py("research/solid/fit_heldout.py", "profiles", "--check")]
    if len(datasets) == len(DATASETS):
        commands.append(py("research/solid/profiles.py", "--check"))
    # ADE-BP's error laws, which ade_training.py fits on TWR-CLOUD, TIERS and IDLab
    ade = [name for dataset, name in (("tiers", "tiers"), ("idlab", "idlab"), ("twr-cloud", "bern"))
           if dataset in datasets]
    if ade:
        commands.append(py(f"{BUILDERS}/ade_training.py", "--datasets", ",".join(ade), "--check"))
    return commands


def runs(datasets, workers):
    bern = [py(f"{BERN}/run_bern.py", "--method", "solid", "--mode", mode, "--output", f"out/solid/bern/{mode}",
               "--workers", workers, "--score") for mode in MODES]
    anchored = ("aac", "anc")
    for method, modes in (("jin-2021", MODES), ("ade-bp-2015", anchored), ("fnl", MODES), ("strong", anchored)):
        bern += [py(f"{BERN}/run_bern.py", "--method", method, "--mode", mode, "--workers", workers, "--score")
                 for mode in modes]
    steps = {
        "twr-cloud": bern,
        "tiers": [py("research/solid/run_tiers.py", "--mode", mode, "--split", "evaluation", "--workers", workers,
                     "--profile", f"{PROFILES}/tiers-{PROFILE[mode]}.json") for mode in MODES]
                 + [py(f"{BUILDERS}/layout_baselines.py", "tiers", "run", "--split", "evaluation", "--mode", mode,
                       "--workers", workers) for mode in MODES],
        "bregar": [py("research/solid/run_bregar.py", "--split", "evaluation", "--workers", workers,
                      "--profile", f"{PROFILES}/bregar-anchor-only.json"),
                   py(f"{BUILDERS}/bregar_layout.py", "evaluate", "--split", "evaluation", "--workers", workers)],
        "idlab": [py("research/solid/run_idlab.py", "--mode", mode, "--split", "evaluation", "--workers", workers,
                     "--profile", f"{PROFILES}/idlab-{PROFILE[mode]}.json") for mode in MODES]
                 + [py(f"{BUILDERS}/idlab_layout_baselines.py", "--split", "evaluation", "--methods",
                       "jin-2021,ade-bp-2015,fnl,strong", "--modes", "af,aac,anc", "--workers", workers)],
    }
    for dataset in ("hymn", "aau"):
        steps[dataset] = [py(f"research/solid/run_{dataset}.py", "--split", "evaluation", "--workers", workers),
                          py(f"{BUILDERS}/heldout_baselines.py", dataset, "run", "--split", "evaluation",
                             "--workers", workers)]
    return [command for dataset in datasets for command in steps[dataset]]


def tables(datasets, workers):
    commands = []
    for dataset in datasets:
        name = "bern" if dataset == "twr-cloud" else dataset
        commands += [py("research/solid/final_tables.py", "build", "--dataset", name),
                     py("experiments/figures/export_cdfs.py", "--dataset", name, "--out", CURVES)]
        if dataset == "twr-cloud":
            commands += [py("experiments/figures/scene_link.py"),
                         py("experiments/figures/scene_curve.py", "--out", f"{CURVES}/scene-curve.json",
                            "--plot", FIGURES)]
        if dataset == "tiers":
            commands.append(py("experiments/figures/tiers_line.py", "--out", f"{CURVES}/tiers-line.json"))
        if dataset == "idlab":
            commands.append(py("experiments/figures/idlab_two_anchors.py", "--out", f"{CURVES}/idlab-two-anchors.json"))
    if len(datasets) == len(DATASETS):
        posterior = "experiments/figures/posterior_illustration"
        record = ("work/figures/posterior-illustration/runs/"
                  "scan-01-scenario-10024-samples-10-iteration-1-bern-anchor-only")
        commands += [py("experiments/figures/plot_cdfs.py", "--data", CURVES, "--out", FIGURES),
                     py("experiments/figures/advantage_share.py", "--out", f"{CURVES}/advantage-shares.json"),
                     py("experiments/figures/summary_table.py", "--tables", TABLES, "--out", TEX),
                     py("experiments/figures/fit_table.py"),
                     py("experiments/figures/full_tables.py", "--tables", TABLES, "--out", TEX),
                     py("experiments/figures/gen_tables.py", f"{TEX}/s3-tables.tex",
                        "--full", f"{TEX}/full-tables.tex"),
                     py("experiments/figures/anchor_starts.py", "--out", f"{CURVES}/anchor-starts.json",
                        "--table", f"{TEX}/s5-stays-table.tex"),
                     py("experiments/figures/anchor_sides.py", "--out", f"{CURVES}/anchor-sides.json"),
                     py("experiments/figures/well_linked.py", "--out", f"{CURVES}/well-linked.json"),
                     # Fig. 2 shows agent 122 of scenario 10024 of the first held scan's fold archive, which
                     # baselines_bern_training.py writes, and the counts read its record
                     py("research/solid/baselines_bern_training.py", "--freeze-only"),
                     ["mkdir", "-p", "work/tmp"],
                     ["sh", f"{posterior}/run_illustration.sh", "run",
                      str(ROOT / "out/solid/unify/baselines-training-v3/scan-01/calibrated.zip"), "10024", "10", "1"],
                     ["cp", f"{record}/record.json", f"{CURVES}/posterior-illustration.json"],
                     py(f"{posterior}/paper-v10/plot_posterior.py", "--agent", 122),
                     py("experiments/figures/paper_counts.py", "--out", f"{CURVES}/paper-counts.json")]
    return commands


def ablation(datasets, workers):
    if not set(DESIGN) <= set(datasets):
        return []
    driver = "research/solid/ablation.py"
    return [py(driver, "plan"), py(driver, "fit", "--workers", workers), py(driver, "run", "--workers", workers),
            py(driver, "baselines", "--workers", workers), py(driver, "score"), py(driver, "table")]


def timing(datasets, workers):
    if "twr-cloud" not in datasets:
        return []
    runs = ["solid:af", "solid:aac", "solid:anc", "jin-2021:af", "jin-2021:aac", "jin-2021:anc", "ade-bp-2015:aac",
            "ade-bp-2015:anc", "fnl:af", "fnl:aac", "fnl:anc", "strong:aac", "strong:anc"]
    return [["sh", "experiments/timing/timing_bern.sh", *runs],
            py("experiments/timing/summarize_timing.py", "--timing", "out/timing/bern", "--evaluation", "out",
               "--out", f"{CURVES}/timing-bern.json"),
            py("experiments/figures/plot_speed.py", "--summary", f"{CURVES}/timing-bern.json", "--out", FIGURES)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", choices=(*DATASETS, "all"), default=["all"])
    parser.add_argument("--stage", nargs="+", choices=(*STAGES, "all"), default=list(DEFAULT))
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="the processes or threads of a run or fit")
    parser.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = parser.parse_args()
    datasets = DATASETS if "all" in args.dataset else tuple(d for d in DATASETS if d in args.dataset)
    stages = STAGES if "all" in args.stage else tuple(s for s in STAGES if s in args.stage)
    environment = dict(os.environ, SOLID_READER="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                       RAYON_NUM_THREADS="1")
    print(f"data folder {DATA}", flush=True)
    for stage in stages:
        for command in globals()[stage](datasets, args.workers):
            print(f"$ {shlex.join(command)}", flush=True)
            if args.dry_run:
                continue
            if command[0] == "sh" and command[1] in LINUX and platform.system() != "Linux":
                sys.exit(f"stopped, {command[1]} needs Linux, for GNU tools")
            argv = [sys.executable, *command[1:]] if command[0] == PYTHON else command
            if subprocess.run(argv, cwd=ROOT, env=environment).returncode:
                sys.exit("stopped, the command above failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
