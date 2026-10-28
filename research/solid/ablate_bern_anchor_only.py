"""Ablation of the unified anchor only configuration on the held training scans.

For every held scan the campaign's calibrated scenarios (anchor ranges only, fold calibration applied, identical to raw
timings for this estimator) are solved with the campaign's fold candidate profile whose four link parameters are the
fold's mixed fit with the fitted excess probability, the unified base, and with one component removed or replaced per
candidate. The released scorer scores every run and the balanced percentiles pool all held scans with equal weight per
scan, anchor count and agent count cell. Same removal rule as the cooperative ablation. Measurement, not selection.

Usage: ablate_bern_anchor_only.py [--workers 12] [--candidates a,b,c]
"""
import argparse
import csv
import gzip
import math
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import fit_bern  # noqa: E402
import profiles  # noqa: E402
import select_bern  # noqa: E402
import scoring  # noqa: E402
from select_bern_cooperative import carried_link_fields  # noqa: E402

CAMPAIGN = common.BERN_ANCHOR_ONLY_CORPUS
WORK = common.OUT / "unify/ablation-anchor-only"
CLI = common.ROOT / "research/target/release/solid"
LINK_KEYS = ("kappa", "excess_probability", "excess_mean_rel", "persistent_sd_rel")
CANDIDATES = {
    "unified": {},
    "no_offset": {"persistent_sd_rel": 0.0},
    "no_excess": {"excess_probability": 0.0},
    "gaussian_core": {"nu": 1.0e6},
    "independent_packets": {"independent_packets": True},
    "single_height": {"single_height": True},
    "posterior_mean": {"decision": "mean"},
}
RULE = ("Ablation of the unified anchor only configuration on the eleven held scans with the campaign's fold calibrated scenarios and the "
        "released scorer. A component is removable when the candidate without it has balanced S75 not higher in either sample track and S90 "
        "not higher in either track than the unified profile; ties count as removable. Measurement, not selection.")


def base_profile(scan, budget=None):
    """The held scan's deployment: the operating area and the settings that deployment.py gives the fold of `scan`, with
    the anchor only budgets of the factor `budget`, by default the converged factor of the budget check record."""
    return dict(area=deployment.bern_area(held=scan), anchor_only=deployment.budgeted(deployment.settings("bern", held=scan), budget))


def profile_for(scan, name, candidates=CANDIDATES):
    """The profile of a candidate on a held scan with every key the engine requires, the fold's link model with coupled
    packets, the pairwise risk decision and the method's presence law unless the candidate changes them. A candidate may
    name its budget factor, the converged one otherwise."""
    change = dict(candidates[name])
    base = base_profile(scan, change.pop("budget", None))
    fold_fits = change.pop("fold_fits", "bern_mixed_fold_fits.json")
    fit = common.read_json(common.FITS / fold_fits)[str(scan)]["four"]
    # a reduced model of the ablation: the link model without one or more components, refitted on the same blocks, from
    # the reduced fits of the fold fits or from the file the candidate names under fits/reduced
    reduced = change.pop("reduced", None)
    reduced_file = change.pop("reduced_file", fold_fits)
    if reduced:
        fit = common.read_json(common.FITS / "reduced" / reduced_file)["folds"][str(scan)][reduced]["plain"]
    anchor_only = dict(base["anchor_only"], **{key: fit[key] for key in LINK_KEYS}, **carried_link_fields(fit), independent_packets=False, decision="risk",
                       presence=profiles.presence_law())
    if change.pop("single_height", False):
        anchor_only.update(deployment.single_height(anchor_only))
    anchor_only.update(change)
    return dict(base, anchor_only=anchor_only)


def run(scan, name, workers, candidates=CANDIDATES, work=None, profile=None):
    """One candidate on one held scan. profile, when given, is the candidate's profile composed by the caller."""
    directory = (work or WORK) / f"scan-{scan:02d}" / name
    directory.mkdir(parents=True, exist_ok=True)
    scenarios = CAMPAIGN / f"scan-{scan:02d}/calibrated.zip"
    profile = profile_for(scan, name, candidates) if profile is None else profile
    common.write_json(directory / "profile.json", profile)
    identity = dict(binary_sha256=common.sha256(CLI), scenarios_sha256=common.sha256(scenarios), profile=profile, workers=workers,
                    scorer_sha256=common.sha256(select_bern.SCORER), source_sha256=common.sha256(Path(__file__)))
    output = directory / "scenario_scores.csv.gz"
    # the predictions stay beside the scores, so any pair set can score them again without solving, and the report of
    # the solves with the engine's metrics of every unit, so the guarded supports can be counted
    outputs = [output, directory / "predictions.zip", directory / "benchmark.json.gz"]
    manifest = directory / "manifest.json"
    if not common.cache_matches(manifest, identity, outputs):
        with tempfile.TemporaryDirectory(prefix="run-", dir=directory) as temporary:
            stage = Path(temporary)
            started = time.time()
            with (directory / "run.log").open("w") as log:
                subprocess.run([str(CLI), "benchmark", "--scenarios", str(scenarios), "--method", "solid", "--deployment-profile", str(directory / "profile.json"),
                                "--output", str(stage / "benchmark.json"), "--predictions-output", str(stage / "predictions.zip"), "--mode", "anc",
                                "--sample-counts", "1,10", "--iterations", "1,2", "--include-cases", "--time-budget-ms", "0", "--workers", str(workers)],
                               check=True, stdout=log, stderr=subprocess.STDOUT)
            with zipfile.ZipFile(stage / "predictions.zip") as archive:
                archive.extract("predictions.csv", stage)
            # the coordinate rule of every scorer on the extracted copy the released scorer reads (scoring.scored_point)
            scoring.bound_prediction_csv(stage / "predictions.csv")
            subprocess.run([sys.executable, str(select_bern.SCORER), "score-predictions", "--scenarios", str(scenarios), "--predictions", str(stage / "predictions.csv"),
                            "--output-dir", str(stage), "--modes", "ANC", "--samples", "1,10"], check=True, capture_output=True)
            if identity["binary_sha256"] != common.sha256(CLI):
                raise ValueError("Runner changed during the anchor only run")
            shutil.move(str(stage / "scenario_scores.csv.gz"), output)
            shutil.move(str(stage / "predictions.zip"), outputs[1])
            with open(stage / "benchmark.json", "rb") as source, gzip.open(outputs[2], "wb", compresslevel=6) as target:
                shutil.copyfileobj(source, target)
            common.write_json(directory / "timing.json", dict(seconds=time.time() - started, workers=workers))
        common.seal_cache(manifest, identity, outputs)
    rows = {}
    with gzip.open(output, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["mode"] == "ANC" and int(row["q"]) == 2:
                error = float(row["mean_pair_error"])
                if not math.isfinite(error):
                    raise ValueError(f"non finite score in {output}")
                rows[(scan, int(row["scenario_id"]), int(row["iteration"]), int(row["samples"]))] = (error, int(row["anchors"]), int(row["agents"]))
    if len(rows) != 13200:
        raise ValueError(f"{output} holds {len(rows)} units instead of 13200")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--candidates", default=",".join(CANDIDATES))
    args = parser.parse_args()
    names = args.candidates.split(",")
    WORK.mkdir(parents=True, exist_ok=True)
    freeze = dict(schema="solid.ablation.bern-anchor-only.v1", scans=list(fit_bern.SCAN_IDS), candidates={n: CANDIDATES[n] for n in names}, rule=RULE,
                  fold_fits_sha256=common.sha256(common.FITS / "bern_mixed_fold_fits.json"), binary_sha256=common.sha256(CLI),
                  scorer_sha256=common.sha256(select_bern.SCORER), workers=args.workers,
                  scenarios={f"scan-{s:02d}": common.sha256(CAMPAIGN / f"scan-{s:02d}/calibrated.zip") for s in fit_bern.SCAN_IDS},
                  source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_bern.__file__))})
    freeze_path = WORK / "freeze.json"
    if freeze_path.exists() and common.read_json(freeze_path) != freeze:
        raise ValueError("Frozen ablation changed. Review the differences before starting another campaign.")
    common.write_json(freeze_path, freeze)
    scorer = common.load_scorer()
    scores = defaultdict(dict)
    done, total = 0, len(names) * len(fit_bern.SCAN_IDS)
    for name in names:
        for scan in fit_bern.SCAN_IDS:
            scores[name].update(run(scan, name, args.workers))
            done += 1
            common.write_json(WORK / "progress.json", dict(done=done, total=total, complete=done == total))
            print(f"{done}/{total} {name} scan {scan}", flush=True)
    results = {}
    for name in names:
        results[name] = dict(change=CANDIDATES[name], tracks=select_bern.summarize_tracks(scores[name], scorer))
    base = results["unified"]["tracks"]
    for name, entry in results.items():
        if name != "unified":
            entry["removable"] = all(entry["tracks"][q]["balanced"][1] <= base[q]["balanced"][1] and entry["tracks"][q]["balanced"][2] <= base[q]["balanced"][2]
                                     for q in ("1", "10"))
    record = dict(freeze, freeze_sha256=common.sha256(freeze_path), candidates=results, complete=True)
    common.write_json(common.SELECTIONS / "bern-ablation-anchor-only.json", record)
    for name, entry in results.items():
        b1, b10 = entry["tracks"]["1"]["balanced"], entry["tracks"]["10"]["balanced"]
        print(f"  {name:>20}  one sample {b1[0]:6.1f}/{b1[1]:6.1f}/{b1[2]:6.1f}   ten samples {b10[0]:6.1f}/{b10[1]:6.1f}/{b10[2]:6.1f}   removable {entry.get('removable')}")


if __name__ == "__main__":
    main()
