"""Held-scan selection of the cooperative block range model on the TWR-CLOUD-Bern training scans.

The anchor only campaign's protocol applied to the agent only and anchor aided modes. Each training scan is held out
in turn. Its generated training scenarios are positioned from their raw timings, with that fold's calibration applied
inside the engine through the profile (the cooperative median density evaluates the scale law at the candidate distance,
so pre-calibrated packets would change the reference's objective), the reference profile and the block candidates use
the fold nested fits (cooperative-held-XX.json and held-XX.json under out/solid/unify/fits),
every candidate positions all held units for both iterations and sample counts, the released scorer scores them, and
the balanced percentiles pool all held scans with equal weight per scan, anchor count and agent count cell.

Rule. The excess probability is selected jointly across the three modes by the lowest mean balanced S75 over modes and
sample tracks, the anchor only values coming from the campaign's completed record. A cooperative mode adopts the
selected block candidate only if its S75 is lower than the fold nested reference in both tracks and its S90 is not
higher in either; otherwise that mode keeps the reference and the record says so. Ties keep candidate order. The
candidate list, fits, binary, scorer and sources are frozen before any positioning, and no evaluation data enters.

Rounds one to seven ran with the anchor aided shared range bias of the production profile (half normal prior of
10 cm); its ablation in round four found no effect and the component left the method, so the profiles built here
no longer carry it and the round four candidate that removed it is not listed. Rounds one to seven also ran with the
fitted presence laws (the availability law and the anchor detection law), replaced by the method's presence law after
the presence round; the candidates of rounds four, six and seven that switched those laws are not listed either.

Usage: select_bern_cooperative.py [--workers 4] [--freeze-only] [--modes af,aac] [--anc-record PATH]
"""
import argparse
import csv
import gzip
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import fit_bern  # noqa: E402
import profiles  # noqa: E402
import select_bern  # noqa: E402
import scoring  # noqa: E402

ARCHIVE = common.OUT / "unify/bern-training-scenarios-v2.zip"
FITS = common.OUT / "unify/fits"
WORK = common.OUT / "unify/selection-v2"
RUNNER = common.ROOT / "research/target/release/solid-archive"
SCORER = select_bern.SCORER
SCHEMA = "solid.selection.bern-cooperative-block.v2"
CALIBRATION_KEYS = ("range_slope_cm_per_ns", "range_offset_cm", "scale_slope", "scale_intercept_cm", "scale_min_cm", "scale_max_cm")
CANDIDATES = {"reference": None, "excess_fitted": "fitted", "excess_0.05": 0.05, "excess_0.10": 0.10, "excess_0.20": 0.20}
ROUND_TWO = {"reference": None, "excess_fitted": "fitted", "excess_fitted_starts64": dict(excess="fitted", starts=64)}
ANC_NAMES = {"excess_fitted": "excess_fitted", "excess_0.05": "excess_0.05", "excess_0.10": "reference", "excess_0.20": "excess_0.20",
             "excess_fitted_starts64": "excess_fitted"}
ROUND_THREE = {"reference": None, "excess_fitted": "fitted", "excess_fitted_starts64": dict(excess="fitted", starts=64)}
ABLATION = {
    "unified": dict(excess="fitted", starts=64, modes=("af", "aac")),
    "reference": None,
    "no_offset": dict(excess="fitted", starts=64, modes=("af", "aac"), overrides={"block_range.persistent_sd_rel": 0.0}),
    "no_excess": dict(excess="fitted", starts=64, modes=("af", "aac"), overrides={"block_range.excess_probability": 0.0}),
    "gaussian_core": dict(excess="fitted", starts=64, modes=("af", "aac"), overrides={"block_range.nu": 1.0e6}),
    "independent_packets": dict(excess="fitted", starts=64, modes=("af", "aac"), overrides={"block_range.independent_packets": True}),
    "single_height": dict(excess="fitted", starts=64, modes=("aac",), overrides={"block_range.height_nodes": [-61.0], "block_range.height_weights": [1.0]}),
    "warm_start_same_likelihood": dict(excess="fitted", starts=64, modes=("af", "aac"),
                                       overrides={"block_range.refine_from_median": False, "block_range.warm_start_scale": 4.0}),
}
WARM = {
    "unified": dict(excess="fitted", starts=64, modes=("af",)),
    "warm_8": dict(excess="fitted", starts=64, modes=("af",), overrides={"block_range.refine_from_median": False, "block_range.warm_start_scale": 8.0}),
    "warm_16": dict(excess="fitted", starts=64, modes=("af",), overrides={"block_range.refine_from_median": False, "block_range.warm_start_scale": 16.0}),
    "warm_16_4": dict(excess="fitted", starts=64, modes=("af",), overrides={"block_range.refine_from_median": False, "block_range.warm_start_schedule": [16.0, 4.0]}),
    "warm_32_8_2": dict(excess="fitted", starts=64, modes=("af",), overrides={"block_range.refine_from_median": False, "block_range.warm_start_schedule": [32.0, 8.0, 2.0]}),
}
FINAL = {
    "unified": dict(excess="fitted", starts=64, modes=("af", "aac")),
    "no_initializer": dict(excess="fitted", starts=64, modes=("af", "aac"), overrides={"use_provided_start": True}),
}
PRESENCE = {
    "unified": dict(excess="fitted", starts=64, modes=("aac",)),
}
PRESENCE_RULE = ("The presence component complete in the anchor aided mode: the agent pair law and the anchor-link law of the anchor only "
                 "estimator with the fold's fitted range and unit weight, marginalized over the height nodes. Same folds, scorer and cells. "
                 "Cost-aware rule declared before the run: adopted unless it worsens balanced S75 or S90 by more than 0.5 cm times its "
                 "solve-time factor in either track. Measurement, not selection.")
FINAL_RULE = ("Final round of the component inventory, same folds, scorer and cells. Cost-aware rule declared before the run: a component "
              "is kept when its removal worsens balanced S75 or S90 by at least 0.5 cm times its solve-time factor in some track of some "
              "mode where it applies. The geometric initializer (negligible solve time, factor 1) is removed if the released starting "
              "coordinates are within 0.5 cm of it on every percentile and track of both cooperative modes. The missing-link factor, kept "
              "everywhere its inputs exist, is added to the anchor aided objective unless it worsens S75 or S90 by more than 0.5 cm times "
              "its factor in either track. Measurement, not selection.")
WARM_RULE = ("Graduated warm start on the same likelihood, agent only, same folds, scorer and cells. A schedule replaces the median first "
             "stage when its balanced S75 is not higher in either sample track and its S90 not higher in either track than the unified "
             "profile; ties count as replaceable. Measurement, not selection.")
ABLATION_RULE = ("Ablation of the unified configuration on the held scans, same folds, scorer and cells. Each candidate removes or replaces "
                 "one component of the unified profile. A component is removable when the candidate without it has balanced S75 not "
                 "higher in either sample track and S90 not higher in either track than the unified profile, in every mode the component "
                 "applies to; ties count as removable, the simpler model wins. The warm start on the same likelihood replaces the median "
                 "first stage under the same criterion. Measurement, not selection: the unified profile stays the base for every comparison.")
ROUND_THREE_RULE = ("Anchor aided check of the single configuration: the round one winner with 64 starts per multistart stage, the count "
                    "adopted for the agent only mode. Same folds, gate and scorer. Ordering as in the shared bias selection: a candidate "
                    "is adopted when its S75 is lower than the fold nested reference in both tracks and its S90 not higher in either, "
                    "and the adopted candidate with the lowest mean S75 over the two anchor aided tracks is selected.")
ROUND_TWO_RULE = ("Second frozen round for the agent only mode after round one selected the fitted excess probability jointly and the "
                  "agent only gate failed by 1.8 cm of ten sample S90. Same gate, same folds, same scorer. One added candidate, the "
                  "round one winner with 64 starts per multistart stage instead of 32. Selected: the block candidate with the lowest "
                  "mean S75 over the two agent only tracks, adopted only if it passes the gate against the fold nested reference.")
RULE = ("Joint selection of the excess probability: lowest mean balanced S75 over the three modes and both sample tracks, "
        "anchor only values from the campaign record. A cooperative mode adopts the selected block candidate only if its S75 "
        "is lower than the fold nested reference in both tracks and its S90 not higher in either. Ties keep candidate order. "
        "Every held scan, anchor count and agent count cell has equal weight. No evaluation scores enter selection.")


def heights(held=None):
    """The height prior of the anchor links by the deployment rule, over the training scans of the fold of `held`."""
    settings = deployment.settings("bern", held=held)
    return settings["height_nodes"], settings["height_weights"]


# the fields of the link model a fit may carry beyond the four parameters
CARRIED_FIELDS = ("nu",) + profiles.CARRIED_LINK_KEYS


def carried_link_fields(fit):
    """The fields of the link model beyond the four parameters, the degrees of freedom, the offset shape and the false
    early detection rate, each the fit's or, for a fit that does not carry it, the method's, since the engine requires
    every field. Fits of the earlier fitter carry none and take the method's values."""
    return {key: fit.get(key, profiles.LINK_DEFAULTS[key]) for key in CARRIED_FIELDS}


def candidate_profile(mode, held, name, candidates=CANDIDATES):
    """The cooperative profiles per sample count of one candidate on one fold, carrying the fold's calibration and the
    method's presence law, with every key the engine requires. A candidate with a link model is refused unless every
    track names its block likelihood and its calibration (profiles.complete_cooperative)."""
    coop = common.read_json(FITS / f"cooperative-held-{held:02d}.json")
    calibration = {key: coop["identity"]["calibration"][key] for key in CALIBRATION_KEYS}
    spec = candidates[name]
    law = profiles.presence_law()
    base = {n: dict(range_error=coop["range_error"][n], calibration=calibration, presence=dict(law)) for n in ("1", "10")}
    if spec is None:
        # the median only reference of the completed rounds, which has no block likelihood by design and is no profile
        # of the method
        return base
    if isinstance(spec, dict) and mode not in spec.get("modes", ("af", "aac")):
        return None
    excess = spec["excess"] if isinstance(spec, dict) else spec
    record = (common.read_json(common.FITS / spec["fold_fits"])[str(held)] if isinstance(spec, dict) and spec.get("fold_fits")
              else common.read_json(FITS / f"held-{held:02d}.json"))
    fit = record["four"]
    # a reduced model of the ablation: the link model without one or more components, refitted on the same blocks, from
    # the reduced fits of the fold fits or from the file the spec names under fits/reduced
    reduced = isinstance(spec, dict) and spec.get("reduced")
    if reduced:
        path = common.FITS / "reduced" / spec.get("reduced_file", spec["fold_fits"])
        fit = common.read_json(path)["folds"][str(held)][reduced]["plain"]
    nodes, weights = heights(held)
    block = dict(kappa=fit["kappa"], excess_probability=fit["excess_probability"] if excess == "fitted" else excess,
                 excess_mean_rel=fit["excess_mean_rel"], persistent_sd_rel=fit["persistent_sd_rel"], height_nodes=nodes,
                 height_weights=weights, refine_from_median=True, independent_packets=False)
    block.update(carried_link_fields(fit))
    if isinstance(spec, dict) and spec.get("single_height"):
        block.update(deployment.single_height(block))
    extra = {"starts": spec["starts"]} if isinstance(spec, dict) and spec.get("starts") else {}
    composed = {n: dict(profile, block_range=dict(block), **extra) for n, profile in base.items()}
    for key, value in (spec.get("overrides", {}) if isinstance(spec, dict) else {}).items():
        for profile in composed.values():
            target = profile
            parts = key.split(".")
            for part in parts[:-1]:
                target = target[part]
            if value is None:
                target.pop(parts[-1], None)
            else:
                target[parts[-1]] = value
    return profiles.complete_cooperative(composed, f"the Bern cooperative profile of {name} on held scan {held}")


ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


def zip_entry(name):
    """Fixed metadata for an archive entry, so the same scenarios always give the same archive bytes."""
    info = zipfile.ZipInfo(name, date_time=ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o644 << 16
    return info


def prepare_scan(scan):
    """The held scan's generated training scenarios with their raw timings, and the expected units.

    The archive is deterministic: sorted entries with a fixed date, attributes and compression level, so a rebuild
    of the same scenarios reproduces its bytes and the positioning caches keyed on them stay valid. A round prepares
    each scan once and hands the result to every candidate.
    """
    path = WORK / f"scan-{scan:02d}/raw.zip"
    manifest = path.with_suffix(".manifest.json")
    identity = dict(archive_sha256=common.sha256(ARCHIVE), source_sha256=common.sha256(Path(fit_bern.__file__)))
    expected = {"af": set(), "aac": set()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith(f"scan-{scan:02d}/") and n.endswith(".json"))
        for name in names:
            scenario = json.loads(archive.read(name))
            fit_bern.check_scenario(scenario, scan)
            mode = "af" if not scenario.get("True Positions of Anchors") else "aac"
            scenario_id = int(Path(name).stem)
            for iteration in scenario["Iterations"]:
                expected[mode].update((scenario_id, int(iteration["ID"]), samples) for samples in (1, 10))
        if not common.cache_matches(manifest, identity, [path]):
            temporary = path.with_suffix(".partial.zip")
            with zipfile.ZipFile(temporary, "w") as target:
                for name in names:
                    target.writestr(zip_entry(name), archive.read(name), compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
            temporary.replace(path)
            common.seal_cache(manifest, identity, [path])
    return path, expected


def read_official_scores(path, expected, mode):
    rows = {}
    with gzip.open(path, "rt") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["scenario_id"]), int(row["iteration"]), int(row["samples"]))
            if row["mode"] != mode.upper() or int(row["q"]) != 2:
                continue
            error = float(row["mean_pair_error"])
            if key in rows or key not in expected or not math.isfinite(error):
                raise ValueError(f"Invalid or duplicate score in {path}")
            rows[key] = (error, int(row["anchors"]), int(row["agents"]))
    if rows.keys() != expected:
        raise ValueError(f"Missing held training units in {path}: {len(rows)} of {len(expected)}")
    return rows


def run(scan, mode, name, scenarios, expected, workers, candidates=CANDIDATES, work=None, profile=None):
    """One candidate on one held scan and mode. profile, when given, is the candidate's profile composed by the caller."""
    work = work or WORK
    directory = work / f"scan-{scan:02d}" / mode / name
    directory.mkdir(parents=True, exist_ok=True)
    profile = candidate_profile(mode, scan, name, candidates) if profile is None else profile
    if profile is None:
        return None
    common.write_json(directory / "profiles.json", profile)
    identity = dict(binary_sha256=common.sha256(RUNNER), scenarios_sha256=common.sha256(scenarios), profile=profile, workers=workers,
                    samples=[1, 10], iterations=[1, 2], scorer_sha256=common.sha256(SCORER), source_sha256=common.sha256(Path(__file__)))
    outputs = [directory / n for n in ("run-scores.csv", "run-predictions.zip", "scenario_scores.csv.gz")]
    manifest = directory / "manifest.json"
    if not common.cache_matches(manifest, identity, outputs):
        with tempfile.TemporaryDirectory(prefix="run-", dir=directory) as temporary:
            stage = Path(temporary)
            started = time.time()
            with (directory / "run.log").open("w") as log:
                subprocess.run([str(RUNNER), str(scenarios), mode, str(directory / "profiles.json"), "1,10", "1,2", "0", "0", str(workers),
                                str(stage / "run")], check=True, stdout=log, stderr=subprocess.STDOUT)
            with zipfile.ZipFile(stage / "run-predictions.zip") as archive:
                archive.extract("predictions.csv", stage)
            # the coordinate rule of every scorer on the extracted copy the released scorer reads (scoring.scored_point)
            scoring.bound_prediction_csv(stage / "predictions.csv")
            subprocess.run([sys.executable, str(SCORER), "score-predictions", "--scenarios", str(scenarios), "--predictions", str(stage / "predictions.csv"),
                            "--output-dir", str(stage), "--modes", mode.upper(), "--samples", "1,10"], check=True, capture_output=True)
            read_official_scores(stage / "scenario_scores.csv.gz", expected, mode)
            if identity["binary_sha256"] != common.sha256(RUNNER):
                raise ValueError("Runner changed during selection")
            for path in outputs:
                (stage / path.name).replace(path)
            common.write_json(directory / "timing.json", dict(seconds=time.time() - started, workers=workers))
        common.seal_cache(manifest, identity, outputs)
    return read_official_scores(outputs[-1], expected, mode)


def summarize(scores, scorer):
    tracks = {}
    for samples in (1, 10):
        cells = defaultdict(list)
        for (scan, scenario, iteration, q), (error, anchors, agents) in scores.items():
            if q == samples:
                cells[(scan, anchors, agents)].append(error)
        tracks[str(samples)] = dict(units=sum(map(len, cells.values())), cells=len(cells),
                                    balanced=[scorer.balanced_percentile(cells, p) for p in (.5, .75, .9)])
    return tracks


def anc_means(record):
    if record.get("schema") != "solid.selection.bern-anchor-only.v2" or not record.get("complete"):
        raise ValueError("The anchor only selection record must be the completed eleven-fold record")
    return {name: {q: record["candidates"][anc]["tracks"][q]["balanced"][1] for q in ("1", "10")} for name, anc in ANC_NAMES.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--modes", default="af,aac")
    parser.add_argument("--freeze-only", action="store_true", help="write the freeze record and the fold 1 candidate profiles, position nothing")
    parser.add_argument("--anc-record", type=Path, default=common.SELECTIONS / "bern-anchor-only.json")
    parser.add_argument("--round", type=int, default=1, choices=(1, 2, 3, 4, 5, 6, 7),
                        help="2 the agent only second round, 3 the anchor aided 64 start check, 4 the ablation, 5 the warm start schedules, 6 the final round, 7 the presence component")
    args = parser.parse_args()
    modes = args.modes.split(",")
    candidates = {1: CANDIDATES, 2: ROUND_TWO, 3: ROUND_THREE, 4: ABLATION, 5: WARM, 6: FINAL, 7: PRESENCE}[args.round]
    work = WORK if args.round == 1 else WORK.with_name(WORK.name + f"-round{args.round}")
    record_path = common.SELECTIONS / {1: "bern-cooperative-block.json", 2: "bern-cooperative-block-af-round2.json",
                                       3: "bern-cooperative-block-aac-starts64.json", 4: "bern-ablation-cooperative.json",
                                       5: "bern-ablation-warm-start-af.json", 6: "bern-ablation-final.json",
                                       7: "bern-ablation-presence-aac.json"}[args.round]
    rule = {1: RULE, 2: ROUND_TWO_RULE, 3: ROUND_THREE_RULE, 4: ABLATION_RULE, 5: WARM_RULE, 6: FINAL_RULE, 7: PRESENCE_RULE}[args.round]
    work.mkdir(parents=True, exist_ok=True)
    anc_record = common.read_json(args.anc_record)
    anc = anc_means(anc_record)
    fits = {p.name: common.sha256(p) for p in sorted(FITS.glob("*held-*.json")) if not p.name.endswith(".manifest.json")}
    freeze = dict(schema=SCHEMA, round=args.round, scans=list(fit_bern.SCAN_IDS), modes=modes, samples=[1, 10], iterations=[1, 2], candidates=candidates,
                  inputs="raw timings of the held scan's generated training scenarios, fold calibration applied inside the engine through the profile",
                  anchor_only_candidate_names=ANC_NAMES, anchor_only_record_sha256=common.sha256(args.anc_record), anchor_only_mean_s75=anc,
                  height_nodes_from="research/artifacts/solid/profiles/bern-anchor-only.json",
                  archive_sha256=common.sha256(ARCHIVE), fits_sha256=fits, binary_sha256=common.sha256(RUNNER), scorer_sha256=common.sha256(SCORER),
                  workers=args.workers, rule=rule,
                  source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_bern.__file__), Path(select_bern.__file__))})
    freeze_path = work / "freeze.json"
    if freeze_path.exists() and common.read_json(freeze_path) != freeze:
        raise ValueError("Frozen selection changed. Review the differences before starting another campaign.")
    common.write_json(freeze_path, freeze)
    if args.freeze_only:
        preview = work / "preview-scan-01"
        for mode in modes:
            for name in candidates:
                profile = candidate_profile(mode, 1, name, candidates)
                if profile is not None:
                    common.write_json(preview / f"{mode}-{name}.json", profile)
        print(f"freeze written to {freeze_path.relative_to(common.ROOT)}, candidate profiles of fold 1 under {preview.relative_to(common.ROOT)}")
        return
    scorer = common.load_scorer()
    scores = {mode: defaultdict(dict) for mode in modes}
    applicable = {mode: [n for n in candidates if candidate_profile(mode, 1, n, candidates) is not None] for mode in modes}
    total = len(fit_bern.SCAN_IDS) * sum(len(v) for v in applicable.values())
    done = 0
    for scan in fit_bern.SCAN_IDS:
        scenarios, expected = prepare_scan(scan)
        for mode in modes:
            for name in applicable[mode]:
                result = run(scan, mode, name, scenarios, expected[mode], args.workers, candidates, work)
                scores[mode][name].update({(scan, *key): value for key, value in result.items()})
                done += 1
                common.write_json(work / "progress.json", dict(phase="selection", done=done, total=total, complete=done == total))
                print(f"{done}/{total} scan {scan} {mode} {name}: {len(result)} units", flush=True)
    results = {}
    for name in candidates:
        entry = dict(change=candidates[name], tracks={mode: summarize(scores[mode][name], scorer) for mode in modes if name in applicable[mode]})
        if name in anc:
            values = [entry["tracks"][mode][q]["balanced"][1] for mode in entry["tracks"] for q in ("1", "10")] + [anc[name][q] for q in ("1", "10")]
            entry["joint_mean_s75"] = float(np.mean(values))
        if entry["tracks"]:
            entry["mode_mean_s75"] = float(np.mean([entry["tracks"][mode][q]["balanced"][1] for mode in entry["tracks"] for q in ("1", "10")]))
        results[name] = entry
    def passes(name, mode):
        reference, chosen = results["reference"]["tracks"][mode], results[name]["tracks"][mode]
        return all(chosen[q]["balanced"][1] < reference[q]["balanced"][1] and chosen[q]["balanced"][2] <= reference[q]["balanced"][2]
                   for q in ("1", "10"))
    if args.round in (4, 5, 6, 7):
        basis = "ablation"
        base = results["unified"]["tracks"]
        margin = 0.5 if args.round in (6, 7) else 0.0
        for name, entry in results.items():
            if name in ("unified", "reference"):
                continue
            verdict = {}
            for mode, tracks in entry["tracks"].items():
                verdict[mode] = all(tracks[q]["balanced"][1] <= base[mode][q]["balanced"][1] + margin
                                    and tracks[q]["balanced"][2] <= base[mode][q]["balanced"][2] + margin for q in ("1", "10"))
            entry["adoptable" if name.startswith("presence") else "removable"] = verdict
        selected = "unified"
    elif args.round == 3:
        basis = "gate_then_mode_mean_s75"
        eligible = [n for n in results if n != "reference" and all(passes(n, mode) for mode in modes)]
        selected = min(eligible, key=lambda n: results[n]["mode_mean_s75"]) if eligible else "reference"
        for name in results:
            results[name]["passes_gate"] = None if name == "reference" else {mode: passes(name, mode) for mode in modes}
    else:
        basis = "joint_mean_s75" if args.round == 1 else "mode_mean_s75"
        selected = min((n for n in results if basis in results[n]), key=lambda n: results[n][basis])
    adopted = {mode: selected not in ("reference", "unified") and passes(selected, mode) for mode in modes}
    record = dict(freeze, freeze_sha256=common.sha256(freeze_path), candidates=results, selected=selected, selection_basis=basis,
                  adopted=adopted, complete=True)
    common.write_json(record_path, record)
    print(f"selected {selected} by {basis}, adopted per mode {adopted}", flush=True)
    candidates = results
    for name, entry in candidates.items():
        line = "  ".join(f"{mode} {entry['tracks'][mode]['1']['balanced'][1]:.1f}/{entry['tracks'][mode]['10']['balanced'][1]:.1f}" for mode in entry["tracks"])
        if "removable" in entry:
            line += f"  removable {entry['removable']}"
        print(f"  {name:>14} S75 {line}  joint {entry.get('joint_mean_s75', float('nan')):.2f}")


if __name__ == "__main__":
    main()
