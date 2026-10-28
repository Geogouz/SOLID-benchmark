"""The ablation of Table S-XVII of the companion document, S75 on the training parts with one part of the block
likelihood removed, recomputed from the public data.

The method's link model is refitted without each of three parts, a Gaussian core in place of the Student t core
(gaussian_core, nu 1e6), no excess (no_excess, an excess probability of zero with a mean of one reference scale) and no
offset (no_offset, a persistent offset of zero), and the method and each removal are positioned on the training parts of
TWR-CLOUD-Bern, TIERS, Bregar and IDLab. Every candidate is the method's spec of research/solid/method_constants.py, the
suffix _r0, the maximum likelihood reach and twice the converged budgets, with the one part refitted out
(ablation_round.removal_spec). Every scenario is positioned with fits that its own measurements did not enter: every
held TWR-CLOUD scan with the fits of the other ten scans, the windows of every TIERS recording, the channels of every
Bregar environment and the radio configurations of IDLab split into the two folds of research/solid/folds.py and each
fold positioned with the fits of the other. On IDLab the parts hold the sweeps, the panels whose key panel-NNNN hashes
to class zero modulo eight. The compositions, the runner and the decision are those of the round (ablation_round.py),
and the S75 of the method and of each removal with the standard error of their difference come from decision.decide, the
paired cluster bootstrap of research/solid/decision.py with 2000 draws and seed 20260925, over the joined pairs, the
anchor only rows of TIERS and IDLab over three or more anchors.

The table's last column gives, on the same scenarios, the baseline of least S75 in that row of Table II, the headline of
the evaluation half, read from the tables of research/solid/final_tables.py build under --tables, or as the paper prints
it (TABLE_II) for a dataset whose tables the folder does not hold, with the options and the error law of ADE-BP that
the evaluation half takes (settings/baseline-options.json). On TWR-CLOUD-Bern these are the baselines' runs
on the eleven held scans of baselines_bern_training.py, gathered by gather_bern, on TIERS the
training half of layout_baselines.py, on IDLab FNL on the training half of idlab_layout_baselines.py scored on the same
sweeps, and on Bregar ADE-BP alone on the 48 training units with the error law of each environment. The paper's
training runs of ADE-BP on IDLab took the error law of model 2, while the evaluation half takes model 1 in the anchor
only mode, so that row shows a dash and the record keeps the reason. Where the baseline has the lower S75, the record
keeps SOLID minus that baseline with its paired 95 percent interval, on TIERS from the tables' own bootstrap
(final_tables.analyse of the training half), elsewhere from decision.compare, and on TIERS also the S75 of SOLID and
ADE-BP per recording.

Stages, each a command, every output under out/ablation of the checkout unless --out names another folder:

  plan       every fit and refit and every run with its dataset, removal, fold or held scan, mode, readings and
             scenarios and their units, the solves in all and the inputs every stage reads, present or missing. It runs
             nothing.
  fit        the link model of the round, R0, on each fold of TIERS, Bregar and IDLab and on the ten other scans of
             every held TWR-CLOUD scan, by the fitters of fit_layout.py, fit_idlab.py and fit_bern_mixed_blocks.py, then
             its refit without each part on the same blocks and weights by fit_reduced.py's run_fit, under fits/ with the
             names of research/artifacts/solid/fits. A fit already written is kept.
  run        the method and every removal on every part, the round's runner, cached by the content of every profile and
             its inputs under runs/
  baselines  the baselines' training runs the last column reads. Their runners keep their own folders: the held Bern
             scans under out/solid/unify/baselines-training-v3, gathered under baselines/bern, TIERS and IDLab under
             out/baselines/<dataset>/training, and ADE-BP on Bregar under baselines/bregar.
  score      the decisions and the baselines' values from the stored runs, written to ablation.json, the record, with
             every decision under decisions.json. It solves nothing.
  table      Table S-XVII in the paper's LaTeX from a record, ablation-table.tex

The cooperative TWR-CLOUD runs read the held-scan corpora of generate_heldscan_corpora.py cooperative and the fold
first stages of fit_bern_cooperative_folds.py under out/solid/unify, and the anchor only runs the fold calibrated
corpora of generate_heldscan_corpora.py anchor-only, which need the fold calibrations of
research/benchmarks/bern/calibration/write_calibration.py. The cooperative first stage of each IDLab fold is the shipped
fits/idlab_first_stage.json, which fit_idlab.py --first-stage fits again. plan names every input with its command.

Usage: ablation.py plan [--json FILE]
       ablation.py fit [--workers 64] [--datasets bern,tiers,bregar,idlab] [--removals a,b] [--parts 1,2] [--full-only]
       ablation.py run [--workers 12] [--datasets ...] [--candidates reference,gaussian_core,no_excess,no_offset]
       ablation.py baselines [--workers 12] [--datasets ...]
       ablation.py score [--tables research/artifacts/solid/tables] [--datasets ...]
       ablation.py table [--record out/ablation/ablation.json] [--latex out/ablation/ablation-table.tex]
       every command takes --out DIR, the folder of the stages, out/ablation by default
"""
import argparse
import csv
import functools
import gzip
import hashlib
import json
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablation_round as rounds  # noqa: E402
import common  # noqa: E402
import decision  # noqa: E402
import deployment  # noqa: E402
import fit_bern  # noqa: E402
import folds  # noqa: E402
import layout_split  # noqa: E402
import method_constants  # noqa: E402
import run_idlab  # noqa: E402

OUT = common.ROOT / "out/ablation"
SCHEMA = "solid.figures.ablation.v2"
STATEMENT = "research/solid/ablation.py"
SUFFIX = method_constants.SUFFIX
REMOVALS = rounds.REMOVALS
CANDIDATES = ("reference",) + REMOVALS
DATASETS = (("bern", "TWR-CLOUD"), ("tiers", "TIERS"), ("bregar", "Bregar"), ("idlab", "IDLab"))
MODES = ("AF", "AAC", "ANC")
NAMES = {"jin-2021": "Jin", "fnl": "FNL", "ade-bp-2015": "ADE-BP", "strong": "STRONG"}
# the deployments whose anchor only rows the record scores over three or more anchors
SCOPED = {"tiers", "idlab"}
LEVEL = 0.75
# the link model of the round, R0: the method's degrees of freedom, a Gaussian offset and no early detection rate, the
# defaults of fit_layout.py, fit_idlab.py and fit_bern_mixed_blocks.py, and at most 2000 blocks per track of a layout fit
MODEL = dict(offset_shape=method_constants.OFFSET_SHAPE, fit_early_detection=False, nu=method_constants.NU)
LAYOUT_BLOCKS = 2000
# the blocks of a Bern fold fit as fit_bern_mixed_blocks.py draws them by default
BERN_BLOCKS = dict(per_scan=40, blocks_per_kind=600, seed=11)
# the scenarios per held scan of the corpora of generate_heldscan_corpora.py, for a plan without the corpora
BERN_DESIGN = dict(af=48, aac=240, anc=3300)
# The baseline of least S75 in every row of Table II, the headline of the evaluation half, as the paper prints it. The
# score takes the choice of the tables of final_tables.py build under --tables, and this one for a dataset whose
# evaluation tables the folder does not hold.
TABLE_II = {("bern", "AF", 1): "fnl", ("bern", "AF", 10): "fnl", ("bern", "AAC", 1): "ade-bp-2015", ("bern", "AAC", 10): "ade-bp-2015",
            ("bern", "ANC", 1): "ade-bp-2015", ("bern", "ANC", 10): "jin-2021", ("tiers", "AF", 1): "jin-2021", ("tiers", "AF", 10): "jin-2021",
            ("tiers", "AAC", 1): "ade-bp-2015", ("tiers", "AAC", 10): "ade-bp-2015", ("tiers", "ANC", 1): "ade-bp-2015", ("tiers", "ANC", 10): "ade-bp-2015",
            ("bregar", "ANC", 1): "ade-bp-2015", ("bregar", "ANC", 10): "ade-bp-2015", ("idlab", "AF", 1): "fnl", ("idlab", "AF", 10): "fnl",
            ("idlab", "AAC", 1): "fnl", ("idlab", "AAC", 10): "fnl", ("idlab", "ANC", 1): "ade-bp-2015", ("idlab", "ANC", 10): "ade-bp-2015"}
# The rows whose baseline has no training run with the options the evaluation half takes, with the reason the record
# keeps. The paper's training runs of ADE-BP on IDLab took the error law of model 2 in every mode, while the evaluation
# half takes model 1 in the anchor only mode (settings/baseline-options.json).
NO_RUN = {("idlab", "ANC", "ade-bp-2015"): "its error law in the gate is not the evaluation's"}
# The baselines whose training runs the last column reads, per dataset and mode, those of TABLE_II with a run. The score
# refuses a row whose baseline in the reader's Table II has no run here. TIERS runs every baseline of a mode.
BASELINES = {dataset: {mode: tuple(sorted({m for (d, o, _), m in TABLE_II.items() if (d, o) == (dataset, mode) and (d, o, m) not in NO_RUN}))
                       for mode in MODES if any((d, o, m) not in NO_RUN for (d, o, _), m in TABLE_II.items() if (d, o) == (dataset, mode))}
             for dataset in ("bern", "tiers", "bregar", "idlab")}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    """A path relative to the checkout when it lies inside it."""
    path = Path(path)
    return str(path.relative_to(common.ROOT)) if path.resolve().is_relative_to(common.ROOT) else str(path)


def candidate_specs():
    """The method, the reference of every removal, and the method without each part of its link model."""
    return dict(reference=rounds.method_spec(), **{name: rounds.removal_spec(name) for name in REMOVALS})


def bregar_environments():
    """The Bregar environments of the deployment record, each with its own link model."""
    return sorted(deployment.load()["datasets"]["bregar"]["environments"])


# ------------------------------------------------------------------------------------------- the plan
def fit_records():
    """Every record of a fit of the round's link model that the removals refit, (dataset, part, environment): every
    held Bern scan, each fold of TIERS and IDLab and each fold and environment of Bregar."""
    records = [("bern", scan, None) for scan in fit_bern.SCAN_IDS]
    records += [("tiers", fold, None) for fold in folds.FOLDS]
    records += [("bregar", fold, environment) for fold in folds.FOLDS for environment in bregar_environments()]
    records += [("idlab", fold, None) for fold in folds.FOLDS]
    return records


def fit_file(dataset, part):
    """The fit file of a part under fits, the Bern file holding every held scan."""
    return rounds.fit_name(dataset, SUFFIX, None if dataset == "bern" else part)


def record_label(dataset, part, environment):
    where = f"scan {part:02d}" if dataset == "bern" else f"fold {part}"
    return f"{dataset} {where}" + (f" {environment}" if environment else "")


def fitted_on(dataset, part):
    if dataset == "bern":
        return f"the ten training scans other than scan {part}"
    return "the clusters of fold %d: %s" % (part, folds.CLUSTERS[dataset])


def plan_fits():
    """The fits of the round's link model and their refits without each part."""
    fits, refits = [], []
    for dataset, part, environment in fit_records():
        entry = dict(dataset=dataset, part=part, environment=environment, file=fit_file(dataset, part), fitted_on=fitted_on(dataset, part),
                     model=dict(MODEL, suffix=SUFFIX), positions=(f"held scan {part}" if dataset == "bern" else f"fold {folds.other(part)}"))
        fits.append(entry)
        for name in REMOVALS:
            refits.append(dict(entry, removal=name, file=f"reduced/{entry['file']}", key=name,
                               refit="fit_reduced.run_fit on the blocks and weights of the fit, its optimum one more start"))
    return fits, refits


def tiers_units(mode, held):
    catalog = json.load(gzip.open(common.EXTERNAL / "tiers-static-campaign/catalog.json.gz"))
    half = layout_split.tiers_blocks("training") & folds.tiers_blocks(held)
    batches = [b for b in catalog if b["track"] == "calibrated" and b["mode"] == mode.upper() and b["block"] in half]
    return sum(b["units"] for b in batches), dict(blocks=sorted(half), batches=len(batches), track="calibrated")


def bregar_units(held):
    meta = {unit["id"]: unit for unit in common.read_json(rounds.run_bregar.E / "training/reference.json")["units"]}
    acquisitions = folds.bregar_acquisitions(held)
    kept = [unit for unit in sorted(layout_split.bregar_units("training")) if meta[unit]["acquisition"] in acquisitions]
    return 2 * len(kept), dict(acquisitions=sorted(acquisitions), scenarios=len(kept))


def idlab_units(mode, held):
    catalog = json.load(gzip.open(common.EXTERNAL / run_idlab.CAMPAIGNS["training"] / "catalog.json.gz"))
    configurations = folds.idlab_configurations(held)
    batches = run_idlab.batches_of(catalog, mode.upper(), 1, rounds.PARAMETERS["idlab_panel_stride"], configurations)
    return sum(b["units"] for b in batches), dict(configurations=sorted(configurations), panels=len({b["panel"] for b in batches}),
                                                  batches=len(batches), panel_stride=rounds.PARAMETERS["idlab_panel_stride"],
                                                  panel_key=run_idlab.PANEL_KEY)


@functools.lru_cache(maxsize=None)
def cooperative_counts(scan):
    """The scenarios and units of every cooperative mode of one held scan in the held-scan corpora, every scenario with
    each of its iterations and both sample tracks."""
    counts = dict(af=[0, 0], aac=[0, 0])
    with zipfile.ZipFile(rounds.cooperative.ARCHIVE) as archive:
        for name in archive.namelist():
            if name.startswith(f"scan-{scan:02d}/") and name.endswith(".json"):
                scenario = json.loads(archive.read(name))
                entry = counts["aac" if scenario.get("True Positions of Anchors") else "af"]
                entry[0] += 1
                entry[1] += 2 * len(scenario["Iterations"])
    return counts


def bern_units(mode, scan):
    """The units of one held Bern scan and mode, counted from the corpora when they are present and from their
    declared design otherwise, every scenario with both iterations and both sample tracks."""
    if mode == "anc":
        path = rounds.anchor_only.CAMPAIGN / f"scan-{scan:02d}/calibrated.zip"
        if path.exists():
            with zipfile.ZipFile(path) as archive:
                scenarios = sum(1 for name in archive.namelist() if name.endswith(".json"))
            return 4 * scenarios, dict(corpus=relative(path), scenarios=scenarios, counted=True)
    else:
        path = rounds.cooperative.ARCHIVE
        if path.exists():
            scenarios, units = cooperative_counts(scan)[mode]
            return units, dict(corpus=f"{relative(path)} scan-{scan:02d}", scenarios=scenarios, counted=True)
    return 4 * BERN_DESIGN[mode], dict(corpus=relative(path), scenarios=BERN_DESIGN[mode], counted=False)


def plan_runs():
    """Every run of every candidate: dataset, mode, held part, the fit that positions it, the readings and the
    scenarios with their units, which the round's runner solves once per candidate."""
    runs, counted = [], {}
    for name, candidate in candidate_specs().items():
        for dataset in rounds.DEPLOYMENTS:
            for mode in rounds.MODES[dataset]:
                for held in rounds.held_parts(dataset):
                    key = (dataset, mode, held)
                    if key not in counted:
                        try:
                            counted[key] = (bern_units(mode, held) if dataset == "bern" else tiers_units(mode, held) if dataset == "tiers"
                                            else bregar_units(held) if dataset == "bregar" else idlab_units(mode, held))
                        except OSError as error:
                            counted[key] = (None, dict(missing=str(error)))
                    units, scenarios = counted[key]
                    fit = rounds.fit_fold(held) if dataset != "bern" else held
                    link = dict(file=fit_file(dataset, fit), record=str(held) if dataset == "bern" else f"fold {fit}")
                    if candidate["reduced"]:
                        link["refit"] = dict(file=f"reduced/{link['file']}", key="+".join(candidate["reduced"]))
                    runs.append(dict(candidate=name, spec=candidate, dataset=dataset, mode=mode, held=held,
                                     positioned_with=(f"the fits of the ten scans other than {held}" if dataset == "bern" else f"the fits of fold {fit}"),
                                     link_model=link, readings=[1, 10], units=units, scenarios=scenarios))
    return runs


def inputs():
    """The inputs of the stages, each with the command that writes it, and whether it is present."""
    unify = common.OUT / "unify"
    entries = [("the training half of TIERS", common.EXTERNAL / name, "research/benchmarks/datasets (the TIERS builders of the data folder)")
               for name in rounds.EXTERNAL_INPUTS["tiers"]]
    entries += [("the training half of Bregar", common.EXTERNAL / name, "research/benchmarks/datasets/bregar_layout.py build") for name in rounds.EXTERNAL_INPUTS["bregar"]]
    entries += [("the IDLab training campaign", common.EXTERNAL / name, "research/benchmarks/datasets (the IDLab builders of the data folder)")
                for name in rounds.EXTERNAL_INPUTS["idlab"]]
    entries += [("the reserved IDLab rows of the fits", common.EXTERNAL / "idlab-training.json", "research/benchmarks/datasets/idlab_training.py"),
                ("the IDLab first stage of each fold", common.FITS / rounds.IDLAB_FIRST_STAGE, "shipped, or research/solid/fit_idlab.py --first-stage")]
    entries += [("the full calibration of TWR-CLOUD", fit_bern.FULL_CALIBRATION, "research/benchmarks/bern/calibration/write_calibration.py")]
    entries += [(f"the fold calibration of scan {scan}", fit_bern.SCANS / f"scan-{scan:02d}/fit.json", "research/benchmarks/bern/calibration/write_calibration.py")
                for scan in fit_bern.SCAN_IDS]
    entries += [("the cooperative held-scan corpora", rounds.cooperative.ARCHIVE, "research/benchmarks/bern/training/generate_heldscan_corpora.py cooperative")]
    entries += [(f"the anchor only corpus of scan {scan}", rounds.anchor_only.CAMPAIGN / f"scan-{scan:02d}/calibrated.zip",
                 "research/benchmarks/bern/training/generate_heldscan_corpora.py anchor-only") for scan in fit_bern.SCAN_IDS]
    entries += [(f"the cooperative first stage of fold {held}", rounds.cooperative.FITS / f"cooperative-held-{held:02d}.json",
                 "research/solid/fit_bern_cooperative_folds.py") for held in fit_bern.SCAN_IDS]
    entries += [("the released scorer", common.DATA / "inputs/0068d72b94db6b1c/score_twr_cloud_bern.py", "tools/fetch_data.py")]
    build = "cargo build --release --locked --manifest-path research/Cargo.toml -p solid-cli"
    entries += [(f"the runner {path.name}", path, build) for path in (common.BINARY, rounds.cooperative.RUNNER, rounds.anchor_only.CLI)]
    return [dict(input=what, path=relative(path), present=Path(path).exists(), written_by=command) for what, path, command in entries]


def plan(out):
    """The whole plan, which runs nothing."""
    fits, refits = plan_fits()
    runs = plan_runs()
    for entry in fits + refits:
        entry["present"] = fit_present(out / "fits", entry)
    solves = sum(run["units"] or 0 for run in runs)
    return dict(statement=STATEMENT, candidates=candidate_specs(), budget=rounds.method_spec()["budget"], removals=list(REMOVALS),
                deployments={dataset: list(modes) for dataset, modes in rounds.MODES.items()}, parts={d: list(rounds.held_parts(d)) for d in rounds.DEPLOYMENTS},
                fits=fits, refits=refits, runs=runs, solves=dict(per_candidate=solves // len(CANDIDATES), total=solves,
                                                                complete=all(run["units"] is not None for run in runs)),
                inputs=inputs(), out=relative(out))


def print_plan(result):
    print(f"candidates at budget factor {result['budget']}: " + "; ".join(f"{name} {spec}" for name, spec in result["candidates"].items()))
    print(f"{len(result['fits'])} fits of the round's link model, {len(result['refits'])} refits, under {result['out']}/fits")
    for entry in result["fits"]:
        print(f"  fit   {record_label(entry['dataset'], entry['part'], entry['environment'])}: {entry['file']}, on {entry['fitted_on'].split(':')[0]}, "
              f"positions {entry['positions']}, {'present' if entry['present'] else 'missing'}")
    for entry in result["refits"]:
        print(f"  refit {record_label(entry['dataset'], entry['part'], entry['environment'])} without {entry['removal']}: {entry['file']}, "
              f"{'present' if entry['present'] else 'missing'}")
    print(f"{len(result['runs'])} runs, readings 1 and 10, every one positioned with fits that did not see it")
    for run in result["runs"]:
        where = f"scan {run['held']:02d}" if run["dataset"] == "bern" else f"fold {run['held']}"
        scenarios = run["scenarios"]
        what = (f"{scenarios.get('scenarios')} scenarios of {scenarios.get('corpus')}" if run["dataset"] == "bern"
                else f"blocks {scenarios.get('blocks')}" if run["dataset"] == "tiers"
                else f"acquisitions {', '.join(scenarios.get('acquisitions', []))}" if run["dataset"] == "bregar"
                else f"{len(scenarios.get('configurations', []))} configurations, {scenarios.get('panels')} panels of stride {scenarios.get('panel_stride')}")
        print(f"  {run['candidate']:13s} {run['dataset']:6s} {run['mode']:3s} {where}: {run['units']} units, {run['positioned_with']}, {what}")
    solves = result["solves"]
    print(f"{solves['per_candidate']} solves per candidate, {solves['total']} in all" + ("" if solves["complete"] else ", some inputs missing"))
    for entry in result["inputs"]:
        print(f"  {'present' if entry['present'] else 'MISSING'} {entry['path']}, {entry['input']} ({entry['written_by']})")


# ------------------------------------------------------------------------------------------- the fits
def fit_present(fits, entry):
    """Whether the fit or refit of a plan entry is written, a refit only from the fit file as it stands."""
    path = Path(fits) / entry["file"]
    if not path.exists():
        return False
    data = common.read_json(path)
    if "removal" not in entry:
        if entry["dataset"] == "bern":
            return str(entry["part"]) in data
        return entry["environment"] in data["environments"] if entry["dataset"] == "bregar" else True
    if not refit_current(fits, entry["dataset"], entry["part"], data):
        return False
    if entry["dataset"] == "bern":
        return "plain" in ((data.get("folds") or {}).get(str(entry["part"])) or {}).get(entry["removal"], {})
    holder = data["environments"].get(entry["environment"], {}) if entry["dataset"] == "bregar" else data["b10"] if entry["dataset"] == "idlab" else data
    return "plain" in holder.get(entry["removal"], {})


def bern_identity(held, coefficients, calibration_path, archive):
    """The identity of a Bern fold fit as fit_bern_mixed_blocks.py records it with its default settings."""
    import fit_bern_mixed_blocks as mixed
    import fit_layout
    import fit_links
    training = [scan for scan in fit_bern.SCAN_IDS if scan != held]
    return dict(held_scan=held or None, fit_scans=training, per_scan=BERN_BLOCKS["per_scan"], blocks_per_kind=BERN_BLOCKS["blocks_per_kind"],
                seed=BERN_BLOCKS["seed"], streams="one per scan, [seed, scan], and [seed, 0] for the picks", nu=MODEL["nu"],
                offset_shape=MODEL["offset_shape"], fit_early_detection=MODEL["fit_early_detection"], pooling=mixed.POOLING, strata="tracks",
                calibration=coefficients, calibration_sha256=common.sha256(calibration_path) if calibration_path else None,
                archive_sha256=common.sha256(archive), kinds=[f"{k[0]}/{k[1]}" for k in mixed.KINDS], evaluation_rows_used=0,
                likelihood=fit_links.likelihood_identity(),
                source={p.name: common.sha256(p) for p in (Path(mixed.__file__), Path(fit_links.__file__), Path(fit_bern.__file__), Path(fit_layout.__file__))})


def full_fit(dataset, part, environment, archive):
    """One record of the round's link model on one part, as fit_bern_mixed_blocks.py --held, fit_layout.py --fold or
    fit_idlab.py --fold writes it with their defaults, its starts in this process. Returns the record and, for a layout
    fit, the identity of its sources."""
    import fit_layout
    import fit_links
    import fit_reduced
    started = time.time()
    if dataset == "bern":
        import fit_bern_mixed_blocks as mixed
        coefficients, calibration_path = fit_bern.calibration(part)
        chosen, strata = mixed.collect(part, BERN_BLOCKS["per_scan"], BERN_BLOCKS["blocks_per_kind"], BERN_BLOCKS["seed"], archive)
        weights = fit_links.stratified_weights(strata)
        fit = fit_links.fit_link_model(chosen, weights, workers=1, **MODEL)
        residuals = np.concatenate([block[0] - block[2] for block in chosen])
        counts = {track: strata.count(track) for track in mixed.TRACKS}
        record = dict(identity=bern_identity(part, coefficients, calibration_path, archive), four=fit, blocks=len(chosen), blocks_per_track=counts,
                      weight_per_track=fit_links.stratum_totals(strata, weights), blocks_per_length=fit_layout.packet_counts(chosen),
                      residual_quantiles=np.percentile(residuals, [5, 25, 50, 75, 95]).round(1).tolist())
        record["seconds"] = time.time() - started
        return record, None
    if dataset == "idlab":
        blocks, weights, strata = fit_reduced.idlab_blocks(fold=part)
        fit = fit_links.fit_link_model(blocks, weights, workers=1, **MODEL)
        counts = {str(track): strata.count(track) for track in sorted(set(strata))}
        return dict(four=fit, blocks=len(blocks), blocks_per_track=counts, weight_per_track=fit_links.stratum_totals(strata, weights),
                    fold=folds.record("idlab", part)), None
    if dataset == "tiers":
        groups, source = fit_layout.tiers_groups(part)
        return dict(fit_layout.fit_groups(groups, LAYOUT_BLOCKS, fit_layout.SEEDS["tiers"], MODEL, 1), fold=folds.record("tiers", part)), source
    groups, source = fit_layout.bregar_groups(part)
    seed = fit_layout.SEEDS["bregar"] + int(environment[-1])
    return dict(fit_layout.fit_groups(groups[environment], LAYOUT_BLOCKS, seed, MODEL, 1), fold=folds.record("bregar", part)), source


def refit_inputs(fits, dataset, part, environment, archive):
    """The blocks, weights and full fit of one record, drawn again as fit_reduced.py draws them and verified against
    the record: the blocks per track and the negative log likelihood at the record's parameters within 1e-9."""
    import fit_layout
    import fit_reduced
    fits = Path(fits)
    label = record_label(dataset, part, environment)
    if dataset == "bern":
        record = common.read_json(fits / fit_file("bern", part))[str(part)]
        _, blocks, weights = fit_reduced.collect_bern(part, record["identity"], archive)
        if len(blocks) != record["blocks"]:
            raise ValueError(f"{label}: the reconstructed blocks differ from the fit record")
        fit_reduced.verify(blocks, weights, record["four"], label)
        full = record["four"]
    elif dataset == "idlab":
        record = common.read_json(fits / fit_file("idlab", part))["b10"]
        fit_reduced.fold_of("idlab", record, part, label)
        blocks, weights, _ = fit_reduced.idlab_set(record, label)
        full = record["four"]
    elif dataset == "tiers":
        record = common.read_json(fits / fit_file("tiers", part))
        groups, _ = fit_layout.tiers_groups(fit_reduced.fold_of("tiers", record, part, label))
        blocks, weights, full = fit_reduced.layout_set(groups, record, fit_layout.SEEDS["tiers"], label)
    else:
        record = common.read_json(fits / fit_file("bregar", part))["environments"][environment]
        groups, _ = fit_layout.bregar_groups(part)
        fit_reduced.fold_of("bregar", record, part, label)
        blocks, weights, full = fit_reduced.layout_set(groups[environment], record, fit_layout.SEEDS["bregar"] + int(environment[-1]), label)
    return blocks, weights, full


def refit_job(fits, dataset, part, environment, removal, archive):
    """The refit of one record without one part of the link model, on the blocks and weights of the record drawn again
    and verified against it (refit_inputs), with fit_reduced.py's checks of the result."""
    import fit_links
    import fit_reduced
    label = record_label(dataset, part, environment)
    blocks, weights, full = refit_inputs(fits, dataset, part, environment, archive)
    _, _, _, fit = fit_reduced.run_fit(dataset, environment, removal, blocks, weights, full)
    fit_reduced.check_settings(full, fit, removal, label)
    if removal in fit_links.NESTED:
        fit["nested_check"] = fit_reduced.nested_check(blocks, weights, full, fit, removal, label)
    fit["nll_difference_to_full"] = fit["nll"] - full["nll"]
    return fit


def reduced_identity(dataset, part):
    """The identity of a refit file as fit_reduced.py writes it, of the removals of the table."""
    import fit_layout
    import fit_links
    import fit_reduced
    identity = dict(variants=list(REMOVALS), nested=list(fit_links.NESTED), gaussian_nu=fit_links.GAUSSIAN_NU, nested_tolerance_nats=fit_reduced.NESTED_TOLERANCE,
                    likelihood=fit_links.likelihood_identity(), split_record_sha256=common.sha256(layout_split.RECORD), evaluation_rows_used=0,
                    source={p.name: common.sha256(p) for p in (Path(fit_reduced.__file__), Path(fit_links.__file__), Path(fit_layout.__file__), Path(__file__))})
    if dataset != "bern":
        identity["fold"] = part
    return identity


def write_full(fits, dataset, part, environment, record, source):
    """A full fit into its file, beside the records already there."""
    path = Path(fits) / fit_file(dataset, part)
    if dataset == "bern":
        data = common.read_json(path) if path.exists() else {}
        data[str(part)] = record
        data = {key: data[key] for key in sorted(data, key=int)}
    elif dataset == "bregar":
        data = common.read_json(path) if path.exists() else dict(environments={}, identity=source)
        data["environments"][environment] = record
        data["environments"] = dict(sorted(data["environments"].items()))
    elif dataset == "idlab":
        training = common.read_json(common.EXTERNAL / "idlab-training.json")
        data = dict(b10=record, scale_cm=training["profiles"]["scale_cm"], correction=training["correction"])
    else:
        data = dict(record, identity=source)
    common.write_json(path, data)


def refit_current(fits, dataset, part, data):
    """Whether a refit file holds the refits of the fit file as it stands, by the SHA-256 it names of it as
    fit_reduced.py names it, the Bern file among its sources."""
    digest = common.sha256(Path(fits) / fit_file(dataset, part))
    return ((data.get("sources") or {}).get(fit_file(dataset, part)) if dataset == "bern" else data.get("source_sha256")) == digest


def write_refit(fits, dataset, part, environment, removal, fit):
    """A refit into its file under fits/reduced, in the layout of fit_reduced.py, beside the refits already there of
    the same fit file. Refits of another fit file are dropped."""
    fits = Path(fits)
    path = fits / "reduced" / fit_file(dataset, part)
    digest = common.sha256(fits / fit_file(dataset, part))
    data = common.read_json(path) if path.exists() else None
    if data is None or not refit_current(fits, dataset, part, data):
        data = dict(reduced_identity(dataset, part))
    if dataset == "bern":
        data.setdefault("folds", {}).setdefault(str(part), {})[removal] = dict(plain=fit)
        data["folds"] = {key: data["folds"][key] for key in sorted(data["folds"], key=int)}
    elif dataset == "bregar":
        data.setdefault("environments", {}).setdefault(environment, {})[removal] = dict(plain=fit)
        data["environments"] = dict(sorted(data["environments"].items()))
    elif dataset == "idlab":
        data.setdefault("b10", {})[removal] = dict(plain=fit)
    else:
        data[removal] = dict(plain=fit)
    if dataset == "bern":
        data["sources"] = {fit_file(dataset, part): digest}
    else:
        data["source_sha256"] = digest
    common.write_json(path, data)


def unconverged(fits):
    """The fits and refits of the folder that did not converge, which no run takes."""
    result = []
    full, refits = plan_fits()
    for entry in full + refits:
        if not fit_present(fits, entry):
            continue
        data = common.read_json(Path(fits) / entry["file"])
        if "removal" in entry:
            holder = (data["folds"][str(entry["part"])] if entry["dataset"] == "bern" else data["environments"][entry["environment"]]
                      if entry["dataset"] == "bregar" else data["b10"] if entry["dataset"] == "idlab" else data)
            fit = holder[entry["removal"]]["plain"]
        else:
            fit = (data[str(entry["part"])] if entry["dataset"] == "bern" else data["environments"][entry["environment"]]
                   if entry["dataset"] == "bregar" else data["b10"] if entry["dataset"] == "idlab" else data)["four"]
        if not fit.get("converged"):
            result.append(record_label(entry["dataset"], entry["part"], entry["environment"]) + (f" without {entry['removal']}" if "removal" in entry else ""))
    return result


def execute(jobs, processes, function, done):
    """Run jobs, (key, arguments), in that many processes, or in this one for one, and hand every result to done."""
    if processes <= 1:
        for key, arguments in jobs:
            done(key, function(*arguments))
        return
    with ProcessPoolExecutor(max_workers=processes) as pool:
        futures = {pool.submit(function, *arguments): key for key, arguments in jobs}
        for future in as_completed(futures):
            done(futures[future], future.result())


def fit_stage(out, datasets, removals, processes, full_only=False, parts=None, archive=None):
    """The fits of the round's link model on every part, or on the held scans and folds of parts, then their refits
    without each removal, each written as it completes and kept when already written."""
    fits = out / "fits"
    archive = Path(archive or rounds.cooperative.ARCHIVE)
    full, refits = plan_fits()
    jobs = [((e["dataset"], e["part"], e["environment"]), (e["dataset"], e["part"], e["environment"], archive))
            for e in full if e["dataset"] in datasets and (parts is None or e["part"] in parts) and not fit_present(fits, e)]
    started = time.time()

    def full_done(key, result):
        record, source = result
        write_full(fits, *key, record, source)
        print(f"fitted {record_label(*key)}: " + str({k: round(record['four'][k], 4) for k in ('kappa', 'excess_probability', 'excess_mean_rel',
                                                                                                'persistent_sd_rel')})
              + f", converged {record['four']['converged']}, after {time.time() - started:.0f}s", flush=True)
    print(f"{len(jobs)} fits of the round's link model to fit", flush=True)
    execute(jobs, processes, full_fit, full_done)
    if full_only:
        return
    jobs = [((e["dataset"], e["part"], e["environment"], e["removal"]), (fits, e["dataset"], e["part"], e["environment"], e["removal"], archive))
            for e in refits if e["dataset"] in datasets and e["removal"] in removals and (parts is None or e["part"] in parts)
            and not fit_present(fits, e)]

    def refit_done(key, fit):
        write_refit(fits, *key, fit)
        print(f"refitted {record_label(*key[:3])} without {key[3]}: NLL {fit['nll_difference_to_full']:+.3f} to the full fit, converged "
              f"{fit['converged']}, {fit['seconds']:.0f}s", flush=True)
    print(f"{len(jobs)} refits to fit", flush=True)
    execute(jobs, processes, refit_job, refit_done)
    left = unconverged(fits)
    if left:
        print(f"WARNING {len(left)} fits did not converge, and the runs refuse them: {', '.join(left)}. The round fits such a fit once more from "
              "its declared start with twice the iteration cap (restart_fits.py of the research record); every fit of the paper converged.",
              file=sys.stderr, flush=True)


# ------------------------------------------------------------------------------------------- the runs
def runner(out, workers=12, solve=True):
    left = unconverged(out / "fits")
    if left:
        raise SystemExit(f"{len(left)} fits under {out / 'fits'} did not converge: {', '.join(left)}")
    return rounds.Runner(out, out / "fits", workers, solve)


def write_index(out, run):
    """The runs of every candidate with their scores' hashes and profile fingerprints, beside the earlier ones."""
    path = out / "runs/index.json"
    index = common.read_json(path) if path.exists() else {}
    for name, entry in run.index.items():
        index.setdefault(name, dict(spec=entry["spec"], runs={}))["runs"].update(entry["runs"])
    common.write_json(path, index)


def run_stage(out, datasets, names, workers):
    """Position every candidate on every part of the datasets, every run cached by its profile and inputs."""
    run = runner(out, workers)
    specs = candidate_specs()
    try:
        for name in names:
            for dataset in datasets:
                for mode in rounds.MODES[dataset]:
                    run.rows(name, specs[name], dataset, mode)
    finally:
        write_index(out, run)


# ------------------------------------------------------------------------------------------- the baselines
def bern_baseline_runs(scan_workers, scorer):
    """The runs of the baselines of Table II on the eleven held Bern scans (baselines_bern_training.run), each method
    with its final option of the mode and ADE-BP with its fold fits. They are the script's own runs below its folder,
    each cached by its identity, and no freeze is written or read: baselines_bern_training.py run and --freeze-only keep
    the freeze of their four methods, and either reuses the runs of the other."""
    import baselines_bern_training as bern_training
    for scan in fit_bern.SCAN_IDS:
        scenarios = bern_training.fold_archive(scan)
        bern_training.fold_config(scan)
        anchor_scenarios = bern_training.CAMPAIGN / f"scan-{scan:02d}/calibrated.zip"
        for mode, methods in BASELINES["bern"].items():
            for method in methods:
                bern_training.run(scan, method, mode.lower(), scan_workers, anchor_scenarios if mode == "ANC" else scenarios, scorer)
                print(f"bern scan {scan} {method} {mode} scored", flush=True)


def bern_run_folder(method, mode, scan):
    """The folder of a baseline's run on a held Bern scan, named as baselines_bern_training.run names it."""
    import baseline_selection
    import baselines_bern_training as bern_training
    candidate = bern_training.selected(method, mode) if baseline_selection.selects(method, mode.upper()) else None
    return bern_training.WORK / f"scan-{scan:02d}" / (f"{method}-{mode}" if candidate is None else f"{method}-{mode}-{candidate['name']}")


def gather_bern(out):
    """The rows of the baselines' runs on the eleven held Bern scans per mode, in the layout of common.write_run:
    every unit keyed by its scan and its number in the scan."""
    import scoring
    for upper, methods in BASELINES["bern"].items():
        mode, rows = upper.lower(), []
        for method in methods:
            for scan in fit_bern.SCAN_IDS:
                directory = bern_run_folder(method, mode, scan)
                outputs = [directory / "scenario_scores.csv.gz", directory / "joined_scores.csv.gz"]
                manifest = common.read_json(directory / "manifest.json")["identity"]
                if not common.cache_matches(directory / "manifest.json", manifest, outputs):
                    raise ValueError(f"{directory} does not hold the outputs of its run")
                timing = common.read_json(directory / "timing.json")
                offset = 0 if mode == "anc" else scan * 10000
                aborted = rounds.numbered({(a["scenario"], a["iteration"], a["samples"]): a["failure"] for a in timing.get("aborted_solves") or []},
                                          offset, scan)
                units = rounds.numbered(scoring.read_scenario_units(outputs[1]), offset, scan)
                rows += rounds.bern_rows(scan, units, rounds.numbered(rounds.penalized(outputs[0], mode), offset, scan), aborted, method)
        common.write_json(out / "baselines/bern" / mode / "scores.json", rows, indent=None)
        print(f"bern {mode}: {', '.join(methods)} gathered, {len(rows)} rows", flush=True)


def bregar_ade(out, workers):
    """ADE-BP alone on the 48 training units of Bregar with the candidate and the error law of each environment from
    the shipped settings, as bregar_layout.py evaluate runs it beside the other baselines and its selection ran it
    alone, scored with bregar_layout.score."""
    import bregar_layout
    settled = {environment: bregar_layout.settled(environment) for environment in bregar_layout.ENVIRONMENTS}
    request = bregar_layout.half_request("training")
    reference = bregar_layout.read(bregar_layout.OUTPUT / "training" / "reference.json")
    units = bregar_layout.with_settings(request["units"], lambda environment: {"ade-bp-2015": settled[environment]["ade-bp-2015"]})
    folder = out / "baselines/bregar"
    predictions = bregar_layout.run_adapter(units, ["ade-bp-2015"], workers, folder / "work")
    scores = bregar_layout.score(dict(units=units), reference, predictions, methods=("ade-bp-2015",))
    (folder / "predictions.jsonl.gz").write_bytes(gzip.compress(b"".join(bregar_layout.encoded(r) + b"\n" for r in predictions), mtime=0))
    (folder / "scores.jsonl.gz").write_bytes(gzip.compress(b"".join(bregar_layout.encoded(r) + b"\n" for r in scores), mtime=0))
    common.write_json(folder / "run.json", dict(method="ade-bp-2015", split="training", units=len(units), workers=workers,
                                                options={environment: settled[environment]["ade-bp-2015"] for environment in bregar_layout.ENVIRONMENTS},
                                                options_sha256=common.sha256(common.ROOT / "settings/baseline-options.json")))
    print(f"bregar: ADE-BP on {len(units)} training units scored", flush=True)


def baselines_stage(out, datasets, workers):
    """The training runs of the baselines that the last column reads."""
    datasets_folder = common.ROOT / "research/benchmarks/datasets"
    if "bern" in datasets:
        bern_baseline_runs(workers, common.load_scorer())
        gather_bern(out)
    if "tiers" in datasets:
        for mode in ("af", "aac", "anc"):
            subprocess.run([sys.executable, str(datasets_folder / "layout_baselines.py"), "tiers", "run", "--split", "training", "--mode", mode,
                            "--workers", str(workers)], check=True)
    if "idlab" in datasets:
        modes = ",".join(mode.lower() for mode in BASELINES["idlab"])
        methods = ",".join(sorted({m for methods in BASELINES["idlab"].values() for m in methods}))
        subprocess.run([sys.executable, str(datasets_folder / "idlab_layout_baselines.py"), "run", "--split", "training", "--methods", methods,
                        "--modes", modes, "--workers", str(workers)], check=True)
    if "bregar" in datasets:
        bregar_ade(out, workers)


def baseline_rows(out, dataset, mode):
    """The score rows of the baselines' training runs of one dataset and mode, and the file they come from."""
    if dataset == "bern":
        path = out / "baselines/bern" / mode.lower() / "scores.json"
        return json.loads(path.read_text()), path
    if dataset == "bregar":
        path = out / "baselines/bregar/scores.jsonl.gz"
        return [json.loads(line) for line in gzip.open(path, "rt")], path
    path = common.ROOT / "out/baselines" / dataset / "training" / mode.lower() / "scores.json"
    return json.loads(path.read_text()), path


# ------------------------------------------------------------------------------------------- the scores
def removal_rows(records):
    """Per (dataset, mode, track) the method's S75 and every removal's S75, difference and standard error, over the
    headline cells of the paper's tables, from the decision record of every removal (ablation_table.removals)."""
    rows = {}
    for name in REMOVALS:
        for c in records[name]["comparisons"]:
            if c["level"] != "S75" or (c["dataset"] in SCOPED and c["mode"] == "anc" and c.get("scope") != "anchors>=3"):
                continue
            if c["released_reference_cm"] != c["reference_cm"] or c["released_candidate_cm"] != c["candidate_cm"]:
                raise SystemExit(f"the released and exact values of {name} differ in {c['dataset']} {c['mode']} {c['track']}")
            key = (c["dataset"], c["mode"].upper(), int(c["track"]))
            row = rows.setdefault(key, dict(solid_cm=c["reference_cm"], units=c["units"], cells=c["cells"], removals={}))
            if (row["solid_cm"], row["units"], row["cells"]) != (c["reference_cm"], c["units"], c["cells"]):
                raise SystemExit(f"the removals of {key} compare against different references")
            row["removals"][name] = dict(cm=c["candidate_cm"], difference_cm=c["difference_cm"], se_cm=c["se_cm"])
    return rows


def headline(row, mode):
    """Whether a row of a percentile or difference table is the headline of its mode under the joined pairs."""
    return row["pair_set"] == "joined" and (row["group"] == "every cell" if mode != "ANC" else "3 or more" in row["group"])


def table_ii(tables, dataset):
    """The evaluation tables of a dataset that Table II prints, those of final_tables.py build under tables, None when
    the folder does not hold them."""
    path = Path(tables) / dataset / "evaluation/percentiles.csv"
    return path if path.exists() else None


def table_best(tables, dataset, mode, track):
    """The baseline of least S75 in a row of Table II, the headline of the evaluation half, from the tables of
    final_tables.py build, or as the paper prints it (TABLE_II) when the tables folder does not hold the dataset."""
    path = table_ii(tables, dataset)
    if path is None:
        return TABLE_II[(dataset, mode, track)]
    held = [r for r in csv.DictReader(open(path)) if r["mode"] == mode and int(r["track"]) == track and headline(r, mode) and r["method"] != "solid"]
    return min(held, key=lambda r: float(r["S75_cm"]))["method"]


def in_sweep_stride(panel):
    """Whether a sweep is one of the record's, its key hashing to class zero modulo the stride (run_idlab.in_panel_stride)."""
    return run_idlab.in_panel_stride(panel, rounds.PARAMETERS["idlab_panel_stride"])


def balanced(rows, percentile):
    """The released balanced S75 of score rows over the joined pairs, the cells in the order of the sorted units."""
    grouped = {}
    for r in sorted(rows, key=lambda r: (r["key"], r["id"], r["samples"])):
        joined = r["pair_sets"]["joined"]
        if joined["eligible_pairs"] > 0:
            grouped.setdefault(decision.plain(r["cell"]), []).append(float(joined["mean_pair_error_cm"]))
    return float(percentile(grouped, LEVEL)), sum(len(v) for v in grouped.values())


def interval(reference, candidate, dataset, mode):
    """SOLID minus a baseline at S75 with its paired 95 percent interval, from decision.compare's draws."""
    pairs = {(dataset, mode.lower()): (decision.external_rows(dataset, reference, pair_set="joined"),
                                       decision.external_rows(dataset, candidate, pair_set="joined"))}
    out = {}
    for c in decision.compare(pairs, levels=(("S75", "0.75"),)):
        low, high = np.quantile(-c["replicates"], [0.025, 0.975], method="inverted_cdf")
        out[int(c["track"])] = dict(difference_cm=-c["difference_cm"], interval_cm=[float(low), float(high)])
    return out


def tiers_lead(solid, base, mode, track, method, percentile):
    """SOLID minus a baseline at S75 with its interval as the tables of the TIERS training half give it, the bootstrap
    of final_tables.analyse over the three modes of the half with the declared design, draws and seed."""
    import final_tables
    declared = final_tables.declaration()
    spec = declared["datasets"]["tiers"]
    design = decision.DESIGNS[spec["design"]["name"]]
    modes = {}
    for upper in MODES:
        units = final_tables.Units("tiers", upper, spec["pair_sets"], "solid", final_tables.row_entries(solid[upper], design, spec["pair_sets"]))
        if upper == mode:
            units.add(method, final_tables.row_entries([r for r in base if r["method"] == method], design, spec["pair_sets"]))
        modes[upper] = units
    rosters = {upper: ["solid"] + ([method] if upper == mode else []) for upper in modes}
    sections = final_tables.analyse(modes, rosters, {}, spec["pair_sets"], spec["design"]["name"], design, spec["replicates"], spec["seed"],
                                    declared["percentiles"], declared["interval"], declared["groups"]["minimum_anchors"], declared["tracks"], percentile)
    for group, entry in sections[mode]["groups"].items():
        if headline(dict(pair_set="joined", group=group), mode):
            value = entry["tracks"][str(track)]["pair_sets"]["joined"]["differences"][f"solid minus {method}"]["S75"]
            return dict(difference_cm=value["value_cm"], interval_cm=list(value["interval_cm"]))
    raise SystemExit(f"the TIERS tables hold no headline of {mode}")


def tiers_recordings(solid, base, percentile):
    """SOLID and ADE-BP at S75 per recording in the headline of ANC on the training part of TIERS."""
    base = [r for r in base if r["method"] == "ade-bp-2015"]
    out = []
    for track in (1, 10):
        for label, held in (("army and corner", ("army", "corner")), ("diagonal", ("diagonal",))):
            keep = lambda r: r["samples"] == track and r["anchors"] >= 3 and r["acquisition"] in held  # noqa: E731
            out.append(dict(track=track, recordings=label, solid_S75_cm=balanced([r for r in solid if keep(r)], percentile)[0],
                            ade_bp_S75_cm=balanced([r for r in base if keep(r)], percentile)[0]))
    return out


def score_rows(rows, raw, baselines, tables, percentile, scorer, datasets, sources=None):
    """The record's rows, leads and recordings, and the decision of every removal, from the unit rows of every candidate
    {name: {(dataset, mode): rows}}, the score rows of SOLID's runs of TIERS, Bregar and IDLab {(dataset, MODE): rows},
    the baselines' score rows {(dataset, MODE): rows} with the files they come from, sources, and the folder of the
    tables of Table II."""
    specs = candidate_specs()
    records = {}
    for name in REMOVALS:
        pairs = {(dataset, mode): (rows["reference"][(dataset, mode)], rows[name][(dataset, mode)])
                 for dataset in rounds.DEPLOYMENTS if dataset in datasets for mode in rounds.MODES[dataset]}
        records[name] = rounds.judged(pairs, ("R**", specs["reference"]), (name, specs[name]), rounds.MODES["bern"], "remove", scorer)
        print(f"{name}: {len(records[name]['comparisons'])} comparisons", flush=True)
    table = removal_rows(records)
    leads = []
    for (dataset, mode, track), row in table.items():
        best = table_best(tables, dataset, mode, track)
        row["best"] = dict(method=best)
        if (dataset, mode, best) in NO_RUN:
            row["best"]["missing"] = NO_RUN[(dataset, mode, best)]
            continue
        if best not in BASELINES[dataset].get(mode, ()) and dataset != "tiers" or mode not in BASELINES[dataset]:
            raise SystemExit(f"Table II names {best} best in {dataset} {mode} with {track} readings, which has no training run here")
        base = [r for r in baselines[(dataset, mode)] if r.get("method", best) == best]
        if dataset == "idlab":
            base = [r for r in base if in_sweep_stride(r["panel"])]
        if dataset == "bregar":
            solid = raw[("bregar", "ANC")]
            if {(r["key"], r["id"], r["samples"]): r["cell"] for r in solid} != {(r["key"], r["id"], r["samples"]): r["cell"] for r in base}:
                raise SystemExit("ADE-BP's Bregar run solved other scenarios than SOLID's")
        value = balanced([r for r in base if r["samples"] == track and (mode != "ANC" or r["anchors"] >= 3)], percentile)[0]
        row["best"].update(cm=value, source=(sources or {}).get((dataset, mode), STATEMENT))
        if value <= row["solid_cm"]:
            if dataset in ("bregar", "idlab"):
                scoped = lambda rs: [r for r in rs if mode != "ANC" or r["anchors"] >= 3]  # noqa: E731
                drawn = interval(scoped(raw[(dataset, mode)]), scoped(base), dataset, mode)[track]
            elif dataset == "tiers":
                drawn = tiers_lead({upper: raw[("tiers", upper)] for upper in MODES}, base, mode, track, best, percentile)
            else:
                raise SystemExit("a baseline leads SOLID on the held Bern scans, whose interval the tables of the gate gave, which this driver does "
                                 "not carry")
            leads.append(dict(dataset=dataset, mode=mode, track=track, baseline=best, **drawn))
    order = [d for d, _ in DATASETS]
    result = dict(rows=[dict(dataset=k[0], mode=k[1], track=k[2], **row) for k, row in sorted(table.items(), key=lambda kv: (
                      order.index(kv[0][0]), MODES.index(kv[0][1]), kv[0][2]))],
                  leads=sorted(leads, key=lambda l: (order.index(l["dataset"]), MODES.index(l["mode"]), l["track"])))
    if "tiers" in datasets:
        result["tiers_recordings"] = tiers_recordings(raw[("tiers", "ANC")], baselines[("tiers", "ANC")], percentile)
    return result, records


def score_stage(out, tables, datasets):
    """The decisions and the baselines' values from the stored runs, written to the record ablation.json and
    decisions.json. It solves nothing."""
    run = runner(out, solve=False)
    specs = candidate_specs()
    rows = {name: {(dataset, mode): run.rows(name, specs[name], dataset, mode) for dataset in rounds.DEPLOYMENTS if dataset in datasets
                   for mode in rounds.MODES[dataset]} for name in CANDIDATES}
    raw = {(dataset, mode.upper()): run.raw[("reference", dataset, mode)] for dataset in ("tiers", "bregar", "idlab") if dataset in datasets
           for mode in rounds.MODES[dataset]}
    baselines, files, named = {}, {}, {}
    for dataset in datasets:
        for mode in rounds.MODES[dataset]:
            if dataset in BASELINES and mode.upper() in BASELINES[dataset]:
                baselines[(dataset, mode.upper())], path = baseline_rows(out, dataset, mode)
                files[relative(path)] = sha256(path)
                named[(dataset, mode.upper())] = relative(path)
    percentile = run.released_scorer().balanced_percentile
    result, records = score_rows(rows, raw, baselines, tables, percentile, run.released_scorer(), datasets, named)
    write_index(out, run)
    sources = {relative(path): sha256(path) for path in [common.ROOT / "research/solid/method_constants.py", out / "runs/index.json"]
               + sorted((out / "fits").rglob("*.json"))}
    sources.update(files)
    sources.update({relative(table_ii(tables, d)): sha256(table_ii(tables, d)) for d in datasets if table_ii(tables, d)})
    choices = {d: relative(table_ii(tables, d)) if table_ii(tables, d) else "the paper's Table II, ablation.TABLE_II" for d in datasets}
    record = dict(schema=SCHEMA, statement=STATEMENT, sources=sources, removals=list(REMOVALS), level="S75",
                  sweep_stride=rounds.PARAMETERS["idlab_panel_stride"], **result, table_ii=choices)
    common.write_json(out / "decisions.json", records)
    (out / "ablation.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {relative(out / 'ablation.json')} and {relative(out / 'decisions.json')}", flush=True)
    return record


# ------------------------------------------------------------------------------------------- the table
def cell(value, se=None):
    text = f"{value:.1f}"
    return text if se is None else f"{text} ({se:.1f})"


def latex(rows):
    # the full text width with the gaps stretched evenly, so that the one-line note widens no single column
    lines = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcrrrrl@{}}", r"\hline",
             r"Dataset & Mode & Readings & SOLID & No heavy tail & No excess & No offset & Best baseline of Table~II \\",
             r"\hline"]
    for dataset, label in DATASETS:
        keys = [k for k in sorted(rows, key=lambda k: (MODES.index(k[1]), k[2])) if k[0] == dataset]
        for i, key in enumerate(keys):
            row = rows[key]
            first_of_mode = i == 0 or keys[i - 1][1] != key[1]
            cells = [label if i == 0 else "", key[1] if first_of_mode else "", str(key[2]), cell(row["solid_cm"])]
            cells += [cell(row["removals"][n]["cm"], row["removals"][n]["se_cm"]) for n in REMOVALS]
            best = row["best"]
            cells.append(f"{NAMES[best['method']]} {cell(best['cm'])}" if best.get("cm") is not None else f"{NAMES[best['method']]} --")
            lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\hline")
    dash = " and a dash where no full run used the evaluated options" if any(row["best"].get("cm") is None for row in rows.values()) else ""
    # the note takes no width of its own, so the columns keep their even gaps and the note stays within the text width
    lines += [r"\multicolumn{8}{@{}l@{}}{\rule{0pt}{2.4ex}\makebox[0pt][l]{\scriptsize S75 in cm, in ANC over three or more anchors, with the standard error of the difference from SOLID in parentheses" + dash + r".}}\\",
              r"\end{tabular*}"]
    return lines


def table_stage(record_path, latex_path):
    """Table S-XVII from a record of the score, the rows it holds."""
    record = json.loads(Path(record_path).read_text())
    rows = {(r["dataset"], r["mode"], r["track"]): r for r in record["rows"]}
    table = latex(rows)
    Path(latex_path).parent.mkdir(parents=True, exist_ok=True)
    Path(latex_path).write_text("\n".join(table) + "\n")
    print("\n".join(table))
    print(f"wrote {relative(latex_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "fit", "run", "baselines", "score", "table"))
    parser.add_argument("--out", type=Path, default=OUT, help="the folder of the stages, out/ablation of the checkout by default")
    parser.add_argument("--datasets", default=",".join(rounds.DEPLOYMENTS))
    parser.add_argument("--removals", default=",".join(REMOVALS), help="the removals fit refits")
    parser.add_argument("--candidates", default=",".join(CANDIDATES), help="the candidates run positions, reference the method")
    parser.add_argument("--full-only", action="store_true", help="fit the round's link model on the parts without refitting it")
    parser.add_argument("--parts", default=None, help="the held Bern scans and the folds fit fits, every one by default, for example 1 or 0,1")
    parser.add_argument("--workers", type=int, default=12, help="threads of a run, processes of a baseline run, or the fits of fit at once, "
                        "each in a process of its own, one fitting in this process")
    parser.add_argument("--tables", type=Path, default=common.ROOT / "research/artifacts/solid/tables", help="the tables of Table II")
    parser.add_argument("--record", type=Path, default=None, help="the record the table is written from, ablation.json under --out by default")
    parser.add_argument("--latex", type=Path, default=None, help="the table, ablation-table.tex under --out by default")
    parser.add_argument("--json", type=Path, default=None, help="write the plan to this file as well")
    args = parser.parse_args()
    out = args.out.resolve()
    datasets = args.datasets.split(",")
    if set(datasets) - set(rounds.DEPLOYMENTS):
        parser.error(f"--datasets names others than {', '.join(rounds.DEPLOYMENTS)}")
    if args.command == "plan":
        result = plan(out)
        print_plan(result)
        if args.json:
            common.write_json(args.json, result)
    elif args.command == "fit":
        removals_ = args.removals.split(",")
        if set(removals_) - set(REMOVALS):
            parser.error(f"--removals names others than {', '.join(REMOVALS)}")
        fit_stage(out, datasets, removals_, args.workers, args.full_only, None if args.parts is None else [int(p) for p in args.parts.split(",")])
    elif args.command == "run":
        names = args.candidates.split(",")
        if set(names) - set(CANDIDATES):
            parser.error(f"--candidates names others than {', '.join(CANDIDATES)}")
        run_stage(out, datasets, names, args.workers)
    elif args.command == "baselines":
        baselines_stage(out, datasets, args.workers)
    elif args.command == "score":
        score_stage(out, args.tables, datasets)
    else:
        table_stage(args.record or out / "ablation.json", args.latex or out / "ablation-table.tex")


if __name__ == "__main__":
    main()
