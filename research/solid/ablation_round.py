"""The pieces that the ablation driver research/solid/ablation.py uses to position the method and each removal of
Table S-XVII on the training parts. The round, in the names of this module, is that set of runs: the method and every
removal on every held part.

They are the specs of the method and of its removals, the names of their fit files, the compositions of a candidate's
profile on every held part (compose and the functions it calls), the round's runner, which positions a candidate on
every part and caches every run by the content of its profile and inputs (Runner, bern_joined and the identities of the
runs), and the scoped pairs, the rule that decides which comparisons count (voting_rule) and the record of each
comparison (judged). They are the code of the runs behind Table S-XVII with these differences, none of which changes a
number:

- The fit files of the folds and their refits are read from a folder the caller names, out/ablation/fits for the
  driver, in place of research/artifacts/solid/fits. The Bern compositions of select_bern_cooperative.py and
  ablate_bern_anchor_only.py read the files a spec names under research/artifacts/solid/fits, so the specs here name
  them by absolute path, which common.FITS / path leaves as it is.
- The method's one switch, the maximum likelihood reach, is applied by profiles.switched. The removals of Table
  S-XVII switch nothing else.
- The freeze of those runs, the worker caps of their machine, the derived first stages and the engine's guarded
  supports, which the removals neither need nor change, are left out.

The score rows of the baselines' runs on the eleven held Bern scans, numbered, penalized and bern_rows, are those of
the paper's training records, unchanged.
"""
import csv
import gzip
import json
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate_bern_anchor_only as anchor_only  # noqa: E402
import common  # noqa: E402
import decision  # noqa: E402
import deployment  # noqa: E402
import fit_bern  # noqa: E402
import fit_reduced  # noqa: E402
import folds  # noqa: E402
import layout_split  # noqa: E402
import method_constants  # noqa: E402
import profiles  # noqa: E402
import run_bregar  # noqa: E402
import run_idlab  # noqa: E402
import run_tiers  # noqa: E402
import scoring  # noqa: E402
import select_bern_cooperative as cooperative  # noqa: E402

DEPLOYMENTS = ("bern", "tiers", "bregar", "idlab")
MODES = dict(bern=("af", "aac", "anc"), tiers=("af", "aac", "anc"), bregar=("anc",), idlab=("af", "aac", "anc"))
TRACKS = ("1", "10")
PAIR_SET = "joined"
PROFILE_NAMES = {"af": "agent-only", "aac": "anchor-aided", "anc": "anchor-only"}
# the fit files of every deployment, the suffix of a model variant inserted before .json
FIT_FILES = dict(bern="bern_mixed_fold_fits_nu3.json", tiers="tiers_layout_fit.json", bregar="bregar_layout_fits.json", idlab="idlab_link_fit.json")
# the cooperative first stage of IDLab on the reserved rows of each fold alone, written by fit_idlab.py --first-stage
IDLAB_FIRST_STAGE = "idlab_first_stage.json"
# the removals of the link model that Table S-XVII reports, each a refit (fit_links.VARIANTS)
REMOVALS = ("gaussian_core", "no_excess", "no_offset")
# the declared parameters of the round that its decisions read
PARAMETERS = dict(decision.PARAMETERS, deployments=list(DEPLOYMENTS), pair_set=PAIR_SET, anc_minimum_anchors=dict(tiers=3, idlab=3),
                  idlab_panel_stride=8, idlab_panel_key=run_idlab.PANEL_KEY)
DECISION_KEYS = tuple(decision.PARAMETERS)
# the deployment whose votes every decision record also leaves out, a diagnostic that no verdict reads
UNCROSSED = "bregar"


# ------------------------------------------------------------------------------------------- candidates
def spec(suffix="", reduced=(), changes=(), budget=1):
    """A candidate: the suffix of its fit files, the link components it refits without, the changes it makes, and the
    factor of its anchor only budgets over the deployment budgets, a power of two."""
    return dict(suffix=suffix, reduced=sorted(set(reduced)), changes=sorted(set(changes)), budget=deployment.budget_factor(budget))


def with_changes(base, reduced=(), changes=()):
    """A candidate built on another: its reduced components and changes added to the base's, with the base's budgets."""
    return spec(base["suffix"], set(base["reduced"]) | set(reduced), set(base["changes"]) | set(changes), base["budget"])


def method_spec():
    """The method, the reference of every removal: R0 with the maximum likelihood reach, at twice the converged
    budgets, the spec of research/solid/method_constants.py."""
    return spec(**method_constants.SPEC)


def removal_spec(name):
    """The method without one part of its link model, a refit of the link model without it."""
    if name not in REMOVALS:
        raise ValueError(f"the ablation removes one of {', '.join(REMOVALS)}, not {name}")
    return with_changes(method_spec(), reduced=[name])


# ------------------------------------------------------------------------------------------- fit files
def fit_name(dataset, suffix, fold=None):
    """The fit file of a deployment and model variant, of one fold of the round when a fold is given. Bern's file holds
    its own held-scan folds."""
    if fold is not None and dataset == "bern":
        raise ValueError("the Bern fit file holds its own held-scan folds")
    return FIT_FILES[dataset].replace(".json", f"{suffix}{folds.name(fold)}.json")


def held_parts(dataset):
    """The parts of a deployment on which the round composes and positions a candidate: every held Bern scan, and each
    fold of TIERS, Bregar and IDLab."""
    return fit_bern.SCAN_IDS if dataset == "bern" else folds.FOLDS


def fit_fold(held):
    """The fold whose fit positions the units of a held fold, the other one, and the whole training half for none."""
    return None if held is None else folds.other(held)


def reduced_name(dataset, suffix, components, fold=None):
    """The file under fits/reduced that holds a reduced model, fits/reduced/combined for two or more components, of one
    fold of the round when a fold is given."""
    return ("combined/" if len(components) > 1 else "") + fit_name(dataset, suffix, fold)


def external_record(fits, dataset, suffix, reduced=(), fold=None):
    """The fit record a profile composition of an external deployment takes, that of one fold of the round when a fold
    is given, with the link model of the reduced fit in place of the full one when the candidate refits without
    components. fits is the folder of the fit files."""
    fits = Path(fits)
    data = common.read_json(fits / fit_name(dataset, suffix, fold))
    record = data["b10"] if dataset == "idlab" else data["environments"] if dataset == "bregar" else data
    if not reduced:
        return record
    key = fit_reduced.combination_key(reduced)
    stored = common.read_json(fits / "reduced" / reduced_name(dataset, suffix, reduced, fold))
    if dataset == "bregar":
        return {environment: dict(entry, four=stored["environments"][environment][key]["plain"]) for environment, entry in record.items()}
    if dataset == "idlab":
        return dict(record, four=stored["b10"][key]["plain"])
    return dict(record, four=stored[key]["plain"])


# ------------------------------------------------------------------------------------------- profiles
def idlab_first_stage(fold):
    """The cooperative first stage of IDLab fitted on the reserved rows of one fold of the round alone, {mode:
    multipliers}, from fits/idlab_first_stage.json (fit_idlab.py --first-stage). A fold of another assignment is
    refused."""
    entry = common.read_json(common.FITS / IDLAB_FIRST_STAGE)["folds"][str(folds.check(fold))]
    if entry.get("fold") != folds.record("idlab", fold):
        raise ValueError(f"{IDLAB_FIRST_STAGE} holds the first stage of fold {fold} of another assignment than folds.py declares. Fit it again")
    return {mode: dict(positive_scale=entry[mode]["positive_scale"], negative_scale=entry[mode]["negative_scale"]) for mode in ("af", "aac")}


def external_profile(fits, dataset, mode, candidate, held=None):
    """The profile of a candidate's link model on an external deployment and mode, with the candidate's budgets and
    before its switches. For a held fold of the round it is composed from the fit of the other fold, so it positions
    the units of the held fold with a link model fitted without them. A cooperative profile of an IDLab fold takes the
    first stage fitted on the other fold's reserved rows in place of the campaign's. The deployment settings are those
    of the whole training half either way."""
    import copy
    record = external_record(fits, dataset, candidate["suffix"], candidate["reduced"], fit_fold(held))
    compositions = dict(tiers=profiles.tiers, idlab=profiles.idlab, bregar=profiles.bregar)
    composed = dict(compositions[dataset](record, budget=candidate["budget"]))
    profile = copy.deepcopy(composed[f"{dataset}-{PROFILE_NAMES[mode]}.json"])
    if dataset == "idlab" and held is not None and mode != "anc":
        multipliers = idlab_first_stage(fit_fold(held))[mode]
        for n in TRACKS:
            profile[n]["range_error"] = dict(multipliers)
    return profile


def bern_specs(fits, candidate, law):
    """The held-scan composition specs of one candidate: the fold fits of its suffix, its reduced link model, the
    method's presence law and the candidate's budgets, which the anchor only composition takes as a parameter. The fit
    files are named by absolute path in the folder fits, which the compositions of select_bern_cooperative.py and
    ablate_bern_anchor_only.py read as they are. The candidate's switches are applied after composition."""
    fits = Path(fits).resolve()
    fold_fits = str(fits / fit_name("bern", candidate["suffix"]))
    coop = dict(excess="fitted", starts=profiles.STARTS, fold_fits=fold_fits, overrides={"presence": dict(law)})
    anc = dict(fold_fits=fold_fits, presence=dict(law), budget=candidate["budget"])
    if candidate["reduced"]:
        reduced = dict(reduced=fit_reduced.combination_key(candidate["reduced"]),
                       reduced_file=str(fits / "reduced" / reduced_name("bern", candidate["suffix"], candidate["reduced"])))
        coop.update(reduced)
        anc.update(reduced)
    return {"candidate": coop}, {"candidate": anc}


def bern_profile(fits, mode, scan, candidate, law=None):
    """The profile of a candidate on one held Bern scan: the held scan's composition, the anchor only campaign's fold
    profile (ablate_bern_anchor_only.profile_for) or the cooperative fold profile
    (select_bern_cooperative.candidate_profile) as select_union_ablation.bern_profile chooses them, with the candidate's
    fit, switches and budgets."""
    law = law or profiles.presence_law()
    coop, anc = bern_specs(fits, candidate, law)
    profile = anchor_only.profile_for(scan, "candidate", anc) if mode == "anc" else cooperative.candidate_profile(mode, scan, "candidate", coop)
    return profiles.switched(profile, mode, candidate["changes"])


def compose(fits, dataset, mode, candidate, held):
    """The profile of a candidate on one deployment and mode that positions the units of `held`: a held Bern scan, or
    a fold of TIERS, Bregar or IDLab, composed from the fit of the other fold."""
    if dataset == "bern":
        return bern_profile(fits, mode, held, candidate)
    return profiles.switched(external_profile(fits, dataset, mode, candidate, held), mode, candidate["changes"])


# ------------------------------------------------------------------------------------------- the runs
def stored(manifest, profile, binary, scenarios_sha256=None):
    """Whether a Bern run directory holds the verified outputs of this profile, runner and, when given, scenarios."""
    try:
        record = common.read_json(manifest)
        identity = record["identity"]
        return (identity["profile"] == profile and identity["binary_sha256"] == common.sha256(binary)
                and (scenarios_sha256 is None or identity["scenarios_sha256"] == scenarios_sha256)
                and common.cache_matches(manifest, identity, [manifest.parent / name for name in record["outputs"]]))
    except (OSError, ValueError, KeyError):
        return False


# the campaign inputs of the external training halves the runners read, part of every run's key
EXTERNAL_INPUTS = dict(tiers=("tiers-static-campaign/base.json.gz", "tiers-static-campaign/catalog.json.gz", "tiers-static-campaign/freeze.json"),
                       bregar=("bregar-layout/training/request.json.gz", "bregar-layout/training/reference.json"),
                       idlab=("idlab-training-campaign/base.json.gz", "idlab-training-campaign/catalog.json.gz", "idlab-training-campaign/freeze.json"))


def external_inputs():
    """The hashes of the external campaign inputs of every deployment."""
    return {dataset: {name: common.sha256(common.EXTERNAL / name) for name in names} for dataset, names in EXTERNAL_INPUTS.items()}


def scoring_code(dataset):
    """The hashes of the code that builds an external run's requests from the campaign inputs and turns its predictions
    into per unit scores, with the deployment record that sets every anchor only unit's area."""
    runner = dict(tiers=run_tiers, bregar=run_bregar, idlab=run_idlab)[dataset]
    import pilot
    modules = [runner, common, scoring, pilot, deployment, layout_split]
    if dataset == "tiers":
        import tiers_campaign
        modules.append(tiers_campaign)
    elif dataset == "idlab":
        import idlab_campaign
        modules.append(idlab_campaign)
    code = {Path(module.__file__).name: common.sha256(Path(module.__file__)) for module in modules}
    return dict(code, **{"deployment record": common.sha256(deployment.RECORD)})


def external_identity(dataset, profile, out):
    """What an external run stands for: the profile, the runner binary, the code that builds and scores its requests,
    and the scores."""
    return dict(profile_fingerprint=common.fingerprint(profile), binary_sha256=common.sha256(common.BINARY), code=scoring_code(dataset),
                scores_sha256=common.sha256(out / "scores.json"))


def external_stored(dataset, out, profile):
    """Whether an external run directory holds the scores of this profile, runner and scoring code."""
    try:
        return common.read_json(out / "round.json") == external_identity(dataset, profile, out)
    except (OSError, ValueError, KeyError):
        return False


def bern_joined(directory, scenarios, predictions, mode, scorer):
    """The per unit scores of one Bern run under the joined pair set, from its stored predictions, cached beside them.

    The released error recomputed from the same predictions must equal the released scorer's file unit for unit, by
    scoring.stored_run, so the predictions and scenarios are those the run scored. A unit without a released pair has
    no released error on either side. Returns {(scenario, iteration, samples): (error, anchors, agents)}, the error None
    for a unit without a joined pair."""
    output, manifest = directory / "joined_scores.csv.gz", directory / "joined.manifest.json"
    identity = dict(predictions_sha256=common.sha256(predictions), scenarios_sha256=common.sha256(scenarios), mode=mode, pair_set=PAIR_SET,
                    scoring_sha256=common.sha256(Path(scoring.__file__)))
    if not common.cache_matches(manifest, identity, [output]):
        with tempfile.TemporaryDirectory(prefix="joined-", dir=directory) as temporary:
            with zipfile.ZipFile(predictions) as archive:
                archive.extract("predictions.csv", temporary)
            units = scoring.stored_run(scenarios, Path(temporary) / "predictions.csv", mode, scorer, directory / "scenario_scores.csv.gz")
        staged = output.with_name(output.name + ".partial")
        with gzip.open(staged, "wt", compresslevel=6) as stream:
            stream.write("scenario_id,iteration,samples,anchors,agents,released_pairs,released_error,joined_pairs,joined_error\n")
            for key in sorted(units):
                unit = units[key]
                values = [*key, unit["anchors"], unit["agents"], unit["released"][0], "" if unit["released"][1] is None else repr(unit["released"][1]),
                          unit["joined"][0], "" if unit["joined"][1] is None else repr(unit["joined"][1])]
                stream.write(",".join(str(v) for v in values) + "\n")
        staged.replace(output)
        common.seal_cache(manifest, identity, [output])
    with gzip.open(output, "rt") as stream:
        return {(int(r["scenario_id"]), int(r["iteration"]), int(r["samples"])): (float(r["joined_error"]) if r["joined_error"] else None,
                                                                                 int(r["anchors"]), int(r["agents"])) for r in csv.DictReader(stream)}


class Runner:
    """Positions candidates on the deployments of the round, every run cached by the content of its profile and inputs
    under work, the round's runner with the fit files of the folder fits.

    With solve false nothing is solved and a run that is not stored raises, for the scores of stored runs. IDLab keeps
    the panels whose key hashes to class zero modulo the round's panel stride, eight. Every external run keeps its
    score rows (raw), which the leads and the recordings of the table read beside the decisions."""

    def __init__(self, work, fits, workers=12, solve=True, scorer=None):
        self.work, self.fits, self.workers, self.solve = Path(work), Path(fits), workers, solve
        self.scorer = scorer
        self.panel_stride = PARAMETERS["idlab_panel_stride"]
        self.prepared, self.hashes, self.inputs = {}, {}, None
        self.index, self.raw = {}, {}

    def sha256(self, path):
        """The hash of an input file, computed once per runner."""
        if path not in self.hashes:
            self.hashes[path] = common.sha256(path)
        return self.hashes[path]

    def released_scorer(self):
        if self.scorer is None:
            self.scorer = common.load_scorer()
        return self.scorer

    def refuse(self, what):
        if not self.solve:
            raise ValueError(f"{what} is not stored under {self.work}. Run research/solid/ablation.py run first.")

    def key(self, dataset, profile, held):
        """The key of an external run, the fingerprint of its profile and inputs, the fold and its clusters among them."""
        if self.inputs is None:
            self.inputs = external_inputs()
        units = None if held is None else dict(fold=held, clusters=folds.listed(folds.clusters(dataset, held)))
        inputs = dict(split=common.sha256(layout_split.RECORD), campaign=self.inputs[dataset],
                      panel_stride=self.panel_stride if dataset == "idlab" else None, fold=units)
        return common.fingerprint(dict(profile=profile, inputs=inputs))[:16]

    def external(self, dataset, mode, profile, held=None, campaign_scales=True, workers=None):
        """The score rows of one candidate profile on an external deployment and mode, on the units of a held fold of the
        round alone when one is given, which every row must belong to, and the run's entry. A stored run whose scoring
        code changed is scored again, its predictions coming from the runner's own cache."""
        workers = workers or self.workers
        key = self.key(dataset, profile, held)
        out = self.work / "runs" / dataset / mode / key
        if not external_stored(dataset, out, profile):
            self.refuse(f"{dataset} {mode}{'' if held is None else f' fold {held}'} run {key}")
            path = self.work / "profiles" / f"{dataset}-{mode}-{key}.json"
            common.write_json(path, profile)
            if dataset == "tiers":
                run_tiers.run(mode, "training", workers, path, out, blocks=None if held is None else sorted(folds.tiers_blocks(held)),
                              campaign_scales=campaign_scales)
            elif dataset == "bregar":
                run_bregar.run("training", workers, path, out, acquisitions=None if held is None else sorted(folds.bregar_acquisitions(held)))
            else:
                run_idlab.run(mode, "training", workers, profile_path=path, out_dir=out, panel_stride=self.panel_stride,
                              configurations=None if held is None else sorted(folds.idlab_configurations(held)))
            common.write_json(out / "round.json", external_identity(dataset, profile, out))
        rows = json.loads((out / "scores.json").read_text())
        if held is not None:
            folds.check_rows(dataset, held, rows)
        return rows, dict(run=str(out.relative_to(self.work)), scores_sha256=common.sha256(out / "scores.json"))

    def bern_scan(self, scan):
        if scan not in self.prepared:
            self.prepared[scan] = cooperative.prepare_scan(scan)
        return self.prepared[scan]

    def bern_directory(self, mode, scan, key):
        """The run directory of a candidate profile on one held Bern scan and mode."""
        return self.work / "runs/bern/anc" / f"scan-{scan:02d}" / key if mode == "anc" else self.work / "runs/bern/cooperative" / f"scan-{scan:02d}" / mode / key

    def bern(self, mode, scan, profile, workers=None):
        """The per unit scores of one candidate profile on one held Bern scan and mode, under the joined pair set."""
        workers = workers or self.workers
        if mode == "anc":
            scenarios, expected = anchor_only.CAMPAIGN / f"scan-{scan:02d}/calibrated.zip", None
        else:
            scenarios, expected = self.bern_scan(scan)
        key = common.fingerprint(dict(profile=profile, scenarios=self.sha256(scenarios)))[:16]
        directory = self.bern_directory(mode, scan, key)
        if mode == "anc":
            work = self.work / "runs/bern/anc"
            predictions, binary = directory / "predictions.zip", anchor_only.CLI
            if not stored(directory / "manifest.json", profile, binary, self.sha256(scenarios)):
                self.refuse(f"bern anc scan {scan} run {key}")
                anchor_only.run(scan, key, workers, {}, work, profile=profile)
        else:
            work = self.work / "runs/bern/cooperative"
            predictions, binary = directory / "run-predictions.zip", cooperative.RUNNER
            if not stored(directory / "manifest.json", profile, binary, self.sha256(scenarios)):
                self.refuse(f"bern {mode} scan {scan} run {key}")
                cooperative.run(scan, mode, key, scenarios, expected[mode], workers, {}, work, profile=profile)
        units = bern_joined(directory, scenarios, predictions, mode, self.released_scorer())
        entry = dict(run=str(directory.relative_to(self.work)), joined_sha256=common.sha256(directory / "joined_scores.csv.gz"))
        return {(scan, *unit): value for unit, value in units.items()}, entry

    def rows(self, name, candidate, dataset, mode):
        """The unit rows of a candidate on one deployment and mode, positioning it when it is not stored. Every unit is
        positioned with the fit that did not see it, a held Bern scan with the fit of the other scans and a fold of
        TIERS, Bregar or IDLab with the fit of the other fold."""
        if dataset == "bern":
            scores, runs = {}, {}
            for scan in fit_bern.SCAN_IDS:
                profile = compose(self.fits, "bern", mode, candidate, scan)
                units, run = self.bern(mode, scan, profile, self.workers)
                scores.update({key: value for key, value in units.items() if value[0] is not None})
                runs[str(scan)] = dict(run, profile_fingerprint=common.fingerprint(profile))
            rows, entry = decision.bern_rows(scores), dict(scans=runs)
        else:
            rows, raw, runs = [], [], {}
            for held in folds.FOLDS:
                profile = compose(self.fits, dataset, mode, candidate, held)
                part, run = self.external(dataset, mode, profile, held, campaign_scales="derived_first_stage" not in candidate["changes"],
                                          workers=self.workers)
                rows += decision.external_rows(dataset, part, pair_set=PARAMETERS["pair_set"])
                raw += part
                runs[str(held)] = dict(run, profile_fingerprint=common.fingerprint(profile))
            self.raw[(name, dataset, mode)] = raw
            entry = dict(folds=runs)
        self.index.setdefault(name, dict(spec=candidate, runs={}))["runs"][f"{dataset}/{mode}"] = dict(entry, units=len(rows))
        print(f"{name} {dataset} {mode}: {len(rows)} units", flush=True)
        return rows


# ------------------------------------------------------------------------------------------- decisions
def scoped_pairs(pairs, parameters=PARAMETERS):
    """The pairs of every deployment and mode, with the anchor only pairs of the declared deployments also over the
    cells with enough anchors."""
    result = dict(pairs)
    for (dataset, mode), (reference, candidate) in pairs.items():
        minimum = parameters["anc_minimum_anchors"].get(dataset) if mode == "anc" else None
        if minimum is not None:
            keep = lambda rows: [row for row in rows if row["anchors"] >= minimum]  # noqa: E731
            result[(dataset, mode, f"anchors>={minimum}")] = (keep(reference), keep(candidate))
    return result


def voting_rule(parameters=PARAMETERS, multi_packet_only=False):
    """Whether a comparison votes: a voting deployment, the scope of enough anchors for the anchor only comparisons of
    the declared deployments, and a multi-packet track for the removal of independent packets."""
    def votes(comparison):
        if comparison["dataset"] not in parameters["deployments"]:
            return False
        if comparison["mode"] == "anc" and comparison["dataset"] in parameters["anc_minimum_anchors"] and comparison.get("scope") is None:
            return False
        if multi_packet_only and int(comparison["track"]) < parameters["multi_packet_minimum_samples"]:
            return False
        return True
    return votes


def judged(pairs, reference, candidate, modes, direction, scorer):
    """The decision record of one comparison of the round, given the rows of both sides
    per deployment and mode, {(dataset, mode): (reference rows, candidate rows)}. reference and candidate are (name,
    spec)."""
    parameters = {key: PARAMETERS[key] for key in DECISION_KEYS}
    record = decision.decide(scoped_pairs(pairs, PARAMETERS), direction, parameters, votes=voting_rule(PARAMETERS), scorer=scorer)
    return dict(record, reference=reference[0], candidate=candidate[0], reference_spec=reference[1], candidate_spec=candidate[1],
                modes=list(modes), multi_packet_only=False, without_bregar=decision.verdict_without(record, UNCROSSED))


# ------------------------------------------------------------------------------------------- the baselines on the held Bern scans
def penalized(path, mode):
    """The pairs the released scorer penalizes for a missing or invalid position per unit of one mode of its scenario
    scores, {(scenario, iteration, samples): pairs}."""
    with gzip.open(path, "rt", newline="") as stream:
        return {(int(row["scenario_id"]), int(row["iteration"]), int(row["samples"])): int(row["penalized_pairs"])
                for row in csv.DictReader(stream) if row["mode"] == mode.upper() and int(row["q"]) == 2}


def numbered(values, offset, scan):
    """Values keyed by (scenario, iteration, samples) with every scenario renumbered by taking off offset, the part of a
    scenario identifier that names the scan, which the baselines' cooperative corpora add to every scenario of the scan
    (baselines_bern_training.fold_archive)."""
    result = {}
    for (scenario, iteration, samples), value in values.items():
        if scenario - offset < 0:
            raise ValueError(f"scenario {scenario} of scan {scan} does not carry the identifier offset {offset} of its scan")
        result[(scenario - offset, iteration, samples)] = value
    return result


def bern_rows(scan, units, penalties, aborted=None, method=None):
    """The score rows of the units of one held Bern scan in the layout of common.write_run, a unit keyed by its scan,
    scenario, iteration and sample track and scored under the released rule and the joined and ranged pair sets. A unit
    fails when the released scorer penalizes a pair of it, as a Bern unit of the final tables fails, or when its solve
    crashed its process, aborted mapping its key to the failure."""
    aborted = aborted or {}
    if set(penalties) != set(units):
        raise ValueError(f"the released scores of scan {scan} hold other units than its scored predictions")
    rows = []
    for key, unit in sorted(units.items()):
        scenario, iteration, samples = key
        failure = aborted.get(key) or (f"the released score penalizes {penalties[key]} pairs for a missing or invalid position" if penalties[key] else None)
        sets = {name: dict(eligible_pairs=unit[name][0], mean_pair_error_cm=unit[name][1]) for name in scoring.SCENARIO_RULES}
        sets["all"] = dict(eligible_pairs=unit["agents"] * (unit["agents"] - 1) // 2, mean_pair_error_cm=None)
        row = dict(key=f"scan-{scan:02d}", id=f"{scenario}/{iteration}", samples=samples, scan=scan, scenario=scenario, iteration=iteration,
                   cell=[scan, unit["anchors"], unit["agents"]], anchors=unit["anchors"], agents=unit["agents"], error_cm=unit[scoring.PAIR_SET][1],
                   elapsed_ms=None, failure=failure, pair_sets=sets)
        if method is not None:
            row["method"] = method
        rows.append(row)
    return rows
