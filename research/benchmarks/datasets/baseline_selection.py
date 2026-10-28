"""The options of the baselines that earlier data chose, selected again on every training half by one rule, and the
declared base configuration of every baseline.

TWR-CLOUD-Bern's training scans chose the planar or the spatial model of Jin 2021, FNL 2025 and STRONG 2021, how FNL
aggregates a link's readings, and where FNL and STRONG start, and some of these choices came after Bern's evaluation.
ADE-BP 2015 ran four rounds of message passing in the anchor aided mode on Bern, the number of its paper, and ten on
TIERS and IDLab, a number chosen on synthetic TIERS layouts. Each is now an option of a grid that every training half
selects from, per mode wherever the method runs and for ADE-BP in the anchor aided mode, by the rule STRONG follows. The
rule takes the lowest mean over the two sample tracks of the balanced S75 under the joined pair set, over the anchor
only headline units with three or more anchors and over every unit in the anchor free and anchor aided modes, ties by
declared order, where mean S75 values within 0.01 cm of the lowest count as equal (TOLERANCE_CM). The
grids follow in declared order.

- Jin 2021 takes the planar or the spatial model, each with the median or the mean of a link's readings, four
  candidates.
- FNL 2025 takes the planar or the spatial model, each with the median of a link's readings or every reading as its
  own term, each from the provided shared start or from its geometry start, eight candidates. FNL implements no mean of
  the readings. Every reading as its own term is the alternative the Bern screen compared with the median, and under
  FNL's squared loss it weighs a link by its readings around their mean.
- In the anchor free mode no anchor fixes a height, so the spatial model gives FNL the solve of the planar one bit for
  bit and Jin's relaxation the same optimum. Both keep the planar model there, Jin two candidates and FNL four.
- STRONG 2021 takes the Huber radius among 10, 20, 40, 80 and 160 cm or no robust loss, each with the planar or the
  spatial model and each from the provided shared start or from its geometry start, 24 candidates
  (joint_candidates).
- ADE-BP 2015 takes four rounds of message passing, the number of its paper, or ten in the anchor aided mode, two
  candidates. The anchor only mode keeps the four rounds of its paper.

The geometry start is SOLID's geometric initializer, and the provided start is the request's own, which every method
shares (layout_inputs.py). Every other setting is fixed and comes from the declared record
baseline-configurations.json in this folder, never from an evaluation run. Jin keeps its published residual multiplier
of two, FNL its 10,000 updates, STRONG its iteration cap of 10,000 and relative tolerance of 1e-5, and ADE-BP the 500
particles of its paper and its cap of 256 importance refinements from a Bern training screen. ADE-BP's ranging model is
fitted on each training half (ade_training.py), and its area is SOLID's operating area of the unit. TWR-CLOUD-Bern runs
ADE-BP from its own fitted configuration (ade_training.bern_config) with the rounds its selection chose.

The runners of TIERS, Bregar, IDLab and the held-out datasets select with these grids on their training halves, and
TWR-CLOUD-Bern on its eleven held training scans by BERN_RULE (research/solid/baselines_bern_training.py). Each writes
one selection record per dataset and method, and a run refuses a record whose rule, candidates, pair set, anchor rule,
base configurations, shared inputs or build differ from the declared ones or from those of the run. A run applies the
declared settings of the candidate a record names and refuses an entry that stores others (chosen).

The reselections of STRONG and ADE-BP added candidates that the evaluated runs took. STRONG's 24 candidates each have a
twin that reads every reading of a link as its own Huber term, named after its median twin with the suffix -all
(strong_candidates), and ADE-BP takes model 2 or model 1 of its paper with the median of a link's readings or every
reading, with four rounds or in the anchor aided mode ten (ade_grid, model2-median-rounds-10 and so on). The grids a
selection ranks (candidates) stay as they are, and a choice may name any declared candidate (declared_grid).

In reader mode (SOLID_READER=1) no selection record exists, and every runner takes the final options of every baseline
per dataset, mode and Bregar environment, and ADE-BP's fitted error law, from the shipped settings
settings/baseline-options.json (Table S-XVIII), through options and chosen, so that a run applies the declared
settings of the candidate the settings name.
"""
import copy
import hashlib
import json
import math
import os
import re
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE / "baseline-configurations.json"
BASE_SCHEMA = "solid.baselines.base-configurations.v1"
# A reader's reproduction (SOLID_READER=1) has no selection record, so its runners take the final options of every
# baseline and ADE-BP's fitted error laws from this shipped settings file (options, chosen)
READER = os.environ.get("SOLID_READER") == "1"
OPTIONS = ROOT / "settings/baseline-options.json"
OPTIONS_SCHEMA = "solid.baselines.options.v1"
# the schema of every selection record of these grids, one record per dataset and method
SELECTION_SCHEMA = "solid.baselines.selection.v3"
# the methods whose options every training half selects, and the modes it selects them in, ADE-BP's rounds in the anchor
# aided mode alone
SELECTED = ("jin-2021", "fnl", "strong", "ade-bp-2015")
MODES = {"jin-2021": ("AF", "AAC", "ANC"), "fnl": ("AF", "AAC", "ANC"), "strong": ("AAC", "ANC"), "ade-bp-2015": ("AAC",)}
LABELS = {"jin-2021": "Jin 2021", "fnl": "FNL 2025", "strong": "STRONG 2021", "ade-bp-2015": "ADE-BP 2015"}
# ADE-BP's rounds of message passing in the anchor aided mode, those of its paper and those chosen on synthetic TIERS layouts
ADE_ROUNDS = (4, 10)
# The rule of every ranking of candidates, here so that a runner needs none of the design modules of strong_threshold.py,
# which takes these names from here. STRONG's same-layout candidates are a geometric grid of Huber radii and the plain
# squared loss, each with the planar and the spatial model and each from STRONG's own geometry start or from the
# provided shared start. A radius no residual reaches never binds the Huber ball, so the loss is the square.
GRID_CM = (10., 20., 40., 80., 160.)
NO_ROBUST_LOSS_CM = 1.0e9
DIMENSIONS = ("planar", "spatial")
# the starts of a baseline, the request's shared start or SOLID's geometry initializer, by label and initialization
STARTS = (("provided", "released"), ("geometry", "solid-geometry"))
# The declared tolerance of every ranking of candidates. Mean S75 values within this many centimetres of the lowest one
# left count as equal, and the declared order decides among them, so that the noise of a solver picks no candidate.
TOLERANCE_CM = 0.01
# The candidates the reselections added (experiments/strong-reselection/grid.py, experiments/ade-reselection/
# declaration.json). STRONG's every-reading twins carry this suffix and aggregation, and ADE-BP's grid of model, readings
# and rounds names its candidates model<model>-<readings>-rounds-<rounds>, model 2 first, then the median of a link's
# readings before every reading, then four rounds before ten, ten in the anchor aided mode alone.
ALL_SUFFIX = "-all"
ALL_AGGREGATION = "AllSamples"
ADE_MODELS = (2, 1)
ADE_READINGS = {"median": "Median", "all": "AllSamples"}
ADE_NAME = re.compile(r"model(?P<model>[12])-(?P<readings>median|all)-rounds-(?P<rounds>[0-9]+)")
# the ranging model's parameters of ADE-BP's error law
LAW_PARAMETERS = ("positive_slope", "positive_intercept_cm", "negative_slope", "negative_intercept_cm")


def joint_candidates():
    """STRONG's same-layout candidates in declared order, per dimension every grid radius ascending and then no robust
    loss, each from the provided start first and then from the geometry start, 24 candidates.

    Each candidate carries the solve settings and the method settings it changes, so a record can extend the grid with
    other settings.
    """
    return [dict(name=f"{dimension}-{'none' if radius is None else f'{radius:g}'}-{start}", dimension_policy=dimension, radius_cm=radius,
                 start=start, solve_config=dict(dimension_policy=dimension, robust_delta=NO_ROBUST_LOSS_CM if radius is None else radius),
                 method_config=dict(initialization=initialization))
            for dimension in DIMENSIONS for radius in GRID_CM + (None,) for start, initialization in STARTS]


def rank_candidates(candidates, mean_s75, tolerance=TOLERANCE_CM):
    """The declared rule. The candidate first in declared order among those whose mean S75 over the sample tracks lies
    within the tolerance of the lowest mean ranks first, and the others are ranked in the same way after it. A
    candidate without a finite mean is refused."""
    infinite = [c["name"] for c in candidates if not math.isfinite(mean_s75[c["name"]])]
    if infinite:
        raise ValueError(f"no finite mean S75 of the candidates {', '.join(infinite)}")
    left, ranked = list(candidates), []
    while left:
        lowest = min(mean_s75[c["name"]] for c in left)
        first = next(c for c in left if mean_s75[c["name"]] <= lowest + tolerance)
        ranked.append(first)
        left.remove(first)
    return ranked


def tied_at_top(ranking, tolerance=TOLERANCE_CM):
    """The candidates of a ranking, rows with mean_s75, whose mean lies within the tolerance of the lowest one, so that
    the declared order chose among them."""
    lowest = min(row["mean_s75"] for row in ranking)
    return sum(row["mean_s75"] <= lowest + tolerance for row in ranking)


def jin_candidates():
    """Jin's candidates in declared order, per dimension the median of a link's readings and then their mean."""
    return [dict(name=f"{dimension}-{aggregation}", dimension_policy=dimension, readings=aggregation,
                 solve_config=dict(dimension_policy=dimension), method_config=dict(range_aggregation=aggregation))
            for dimension in DIMENSIONS for aggregation in ("median", "mean")]


def fnl_candidates():
    """FNL's candidates in declared order, per dimension the median of a link's readings and then every reading as its
    own term, each from the provided start and then from the geometry start."""
    return [dict(name=f"{dimension}-{readings}-{start}", dimension_policy=dimension, readings=readings, start=start,
                 solve_config=dict(dimension_policy=dimension, aggregation=aggregation), method_config=dict(initialization=initialization))
            for dimension in DIMENSIONS for readings, aggregation in (("median", "Median"), ("all", "AllSamples"))
            for start, initialization in STARTS]


def ade_candidates():
    """ADE-BP's candidates in declared order, the rounds of its paper and then the rounds chosen on synthetic TIERS
    layouts."""
    return [dict(name=f"rounds-{rounds}", rounds=rounds, solve_config={}, method_config=dict(iterations=rounds)) for rounds in ADE_ROUNDS]


def strong_candidates():
    """STRONG's 48 declared candidates in declared order, the 24 of joint_candidates and then each of them with every
    reading of a link as its own Huber term, named after its median twin with the suffix -all
    (experiments/strong-reselection/grid.py)."""
    median = joint_candidates()
    every = [dict(copy.deepcopy(entry), name=entry["name"] + ALL_SUFFIX, readings="all",
                  solve_config=dict(entry["solve_config"], aggregation=ALL_AGGREGATION)) for entry in median]
    return median + every


def ade_candidate(name):
    """ADE-BP's candidate of the grid of model, readings and rounds by its name, with the solve and method settings it
    sets and the number of the model of its paper whose fitted law it takes."""
    match = ADE_NAME.fullmatch(name)
    if not match:
        raise ValueError(f"{name} names no candidate of ADE-BP's grid of model, readings and rounds")
    rounds = int(match["rounds"])
    return dict(name=name, model=int(match["model"]), readings=match["readings"], rounds=rounds,
                solve_config=dict(aggregation=ADE_READINGS[match["readings"]]), method_config=dict(iterations=rounds))


def ade_grid(mode=None):
    """ADE-BP's grid of model, readings and rounds in declared order (experiments/ade-reselection/declaration.json), in
    the anchor aided mode with four or ten rounds, eight candidates, and in the anchor only mode with the four of its
    paper, four candidates. Without a mode, every candidate of both."""
    rounds = {"AAC": ADE_ROUNDS, "ANC": (4,)}
    names = [f"model{model}-{readings}-rounds-{count}" for model in ADE_MODELS for readings in ADE_READINGS
             for count in rounds.get(mode, ADE_ROUNDS)]
    return [ade_candidate(name) for name in names]


def selects(method, mode):
    """Whether a method's options are selected in a mode, so that its runs there need a selection."""
    return mode in MODES.get(method, ())


def candidates(method, mode=None):
    """The declared candidates of a method's options in a mode, in declared order. In the anchor free mode the planar
    model stays alone, since without anchors the spatial model changes no solve. Without a mode, the candidates of the
    anchored modes."""
    grids = {"jin-2021": jin_candidates, "fnl": fnl_candidates, "strong": joint_candidates, "ade-bp-2015": ade_candidates}
    if method not in grids:
        raise ValueError(f"{method} has no options to select")
    grid = grids[method]()
    if mode == "AF":
        grid = [entry for entry in grid if entry["solve_config"].get("dimension_policy") != "spatial"]
    return grid


def declared_grid(method, mode=None):
    """Every declared candidate of a method in a mode, in declared order, the candidates a selection ranks and after them
    those the reselections added: STRONG's every-reading twins (strong_candidates) and ADE-BP's grid of model, readings
    and rounds of the mode (ade_grid). A choice names one of these."""
    grid = candidates(method, mode)
    if method == "strong":
        grid = strong_candidates()
    elif method == "ade-bp-2015":
        grid = grid + ade_grid(mode)
    if mode == "AF":
        grid = [entry for entry in grid if entry["solve_config"].get("dimension_policy") != "spatial"]
    return grid


def grid(method, modes=None):
    """The declared candidates of a method per mode that selects its options, {mode: candidates}, over the modes given
    or every mode of MODES. A selection record names this grid."""
    return {mode: candidates(method, mode) for mode in (modes or MODES[method]) if selects(method, mode)}


def candidate(method, name, mode=None):
    """A method's candidate of this name among its declared candidates in the mode (declared_grid)."""
    for entry in declared_grid(method, mode):
        if entry["name"] == name:
            return entry
    raise ValueError(f"{method} has no candidate {name}" + (f" in {mode}" if mode else ""))


def chosen(method, mode, choice):
    """The declared candidate a choice names, a candidate itself, the entry of a selection record or the entry of the
    shipped settings (options), refused when the settings the entry holds differ from those of the declared candidate,
    so that a run applies the declared settings of the name its record or the settings selected. ADE-BP's entry of the
    settings carries its fitted error law, which the candidate then carries too, refused unless the law is of the model
    the candidate names."""
    name = choice.get("selected") or choice.get("name")
    entry = candidate(method, name, mode)
    for key in ("solve_config", "method_config"):
        if key in choice and json.loads(json.dumps(choice[key] or {})) != json.loads(json.dumps(entry.get(key) or {})):
            raise ValueError(f"the choice of {method} in {mode} names {name} with other {key} than the declared candidate")
    if "ranging_model" in choice:
        if choice.get("model") != entry.get("model", 2):
            raise ValueError(f"the choice of {method} in {mode} names {name} with the law of model {choice.get('model')}")
        entry = dict(entry, model=choice["model"], ranging_model=copy.deepcopy(choice["ranging_model"]))
    return entry


def with_settings(baselines, choices, alone=None):
    """The baselines with every chosen method's candidate applied, its solve and method settings replacing the entry's,
    every other method unchanged. choices maps a method to a candidate. alone keeps that method's entry only, the request
    of a selection run, since the other baselines do not change with its settings."""
    result = copy.deepcopy(baselines)
    for entry in result:
        chosen = choices.get(entry["method"])
        if chosen is None:
            continue
        entry["solve_config"].update(chosen["solve_config"])
        entry["method_config"].update(chosen.get("method_config") or {})
    if alone is not None:
        result = [entry for entry in result if entry["method"] == alone]
        if len(result) != 1:
            raise ValueError(f"the request holds no {alone} baseline")
    return result


def base_record():
    """The declared base configurations, refused unless they are of the declared schema."""
    record = json.loads(BASE.read_text())
    if record.get("schema") != BASE_SCHEMA:
        raise ValueError(f"{BASE} is not a record of the declared base configurations")
    return record


def declared(method, mode, bern=False):
    """One baseline entry of a request from the declared record, its fixed settings of every mode and of this mode, with
    the options TWR-CLOUD-Bern chose when bern. A selected method takes its options from its selection instead
    (with_settings)."""
    record = base_record()
    entry = record["methods"][method]
    parts = [entry, (entry.get("modes") or {}).get(mode) or {}]
    if bern:
        parts.append(entry["bern"][mode])
    solve, config = dict(record["solve_config"]), {}
    for part in parts:
        solve.update(part.get("solve_config") or {})
        config.update(part.get("method_config") or {})
    return dict(method=method, method_config=config, solve_config=solve)


def declared_baselines(roster, mode, choices, ranging_model=None):
    """The baselines of a request in the order of its roster, every one from the declared record, each method whose
    options the mode selects with the declared candidate its choice names (chosen) and ADE-BP with the ranging model of
    the training half. A method without the choice its mode selects is refused, so no run takes an option that earlier
    data chose."""
    result = []
    for method in roster:
        entry = declared(method, mode)
        if selects(method, mode) and method not in choices:
            raise ValueError(f"{method} runs in {mode} without a selection of its options")
        # a mode that selects no option of the method takes a choice only from the settings of reader mode (options),
        # ADE-BP's anchor only candidate, whose settings are those of the declared base configuration
        if method in choices:
            entry = with_settings([entry], {method: chosen(method, mode, choices[method])})[0]
        if method == "ade-bp-2015":
            if ranging_model is None:
                raise ValueError("ADE-BP runs without the ranging model of its training half")
            entry["method_config"]["ranging_model"] = copy.deepcopy(ranging_model)
        result.append(entry)
    return result


def provenance(mode):
    """The Bern evaluation run records the declared values were once read from, {relative path: SHA-256}, provenance only."""
    return {path: digest for path, digest in base_record()["provenance"][mode].values()}


# ------------------------------------------------------------------------------------------- the shipped settings
def options_record():
    """The shipped settings of the baselines' final options and ADE-BP's laws, refused unless they are of the declared
    schema and every option names a declared candidate whose ADE-BP model is that of its law."""
    record = json.loads(OPTIONS.read_text())
    if record.get("schema") != OPTIONS_SCHEMA:
        raise ValueError(f"{OPTIONS} is not a record of the baselines' options of the declared schema {OPTIONS_SCHEMA}")
    return record


def options_sha256():
    """The SHA-256 of the shipped settings, which every run record of reader mode keeps."""
    return hashlib.sha256(OPTIONS.read_bytes()).hexdigest()


def options(dataset, mode, environment=None):
    """The choices of the shipped settings for one dataset, mode and on Bregar environment, {method: entry}, every entry
    naming the declared candidate of the final option (Table S-XVIII), ADE-BP's with its fitted error law and the number
    of its model. A run applies them through declared_baselines and chosen."""
    record = options_record()
    result = {}
    for method, modes in record["options"][dataset].items():
        name = modes.get(mode)
        if isinstance(name, dict):
            name = name[environment]
        if name is None:
            continue
        entry = dict(selected=name)
        if method == "ade-bp-2015":
            entry.update(law_entry(record, dataset, mode, environment))
        chosen(method, mode, entry)
        result[method] = entry
    return result


def law_entry(record, dataset, mode, environment=None):
    """ADE-BP's fitted error law of the shipped settings for a dataset, mode and Bregar environment, its model and its
    ranging model."""
    entry = record["ade_bp"]["laws"][dataset][mode]
    if environment is not None or "model" not in entry:
        entry = entry[environment]
    if set(entry["ranging_model"]) != set(LAW_PARAMETERS):
        raise ValueError(f"the law of ADE-BP on {dataset} in {mode} has other parameters than {', '.join(LAW_PARAMETERS)}")
    return dict(model=entry["model"], ranging_model=copy.deepcopy(entry["ranging_model"]))


def law(dataset, mode, environment=None):
    """ADE-BP's ranging model on a dataset, mode and Bregar environment from the shipped settings, the law of the model
    its final option names (chosen)."""
    return chosen("ade-bp-2015", mode, options(dataset, mode, environment)["ade-bp-2015"])["ranging_model"]


def kept(choices, mode):
    """What a run record of reader mode keeps of the choices of the shipped settings (options), per method the name and
    settings of the declared candidate, ADE-BP's model and law, and the SHA-256 of the settings."""
    digest = options_sha256()
    result = {}
    for method, choice in choices.items():
        entry = chosen(method, mode, choice)
        result[method] = dict(selected=entry["name"], solve_config=entry["solve_config"], method_config=entry.get("method_config") or {},
                              options_sha256=digest, **{key: entry[key] for key in ("model", "ranging_model") if key in entry})
    return result


def bern_area():
    """ADE-BP's operating area on TWR-CLOUD-Bern from the shipped settings, the one area of its released inputs."""
    return copy.deepcopy(options_record()["ade_bp"]["bern_area"])


# ------------------------------------------------------------------------------------------- TWR-CLOUD-Bern
# The solid CLI's flags of a baseline's solve settings, and the settings it has no flag for, which it keeps at these
# values
AGGREGATION_FLAGS = {"Median": "median", "AllSamples": "all"}
DIMENSION_FLAGS = {"planar": "two-d", "spatial": "three-d"}
CLI_FIXED = dict(gradient_tolerance=1e-5, history_size=10)
BERN_RULE = ("Per mode wherever the method runs, on the eleven held TWR-CLOUD-Bern training scans with the inputs of SOLID's held-scan "
             "rounds, each of Jin 2021, FNL 2025 and STRONG 2021 on its own among the declared candidates of the options TWR-CLOUD-Bern "
             "chose for it, and ADE-BP 2015 among four and ten rounds in the anchor aided mode (baseline_selection.py), by the lowest mean "
             "over the two sample tracks of the balanced S75 over the cells of held scan, anchor count and agent count, in the anchor only "
             "mode over the units with three or more anchors of its headline and in the other modes over every unit, ties by declared order "
             "among the means within 0.01 cm of the lowest. A unit's error is its mean capped pair error over the pairs its measured links "
             "join, and a unit without such a pair is left out. Every other setting is the declared base configuration.")


def cli_flags(entry, config_path):
    """The flags of the solid CLI's benchmark command for a baseline entry of a request (declared_baselines), its solve
    settings as flags and its method settings through the file config_path, which the caller writes with the entry's
    method_config. The CLI has no flag for the gradient tolerance and the history size and keeps its own, which the
    entry must hold."""
    solve = entry["solve_config"]
    for key, value in CLI_FIXED.items():
        if solve.get(key) != value:
            raise ValueError(f"the solid CLI keeps {key} at {value}, not {solve.get(key)}")
    if solve["max_time_micros"] % 1000:
        raise ValueError("the solid CLI takes its time budget in whole milliseconds")
    return ["--method", entry["method"], "--solver-iterations", str(solve["max_iterations"]), "--time-budget-ms", str(solve["max_time_micros"] // 1000),
            "--dimensions", DIMENSION_FLAGS[solve["dimension_policy"]], "--aggregation", AGGREGATION_FLAGS[solve["aggregation"]],
            "--robust-delta", repr(float(solve["robust_delta"])), "--max-tof", repr(float(solve["max_tof_ns"])), "--method-config", str(config_path)]


def write_config(path, entry):
    """The method settings of a baseline entry in the file the CLI reads, written when it holds other settings, staged
    beside it and moved in place, so that runs at the same time never read a part of it."""
    path = Path(path)
    text = json.dumps(entry["method_config"], indent=1, sort_keys=True) + "\n"
    if not path.exists() or path.read_text() != text:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.partial")
        staged.write_text(text)
        staged.replace(path)
    return path


def bern_record(record, path, method, cli_sha256, pair_set, minimum_anchors):
    """A TWR-CLOUD-Bern selection record of a method, refused unless it has the declared schema, names the method, was
    made by BERN_RULE over the method's declared candidates of every mode that selects its options, ranked them under
    the headline's pair set, the anchor only mode over the units of minimum_anchors or more anchors, from the declared
    base configurations as they are, and the solid CLI whose SHA-256 is cli_sha256 made it, the build that runs now.
    The record's content decides, so a record kept from an earlier rule, grid, base record or build runs nothing."""
    if record.get("schema") != SELECTION_SCHEMA or record.get("dataset") != "bern":
        raise ValueError(f"{path} is not a TWR-CLOUD-Bern selection record of the declared schema {SELECTION_SCHEMA}")
    if record.get("method") != method:
        raise ValueError(f"{path} is not a selection of {method}")
    if record.get("rule") != BERN_RULE or record.get("candidates") != json.loads(json.dumps(grid(method))):
        raise ValueError(f"{path} was made under another rule or candidate grid than the declared ones")
    if record.get("pair_set") != pair_set or any(entry.get("pair_set") != pair_set for entry in (record.get("selections") or {}).values()):
        raise ValueError(f"{path} ranked {method}'s candidates under another pair set than the {pair_set} pairs of the headline")
    if record.get("anchor_only_headline_min_anchors") != minimum_anchors:
        raise ValueError(f"{path} ranked the anchor only mode over other units than those of {minimum_anchors} or more anchors")
    # a reader's build and base record are never the ones that made our records, and reader mode takes the shipped
    # settings instead (options)
    if READER:
        return record
    if record.get("base_configurations_sha256") != hashlib.sha256(BASE.read_bytes()).hexdigest():
        raise ValueError(f"{path} was made with other declared base configurations than those of this run")
    if record.get("cli_sha256") != cli_sha256:
        raise ValueError(f"{path} was made by another build of the solid CLI than the one that runs now. Every method of a dataset runs with "
                         "one build, so select again with this build after moving the record aside")
    return record
