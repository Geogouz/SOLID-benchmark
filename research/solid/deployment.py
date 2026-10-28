"""Deployment rules of the SOLID anchor only estimator, one set for every dataset, and the record of the training halves
they read.

A profile holds the link model of a fit record and the deployment settings of the rules below. The rules read only the
training half of a deployment. The statistics they read are written, with a check of the operating areas on every
training unit, to research/artifacts/solid/fits/deployment.json, and profiles.py and compose_unified_profiles.py compose
from that record. Nothing is set by hand per dataset.

Operating area of a unit. The bounding box of the unit's known anchors, padded on every side by the unit's largest
measured anchor range plus three times the largest reported packet scale of its anchor links. An agent with an anchor
link is no farther from that anchor than its true slant distance, which exceeds a reading by more than three packet
scales only after a gross negative error, so no agent with an anchor link falls outside. A dataset that publishes its
site extent intersects the padded box with the bound the extent puts on the unit, which holds every point of the site.
An agent without any anchor range can lie anywhere on the site, so a unit holding one takes that bound alone, as does a
unit without any anchor range. A dataset without a published extent keeps the padded box for such a unit and cannot
solve a unit without any anchor range. TIERS (the published 8 m by 9 m arena) and IDLab (the published 41 m by 26 m
floor) bound every coordinate by the extent's diagonal around every known anchor, the declared support of their
campaigns. Bern uses the bounding box of its training scans' node positions (fits/bern_deployment.json), the area its
baselines share. Its harness applies one area to the whole deployment, so the Bern profile carries that box, which
equals the padded box intersected with it on all but a few training units (the record counts them). Bregar publishes
no extent. The record checks on every training half that every true agent position lies inside its unit's area.

Height prior. The true heights of the training half's unknown nodes, one value per distinct unknown position. Five
equally spaced nodes from the 5th to the 95th percentile with equal weights, or one node at the median when that span
is under 10 cm. Bern takes the unknown nodes of its training scans, and a held-scan fold those of the other ten scans.

Budgets. The grid step is one quarter of the median reported packet scale of the training half's anchor links, within
5 cm and 40 cm, so that the cells resolve the width of a link's likelihood. It is the step of the finest cells of the
engine's anchor posterior. Every deployment draws 512 joint samples and uses the engine's seed. The grid step and the
joint samples are the base budgets. The engine computes the posterior adaptively on the finest grid and keeps every
cell whose mass may matter, with the value the exhaustive grid gives it, dropping a share of the posterior mass that it
certifies to be at most 1e-4, so no count of kept cells is a budget and no profile holds one. The budget check of
research/solid/confirm_round.py doubles the budgets together on the training halves, the grid step over the square root
of two and the joint samples twice, so that every doubling doubles the finest cells per unit area and the joint samples
alike, until a doubling changes no anchor only result beyond its noise. Its record
research/artifacts/solid/selections/budget-check.json holds the factor of the converged budgets and the hash of the
deployment record it ran on, and the compositions of the method scale the base budgets by it. They take the factor one
while no record exists and no record of the round's Run I either, and they refuse a missing or incomplete record once
the round has run. The budgets are then numerical accuracy settings shown to be converged on training data, not design
choices.

Usage: deployment.py [--datasets bern,tiers,bregar,idlab] [--check]
       writes research/artifacts/solid/fits/deployment.json and prints the areas, containment, heights and budgets.
       --check compares instead, in a reader's reproduction (SOLID_READER=1) the statistics without the SHA-256 of
       the halves they were read from
"""
import argparse
import gzip
import json
import math
import sys
import time
import zipfile
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

RECORD = common.FITS / "deployment.json"
BERN_DEPLOYMENT = common.FITS / "bern_deployment.json"
SCHEMA = "solid.deployment.v1"
# the record of the budget check of research/solid/confirm_round.py, which the compositions of the method read
BUDGET_RECORD = common.SELECTIONS / "budget-check.json"
BUDGET_SCHEMA = "solid.selection.budget-check.v1"
# the record of Run I of the confirmatory round, whose existence means the budget check has run
ROUND_RECORD = common.SELECTIONS / "confirmatory-run-1.json"
# the numerical budgets of the anchor only estimator that the budget check scales together, the step of the finest
# cells of the engine's adaptive posterior and the joint samples
BUDGET_KEYS = ("grid_step", "joint_samples")
DATASETS = ("bern", "tiers", "bregar", "idlab")
PADDING_SCALES = 3.0
HEIGHT_PERCENTILES = (5.0, 95.0)
HEIGHT_NODES = 5
SINGLE_HEIGHT_SPAN_CM = 10.0
GRID_FRACTION = 0.25
GRID_STEP_BOUNDS_CM = (5.0, 40.0)
JOINT_SAMPLES = 512
SEED = 20260918
DECIMALS = 1
# width and depth of the published site of a dataset, placed by its diagonal around every known anchor
PUBLISHED_EXTENTS_CM = {"tiers": (800.0, 900.0), "idlab": (4100.0, 2600.0)}
EXTENT_SOURCES = {"tiers": "published 8 m by 9 m arena, diagonal around every known anchor",
                  "idlab": "published 41 m by 26 m floor, diagonal around every known anchor",
                  "bern": "bounding box of the node positions of the training scans, research/artifacts/solid/fits/bern_deployment.json"}


def rules():
    """The constants of the rules, stored with the record so that a record built under other rules is refused."""
    return dict(padding_scales=PADDING_SCALES, height_percentiles=list(HEIGHT_PERCENTILES), height_nodes=HEIGHT_NODES,
                single_height_span_cm=SINGLE_HEIGHT_SPAN_CM, grid_fraction=GRID_FRACTION, grid_step_bounds_cm=list(GRID_STEP_BOUNDS_CM),
                joint_samples=JOINT_SAMPLES, seed=SEED, decimals=DECIMALS,
                published_extents_cm={name: list(size) for name, size in PUBLISHED_EXTENTS_CM.items()})


# ---------------------------------------------------------------------------------------------------------------------
# The rules


def valid_sample(sample):
    """A calibrated sample the engine accepts, finite with a nonnegative distance and a positive scale."""
    distance, scale = sample.get("distance_cm"), sample.get("scale_cm")
    return (isinstance(distance, (int, float)) and isinstance(scale, (int, float)) and math.isfinite(distance) and distance >= 0
            and math.isfinite(scale) and scale > 0)


def anchor_evidence(problem):
    """The largest measured anchor range, the largest reported packet scale of the anchor links and the agents with at
    least one admissible anchor sample of one prepared unit. Both values are None without any admissible anchor sample."""
    largest_range = largest_scale = None
    linked = set()
    for observation in problem["observations"]:
        if not observation["to_is_anchor"]:
            continue
        if observation.get("tof_ns") and not observation.get("calibrated_samples"):
            raise ValueError("raw timings reach the area rule only through the deployment's calibration")
        samples = [s for s in observation.get("calibrated_samples") or [] if valid_sample(s)]
        if not samples:
            continue
        linked.add(observation["from"])
        distance = max(s["distance_cm"] for s in samples)
        scale = max(s["scale_cm"] for s in samples)
        largest_range = distance if largest_range is None else max(largest_range, distance)
        largest_scale = scale if largest_scale is None else max(largest_scale, scale)
    return largest_range, largest_scale, linked


def padded_box(anchors, padding):
    """The bounding box of the anchors (x, y) padded by `padding` centimetres on every side."""
    xs, ys = [x for x, _ in anchors], [y for _, y in anchors]
    return dict(minimum_x_cm=min(xs) - padding, maximum_x_cm=max(xs) + padding, minimum_y_cm=min(ys) - padding,
                maximum_y_cm=max(ys) + padding)


def diagonal_bound(anchors, width_cm, depth_cm):
    """The points within the diagonal of a published site of every known anchor, per axis. Every point of a site of
    that width and depth which holds the anchors lies inside."""
    diagonal = math.hypot(width_cm, depth_cm)
    xs, ys = [x for x, _ in anchors], [y for _, y in anchors]
    bound = dict(minimum_x_cm=max(xs) - diagonal, maximum_x_cm=min(xs) + diagonal, minimum_y_cm=max(ys) - diagonal,
                 maximum_y_cm=min(ys) + diagonal)
    if not (bound["minimum_x_cm"] < bound["maximum_x_cm"] and bound["minimum_y_cm"] < bound["maximum_y_cm"]):
        raise ValueError("the anchors spread wider than the published site")
    return bound


def intersect(area, bound):
    result = dict(minimum_x_cm=max(area["minimum_x_cm"], bound["minimum_x_cm"]), maximum_x_cm=min(area["maximum_x_cm"], bound["maximum_x_cm"]),
                  minimum_y_cm=max(area["minimum_y_cm"], bound["minimum_y_cm"]), maximum_y_cm=min(area["maximum_y_cm"], bound["maximum_y_cm"]))
    if not (result["minimum_x_cm"] < result["maximum_x_cm"] and result["minimum_y_cm"] < result["maximum_y_cm"]):
        raise ValueError("the padded anchor box and the site extent do not overlap")
    return result


def operating_area(anchors, largest_range, largest_scale, extent=None, every_agent_ranged=True):
    """The operating area of one unit from its known anchors (x, y) in centimetres, its largest measured anchor range
    and largest reported packet scale, intersected with the bound of the site extent when the dataset publishes one. A
    unit holding an agent without any anchor range takes the extent's bound alone, since such an agent can lie
    anywhere on the site."""
    if not anchors:
        raise ValueError("the operating area needs at least one known anchor")
    if largest_range is None or (extent is not None and not every_agent_ranged):
        if extent is None:
            raise ValueError("a unit without any anchor range has no operating area unless its site extent is published")
        return dict(extent)
    box = padded_box(anchors, largest_range + PADDING_SCALES * largest_scale)
    return box if extent is None else intersect(box, extent)


def extent_bound(dataset, anchors):
    """The bound a published site extent puts on a unit with these known anchors, None without a published extent."""
    if dataset in PUBLISHED_EXTENTS_CM:
        return diagonal_bound(anchors, *PUBLISHED_EXTENTS_CM[dataset])
    return None


def unit_area(problem, dataset):
    """The operating area of one prepared anchor only unit of an external dataset."""
    if dataset == "bern":
        raise ValueError("the Bern harness takes the area of the Bern profile")
    anchors = [(a["position"]["x"], a["position"]["y"]) for a in problem["anchors"]]
    largest_range, largest_scale, ranged = anchor_evidence(problem)
    return operating_area(anchors, largest_range, largest_scale, extent_bound(dataset, anchors), set(problem["agent_ids"]) <= ranged)


def contains(area, x, y):
    return area["minimum_x_cm"] <= x <= area["maximum_x_cm"] and area["minimum_y_cm"] <= y <= area["maximum_y_cm"]


def height_statistics(heights):
    """The distribution of the unknown nodes' true heights the height rule reads, one value per distinct position."""
    values = np.asarray(list(heights), dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("the height rule needs finite heights of unknown nodes")
    low, median, high = np.percentile(values, [HEIGHT_PERCENTILES[0], 50.0, HEIGHT_PERCENTILES[1]])
    return dict(positions=int(len(values)), minimum=float(values.min()), p5=float(low), median=float(median), p95=float(high),
                maximum=float(values.max()))


def height_prior(statistics):
    """Height nodes and weights: five equally spaced nodes from the 5th to the 95th percentile, or one node at the median
    when that span is under 10 cm, equal weights, rounded to the millimetre."""
    low, high = statistics["p5"], statistics["p95"]
    nodes = [statistics["median"]] if high - low < SINGLE_HEIGHT_SPAN_CM else np.linspace(low, high, HEIGHT_NODES).tolist()
    nodes = [round(float(value), DECIMALS) + 0.0 for value in nodes]
    return nodes, [1.0] * len(nodes)


def single_height(settings):
    """One height node at the weighted mean of a deployment's nodes, the prior of the single height ablation."""
    nodes, weights = np.asarray(settings["height_nodes"], dtype=float), np.asarray(settings["height_weights"], dtype=float)
    return dict(height_nodes=[float((nodes * weights).sum() / weights.sum())], height_weights=[1.0])


def grid_step(median_scale):
    """One quarter of the median reported packet scale within the bounds, rounded to the millimetre."""
    if not (math.isfinite(median_scale) and median_scale > 0):
        raise ValueError("the grid rule needs a positive median packet scale")
    low, high = GRID_STEP_BOUNDS_CM
    return round(min(max(GRID_FRACTION * median_scale, low), high), DECIMALS)


def settings_from(entry):
    """The anchor only deployment settings of one record entry by the rules."""
    nodes, weights = height_prior(entry["heights_cm"])
    return dict(height_nodes=nodes, height_weights=weights, grid_step=grid_step(entry["anchor_scale_cm"]["median"]),
                joint_samples=JOINT_SAMPLES, seed=SEED)


# ---------------------------------------------------------------------------------------------------------------------
# The record


def split_record_sha256():
    import layout_split
    return common.sha256(layout_split.RECORD)


def load(path=None):
    """The tracked deployment record, refused when it was built under other rules or another split."""
    record = common.read_json(path or RECORD)
    if record.get("schema") != SCHEMA:
        raise ValueError(f"{path or RECORD} is not a deployment record of schema {SCHEMA}")
    if record.get("rules") != rules():
        raise ValueError("The deployment record was built under other rules. Rebuild it with research/solid/deployment.py")
    if record.get("split_record_sha256") != split_record_sha256():
        raise ValueError("The deployment record was built on another same-layout split. Rebuild it with research/solid/deployment.py")
    return record


def entry(dataset, environment=None, held=None, record=None):
    """The record entry of a deployment: a dataset, a Bregar environment or a Bern held-scan fold."""
    record = record or load()
    if dataset not in record["datasets"]:
        raise ValueError(f"the deployment record holds no entry for {dataset}")
    item = record["datasets"][dataset]
    if dataset == "bregar":
        if environment is None:
            raise ValueError("Bregar deployments are its environments")
        item = item["environments"][environment]
    elif environment is not None:
        raise ValueError(f"{dataset} is one deployment")
    if held is not None:
        if dataset != "bern":
            raise ValueError("held-scan folds exist on Bern only")
        item = item["folds"][str(held)]
    return item


def settings(dataset, environment=None, held=None, record=None):
    """The anchor only deployment settings of a deployment: the height prior, the grid step and the budgets."""
    item = entry(dataset, environment, held, record)
    derived = settings_from(item)
    if item.get("settings") != derived:
        raise ValueError(f"The deployment record's settings of {dataset} differ from its rules. Rebuild it with research/solid/deployment.py")
    return derived


def bern_area(held=None, record=None):
    """The one operating area of the Bern profile, the fold's area for a held scan."""
    return dict(entry("bern", held=held, record=record)["area"])


# ---------------------------------------------------------------------------------------------------------------------
# The budgets of the method


# the power of the factor each count budget grows by, the joint samples by the factor itself, while the grid step
# shrinks by the square root of the factor (scale_budgets)
BUDGET_GROWTH = dict(joint_samples=1)


def budget_base():
    """The constants that set the base budgets and their growth with a factor, which a budget check record must have
    been built on."""
    return dict(grid_fraction=GRID_FRACTION, grid_step_bounds_cm=list(GRID_STEP_BOUNDS_CM), joint_samples=JOINT_SAMPLES,
                decimals=DECIMALS, growth=dict(BUDGET_GROWTH))


def budget_factor(factor):
    """A factor of the budgets over the base budgets, a power of two, refused otherwise."""
    if isinstance(factor, bool) or not isinstance(factor, int) or factor < 1 or factor & (factor - 1):
        raise ValueError(f"a budget factor is a power of two, not {factor!r}")
    return factor


def scale_budgets(settings, factor):
    """The settings with every numerical budget of the anchor only estimator that they hold scaled by `factor`, a power
    of two. The grid step shrinks by the square root of the factor, so the finest cells per unit area grow by the
    factor, and the joint samples grow by the factor. The scaled step is not held to the bounds of the grid rule, which
    set the base step only. Every other setting stays as it is."""
    factor = budget_factor(factor)
    result = dict(settings)
    if factor == 1:
        return result
    if "grid_step" in result:
        result["grid_step"] = result["grid_step"] / math.sqrt(factor)
    for key, power in BUDGET_GROWTH.items():
        if key in result:
            result[key] = int(result[key]) * factor ** power
    return result


def budget_record(path=None):
    """The record of the budget check, None while none exists. A record of another schema, built on other base budgets
    or on another deployment record, whose settings its runs scaled, is refused."""
    path = Path(path or BUDGET_RECORD)
    if not path.exists():
        return None
    record = common.read_json(path)
    if record.get("schema") != BUDGET_SCHEMA:
        raise ValueError(f"{path} is not a budget check record of schema {BUDGET_SCHEMA}")
    if record.get("base") != budget_base():
        raise ValueError(f"The budget check of {path} ran on other base budgets. Run the budget check of research/solid/confirm_round.py again")
    if record.get("deployment_record_sha256") != common.sha256(RECORD):
        raise ValueError(f"The budget check of {path} ran on another deployment record than {RECORD}. Run the budget check of "
                         "research/solid/confirm_round.py again")
    return record


def converged_budget(path=None):
    """The factor of the method's budgets from the budget check record, one while neither that record nor the record
    of the round's Run I exists: the converged budgets, or when no doubling converged the largest budgets the check
    tried, the declared outcome, which the record flags. A missing record is refused once the round has run, and the
    record of a check that did not complete is refused always, since the method then has no budgets."""
    record = budget_record(path)
    if record is None:
        if ROUND_RECORD.exists():
            raise ValueError(f"The confirmatory round has run ({ROUND_RECORD.name} exists), so the method's budgets come from its budget "
                             f"check record {path or BUDGET_RECORD}, which is missing")
        return 1
    if not record.get("complete") or not record.get("result"):
        raise ValueError(f"The budget check of {path or BUDGET_RECORD} did not complete, so the method has no budgets. Its record holds every step")
    return budget_factor(record["result"]["factor"])


def budget_provenance(path=None):
    """What a run record keeps of the method's budgets, the factor its profiles take with the budget check record and
    its hash. Once the round has confirmed a method, the factor is the one of its method record (profiles.METHOD_RECORD),
    twice the converged factor when a verdict of the round differed at twice the budgets, and the method record and its
    hash stand beside it. Before, it is the converged factor of the budget check record, and the factor one with no hash
    while no record exists. A record that the compositions refuse is kept with its hash and no factor, so a run record
    never fails for it. A reader's run, which has neither record, keeps the factor of the method's constants
    (profiles.method_record), twice the base budgets."""
    import profiles

    def relative(record):
        return str(record.relative_to(common.ROOT)) if record.is_relative_to(common.ROOT) else str(record)
    if common.READER:
        return dict(record=None, record_sha256=None, factor=budget_factor(profiles.method_record()["budget"]["factor"]),
                    constants="research/solid/method_constants.py")
    path = Path(path or BUDGET_RECORD)
    result = dict(record=relative(path), record_sha256=common.sha256(path) if path.exists() else None)
    try:
        if profiles.METHOD_RECORD.exists():
            result.update(method_record=relative(profiles.METHOD_RECORD), method_record_sha256=common.sha256(profiles.METHOD_RECORD))
            result["factor"] = budget_factor((profiles.method_record().get("budget") or {}).get("factor"))
        else:
            result["factor"] = converged_budget(path)
    except ValueError as error:
        result.update(factor=None, refused=str(error))
    return result


def budgeted(settings, budget=None):
    """Anchor only settings with the budgets of the factor `budget`, by default the converged budgets of the budget
    check record."""
    return scale_budgets(settings, converged_budget() if budget is None else budget)


# ---------------------------------------------------------------------------------------------------------------------
# Training halves


def median_of_counts(counts):
    """The median of a multiset given as value counts, the mean of the two middle values for an even count."""
    if not counts:
        raise ValueError("no packet scale to take the median of")
    values = sorted(counts)
    total = sum(counts.values())
    wanted = [(total - 1) // 2, total // 2]
    found, seen = [], 0
    for value in values:
        upper = seen + counts[value]
        while len(found) < 2 and wanted[len(found)] < upper:
            found.append(value)
        seen = upper
    return float(0.5 * (found[0] + found[1]))


def spread(values):
    values = np.asarray(values, dtype=float)
    return [float(values.min()), float(np.median(values)), float(values.max())] if len(values) else None


class Tally:
    """Containment of the true agent positions in the operating areas of a training half, with the statistics the
    height and grid rules read. `collect_scales` counts the reported packet scales, which Bern takes from its raw
    timings instead."""

    def __init__(self, collect_scales=True):
        self.units = self.agents = self.linked = self.linked_outside = self.unlinked = self.unlinked_outside = 0
        self.without_range = self.with_unranged_agent = self.extent_binding = 0
        self.collect_scales = collect_scales
        self.heights, self.scales = {}, Counter()
        self.widths, self.depths, self.paddings = array("d"), array("d"), array("d")
        self.outside = []

    def add(self, key, problem, truth, dataset, area=None, evidence=None):
        """One prepared unit: `truth` maps every agent to its true (x, y). The area is the rule's unless given, and
        `evidence` the unit's anchor evidence when already computed."""
        anchors = [(a["position"]["x"], a["position"]["y"]) for a in problem["anchors"]]
        largest_range, largest_scale, linked = evidence or anchor_evidence(problem)
        every_agent_ranged = set(problem["agent_ids"]) <= linked
        if area is None:
            extent = extent_bound(dataset, anchors)
            area = operating_area(anchors, largest_range, largest_scale, extent, every_agent_ranged)
            if extent is not None and largest_range is not None and area != padded_box(anchors, largest_range + PADDING_SCALES * largest_scale):
                self.extent_binding += 1
        self.units += 1
        self.without_range += int(largest_range is None)
        self.with_unranged_agent += int(not every_agent_ranged)
        if largest_range is not None:
            self.paddings.append(largest_range + PADDING_SCALES * largest_scale)
        self.widths.append(area["maximum_x_cm"] - area["minimum_x_cm"])
        self.depths.append(area["maximum_y_cm"] - area["minimum_y_cm"])
        if self.collect_scales:
            for observation in problem["observations"]:
                if observation["to_is_anchor"]:
                    self.scales.update(s["scale_cm"] for s in observation.get("calibrated_samples") or [] if valid_sample(s))
        for agent in problem["agent_ids"]:
            x, y = truth[agent][:2]
            inside = contains(area, x, y)
            self.agents += 1
            if agent in linked:
                self.linked += 1
                self.linked_outside += int(not inside)
            else:
                self.unlinked += 1
                self.unlinked_outside += int(not inside)
            if not inside and len(self.outside) < 20:
                self.outside.append(dict(unit=str(key), agent=agent, linked=agent in linked, x_cm=x, y_cm=y, area=area))

    def containment(self):
        return dict(units=self.units, agents=self.agents, linked_agents=self.linked, linked_outside=self.linked_outside,
                    unlinked_agents=self.unlinked, unlinked_outside=self.unlinked_outside, units_without_anchor_range=self.without_range,
                    units_with_an_unranged_agent=self.with_unranged_agent, units_bound_by_extent=self.extent_binding, first_outside=self.outside)

    def entry(self, heights=None, scales=None, **extra):
        """The record entry: the statistics of the rules, the settings they give and the containment check."""
        result = dict(heights_cm=height_statistics(heights if heights is not None else self.heights.values()),
                      anchor_scale_cm=dict(packets=int(sum(self.scales.values())), median=median_of_counts(self.scales))
                      if scales is None else scales, **extra)
        result["settings"] = settings_from(result)
        step = result["settings"]["grid_step"]
        result["containment"] = self.containment()
        result["area_cm"] = dict(width=spread(self.widths), depth=spread(self.depths), padding=spread(self.paddings),
                                 cells=spread(np.ceil(np.asarray(self.widths) / step) * np.ceil(np.asarray(self.depths) / step)))
        return result


def tiers_entry():
    """The anchor only units of the TIERS training blocks, calibrated track, as the runner builds them."""
    import layout_split
    import tiers_campaign as campaign
    folder = common.EXTERNAL / "tiers-static-campaign"
    base = json.load(gzip.open(folder / "base.json.gz"))
    catalog = json.load(gzip.open(folder / "catalog.json.gz"))
    half = layout_split.tiers_blocks("training")
    tally = Tally()
    for batch in catalog:
        if batch["track"] != "calibrated" or batch["mode"] != "ANC" or batch["block"] not in half:
            continue
        request, _ = campaign.batch_inputs(base, batch)
        coordinates = base["coordinates_cm"][batch["acquisition"]]
        for unit in request["units"]:
            truth = {agent: coordinates[str(agent)] for agent in unit["problem"]["agent_ids"]}
            tally.add((batch["key"], unit["id"], unit["samples"]), unit["problem"], truth, "tiers")
            for agent in unit["problem"]["agent_ids"]:
                tally.heights[(batch["acquisition"], agent)] = coordinates[str(agent)][2]
    sources = {name: common.sha256(folder / name) for name in ("base.json.gz", "catalog.json.gz")}
    return tally.entry(extent=EXTENT_SOURCES["tiers"], sources=sources)


def bregar_entry():
    """Per environment, the same-layout Bregar units of the training block, truth and heights from their reference."""
    import layout_split
    folder = common.EXTERNAL / "bregar-layout/training"
    request = json.load(gzip.open(folder / "request.json.gz"))
    reference = common.read_json(folder / "reference.json")
    meta = {u["id"]: u for u in reference["units"]}
    if {u["id"] for u in request["units"]} != layout_split.bregar_units("training"):
        raise ValueError("the Bregar training request does not hold the training identities")
    tallies = defaultdict(Tally)
    for unit in request["units"]:
        environment = meta[unit["id"]]["reference"]
        coordinates, heights = reference["coordinates_cm"][environment], reference["heights_cm"][environment]
        truth = {agent: coordinates[str(agent)] for agent in unit["problem"]["agent_ids"]}
        tallies[environment].add((unit["id"], unit["samples"]), unit["problem"], truth, "bregar")
        for agent in unit["problem"]["agent_ids"]:
            tallies[environment].heights[agent] = heights[str(agent)]
    sources = dict(request=common.sha256(folder / "request.json.gz"), reference=common.sha256(folder / "reference.json"))
    return dict(environments={name: tally.entry(extent=None) for name, tally in sorted(tallies.items())}, sources=sources)


def idlab_entry():
    """The anchor only units of the IDLab training campaign, built from the reserved calibration rows."""
    import idlab_campaign as campaign
    folder = common.EXTERNAL / "idlab-training-campaign"
    base = json.load(gzip.open(folder / "base.json.gz"))
    base["values"] = {(v[0], v[1]): v[2] for v in base["values"]}
    catalog = json.load(gzip.open(folder / "catalog.json.gz"))
    coordinates = base["coordinates_cm"]
    tally = Tally()
    for batch in catalog:
        if batch["mode"] != "ANC":
            continue
        request, _ = campaign.batch_inputs(base, batch)
        for unit in request["units"]:
            truth = {agent: coordinates[str(agent)] for agent in unit["problem"]["agent_ids"]}
            tally.add((batch["key"], unit["id"], unit["samples"]), unit["problem"], truth, "idlab")
            for agent in unit["problem"]["agent_ids"]:
                tally.heights[agent] = coordinates[str(agent)][2]
    sources = {name: common.sha256(folder / name) for name in ("base.json.gz", "catalog.json.gz", "freeze.json")}
    return tally.entry(extent=EXTENT_SOURCES["idlab"], sources=sources)


def bern_calibrated(values, coefficients):
    """Calibrated distances and scales of raw Bern timings, the conversion of fit_bern.calibrated_packets vectorized."""
    tof = np.asarray(values, dtype=float)
    tof = tof[np.isfinite(tof) & (tof >= coefficients["runtime_tof_floor_ns"]) & (tof <= coefficients["runtime_tof_max_ns"])]
    distance = np.maximum(0.0, coefficients["range_slope_cm_per_ns"] * tof + coefficients["range_offset_cm"])
    scale = np.clip(coefficients["scale_slope"] * distance + coefficients["scale_intercept_cm"], coefficients["scale_min_cm"], coefficients["scale_max_cm"])
    return distance, scale


def bern_problem(scenario, iteration, samples, coefficients):
    """One Bern anchor only unit in the prepared layout, calibrated with the given coefficients."""
    import fit_bern
    observations = []
    for key, value in iteration["Ranging samples to Anchors"].items():
        packets = fit_bern.calibrated_packets((value or {}).get(samples), coefficients)
        if packets:
            agent, anchor = map(int, key.split("-"))
            observations.append({"from": agent, "to": anchor, "to_is_anchor": True, "calibrated_samples": packets})
    anchors = [dict(id=int(k), position=dict(x=v["x"], y=v["y"]), height_cm=v["z"]) for k, v in scenario["True Positions of Anchors"].items()]
    return dict(agent_ids=[int(k) for k in scenario["True Positions of Agents"]], anchors=anchors, observations=observations)


def bern_entry():
    """Bern's training scans: every anchor only unit (scenario, iteration and sample count) against the profile's area,
    the fold areas against the units of their held scan, and the statistics of the rules for all scans and per fold.
    The units are those of the held-scan corpora built by the evaluation release's rules, the fold areas those of the
    fold fits. A reader's reproduction has neither the fold calibrations nor the fold fits, records of the held-scan
    rounds, so it builds the part of all eleven scans alone, the part the profiles read, and keeps the folds of the
    tracked record, which only the held-scan rounds and the ablation read."""
    import fit_bern
    full, _ = fit_bern.calibration(0)
    area = dict(common.read_json(BERN_DEPLOYMENT)["area"])
    folded = not common.READER
    fold_calibrations = {scan: fit_bern.calibration(scan)[0] for scan in fit_bern.SCAN_IDS} if folded else {}
    fold_areas = {scan: dict(common.read_json(fit_bern.SCANS / f"scan-{scan:02d}/fit.json")["profile"]["area"]) for scan in fit_bern.SCAN_IDS} \
        if folded else {}
    tally, rule_covers, rule_units, rule_outside = Tally(collect_scales=False), 0, 0, 0
    fold_tallies = {scan: Tally(collect_scales=False) for scan in fit_bern.SCAN_IDS}
    heights, timings = {}, {}
    for scan in fit_bern.SCAN_IDS:
        heights[scan], raw = {}, []
        with zipfile.ZipFile(common.BERN_ANCHOR_ONLY_CORPUS / f"scan-{scan:02d}/scenarios.zip") as archive:
            for name in sorted(archive.namelist()):
                scenario = json.loads(archive.read(name))
                fit_bern.check_scenario(scenario, scan)
                agents = {int(k): (v["x"], v["y"], v["z"]) for k, v in scenario["True Positions of Agents"].items()}
                heights[scan].update({agent: position[2] for agent, position in agents.items()})
                if not scenario["True Positions of Anchors"]:
                    continue
                for iteration in scenario["Iterations"]:
                    for samples in ("1", "10"):
                        for value in iteration["Ranging samples to Anchors"].values():
                            packet = (value or {}).get(samples)
                            if packet is not None:
                                raw.extend(packet if isinstance(packet, list) else [packet])
                        problem = bern_problem(scenario, iteration, samples, full)
                        key = (scan, scenario["Research Metadata"]["scenario_id"], iteration["ID"], samples)
                        evidence = anchor_evidence(problem)
                        tally.add(key, problem, agents, "bern", area=area, evidence=evidence)
                        if folded:
                            fold_tallies[scan].add(key, problem, agents, "bern", area=fold_areas[scan], evidence=evidence)
                        anchors = [(a["position"]["x"], a["position"]["y"]) for a in problem["anchors"]]
                        largest_range, largest_scale, linked = evidence
                        if largest_range is None:
                            continue
                        box = padded_box(anchors, largest_range + PADDING_SCALES * largest_scale)
                        rule_units += 1
                        rule_covers += int(intersect(box, area) == area)
                        rule_outside += sum(not contains(box, *agents[agent][:2]) for agent in linked)
        timings[scan] = np.asarray(raw, dtype=float)
        print(f"   Bern scan {scan} read", flush=True)

    def scales(scans, coefficients):
        _, scale = bern_calibrated(np.concatenate([timings[s] for s in scans]), coefficients)
        return dict(packets=int(len(scale)), median=float(np.median(scale)))

    def nodes(scans):
        merged = {}
        for s in scans:
            merged.update(heights[s])
        return list(merged.values())

    result = tally.entry(heights=nodes(fit_bern.SCAN_IDS), scales=scales(fit_bern.SCAN_IDS, full), extent=EXTENT_SOURCES["bern"], area=area,
                         rule=dict(units_with_anchor_range=rule_units, units_whose_padded_box_covers_the_area=rule_covers,
                                   linked_agents_outside_the_padded_box=rule_outside),
                         sources=dict(bern_deployment=common.sha256(BERN_DEPLOYMENT),
                                      scans={f"scan-{s:02d}": common.sha256(common.BERN_ANCHOR_ONLY_CORPUS / f"scan-{s:02d}/scenarios.zip") for s in fit_bern.SCAN_IDS}))
    if not folded:
        tracked = common.read_json(RECORD)["datasets"].get("bern", {}).get("folds") if RECORD.exists() else None
        if tracked is not None:
            result["folds"] = tracked
        print(f"   Bern folds {'kept from the tracked record' if tracked is not None else 'left out'}, a reader's data holds no fold "
              "calibrations", flush=True)
        return result
    result["folds"] = {}
    for held in fit_bern.SCAN_IDS:
        training = [s for s in fit_bern.SCAN_IDS if s != held]
        result["folds"][str(held)] = fold_tallies[held].entry(
            heights=nodes(training), scales=scales(training, fold_calibrations[held]), area=fold_areas[held],
            extent=f"fold area of held scan {held}, research/data/anc-tail-generalization/scan-{held:02d}/fit.json",
            containment_scope=f"the units of held scan {held}, where the fold profile is applied")
    return result


BUILDERS = dict(bern=bern_entry, tiers=tiers_entry, bregar=bregar_entry, idlab=idlab_entry)


def statistics(item):
    """A dataset's entry without the SHA-256 of the files it was read from, what a reader's check compares (SOLID_READER=1):
    a reader's halves hold the paper's values in files whose bytes can differ by platform, the gzip header for one."""
    return item if item is None else {key: value for key, value in item.items() if key != "sources"}


def build(datasets, previous=None):
    """The record with fresh entries for `datasets` and the previous record's entries for the rest."""
    kept = {} if previous is None else {name: value for name, value in previous["datasets"].items() if name not in datasets}
    fresh = {}
    for dataset in datasets:
        started = time.time()
        fresh[dataset] = BUILDERS[dataset]()
        print(f"{dataset} read in {time.time() - started:.0f}s", flush=True)
    merged = {**kept, **fresh}
    return dict(schema=SCHEMA, rules=rules(), split_record_sha256=split_record_sha256(), statement="research/solid/deployment.py",
                datasets={name: merged[name] for name in DATASETS if name in merged})


def entries(record):
    for dataset, item in record["datasets"].items():
        if dataset == "bregar":
            for environment, value in item["environments"].items():
                yield f"bregar {environment}", value
        else:
            yield dataset, item
            for held, value in item.get("folds", {}).items():
                yield f"bern held scan {held}", value


def report(record):
    """Print the areas, the containment, the heights and the budgets per deployment. Returns the count of agents outside."""
    outside = 0
    for name, item in entries(record):
        c, s, a = item["containment"], item["settings"], item["area_cm"]
        outside += c["linked_outside"] + c["unlinked_outside"]
        print(f"{name}: {c['units']} units, {c['agents']} agents ({c['linked_agents']} with an anchor link), outside the area "
              f"{c['linked_outside']} linked and {c['unlinked_outside']} of {c['unlinked_agents']} unlinked, "
              f"{c['units_without_anchor_range']} units without anchor range, {c['units_with_an_unranged_agent']} with an agent without, "
              f"{c['units_bound_by_extent']} bound by the extent")
        print(f"   area width {a['width'][0]:.0f}/{a['width'][1]:.0f}/{a['width'][2]:.0f} cm, depth {a['depth'][0]:.0f}/{a['depth'][1]:.0f}/{a['depth'][2]:.0f} cm, "
              f"cells {a['cells'][0]:.0f}/{a['cells'][1]:.0f}/{a['cells'][2]:.0f} (min/median/max)")
        print(f"   heights p5 {item['heights_cm']['p5']:.1f} median {item['heights_cm']['median']:.1f} p95 {item['heights_cm']['p95']:.1f} "
              f"over {item['heights_cm']['positions']} positions, nodes {s['height_nodes']}")
        print(f"   median packet scale {item['anchor_scale_cm']['median']:.2f} cm, grid step {s['grid_step']} cm, "
              f"joint samples {s['joint_samples']}, seed {s['seed']}")
        if "rule" in item:
            print(f"   padded box covers the area on {item['rule']['units_whose_padded_box_covers_the_area']} of "
                  f"{item['rule']['units_with_anchor_range']} units, linked agents outside the padded box {item['rule']['linked_agents_outside_the_padded_box']}")
    return outside


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--check", action="store_true", help="rebuild and compare with the tracked record instead of writing it")
    args = parser.parse_args()
    datasets = [d for d in args.datasets.split(",") if d]
    if set(datasets) - set(DATASETS):
        parser.error(f"unknown datasets {sorted(set(datasets) - set(DATASETS))}")
    previous = common.read_json(RECORD) if RECORD.exists() else None
    record = build(datasets, previous if previous and previous.get("schema") == SCHEMA and previous.get("rules") == rules() else None)
    outside = report(record)
    if args.check:
        # a reader's reproduction compares the statistics the rules compute with the tracked record's
        view = statistics if common.READER else (lambda item: item)
        if previous is None or {k: view(previous["datasets"].get(k)) for k in datasets} != {k: view(record["datasets"][k]) for k in datasets} \
                or {k: v for k, v in previous.items() if k != "datasets"} != {k: v for k, v in record.items() if k != "datasets"}:
            raise SystemExit("the tracked deployment record differs from the rules applied to the training halves")
        print("deployment record verified")
    else:
        common.write_json(RECORD, record)
        print(f"written {RECORD.relative_to(common.ROOT)}")
    if outside:
        raise SystemExit(f"{outside} true agent positions lie outside their unit's operating area")


# ---------------------------------------------------------------------------------------------------------------------
# Held-out benchmarks
#
# HYMN and AAU Meshed UWB are declared in research/artifacts/solid/splits/heldout-benchmarks.json and never inform a
# design decision. Their deployment settings follow the same rules from their training halves, in a record of their own,
# research/artifacts/solid/fits/heldout-deployment.json, so the tracked record above and its readers stay unchanged. AAU
# publishes the 5 m by 8 m area of its static nodes, bounded like TIERS and IDLab by its diagonal around every known
# anchor. HYMN publishes no hall extent in its archive or pages, so its units keep the padded anchor box, and a unit with
# an agent without any admissible anchor sample takes the anchor box padded by the end of the admissible interval.

HELDOUT_RECORD = common.FITS / "heldout-deployment.json"
HELDOUT_SCHEMA = "solid.deployment.heldout.v1"
HELDOUT_DATASETS = ("hymn", "aau")
HELDOUT_EXTENTS_CM = {"aau": (500.0, 800.0)}
HELDOUT_EXTENT_SOURCES = {"aau": "published 5 m by 8 m area of the static nodes, diagonal around every known anchor"}


def heldout_rules():
    """The shared rules with the held-out extents, stored with the held-out record."""
    return dict(rules(), heldout_extents_cm={name: list(size) for name, size in HELDOUT_EXTENTS_CM.items()})


def heldout_extent_bound(dataset, anchors):
    """The bound a held-out dataset's published extent puts on a unit with these known anchors, None without one."""
    if dataset not in HELDOUT_DATASETS:
        raise ValueError(f"{dataset} is not a held-out dataset")
    if dataset in HELDOUT_EXTENTS_CM:
        return diagonal_bound(anchors, *HELDOUT_EXTENTS_CM[dataset])
    return None


def heldout_unit_area(problem, dataset):
    """The operating area of one prepared anchor only unit of a held-out dataset, the rule of unit_area. A unit of a
    dataset without a published extent that holds an agent without any admissible anchor sample, or no admissible
    anchor sample at all, takes the box of its anchors padded by the end of the admissible interval, the longest range
    an admissible reading can take. Such an agent carries no range that bounds its place, so the box padded by the
    ranges of the other agents need not hold it, and unit_area has no area without any range."""
    anchors = [(a["position"]["x"], a["position"]["y"]) for a in problem["anchors"]]
    largest_range, largest_scale, ranged = anchor_evidence(problem)
    extent = heldout_extent_bound(dataset, anchors)
    every_agent_ranged = set(problem["agent_ids"]) <= ranged
    if extent is None and (largest_range is None or not every_agent_ranged):
        import heldout
        return padded_box(anchors, heldout.MAXIMUM_RANGE_CM)
    return operating_area(anchors, largest_range, largest_scale, extent, every_agent_ranged)


def heldout_declaration_sha256():
    import heldout
    return common.sha256(heldout.RECORD)


def heldout_training_inputs(dataset):
    """The request and the reference of a held-out training half as its builder wrote them, checked against the
    declared plan less the units of the flights the build left out beyond the relocation guard."""
    import heldout
    request, reference, identity = heldout.built_half(dataset, "training")
    left_out = sum(entry["units"] for entry in identity["excluded_flights"].values())
    if len({u["id"] for u in request["units"]}) + left_out != heldout.section(dataset)["plan"]["training"]["identities"]:
        raise ValueError(f"the {dataset} training request does not hold the declared units")
    return request, reference, identity


def heldout_tally(dataset, units, truth, heights):
    """Containment and the rules' statistics over the units of a held-out training half, with the held-out areas.
    truth maps an agent to its true (x, y) and heights to its true height."""
    tally = Tally()
    for unit in units:
        problem = unit["problem"]
        anchors = [(a["position"]["x"], a["position"]["y"]) for a in problem["anchors"]]
        evidence = anchor_evidence(problem)
        area = heldout_unit_area(problem, dataset)
        extent = heldout_extent_bound(dataset, anchors)
        if extent is not None and evidence[0] is not None and area != padded_box(anchors, evidence[0] + PADDING_SCALES * evidence[1]):
            tally.extent_binding += 1
        tally.add((unit["id"], unit["samples"]), problem, {agent: truth[agent] for agent in problem["agent_ids"]}, dataset, area=area, evidence=evidence)
        for agent in problem["agent_ids"]:
            tally.heights[agent] = heights[agent]
    return tally


def heldout_entry(dataset):
    """The deployment entry of a held-out dataset from the units of its training half, truth and heights from their
    reference. The HYMN agents are the training points and the AAU agents the nine static nodes."""
    request, reference, sources = heldout_training_inputs(dataset)
    truth = {int(k): tuple(v[:2]) for k, v in reference["positions_cm"].items()}
    heights = {int(k): float(v[2]) for k, v in reference["positions_cm"].items()}
    tally = heldout_tally(dataset, request["units"], truth, heights)
    return tally.entry(extent=HELDOUT_EXTENT_SOURCES.get(dataset), sources=sources)


def heldout_build(datasets, previous=None):
    """The held-out record with fresh entries for `datasets` and the previous record's entries for the rest."""
    kept = {} if previous is None else {name: value for name, value in previous["datasets"].items() if name not in datasets}
    fresh = {dataset: heldout_entry(dataset) for dataset in datasets}
    merged = {**kept, **fresh}
    return dict(schema=HELDOUT_SCHEMA, rules=heldout_rules(), declaration_sha256=heldout_declaration_sha256(),
                statement="research/solid/deployment.py", datasets={name: merged[name] for name in HELDOUT_DATASETS if name in merged})


def heldout_load(path=None):
    """The held-out deployment record, refused when it was built under other rules or another declaration."""
    record = common.read_json(path or HELDOUT_RECORD)
    if record.get("schema") != HELDOUT_SCHEMA:
        raise ValueError(f"{path or HELDOUT_RECORD} is not a held-out deployment record")
    if record.get("rules") != heldout_rules():
        raise ValueError("The held-out deployment record was built under other rules. Rebuild it with research/solid/fit_heldout.py deployment")
    if record.get("declaration_sha256") != heldout_declaration_sha256():
        raise ValueError("The held-out deployment record was built under another declaration. Rebuild it with research/solid/fit_heldout.py deployment")
    return record


def heldout_settings(dataset, record=None):
    """The anchor only deployment settings of a held-out dataset, its height prior, grid step and budgets."""
    record = record or heldout_load()
    if dataset not in record["datasets"]:
        raise ValueError(f"the held-out deployment record holds no entry for {dataset}")
    item = record["datasets"][dataset]
    derived = settings_from(item)
    if item.get("settings") != derived:
        raise ValueError(f"The held-out deployment record's settings of {dataset} differ from its rules")
    return derived


if __name__ == "__main__":
    main()
