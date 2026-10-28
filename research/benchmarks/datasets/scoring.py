"""Pair sets of every dataset from the inputs alone, their coverage, and tables per anchor count.

The released TWR-CLOUD-Bern scorer scores anchor free pairs only within one connected component of the successful
agent links, and every agent pair in the anchored modes. The external campaigns score every agent pair in every
mode (pilot.score_records), including pairs that no method can resolve from its inputs: agents of different
components in the anchor free mode, and agents with no path to an anchor in the anchored modes. The functions here
derive the pair sets from a unit's inputs alone, never from truth or from any method's output:

- connected: anchor free pairs within one connected component of successful agent links, the released rule.
- anchored-1: anchor aided and anchor only pairs whose two agents both reach at least one successful anchor link
  through successful links.
- anchored-3: the same with at least three reachable anchor links, the count that fixes a planar position.
- joined: pairs whose two agents the unit's measured links connect, directly or through other agents or anchors,
  with every anchor counted as one node, so that the pair is connected by measured links. The measurements need not
  fix the distance of such a pair. In the anchor only mode one anchor link per agent joins a pair, and in the anchor
  aided mode two agents may be joined through two different anchors. In the anchor free mode this is the released
  component rule. In the anchor aided mode it also holds the pairs inside a component without any anchor link and
  every pair of agents that reach the anchors. In the anchor only mode every agent is positioned alone, so a pair is
  joined when both agents have at least one measured anchor link.
- ranged: the joined pairs with only successful links joining, so that the pair is connected by links that carry a
  range. An answered link without a range joins a pair of the joined set, and only SOLID's presence term reads such
  a link. In the anchor free mode this is the connected variant, and in the anchor only mode anchored-1.

For every variant but joined a link is successful when its observation holds at least one valid sample: a finite,
non negative calibrated distance with a positive, finite scale, or a finite time of flight between 0 and max_tof_ns.
For the joined variant a link is measured when its observation exists, even with no admissible sample, as the engine
counts an observation as a responded exchange. Observations between unknown nodes are ignored, as the released
scorer ignores edges outside the agent set. Agent links connect agents in the anchor aided mode. In the anchor only
mode every agent reaches only its own anchor links. Anchor links are counted as distinct (agent, anchor) pairs of the
agent's component.

The module also rescores units from the per pair errors pilot.score_records keeps, reports the coverage of every
variant, and tabulates balanced percentiles per anchor count with a headline over units with three or more anchors.
For scenarios in the released format (TWR-CLOUD-Bern and its held-scan corpora) released_problem reads a unit's
measured links and their samples as the engine prepares them, and score_scenarios scores stored predictions under the
released rule and the joined and ranged pair sets with the released scorer's arithmetic, without solving again.

One coordinate rule holds for every method and dataset wherever predictions are read for scoring, before any scorer,
the released one included. A predicted coordinate that is not finite, or whose magnitude exceeds COORDINATE_BOUND_CM,
is missing, and so is its point (scored_point). The released rule then charges the point as it charges a missing one,
the penalty on every pair it joins, counted among the penalized pairs. score_pairs applies the rule before the
released scorer's arithmetic, which the external campaigns (pilot.score_records) and score_scenarios score with, and
bound_prediction_csv applies it to a prediction file in the released layout before the released scorer reads it.
Below the bound every score is the released scorer's, bit for bit.

Every reported table and every selection of a baseline's settings scores the joined pair set (PAIR_SET). A pair is
scored only when the measurements connect its two devices, and the table reports the share it scores. The ranged pair
set is reported beside it as a supplementary figure (SUPPLEMENTARY), and the headline stays joined. The anchor only
headline covers the units with three or more anchors (HEADLINE_MIN_ANCHORS), and the units with one and with two
anchors are reported alone. summary gives, per sample track and reported group, the balanced S50, S75 and S90 under
each reported pair set with the share of pairs and units it scores, its pairs over every agent pair of the same units
and the units that hold at least one of its pairs. Both shares follow from the inputs alone, so every method has the
same. TWR-CLOUD-Bern keeps the released scorer's figure, the benchmark's own, and scenario_tables reports the joined
and ranged figures and their shares beside it.
"""
import csv
import gzip
import io
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path

MAX_TOF_NS = 200.0
MODES = {"AnchorFree": "AF", "AnchorAided": "AAC", "AnchorOnly": "ANC"}
ENGINE_MODES = {short: name for name, short in MODES.items()}
VARIANTS = {"AF": ("connected", "joined", "ranged"), "AAC": ("anchored-1", "anchored-3", "joined", "ranged"),
            "ANC": ("anchored-1", "anchored-3", "joined", "ranged")}
MINIMUM_LINKS = {"anchored-1": 1, "anchored-3": 3}
# The pair set of every reported table and of every selection of a baseline's settings, and the fewest anchors of a unit
# in the anchor only headline. The anchor only units with fewer anchors are reported alone, one row per anchor count.
PAIR_SET = "joined"
HEADLINE_MIN_ANCHORS = 3
# The pair sets every table reports beside the headline's, supplementary figures that decide nothing
SUPPLEMENTARY = ("ranged",)
REPORTED = (PAIR_SET, *SUPPLEMENTARY)
# The largest magnitude in centimetres of a predicted coordinate that the scoring reads, the engine's bound on a sane
# coordinate (MAX_COORDINATE_CM of runtime/engine/src/model.rs), 100 km. A coordinate beyond it is missing (scored_point).
COORDINATE_BOUND_CM = 1.0e7


def mode_of(problem):
    return MODES[problem["mode"]]


def valid_samples(observation, max_tof_ns=MAX_TOF_NS):
    """The number of usable samples of one observation."""
    count = 0
    for sample in observation.get("calibrated_samples") or []:
        distance, scale = sample.get("distance_cm"), sample.get("scale_cm")
        if (distance is not None and scale is not None and math.isfinite(distance) and distance >= 0
                and math.isfinite(scale) and scale > 0):
            count += 1
    for tof in observation.get("tof_ns") or []:
        if tof is not None and math.isfinite(tof) and 0 <= tof <= max_tof_ns:
            count += 1
    return count


def successful_links(problem, max_tof_ns=MAX_TOF_NS):
    """Agent links as sorted pairs and anchor links per agent, of the observations with a usable sample."""
    agents = set(problem["agent_ids"])
    anchors = {anchor["id"] for anchor in problem.get("anchors") or []}
    agent_links, anchor_links = set(), defaultdict(set)
    for observation in problem["observations"]:
        if not valid_samples(observation, max_tof_ns):
            continue
        a, b = observation["from"], observation["to"]
        if observation["to_is_anchor"]:
            if a in agents and b in anchors:
                anchor_links[a].add(b)
        elif a in agents and b in agents and a != b:
            agent_links.add((min(a, b), max(a, b)))
    return agent_links, dict(anchor_links)


def components(agents, links):
    """Connected components of the agents over the links, each sorted, in the order of their smallest agent."""
    parent = {agent: agent for agent in agents}

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for a, b in sorted(links):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = defaultdict(list)
    for agent in agents:
        groups[root(agent)].append(agent)
    return sorted((sorted(group) for group in groups.values()), key=lambda group: group[0])


def connected_pairs(problem, max_tof_ns=MAX_TOF_NS):
    """Anchor free pairs within one component of successful agent links, the released scorer's rule."""
    links, _ = successful_links(problem, max_tof_ns)
    return sorted(pair for group in components(problem["agent_ids"], links) for pair in combinations(group, 2))


def anchor_reach(problem, max_tof_ns=MAX_TOF_NS):
    """Per agent, the anchor links and the distinct anchors reachable through successful links."""
    links, anchor_links = successful_links(problem, max_tof_ns)
    if mode_of(problem) == "ANC":
        links = set()  # positioned alone: an agent reaches its own anchor links only
    reach = {}
    for group in components(problem["agent_ids"], links):
        pairs = {(agent, anchor) for agent in group for anchor in anchor_links.get(agent, ())}
        entry = dict(links=len(pairs), anchors=len({anchor for _, anchor in pairs}), component=len(group))
        reach.update({agent: entry for agent in group})
    return reach


def anchored_pairs(problem, minimum_links, max_tof_ns=MAX_TOF_NS):
    """Pairs whose two agents both reach at least minimum_links successful anchor links."""
    reach = anchor_reach(problem, max_tof_ns)
    anchored = sorted(agent for agent in problem["agent_ids"] if reach[agent]["links"] >= minimum_links)
    return list(combinations(anchored, 2))


def joined_pairs(problem, ranged=False, max_tof_ns=MAX_TOF_NS):
    """Pairs of agents joined by the unit's measured links, in ascending order.

    Every agent and one node standing for all anchors form a graph. An observation joins its two ends whenever it
    exists, whatever its samples: an agent link in the anchor free and anchor aided modes, an anchor link in the
    anchor aided and anchor only modes. Two agents form a pair when a path of such links connects them. With ranged an
    observation joins only when it holds at least one valid sample, the ranged pair set."""
    mode = mode_of(problem)
    agents = sorted(problem["agent_ids"])
    index = {agent: i for i, agent in enumerate(agents)}
    frame = len(agents)
    parent = list(range(len(agents) + 1))

    def root(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def join(a, b):
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    anchors = {anchor["id"] for anchor in problem.get("anchors") or []}
    for observation in problem["observations"]:
        a, b = observation["from"], observation["to"]
        if a not in index or (ranged and not valid_samples(observation, max_tof_ns)):
            continue
        if observation["to_is_anchor"]:
            if mode != "AF" and b in anchors:
                join(index[a], frame)
        elif mode != "ANC" and b in index and a != b:
            join(index[a], index[b])
    groups = defaultdict(list)
    for agent in agents:
        groups[root(index[agent])].append(agent)
    return sorted(pair for group in groups.values() for pair in combinations(group, 2))


def variant_pairs(problem, variant, max_tof_ns=MAX_TOF_NS):
    """The pairs of one variant of one unit."""
    if variant == "all":
        return list(combinations(sorted(problem["agent_ids"]), 2))
    if variant == "connected":
        return connected_pairs(problem, max_tof_ns)
    if variant == "joined":
        return joined_pairs(problem)
    if variant == "ranged":
        return joined_pairs(problem, ranged=True, max_tof_ns=max_tof_ns)
    return anchored_pairs(problem, MINIMUM_LINKS[variant], max_tof_ns)


def pair_sets(problem, max_tof_ns=MAX_TOF_NS):
    """Every pair set of one unit: all pairs and the variants of its mode, each a sorted list of agent pairs."""
    mode = mode_of(problem)
    result = {"all": variant_pairs(problem, "all", max_tof_ns)}
    for variant in VARIANTS[mode]:
        result[variant] = variant_pairs(problem, variant, max_tof_ns)
    return result


def coverage(units, max_tof_ns=MAX_TOF_NS):
    """Per variant over the units of one mode: possible and eligible pairs, their ratio and the units left without pairs."""
    totals = {}
    for unit in units:
        sets = pair_sets(unit["problem"], max_tof_ns)
        possible = len(sets["all"])
        for variant, pairs in sets.items():
            entry = totals.setdefault(variant, dict(units=0, possible_pairs=0, eligible_pairs=0, units_without_pairs=0))
            entry["units"] += 1
            entry["possible_pairs"] += possible
            entry["eligible_pairs"] += len(pairs)
            entry["units_without_pairs"] += not pairs
    for entry in totals.values():
        entry["coverage"] = entry["eligible_pairs"] / entry["possible_pairs"] if entry["possible_pairs"] else math.nan
    return totals


def scored_coordinate(value):
    """Whether the scoring reads a predicted coordinate, a finite number of magnitude at most COORDINATE_BOUND_CM."""
    return value is not None and math.isfinite(value) and abs(value) <= COORDINATE_BOUND_CM


def scored_point(point, q=2):
    """A predicted point as every scorer reads it, the point itself or None where it is missing.

    The coordinate rule of every method and dataset: a predicted coordinate that is not finite, or whose magnitude
    exceeds COORDINATE_BOUND_CM, is missing, and a point with such a coordinate, or without q coordinates, is missing.
    The released TWR-CLOUD-Bern scorer charges a missing point the penalty, the cap of every pair error, on every pair
    it joins and counts those pairs as penalized, so a point beyond the bound is charged and counted as a missing one.
    Its arithmetic squares coordinate differences, which Python refuses beyond about 1.3e154, so without the rule a
    finite but enormous coordinate would stop the scoring. A coordinate that is not finite is missing under the
    released rule already, and a point within the bound is read as it is."""
    if point is None or len(point) != q or not all(scored_coordinate(value) for value in point):
        return None
    return point


def released_coordinate(text):
    """A coordinate of a prediction file as the released scorer parses it (parse_float), not a number where it is blank,
    a dash or no number."""
    text = text.strip()
    if text in ("-", ""):
        return math.nan
    try:
        return float(text)
    except ValueError:
        return math.nan


def bound_prediction_csv(path):
    """Applies the coordinate rule of scored_point, in place, to a prediction file in the released scorer's layout before
    a scorer reads it, the released scorer included.

    A row whose coordinates the released scorer reads as finite, x and y and also z where q is 3, one of them beyond
    COORDINATE_BOUND_CM, gets its coordinates blank, as the solid CLI writes a missing prediction, so that every scorer
    of the file charges the point as missing and counts its pairs among the penalized ones. A row that the released rule
    already treats as missing keeps its bytes, and so does every other row and a file without the coordinate columns,
    which the released scorer refuses. The file is not written unless a row changes, so a file within the bound stays as
    it is. The file is a copy the caller extracted for the scorer, and the
    stored predictions keep what the method returned. Returns the number of rows made missing."""
    path = Path(path)

    def lines(stream):
        """Every line of the file with its replacement, None for a line that keeps its bytes, the header first."""
        header = stream.readline()
        yield header, None
        columns = next(csv.reader([header]), [])
        # a file without the coordinate columns is the released scorer's to refuse
        if "x" not in columns or "y" not in columns:
            for line in stream:
                yield line, None
            return
        x, y = columns.index("x"), columns.index("y")
        q = columns.index("q") if "q" in columns else None
        z = columns.index("z") if "z" in columns else None
        for line in stream:
            text = line.rstrip("\r\n")
            quoted = '"' in text
            fields = next(csv.reader([text])) if quoted else text.split(",")
            three = q is not None and q < len(fields) and fields[q].strip() and int(fields[q]) == 3
            indexes = (x, y, z) if three else (x, y)
            # a blank or short line and a point of three coordinates without a z column are the released scorer's to judge
            if not text.strip() or None in indexes or max(indexes) >= len(fields):
                yield line, None
                continue
            values = [released_coordinate(fields[i]) for i in indexes]
            if all(math.isfinite(value) for value in values) and scored_point(values, len(values)) is None:
                for i in indexes:
                    fields[i] = ""
                if quoted:
                    buffer = io.StringIO()
                    csv.writer(buffer, lineterminator="").writerow(fields)
                    yield line, buffer.getvalue() + line[len(text):]
                else:
                    yield line, ",".join(fields) + line[len(text):]
            else:
                yield line, None

    with path.open(newline="") as stream:
        changed = sum(1 for _, replacement in lines(stream) if replacement is not None)
    if changed:
        staged = path.with_name(path.name + ".bounded")
        with path.open(newline="") as stream, staged.open("w", newline="") as target:
            for line, replacement in lines(stream):
                target.write(line if replacement is None else replacement)
        staged.replace(path)
    return changed


def score_pairs(scorer, truth, estimated, pairs, penalty, q=2):
    """The released scorer's score_pairs under the coordinate rule, its pair count, penalized pair count and mean pair
    error.

    Every estimated point that the rule makes missing (scored_point) is left out before the released scorer's
    arithmetic reads the points, so the released rule charges it the penalty on every pair it joins and counts those
    pairs as penalized, as it does a missing point, whatever the method. Within the bound the result is the released
    scorer's, bit for bit. A pair whose estimated distance still overflows the released arithmetic, which the bound
    leaves no room for, is charged the penalty and counted as penalized, while every other pair keeps the released
    scorer's own error and the mean sums the errors in the order of `pairs`, as the released scorer does."""
    pairs = list(pairs)
    estimated = {node: point for node, point in estimated.items() if scored_point(point, q) is not None}
    try:
        return scorer.score_pairs(truth, estimated, pairs, penalty, q)
    except OverflowError:
        pass
    errors, penalized = [], 0
    for pair in pairs:
        try:
            _, missing, error = scorer.score_pairs(truth, estimated, [pair], penalty, q)
        except OverflowError:
            missing, error = 1, penalty
        errors.append(error)
        penalized += missing
    return len(errors), penalized, sum(errors) / len(errors)


def rescore(row, pairs):
    """The mean pair error of one scored unit over a subset of its pairs, from the per pair errors; None when empty.

    row holds pilot.score_records' pairs and pair_errors_cm, so an estimate missing a position keeps the fixed
    penalty on the pairs it touches, as in the released scorer.
    """
    if not pairs:
        return None
    # the pilot lists pairs in the unit's agent order, the pair sets in ascending order
    errors = {tuple(sorted(pair)): error for pair, error in zip(row["pairs"], row["pair_errors_cm"])}
    return sum(errors[tuple(sorted(pair))] for pair in pairs) / len(pairs)


def score_with_pair_sets(request, reference, records, scorer, methods, max_tof_ns=MAX_TOF_NS):
    """pilot.score_records rows with the mean pair error and the eligible pair count of every pair set of the unit."""
    import pilot
    units = {(unit["id"], unit["samples"]): unit for unit in request["units"]}
    rows = pilot.score_records(request, reference, records, scorer, methods)
    cache = {}
    for row in rows:
        key = (row["id"], row["samples"])
        if key not in cache:
            cache[key] = pair_sets(units[key]["problem"], max_tof_ns)
        row["pair_sets"] = {variant: dict(eligible_pairs=len(pairs), mean_pair_error_cm=rescore(row, pairs))
                            for variant, pairs in cache[key].items()}
    return rows


def rescore_variant(row, problem, variant):
    """The mean pair error of one scored unit over one pair set of its inputs, from pilot.score_records' per pair
    errors, or None when the unit has no pair of the set. Stored per pair errors score any variant again without
    solving."""
    return rescore(row, variant_pairs(problem, variant))


# ------------------------------------------------------------------------------------------- reported tables
def report_groups(rows, mode, minimum_anchors=HEADLINE_MIN_ANCHORS):
    """The rows of one mode as a table reports them, each group as (label, rows) and empty groups left out. The anchor
    only mode reports the headline over the units with minimum_anchors or more anchors first and then every smaller
    anchor count alone, the other modes every unit together. Rows of the anchor only mode carry their anchor count."""
    rows = list(rows)
    if mode != "ANC":
        groups = [("every cell", rows)]
    else:
        groups = [(f"headline, {minimum_anchors} or more anchors", [row for row in rows if row["anchors"] >= minimum_anchors])]
        groups += [(f"anchors-{k}", [row for row in rows if row["anchors"] == k])
                   for k in sorted({row["anchors"] for row in rows if row["anchors"] < minimum_anchors})]
    return [(label, part) for label, part in groups if part]


def unit_scores(rows, pair_set=PAIR_SET):
    """(cell, error, pairs, possible pairs) of every scored row under a pair set, from the pair sets the row carries
    (score_with_pair_sets). The error is the unit's mean pair error over the set, None without any pair of it, and
    the possible pairs are every agent pair of the unit."""
    result = []
    for row in rows:
        entry = row["pair_sets"][pair_set]
        result.append((row["cell"], entry["mean_pair_error_cm"], entry["eligible_pairs"], row["pair_sets"]["all"]["eligible_pairs"]))
    return result


def table(units, percentile, pair_set=PAIR_SET, levels=(0.5, 0.75, 0.9)):
    """The balanced percentiles of one group of units under a pair set and the share of pairs and units it scores.

    units are (cell, error, pairs, possible pairs) tuples, as unit_scores gives them. A unit without a pair of the set
    has no error and is left out of the percentiles, as the released scorer leaves out a scenario without eligible
    pairs. The pair share is the pairs of the set over every agent pair of the units, and the unit share the units
    holding at least one of its pairs over all units. Both follow from the inputs alone, so every method has the same.
    The cells enter the balance in the order they first appear."""
    cells = defaultdict(list)
    for cell, error, _, _ in units:
        if error is not None:
            cells[cell].append(error)
    pairs, possible = sum(unit[2] for unit in units), sum(unit[3] for unit in units)
    scored = sum(1 for unit in units if unit[2])
    return dict(pair_set=pair_set, units=len(units), scored_units=scored, unit_share=scored / len(units) if units else None,
                pairs=pairs, possible_pairs=possible, pair_share=pairs / possible if possible else None, cells=len(cells),
                balanced=[float(percentile(cells, q)) for q in levels] if cells else None)


def summary(rows, mode, percentile=None, groups=None, pair_sets=REPORTED):
    """The tables of one method's scored rows of a mode per sample track, reported group and pair set, the joined pairs
    of the headline and the supplementary ranged pairs by default. Each holds the balanced S50, S75 and S90 and the
    share of pairs and units the set scores.

    groups splits the rows of a track into (label, rows), as report_groups does by default. The released scorer's
    balanced percentile unless another is given. Returns {track: {group: {pair set: table}}}."""
    percentile = percentile or released_percentile()
    groups = groups or (lambda part: report_groups(part, mode))
    result = {}
    for samples in sorted({row["samples"] for row in rows}):
        track = [row for row in rows if row["samples"] == samples]
        result[str(samples)] = {label: {name: table(unit_scores(part, name), percentile, name) for name in pair_sets}
                                for label, part in groups(track) if part}
    return result


def describe(entry):
    """One table in words, its balanced S50, S75 and S90 and the share of pairs and units its pair set scores."""
    values = " / ".join(f"{v:.1f}" for v in entry["balanced"]) if entry["balanced"] else "none"

    def share(value):
        return "none" if value is None else f"{value:.1%}"
    return (f"{entry['pair_set']} S50/S75/S90 = {values}, {entry['pairs']} of {entry['possible_pairs']} pairs ({share(entry['pair_share'])}) "
            f"and {entry['scored_units']} of {entry['units']} units ({share(entry['unit_share'])}) scored")


# ------------------------------------------------------------------------------------------- the released format
def measured(value):
    """Whether one sample value of a scenario in the released format gives the engine an observation: a number, a
    list holding a number, or calibrated samples. The engine skips a missing value, a list without any number and an
    empty set of calibrated samples, and keeps every other value whether or not its samples are admissible."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, list):
        return any(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    if isinstance(value, dict):
        return bool(value.get("calibrated_samples"))
    return False


def released_samples(value):
    """The times of flight and the calibrated samples of one measured value in the released format, as the engine reads
    them. Every number of a number or of a list is a time of flight in nanoseconds, and a set holds calibrated samples."""
    if isinstance(value, dict):
        return [], list(value.get("calibrated_samples") or [])
    items = value if isinstance(value, list) else [value]
    return [float(item) for item in items if isinstance(item, (int, float)) and not isinstance(item, bool)], []


def released_problem(scenario, iteration, samples, mode):
    """One unit of a scenario in the released format with the links the engine prepares from it and their samples,
    enough for the pair sets: agent links except in the anchor only mode, anchor links except in the anchor free mode,
    each present when its value for the sample count is measured. mode is AF, AAC or ANC."""
    agents = [int(node) for node in scenario.get("Agents", [])]
    anchors = [] if mode == "AF" else [dict(id=int(node)) for node in scenario.get("True Positions of Anchors") or {}]
    fields = ([("Ranging samples between Agents", False)] if mode != "ANC" else []) + ([("Ranging samples to Anchors", True)] if mode != "AF" else [])
    observations = []
    for field, to_anchor in fields:
        for pair, configurations in (iteration.get(field) or {}).items():
            value = (configurations or {}).get(str(samples))
            if not measured(value):
                continue
            left, right = pair.split("-", 1)
            tof, calibrated = released_samples(value)
            observations.append({"from": int(left), "to": int(right), "to_is_anchor": to_anchor, "tof_ns": tof, "calibrated_samples": calibrated})
    return dict(mode=ENGINE_MODES[mode], agent_ids=agents, anchors=anchors, observations=observations)


def score_scenarios(scenarios, predictions, mode, scorer, samples=(1, 10)):
    """Per unit scores of stored predictions on scenarios in the released format, under the released rule and under the
    joined and ranged pair sets, with the released scorer's pair errors, cap and penalty in two dimensions.

    scenarios is an archive, a directory or a file the released scorer reads, predictions its prediction CSV and scorer
    the released scorer module. Returns {(scenario, iteration, samples): dict(anchors, agents, released=(pairs, error),
    joined=(pairs, error), ranged=(pairs, error))}, the error None without any pair. Units are those the released
    scorer scores by default. In the anchor free mode the joined pairs must equal the released pairs, which is checked,
    and they are summed in the released order, so both errors agree bit for bit, as do the ranged pairs when they are
    the released ones. In the anchored modes the pairs are summed in ascending order, the released order, so a unit
    whose pairs are all joined has the released error too. Every pair is scored under the coordinate rule (score_pairs),
    so the predictions score as the released scorer scores the copy of them that bound_prediction_csv leaves it."""
    mode = scorer.canonical_mode(mode)
    table = scorer.load_predictions(Path(predictions))
    result = {}
    for scenario_id, data in scorer.iter_json_scenarios(Path(scenarios)):
        agents = [int(node) for node in data.get("Agents", [])]
        anchors = [int(node) for node in data.get("Anchors", [])]
        if (mode == "AF" and anchors) or (mode in ("AAC", "ANC") and not anchors):
            continue
        members = set(agents)
        truth = {int(node): scorer.true_point(position, 2) for node, position in data.get("True Positions of Agents", {}).items() if int(node) in members}
        penalty = scorer.scenario_diameter(truth)
        iterations = data.get("Iterations", [])
        for number, iteration in enumerate(iterations.values() if isinstance(iterations, dict) else iterations, start=1):
            iteration_id = int(iteration.get("ID", number))
            for count in samples:
                estimated = {node: table.get((scenario_id, iteration_id, count, mode, 2, node)) for node in agents}
                released = scorer.eligible_pairs_for_mode(agents, mode, iteration, count)
                problem = released_problem(data, iteration, count, mode)
                joined, ranged = joined_pairs(problem), joined_pairs(problem, ranged=True)
                if mode == "AF":
                    components = {tuple(sorted(pair)) for pair in released}
                    if set(joined) != components:
                        raise ValueError(f"scenario {scenario_id} iteration {iteration_id}: the joined pairs differ from the released components")
                    joined = released
                    if set(ranged) == components:
                        ranged = released
                entry = dict(anchors=0 if mode == "AF" else len(anchors), agents=len(agents))
                for name, pairs in (("released", released), ("joined", joined), ("ranged", ranged)):
                    number_of_pairs, _, error = score_pairs(scorer, truth, estimated, pairs, penalty, 2)
                    entry[name] = (number_of_pairs, error if number_of_pairs else None)
                key = (scenario_id, iteration_id, count)
                if key in result:
                    raise ValueError(f"duplicate unit {key} in {scenarios}")
                result[key] = entry
    return result


def released_scores(path, mode):
    """The mean pair error of every unit of one mode in the released scorer's scenario scores file, as the file writes
    it. mode is AF, AAC or ANC."""
    with gzip.open(path, "rt") as stream:
        return {(int(row["scenario_id"]), int(row["iteration"]), int(row["samples"])): row["mean_pair_error"]
                for row in csv.DictReader(stream) if row["mode"] == mode and int(row["q"]) == 2}


def stored_run(scenarios, predictions, mode, scorer, released):
    """The units of one stored run under the released rule and the joined and ranged pair sets (score_scenarios),
    refused unless its released errors equal those of the released scorer's scenario scores file `released` unit for
    unit, so that every rule scores the predictions and scenarios the released scorer read."""
    units = score_scenarios(scenarios, predictions, mode, scorer)
    stored = released_scores(released, scorer.canonical_mode(mode))
    if set(stored) != set(units) or any(("nan" if units[key]["released"][1] is None else f"{units[key]['released'][1]:.12g}") != value
                                        for key, value in stored.items()):
        raise ValueError(f"the predictions {predictions} do not reproduce the released scores {released}")
    return units


# the rules of a unit in the released format, the released one first
SCENARIO_RULES = ("released", *REPORTED)
SCENARIO_COLUMNS = ("scenario_id", "iteration", "samples", "anchors", "agents", *(f"{name}_{field}" for name in SCENARIO_RULES for field in ("pairs", "error")))


def write_scenario_units(path, units):
    """The units of score_scenarios as a compressed table in key order, every error exact and empty without a pair."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_name(path.name + ".partial")
    with gzip.open(staged, "wt", compresslevel=6) as stream:
        stream.write(",".join(SCENARIO_COLUMNS) + "\n")
        for key in sorted(units):
            unit = units[key]
            values = [*key, unit["anchors"], unit["agents"]]
            for name in SCENARIO_RULES:
                pairs, error = unit[name]
                values += [pairs, "" if error is None else repr(error)]
            stream.write(",".join(str(v) for v in values) + "\n")
    staged.replace(path)


def read_scenario_units(path):
    """The units write_scenario_units wrote, {(scenario, iteration, samples): dict(anchors, agents, released, joined,
    ranged)}."""
    with gzip.open(path, "rt") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != SCENARIO_COLUMNS:
            raise ValueError(f"{path} is not a table of unit scores under {', '.join(SCENARIO_RULES)}")
        return {(int(row["scenario_id"]), int(row["iteration"]), int(row["samples"])): dict(
                    anchors=int(row["anchors"]), agents=int(row["agents"]),
                    **{name: (int(row[f"{name}_pairs"]), float(row[f"{name}_error"]) if row[f"{name}_error"] else None) for name in SCENARIO_RULES})
                for row in reader}


def scenario_tables(units, percentile, cell=None):
    """Per sample track the tables of units in the released format under the released rule, the benchmark's own
    figure, and under the joined and the supplementary ranged pair sets beside it, each with the share of pairs and
    units it scores.

    units map keys ending in the sample count to the entries of score_scenarios. They are balanced over the released
    reporting cells of anchor and agent count, or over cell(key, unit), and enter in their order, the order in which the
    released scorer meets them, so the released tables of a whole corpus equal its track summary. In the anchor free
    mode the released and joined rules score the released components. Returns {track: {rule: table}}."""
    cell = cell or (lambda key, unit: (unit["anchors"], unit["agents"]))
    result = {}
    for samples in sorted({key[-1] for key in units}):
        track = [(key, unit) for key, unit in units.items() if key[-1] == samples]
        result[str(samples)] = {name: table([(cell(key, unit), unit[name][1], unit[name][0], unit["agents"] * (unit["agents"] - 1) // 2)
                                             for key, unit in track], percentile, name) for name in SCENARIO_RULES}
    return result


def released_percentile():
    import pilot
    return pilot.evidence.load_scorer().balanced_percentile


def anchor_tables(rows, percentile=None, levels=(0.5, 0.75, 0.9), minimum_anchors=3, error="error_cm"):
    """Balanced percentiles per sample track and anchor count, and the headline over the units with enough anchors.

    rows carry samples, cell, anchors and the error field; units without an error, a pair set left empty, are left
    out as the released scorer leaves out non finite scenario means. The released scorer's percentile by default.
    """
    percentile = percentile or released_percentile()

    def table(selected):
        cells = defaultdict(list)
        for row in selected:
            cells[row["cell"]].append(row[error])
        return dict(units=len(selected), cells=len(cells), balanced=[float(percentile(cells, q)) for q in levels] if cells else None)

    result = {}
    for samples in sorted({row["samples"] for row in rows}):
        track = [row for row in rows if row["samples"] == samples and row[error] is not None and math.isfinite(row[error])]
        result[str(samples)] = dict(
            per_anchor_count={str(k): table([row for row in track if row["anchors"] == k]) for k in sorted({row["anchors"] for row in track})},
            headline=dict(minimum_anchors=minimum_anchors, **table([row for row in track if row["anchors"] >= minimum_anchors])))
    return result
