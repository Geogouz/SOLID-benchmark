"""Which side of the line through the two anchors the agents take in the anchor-only mode, in the true layouts and in
the scored estimates of FNL, STRONG and SOLID, from the inputs behind the final tables.

With two anchors in ANC, a range to each anchor fixes an agent's horizontal position only up to its mirror image across
the line through the anchors, and the pair error (eq. pair-error) does not change under a reflection of the whole
layout, so an estimate scores a pair well only when it puts the two agents on one side of the line if they lie on one
side and on opposite sides if they do not. FNL and STRONG start from the engine's geometric layout
(research/methods/src/lib.rs, BaselineInitialization::SolidGeometry), which in ANC sits every agent at its own
lateration (runtime/engine/src/initializer.rs). An agent with an admissible reading to each of the two anchors goes to
the intersection of its two circles on the left of the line from the first anchor to the second, or onto the line
where the circles do not meet (two_anchor_initialization), the anchors taken in the order of anchor_ranges, a BTreeMap
keyed by the bit patterns of x, y and the height less the median anchor height. An agent that ranges one anchor goes
to that anchor's x plus the range, at the anchor's y, and an agent without a range keeps the start its unit provides.
SOLID evaluates every agent's posterior alone (paper Section V-B), so nothing ties the mirror choices of two agents.
FNL or STRONG has the lowest S75 in every two-anchor combination where SOLID does not (anchor-starts.json), and this
record gives the facts that bear on why, per dataset over the two-anchor scenarios of the evaluation half:
1. in the true layouts, the agent pairs whose two agents both lie more than 1 cm off the line, and how many of them
   lie on one side;
2. in the scored estimates of FNL, STRONG and SOLID, how many of these pairs the estimate puts on one side, on opposite
   sides, or leaves undetermined, with an agent within 1 cm of the line or without a position, how many of the pairs it
   places keep their true relation, in how many scenarios the agents it places off the line all lie on one side, and
   how many agents lie on the initializer's side;
3. the S75 of every method in each two-anchor combination, copied from the committed tables.
Every count is given over all agent pairs, over the joined pairs, whose two agents each have a measured anchor link
(the pairs the tables score, scoring.joined_pairs), and over the pairs whose two agents each have an admissible reading
to each of the two anchors, the agents the initializer places by two circles. The last two depend on the reading count.
The one-reading and ten-reading units of a scenario share its agents, anchors and layout, so the counts of the true
layouts over all pairs take every scenario once. Several scenarios can share one physical layout, since TIERS repeats
each recording over its blocks and AAU and IDLab keep their nodes in place, so those counts are also given over the
distinct layouts.

Every score file is an input of the final tables of its dataset and is checked against the hash they record (a recorded
path below out/ lies below the runs root, as in anchor_starts.py). The baselines' estimates come from the work folders of
the runs, which can keep more than one run of a method. An estimate counts only when the pair error it gives under the scorer's
arithmetic (pilot.score_records: the mean over every agent pair of the unit of the error of the pair's distance, capped
at the diameter of the true layout, the cap where a point is missing) equals the scored error of its id, reading count
and method within a relative 1e-12, which the scored runs' estimates meet to the last bits of the arithmetic. The record
gives the largest relative gap of a matched estimate and the smallest of an estimate left out, and two estimates that
match one scored row must be equal. SOLID's prediction files are those its run records list (prediction_manifests,
every hash checked), and since they number their scenarios within each file, a SOLID estimate counts when exactly one
scenario of its label, reading count and agents has the scored error it gives.
HYMN and AAU take their layouts, anchors and links from the request and the reference of their evaluation half. TIERS
and IDLab take the layout from their campaign base, their anchors are the radios missing from the estimates' agents,
which must be the anchors of the campaign's numbering of anchor roles, and their links follow from the construction:
on TIERS every unit holds the first one or ten forward reports of every radio pair (aac_preflight.construct), and on
IDLab the panel's links and readings make the observations (idlab_check.unit). Every reconstruction of the links is
checked against the joined and ranged pair counts of the score rows. Nothing is solved.

Usage: anchor_sides.py [--runs ROOT] [--tables DIR] [--out FILE]
       ROOT is the runs root the tables read, out/ of the repository unless given, and DIR the folder of the final
       tables, research/artifacts/solid/tables unless given. The data folder is SOLID_DATA, research/data unless set.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import statistics
import struct
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "out"
TABLES = ROOT / "research/artifacts/solid/tables"
DATA = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data") / "external-benchmarks"
DATASETS = (("tiers", "TIERS"), ("aau", "AAU"), ("hymn", "HYMN"), ("idlab", "IDLab"))
BASELINES = ("fnl", "strong")
METHODS = (*BASELINES, "solid")
TRACKS = (1, 10)
SUBSETS = ("all", "joined", "both_anchors")
OFF_LINE_CM = 1.0
# the relative gap within which a recomputed pair error matches a scored one, and the looser gap of a near miss
MATCH = 1e-12
NEAR = 1e-6
# the engine's bound on an admissible range and scale (runtime/engine/src/model.rs, MAX_RANGE_CM) and the scorer's bound
# on a predicted coordinate (scoring.COORDINATE_BOUND_CM)
MAX_RANGE_CM = 1.0e5
COORDINATE_BOUND_CM = 1.0e7
# the folders below the runs root that hold the baselines' estimates, and whether a file there is named by the key of
# its batch, so that only the files of two-anchor batches are read
ESTIMATES = {
    "tiers": (("baselines/tiers/work/evaluation-anc",), True),
    "aau": (("baselines/aau/work/evaluation",), False),
    "hymn": (("baselines/hymn/work/evaluation",), False),
    "idlab": (("baselines/idlab/evaluation/anc/predictions",), True),
}
# the anchor roles of the TIERS campaign (tiers_campaign.ROLES), scenario (block - 1) * len(ROLES) + i + 1 has ROLES[i]
TIERS_ROLES = tuple(c for k in range(1, 7) for c in itertools.combinations(range(1, 9), k))
TIERS_RADIOS = tuple(range(1, 9))
IDLAB_RADIOS = 15


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def local(path, runs):
    """A path that the tables record: below the runs root in place of out/, relative to the repository, or as it is
    when absolute (final_tables.py build --runs)."""
    if Path(path).is_absolute():
        return Path(path)
    if path.startswith("out/"):
        return runs / path[len("out/"):]
    return ROOT / path


def relative(path, root=ROOT):
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def array(path, chunk=1 << 24):
    """The objects of a JSON array one at a time, so that a score file of gigabytes holds no more than one chunk."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        text = f.read(chunk)
        at = len(text) - len(text.lstrip())
        if text[at:at + 1] != "[":
            raise ValueError(f"{path} is not a JSON array")
        at += 1
        while True:
            while at < len(text) and text[at] in " \t\r\n,":
                at += 1
            if at == len(text):
                more = f.read(chunk)
                if not more:
                    raise ValueError(f"{path} ends inside its array")
                text, at = more, 0
                continue
            if text[at] == "]":
                return
            try:
                value, end = decoder.raw_decode(text, at)
            except json.JSONDecodeError:
                more = f.read(chunk)
                if not more:
                    raise
                text, at = text[at:] + more, 0
                continue
            yield value
            at = end


def lines(path):
    """The records of a JSON lines file, compressed or not."""
    with (gzip.open(path, "rt", encoding="utf-8") if path.suffix == ".gz" else open(path, encoding="utf-8")) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def checked(entry, runs, sources, name):
    """The file of an input the tables record (local), refused unless it is the file they read."""
    path = local(entry["path"], runs)
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"{path} is not the file the tables read")
    sources[name] = dict(path=entry["path"], sha256=entry["sha256"])
    return path


def hashed(path, sources, name):
    sources[name] = dict(path=relative(path), sha256=sha256(path))
    return sources[name]["sha256"]


def distance(a, b):
    """The released scorer's distance (dist of score_twr_cloud_bern.py), bit for bit."""
    return math.sqrt(sum((left - right) ** 2 for left, right in zip(a, b)))


def scored_point(point):
    """A predicted point as every scorer reads it, None where it is missing (scoring.scored_point)."""
    if point is None or len(point) != 2 or not all(
            v is not None and math.isfinite(v) and abs(v) <= COORDINATE_BOUND_CM for v in point):
        return None
    return tuple(point)


def prepared(scenario):
    """A scenario with the true distance of every agent pair in the unit's order and the cap of every pair error, the
    diameter of the true layout (scenario_diameter of the released scorer)."""
    truth, agents = scenario["truth"], scenario["agents"]
    scenario["true_distances"] = [distance(truth[i], truth[j]) for i, j in itertools.combinations(agents, 2)]
    scenario["cap"] = max((distance(truth[i], truth[j]) for i, j in itertools.combinations(truth, 2)), default=0.0)
    return scenario


def pair_error(scenario, positions):
    """The scored error of an estimate as pilot.score_records computes it: the mean over every agent pair of the unit, in
    its order, of the error of the pair's distance capped at the diameter of the true layout, the cap where a point is
    missing (score_pairs of the released scorer)."""
    cap = scenario["cap"]
    errors = [cap if positions[i] is None or positions[j] is None else
              min(abs(distance(positions[i], positions[j]) - true), cap)
              for (i, j), true in zip(itertools.combinations(range(len(positions)), 2), scenario["true_distances"])]
    return sum(errors) / len(errors)


def gap(error, scored):
    """The relative gap between a recomputed and a scored pair error."""
    return abs(error - scored) / max(1.0, abs(scored))


def matches(error, scored, tolerance=MATCH):
    return gap(error, scored) <= tolerance


def bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def initializer_order(anchors):
    """The two anchors (id, x, y, height) in the order the initializer takes them: anchor_ranges keys a BTreeMap by the
    bit patterns of x, y and the height less the median height of the unit's anchors."""
    low, high = sorted(a[3] for a in anchors)
    median = (low + high) * 0.5
    ordered = sorted(anchors, key=lambda a: (bits(a[1]), bits(a[2]), bits(a[3] - median)))
    if math.hypot(ordered[1][1] - ordered[0][1], ordered[1][2] - ordered[0][2]) == 0:
        raise ValueError(f"anchors {anchors} share one horizontal place and define no line")
    return ordered


def offset(anchors, point):
    """The signed distance of a point from the line through the two anchors in the initializer's order, positive on the
    left of the line from the first to the second, the side where the initializer places an agent by two circles."""
    (_, ax, ay, _), (_, bx, by, _) = anchors
    dx, dy = bx - ax, by - ay
    return (dx * (point[1] - ay) - dy * (point[0] - ax)) / math.hypot(dx, dy)


def valid(distance_cm, scale_cm):
    """A usable sample of the pair sets (scoring.valid_samples)."""
    return math.isfinite(distance_cm) and distance_cm >= 0 and math.isfinite(scale_cm) and scale_cm > 0


def admissible(distance_cm, scale_cm):
    """A sample the engine admits into a range likelihood, and so into a lateration (CalibratedRangeSample)."""
    return valid(distance_cm, scale_cm) and distance_cm <= MAX_RANGE_CM and scale_cm <= MAX_RANGE_CM


def comb2(n):
    return n * (n - 1) // 2


def ratio(part, whole, digits=4):
    return round(part / whole, digits) if whole else None


def one(entries, what):
    entries = list(entries)
    if len(entries) != 1:
        raise ValueError(f"expected one {what}, found {len(entries)}")
    return entries[0]


# ------------------------------------------------------------------------------------------- layouts and links
def links_of(agents, anchors, observations):
    """The anchors every agent has a measured link to, a valid sample from and an admissible sample from."""
    measured, ranged, admitted = defaultdict(set), defaultdict(set), defaultdict(set)
    for agent, anchor, samples in observations:
        if agent not in agents or anchor not in anchors:
            continue
        measured[agent].add(anchor)
        if any(valid(d, s) for d, s in samples):
            ranged[agent].add(anchor)
        if any(admissible(d, s) for d, s in samples):
            admitted[agent].add(anchor)
    return dict(joined={g for g in agents if measured[g]}, ranged={g for g in agents if ranged[g]},
                both_anchors={g for g in agents if admitted[g] >= set(anchors)},
                valid_not_admissible=sum(ranged[g] != admitted[g] for g in agents))


def held_out(dataset, units, tables, runs, sources):
    """HYMN and AAU: agents, anchors and links from the request of the evaluation half, positions from its reference."""
    folder = DATA / f"{dataset}-layout/evaluation"
    run = json.load(open(checked(one((e for e in tables["inputs"] if e["path"].endswith(
        f"/baselines/{dataset}/evaluation/anc/run.json")), "baseline run record"), runs, sources, f"{dataset}/baseline_run")))
    if hashed(folder / "request.json.gz", sources, f"{dataset}/request") != run["request_sha256"]:
        raise ValueError(f"{dataset}: the request is not the one the baselines solved")
    reference_entry = one((e for e in tables["inputs"] if e["path"].endswith(f"{dataset}-layout/evaluation/reference.json")),
                          "reference")
    if hashed(folder / "reference.json", sources, f"{dataset}/reference") != reference_entry["sha256"]:
        raise ValueError(f"{dataset}: the reference is not the one the tables read")
    request = json.loads(gzip.decompress((folder / "request.json.gz").read_bytes()))
    reference = json.load(open(folder / "reference.json"))
    metadata = {u["id"]: u for u in reference["units"]}
    scenarios, links = {}, {}
    for unit in request["units"]:
        key = (unit["id"], unit["samples"])
        if key not in units:
            continue
        problem = unit["problem"]
        anchors = initializer_order([(a["id"], a["position"]["x"], a["position"]["y"], a["height_cm"]) for a in problem["anchors"]])
        if sorted(a[0] for a in anchors) != sorted(metadata[unit["id"]]["anchor_ids"]):
            raise ValueError(f"{dataset} unit {unit['id']}: the request and the reference name other anchors")
        truth = {int(n): tuple(p) for n, p in reference["coordinates_cm"][metadata[unit["id"]]["reference"]].items()}
        scenario = dict(agents=problem["agent_ids"], anchors=anchors, truth=truth, group=None)
        if set(truth) != set(problem["agent_ids"]):
            raise ValueError(f"{dataset} unit {unit['id']}: the reference holds other agents")
        if scenarios.setdefault(unit["id"], scenario) != scenario:
            raise ValueError(f"{dataset} unit {unit['id']}: its tracks differ in agents, anchors or layout")
        if any(o["tof_ns"] for o in problem["observations"]):
            raise ValueError(f"{dataset} unit {unit['id']}: raw timings where calibrated readings were expected")
        links[key] = links_of(set(problem["agent_ids"]), {a[0] for a in anchors},
                              [(o["from"], o["to"], [(s["distance_cm"], s["scale_cm"]) for s in o["calibrated_samples"]])
                               for o in problem["observations"] if o["to_is_anchor"]])
    if set(links) != set(units):
        raise ValueError(f"{dataset}: the request lacks scored two-anchor units")
    rule = ("agents, anchors and the readings of every agent and anchor link from the request of the evaluation half, "
            "true positions from its reference")
    return scenarios, links, rule


def tiers(units, tables, runs, sources):
    """TIERS: positions from the campaign base, anchors by the campaign's anchor roles, links from the block reports."""
    folder = DATA / "tiers-static-campaign"
    run = json.load(open(checked(one((e for e in tables["inputs"] if e["path"].endswith(
        "/baselines/tiers/evaluation/anc/run.json")), "baseline run record"), runs, sources, "tiers/baseline_run")))
    if hashed(folder / "freeze.json", sources, "tiers/campaign_freeze") != run["campaign_freeze_sha256"]:
        raise ValueError("tiers: the campaign freeze is not the one the baselines ran under")
    if hashed(folder / "base.json.gz", sources, "tiers/base") != json.load(open(folder / "freeze.json"))["files"]["base.json.gz"]:
        raise ValueError("tiers: the base is not the one the campaign froze")
    base = json.loads(gzip.decompress((folder / "base.json.gz").read_bytes()))
    fitted = base["fitted"]
    blocks = {b["id"]: b for b in base["blocks"]}
    scenarios, links = {}, {}
    for (sid, samples), unit in sorted(units.items()):
        role = sid - (unit["block"] - 1) * len(TIERS_ROLES)
        if not 1 <= role <= len(TIERS_ROLES) or len(TIERS_ROLES[role - 1]) != 2:
            raise ValueError(f"tiers scenario {sid} is not a two-anchor role of block {unit['block']}")
        coordinates = {int(n): p for n, p in base["coordinates_cm"][unit["acquisition"]].items()}
        if sorted(coordinates) != list(TIERS_RADIOS):
            raise ValueError(f"tiers recording {unit['acquisition']} does not hold the eight radios")
        named = TIERS_ROLES[role - 1]
        agents = [n for n in TIERS_RADIOS if n not in named]
        scenario = dict(agents=agents, anchors=initializer_order([(n, *coordinates[n][:3]) for n in named]),
                        truth={n: tuple(coordinates[n][:2]) for n in agents}, group=unit["acquisition"])
        if scenarios.setdefault(sid, scenario) != scenario:
            raise ValueError(f"tiers scenario {sid}: its tracks differ")
        reports = blocks[unit["block"]]["reports"]
        observations = []
        for agent in agents:
            for anchor in named:
                rows = reports[f"{min(agent, anchor)}-{max(agent, anchor)}"][:samples]
                if len(rows) != samples or not all(math.isfinite(r[1]) and r[1] > 0 for r in rows):
                    raise ValueError(f"tiers block {unit['block']}: link {agent}-{anchor} lacks {samples} positive reports")
                corrected = [max(0.0, fitted["correction"]["gain"] * (r[1] * 100) + fitted["correction"]["offset_cm"]) for r in rows]
                observations.append((agent, anchor, [(d, fitted["scale_cm"]) for d in corrected]))
        links[(sid, samples)] = links_of(set(agents), set(named), observations)
    rule = ("positions of the recording from the campaign base, the anchors of the campaign's role numbering, and the "
            "first one or ten forward reports of every agent and anchor pair of the block, calibrated as the campaign does "
            "(tiers_static_check.metric_input)")
    return scenarios, links, rule


def idlab(units, tables, runs, sources):
    """IDLab: positions from the frozen campaign base, anchors by the panel's anchor order, links from the panel."""
    folder = DATA / "idlab-campaign"
    run = json.load(open(checked(one((e for e in tables["inputs"] if e["path"].endswith(
        "/baselines/idlab/evaluation/anc/run.json")), "baseline run record"), runs, sources, "idlab/baseline_run")))
    freezes = {run["runs"][m]["campaign_freeze_sha256"] for m in BASELINES}
    manifest = json.load(open(folder / "manifest.json"))
    hashed(folder / "manifest.json", sources, "idlab/campaign_manifest")
    if freezes != {manifest["files"]["freeze.json"]} or hashed(folder / "freeze.json", sources, "idlab/campaign_freeze") not in freezes:
        raise ValueError("idlab: the campaign freeze is not the one FNL and STRONG ran under")
    if hashed(folder / "base.json.gz", sources, "idlab/base") != manifest["files"]["base.json.gz"]:
        raise ValueError("idlab: the base is not the one the campaign's manifest lists")
    base = json.loads(gzip.decompress((folder / "base.json.gz").read_bytes()))
    coordinates = {int(n): p for n, p in base["coordinates_cm"].items()}
    values = {(node, row): cm for node, row, cm in base["values"]}
    correction, scale = base["fitted"]["correction"], base["fitted"]["profiles"]["scale_cm"]
    scenarios, links = {}, {}
    for (sid, samples), unit in sorted(units.items()):
        panel = base["panels"][unit["panel"]]
        nodes = panel["nodes"]
        if panel["configuration"] != unit["configuration"] or len(nodes) != IDLAB_RADIOS:
            raise ValueError(f"idlab scenario {sid}: panel {unit['panel']} is not a panel of the 15 office radios")
        order = panel["anchor_order"]
        # idlab_population.anchor_subsets and idlab_campaign.batch_inputs: scenario panel * (roles + 1) + i + 2 has subset i
        subsets = [sorted((order + order)[start:start + count]) for count in range(1, len(order) - 1) for start in range(len(order))]
        index = sid - unit["panel"] * (len(nodes) * (len(nodes) - 2) + 1) - 2
        named = subsets[index] if 0 <= index < len(subsets) else ()
        if len(named) != 2:
            raise ValueError(f"idlab scenario {sid} is not a two-anchor role of panel {unit['panel']}")
        agents = [n for n in nodes if n not in named]
        scenario = dict(agents=agents, anchors=initializer_order([(n, *coordinates[n][:3]) for n in named]),
                        truth={n: tuple(coordinates[n][:2]) for n in agents}, group=None)
        if scenarios.setdefault(sid, scenario) != scenario:
            raise ValueError(f"idlab scenario {sid}: its tracks differ")
        observations = []
        for link in panel["links"]:
            a, b = link["nodes"]
            if (a in named) == (b in named):
                continue
            agent, anchor = (b, a) if a in named else (a, b)
            identities = link["one" if samples == 1 else "up_to_ten"]
            if identities:
                corrected = [max(0.0, correction["gain"] * values[tuple(i)] + correction["offset_cm"]) for i in identities]
                observations.append((agent, anchor, [(d, scale) for d in corrected]))
        links[(sid, samples)] = links_of(set(agents), set(named), observations)
    rule = ("positions of the 15 office radios from the frozen campaign base, the anchors of the panel's anchor order "
            "(idlab_population.anchor_subsets), and the panel's link records of the track, calibrated as the campaign does "
            "(idlab_check.unit)")
    return scenarios, links, rule


LAYOUTS = {"tiers": tiers, "aau": lambda *a: held_out("aau", *a), "hymn": lambda *a: held_out("hymn", *a), "idlab": idlab}


# ------------------------------------------------------------------------------------------- counting
def tally_truth(counter, agents, offsets):
    """The pairs of the agents both more than 1 cm off the line and those on one side, and the scenarios with two or
    more such agents and those with all of them on one side."""
    off = [offsets[g] for g in agents if abs(offsets[g]) > OFF_LINE_CM]
    left = sum(t > 0 for t in off)
    counter["scenarios"] += 1
    counter["agents"] += len(agents)
    counter["agents_off_line"] += len(off)
    counter["pairs"] += comb2(len(off))
    counter["pairs_same_side"] += comb2(left) + comb2(len(off) - left)
    if len(off) >= 2:
        counter["scenarios_two_or_more_off_line"] += 1
        counter["scenarios_all_one_side"] += left in (0, len(off))


def tally_estimate(counter, agents, offsets, placed):
    """Where an estimate puts the agents and the pairs of agents both more than 1 cm off the line in truth. placed holds
    each agent's offset in the estimate, None without a position."""
    counter["scenarios"] += 1
    counter["agents"] += len(agents)
    left = sum(1 for g in agents if placed[g] is not None and placed[g] > OFF_LINE_CM)
    right = sum(1 for g in agents if placed[g] is not None and placed[g] < -OFF_LINE_CM)
    counter["agents_initializer_side"] += left
    counter["agents_other_side"] += right
    counter["agents_within_1cm"] += sum(1 for g in agents if placed[g] is not None and abs(placed[g]) <= OFF_LINE_CM)
    counter["agents_without_position"] += sum(1 for g in agents if placed[g] is None)
    if left + right >= 2:
        counter["scenarios_two_or_more_placed_off_line"] += 1
        counter["scenarios_placed_one_side"] += not (left and right)
        counter["scenarios_placed_initializer_side"] += not right
    # agents both off the line in truth, by their true side and the side the estimate gives them
    classes, pairs = Counter(), 0
    for g in agents:
        if abs(offsets[g]) <= OFF_LINE_CM:
            continue
        pairs += 1
        if placed[g] is not None and abs(placed[g]) > OFF_LINE_CM:
            classes[(offsets[g] > 0, placed[g] > 0)] += 1
    a, b, c, d = (classes[k] for k in ((True, True), (True, False), (False, True), (False, False)))
    counter["pairs"] += comb2(pairs)
    counter["pairs_same_side"] += comb2(a + c) + comb2(b + d)
    counter["pairs_opposite_sides"] += (a + c) * (b + d)
    counter["pairs_undetermined"] += comb2(pairs) - comb2(a + b + c + d)
    counter["pairs_relation_as_true"] += comb2(a) + comb2(b) + comb2(c) + comb2(d) + a * d + b * c


TRUTH_FIELDS = ("scenarios", "agents", "agents_off_line", "median_offset_cm", "pairs", "pairs_same_side",
                "share_same_side", "scenarios_two_or_more_off_line", "scenarios_all_one_side")
ESTIMATE_FIELDS = ("scenarios", "agents", "agents_initializer_side", "agents_other_side", "agents_within_1cm",
                   "agents_without_position", "pairs", "pairs_same_side", "pairs_opposite_sides", "pairs_undetermined",
                   "pairs_relation_as_true", "share_same_side", "share_same_side_of_placed",
                   "share_relation_as_true_of_placed", "scenarios_two_or_more_placed_off_line",
                   "scenarios_placed_one_side", "scenarios_placed_initializer_side", "share_scenarios_placed_one_side",
                   "median_anchor_distance_gap_cm")


def truth_entry(counter, offsets=None):
    """The counts of the true layouts in a fixed order, with the share of pairs on one side and, given the agents'
    distances from the line, their median."""
    entry = dict(counter, share_same_side=ratio(counter["pairs_same_side"], counter["pairs"]))
    if offsets is not None:
        entry["median_offset_cm"] = round(statistics.median(offsets), 1)
    return {k: entry.get(k, 0) for k in TRUTH_FIELDS if k != "median_offset_cm" or offsets is not None}


def estimate_entry(counter, gaps=None):
    """The counts of the estimates in a fixed order with their shares and, given the gaps between estimated and true
    distances to the anchors, their median."""
    placed = counter["pairs_same_side"] + counter["pairs_opposite_sides"]
    entry = dict(counter, share_same_side=ratio(counter["pairs_same_side"], counter["pairs"]),
                 share_same_side_of_placed=ratio(counter["pairs_same_side"], placed),
                 share_relation_as_true_of_placed=ratio(counter["pairs_relation_as_true"], placed),
                 share_scenarios_placed_one_side=ratio(counter["scenarios_placed_one_side"],
                                                       counter["scenarios_two_or_more_placed_off_line"]))
    if gaps is not None:
        entry["median_anchor_distance_gap_cm"] = round(statistics.median(gaps), 1) if gaps else None
    return {k: entry.get(k, 0) for k in ESTIMATE_FIELDS if k != "median_anchor_distance_gap_cm" or gaps is not None}


# ------------------------------------------------------------------------------------------- one dataset
def scores_of(dataset, tables, runs, sources):
    """The two-anchor rows of FNL, STRONG and SOLID in the score files the tables read: the scored error per (id,
    reading count, method) and per (id, reading count) the unit's metadata and pair counts, the same for every method."""
    scored, units = {}, {}
    files = (("baseline_scores", f"/baselines/{dataset}/evaluation/anc/scores.json", BASELINES),
             ("solid_scores", f"/solid/{dataset}/evaluation/anc/scores.json", ("solid",)))
    for name, ending, methods in files:
        path = checked(one((e for e in tables["inputs"] if e["path"].endswith(ending)), name), runs, sources, f"{dataset}/{name}")
        for row in array(path):
            method = row.get("method", "solid")
            if row["anchors"] != 2 or method not in methods:
                continue
            key = (row["id"], row["samples"], method)
            if key in scored:
                raise ValueError(f"{dataset}: two scored rows of {key}")
            scored[key] = row["error_cm"]
            unit = dict(key=row["key"], cell=row["cell"], acquisition=row["acquisition"], block=row.get("block"),
                        panel=row.get("panel"), configuration=row.get("configuration"),
                        **{f"{s}_pairs": row["pair_sets"][s]["eligible_pairs"] for s in ("all", "joined", "ranged")})
            if units.setdefault((row["id"], row["samples"]), unit) != unit:
                raise ValueError(f"{dataset}: the rows of unit {row['id']} disagree on its metadata or pair counts")
    for method in METHODS:
        if {k[:2] for k in scored if k[2] == method} != set(units):
            raise ValueError(f"{dataset}: {method} lacks scored two-anchor units")
    return scored, units


def baseline_estimates(dataset, scored, units, scenarios, runs, sources, confirmed):
    """The positions of every scored row of FNL and STRONG whose estimate in the work folders gives its scored error."""
    folders, by_key = ESTIMATES[dataset]
    found, conflicts, read = {}, set(), {}
    counts = {m: Counter({f"rows_{what}_in {folder}": 0 for folder in folders for what in ("read", "matched")})
              for m in BASELINES}
    extremes = {m: dict(largest_relative_gap_matched=0.0, smallest_relative_gap_not_matched=None) for m in BASELINES}
    keys = {u["key"] for u in units.values()}
    for folder in folders:
        hashes = read[folder] = {}
        for path in sorted((runs / folder).glob("*.jsonl*")):
            if by_key and path.name.split(".")[0].rsplit("-", 1)[0] not in keys:
                continue
            hashes[path.name] = sha256(path)
            for row in lines(path):
                key = (row["id"], row["samples"], row.get("method"))
                if key[2] not in BASELINES or key not in scored:
                    continue
                scenario = scenarios[row["id"]]
                estimate = row["estimate"]
                if row["agent_ids"] != scenario["agents"] or (estimate and estimate["agent_ids"] != scenario["agents"]):
                    raise ValueError(f"{dataset} {key}: the estimate's agents are not the scenario's")
                confirmed[row["id"]] += 1
                positions = tuple(map(scored_point, estimate["positions"])) if estimate else (None,) * len(scenario["agents"])
                if len(positions) != len(scenario["agents"]):
                    raise ValueError(f"{dataset} {key}: the estimate holds another number of positions")
                error = pair_error(scenario, positions)
                tally, extreme = counts[key[2]], extremes[key[2]]
                tally["rows_read"] += 1
                tally[f"rows_read_in {folder}"] += 1
                relative_gap = gap(error, scored[key])
                if not matches(error, scored[key]):
                    tally["rows_near_but_not_matched" if matches(error, scored[key], NEAR) else "rows_of_other_runs"] += 1
                    if extreme["smallest_relative_gap_not_matched"] is None or relative_gap < extreme["smallest_relative_gap_not_matched"]:
                        extreme["smallest_relative_gap_not_matched"] = relative_gap
                    continue
                extreme["largest_relative_gap_matched"] = max(extreme["largest_relative_gap_matched"], relative_gap)
                tally[f"rows_matched_in {folder}"] += 1
                if key in found:
                    tally["rows_matched_again_identical" if found[key] == positions else "rows_matched_again_different"] += 1
                    if found[key] != positions:
                        conflicts.add(key)
                    continue
                found[key] = positions
    sources[f"{dataset}/baseline_estimates"] = {folder: dict(files=len(hashes), sha256=hashes) for folder, hashes in read.items()}
    for key in conflicts:
        del found[key]
    for method, tally in counts.items():
        tally["scored_rows"] = sum(1 for k in scored if k[2] == method)
        tally["scored_rows_matched"] = sum(1 for k in found if k[2] == method)
    return found, {m: dict(sorted(c.items()), **extremes[m]) for m, c in counts.items()}


def solid_estimates(dataset, tables, scored, units, scenarios, runs, sources, confirmed):
    """The positions of every scored row of SOLID from the prediction files of its run record, a prediction matched to
    the one scenario of its label, reading count and agents whose scored error it gives."""
    run_path = checked(one((e for e in tables["inputs"] if e["path"].endswith(f"/solid/{dataset}/evaluation/anc/run.json")),
                           "solid run record"), runs, sources, f"{dataset}/solid_run")
    candidates = defaultdict(list)
    for (sid, samples), unit in units.items():
        for label in (unit["key"], unit["cell"]):
            candidates[(label, samples, tuple(scenarios[sid]["agents"]))].append(sid)
    found, tally, hashes = {}, Counter(), {}
    extremes = dict(largest_relative_gap_matched=0.0, smallest_relative_gap_of_other_candidates=None)
    for manifest_name, manifest_sha in sorted(json.load(open(run_path))["prediction_manifests"].items()):
        manifest_path = run_path.parent / manifest_name
        if sha256(manifest_path) != manifest_sha:
            raise ValueError(f"{manifest_path} is not the manifest the run record lists")
        hashes[manifest_name] = manifest_sha
        for output, output_sha in sorted(json.load(open(manifest_path))["outputs"].items()):
            path = run_path.parent / output
            if sha256(path) != output_sha:
                raise ValueError(f"{path} is not the prediction file its manifest lists")
            hashes[output] = output_sha
            for row in lines(path):
                options = candidates.get((row["label"], row["samples"], tuple(row["agent_ids"])))
                if not options:
                    continue
                estimate = row["estimate"]
                if estimate and estimate["agent_ids"] != row["agent_ids"]:
                    raise ValueError(f"{path}: an estimate reorders its agents")
                positions = tuple(map(scored_point, estimate["positions"])) if estimate else (None,) * len(row["agent_ids"])
                if len(positions) != len(row["agent_ids"]):
                    raise ValueError(f"{path}: an estimate holds another number of positions")
                tally["predictions_read"] += 1
                gaps = {sid: gap(pair_error(scenarios[sid], positions), scored[(sid, row["samples"], "solid")]) for sid in options}
                hits = [sid for sid, g in gaps.items() if g <= MATCH]
                for sid, g in gaps.items():
                    if g <= MATCH:
                        extremes["largest_relative_gap_matched"] = max(extremes["largest_relative_gap_matched"], g)
                    elif extremes["smallest_relative_gap_of_other_candidates"] is None or g < extremes["smallest_relative_gap_of_other_candidates"]:
                        extremes["smallest_relative_gap_of_other_candidates"] = g
                if len(hits) != 1:
                    tally["predictions_unmatched" if not hits else "predictions_matched_ambiguously"] += 1
                    continue
                key = (hits[0], row["samples"], "solid")
                if key in found:
                    tally["predictions_matched_to_a_matched_row"] += 1
                    continue
                confirmed[hits[0]] += 1
                found[key] = positions
    sources[f"{dataset}/solid_predictions"] = dict(folder=relative(run_path.parent, runs), files=len(hashes), sha256=hashes)
    tally["scored_rows"] = sum(1 for k in scored if k[2] == "solid")
    tally["scored_rows_matched"] = len(found)
    return found, dict(sorted(tally.items()), **extremes)


def dataset_record(dataset, name, runs, sources, tables_root=TABLES):
    tables_path = tables_root / f"{dataset}/evaluation/tables.json"
    hashed(tables_path, sources, f"{dataset}/tables")
    tables = json.load(open(tables_path))
    scored, units = scores_of(dataset, tables, runs, sources)
    scenarios, links, layout_rule = LAYOUTS[dataset](units, tables, runs, sources)
    for scenario in scenarios.values():
        prepared(scenario)

    # the links against the pair counts of the score rows
    link_checks = Counter()
    for key, unit in units.items():
        link = links[key]
        link_checks["units"] += 1
        link_checks["all_pairs_agree"] += comb2(len(scenarios[key[0]]["agents"])) == unit["all_pairs"]
        link_checks["joined_pairs_agree"] += comb2(len(link["joined"])) == unit["joined_pairs"]
        link_checks["ranged_pairs_agree"] += comb2(len(link["ranged"])) == unit["ranged_pairs"]
        link_checks["agents_valid_but_not_admissible"] += link["valid_not_admissible"]
        for track_key in ("joined", "both_anchors"):
            link_checks[f"agents_{track_key}_{key[1]}"] += len(link[track_key])
        link_checks[f"agents_{key[1]}"] += len(scenarios[key[0]]["agents"])
    if not link_checks["units"] == link_checks["all_pairs_agree"] == link_checks["joined_pairs_agree"] == link_checks["ranged_pairs_agree"]:
        raise ValueError(f"{dataset}: the reconstructed links disagree with the pair counts of the score rows {dict(link_checks)}")

    confirmed = Counter()
    estimates, matching = baseline_estimates(dataset, scored, units, scenarios, runs, sources, confirmed)
    estimates_solid, matching["solid"] = solid_estimates(dataset, tables, scored, units, scenarios, runs, sources, confirmed)
    estimates.update(estimates_solid)

    # 1. the true layouts
    offsets = {sid: {g: offset(s["anchors"], s["truth"][g]) for g in s["agents"]} for sid, s in scenarios.items()}
    groups = sorted({s["group"] for s in scenarios.values() if s["group"]})
    truth, truth_offsets, layouts = defaultdict(Counter), defaultdict(list), {}
    for sid, s in scenarios.items():
        for group in ("all", *([f"recording {s['group']}"] if s["group"] else [])):
            tally_truth(truth[group], s["agents"], offsets[sid])
            truth_offsets[group] += [abs(offsets[sid][g]) for g in s["agents"]]
        layouts.setdefault((tuple(sorted(s["truth"].items())), tuple(sorted(s["anchors"]))), sid)
    for sid in layouts.values():
        tally_truth(truth["distinct layouts"], scenarios[sid]["agents"], offsets[sid])
        truth_offsets["distinct layouts"] += [abs(v) for v in offsets[sid].values()]
    for (sid, samples), link in links.items():
        for subset in ("joined", "both_anchors"):
            agents = [g for g in scenarios[sid]["agents"] if g in link[subset]]
            for track in (str(samples), "both"):
                tally_truth(truth[(subset, track)], agents, offsets[sid])
    true_layout = dict(all=truth_entry(truth["all"], truth_offsets["all"]),
                       distinct_layouts=truth_entry(truth["distinct layouts"], truth_offsets["distinct layouts"]),
                       **{subset: {track: truth_entry(truth[(subset, track)]) for track in ("1", "10", "both")}
                          for subset in ("joined", "both_anchors")})
    if groups:
        true_layout["by_recording"] = {g: truth_entry(truth[f"recording {g}"], truth_offsets[f"recording {g}"]) for g in groups}

    # 2. the estimates
    counts, gaps = defaultdict(Counter), defaultdict(list)
    for (sid, samples, method), positions in estimates.items():
        s = scenarios[sid]
        link = links[(sid, samples)]
        placed = {g: offset(s["anchors"], p) if p is not None else None for g, p in zip(s["agents"], positions)}
        point = dict(zip(s["agents"], positions))
        for subset in SUBSETS:
            agents = s["agents"] if subset == "all" else [g for g in s["agents"] if g in link[subset]]
            for track in (str(samples), "both"):
                tally_estimate(counts[(method, track, subset)], agents, offsets[sid], placed)
                if subset == "both_anchors":
                    gaps[(method, track)] += [abs(distance(point[g], a[1:3]) - distance(s["truth"][g], a[1:3]))
                                              for g in agents if point[g] is not None for a in s["anchors"]]
        if s["group"]:
            tally_estimate(counts[(method, "both", f"recording {s['group']}")], s["agents"], offsets[sid], placed)
    estimate_record = {}
    for method in METHODS:
        entry = dict(matching=matching[method])
        for track in ("1", "10", "both"):
            entry[track] = {subset: estimate_entry(counts[(method, track, subset)],
                                                   gaps[(method, track)] if subset == "both_anchors" else None)
                            for subset in SUBSETS}
        if groups:
            entry["by_recording_both_tracks_all_agents"] = {g: estimate_entry(counts[(method, "both", f"recording {g}")])
                                                            for g in groups}
        estimate_record[method] = entry

    # 3. S75 from the tables
    s75, lowest = {}, {}
    for readings, track in sorted(tables["modes"]["ANC"]["groups"]["anchors-2"]["tracks"].items(), key=lambda kv: int(kv[0])):
        if track["units"] != sum(1 for k in units if k[1] == int(readings)):
            raise ValueError(f"{dataset}: the tables count other two-anchor units")
        values = {m: v["S75"]["value_cm"] for m, v in track["pair_sets"]["joined"]["methods"].items()}
        printed = {m: round(v, 1) for m, v in values.items()}
        s75[readings] = dict(values, scored_units=track["pair_sets"]["joined"]["scored_units"],
                             pair_share=track["pair_sets"]["joined"]["pair_share"])
        lowest[readings] = sorted(m for m in printed if printed[m] == min(printed.values()))

    record = dict(
        name=name,
        scenarios=dict(ids=len(scenarios), units={str(t): sum(1 for k in units if k[1] == t) for t in TRACKS},
                       distinct_layouts=len(layouts)),
        layout_and_links=layout_rule,
        anchors_confirmed=dict(scenarios=len(scenarios), scenarios_confirmed=sum(1 for sid in scenarios if confirmed[sid]),
                               estimates_checked=sum(confirmed.values()),
                               rule="every estimate read lists the scenario's agents, so its anchors are the radios it "
                                    "leaves out" if dataset in ("tiers", "idlab") else
                                    "every estimate read lists the agents of the request's unit"),
        link_checks=dict(sorted(link_checks.items())),
        true_layout=true_layout,
        estimates=estimate_record,
        S75_cm=s75,
        lowest_S75_at_0_1_cm=lowest,
    )
    if dataset == "idlab" and {len(s["agents"]) for s in scenarios.values()} != {IDLAB_RADIOS - 2}:
        raise ValueError("idlab: a two-anchor scenario without 13 agents")
    return record


def summary_of(record):
    """The headline shares of one dataset, both reading counts together where a count depends on them."""
    true_layout, estimates = record["true_layout"], record["estimates"]
    return dict(
        true_share_same_side=dict(all=true_layout["all"]["share_same_side"],
                                  joined=true_layout["joined"]["both"]["share_same_side"],
                                  both_anchors=true_layout["both_anchors"]["both"]["share_same_side"]),
        estimate_share_same_side={m: {subset: estimates[m]["both"][subset]["share_same_side"] for subset in SUBSETS}
                                  for m in METHODS},
        estimate_share_relation_as_true_of_placed={m: {subset: estimates[m]["both"][subset]["share_relation_as_true_of_placed"]
                                                      for subset in SUBSETS} for m in METHODS},
        estimate_share_scenarios_placed_one_side={m: {subset: estimates[m]["both"][subset]["share_scenarios_placed_one_side"]
                                                     for subset in SUBSETS} for m in METHODS},
        scored_rows_matched={m: f"{estimates[m]['matching']['scored_rows_matched']}/{estimates[m]['matching']['scored_rows']}"
                             for m in METHODS},
        S75_cm={t: {m: round(v, 1) for m, v in record["S75_cm"][t].items() if m in (*METHODS, "jin-2021", "ade-bp-2015")}
                for t in record["S75_cm"]},
        lowest_S75_at_0_1_cm=record["lowest_S75_at_0_1_cm"],
    )


NOTES = [
    "A pair counts when the true positions of both its agents lie more than 1 cm off the line through the two anchors. In "
    "an estimate, an agent within 1 cm of the line or without a position leaves each of its pairs undetermined. "
    "share_same_side divides the pairs an estimate puts on one side by every counted pair, undetermined ones included, and "
    "the shares 'of_placed' divide by the pairs it puts on one side or on opposite sides.",
    "Offsets are signed so that the initializer's side is positive, left of the line from the first anchor to the second "
    "in the order of anchor_ranges (runtime/engine/src/initializer.rs), a BTreeMap keyed by the bit patterns of x, y and "
    "the height less the median of the two anchor heights. Pair counts do not depend on this sign.",
    "Subsets: all is every agent of a scenario. joined is the agents with at least one measured anchor link, an "
    "observation with or without an admissible sample, and its pairs are the joined pairs the tables score "
    "(scoring.joined_pairs). both_anchors is the agents with at least one admissible reading to each of the two anchors, "
    "admissible as the engine admits a calibrated sample (finite, non negative distance and positive scale, both at most "
    "1e5 cm), the agents the initializer places by two circles. Within a subset every agent pair counts.",
    "Deduplication: the one-reading and ten-reading units of a scenario id hold the same agents, anchors and true layout "
    "(checked), so the true counts over all agents take every id once. The counts over joined and both_anchors take every "
    "(id, reading count) once, since the links depend on the reading count, and 'both' adds the two reading counts. "
    "distinct_layouts takes every distinct set of true agent positions and anchors once, with its first scenario.",
    "An estimate is matched to a scored row when the pair error it gives under the scorer's arithmetic is within a "
    "relative 1e-12 of the scored error. rows_near_but_not_matched counts estimates of other runs within 1e-6 but not "
    "1e-12, which a tolerance of 1e-6 would take. A scored row with two different matching estimates is dropped "
    "(rows_matched_again_different).",
    "SOLID reports the better of two descents on its estimated pair-distance risk, started from the coordinate mean and "
    "from the medoid of joint draws in the anchors' frame (runtime/engine/src/anchor_only.rs, decide). Every step moves "
    "the agents along the gradient of a function of their pair distances, whose terms sum to zero and exert no torque, "
    "so the layout keeps the centroid of its start exactly and its orientation to first order, and its sides can be read "
    "against the anchors' line. median_anchor_distance_gap_cm is the median, over the both_anchors agents and both "
    "anchors, of the gap between an agent's estimated and true horizontal distance to the anchor. A large gap means the "
    "layout leaves these agents off their range circles, not that it leaves the frame.",
    "TIERS and IDLab: the runners build their requests from the campaign base and do not store them, so agents, anchors "
    "and links come from the base by the campaign's construction, and every estimate read lists exactly the agents this "
    "gives. On TIERS every agent has an admissible reading to every anchor in both reading counts, since every block holds "
    "ten positive forward reports of each of the 28 radio pairs. On IDLab every panel holds the 15 office radios, so a "
    "two-anchor scenario has 13 agents, and its anchors are the two radios its estimates leave out.",
    "Older runs beside the scored ones: the TIERS work folder keeps two runs of every batch, and the IDLab predictions of "
    "FNL's run keep STRONG's run from before its reselection. A scored row is matched in one run only or by identical "
    "estimates of two runs (rows_matched_again_identical). The IDLab ADE-BP rerun is read too and holds no FNL or STRONG "
    "row (rows_read_in).",
    "S75_cm copies modes.ANC.groups['anchors-2'].tracks[readings].pair_sets.joined.methods[m].S75.value_cm of the "
    "committed tables with the units its joined pairs score and their share of all pairs, and lowest_S75_at_0_1_cm names "
    "the methods lowest at the printed precision.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sources, datasets = {}, {}
    for dataset, name in DATASETS:
        datasets[dataset] = dataset_record(dataset, name, args.runs, sources, args.tables)
        print(dataset, json.dumps(summary_of(datasets[dataset])), flush=True)
    result = dict(
        sources=sources,
        rule=dict(off_line_cm=OFF_LINE_CM, match_relative=MATCH, near_relative=NEAR, subsets=list(SUBSETS),
                  tracks=["1", "10", "both"]),
        summary={d: summary_of(r) for d, r in datasets.items()},
        datasets=datasets,
        notes=NOTES,
    )
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
