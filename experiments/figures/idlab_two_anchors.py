"""Why STRONG has a lower S75 than SOLID in the anchor-only mode (ANC) of IDLab with two anchors, while SOLID has the
lower S75 with one, from the inputs behind the final tables.

The scored estimates are those of anchor_sides.py, whose functions this script reuses: the score files that the
committed IDLab tables read, each checked against the hash the tables record, every scored row of FNL, STRONG and
SOLID matched to its estimate in the work folders or the prediction files when the pair error the estimate gives under
the scorer's arithmetic equals the scored error within a relative 1e-12, and every scenario rebuilt from the frozen
campaign base: the 15 office radios at their surveyed positions, the anchors of the panel's anchor order
(idlab_population.anchor_subsets), and the panel's links and readings of the track, calibrated as the campaign does
(idlab_check.unit). Here the rebuild covers the one-anchor and the two-anchor scenarios and keeps every reading, and
each estimate is also scored on the joined pairs, the pairs the tables score (scoring.joined_pairs: in ANC the pairs of
two agents that each have a measured anchor link), with the arithmetic of its score file (the released scorer's dist
for the composed baseline scores, pilot.score_records' math.dist for SOLID's), which must give the joined error of the
score row; the released scorer's balanced percentile of these errors must give the S75 of the tables.

Per anchor count (1, 2) and reading count (1, 10) the record gives, for FNL, STRONG and SOLID:
1. the joined pairs by type: (a) both agents range the same single anchor, (b) the two agents range different single
   anchors, (c) at least one agent ranges both anchors, and any other type found. A type's share of the joined pairs and
   per method the mean over its pairs of the pair error (eq. pair-error: the absolute error of the horizontal pair
   distance, capped at the diameter of the true layout), with the mean estimated and true distance.
2. which type carries the difference between SOLID's and STRONG's errors. A scenario's joined error is the mean of its
   pair errors, so its difference splits exactly into one term per type, the type's pairs' error differences over the
   scenario's joined pairs. The terms are given over every scored scenario and over the scenarios that set S75, those
   whose error lies between the method's balanced S65 and S85, with the balanced weights of the S75. The S75 gap itself
   is split by the S75 of hybrid errors, SOLID's pair errors with those of some types replaced by STRONG's: the gap
   closed by replacing one type, and the Shapley shares of the types, which add up to the gap.
3. for the agents that range one anchor, the distance of each estimate from the range circle the start uses (the
   horizontal distance to the anchor less the median horizontal range of initializer.rs) and the bearing of the
   estimate from the anchor relative to the true bearing; for the agents without a measured link, where each method
   puts them and whether their pairs are joined.
4. the geometry behind the pairs of type (b): the distance between the two anchors, the true pair distances against it,
   the sides of their anchors the agents take, and the one-anchor pairs' angles at their anchor.
Nothing is solved or fitted.

Usage: idlab_two_anchors.py [--runs ROOT] [--tables DIR] [--out FILE]
       ROOT is the runs root the tables read, out/ of the repository unless given, and DIR the folder of the final
       tables, research/artifacts/solid/tables unless given (anchor_sides.py)
"""
import argparse
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import anchor_sides as sides

ROOT = sides.ROOT
DATASET = "idlab"
CAMPAIGN = sides.DATA / "idlab-campaign"
# the released TWR-CLOUD-Bern scorer, whose balanced percentile every table reports (research/benchmarks/bern/
# official_scorer.py, SCORER_SHA256), through research/data like every retained input
SCORER = sides.DATA.parent / "inputs/0068d72b94db6b1c/score_twr_cloud_bern.py"
SCORER_SHA256 = "0068d72b94db6b1c40414829390025a3b2b12e9b9d1fd7d939fff5fc97659073"
ANCHOR_COUNTS = (1, 2)
TRACKS = sides.TRACKS
METHODS = sides.METHODS
BASELINES = sides.BASELINES
# idlab_campaign.batch_inputs: a panel of n radios has n (n - 2) anchor roles, and scenario panel * (roles + 1) + i + 2
# has the anchors of subset i of idlab_population.anchor_subsets
IDLAB_RADIOS = sides.IDLAB_RADIOS
ROLES = IDLAB_RADIOS * (IDLAB_RADIOS - 2)
# the types of a joined pair by the anchors its two agents range, an anchor counting when the agent has an admissible
# reading of it, and the parts of the third type
TYPES = ("same_anchor", "different_anchors", "with_both", "other")
TYPE_RULES = dict(same_anchor="(a) each agent ranges exactly one anchor, the same one",
                  different_anchors="(b) each agent ranges exactly one anchor, the two anchors differ",
                  with_both="(c) at least one agent ranges both anchors",
                  other="any other pair, an agent with a measured link but no admissible reading")
WITH_BOTH_PARTS = ("both_agents_range_both", "one_agent_ranges_both")
# the arithmetic of the joined errors of each score file, which they reproduce bit for bit (load checks the gaps): the
# composed baseline scores take the released scorer's dist, SOLID's take pilot.score_records' per pair errors (math.dist)
PAIR_DISTANCE = {"fnl": sides.distance, "strong": sides.distance, "solid": math.dist}
COMPARISONS = (("solid", "strong"), ("solid", "fnl"))
# the scenarios that set a method's S75: errors between its balanced S65 and S85
BAND = (0.65, 0.85)
# the bins of the distance between the two anchors over the sum of the start ranges of a pair of type (b)
SEPARATION_BINS = (0.75, 1.0, 1.5)


def subsets_of(order):
    """idlab_population.anchor_subsets."""
    return [sorted((order + order)[start:start + count]) for count in range(1, len(order) - 1) for start in range(len(order))]


def engine_median(values):
    """The median of runtime/engine/src/initializer.rs: the middle value, or half the sum of the two middle values."""
    values = sorted(values)
    middle = len(values) // 2
    return (values[middle - 1] + values[middle]) * 0.5 if len(values) % 2 == 0 else values[middle]


def released_scorer(sources):
    if sides.hashed(SCORER, sources, "released_scorer") != SCORER_SHA256:
        raise ValueError(f"{SCORER} is not the released scorer")
    spec = importlib.util.spec_from_file_location("released_scorer", SCORER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # as research/benchmarks/datasets/evidence.py load_scorer, for its dataclasses
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------- inputs
def scores_of(tables, runs, sources):
    """The one-anchor and two-anchor rows of FNL, STRONG and SOLID in the score files the tables read: per (id, reading
    count, method) the scored error over every pair, the joined error and the stay flag, per (id, reading count) the
    unit's metadata and pair counts, the same for every method, and per (method, anchors, reading count) the cells in
    the order the tables meet them (scoring.table), first appearance among the rows with a joined error."""
    scored, joined, stayed, units, cells = {}, {}, {}, {}, defaultdict(dict)
    files = (("baseline_scores", f"/baselines/{DATASET}/evaluation/anc/scores.json", BASELINES),
             ("solid_scores", f"/solid/{DATASET}/evaluation/anc/scores.json", ("solid",)))
    for name, ending, methods in files:
        path = sides.checked(sides.one((e for e in tables["inputs"] if e["path"].endswith(ending)), name), runs, sources,
                             f"{DATASET}/{name}")
        for row in sides.array(path):
            method = row.get("method", "solid")
            if row["anchors"] not in ANCHOR_COUNTS or method not in methods:
                continue
            key = (row["id"], row["samples"], method)
            if key in scored:
                raise ValueError(f"two scored rows of {key}")
            if row["failure"] is not None:
                raise ValueError(f"{key} is a failed solve")
            scored[key] = row["error_cm"]
            joined[key] = row["pair_sets"]["joined"]["mean_pair_error_cm"]
            stayed[key] = row.get("stayed_at_start")
            if joined[key] is not None:
                cells[(method, row["anchors"], row["samples"])].setdefault(row["cell"], None)
            unit = dict(key=row["key"], cell=row["cell"], acquisition=row["acquisition"], anchors=row["anchors"],
                        panel=row["panel"], configuration=row["configuration"],
                        **{f"{s}_pairs": row["pair_sets"][s]["eligible_pairs"] for s in ("all", "joined", "ranged")})
            if units.setdefault((row["id"], row["samples"]), unit) != unit:
                raise ValueError(f"the rows of unit {row['id']} disagree on its metadata or pair counts")
    for method in METHODS:
        if {k[:2] for k in scored if k[2] == method} != set(units):
            raise ValueError(f"{method} lacks scored one-anchor or two-anchor units")
    return scored, joined, stayed, units, {k: list(v) for k, v in cells.items()}


def scenarios_of(units, tables, runs, sources):
    """Positions from the frozen campaign base, anchors by the panel's anchor order, and every reading of every agent
    and anchor link of the track (sides.idlab, for one and two anchors and keeping the readings)."""
    run = json.load(open(sides.checked(sides.one((e for e in tables["inputs"] if e["path"].endswith(
        f"/baselines/{DATASET}/evaluation/anc/run.json")), "baseline run record"), runs, sources, f"{DATASET}/baseline_run")))
    freezes = {run["runs"][m]["campaign_freeze_sha256"] for m in BASELINES}
    manifest = json.load(open(CAMPAIGN / "manifest.json"))
    sides.hashed(CAMPAIGN / "manifest.json", sources, f"{DATASET}/campaign_manifest")
    if freezes != {manifest["files"]["freeze.json"]} or sides.hashed(CAMPAIGN / "freeze.json", sources, f"{DATASET}/campaign_freeze") not in freezes:
        raise ValueError("the campaign freeze is not the one FNL and STRONG ran under")
    if sides.hashed(CAMPAIGN / "base.json.gz", sources, f"{DATASET}/base") != manifest["files"]["base.json.gz"]:
        raise ValueError("the base is not the one the campaign's manifest lists")
    settings = {m: run["runs"][m]["selections"][m] for m in BASELINES}
    for m in BASELINES:
        if settings[m]["solve_config"].get("aggregation") != "AllSamples" or settings[m]["method_config"].get("initialization") != "solid-geometry":
            raise ValueError(f"{m} did not run from the geometry start with every sample, which the start rule below assumes")
    base = json.loads(gzip.decompress((CAMPAIGN / "base.json.gz").read_bytes()))
    coordinates = {int(n): p for n, p in base["coordinates_cm"].items()}
    values = {(node, row): cm for node, row, cm in base["values"]}
    correction, scale = base["fitted"]["correction"], base["fitted"]["profiles"]["scale_cm"]
    scenarios, readings = {}, {}
    for (sid, samples), unit in sorted(units.items()):
        panel = base["panels"][unit["panel"]]
        nodes = panel["nodes"]
        if panel["configuration"] != unit["configuration"] or len(nodes) != IDLAB_RADIOS:
            raise ValueError(f"scenario {sid}: panel {unit['panel']} is not a panel of the 15 office radios")
        subsets = subsets_of(panel["anchor_order"])
        index = sid - unit["panel"] * (ROLES + 1) - 2
        named = subsets[index] if 0 <= index < len(subsets) else ()
        if len(named) != unit["anchors"]:
            raise ValueError(f"scenario {sid} is not a {unit['anchors']}-anchor role of panel {unit['panel']}")
        agents = [n for n in nodes if n not in named]
        anchors = [(n, *coordinates[n][:3]) for n in named]
        scenario = dict(agents=agents, anchors=sides.initializer_order(anchors) if len(anchors) == 2 else anchors,
                        truth={n: tuple(coordinates[n][:2]) for n in agents}, group=None)
        if scenarios.setdefault(sid, scenario) != scenario:
            raise ValueError(f"scenario {sid}: its tracks differ")
        links = {}
        for link in panel["links"]:
            a, b = link["nodes"]
            if (a in named) == (b in named):
                continue
            agent, anchor = (b, a) if a in named else (a, b)
            identities = link["one" if samples == 1 else "up_to_ten"]
            if not identities:
                continue
            if (agent, anchor) in links or len(identities) > samples:
                raise ValueError(f"scenario {sid}: link {agent}-{anchor} is not one link of at most {samples} readings")
            links[(agent, anchor)] = [max(0.0, correction["gain"] * values[tuple(i)] + correction["offset_cm"]) for i in identities]
        readings[(sid, samples)] = dict(links=links, scale=scale)
    return scenarios, readings


def unit_links(scenario, reading):
    """Per agent the anchors of its measured links and of its admissible readings, and per admissible link the median
    horizontal range of the start: initializer.rs takes every admissible sample (aggregation AllSamples), its
    horizontal part sqrt(max(0, d^2 - v^2)) with v the anchor's height less the median of the anchor heights, and
    their median. An agent that ranges one anchor starts at (x + range, y) of that anchor (laterate)."""
    heights = [a[3] for a in scenario["anchors"]]
    plane = engine_median(heights)
    place = {a[0]: a for a in scenario["anchors"]}
    measured, ranged, start_range = defaultdict(set), defaultdict(set), {}
    for (agent, anchor), distances in reading["links"].items():
        measured[agent].add(anchor)
        admitted = [d for d in distances if sides.admissible(d, reading["scale"])]
        if admitted:
            ranged[agent].add(anchor)
            vertical = place[anchor][3] - plane
            start_range[(agent, anchor)] = engine_median(
                [math.sqrt(max(0.0, d * d - vertical * vertical)) for d in admitted])
    return {g: frozenset(measured[g]) for g in scenario["agents"]}, {g: frozenset(ranged[g]) for g in scenario["agents"]}, start_range


def start_of(scenario, ranged, start_range, agent):
    """The start initializer.rs gives an agent that ranges one or two anchors (laterate, two_anchor_initialization),
    None for an agent without a range."""
    place = {a[0]: a for a in scenario["anchors"]}
    anchors = [a for a in scenario["anchors"] if a[0] in ranged[agent]]
    if len(anchors) == 1:
        (anchor, x, y, _), = anchors
        return (x + start_range[(agent, anchor)], y)
    if len(anchors) == 2:
        (first, ax, ay, _), (second, bx, by, _) = anchors
        ra, rb = start_range[(agent, first)], start_range[(agent, second)]
        dx, dy = bx - ax, by - ay
        separation = max(math.hypot(dx, dy), 1e-9)
        along = (ra * ra - rb * rb + separation * separation) / (2.0 * separation)
        height = math.sqrt(max(0.0, ra * ra - along * along))
        return (ax + along * dx / separation - height * dy / separation, ay + along * dy / separation + height * dx / separation)
    return None


def load(runs, sources, tables_root=sides.TABLES):
    """Every input of the record: the scores, the rebuilt scenarios and links, and the matched estimates."""
    tables_path = tables_root / f"{DATASET}/evaluation/tables.json"
    sides.hashed(tables_path, sources, f"{DATASET}/tables")
    tables = json.load(open(tables_path))
    scored, joined, stayed, units, cells = scores_of(tables, runs, sources)
    scenarios, readings = scenarios_of(units, tables, runs, sources)
    for scenario in scenarios.values():
        sides.prepared(scenario)
    links = {key: unit_links(scenarios[key[0]], reading) for key, reading in readings.items()}
    checks = Counter()
    for key, unit in units.items():
        measured, ranged, _ = links[key]
        agents = scenarios[key[0]]["agents"]
        checks["units"] += 1
        checks["all_pairs_agree"] += sides.comb2(len(agents)) == unit["all_pairs"]
        checks["joined_pairs_agree"] += sides.comb2(sum(1 for g in agents if measured[g])) == unit["joined_pairs"]
        checks["ranged_pairs_agree"] += sides.comb2(sum(1 for g in agents if ranged[g])) == unit["ranged_pairs"]
        checks["agents_measured_without_admissible_reading"] += sum(1 for g in agents if measured[g] and not ranged[g])
    if not checks["units"] == checks["all_pairs_agree"] == checks["joined_pairs_agree"] == checks["ranged_pairs_agree"]:
        raise ValueError(f"the rebuilt links disagree with the pair counts of the score rows {dict(checks)}")
    confirmed = Counter()
    estimates, matching = sides.baseline_estimates(DATASET, scored, units, scenarios, runs, sources, confirmed)
    estimates_solid, matching["solid"] = sides.solid_estimates(DATASET, tables, scored, units, scenarios, runs, sources, confirmed)
    estimates.update(estimates_solid)
    for method in METHODS:
        if sum(1 for k in estimates if k[2] == method) != sum(1 for k in scored if k[2] == method):
            raise ValueError(f"{method}: not every scored row has its estimate {matching[method]}")
    # the baselines' prediction folders hold thousands of files: keep their count and the digest of their listing
    for folder, entry in sources[f"{DATASET}/baseline_estimates"].items():
        listing = "".join(f"{name}  {digest}\n" for name, digest in sorted(entry["sha256"].items()))
        sources[f"{DATASET}/baseline_estimates"][folder] = dict(files=entry["files"], listing_sha256=hashlib.sha256(listing.encode()).hexdigest())
    return dict(tables=tables, scored=scored, joined=joined, stayed=stayed, units=units, cells=cells, scenarios=scenarios,
                links=links, estimates=estimates, matching=matching, link_checks=dict(sorted(checks.items())),
                scenarios_confirmed=sum(1 for sid in scenarios if confirmed[sid]), estimates_checked=sum(confirmed.values()))


# ------------------------------------------------------------------------------------------- statistics
def spread(values, digits=1):
    """Count, mean and quartiles (statistics.quantiles, inclusive) of a list."""
    values = list(values)
    if not values:
        return dict(count=0)
    p25, p50, p75 = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return dict(count=len(values), mean=round(statistics.fmean(values), digits), p25=round(p25, digits),
                median=round(p50, digits), p75=round(p75, digits))


def mean(values, digits=1):
    values = list(values)
    return round(statistics.fmean(values), digits) if values else None


def wrapped(degrees):
    """An angle in degrees within [-180, 180)."""
    return (degrees + 180.0) % 360.0 - 180.0


def bearing(origin, point):
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0]))


def bearing_errors(values):
    """The absolute differences between estimated and true bearings, with the shares a uniform bearing gives 0.25 each."""
    entry = spread(values)
    if values:
        entry.update(share_within_45_deg=sides.ratio(sum(v <= 45 for v in values), len(values)),
                     share_beyond_135_deg=sides.ratio(sum(v >= 135 for v in values), len(values)))
    return entry


# ------------------------------------------------------------------------------------------- the joined pairs
def pair_type(ranged, g, h):
    a, b = ranged[g], ranged[h]
    if len(a) == 2 or len(b) == 2:
        return "with_both"
    if len(a) == 1 and len(b) == 1:
        return "same_anchor" if a == b else "different_anchors"
    return "other"


def scored_units(data, anchors, samples, checks):
    """Every unit of a track that has a joined pair: its joined pairs in ascending order (scoring.joined_pairs), their
    types and true distances, and per method every pair's estimated distance and error (eq. pair-error, the cap where a
    point is missing) and their mean, which must give the joined error of the score row."""
    units = []
    for key in sorted(k for k, u in data["units"].items() if u["anchors"] == anchors and k[1] == samples):
        s = data["scenarios"][key[0]]
        measured, ranged, _ = data["links"][key]
        pairs = list(itertools.combinations(sorted(g for g in s["agents"] if measured[g]), 2))
        if not pairs:
            if any(data["joined"][(*key, m)] is not None for m in METHODS):
                raise ValueError(f"{key}: a joined error without a joined pair")
            continue
        unit = dict(key=key, cell=data["units"][key]["cell"], pairs=pairs, types=[pair_type(ranged, g, h) for g, h in pairs],
                    true=[math.dist(s["truth"][g], s["truth"][h]) for g, h in pairs], est={}, err={}, error={})
        for m in METHODS:
            distance = PAIR_DISTANCE[m]
            point = dict(zip(s["agents"], data["estimates"][(*key, m)]))
            est = [None if point[g] is None or point[h] is None else distance(point[g], point[h]) for g, h in pairs]
            err = [s["cap"] if e is None else min(abs(e - distance(s["truth"][g], s["truth"][h])), s["cap"])
                   for e, (g, h) in zip(est, pairs)]
            error, stored = sum(err) / len(err), data["joined"][(*key, m)]
            relative_gap = sides.gap(error, stored)
            if relative_gap > sides.MATCH:
                raise ValueError(f"{key} {m}: the joined error {error} is not the score row's {stored}")
            tally = checks[(anchors, samples, m)]
            tally["units"] += 1
            tally["bit_for_bit"] += error == stored
            tally["largest_relative_gap"] = max(tally["largest_relative_gap"], relative_gap)
            tally["missing_points"] += sum(e is None for e in est)
            unit["est"][m], unit["err"][m], unit["error"][m] = est, err, error
        units.append(unit)
    counts = Counter(u["cell"] for u in units)
    for u in units:
        u["weight"] = 1.0 / (len(counts) * counts[u["cell"]])  # the weight of a unit in the balanced percentile
    return units


def balanced(scorer, order, units, value, alpha):
    """The released balanced percentile of one value of the units, the cells in the order the tables meet them."""
    cells = {c: [] for c in order}
    for u in units:
        cells[u["cell"]].append(value(u))
    if set(cells) != {u["cell"] for u in units}:
        raise ValueError("the cell order lacks a cell of the units")
    return scorer.balanced_percentile(cells, alpha)


def type_table(units, ranged_of):
    """Per type its pairs and share of the joined pairs, the mean true distance, and per method the mean pair error, the
    mean estimated distance, the mean of estimated less true distance and the share of pairs estimated shorter."""
    groups = defaultdict(list)
    for u in units:
        ranged = ranged_of(u)
        for i, ((g, h), t) in enumerate(zip(u["pairs"], u["types"])):
            groups[t].append((u, i))
            if t == "with_both":
                groups["both_agents_range_both" if len(ranged[g]) == 2 and len(ranged[h]) == 2 else "one_agent_ranges_both"].append((u, i))
    total = sum(len(groups[t]) for t in TYPES)

    def entry(members):
        if not members:
            return dict(pairs=0)
        result = dict(pairs=len(members), mean_true_distance_cm=mean(u["true"][i] for u, i in members), methods={})
        for m in METHODS:
            placed = [(u["est"][m][i], u["true"][i]) for u, i in members if u["est"][m][i] is not None]
            result["methods"][m] = dict(
                mean_pair_error_cm=mean(u["err"][m][i] for u, i in members),
                mean_estimated_distance_cm=mean(e for e, _ in placed),
                mean_estimated_minus_true_cm=mean(e - t for e, t in placed),
                share_estimated_shorter=sides.ratio(sum(e < t for e, t in placed), len(placed)))
        return result

    table = {t: dict(rule=TYPE_RULES[t], share_of_joined_pairs=sides.ratio(len(groups[t]), total), **entry(groups[t])) for t in TYPES}
    table["with_both"]["parts"] = {p: entry(groups[p]) for p in WITH_BOTH_PARTS}
    return table


def contributions(u, m):
    """Per type the sum of a method's pair errors of the type over the unit's joined pairs, which add up to its error."""
    total = dict.fromkeys(TYPES, 0.0)
    for t, e in zip(u["types"], u["err"][m]):
        total[t] += e
    return {t: v / len(u["pairs"]) for t, v in total.items()}


def type_shares(u):
    counts = Counter(u["types"])
    return {t: counts[t] / len(u["pairs"]) for t in TYPES}


def weighted(units, value):
    """The balanced mean of a value per type over a set of units, with the weights of the balanced percentile."""
    mass = sum(u["weight"] for u in units)
    sums = dict.fromkeys(TYPES, 0.0)
    for u in units:
        for t, v in value(u).items():
            sums[t] += u["weight"] * v
    return {t: v / mass for t, v in sums.items()}, mass


def rounded(values, digits=1):
    return {k: round(v, digits) for k, v in values.items()}


def difference(units, scorer, cells, a, b, anchors, samples, tables_s75):
    """How much of the difference between a's and b's errors each pair type carries, over every scored unit, over the
    units that set each method's S75, and in the S75 itself through hybrid errors."""
    def terms(u):
        ca, cb = contributions(u, a), contributions(u, b)
        return {t: ca[t] - cb[t] for t in TYPES}

    mean_terms, _ = weighted(units, terms)
    plain = {t: round(statistics.fmean(terms(u)[t] for u in units), 1) for t in TYPES}
    signs = Counter("a_higher" if u["error"][a] > u["error"][b] else "b_higher" if u["error"][a] < u["error"][b] else "equal" for u in units)
    all_scored = dict(units=len(units), balanced_mean_terms_cm=rounded(mean_terms), balanced_mean_difference_cm=round(sum(mean_terms.values()), 1),
                      mean_terms_cm=plain, units_with_higher_error={a: signs["a_higher"], b: signs["b_higher"], "equal": signs["equal"]})
    bands = {}
    for m in (a, b):
        low, high = (balanced(scorer, cells[(m, anchors, samples)], units, lambda u: u["error"][m], q) for q in BAND)
        members = [u for u in units if low <= u["error"][m] <= high]
        composition, mass = weighted(members, lambda u: contributions(u, m))
        shares, _ = weighted(members, lambda u: {t: 100.0 * v for t, v in type_shares(u).items()})
        paired, _ = weighted(members, terms)
        bands[m] = dict(S65_cm=round(low, 1), S85_cm=round(high, 1), units=len(members), balanced_mass=round(mass, 4),
                        balanced_mean_error_cm=round(sum(composition.values()), 1), balanced_mean_error_by_type_cm=rounded(composition),
                        balanced_mean_type_share_percent=rounded(shares), balanced_mean_terms_cm=rounded(paired), composition=composition)
    band_difference = {t: bands[a]["composition"][t] - bands[b]["composition"][t] for t in TYPES}
    for m in (a, b):
        del bands[m]["composition"]
    # hybrid errors: a's pair errors, b's on the pairs of the types in S
    present = [t for t in TYPES if any(t in u["types"] for u in units)]

    def hybrid(replaced):
        return balanced(scorer, cells[(a, anchors, samples)], units,
                        lambda u: sum(eb if t in replaced else ea for t, ea, eb in zip(u["types"], u["err"][a], u["err"][b])) / len(u["pairs"]), 0.75)
    s75 = {frozenset(S): hybrid(set(S)) for size in range(len(present) + 1) for S in itertools.combinations(present, size)}
    if s75[frozenset()] != tables_s75[a] or s75[frozenset(present)] != tables_s75[b]:
        raise ValueError(f"the hybrid errors do not give the S75 of {a} and {b} at both ends {s75[frozenset()]} {s75[frozenset(present)]}")
    shapley = {}
    for t in present:
        others, value = [x for x in present if x != t], 0.0
        for size in range(len(others) + 1):
            for S in itertools.combinations(others, size):
                weight = math.factorial(size) * math.factorial(len(present) - size - 1) / math.factorial(len(present))
                value += weight * (s75[frozenset(S)] - s75[frozenset(S) | {t}])
        shapley[t] = round(value, 1)
    name = lambda S: "+".join(t for t in present if t in S) or "none"
    hybrid_record = dict(
        S75_cm_with_b_errors_on={name(S): round(v, 1) for S, v in s75.items()},
        gap_cm=round(tables_s75[a] - tables_s75[b], 1),
        closed_by_replacing_one_type_cm={t: round(tables_s75[a] - s75[frozenset([t])], 1) for t in present},
        shapley_share_of_gap_cm=shapley)
    return dict(all_scored=all_scored, S75_bands=bands, S75_bands_difference_by_type_cm=rounded(band_difference),
                S75_bands_difference_cm=round(sum(band_difference.values()), 1), hybrid_S75=hybrid_record)


# ------------------------------------------------------------------------------------------- agents and geometry
def agents_record(data, anchors, samples, starts):
    """Where every method puts the agents that range one anchor, both anchors or none, against the start and the truth."""
    single = dict(offset=defaultdict(list), bearing=defaultdict(list), at_start=Counter(), on_ray=Counter(), toward=defaultdict(list),
                  toward_by_partner=defaultdict(list), true_bearing=[], ranges=[])
    both = defaultdict(list)
    unlinked = dict(agents=0, pairs=0, joined_pairs=0, fnl_equals_strong=0, from_truth=defaultdict(list), from_anchor=defaultdict(list))
    for key in sorted(k for k, u in data["units"].items() if u["anchors"] == anchors and k[1] == samples):
        s = data["scenarios"][key[0]]
        measured, ranged, start_range = data["links"][key]
        place = {a[0]: a for a in s["anchors"]}
        point = {m: dict(zip(s["agents"], data["estimates"][(*key, m)])) for m in METHODS}
        point["truth"] = s["truth"]
        # a ranged agent of a solve the scorer flags as stayed sits at the start rebuilt here (start_of)
        for m in BASELINES:
            if data["stayed"][(*key, m)]:
                for g in s["agents"]:
                    if ranged[g]:
                        start, tally = start_of(s, ranged, start_range, g), starts[(m, len(ranged[g]))]
                        tally["agents_of_stayed_solves"] += 1
                        tally["at_rebuilt_start_bit_for_bit"] += point[m][g] == start
                        tally["largest_distance_from_rebuilt_start_cm"] = max(tally["largest_distance_from_rebuilt_start_cm"], math.dist(point[m][g], start))
        lone = [g for g in s["agents"] if len(ranged[g]) == 1]
        for g in s["agents"]:
            if len(ranged[g]) == 1:
                (anchor,) = ranged[g]
                origin = place[anchor][1:3]
                radius = start_range[(g, anchor)]
                start = start_of(s, ranged, start_range, g)
                true_bearing = bearing(origin, s["truth"][g])
                single["true_bearing"].append(abs(wrapped(true_bearing)))
                single["ranges"].append(radius)
                other = [a for a in place if a != anchor]
                partners = sum(1 for h in lone if ranged[h] != ranged[g])
                for m in ("truth", *METHODS):
                    p = point[m][g]
                    if p is None:
                        continue
                    single["offset"][m].append(math.dist(p, origin) - radius)
                    if m != "truth":
                        single["bearing"][m].append(abs(wrapped(bearing(origin, p) - true_bearing)))
                        single["at_start"][m] += p == start
                        single["on_ray"][m] += p[1] == origin[1] and p[0] > origin[0]
                    if other:
                        (b,) = other
                        far = place[b][1:3]
                        separation = math.dist(origin, far)
                        toward = ((p[0] - origin[0]) * (far[0] - origin[0]) + (p[1] - origin[1]) * (far[1] - origin[1])) / separation
                        single["toward"][m].append(toward)
                        single["toward_by_partner"][m] += [toward] * partners
            elif len(ranged[g]) == 2:
                for m in ("truth", *METHODS):
                    p = point[m][g]
                    if p is not None:
                        both[m] += [abs(math.dist(p, place[a][1:3]) - start_range[(g, a)]) for a in sorted(ranged[g])]
            elif not measured[g]:
                unlinked["agents"] += 1
                unlinked["fnl_equals_strong"] += point["fnl"][g] == point["strong"][g]
                for m in METHODS:
                    p = point[m][g]
                    if p is not None:
                        unlinked["from_truth"][m].append(math.dist(p, s["truth"][g]))
                        unlinked["from_anchor"][m].append(min(math.dist(p, a[1:3]) for a in s["anchors"]))
        linked = sum(1 for g in s["agents"] if measured[g])
        unlinked["pairs"] += sides.comb2(len(s["agents"])) - sides.comb2(linked)
        # the joined pairs of the score row beyond the pairs of two linked agents, the joined pairs with an unlinked agent
        unlinked["joined_pairs"] += data["units"][key]["joined_pairs"] - sides.comb2(linked)
    count = len(single["ranges"])
    record = dict(
        single_anchor_agents=dict(
            agents=count,
            start_range_cm=spread(single["ranges"]),
            true_bearing_from_x_axis_deg=dict(spread(single["true_bearing"]), share_east=sides.ratio(sum(v < 90 for v in single["true_bearing"]), count)),
            circle_offset_cm={m: dict(spread(single["offset"][m]), median_absolute=round(statistics.median(abs(v) for v in single["offset"][m]), 1),
                                      share_within_1cm=sides.ratio(sum(abs(v) <= 1.0 for v in single["offset"][m]), len(single["offset"][m])))
                              for m in ("truth", *METHODS)},
            bearing_error_deg={m: bearing_errors(single["bearing"][m]) for m in METHODS},
            at_start={m: single["at_start"][m] for m in METHODS},
            on_x_axis_ray_of_anchor={m: single["on_ray"][m] for m in METHODS}),
        both_anchor_agents=dict(agents=len(both["truth"]) // 2,
                                circle_offset_absolute_cm={m: spread(both[m]) for m in ("truth", *METHODS)}) if anchors == 2 else None,
        agents_without_link=dict(
            agents=unlinked["agents"], pairs_with_them=unlinked["pairs"], joined_pairs_with_them=unlinked["joined_pairs"],
            fnl_and_strong_identical=unlinked["fnl_equals_strong"],
            distance_from_true_position_cm={m: spread(unlinked["from_truth"][m]) for m in METHODS},
            distance_from_nearest_anchor_cm={m: spread(unlinked["from_anchor"][m]) for m in METHODS}))
    if anchors == 2:
        record["single_anchor_agents"]["toward_other_anchor_cm"] = {
            m: dict(spread(single["toward"][m]), share_toward=sides.ratio(sum(v > 0 for v in single["toward"][m]), len(single["toward"][m])))
            for m in ("truth", *METHODS)}
        record["single_anchor_agents"]["toward_other_anchor_per_different_anchor_partner_cm"] = {
            m: dict(count=len(single["toward_by_partner"][m]), mean=mean(single["toward_by_partner"][m]),
                    share_toward=sides.ratio(sum(v > 0 for v in single["toward_by_partner"][m]), len(single["toward_by_partner"][m])))
            for m in ("truth", *METHODS)}
    return record


def same_anchor_angles(data, units):
    """The angle at their common anchor between the two agents of every pair of type (a), true and estimated."""
    angles = defaultdict(list)
    for u in units:
        s = data["scenarios"][u["key"][0]]
        _, ranged, _ = data["links"][u["key"]]
        place = {a[0]: a[1:3] for a in s["anchors"]}
        point = {m: dict(zip(s["agents"], data["estimates"][(*u["key"], m)])) for m in METHODS}
        point["truth"] = s["truth"]
        for (g, h), t in zip(u["pairs"], u["types"]):
            if t != "same_anchor":
                continue
            (anchor,) = ranged[g]
            for m in ("truth", *METHODS):
                if point[m][g] is not None and point[m][h] is not None:
                    angles[m].append(abs(wrapped(bearing(place[anchor], point[m][g]) - bearing(place[anchor], point[m][h]))))
    return {m: spread(angles[m]) for m in ("truth", *METHODS)}


def separation_labels():
    """The bins of the separation over the summed start ranges: below the first edge, between two edges, beyond the last."""
    edges = SEPARATION_BINS
    return [f"below {edges[0]}", *(f"{low} to {high}" for low, high in zip(edges, edges[1:])), f"{edges[-1]} or more"]


def different_anchor_pairs(data, units):
    """The pairs of type (b) against the distance L between their two anchors: the true and estimated distances less L,
    the parts of the pair's vector along and across the line through the anchors, and the errors by L over the sum of
    the two start ranges."""
    rows = defaultdict(lambda: defaultdict(list))
    for u in units:
        s = data["scenarios"][u["key"][0]]
        _, ranged, start_range = data["links"][u["key"]]
        place = {a[0]: a[1:3] for a in s["anchors"]}
        point = {m: dict(zip(s["agents"], data["estimates"][(*u["key"], m)])) for m in METHODS}
        point["truth"] = s["truth"]
        for i, ((g, h), t) in enumerate(zip(u["pairs"], u["types"])):
            if t != "different_anchors":
                continue
            (a,), (b,) = ranged[g], ranged[h]
            A, B = place[a], place[b]
            separation = math.dist(A, B)
            ex, ey = (B[0] - A[0]) / separation, (B[1] - A[1]) / separation
            ratio = separation / (start_range[(g, a)] + start_range[(h, b)])
            label = next((name for name, edge in zip(separation_labels(), SEPARATION_BINS) if ratio < edge), separation_labels()[-1])
            for group in ("all", label):
                rows[group]["separation"].append(separation)
                for m in ("truth", *METHODS):
                    pg, ph = point[m][g], point[m][h]
                    if pg is None or ph is None:
                        continue
                    og, oh = (pg[0] - A[0], pg[1] - A[1]), (ph[0] - B[0], ph[1] - B[1])
                    rows[group][(m, "minus_separation")].append(math.dist(pg, ph) - separation)
                    # how far the two agents sit toward each other's anchor along the line, and their offset across it
                    rows[group][(m, "toward")].append((og[0] * ex + og[1] * ey) - (oh[0] * ex + oh[1] * ey))
                    rows[group][(m, "across")].append((og[0] * ey - og[1] * ex) - (oh[0] * ey - oh[1] * ex))
                    if m != "truth":
                        rows[group][(m, "error")].append(u["err"][m][i])
    record = {}
    for group in ("all", *separation_labels()):
        r = rows[group]
        record[group] = dict(
            pairs=len(r["separation"]), separation_cm=spread(r["separation"]),
            distance_minus_separation_cm={m: dict(spread(r[(m, "minus_separation")]),
                                                  share_shorter=sides.ratio(sum(v < 0 for v in r[(m, "minus_separation")]), len(r[(m, "minus_separation")])))
                                          for m in ("truth", *METHODS)},
            toward_each_other_along_line_cm={m: mean(r[(m, "toward")]) for m in ("truth", *METHODS)},
            rms_across_line_cm={m: round(math.sqrt(statistics.fmean(v * v for v in r[(m, "across")])), 1) if r[(m, "across")] else None
                                for m in ("truth", *METHODS)},
            mean_pair_error_cm={m: mean(r[(m, "error")]) for m in METHODS})
    return record


def analyze(data, scorer):
    tables = data["tables"]
    checks = defaultdict(lambda: Counter(largest_relative_gap=0.0))
    starts = defaultdict(lambda: Counter(largest_distance_from_rebuilt_start_cm=0.0))
    result, reproduced = {}, {}
    for anchors in ANCHOR_COUNTS:
        for samples in TRACKS:
            track = tables["modes"]["ANC"]["groups"][f"anchors-{anchors}"]["tracks"][str(samples)]
            joined_table = track["pair_sets"]["joined"]
            units = scored_units(data, anchors, samples, checks)
            if len(units) != joined_table["scored_units"] or len({u["cell"] for u in units}) != joined_table["cells"]:
                raise ValueError(f"anchors {anchors}, {samples} readings: other scored units or cells than the tables")
            if sum(len(u["pairs"]) for u in units) != joined_table["pairs"]:
                raise ValueError(f"anchors {anchors}, {samples} readings: other joined pairs than the tables")
            s75 = {m: joined_table["methods"][m]["S75"]["value_cm"] for m in METHODS}
            for m in METHODS:
                order = data["cells"][(m, anchors, samples)]
                from_rows = balanced(scorer, order, units, lambda u: data["joined"][(*u["key"], m)], 0.75)
                from_estimates = balanced(scorer, order, units, lambda u: u["error"][m], 0.75)
                reproduced[f"{anchors} anchors, {samples} readings, {m}"] = dict(
                    tables=s75[m], from_score_rows=from_rows, from_estimates=from_estimates)
                if from_rows != s75[m] or from_estimates != s75[m]:
                    raise ValueError(f"anchors {anchors}, {samples} readings, {m}: the S75 is not the tables' {s75[m]}")
            ranged_of = lambda u: data["links"][u["key"]][1]
            entry = dict(
                scored_units=len(units), cells=joined_table["cells"], joined_pairs=joined_table["pairs"],
                S75_cm={m: round(v, 1) for m, v in s75.items()},
                pair_types=type_table(units, ranged_of),
                **{f"{a}_minus_{b}": difference(units, scorer, data["cells"], a, b, anchors, samples, s75) for a, b in COMPARISONS},
                same_anchor_pair_angle_at_anchor_deg=same_anchor_angles(data, units),
                **agents_record(data, anchors, samples, starts))
            if anchors == 2:
                entry["different_anchor_pairs"] = different_anchor_pairs(data, units)
            else:
                del entry["both_anchor_agents"]
            # the premises the notes state
            unlinked = entry["agents_without_link"]
            if entry["pair_types"]["other"]["pairs"] or unlinked["joined_pairs_with_them"] or unlinked["fnl_and_strong_identical"] != unlinked["agents"]:
                raise ValueError(f"anchors {anchors}, {samples} readings: a pair of type other, a joined pair with an unlinked agent, "
                                 "or an unlinked agent that FNL and STRONG place apart")
            result.setdefault(str(anchors), {})[str(samples)] = entry
            print(f"anchors {anchors}, {samples} readings:", json.dumps(dict(
                S75=entry["S75_cm"], shares={t: entry["pair_types"][t]["share_of_joined_pairs"] for t in TYPES},
                errors={t: {m: entry["pair_types"][t]["methods"][m]["mean_pair_error_cm"] for m in METHODS}
                        for t in TYPES if entry["pair_types"][t]["pairs"]},
                shapley=entry["solid_minus_strong"]["hybrid_S75"]["shapley_share_of_gap_cm"])), flush=True)
    joined_checks = {f"{a} anchors, {s} readings, {m}": dict(v) for (a, s, m), v in sorted(checks.items())}
    for (m, n), v in starts.items():
        if (n == 1 and v["at_rebuilt_start_bit_for_bit"] != v["agents_of_stayed_solves"]) or v["largest_distance_from_rebuilt_start_cm"] > 1e-9:
            raise ValueError(f"{m}: a stayed agent that ranges {n} anchors is not at the rebuilt start {dict(v)}")
    start_checks = {f"{m}, agents ranging {n} anchor{'s' if n > 1 else ''}": dict(v) for (m, n), v in sorted(starts.items())}
    return result, dict(joined_errors=joined_checks, S75=reproduced, starts=start_checks)


def summary_of(result):
    """The headline numbers per anchor count and reading count."""
    summary = {}
    for anchors, tracks in result.items():
        for samples, e in tracks.items():
            present = [t for t in TYPES if e["pair_types"][t]["pairs"]]
            gap = e["solid_minus_strong"]
            entry = dict(
                S75_cm=e["S75_cm"],
                share_of_joined_pairs={t: e["pair_types"][t]["share_of_joined_pairs"] for t in present},
                mean_pair_error_cm={t: {m: e["pair_types"][t]["methods"][m]["mean_pair_error_cm"] for m in METHODS} for t in present},
                mean_estimated_minus_true_distance_cm={t: {m: e["pair_types"][t]["methods"][m]["mean_estimated_minus_true_cm"] for m in METHODS}
                                                       for t in present},
                solid_minus_strong=dict(
                    S75_gap_cm=gap["hybrid_S75"]["gap_cm"],
                    shapley_share_of_gap_cm=gap["hybrid_S75"]["shapley_share_of_gap_cm"],
                    solid_S75_with_strong_errors_on_one_type_cm={t: gap["hybrid_S75"]["S75_cm_with_b_errors_on"][t] for t in present},
                    S75_band_mean_error_cm={m: gap["S75_bands"][m]["balanced_mean_error_cm"] for m in ("solid", "strong")},
                    S75_band_mean_error_by_type_cm={m: {t: gap["S75_bands"][m]["balanced_mean_error_by_type_cm"][t] for t in present}
                                                    for m in ("solid", "strong")},
                    balanced_mean_terms_over_scored_units_cm={t: gap["all_scored"]["balanced_mean_terms_cm"][t] for t in present}),
                same_anchor_pairs_median_angle_at_anchor_deg={m: e["same_anchor_pair_angle_at_anchor_deg"][m]["median"]
                                                              for m in ("truth", *METHODS)},
                single_anchor_agents=dict(
                    agents=e["single_anchor_agents"]["agents"],
                    on_x_axis_ray_of_anchor=e["single_anchor_agents"]["on_x_axis_ray_of_anchor"],
                    median_circle_offset_cm={m: e["single_anchor_agents"]["circle_offset_cm"][m]["median"] for m in ("truth", *METHODS)},
                    bearing_error_share_within_45_deg={m: e["single_anchor_agents"]["bearing_error_deg"][m]["share_within_45_deg"] for m in METHODS}))
            if anchors == "2":
                pairs = e["different_anchor_pairs"]
                entry["different_anchor_pairs"] = dict(
                    distance_minus_anchor_separation_cm={m: dict(mean=pairs["all"]["distance_minus_separation_cm"][m]["mean"],
                                                                 median=pairs["all"]["distance_minus_separation_cm"][m]["median"])
                                                         for m in ("truth", *METHODS)},
                    toward_each_other_along_line_cm=pairs["all"]["toward_each_other_along_line_cm"],
                    rms_across_line_cm=pairs["all"]["rms_across_line_cm"],
                    mean_pair_error_by_separation_over_summed_ranges_cm={
                        group: dict(pairs=v["pairs"], **v["mean_pair_error_cm"]) for group, v in pairs.items() if group != "all"})
            summary[f"{anchors} anchors, {samples} readings"] = entry
    return summary


NOTES = [
    "Inputs: the one-anchor and two-anchor rows of the score files the committed IDLab tables read (hashes checked), "
    "and every scored row of FNL, STRONG and SOLID matched to its estimate as in anchor_sides.py: an estimate counts when "
    "the pair error it gives over every agent pair under the scorer's arithmetic equals the scored error_cm within a "
    "relative 1e-12 (checks.matching, as anchor_sides.py records it, the older STRONG run in FNL's folder included). "
    "Scenarios are rebuilt from the frozen campaign base as anchor_sides.idlab does, for one and two anchors, keeping every "
    "reading, and the rebuilt links give the all, joined and ranged pair counts of every score row (checks.link_checks).",
    "Joined pairs and errors: in ANC a pair is joined when both agents have a measured anchor link, an observation "
    "whatever its readings (scoring.joined_pairs), and a unit is scored when it has a joined pair. A pair's error is "
    "min(|estimated - true horizontal distance|, diameter of the true layout) (eq. pair-error), the diameter where a "
    "point is missing (none is). Each file's joined errors are reproduced with its own arithmetic: the composed baseline "
    "scores with the released scorer's dist, SOLID's with pilot.score_records' per pair errors (math.dist), averaged over "
    "the joined pairs in ascending order (scoring.rescore). checks.joined_errors counts the units reproduced bit for bit "
    "and the largest relative gap of the others, and the tables' S75 follows bit for bit from the score rows' joined "
    "errors and from the recomputed ones (checks.S75), with the released balanced percentile and the cells in the order "
    "the tables meet them.",
    "Ranging: an agent ranges an anchor when it has an admissible reading of it (finite, not negative, positive scale, "
    "both at most 1e5 cm, CalibratedRangeSample::is_admissible). On IDLab every measured link holds an admissible "
    "reading (checks.link_checks.agents_measured_without_admissible_reading is 0), so the joined agents are the ranged "
    "ones and the type 'other' has no pair. with_both.parts splits type (c) into pairs of two agents that range both "
    "anchors and pairs where only one does.",
    "Start: the FNL and STRONG runs behind the tables start from the engine's geometric layout and take every reading "
    "(run record selections: initialization solid-geometry, aggregation AllSamples, checked). initializer.rs then gives "
    "an agent's horizontal range to an anchor as the median, over its admissible readings d, of sqrt(max(0, d^2 - v^2)) "
    "with v the anchor's height less the median of the anchor heights (0 with one anchor, half the height difference with "
    "two), its median taking the mean of the two middle values of an even count. An agent that ranges one anchor starts at "
    "(x + range, y) of that anchor, one that ranges two at the intersection of the two circles (two_anchor_initialization). "
    "checks.starts compares this rebuilt start with every ranged agent of the solves the score rows flag as stayed at their "
    "start: it holds bit for bit for every agent that ranges one anchor, and the circle intersections differ in the last "
    "bits of their arithmetic only (the largest distance given there, the script refuses more than 1e-9 cm).",
    "Circle offset: the horizontal distance of a position from the anchor the agent ranges, less the start's range; for "
    "the truth it is the error of the start's range. Bearing: the direction of the horizontal offset from that anchor; "
    "bearing_error_deg is |estimated - true bearing| wrapped to [0, 180], where a bearing independent of the true one "
    "puts 25 percent within 45 degrees and 25 percent beyond 135. on_x_axis_ray_of_anchor counts estimates with the "
    "anchor's y and a larger x. toward_other_anchor_cm is the component of an agent's offset from its anchor along the "
    "direction to the other anchor (positive toward it), per agent and once per agent of the other anchor in its scenario, "
    "the weighting of the pairs of type (b).",
    "Balanced weights: a scored unit weighs 1/(C n) with C the cells of its track and n the scored units of its cell, "
    "the weights of the released balanced percentile, whose cells on IDLab are the configurations (cell 'configuration/"
    "anchors-k'). Balanced means use these weights, mean_terms_cm gives the plain mean.",
    "Decomposition: a unit's joined error is the mean of its joined pair errors, so the difference between two methods' "
    "errors of a unit is the sum over the pair types of the type's term, the sum of the differences of its pairs' errors "
    "divided by the unit's joined pairs. all_scored gives the terms' means over every scored unit and how many units "
    "each method errs more on.",
    "S75 bands: the scenarios that set a method's S75 are its scored units whose error lies between its balanced S65 and "
    "S85, both included, a balanced mass of 0.2. balanced_mean_error_by_type_cm splits their balanced mean error into "
    "the types' contributions, and S75_bands_difference compares the scenarios that set SOLID's S75 with those that set "
    "the other method's, which differ. balanced_mean_terms_cm gives the paired terms within one method's band, which select "
    "on that method's error and so lean toward the other method; the hybrid S75 selects nothing.",
    "Hybrid S75: for a set S of pair types, the released balanced percentile at 0.75, over the same cells, of every scored "
    "unit's mean over its joined pairs of SOLID's pair error, the other method's on the pairs of the types in S. With S "
    "empty it is SOLID's S75 and with every type the other method's (checked). closed_by_replacing_one_type is SOLID's "
    "S75 less the hybrid S75 of one type, and the Shapley share of a type is the mean over the orders of the types of the "
    "fall of the hybrid S75 when the type is added, so the shares add up to SOLID's S75 less the other method's.",
    "Pairs of type (b): the separation is the horizontal distance between the two anchors. distance_minus_separation_cm "
    "takes the true and each estimated pair distance. toward_each_other_along_line_cm is the mean of the component of "
    "the first agent's offset from its anchor toward the second anchor plus the component of the second agent's offset "
    "toward the first anchor (positive: the agents closer along the line than their anchors), rms_across_line_cm the root "
    "mean square of the difference of the two offsets' components across the line. To first order a pair's distance is "
    "the separation less the first plus the square of the second over twice the separation. The bins of the separation "
    "over the sum of the two start ranges have the round edges 0.75, 1 and 1.5, and the record gives every bin.",
    "Frame: SOLID's anchor-only estimate is the layout of least capped pair-error risk over joint posterior draws "
    "(runtime/engine/src/anchor_only.rs, decide), whose descent moves pair distances only and keeps the centroid of its "
    "candidate, the draws' coordinate mean or medoid in the anchors' frame. Positions against the anchors (circle "
    "offsets, bearings, the parts along and across the line) describe the layout as it is placed in that frame, pair "
    "distances do not depend on it.",
    "Agents without a measured link: none of their pairs is joined (joined_pairs_with_them counts the joined pairs of the "
    "score rows beyond the pairs of two linked agents), so their positions do not enter the joined errors or the S75. FNL "
    "and STRONG give them identical coordinates (counted), the unit's start, which initializer.rs keeps for an agent "
    "without a range and neither solver has a factor to move. SOLID places them by its pair risk, which takes every "
    "agent pair of the unit.",
    "Not established here: SOLID's posterior holds a presence law on failed anchor attempts (runtime/engine/src/"
    "presence.rs, profile presence enabled with the maximum likelihood reach), under which an agent that ranges one anchor "
    "but not the other, which answered another agent, is less likely near the other anchor. That agrees with SOLID "
    "displacing such agents away from the other anchor, but the predictions record no reach and nothing is solved, so "
    "the record does not show that the presence law causes it.",
    "Rounding: centimetres and degrees to 0.1, shares to 4 decimals, counts exact; quartiles are statistics.quantiles "
    "with the inclusive method. S75_cm copies modes.ANC.groups['anchors-k'].tracks[readings].pair_sets.joined.methods[m]"
    ".S75.value_cm of the committed tables, and checks.S75 gives them unrounded.",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=sides.RUNS)
    parser.add_argument("--tables", type=Path, default=sides.TABLES)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    sources = {}
    sides.hashed(Path(__file__).resolve(), sources, "script")
    sides.hashed(Path(sides.__file__).resolve(), sources, "anchor_sides.py")
    scorer = released_scorer(sources)
    data = load(args.runs, sources, args.tables)
    result, checks = analyze(data, scorer)
    record = dict(
        sources=sources,
        rule=dict(match_relative=sides.MATCH, types=TYPE_RULES, with_both_parts=list(WITH_BOTH_PARTS), S75_band=list(BAND),
                  separation_bins=list(SEPARATION_BINS), comparisons=[f"{a}_minus_{b}" for a, b in COMPARISONS],
                  anchor_counts=[str(a) for a in ANCHOR_COUNTS], tracks=[str(t) for t in TRACKS]),
        checks=dict(matching=data["matching"], link_checks=data["link_checks"],
                    scenarios=dict(ids=len(data["scenarios"]), confirmed=data["scenarios_confirmed"], estimates_checked=data["estimates_checked"]),
                    **checks),
        summary=summary_of(result),
        anchors=result,
        notes=NOTES,
    )
    print(json.dumps(record["summary"], indent=1), flush=True)
    if args.out:
        args.out.write_text(json.dumps(record, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
