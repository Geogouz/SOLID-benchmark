"""In the anchor-only mode (ANC), how many agents with an anchor link have ranged links to fewer than three anchors, per
dataset, reading count and anchor count, and on TWR-CLOUD every method's S50, S75 and S90 over the scenarios whose agents
all have ranged links to three or more anchors, the scenarios with eight or more anchors, from the inputs and the unit
scores of the final tables. These are the numbers of the paragraph on well-linked agents in Section VIII of the paper.

An agent of a unit is well linked when it has ranged links to three or more distinct anchors of the unit, a ranged link
being an observation between the two with at least one sample of the unit's reading count that scoring.valid_samples
counts (scoring.successful_links, the rule of the paper's ranged pairs, a calibrated sample with a finite distance of at
least 0 and a finite scale above 0 or a raw time of flight within 0 and 200 ns). An agent has an anchor link when the
unit's measured links join it to the anchors, which makes it an agent of the unit's joined pairs (scoring.joined_pairs,
in ANC an agent with an observation to an anchor, whatever its samples). A pair of well-linked agents is a joined pair
whose two agents are well linked. The rule reads the inputs alone, so it is the same for every method.

Over the headline of the final tables' ANC tables, the units with three or more anchors, the record gives per dataset
and reading count the agents with an anchor link of the units that hold a joined pair and those of them that are not well
linked with their share, overall and per anchor count. It gives the units, the units that keep a pair of well-linked
agents and their reporting cells, the cells and the joined pairs of the tables, and the pairs of well-linked agents with
their share of the joined pairs.

The units are those of SOLID's anchor-only score file, an input of research/artifacts/solid/tables/<dataset>/evaluation/
tables.json that is checked against the SHA-256 the tables record (a recorded path below out/ lies below the runs root,
as in anchor_starts.py). Every unit's problem is built again from the data folder as its runner built it, on TWR-CLOUD
the released problem of every scenario of the scorer input (scoring.released_problem, read as scoring.score_scenarios
reads it), on TIERS and IDLab the campaign's request of every batch (batch_inputs of tiers_campaign.py and
idlab_campaign.py), and on HYMN, AAU and Bregar the request of the evaluation half. Every input is the one the runs read.
The scorer input has the SHA-256 that the joined scores of every TWR-CLOUD run record, a campaign's base and catalog are
those that its freeze, or on IDLab the manifest beside it, lists for the freeze that SOLID's run record names, and a
request has the SHA-256 that SOLID's run record names. The problems and the score rows must hold the same units, every
problem must give the joined and ranged pair counts of its score row, and its pairs of well-linked agents must be the
row's anchored-3 pairs where the row keeps them (scoring.pair_sets). Without the filter, the units, the scored units,
the joined pairs, the possible pairs and the cells must be those of the tables' headline.

Only TWR-CLOUD's errors are given. There the filter keeps whole scenarios, since every unit keeps all its agents, and so
all its joined pairs, or keeps no pair of well-linked agents. The script checks this and refuses otherwise, and it checks
that the kept units are exactly those with eight or more anchors. The errors over the pairs of well-linked agents are
then the tables' unit errors over the scenarios with eight or more anchors, and no estimate is read. The script builds
final_tables.Units of the five methods from the unit scores the tables read (bern_run, bern_entries) and passes the
joined pairs of the kept units through final_tables.analyse, with the paired cluster bootstrap of the dataset's design,
2000 draws and seed 20260926, as the tables run it. Its units hold every agent group of the release (release_cells), the
clusters of every mode of the tables, so the draws are those of the tables. The record gives every method's S50, S75 and
S90 with their 95 percent intervals, SOLID minus every baseline with its interval and whether the interval excludes zero,
the method and the baseline with the lowest S75, and every method's S75 without the filter, which must be the tables'.
On the other datasets the filter drops pairs within units, so their errors would need every method's estimates, and the
paper prints their shares alone.

Nothing is solved or fitted.

Usage: well_linked.py [--runs ROOT] [--tables DIR] [--out FILE]
       ROOT is the runs root the tables read, out/ of the repository unless given, and DIR the folder of the final
       tables, research/artifacts/solid/tables unless given. The data folder is SOLID_DATA, research/data unless set.
"""
import argparse
import gzip
import json
import sys
from collections import Counter, namedtuple
from pathlib import Path

import numpy as np

from anchor_sides import array, checked, hashed, local, one, relative

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "out"
TABLES = ROOT / "research/artifacts/solid/tables"
OUT = ROOT / "research/artifacts/solid/figures/data/well-linked.json"
sys.path.insert(0, str(ROOT / "research/solid"))
import common  # noqa: E402
import decision  # noqa: E402
import final_tables  # noqa: E402
import scoring  # noqa: E402

DATASETS = (("bern", "TWR-CLOUD"), ("idlab", "IDLab"), ("hymn", "HYMN"), ("aau", "AAU"), ("tiers", "TIERS"),
            ("bregar", "Bregar"))
MODE = "ANC"
MINIMUM_LINKS = 3
HEADLINE = f"headline, {final_tables.HEADLINE_MIN_ANCHORS} or more anchors"
TRACKS = ("1", "10")
LEVELS = ("S50", "S75", "S90")
# the pair sets whose counts a problem must give, those of the TWR-CLOUD unit scores
PAIR_SETS = (scoring.PAIR_SET, *scoring.SUPPLEMENTARY)
# on TWR-CLOUD the release's ANC scenarios have 4, 8, 12, 16 or 20 anchors
BERN_FEW_ANCHORS = 4
# the files of a TWR-CLOUD run its units are read from (final_tables.bern_run and bern_entries), and the summary of its
# joined scores, which names the scorer input
BERN_FILES = ("command.json", "benchmark.json", "official/scenario_scores.csv.gz", "joined/scenario_scores.csv.gz",
              "joined/summary.json")
RULE = (f"an agent of an anchor-only unit with ranged links to {MINIMUM_LINKS} or more distinct anchors of the unit, a ranged "
        "link being an observation between the two with at least one sample of the unit's reading count that "
        "scoring.valid_samples counts (scoring.successful_links, the rule of the paper's ranged pairs, a calibrated sample with "
        "a finite distance of at least 0 and a finite scale above 0 or a raw time of flight within 0 and 200 ns). The filter "
        "depends on the inputs alone and is the same for every method")

# one unit of the tables, its reading count, anchor count, reporting cell, joined pairs and agent pairs
Unit = namedtuple("Unit", "key track anchors cell joined possible")
# one unit's problem, its agents, joined and ranged pairs, agents of its joined pairs, well-linked agents among them and
# joined pairs of two well-linked agents
Links = namedtuple("Links", "agents joined ranged linked well well_pairs")


def links_of(problem):
    """One ANC unit's agents, its joined and ranged pairs (scoring.joined_pairs), its agents with an anchor link, the
    agents of its joined pairs, those of them with ranged links to MINIMUM_LINKS or more distinct anchors
    (scoring.successful_links), and its joined pairs of two such agents."""
    if scoring.mode_of(problem) != MODE:
        raise ValueError(f"a problem of mode {problem['mode']} among those of {MODE}")
    joined = scoring.joined_pairs(problem)
    _, anchor_links = scoring.successful_links(problem)
    linked = {agent for pair in joined for agent in pair}
    well = {agent for agent in linked if len(anchor_links.get(agent, ())) >= MINIMUM_LINKS}
    return Links(len(problem["agent_ids"]), len(joined), len(scoring.joined_pairs(problem, ranged=True)), len(linked), len(well),
                 sum(a in well and b in well for a, b in joined))


def compared(links, pairs, where):
    """Refused unless a unit's problem gives the pair counts of its score row, every agent pair, the joined and the ranged
    pairs, and the anchored-3 pairs where the row keeps them, the pairs of two agents with ranged links to three or more
    anchors (scoring.pair_sets)."""
    found = {"all": links.agents * (links.agents - 1) // 2, "joined": links.joined, "ranged": links.ranged,
             f"anchored-{MINIMUM_LINKS}": links.well_pairs}
    differing = [name for name, count in pairs.items() if name in found and found[name] != count]
    if differing:
        raise ValueError(f"{where}: its problem gives other {', '.join(differing)} pair counts than its score row")


# ------------------------------------------------------------------------------------------- the units and their problems
def score_rows(path):
    """Per unit of SOLID's ANC score rows, (key, id, reading count), its reading count, anchor count and reporting cell,
    as the tables take them (final_tables.row_entries), and the pair counts of its row."""
    rows = {}
    for row in array(path):
        key = (row["key"], row["id"], row["samples"])
        if key in rows:
            raise ValueError(f"{path} holds two rows of unit {key}")
        rows[key] = (row["samples"], row["anchors"], decision.plain(row["cell"] if "cell" in row else row["acquisition"]),
                     {name: entry["eligible_pairs"] for name, entry in row["pair_sets"].items()})
    return rows


def solid_run(dataset, tables, runs, sources):
    """SOLID's ANC run record and score file, inputs of the tables, each checked against the SHA-256 the tables record."""
    folder = f"/solid/{dataset}/evaluation/{MODE.lower()}/"
    run, scores = (checked(one((e for e in tables["inputs"] if e["path"].endswith(folder + name)), f"{dataset} {name}"), runs,
                           sources, f"{dataset}/solid_{name.split('.')[0]}") for name in ("run.json", "scores.json"))
    return json.loads(run.read_text()), scores


def problems(dataset, run, rows, sources):
    """(key, problem) of every ANC unit of TIERS, IDLab, HYMN, AAU or Bregar as its runner built it from the data folder,
    each input refused unless it is the one SOLID's run read (run). TIERS and IDLab build the campaign's request of every
    batch the rows name (batch_inputs), with the campaign whose freeze the run names, and HYMN, AAU and Bregar read the
    request of the evaluation half with the SHA-256 the run names. layout_inputs.levelled, which the runners pass every
    request of TIERS, IDLab and Bregar through, sets starts and areas alone and is left out."""
    if dataset in ("tiers", "idlab"):
        if dataset == "tiers":
            import tiers_campaign as campaign
            folder = common.EXTERNAL / "tiers-static-campaign"
            files = json.loads((folder / "freeze.json").read_text())["files"]
        else:
            import idlab_campaign as campaign
            folder = common.EXTERNAL / run["campaign"]
            files = json.loads((folder / "manifest.json").read_text())["files"]
            hashed(folder / "manifest.json", sources, f"{dataset}/campaign_manifest")
            if files["freeze.json"] != run["campaign_freeze_sha256"]:
                raise ValueError(f"{dataset}: the campaign's manifest names another freeze than SOLID's run")
        if hashed(folder / "freeze.json", sources, f"{dataset}/campaign_freeze") != run["campaign_freeze_sha256"]:
            raise ValueError(f"{dataset}: the campaign's freeze is not the one SOLID's run names")
        for name in ("base.json.gz", "catalog.json.gz"):
            if hashed(folder / name, sources, f"{dataset}/campaign_{name.split('.')[0]}") != files[name]:
                raise ValueError(f"{dataset}: the campaign's {name} is not the one its freeze or manifest lists")
        base = json.loads(gzip.decompress((folder / "base.json.gz").read_bytes()))
        if dataset == "idlab":
            base["values"] = {(v[0], v[1]): v[2] for v in base["values"]}
        catalog = json.loads(gzip.decompress((folder / "catalog.json.gz").read_bytes()))
        # SOLID's runs of TIERS solve its calibrated track (run_tiers.py), and IDLab's campaign holds one track
        batches = {b["key"]: b for b in catalog if b["mode"] == MODE and (dataset == "idlab" or b["track"] == "calibrated")}
        for key in sorted({key for key, _, _ in rows}):
            request, _ = campaign.batch_inputs(base, batches[key])
            for unit in request["units"]:
                yield (key, unit["id"], unit["samples"]), unit["problem"]
        return
    path = common.EXTERNAL / f"{dataset}-layout/evaluation/request.json.gz"
    if hashed(path, sources, f"{dataset}/request") != run["request_sha256"]:
        raise ValueError(f"{dataset}: the request of the evaluation half is not the one SOLID's run names")
    keys = Counter((identity, samples) for _, identity, samples in rows)
    if max(keys.values()) > 1:
        raise ValueError(f"{dataset}: an identity and reading count of two units")
    named = {(identity, samples): key for key, identity, samples in rows}
    for unit in json.loads(gzip.decompress(path.read_bytes()))["units"]:
        key = named.get((unit["id"], unit["samples"]))
        if key is None:
            raise ValueError(f"{dataset}: unit {unit['id']} with {unit['samples']} readings of the request has no score row")
        yield (key, unit["id"], unit["samples"]), unit["problem"]


def campaign_units(dataset, tables, runs, sources):
    """The units of SOLID's ANC score rows of one dataset outside TWR-CLOUD with the links of their problems, every
    problem checked against its row."""
    run, path = solid_run(dataset, tables, runs, sources)
    rows = score_rows(path)
    links = {}
    for key, problem in problems(dataset, run, rows, sources):
        if key not in rows or key in links:
            raise ValueError(f"{dataset}: a problem of unit {key}, which the score rows do not hold or which appears twice")
        links[key] = links_of(problem)
        compared(links[key], rows[key][3], f"{dataset} unit {key}")
    if len(links) != len(rows):
        raise ValueError(f"{dataset}: {len(rows) - len(links)} units of the score rows have no problem")
    units = [Unit(key, track, anchors, cell, pairs["joined"], pairs["all"]) for key, (track, anchors, cell, pairs) in rows.items()]
    return units, links


def bern_units(tables, runs, declared, scorer, sources):
    """final_tables.Units of TWR-CLOUD's ANC units with every method's unit scores under the joined and ranged pairs, read
    by final_tables' own readers from the runs the tables read, every file checked against the SHA-256 the tables record,
    and the links of every unit's released problem, from the scorer input that the joined scores of every run name, each
    checked against its unit."""
    spec = declared["datasets"]["bern"]
    design = decision.Design(**{k: tuple(v) if isinstance(v, list) else v for k, v in spec["design"].items() if k not in ("name", "reason")})
    methods = tables["modes"][MODE]["methods"]
    if methods != spec["modes"][MODE]:
        raise ValueError(f"the TWR-CLOUD tables hold the methods {methods}, not the declared {spec['modes'][MODE]}")
    reader, units, scenarios = final_tables.Reader(ROOT), None, set()
    for method in methods:
        folder = tables["runs"][MODE][method]["record"].rsplit("/", 1)[0]
        for name in BERN_FILES:
            checked(one((e for e in tables["inputs"] if e["path"] == f"{folder}/{name}"), f"{folder}/{name}"), runs, sources,
                    f"bern/{method}/{name}")
        folder = local(folder, runs)
        scenarios.add(json.loads((folder / "joined/summary.json").read_text())["scenarios_sha256"])
        run, _ = final_tables.bern_run(reader, folder, method, MODE, method == "solid")
        entries = final_tables.bern_entries(reader, folder, MODE, design, PAIR_SETS, run.aborted)
        if units is None:
            units = final_tables.Units("bern", MODE, PAIR_SETS, method, entries, sort=False)
        else:
            units.add(method, entries)
    counts = {int(track): int((units.track == track).sum()) for track in sorted(set(units.track.tolist()))}
    if any(count != spec["units_per_track"][MODE] for count in counts.values()):
        raise ValueError(f"the TWR-CLOUD {MODE} units per track {counts} are not the corpus's {spec['units_per_track'][MODE]}")
    # the clusters of the bootstrap, every agent group of the release, which every anchor count and the anchor-free mode repeat
    if set(units.top) != {(agents, group) for _, count, agents, _ in final_tables.release_cells() for group in range(count)}:
        raise ValueError("the TWR-CLOUD ANC units do not hold every agent group of the release, so their draws would not be the tables'")
    archive = common.DATA / json.loads((ROOT / "research/benchmarks/bern/baselines.json").read_text())["archives"]["scorer_scenarios"]
    if len(scenarios) != 1 or hashed(archive, sources, "bern/scorer_input") not in scenarios:
        raise ValueError(f"{relative(archive)} is not the scorer input that the joined scores of every run name")
    links = {}
    for scenario, data in scorer.iter_json_scenarios(archive):
        if not data.get("Anchors"):
            continue
        iterations = data.get("Iterations", [])
        for number, iteration in enumerate(iterations.values() if isinstance(iterations, dict) else iterations, start=1):
            for samples in (1, 10):
                key = (scenario, int(iteration.get("ID", number)), samples)
                i = units.index.get(key)
                if i is None or key in links:
                    raise ValueError(f"TWR-CLOUD: a problem of unit {key}, which the unit scores do not hold or which appears twice")
                links[key] = links_of(scoring.released_problem(data, iteration, samples, MODE))
                compared(links[key], {"all": int(units.possible[i]), **{name: int(units.pairs[p][i]) for p, name in enumerate(PAIR_SETS)}},
                         f"TWR-CLOUD unit {key}")
    if len(links) != len(units.keys):
        raise ValueError(f"TWR-CLOUD: {len(units.keys) - len(links)} units of the unit scores have no problem")
    return units, methods, design, links


# ------------------------------------------------------------------------------------------- the counts and the tables
def tallied(units, links):
    """Per reading count over the headline of the tables, the units with three or more anchors, the agents with an anchor
    link of the units with a joined pair and those of them that are not well linked, overall and per anchor count, and the
    units, the units and cells that keep a pair of well-linked agents, the cells, joined pairs and possible pairs of the
    tables and the pairs of well-linked agents."""
    result = {}
    for track in TRACKS:
        members = [u for u in units if str(u.track) == track and u.anchors >= final_tables.HEADLINE_MIN_ANCHORS]
        agents, well = Counter(), Counter()
        for u in members:
            if u.joined:
                agents[u.anchors] += links[u.key].linked
                well[u.anchors] += links[u.key].well
        by_count = {str(count): dict(agents=agents[count], without_three=agents[count] - well[count],
                                     share_without_three=(agents[count] - well[count]) / agents[count]) for count in sorted(agents)}
        n, w = sum(agents.values()), sum(well.values())
        kept = [u for u in members if links[u.key].well_pairs]
        joined, pairs = sum(u.joined for u in members), sum(links[u.key].well_pairs for u in members)
        result[track] = dict(agents=n, without_three=n - w, share_without_three=(n - w) / n if n else None, by_anchor_count=by_count,
                             units=len(members), scored_units=len(kept), cells=len({u.cell for u in kept}),
                             cells_of_the_tables=len({u.cell for u in members if u.joined}), joined_pairs_of_the_tables=joined,
                             pairs_of_well_linked_agents=pairs, pair_share=pairs / joined if joined else None,
                             unfiltered=dict(units=len(members), scored_units=sum(1 for u in members if u.joined), pairs=joined,
                                             possible_pairs=sum(u.possible for u in members),
                                             cells=len({u.cell for u in members if u.joined})))
    return result


def headline(tables, track):
    return tables["modes"][MODE]["groups"][HEADLINE]["tracks"][track]["pair_sets"][scoring.PAIR_SET]


def eight_or_more(units, links):
    """On TWR-CLOUD, refused unless every headline unit with a joined pair keeps all its joined pairs or no pair of
    well-linked agents, and unless the units the filter keeps are those with eight or more anchors, each with every
    joined pair."""
    entries = [(u.anchors, u.joined, links[u.key].well_pairs) for u in units if u.anchors >= final_tables.HEADLINE_MIN_ANCHORS and u.joined]
    split = [entry for entry in entries if 0 < entry[2] < entry[1]]
    if split:
        raise SystemExit(f"TWR-CLOUD: {len(split)} units keep some of their joined pairs and drop others, so the filter does not keep "
                         "whole scenarios and the tables' unit errors cannot give its errors")
    kept = [(count, pairs == well) for count, pairs, well in entries if well > 0]
    many = [(pairs, well) for count, pairs, well in entries if count > BERN_FEW_ANCHORS]
    few = [well for count, pairs, well in entries if count == BERN_FEW_ANCHORS]
    if not (all(count > BERN_FEW_ANCHORS and whole for count, whole in kept) and len(kept) == len(many)
            and all(well == 0 for well in few) and all(well == pairs for pairs, well in many)):
        raise SystemExit("TWR-CLOUD: the kept units are not exactly those with eight or more anchors")
    return {"units_with_eight_or_more_anchors": len(many), "units_kept": len(kept), "units_with_four_anchors": len(few),
            "every_joined_pair_kept": True}


def kept_units(units, methods, kept):
    """final_tables.Units of the joined pairs of the kept units alone, in the order of the tables' units, every other unit
    without a pair and without an error, as the tables hold a unit without a pair of a set."""
    result = None
    for method in methods:
        errors = units.errors[method][0]
        entries = (final_tables.Entry(key=key, track=int(units.track[i]), cell=units.cell[i], anchors=int(units.anchors[i]), top=units.top[i],
                                      stratum=units.top_stratum[i], pairs=(int(units.pairs[0][i]) if kept[i] else 0,),
                                      possible=int(units.possible[i]), errors=(float(errors[i]) if kept[i] else None,),
                                      failed=bool(units.failed[method][i]), aborted=bool(units.aborted[method][i]), elapsed=None)
                   for i, key in enumerate(units.keys))
        if result is None:
            result = final_tables.Units(units.dataset, units.mode, (scoring.PAIR_SET,), method, entries, sort=False)
        else:
            result.add(method, entries)
    return result


def bern_tables(units, methods, design, links, declared, percentile, tables):
    """TWR-CLOUD's tables over the kept units (final_tables.analyse) and every method's percentiles without the filter
    (final_tables.point), the latter refused unless they are the tables'."""
    spec, levels = declared["datasets"]["bern"], decision.levels_of(declared["percentiles"])
    if [name for name, _ in levels] != list(LEVELS):
        raise ValueError(f"the tables declare the percentiles {declared['percentiles']}, not {LEVELS}")
    mask = dict(final_tables.report_groups(MODE, units.anchors, declared["groups"]["minimum_anchors"]))[HEADLINE]
    unfiltered, values = {}, 0
    for track in declared["tracks"]:
        scored = np.flatnonzero(mask & (units.track == track) & (units.pairs[0] > 0))
        cells = [units.cell[i] for i in scored]
        unfiltered[str(track)] = {m: dict(zip(LEVELS, final_tables.point(cells, units.errors[m][0][scored], levels, percentile))) for m in methods}
        for m in methods:
            for level in LEVELS:
                value, exact = unfiltered[str(track)][m][level]
                theirs = headline(tables, str(track))["methods"][m][level]
                if (value, exact) != (theirs["value_cm"], theirs["exact_cm"]):
                    raise ValueError(f"TWR-CLOUD {track} {m} {level}: {value}, {exact} without the filter against the tables' "
                                     f"{theirs['value_cm']}, {theirs['exact_cm']}")
                values += 2
    kept = np.array([links[key].well_pairs > 0 for key in units.keys])
    sections = final_tables.analyse({MODE: kept_units(units, methods, kept)}, {MODE: list(methods)}, {}, (scoring.PAIR_SET,), spec["design"]["name"],
                                    design, spec["replicates"], spec["seed"], declared["percentiles"], declared["interval"],
                                    declared["groups"]["minimum_anchors"], declared["tracks"], percentile)
    return sections[MODE]["groups"][HEADLINE]["tracks"], unfiltered, values


def errors_entry(table, unfiltered):
    """Every method's S50, S75 and S90 of one filtered table with their intervals, SOLID minus every baseline with its
    interval and significance, the method and the baseline with the lowest S75, and every method's S75 without the
    filter."""
    methods = {m: {level: {"value_cm": v[level]["value_cm"], "interval_cm": v[level]["interval_cm"]} for level in LEVELS}
               for m, v in table["methods"].items()}
    differences = {}
    for name, v in table["differences"].items():
        differences[name] = {}
        for level in LEVELS:
            interval = v[level]["interval_cm"]
            differences[name][level] = {"value_cm": v[level]["value_cm"], "interval_cm": interval,
                                        "significant": None if interval is None else (interval[0] > 0 or interval[1] < 0)}
    baselines = [m for m in methods if m != "solid"]
    full = {m: unfiltered[m]["S75"][0] for m in methods}
    return dict(methods=methods, differences=differences, lowest_S75=min(methods, key=lambda m: methods[m]["S75"]["value_cm"]),
                best_baseline_S75=min(baselines, key=lambda m: methods[m]["S75"]["value_cm"]), tables_S75_cm=full,
                tables_best_baseline_S75=min(baselines, key=full.get))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS, help="the runs root the tables read, in place of out/ of the paths they record")
    parser.add_argument("--tables", type=Path, default=TABLES, help="the folder of the final tables")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    declared = json.loads(final_tables.encoded(final_tables.declaration()))
    scorer, sources = common.load_scorer(), {}
    percentile = scorer.balanced_percentile
    hashed(Path(scorer.__file__), sources, "released_scorer")
    result = dict(schema="solid.figures.well-linked.v1", minimum_links=MINIMUM_LINKS, headline=HEADLINE, sources=sources, rule=RULE,
                  checks={}, datasets={})
    for dataset, label in DATASETS:
        tables = json.loads((args.tables / dataset / "evaluation/tables.json").read_text())
        before = set(sources)
        if dataset == "bern":
            bern, methods, design, links = bern_units(tables, args.runs, declared, scorer, sources)
            units = [Unit(key, int(bern.track[i]), int(bern.anchors[i]), bern.cell[i], int(bern.pairs[0][i]), int(bern.possible[i]))
                     for i, key in enumerate(bern.keys)]
        else:
            units, links = campaign_units(dataset, tables, args.runs, sources)
        counts = tallied(units, links)
        # the files of this dataset, those the tables record and those of the data folder that SOLID's run names
        recorded, read = {e["path"] for e in tables["inputs"]}, [sources[name]["path"] for name in sources if name not in before]
        checks = {"tables_inputs": {"files": sum(path in recorded for path in read), "equal_sha256": True},
                  "data_inputs": {"files": sum(path not in recorded for path in read), "equal_sha256": True},
                  "problems": {"units": len(links), "every_pair_count_equal": True}}
        values_compared = 0
        for track, entry in counts.items():
            theirs = headline(tables, track)
            for name, value in entry.pop("unfiltered").items():
                if value != theirs[name]:
                    raise ValueError(f"{dataset} {track}: {name} {value} without the filter against the tables' {theirs[name]}")
                values_compared += 1
        if dataset == "bern":
            checks["eight_or_more_anchors"] = eight_or_more(units, links)
            filtered, unfiltered, points = bern_tables(bern, methods, design, links, declared, percentile, tables)
            values_compared += points
            for track in TRACKS:
                table = filtered[track]["pair_sets"][scoring.PAIR_SET]
                mine = counts[track]
                if (table["units"], table["scored_units"], table["cells"], table["pairs"]) != (mine["units"], mine["scored_units"], mine["cells"],
                                                                                               mine["pairs_of_well_linked_agents"]):
                    raise ValueError(f"TWR-CLOUD {track}: the filtered table holds other units, cells or pairs than the counts")
                counts[track].update(errors_entry(table, unfiltered[track]))
            del bern
        checks["tables_without_filter"] = {"values_compared": values_compared, "equal": True}
        result["checks"][dataset] = checks
        result["datasets"][dataset] = {"label": label, "tracks": counts}
        print(dataset, {t: (e["agents"], e["without_three"], e["scored_units"], e["units"], e["pairs_of_well_linked_agents"],
                            e["joined_pairs_of_the_tables"], e.get("lowest_S75"), e.get("best_baseline_S75"))
                        for t, e in counts.items()}, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("wrote", relative(args.out))


if __name__ == "__main__":
    main()
