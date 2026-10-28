"""Where ADE-BP leads SOLID in the anchor-only mode of TIERS, from the scored runs of the evaluation half.

TIERS records one static layout of eight radios in each of three recordings, army, corner and diagonal, and in the
diagonal recording all eight stand on one line. Anchors on a line leave every agent a mirror image across it. This
script compares SOLID with ADE-BP over the headline scenarios of ANC (three or more anchors) per recording and anchor
count, and in the diagonal recording it measures how far each method places the agents from the line of the radios.

A scenario is matched across the two methods by its unique id (the key of a scenario repeats across its samples).
SOLID's prediction files number their scenarios within each file, so a SOLID prediction counts only when the pair
error it gives equals a scored error of its label and reading count. An ADE-BP prediction counts only when it equals
its own scored error, which also picks the scored run among the two runs that the work folder keeps. The pair error is
the mean over the agent pairs of the error of their horizontal distance, as the scorer computes it. Within a cell a
percentile is the smallest error at which the share of the cell's scenarios at or below it reaches the level, and the
balanced percentile weights every cell equally, as the paper's balanced CDF does. The line is the least-squares line
through the eight radios of the diagonal recording. Both score files must be the ones the final tables of TIERS read,
with the hashes their tables.json records. Nothing is solved.

Usage: tiers_line.py [--runs OUT] [--tables DIR] [--coordinates BASE] [--out FILE]
       OUT holds solid/tiers and baselines/tiers of the evaluation runs, out/ of the repository unless given, DIR the
       folder of the final tables, research/artifacts/solid/tables unless given, and BASE the campaign base in the data
       folder, SOLID_DATA or research/data
"""
import argparse
import gzip
import hashlib
import itertools
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "out"
TABLES = ROOT / "research/artifacts/solid/tables"
BASE = Path(os.environ.get("SOLID_DATA") or ROOT / "research/data") / "external-benchmarks/tiers-static-campaign/base.json.gz"
LEVELS = (("S50", 0.5), ("S75", 0.75), ("S90", 0.9))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(errors, level):
    errors = sorted(errors)
    return errors[max(0, math.ceil(level * len(errors)) - 1)]


def balanced(cells, level):
    """The smallest error at which the mean over the cells of the share at or below it reaches the level."""
    points = sorted((e, 1 / (len(cells) * len(v))) for v in cells.values() for e in v)
    total = 0.0
    for error, weight in points:
        total += weight
        if total >= level - 1e-12:
            return error
    return points[-1][0]


def pair_error(truth, ids, positions):
    pairs = itertools.combinations(list(zip(ids, positions)), 2)
    return statistics.mean(abs(math.dist(a[:2], b[:2]) - math.dist(truth[i], truth[j])) for (i, a), (j, b) in pairs)


def line_distance(points):
    mx = statistics.mean(p[0] for p in points)
    my = statistics.mean(p[1] for p in points)
    sxx = sum((p[0] - mx) ** 2 for p in points)
    syy = sum((p[1] - my) ** 2 for p in points)
    sxy = sum((p[0] - mx) * (p[1] - my) for p in points)
    angle = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(angle), math.sin(angle)
    return lambda p: abs((p[1] - my) * ux - (p[0] - mx) * uy)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=RUNS)
    parser.add_argument("--tables", type=Path, default=TABLES)
    parser.add_argument("--coordinates", type=Path, default=BASE)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    solid_scores = args.runs / "solid/tiers/evaluation/anc/scores.json"
    baseline_scores = args.runs / "baselines/tiers/evaluation/anc/scores.json"
    read = json.load(open(args.tables / "tiers/evaluation/tables.json"))["inputs"]
    for path, ending in ((solid_scores, "/solid/tiers/evaluation/anc/scores.json"), (baseline_scores, "/baselines/tiers/evaluation/anc/scores.json")):
        recorded = [e["sha256"] for e in read if e["path"].endswith(ending)]
        assert len(recorded) == 1 and sha256(path) == recorded[0], f"{path} is not the file the tables read"
    truth = {acq: {int(k): v[:2] for k, v in radios.items()}
             for acq, radios in json.load(gzip.open(args.coordinates))["coordinates_cm"].items()}
    off_line = line_distance(list(truth["diagonal"].values()))

    solid = {(r["id"], r["samples"]): r for r in json.load(open(solid_scores))}
    ade = {(r["id"], r["samples"]): r for r in json.load(open(baseline_scores)) if r["method"] == "ade-bp-2015"}
    cells = defaultdict(lambda: {"solid": [], "ade-bp": [], "solid_worse": 0})
    for key, s in solid.items():
        if s["anchors"] >= 3:
            a = ade[key]
            assert (a["key"], a["cell"], a["anchors"]) == (s["key"], s["cell"], s["anchors"])
            c = cells[(s["samples"], s["acquisition"], s["anchors"])]
            c["solid"].append(s["error_cm"])
            c["ade-bp"].append(a["error_cm"])
            c["solid_worse"] += s["error_cm"] > a["error_cm"]

    per_cell = []
    for (samples, acquisition, anchors), c in sorted(cells.items()):
        row = dict(readings=samples, recording=acquisition, anchors=anchors, scenarios=len(c["solid"]),
                   solid_worse_share=round(c["solid_worse"] / len(c["solid"]), 3))
        for name, level in LEVELS:
            row[f"solid_{name}_cm"] = round(percentile(c["solid"], level), 2)
            row[f"ade_bp_{name}_cm"] = round(percentile(c["ade-bp"], level), 2)
        per_cell.append(row)

    by_recordings = {}
    for samples in (1, 10):
        for label, keep in (("all three", None), ("army and corner", ("army", "corner")), ("diagonal", ("diagonal",))):
            chosen = {k: v for k, v in cells.items() if k[0] == samples and (keep is None or k[1] in keep)}
            by_recordings[f"{label}, {samples} reading{'s' if samples > 1 else ''}"] = {
                f"{m}_S75_cm": round(balanced({k: v[m] for k, v in chosen.items()}, 0.75), 2) for m in ("solid", "ade-bp")}

    scored_by_label = defaultdict(Counter)
    acquisition_of = {}
    for s in solid.values():
        scored_by_label[(s["key"], s["samples"])][round(s["error_cm"], 6)] += 1
        acquisition_of[s["key"]] = s["acquisition"]
    distances = defaultdict(lambda: defaultdict(list))
    checks = defaultdict(lambda: dict(read=0, matched=0))
    for path in sorted((args.runs / "solid/tiers/evaluation/anc").glob("predictions-b*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            anchors = int(r["label"].rsplit("-", 1)[1])
            if acquisition_of.get(r["label"]) != "diagonal" or anchors < 3 or not r.get("estimate"):
                continue
            ids, positions = r["estimate"]["agent_ids"], r["estimate"]["positions"]
            checks["solid"]["read"] += 1
            if scored_by_label[(r["label"], r["samples"])][round(pair_error(truth["diagonal"], ids, positions), 6)]:
                checks["solid"]["matched"] += 1
                distances["solid"][(anchors, r["samples"])].append(statistics.median(map(off_line, positions)))
    for path in sorted((args.runs / "baselines/tiers/work/evaluation-anc").glob("*.jsonl")):
        for line in open(path):
            r = json.loads(line)
            a = ade.get((r["id"], r["samples"])) if r.get("method") == "ade-bp-2015" else None
            if not a or a["acquisition"] != "diagonal" or a["anchors"] < 3 or not r.get("estimate"):
                continue
            ids, positions = r["estimate"]["agent_ids"], r["estimate"]["positions"]
            checks["ade-bp"]["read"] += 1
            if abs(pair_error(truth["diagonal"], ids, positions) - a["error_cm"]) <= 1e-6 * max(1.0, a["error_cm"]):
                checks["ade-bp"]["matched"] += 1
                distances["ade-bp"][(a["anchors"], r["samples"])].append(statistics.median(map(off_line, positions)))

    off_line_rows = []
    for anchors, samples in sorted(set(distances["solid"]) | set(distances["ade-bp"])):
        row = dict(anchors=anchors, readings=samples)
        for m in ("solid", "ade-bp"):
            v = distances[m][(anchors, samples)]
            row[f"{m.replace('-', '_')}_median_cm"] = round(statistics.median(v), 2)
            row[f"{m.replace('-', '_')}_scenarios"] = len(v)
        off_line_rows.append(row)

    result = dict(
        sources=dict(solid_scores=sha256(solid_scores), baseline_scores=sha256(baseline_scores),
                     coordinates=sha256(args.coordinates)),
        radios_off_line_cm=[round(off_line(p), 1) for p in truth["diagonal"].values()],
        balanced_S75_by_recordings=by_recordings,
        per_cell=per_cell,
        off_line_of_estimates=off_line_rows,
        checks={m: dict(c) for m, c in checks.items()},
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_cell"}, indent=1))
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
