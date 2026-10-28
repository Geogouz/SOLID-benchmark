"""Export the balanced CDF of every table of the final tables, for the paper's figures.

The script runs the final tables' own reading path (research/solid/final_tables.py build) on one dataset's evaluation
half, with every guard of the build, and replaces the analysis and the writing of the tables. For every mode, group,
sample track, pair set and method of the tables it takes the units the table scores, their cells and their errors, and
writes the balanced CDF as its quantile function at LEVELS: the smallest error at which the mean over the cells of the
share of the cell's units at or below it reaches the level, the definition of the released scorer's balanced
percentile. The S50, S75 and S90 of every curve are then taken with the scorer's own function, as the tables take them,
and must equal the committed percentiles.csv, so no figure can disagree with a table. Nothing is solved, and an audit
hook refuses every write outside OUT.

In a reader's reproduction (SOLID_READER=1) the build reads the reader's own runs below the runs root, out/ of REPO
unless --runs names another, as final_tables.py build --runs reads them, and the curves must equal the reader's own
percentiles.csv, the one that build wrote to the tables folder, research/artifacts/solid/tables of REPO unless --tables
names another.

Usage: export_cdfs.py [--root REPO] --dataset bern|tiers|bregar|idlab|hymn|aau --out OUT [--runs DIR] [--tables DIR]
       REPO is the checkout of the final tables, the one that holds this script unless given, writes
       OUT/cdfs-<dataset>.json
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

LEVELS = np.round(np.linspace(0.0, 1.0, 1001), 3)
CHECKED = (("S50", 0.5), ("S75", 0.75), ("S90", 0.9))


def guard(out):
    """Refuse every write outside `out` from here on."""
    out = os.path.realpath(out)
    writing = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC

    def inside(path):
        try:
            return os.path.realpath(os.fsdecode(path)).startswith(out + os.sep)
        except TypeError:
            return True

    def hook(event, args):
        if event == "open":
            path, mode, flags = args
            wants = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & writing)
            if wants and path is not None and not inside(path):
                raise PermissionError(f"the export writes nothing outside {out}, not {path}")
        elif event in ("os.rename", "os.replace", "os.remove", "os.rmdir", "os.mkdir", "shutil.rmtree", "os.unlink", "os.truncate"):
            for path in args[:2]:
                if isinstance(path, (str, bytes, os.PathLike)) and not inside(path):
                    raise PermissionError(f"the export changes nothing outside {out}, not {path} ({event})")
    sys.addaudithook(hook)


def quantiles(grouped, levels):
    """The balanced quantile function of cells of values at the levels: every nonempty cell weighs the same, and each of
    its values an equal share of it."""
    # The released scorer's arithmetic (balanced_percentile): per value, the shares of the cells at or below it summed in
    # cell order and divided by the number of cells. A cumulative sum of per-unit weights, used before 16 October 2026,
    # could fall short of a level that the balanced CDF reaches exactly.
    cells = [np.sort(np.asarray(values, dtype=float)) for values in grouped.values() if len(values)]
    distinct = np.unique(np.concatenate(cells))
    cdf = np.zeros(len(distinct))
    for c in cells:
        cdf += np.searchsorted(c, distinct, side="right") / len(c)
    cdf /= len(cells)
    index = np.minimum(np.searchsorted(cdf, levels, side="left"), len(distinct) - 1)
    return distinct[index]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=Path, help="the runs root in place of out/ of REPO, in a reader's reproduction")
    parser.add_argument("--tables", type=Path, help="the folder of the reader's tables, in a reader's reproduction")
    args = parser.parse_args()
    reader = os.environ.get("SOLID_READER") == "1"
    if (args.runs or args.tables) and not reader:
        parser.error("--runs and --tables serve a reader's reproduction, SOLID_READER=1")
    root, out = args.root.resolve(), args.out.resolve()
    tables_root = (args.tables or root / "research/artifacts/solid/tables").resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    guard(out)
    sys.path.insert(0, str(root / "research/solid"))
    import final_tables as tables

    curves = []

    def analyse(modes, rosters, extras, pair_sets, design_name, design, replicates, seed, levels, interval, minimum, tracks, percentile, runs=None):
        for mode, units in modes.items():
            methods = [*rosters[mode], *extras.get(mode, ())]
            for label, mask in tables.report_groups(mode, units.anchors, minimum):
                for track in tracks:
                    members = mask & (units.track == track)
                    for p, name in enumerate(pair_sets):
                        scored = np.flatnonzero(members & (units.pairs[p] > 0))
                        if not len(scored):
                            continue
                        cells = [units.cell[i] for i in scored]
                        for method in methods:
                            grouped = {}
                            for cell, value in zip(cells, units.errors[method][p][scored]):
                                grouped.setdefault(cell, []).append(float(value))
                            curves.append(dict(mode=mode, group=label, track=int(track), pair_set=name, method=method, units=int(len(scored)),
                                               cells=len(grouped), quantiles_cm=[float(v) for v in quantiles(grouped, LEVELS)],
                                               **{level: float(percentile(grouped, q)) for level, q in CHECKED}))
        return {}

    tables.analyse = analyse
    tables.write_outputs = lambda *args, **kwargs: None
    if reader:
        tables.build(args.dataset, "evaluation", root=root, runs_root=args.runs)
    else:
        tables.build(args.dataset, "evaluation", root=root)

    committed = {}
    with open(tables_root / f"{args.dataset}/evaluation/percentiles.csv", newline="") as stream:
        for row in csv.DictReader(stream):
            committed[(row["mode"], row["group"], int(row["track"]), row["pair_set"], row["method"])] = row
    mismatched = []
    for curve in curves:
        row = committed.get((curve["mode"], curve["group"], curve["track"], curve["pair_set"], curve["method"]))
        curve["matches_committed"] = row is not None and all(float(row[f"{level}_cm"]) == curve[level] for level, _ in CHECKED)
        if not curve["matches_committed"]:
            mismatched.append({key: curve[key] for key in ("mode", "group", "track", "pair_set", "method")})
    # a reader's checkout need not be a git repository, so a reader's curves name no commit
    commit = None if reader else os.popen(f"git -C {root} rev-parse HEAD").read().strip()
    record = dict(schema="solid.figures.cdfs.v1", dataset=args.dataset, split="evaluation", commit=commit,
                  definition=("Balanced quantile function: at every level, the smallest error at which the mean over the cells of the share "
                              "of the cell's units at or below it reaches the level. S50, S75 and S90 by the released scorer's function."),
                  levels=LEVELS.tolist(), curves=curves, tables=len(committed), curves_matching_committed=sum(c["matches_committed"] for c in curves),
                  mismatched=mismatched, every_table_has_a_curve=set(committed) <= {(c["mode"], c["group"], c["track"], c["pair_set"], c["method"])
                                                                                    for c in curves})
    (out / f"cdfs-{args.dataset}.json").write_text(json.dumps(record) + "\n")
    print(f"{args.dataset}: {len(curves)} curves, {record['curves_matching_committed']} match the committed tables, "
          f"every table has a curve: {record['every_table_has_a_curve']}, mismatched {mismatched[:3]}")
    return 0 if not mismatched and record["every_table_has_a_curve"] else 1


if __name__ == "__main__":
    sys.exit(main())
