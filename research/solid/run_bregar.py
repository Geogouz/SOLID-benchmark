"""SOLID anchor only on the same-layout Bregar inputs, one half of the split, four environments with six channels each.

Reads the request and reference of the half under external-benchmarks/bregar-layout of the data folder, built by
research/benchmarks/datasets/bregar_layout.py. Every unit takes the profile of its environment (link model fitted on the
environment's own training block), composed in memory by profiles.py from the record of the method the confirmatory
round confirmed, or before the round from the fit records and the deployment record, or taken from the profile passed
with --profile, and the operating area of the deployment rules of deployment.py. The request passes through
layout_inputs.levelled, so SOLID's units start where every baseline's start. A tracked profile file is solved only
when it holds that composition. Solves with solid-units and scores with the campaign scorer balanced over the
environment and channel cells. Output: out/solid/bregar/<split>/anc/scores.json with the solved profile beside it, or
--out. Every row keeps the mean pair error of each pair set of its unit's inputs and the unit's anchor count, and the
summary reports the joined pairs and the supplementary ranged pairs. --acquisitions solves the units of the named
acquisitions alone, those of one fold of the confirmatory round for example.

Usage: run_bregar.py --split training|evaluation [--workers 8] [--profile PATH] [--out DIR] [--acquisitions NAME ...]
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import layout_inputs  # noqa: E402
import layout_split  # noqa: E402
import profiles  # noqa: E402

E = common.EXTERNAL / "bregar-layout"
PROFILE = "bregar-anchor-only.json"


def prepared(unit, profile, environment):
    """The profile part of one unit of the request: the operating area and the anchor only block of its environment."""
    return dict(deployment_profile=dict(area=deployment.unit_area(unit["problem"], "bregar"), anchor_only=profile["environments"][environment]["anchor_only"]))


def run(split, workers, profile_path=None, out_dir=None, acquisitions=None):
    """Position the units of one half, of the named acquisitions alone when given, and return the scored rows. A
    selection of no acquisition is refused, never read as every acquisition."""
    if acquisitions is not None and not set(acquisitions):
        raise ValueError("the selection names no Bregar acquisition, so the run would position nothing")
    # every method's starts by SOLID's deployment rules, the request the baselines receive too
    request = layout_inputs.levelled(json.load(gzip.open(E / split / "request.json.gz")), "bregar", "ANC")
    reference = common.read_json(E / split / "reference.json")
    meta = {u["id"]: u for u in reference["units"]}
    profile, source = profiles.resolve(PROFILE, profile_path)
    ids = layout_split.bregar_units(split)
    if {u["id"] for u in request["units"]} != ids:
        raise ValueError("the Bregar request does not hold the identities of the half")
    if acquisitions is not None and not set(acquisitions) <= {meta[i]["acquisition"] for i in ids}:
        raise ValueError(f"acquisitions {sorted(set(acquisitions) - {meta[i]['acquisition'] for i in ids})} are not in the {split} half")
    request = dict(request, units=[u for u in request["units"] if acquisitions is None or meta[u["id"]]["acquisition"] in acquisitions])
    units, lookup, anchors = [], {}, {}
    for unit in request["units"]:
        environment = meta[unit["id"]]["reference"]
        anchors[unit["id"]] = len(unit["problem"]["anchors"])
        lookup[(unit["id"], unit["samples"])] = len(units)
        units.append(dict(id=len(units), samples=unit["samples"], problem=unit["problem"], label=meta[unit["id"]]["acquisition"],
                          **prepared(unit, profile, environment)))
    out_dir = Path(out_dir) if out_dir else common.OUT / "bregar" / split / "anc"
    solved_profile = profiles.provenance(out_dir, profile, source)
    records = common.solve(units, out_dir / "predictions.jsonl", workers)
    solved = {key: records[index] for key, index in lookup.items()}
    rows = []
    for s in common.score(request, reference, solved, common.load_scorer()):
        environment = meta[s["id"]]["reference"]
        rows.append(dict(key=environment, id=s["id"], samples=s["samples"], cell=s["acquisition"], level=environment,
                         acquisition=s["acquisition"], environment=environment, block=meta[s["id"]]["block"], channel=meta[s["id"]]["channel"],
                         anchors=anchors[s["id"]], error_cm=s["mean_pair_error_cm"], elapsed_ms=s["elapsed_ms"], failure=s["failure"],
                         pair_sets=s["pair_sets"]))
    common.write_run(out_dir, rows, workers=workers, dataset="bregar", mode="ANC", split=split, **solved_profile,
                     split_record_sha256=common.sha256(layout_split.RECORD), request_sha256=common.sha256(E / split / "request.json.gz"),
                     acquisitions=None if acquisitions is None else sorted(acquisitions), complete=acquisitions is None,
                     shared_inputs=layout_inputs.record("bregar"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["training", "evaluation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile", default=None, help="profile file, the composition of the fit records by default")
    parser.add_argument("--out", default=None)
    parser.add_argument("--acquisitions", nargs="+", default=None, help="the acquisitions to solve, every acquisition of the half by default")
    args = parser.parse_args()
    rows = run(args.split, args.workers, args.profile, args.out, args.acquisitions)
    common.summarize(rows, f"Bregar ANC {args.split}", "ANC")


if __name__ == "__main__":
    main()
