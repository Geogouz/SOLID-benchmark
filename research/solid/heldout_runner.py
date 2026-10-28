"""SOLID anchor only on one half of a held-out benchmark, HYMN or AAU Meshed UWB, the body of run_hymn.py and run_aau.py.

Reads the request and the reference of the half that research/benchmarks/datasets/<dataset>_layout.py built under
external-benchmarks/<dataset>-layout of the data folder, refused when a file changed after its build or the half was
built under another declaration. Either half runs only after the method freeze, with the records of held-out stage one
committed as the ledger closed them (heldout.require_stage_one) and while the committed ledger opens the dataset
(heldout.require_run), so the ledger records every run. Every unit takes the dataset's tracked profile, the record of
stage one, which must equal its composition from the method record and the held-out fit and deployment records by
profiles.py, and the operating area of the deployment rules, which must equal the area the baselines received. Solves
with solid-units and scores with the campaign scorer. The scores of every cell are kept, every row with the mean pair
error of each pair set of its unit's inputs. The summary prints the headline, balanced over the cells with three or
more anchors, and every separate cell with one or two anchors, each under the pairs the measured links join and the
supplementary ranged pairs with the share of pairs and units each scores, and names the flights the build excluded. A
run of either half goes to out/solid/<dataset>/<split>/anc with the solved profile beside the predictions, and no run
takes another profile or folder.

The declared outcomes of the training half reach every run record (the declaration's rules.outcomes). A link fit that
did not converge is solved as it stands, and a link model with a value that is not finite solves no unit, so every
unit is recorded as SOLID's failure and scored with the failure penalty. The run record names every such outcome,
with a failed nested check of the refit and the true agent positions of the training half outside their operating
area, under marks, which the tables print beside SOLID's rows.

A reader's run (SOLID_READER=1) has no method freeze, ledger or method record, so heldout.require_run and
heldout.require_stage_one let it pass, and it solves the tracked profile as written (profiles.heldout_resolve), with the
marks of the tracked fit and deployment records.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import heldout  # noqa: E402
import profiles  # noqa: E402

# the failure of every unit of a run whose link model holds a value that is not finite
UNSOLVED = "not solved, the link model fitted on the training half holds a value that is not finite"


def inputs(dataset, split):
    """The request, the reference and the identity of one built half, either half only after the freeze."""
    return heldout.built_half(dataset, split)


def marks(dataset):
    """The declared outcomes of a dataset's training half that its tables mark beside SOLID's rows. These are a link fit
    that did not converge and is used as it stands, a link fit with a value that is not finite, whose solves all fail, a
    failed nested check of the refit, and the true agent positions of the training half outside their unit's operating
    area, which SOLID alone receives. None of them stops a run."""
    record = profiles.heldout_fit(dataset)
    names = dict(full="the full link fit", four="the refit without the removed components") if "full" in record else dict(four="the link fit")
    found = []
    for key, name in names.items():
        fit = record[key]
        if not fit.get("converged"):
            found.append(f"{name.capitalize()} of the training half did not converge and is used as it stands")
        if not profiles.finite_link(fit):
            found.append(f"{name.capitalize()} of the training half holds a value that is not finite, so every solve of SOLID fails")
    check = record["four"].get("nested_check") or {}
    if check.get("passed") is False:
        found.append(f"The nested check of the refit failed, {check.get('failure')}")
    containment = deployment.heldout_load()["datasets"][dataset]["containment"]
    outside = containment["linked_outside"] + containment["unlinked_outside"]
    if outside:
        found.append(f"{outside} true agent positions of the training half lie outside their unit's operating area, which SOLID alone receives")
    return found


def unsolved(units, path, workers):
    """Every unit recorded as SOLID's failure without a solve, for a link model with a value that is not finite, sealed as
    common.solve seals the predictions of a run so that the run record verifies them. The scorer charges each the
    failure penalty."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(id=unit["id"], samples=unit["samples"], label=unit.get("label"), method="solid",
                                            agent_ids=unit["problem"]["agent_ids"], estimate=None, elapsed_ns=0, failure=UNSOLVED)) + "\n"
                            for unit in units))
    identity = dict(request_sha256=common.fingerprint(dict(units=units)), binary_sha256=common.sha256(common.BINARY), workers=workers)
    common.seal_cache(path.with_suffix(".manifest.json"), identity, [path], unsolved_units=len(units))
    return common.prediction_records(path, units)


def run(dataset, split, workers):
    freeze = heldout.require_run(dataset, split)
    heldout.require_stage_one()
    out_dir = common.OUT / dataset / split / "anc"
    request, reference, identity = inputs(dataset, split)
    metadata = {m["id"]: m for m in reference["units"]}
    name = f"{dataset}-anchor-only.json"
    # the tracked profile of stage one, refused unless it holds its composition from the method and held-out records
    profile, source = profiles.heldout_resolve(name, common.PROFILES / name)
    units, lookup = [], {}
    for unit in request["units"]:
        area = deployment.heldout_unit_area(unit["problem"], dataset)
        if area != unit["area"]:
            raise ValueError("a unit's operating area differs from the one its baselines received")
        lookup[(unit["id"], unit["samples"])] = len(units)
        units.append(dict(id=len(units), samples=unit["samples"], problem=unit["problem"], label=metadata[unit["id"]]["acquisition"],
                          deployment_profile=dict(area=area, anchor_only=profile["anchor_only"])))
    solved_profile = profiles.heldout_provenance(out_dir, profile, source)
    outcomes = marks(dataset)
    if profiles.finite_link(profile["anchor_only"]):
        records = common.solve(units, out_dir / "predictions.jsonl", workers)
    else:
        records = unsolved(units, out_dir / "predictions.jsonl", workers)
    solved = {key: records[index] for key, index in lookup.items()}
    rows = []
    for s in common.score(request, reference, solved, common.load_scorer()):
        entry = metadata[s["id"]]
        rows.append(dict(key=f"{dataset}-{s['id']}", id=s["id"], samples=s["samples"], cell=s["acquisition"], level=entry["anchors"],
                         acquisition=s["acquisition"], anchors=entry["anchors"], error_cm=s["mean_pair_error_cm"], elapsed_ms=s["elapsed_ms"],
                         failure=s["failure"], pair_sets=s["pair_sets"]))
    common.write_run(out_dir, rows, workers=workers, dataset=dataset, mode="ANC", split=split, **solved_profile, **identity, **freeze,
                     declaration_sha256=heldout.sha256(heldout.RECORD), headline=dict(zip(("cells", "separate_rows"), heldout.headline(dataset))),
                     marks=outcomes, complete=True)
    for mark in outcomes:
        print(f"{dataset} {split}: {mark} (the declaration's rules.outcomes)", flush=True)
    if identity["excluded_flights"]:
        print(f"{dataset} {split}: excluded by the relocation guard " + ", ".join(f"{flight} ({entry['change_cm']} cm, {entry['units']} units)"
                                                                            for flight, entry in identity["excluded_flights"].items()), flush=True)
    if split != "training" and identity.get("training_excluded_flights"):
        print(f"{dataset} {split}: training flights excluded by the relocation guard, out of every fit and selection, "
              + ", ".join(f"{flight} ({entry['change_cm']} cm, {entry['units']} units)" for flight, entry in identity["training_excluded_flights"].items()),
              flush=True)
    if identity.get("unguarded_pairs"):
        print(f"{dataset} {split}: pairs without an admissible report, left out of the relocation guard, "
              + ", ".join(f"{flight} ({', '.join(pairs)})" for flight, pairs in identity["unguarded_pairs"].items()), flush=True)
    return rows


def main(dataset, doc):
    import argparse
    parser = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=["training", "evaluation"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    rows = run(dataset, args.split, args.workers)
    common.summarize(rows, f"{dataset} ANC {args.split}", "ANC", groups=lambda part: heldout.report_groups(part, dataset))
