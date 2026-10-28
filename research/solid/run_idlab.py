"""SOLID on the frozen IDLab campaign inputs, one half of the same-layout split, one mode per call.

The evaluation half is the frozen campaign (idlab-campaign), the training half the campaign built from the reserved
calibration rows (idlab-training-campaign). Builds the exact campaign requests from the half's base and catalog and
gives every unit the IDLab profile of the mode, composed in memory by profiles.py from the record of the method the
confirmatory round confirmed, or before the round from the fit records and the deployment record, or the profile passed
with --profile. A tracked profile file is solved only when it holds that composition. Every anchor only unit takes the
operating area of the deployment rules of deployment.py. Every request passes through layout_inputs.levelled, so SOLID's
units start where every baseline's start. Solves with solid-units and scores every unit with the campaign
scorer. A stride keeps every batch whose key hash falls in class zero modulo the stride, a fixed subsample spread over
every configuration and anchor count, for design rounds on the training half. A panel stride keeps every batch of the
panels whose key, panel-NNNN, hashes to class zero, so whole panels stay together in every mode within the radio
configurations, the clusters of the decision's IDLab design. --configurations keeps the batches of the named radio configurations alone, those of one fold
of the confirmatory round for example. Output: out/solid/idlab/<split>/<mode>/scores.json with the prediction chunks
and the solved profile beside it, or --out. Every row keeps the mean pair error of each pair set of its unit's inputs.
Baseline scores on the same units come from the campaign checkpoints when the dashboard evidence is collected.

Usage: run_idlab.py --mode af|aac|anc --split training|evaluation [--workers 8] [--chunk 400] [--limit N] [--stride K]
                    [--panel-stride K] [--configurations C ...] [--profile PATH] [--out DIR]
"""
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import deployment  # noqa: E402
import idlab_campaign as campaign  # noqa: E402
import layout_inputs  # noqa: E402
import profiles  # noqa: E402

CAMPAIGNS = {"evaluation": "idlab-campaign", "training": "idlab-training-campaign"}
PROFILE = {"AF": "idlab-agent-only.json", "AAC": "idlab-anchor-aided.json", "ANC": "idlab-anchor-only.json"}


PANEL_KEY = "panel-{:04d}"


def in_stride(key, stride):
    return stride <= 1 or int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % stride == 0


def in_panel_stride(panel, stride):
    """Whether a panel's key hash falls in class zero modulo the stride, the same panels in every mode."""
    return in_stride(PANEL_KEY.format(panel), stride)


def batches_of(catalog, mode, stride=1, panel_stride=1, configurations=None):
    """The batches of one mode kept by the batch key stride and the panel stride, of the named configurations alone when
    given."""
    return [b for b in catalog if b["mode"] == mode and in_stride(b["key"], stride) and in_panel_stride(b["panel"], panel_stride)
            and (configurations is None or b["configuration"] in configurations)]


def prepared(unit, profile, mode):
    """The profile part of one unit of the request: the operating area and the anchor only block, or the cooperative
    profile of the unit's sample count."""
    if mode == "ANC":
        return dict(deployment_profile=dict(area=deployment.unit_area(unit["problem"], "idlab"), anchor_only=profile["anchor_only"]))
    return dict(cooperative_profile=profile[str(unit["samples"])])


def run(mode, split, workers, chunk=400, limit=0, stride=1, profile_path=None, out_dir=None, panel_stride=1, configurations=None):
    mode = mode.upper()
    # a selection of no configuration positions nothing, never every configuration of the campaign
    if configurations is not None and not set(configurations):
        raise ValueError("the selection names no IDLab configuration, so the run would position nothing")
    D = common.EXTERNAL / CAMPAIGNS[split]
    base = json.load(gzip.open(D / "base.json.gz"))
    base["values"] = {(v[0], v[1]): v[2] for v in base["values"]}
    catalog = json.load(gzip.open(D / "catalog.json.gz"))
    if configurations is not None and not set(configurations) <= {b["configuration"] for b in catalog}:
        raise ValueError(f"configurations {sorted(set(configurations) - {b['configuration'] for b in catalog})} are not in the {split} campaign")
    profile, source = profiles.resolve(PROFILE[mode], profile_path)
    scorer = common.load_scorer()
    batches = batches_of(catalog, mode, stride, panel_stride, configurations)
    if limit:
        batches = batches[:limit]
    out_dir = Path(out_dir) if out_dir else common.OUT / "idlab" / split / mode.lower()
    solved_profile = profiles.provenance(out_dir, profile, source)
    chunks = [batches[i:i + chunk] for i in range(0, len(batches), chunk)]
    rows = []
    for index, part in enumerate(chunks):
        built, units, lookup = [], [], {}
        for batch in part:
            request, reference = campaign.batch_inputs(base, batch)
            # every method's starts by SOLID's deployment rules, the request the baselines receive too
            request = layout_inputs.levelled(request, "idlab", mode)
            built.append((batch, request, reference))
            for unit in request["units"]:
                lookup[(batch["key"], unit["id"], unit["samples"])] = len(units)
                units.append(dict(id=len(units), samples=unit["samples"], problem=unit["problem"], label=batch["key"], **prepared(unit, profile, mode)))
        records = common.solve(units, out_dir / f"predictions-c{index:03d}.jsonl", workers)
        for batch, request, reference in built:
            solved = {(u["id"], u["samples"]): records[lookup[(batch["key"], u["id"], u["samples"])]] for u in request["units"]}
            for s in common.score(request, reference, solved, scorer):
                rows.append(dict(key=batch["key"], id=s["id"], samples=s["samples"], cell=s["acquisition"], level=batch["anchors"],
                                 acquisition=s["acquisition"], anchors=batch["anchors"], configuration=batch["configuration"],
                                 panel=batch["panel"], error_cm=s["mean_pair_error_cm"], elapsed_ms=s["elapsed_ms"], failure=s["failure"],
                                 pair_sets=s["pair_sets"]))
        print(f"chunk {index + 1}/{len(chunks)} scored, {len(rows)} units so far", file=sys.stderr, flush=True)
    common.write_run(out_dir, rows, workers=workers, dataset="idlab", mode=mode, split=split, stride=stride, panel_stride=panel_stride,
                     configurations=None if configurations is None else sorted(configurations), batches=len(batches), **solved_profile,
                     campaign=CAMPAIGNS[split], campaign_freeze_sha256=common.sha256(D / "freeze.json"),
                     complete=not limit and stride <= 1 and panel_stride <= 1 and configurations is None,
                     shared_inputs=layout_inputs.record("idlab"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["af", "aac", "anc"])
    parser.add_argument("--split", required=True, choices=["training", "evaluation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=400, help="batches per prediction file")
    parser.add_argument("--limit", type=int, default=0, help="first N batches only")
    parser.add_argument("--stride", type=int, default=1, help="keep the batches whose key hash falls in class zero modulo the stride")
    parser.add_argument("--panel-stride", type=int, default=1, help="keep the panels whose key hash falls in class zero modulo the stride")
    parser.add_argument("--configurations", type=int, nargs="+", default=None, help="keep the batches of these radio configurations alone")
    parser.add_argument("--profile", default=None, help="profile file, the composition of the fit records by default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    rows = run(args.mode, args.split, args.workers, args.chunk, args.limit, args.stride, args.profile, args.out, args.panel_stride, args.configurations)
    common.summarize(rows, f"IDLab {args.mode.upper()} {args.split}", args.mode.upper())


if __name__ == "__main__":
    main()
