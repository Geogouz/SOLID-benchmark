"""SOLID on the frozen TIERS static campaign inputs, calibrated track, one half of the same-layout split, one mode per call.

Builds the exact campaign requests of the half's blocks and gives every unit the TIERS profile of the mode, composed in
memory by profiles.py from the record of the method the confirmatory round confirmed, or before the round from the fit
records and the deployment record, or the profile passed with --profile. A tracked profile file is solved only when it
holds that composition. The first stage of the cooperative profiles must be the frozen campaign fit unless the method
derives it from its link model, or --no-campaign-scales is given for a profile whose first stage is derived. Every
anchor only unit takes the operating area of the deployment rules of deployment.py. Every request passes through
layout_inputs.levelled, so SOLID's units start where every baseline's start. Solves with solid-units and scores every
unit with the campaign scorer. Cells follow the campaign weighting, acquisition and anchor count. The half comes
from the tracked split record research/artifacts/solid/splits/layout-split.json. The training half feeds fits and design
decisions, the evaluation half is reported. Output: out/solid/tiers/<split>/<mode>/scores.json with the prediction
chunks and the solved profile beside it, or the directory passed with --out. Every row keeps the mean pair error of each
pair set of its unit's inputs.

Usage: run_tiers.py --mode af|aac|anc --split training|evaluation [--workers 8] [--profile PATH] [--out DIR] [--blocks id ...]
                    [--no-campaign-scales]
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
import tiers_campaign as campaign  # noqa: E402

D = common.EXTERNAL / "tiers-static-campaign"
PROFILE = {"AF": "tiers-agent-only.json", "AAC": "tiers-anchor-aided.json", "ANC": "tiers-anchor-only.json"}
CHUNK_BLOCKS = 6


def prepared(unit, profile, mode):
    """The profile part of one unit of the request: the operating area and the anchor only block, or the cooperative
    profile of the unit's sample count."""
    if mode == "ANC":
        return dict(deployment_profile=dict(area=deployment.unit_area(unit["problem"], "tiers"), anchor_only=profile["anchor_only"]))
    return dict(cooperative_profile=profile[str(unit["samples"])])


def check_campaign_profile(request, profile, mode):
    """The cooperative scales of the profile must equal the frozen campaign fit."""
    if mode == "ANC":
        return
    for samples in ("1", "10"):
        fitted = request["profile"]["free"][samples] if mode == "AF" else request["profile"]["anchored"][samples]["2"]
        mine = profile[samples]["range_error"]
        if (fitted["positive_scale"], fitted["negative_scale"]) != (mine["positive_scale"], mine["negative_scale"]):
            raise ValueError("the profile does not match the frozen campaign fit")


def run(mode, split, workers, profile_path=None, out_dir=None, blocks=None, campaign_scales=None):
    """Position the blocks of one half in one mode and return the scored rows. With campaign_scales the first stage
    multipliers of a cooperative profile must be the frozen campaign fit, by default unless the method record derives
    them. A design round's candidate whose first stage is derived from its link model runs without that check."""
    mode = mode.upper()
    # a selection of no block positions nothing, never every block of the half
    if blocks is not None and not set(blocks):
        raise ValueError("the selection names no TIERS block, so the run would position nothing")
    if campaign_scales is None:
        campaign_scales = profiles.campaign_first_stage()
    base = json.load(gzip.open(D / "base.json.gz"))
    catalog = json.load(gzip.open(D / "catalog.json.gz"))
    half = layout_split.tiers_blocks(split)
    every = sorted({b["block"] for b in catalog if b["track"] == "calibrated" and b["mode"] == mode} & half)
    if blocks is None:
        blocks = every
    else:
        blocks = sorted(set(blocks) & half)
        if not blocks:
            raise ValueError(f"no block of the selection lies in the TIERS {split} half, so the run would position nothing")
    profile, source = profiles.resolve(PROFILE[mode], profile_path)
    scorer = common.load_scorer()
    out_dir = Path(out_dir) if out_dir else common.OUT / "tiers" / split / mode.lower()
    solved_profile = profiles.provenance(out_dir, profile, source)
    rows = []
    # prediction files cover windows of six consecutive block identifiers, whether or not every identifier exists
    for start in range(1, max(blocks) + 1, CHUNK_BLOCKS):
        chunk = [b for b in range(start, start + CHUNK_BLOCKS) if b in blocks]
        if not chunk:
            continue
        batches = [b for b in catalog if b["track"] == "calibrated" and b["mode"] == mode and b["block"] in chunk]
        built, units, lookup = [], [], {}
        for batch in batches:
            request, reference = campaign.batch_inputs(base, batch)
            # every method's starts by SOLID's deployment rules, the request the baselines receive too
            request = layout_inputs.levelled(request, "tiers", mode)
            if campaign_scales:
                check_campaign_profile(request, profile, mode)
            built.append((batch, request, reference))
            for unit in request["units"]:
                lookup[(batch["key"], unit["id"], unit["samples"])] = len(units)
                units.append(dict(id=len(units), samples=unit["samples"], problem=unit["problem"], label=batch["key"], **prepared(unit, profile, mode)))
        records = common.solve(units, out_dir / f"predictions-b{start:03d}.jsonl", workers)
        for batch, request, reference in built:
            solved = {(u["id"], u["samples"]): records[lookup[(batch["key"], u["id"], u["samples"])]] for u in request["units"]}
            for s in common.score(request, reference, solved, scorer):
                cell = s["acquisition"] if mode == "AF" else f'{s["acquisition"]}/anchors-{batch["anchors"]}'
                rows.append(dict(key=batch["key"], id=s["id"], samples=s["samples"], cell=cell, level=batch["anchors"],
                                 acquisition=s["acquisition"], block=batch["block"], anchors=batch["anchors"],
                                 error_cm=s["mean_pair_error_cm"], elapsed_ms=s["elapsed_ms"], failure=s["failure"], pair_sets=s["pair_sets"]))
    common.write_run(out_dir, rows, workers=workers, dataset="tiers", track="calibrated", mode=mode, split=split, **solved_profile,
                     blocks=blocks, split_record_sha256=common.sha256(layout_split.RECORD),
                     campaign_freeze_sha256=common.sha256(D / "freeze.json"), campaign_scales=campaign_scales, complete=blocks == every,
                     shared_inputs=layout_inputs.record("tiers"))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["af", "aac", "anc"])
    parser.add_argument("--split", required=True, choices=["training", "evaluation"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--profile", default=None, help="profile file, the composition of the fit records by default")
    parser.add_argument("--out", default=None, help="output directory, out/solid/tiers/<split>/<mode> by default")
    parser.add_argument("--blocks", type=int, nargs="+", default=None, help="block ids, every block of the half by default")
    parser.add_argument("--no-campaign-scales", action="store_true",
                        help="solve a first stage other than the frozen campaign fit, the method's derived one being taken without it")
    args = parser.parse_args()
    rows = run(args.mode, args.split, args.workers, args.profile, args.out, args.blocks, campaign_scales=False if args.no_campaign_scales else None)
    common.summarize(rows, f"TIERS {args.mode.upper()} {args.split}", args.mode.upper())


if __name__ == "__main__":
    main()
