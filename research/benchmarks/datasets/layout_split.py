"""Same-layout training and evaluation halves of the external datasets.

Every external dataset records one deployment. The paper's protocol reserves the first part of each recording for
training, by the order of its reports, and scores only the rest, so both halves share the layout, the devices and the
radio settings, and no positioning result is ever scored on rows that a fit, a selection or a design decision saw.
IDLab's split already works this way. This script states the same rule for the other two datasets over their frozen
populations and writes one tracked record, research/artifacts/solid/splits/layout-split.json, that every runner reads:

- TIERS: the blocks of ten reports of each static recording by rank, the block of rank k holding the reports 10k to
  10k+9 of every link. The first half of the blocks trains and the rest evaluate (22 of 44, 22 of 45 and 25 of 50),
  every anchor subset of a block on the same side. The halves split every link's reports by their order, and since the
  links report at different rates they overlap in time across links.
- Bregar: the three blocks of ten reports of every environment and channel; the first block trains and the other two
  evaluate, the whole constellation on the same side. bregar_layout.py builds the populations from that rule.
- IDLab: the reserved calibration rows form the training campaign and the evaluation rows the frozen campaign.

Usage: layout_split.py [--check]
"""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import audit

DATA = audit.DATA
RECORD = audit.ROOT / "research/artifacts/solid/splits/layout-split.json"
SCHEMA = "solid.split.layout.v1"
BREGAR_TRAINING_BLOCKS = (0,)
BREGAR_EVALUATION_BLOCKS = (1, 2)
SIDES = ("training", "evaluation")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    path = Path(path)
    if path.suffix == ".gz":
        return json.loads(gzip.decompress(path.read_bytes()))
    return json.loads(path.read_text())


def tiers_split(base, catalog):
    """The first half of every recording's blocks by rank trains, the rest evaluate."""
    by_acquisition = defaultdict(list)
    for block in base["blocks"]:
        by_acquisition[block["acquisition"]].append(block)
    acquisitions, sides = {}, {}
    for acquisition, blocks in sorted(by_acquisition.items()):
        ordered = sorted(blocks, key=lambda b: b["rank"])
        if [b["rank"] for b in ordered] != list(range(len(ordered))):
            raise ValueError(f"block ranks of {acquisition} are not consecutive from zero")
        half = len(ordered) // 2
        training, evaluation = [b["id"] for b in ordered[:half]], [b["id"] for b in ordered[half:]]
        acquisitions[acquisition] = dict(blocks=len(ordered), training_blocks=training, evaluation_blocks=evaluation)
        for side, ids in (("training", training), ("evaluation", evaluation)):
            for block_id in ids:
                if block_id in sides:
                    raise ValueError("a block identifier appears twice")
                sides[block_id] = side
    if set(sides) != {b["id"] for b in base["blocks"]}:
        raise ValueError("the split does not cover every block exactly once")
    counts = {side: defaultdict(lambda: dict(batches=0, units=0)) for side in SIDES}
    for batch in catalog:
        if batch["block"] not in sides:
            raise ValueError("catalog batch of an unknown block")
        if batch["track"] != "calibrated":
            continue
        entry = counts[sides[batch["block"]]][batch["mode"]]
        entry["batches"] += 1
        entry["units"] += batch["units"]
    return dict(rule="The first floor(n/2) blocks of each recording by rank train, the remaining blocks evaluate. All anchor subsets of a block are on its side.",
                acquisitions=acquisitions,
                training_blocks=sorted(b for b, s in sides.items() if s == "training"),
                evaluation_blocks=sorted(b for b, s in sides.items() if s == "evaluation"),
                calibrated_track={side: dict(counts[side]) for side in SIDES})


def bregar_sid(environment, channel_index, block):
    return 1 + 18 * environment + 3 * channel_index + block


def bregar_split(reference=None):
    """The first block of ten reports of every environment and channel trains, the other two evaluate."""
    training, evaluation = [], []
    for environment in range(4):
        for channel_index in range(6):
            training.extend(bregar_sid(environment, channel_index, b) for b in BREGAR_TRAINING_BLOCKS)
            evaluation.extend(bregar_sid(environment, channel_index, b) for b in BREGAR_EVALUATION_BLOCKS)
    if reference is not None:
        by_block = {side: sorted(u["id"] for u in reference["units"] if u["block"] in blocks)
                    for side, blocks in (("training", BREGAR_TRAINING_BLOCKS), ("evaluation", BREGAR_EVALUATION_BLOCKS))}
        if by_block != dict(training=sorted(training), evaluation=sorted(evaluation)):
            raise ValueError("the frozen Bregar identities do not follow the block scheme")
    return dict(rule="Reports 1 to 10 of every link (block 0) train, reports 11 to 30 (blocks 1 and 2) evaluate, in every environment and channel. "
                     "Unit identity 1 + 18 environment + 3 channel + block, both sample counts of a unit on its side.",
                training_blocks=list(BREGAR_TRAINING_BLOCKS), evaluation_blocks=list(BREGAR_EVALUATION_BLOCKS),
                training_units=sorted(training), evaluation_units=sorted(evaluation))


def idlab_split():
    split = read(DATA / "idlab-split.json")
    result = dict(rule=split["split_rule"], split_sha256=sha256(DATA / "idlab-split.json"),
                  totals=split["totals"], campaigns={})
    for side, folder in (("training", "idlab-training-campaign"), ("evaluation", "idlab-campaign")):
        freeze = DATA / folder / "freeze.json"
        if freeze.exists():
            record = read(freeze)
            result["campaigns"][side] = dict(folder=folder, freeze_sha256=sha256(freeze), inputs=record.get("inputs"),
                                             batches=record.get("batches", record.get("budget", {}).get("checkpoints")))
    return result


def build():
    tiers_dir, bregar_dir = DATA / "tiers-static-campaign", DATA / "bregar-evaluation"
    tiers = tiers_split(read(tiers_dir / "base.json.gz"), read(tiers_dir / "catalog.json.gz"))
    bregar = bregar_split(read(bregar_dir / "reference.json") if (bregar_dir / "reference.json").exists() else None)
    sources = {str(p.relative_to(DATA)): sha256(p) for p in (tiers_dir / "base.json.gz", tiers_dir / "catalog.json.gz", tiers_dir / "freeze.json")}
    # The tracked record holds this sentence as it was written, and --check compares the record with it. For TIERS the
    # record's own rule, by rank, states the split, whose halves overlap in time across links.
    return dict(schema=SCHEMA,
                rule="The first part of every recording in time trains, the rest evaluates. Fits, selections and design decisions "
                     "use training rows only. Reported accuracy comes from the evaluation half alone.",
                tiers=tiers, bregar=bregar, idlab=idlab_split(), sources=sources, source_sha256=sha256(Path(__file__)))


def load():
    return json.loads(RECORD.read_text())


def tiers_blocks(side):
    return set(load()["tiers"][f"{side}_blocks"])


def bregar_units(side):
    return set(load()["bregar"][f"{side}_units"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="verify that the tracked record matches the frozen populations")
    args = parser.parse_args()
    record = build()
    if args.check:
        current = load()
        if {k: v for k, v in current.items() if k != "source_sha256"} != {k: v for k, v in record.items() if k != "source_sha256"}:
            raise ValueError("the tracked split record differs from the frozen populations")
        print("split record verified")
        return
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    tiers, bregar = record["tiers"], record["bregar"]
    print(f"TIERS blocks {len(tiers['training_blocks'])} training / {len(tiers['evaluation_blocks'])} evaluation, "
          + ", ".join(f"{a} {len(v['training_blocks'])}/{len(v['evaluation_blocks'])}" for a, v in tiers["acquisitions"].items()))
    print("TIERS calibrated units", json.dumps(tiers["calibrated_track"]))
    print(f"Bregar units {len(bregar['training_units'])} training / {len(bregar['evaluation_units'])} evaluation (one identity per block, two sample counts each)")
    print("IDLab", json.dumps({k: v for k, v in record["idlab"]["campaigns"].items()}))
    print(f"written {RECORD.relative_to(audit.ROOT)}")


if __name__ == "__main__":
    main()
