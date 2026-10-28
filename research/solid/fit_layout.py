"""Link model fits on the same-layout training halves of TIERS and Bregar.

The block model of the method is fitted on residual blocks of the deployment itself, with the truth of the training
half, so the profile reflects the devices and the environment the evaluation half is scored in. The training half is
the one named in research/artifacts/solid/splits/layout-split.json and no evaluation unit is read.

Strata are sample tracks. A block belongs to the track of the unit it comes from, one sample or the full track of ten,
never to its raw packet count, so a full-track block that lost packets stays in the full track. Every track receives
the same total weight and the weights sum to the block count. At most --blocks blocks are drawn per track.

- TIERS: the agent to agent links of the training blocks of the three static recordings, calibrated track, the ten
  sample units and the one sample units. Truth from the audited node coordinates in three dimensions.
- Bregar: per environment, the ten sample and the one sample units of the training block. Truth from the surveyed
  positions and heights of the same-layout reference.

An observation without samples (a link the radio answered with no admissible reading) forms no block. Every fit record
holds `four`, the fit of the link model on the residual blocks.

With --fold the fit reads the clusters of one fold of the confirmatory round alone, research/solid/folds.py, the TIERS
training blocks or the Bregar acquisitions of that fold, and every record holds the fold's entry. The round positions
the units of the other fold with it.

Usage: fit_layout.py [--datasets tiers,bregar] [--blocks 2000] [--workers 15] [--offset-shape gaussian] [--early-detection]
                     [--nu 3] [--suffix S] [--fold 0|1]
       writes fits/tiers_layout_fit<S>.json and fits/bregar_layout_fits<S>.json, the suffix empty by default, so the fit
       of a model variant can sit next to the tracked fit, and with --fold K the files end in _foldK before .json
"""
import argparse
import gzip
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_links  # noqa: E402
import folds  # noqa: E402
import layout_split  # noqa: E402

SEEDS = dict(tiers=11, bregar=13)


def answered(observations):
    """The observations that carry samples. A layout builder may emit a link the radio answered without any admissible
    reading, which holds no range and forms no block."""
    return [observation for observation in observations if observation["calibrated_samples"]]


def bregar_unit_blocks(unit, truth, heights):
    """(agent, anchor, True, block) for every Bregar observation of a unit that carries samples, with the slant
    distance from the surveyed position and height of the agent to the anchor."""
    anchors = {a["id"]: (a["position"]["x"], a["position"]["y"], a["height_cm"]) for a in unit["problem"]["anchors"]}
    items = []
    for observation in answered(unit["problem"]["observations"]):
        agent = str(observation["from"])
        distance = math.dist((truth[agent][0], truth[agent][1], heights[agent]), anchors[observation["to"]])
        items.append((observation["from"], observation["to"], True, block(observation, distance)))
    return items


def block(observation, distance):
    samples = observation["calibrated_samples"]
    return (np.array([s["distance_cm"] for s in samples], dtype=float), np.array([s["scale_cm"] for s in samples], dtype=float), float(distance))


# ---------------------------------------------------------------------------------------------------------------
# Strata
def stratified_sample(groups, cap, rng):
    """At most `cap` blocks per track drawn without replacement, and weights with the same total weight in every track
    summing to the block count. groups maps a track (the unit's sample count) to its blocks."""
    chosen, strata = [], []
    for track, blocks in sorted(groups.items()):
        picked = [blocks[i] for i in rng.choice(len(blocks), min(cap, len(blocks)), replace=False)]
        chosen.extend(picked)
        strata.extend([track] * len(picked))
    counts = {str(track): strata.count(track) for track in sorted(set(strata))}
    return chosen, fit_links.stratified_weights(strata), counts, strata


def packet_counts(blocks):
    return {str(k): v for k, v in sorted(Counter(len(b[0]) for b in blocks).items())}


def fit_groups(groups, cap, seed, model=None, workers=1):
    """The fit of the link model on the residual blocks. groups maps a track to (a, b, to_is_anchor, block) items."""
    model = dict(model or {})
    plain = {track: [b for _, _, _, b in tagged] for track, tagged in groups.items()}
    chosen, weights, counts, strata = stratified_sample(plain, cap, np.random.default_rng(seed))
    fit = fit_links.fit_link_model(chosen, weights, workers=workers, **model)
    return dict(four=fit, blocks=len(chosen), blocks_cap=cap, blocks_per_track=counts, blocks_per_length=packet_counts(chosen),
                weight_per_track=fit_links.stratum_totals(strata, weights), nu=fit["nu"])


def identity(**sources):
    return dict(split="training", split_record_sha256=common.sha256(layout_split.RECORD), evaluation_rows_used=0,
                sources={name: common.sha256(path) for name, path in sources.items()})


# ---------------------------------------------------------------------------------------------------------------
# Datasets
def tiers_blocks(fold=None):
    """The TIERS training blocks a fit reads, those of one fold of the confirmatory round when a fold is given."""
    blocks = layout_split.tiers_blocks("training")
    return blocks if fold is None else blocks & folds.tiers_blocks(fold)


def tiers_groups(fold=None):
    """Blocks per track of the TIERS training half, or of one fold of it, and the identity of their sources."""
    import tiers_campaign as campaign
    D = common.EXTERNAL / "tiers-static-campaign"
    base = json.load(gzip.open(D / "base.json.gz"))
    catalog = json.load(gzip.open(D / "catalog.json.gz"))
    blocks = tiers_blocks(fold)
    groups = defaultdict(list)
    for batch in catalog:
        if batch["track"] != "calibrated" or batch["mode"] != "AF" or batch["block"] not in blocks:
            continue
        request, _ = campaign.batch_inputs(base, batch)
        truth = base["coordinates_cm"][batch["acquisition"]]
        for unit in request["units"]:
            for observation in answered(unit["problem"]["observations"]):
                if observation["to_is_anchor"]:
                    continue
                distance = math.dist(truth[str(observation["from"])], truth[str(observation["to"])])
                groups[unit["samples"]].append((observation["from"], observation["to"], False, block(observation, distance)))
    return groups, identity(base=D / "base.json.gz", catalog=D / "catalog.json.gz")


def bregar_groups(fold=None):
    """Blocks per environment and track of the Bregar training half, or of the acquisitions of one fold of it, and the
    identity of their sources."""
    D = common.EXTERNAL / "bregar-layout"
    request = json.load(gzip.open(D / "training" / "request.json.gz"))
    reference = common.read_json(D / "training" / "reference.json")
    meta = {u["id"]: u for u in reference["units"]}
    if {u["id"] for u in request["units"]} != layout_split.bregar_units("training"):
        raise ValueError("the Bregar training request does not hold the training identities")
    acquisitions = None if fold is None else folds.bregar_acquisitions(fold)
    groups = defaultdict(lambda: defaultdict(list))
    for unit in request["units"]:
        if acquisitions is not None and meta[unit["id"]]["acquisition"] not in acquisitions:
            continue
        environment = meta[unit["id"]]["reference"]
        truth, heights = reference["coordinates_cm"][environment], reference["heights_cm"][environment]
        for item in bregar_unit_blocks(unit, truth, heights):
            groups[environment][unit["samples"]].append(item)
    return groups, identity(request=D / "training" / "request.json.gz", reference=D / "training" / "reference.json")


def link_summary(record):
    return {key: round(fit_links.parameter(record, key), 4) for key in fit_links.PARAMETERS} | dict(nll=round(record["nll"], 3), converged=record["converged"],
                                                                                optima=record["distinct_optima"])


# ---------------------------------------------------------------------------------------------------------------
# Held-out benchmarks
#
# HYMN and AAU Meshed UWB, declared in research/artifacts/solid/splits/heldout-benchmarks.json, are fitted like the
# layouts above on the blocks of their training halves only. research/solid/fit_heldout.py runs these collectors with
# fit_groups. Every training link and block enters once, with both tracks, rather than as often as the units repeat it.

HELDOUT_SEEDS = dict(hymn=14, aau=15)


def heldout_identity(dataset, evaluation_rows_used, **sources):
    """The training identity of a held-out fit, tied to the held-out declaration instead of the layout split, with the
    readings of the declared evaluation half its collector took (evaluation_rows)."""
    import heldout
    return dict(split="training", dataset=dataset, declaration_sha256=common.sha256(heldout.RECORD), evaluation_rows_used=evaluation_rows_used,
                sources={name: common.sha256(path) for name, path in sources.items()})


def evaluation_rows(dataset, read):
    """The readings a held-out collector took from the declared evaluation half, those of the HYMN points or of the AAU
    flights the declaration names for evaluation, counted as heldout.evaluation_units counts the units of a selection.
    read maps a point label or a flight name to the readings taken from it. A collector that took any is refused."""
    import heldout
    split = heldout.section(dataset)["split"]
    evaluation = set(split["evaluation_points"] if dataset == "hymn" else split["evaluation_flights"])
    count = sum(value for name, value in read.items() if name in evaluation)
    if count:
        raise ValueError(f"the {dataset} training fit read {count} readings of the declared evaluation half")
    return count


def heldout_calibration(module):
    """The calibration a held-out builder wrote from its training half, checked against the current declaration."""
    import heldout
    return heldout.training_calibration(module.calibration_path(), module.ARCHIVE_SHA256)


def heldout_blocks(groups, a, b, values, scale_cm, distance, first=None):
    """Append the one-reading and the ten-reading block of one link, each only when it holds an admissible reading. The
    one-reading block holds first, the reading of the block's first attempt as the held-out units take it
    (heldout.units_of), by default the first of values, and a link whose first attempt holds no reading forms none."""
    import heldout
    first = values[:1] if first is None else first
    if len(first) > 1 or (first and first[0] != values[0]):
        raise ValueError("the one-reading block of a link is the first reading of its ten-reading block")
    for samples in heldout.TRACKS:
        readings = first if samples == 1 else values[:samples]
        if not readings:
            continue
        observation = heldout.observation(a, b, readings, scale_cm)
        if observation["calibrated_samples"]:
            groups[samples].append((a, b, True, block(observation, distance)))


def hymn_groups():
    """Blocks per track of the HYMN training half. Every training point, anchor and declared window with a published
    reading gives a ten-epoch block of the window's readings and a one-reading block of the reading of the window's first
    epoch, none when that epoch holds no reading, as the held-out units take them, with the slant distance from the tag's
    surveyed position to the anchor."""
    import hymn_layout
    data = hymn_layout.load()
    scale = heldout_calibration(hymn_layout)["scale_cm"]
    identities, halves = hymn_layout.split(data["points"])
    labels = {identity: label for label, identity in identities.items()}
    groups, read = defaultdict(list), defaultdict(int)
    for point in halves["training"]:
        epochs = data["epochs"][labels[point]]
        for anchor, position in sorted(data["anchors"].items()):
            distance = math.dist(data["points"][labels[point]], position)
            for window in range(hymn_layout.WINDOWS):
                values = hymn_layout.window_readings(epochs, anchor, window)
                if values:
                    reading = hymn_layout.first_reading(epochs, anchor, window)
                    heldout_blocks(groups, point, anchor, values, scale, distance, [] if reading is None else [reading])
                    read[labels[point]] += len(values)
    return groups, heldout_identity("hymn", evaluation_rows_used=evaluation_rows("hymn", read), archive=hymn_layout.ARCHIVE,
                                    calibration=hymn_layout.calibration_path())


def aau_groups():
    """Blocks per track of the AAU training flights. Every block of every ordered pair of static nodes gives its first
    report and its ten reports, calibrated by the published pair biases, with the published distance between the nodes.
    A training flight beyond the relocation guard, which the calibration left out (aau_layout.calibrate), stays out of the
    fit, as out of the units and the scale."""
    import aau_layout
    data = aau_layout.load()
    calibration = heldout_calibration(aau_layout)
    scale = calibration["scale_cm"]
    training = set(aau_layout.split(data["flights"])["training"]) - set(calibration.get("excluded_flights") or {})
    nodes = data["positions"]
    groups, read = defaultdict(list), defaultdict(int)
    for flight in data["flights"]:
        if flight["name"] not in training:
            continue
        for index in range(aau_layout.blocks(flight)):
            first = 10 * index
            for (a, b), reports in sorted(flight["streams"].items()):
                values = aau_layout.calibrated([r[2] for r in reports[first:first + 10]], (a, b), data["biases"]).tolist()
                heldout_blocks(groups, a, b, values, scale, math.dist(nodes[a], nodes[b]))
                read[flight["name"]] += len(values)
    return groups, heldout_identity("aau", evaluation_rows_used=evaluation_rows("aau", read), archive=aau_layout.ARCHIVE,
                                    calibration=aau_layout.calibration_path())


HELDOUT_GROUPS = dict(hymn=hymn_groups, aau=aau_groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="tiers,bregar")
    parser.add_argument("--blocks", type=int, default=2000, help="at most this many blocks per track")
    parser.add_argument("--workers", type=int, default=15, help="processes for the starts of every fit")
    parser.add_argument("--offset-shape", default="gaussian", choices=("gaussian", "student_t"))
    parser.add_argument("--early-detection", action="store_true", help="fit the rate of false early detections of the link model")
    parser.add_argument("--nu", type=float, default=fit_links.NU, help="Student t degrees of freedom of the packet core")
    parser.add_argument("--suffix", default="", help="appended to the output file names, for the fit of a model variant")
    parser.add_argument("--fold", type=int, default=None, choices=folds.FOLDS, help="fit the clusters of one fold of the confirmatory round alone")
    args = parser.parse_args()
    datasets = args.datasets.split(",")
    if set(datasets) - set(SEEDS):
        parser.error(f"the layout fits are those of {', '.join(SEEDS)}")
    model = dict(offset_shape=args.offset_shape, fit_early_detection=args.early_detection, nu=args.nu)
    for dataset in datasets:
        # every record of a fold's fit names the fold, the fits of the whole training halves name none
        entry = {} if args.fold is None else dict(fold=folds.record(dataset, args.fold))
        if dataset == "bregar":
            groups, source = bregar_groups(args.fold)
            environments = {}
            for environment, parts in sorted(groups.items()):
                environments[environment] = dict(fit_groups(parts, args.blocks, SEEDS[dataset] + int(environment[-1]), model, args.workers), **entry)
                result = environments[environment]
                print(dataset, environment, link_summary(result["four"]), result["blocks_per_track"], flush=True)
            common.write_json(common.FITS / f"bregar_layout_fits{args.suffix}{folds.name(args.fold)}.json", dict(environments=environments, identity=source))
        else:
            groups, source = tiers_groups(args.fold)
            result = dict(fit_groups(groups, args.blocks, SEEDS[dataset], model, args.workers), **entry)
            result["identity"] = source
            print(dataset, link_summary(result["four"]), result["blocks_per_track"], flush=True)
            common.write_json(common.FITS / f"{dataset}_layout_fit{args.suffix}{folds.name(args.fold)}.json", result)


if __name__ == "__main__":
    main()
