"""Link model fit on the reserved IDLab training partition, with the frozen affine correction applied.

Two tracks enter as strata with the same total weight. The ten sample track holds blocks of ten consecutive valid rows
per link and radio setting, at most four per link setting. The one sample track holds the first valid row of every
such block. Within a track every link setting has the same total weight, and the weights sum to the block count.

With --fold the fit keeps the blocks of the link settings whose radio configuration lies in one fold of the
confirmatory round, research/solid/folds.py, the blocks the fit of the whole partition draws for them, and its record
holds the fold's entry. The round positions the units of the other fold's configurations with it.

With --first-stage the script fits no link model. It fits the cooperative first stage of IDLab, the multipliers of the
agent only and the anchor aided modes that idlab_training.range_profiles fits on every reserved row, on the rows of each
fold of the round alone, by the same procedure, and writes them to fits/idlab_first_stage.json. The affine correction
and the packet scale stay those of the whole partition, the calibration on the same layout that every method receives.
The same procedure on every reserved row must give back the multipliers of idlab-training.json, and the round positions
the units of each fold with the first stage of the other.

Usage: fit_idlab.py [--workers 15] [--offset-shape gaussian] [--early-detection] [--nu 3] [--suffix S] [--fold 0|1]
       writes fits/idlab_link_fit<S>.json, the suffix empty by default, and with --fold K the file ends in _foldK
       before .json
       fit_idlab.py --first-stage
       writes fits/idlab_first_stage.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_layout  # noqa: E402
import fit_links  # noqa: E402
import folds  # noqa: E402
import aac_training  # noqa: E402
import calibration  # noqa: E402
import idlab_admission  # noqa: E402
import idlab_split  # noqa: E402
import idlab_training  # noqa: E402
import pilot  # noqa: E402

BLOCK = 10
# the cooperative first stage of IDLab fitted on the reserved rows of each fold of the round alone
FIRST_STAGE = "idlab_first_stage.json"
FIRST_STAGE_SCHEMA = "solid.fit.idlab-first-stage.v1"
FIRST_STAGE_MODES = ("af", "aac")
# the relative difference within which the same procedure on every reserved row gives back the training fit's multipliers
FIRST_STAGE_TOLERANCE = 1e-6


def training_blocks(cells, correction, scale):
    """The ten sample blocks and their first rows as one sample blocks, with the track, the relative weight within the
    track (equal total per link setting) and the link setting of every block."""
    rng = np.random.default_rng(5)
    blocks, strata, within, tagged = [], [], [], []
    for key, rows in cells.items():
        rows = rows[np.argsort(rows[:, 0])]
        corrected, distances = calibration.corrected(rows[:, 1], correction), rows[:, 2]
        starts = list(range(0, len(rows) - BLOCK + 1, BLOCK))
        if not starts:
            continue
        picked = rng.choice(starts, min(len(starts), max(1, 40 // BLOCK)), replace=False)
        for start in picked:
            for track, size in ((BLOCK, BLOCK), (1, 1)):
                blocks.append((np.asarray(corrected[start:start + size], dtype=float), np.full(size, scale), float(distances[start])))
                strata.append(track)
                within.append(1.0 / len(picked))
                tagged.append(key)
    return blocks, strata, np.asarray(within), tagged


def configuration_of(key):
    """The radio configuration of a link setting of the reserved rows, keyed (node, peer, settings) as
    idlab_training.load_training keys it."""
    import static_preflight
    return folds.idlab_configuration(dict(zip(static_preflight.SETTINGS, key[2:])))


def fold_blocks(blocks, strata, within, tagged, fold):
    """The blocks of training_blocks whose link setting lies in a configuration of one fold of the round, with their
    tracks, relative weights and link settings. Every link setting keeps the blocks and the weight within its track
    that the fit of the whole partition gives it."""
    kept = folds.idlab_configurations(fold)
    keep = [index for index, key in enumerate(tagged) if configuration_of(key) in kept]
    return [blocks[i] for i in keep], [strata[i] for i in keep], np.asarray(within)[keep], [tagged[i] for i in keep]


def first_stage(cells, correction, scale):
    """The cooperative first stage fitted on the rows of the given link settings, as idlab_training.range_profiles fits
    it on every reserved row: the asymmetric shape of the corrected residuals over the packet scale for the agent only
    mode and over the anchor aided scale law for the anchor aided mode, every link setting with the same total weight
    and every row of it the same share. The affine correction and the packet scale are given."""
    measured, reference, weights = calibration.pooled(cells)
    residual = calibration.corrected(measured, correction) - reference
    return dict(af=calibration.fit_shape(residual, scale, weights), aac=calibration.fit_shape(residual, aac_training.aac_scale(reference), weights))


def relative_difference(first, second):
    """The largest relative difference between the two multipliers of two first stages."""
    return max(abs(first[key] - second[key]) / abs(second[key]) for key in ("positive_scale", "negative_scale"))


def first_stages(cells, training):
    """The record of fits/idlab_first_stage.json from the reserved rows and the IDLab training fit: the first stage of
    every reserved row, which must give back the training fit's multipliers within FIRST_STAGE_TOLERANCE, so the
    procedure is the training fit's, and the first stage of each fold of the round on the rows of its configurations
    alone, all with the affine correction and the packet scale of the whole partition."""
    correction, scale = training["correction"], training["profiles"]["scale_cm"]
    whole = first_stage(cells, correction, scale)
    recorded = {mode: training["profiles"][f"{mode}_range_error"] for mode in FIRST_STAGE_MODES}
    differences = {mode: relative_difference(whole[mode], recorded[mode]) for mode in FIRST_STAGE_MODES}
    if max(differences.values()) > FIRST_STAGE_TOLERANCE:
        raise ValueError(f"the first stage of every reserved row differs from the training fit's by {differences}, so the procedure is not the "
                         "training fit's")
    parts = {}
    for fold in folds.FOLDS:
        kept = folds.idlab_configurations(fold)
        rows = {key: values for key, values in cells.items() if configuration_of(key) in kept}
        parts[str(fold)] = dict(first_stage(rows, correction, scale), fold=folds.record("idlab", fold), link_settings=len(rows),
                                rows=int(sum(len(values) for values in rows.values())))
    return dict(schema=FIRST_STAGE_SCHEMA, correction=correction, scale_cm=scale, whole=dict(whole, differences=differences), folds=parts,
                rule=("The multipliers of the agent only and the anchor aided first stage of idlab_training.range_profiles, fitted on the "
                      "reserved rows of the configurations of each fold of the round alone, with the affine correction and the packet "
                      "scale of every reserved row. The round positions the units of each fold with the first stage of the other."))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=15, help="processes for the starts of every fit")
    parser.add_argument("--offset-shape", default="gaussian", choices=("gaussian", "student_t"))
    parser.add_argument("--early-detection", action="store_true", help="fit the rate of false early detections of the link model")
    parser.add_argument("--nu", type=float, default=fit_links.NU, help="Student t degrees of freedom of the packet core")
    parser.add_argument("--suffix", default="", help="appended to the output file name, for the fit of a model variant")
    parser.add_argument("--fold", type=int, default=None, choices=folds.FOLDS, help="fit the configurations of one fold of the confirmatory round alone")
    parser.add_argument("--first-stage", action="store_true", help="fit the cooperative first stage on each fold of the round, no link model")
    args = parser.parse_args()
    if args.first_stage and (args.fold is not None or args.suffix):
        parser.error("--first-stage fits every fold at once and belongs to no model variant")
    model = dict(offset_shape=args.offset_shape, fit_early_detection=args.early_detection, nu=args.nu)
    training = common.read_json(common.EXTERNAL / "idlab-training.json")
    correction, scale = training["correction"], training["profiles"]["scale_cm"]
    split = pilot.evidence.read(idlab_split.OUTPUT)
    surveyed = np.loadtxt(idlab_admission.FOLDER / "groundtruth9.csv", delimiter=",")
    coordinates = {int(row[0]): row[1:] for row in surveyed}
    cells = idlab_training.load_training(split, idlab_admission.FOLDER, coordinates)
    print(f"{len(cells)} link settings, {sum(map(len, cells.values()))} reserved ranges, scale {scale:.2f} cm", flush=True)
    if args.first_stage:
        record = dict(first_stages(cells, training), identity=dict(training_sha256=common.sha256(common.EXTERNAL / "idlab-training.json"),
                                                                    split_sha256=common.sha256(idlab_split.OUTPUT), evaluation_rows_used=0))
        for part, entry in [("every reserved row", record["whole"])] + [(f"fold {fold}", record["folds"][str(fold)]) for fold in folds.FOLDS]:
            print(part, {mode: {key: round(value, 6) for key, value in entry[mode].items()} for mode in FIRST_STAGE_MODES}, flush=True)
        common.write_json(common.FITS / FIRST_STAGE, record)
        return
    blocks, strata, within, tagged = training_blocks(cells, correction, scale)
    if args.fold is not None:
        blocks, strata, within, tagged = fold_blocks(blocks, strata, within, tagged, args.fold)
        print(f"fold {args.fold}: {len(set(tagged))} link settings of {len(folds.idlab_configurations(args.fold))} configurations", flush=True)
    weights = fit_links.stratified_weights(strata, within)
    fit = fit_links.fit_link_model(blocks, weights, workers=args.workers, **model)
    print(f"{len(blocks)} blocks", fit_layout.link_summary(fit), flush=True)
    counts = {str(track): strata.count(track) for track in sorted(set(strata))}
    b10 = dict(four=fit, blocks=len(blocks), blocks_per_track=counts, weight_per_track=fit_links.stratum_totals(strata, weights))
    if args.fold is not None:
        b10["fold"] = folds.record("idlab", args.fold)
    common.write_json(common.FITS / f"idlab_link_fit{args.suffix}{folds.name(args.fold)}.json", dict(b10=b10, scale_cm=scale, correction=correction))


if __name__ == "__main__":
    main()
