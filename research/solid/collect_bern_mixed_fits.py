"""Collect the fold fits of fit_bern_mixed_blocks.py into tracked artifacts.

Reads out/solid/unify/<fits directory>/held-NN.json for the eleven held scans and the all-scan fit (held 0) and
writes research/artifacts/solid/fits/<name> with the eleven folds keyed by scan and <all name> with the all-scan fit
under "0", the layout the candidate builders read. The twelve records must come from one likelihood and one model
setting, and every fit must be converged unless --allow-unconverged is given. With --all-only the all-scan fit alone is
collected, the one fit of the method's Bern link model, which a reproduction fits with fit_bern_mixed_blocks.py --held 0
without the eleven fold fits of the held-scan rounds.

Usage: collect_bern_mixed_fits.py [--fits fits-ml-nu3] --name bern_mixed_fold_fits_nu3_device.json --all bern_mixed_fit_nu3_device.json
       collect_bern_mixed_fits.py [--fits fits-ml-nu3] --all-only --all bern_mixed_fit_nu3_r0.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_bern  # noqa: E402
import fit_layout  # noqa: E402

SETTING_KEYS = ("likelihood", "nu", "offset_shape", "fit_early_detection", "pooling", "strata")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits", default="fits-ml-nu3", help="directory under out/solid/unify with the held-NN.json records")
    parser.add_argument("--name", help="artifact name of the eleven fold fits, required unless --all-only")
    parser.add_argument("--all", dest="all_name", required=True, help="artifact name of the all-scan fit")
    parser.add_argument("--all-only", action="store_true", help="collect the all-scan fit alone, without the eleven fold fits")
    parser.add_argument("--allow-unconverged", action="store_true", help="collect even when a fit did not converge")
    args = parser.parse_args()
    if not args.all_only and not args.name:
        parser.error("the eleven fold fits need --name, or collect the all-scan fit alone with --all-only")
    source = common.OUT / "unify" / args.fits
    folds = {}
    for scan in () if args.all_only else fit_bern.SCAN_IDS:
        record = common.read_json(source / f"held-{scan:02d}.json")
        if record["identity"]["held_scan"] != scan or scan in record["identity"]["fit_scans"]:
            raise ValueError(f"held-{scan:02d}.json is not the fold of scan {scan}")
        folds[str(scan)] = record
    whole = common.read_json(source / "held-00.json")
    if whole["identity"]["held_scan"] is not None or sorted(whole["identity"]["fit_scans"]) != sorted(fit_bern.SCAN_IDS):
        raise ValueError("held-00.json is not the all-scan fit")
    records = dict(folds, **{"0": whole})
    settings = {key: whole["identity"].get(key) for key in SETTING_KEYS}
    for scan, record in records.items():
        if {key: record["identity"].get(key) for key in SETTING_KEYS} != settings:
            raise ValueError(f"fold {scan} was fitted with other settings or another likelihood than the all-scan fit")
    unconverged = [scan for scan, record in records.items() if not record["four"]["converged"]]
    if unconverged and not args.allow_unconverged:
        raise ValueError(f"unconverged fits: {', '.join(unconverged)}")
    if not args.all_only:
        common.write_json(common.FITS / args.name, folds)
    common.write_json(common.FITS / args.all_name, {"0": whole})
    for scan, record in list(folds.items())[:3] + [("0", whole)]:
        print(scan, fit_layout.link_summary(record["four"]))
    if unconverged:
        print(f"collected with unconverged fits: {', '.join(unconverged)}")
    print(f"written fits/{args.all_name}" if args.all_only else f"written fits/{args.name} and fits/{args.all_name}")


if __name__ == "__main__":
    main()
