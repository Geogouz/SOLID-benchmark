"""Fit the block link model on mixed Bern training blocks, the link model of the three Bern modes.

The blocks come from the generated training scenarios (members scan-XX/N.json of the cooperative held-scan archive),
which hold agent to agent and agent to anchor links. From a deterministic sample of each training scan's scenarios
the fit draws the same number of blocks of each link type and track (one and ten samples), --blocks-per-kind each,
with the production calibration or the fold's. Training scans only. Every scan samples its scenarios from a random
stream of its own, seeded by [seed, scan], so a scan contributes the same scenarios to every fold that trains on it,
and the picks of each link type and track take one more stream, [seed, 0], scan 0 being no scan.

Agent links pool both directions. The two directions of an agent pair measure the same distance between the same two
devices within the same iteration, so their persistent errors are shared, and two blocks would count that shared
error twice. Every unordered agent pair therefore forms one block per track that holds the packets of both
directions, each packet with its own scale, the way the engine's cooperative block model pools them. A one sample
block of a pair so holds up to two packets. Anchor links stay one block per link.

Strata are the tracks. Every track receives the same total weight and the weights sum to the block count, which with
the same number of blocks per link type and track makes every block weigh one.

With --held N the fit excludes training scan N and uses that fold's calibration from the campaign's fold records,
the same nesting as the anchor only selection. --held 0 fits all eleven scans with the production calibration.

Usage: fit_bern_mixed_blocks.py [--held 0] [--per-scan 40] [--blocks-per-kind 600] [--seed 11] [--nu 3] [--workers 15]
                                [--offset-shape gaussian] [--early-detection] [--archive PATH] [--fits-dir NAME]

The record is written to out/solid/unify/<fits dir>/held-NN.json, by default fits-ml-nu<nu> with -early and -student
appended for those settings, so the fits of the earlier fitter under fits and fits-nu3 stay untouched.
"""
import argparse
import json
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import fit_bern  # noqa: E402
import fit_layout  # noqa: E402
import fit_links  # noqa: E402

OUT = common.OUT / "unify"
ARCHIVE = OUT / "bern-training-scenarios-v2.zip"      # the held-scan corpora built by the evaluation release's rules
TRACKS = ("1", "10")
KINDS = (("agents", "1"), ("agents", "10"), ("anchors", "1"), ("anchors", "10"))
POOLING = ("both directions of an agent pair in one block per unordered pair and track, every packet with its own scale, "
           "and one block per anchor link and track")


def fits_directory(nu, offset_shape="gaussian", early_detection=False):
    return f"fits-ml-nu{nu:g}" + ("-early" if early_detection else "") + ("-student" if offset_shape == "student_t" else "")


def pooled_agent_blocks(iteration, agents, coefficients):
    """(track, left, right, block) for every unordered agent pair of an iteration with packets on that track, the
    packets of both directions in one block, the direction from the lower to the higher identifier first."""
    field = iteration["Ranging samples between Agents"]
    pairs = defaultdict(lambda: {track: [] for track in TRACKS})
    for key in sorted(field, key=lambda k: tuple(map(int, k.split("-")))):
        left, right = map(int, key.split("-"))
        for track in TRACKS:
            pairs[(min(left, right), max(left, right))][track].extend(fit_bern.calibrated_packets((field[key] or {}).get(track), coefficients))
    for (left, right), tracks in sorted(pairs.items()):
        distance = float(np.linalg.norm((agents[left] - agents[right])[:2]))
        for track in TRACKS:
            packets = tracks[track]
            if packets:
                yield track, left, right, (np.array([p["distance_cm"] for p in packets]), np.array([p["scale_cm"] for p in packets]), distance)


def scan_stream(seed, scan):
    """The random stream of one scan's sample of scenarios, the same in every fold. Scan 0 gives the stream of the
    picks of each link type and track."""
    return np.random.default_rng([seed, scan])


def blocks_from_scan(scan, rng, scenarios, coefficients, archive_path):
    """Blocks per (link type, track) from a deterministic sample of one scan's training scenarios, the agent pairs
    pooled over both directions."""
    blocks = {kind: [] for kind in KINDS}
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(n for n in archive.namelist() if n.startswith(f"scan-{scan:02d}/") and n.endswith(".json"))
        for name in rng.choice(names, min(scenarios, len(names)), replace=False):
            scenario = json.loads(archive.read(name))
            fit_bern.check_scenario(scenario, scan)
            anchors = {int(k): np.array([v["x"], v["y"], v["z"]]) for k, v in (scenario["True Positions of Anchors"] or {}).items()}
            agents = {int(k): np.array([v["x"], v["y"], v["z"]]) for k, v in scenario["True Positions of Agents"].items()}
            iteration = scenario["Iterations"][0]
            for track, _, _, item in pooled_agent_blocks(iteration, agents, coefficients):
                blocks[("agents", track)].append(item)
            for key, value in iteration["Ranging samples to Anchors"].items():
                left, right = map(int, key.split("-"))
                for track in TRACKS:
                    packets = fit_bern.calibrated_packets((value or {}).get(track), coefficients)
                    if not packets:
                        continue
                    distance = float(np.linalg.norm(agents[left] - anchors[right]))
                    blocks[("anchors", track)].append((np.array([p["distance_cm"] for p in packets]), np.array([p["scale_cm"] for p in packets]), distance))
    return blocks


def collect(held, per_scan=40, blocks_per_kind=600, seed=11, archive=ARCHIVE, verbose=False):
    """The chosen blocks of one fold exactly as main draws them and the track of every chosen block."""
    coefficients, _ = fit_bern.calibration(held)
    training = [scan for scan in fit_bern.SCAN_IDS if scan != held]
    pool = {kind: [] for kind in KINDS}
    for scan in training:
        blocks = blocks_from_scan(scan, scan_stream(seed, scan), per_scan, coefficients, archive)
        for kind in KINDS:
            pool[kind].extend(blocks[kind])
        if verbose:
            print(f"scan {scan}: " + ", ".join(f"{k[0]}/{k[1]} {len(v)}" for k, v in pool.items()), flush=True)
    rng = scan_stream(seed, 0)
    picks = {kind: rng.choice(len(pool[kind]), min(blocks_per_kind, len(pool[kind])), replace=False) for kind in KINDS}
    chosen = [pool[kind][i] for kind in KINDS for i in picks[kind]]
    strata = [kind[1] for kind in KINDS for _ in picks[kind]]
    return chosen, strata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-scan", type=int, default=40)
    parser.add_argument("--blocks-per-kind", type=int, default=600)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--held", type=int, default=0, help="held training scan 1 to 11, 0 fits all eleven")
    parser.add_argument("--nu", type=float, default=3.0, help="Student t degrees of freedom of the packet core, 3 in the method")
    parser.add_argument("--workers", type=int, default=15, help="processes for the starts of every fit")
    parser.add_argument("--offset-shape", default="gaussian", choices=("gaussian", "student_t"))
    parser.add_argument("--early-detection", action="store_true", help="fit the rate of false early detections of the link model")
    parser.add_argument("--fits-dir", default=None, help="directory under out/solid/unify, by default from the model settings")
    args = parser.parse_args()
    if args.held not in range(12):
        raise ValueError("held must be 0 or a training scan from 1 to 11")
    coefficients, calibration_path = fit_bern.calibration(args.held)
    training = [scan for scan in fit_bern.SCAN_IDS if scan != args.held]
    model = dict(offset_shape=args.offset_shape, fit_early_detection=args.early_detection, nu=args.nu)
    output = OUT / (args.fits_dir or fits_directory(args.nu, args.offset_shape, args.early_detection)) / f"held-{args.held:02d}.json"
    manifest = output.with_suffix(".manifest.json")
    identity = dict(held_scan=args.held or None, fit_scans=training, per_scan=args.per_scan, blocks_per_kind=args.blocks_per_kind,
                    seed=args.seed, streams="one per scan, [seed, scan], and [seed, 0] for the picks", nu=args.nu,
                    offset_shape=args.offset_shape, fit_early_detection=args.early_detection, pooling=POOLING, strata="tracks",
                    calibration=coefficients, calibration_sha256=common.sha256(calibration_path) if calibration_path else None,
                    archive_sha256=common.sha256(args.archive),
                    kinds=[f"{k[0]}/{k[1]}" for k in KINDS], evaluation_rows_used=0, likelihood=fit_links.likelihood_identity(),
                    source={p.name: common.sha256(p) for p in (Path(__file__), Path(fit_links.__file__), Path(fit_bern.__file__), Path(fit_layout.__file__))})
    if common.cache_matches(manifest, identity, [output]):
        print(f"held {args.held}: cached fit {fit_layout.link_summary(common.read_json(output)['four'])}", flush=True)
        return
    chosen, strata = collect(args.held, args.per_scan, args.blocks_per_kind, args.seed, args.archive, verbose=True)
    weights = fit_links.stratified_weights(strata)
    started = time.time()
    fit = fit_links.fit_link_model(chosen, weights, workers=args.workers, **model)
    residuals = np.concatenate([block[0] - block[2] for block in chosen])
    counts = {track: strata.count(track) for track in TRACKS}
    result = dict(identity=identity, four=fit, blocks=len(chosen), blocks_per_track=counts, weight_per_track=fit_links.stratum_totals(strata, weights),
                  blocks_per_length=fit_layout.packet_counts(chosen), residual_quantiles=np.percentile(residuals, [5, 25, 50, 75, 95]).round(1).tolist())
    result["seconds"] = time.time() - started
    common.write_json(output, result)
    common.seal_cache(manifest, identity, [output])
    print(f"held {args.held}: {fit_layout.link_summary(fit)}", flush=True)


if __name__ == "__main__":
    main()
