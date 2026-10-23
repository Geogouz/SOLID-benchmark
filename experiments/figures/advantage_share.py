"""The exclusive advantage share of every method in every CDF panel of the paper, from the curves of export_cdfs.py.

A panel is a dataset, a mode it supports and a reading count, 24 in all, over the headline group (every cell in AF and
AAC, the scenarios with three or more anchors in ANC) and the joined pairs, as plot_cdfs.py draws it. With F_m^-1 the
balanced quantile function of the pair error of method m, the exclusive advantage of method m is

    A_m = integral over the levels v from 0 to 1 of max(0, min over the other methods of their quantile - F_m^-1(v)),

in cm, how far method m lies below the best other method, or zero where it does not, averaged over all levels. At each
level only the lowest method holds an advantage, so the advantages of all methods add up to the area between the lowest
and the second lowest quantile curve, and the exclusive advantage share S_m = 100 A_m / sum of A is the part of it that
method m holds (the author's score of 4 October 2026, which Table II gives for SOLID). The share weighs how often a
method leads by how far it leads, so each panel also records, per method, the share of levels where it lies lowest and
its mean lead there. Every integral is exact for the curves drawn as straight lines between the 1001 exported levels,
since each step between two levels is split wherever two curves cross, so the order of the methods stays fixed within
each piece.

Usage: advantage_share.py [--curves research/artifacts/solid/figures/data] [--out FILE]
       prints SOLID's share per panel and, with --out, writes every panel as JSON
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PANELS = (("bern", "AF"), ("bern", "AAC"), ("bern", "ANC"), ("hymn", "ANC"), ("aau", "ANC"), ("tiers", "AF"),
          ("tiers", "AAC"), ("tiers", "ANC"), ("bregar", "ANC"), ("idlab", "AF"), ("idlab", "AAC"), ("idlab", "ANC"))
TRACKS = (1, 10)
HEADLINE = {"AF": "every cell", "AAC": "every cell", "ANC": "headline, 3 or more anchors"}


def curves(folder, dataset, mode, track):
    record = json.loads((folder / f"cdfs-{dataset}.json").read_text())
    levels = [round(v, 3) for v in record["levels"]]
    return levels, {c["method"]: c["quantiles_cm"] for c in record["curves"]
                    if c["mode"] == mode and c["track"] == track and c["group"] == HEADLINE[mode] and c["pair_set"] == "joined"}


def pieces(levels, c):
    """The pieces between consecutive levels, split where two curves cross, as (v0, v1, values at v0, values at v1)."""
    methods = list(c)
    for k in range(len(levels) - 1):
        cuts = {0.0, 1.0}
        for a in range(len(methods)):
            for b in range(a + 1, len(methods)):
                d0 = c[methods[a]][k] - c[methods[b]][k]
                d1 = c[methods[a]][k + 1] - c[methods[b]][k + 1]
                if d0 * d1 < 0:
                    cuts.add(d0 / (d0 - d1))
        cuts = sorted(cuts)
        at = lambda t: {m: c[m][k] + t * (c[m][k + 1] - c[m][k]) for m in methods}
        for t0, t1 in zip(cuts, cuts[1:]):
            step = levels[k + 1] - levels[k]
            yield levels[k] + t0 * step, levels[k] + t1 * step, at(t0), at(t1)


def panel(levels, c):
    methods = list(c)
    areas, width = dict.fromkeys(methods, 0.0), dict.fromkeys(methods, 0.0)
    for v0, v1, q0, q1 in pieces(levels, c):
        for m in methods:
            g0 = max(0.0, min(q0[j] for j in methods if j != m) - q0[m])
            g1 = max(0.0, min(q1[j] for j in methods if j != m) - q1[m])
            areas[m] += (v1 - v0) * (g0 + g1) / 2
            if g0 + g1 > 0:
                width[m] += v1 - v0
    total = sum(areas.values())
    return {m: dict(area_cm=round(areas[m], 3), share=round(100 * areas[m] / total, 2) if total > 0 else None,
                    lowest_levels=round(100 * width[m], 1),
                    mean_lead_cm=round(areas[m] / width[m], 2) if width[m] > 0 else 0.0) for m in methods}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curves", type=Path, default=ROOT / "research/artifacts/solid/figures/data")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {}
    for dataset, mode in PANELS:
        for track in TRACKS:
            levels, c = curves(args.curves, dataset, mode, track)
            result[f"{dataset} {mode} {track}"] = dict(methods=panel(levels, c))
    for name, p in result.items():
        s = p["methods"]["solid"]
        print(f"{name:16s} share of SOLID {s['share']:6.1f}%, lowest at {s['lowest_levels']:5.1f}% of levels by {s['mean_lead_cm']:6.2f} cm")
    if args.out:
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print("wrote", args.out)


if __name__ == "__main__":
    main()
