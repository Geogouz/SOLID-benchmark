"""Draw the paper's balanced CDF figures from the curves export_cdfs.py wrote.

Every panel holds the balanced CDF of every method of one mode and sample track, over the headline group of the final
tables (every cell in the agent only and anchor aided modes, the units with three or more anchors in the anchor only
mode) and the joined pairs, the headline pair set. The light guides mark the tables' levels 0.5, 0.75 and 0.9, so each
curve crosses them at the S50, S75 and S90 of its table.

- bern-cdfs.pdf, page wide: TWR-CLOUD, the three modes by the two sample tracks.
- annex-hymn-aau-cdfs.pdf, column wide: HYMN and AAU Meshed UWB, the anchor only mode by the two sample tracks.
- annex-tiers-cdfs.pdf and annex-idlab-cdfs.pdf, page wide, and annex-bregar-cdfs.pdf, column wide: the other datasets of the
  companion document.

Usage: plot_cdfs.py --data DIR --out DIR
       DIR holds cdfs-<dataset>.json of every dataset, the figures go to OUT as vector PDF
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

import style

HEADLINE = {"AF": "every cell", "AAC": "every cell", "ANC": "headline, 3 or more anchors"}
TRACKS = (1, 10)
PAIR_SET = "joined"
NAMES = dict(bern="TWR-CLOUD", tiers="TIERS", bregar="Bregar", idlab="IDLab", hymn="HYMN", aau="AAU Meshed UWB")


def track_label(dataset, track):
    """The nominal reading count of a track, which Section III counts per direction on TWR-CLOUD and per link elsewhere.
    The label leaves that unit to the text, since a longer label runs past the height of its row."""
    return f"{track} reading{'' if track == 1 else 's'}"


def headline(record, mode, track):
    """The curves of one mode and track of the headline group and pair set, {method: curve}."""
    return {c["method"]: c for c in record["curves"]
            if c["mode"] == mode and c["track"] == track and c["group"] == HEADLINE[mode] and c["pair_set"] == PAIR_SET}


def panel(ax, record, mode, track, levels, xmax=None):
    curves = headline(record, mode, track)
    style.guides(ax)
    for method in style.METHODS:
        if method in curves:
            style.line(ax, method, curves[method]["quantiles_cm"], levels)
    top = xmax or style.nice_ceiling(max(c["quantiles_cm"][levels.index(0.95)] for c in curves.values()))
    ax.set_xlim(0, top)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.25))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], prune="upper"))
    return curves


def grid(records, columns, out, width, height_per_row=1.0):
    """A figure of one column per (dataset, mode) and one row per sample track."""
    fig, axes = plt.subplots(len(TRACKS), len(columns), figsize=(width, height_per_row * len(TRACKS) + 0.45), squeeze=False,
                             sharey=True, constrained_layout=True)
    present = set()
    for c, (dataset, mode) in enumerate(columns):
        record = records[dataset]
        levels = [round(v, 3) for v in record["levels"]]
        for r, track in enumerate(TRACKS):
            ax = axes[r][c]
            present |= set(panel(ax, record, mode, track, levels))
            if r == 0:
                title = style.MODES[mode] if len({d for d, _ in columns}) == 1 else f"{NAMES[dataset]}, {mode}"
                ax.set_title(title, pad=3)
            if c == 0:
                ax.set_ylabel(f"{track_label(dataset, track)}\nBalanced CDF")
            if r == len(TRACKS) - 1:
                ax.set_xlabel("Pair error (cm)")
    handles = [plt.Line2D([], [], color=style.COLORS[m], linestyle=style.DASHES[m], linewidth=style.WIDTHS.get(m, 0.9), label=style.LABELS[m])
               for m in style.METHODS if m in present]
    # one row of every method across a page, and rows of three or two methods on narrower figures, so that the legend
    # never widens a figure beyond its width
    columns_of_legend = len(handles) if width >= style.PAGE else 3 if width >= style.COLUMN else 2
    fig.legend(handles=handles, loc="outside upper center", ncol=columns_of_legend)
    # no creation date, so that a rerun gives the same bytes
    fig.savefig(out, metadata={"CreationDate": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    style.setup()
    args.out.mkdir(parents=True, exist_ok=True)
    records = {path.stem.removeprefix("cdfs-"): json.loads(path.read_text()) for path in sorted(args.data.glob("cdfs-*.json"))}
    figures = [("bern-cdfs.pdf", [("bern", m) for m in ("AF", "AAC", "ANC")], style.PAGE),
               ("annex-hymn-aau-cdfs.pdf", [("hymn", "ANC"), ("aau", "ANC")], style.COLUMN),
               ("annex-tiers-cdfs.pdf", [("tiers", m) for m in ("AF", "AAC", "ANC")], style.PAGE),
               ("annex-bregar-cdfs.pdf", [("bregar", "ANC")], style.COLUMN * 0.6),
               ("annex-idlab-cdfs.pdf", [("idlab", m) for m in ("AF", "AAC", "ANC")], style.PAGE)]
    for name, columns, width in figures:
        if all(dataset in records for dataset, _ in columns):
            grid(records, columns, args.out / name, width)
            print("wrote", args.out / name)
        else:
            print("skipped", name, "without", sorted({d for d, _ in columns} - set(records)))


if __name__ == "__main__":
    main()
