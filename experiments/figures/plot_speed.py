"""Draw the controlled speed figure of the paper from the timing summary of experiments/timing.

One column wide figure with a panel per mode of TWR-CLOUD: every method's median time of one solve as a dot and
the range up to its 95th percentile as a line, on a logarithmic axis, from the controlled run on an otherwise idle node
with one solve per physical core.

Usage: plot_speed.py --summary timing-bern.json --out DIR
       writes DIR/speed.pdf
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

import style


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    style.setup()
    runs = json.loads(args.summary.read_text())["runs"]
    modes = [mode for mode in ("AF", "AAC", "ANC") if any(r["mode"] == mode for r in runs)]
    counts = [sum(1 for r in runs if r["mode"] == mode) for mode in modes]
    fig, axes = plt.subplots(len(modes), 1, figsize=(style.COLUMN, 0.32 * sum(counts) + 0.5), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=counts))
    for ax, mode in zip(axes, modes):
        found = {r["method"]: r for r in runs if r["mode"] == mode}
        methods = [m for m in style.METHODS if m in found]
        for y, method in enumerate(reversed(methods)):
            run = found[method]
            ax.plot([run["median_ms"], run["p95_ms"]], [y, y], color=style.COLORS[method], linewidth=1.2, solid_capstyle="butt")
            ax.plot([run["median_ms"]], [y], marker="o", markersize=3.5, color=style.COLORS[method])
        ax.set_yticks(range(len(methods)), [style.LABELS[m] for m in reversed(methods)])
        ax.set_ylim(-0.6, len(methods) - 0.4)
        ax.set_xscale("log")
        ax.set_title(style.MODES[mode], pad=2, loc="left")
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.4)
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("Time of one solve (ms), median to 95th percentile")
    args.out.mkdir(parents=True, exist_ok=True)
    # no creation date, so that a rerun gives the same bytes
    fig.savefig(args.out / "speed.pdf", metadata={"CreationDate": None})
    print("wrote", args.out / "speed.pdf")


if __name__ == "__main__":
    main()
