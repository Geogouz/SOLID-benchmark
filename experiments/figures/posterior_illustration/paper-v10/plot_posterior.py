"""Draw the adaptive posterior of one agent of an anchor only unit from the record of the posterior illustration.

The record comes from the engine test of illustration.rs, which run_illustration.sh runs. It holds every
quadtree node the certified refinement decided about, for an agent whose posterior the frozen engine keeps bitwise.

Panel (a) shows the whole operating area. The nodes the refinement dropped are light rectangles shaded by the level at
which they were dropped, coarse far away and fine near the modes, and the cells it kept are one teal region. Panel (b)
zooms on the kept region, where every kept leaf is shaded by its log posterior less the best value seen. The scale runs
from minus delta, the drop threshold with delta = ln(N / 1e-4) and N the grid's cells, up to zero. A single cell carries
its exact value and a coarse leaf its upper bound, which lies within one nat of every cell in it. Filled triangles mark the anchors the agent ranged, open triangles the
anchors it failed to reach, the cross its true position and the circle SOLID's estimate. Coordinates are metres from the
minimum corner of the operating area. Every node is a vector rectangle, and a thin white gap separates the nodes that are
large enough on paper to show one. Nodes too small to see in panel (a) merge into row runs of equal level, which keeps
the PDF small, and panel (b) draws the nodes it shows one by one.

The caption's numbers go to the standard output and, with --caption, to a file.

Usage: plot_posterior.py [--record RECORD] [--agent ID] [--output PDF] [--width column|page] [--zoom X0 X1 Y0 Y1]
                         [--caption TXT]
       RECORD defaults to research/artifacts/solid/figures/data/posterior-illustration.json and PDF to
       research/artifacts/solid/figures/posterior.pdf. Without --agent the first informative agent with decisions.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import style  # noqa: E402  (sets the PDF backend before pyplot loads)

import matplotlib.colorbar  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

# A colorbar of 50 or more colors rasterizes its solids by default, which embeds an image in the PDF. The paper's
# figures are vector only, so every colorbar keeps its solids as vector patches.
matplotlib.colorbar.Colorbar.n_rasterize = float("inf")

RECORD = ROOT / "research/artifacts/solid/figures/data/posterior-illustration.json"
OUTPUT = ROOT / "research/artifacts/solid/figures/posterior.pdf"
# the fate codes of the record
SPLIT, DROPPED, GUARDED, CELL, COARSE, FINAL = range(6)
DROPS = (DROPPED, GUARDED, FINAL)
INK, SURFACE = "#0b0b0b", "#ffffff"
# the log posterior of the kept leaves, one hue around SOLID's teal from light to dark, and the kept region of panel (a)
TEAL = LinearSegmentedColormap.from_list("kept", ["#b9e6de", "#6ccbbb", "#1fa898", "#007f73", "#1f5e56", "#27433f"])
KEPT = style.COLORS["solid"]
# neutral grays by drop level, even steps of OKLab lightness from the coarsest level, lightest, to single cells. The
# size of a node shows its level exactly, so the shade needs only show the way from coarse to fine and stays light.
GRAY_COARSE, GRAY_FINE = 0.97, 0.60
# a node narrower than this many points in panel (a) merges into row runs without its gap
VISIBLE_POINTS = 2.2
# the surface gaps between nodes and around the coarse kept leaves, in points
GAP, LEAF_GAP = 0.3, 0.2


def gray(lightness):
    """The neutral gray of an OKLab lightness, whose linear luminance is its cube."""
    y = min(max(lightness, 0.0), 1.0) ** 3
    v = 12.92 * y if y <= 0.0031308 else 1.055 * y ** (1 / 2.4) - 0.055
    return (v, v, v)


def numbers(column):
    """A column of bounds, whose values that are not finite the record writes as strings."""
    return np.array([float(value) for value in column], dtype=float)


class Grid:
    """The cell centres of the record's grid and the conversion of node indices to metres from the area's corner."""

    def __init__(self, record):
        grid, area = record["grid"], record["area"]
        self.xs, self.ys = np.asarray(grid["xs"], dtype=float), np.asarray(grid["ys"], dtype=float)
        self.step, self.nx, self.ny = float(grid["step"]), int(grid["nx"]), int(grid["ny"])
        self.origin = (area["minimum_x_cm"], area["minimum_y_cm"])
        self.size = ((area["maximum_x_cm"] - area["minimum_x_cm"]) / 100.0, (area["maximum_y_cm"] - area["minimum_y_cm"]) / 100.0)

    def point(self, x_cm, y_cm):
        return (x_cm - self.origin[0]) / 100.0, (y_cm - self.origin[1]) / 100.0

    def span(self, i0, i1, j0, j1):
        """The rectangles in metres of the cells i0 to i1 by j0 to j1, arrays of first and last indices."""
        half = self.step / 2.0
        x0 = (self.xs[i0] - half - self.origin[0]) / 100.0
        x1 = (self.xs[i1] + half - self.origin[0]) / 100.0
        y0 = (self.ys[j0] - half - self.origin[1]) / 100.0
        y1 = (self.ys[j1] + half - self.origin[1]) / 100.0
        return x0, x1, y0, y1

    def nodes(self, level, i, j):
        """The last cell of every node along each axis, a node of level k covering 2^k cells clipped to the grid."""
        side = np.left_shift(1, level.astype(np.int64))
        return np.minimum(i + side, self.nx) - 1, np.minimum(j + side, self.ny) - 1


def quads(x0, x1, y0, y1):
    return np.stack([np.stack([x0, y0], -1), np.stack([x1, y0], -1), np.stack([x1, y1], -1), np.stack([x0, y1], -1)], 1)


def runs(cells, grid):
    """Row runs of equal nonnegative labels of a cell map, as the rectangles in metres and their labels."""
    x0, x1, y0, y1, labels = [], [], [], [], []
    for j in np.flatnonzero((cells >= 0).any(axis=1)):
        row = cells[j]
        edges = np.flatnonzero(np.diff(np.concatenate([[-2], row, [-2]])) != 0)
        for start, stop in zip(edges[:-1], edges[1:]):
            if row[start] >= 0:
                a, b, c, d = grid.span(start, stop - 1, j, j)
                x0.append(a), x1.append(b), y0.append(c), y1.append(d), labels.append(row[start])
    return quads(np.array(x0), np.array(x1), np.array(y0), np.array(y1)), np.array(labels)


def choose(record, wanted):
    """The agent to draw, the one of identifier `wanted` or the first informative agent with its decisions."""
    for agent in record["agents"]:
        eligible = agent["informative"] and "decisions" in agent
        if (wanted is None and eligible) or agent["agent_id"] == wanted:
            if not eligible:
                raise SystemExit(f"agent {wanted} has no recorded decisions, run the test without ILLUSTRATION_AGENT or with it")
            return agent
    raise SystemExit(f"no agent {wanted} with recorded decisions in the record" if wanted is not None else "no informative agent with decisions")


def caption(record, agent):
    unit = record.get("unit") or {}
    where = f"scenario {unit['scenario']}, {unit['samples']} readings per direction, iteration {unit['iteration']}" if unit else "the unit"
    law = agent.get("law")
    reach = f", presence reach {law['reach'] / 100:.2f} m" if law else ", no presence law"
    share = float(agent["dropped_share"])
    epsilon = record["constants"]["epsilon"]
    bound = (f". The node guard cut this posterior, so the bound of {epsilon:g} does not hold for it." if agent["guarded"]
             else f", at most {epsilon:g}.")
    return (f"Agent {agent['agent_id']} of {where}{reach}. The grid holds {agent['grid_cells']:,} cells. The refinement made "
            f"{agent['cells']:,} exact cell evaluations and bounded {agent['nodes']:,} nodes, and it kept {agent['kept_cells']:,} "
            f"cells, {agent['single_cells']:,} single cells and {agent['coarse_cells']:,} cells in {agent['coarse_leaves']:,} "
            f"coarse leaves. Delta is {float(agent['delta']):.1f} nats. The certified share of the posterior mass left out is "
            f"{share:.2e}{bound}")


def zoom_window(grid, kept, args):
    """The zoom of panel (b), the kept leaves with a margin, square where the area allows."""
    if args.zoom:
        return tuple(args.zoom)
    if not len(kept):
        x, y = grid.point(*args.estimate)
        kept = np.array([[[x - 1.0, y - 1.0], [x + 1.0, y + 1.0]]])
    x0, x1, y0, y1 = kept[..., 0].min(), kept[..., 0].max(), kept[..., 1].min(), kept[..., 1].max()
    side = max(x1 - x0, y1 - y0)
    side += max(0.5 * side, 1.0)
    # square unless the area is narrower, each side at most the area's
    width, height = min(side, grid.size[0]), min(side, grid.size[1])
    cx = min(max((x0 + x1) / 2, width / 2), grid.size[0] - width / 2)
    cy = min(max((y0 + y1) / 2, height / 2), grid.size[1] - height / 2)
    return cx - width / 2, cx + width / 2, cy - height / 2, cy + height / 2


def idle(record, agent):
    """The anchors the agent neither ranged nor failed to reach, those no agent of the unit exchanged with."""
    attempted = {(round(a[0], 6), round(a[1], 6)) for a in agent["present"] + agent["missing"]}
    return [a for a in record["anchors"] if (round(a[1], 6), round(a[2], 6)) not in attempted]


def markers(ax, record, agent, grid, size):
    """The anchors, the true position and the estimate, with a surface ring so they stay legible over the nodes."""
    ring = [patheffects.withStroke(linewidth=1.6, foreground=SURFACE)]
    ranged = {(round(a[0], 6), round(a[1], 6)) for a in agent["present"]}
    failed = {(round(a[0], 6), round(a[1], 6)) for a in agent["missing"]}
    for _, x, y, _ in record["anchors"]:
        key = (round(x, 6), round(y, 6))
        px, py = grid.point(x, y)
        if key in ranged:
            ax.plot(px, py, marker="^", markersize=size, color=INK, markeredgecolor=SURFACE, markeredgewidth=0.5, zorder=6)
        elif key in failed:
            ax.plot(px, py, marker="^", markersize=size, markerfacecolor=SURFACE, markeredgecolor=INK, markeredgewidth=0.6, zorder=6)
        else:
            ax.plot(px, py, marker="^", markersize=size * 0.7, color="#9a9a96", markeredgewidth=0, zorder=6)
    if agent.get("truth"):
        ax.plot(*grid.point(*agent["truth"]), marker="x", markersize=size, color=INK, markeredgewidth=0.9, path_effects=ring, zorder=7)
    ax.plot(*grid.point(*agent["estimate"]), marker="o", markersize=size, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.9,
            path_effects=ring, zorder=7)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", type=Path, default=RECORD)
    parser.add_argument("--agent", type=int)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--width", choices=("column", "page"), default="column")
    parser.add_argument("--zoom", type=float, nargs=4, metavar=("X0", "X1", "Y0", "Y1"), help="zoom window in metres from the area's corner")
    parser.add_argument("--caption", type=Path, help="also write the caption's numbers to this file")
    args = parser.parse_args()
    style.setup()
    record = json.loads(args.record.read_text())
    for check, state in record["checks"].items():
        print(f"check {check} {state}")
    for other in record["agents"]:
        print(f"agent {other['agent_id']} informative {other['informative']}, kept cells {other.get('kept_cells', '-')}, "
              f"cells {other['cells']}, nodes {other['nodes']}, decisions {'yes' if 'decisions' in other else 'no'}")
    agent = choose(record, args.agent)
    grid = Grid(record)
    decisions = agent["decisions"]
    level, fate = np.asarray(decisions["level"], dtype=np.int64), np.asarray(decisions["fate"], dtype=np.int64)
    i, j = np.asarray(decisions["i"], dtype=np.int64), np.asarray(decisions["j"], dtype=np.int64)
    upper = numbers(decisions["upper"])
    i1, j1 = grid.nodes(level, i, j)
    x0, x1, y0, y1 = grid.span(i, i1, j, j1)
    rects = quads(x0, x1, y0, y1)
    drop = np.isin(fate, DROPS)
    kept = (fate == CELL) | (fate == COARSE)
    relative = np.minimum(upper - float(agent["best"]), 0.0)
    delta = float(agent["delta"])
    levels = np.unique(level[drop])
    top = int(levels.max()) if levels.size else 0
    shades = {k: gray(GRAY_FINE + (GRAY_COARSE - GRAY_FINE) * k / max(top, 1)) for k in range(top + 1)}

    # the figure, panel (a) as tall as the area and panel (b) square, one colorbar under each
    args.estimate = agent["estimate"]
    zoom = zoom_window(grid, rects[kept], args)
    aspect = grid.size[0] / grid.size[1]
    zoom_aspect = (zoom[1] - zoom[0]) / (zoom[3] - zoom[2])
    # the panels' height fills the width but for a cap, and the figure is as wide as the panels need, at most that width
    width = style.COLUMN if args.width == "column" else style.PAGE
    height = min((width - 0.75) / (aspect + zoom_aspect), 3.4 if args.width == "page" else 2.4)
    width = min(width, (aspect + zoom_aspect) * height + 0.75)
    fig = plt.figure(figsize=(width, height + 1.0), layout="constrained")
    layout = fig.add_gridspec(2, 2, width_ratios=[aspect, zoom_aspect], height_ratios=[1, 0.045])
    overview, detail = fig.add_subplot(layout[0, 0]), fig.add_subplot(layout[0, 1])
    level_bar, value_bar = fig.add_subplot(layout[1, 0]), fig.add_subplot(layout[1, 1])

    # panel (a): the nodes large enough to see one by one, the smaller ones and the kept cells as row runs
    points_per_metre = 72.0 * height / grid.size[1]
    visible = drop & ((x1 - x0) * points_per_metre >= VISIBLE_POINTS)
    overview.add_collection(PolyCollection(rects[visible], facecolors=[shades[k] for k in level[visible]], edgecolors=SURFACE,
                                           linewidths=GAP, zorder=1))
    cells = np.full((grid.ny, grid.nx), -1, dtype=np.int16)
    for k in np.flatnonzero(drop & ~visible):
        cells[j[k]:j1[k] + 1, i[k]:i1[k] + 1] = level[k]
    small, small_levels = runs(cells, grid)
    if len(small):
        overview.add_collection(PolyCollection(small, facecolors=[shades[k] for k in small_levels], edgecolors="none", zorder=1))
    region = np.full((grid.ny, grid.nx), -1, dtype=np.int16)
    for k in np.flatnonzero(kept):
        region[j[k]:j1[k] + 1, i[k]:i1[k] + 1] = 0
    blob, _ = runs(region, grid)
    overview.add_collection(PolyCollection(blob, facecolors=KEPT, edgecolors="none", zorder=2))
    overview.add_patch(Rectangle((zoom[0], zoom[2]), zoom[1] - zoom[0], zoom[3] - zoom[2], fill=False, edgecolor=INK, linewidth=0.6, zorder=5))
    overview.set_xlim(0, grid.size[0])
    overview.set_ylim(0, grid.size[1])
    markers(overview, record, agent, grid, 3.2)

    # panel (b): every node the zoom shows, the kept leaves by their value, with a gap only around a node large enough on
    # paper to show one
    shown = (x1 > zoom[0]) & (x0 < zoom[1]) & (y1 > zoom[2]) & (y0 < zoom[3])
    large = (x1 - x0) * 72.0 * height / (zoom[3] - zoom[2]) >= VISIBLE_POINTS
    dropped_shown = shown & drop
    detail.add_collection(PolyCollection(rects[dropped_shown], facecolors=[shades[k] for k in level[dropped_shown]], edgecolors=SURFACE,
                                         linewidths=np.where(large[dropped_shown], GAP, 0.0), zorder=1))
    norm = Normalize(vmin=-delta, vmax=0.0)
    for code, gap in ((CELL, 0.0), (COARSE, LEAF_GAP)):
        chosen = shown & (fate == code)
        if chosen.any():
            detail.add_collection(PolyCollection(rects[chosen], array=relative[chosen], cmap=TEAL, norm=norm, edgecolors=SURFACE,
                                                 linewidths=np.where(large[chosen], gap, 0.0), zorder=2))
    detail.set_xlim(zoom[0], zoom[1])
    detail.set_ylim(zoom[2], zoom[3])
    markers(detail, record, agent, grid, 4.2)

    for ax in (overview, detail):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("First coordinate (m)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3 if ax is overview else 4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6 if ax is overview else 4))
    overview.set_ylabel("Second coordinate (m)")

    bounds = np.arange(-0.5, top + 1.0)
    level_map = ScalarMappable(norm=BoundaryNorm(bounds, top + 1), cmap=ListedColormap([shades[k] for k in range(top + 1)]))
    bar = fig.colorbar(level_map, cax=level_bar, orientation="horizontal", ticks=range(0, top + 1, 2 if top > 5 else 1))
    bar.set_label("Drop level\n(a) Operating area", labelpad=1)
    bar.minorticks_off()
    bar.outline.set_linewidth(0.4)
    bar = fig.colorbar(ScalarMappable(norm=norm, cmap=TEAL), cax=value_bar, orientation="horizontal")
    bar.set_label("Log posterior less the best (nats)\n(b) Kept region", labelpad=1)
    bar.ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    bar.minorticks_off()
    bar.outline.set_linewidth(0.4)

    handles = [Line2D([], [], marker="^", color=INK, linestyle="none", markersize=4, label="Anchor heard"),
               Line2D([], [], marker="^", markerfacecolor=SURFACE, markeredgecolor=INK, linestyle="none", markersize=4, label="Anchor not heard"),
               Line2D([], [], marker="x", color=INK, linestyle="none", markersize=4, markeredgewidth=0.9, label="True position"),
               Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor=INK, linestyle="none", markersize=4, label="Reported position"),
               Patch(facecolor=KEPT, edgecolor="none", label="Kept cells")]
    if idle(record, agent):
        handles.insert(2, Line2D([], [], marker="^", color="#9a9a96", linestyle="none", markersize=3, label="Anchor heard by none"))
    # the legend's spacing keeps a column-wide figure within the 252 pt column, so it prints at its natural size and its
    # labels stay at 8 pt
    fig.legend(handles=handles, loc="outside upper center", ncol=3 if args.width == "column" else 5, handletextpad=0.3,
               columnspacing=0.6, handlelength=1.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # no creation date, so that a rerun gives the same bytes
    fig.savefig(args.output, metadata={"CreationDate": None})
    plt.close(fig)
    text = caption(record, agent)
    print(text)
    print("wrote", args.output, f"{args.output.stat().st_size / 1024:.0f} KiB,", int(visible.sum()), "nodes and", len(small),
          "runs in panel (a),", int(shown.sum()), "nodes in panel (b)")
    if args.caption:
        args.caption.write_text(text + "\n")


if __name__ == "__main__":
    main()
