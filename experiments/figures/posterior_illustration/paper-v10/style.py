"""The style of the paper's figures: IEEE Transactions sizes, embedded Times-like fonts, vector output, and one color
and line style per method, those of the earlier dashboard, so a method looks the same in every figure."""
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

# widths of an IEEE Transactions column and page, in inches
COLUMN = 3.5
PAGE = 7.16

METHODS = ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")
LABELS = {"solid": "SOLID", "jin-2021": "Jin", "ade-bp-2015": "ADE-BP", "fnl": "FNL", "strong": "STRONG"}
COLORS = {"solid": "#007f73", "jin-2021": "#d97706", "ade-bp-2015": "#65a30d", "fnl": "#4f46e5", "strong": "#64748b"}
DASHES = {"solid": "-", "jin-2021": "--", "ade-bp-2015": "-.", "fnl": (0, (5, 1.5, 1, 1.5, 1, 1.5)), "strong": ":"}
WIDTHS = {"solid": 1.4}
MODES = {"AF": "Anchor-free (AF)", "AAC": "Anchor-aided cooperative (AAC)", "ANC": "Anchor-only non-cooperative (ANC)"}
# the percentiles of the final tables, drawn as light guides on every CDF
LEVELS = (0.5, 0.75, 0.9)


def setup():
    # a colorbar of 50 or more colors rasterizes its solids by default, which embeds an image in the PDF, and the
    # paper's figures are vector only
    import matplotlib.colorbar
    matplotlib.colorbar.Colorbar.n_rasterize = float("inf")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.direction": "in", "ytick.direction": "in", "lines.linewidth": 0.9, "legend.frameon": False, "legend.handlelength": 2.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "axes.spines.top": True,
        "axes.spines.right": True,
    })


def line(ax, method, x, y, **kwargs):
    """One method's line in its color, dash and width."""
    return ax.plot(x, y, color=COLORS[method], linestyle=DASHES[method], linewidth=WIDTHS.get(method, 0.9), label=LABELS[method],
                   zorder=3 if method == "solid" else 2, **kwargs)[0]


def guides(ax):
    for level in LEVELS:
        ax.axhline(level, color="#d9d9d9", linewidth=0.4, zorder=0)


def nice_ceiling(value):
    """The smallest of 1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8 times a power of ten at or above a positive value."""
    import math
    power = 10 ** math.floor(math.log10(value))
    for step in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * power >= value:
            return step * power
    return 10 * power
