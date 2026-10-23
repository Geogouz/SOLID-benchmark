"""The paper's one results table: the datasets against the methods, mode by mode, from the final tables.

Every row is a case, one dataset, mode and sample track, and every column a method, so a dash marks a mode that the
method does not run, and a dataset without a mode has no row in that mode (the author's layout of 4 October 2026).
Every cell is a method's balanced S75 in cm over the headline group and the joined pairs of the final tables. The
lowest value of every row is bold, and an asterisk on SOLID's value marks a difference from the best baseline of the row
whose paired bootstrap interval excludes zero. The last column, headed by the italic S, is SOLID's exclusive
advantage share in percent from advantage_share.py, its part of the summed exclusive advantages of all methods of the
panel (the author's score of 4 October 2026). The table is one column wide. The CDF figures show the whole
distributions, and the full tables are in the companion document.

Usage: summary_table.py --tables research/artifacts/solid/tables [--advantage FILE] --out DIR [--compile]
       writes DIR/summary-table.tex, and with --compile a standalone DIR/summary-table.pdf to preview it
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

DATASETS = (("bern", "TWR-CLOUD"), ("hymn", "HYMN"), ("aau", "AAU"), ("tiers", "TIERS"), ("bregar", "Bregar"), ("idlab", "IDLab"))
MODES = (("AF", ("solid", "jin-2021", "fnl")), ("AAC", ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")),
         ("ANC", ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")))
METHODS = ("solid", "jin-2021", "ade-bp-2015", "fnl", "strong")
LABELS = {"solid": "SOLID", "jin-2021": "Jin", "ade-bp-2015": "ADE-BP", "fnl": "FNL", "strong": "STRONG"}
HEADLINE = {"AF": "every cell", "AAC": "every cell", "ANC": "headline, 3 or more anchors"}
TRACKS = ("1", "10")


def read(tables):
    """{(dataset, mode, track, method): S75} and {(dataset, mode, track, method): (low, high)} of SOLID minus a method."""
    values, intervals = {}, {}
    for dataset, _ in DATASETS:
        folder = tables / dataset / "evaluation"
        for row in csv.DictReader(open(folder / "percentiles.csv", newline="")):
            if row["pair_set"] == "joined" and row["group"] == HEADLINE[row["mode"]]:
                values[(dataset, row["mode"], row["track"], row["method"])] = float(row["S75_cm"])
        for row in csv.DictReader(open(folder / "differences.csv", newline="")):
            if row["pair_set"] == "joined" and row["group"] == HEADLINE[row["mode"]] and row["difference"].startswith("solid minus "):
                method = row["difference"].removeprefix("solid minus ")
                intervals[(dataset, row["mode"], row["track"], method)] = (float(row["S75_low_cm"]), float(row["S75_high_cm"]))
    return values, intervals


def number(value):
    return f"{value:.0f}" if value >= 100 else f"{value:.1f}"


def significant(values, intervals, dataset, mode, track):
    """Whether SOLID differs significantly from the best baseline of one case."""
    baselines = [method for method in METHODS[1:] if (dataset, mode, track, method) in values]
    rival = min(baselines, key=lambda method: values[(dataset, mode, track, method)])
    low, high = intervals[(dataset, mode, track, rival)]
    return low > 0 or high < 0


def latex(values, intervals, advantage):
    width = 3 + len(METHODS)
    lines = [r"\begin{tabular}{@{}l@{\hspace{3pt}}rr@{\hspace{9pt}}" + "r" * (len(METHODS) - 1) + "r@{}}", r"\toprule",
             r"Dataset & Readings & " + " & ".join(LABELS[method] for method in METHODS) + r" & $S$ \\", r"\midrule"]
    for m, (mode, _) in enumerate(MODES):
        lines.append(rf"\multicolumn{{{width}}}{{@{{}}l}}{{\textit{{{mode}}}}} \\")
        for dataset, label in DATASETS:
            for track in TRACKS:
                if (dataset, mode, track, "solid") not in values:
                    continue
                present = [method for method in METHODS if (dataset, mode, track, method) in values]
                best = min(present, key=lambda method: values[(dataset, mode, track, method)])
                cells = []
                for method in METHODS:
                    if method not in present:
                        cells.append(r"\multicolumn{1}{c}{--}")
                        continue
                    text = number(values[(dataset, mode, track, method)])
                    if method == "solid" and significant(values, intervals, dataset, mode, track):
                        text += r"\rlap{*}"
                    cells.append(rf"\textbf{{{text}}}" if method == best else text)
                share = advantage[f"{dataset} {mode} {track}"]["methods"]["solid"]["share"]
                lines.append(f"{label if track == TRACKS[0] else ''} & {track} & " + " & ".join(cells) + f" & {share:.0f}\\% " + r"\\")
        lines.append(r"\midrule" if m < len(MODES) - 1 else r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


STANDALONE = r"""\documentclass[10pt]{article}
\usepackage[paperwidth=4in,paperheight=5in,margin=0.25in]{geometry}
\usepackage{booktabs}
\usepackage{newtxtext,newtxmath}
\pagestyle{empty}
\begin{document}
\noindent\begin{minipage}{252pt}\centering
{\footnotesize\setlength{\tabcolsep}{2.6pt}\input{summary-table.tex}\par}
\end{minipage}
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--advantage", type=Path, default=Path(__file__).resolve().parents[2] / "research/artifacts/solid/figures/data/advantage-shares.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    values, intervals = read(args.tables)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary-table.tex").write_text(latex(values, intervals, json.loads(args.advantage.read_text())))
    print("wrote", args.out / "summary-table.tex")
    if args.compile:
        (args.out / "summary-preview.tex").write_text(STANDALONE)
        run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "summary-preview.tex"], cwd=args.out, capture_output=True, text=True)
        if run.returncode:
            print(run.stdout[-2500:])
            raise SystemExit("pdflatex failed")
        (args.out / "summary-preview.pdf").replace(args.out / "summary-table.pdf")
        print("compiled", args.out / "summary-table.pdf")


if __name__ == "__main__":
    main()
