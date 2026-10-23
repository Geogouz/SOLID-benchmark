"""Assemble the full tables of the evaluation into one LaTeX document, which the companion document's tables follow.

Every table of research/artifacts/solid/tables/<dataset>/evaluation/tables.tex keeps its numbers exactly as the final
tables wrote them. The comment line the builder writes above each table becomes its caption, and the datasets follow
the order of the paper, TWR-CLOUD first, then the held-out datasets and the annex.

Usage: full_tables.py --tables research/artifacts/solid/tables --out DIR [--compile]
       writes DIR/full-tables.tex, and with --compile DIR/full-tables.pdf by pdflatex
"""
import argparse
import re
import subprocess
from pathlib import Path

ORDER = (("bern", "TWR-CLOUD"), ("hymn", "HYMN"), ("aau", "AAU Meshed UWB"), ("tiers", "TIERS"), ("bregar", "Bregar"), ("idlab", "IDLab"))
PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{booktabs}
\usepackage{newtxtext,newtxmath}
\usepackage{caption}
\captionsetup{font=small,labelformat=empty,skip=3pt}
\setlength{\tabcolsep}{4pt}
\begin{document}
\section*{Full tables of the evaluation}
Balanced percentiles in cm, each with the 95\,\% interval of its paired cluster bootstrap in brackets, and the paired
differences of SOLID minus every baseline. The joined pairs are the headline pair set. Written by
\texttt{research/solid/final\_tables.py} under the declared analysis \texttt{research/artifacts/solid/splits/final-analysis.json}.
"""


def blocks(text):
    """(caption, tabular) of every table of one tables.tex, the caption the comment line above the table."""
    found = []
    for match in re.finditer(r"^% ([^\n]+)\n(\\begin\{tabular\}.*?\\end\{tabular\})", text, flags=re.S | re.M):
        found.append((match.group(1).strip(), match.group(2)))
    return found


def escape(caption):
    return caption.replace("%", r"\%").replace("_", r"\_").replace("&", r"\&")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    parts, count = [PREAMBLE], 0
    for dataset, label in ORDER:
        tables = blocks((args.tables / dataset / "evaluation/tables.tex").read_text())
        parts.append(f"\\subsection*{{{label}}}\n")
        for caption, tabular in tables:
            parts.append("\\begin{table}[h!]\\centering\\small\n" + f"\\caption{{{escape(caption)}}}\n" + tabular + "\n\\end{table}\n")
            count += 1
        parts.append("\\clearpage\n")
    parts.append("\\end{document}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.out / "full-tables.tex"
    source.write_text("".join(parts))
    print(f"{count} tables written to {source}")
    if args.compile:
        for _ in range(2):
            run = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", source.name], cwd=args.out, capture_output=True, text=True)
            if run.returncode:
                print(run.stdout[-3000:])
                raise SystemExit("pdflatex failed")
        print("compiled", args.out / "full-tables.pdf")


if __name__ == "__main__":
    main()
