"""Write the full result tables of the companion document from the final tables.

Reads DIR/<dataset>/evaluation/{percentiles,differences}.csv (read only), DIR research/artifacts/solid/tables of the
repository that holds this script unless --tables names another, and writes one table* per dataset and mode under the
joined pairs, with SOLID minus each baseline under that baseline, and one table of the ranged pairs where they differ
from the joined pairs. Every printed number is checked against FULL, the full tables that full_tables.py writes
(full-tables.tex), before anything is written. The comment lines of the output keep the names of the paper's sources.
Usage: python3 experiments/figures/gen_tables.py OUT --full FULL [--tables DIR]
       writes OUT, the companion document's sections/s3-tables.tex
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TABLES = REPO / 'research/artifacts/solid/tables'
# the text of the full tables, which main reads
FULL = None

DATASETS = (('bern', 'TWR-CLOUD'), ('hymn', 'HYMN'), ('aau', 'AAU Meshed UWB'), ('tiers', 'TIERS'),
            ('bregar', 'Bregar'), ('idlab', 'IDLab'))
MODES = ('AF', 'AAC', 'ANC')
GROUPS = {'AF': ('every cell',), 'AAC': ('every cell', 'anchors-1', 'anchors-2'),
          'ANC': ('headline, 3 or more anchors', 'anchors-1', 'anchors-2')}
GROUP_LABEL = {'every cell': 'All reporting cells', 'headline, 3 or more anchors': '3 or more anchors',
               'anchors-1': '1 anchor', 'anchors-2': '2 anchors'}
METHODS = ('solid', 'jin-2021', 'ade-bp-2015', 'fnl', 'strong')
LABEL = {'solid': 'SOLID', 'jin-2021': 'Jin', 'ade-bp-2015': 'ADE-BP', 'fnl': 'FNL', 'strong': 'STRONG'}
PCT = ('S50', 'S75', 'S90')


def fmt(x, signed=False):
    text = f'{x:+.1f}' if signed else f'{x:.1f}'
    return text


def tex_number(text):
    return text.replace('-', '$-$')


def cell(row, p, signed=False):
    v, lo, hi = float(row[f'{p}_cm']), float(row[f'{p}_low_cm']), float(row[f'{p}_high_cm'])
    plain = f'{fmt(v, signed)} [{fmt(lo, signed)}, {fmt(hi, signed)}]'
    return plain, tex_number(plain)


def read(dataset):
    values, diffs = defaultdict(dict), defaultdict(dict)
    for row in csv.DictReader(open(TABLES / dataset / 'evaluation/percentiles.csv', newline='')):
        values[(row['mode'], row['group'], row['track'], row['pair_set'])][row['method']] = row
    for row in csv.DictReader(open(TABLES / dataset / 'evaluation/differences.csv', newline='')):
        method = row['difference'].removeprefix('solid minus ')
        diffs[(row['mode'], row['group'], row['track'], row['pair_set'])][method] = row
    return values, diffs


CHECKED = []


def check(plain):
    """The printed text must appear in the committed full tables, as the full tables print it."""
    if plain not in FULL:
        raise SystemExit(f'not in full-tables.tex: {plain}')
    CHECKED.append(plain)


def share_line(row, ncols, skip):
    pairs, units = 100 * float(row['pair_share']), 100 * float(row['unit_share'])
    text = f'{pairs:.1f}\\,\\% of pairs and {units:.1f}\\,\\% of scenarios scored'
    check(f'{pairs:.1f}\\,\\% of pairs and {units:.1f}\\,\\% of units scored')
    return '&' * skip + f' \\multicolumn{{{ncols - skip}}}{{@{{}}l}}{{\\scriptsize {text}}} \\\\'


def block_rows(vals, difs, notes, first_cols):
    """Rows of one group and track: SOLID, then each baseline with SOLID minus it below."""
    rows = []
    present = [m for m in METHODS if m in vals]
    for m in present:
        r = vals[m]
        cells = []
        for p in PCT:
            plain, tex = cell(r, p)
            check(plain)
            cells.append(tex)
        label = LABEL[m]
        failed, aborted = int(r['failed_units']), int(r['aborted_units'])
        if failed:
            key = (failed, aborted)
            notes.setdefault(key, len(notes))
            label += f'$^{{\\mathrm{{{chr(ord("a") + notes[key])}}}}}$'
        rows.append([label] + cells)
        if m != 'solid':
            d = difs[m]
            dcells = []
            for p in PCT:
                plain, tex = cell(d, p, signed=True)
                check(plain)
                dcells.append(tex)
            rows.append([f'\\quad SOLID $-$ {LABEL[m]}'] + dcells)
    return rows


def table(dataset, name, mode, values, diffs, pair_set='joined', groups=None, tracks=('1', '10'), label=None,
          caption=None):
    groups = groups or GROUPS[mode]
    lines, notes = [], {}
    multi = len(groups) > 1
    ncols = 6 if multi else 5
    first = True
    for g in groups:
        for t in tracks:
            key = (mode, g, t, pair_set)
            if key not in values:
                continue
            body = block_rows(values[key], diffs[key], notes, None)
            if not first:
                lines.append('\\hline')
            first = False
            for i, row in enumerate(body):
                lead = []
                if multi:
                    lead.append(GROUP_LABEL[g] if (i == 0 and t == tracks[0]) else '')
                lead.append(t if i == 0 else '')
                lines.append(' & '.join(lead + row) + r' \\')
            solid = values[key]['solid']
            lines.append(share_line(solid, ncols, 1 if not multi else 2))
    head = (['Anchors'] if multi else []) + ['Readings', 'Method or difference', 'S50 (cm)', 'S75 (cm)', 'S90 (cm)']
    spec = ('l' if multi else '') + 'clrrr'
    out = ['\\begin{table*}[!t]', f'\\caption{{{caption}}}', f'\\label{{{label}}}', '\\centering', '\\footnotesize',
           '\\setlength{\\tabcolsep}{5pt}', '\\renewcommand{\\arraystretch}{1.05}',
           f'\\begin{{tabular}}{{@{{}}{spec}@{{}}}}', '\\hline', ' & '.join(head) + r' \\', '\\hline']
    out += lines
    out.append('\\hline')
    if notes:
        parts = []
        for (failed, aborted), k in sorted(notes.items(), key=lambda kv: kv[1]):
            sc = 'solve' if failed == 1 else 'solves'
            text = f'$^{{\\mathrm{{{chr(ord("a") + k)}}}}}$Failed on {failed} {sc}'
            if aborted:
                text += f', {aborted} by a crash of the solver process'
            text += '.'
            parts.append(text)
        out.append(f'\\multicolumn{{{ncols}}}{{@{{}}l@{{}}}}{{\\rule{{0pt}}{{2.4ex}}' + ' '.join(parts) + '}\\\\')
    out += ['\\end{tabular}', '\\end{table*}', '']
    return '\n'.join(out)


MODE_NAME = {'AF': 'AF', 'AAC': 'AAC', 'ANC': 'ANC'}


def main():
    global TABLES, FULL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('out', type=Path)
    parser.add_argument('--full', type=Path, required=True)
    parser.add_argument('--tables', type=Path, default=TABLES)
    args = parser.parse_args()
    TABLES, FULL = args.tables, args.full.read_text()
    out = ['% Written by companion/tools/gen_tables.py from research/artifacts/solid/tables/<dataset>/evaluation/'
           'percentiles.csv and differences.csv.', '% Every printed value equals its value in paper/tables/full-tables.tex.', '']
    count = 0
    for dataset, name in DATASETS:
        values, diffs = read(dataset)
        for mode in MODES:
            if not any(k[0] == mode for k in values):
                continue
            label = f'supp:tab-{dataset}-{mode.lower()}'
            caption = f'Full Results of {name} in {MODE_NAME[mode]}'
            out.append(table(dataset, name, mode, values, diffs, label=label, caption=caption))
            count += 1
    # The ranged pairs where they differ from the joined pairs, HYMN with one anchor and AAU with one reading.
    ranged = []
    for dataset, name in (('hymn', 'HYMN'), ('aau', 'AAU')):
        values, diffs = read(dataset)
        for key in sorted(values):
            mode, g, t, ps = key
            if ps != 'ranged':
                continue
            joined = values[(mode, g, t, 'joined')]
            same = all(joined[m][c] == values[key][m][c] for m in joined for c in joined[m]
                       if c.startswith('S') or c in ('pairs', 'scored_units'))
            if not same:
                ranged.append((dataset, name, mode, g, t))
    out.append('% The ranged pairs where they differ from the joined pairs.')
    lines, notes = [], {}
    first = True
    order = {g: i for i, g in enumerate(GROUPS['ANC'])}
    for dataset, name, mode, g, t in sorted(ranged, key=lambda r: (r[0] != 'hymn', order[r[3]], r[4])):
        values, diffs = read(dataset)
        key = (mode, g, t, 'ranged')
        body = block_rows(values[key], diffs[key], notes, None)
        if not first:
            lines.append('\\hline')
        first = False
        for i, row in enumerate(body):
            lead = [name if i == 0 else '', GROUP_LABEL[g] if i == 0 else '', t if i == 0 else '']
            lines.append(' & '.join(lead + row) + r' \\')
        lines.append(share_line(values[key]['solid'], 7, 3))
    head = ['Dataset', 'Anchors', 'Readings', 'Method or difference', 'S50 (cm)', 'S75 (cm)', 'S90 (cm)']
    tab = ['\\begin{table*}[!t]', '\\caption{Results Under the Ranged Pairs Where They Differ From the Joined Pairs}',
           '\\label{supp:tab-ranged}', '\\centering', '\\footnotesize', '\\setlength{\\tabcolsep}{5pt}',
           '\\renewcommand{\\arraystretch}{1.05}', '\\setlength{\\tabcolsep}{3.5pt}', '\\begin{tabular}{@{}llclrrr@{}}', '\\hline', ' & '.join(head) + r' \\',
           '\\hline'] + lines + ['\\hline', '\\end{tabular}', '\\end{table*}', '']
    out.append('\n'.join(tab))
    args.out.write_text('\n'.join(out))
    print(f'{count} tables of the joined pairs and one of the ranged pairs, {len(CHECKED)} printed values checked, '
          f'{len(ranged)} ranged blocks')


if __name__ == '__main__':
    main()
