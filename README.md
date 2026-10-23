# SOLID-benchmark

SOLID, Spatial Optimization with Likelihood-Informed Distance Geometry, positions radios from ultra-wideband two-way
ranging. It describes the readings of every pair of radios by a likelihood over their distance, adds what failed
exchanges say about that distance, and fits the four parameters of the likelihood per site. One model serves the three
modes of the paper, in which radios of unknown position, the agents, range only each other (AF), each other and radios
at surveyed positions, the anchors (AAC), or only the anchors (ANC).

This repository holds the method, the four baselines of the paper (Jin et al., FNL, STRONG and ADE-BP) and the scripts
that reproduce the tables, figures and printed counts of the paper and its companion document from the six public
datasets. The companion document, `results/SOLID-details-and-full-results.pdf`, holds the full results, the ablation on
the training parts and the details of the method, the datasets and the comparison, and `results/tables/` holds the final
tables as CSV files. The repository holds no data. The data come from their publishers, and your runs write their own
results.

## Repository map

| Path | What it holds |
| --- | --- |
| `runtime/` | The engine of SOLID, the crate `solid-core`, and its C interface |
| `research/cli/`, `research/io/`, `research/evaluation/` | The programs that solve scenarios and units, the adapters that feed the baselines, and their scoring |
| `research/methods/` | The four baselines |
| `research/solid/` | SOLID's link fits, deployment settings and profiles, its runners on every dataset, and the final tables |
| `research/benchmarks/bern/` | TWR-CLOUD, which the scripts call bern, with its scenario runner, calibration, training corpora and ADE-BP's fits |
| `research/benchmarks/datasets/` | The builders of TIERS, Bregar, IDLab, HYMN and AAU Meshed UWB, and the runners of the baselines on them |
| `research/artifacts/` | The shipped settings of SOLID (the 12 fitted profiles, the link fits, the deployment settings and the two split records) and ADE-BP's fitted error laws |
| `experiments/ade-heightspread/`, `experiments/ade-reselection/` | ADE-BP's fitted error laws for Bregar, HYMN, AAU and IDLab's ANC mode |
| `settings/` | `baseline-options.json`, the final options of the baselines (Table S-XVIII), and `data/`, the pins of the public files and the settings that the builders read, the settings of the TIERS and IDLab campaigns among them |
| `experiments/figures/`, `experiments/timing/` | The scripts of the paper's tables, figures and counts, and the controlled timing |
| `tools/` | `fetch_data.py`, which fetches and verifies the data, and `reproduce.py`, which runs the stages below |
| `results/` | `SOLID-details-and-full-results.pdf`, the paper's companion document, and `tables/<dataset>/`, the final tables of each dataset as CSV files, `percentiles.csv` with S50, S75 and S90 and their intervals for every method and `differences.csv` with SOLID minus each baseline, which your runs rebuild in `research/artifacts/solid/tables` |

In `results/tables/`, the folder `bern` holds TWR-CLOUD and the column `track` gives the reading count. The columns
`S50_cm`, `S75_cm` and `S90_cm` hold the released scorer's values, which the tables print, and the `_exact_cm` columns
the same percentiles in exact arithmetic, which differ only in three HYMN medians, by 0.03 to 0.42 cm. The columns
`median_ms`, `threads_per_process` and `processes` describe the evaluation runs, which solved 180 or 190 scenarios at
once, as processes or as threads of one process, so their times do not compare the speed of the methods. The
controlled timing of Fig. S5 does.

## Platform

The paper's runs were made on Linux x86-64, and there the scripts give their results bit for bit. On macOS arm64 some
results differ. numpy's uniform draw is compiled with a fused multiply-add there, so the starting coordinates of many
units differ in their last bit, which moves FNL and STRONG by less than 1e-10 cm. Clarabel, the semidefinite solver of
Jin et al., links Apple's Accelerate on macOS and OpenBLAS on Linux, which moves Jin et al.'s estimates by up to about
0.04 cm. In Bregar's environment 2, two training positions lie at the same distance from anchor A1 up to one unit in the
last place, and macOS takes the other one as the shortest line-of-sight link. The calibration of that environment then
differs, its anchor offsets by up to 34.2 cm, and so do both halves of Bregar. ADE-BP, a particle method, turns the tiny
differences between the platforms' mathematical functions into other draws of its particles. On HYMN it received the
paper's inputs bit for bit, apart from the starting coordinates that it does not read, and it repeats exactly on one
machine. Yet on macOS 1,196 of its 1,200 units moved, by up to 953 cm with one or two anchors and by less than 0.05 cm
with eight or more anchors and ten readings, and its percentiles by up to 16 percent with two anchors and by at most 3 percent with three or more. SOLID's own estimates on HYMN were
the paper's bit for bit on macOS, as are its anchor-only solves elsewhere, while its link fits differ by about
1e-8 relative and its cooperative solves by up to 1e-4 cm on TIERS and up to about 0.05 cm on TWR-CLOUD, where an AF
solve can also return the same layout moved rigidly, which the pair error does not see. Use Linux x86-64 to reproduce
the printed numbers.

## Setup

`rust-toolchain.toml` pins the Rust toolchain 1.97.1. Python is 3.12 with the packages of `research/requirements.txt`,
numpy 2.5.3, scipy 1.18.1, pandas 3.0.1, rosbags 0.11.0 and matplotlib 3.10.8, the versions of the paper's runs.

```sh
git clone https://github.com/Geogouz/SOLID-benchmark
cd SOLID-benchmark
rustup toolchain install 1.97.1 --profile minimal
python3.12 -m venv .venv
.venv/bin/python -m pip install -r research/requirements.txt
.venv/bin/python tools/reproduce.py --stage build
```

The build runs `cargo build --release --locked --manifest-path research/Cargo.toml` for the package `solid-cli` and for
`solid-bern-calibration`. The first gives `solid`, `solid-units`, `solid-archive` and the adapters `tiers-adapter`,
`bregar-adapter`, `idlab-layout-adapter` and `heldout-adapter`, and the second `fit_corrected_training`, the fitter of
the TWR-CLOUD calibration. They land in `research/target/release`, where the scripts look for them. On Linux the build
downloads OpenBLAS 0.3.32 and compiles it for Clarabel with the settings of `.cargo/config.toml`, so it needs network
access, a C compiler and make. The build took 40 seconds on a laptop, and compiling OpenBLAS adds some minutes on
Linux.

## Data

The data folder is `research/data` of the checkout, or the folder that the variable `SOLID_DATA` names.
`tools/fetch_data.py` downloads every file the scripts read into it, 9.16 GB in all, and checks each against its size,
SHA-256 and published MD5 in `settings/data/pins.json`. Every dataset needs the released scorer of TWR-CLOUD, whose
pair loss scores them all, so fetch `scorer` with any dataset. TIERS also needs the IDLab files, since its preflight
inventories them.

```sh
.venv/bin/python tools/fetch_data.py                            # every file
.venv/bin/python tools/fetch_data.py --dataset hymn scorer      # one dataset and the scorer
.venv/bin/python tools/fetch_data.py --check --strict           # verify without writing
```

| Dataset | Source | Licence |
| --- | --- | --- |
| TWR-CLOUD | Zenodo record 23434390, version 6, [doi:10.5281/zenodo.23434390](https://doi.org/10.5281/zenodo.23434390), with the released scorer. The fetch also takes the responder table of version 5, [doi:10.5281/zenodo.20277908](https://doi.org/10.5281/zenodo.20277908), which no result reads and only optional comparisons open | MIT |
| TIERS | [TIERS/uwb-relative-localization-dataset](https://github.com/TIERS/uwb-relative-localization-dataset) at revision 5dceab32c4d2b1dc32016a80a5b2406d973cea2d | MIT |
| Bregar | Zenodo record 8280736, version 2.0, [doi:10.5281/zenodo.8280736](https://doi.org/10.5281/zenodo.8280736), with the publisher's preprocessing code of [KlemenBr/uwb_positioning](https://github.com/KlemenBr/uwb_positioning) tag v1.0.0, revision ef1c7906344ce54ab1b0de9c3e7b4e1e5f4dc4cc | CC BY 4.0 |
| IDLab | [doi:10.21227/79pv-0963](https://doi.org/10.21227/79pv-0963) on IEEE DataPort, read from its open copy, the imec share that the card of [iwine-ugent/uwb-positioning-datasets](https://huggingface.co/datasets/iwine-ugent/uwb-positioning-datasets) at revision d813ce72783241142102723b77c09c0f5fa165d1 links, and the publisher's aggregate of [dietercoppens/UWB-DRL-PHY-Runtime-Adaptation-dataset](https://github.com/dietercoppens/UWB-DRL-PHY-Runtime-Adaptation-dataset) at revision c1b2298534d3bcbc4894b8a00d3f06186a81ceba | Apache 2.0 on the card |
| HYMN | Zenodo record 19566178, release 1.3, [doi:10.5281/zenodo.19566178](https://doi.org/10.5281/zenodo.19566178) | Data CC BY 4.0, code MIT |
| AAU Meshed UWB | Zenodo record 10819706, [doi:10.5281/zenodo.10819706](https://doi.org/10.5281/zenodo.10819706) | CC BY 4.0 on the record, with a BSD-2 licence in the archive for the publisher's software |

The raw IDLab logs of about 34 GB are not fetched. Only a check of the published files against them reads them, and its
record ships as `settings/data/idlab-availability.json.gz`. The built TWR-CLOUD data take about 14 GB beside the
downloads, and preparing the scorer's input needs Info-ZIP `zip`. The runs of all datasets take about 36 GB.

## Reader mode

Every command of the reproduction runs with `SOLID_READER=1`, which `tools/reproduce.py` sets. It skips the provenance
checks that only the authors' own checkout, builds and commits can pass, and it keeps every check of your downloads
against the pins, of the shipped settings and of the consistency of your own runs.

The TIERS and IDLab campaigns, the inputs of every run on those two datasets, are frozen without running the adapter
checks of `tiers_static_check.py` and `idlab_check.py`. Their `prepare` stages write the inputs of those checks, and the
campaigns take the settings of the paper's campaigns from `settings/data/tiers-campaign-settings.json` and
`idlab-campaign-settings.json`, which must hold every setting that your build prepares. They also hold a few fields
that the builders here no longer write, so the campaigns you build hold exactly the paper's settings. The checks of the
settings stage compare values with the shipped files: `deployment.py --check` the statistics of the deployment rules,
not the SHA-256 of the halves they read, and `ade_training.py --check` ADE-BP's fitted laws, rows and conversions, not
the SHA-256 of its sources and code.

## Stages

`tools/reproduce.py` runs the stages in order, prints every command before it runs it and stops at the first that
fails. It computes nothing itself, so every printed command can also be run alone from the repository root with
`SOLID_READER=1`.

```sh
.venv/bin/python tools/reproduce.py [--dataset NAME ...] [--stage NAME ...] [--workers N] [--dry-run]
```

The datasets are `twr-cloud`, `tiers`, `bregar`, `idlab`, `hymn` and `aau`, all six unless `--dataset` names some.
Without `--stage` the driver runs build, data, runs and tables. `--workers` sets the processes or threads of every run
and fit, the processors of the machine unless given, and `--dry-run` prints the commands without running them. A command
that reads several datasets runs only when all of them are selected. The checks of the deployment settings and profiles
of HYMN and AAU need both datasets, the ablation the four datasets of its table, and the check of all profiles and the
paper's own tables, figures and counts all six. The driver sets `OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS` and
`RAYON_NUM_THREADS` to 1 as the paper's runs did, and the solvers size their own pools from `--workers`. Runs go to
`out/`, the final tables to `research/artifacts/solid/tables/<dataset>/evaluation`, curves, counts and records to
`research/artifacts/solid/figures/data`, figures to `research/artifacts/solid/figures` and the LaTeX tables to
`out/tables`. A run resumes from its caches when it is started again, while a builder refuses to overwrite what it
wrote, so move a builder's output aside before you build it again.

| Stage | What it runs | Cost |
| --- | --- | --- |
| build | cargo, see Setup | about a minute, more on Linux |
| data | `fetch_data.py --check --strict` for the dataset and the scorer, then its builders, which write the training and evaluation parts into the data folder. On TWR-CLOUD these prepare the scorer's input, the calibration of all training scans and of each held scan, a training scan calibrated by the range line of the other ten, the calibrated evaluation archive and the training corpora of the held scans. On TIERS and IDLab they end with the frozen campaigns, `tiers-static-campaign`, `idlab-campaign` and `idlab-training-campaign`, whose base and catalog every run, fit and figure of those datasets reads | seconds for HYMN and AAU, about two minutes for Bregar, about four minutes for TIERS and one for IDLab on a laptop, longer for the archives of TWR-CLOUD |
| settings | Optional. The refits of SOLID's link fits, deployment settings and profiles and of ADE-BP's error laws. A script with `--check` compares its refit with the shipped file and writes nothing over it, and the link fits of the other scripts go beside the shipped files. Without this stage, and after it, the runs read the shipped settings | the TWR-CLOUD link fit alone took about 6 hours |
| runs | SOLID and the four baselines on the evaluation part of the dataset, in every mode the paper runs | SOLID on TWR-CLOUD took 9, 89 and 150 minutes in AF, AAC and ANC on 90 cores, and ADE-BP 89 and 18 minutes in AAC and ANC. SOLID's solves on IDLab took about 470 CPU hours in AAC and 365 in ANC. HYMN took about 4 minutes for SOLID and 75 seconds for the baselines on one core of a laptop |
| tables | `final_tables.py build` and `export_cdfs.py` for each dataset, then, for all six, the scripts of the next section | the paired bootstrap of the TWR-CLOUD table took 26.5 minutes on a laptop, and the IDLab table reads its 3 GB score files whole and needs plenty of memory |
| ablation | `research/solid/ablation.py plan`, `fit`, `run`, `baselines`, `score` and `table`, which refit SOLID with one part of its block likelihood removed on the training parts of TWR-CLOUD, TIERS, Bregar and IDLab and score it, under `out/ablation/` | about 92 CPU hours of fits and 1.96 million solves |
| timing | Linux only. `experiments/timing/timing_bern.sh` times every method and mode on the TWR-CLOUD evaluation scenarios on an idle machine, then `summarize_timing.py` and `plot_speed.py` draw Fig. S5. `SOLID_TIMING_WORKERS` and `SOLID_TIMING_PIN` set the processes and their pinning, which were 90 and `taskset -c 0-89` for the paper | about 6 hours on 90 cores |

## Where each result comes from

| Result of the paper or its companion document | Script | Output |
| --- | --- | --- |
| Table II, balanced S75 and exclusive advantage share | `summary_table.py`, with the shares of `advantage_share.py` | `out/tables/summary-table.tex` |
| Fig. 3, the CDFs of TWR-CLOUD, and Figs. S1 to S4, those of HYMN and AAU, TIERS, Bregar and IDLab | `export_cdfs.py`, then `plot_cdfs.py` | `bern-cdfs.pdf` and `annex-*-cdfs.pdf` in `research/artifacts/solid/figures` |
| Table III, the fitted link parameters of each site | `fit_table.py`, from the shipped fits | `out/tables/fit-table.tex` |
| Tables S-IV to S-XVI, the full results and the ranged pairs | `full_tables.py`, then `gen_tables.py` | `out/tables/full-tables.tex` and `out/tables/s3-tables.tex` |
| Table S-XVII, the ablation | `research/solid/ablation.py table` | `out/ablation/` |
| Table S-XIX, the solves that stayed at their start | `anchor_starts.py --table` | `out/tables/s5-stays-table.tex` |
| Fig. 2, the posterior of one TWR-CLOUD agent, and Table S-II | `baselines_bern_training.py --freeze-only` writes the archive of the first held scan, then `posterior_illustration/run_illustration.sh` (Linux only) and `plot_posterior.py --agent 122` | `research/artifacts/solid/figures/posterior.pdf` and the record `posterior-illustration.json` |
| Fig. S5, the time of one solve on TWR-CLOUD | the timing stage | `research/artifacts/solid/figures/speed.pdf` |
| Fig. S6, the block likelihood of the TWR-CLOUD link of Section I | `scene_link.py`, then `scene_curve.py --plot` | `research/artifacts/solid/figures/scene-link.pdf` |
| The counts that the text prints | `paper_counts.py`, `anchor_starts.py`, `anchor_sides.py`, `idlab_two_anchors.py` and `tiers_line.py` | `paper-counts.json`, `anchor-starts.json`, `anchor-sides.json`, `idlab-two-anchors.json` and `tiers-line.json` in `research/artifacts/solid/figures/data` |
| Table S-XVIII, the options of the baselines | shipped as `settings/baseline-options.json` | |
| Tables S-I and S-III, the settings and multipliers of each site and the fixed settings of SOLID | the shipped fits and profiles in `research/artifacts/solid`, and `research/solid/method_constants.py` | |

The scripts without a folder are in `experiments/figures`. Numbers with the prefix S belong to the companion document.
Table I and Fig. 1 hold no computed result. The record of
Fig. 2 holds training scenario 10024 of the first held scan with ten readings and iteration 1, and `paper_counts.py` reads it as
well, so the counts follow Fig. 2 at the end of the tables stage.

## Licence and citation

The code is under the MIT licence of `LICENSE`. It contains no third-party code, and the baselines are our
implementations from their published papers. The datasets keep their own licences, listed above. If you use this
repository, please cite the paper as `CITATION.cff` gives it, D. Xenakis, A. Di Maio and T. Braun, "SOLID: Spatial
Optimization with Likelihood-Informed Distance Geometry for Range-Based Positioning", IEEE Journal of Indoor and
Seamless Positioning and Navigation, in preparation.
