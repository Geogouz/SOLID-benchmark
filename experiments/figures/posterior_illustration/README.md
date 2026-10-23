# Posterior illustration

This folder makes the paper's figure of SOLID's anchor only posterior. The figure shows how the certified adaptive refinement treats the posterior of one agent on a real TWR-CLOUD-Bern training unit.

Panel (a) shows the whole operating area. The quadtree nodes the refinement dropped are light rectangles, shaded by the level at which they were dropped. They are coarse far from the agent and fine near its modes. The cells the refinement kept form one teal region. Panel (b) zooms on that region and shades every kept leaf by its log posterior less the best value seen. A single cell carries its exact value, and a coarse leaf carries its upper bound. Both panels mark the anchors the agent ranged, the anchors it failed to reach, its true position and SOLID's estimate. Coordinates are metres from the minimum corner of the operating area.

The caption takes its numbers from the record. Among them are the cells of the grid, the exact cell evaluations, the nodes bounded, the cells kept and the certified share of the posterior mass left out, which is at most 1e-4 unless the node guard cut the posterior. The engine counts a probed cell that is later evaluated again at the finest level twice, so the evaluations can exceed the distinct cells by at most one per level of the quadtree.

## The frozen engine stays as it is

The frozen engine is used unchanged. The tool never edits a source file of the repository. Its plotting script writes the figure, and the caption's numbers when asked. Its run script copies the frozen sources of the checkout into its own folder and adds two files to that copy. The first is `illustration.rs`, a test module that the copy's `lib.rs` declares under `cfg(test)`, so the engine the builds link is the frozen one and only its own tests see the module. The second is `dump_problem.rs`, an example of the research CLI's package that cargo finds by itself, so no manifest and no lock file changes and `--locked` holds.

The helpers of the solve in `anchor_only.rs` and the node helpers in `posterior.rs` are private to their modules, so `illustration.rs` copies them line for line and names the lines it copies. The only changes turn private methods into functions with the same bodies, `LinkTable::build` and five methods of `Node`. Its recording copy of the refinement also writes down the fate of every node it decides about. Four assertions tie the record to the frozen engine.

1. The digest of every number of the problem and the profile, which the dump computes before it writes them and the test after it reads them, is the same, so both crossed the JSON file bitwise.
2. The frozen solve of the input gives bitwise the result that the dump computed through the research CLI's own path, the elapsed time aside.
3. The test's copy of the solve gives bitwise the result of the frozen solve, the elapsed time aside. That covers every position, the risk, the iterations, the evaluations, the guarded agents, the largest dropped share, and the nodes and cells summed over every posterior of the solve.
4. For every informative agent, the recording refinement keeps the same leaves in the same order with the same bounds as the frozen `adaptive` on the same evaluator, with the same dropped mass and share, guard flag, nodes and cells. The kept mass is private to the engine and enters the solve only through the share, which is compared.

For an input the dump wrote, the record is written only when all four hold, so the figure provably shows the frozen algorithm. The record's `checks` entry states each of them.

## Files

| File | Role |
| --- | --- |
| `illustration.rs` | The ignored engine test that writes the record, placed as `engine/src/illustration.rs` in the copy |
| `dump_problem.rs` | Writes the input of one anchor only unit and lists the anchor only scenarios of an archive, placed as `cli/examples/illustration_dump.rs` in the copy |
| `run_illustration.sh` | Copies the sources, builds with the pinned toolchain and `--locked`, and runs the dump and the test, on Linux |
| `paper-v10/plot_posterior.py`, `paper-v10/style.py` | Draw the figure from the record |

## Making the record

The script copies the sources of the checkout that holds it, or of the one `REPO` names, and writes its files only under `$WORK/figures/posterior-illustration`, where `WORK` is `./work` unless set. It resolves every path it writes with `realpath` first and stops before writing if one resolves elsewhere. It deletes only its previous copy of the sources, a stale record and its own lock, all inside that folder. The lock keeps two runs from sharing the one copy of the sources. It builds with the toolchain that `rust-toolchain.toml` pins and runs with `RAYON_NUM_THREADS=1` and `TMPDIR=$WORK/tmp`, a folder that must exist, where the compiler and the linker keep their temporary files. Cargo and rustup keep using their own caches as for every build, and cargo's automatic cleaning of its cache is off while the script runs. It needs Linux, for GNU `realpath` and `sha256sum`.

Give the archive as an absolute path. The calibrated archives of the held scans lie at `out/solid/unify/bern-anchor-only-scenarios-v2/scan-NN/calibrated.zip` of the checkout if the corpus sits at its default place, where `research/benchmarks/bern/training/generate_heldscan_corpora.py anchor-only` writes it, and `SOLID_ANCHOR_ONLY_CAMPAIGN` names another place where it was built elsewhere.

First list the anchor only scenarios of one scan with the anchors their agents ranged, at 10 packets per link and iteration 1, from the root of the checkout.

```sh
mkdir -p work/tmp
sh experiments/figures/posterior_illustration/run_illustration.sh list ARCHIVE 10 1
```

The list goes to `$WORK/figures/posterior-illustration/list-FOLDER-samples-10-iteration-1.tsv`, where `FOLDER` is the name of the archive's folder, `scan-01` for example. Choose a training scenario whose agents ranged a typical number of anchors, then write its record.

```sh
sh experiments/figures/posterior_illustration/run_illustration.sh run ARCHIVE SCENARIO 10 1
```

The profile defaults to `research/artifacts/solid/profiles/bern-anchor-only.json` of the checkout, the final Bern profile, and a sixth argument names another. The research CLI applies one profile file to every unit it solves and changes nothing per scenario, so the unit is solved exactly as the CLI solves it under that file. A held scan's fold has its own profile, with its own area, height prior, grid step and fitted values, so reproducing the fold's own run of a unit takes that fold's profile file. A seventh argument, an agent identifier, keeps the decisions of that agent alone in the record, which makes it smaller, and the test stops if that agent is not an informative agent of the unit. The run writes `input.json`, `record.json` and their logs to `$WORK/figures/posterior-illustration/runs/FOLDER-scenario-SCENARIO-samples-10-iteration-1-PROFILE`, where `PROFILE` is the stem of the profile's file, and the log of the test prints the caption's numbers of every agent.

## Drawing the figure

Copy `record.json` to `research/artifacts/solid/figures/data/posterior-illustration.json` and draw it.

```sh
python experiments/figures/posterior_illustration/paper-v10/plot_posterior.py --agent AGENT
```

The figure goes to `research/artifacts/solid/figures/posterior.pdf`, one column wide, and `--output` names another file. `--width page` draws it for two columns, `--zoom X0 X1 Y0 Y1` sets the window of panel (b) in metres, and `--caption FILE` writes the caption's numbers to a file as well. Without `--agent` the script draws the first informative agent. It prints every agent's numbers first, so an agent can be chosen from them.

## The record

The record holds the grid, the area, the anchors, the checks, the frozen solve's metrics, its provenance and one entry per agent. The provenance names the archive and the profile with their SHA-256, the commit of the checkout, none in a reader's checkout that is not a git repository (`SOLID_READER=1`), and the identity of its frozen sources. An agent's entry holds its law, the anchors of its links, of its successful attempts and of its failed ones, its estimate and true position, its certificate and work, its kept leaves, and its decisions. The decisions are columns of the level, the first cell `i` and `j`, the lower and upper bound and the fate of every node, level by level in the order of the frontier. A node of level `k` covers the cells from `i` to `i + 2^k - 1` and from `j` to `j + 2^k - 1`, clipped to the grid. The fates are 0 for a split, 1 for a drop at the node's level, 2 for a drop by the node guard, 3 for a kept single cell, 4 for a kept coarse leaf and 5 for a leaf the last pass dropped. A bound that is not finite is the string `Infinity`, `-Infinity` or `NaN`.
