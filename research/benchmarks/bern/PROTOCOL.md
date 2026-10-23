# TWR-CLOUD-Bern evaluation protocol

This document states the evaluation of the TWR-CLOUD-Bern comparison as the paper describes it: the question, the
population, the inputs of each mode, the score and the failure rule. `official_scorer.py` records its SHA-256 in the
manifest of every scored run.

## Question and methods

The comparison measures two-dimensional inter-agent constellation error. Absolute coordinate error is not an endpoint.
It compares SOLID with Jin 2021, Empirical ADE-BP 2015, FNL 2025 and STRONG 2021, one implementation and one
configuration per method and mode. Each baseline runs with the fixed settings of
`research/benchmarks/datasets/baseline-configurations.json` and the options that its selection on the eleven training
scans chose per mode, and every method runs in each mode it can process (`baselines.json`); a method without an
estimator for a mode is reported as N/A, never as a numerical error.

## Data

The data are TWR-CLOUD-Bern version 6, [Zenodo record 23434390](https://zenodo.org/records/23434390).
`tools/fetch_data.py` downloads every file and checks it against these identities.

```text
Evaluation archive  TWR-CLOUD Bern Eval Scenarios.zip
                    SHA256 8617b54db218e0999ebda2f6c36107e5c79f3f5e7e84eb1047ecf997551c110b
                    MD5    757319458e2c159caddbb27bbc2d49c0
Released scorer     score_twr_cloud_bern.py
                    SHA256 0068d72b94db6b1c40414829390025a3b2b12e9b9d1fd7d939fff5fc97659073
                    MD5    dce91f7dcd472f4cfbabb46f1c49e012
Responder table     Rangings at Responders.pkl
                    SHA256 65c077bb1aa9b8793d9b206a406ecfd574047ad345fbafdbf93cfcdfad26fa21
                    MD5    d5d71fe55c758c2a87d7efd812357c5b
```

Every fit and every selection uses only the training scans 1 to 11 of the responder table, 1,947,886 rows. No evaluation
row enters a fit or a selection. One range calibration fitted on the training scans converts the released times of
flight to centimetres for every method (`runtime/engine/src/calibration.rs`), from the raw timings or from the
calibrated archive made with the same law. A held training scan is solved with the calibration and the fits of the
other ten.

The released archive also holds macOS resource-fork members that the released scorer cannot read.
`prepare_scenarios.py` removes only the `__MACOSX` members and keeps every scenario member in its original compressed
form; the fingerprint of the kept compressed members must equal that of the real members of the released archive.

## Population

The archive holds 148,260 physical scenarios, and both released iterations and both released sample counts (1 and 10)
are evaluated. AF uses the 24,710 scenarios without anchors, 49,420 units per sample count. AAC and ANC use the 123,550
scenarios with declared anchors, 247,100 units per sample count. No scenario or node is removed for its geometry,
observability, solver status or any method's outcome, and every method of a table column covers the same units.

## Inputs of each mode

AF receives the agent-to-agent ranges. AAC receives the agent-to-agent and agent-to-anchor ranges and the declared anchor
coordinates. ANC receives the agent-to-anchor ranges and the declared anchor coordinates only. Every method receives the
same information for a mode; a method may ignore part of it as its design states, and may process it with its own
initialization, starts, aggregation of repeated ranges, likelihood, optimization and fallback.

## Score

For each unit the released scorer takes the absolute error of every eligible estimated inter-agent distance in the x-y
plane and caps it at the true diameter of the agent constellation. A pair with a missing, invalid or nonfinite
coordinate receives the same cap, and a coordinate that is not finite or whose magnitude exceeds 1e7 cm is treated as
missing for every method alike. The unit's score is the mean of its eligible pair errors.

The released figure keeps the released eligibility: the pairs within the observed agent graph component in AF and all
agent pairs in AAC and ANC. The headline figure scores, with the released scorer's arithmetic, the pairs that the unit's
measured links join with every anchor counted as one node (`research/benchmarks/datasets/scoring.py`), which equals
the released eligibility in AF; a unit without such a pair leaves that figure. Both figures come from the same
predictions.

Reporting cells are defined by mode, sample count, declared anchor count and agent count. The balanced empirical
distribution gives every nonempty cell equal weight; the tables report its 75th percentile S75 in centimetres with S50
and S90, and the CDF figures show the same distributions.

The released diameter cap is the only failure rule of the tables and figures. Output success is counted per expected
agent and sample count, and the valid-output diagnostic (`valid_output_diagnostics.py`), which scores only pairs whose
endpoints have finite outputs, describes reliability and never ranks methods.

## Run outputs

Every run keeps its predictions in the released scorer's CSV format, the scorer's outputs and the valid-output
diagnostic, bound by SHA-256 in a manifest to the scenario archive, the scorer, this protocol, the method configuration
and the executable.
