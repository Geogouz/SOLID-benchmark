# Runtime

The deployable SOLID product.

| Path | Purpose |
| --- | --- |
| `engine` | The `solid-core` crate, a truth free numerical localization engine |
| `ffi` | C compatible static and dynamic libraries around it |

The engine accepts a `PreparedProblem` with range observations, optional fixed anchors, starting positions, a localization mode and optionally the ranging attempts that the presence law reads, together with the fitted profile of that mode, and returns estimated horizontal positions with solver diagnostics. There is no ground truth field.

## Interface

```rust
let engine = LocalizationEngine::new();
let cooperative = engine.solve_with_cooperative_profile(&problem, profile)?;   // AF and AAC
let anchor_only = engine.solve_with_profile(&problem, deployment)?;           // ANC
```

`CooperativeProfile` holds, for one sample track, the two multipliers of the median first stage, the block likelihood of the link model with its height prior, the number of starts, the presence law and the law of raw timings. `DeploymentProfile` holds the operating area of the unit and the anchor only estimator's link model, height prior, budgets, decision rule and presence law. Both are fitted offline on a site's training half and fixed during inference. A request without the profile of its mode is an error.

Raw time of flight observations are converted with the law a cooperative profile names, and in the anchor only mode always with the engine's built-in TWR-CLOUD-Bern law. Calibrated samples with their own scales are accepted from any ranging technology. The paper, which [CITATION.cff](../CITATION.cff) cites, and its companion document in [results/](../results) state the
method as the engine computes it.

## Declared conventions

The cooperative solver keeps three conventions that differ between its stages or its modes. They are the engine's declared conventions, and the code keeps them as they are.

| Convention | What the engine does | When it matters |
| --- | --- | --- |
| Metrics of a two stage solve | A profile with `refine_from_median` runs the median residual stage before the block stage. The metrics a solve reports, its iterations, evaluations, starts, convergence and termination, are those of its last stage, the block stage, and those of the median stage are not kept (`cooperative/mod.rs`, `solve`). | The positions do not depend on it. A median stage that ended without converging does not show in the metrics. |
| Scale of a link with calibrated samples | AF takes the scale of the sample at the median distance of the link, and the mean scale of the two middle samples for an even count (`cooperative/af.rs`). AAC takes the median of the link's sample scales sorted on their own (`cooperative/aac.rs`). The samples of a link are sorted by distance and then by scale. | Both give the same scale whenever the scales of a link's samples rise or fall with their distances, a constant scale per link among them, and they can differ otherwise. |
| Typical range of the multistart | The jitter of the cooperative multistart grows with the typical range of the measured links. AF takes their median, the mean of the two middle values for an even count. AAC takes the upper of the two middle values for an even count (`typical_range` in both files). | Both agree for an odd number of measured links. For an even number the jitter of AAC's starts is at least as wide as AF's rule would make it. |

## Build and test

```sh
cargo test --manifest-path runtime/Cargo.toml --workspace --locked
```
