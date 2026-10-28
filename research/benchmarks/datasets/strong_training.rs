use super::*;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Variant {
    name: String,
    solve_config: SolveConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Training {
    data_role: String,
    method_config: MethodConfig,
    variants: Vec<Variant>,
    units: Vec<Unit>,
}

#[test]
#[ignore = "explicit calibration-only threshold check"]
fn threshold_check() -> Result<()> {
    let request: Training =
        serde_json::from_reader(File::open(std::env::var("STRONG_TRAINING_INPUT")?)?)?;
    if request.data_role == "bregar-training-strong-threshold" {
        ensure!(
            request.units.len() == 6 && request.variants.len() == 4,
            "three paired raw blocks and four training thresholds required"
        );
        for unit in &request.units {
            ensure!(
                unit.problem.mode == LocalizationMode::AnchorOnly
                    && matches!(unit.samples, 1 | 10)
                    && matches!(unit.problem.anchors.len(), 7 | 8)
                    && unit.deployment_profile.is_none()
                    && unit.area.is_none(),
                "Bregar ANC training without transferred profiles or area priors"
            );
        }
    } else if request.data_role == "idlab-training-strong-threshold" {
        ensure!(
            request.units.len() == 16 && request.variants.len() == 4,
            "four paired anchor counts in both modes and four thresholds required"
        );
        let mut keys = BTreeSet::new();
        for unit in &request.units {
            let p = &unit.problem;
            let agents: BTreeSet<_> = p.agent_ids.iter().copied().collect();
            let anchors: BTreeSet<_> = p.anchors.iter().map(|a| a.id).collect();
            ensure!(
                matches!(unit.samples, 1 | 10)
                    && matches!(anchors.len(), 1 | 4 | 8 | 13)
                    && agents.len() == p.agent_ids.len()
                    && anchors.len() == p.anchors.len()
                    && agents.is_disjoint(&anchors)
                    && agents.union(&anchors).copied().collect::<BTreeSet<_>>()
                        == (1..=15).collect()
                    && unit.area.is_none()
                    && unit.deployment_profile.is_none()
                    && keys.insert((
                        p.mode == LocalizationMode::AnchorOnly,
                        anchors.len(),
                        unit.samples
                    )),
                "IDLab training retains every node and paired input without extra priors"
            );
        }
    } else {
        ensure!(
            request.data_role == "synthetic-training-strong-threshold"
                && request.units.len() == 60
                && request.variants.len() == 4,
            "bounded training check only"
        );
    }
    ensure!(
        matches!(request.method_config, MethodConfig::Strong { .. }),
        "STRONG only"
    );
    let method = MethodRunner::configured(MethodKind::Strong, request.method_config)?;
    let mut writer = BufWriter::new(File::create_new(std::env::var("STRONG_TRAINING_OUTPUT")?)?);
    for unit in request.units {
        ensure!(
            unit.problem.mode != LocalizationMode::AnchorFree,
            "anchored training only"
        );
        ensure!(
            unit.problem.agent_ids.len() >= 2 && !unit.problem.anchors.is_empty(),
            "anchored agents required"
        );
        for variant in &request.variants {
            let config = &variant.solve_config;
            ensure!(
                config.max_time_micros == 0
                    && config.robust_delta > 0.0
                    && config.dimension_policy == DimensionPolicy::Spatial,
                "fixed spatial solver without time cap"
            );
            let started = Instant::now();
            let result = method
                .solve(&unit.problem, config)
                .map(|r| {
                    json!({
                        "positions": r.positions.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>(),
                        "agent_ids": r.agent_ids, "metrics": r.metrics
                    })
                })
                .map_err(Into::into);
            write_record(&mut writer, &unit, &variant.name, started, result)?;
        }
        writer.flush()?;
    }
    writer.get_ref().sync_all()?;
    Ok(())
}
