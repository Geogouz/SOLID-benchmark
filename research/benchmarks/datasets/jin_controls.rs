use serde_json::{json, Value};
use solid_core::{
    Anchor, CalibratedRangeSample, LocalizationMode, Point2, PreparedProblem, RangeObservation,
};

pub(super) struct Control {
    pub name: &'static str,
    pub samples: usize,
    pub problem: PreparedProblem,
    pub reference: Vec<Point2>,
}

pub(super) fn cases() -> Vec<Control> {
    let mut cases = Vec::new();
    for (name, anchor_xy, shift) in [
        ("one", vec![(0.0, 0.0)], (0.0, 0.0)),
        ("one-translated", vec![(0.0, 0.0)], (500.0, -700.0)),
        ("two", vec![(0.0, 0.0), (400.0, 0.0)], (0.0, 0.0)),
        (
            "four-collinear",
            vec![(0.0, 0.0), (400.0, 0.0), (800.0, 0.0), (1200.0, 0.0)],
            (0.0, 0.0),
        ),
        (
            "three-noncollinear",
            vec![(0.0, 0.0), (800.0, 0.0), (0.0, 800.0)],
            (0.0, 0.0),
        ),
    ] {
        let reference: Vec<_> = [
            (100.0, 100.0),
            (300.0, 100.0),
            (100.0, 300.0),
            (300.0, 300.0),
        ]
        .map(|(x, y)| Point2::new(x + shift.0, y + shift.1))
        .to_vec();
        let anchors: Vec<_> = anchor_xy
            .iter()
            .enumerate()
            .map(|(i, &(x, y))| Anchor {
                id: 101 + i as u32,
                position: Point2::new(x + shift.0, y + shift.1),
                height_cm: 0.0,
            })
            .collect();
        for samples in [1, 10] {
            let mut observations = Vec::new();
            for (i, &point) in reference.iter().enumerate() {
                for (to, position, to_is_anchor) in
                    anchors.iter().map(|a| (a.id, a.position, true)).chain(
                        reference
                            .iter()
                            .enumerate()
                            .skip(i + 1)
                            .map(|(j, &p)| (j as u32 + 1, p, false)),
                    )
                {
                    observations.push(RangeObservation {
                        from: i as u32 + 1,
                        to,
                        to_is_anchor,
                        tof_ns: vec![],
                        calibrated_samples: vec![
                            CalibratedRangeSample::new(
                                point.distance(position),
                                10.0
                            );
                            samples
                        ],
                    });
                }
            }
            cases.push(Control {
                name,
                samples,
                reference: reference.clone(),
                problem: PreparedProblem {
                    attempts: None,
                    agent_ids: vec![1, 2, 3, 4],
                    anchors: anchors.clone(),
                    observations,
                    initial_positions: vec![Point2::new(-500.0 + shift.0, 800.0 + shift.1); 4],
                    mode: LocalizationMode::AnchorAided,
                },
            });
        }
    }
    cases
}

pub(super) fn pairs(reference: &[Point2], positions: &[Point2]) -> Vec<Value> {
    let mut pairs = Vec::new();
    for i in 0..reference.len() {
        for j in i + 1..reference.len() {
            let actual = reference[i].distance(reference[j]);
            let predicted = positions[i].distance(positions[j]);
            pairs.push(json!({"agents": [i + 1, j + 1], "reference_cm": actual,
                "predicted_cm": predicted, "absolute_error_cm": (predicted - actual).abs()}));
        }
    }
    pairs
}

pub(super) fn mean_error(pairs: &[Value]) -> f64 {
    pairs
        .iter()
        .map(|p| p["absolute_error_cm"].as_f64().unwrap())
        .sum::<f64>()
        / pairs.len() as f64
}
