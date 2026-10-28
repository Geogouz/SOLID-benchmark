use std::collections::BTreeMap;

use thiserror::Error;

use super::{
    balanced_percentile, mean, nonnegative_finite, passes_inclusive_ratio_limit,
    BalancedPercentile, InclusiveRatioLimit, Neumaier, ScoringError,
};

const BERN_TRAINING_SAMPLES: &[u32] = &[1, 5, 10, 15, 20];
const BERN_EVALUATION_SAMPLES: &[u32] = &[1, 10];
const CTTC_SAMPLES: &[u32] = &[1, 5, 10, 15, 20];
const MILUV_SAMPLES: &[u32] = &[1];
const BREGAR_SAMPLES: &[u32] = &[1, 5, 10, 15, 20];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum G11Dataset {
    BernTraining,
    BernEvaluation,
    Cttc,
    Miluv,
    Bregar,
}

impl G11Dataset {
    pub const fn sample_counts(self) -> &'static [u32] {
        match self {
            Self::BernTraining => BERN_TRAINING_SAMPLES,
            Self::BernEvaluation => BERN_EVALUATION_SAMPLES,
            Self::Cttc => CTTC_SAMPLES,
            Self::Miluv => MILUV_SAMPLES,
            Self::Bregar => BREGAR_SAMPLES,
        }
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum G11Cell {
    BernTraining {
        scan_id: u8,
        anchor_count: u32,
        agent_count: u32,
    },
    BernEvaluation {
        anchor_count: u32,
        agent_count: u32,
    },
    Cttc {
        partition_identifier: String,
    },
    Miluv,
    Bregar {
        environment_identifier: String,
        channel_identifier: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct G11Case {
    initialization_identifier: u8,
    sample_count: u32,
    cell: G11Cell,
    error_cm: f64,
}

impl G11Case {
    pub fn new(
        initialization_identifier: u8,
        sample_count: u32,
        cell: G11Cell,
        error_cm: f64,
    ) -> Result<Self, G11Error> {
        if !matches!(initialization_identifier, 1 | 2) || sample_count == 0 {
            return Err(G11Error::InvalidCase);
        }
        validate_cell(&cell)?;
        let error_cm = nonnegative_finite(error_cm)?;
        Ok(Self {
            initialization_identifier,
            sample_count,
            cell,
            error_cm,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct G11Comparison {
    pub initialization_identifier: u8,
    pub sample_count: u32,
    pub candidate_t90_cm: f64,
    pub jin_t90_cm: f64,
    pub passes: bool,
}

#[derive(Debug, Error, Eq, PartialEq)]
pub enum G11Error {
    #[error(transparent)]
    Score(#[from] ScoringError),
    #[error("G11 case metadata is invalid for the declared dataset")]
    InvalidCase,
    #[error("G11 is missing an expected initialization and sample slice")]
    MissingSlice,
    #[error("Bern training G11 requires every scan from 01 through 11")]
    MissingBernScan,
}

pub fn assess_g11(
    dataset: G11Dataset,
    candidate_cases: &[G11Case],
    jin_cases: &[G11Case],
) -> Result<Vec<G11Comparison>, G11Error> {
    validate_dataset_cases(dataset, candidate_cases)?;
    validate_dataset_cases(dataset, jin_cases)?;
    let mut comparisons = Vec::with_capacity(dataset.sample_counts().len() * 2);
    for initialization_identifier in [1, 2] {
        for &sample_count in dataset.sample_counts() {
            let candidate_t90_cm = t90(
                dataset,
                candidate_cases,
                initialization_identifier,
                sample_count,
            )?;
            let jin_t90_cm = t90(dataset, jin_cases, initialization_identifier, sample_count)?;
            comparisons.push(G11Comparison {
                initialization_identifier,
                sample_count,
                candidate_t90_cm,
                jin_t90_cm,
                passes: passes_inclusive_ratio_limit(
                    candidate_t90_cm,
                    jin_t90_cm,
                    InclusiveRatioLimit::TwoPercent,
                )?,
            });
        }
    }
    Ok(comparisons)
}

fn t90(
    dataset: G11Dataset,
    cases: &[G11Case],
    initialization_identifier: u8,
    sample_count: u32,
) -> Result<f64, G11Error> {
    let selected = cases
        .iter()
        .filter(|case| {
            case.initialization_identifier == initialization_identifier
                && case.sample_count == sample_count
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return Err(G11Error::MissingSlice);
    }
    match dataset {
        G11Dataset::BernTraining => bern_training_t90(&selected),
        G11Dataset::BernEvaluation => balanced_t90(selected, |cell| match cell {
            G11Cell::BernEvaluation {
                anchor_count,
                agent_count,
            } => Some((*anchor_count, *agent_count)),
            _ => None,
        }),
        G11Dataset::Cttc => balanced_t90(selected, |cell| match cell {
            G11Cell::Cttc {
                partition_identifier,
            } => Some(partition_identifier.clone()),
            _ => None,
        }),
        G11Dataset::Miluv => balanced_t90(selected, |cell| match cell {
            G11Cell::Miluv => Some(()),
            _ => None,
        }),
        G11Dataset::Bregar => balanced_t90(selected, |cell| match cell {
            G11Cell::Bregar {
                environment_identifier,
                channel_identifier,
            } => Some((environment_identifier.clone(), channel_identifier.clone())),
            _ => None,
        }),
    }
}

fn bern_training_t90(cases: &[&G11Case]) -> Result<f64, G11Error> {
    let mut scans = BTreeMap::<u8, BTreeMap<(u32, u32), Vec<f64>>>::new();
    for case in cases {
        let G11Cell::BernTraining {
            scan_id,
            anchor_count,
            agent_count,
        } = case.cell
        else {
            return Err(G11Error::InvalidCase);
        };
        scans
            .entry(scan_id)
            .or_default()
            .entry((anchor_count, agent_count))
            .or_default()
            .push(case.error_cm);
    }
    if scans.keys().copied().collect::<Vec<_>>() != (1..=11).collect::<Vec<_>>() {
        return Err(G11Error::MissingBernScan);
    }
    let mut scan_sum = Neumaier::default();
    for cells in scans.values() {
        scan_sum.add(balanced_percentile(cells, BalancedPercentile::P90)?)?;
    }
    Ok(mean(scan_sum, scans.len())?)
}

fn balanced_t90<K: Ord, F>(cases: Vec<&G11Case>, key: F) -> Result<f64, G11Error>
where
    F: Fn(&G11Cell) -> Option<K>,
{
    let mut cells = BTreeMap::<K, Vec<f64>>::new();
    for case in cases {
        cells
            .entry(key(&case.cell).ok_or(G11Error::InvalidCase)?)
            .or_default()
            .push(case.error_cm);
    }
    Ok(balanced_percentile(&cells, BalancedPercentile::P90)?)
}

fn validate_dataset_cases(dataset: G11Dataset, cases: &[G11Case]) -> Result<(), G11Error> {
    if cases.iter().any(|case| {
        !dataset.sample_counts().contains(&case.sample_count)
            || !matches!(
                (dataset, &case.cell),
                (G11Dataset::BernTraining, G11Cell::BernTraining { .. })
                    | (G11Dataset::BernEvaluation, G11Cell::BernEvaluation { .. })
                    | (G11Dataset::Cttc, G11Cell::Cttc { .. })
                    | (G11Dataset::Miluv, G11Cell::Miluv)
                    | (G11Dataset::Bregar, G11Cell::Bregar { .. })
            )
    }) {
        return Err(G11Error::InvalidCase);
    }
    Ok(())
}

fn validate_cell(cell: &G11Cell) -> Result<(), G11Error> {
    let valid = match cell {
        G11Cell::BernTraining {
            scan_id,
            anchor_count,
            agent_count,
        } => (1..=11).contains(scan_id) && *anchor_count > 0 && *agent_count > 0,
        G11Cell::BernEvaluation {
            anchor_count,
            agent_count,
        } => *anchor_count > 0 && *agent_count > 0,
        G11Cell::Cttc {
            partition_identifier,
        } => !partition_identifier.is_empty(),
        G11Cell::Miluv => true,
        G11Cell::Bregar {
            environment_identifier,
            channel_identifier,
        } => !environment_identifier.is_empty() && !channel_identifier.is_empty(),
    };
    valid.then_some(()).ok_or(G11Error::InvalidCase)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete_cases(dataset: G11Dataset, base: f64) -> Vec<G11Case> {
        let mut cases = Vec::new();
        for initialization in [1, 2] {
            for &sample_count in dataset.sample_counts() {
                match dataset {
                    G11Dataset::BernTraining => {
                        for scan_id in 1..=11 {
                            cases.push(
                                G11Case::new(
                                    initialization,
                                    sample_count,
                                    G11Cell::BernTraining {
                                        scan_id,
                                        anchor_count: 4,
                                        agent_count: 2,
                                    },
                                    base + f64::from(scan_id),
                                )
                                .unwrap(),
                            );
                        }
                    }
                    G11Dataset::BernEvaluation => cases.push(
                        G11Case::new(
                            initialization,
                            sample_count,
                            G11Cell::BernEvaluation {
                                anchor_count: 4,
                                agent_count: 2,
                            },
                            base,
                        )
                        .unwrap(),
                    ),
                    G11Dataset::Cttc => cases.push(
                        G11Case::new(
                            initialization,
                            sample_count,
                            G11Cell::Cttc {
                                partition_identifier: "C1".to_owned(),
                            },
                            base,
                        )
                        .unwrap(),
                    ),
                    G11Dataset::Miluv => cases.push(
                        G11Case::new(initialization, sample_count, G11Cell::Miluv, base).unwrap(),
                    ),
                    G11Dataset::Bregar => cases.push(
                        G11Case::new(
                            initialization,
                            sample_count,
                            G11Cell::Bregar {
                                environment_identifier: "E1".to_owned(),
                                channel_identifier: "C1".to_owned(),
                            },
                            base,
                        )
                        .unwrap(),
                    ),
                }
            }
        }
        cases
    }

    #[test]
    fn every_dataset_rule_emits_every_expected_slice() {
        for dataset in [
            G11Dataset::BernTraining,
            G11Dataset::BernEvaluation,
            G11Dataset::Cttc,
            G11Dataset::Miluv,
            G11Dataset::Bregar,
        ] {
            let candidate = complete_cases(dataset, 1.0);
            let jin = complete_cases(dataset, 2.0);
            let comparisons = assess_g11(dataset, &candidate, &jin).unwrap();
            assert_eq!(comparisons.len(), dataset.sample_counts().len() * 2);
            assert!(comparisons.iter().all(|comparison| comparison.passes));
        }
    }

    #[test]
    fn bern_training_balances_geometry_within_scan_then_means_scans() {
        let mut cases = complete_cases(G11Dataset::BernTraining, 0.0);
        cases.extend([
            G11Case::new(
                1,
                1,
                G11Cell::BernTraining {
                    scan_id: 1,
                    anchor_count: 5,
                    agent_count: 3,
                },
                100.0,
            )
            .unwrap(),
            G11Case::new(
                1,
                1,
                G11Cell::BernTraining {
                    scan_id: 1,
                    anchor_count: 5,
                    agent_count: 3,
                },
                100.0,
            )
            .unwrap(),
        ]);
        assert_eq!(t90(G11Dataset::BernTraining, &cases, 1, 1).unwrap(), 15.0);
    }

    #[test]
    fn missing_slice_and_missing_bern_scan_fail_closed() {
        let mut cases = complete_cases(G11Dataset::BernTraining, 1.0);
        cases.retain(|case| !(case.initialization_identifier == 2 && case.sample_count == 20));
        assert_eq!(
            assess_g11(G11Dataset::BernTraining, &cases, &cases),
            Err(G11Error::MissingSlice)
        );

        let mut cases = complete_cases(G11Dataset::BernTraining, 1.0);
        cases.retain(|case| !matches!(case.cell, G11Cell::BernTraining { scan_id: 11, .. }));
        assert_eq!(
            assess_g11(G11Dataset::BernTraining, &cases, &cases),
            Err(G11Error::MissingBernScan)
        );
    }

    #[test]
    fn exact_two_percent_boundary_passes_and_next_ulp_fails() {
        let jin = complete_cases(G11Dataset::Miluv, 1.0);
        let passing = complete_cases(G11Dataset::Miluv, 1.02);
        assert!(assess_g11(G11Dataset::Miluv, &passing, &jin)
            .unwrap()
            .iter()
            .all(|comparison| comparison.passes));

        let failing = complete_cases(G11Dataset::Miluv, f64::from_bits(1.02_f64.to_bits() + 1));
        assert!(assess_g11(G11Dataset::Miluv, &failing, &jin)
            .unwrap()
            .iter()
            .all(|comparison| !comparison.passes));
    }
}
