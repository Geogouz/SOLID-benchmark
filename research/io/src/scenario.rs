use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde::Deserialize;
use serde_json::Value;
use solid_core::{
    Anchor, CalibratedRangeSample, LocalizationMode, Point2, PreparedProblem, RangeObservation,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ScenarioError {
    #[error("failed to read scenario {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("invalid scenario JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("scenario has no iteration {0}")]
    MissingIteration(u32),
    #[error("invalid node pair {0}")]
    InvalidPair(String),
    #[error("invalid node ID {0}")]
    InvalidNodeId(String),
}

#[derive(Clone, Debug, Deserialize)]
pub struct EvaluationScenario {
    #[serde(rename = "Research Metadata", default)]
    pub research_metadata: Option<ResearchMetadata>,
    #[serde(rename = "Agents")]
    pub agents: Vec<u32>,
    #[serde(rename = "Anchors")]
    pub anchors: Vec<u32>,
    #[serde(rename = "True Positions of Agents")]
    pub true_positions_of_agents: BTreeMap<String, Position>,
    #[serde(rename = "True Positions of Anchors")]
    pub true_positions_of_anchors: BTreeMap<String, Position>,
    #[serde(rename = "Iterations")]
    pub iterations: Vec<Iteration>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ResearchMetadata {
    pub schema_version: String,
    pub data_role: String,
    pub scan_id: u32,
    pub scenario_id: u32,
    pub group_index: usize,
    pub sample_counts: Vec<u32>,
    pub seed: u64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Iteration {
    #[serde(rename = "ID")]
    pub id: u32,
    #[serde(rename = "Initial positions of Agents")]
    pub initial_positions_of_agents: BTreeMap<String, Position>,
    #[serde(rename = "Ranging samples between Agents")]
    pub ranging_samples_between_agents: BTreeMap<String, BTreeMap<String, Value>>,
    #[serde(rename = "Ranging samples to Anchors")]
    pub ranging_samples_to_anchors: BTreeMap<String, BTreeMap<String, Value>>,
}

impl EvaluationScenario {
    pub fn from_path(path: &Path) -> Result<Self, ScenarioError> {
        let contents = fs::read_to_string(path).map_err(|source| ScenarioError::Read {
            path: path.display().to_string(),
            source,
        })?;
        Self::from_slice(contents.as_bytes())
    }

    pub fn from_slice(contents: &[u8]) -> Result<Self, ScenarioError> {
        Ok(serde_json::from_slice(contents)?)
    }

    pub fn prepare_problem(
        &self,
        iteration_id: u32,
        samples: u32,
        mode: LocalizationMode,
    ) -> Result<PreparedProblem, ScenarioError> {
        let iteration = self
            .iterations
            .iter()
            .find(|iteration| iteration.id == iteration_id)
            .ok_or(ScenarioError::MissingIteration(iteration_id))?;
        let anchors = if mode == LocalizationMode::AnchorFree {
            Vec::new()
        } else {
            self.true_positions_of_anchors
                .iter()
                .map(|(id, position)| {
                    Ok(Anchor {
                        id: id
                            .parse()
                            .map_err(|_| ScenarioError::InvalidNodeId(id.clone()))?,
                        position: Point2::new(position.x, position.y),
                        height_cm: position.z,
                    })
                })
                .collect::<Result<_, ScenarioError>>()?
        };
        let initial_positions = self
            .agents
            .iter()
            .map(|id| {
                iteration
                    .initial_positions_of_agents
                    .get(&id.to_string())
                    .map_or(Point2::default(), |position| {
                        Point2::new(position.x, position.y)
                    })
            })
            .collect();
        let mut observations = Vec::new();

        if mode != LocalizationMode::AnchorOnly {
            append_observations(
                &mut observations,
                &iteration.ranging_samples_between_agents,
                samples,
                false,
            )?;
        }
        if mode != LocalizationMode::AnchorFree {
            append_observations(
                &mut observations,
                &iteration.ranging_samples_to_anchors,
                samples,
                true,
            )?;
        }

        Ok(PreparedProblem {
            attempts: None,
            agent_ids: self.agents.clone(),
            anchors,
            observations,
            initial_positions,
            mode,
        })
    }

    pub fn true_agent_positions(&self) -> Result<BTreeMap<u32, Point2>, ScenarioError> {
        parse_positions(&self.true_positions_of_agents)
    }

    pub fn true_anchor_positions(&self) -> Result<BTreeMap<u32, Point2>, ScenarioError> {
        parse_positions(&self.true_positions_of_anchors)
    }
}

fn parse_positions(
    positions: &BTreeMap<String, Position>,
) -> Result<BTreeMap<u32, Point2>, ScenarioError> {
    positions
        .iter()
        .map(|(id, position)| {
            Ok((
                id.parse()
                    .map_err(|_| ScenarioError::InvalidNodeId(id.clone()))?,
                Point2::new(position.x, position.y),
            ))
        })
        .collect()
}

fn append_observations(
    target: &mut Vec<RangeObservation>,
    source: &BTreeMap<String, BTreeMap<String, Value>>,
    samples: u32,
    to_is_anchor: bool,
) -> Result<(), ScenarioError> {
    for (pair, configurations) in source {
        let Some(value) = configurations.get(&samples.to_string()) else {
            continue;
        };
        let (tof_ns, calibrated_samples) = parse_samples(value)?;
        if tof_ns.is_empty() && calibrated_samples.is_empty() {
            continue;
        }
        let (from, to) = parse_pair(pair)?;
        target.push(RangeObservation {
            from,
            to,
            tof_ns,
            calibrated_samples,
            to_is_anchor,
        });
    }
    Ok(())
}

fn parse_pair(pair: &str) -> Result<(u32, u32), ScenarioError> {
    let (from, to) = pair
        .split_once('-')
        .ok_or_else(|| ScenarioError::InvalidPair(pair.to_owned()))?;
    let from = from
        .parse()
        .map_err(|_| ScenarioError::InvalidPair(pair.to_owned()))?;
    let to = to
        .parse()
        .map_err(|_| ScenarioError::InvalidPair(pair.to_owned()))?;
    Ok((from, to))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalibratedSampleSet {
    calibrated_samples: Vec<CalibratedRangeSample>,
}

fn parse_samples(value: &Value) -> Result<(Vec<f64>, Vec<CalibratedRangeSample>), ScenarioError> {
    match value {
        Value::Number(number) => Ok((number.as_f64().into_iter().collect(), Vec::new())),
        Value::Array(values) => Ok((
            values.iter().filter_map(Value::as_f64).collect(),
            Vec::new(),
        )),
        Value::Object(_) => {
            let samples: CalibratedSampleSet = serde_json::from_value(value.clone())?;
            Ok((Vec::new(), samples.calibrated_samples))
        }
        _ => Ok((Vec::new(), Vec::new())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_scenario() {
        let scenario = EvaluationScenario::from_slice(
            br#"{
              "Agents":[1,2], "Anchors":[10],
              "True Positions of Agents":{"1":{"x":0,"y":0,"z":0},"2":{"x":1,"y":0,"z":0}},
              "True Positions of Anchors":{"10":{"x":0,"y":1,"z":0}},
              "Iterations":[{"ID":1,"Initial positions of Agents":{"1":{"x":0,"y":0,"z":0},"2":{"x":0,"y":0,"z":0}},
              "Ranging samples between Agents":{"1-2":{"1":3.2,"10":[3.1,3.2]}},
              "Ranging samples to Anchors":{"1-10":{"1":4.2,"10":[4.1,4.2]}}}]
            }"#,
        )
        .unwrap();
        let problem = scenario
            .prepare_problem(1, 10, LocalizationMode::AnchorAided)
            .unwrap();
        assert_eq!(problem.observations.len(), 2);
        assert_eq!(problem.observations[0].tof_ns, vec![3.1, 3.2]);
        assert!(problem.observations[0].calibrated_samples.is_empty());
    }

    #[test]
    fn parses_calibrated_samples_without_recalibration() {
        let scenario = EvaluationScenario::from_slice(
            br#"{
              "Agents":[1], "Anchors":[10],
              "True Positions of Agents":{"1":{"x":0,"y":0,"z":0}},
              "True Positions of Anchors":{"10":{"x":0,"y":1,"z":0}},
              "Iterations":[{"ID":1,"Initial positions of Agents":{"1":{"x":0,"y":0,"z":0}},
              "Ranging samples between Agents":{},
              "Ranging samples to Anchors":{"1-10":{"5":{"calibrated_samples":[
                {"distance_cm":123.0,"scale_cm":8.0},
                {"distance_cm":125.0,"scale_cm":9.0}
              ]}}}}]
            }"#,
        )
        .unwrap();
        let problem = scenario
            .prepare_problem(1, 5, LocalizationMode::AnchorOnly)
            .unwrap();

        assert_eq!(problem.observations.len(), 1);
        assert!(problem.observations[0].tof_ns.is_empty());
        assert_eq!(
            problem.observations[0].calibrated_samples,
            vec![
                CalibratedRangeSample::new(123.0, 8.0),
                CalibratedRangeSample::new(125.0, 9.0),
            ]
        );
    }
}
