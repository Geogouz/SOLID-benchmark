use std::ffi::{c_char, CString};
use std::slice;

use serde::{Deserialize, Serialize};
use solid_core::{
    CooperativeProfile, DeploymentProfile, LocalizationEngine, PreparedProblem, SolveResult,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FfiRequest {
    problem: PreparedProblem,
    profile: Option<FfiProfile>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum FfiProfile {
    Cooperative(CooperativeProfile),
    Anc(DeploymentProfile),
}

#[derive(Serialize)]
struct FfiResponse {
    result: Option<SolveResult>,
    error: Option<String>,
}

#[no_mangle]
/// Solves one localization request with the fitted profile of its mode.
///
/// # Safety
///
/// `input` must reference `input_len` readable bytes. The returned pointer must
/// be released exactly once with [`solid_string_free`].
pub unsafe extern "C" fn solid_solve_json(input: *const u8, input_len: usize) -> *mut c_char {
    let response = if input.is_null() {
        error_response("input pointer is null")
    } else {
        let input = unsafe { slice::from_raw_parts(input, input_len) };
        solve_json(input)
    };
    CString::new(response)
        .expect("serialized JSON contains no null bytes")
        .into_raw()
}

#[no_mangle]
/// Releases a response returned by [`solid_solve_json`].
///
/// # Safety
///
/// `value` must be null or a live pointer returned by [`solid_solve_json`] that
/// has not already been released.
pub unsafe extern "C" fn solid_string_free(value: *mut c_char) {
    if !value.is_null() {
        unsafe {
            drop(CString::from_raw(value));
        }
    }
}

fn solve_json(input: &[u8]) -> String {
    let request: FfiRequest = match serde_json::from_slice(input) {
        Ok(request) => request,
        Err(error) => return error_response(&format!("invalid request JSON. {error}")),
    };
    let engine = LocalizationEngine::new();
    let solved = match request.profile {
        Some(FfiProfile::Cooperative(profile)) => {
            engine.solve_with_cooperative_profile(&request.problem, profile)
        }
        Some(FfiProfile::Anc(profile)) => engine.solve_with_profile(&request.problem, profile),
        None => return error_response("every request needs the fitted profile of its mode"),
    };
    match solved {
        Ok(result) => serialize_response(FfiResponse {
            result: Some(result),
            error: None,
        }),
        Err(error) => error_response(&error.to_string()),
    }
}

fn error_response(error: &str) -> String {
    serialize_response(FfiResponse {
        result: None,
        error: Some(error.to_owned()),
    })
}

fn serialize_response(response: FfiResponse) -> String {
    serde_json::to_string(&response).expect("FFI response serialization cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use solid_core::{
        Anchor, AnchorOnlyProfile, LocalizationMode, PlanarArea, Point2, RangeObservation,
    };

    #[test]
    fn reports_invalid_json() {
        let response: serde_json::Value = serde_json::from_str(&solve_json(b"{")).unwrap();
        assert!(response["result"].is_null());
        assert!(response["error"]
            .as_str()
            .unwrap()
            .starts_with("invalid request JSON"));
    }

    #[test]
    fn rejects_solver_configuration_overrides() {
        let request = serde_json::json!({
            "problem": {
                "agent_ids": [],
                "anchors": [],
                "observations": [],
                "initial_positions": [],
                "mode": "AnchorFree"
            },
            "config": {"dimension_policy": "spatial"}
        });
        let response: serde_json::Value =
            serde_json::from_str(&solve_json(request.to_string().as_bytes())).unwrap();

        assert!(response["result"].is_null());
        assert!(response["error"]
            .as_str()
            .unwrap()
            .contains("unknown field `config`"));
    }

    #[test]
    fn accepts_a_deployment_profile_for_anchor_only() {
        let problem = PreparedProblem {
            attempts: None,
            agent_ids: vec![1],
            anchors: vec![
                anchor(10, 0.0, 0.0),
                anchor(11, 300.0, 0.0),
                anchor(12, 0.0, 300.0),
                anchor(13, 300.0, 300.0),
            ],
            observations: vec![observation(10, 4.0), observation(11, 5.0)],
            initial_positions: vec![Point2::default()],
            mode: LocalizationMode::AnchorOnly,
        };
        let request = serde_json::json!({
            "problem": problem,
            "profile": deployment_profile()
        });

        let response: serde_json::Value =
            serde_json::from_str(&solve_json(request.to_string().as_bytes())).unwrap();

        assert!(response["error"].is_null(), "{}", response["error"]);
        assert_eq!(response["result"]["agent_ids"][0], 1);
    }

    fn anchor(id: u32, x: f64, y: f64) -> Anchor {
        Anchor {
            id,
            position: Point2::new(x, y),
            height_cm: 0.0,
        }
    }

    fn observation(anchor: u32, tof_ns: f64) -> RangeObservation {
        RangeObservation {
            from: 1,
            to: anchor,
            tof_ns: vec![tof_ns],
            calibrated_samples: Vec::new(),
            to_is_anchor: true,
        }
    }

    fn deployment_profile() -> DeploymentProfile {
        DeploymentProfile {
            area: PlanarArea {
                minimum_x_cm: -50.0,
                maximum_x_cm: 350.0,
                minimum_y_cm: -50.0,
                maximum_y_cm: 350.0,
            },
            anchor_only: AnchorOnlyProfile::default(),
        }
    }

    fn cooperative_request(mode: LocalizationMode) -> serde_json::Value {
        serde_json::json!({
            "problem": {
                "agent_ids": [1, 2], "anchors": [],
                "observations": [{"from": 1, "to": 2, "to_is_anchor": false, "tof_ns": [5.2, 6.8]}],
                "initial_positions": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}],
                "mode": mode,
            },
            "profile": {
                "range_error": {"positive_scale": 1.2, "negative_scale": 0.8},
                "starts": 32,
                "presence": {"shape": "gaussian", "quantile": 0.9, "factor": 0.75, "width": 0.15},
            },
        })
    }

    fn response(request: &serde_json::Value) -> serde_json::Value {
        serde_json::from_str(&solve_json(request.to_string().as_bytes())).unwrap()
    }

    #[test]
    fn cooperative_profiles_reach_the_selected_solver() {
        for mode in [LocalizationMode::AnchorFree, LocalizationMode::AnchorAided] {
            let request = cooperative_request(mode);
            let problem = serde_json::from_value(request["problem"].clone()).unwrap();
            let profile = serde_json::from_value(request["profile"].clone()).unwrap();
            let mut expected = LocalizationEngine::new()
                .solve_with_cooperative_profile(&problem, profile)
                .unwrap();
            expected.metrics.elapsed_micros = 0;
            let result = response(&request);
            assert!(result["error"].is_null());
            let mut actual: SolveResult = serde_json::from_value(result["result"].clone()).unwrap();
            actual.metrics.elapsed_micros = 0;
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn a_request_without_a_profile_is_rejected() {
        for mode in [
            LocalizationMode::AnchorFree,
            LocalizationMode::AnchorAided,
            LocalizationMode::AnchorOnly,
        ] {
            let mut request = cooperative_request(mode);
            request.as_object_mut().unwrap().remove("profile");
            let result = response(&request);
            assert!(result["result"].is_null());
            assert_eq!(
                result["error"],
                "every request needs the fitted profile of its mode"
            );
        }
    }

    #[test]
    fn invalid_or_incompatible_profiles_are_rejected() {
        let mut request = cooperative_request(LocalizationMode::AnchorFree);
        request["profile"]["presence"]["factor"] = serde_json::json!(0.0);
        assert!(response(&request)["error"]
            .as_str()
            .unwrap()
            .contains("presence"));
        request["profile"]["presence"]["factor"] = serde_json::json!(0.75);
        request["problem"]["mode"] = serde_json::json!("AnchorAided");
        assert!(response(&request)["error"].is_null());
        request["profile"]["range_error"]["positive_scale"] = serde_json::json!(0.0);
        assert!(response(&request)["result"].is_null());
        request = cooperative_request(LocalizationMode::AnchorFree);
        request["profile"]["requested_starts"] = serde_json::json!(1);
        assert!(response(&request)["result"].is_null());
        request = cooperative_request(LocalizationMode::AnchorFree);
        request["profile"] = serde_json::to_value(deployment_profile()).unwrap();
        assert!(response(&request)["error"]
            .as_str()
            .unwrap()
            .contains("ANC profile"));
        request = cooperative_request(LocalizationMode::AnchorOnly);
        assert!(response(&request)["result"].is_null());
    }
}
