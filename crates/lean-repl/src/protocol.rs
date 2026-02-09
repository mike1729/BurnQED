use serde::{Deserialize, Serialize};

/// A request to send to Pantograph via JSON lines.
///
/// Pantograph expects `{"cmd": "<command>", "payload": {<args>}}` format.
#[derive(Debug, Clone)]
pub enum PantographRequest {
    /// Start a new proof environment for an expression.
    GoalStart { expr: String },
    /// Apply a tactic to a goal within a proof state.
    GoalTactic {
        state_id: u64,
        goal_id: Option<u64>,
        tactic: String,
    },
}

/// Wire format for the outer command envelope.
#[derive(Serialize)]
struct CommandWire {
    cmd: &'static str,
    payload: serde_json::Value,
}

/// Wire format for GoalStart payload.
#[derive(Serialize)]
struct GoalStartPayload {
    expr: String,
}

/// Wire format for GoalTactic payload.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GoalTacticPayload {
    state_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "goalId")]
    goal_id: Option<u64>,
    tactic: String,
}

impl PantographRequest {
    /// Serialize this request to a JSON string suitable for Pantograph.
    ///
    /// Format: `{"cmd": "goal.start", "payload": {"expr": "..."}}`
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        match self {
            PantographRequest::GoalStart { expr } => {
                let payload = serde_json::to_value(GoalStartPayload { expr: expr.clone() })?;
                serde_json::to_string(&CommandWire {
                    cmd: "goal.start",
                    payload,
                })
            }
            PantographRequest::GoalTactic {
                state_id,
                goal_id,
                tactic,
            } => {
                let payload = serde_json::to_value(GoalTacticPayload {
                    state_id: *state_id,
                    goal_id: *goal_id,
                    tactic: tactic.clone(),
                })?;
                serde_json::to_string(&CommandWire {
                    cmd: "goal.tactic",
                    payload,
                })
            }
        }
    }
}

// --- Response types ---

/// A goal variable (hypothesis) in a Pantograph response.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PantographVariable {
    /// Internal name.
    pub name: String,
    /// User-facing name.
    pub user_name: String,
    /// Whether the name contains a dagger (is inaccessible).
    #[serde(default)]
    pub is_inaccessible: bool,
    /// Type expression.
    #[serde(rename = "type")]
    pub type_expr: Option<PantographExpression>,
    /// Value expression (for let-bindings).
    pub value: Option<PantographExpression>,
}

/// An expression in a Pantograph response.
#[derive(Debug, Clone, Deserialize)]
pub struct PantographExpression {
    /// Pretty-printed expression.
    pub pp: Option<String>,
    /// S-expression form.
    pub sexp: Option<String>,
}

/// A structured goal in a Pantograph response.
#[derive(Debug, Clone, Deserialize)]
pub struct PantographGoal {
    /// Metavariable name.
    pub name: String,
    /// Target expression.
    pub target: PantographExpression,
    /// Variables (hypotheses) in scope.
    #[serde(default)]
    pub vars: Vec<PantographVariable>,
}

/// Response from `goal.start`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalStartResult {
    /// The state ID for this proof.
    pub state_id: u64,
    /// Root metavariable name.
    pub root: String,
}

/// Response from `goal.tactic`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalTacticResult {
    /// Next state ID (present on success).
    pub next_state_id: Option<u64>,
    /// Goals after tactic application (empty = proof complete).
    #[serde(default)]
    pub goals: Option<Vec<PantographGoal>>,
    /// Parse error message (tactic parsing failed).
    pub parse_error: Option<String>,
}

/// Error response from Pantograph.
#[derive(Debug, Clone, Deserialize)]
pub struct PantographError {
    /// Error category (e.g., "command", "index", "parse", "elab").
    pub error: String,
    /// Error description.
    pub desc: String,
}

/// A response received from Pantograph.
///
/// Pantograph responses can be:
/// - A successful `GoalStartResult` or `GoalTacticResult`
/// - An error `PantographError`
///
/// The response type depends on the command sent.
#[derive(Debug, Clone)]
pub enum PantographResponse {
    /// Response to `goal.start`.
    GoalStarted(GoalStartResult),
    /// Response to `goal.tactic`.
    TacticResult(GoalTacticResult),
    /// An error from Pantograph.
    Error(PantographError),
}

impl PantographResponse {
    /// Parse a JSON response line from Pantograph as a `goal.start` response.
    pub fn parse_goal_start(json: &str) -> Result<Self, crate::types::LeanError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::types::LeanError::Protocol(format!("Invalid JSON: {e}. Raw: {json}")))?;

        // Check if response has "error" + "desc" keys (Pantograph error format)
        if value.get("error").is_some() && value.get("desc").is_some() {
            let err: PantographError = serde_json::from_value(value)
                .map_err(|e| crate::types::LeanError::Protocol(format!("Failed to parse error: {e}")))?;
            return Ok(PantographResponse::Error(err));
        }

        let result: GoalStartResult = serde_json::from_value(value)
            .map_err(|e| crate::types::LeanError::Protocol(format!(
                "Failed to parse goal.start response: {e}. Raw: {json}"
            )))?;
        Ok(PantographResponse::GoalStarted(result))
    }

    /// Parse a JSON response line from Pantograph as a `goal.tactic` response.
    pub fn parse_goal_tactic(json: &str) -> Result<Self, crate::types::LeanError> {
        let value: serde_json::Value = serde_json::from_str(json)
            .map_err(|e| crate::types::LeanError::Protocol(format!("Invalid JSON: {e}. Raw: {json}")))?;

        // Check if response has "error" + "desc" keys (Pantograph error format)
        if value.get("error").is_some() && value.get("desc").is_some() {
            let err: PantographError = serde_json::from_value(value)
                .map_err(|e| crate::types::LeanError::Protocol(format!("Failed to parse error: {e}")))?;
            return Ok(PantographResponse::Error(err));
        }

        let result: GoalTacticResult = serde_json::from_value(value)
            .map_err(|e| crate::types::LeanError::Protocol(format!(
                "Failed to parse goal.tactic response: {e}. Raw: {json}"
            )))?;
        Ok(PantographResponse::TacticResult(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize_goal_start() {
        let req = PantographRequest::GoalStart {
            expr: "forall (n : Nat), n = n".to_string(),
        };
        let json = req.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["cmd"], "goal.start");
        assert_eq!(parsed["payload"]["expr"], "forall (n : Nat), n = n");
    }

    #[test]
    fn serialize_goal_tactic() {
        let req = PantographRequest::GoalTactic {
            state_id: 5,
            goal_id: None,
            tactic: "simp".to_string(),
        };
        let json = req.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["cmd"], "goal.tactic");
        assert_eq!(parsed["payload"]["stateId"], 5);
        assert_eq!(parsed["payload"]["tactic"], "simp");
        // goalId should be absent when None
        assert!(parsed["payload"].get("goalId").is_none());
    }

    #[test]
    fn serialize_goal_tactic_with_goal_id() {
        let req = PantographRequest::GoalTactic {
            state_id: 0,
            goal_id: Some(1),
            tactic: "intro n".to_string(),
        };
        let json = req.to_json().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed["payload"]["stateId"], 0);
        assert_eq!(parsed["payload"]["goalId"], 1);
        assert_eq!(parsed["payload"]["tactic"], "intro n");
    }

    #[test]
    fn serialize_uses_camel_case() {
        let req = PantographRequest::GoalTactic {
            state_id: 0,
            goal_id: None,
            tactic: "rfl".to_string(),
        };
        let json = req.to_json().unwrap();
        assert!(json.contains("\"stateId\""));
        assert!(!json.contains("\"state_id\""));
    }

    #[test]
    fn deserialize_goal_start_success() {
        let json = r#"{"root":"_uniq.7","stateId":0}"#;
        let resp = PantographResponse::parse_goal_start(json).unwrap();

        match resp {
            PantographResponse::GoalStarted(r) => {
                assert_eq!(r.state_id, 0);
                assert_eq!(r.root, "_uniq.7");
            }
            _ => panic!("Expected GoalStarted, got {:?}", resp),
        }
    }

    #[test]
    fn deserialize_goal_tactic_success_with_goals() {
        let json = r#"{"goals":[{"fragment":"tactic","name":"_uniq.9","target":{"pp":"n = n"},"vars":[{"isInaccessible":false,"name":"_uniq.8","type":{"pp":"Nat"},"userName":"n"}]}],"hasSorry":false,"hasUnsafe":false,"messages":[],"nextStateId":1}"#;
        let resp = PantographResponse::parse_goal_tactic(json).unwrap();

        match resp {
            PantographResponse::TacticResult(r) => {
                assert_eq!(r.next_state_id, Some(1));
                let goals = r.goals.unwrap();
                assert_eq!(goals.len(), 1);
                assert_eq!(goals[0].target.pp.as_deref(), Some("n = n"));
                assert_eq!(goals[0].vars.len(), 1);
                assert_eq!(goals[0].vars[0].user_name, "n");
            }
            _ => panic!("Expected TacticResult, got {:?}", resp),
        }
    }

    #[test]
    fn deserialize_goal_tactic_proof_complete() {
        let json = r#"{"goals":[],"hasSorry":false,"hasUnsafe":false,"messages":[],"nextStateId":2}"#;
        let resp = PantographResponse::parse_goal_tactic(json).unwrap();

        match resp {
            PantographResponse::TacticResult(r) => {
                assert_eq!(r.next_state_id, Some(2));
                assert!(r.goals.unwrap().is_empty());
                assert!(r.parse_error.is_none());
            }
            _ => panic!("Expected TacticResult with empty goals"),
        }
    }

    #[test]
    fn deserialize_goal_tactic_parse_error() {
        let json = r#"{"hasSorry":false,"hasUnsafe":false,"parseError":"<Pantograph>:1:1: unknown tactic"}"#;
        let resp = PantographResponse::parse_goal_tactic(json).unwrap();

        match resp {
            PantographResponse::TacticResult(r) => {
                assert!(r.next_state_id.is_none());
                assert!(r.parse_error.is_some());
                assert!(r.parse_error.unwrap().contains("unknown tactic"));
            }
            _ => panic!("Expected TacticResult with parseError"),
        }
    }

    #[test]
    fn deserialize_pantograph_error() {
        let json = r#"{"error":"command","desc":"Exactly one of {expr, copyFrom} must be supplied"}"#;
        let resp = PantographResponse::parse_goal_start(json).unwrap();

        match resp {
            PantographResponse::Error(e) => {
                assert_eq!(e.error, "command");
                assert!(e.desc.contains("Exactly one"));
            }
            _ => panic!("Expected Error, got {:?}", resp),
        }
    }

    #[test]
    fn deserialize_malformed_json() {
        let json = "not valid json {{{";
        let result = PantographResponse::parse_goal_start(json);
        assert!(result.is_err());
    }
}
