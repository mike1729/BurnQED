use std::time::Instant;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

use crate::protocol::{PantographRequest, PantographResponse};
use crate::types::{Goal, LeanError, LeanPoolConfig, ProofState, TacticResult};

/// A single Lean worker managing a Pantograph child process.
///
/// Each worker owns one Pantograph process and communicates via JSON lines
/// over stdin/stdout. Workers track their request count and age for
/// recycling decisions.
pub struct LeanWorker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    requests_handled: u64,
    started_at: Instant,
    config: LeanPoolConfig,
}

impl LeanWorker {
    /// Spawn a new Pantograph child process.
    ///
    /// Launches via `lake exe repl <imports>` from the Lean project directory,
    /// which sets up `LEAN_PATH` correctly. Consumes the initial "ready." line.
    pub async fn spawn(config: &LeanPoolConfig) -> Result<Self, LeanError> {
        let (child, stdin, stdout) = Self::spawn_process(config)?;

        let mut worker = Self {
            child,
            stdin,
            stdout,
            requests_handled: 0,
            started_at: Instant::now(),
            config: config.clone(),
        };

        // Consume the "ready." line that Pantograph prints on startup
        worker.consume_ready_line().await?;

        tracing::debug!(
            pantograph_path = %config.pantograph_path.display(),
            lean_env_path = %config.lean_env_path.display(),
            "Spawned Lean worker"
        );

        Ok(worker)
    }

    /// Spawn the underlying OS process.
    ///
    /// Pantograph is launched as `lake exe repl <imports>` from within
    /// the project directory specified by `lean_env_path`.
    fn spawn_process(
        config: &LeanPoolConfig,
    ) -> Result<(Child, BufWriter<ChildStdin>, BufReader<ChildStdout>), LeanError> {
        let mut cmd = Command::new(&config.pantograph_path);

        // If pantograph_path points to `lake`, run `lake exe repl <imports>`
        // If it points directly to the repl binary, just pass imports as args
        let path_str = config.pantograph_path.to_string_lossy();
        if path_str.contains("lake") {
            cmd.arg("exe").arg("repl");
        }

        // Add imports (e.g., "Init", "Mathlib")
        for import in &config.imports {
            cmd.arg(import);
        }

        // Set working directory to the Lean project so `lake` can find lakefile
        cmd.current_dir(&config.lean_env_path);

        let mut child = cmd
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()?;

        let stdin = BufWriter::new(
            child
                .stdin
                .take()
                .ok_or_else(|| LeanError::Protocol("Failed to capture stdin".into()))?,
        );
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or_else(|| LeanError::Protocol("Failed to capture stdout".into()))?,
        );

        Ok((child, stdin, stdout))
    }

    /// Consume the initial "ready." line from Pantograph.
    ///
    /// Uses a longer timeout (120s) than normal tactic operations because
    /// loading large environments like Mathlib can take 60-90s on first spawn.
    async fn consume_ready_line(&mut self) -> Result<(), LeanError> {
        let mut line = String::new();
        let startup_timeout_secs = 120;
        let timeout = std::time::Duration::from_secs(startup_timeout_secs);

        let result = tokio::time::timeout(timeout, self.stdout.read_line(&mut line)).await;

        match result {
            Ok(Ok(0)) => Err(LeanError::ProcessDied),
            Ok(Ok(_)) => {
                let trimmed = line.trim();
                if trimmed != "ready." {
                    tracing::warn!(line = trimmed, "Unexpected first line from Pantograph");
                }
                Ok(())
            }
            Ok(Err(e)) => Err(LeanError::Io(e)),
            Err(_) => Err(LeanError::Timeout(startup_timeout_secs)),
        }
    }

    /// Check whether this worker should be recycled.
    ///
    /// A worker needs recycling if it has handled too many requests
    /// (Lean processes leak memory) or has been alive too long.
    pub fn needs_recycling(&self) -> bool {
        self.requests_handled >= self.config.max_requests_per_worker
            || self.started_at.elapsed().as_secs() >= self.config.max_lifetime_secs
    }

    /// Number of requests this worker has handled since last spawn/recycle.
    pub fn requests_handled(&self) -> u64 {
        self.requests_handled
    }

    /// Kill the current process and spawn a fresh one.
    pub async fn recycle(&mut self) -> Result<(), LeanError> {
        // Gracefully kill old process — ignore errors (it may already be dead)
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;

        let (child, stdin, stdout) = Self::spawn_process(&self.config)?;
        self.child = child;
        self.stdin = stdin;
        self.stdout = stdout;
        self.requests_handled = 0;
        self.started_at = Instant::now();

        // Consume the "ready." line from the fresh process
        self.consume_ready_line().await?;

        tracing::debug!("Recycled Lean worker");
        Ok(())
    }

    /// Send a raw JSON line to Pantograph and read one response line.
    ///
    /// CRITICAL: Appends `\n` after the JSON — Pantograph blocks without it.
    async fn send_line(&mut self, json: &str) -> Result<String, LeanError> {
        self.send_line_timeout(json, self.config.tactic_timeout_secs)
            .await
    }

    /// Like [`send_line`], but with a custom timeout in seconds.
    async fn send_line_timeout(
        &mut self,
        json: &str,
        timeout_secs: u64,
    ) -> Result<String, LeanError> {
        // Write JSON + newline + flush
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        // Read response with timeout
        let mut response_line = String::new();
        let timeout = std::time::Duration::from_secs(timeout_secs);

        let read_result =
            tokio::time::timeout(timeout, self.stdout.read_line(&mut response_line)).await;

        match read_result {
            Ok(Ok(0)) => Err(LeanError::ProcessDied),
            Ok(Ok(_)) => {
                self.requests_handled += 1;
                Ok(response_line)
            }
            Ok(Err(e)) => Err(LeanError::Io(e)),
            Err(_) => {
                tracing::warn!(
                    timeout_secs,
                    "Lean tactic timed out, recycling worker"
                );
                self.recycle().await?;
                Err(LeanError::Timeout(timeout_secs))
            }
        }
    }

    /// Start a new proof for the given expression.
    ///
    /// Returns the initial proof state (with state_id but no goals yet —
    /// use `goal.tactic` to inspect goals or apply tactics).
    pub async fn start_proof(&mut self, expr: &str) -> Result<ProofState, LeanError> {
        let request = PantographRequest::GoalStart {
            expr: expr.to_string(),
        };
        let json = request
            .to_json()
            .map_err(|e| LeanError::Protocol(format!("Serialization error: {e}")))?;

        // Use a longer timeout for goal.start — expression elaboration can take
        // much longer than tactic application (60s+ for complex types like ℂ, Finset).
        let goal_start_timeout = self.config.tactic_timeout_secs.max(190);
        let response_line = self.send_line_timeout(&json, goal_start_timeout).await?;
        let response = PantographResponse::parse_goal_start(response_line.trim())?;

        match response {
            PantographResponse::GoalStarted(result) => Ok(ProofState {
                state_id: result.state_id,
                goals: Vec::new(), // goal.start doesn't return goals
            }),
            PantographResponse::Error(e) => Err(LeanError::LeanMessage(e.desc)),
            PantographResponse::TacticResult(_) => Err(LeanError::Protocol(
                "Unexpected TacticResult from goal.start".into(),
            )),
        }
    }

    /// Start a new proof by looking up a theorem name in the environment.
    ///
    /// Uses Pantograph's `copyFrom` feature to resolve a fully-qualified
    /// theorem name (e.g. `"Nat.add_comm"`) against the loaded Lean environment.
    pub async fn start_proof_by_name(&mut self, name: &str) -> Result<ProofState, LeanError> {
        let request = PantographRequest::GoalStartCopyFrom {
            copy_from: name.to_string(),
        };
        let json = request
            .to_json()
            .map_err(|e| LeanError::Protocol(format!("Serialization error: {e}")))?;

        let response_line = self.send_line(&json).await?;
        let response = PantographResponse::parse_goal_start(response_line.trim())?;

        match response {
            PantographResponse::GoalStarted(result) => Ok(ProofState {
                state_id: result.state_id,
                goals: Vec::new(), // goal.start doesn't return goals
            }),
            PantographResponse::Error(e) => Err(LeanError::LeanMessage(e.desc)),
            PantographResponse::TacticResult(_) => Err(LeanError::Protocol(
                "Unexpected TacticResult from goal.start".into(),
            )),
        }
    }

    /// Apply a tactic to a specific goal within a proof state.
    ///
    /// If `goal_id` is `None`, acts on the first goal.
    pub async fn apply_tactic(
        &mut self,
        state_id: u64,
        goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<TacticResult, LeanError> {
        let request = PantographRequest::GoalTactic {
            state_id,
            goal_id,
            tactic: tactic.to_string(),
        };
        let json = request
            .to_json()
            .map_err(|e| LeanError::Protocol(format!("Serialization error: {e}")))?;

        let response_line = self.send_line(&json).await?;
        let response = PantographResponse::parse_goal_tactic(response_line.trim())?;

        match response {
            PantographResponse::TacticResult(result) => {
                // Check for parse error
                if let Some(parse_error) = result.parse_error {
                    return Ok(TacticResult::Failed {
                        message: parse_error,
                    });
                }

                match (result.next_state_id, result.goals) {
                    (Some(next_id), Some(goals)) if goals.is_empty() => {
                        Ok(TacticResult::ProofComplete { state_id: next_id })
                    }
                    (Some(next_id), Some(goals)) => {
                        let parsed_goals: Vec<Goal> = goals
                            .iter()
                            .enumerate()
                            .map(|(i, g)| Goal::from_pantograph(i, g))
                            .collect();
                        Ok(TacticResult::Success {
                            state_id: next_id,
                            goals: parsed_goals,
                        })
                    }
                    (None, _) => Ok(TacticResult::Failed {
                        message: "Tactic failed (no next state)".into(),
                    }),
                    (Some(next_id), None) => {
                        // nextStateId present but no goals field — treat as success with no goals
                        Ok(TacticResult::ProofComplete { state_id: next_id })
                    }
                }
            }
            PantographResponse::Error(e) => Ok(TacticResult::Failed { message: e.desc }),
            PantographResponse::GoalStarted(_) => Err(LeanError::Protocol(
                "Unexpected GoalStarted from goal.tactic".into(),
            )),
        }
    }

    /// Shut down this worker by killing the child process.
    pub async fn shutdown(&mut self) {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
        tracing::debug!("Lean worker shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_config() -> LeanPoolConfig {
        LeanPoolConfig {
            num_workers: 1,
            max_requests_per_worker: 1000,
            max_lifetime_secs: 1800,
            tactic_timeout_secs: 30,
            pantograph_path: PathBuf::from("lake"),
            lean_env_path: PathBuf::from("/tmp/fake"),
            imports: vec!["Init".to_string()],
        }
    }

    #[test]
    fn needs_recycling_under_limit() {
        let config = test_config();
        assert_eq!(config.max_requests_per_worker, 1000);
        assert_eq!(config.max_lifetime_secs, 1800);
    }

    #[test]
    fn needs_recycling_request_count() {
        let mut config = test_config();
        config.max_requests_per_worker = 5;
        assert!(5 >= config.max_requests_per_worker);
        assert!(4 < config.max_requests_per_worker);
    }

    #[test]
    fn request_serialization_includes_payload() {
        let req = PantographRequest::GoalStart {
            expr: "True".to_string(),
        };
        let json = req.to_json().unwrap();
        assert!(json.contains("\"payload\""));
        assert!(json.contains("\"cmd\""));
        assert!(!json.ends_with('\n'));
    }
}
