use std::path::PathBuf;

/// Errors that can occur during Lean REPL interaction.
#[derive(Debug, thiserror::Error)]
pub enum LeanError {
    /// Lean process exited unexpectedly.
    #[error("Lean process exited unexpectedly")]
    ProcessDied,

    /// Tactic timed out after the specified number of seconds.
    #[error("Tactic timed out after {0}s")]
    Timeout(u64),

    /// JSON parse error or unexpected response format.
    #[error("Protocol error: {0}")]
    Protocol(String),

    /// Lean reported an error (tactic failed, unknown identifier, etc.).
    #[error("Lean error: {0}")]
    LeanMessage(String),

    /// IO error from process communication.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// A single goal in a proof state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Goal {
    /// Zero-indexed goal ID within the proof state.
    pub goal_id: usize,
    /// Hypothesis strings (e.g., "n : Nat").
    pub hypotheses: Vec<String>,
    /// The goal target (the part after `⊢`).
    pub target: String,
    /// The full raw pretty-printed string from Pantograph.
    pub raw: String,
}

impl Goal {
    /// Parse a goal from Pantograph's pretty-printed format.
    ///
    /// Format:
    /// ```text
    /// hyp1 : Type1
    /// hyp2 : Type2
    /// ⊢ target_type
    /// ```
    ///
    /// The `⊢` symbol separates hypotheses from the goal target.
    pub fn parse(goal_id: usize, raw: &str) -> Self {
        let raw = raw.to_string();

        if let Some(turnstile_pos) = raw.find('⊢') {
            let before = &raw[..turnstile_pos];
            // Skip the `⊢` character (3 bytes in UTF-8) and any leading whitespace
            let after = raw[turnstile_pos + '⊢'.len_utf8()..].trim();

            let hypotheses: Vec<String> = before
                .lines()
                .map(|l| l.trim())
                .filter(|l| !l.is_empty())
                .map(|l| l.to_string())
                .collect();

            let target = after.to_string();

            Goal {
                goal_id,
                hypotheses,
                target,
                raw,
            }
        } else {
            // No turnstile found — treat entire string as target
            Goal {
                goal_id,
                hypotheses: Vec::new(),
                target: raw.clone(),
                raw,
            }
        }
    }

    /// Construct a Goal from Pantograph's structured response.
    ///
    /// Pantograph v0.3+ returns goals as structured JSON with separate
    /// `target` and `vars` fields rather than a single pretty-printed string.
    pub fn from_pantograph(goal_id: usize, pg: &crate::protocol::PantographGoal) -> Self {
        let target = pg
            .target
            .pp
            .as_deref()
            .unwrap_or("<no pp>")
            .to_string();

        let hypotheses: Vec<String> = pg
            .vars
            .iter()
            .map(|v| {
                let type_str = v
                    .type_expr
                    .as_ref()
                    .and_then(|e| e.pp.as_deref())
                    .unwrap_or("?");
                format!("{} : {}", v.user_name, type_str)
            })
            .collect();

        // Reconstruct a raw string similar to Lean's pretty-printed format
        let raw = if hypotheses.is_empty() {
            format!("⊢ {target}")
        } else {
            format!("{}\n⊢ {target}", hypotheses.join("\n"))
        };

        Goal {
            goal_id,
            hypotheses,
            target,
            raw,
        }
    }
}

/// Result of applying a tactic to a proof state.
#[derive(Debug, Clone)]
pub enum TacticResult {
    /// Tactic applied successfully, producing new goals.
    Success {
        state_id: u64,
        goals: Vec<Goal>,
    },
    /// Proof is complete (goals list is empty).
    ProofComplete {
        state_id: u64,
    },
    /// Tactic failed with a Lean error message.
    Failed {
        message: String,
    },
}

/// A proof state returned by Pantograph.
#[derive(Debug, Clone)]
pub struct ProofState {
    /// Monotonically increasing state ID from Pantograph.
    pub state_id: u64,
    /// Current goals in this proof state.
    pub goals: Vec<Goal>,
}

/// Configuration for the Lean worker pool.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct LeanPoolConfig {
    /// Number of worker processes to maintain.
    #[serde(default = "default_num_workers")]
    pub num_workers: usize,

    /// Maximum requests before a worker is recycled.
    #[serde(default = "default_max_requests")]
    pub max_requests_per_worker: u64,

    /// Maximum lifetime in seconds before a worker is recycled.
    #[serde(default = "default_max_lifetime")]
    pub max_lifetime_secs: u64,

    /// Timeout in seconds for a single tactic application.
    #[serde(default = "default_tactic_timeout")]
    pub tactic_timeout_secs: u64,

    /// Path to the Pantograph binary (or `lake` if using `lake exe repl`).
    pub pantograph_path: PathBuf,

    /// Path to the Lean project directory (where lakefile.lean lives).
    /// Used as the working directory when spawning Pantograph.
    pub lean_env_path: PathBuf,

    /// Lean modules to import (e.g., `["Init"]` or `["Init", "Mathlib"]`).
    #[serde(default = "default_imports")]
    pub imports: Vec<String>,
}

fn default_imports() -> Vec<String> {
    vec!["Init".to_string()]
}

impl LeanPoolConfig {
    /// Create a `LeanPoolConfig` using auto-discovered Pantograph location.
    ///
    /// Uses [`discover_pantograph()`] to find the Pantograph project directory,
    /// then fills in sensible defaults. Returns `None` if no Pantograph
    /// installation can be found.
    ///
    /// The returned config uses `lake` as the command (running `lake exe repl`)
    /// and imports only `Init`.
    pub fn with_bundled_pantograph() -> Option<Self> {
        let project_dir = discover_pantograph()?;
        Some(Self {
            num_workers: default_num_workers(),
            max_requests_per_worker: default_max_requests(),
            max_lifetime_secs: default_max_lifetime(),
            tactic_timeout_secs: default_tactic_timeout(),
            pantograph_path: PathBuf::from("lake"),
            lean_env_path: project_dir,
            imports: default_imports(),
        })
    }
}

/// Auto-discover the Pantograph project directory.
///
/// Discovery chain (first match wins):
/// 1. `PANTOGRAPH_PROJECT` environment variable
/// 2. `vendor/Pantograph/` relative to the workspace root (detected via
///    `LEAN_REPL_MANIFEST_DIR` set by build.rs, walking up two levels from
///    the `crates/lean-repl/` directory)
///
/// Returns `None` if no valid Pantograph directory is found. A directory is
/// considered valid if it contains a `lakefile.lean` file.
pub fn discover_pantograph() -> Option<PathBuf> {
    // 1. Explicit env var (checked at runtime)
    if let Ok(path) = std::env::var("PANTOGRAPH_PROJECT") {
        let p = PathBuf::from(path);
        if p.join("lakefile.lean").is_file() {
            return Some(p);
        }
        tracing::warn!(
            "PANTOGRAPH_PROJECT={} set but lakefile.lean not found there",
            p.display()
        );
    }

    // 2. Bundled vendor/Pantograph relative to workspace root
    //    LEAN_REPL_MANIFEST_DIR points to crates/lean-repl/
    //    Workspace root is two levels up: ../../
    if let Some(manifest_dir) = option_env!("LEAN_REPL_MANIFEST_DIR") {
        let vendor = PathBuf::from(manifest_dir)
            .join("..")
            .join("..")
            .join("vendor")
            .join("Pantograph");
        if let Ok(canonical) = vendor.canonicalize() {
            if canonical.join("lakefile.lean").is_file() {
                return Some(canonical);
            }
        }
    }

    None
}

fn default_num_workers() -> usize {
    4
}
fn default_max_requests() -> u64 {
    1000
}
fn default_max_lifetime() -> u64 {
    1800
}
fn default_tactic_timeout() -> u64 {
    30
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_goal_with_one_hypothesis() {
        let raw = "n : Nat\n⊢ n + 0 = n";
        let goal = Goal::parse(0, raw);

        assert_eq!(goal.goal_id, 0);
        assert_eq!(goal.hypotheses, vec!["n : Nat"]);
        assert_eq!(goal.target, "n + 0 = n");
        assert_eq!(goal.raw, raw);
    }

    #[test]
    fn parse_goal_no_hypotheses() {
        let raw = "⊢ ∀ (n : Nat), n + 0 = n";
        let goal = Goal::parse(0, raw);

        assert_eq!(goal.goal_id, 0);
        assert!(goal.hypotheses.is_empty());
        assert_eq!(goal.target, "∀ (n : Nat), n + 0 = n");
    }

    #[test]
    fn parse_goal_multiple_hypotheses() {
        let raw = "h : p\nh2 : q\n⊢ p ∧ q";
        let goal = Goal::parse(0, raw);

        assert_eq!(goal.hypotheses, vec!["h : p", "h2 : q"]);
        assert_eq!(goal.target, "p ∧ q");
    }

    #[test]
    fn parse_goal_preserves_goal_id() {
        let raw = "⊢ True";
        let goal = Goal::parse(3, raw);

        assert_eq!(goal.goal_id, 3);
        assert_eq!(goal.target, "True");
    }

    #[test]
    fn parse_goal_complex_types() {
        let raw = "α : Type u_1\nβ : Type u_2\nf : α → β\nx : α\n⊢ f x = f x";
        let goal = Goal::parse(0, raw);

        assert_eq!(goal.hypotheses.len(), 4);
        assert_eq!(goal.hypotheses[0], "α : Type u_1");
        assert_eq!(goal.hypotheses[3], "x : α");
        assert_eq!(goal.target, "f x = f x");
    }

    #[test]
    fn parse_goal_no_turnstile() {
        // Edge case: malformed goal string without ⊢
        let raw = "something unexpected";
        let goal = Goal::parse(0, raw);

        assert!(goal.hypotheses.is_empty());
        assert_eq!(goal.target, "something unexpected");
    }
}
