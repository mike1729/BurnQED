/// Search configuration loaded from TOML.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SearchConfig {
    /// Maximum number of nodes to expand before giving up.
    #[serde(default = "default_max_nodes")]
    pub max_nodes: u32,

    /// Maximum proof depth (tactics from root).
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,

    /// Weight for LLM log-prob in frontier scoring: `alpha * llm + beta * ebm`.
    #[serde(default = "default_alpha")]
    pub alpha: f64,

    /// Weight for EBM score in frontier scoring. Used by `effective_beta()`.
    #[serde(default = "default_beta")]
    pub beta: f64,

    /// Maximum wall-clock seconds per theorem.
    #[serde(default = "default_timeout")]
    pub timeout_per_theorem: u64,

    /// Built-in Lean tactics tried alongside LLM candidates at each expansion.
    #[serde(default = "default_probe_tactics")]
    pub probe_tactics: Vec<String>,

    /// Maximum number of states per encode HTTP request to the EBM embedding server.
    /// Controls the `GlobalEncodeBatcher` coalesce limit. Lower values reduce peak VRAM
    /// on the encode server (important for nf4 quantization). Default: 8.
    #[serde(default = "default_batch_encode_size")]
    pub batch_encode_size: usize,

    /// EBM beta ramps linearly from 0 → full over this many depth levels.
    /// 0 = disabled (use full beta at all depths). E.g., `ebm_ramp_depth = 4` means
    /// effective_beta = beta * min(1.0, depth / 4).
    #[serde(default)]
    pub ebm_ramp_depth: u32,

    /// Skip EBM inference entirely below this depth (saves GPU compute).
    /// 0 = disabled (always score with EBM). E.g., `ebm_min_depth = 2` means
    /// depths 0 and 1 get no EBM scoring at all.
    #[serde(default)]
    pub ebm_min_depth: u32,

    // --- Hybrid whole-proof search config ---

    /// Number of whole proofs to generate in the first (root) round.
    #[serde(default = "default_hybrid_num_proofs")]
    pub hybrid_num_proofs: usize,

    /// Number of whole proofs to generate in subsequent (non-root) rounds.
    #[serde(default = "default_hybrid_expand_proofs")]
    pub hybrid_expand_proofs: usize,

    /// Maximum number of expansion rounds.
    #[serde(default = "default_hybrid_max_rounds")]
    pub hybrid_max_rounds: u32,

    /// Maximum tokens per whole proof generation.
    #[serde(default = "default_hybrid_max_tokens")]
    pub hybrid_max_tokens: usize,

    /// Total proof generation budget (sum of N across all rounds).
    #[serde(default = "default_hybrid_budget")]
    pub hybrid_budget: u32,

    /// UCB exploration constant for PUCT frontier scoring.
    /// Higher values encourage visiting less-explored nodes.
    #[serde(default = "default_exploration_c")]
    pub exploration_c: f64,

    /// Maximum consecutive tactic steps that leave the goal target unchanged.
    /// Blocks context-stuffing loops where the model adds redundant hypotheses.
    /// 0 = disabled. Default: 3.
    #[serde(default = "default_max_goal_unchanged")]
    pub max_goal_unchanged_steps: u32,

    /// Penalty per extra open goal beyond the first in frontier scoring.
    /// Deprioritizes nodes with many subgoals (from `have h : T` decomposition)
    /// in favor of nodes closer to a complete proof.
    /// Score adjustment: `-penalty * max(0, n_goals - 1)`. Default: 0.3.
    #[serde(default = "default_goal_count_penalty")]
    pub goal_count_penalty: f64,

    // --- LLM generation parameters ---

    /// Sampling temperature for tactic generation. Default: 0.6.
    /// CLI `--temperature` overrides this.
    #[serde(default = "default_temperature")]
    pub temperature: f64,

    /// Top-p (nucleus) sampling parameter. Default: 0.95.
    #[serde(default = "default_top_p")]
    pub top_p: f64,

    /// Maximum tokens per generated tactic. Default: 48.
    /// CLI `--max-tactic-tokens` overrides this.
    #[serde(default = "default_max_tactic_tokens")]
    pub max_tactic_tokens: usize,

}

fn default_max_nodes() -> u32 {
    600
}
fn default_max_depth() -> u32 {
    25
}
fn default_alpha() -> f64 {
    0.5
}
fn default_beta() -> f64 {
    0.5
}
fn default_timeout() -> u64 {
    600
}
fn default_probe_tactics() -> Vec<String> {
    [
        "simp", "ring", "omega", "norm_num", "decide", "trivial", "rfl", "tauto",
        "linarith", "push_neg", "contradiction", "exfalso", "constructor", "left",
        "right", "ext", "simp_all",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}
fn default_batch_encode_size() -> usize {
    8
}
fn default_hybrid_num_proofs() -> usize {
    32
}
fn default_hybrid_expand_proofs() -> usize {
    16
}
fn default_hybrid_max_rounds() -> u32 {
    30
}
fn default_hybrid_max_tokens() -> usize {
    1024
}
fn default_hybrid_budget() -> u32 {
    256
}
fn default_exploration_c() -> f64 {
    1.41
}
fn default_max_goal_unchanged() -> u32 {
    3
}
fn default_goal_count_penalty() -> f64 {
    0.3
}
fn default_temperature() -> f64 {
    0.8
}
fn default_top_p() -> f64 {
    0.95
}
fn default_max_tactic_tokens() -> usize {
    48
}

impl SearchConfig {
    /// Log configuration details.
    pub fn validate(&self) {
        if self.ebm_ramp_depth > 0 {
            tracing::info!(
                ebm_ramp_depth = self.ebm_ramp_depth,
                "EBM depth ramp enabled: beta ramps 0→{:.2} over {} depths",
                self.beta,
                self.ebm_ramp_depth,
            );
        }
        if self.ebm_min_depth > 0 {
            tracing::info!(
                ebm_min_depth = self.ebm_min_depth,
                "EBM hard cutoff enabled: skipping EBM inference below depth {}",
                self.ebm_min_depth,
            );
        }
    }

    /// Compute effective EBM beta weight for a given depth.
    ///
    /// Applies `ebm_min_depth` (hard cutoff → 0.0) then `ebm_ramp_depth`
    /// (linear ramp from 0 → beta). Both default to 0 (disabled).
    pub fn effective_beta(&self, depth: u32) -> f64 {
        if self.ebm_min_depth > 0 && depth < self.ebm_min_depth {
            return 0.0;
        }
        if self.ebm_ramp_depth > 0 {
            let lambda = (depth as f64 / self.ebm_ramp_depth as f64).min(1.0);
            self.beta * lambda
        } else {
            self.beta
        }
    }

    /// Whether EBM inference should be skipped entirely at this depth.
    pub fn should_skip_ebm(&self, depth: u32) -> bool {
        self.ebm_min_depth > 0 && depth < self.ebm_min_depth
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_nodes: default_max_nodes(),
            max_depth: default_max_depth(),
            alpha: default_alpha(),
            beta: default_beta(),
            timeout_per_theorem: default_timeout(),
            probe_tactics: default_probe_tactics(),
            batch_encode_size: default_batch_encode_size(),
            ebm_ramp_depth: 0,
            ebm_min_depth: 0,
            hybrid_num_proofs: default_hybrid_num_proofs(),
            hybrid_expand_proofs: default_hybrid_expand_proofs(),
            hybrid_max_rounds: default_hybrid_max_rounds(),
            hybrid_max_tokens: default_hybrid_max_tokens(),
            hybrid_budget: default_hybrid_budget(),
            exploration_c: default_exploration_c(),
            max_goal_unchanged_steps: default_max_goal_unchanged(),
            goal_count_penalty: default_goal_count_penalty(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tactic_tokens: default_max_tactic_tokens(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let cfg = SearchConfig::default();
        assert_eq!(cfg.max_nodes, 600);
        assert_eq!(cfg.max_depth, 25);
        assert!((cfg.alpha - 0.5).abs() < 1e-9);
        assert!((cfg.beta - 0.5).abs() < 1e-9);
        assert_eq!(cfg.timeout_per_theorem, 600);
        assert_eq!(cfg.probe_tactics.len(), 17);
        assert_eq!(cfg.batch_encode_size, 8);
        assert_eq!(cfg.hybrid_num_proofs, 32);
        assert_eq!(cfg.hybrid_expand_proofs, 16);
        assert_eq!(cfg.hybrid_max_rounds, 30);
        assert_eq!(cfg.hybrid_max_tokens, 1024);
        assert_eq!(cfg.hybrid_budget, 256);
        assert!((cfg.exploration_c - 1.41).abs() < 1e-9);
    }

    #[test]
    fn test_partial_toml_override() {
        let toml_str = r#"
            max_nodes = 100
            beta = 0.7
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_nodes, 100);
        assert!((cfg.beta - 0.7).abs() < 1e-9);
        assert_eq!(cfg.max_depth, 25);
    }

    #[test]
    fn test_full_toml() {
        let toml_str = r#"
            max_nodes = 200
            max_depth = 30
            beta = 0.7
            timeout_per_theorem = 300
            hybrid_num_proofs = 64
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_nodes, 200);
        assert_eq!(cfg.max_depth, 30);
        assert!((cfg.beta - 0.7).abs() < 1e-9);
        assert_eq!(cfg.timeout_per_theorem, 300);
        assert_eq!(cfg.hybrid_num_proofs, 64);
    }

    #[test]
    fn test_validate_default_ok() {
        let cfg = SearchConfig::default();
        cfg.validate(); // Should not panic
    }

    #[test]
    fn test_probe_tactics_disabled() {
        let toml_str = r#"
            probe_tactics = []
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.probe_tactics.is_empty());
    }

    #[test]
    fn test_toml_ignores_removed_fields() {
        // Old TOML files with removed fields should still parse (silently ignored)
        let toml_str = r#"
            alpha = 0.5
            num_candidates = 64
            batch_expansion_size = 8
            fallback_tactics = []
            harvest_siblings = true
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert!((cfg.beta - 0.5).abs() < 1e-9); // defaults applied
    }

    #[test]
    fn test_generation_params() {
        let toml_str = r#"
            temperature = 0.9
            top_p = 0.9
            max_tactic_tokens = 64
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert!((cfg.temperature - 0.9).abs() < 1e-9);
        assert!((cfg.top_p - 0.9).abs() < 1e-9);
        assert_eq!(cfg.max_tactic_tokens, 64);
    }

    #[test]
    fn test_generation_defaults() {
        let cfg = SearchConfig::default();
        assert!((cfg.temperature - 0.8).abs() < 1e-9);
        assert!((cfg.top_p - 0.95).abs() < 1e-9);
        assert_eq!(cfg.max_tactic_tokens, 48);
    }

    #[test]
    fn test_depth_ramp_defaults_disabled() {
        let cfg = SearchConfig::default();
        assert_eq!(cfg.ebm_ramp_depth, 0);
        assert_eq!(cfg.ebm_min_depth, 0);
        // With defaults disabled, effective_beta == beta at all depths
        assert!((cfg.effective_beta(0) - cfg.beta).abs() < 1e-9);
        assert!((cfg.effective_beta(5) - cfg.beta).abs() < 1e-9);
        assert!(!cfg.should_skip_ebm(0));
    }

    #[test]
    fn test_depth_ramp_only() {
        let cfg = SearchConfig {
            beta: 0.5,
            ebm_ramp_depth: 4,
            ..Default::default()
        };
        assert!((cfg.effective_beta(0) - 0.0).abs() < 1e-9);
        assert!((cfg.effective_beta(1) - 0.125).abs() < 1e-9); // 0.5 * 1/4
        assert!((cfg.effective_beta(2) - 0.25).abs() < 1e-9);  // 0.5 * 2/4
        assert!((cfg.effective_beta(4) - 0.5).abs() < 1e-9);   // 0.5 * 4/4
        assert!((cfg.effective_beta(10) - 0.5).abs() < 1e-9);  // clamped at 1.0
        assert!(!cfg.should_skip_ebm(0)); // ramp only, no hard cutoff
    }

    #[test]
    fn test_cutoff_only() {
        let cfg = SearchConfig {
            beta: 0.5,
            ebm_min_depth: 2,
            ..Default::default()
        };
        assert!(cfg.should_skip_ebm(0));
        assert!(cfg.should_skip_ebm(1));
        assert!(!cfg.should_skip_ebm(2));
        assert!(!cfg.should_skip_ebm(5));
        assert!((cfg.effective_beta(0) - 0.0).abs() < 1e-9);
        assert!((cfg.effective_beta(1) - 0.0).abs() < 1e-9);
        assert!((cfg.effective_beta(2) - 0.5).abs() < 1e-9); // full beta, no ramp
        assert!((cfg.effective_beta(5) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ramp_and_cutoff_combined() {
        let cfg = SearchConfig {
            beta: 0.5,
            ebm_ramp_depth: 4,
            ebm_min_depth: 2,
            ..Default::default()
        };
        // Below cutoff: skip entirely
        assert!(cfg.should_skip_ebm(0));
        assert!((cfg.effective_beta(0) - 0.0).abs() < 1e-9);
        assert!((cfg.effective_beta(1) - 0.0).abs() < 1e-9);
        // At/above cutoff: ramp applies
        assert!(!cfg.should_skip_ebm(2));
        assert!((cfg.effective_beta(2) - 0.25).abs() < 1e-9);  // 0.5 * 2/4
        assert!((cfg.effective_beta(3) - 0.375).abs() < 1e-9); // 0.5 * 3/4
        assert!((cfg.effective_beta(4) - 0.5).abs() < 1e-9);   // 0.5 * 4/4 = full
        assert!((cfg.effective_beta(10) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_toml_backward_compat() {
        // TOML without new fields should still parse (defaults to 0)
        let toml_str = r#"
            beta = 0.5
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.ebm_ramp_depth, 0);
        assert_eq!(cfg.ebm_min_depth, 0);
    }

    #[test]
    fn test_toml_with_ramp_fields() {
        let toml_str = r#"
            beta = 0.5
            ebm_ramp_depth = 4
            ebm_min_depth = 2
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.ebm_ramp_depth, 4);
        assert_eq!(cfg.ebm_min_depth, 2);
    }
}
