/// Search configuration loaded from TOML.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SearchConfig {
    /// Maximum number of nodes to expand before giving up.
    #[serde(default = "default_max_nodes")]
    pub max_nodes: u32,

    /// Maximum proof depth (tactics from root).
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,

    /// Number of candidate tactics to generate per expansion.
    #[serde(default = "default_num_candidates")]
    pub num_candidates: usize,

    /// Weight for LLM log-probability in combined score.
    #[serde(default = "default_alpha")]
    pub alpha: f64,

    /// Weight for EBM score in combined score.
    #[serde(default = "default_beta")]
    pub beta: f64,

    /// Temperature for LLM log-prob scaling: `llm_log_prob / llm_temperature`.
    /// Higher values flatten the LLM signal. Default 1.0 (no scaling).
    #[serde(default = "default_temperature")]
    pub llm_temperature: f64,

    /// Temperature for EBM score scaling: `ebm_score / ebm_temperature`.
    /// Use this to bring EBM scores to a comparable scale with LLM log-probs.
    /// E.g., if EBM scores range [-10, 10] and log-probs range [-5, -1],
    /// set ebm_temperature=10 to normalize. Default 1.0 (no scaling).
    #[serde(default = "default_temperature")]
    pub ebm_temperature: f64,

    /// Maximum wall-clock seconds per theorem.
    #[serde(default = "default_timeout")]
    pub timeout_per_theorem: u64,

    /// Fallback tactics to try when the LLM generates no candidates.
    #[serde(default = "default_fallback_tactics")]
    pub fallback_tactics: Vec<String>,

    /// Built-in Lean tactics tried alongside LLM candidates at each expansion.
    #[serde(default = "default_probe_tactics")]
    pub probe_tactics: Vec<String>,

    /// Whether to mine sibling states from the proof path after finding a proof.
    #[serde(default)]
    pub harvest_siblings: bool,

    /// Number of nodes to expand per batch (1 = sequential, >1 = batched).
    /// Also controls the generate batcher coalesce limit: `batch_expansion_size × concurrency`.
    #[serde(default = "default_batch_expansion_size")]
    pub batch_expansion_size: usize,

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
}

fn default_max_nodes() -> u32 {
    600
}
fn default_max_depth() -> u32 {
    50
}
fn default_num_candidates() -> usize {
    32
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
fn default_fallback_tactics() -> Vec<String> {
    Vec::new()
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
fn default_batch_expansion_size() -> usize {
    1
}
fn default_batch_encode_size() -> usize {
    8
}
fn default_temperature() -> f64 {
    1.0
}

impl SearchConfig {
    /// Log a warning if alpha + beta don't sum to 1.0.
    pub fn validate(&self) {
        let sum = self.alpha + self.beta;
        if (sum - 1.0).abs() > 1e-6 {
            tracing::warn!(
                alpha = self.alpha,
                beta = self.beta,
                sum,
                "alpha + beta = {sum:.4}, expected 1.0; scores may not be normalized"
            );
        }
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
            num_candidates: default_num_candidates(),
            alpha: default_alpha(),
            beta: default_beta(),
            llm_temperature: default_temperature(),
            ebm_temperature: default_temperature(),
            timeout_per_theorem: default_timeout(),
            fallback_tactics: default_fallback_tactics(),
            probe_tactics: default_probe_tactics(),
            harvest_siblings: false,
            batch_expansion_size: default_batch_expansion_size(),
            batch_encode_size: default_batch_encode_size(),
            ebm_ramp_depth: 0,
            ebm_min_depth: 0,
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
        assert_eq!(cfg.max_depth, 50);
        assert_eq!(cfg.num_candidates, 32);
        assert!((cfg.alpha - 0.5).abs() < 1e-9);
        assert!((cfg.beta - 0.5).abs() < 1e-9);
        assert_eq!(cfg.timeout_per_theorem, 600);
        assert!(cfg.fallback_tactics.is_empty());
        assert_eq!(cfg.probe_tactics.len(), 17);
        assert!(!cfg.harvest_siblings);
        assert_eq!(cfg.batch_expansion_size, 1);
        assert_eq!(cfg.batch_encode_size, 8);
    }

    #[test]
    fn test_partial_toml_override() {
        let toml_str = r#"
            max_nodes = 100
            alpha = 0.7
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_nodes, 100);
        assert!((cfg.alpha - 0.7).abs() < 1e-9);
        // Defaults for unspecified fields
        assert_eq!(cfg.max_depth, 50);
        assert!((cfg.beta - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_full_toml() {
        let toml_str = r#"
            max_nodes = 200
            max_depth = 30
            num_candidates = 64
            alpha = 0.3
            beta = 0.7
            timeout_per_theorem = 300
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_nodes, 200);
        assert_eq!(cfg.max_depth, 30);
        assert_eq!(cfg.num_candidates, 64);
        assert!((cfg.alpha - 0.3).abs() < 1e-9);
        assert!((cfg.beta - 0.7).abs() < 1e-9);
        assert_eq!(cfg.timeout_per_theorem, 300);
    }

    #[test]
    fn test_alpha_beta_sum() {
        let cfg = SearchConfig::default();
        assert!((cfg.alpha + cfg.beta - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_validate_default_ok() {
        // Default config should not warn (alpha + beta == 1.0)
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
    fn test_harvest_siblings_enabled() {
        let toml_str = r#"
            harvest_siblings = true
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert!(cfg.harvest_siblings);
    }

    #[test]
    fn test_batch_expansion_size() {
        let toml_str = r#"
            batch_expansion_size = 8
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.batch_expansion_size, 8);
    }

    #[test]
    fn test_validate_non_normalized() {
        // Non-normalized weights should not panic (just logs a warning)
        let cfg = SearchConfig {
            alpha: 0.3,
            beta: 0.3,
            ..Default::default()
        };
        cfg.validate(); // Should log warning but not panic
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
            alpha = 0.5
            beta = 0.5
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.ebm_ramp_depth, 0);
        assert_eq!(cfg.ebm_min_depth, 0);
    }

    #[test]
    fn test_toml_with_ramp_fields() {
        let toml_str = r#"
            alpha = 0.5
            beta = 0.5
            ebm_ramp_depth = 4
            ebm_min_depth = 2
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.ebm_ramp_depth, 4);
        assert_eq!(cfg.ebm_min_depth, 2);
    }
}
