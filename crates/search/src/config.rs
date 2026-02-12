/// Search configuration loaded from TOML.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SearchConfig {
    /// Maximum number of nodes to expand before giving up.
    #[serde(default = "default_max_nodes")]
    pub max_nodes: u32,

    /// Maximum proof depth (tactics from root).
    #[serde(default = "default_max_depth")]
    pub max_depth: u32,

    /// Top-k candidates by LLM log-prob before Lean + EBM scoring.
    #[serde(default = "default_beam_width")]
    pub beam_width: usize,

    /// Number of candidate tactics to generate per expansion.
    #[serde(default = "default_num_candidates")]
    pub num_candidates: usize,

    /// Weight for LLM log-probability in combined score.
    #[serde(default = "default_alpha")]
    pub alpha: f64,

    /// Weight for EBM score in combined score.
    #[serde(default = "default_beta")]
    pub beta: f64,

    /// Maximum wall-clock seconds per theorem.
    #[serde(default = "default_timeout")]
    pub timeout_per_theorem: u64,

    /// Fallback tactics to try when the LLM generates no candidates.
    #[serde(default = "default_fallback_tactics")]
    pub fallback_tactics: Vec<String>,
}

fn default_max_nodes() -> u32 {
    600
}
fn default_max_depth() -> u32 {
    50
}
fn default_beam_width() -> usize {
    8
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
    ["simp", "omega", "decide", "aesop", "norm_num", "ring", "intro", "tauto"]
        .iter()
        .map(|s| s.to_string())
        .collect()
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
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            max_nodes: default_max_nodes(),
            max_depth: default_max_depth(),
            beam_width: default_beam_width(),
            num_candidates: default_num_candidates(),
            alpha: default_alpha(),
            beta: default_beta(),
            timeout_per_theorem: default_timeout(),
            fallback_tactics: default_fallback_tactics(),
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
        assert_eq!(cfg.beam_width, 8);
        assert_eq!(cfg.num_candidates, 32);
        assert!((cfg.alpha - 0.5).abs() < 1e-9);
        assert!((cfg.beta - 0.5).abs() < 1e-9);
        assert_eq!(cfg.timeout_per_theorem, 600);
        assert_eq!(cfg.fallback_tactics.len(), 8);
        assert_eq!(cfg.fallback_tactics[0], "simp");
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
            beam_width = 16
            num_candidates = 64
            alpha = 0.3
            beta = 0.7
            timeout_per_theorem = 300
        "#;
        let cfg: SearchConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.max_nodes, 200);
        assert_eq!(cfg.max_depth, 30);
        assert_eq!(cfg.beam_width, 16);
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
    fn test_validate_non_normalized() {
        // Non-normalized weights should not panic (just logs a warning)
        let cfg = SearchConfig {
            alpha: 0.3,
            beta: 0.3,
            ..Default::default()
        };
        cfg.validate(); // Should log warning but not panic
    }
}
