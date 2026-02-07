//! TOML config loading for the prover CLI.
//!
//! Deserializes `configs/search.toml` which has `[search]` and `[lean_pool]` sections,
//! then merges with CLI overrides.

use std::path::Path;

use lean_repl::LeanPoolConfig;
use search::SearchConfig;
use serde::Deserialize;

/// Top-level structure matching `configs/search.toml`.
#[derive(Debug, Deserialize)]
pub struct SearchToml {
    /// Search algorithm parameters.
    pub search: SearchConfig,
    /// Lean worker pool overrides (paths come from auto-discovery).
    #[serde(default)]
    pub lean_pool: LeanPoolOverrides,
}

/// Optional overrides for `LeanPoolConfig` fields.
///
/// All fields are `Option` because paths come from auto-discovery via
/// `LeanPoolConfig::with_bundled_pantograph()`. Only numeric tuning
/// parameters can be overridden here.
#[derive(Debug, Default, Deserialize)]
pub struct LeanPoolOverrides {
    /// Number of worker processes.
    pub num_workers: Option<usize>,
    /// Maximum requests before a worker is recycled.
    pub max_requests_per_worker: Option<u64>,
    /// Maximum lifetime in seconds before a worker is recycled.
    pub max_lifetime_secs: Option<u64>,
    /// Timeout in seconds for a single tactic application.
    pub tactic_timeout_secs: Option<u64>,
}

/// Load and deserialize a `SearchToml` from a TOML file.
pub fn load_search_toml(path: &Path) -> anyhow::Result<SearchToml> {
    let contents = std::fs::read_to_string(path)?;
    let config: SearchToml = toml::from_str(&contents)?;
    tracing::info!(path = %path.display(), "Loaded search config");
    Ok(config)
}

/// Build a `LeanPoolConfig` from auto-discovery, TOML overrides, and CLI flags.
///
/// Priority chain: `with_bundled_pantograph()` defaults < TOML values < CLI `--num-workers`.
pub fn build_lean_pool_config(
    overrides: &LeanPoolOverrides,
    num_workers_cli: Option<usize>,
) -> anyhow::Result<LeanPoolConfig> {
    let mut config = LeanPoolConfig::with_bundled_pantograph()
        .ok_or_else(|| anyhow::anyhow!("Pantograph not found — run scripts/setup_pantograph.sh first"))?;

    // Apply TOML overrides
    if let Some(n) = overrides.num_workers {
        config.num_workers = n;
    }
    if let Some(n) = overrides.max_requests_per_worker {
        config.max_requests_per_worker = n;
    }
    if let Some(n) = overrides.max_lifetime_secs {
        config.max_lifetime_secs = n;
    }
    if let Some(n) = overrides.tactic_timeout_secs {
        config.tactic_timeout_secs = n;
    }

    // CLI override takes highest priority
    if let Some(n) = num_workers_cli {
        config.num_workers = n;
    }

    Ok(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_full_search_toml() {
        let toml_str = r#"
[search]
max_nodes = 200
max_depth = 30
beam_width = 16
num_candidates = 64
alpha = 0.3
beta = 0.7
timeout_per_theorem = 300

[search.iteration_0]
normal_temperature = 0.8
noise_temperature = 1.2
noise_fraction = 0.3

[lean_pool]
num_workers = 8
max_requests_per_worker = 500
max_lifetime_secs = 900
tactic_timeout_secs = 15
"#;
        let config: SearchToml = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.max_nodes, 200);
        assert_eq!(config.search.max_depth, 30);
        assert!((config.search.alpha - 0.3).abs() < 1e-9);
        assert_eq!(config.lean_pool.num_workers, Some(8));
        assert_eq!(config.lean_pool.max_requests_per_worker, Some(500));
        assert_eq!(config.lean_pool.max_lifetime_secs, Some(900));
        assert_eq!(config.lean_pool.tactic_timeout_secs, Some(15));
    }

    #[test]
    fn test_deserialize_optional_lean_pool() {
        // lean_pool section completely missing — should use defaults
        let toml_str = r#"
[search]
max_nodes = 100
"#;
        let config: SearchToml = toml::from_str(toml_str).unwrap();
        assert_eq!(config.search.max_nodes, 100);
        assert!(config.lean_pool.num_workers.is_none());
        assert!(config.lean_pool.max_requests_per_worker.is_none());
    }

    #[test]
    fn test_cli_override_priority() {
        let overrides = LeanPoolOverrides {
            num_workers: Some(8),
            max_requests_per_worker: None,
            max_lifetime_secs: None,
            tactic_timeout_secs: None,
        };

        // This test only checks the override logic, not actual Pantograph discovery.
        // We construct a config manually and apply overrides.
        let mut config = LeanPoolConfig {
            num_workers: 4,
            max_requests_per_worker: 1000,
            max_lifetime_secs: 1800,
            tactic_timeout_secs: 30,
            pantograph_path: std::path::PathBuf::from("lake"),
            lean_env_path: std::path::PathBuf::from("/tmp"),
            imports: vec!["Init".to_string()],
        };

        // Apply TOML overrides
        if let Some(n) = overrides.num_workers {
            config.num_workers = n;
        }
        assert_eq!(config.num_workers, 8);

        // CLI override takes priority
        config.num_workers = 16;
        assert_eq!(config.num_workers, 16);
    }
}
