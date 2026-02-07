//! Configuration and data types for the policy crate.

use candle_core::Device;
use serde::Deserialize;
use std::path::PathBuf;

/// Device selection for model inference.
#[derive(Debug, Clone, Default, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum DeviceConfig {
    /// CPU inference.
    #[serde(rename = "cpu")]
    #[default]
    Cpu,
    /// CUDA GPU inference with the given ordinal.
    #[serde(rename = "cuda")]
    Cuda { ordinal: usize },
}

impl DeviceConfig {
    /// Convert to a candle `Device`.
    pub fn to_candle_device(&self) -> anyhow::Result<Device> {
        match self {
            DeviceConfig::Cpu => Ok(Device::Cpu),
            DeviceConfig::Cuda { ordinal } => {
                Ok(Device::new_cuda(*ordinal)?)
            }
        }
    }
}

/// Configuration for the tactic generator.
#[derive(Debug, Clone, Deserialize)]
pub struct PolicyConfig {
    /// Path to the HuggingFace model directory (with config.json, tokenizer.json, *.safetensors).
    pub model_path: PathBuf,
    /// Maximum sequence length for input (prompt) tokens. Defaults to 2048.
    #[serde(default = "default_max_seq_len")]
    pub max_seq_len: usize,
    /// Number of candidate tactics to generate per call. Defaults to 32.
    #[serde(default = "default_num_candidates")]
    pub num_candidates: usize,
    /// Sampling temperature. Defaults to 0.6.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold. Defaults to 0.95.
    #[serde(default = "default_top_p")]
    pub top_p: f64,
    /// Maximum tokens to generate per tactic. Defaults to 256.
    #[serde(default = "default_max_tactic_tokens")]
    pub max_tactic_tokens: usize,
    /// Device to run inference on.
    #[serde(default)]
    pub device: DeviceConfig,
}

fn default_max_seq_len() -> usize {
    2048
}
fn default_num_candidates() -> usize {
    32
}
fn default_temperature() -> f64 {
    0.6
}
fn default_top_p() -> f64 {
    0.95
}
fn default_max_tactic_tokens() -> usize {
    256
}

impl PolicyConfig {
    /// Create a config with the given model path and defaults.
    pub fn new(model_path: PathBuf) -> Self {
        Self {
            model_path,
            max_seq_len: default_max_seq_len(),
            num_candidates: default_num_candidates(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            max_tactic_tokens: default_max_tactic_tokens(),
            device: DeviceConfig::default(),
        }
    }
}

/// A generated tactic candidate with metadata.
#[derive(Debug, Clone)]
pub struct GeneratedTactic {
    /// The tactic text.
    pub text: String,
    /// Sum of log-probabilities of the generated tokens.
    pub log_prob: f64,
    /// The generated token IDs (excluding prompt tokens).
    pub tokens: Vec<u32>,
}

/// A mean-pooled hidden-state embedding from the model.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The embedding vector.
    pub data: Vec<f32>,
    /// Dimensionality of the embedding (should equal hidden_size).
    pub dim: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_config_default() {
        let dc = DeviceConfig::default();
        assert_eq!(dc, DeviceConfig::Cpu);
    }

    #[test]
    fn test_device_config_cpu() {
        let dc = DeviceConfig::Cpu;
        let device = dc.to_candle_device().unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_device_config_deserialize_cpu() {
        let json = r#"{"type": "cpu"}"#;
        let dc: DeviceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(dc, DeviceConfig::Cpu);
    }

    #[test]
    fn test_device_config_deserialize_cuda() {
        let json = r#"{"type": "cuda", "ordinal": 0}"#;
        let dc: DeviceConfig = serde_json::from_str(json).unwrap();
        assert_eq!(dc, DeviceConfig::Cuda { ordinal: 0 });
    }

    #[test]
    fn test_policy_config_defaults() {
        let cfg = PolicyConfig::new(PathBuf::from("/tmp/model"));
        assert_eq!(cfg.max_seq_len, 2048);
        assert_eq!(cfg.num_candidates, 32);
        assert!((cfg.temperature - 0.6).abs() < 1e-9);
        assert!((cfg.top_p - 0.95).abs() < 1e-9);
        assert_eq!(cfg.max_tactic_tokens, 256);
    }

    #[test]
    fn test_policy_config_deserialize() {
        let json = r#"{
            "model_path": "/tmp/model",
            "max_seq_len": 1024,
            "temperature": 0.8
        }"#;
        let cfg: PolicyConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.max_seq_len, 1024);
        assert!((cfg.temperature - 0.8).abs() < 1e-9);
        // Defaults
        assert_eq!(cfg.num_candidates, 32);
        assert!((cfg.top_p - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_generated_tactic() {
        let t = GeneratedTactic {
            text: "intro n".to_string(),
            log_prob: -0.5,
            tokens: vec![1, 2, 3],
        };
        assert_eq!(t.text, "intro n");
        assert_eq!(t.tokens.len(), 3);
    }

    #[test]
    fn test_embedding() {
        let e = Embedding {
            data: vec![1.0, 2.0, 3.0],
            dim: 3,
        };
        assert_eq!(e.data.len(), e.dim);
    }
}
