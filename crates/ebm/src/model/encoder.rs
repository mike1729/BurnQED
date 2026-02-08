//! Encoder backend configuration for the EBM.
//!
//! The `EncoderBackend` enum selects how proof state strings are encoded into
//! embedding vectors. In both variants, the actual encoder lives **outside**
//! the EBM crate — it is injected as a closure `Fn(&str) -> Result<Vec<f32>>`
//! into [`crate::training::trainer::train`] and [`crate::inference::EBMScorer`].
//!
//! This enum is a **configuration** type: it records the embedding dimension
//! so the energy head can be sized correctly, and it deserializes from TOML
//! via `#[serde(tag = "type")]`.
//!
//! # Variants
//!
//! - `Shared` — the default. Uses the policy model's hidden states
//!   (e.g. DeepSeek-Prover-V2-7B, `hidden_dim = 4096`).
//! - `Dedicated` — reserved for a future smaller encoder (e.g. 1.3B).
//!
//! # Example (TOML)
//!
//! ```toml
//! [encoder]
//! type = "Shared"
//! # hidden_dim = 4096  # optional, this is the default
//! ```

use serde::Deserialize;

/// Configurable encoder backend for the EBM.
///
/// Determines the embedding dimension (`hidden_dim`) used to size the energy
/// head's input layer. The actual encoding is performed by an injected closure,
/// not by this enum.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(tag = "type")]
pub enum EncoderBackend {
    /// Share the policy model's backbone (e.g. DeepSeek-7B, hidden_dim=4096).
    Shared {
        /// Embedding dimension. Defaults to 4096 (DeepSeek-Prover-V2-7B).
        #[serde(default = "default_shared_hidden_dim")]
        hidden_dim: usize,
    },
    /// Separate dedicated encoder (future extension).
    Dedicated {
        /// Embedding dimension for the dedicated encoder.
        hidden_dim: usize,
    },
}

fn default_shared_hidden_dim() -> usize {
    4096
}

impl Default for EncoderBackend {
    fn default() -> Self {
        Self::Shared { hidden_dim: 4096 }
    }
}

impl EncoderBackend {
    /// Returns the embedding dimension for this backend.
    pub fn hidden_dim(&self) -> usize {
        match self {
            Self::Shared { hidden_dim } => *hidden_dim,
            Self::Dedicated { hidden_dim } => *hidden_dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_shared_4096() {
        let backend = EncoderBackend::default();
        assert_eq!(backend, EncoderBackend::Shared { hidden_dim: 4096 });
        assert_eq!(backend.hidden_dim(), 4096);
    }

    #[test]
    fn test_dedicated_hidden_dim() {
        let backend = EncoderBackend::Dedicated { hidden_dim: 1024 };
        assert_eq!(backend.hidden_dim(), 1024);
    }

    #[test]
    fn test_deserialize_shared() {
        let toml_str = r#"type = "Shared""#;
        let backend: EncoderBackend = toml::from_str(toml_str).unwrap();
        assert_eq!(backend, EncoderBackend::Shared { hidden_dim: 4096 });
    }

    #[test]
    fn test_deserialize_shared_custom_dim() {
        let toml_str = r#"
type = "Shared"
hidden_dim = 2048
"#;
        let backend: EncoderBackend = toml::from_str(toml_str).unwrap();
        assert_eq!(backend, EncoderBackend::Shared { hidden_dim: 2048 });
    }

    #[test]
    fn test_deserialize_dedicated() {
        let toml_str = r#"
type = "Dedicated"
hidden_dim = 1024
"#;
        let backend: EncoderBackend = toml::from_str(toml_str).unwrap();
        assert_eq!(backend, EncoderBackend::Dedicated { hidden_dim: 1024 });
    }
}
