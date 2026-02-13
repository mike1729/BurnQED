//! Data types for the policy crate.

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
