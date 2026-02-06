/// Wrapper around HuggingFace tokenizer for Lean proof state encoding.
pub struct ProofTokenizer {
    _private: (),
}

impl ProofTokenizer {
    /// Load tokenizer from a HuggingFace model directory.
    pub fn from_pretrained(_model_path: &str) -> anyhow::Result<Self> {
        todo!()
    }
}
