/// LLM-based tactic generator using DeepSeek-Prover-V2-7B via candle.
pub struct TacticGenerator {
    _private: (),
}

impl TacticGenerator {
    /// Generate candidate tactics for a proof state.
    pub fn generate(&self, _proof_state: &str, _num_candidates: usize) -> Vec<String> {
        todo!()
    }

    /// Run the model in encoder-only mode, returning mean-pooled hidden states.
    pub fn encode_only(&self, _proof_state: &str) -> Vec<f32> {
        todo!()
    }
}
