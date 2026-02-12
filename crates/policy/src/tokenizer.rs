//! Wrapper around HuggingFace tokenizer for Lean proof state encoding.

use std::path::Path;
use tokenizers::Tokenizer;

/// Wraps a HuggingFace BPE tokenizer for encoding Lean proof states and tactics.
pub struct LeanTokenizer {
    inner: Tokenizer,
    bos_id: Option<u32>,
    eos_id: Option<u32>,
}

impl LeanTokenizer {
    /// Load a tokenizer from a HuggingFace model directory containing `tokenizer.json`.
    pub fn load(model_path: &Path) -> anyhow::Result<Self> {
        let tokenizer_path = model_path.join("tokenizer.json");
        let inner = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path.display(), e))?;

        // Extract special token IDs — try DeepSeek names first, then standard Llama
        let bos_id = inner.token_to_id("<｜begin▁of▁sentence｜>")
            .or_else(|| inner.token_to_id("<s>"));
        let eos_id = inner.token_to_id("<｜end▁of▁sentence｜>")
            .or_else(|| inner.token_to_id("</s>"));

        tracing::info!(
            vocab_size = inner.get_vocab_size(true),
            bos_id = ?bos_id,
            eos_id = ?eos_id,
            "Loaded LeanTokenizer"
        );

        Ok(Self {
            inner,
            bos_id,
            eos_id,
        })
    }

    /// Encode text into token IDs without adding special tokens.
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self.inner.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with BOS token prepended.
    pub fn encode_with_bos(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let mut ids = Vec::new();
        if let Some(bos) = self.bos_id {
            ids.push(bos);
        }
        let encoding = self.inner.encode(text, false)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
        ids.extend_from_slice(encoding.get_ids());
        Ok(ids)
    }

    /// Decode token IDs back to a string, skipping special tokens.
    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        let text = self.inner.decode(ids, true)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;
        Ok(text)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Get the BOS token ID, if any.
    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// Get the EOS token ID, if any.
    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Look up a token string and return its ID, if it exists in the vocabulary.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Truncate a token sequence to at most `max_len` tokens.
    pub fn truncate(ids: &[u32], max_len: usize) -> Vec<u32> {
        if ids.len() <= max_len {
            ids.to_vec()
        } else {
            ids[..max_len].to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short() {
        let ids = vec![1, 2, 3];
        let result = LeanTokenizer::truncate(&ids, 5);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_exact() {
        let ids = vec![1, 2, 3];
        let result = LeanTokenizer::truncate(&ids, 3);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_long() {
        let ids = vec![1, 2, 3, 4, 5];
        let result = LeanTokenizer::truncate(&ids, 3);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_truncate_empty() {
        let ids: Vec<u32> = vec![];
        let result = LeanTokenizer::truncate(&ids, 5);
        assert!(result.is_empty());
    }

    // Integration-style tests that require model files are in tests/integration.rs
}
