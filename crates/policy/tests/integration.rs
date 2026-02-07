//! Integration tests for the policy crate.
//!
//! All tests are `#[ignore]` by default because they require model weights.
//!
//! ## Running
//!
//! ```bash
//! # Uses default path: models/tinyllama-1.1b (relative to workspace root)
//! cargo test -p policy -- --ignored --nocapture --test-threads=1
//!
//! # Or override with MODEL_PATH for the 7B model:
//! MODEL_PATH=models/deepseek-prover-v2-7b cargo test -p policy -- --ignored --nocapture --test-threads=1
//! ```

use policy::{DeviceConfig, PolicyConfig, TacticGenerator};
use std::path::PathBuf;

/// Default model path relative to the workspace root.
const DEFAULT_MODEL_PATH: &str = "models/tinyllama-1.1b";

/// Get the model path. Uses MODEL_PATH env var if set, otherwise the default
/// path relative to the workspace root.
fn model_path() -> PathBuf {
    if let Ok(p) = std::env::var("MODEL_PATH") {
        return PathBuf::from(p);
    }
    // Resolve relative to workspace root via CARGO_MANIFEST_DIR (crates/policy/)
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    workspace_root.join(DEFAULT_MODEL_PATH)
}

/// Create a PolicyConfig for testing.
fn test_config() -> PolicyConfig {
    let mut cfg = PolicyConfig::new(model_path());
    cfg.max_seq_len = 2048;
    cfg.num_candidates = 3;
    cfg.temperature = 0.6;
    cfg.top_p = 0.95;
    cfg.max_tactic_tokens = 32;
    cfg.device = DeviceConfig::Cpu;
    cfg
}

#[test]
#[ignore]
fn test_model_loads() {
    let config = test_config();
    let gen = TacticGenerator::load(&config).unwrap();
    assert!(gen.hidden_size() > 0);
    // Tokenizer vocab may differ from model vocab — just check it's nonzero
    assert!(gen.tokenizer().vocab_size() > 0);
}

#[test]
#[ignore]
fn test_tokenizer_roundtrip() {
    let config = test_config();
    let gen = TacticGenerator::load(&config).unwrap();
    let tok = gen.tokenizer();

    let text = "intro n";
    let ids = tok.encode(text).unwrap();
    assert!(!ids.is_empty());

    let decoded = tok.decode(&ids).unwrap();
    assert_eq!(decoded.trim(), text);
}

#[test]
#[ignore]
fn test_tokenizer_special_tokens() {
    let config = test_config();
    let gen = TacticGenerator::load(&config).unwrap();
    let tok = gen.tokenizer();

    assert!(tok.bos_token_id().is_some());
    assert!(tok.eos_token_id().is_some());

    let ids_plain = tok.encode("hello").unwrap();
    let ids_bos = tok.encode_with_bos("hello").unwrap();
    assert_eq!(ids_bos.len(), ids_plain.len() + 1);
    assert_eq!(ids_bos[0], tok.bos_token_id().unwrap());
}

#[test]
#[ignore]
fn test_forward_logits_shape() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    // Create a small input tensor
    let input = candle_core::Tensor::new(&[1u32, 2, 3], &candle_core::Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();

    let logits = gen.forward_logits(&input, 0).unwrap();
    let shape = logits.dims();
    // Should be (1, vocab_size) — last token only
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], 1);
    assert_eq!(shape[1], gen.vocab_size());
}

#[test]
#[ignore]
fn test_generate_one_tactic() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    let state = "n : Nat\n⊢ n + 0 = n";
    let tactic = gen.generate_one(state).unwrap();

    println!("Generated tactic: {:?}", tactic.text);
    println!("Log prob: {}", tactic.log_prob);
    println!("Tokens: {:?}", tactic.tokens);

    assert!(!tactic.text.is_empty(), "Tactic should not be empty");
    assert!(tactic.log_prob.is_finite(), "Log prob should be finite");
    assert!(!tactic.tokens.is_empty(), "Should have generated tokens");
}

#[test]
#[ignore]
fn test_generate_candidates_sorted() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    let state = "⊢ ∀ (n : Nat), n = n";
    let candidates = gen.generate_candidates(state, 3).unwrap();

    println!("Generated {} candidates:", candidates.len());
    for (i, c) in candidates.iter().enumerate() {
        println!("  {}: {:?} (log_prob={:.4})", i, c.text, c.log_prob);
    }

    assert!(!candidates.is_empty());

    // Verify sorted by log_prob descending
    for w in candidates.windows(2) {
        assert!(
            w[0].log_prob >= w[1].log_prob,
            "Candidates should be sorted by log_prob descending"
        );
    }
}

#[test]
#[ignore]
fn test_encode_only_shape() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    let state = "n : Nat\n⊢ n + 0 = n";
    let embedding = gen.encode_only(state).unwrap();

    assert_eq!(embedding.dim, gen.hidden_size(), "Embedding dim should match model hidden_size");
    assert_eq!(embedding.data.len(), gen.hidden_size());
    assert!(
        embedding.data.iter().all(|x| x.is_finite()),
        "All embedding values should be finite"
    );
}

#[test]
#[ignore]
fn test_encode_only_distinct() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    let emb1 = gen.encode_only("n : Nat\n⊢ n + 0 = n").unwrap();
    let emb2 = gen.encode_only("⊢ True").unwrap();

    // Compute cosine similarity
    let dot: f32 = emb1.data.iter().zip(&emb2.data).map(|(a, b)| a * b).sum();
    let norm1: f32 = emb1.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2.data.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cosine = dot / (norm1 * norm2);

    println!("Cosine similarity between different states: {cosine:.4}");
    assert!(
        cosine < 0.999,
        "Different proof states should produce different embeddings (cosine={cosine})"
    );
}

#[test]
#[ignore]
fn test_encode_only_deterministic() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    let state = "n : Nat\n⊢ n + 0 = n";
    let emb1 = gen.encode_only(state).unwrap();
    let emb2 = gen.encode_only(state).unwrap();

    // Should be exactly the same (no KV caching, same input)
    let max_diff: f32 = emb1
        .data
        .iter()
        .zip(&emb2.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max diff between identical inputs: {max_diff}");
    assert!(
        max_diff < 1e-5,
        "Same input should produce same embedding (max_diff={max_diff})"
    );
}

#[test]
#[ignore]
fn test_encode_batch() {
    let config = test_config();
    let mut gen = TacticGenerator::load(&config).unwrap();

    let texts = ["⊢ True", "n : Nat\n⊢ n = n", "⊢ False → False"];
    let embeddings = gen.encode_batch(&texts).unwrap();

    assert_eq!(embeddings.len(), 3);
    for emb in &embeddings {
        assert_eq!(emb.dim, gen.hidden_size());
    }
}
