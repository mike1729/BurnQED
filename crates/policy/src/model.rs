//! LLM-based tactic generator using DeepSeek-Prover-V2-7B via candle.

use crate::llama::{Cache, Config, DeepSeekConfig, Llama};
use crate::tokenizer::LeanTokenizer;
use crate::types::{Embedding, GeneratedTactic, PolicyConfig};

use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use rand::distributions::Distribution;
use std::path::PathBuf;

/// LLM-based tactic generator wrapping a Llama model.
///
/// Provides:
/// - `generate_candidates()` — autoregressive tactic generation with top-p sampling
/// - `encode_only()` — mean-pooled hidden-state extraction for the EBM
pub struct TacticGenerator {
    model: Llama,
    tokenizer: LeanTokenizer,
    config: PolicyConfig,
    model_config: Config,
    device: Device,
    cache: Cache,
}

impl TacticGenerator {
    /// Load the model from a HuggingFace model directory.
    ///
    /// The directory must contain:
    /// - `config.json` — model architecture config
    /// - `tokenizer.json` — HuggingFace tokenizer
    /// - `*.safetensors` — model weights (can be sharded)
    pub fn load(config: &PolicyConfig) -> anyhow::Result<Self> {
        let device = config.device.to_candle_device()?;
        let dtype = match &device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        };

        tracing::info!(
            model_path = %config.model_path.display(),
            ?dtype,
            "Loading model"
        );

        // Parse config.json
        let config_path = config.model_path.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", config_path.display(), e))?;
        let ds_config: DeepSeekConfig = serde_json::from_str(&config_json)
            .map_err(|e| anyhow::anyhow!("Failed to parse config.json: {}", e))?;

        tracing::info!(
            hidden_size = ds_config.hidden_size,
            num_layers = ds_config.num_hidden_layers,
            vocab_size = ds_config.vocab_size,
            num_heads = ds_config.num_attention_heads,
            "Model config loaded"
        );

        let model_config = ds_config.into_config(false);

        // Find safetensors files
        let safetensor_files = find_safetensors(&config.model_path)?;
        tracing::info!(
            num_shards = safetensor_files.len(),
            "Loading safetensors weights"
        );

        // Load weights via memory-mapped safetensors
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, dtype, &device)?
        };

        let model = Llama::load(vb, &model_config)?;
        let tokenizer = LeanTokenizer::load(&config.model_path)?;

        let cache = Cache::new(true, dtype, &model_config, &device)?;

        tracing::info!(
            hidden_size = model_config.hidden_size,
            vocab_size = model_config.vocab_size,
            "Model loaded successfully"
        );

        Ok(Self {
            model,
            tokenizer,
            config: config.clone(),
            model_config,
            device,
            cache,
        })
    }

    /// Get the hidden size (embedding dimension) of the model.
    pub fn hidden_size(&self) -> usize {
        self.model_config.hidden_size
    }

    /// Get the vocabulary size of the model.
    pub fn vocab_size(&self) -> usize {
        self.model_config.vocab_size
    }

    /// Get a reference to the tokenizer.
    pub fn tokenizer(&self) -> &LeanTokenizer {
        &self.tokenizer
    }

    /// Run a forward pass on input token IDs, returning logits for the last token.
    ///
    /// Shape: `(batch_size, vocab_size)`.
    pub fn forward_logits(&mut self, input_ids: &Tensor, index_pos: usize) -> anyhow::Result<Tensor> {
        let logits = self.model.forward(input_ids, index_pos, &mut self.cache)?;
        Ok(logits)
    }

    /// Generate a single tactic for the given proof state.
    pub fn generate_one(&mut self, proof_state: &str) -> anyhow::Result<GeneratedTactic> {
        let message = format_tactic_message(proof_state);
        let mut tokens = self.tokenizer.encode_chat(&message)?;
        tokens = LeanTokenizer::truncate(&tokens, self.config.max_seq_len);

        tracing::debug!(
            prompt_tokens = tokens.len(),
            proof_state_len = proof_state.len(),
            "Starting tactic generation"
        );

        // Reset cache for new generation
        self.cache.clear();

        // Prefill: process prompt tokens
        let prompt_len = tokens.len();
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, &mut self.cache)?;

        // Autoregressive decoding
        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut log_prob_sum = 0.0f64;
        let mut next_logits = logits;
        let mut tokens_fed: usize = 0;
        let mut stop_reason = "max_tokens";

        let eos_id = self.tokenizer.eos_token_id();

        for _ in 0..self.config.max_tactic_tokens {
            let next_token = sample_top_p(
                &next_logits,
                self.config.temperature,
                self.config.top_p,
            )?;

            // Accumulate log probability
            let log_probs = candle_nn::ops::log_softmax(&next_logits, candle_core::D::Minus1)?;
            let token_log_prob = log_probs
                .i((0, next_token as usize))?
                .to_scalar::<f32>()? as f64;
            log_prob_sum += token_log_prob;

            // Check for EOS
            if eos_id == Some(next_token) {
                stop_reason = "eos";
                break;
            }

            generated_tokens.push(next_token);

            // Stop if candidate enters a repetitive loop
            if detect_repetition(&generated_tokens, 3, 8, 3) {
                stop_reason = "repetition";
                break;
            }

            // Feed next token
            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let pos = prompt_len + tokens_fed;
            tokens_fed += 1;
            next_logits = self.model.forward(&next_input, pos, &mut self.cache)?;
        }

        let raw_text = self.tokenizer.decode(&generated_tokens)?;
        let text = extract_first_tactic(&raw_text);

        tracing::info!(
            content_tokens = generated_tokens.len(),
            stop_reason,
            raw = %raw_text.escape_debug(),
            extracted = %text,
            "Tactic generation complete"
        );

        Ok(GeneratedTactic {
            text,
            log_prob: log_prob_sum,
            tokens: generated_tokens,
        })
    }

    /// Generate `n` candidate tactics in a single batched decode pass.
    ///
    /// Encodes the prompt once (batch=1), expands the KV cache to batch=N,
    /// then decodes all N sequences simultaneously. Sequences that hit EOS or
    /// a newline early are fed EOS for remaining steps (wasted compute is
    /// negligible for small N and short tactic sequences).
    fn generate_batch(
        &mut self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        let message = format_tactic_message(proof_state);
        let tokens = self.tokenizer.encode_chat(&message)?;
        let tokens = LeanTokenizer::truncate(&tokens, self.config.max_seq_len);
        let prompt_len = tokens.len();

        tracing::debug!(
            prompt_tokens = prompt_len,
            batch_size = n,
            "Starting batched tactic generation"
        );

        // 1. Prefill (batch=1)
        self.cache.clear();
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, &mut self.cache)?;
        // logits shape: (1, vocab_size)

        // 2. Expand cache from batch=1 to batch=N
        self.cache.expand_batch(n)?;

        // 3. Sample N different first tokens from the shared logits
        let eos_id = self.tokenizer.eos_token_id();
        let eos_fallback = eos_id.unwrap_or(0);

        let mut sequences: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut log_probs: Vec<f64> = vec![0.0; n];
        let mut stopped: Vec<bool> = vec![false; n];
        let mut stop_reasons: Vec<&str> = vec!["max_tokens"; n];

        // Compute log-probs from shared prefill logits
        let shared_log_probs =
            candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;

        // Sample N first tokens
        let mut first_tokens: Vec<u32> = Vec::with_capacity(n);
        for i in 0..n {
            let token = sample_top_p(&logits, self.config.temperature, self.config.top_p)?;
            let token_lp = shared_log_probs
                .i((0, token as usize))?
                .to_scalar::<f32>()? as f64;
            log_probs[i] = token_lp;

            // Check EOS
            if eos_id == Some(token) {
                stopped[i] = true;
                stop_reasons[i] = "eos";
                first_tokens.push(eos_fallback);
                continue;
            }

            sequences[i].push(token);
            first_tokens.push(token);
        }

        // 4. Batch decode loop
        let mut batch_input_vec = first_tokens;

        for step in 0..self.config.max_tactic_tokens.saturating_sub(1) {
            if stopped.iter().all(|&s| s) {
                break;
            }

            let input = Tensor::new(batch_input_vec.as_slice(), &self.device)?
                .reshape((n, 1))?;
            let logits =
                self.model
                    .forward(&input, prompt_len + step, &mut self.cache)?;
            // logits shape: (N, vocab_size)

            let all_log_probs =
                candle_nn::ops::log_softmax(&logits, candle_core::D::Minus1)?;

            batch_input_vec = Vec::with_capacity(n);
            for i in 0..n {
                if stopped[i] {
                    batch_input_vec.push(eos_fallback);
                    continue;
                }

                let seq_logits = logits.i(i)?.unsqueeze(0)?;
                let token =
                    sample_top_p(&seq_logits, self.config.temperature, self.config.top_p)?;

                let token_lp = all_log_probs
                    .i((i, token as usize))?
                    .to_scalar::<f32>()? as f64;
                log_probs[i] += token_lp;

                // Check EOS
                if eos_id == Some(token) {
                    stopped[i] = true;
                    stop_reasons[i] = "eos";
                    batch_input_vec.push(eos_fallback);
                    continue;
                }

                sequences[i].push(token);

                // Stop candidate if it enters a repetitive loop
                if detect_repetition(&sequences[i], 3, 8, 3) {
                    stopped[i] = true;
                    stop_reasons[i] = "repetition";
                    batch_input_vec.push(eos_fallback);
                    continue;
                }

                batch_input_vec.push(token);
            }
        }

        // 5. Decode, extract tactics, build results
        let mut candidates = Vec::with_capacity(n);
        for i in 0..n {
            let raw_text = self.tokenizer.decode(&sequences[i])?;
            let text = extract_first_tactic(&raw_text);

            tracing::info!(
                candidate = i,
                content_tokens = sequences[i].len(),
                stop_reason = stop_reasons[i],
                raw = %raw_text.escape_debug(),
                extracted = %text,
                "Batch candidate"
            );

            if text.is_empty() {
                continue;
            }

            // Deduplicate: skip if we already have this tactic (keep highest log_prob)
            if candidates.iter().any(|c: &GeneratedTactic| c.text == text) {
                tracing::debug!(candidate = i, text = %text, "Skipping duplicate tactic");
                continue;
            }

            candidates.push(GeneratedTactic {
                text,
                log_prob: log_probs[i],
                tokens: sequences[i].clone(),
            });
        }

        if candidates.is_empty() {
            tracing::warn!(
                batch_size = n,
                "All batch candidates were empty after extraction"
            );
        }

        // Sort by log_prob descending
        candidates.sort_by(|a, b| {
            b.log_prob
                .partial_cmp(&a.log_prob)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        tracing::info!(
            total = n,
            unique = candidates.len(),
            "Batch generation complete"
        );

        Ok(candidates)
    }

    /// Generate `n` candidate tactics for a proof state, sorted by log-probability (descending).
    ///
    /// For `n > 1`, uses batched decoding (single prefill, N-way parallel decode).
    /// For `n <= 1`, falls back to the simple sequential path.
    pub fn generate_candidates(
        &mut self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        if n <= 1 {
            // Single candidate — use simple path (no batch overhead)
            let mut candidates = Vec::with_capacity(n);
            for i in 0..n {
                tracing::debug!(candidate = i, total = n, "Generating candidate");
                match self.generate_one(proof_state) {
                    Ok(tactic) if tactic.text.is_empty() => {
                        tracing::debug!(candidate = i, "Skipping empty tactic");
                    }
                    Ok(tactic) => candidates.push(tactic),
                    Err(e) => {
                        tracing::warn!(
                            candidate = i,
                            error = %e,
                            "Failed to generate candidate"
                        );
                    }
                }
            }
            return Ok(candidates);
        }

        self.generate_batch(proof_state, n)
    }

    /// Run the model in encoder-only mode, returning mean-pooled hidden states.
    ///
    /// Uses a fresh cache (no KV caching) for each call to ensure deterministic output.
    /// The resulting `Embedding` has `dim == hidden_size` (4096 for DeepSeek-Prover-V2-7B).
    pub fn encode_only(&mut self, text: &str) -> anyhow::Result<Embedding> {
        let message = format_tactic_message(text);
        let tokens = self.tokenizer.encode_chat(&message)?;
        let tokens = LeanTokenizer::truncate(&tokens, self.config.max_seq_len);

        if tokens.is_empty() {
            return Ok(Embedding {
                data: vec![0.0; self.model_config.hidden_size],
                dim: self.model_config.hidden_size,
            });
        }

        // Use a fresh cache (no KV caching for encoding)
        let dtype = match &self.device {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        };
        let mut encode_cache = Cache::new(false, dtype, &self.model_config, &self.device)?;

        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let hidden_states = self.model.forward_hidden_states(&input, 0, &mut encode_cache)?;

        // Mean-pool across the sequence dimension: (1, seq_len, hidden) -> (1, hidden)
        let pooled = hidden_states.mean(1)?;
        let pooled = pooled.squeeze(0)?.to_dtype(DType::F32)?;
        let data = pooled.to_vec1::<f32>()?;

        Ok(Embedding {
            dim: data.len(),
            data,
        })
    }

    /// Encode multiple texts, returning an embedding for each.
    ///
    /// Currently processes sequentially. Batch processing may be added later.
    pub fn encode_batch(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Embedding>> {
        texts.iter().map(|t| self.encode_only(t)).collect()
    }
}

/// Check if the last `pattern_len * min_repeats` tokens consist of
/// the same pattern repeated `min_repeats` times.
/// Checks pattern lengths from `min_pattern` to `max_pattern`.
fn detect_repetition(
    tokens: &[u32],
    min_pattern: usize,
    max_pattern: usize,
    min_repeats: usize,
) -> bool {
    for pattern_len in min_pattern..=max_pattern {
        let required = pattern_len * min_repeats;
        if tokens.len() < required {
            continue;
        }
        let tail = &tokens[tokens.len() - required..];
        let pattern = &tail[..pattern_len];
        if (1..min_repeats).all(|i| &tail[i * pattern_len..(i + 1) * pattern_len] == pattern) {
            return true;
        }
    }
    false
}

/// Format a proof state into a chat message for the model.
///
/// Uses tactic-state comment format matching DeepSeek-Prover-V1.5/V2 training data.
fn format_tactic_message(proof_state: &str) -> String {
    format!(
        "Complete the following Lean 4 code:\n\n\
         ```lean4\n\
         /- tactic state:\n\
         {proof_state}\n\
         -/\n\
         ```"
    )
}

/// Extract the first valid tactic from model output.
///
/// Handles:
/// - Raw tactic text (`"intro h"`)
/// - Code-fenced output (`` ```lean4\nintro h\n``` ``)
/// - Comment lines (`"-- We introduce h\nintro h"`)
/// - Full theorem declarations (`"theorem X : T := by\n  tactic"` → `"tactic"`)
/// - Inline proof (`"theorem X := by trivial"` → `"trivial"`)
fn extract_first_tactic(raw: &str) -> String {
    let text = raw.trim();
    // Strip code fence if present
    let text = if text.starts_with("```") {
        text.lines()
            .skip(1) // skip ```lean4
            .take_while(|l| !l.starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        text.to_string()
    };
    // Take the first non-empty, non-comment, non-declaration line
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("--") && !l.starts_with("/-"))
        .find_map(|l| {
            // Skip theorem/lemma/example declarations
            if l.starts_with("theorem ") || l.starts_with("lemma ") || l.starts_with("example ") {
                // Check for inline tactic after "by": "theorem X := by trivial"
                if let Some(by_pos) = l.rfind(" by ") {
                    let after_by = l[by_pos + 4..].trim();
                    if !after_by.is_empty() {
                        return Some(after_by.to_string());
                    }
                }
                // Declaration ends with "by" — tactic is on the next line
                None
            } else {
                Some(l.to_string())
            }
        })
        .unwrap_or_default()
}

/// Find all `*.safetensors` files in a directory, sorted by name.
fn find_safetensors(model_path: &std::path::Path) -> anyhow::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(model_path)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .collect();

    if files.is_empty() {
        anyhow::bail!(
            "No .safetensors files found in {}",
            model_path.display()
        );
    }

    files.sort();
    tracing::debug!(?files, "Found safetensors shards");
    Ok(files)
}

/// Sample a token from logits using temperature and top-p (nucleus) sampling.
fn sample_top_p(logits: &Tensor, temperature: f64, top_p: f64) -> anyhow::Result<u32> {
    // logits shape: (1, vocab_size) or (vocab_size,)
    let logits = logits.squeeze(0)?;

    if temperature <= 0.0 {
        // Greedy decoding
        let token = logits.argmax(0)?.to_scalar::<u32>()?;
        return Ok(token);
    }

    // Apply temperature
    let logits = (&logits / temperature)?;

    // Softmax to get probabilities
    let probs = candle_nn::ops::softmax(&logits, 0)?;
    let probs_vec = probs.to_vec1::<f32>()?;

    // Sort probabilities in descending order
    let mut sorted_indices: Vec<usize> = (0..probs_vec.len()).collect();
    sorted_indices.sort_by(|&a, &b| probs_vec[b].partial_cmp(&probs_vec[a]).unwrap_or(std::cmp::Ordering::Equal));

    // Compute cumulative probability and find nucleus
    let mut cumulative = 0.0f32;
    let mut nucleus_size = 0;
    for &idx in &sorted_indices {
        cumulative += probs_vec[idx];
        nucleus_size += 1;
        if cumulative >= top_p as f32 {
            break;
        }
    }

    // Re-normalize probabilities within the nucleus
    let nucleus = &sorted_indices[..nucleus_size];
    let nucleus_sum: f32 = nucleus.iter().map(|&i| probs_vec[i]).sum();

    // Sample from the nucleus
    let mut rng = rand::thread_rng();
    let dist = rand::distributions::Uniform::new(0.0f32, nucleus_sum);
    let mut threshold = dist.sample(&mut rng);

    for &idx in nucleus {
        threshold -= probs_vec[idx];
        if threshold <= 0.0 {
            return Ok(idx as u32);
        }
    }

    // Fallback to most probable token
    Ok(sorted_indices[0] as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tactic_message() {
        let state = "n : Nat\n⊢ n + 0 = n";
        let msg = format_tactic_message(state);
        assert!(msg.contains("Complete the following Lean 4 code:"));
        assert!(msg.contains("tactic state:"));
        assert!(msg.contains(state));
        assert!(msg.contains("```lean4"));
    }

    #[test]
    fn test_format_tactic_message_empty() {
        let msg = format_tactic_message("");
        assert!(msg.contains("tactic state:"));
    }

    #[test]
    fn test_extract_first_tactic_raw() {
        assert_eq!(extract_first_tactic("intro h"), "intro h");
        assert_eq!(extract_first_tactic("  simp  "), "simp");
    }

    #[test]
    fn test_extract_first_tactic_code_fence() {
        let raw = "```lean4\nintro h\nexact h\n```";
        assert_eq!(extract_first_tactic(raw), "intro h");
    }

    #[test]
    fn test_extract_first_tactic_with_comments() {
        let raw = "-- We introduce h\nintro h\nexact h";
        assert_eq!(extract_first_tactic(raw), "intro h");
    }

    #[test]
    fn test_extract_first_tactic_empty() {
        assert_eq!(extract_first_tactic(""), "");
        assert_eq!(extract_first_tactic("  \n  "), "");
    }

    #[test]
    fn test_extract_first_tactic_block_comment() {
        let raw = "/- some reasoning -/\nomega";
        assert_eq!(extract_first_tactic(raw), "omega");
    }

    #[test]
    fn test_extract_first_tactic_theorem_declaration() {
        // Model generates "theorem X : T := by\n  tactic" — skip declaration, extract tactic
        let raw = "```lean4\ntheorem proof_of_true : True := by\n  trivial\n```";
        assert_eq!(extract_first_tactic(raw), "trivial");
    }

    #[test]
    fn test_extract_first_tactic_theorem_inline_by() {
        // Model generates "theorem X := by trivial" — extract tactic after "by"
        let raw = "theorem foo : True := by trivial";
        assert_eq!(extract_first_tactic(raw), "trivial");
    }

    #[test]
    fn test_extract_first_tactic_theorem_multi_step() {
        // Multi-step proof: extract only the first tactic
        let raw = "```lean4\ntheorem and_comm : P ∧ Q → Q ∧ P := by\n  intro ⟨hp, hq⟩\n  exact ⟨hq, hp⟩\n```";
        assert_eq!(extract_first_tactic(raw), "intro ⟨hp, hq⟩");
    }

    #[test]
    fn test_extract_first_tactic_lemma_declaration() {
        let raw = "lemma foo : True := by\n  simp";
        assert_eq!(extract_first_tactic(raw), "simp");
    }

    #[test]
    fn test_extract_first_tactic_theorem_only_declaration() {
        // Model only generates the declaration with no tactic body
        let raw = "theorem foo : True := by";
        assert_eq!(extract_first_tactic(raw), "");
    }

    #[test]
    fn test_detect_repetition_short_pattern() {
        // Pattern [1,2,3] repeated 3 times
        let tokens = vec![99, 1, 2, 3, 1, 2, 3, 1, 2, 3];
        assert!(detect_repetition(&tokens, 3, 8, 3));
    }

    #[test]
    fn test_detect_repetition_no_repeat() {
        let tokens = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert!(!detect_repetition(&tokens, 3, 8, 3));
    }

    #[test]
    fn test_detect_repetition_too_few_repeats() {
        // Pattern [1,2,3] repeated only 2 times (need 3)
        let tokens = vec![1, 2, 3, 1, 2, 3];
        assert!(!detect_repetition(&tokens, 3, 8, 3));
    }

    #[test]
    fn test_detect_repetition_single_token() {
        // Same token 3 times (pattern_len=1 is below min_pattern=3)
        let tokens = vec![5, 5, 5];
        assert!(!detect_repetition(&tokens, 3, 8, 3));
    }

    #[test]
    fn test_detect_repetition_longer_pattern() {
        // Pattern of 6 tokens repeated 3 times
        let tokens = vec![0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6];
        assert!(detect_repetition(&tokens, 3, 8, 3));
    }

    #[test]
    fn test_sample_top_p_greedy() {
        // Create logits with one clearly dominant value
        let logits = Tensor::new(&[0.0f32, 0.0, 10.0, 0.0], &Device::Cpu).unwrap();
        let token = sample_top_p(&logits, 0.0, 0.95).unwrap();
        assert_eq!(token, 2); // argmax
    }

    #[test]
    fn test_sample_top_p_with_temperature() {
        // High temperature should still produce valid tokens
        let logits = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &Device::Cpu).unwrap();
        let token = sample_top_p(&logits, 1.0, 0.95).unwrap();
        assert!(token < 4);
    }

    #[test]
    fn test_sample_top_p_narrow_nucleus() {
        // With very low top_p, should almost always pick the most probable
        let logits = Tensor::new(&[0.0f32, 0.0, 100.0, 0.0], &Device::Cpu).unwrap();
        let token = sample_top_p(&logits, 0.5, 0.01).unwrap();
        assert_eq!(token, 2);
    }
}
