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
        let prompt = format_prompt(proof_state);
        let mut tokens = self.tokenizer.encode_with_bos(&prompt)?;
        tokens = LeanTokenizer::truncate(&tokens, self.config.max_seq_len);

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

        let eos_id = self.tokenizer.eos_token_id();
        let newline_id = self.tokenizer.token_to_id("\n");

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
                break;
            }

            // Stop on newline — tactics are single-line
            if newline_id == Some(next_token) {
                break;
            }

            generated_tokens.push(next_token);

            // Feed next token
            let next_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let pos = prompt_len + generated_tokens.len() - 1;
            next_logits = self.model.forward(&next_input, pos, &mut self.cache)?;
        }

        let text = self.tokenizer.decode(&generated_tokens)?;
        // Safety net: only keep first line in case of multi-token newline leaks
        let text = text.lines().next().unwrap_or("").trim().to_string();

        Ok(GeneratedTactic {
            text,
            log_prob: log_prob_sum,
            tokens: generated_tokens,
        })
    }

    /// Generate `n` candidate tactics for a proof state, sorted by log-probability (descending).
    pub fn generate_candidates(
        &mut self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        let mut candidates = Vec::with_capacity(n);

        for i in 0..n {
            tracing::debug!(candidate = i, total = n, "Generating candidate");
            match self.generate_one(proof_state) {
                Ok(tactic) if tactic.text.is_empty() => {
                    tracing::debug!(candidate = i, "Skipping empty tactic");
                }
                Ok(tactic) => candidates.push(tactic),
                Err(e) => {
                    tracing::warn!(candidate = i, error = %e, "Failed to generate candidate");
                }
            }
        }

        // Sort by log_prob descending (higher = more probable)
        candidates.sort_by(|a, b| b.log_prob.partial_cmp(&a.log_prob).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates)
    }

    /// Run the model in encoder-only mode, returning mean-pooled hidden states.
    ///
    /// Uses a fresh cache (no KV caching) for each call to ensure deterministic output.
    /// The resulting `Embedding` has `dim == hidden_size` (4096 for DeepSeek-Prover-V2-7B).
    pub fn encode_only(&mut self, text: &str) -> anyhow::Result<Embedding> {
        let tokens = self.tokenizer.encode(text)?;
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

/// Format a proof state into a prompt for the model.
///
/// Uses a simple structured format. The model will complete after `[PROOFSTEP]`.
fn format_prompt(proof_state: &str) -> String {
    format!("[GOAL]{proof_state}[PROOFSTEP]")
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
    fn test_format_prompt() {
        let state = "n : Nat\n⊢ n + 0 = n";
        let prompt = format_prompt(state);
        assert!(prompt.starts_with("[GOAL]"));
        assert!(prompt.contains(state));
        assert!(prompt.ends_with("[PROOFSTEP]"));
    }

    #[test]
    fn test_format_prompt_empty() {
        let prompt = format_prompt("");
        assert_eq!(prompt, "[GOAL][PROOFSTEP]");
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
