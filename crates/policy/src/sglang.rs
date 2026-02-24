//! SGLang HTTP client for LLM inference.
//!
//! Provides [`SglangClient`] for high-throughput tactic generation and
//! hidden-state extraction via SGLang's continuous batching and PagedAttention.
//!
//! # Server Setup
//!
//! ```bash
//! python -m sglang.launch_server \
//!     --model-path deepseek-ai/DeepSeek-Prover-V2-7B \
//!     --enable-return-hidden-states \
//!     --port 30000
//! ```

use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reqwest::Client;
use serde::Serialize;
use url::Url;

use crate::prompt::{extract_first_tactic, format_tactic_message};
use crate::types::{Embedding, GeneratedTactic};

/// Server encode capability, probed once on first encode call.
const ENCODE_UNKNOWN: u8 = 0;
const ENCODE_POOLED: u8 = 1;
const ENCODE_LEGACY: u8 = 2;

/// Configuration for connecting to an SGLang server.
#[derive(Debug, Clone)]
pub struct SglangConfig {
    /// Base URL of the SGLang server (e.g., "http://localhost:30000").
    pub server_url: String,
    /// Sampling temperature for generation.
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold.
    pub top_p: f64,
    /// Maximum tokens to generate per tactic.
    pub max_tactic_tokens: usize,
    /// Hidden size of the model (for embedding validation). 4096 for DeepSeek-Prover-V2-7B.
    pub hidden_size: usize,
}

/// Circuit breaker for SGLang transport-level failures.
///
/// Trips after `THRESHOLD` consecutive transport errors (connection refused, timeout).
/// Once open, all requests fail immediately until `COOLDOWN` elapses, then half-open
/// allows one probe request through.
#[derive(Debug, Clone)]
struct CircuitBreaker {
    consecutive_failures: std::sync::Arc<AtomicU32>,
    last_failure_epoch_ms: std::sync::Arc<AtomicU64>,
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self {
            consecutive_failures: std::sync::Arc::new(AtomicU32::new(0)),
            last_failure_epoch_ms: std::sync::Arc::new(AtomicU64::new(0)),
        }
    }
}

impl CircuitBreaker {
    const THRESHOLD: u32 = 3;
    const COOLDOWN_MS: u64 = 60_000;

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    /// Check if the circuit is open. Returns Err if requests should be rejected.
    fn check(&self) -> anyhow::Result<()> {
        let failures = self.consecutive_failures.load(Ordering::Relaxed);
        if failures < Self::THRESHOLD {
            return Ok(());
        }
        // Auto-reset after cooldown
        let last = self.last_failure_epoch_ms.load(Ordering::Relaxed);
        if Self::now_ms().saturating_sub(last) > Self::COOLDOWN_MS {
            self.consecutive_failures.store(0, Ordering::Relaxed);
            tracing::info!("SGLang circuit breaker reset after cooldown");
            return Ok(());
        }
        anyhow::bail!(
            "SGLang circuit breaker open ({failures} consecutive transport failures). \
             Will auto-reset after {}s cooldown.",
            Self::COOLDOWN_MS / 1000
        )
    }

    fn record_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    fn record_failure(&self) {
        let prev = self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        self.last_failure_epoch_ms
            .store(Self::now_ms(), Ordering::Relaxed);
        if prev + 1 >= Self::THRESHOLD {
            tracing::error!(
                consecutive_failures = prev + 1,
                "SGLang circuit breaker TRIPPED — server appears down. \
                 All requests will fail for {}s.",
                Self::COOLDOWN_MS / 1000
            );
        }
    }
}

/// HTTP client for SGLang inference server.
///
/// Supports two encoding paths:
/// - **Pooled** (`/encode`): Custom server mean-pools hidden states in-process,
///   returning a compact `(hidden_size,)` embedding (~16KB). ~7x faster.
/// - **Legacy** (`/generate` + `return_hidden_states`): SGLang HTTP server returns
///   the full `(1, num_tokens, hidden_size)` tensor as JSON (~10MB). Slow but works
///   with stock SGLang.
///
/// The client auto-detects which path is available on the first `encode()` call
/// and caches the result.
#[derive(Clone)]
pub struct SglangClient {
    client: Client,
    base_url: Url,
    config: SglangConfig,
    /// Cached encode capability: 0=unknown, 1=pooled (/encode), 2=legacy (/generate).
    encode_capability: std::sync::Arc<AtomicU8>,
    /// Circuit breaker for transport-level failures.
    circuit: CircuitBreaker,
}

// -- SGLang native API request/response types --

#[derive(Serialize)]
struct GenerateRequest {
    text: String,
    sampling_params: SamplingParams,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_logprob: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_hidden_states: bool,
}

/// Batch request: `text` is a list of prompts, SGLang batches them in one GPU forward pass.
#[derive(Serialize)]
struct BatchGenerateRequest {
    text: Vec<String>,
    sampling_params: SamplingParams,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_logprob: bool,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    return_hidden_states: bool,
}

#[derive(Serialize)]
struct SamplingParams {
    max_new_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<usize>,
}


impl SglangClient {
    /// Create a new client and verify the server is reachable.
    pub async fn new(config: SglangConfig) -> anyhow::Result<Self> {
        let base_url = Url::parse(&config.server_url)
            .map_err(|e| anyhow::anyhow!("Invalid server URL '{}': {e}", config.server_url))?;

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()?;

        let this = Self {
            client,
            base_url,
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        this.health_check().await?;
        Ok(this)
    }

    /// Verify the server is reachable and responding.
    pub async fn health_check(&self) -> anyhow::Result<()> {
        let url = self.base_url.join("/health")?;
        let resp = self
            .client
            .get(url.clone())
            .timeout(Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SGLang server unreachable at {}: {e}", self.config.server_url))?;

        if !resp.status().is_success() {
            // Some SGLang versions don't have /health — try a minimal generate
            tracing::debug!("SGLang /health returned {}, trying generate probe", resp.status());
            self.probe_generate().await?;
        }
        tracing::info!(url = %self.config.server_url, "SGLang server is reachable");
        Ok(())
    }

    /// Test whether the server supports returning hidden states.
    pub async fn test_hidden_states_support(&self) -> anyhow::Result<bool> {
        let request = GenerateRequest {
            text: "test".to_string(),
            sampling_params: SamplingParams {
                max_new_tokens: 1,
                temperature: Some(0.0),
                top_p: None,
                n: None,
            },
            return_logprob: false,
            return_hidden_states: true,
        };

        let url = self.base_url.join("/generate")?;
        let resp = self
            .client
            .post(url)
            .json(&request)
            .send()
            .await?;

        let body: serde_json::Value = resp.json().await?;
        let has_hs = body
            .get("meta_info")
            .and_then(|m| m.get("hidden_states"))
            .map(|hs| !hs.is_null())
            .unwrap_or(false);

        if has_hs {
            tracing::info!("SGLang server supports return_hidden_states");
        } else {
            tracing::warn!(
                "SGLang server does NOT support return_hidden_states over HTTP. \
                 EBM encoding will not be available. See SGLang Issue #6528."
            );
        }
        Ok(has_hs)
    }

    /// Generate N tactic candidates for a proof state.
    ///
    /// Sends N sequential requests with `n=1` each. SGLang's native `n>1` response
    /// format is unreliable, and sequential requests are negligibly slower since the
    /// GPU inference time dominates over localhost HTTP roundtrips.
    pub async fn generate_candidates(
        &self,
        proof_state: &str,
        n: usize,
    ) -> anyhow::Result<Vec<GeneratedTactic>> {
        let prompt = self.format_prompt(proof_state);
        let url = self.base_url.join("/generate")?;

        // Collect (text, log_prob) pairs — parse as Value to handle SGLang's
        // varying logprob formats (tuple third element can be string, null, or map).
        let mut responses: Vec<(String, f64)> = Vec::with_capacity(n);
        for i in 0..n {
            let request = GenerateRequest {
                text: prompt.clone(),
                sampling_params: SamplingParams {
                    max_new_tokens: self.config.max_tactic_tokens,
                    temperature: Some(self.config.temperature),
                    top_p: Some(self.config.top_p),
                    n: None,
                },
                return_logprob: true,
                return_hidden_states: false,
            };

            let resp = match self.post_with_retry(&url, &request, None).await {
                Ok(r) => r,
                Err(e) => {
                    tracing::debug!(
                        candidate = i,
                        error = %e,
                        "SGLang request failed, skipping candidate"
                    );
                    continue;
                }
            };
            let body = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    tracing::debug!(
                        candidate = i,
                        error = %e,
                        "Failed to read SGLang response body, skipping candidate"
                    );
                    continue;
                }
            };
            match serde_json::from_slice::<serde_json::Value>(&body) {
                Ok(val) => {
                    let text = val.get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();

                    // Sum logprobs from meta_info.output_token_logprobs: [[lp, id, ...], ...]
                    let log_prob = val.get("meta_info")
                        .and_then(|m| m.get("output_token_logprobs"))
                        .and_then(|lps| lps.as_array())
                        .map(|lps| {
                            lps.iter()
                                .filter_map(|entry| entry.as_array()?.first()?.as_f64())
                                .sum::<f64>()
                        })
                        .unwrap_or(0.0);

                    responses.push((text, log_prob));
                }
                Err(e) => {
                    let preview: String = String::from_utf8_lossy(&body).chars().take(200).collect();
                    tracing::warn!(
                        candidate = i,
                        error = %e,
                        body_preview = %preview,
                        "Failed to decode SGLang response, skipping candidate"
                    );
                }
            }
        }

        let mut tactics = Vec::new();
        for (i, (raw_text, log_prob)) in responses.iter().enumerate() {
            let tactic_text = extract_first_tactic(raw_text);
            let log_prob = *log_prob;

            if tactic_text.is_empty() {
                tracing::debug!(
                    candidate = i,
                    raw = %raw_text.chars().take(80).collect::<String>(),
                    "Empty tactic after extraction, skipping"
                );
                continue;
            }

            // Deduplicate: skip if we already have this tactic text
            if tactics.iter().any(|t: &GeneratedTactic| t.text == tactic_text) {
                continue;
            }

            tactics.push(GeneratedTactic {
                text: tactic_text,
                raw_text: raw_text.clone(),
                log_prob,
                tokens: Vec::new(), // Token IDs not needed for remote inference
            });
        }

        // Sort by log_prob descending (highest probability first)
        tactics.sort_by(|a, b| b.log_prob.partial_cmp(&a.log_prob).unwrap_or(std::cmp::Ordering::Equal));

        tracing::debug!(
            candidates = responses.len(),
            unique = tactics.len(),
            "SGLang generation complete"
        );

        Ok(tactics)
    }

    /// Generate tactic candidates for multiple proof states in a single HTTP request.
    ///
    /// Replicates each prompt `n` times into a flat batch, sends one
    /// `BatchGenerateRequest` to SGLang (RadixAttention caches shared prefixes),
    /// then unflattens and deduplicates per state.
    pub async fn generate_candidates_batch(
        &self,
        proof_states: &[String],
        n: usize,
    ) -> anyhow::Result<Vec<Vec<GeneratedTactic>>> {
        if proof_states.is_empty() {
            return Ok(vec![]);
        }

        // Build flat batch: each state's prompt repeated n times
        let mut flat_prompts: Vec<String> = Vec::with_capacity(proof_states.len() * n);
        for state in proof_states {
            let prompt = self.format_prompt(state);
            for _ in 0..n {
                flat_prompts.push(prompt.clone());
            }
        }

        // Scale timeout with batch size: ~0.5s per prompt, clamped to [15, 120]s
        let timeout_secs = ((flat_prompts.len() as u64) / 2).clamp(15, 120);

        let request = BatchGenerateRequest {
            text: flat_prompts,
            sampling_params: SamplingParams {
                max_new_tokens: self.config.max_tactic_tokens,
                temperature: Some(self.config.temperature),
                top_p: Some(self.config.top_p),
                n: None,
            },
            return_logprob: true,
            return_hidden_states: false,
        };

        let url = self.base_url.join("/generate")?;
        let resp = self
            .post_with_retry(&url, &request, Some(Duration::from_secs(timeout_secs)))
            .await?;
        let body: serde_json::Value = resp.json().await.map_err(|e| {
            anyhow::anyhow!("Failed to decode SGLang batch generate response: {e}")
        })?;

        let items = body.as_array().ok_or_else(|| {
            let preview: String = body.to_string().chars().take(200).collect();
            anyhow::anyhow!(
                "Expected JSON array for batch generate response, got: {preview}"
            )
        })?;

        let expected = proof_states.len() * n;
        if items.len() != expected {
            anyhow::bail!(
                "Batch generate response length mismatch: expected {} ({}×{}), got {}",
                expected,
                proof_states.len(),
                n,
                items.len()
            );
        }

        // Parse each item into (text, log_prob), then unflatten into groups of n
        let mut all_results: Vec<Vec<GeneratedTactic>> = Vec::with_capacity(proof_states.len());
        for (state_idx, chunk) in items.chunks(n).enumerate() {
            let mut responses: Vec<(String, f64)> = Vec::with_capacity(n);
            for (j, item) in chunk.iter().enumerate() {
                let text = item.get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                let log_prob = item.get("meta_info")
                    .and_then(|m| m.get("output_token_logprobs"))
                    .and_then(|lps| lps.as_array())
                    .map(|lps| {
                        lps.iter()
                            .filter_map(|entry| entry.as_array()?.first()?.as_f64())
                            .sum::<f64>()
                    })
                    .unwrap_or(0.0);

                if text.is_empty() {
                    tracing::debug!(
                        state = state_idx,
                        candidate = j,
                        "Empty text in batch response, skipping"
                    );
                    continue;
                }

                responses.push((text, log_prob));
            }

            // Extract, dedup, sort — same logic as generate_candidates()
            let mut tactics = Vec::new();
            for (raw_text, log_prob) in &responses {
                let tactic_text = extract_first_tactic(raw_text);
                if tactic_text.is_empty() {
                    continue;
                }
                if tactics.iter().any(|t: &GeneratedTactic| t.text == tactic_text) {
                    continue;
                }
                tactics.push(GeneratedTactic {
                    text: tactic_text,
                    raw_text: raw_text.clone(),
                    log_prob: *log_prob,
                    tokens: Vec::new(),
                });
            }
            tactics.sort_by(|a, b| {
                b.log_prob.partial_cmp(&a.log_prob).unwrap_or(std::cmp::Ordering::Equal)
            });
            all_results.push(tactics);
        }

        tracing::debug!(
            states = proof_states.len(),
            n = n,
            total_unique = all_results.iter().map(|v| v.len()).sum::<usize>(),
            "SGLang batch generation complete"
        );

        Ok(all_results)
    }

    /// Encode a proof state to a mean-pooled embedding.
    ///
    /// Auto-detects server capability on first call:
    /// - Custom server with `/encode` endpoint → pooled path (~16KB response, ~7x faster)
    /// - Legacy SGLang HTTP server → `/generate` + `return_hidden_states` (~10MB response)
    pub async fn encode(&self, text: &str) -> anyhow::Result<Embedding> {
        let cap = self.encode_capability.load(Ordering::Relaxed);

        match cap {
            ENCODE_POOLED => return self.encode_pooled(text).await,
            ENCODE_LEGACY => return self.encode_legacy(text).await,
            _ => {
                // Probe: try /encode first
                match self.encode_pooled(text).await {
                    Ok(emb) => {
                        self.encode_capability.store(ENCODE_POOLED, Ordering::Relaxed);
                        tracing::info!("Server supports /encode endpoint (pooled path)");
                        return Ok(emb);
                    }
                    Err(e) => {
                        // Fall back to legacy if /encode returned a client error (400, 404).
                        // 404 = endpoint doesn't exist; 400 = exists but --is-embedding not set.
                        // For transient errors (500, timeout), propagate so caller can retry.
                        let err_str = e.to_string();
                        let is_client_error = err_str.contains("404") || err_str.contains("400");
                        if is_client_error {
                            tracing::debug!(error = %e, "Server /encode unavailable, falling back to legacy path");
                            let emb = self.encode_legacy(text).await?;
                            self.encode_capability.store(ENCODE_LEGACY, Ordering::Relaxed);
                            tracing::info!(
                                "Server does not support /encode, using legacy /generate path"
                            );
                            return Ok(emb);
                        } else {
                            return Err(e);
                        }
                    }
                }
            }
        }
    }

    /// Encode a batch of proof states.
    ///
    /// Routes to `/encode` (batch) or legacy `/generate` (batch) based on
    /// the cached server capability. If capability is unknown, probes via
    /// a single `encode()` call first.
    pub async fn encode_batch(&self, texts: &[String]) -> anyhow::Result<Vec<anyhow::Result<Embedding>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Ensure capability is probed
        let cap = self.encode_capability.load(Ordering::Relaxed);
        if cap == ENCODE_UNKNOWN {
            // Probe with first text, then continue with batch for the rest
            let first_result = self.encode(&texts[0]).await;
            let cap = self.encode_capability.load(Ordering::Relaxed);

            if texts.len() == 1 {
                return Ok(vec![first_result]);
            }

            // Route remaining based on detected capability
            let rest = &texts[1..];
            let mut results = vec![first_result];
            let rest_results = if cap == ENCODE_POOLED {
                self.encode_pooled_batch(rest).await?
            } else {
                self.encode_legacy_batch(rest).await?
            };
            results.extend(rest_results);
            return Ok(results);
        }

        if cap == ENCODE_POOLED {
            self.encode_pooled_batch(texts).await
        } else {
            self.encode_legacy_batch(texts).await
        }
    }

    /// Encode via the custom `/encode` endpoint (pre-pooled, ~16KB response).
    async fn encode_pooled(&self, text: &str) -> anyhow::Result<Embedding> {
        let prompt = self.format_prompt(text);
        let request = serde_json::json!({
            "text": prompt,
            "hidden_size": self.config.hidden_size,
        });

        let url = self.base_url.join("/encode")?;
        let resp = self.post_once(&url, &request, Some(Duration::from_secs(10))).await?;

        let body: serde_json::Value = resp.json().await
            .map_err(|e| anyhow::anyhow!("Failed to decode /encode response: {e}"))?;

        self.parse_encode_response(&body)
    }

    /// Batch encode via the custom `/encode` endpoint.
    async fn encode_pooled_batch(&self, texts: &[String]) -> anyhow::Result<Vec<anyhow::Result<Embedding>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let prompts: Vec<String> = texts.iter().map(|t| self.format_prompt(t)).collect();
        let request = serde_json::json!({
            "text": prompts,
            "hidden_size": self.config.hidden_size,
        });

        let url = self.base_url.join("/encode")?;
        // Scale timeout with batch size, cap at 30s
        let timeout_secs = (5 * texts.len() as u64).clamp(10, 30);
        let resp = self.post_once(&url, &request, Some(Duration::from_secs(timeout_secs))).await?;
        let body: serde_json::Value = resp.json().await
            .map_err(|e| anyhow::anyhow!("Failed to decode /encode batch response: {e}"))?;

        // Batch response formats:
        //   SGLang --is-embedding: [{"embedding": [f32...]}, ...]  (top-level array)
        //   Custom /encode server:  {"embeddings": [[f32...], ...]} (object wrapper)
        let items: &Vec<serde_json::Value> = if let Some(arr) = body.as_array() {
            // Top-level array: SGLang native embedding mode
            arr
        } else if let Some(arr) = body.get("embeddings").and_then(|v| v.as_array()) {
            // Object wrapper: custom server
            arr
        } else {
            let preview: String = body.to_string().chars().take(200).collect();
            anyhow::bail!("Expected array or {{\"embeddings\": [...]}} in /encode batch response, got: {preview}");
        };

        if items.len() != texts.len() {
            anyhow::bail!(
                "/encode batch response length mismatch: expected {}, got {}",
                texts.len(),
                items.len()
            );
        }

        let results: Vec<anyhow::Result<Embedding>> = items
            .iter()
            .enumerate()
            .map(|(i, emb_val)| {
                // null → server failed for this prompt, return zero embedding
                if emb_val.is_null() {
                    return Ok(Embedding {
                        data: vec![0.0; self.config.hidden_size],
                        dim: self.config.hidden_size,
                    });
                }
                // Each item is either [f32...] directly or {"embedding": [f32...]}
                let arr = if let Some(inner) = emb_val.get("embedding").and_then(|v| v.as_array()) {
                    inner
                } else if let Some(inner) = emb_val.as_array() {
                    inner
                } else {
                    anyhow::bail!("Batch item {i}: expected array or {{\"embedding\": [...]}}")
                };
                let data: Vec<f32> = arr
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
                if data.len() != self.config.hidden_size {
                    anyhow::bail!(
                        "Batch item {i}: embedding dim mismatch: expected {}, got {}",
                        self.config.hidden_size,
                        data.len()
                    );
                }
                Ok(Embedding { data, dim: self.config.hidden_size })
            })
            .collect();

        Ok(results)
    }

    /// Encode via legacy `/generate` + `return_hidden_states` (SGLang HTTP server).
    async fn encode_legacy(&self, text: &str) -> anyhow::Result<Embedding> {
        let prompt = self.format_prompt(text);

        let request = GenerateRequest {
            text: prompt,
            sampling_params: SamplingParams {
                max_new_tokens: 1,
                temperature: Some(0.0),
                top_p: None,
                n: None,
            },
            return_logprob: false,
            return_hidden_states: true,
        };

        let url = self.base_url.join("/generate")?;
        let resp = self.post_with_retry(&url, &request, None).await?;
        let body: serde_json::Value = resp.json().await
            .map_err(|e| anyhow::anyhow!("Failed to decode SGLang encode response: {e}"))?;

        self.parse_hidden_states(&body)
    }

    /// Batch encode via concurrent individual `/generate` + `return_hidden_states` requests.
    ///
    /// Uses bounded concurrency with `futures::stream::buffered` to preserve input ordering
    /// while avoiding SGLang's buggy batch hidden_states responses.
    async fn encode_legacy_batch(&self, texts: &[String]) -> anyhow::Result<Vec<anyhow::Result<Embedding>>> {
        use futures::stream::{self, StreamExt};

        if texts.is_empty() {
            return Ok(vec![]);
        }

        let concurrency = 16;
        let results: Vec<anyhow::Result<Embedding>> = stream::iter(texts.to_vec())
            .map(|text| async move { self.encode_legacy(&text).await })
            .buffered(concurrency)
            .collect()
            .await;

        Ok(results)
    }

    /// Parse the response from the custom `/encode` endpoint.
    fn parse_encode_response(&self, body: &serde_json::Value) -> anyhow::Result<Embedding> {
        let embedding: Vec<f32> = body
            .get("embedding")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow::anyhow!("Missing 'embedding' in /encode response"))?
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if embedding.len() != self.config.hidden_size {
            anyhow::bail!(
                "Embedding dim mismatch: expected {}, got {}",
                self.config.hidden_size,
                embedding.len()
            );
        }

        Ok(Embedding {
            data: embedding,
            dim: self.config.hidden_size,
        })
    }

    /// Extract hidden states from a response JSON object and mean-pool to an embedding.
    fn parse_hidden_states(&self, body: &serde_json::Value) -> anyhow::Result<Embedding> {
        let hs_value = body
            .get("meta_info")
            .and_then(|m| m.get("hidden_states"))
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Server did not return hidden_states. Ensure --enable-return-hidden-states is set."
                )
            })?;

        let token_states: Vec<Vec<f32>> = hs_value
            .as_array()
            .and_then(|batch| batch.first())
            .and_then(|tokens| tokens.as_array())
            .ok_or_else(|| anyhow::anyhow!("Empty hidden_states batch dimension"))?
            .iter()
            .map(|token| {
                token
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_f64().map(|f| f as f32))
                            .collect()
                    })
                    .unwrap_or_default()
            })
            .collect();

        let num_tokens = token_states.len();
        if num_tokens == 0 {
            anyhow::bail!("Hidden states has 0 tokens");
        }

        let hidden_size = token_states[0].len();
        if hidden_size != self.config.hidden_size {
            anyhow::bail!(
                "Hidden size mismatch: expected {}, got {hidden_size}",
                self.config.hidden_size
            );
        }

        // Mean-pool across tokens
        let mut embedding = vec![0.0f32; hidden_size];
        for token_vec in &token_states {
            for (i, &val) in token_vec.iter().enumerate() {
                embedding[i] += val;
            }
        }
        let scale = 1.0 / num_tokens as f32;
        for val in &mut embedding {
            *val *= scale;
        }

        Ok(Embedding {
            data: embedding,
            dim: hidden_size,
        })
    }

    /// Format a proof state into the full chat prompt for SGLang.
    ///
    /// Uses DeepSeek special token markers as text to produce the chat-formatted
    /// prompt matching the model's training data.
    fn format_prompt(&self, proof_state: &str) -> String {
        let message = format_tactic_message(proof_state);
        // DeepSeek-Prover-V2 chat template:
        // <｜begin▁of▁sentence｜><｜User｜>{message}<｜Assistant｜>
        format!(
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>\
             <\u{ff5c}User\u{ff5c}>{message}\
             <\u{ff5c}Assistant\u{ff5c}>"
        )
    }

    /// POST with retry (up to 3 attempts with exponential backoff on 5xx).
    ///
    /// An optional `timeout` overrides the client default (useful for large
    /// batch requests that produce multi-MB responses).
    async fn post_with_retry<T: Serialize>(
        &self,
        url: &Url,
        body: &T,
        timeout: Option<Duration>,
    ) -> anyhow::Result<reqwest::Response> {
        self.post_with_attempts(url, body, timeout, 3).await
    }

    /// POST without retrying — fails immediately on server error or timeout.
    async fn post_once<T: Serialize>(
        &self,
        url: &Url,
        body: &T,
        timeout: Option<Duration>,
    ) -> anyhow::Result<reqwest::Response> {
        self.post_with_attempts(url, body, timeout, 1).await
    }

    async fn post_with_attempts<T: Serialize>(
        &self,
        url: &Url,
        body: &T,
        timeout: Option<Duration>,
        max_attempts: usize,
    ) -> anyhow::Result<reqwest::Response> {
        self.circuit.check()?;

        let mut last_err = None;
        for attempt in 0..max_attempts {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * (1 << attempt));
                tracing::debug!(attempt, delay_ms = delay.as_millis() as u64, "Retrying SGLang request");
                tokio::time::sleep(delay).await;
            }

            let mut req = self.client.post(url.clone()).json(body);
            if let Some(t) = timeout {
                req = req.timeout(t);
            }

            match req.send().await {
                Ok(resp) if resp.status().is_server_error() => {
                    // 5xx: retry (transient server issue)
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    last_err = Some(anyhow::anyhow!(
                        "SGLang server error {status}: {body_text}"
                    ));
                }
                Ok(resp) if resp.status().is_client_error() => {
                    self.circuit.record_success();
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    anyhow::bail!("SGLang client error {status}: {body_text}");
                }
                Ok(resp) => {
                    self.circuit.record_success();
                    return Ok(resp);
                }
                Err(e) => {
                    // Transport error (connection refused, timeout): don't retry,
                    // record failure for circuit breaker.
                    tracing::warn!(
                        error = %e,
                        url = %url,
                        "SGLang transport error (connection refused / timeout)"
                    );
                    self.circuit.record_failure();
                    anyhow::bail!("SGLang request failed: {e}");
                }
            }
        }
        Err(last_err.unwrap_or_else(|| anyhow::anyhow!("SGLang request failed after retries")))
    }

    /// Minimal generate probe to verify the server can process requests.
    async fn probe_generate(&self) -> anyhow::Result<()> {
        let request = GenerateRequest {
            text: "test".to_string(),
            sampling_params: SamplingParams {
                max_new_tokens: 1,
                temperature: Some(0.0),
                top_p: None,
                n: None,
            },
            return_logprob: false,
            return_hidden_states: false,
        };
        let url = self.base_url.join("/generate")?;
        let resp = self.client.post(url).json(&request).send().await?;
        if !resp.status().is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("SGLang probe failed: {body}");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_prompt() {
        // Construct a throwaway client just to test format_prompt
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 4096,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        let prompt = client.format_prompt("n : Nat\n\u{22a2} n + 0 = n");
        assert!(prompt.contains("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        assert!(prompt.contains("<\u{ff5c}User\u{ff5c}>"));
        assert!(prompt.contains("<\u{ff5c}Assistant\u{ff5c}>"));
        assert!(prompt.contains("tactic state:"));
        assert!(prompt.contains("n + 0 = n"));
    }

    #[test]
    fn test_sampling_params_serialization() {
        let params = SamplingParams {
            max_new_tokens: 128,
            temperature: Some(0.6),
            top_p: Some(0.95),
            n: Some(4),
        };
        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["max_new_tokens"], 128);
        assert_eq!(json["temperature"], 0.6);
        assert_eq!(json["n"], 4);
    }

    #[test]
    fn test_sampling_params_skip_none() {
        let params = SamplingParams {
            max_new_tokens: 1,
            temperature: None,
            top_p: None,
            n: None,
        };
        let json = serde_json::to_value(&params).unwrap();
        assert_eq!(json["max_new_tokens"], 1);
        assert!(json.get("temperature").is_none());
        assert!(json.get("top_p").is_none());
        assert!(json.get("n").is_none());
    }

    #[test]
    fn test_generate_request_skip_false_bools() {
        let req = GenerateRequest {
            text: "test".to_string(),
            sampling_params: SamplingParams {
                max_new_tokens: 1,
                temperature: None,
                top_p: None,
                n: None,
            },
            return_logprob: false,
            return_hidden_states: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        // false bools should be skipped
        assert!(json.get("return_logprob").is_none());
        assert!(json.get("return_hidden_states").is_none());
    }

    #[test]
    fn test_batch_request_serializes_text_as_array() {
        let req = BatchGenerateRequest {
            text: vec!["prompt1".to_string(), "prompt2".to_string()],
            sampling_params: SamplingParams {
                max_new_tokens: 1,
                temperature: Some(0.0),
                top_p: None,
                n: None,
            },
            return_logprob: false,
            return_hidden_states: true,
        };
        let json = serde_json::to_value(&req).unwrap();
        let text_arr = json["text"].as_array().unwrap();
        assert_eq!(text_arr.len(), 2);
        assert_eq!(text_arr[0], "prompt1");
        assert_eq!(text_arr[1], "prompt2");
        assert_eq!(json["return_hidden_states"], true);
        assert!(json.get("return_logprob").is_none());
    }

    #[test]
    fn test_parse_hidden_states_extracts_embedding() {
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 3,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        // Simulated response: 2 tokens × 3 dims
        let body = serde_json::json!({
            "meta_info": {
                "hidden_states": [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]]
            }
        });

        let emb = client.parse_hidden_states(&body).unwrap();
        assert_eq!(emb.dim, 3);
        // Mean of [1,2,3] and [3,4,5] = [2,3,4]
        assert!((emb.data[0] - 2.0).abs() < 1e-6);
        assert!((emb.data[1] - 3.0).abs() < 1e-6);
        assert!((emb.data[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_hidden_states_rejects_wrong_dim() {
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 4096,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        let body = serde_json::json!({
            "meta_info": {
                "hidden_states": [[[1.0, 2.0, 3.0]]]
            }
        });

        let err = client.parse_hidden_states(&body).unwrap_err();
        assert!(err.to_string().contains("Hidden size mismatch"));
    }

    #[test]
    fn test_generate_request_include_true_bools() {
        let req = GenerateRequest {
            text: "test".to_string(),
            sampling_params: SamplingParams {
                max_new_tokens: 1,
                temperature: None,
                top_p: None,
                n: None,
            },
            return_logprob: true,
            return_hidden_states: true,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["return_logprob"], true);
        assert_eq!(json["return_hidden_states"], true);
    }

    #[test]
    fn test_parse_encode_response_single() {
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 3,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        let body = serde_json::json!({
            "embedding": [1.0, 2.0, 3.0]
        });

        let emb = client.parse_encode_response(&body).unwrap();
        assert_eq!(emb.dim, 3);
        assert!((emb.data[0] - 1.0).abs() < 1e-6);
        assert!((emb.data[1] - 2.0).abs() < 1e-6);
        assert!((emb.data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_encode_response_rejects_wrong_dim() {
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 4096,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        let body = serde_json::json!({
            "embedding": [1.0, 2.0, 3.0]
        });

        let err = client.parse_encode_response(&body).unwrap_err();
        assert!(err.to_string().contains("Embedding dim mismatch"));
    }

    #[test]
    fn test_parse_encode_response_rejects_missing_key() {
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 3,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        let body = serde_json::json!({"embeddings": [[1.0, 2.0, 3.0]]});
        let err = client.parse_encode_response(&body).unwrap_err();
        assert!(err.to_string().contains("Missing 'embedding'"));
    }

    #[test]
    fn test_encode_capability_constants() {
        assert_ne!(ENCODE_UNKNOWN, ENCODE_POOLED);
        assert_ne!(ENCODE_UNKNOWN, ENCODE_LEGACY);
        assert_ne!(ENCODE_POOLED, ENCODE_LEGACY);
    }

    #[test]
    fn test_batch_generate_request_replicates_prompts() {
        // Verify that building a batch request with 2 states × 3 candidates
        // produces a flat array of 6 prompts
        let prompts = vec!["prompt_A", "prompt_B"];
        let n = 3;
        let mut flat: Vec<String> = Vec::new();
        for p in &prompts {
            for _ in 0..n {
                flat.push(p.to_string());
            }
        }
        let req = BatchGenerateRequest {
            text: flat,
            sampling_params: SamplingParams {
                max_new_tokens: 128,
                temperature: Some(0.6),
                top_p: Some(0.95),
                n: None,
            },
            return_logprob: true,
            return_hidden_states: false,
        };
        let json = serde_json::to_value(&req).unwrap();
        let text_arr = json["text"].as_array().unwrap();
        assert_eq!(text_arr.len(), 6);
        // First 3 are prompt_A, next 3 are prompt_B
        assert_eq!(text_arr[0], "prompt_A");
        assert_eq!(text_arr[2], "prompt_A");
        assert_eq!(text_arr[3], "prompt_B");
        assert_eq!(text_arr[5], "prompt_B");
        assert_eq!(json["return_logprob"], true);
        assert!(json.get("return_hidden_states").is_none()); // skipped when false
    }

    #[tokio::test]
    async fn test_batch_generate_empty_input() {
        let config = SglangConfig {
            server_url: "http://localhost:30000".to_string(),
            temperature: 0.6,
            top_p: 0.95,
            max_tactic_tokens: 128,
            hidden_size: 4096,
        };
        let client = SglangClient {
            client: Client::new(),
            base_url: Url::parse("http://localhost:30000").unwrap(),
            config,
            encode_capability: std::sync::Arc::new(AtomicU8::new(ENCODE_UNKNOWN)),
            circuit: CircuitBreaker::default(),
        };

        let result = client.generate_candidates_batch(&[], 8).await.unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_generate_parse_and_unflatten() {
        // Simulate: 2 states × 3 candidates = 6 response items
        // State 0: "simp", "ring", "simp" (dup)
        // State 1: "omega", "", "omega" (empty + dup)
        use crate::prompt::extract_first_tactic;

        let items: Vec<serde_json::Value> = vec![
            serde_json::json!({"text": "simp", "meta_info": {"output_token_logprobs": [[-0.5, 1]]}}),
            serde_json::json!({"text": "ring", "meta_info": {"output_token_logprobs": [[-1.0, 2]]}}),
            serde_json::json!({"text": "simp", "meta_info": {"output_token_logprobs": [[-0.3, 1]]}}),
            serde_json::json!({"text": "omega", "meta_info": {"output_token_logprobs": [[-0.2, 3]]}}),
            serde_json::json!({"text": "", "meta_info": {"output_token_logprobs": []}}),
            serde_json::json!({"text": "omega", "meta_info": {"output_token_logprobs": [[-0.8, 3]]}}),
        ];

        let n = 3;
        let num_states = 2;
        let mut all_results: Vec<Vec<GeneratedTactic>> = Vec::new();

        for chunk in items.chunks(n) {
            let mut responses: Vec<(String, f64)> = Vec::new();
            for item in chunk {
                let text = item.get("text").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let log_prob = item.get("meta_info")
                    .and_then(|m| m.get("output_token_logprobs"))
                    .and_then(|lps| lps.as_array())
                    .map(|lps| lps.iter().filter_map(|e| e.as_array()?.first()?.as_f64()).sum::<f64>())
                    .unwrap_or(0.0);
                if !text.is_empty() {
                    responses.push((text, log_prob));
                }
            }

            let mut tactics = Vec::new();
            for (raw_text, log_prob) in &responses {
                let tactic_text = extract_first_tactic(raw_text);
                if tactic_text.is_empty() { continue; }
                if tactics.iter().any(|t: &GeneratedTactic| t.text == tactic_text) { continue; }
                tactics.push(GeneratedTactic {
                    text: tactic_text,
                    raw_text: raw_text.clone(),
                    log_prob: *log_prob,
                    tokens: Vec::new(),
                });
            }
            tactics.sort_by(|a, b| b.log_prob.partial_cmp(&a.log_prob).unwrap_or(std::cmp::Ordering::Equal));
            all_results.push(tactics);
        }

        assert_eq!(all_results.len(), num_states);
        // State 0: "simp" (first occurrence, lp=-0.5) and "ring" (lp=-1.0), sorted by lp desc
        assert_eq!(all_results[0].len(), 2);
        assert_eq!(all_results[0][0].text, "simp"); // -0.5 > -1.0
        assert_eq!(all_results[0][1].text, "ring");
        // State 1: only "omega" (first occurrence, lp=-0.2), empty skipped, dup skipped
        assert_eq!(all_results[1].len(), 1);
        assert_eq!(all_results[1][0].text, "omega");
    }
}
