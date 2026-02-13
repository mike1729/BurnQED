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

use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use url::Url;

use crate::prompt::{extract_first_tactic, format_tactic_message};
use crate::types::{Embedding, GeneratedTactic};

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

/// HTTP client for SGLang inference server.
#[derive(Clone)]
pub struct SglangClient {
    client: Client,
    base_url: Url,
    config: SglangConfig,
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

#[derive(Deserialize, Debug)]
struct GenerateResponse {
    text: String,
    meta_info: MetaInfo,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct MetaInfo {
    #[serde(default)]
    output_token_logprobs: Option<Vec<LogprobEntry>>,
    #[serde(default)]
    hidden_states: Option<Vec<Vec<Vec<f32>>>>,
    #[serde(default)]
    finish_reason: Option<String>,
}

/// Each logprob entry is (logprob, token_id, token_str_or_null).
/// We only need the logprob value.
#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct LogprobEntry(f64, u32, Option<String>);

impl SglangClient {
    /// Create a new client and verify the server is reachable.
    pub async fn new(config: SglangConfig) -> anyhow::Result<Self> {
        let base_url = Url::parse(&config.server_url)
            .map_err(|e| anyhow::anyhow!("Invalid server URL '{}': {e}", config.server_url))?;

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(120))
            .build()?;

        let this = Self {
            client,
            base_url,
            config,
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

        let mut responses = Vec::with_capacity(n);
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

            let resp = self.post_with_retry(&url, &request).await?;
            let body = resp.bytes().await?;
            match serde_json::from_slice::<GenerateResponse>(&body) {
                Ok(gen) => responses.push(gen),
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
        for (i, gen) in responses.iter().enumerate() {
            let raw_text = &gen.text;
            let tactic_text = extract_first_tactic(raw_text);

            let log_prob = gen
                .meta_info
                .output_token_logprobs
                .as_ref()
                .map(|lps| lps.iter().map(|lp| lp.0).sum())
                .unwrap_or(0.0);

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

    /// Encode a proof state to a mean-pooled embedding via hidden states.
    pub async fn encode(&self, text: &str) -> anyhow::Result<Embedding> {
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
        let resp = self.post_with_retry(&url, &request).await?;
        let gen: GenerateResponse = resp.json().await?;

        let hidden_states = gen
            .meta_info
            .hidden_states
            .ok_or_else(|| anyhow::anyhow!(
                "Server did not return hidden_states. Ensure --enable-return-hidden-states is set."
            ))?;

        // Shape: (1, num_tokens, hidden_size) — index [0] to remove batch dim
        let token_states = hidden_states
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty hidden_states batch dimension"))?;

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
    async fn post_with_retry(
        &self,
        url: &Url,
        body: &GenerateRequest,
    ) -> anyhow::Result<reqwest::Response> {
        let mut last_err = None;
        for attempt in 0..3 {
            if attempt > 0 {
                let delay = Duration::from_millis(500 * (1 << attempt));
                tracing::warn!(attempt, delay_ms = delay.as_millis() as u64, "Retrying SGLang request");
                tokio::time::sleep(delay).await;
            }

            match self.client.post(url.clone()).json(body).send().await {
                Ok(resp) if resp.status().is_server_error() => {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    last_err = Some(anyhow::anyhow!(
                        "SGLang server error {status}: {body_text}"
                    ));
                }
                Ok(resp) if resp.status().is_client_error() => {
                    let status = resp.status();
                    let body_text = resp.text().await.unwrap_or_default();
                    anyhow::bail!("SGLang client error {status}: {body_text}");
                }
                Ok(resp) => return Ok(resp),
                Err(e) => {
                    last_err = Some(anyhow::anyhow!("SGLang request failed: {e}"));
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
        };

        let prompt = client.format_prompt("n : Nat\n\u{22a2} n + 0 = n");
        assert!(prompt.contains("<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"));
        assert!(prompt.contains("<\u{ff5c}User\u{ff5c}>"));
        assert!(prompt.contains("<\u{ff5c}Assistant\u{ff5c}>"));
        assert!(prompt.contains("tactic state:"));
        assert!(prompt.contains("n + 0 = n"));
    }

    #[test]
    fn test_logprob_entry_deserialize() {
        let json = r#"[-0.123, 42, null]"#;
        let entry: LogprobEntry = serde_json::from_str(json).unwrap();
        assert!((entry.0 - (-0.123)).abs() < 1e-6);
        assert_eq!(entry.1, 42);
        assert!(entry.2.is_none());
    }

    #[test]
    fn test_logprob_entry_with_token_str() {
        let json = r#"[-0.5, 100, "hello"]"#;
        let entry: LogprobEntry = serde_json::from_str(json).unwrap();
        assert!((entry.0 - (-0.5)).abs() < 1e-6);
        assert_eq!(entry.1, 100);
        assert_eq!(entry.2.as_deref(), Some("hello"));
    }

    #[test]
    fn test_generate_response_deserialize() {
        let json = r#"{
            "text": "intro n",
            "meta_info": {
                "output_token_logprobs": [[-0.1, 42, null], [-0.2, 43, null]],
                "finish_reason": "stop"
            }
        }"#;
        let resp: GenerateResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "intro n");
        let lps = resp.meta_info.output_token_logprobs.unwrap();
        assert_eq!(lps.len(), 2);
        let total: f64 = lps.iter().map(|lp| lp.0).sum();
        assert!((total - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_hidden_states_response_deserialize() {
        // Shape: (1, 2, 3) — batch=1, tokens=2, hidden_size=3
        let json = r#"{
            "text": "",
            "meta_info": {
                "hidden_states": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
            }
        }"#;
        let resp: GenerateResponse = serde_json::from_str(json).unwrap();
        let hs = resp.meta_info.hidden_states.unwrap();
        assert_eq!(hs.len(), 1);       // batch
        assert_eq!(hs[0].len(), 2);    // tokens
        assert_eq!(hs[0][0].len(), 3); // hidden_size
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
}
