# Spindle: Lean 4 Theorem Prover — Final Implementation Plan

## 1. Project Overview

**Goal:** Build an automated theorem prover for Lean 4 that combines an autoregressive LLM (tactic generator / policy) with an Energy-Based Model (value function) to guide best-first proof search. Trained via expert iteration.

**Key motivation for training the EBM in burn-rs:** The EBM is a custom architecture (frozen 7B encoder + trainable energy head) that requires several features burn-rs doesn't yet have. This is an opportunity to learn burn-rs deeply, identify and fill gaps, and contribute high-quality PRs upstream.

### 1.1 Architecture Split

| Concern | Language | Framework | Rationale |
|---------|----------|-----------|-----------|
| LLM fine-tuning | Python | PyTorch + HuggingFace + PEFT | Mature LoRA tooling, no reason to rewrite |
| **EBM training** | **Rust** | **burn-rs** | **Learn the framework, contribute missing pieces upstream** |
| LLM inference in search | Python/Rust | SGLang server + reqwest HTTP client | GPU-optimized batching, hidden-state extraction |
| EBM inference in search | Rust | burn-rs | Same crate as training — zero friction |
| Search engine | Rust | tokio + custom | Zero-cost async, concurrent Lean REPL management |
| Trajectory I/O | Rust | arrow-rs / parquet | Efficient columnar storage between iterations |
| Data preparation | Python | LeanDojo | One-time Mathlib tracing |

### 1.2 System Diagram (Revised: Shared 7B Backbone)

```
┌─────────────────────────────────────────────────┐
│          DeepSeek-Prover-V2-7B                  │
│         (SGLang server, frozen)                 │
│                                                 │
│  ┌───────────────────────────────────────┐      │
│  │       Transformer Backbone            │      │
│  │       (shared computation)            │      │
│  └─────────┬────────────────┬────────────┘      │
│            │                │                   │
│    ┌───────▼──────┐  ┌─────▼──────────┐        │
│    │   LM Head    │  │  Mean Pool     │        │
│    │  (policy)    │  │  → Vec<f32>    │        │
│    │  /generate   │  │  /encode       │        │
│    │  endpoint    │  └──────┬─────────┘        │
│    └──────────────┘         │                   │
└─────────────────────────────┼───────────────────┘
                              │ detached embeddings (HTTP)
                              ▼
                ┌──────────────────────────────┐
                │  Energy Head                 │
                │  (burn-rs, trainable)        │
                │  SpectralNorm MLP            │
                │  4096 → 2048 → 1024 → 512→1 │
                │                              │
                │  ← ONLY THIS TRAINS →        │
                └──────────────────────────────┘
```

This is the AlphaZero/MuZero architecture: one network with a policy head and a value head sharing the same trunk. LLM inference is delegated to an SGLang server for GPU-optimized batching; the Rust side communicates via HTTP.

### 1.3 Data Flow (Revised)

```
Python: LLM fine-tune → export safetensors → restart SGLang server
Python: One-time Mathlib trace → theorem index + tactic pairs
Rust (reqwest → SGLang): /encode endpoint provides embeddings as Vec<f32>
Rust (burn): Train energy head on embeddings from SGLang
Rust (search → SGLang): Run search on 75K theorems → write Parquet trajectories
Rust (burn): Retrain energy head on accumulated trajectories
...repeat
```

The Rust side communicates with SGLang via HTTP. The EBM head lives entirely in Rust.

### 1.4 Key Architecture Decisions (Settled After Two Review Rounds)

**Shared 7B backbone, not separate 1.3B encoder.**

| Aspect | Original (v3) | Revised (final) |
|--------|---------------|-----------------|
| EBM encoder | Separate 1.3B (DeepSeek-Coder) | Shared 7B (DeepSeek-Prover) via SGLang |
| Encoder framework | tch (TorchScript) | SGLang server (HTTP /encode endpoint) |
| GPU memory for models | ~17 GB (7B + 1.3B) | ~14 GB (7B only, served by SGLang) |
| Encoder quality on Lean | Code model (mediocre) | Lean-specialized (excellent) |
| EBM training complexity | Needs per-param-group optimizer | Single optimizer on head only |
| Tokenizer | Two (LLM + encoder) | One (managed by SGLang server) |
| Python encoder export | Required (TorchScript/ONNX) | Not needed (SGLang serves model directly) |
| EBM inference during search | Separate 1.3B forward passes | 7B /encode calls (batched by SGLang) |

**Latency trade-off analysis:**

| Scenario | 7B forward passes per expansion | Time |
|----------|--------------------------------|------|
| Separate 1.3B encoder | 1 (7B for tactics) + 8 (1.3B for EBM) | ~15ms + ~24ms = ~39ms |
| Shared 7B backbone | 1 (7B for tactics, free EBM on root) + 8 (7B for child EBM) | ~15ms + ~120ms = ~135ms |
| Shared 7B + encoder-only mode | 1 + 8 (encoder-only, no generation) | ~15ms + ~50ms = ~65ms |

The shared backbone is slower but the better representation quality of the Lean-native 7B model likely outweighs this for hard theorems. Made configurable — start with shared 7B, fall back to dedicated 1.3B if profiling shows bottleneck.

**No ONNX import.** Both reviewers flagged burn-import's ONNX support as brittle for transformers with RoPE/GQA. With shared backbone served by SGLang, no Rust-side model loading is needed.

**No dual tokenizer.** Custom Lean tokenizer was considered (bag-of-embeddings path) but cut — without self-attention it adds noise, not signal. If context length becomes an issue, use smart truncation (keep goal, truncate hypotheses) rather than parallel tokenizer path. Custom tokenizer training remains as future option for native burn-rs encoder.

**No "skip easy" EBM bypass.** LLMs are confidently wrong on formal math — `simp` will always have high log-prob but often fails. The EBM is most valuable precisely when the LLM is confident, because that's where log-prob alone gives misleading signal.

**Worker recycling for Lean processes.** Lean 4 processes leak memory via cached environments, elaboration state, and type-class resolution caches. Workers restart after 1000 requests or 30 minutes.

**ProofHandle pattern for state ID routing.** Pantograph `stateId` values are process-local — each child process has its own monotonically increasing counter. A standalone `pool.run_tactic(state_id, ...)` is always wrong with >1 worker because it could route to a different process than the one that created that `state_id`. Instead, `pool.start_proof()` returns a `ProofHandle` that holds the worker for the entire proof attempt. All tactics go through the handle. `ProofHandleOwned` (with `Arc<LeanPool>`) provides a `'static` variant for `tokio::spawn`. Raw `checkout()` / `WorkerGuard` is available for advanced multi-proof-on-one-worker scenarios (e.g., proof search simulation).

---

## 2. burn-rs Gap Analysis

### 2.1 What burn-rs Already Has

| Feature | burn-rs Status | Our Use |
|---------|---------------|---------|
| `Linear`, `LayerNorm`, `Dropout`, `Embedding` | ✅ Stable | Energy head, encoder components |
| `Module` derive macro | ✅ Stable | All model definitions |
| `AutodiffBackend` | ✅ Stable | Gradient computation |
| `AdamW` optimizer | ✅ Stable | Primary optimizer |
| `GradientClipping` | ✅ Stable | Clip grad norm to 1.0 |
| `CosineAnnealingLrScheduler` | ✅ Stable | LR schedule |
| `Learner` training abstraction | ✅ Stable | Training loop orchestration |
| `Dataset` trait | ✅ Stable | Data loading |
| ONNX import (`burn-import`) | ✅ Stable | Fallback encoder import |
| PyTorch import | ✅ Stable | Alternative encoder import |
| `burn-tch` backend (LibTorch) | ✅ Stable | GPU training via CUDA |
| Multi-head attention | ✅ Stable | Inside imported encoder |
| SiLU / Swish activation | ✅ Stable | Energy head activation |
| `CrossEntropyLoss` | ✅ Stable | Part of InfoNCE |

### 2.2 What's Missing — PR Opportunities

| Missing Feature | Complexity | Impact | Priority |
|----------------|-----------|--------|----------|
| **SpectralNorm module** | Medium | High — needed for GANs, EBMs, Lipschitz networks | PR #1 |
| **InfoNCE / contrastive loss** | Low-Medium | High — used in CLIP, SimCLR, contrastive learning | PR #2 |
| **Per-parameter-group optimizer** | Medium-High | Very high — needed for any transfer learning | PR #3 (**deprioritized** — not needed with shared backbone) |
| **Warmup + scheduler composition** | Low | Medium — common training pattern | PR #4 |
| **Parquet Dataset adapter** | Medium | Medium — useful for large-scale data pipelines | PR #5 |
| **Mean pooling utility** | Low | Low — simple but commonly needed | PR #6 |

### 2.3 Detailed PR Specifications

#### PR #1: SpectralNorm Module

**What it is:** A wrapper that normalizes a layer's weight matrix by its largest singular value (spectral norm) at each forward pass, ensuring the layer is 1-Lipschitz. Uses power iteration to approximate the spectral norm efficiently.

**Why burn-rs needs it:** Standard technique for stabilizing GAN discriminators, EBMs, and any network where Lipschitz continuity matters. PyTorch has `torch.nn.utils.spectral_norm`.

**Proposed API:**

```rust
/// Configuration for spectral normalization wrapper.
#[derive(Config, Debug)]
pub struct SpectralNormConfig {
    /// Number of power iteration steps per forward pass.
    #[config(default = 1)]
    pub n_power_iterations: usize,
    /// Small constant for numerical stability.
    #[config(default = 1e-12)]
    pub eps: f64,
}

/// Wraps a Linear layer with spectral normalization.
/// On each forward pass, the weight is divided by its estimated spectral norm.
///
/// The spectral norm σ(W) is approximated via power iteration:
///   v ← W^T u / ‖W^T u‖
///   u ← W v / ‖W v‖
///   σ(W) ≈ u^T W v
///
/// The normalized weight is: W_SN = W / σ(W)
#[derive(Module, Debug)]
pub struct SpectralNormLinear<B: Backend> {
    /// The underlying linear layer (holds weight_orig, bias)
    linear: Linear<B>,
    /// Left singular vector estimate (not a gradient-tracked parameter)
    u: Tensor<B, 1>,
    /// Right singular vector estimate
    v: Tensor<B, 1>,
    /// Number of power iteration steps
    n_power_iterations: usize,
    /// Numerical stability epsilon
    eps: f64,
}
```

**Implementation notes for the PR:**
- `u` and `v` should be registered as buffers (not parameters) — they're updated in-place during forward but have no gradients
- burn-rs doesn't have a buffer concept distinct from `Param` — this might require either (a) using raw `Tensor` fields that survive serialization or (b) proposing a `Buffer` type to burn-rs (bigger PR)
- The detach/no-grad distinction is handled by burn's autodiff backend — tensors not wrapped in `Param` don't get gradients
- Should support wrapping Conv2d too (reshape weight to 2D internally)

**SpectralNorm persistence — three options for u/v vectors:**

**Option A: Return updated model** (burn-idiomatic but awkward in training loop)
```rust
fn forward(&self, input: Tensor<B, 2>) -> (Tensor<B, 2>, Self) {
    // ... power iteration ...
    let updated_self = Self { u: Param::from(new_u), v: Param::from(new_v), ..self.clone() };
    (output, updated_self)
}
```

**Option B: Param + detach** (for upstream PR)
```rust
#[derive(Module, Debug)]
pub struct SpectralNormLinear<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Option<Param<Tensor<B, 1>>>,
    // Wrap in Param for automatic serialization, but detach in forward
    u: Param<Tensor<B, 1>>,
    v: Param<Tensor<B, 1>>,
    n_power_iterations: usize,
    eps: f64,
}
```

The key insight: burn's `Param` gives you serialization for free. Calling `.val().detach()` in forward prevents gradients from flowing back into u/v. The optimizer will technically "see" u and v as parameters, but with zero gradients they won't be updated by the optimizer — only by our explicit power iteration logic.

**Option C: Recompute from random init each forward** (simplest, use for v1)
```rust
fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let device = input.device();
    let mut u = Tensor::random([self.d_output], Distribution::Normal(0.0, 1.0), &device);
    let mut v = Tensor::random([self.d_input], Distribution::Normal(0.0, 1.0), &device);
    // 5 iterations instead of 1 (compensates for random init)
    for _ in 0..5 { /* power iteration */ }
    // ... normalize weight and apply
}
```

**Decision for our codebase:** Start with Option C (simplest, no persistence headaches). Switch to Option A (return updated model) when submitting the upstream PR, as it's more idiomatic for burn-rs.

**For the upstream PR (#1):** Propose a `Buffer` trait discussion alongside the SpectralNorm implementation. This is a known gap in burn-rs — non-gradient persistent state — and the spectral norm u/v vectors are a perfect motivating example.

**Workaround if PR takes time:** Implement power iteration inline in our EBM forward pass. Functionally identical, just not reusable as a generic module.

---

#### PR #2: InfoNCE / Contrastive Loss

**What it is:** The InfoNCE loss treats one sample as "positive" and K others as "negatives", optimizing the model to assign the highest score to the positive. Core loss for CLIP, SimCLR, MoCo, and contrastive representation learning.

**Why burn-rs needs it:** Contrastive learning is one of the most common training paradigms in modern ML. Having this in `burn::nn::loss` would be broadly useful.

**Proposed API:**

```rust
/// InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.
///
/// Given a similarity matrix where entry (i,j) represents the similarity
/// between anchor i and candidate j, treats the diagonal (or specified
/// positive indices) as positives and all others as negatives.
///
/// L = -log( exp(sim(anchor, positive) / τ) / Σ_j exp(sim(anchor, j) / τ) )
///
/// This is equivalent to cross-entropy where the "correct class" is the positive.
#[derive(Config, Debug)]
pub struct InfoNceLossConfig {
    /// Temperature scaling parameter τ. Lower = sharper distribution.
    #[config(default = 0.07)]
    pub temperature: f64,
    /// Reduction method for batch of losses.
    #[config(default = "Reduction::Mean")]
    pub reduction: Reduction,
}

#[derive(Module, Debug)]
pub struct InfoNceLoss {
    temperature: f64,
    reduction: Reduction,
}

impl InfoNceLoss {
    /// Compute InfoNCE loss from pre-computed logits.
    ///
    /// # Arguments
    /// * `logits` - Shape (B, K+1) where column 0 is the positive score
    ///   and columns 1..K+1 are negative scores. These are raw scores
    ///   (NOT pre-divided by temperature).
    pub fn forward<B: Backend>(&self, logits: Tensor<B, 2>) -> Tensor<B, 1> {
        let [batch_size, _num_candidates] = logits.dims();
        let device = logits.device();
        let scaled_logits = logits / self.temperature;
        let labels = Tensor::<B, 1, Int>::zeros([batch_size], &device);
        cross_entropy_with_logits(scaled_logits, labels, self.reduction)
    }

    /// Compute InfoNCE from separate positive and negative score tensors.
    pub fn forward_from_scores<B: Backend>(
        &self,
        positive_scores: Tensor<B, 1>,
        negative_scores: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let pos = positive_scores.unsqueeze_dim(1);
        let logits = Tensor::cat(vec![pos, negative_scores], 1);
        self.forward(logits)
    }
}
```

**Implementation notes:**
- This is essentially cross-entropy with dynamic "class" construction, so it reuses existing cross-entropy internals
- Should include both the logits form and the separate pos/neg form
- Could also include a `forward_from_embeddings` variant that computes cosine similarity internally (for CLIP-style use)
- Temperature should be either a fixed config value or a learnable parameter — the PR should support both

---

#### PR #3: Per-Parameter-Group Optimizer (Deprioritized)

**Status:** No longer needed for our use case with the shared backbone architecture (encoder served by SGLang, only energy head trains in burn-rs). But still a high-impact contribution for the burn-rs ecosystem.

**Proposed approach (path of least resistance):**

```rust
/// Maps parameter names (or prefixes) to learning rate multipliers.
///
/// Example:
///   encoder.* → 0.1  (10× lower LR)
///   energy_head.* → 1.0  (full LR)
///   log_temperature → 0.5
#[derive(Config, Debug, Clone)]
pub struct ParameterGroupConfig {
    #[config(default = 1.0)]
    pub default_multiplier: f64,
    pub groups: Vec<ParameterGroup>,
}

#[derive(Config, Debug, Clone)]
pub struct ParameterGroup {
    pub name_prefix: String,
    pub lr_multiplier: f64,
    pub weight_decay: Option<f64>,
}

pub struct GroupedOptimizer<O> {
    inner: O,
    group_config: ParameterGroupConfig,
}
```

**Why this is the hardest PR:** burn-rs's optimizer trait is deeply integrated with the `Module` derive macro and `GradientsParams`. Changing how LR is dispatched requires touching the optimizer core.

**Workaround (if we ever need it):** Gradient scaling trick — multiply encoder grads by `encoder_lr / head_lr` before calling `optimizer.step(head_lr)`. This makes the encoder effectively get `encoder_lr`.

---

#### PR #4: Warmup + Scheduler Composition

```rust
#[derive(Config, Debug)]
pub struct LinearWarmupConfig {
    pub warmup_steps: usize,
}

/// Composes a warmup phase with any inner LR scheduler.
/// During warmup: lr = base_lr * (step / warmup_steps)
/// After warmup: delegates to inner scheduler
pub struct LinearWarmupScheduler<S: LrScheduler> {
    warmup_steps: usize,
    inner: S,
    base_lr: f64,
}

impl<S: LrScheduler> LrScheduler for LinearWarmupScheduler<S> {
    type Record = (usize, S::Record);

    fn step(&mut self) -> f64 {
        self.current_step += 1;
        if self.current_step <= self.warmup_steps {
            self.base_lr * (self.current_step as f64 / self.warmup_steps as f64)
        } else {
            self.inner.step()
        }
    }
}
```

**Note:** Check `burn::lr_scheduler::composed` before implementing — burn-rs may already have this.

---

#### PR #5: Parquet Dataset Adapter

```rust
use burn::data::dataset::Dataset;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;

pub struct ParquetDataset<T> {
    records: Vec<T>,  // Pre-loaded for simplicity; could be memory-mapped
}

impl<T> ParquetDataset<T>
where
    T: TryFrom<arrow::record_batch::RecordBatch>,
{
    pub fn from_file(path: &str) -> Self {
        // Read all row groups into memory
        // For very large datasets, implement lazy loading with memory mapping
        todo!()
    }
}

impl<T: Clone + Send + Sync> Dataset<T> for ParquetDataset<T> {
    fn get(&self, index: usize) -> Option<T> {
        self.records.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.records.len()
    }
}
```

---

#### PR #6: Mean Pooling Utility

```rust
/// Mean pooling over sequence dimension, respecting attention mask.
/// input: (batch, seq_len, hidden) 
/// mask:  (batch, seq_len) — 1 for real tokens, 0 for padding
/// output: (batch, hidden)
pub fn mean_pool<B: Backend>(
    hidden_states: Tensor<B, 3>,
    attention_mask: Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let mask_f = attention_mask.float().unsqueeze_dim(2);    // (B, S, 1)
    let masked = hidden_states * mask_f.clone();
    let summed = masked.sum_dim(1);                           // (B, H)
    let counts = mask_f.sum_dim(1).clamp_min(1e-9);          // (B, 1)
    summed / counts
}
```

---

## 3. Project Structure

```
lean-prover/
├── Cargo.toml                          # Workspace root
├── vendor/
│   └── Pantograph/                     # Git submodule (pinned d047b1d, v0.3.11)
├── crates/
│   ├── prover-core/                    # Main binary + CLI
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── config.rs
│   │       └── pipeline.rs            # Expert iteration orchestrator
│   │
│   ├── search/                         # Search algorithms
│   │   └── src/ ...
│   │
│   ├── lean-repl/                      # Lean 4 REPL async client
│   │   ├── build.rs                   # Emits LEAN_REPL_MANIFEST_DIR for auto-discovery
│   │   └── src/
│   │       ├── lib.rs                # Public API: LeanPool, ProofHandle, ProofHandleOwned
│   │       ├── worker.rs             # LeanWorker: spawn, communicate, recycle
│   │       ├── pool.rs               # LeanPool, ProofHandle, WorkerGuard (+Owned variants)
│   │       ├── session.rs            # ProofSession: stateful proof tracking (holds ProofHandle)
│   │       ├── protocol.rs           # Pantograph JSON request/response serde types
│   │       └── types.rs              # Types + discover_pantograph() auto-discovery
│   │
│   ├── policy/                         # SGLang HTTP client (tactic gen + encode)
│   │   └── src/
│   │       ├── model.rs               # InferenceHandle, SglangClient
│   │       └── types.rs               # PolicyConfig, GeneratedTactic, Embedding
│   │
│   ├── ebm/                            # EBM: model + training + inference (ALL burn-rs)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── model/
│   │       │   ├── mod.rs
│   │       │   ├── encoder.rs          # EncoderBackend enum (shared vs dedicated)
│   │       │   ├── energy_head.rs      # Spectral-normed MLP
│   │       │   ├── spectral_norm.rs    # SpectralNormLinear (→ PR #1)
│   │       │   └── ebm.rs             # Full EBM: encoder + head
│   │       ├── training/
│   │       │   ├── mod.rs
│   │       │   ├── loss.rs             # InfoNCE + depth regression (→ PR #2)
│   │       │   ├── data.rs             # Parquet dataset + negative mining batcher
│   │       │   ├── trainer.rs          # Training loop
│   │       │   └── metrics.rs          # EBM-specific metrics + health checks
│   │       └── inference.rs            # Batch scoring for search
│   │
│   ├── burn-contrib/                   # Reusable modules destined for upstream PRs
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── spectral_norm.rs        # PR #1
│   │       ├── info_nce_loss.rs        # PR #2
│   │       ├── grouped_optimizer.rs    # PR #3 (deprioritized)
│   │       ├── warmup_scheduler.rs     # PR #4
│   │       ├── parquet_dataset.rs      # PR #5
│   │       └── pooling.rs             # PR #6
│   │
│   └── trajectory/                     # Data collection + Parquet I/O
│       └── src/ ...
│
├── python/                             # LLM training + data prep ONLY
│   ├── training/
│   │   ├── train_llm.py
│   │   └── export_llm.py              # safetensors export for SGLang
│   ├── data/
│   │   ├── trace_mathlib.py
│   │   ├── prepare_tactic_pairs.py
│   │   └── train_tokenizer.py         # Future: custom Lean tokenizer
│   └── eval/
│       └── evaluate.py
│
├── configs/
│   ├── models.toml
│   └── search.toml
└── scripts/
    ├── setup_pantograph.sh             # One-time: init submodule + lake build
    ├── run_iteration.sh
    └── setup_environment.sh
```

Note the `burn-contrib` crate: this holds the generic reusable modules that will eventually become PRs to burn-rs upstream. During development, `ebm` depends on `burn-contrib`. Once PRs are merged, we replace the dependency with the upstream burn version.

### 3.1 Key Rust Dependencies

```toml
[workspace]
members = ["crates/*"]

[workspace.dependencies]
# LLM inference (SGLang HTTP client)
reqwest = { version = "0.12", features = ["json"] }

# EBM training + inference (burn)
burn = { version = "0.16", features = ["train", "autodiff", "metrics", "dataset"] }
burn-ndarray = "0.16"          # CPU backend (NdArray — avoids WGPU Windows issues)

# Trajectory I/O
arrow = "53"
parquet = "53"

# Async
tokio = { version = "1", features = ["full"] }

# Data I/O
arrow = { version = "53", features = ["prettyprint"] }
parquet = { version = "53", features = ["async"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
toml = "0.8"

# Utilities
anyhow = "1"
tracing = "0.1"
tracing-subscriber = "0.3"
rand = "0.8"
dashmap = "6"
indicatif = "0.17"
clap = { version = "4", features = ["derive"] }
ordered-float = "4"
```

### 3.2 Configuration Files

```toml
# configs/models.toml

[encoder]
# "shared" = use the policy model's backbone (7B, via SGLang /encode)
# "dedicated" = load a separate encoder (not currently used)
mode = "shared"

# Only used if mode = "shared"
shared_hidden_dim = 4096      # DeepSeek-Prover-V2-7B

# Only used if mode = "dedicated"
# dedicated_model_path = "models/encoder/encoder.pt"
# dedicated_hidden_dim = 2048   # 1.3B

[energy_head]
d_hidden1 = 512
d_hidden2 = 256
dropout = 0.1
n_power_iterations = 5        # Option C: random reinit needs more iterations

[llm]
model_name = "deepseek-ai/DeepSeek-Prover-V2-7B"
max_seq_len = 2048
num_candidates = 32           # Tactics generated per expansion
temperature = 0.8
```

```toml
# configs/search.toml

[search]
max_nodes = 600
max_depth = 50
beam_width = 8                # Top-k by LLM log-prob before Lean + EBM
alpha = 0.5                   # LLM log-prob weight
beta = 0.5                    # EBM score weight
timeout_per_theorem = 600     # seconds
# No ebm_skip_easy. Always score with EBM when available.

[search.iteration_0]
# Run search twice: once normal, once with high temperature for diverse negatives
normal_temperature = 0.8
noise_temperature = 1.2
noise_fraction = 0.3          # 30% of theorems get the high-T run

[lean_pool]
num_workers = 64
max_requests_per_worker = 1000
max_lifetime_secs = 1800      # 30 minutes
tactic_timeout_secs = 30
```

---

## 4. EBM Model Definition in burn-rs (crates/ebm/)

### 4.1 Spectral-Normalized Energy Head

```rust
// crates/ebm/src/model/spectral_norm.rs

use burn::prelude::*;
use burn::nn::{Linear, LinearConfig};

#[derive(Config, Debug)]
pub struct SpectralNormLinearConfig {
    pub d_input: usize,
    pub d_output: usize,
    #[config(default = 5)]
    pub n_power_iterations: usize,  // 5 for Option C (random reinit)
    #[config(default = 1e-12)]
    pub eps: f64,
    #[config(default = true)]
    pub bias: bool,
}

/// Linear layer with spectral normalization (Option C: random reinit per forward).
/// Weight is normalized by its spectral norm (largest singular value)
/// at each forward pass using power iteration from fresh random init.
#[derive(Module, Debug)]
pub struct SpectralNormLinear<B: Backend> {
    weight: Param<Tensor<B, 2>>,
    bias: Option<Param<Tensor<B, 1>>>,
    n_power_iterations: usize,
    eps: f64,
}

impl SpectralNormLinearConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SpectralNormLinear<B> {
        let weight = Tensor::random(
            [self.d_output, self.d_input],
            burn::tensor::Distribution::Normal(0.0, (2.0 / self.d_input as f64).sqrt()),
            device,
        );
        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([self.d_output], device)))
        } else {
            None
        };

        SpectralNormLinear {
            weight: Param::from_tensor(weight),
            bias,
            n_power_iterations: self.n_power_iterations,
            eps: self.eps,
        }
    }
}

impl<B: Backend> SpectralNormLinear<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let weight = self.weight.val();
        let [d_out, d_in] = weight.dims();
        let device = input.device();

        // Option C: fresh random init each call — converges in 3-5 power iterations
        let mut u = Tensor::random([d_out], burn::tensor::Distribution::Normal(0.0, 1.0), &device);
        let mut v = Tensor::random([d_in], burn::tensor::Distribution::Normal(0.0, 1.0), &device);

        for _ in 0..self.n_power_iterations {
            // v = W^T u / ‖W^T u‖
            let wt_u = weight.clone().transpose().matmul(u.clone().unsqueeze_dim(1)).squeeze(1);
            v = wt_u.clone() / (wt_u.powf_scalar(2.0).sum().sqrt() + self.eps);

            // u = W v / ‖W v‖
            let w_v = weight.clone().matmul(v.clone().unsqueeze_dim(1)).squeeze(1);
            u = w_v.clone() / (w_v.powf_scalar(2.0).sum().sqrt() + self.eps);
        }

        // σ = u^T W v
        let sigma = u.clone().unsqueeze_dim(0)
            .matmul(weight.clone().matmul(v.clone().unsqueeze_dim(1)))
            .reshape([1])
            .squeeze(0);

        // W_normed = W / σ
        let w_normed = weight / sigma;

        let output = input.matmul(w_normed.transpose());
        match &self.bias {
            Some(b) => output + b.val().unsqueeze_dim(0),
            None => output,
        }
    }
}
```

### 4.2 Energy Head

```rust
// crates/ebm/src/model/energy_head.rs
use burn::prelude::*;
use crate::model::spectral_norm::{SpectralNormLinear, SpectralNormLinearConfig};

#[derive(Config, Debug)]
pub struct EnergyHeadConfig {
    pub d_encoder: usize,     // 4096 for DeepSeek-Prover-V2-7B
    #[config(default = 512)]
    pub d_hidden1: usize,
    #[config(default = 256)]
    pub d_hidden2: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// Energy head: spectral-normed MLP mapping encoder output to scalar energy.
/// Architecture: d_encoder → d_hidden1 → d_hidden2 → 1
/// All Linear layers are spectral-normalized for Lipschitz continuity.
#[derive(Module, Debug)]
pub struct EnergyHead<B: Backend> {
    sn_linear1: SpectralNormLinear<B>,
    sn_linear2: SpectralNormLinear<B>,
    sn_linear3: SpectralNormLinear<B>,
    dropout1: burn::nn::Dropout,
    dropout2: burn::nn::Dropout,
    log_temperature: Param<Tensor<B, 1>>,
}

impl EnergyHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EnergyHead<B> {
        EnergyHead {
            sn_linear1: SpectralNormLinearConfig::new(self.d_encoder, self.d_hidden1).init(device),
            sn_linear2: SpectralNormLinearConfig::new(self.d_hidden1, self.d_hidden2).init(device),
            sn_linear3: SpectralNormLinearConfig::new(self.d_hidden2, 1)
                .with_bias(false)
                .init(device),
            dropout1: burn::nn::DropoutConfig::new(self.dropout).init(),
            dropout2: burn::nn::DropoutConfig::new(self.dropout).init(),
            log_temperature: Param::from_tensor(Tensor::zeros([1], device)),
        }
    }
}

impl<B: Backend> EnergyHead<B> {
    /// Compute scalar energy from pooled encoder output.
    /// Input: (batch, d_encoder) → Output: (batch,)
    pub fn forward(&self, h: Tensor<B, 2>) -> Tensor<B, 1> {
        let x = self.sn_linear1.forward(h);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout1.forward(x);

        let x = self.sn_linear2.forward(x);
        let x = burn::tensor::activation::silu(x);
        let x = self.dropout2.forward(x);

        let raw_energy = self.sn_linear3.forward(x).squeeze(1); // (batch,)
        let temperature = self.log_temperature.val().exp();
        raw_energy / temperature
    }
}
```

### 4.3 Encoder Backend (Configurable: Shared 7B or Dedicated 1.3B)

> **Implementation note:** The actual implementation uses SGLang's HTTP `/encode` endpoint for shared-backbone encoding instead of in-process candle/tch. The `EncoderBackend` enum in `crates/ebm/src/model/encoder.rs` is a config enum; the real encoding is done via `InferenceHandle::encode_blocking()` in the policy crate. See `crates/policy/src/model.rs`.

### 4.4 Encode-Only Mode

> **Implementation note:** The original plan called for candle in-process inference. The actual implementation uses an SGLang server with an HTTP `/encode` endpoint that returns mean-pooled hidden states as `Vec<f32>`. See `crates/policy/src/model.rs` (`InferenceHandle::encode_blocking()`). The code sample below is the original plan sketch and was not implemented as written.

```rust
// HISTORICAL PLAN — actual implementation uses SGLang HTTP /encode endpoint
// See crates/policy/src/model.rs for the real implementation
```

### 4.5 Tensor Bridge: SGLang → burn

```rust
fn tch_to_burn<B: Backend>(tch_tensor: &TchTensor, device: &B::Device) -> Tensor<B, 2> {
    // TODO: Zero-copy sharing when B = TchBackend.
    // Both burn-tch and raw tch use LibTorch underneath.
    // There should be a way to share the underlying storage pointer
    // without a GPU→CPU→GPU round-trip.
    // Current cost: ~50µs per batch (negligible vs 15ms encoder forward).
    // Revisit if EBM inference is optimized below 1ms.
    let data: Vec<f32> = Vec::from(tch_tensor.flatten(0, -1));
    let shape: Vec<usize> = tch_tensor.size().iter().map(|&s| s as usize).collect();
    Tensor::from_data(burn::tensor::TensorData::new(data, shape), device)
}

fn embeddings_to_burn_tensor<B: Backend>(
    embeddings: &[Vec<f32>],
    device: &B::Device,
) -> Tensor<B, 2> {
    let batch_size = embeddings.len();
    let d_model = embeddings[0].len();
    let flat: Vec<f32> = embeddings.iter().flat_map(|e| e.iter().copied()).collect();
    Tensor::from_data(
        burn::tensor::TensorData::new(flat, [batch_size, d_model]),
        device,
    )
}
```

### 4.6 Full EBM Model (with shared backbone)

```rust
// crates/ebm/src/model/ebm.rs

pub struct EBMValueFunction<B: Backend> {
    encoder: EncoderBackend,
    energy_head: EnergyHead<B>,
    tokenizer: tokenizers::Tokenizer,
    max_seq_len: usize,
}

impl<B: Backend> EBMValueFunction<B> {
    /// Compute energy for raw proof state strings.
    /// Lower energy = more provable.
    pub fn batch_energy_from_text(&self, proof_states: &[&str], device: &B::Device) -> Vec<f64> {
        // Encode all states with the encoder backend
        let embeddings: Vec<Vec<f32>> = proof_states.iter()
            .map(|s| {
                let enc = self.tokenizer.encode(*s, true).unwrap();
                let ids: Vec<u32> = enc.get_ids().iter().copied().collect();
                let mask: Vec<u32> = vec![1; ids.len()];
                self.encoder.encode(&ids, &mask)
            })
            .collect();

        // Convert to burn tensor
        let tensor = embeddings_to_burn_tensor::<B>(&embeddings, device);

        // Forward through energy head
        let energies = self.energy_head.forward(tensor);
        energies.into_data().to_vec::<f64>().unwrap()
    }

    /// Provability score = -energy. Higher = more provable.
    pub fn batch_score_from_text(&self, proof_states: &[&str], device: &B::Device) -> Vec<f64> {
        self.batch_energy_from_text(proof_states, device)
            .into_iter()
            .map(|e| -e)
            .collect()
    }
}
```

## 5. EBM Training Data Pipeline in burn-rs

### 5.1 Parquet Dataset

```rust
// crates/ebm/src/training/data.rs
use burn::data::dataset::Dataset;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use arrow::array::{StringArray, UInt32Array, Int32Array, Float64Array};
use std::fs::File;

#[derive(Clone, Debug)]
pub struct ProofStateRecord {
    pub theorem_name: String,
    pub state_pp: String,
    pub label: String,          // "positive" or "negative"
    pub depth_from_root: u32,
    pub remaining_depth: i32,   // -1 if unknown
    pub ebm_score: f64,
}

pub struct TrajectoryDataset {
    records: Vec<ProofStateRecord>,
}

impl TrajectoryDataset {
    pub fn from_parquet(paths: &[&str]) -> Self {
        let mut records = Vec::new();

        for path in paths {
            let file = File::open(path).expect("Failed to open parquet file");
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)
                .expect("Failed to create reader");
            let reader = builder.build().expect("Failed to build reader");

            for batch in reader {
                let batch = batch.expect("Failed to read batch");
                let names = batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
                let states = batch.column(2).as_any().downcast_ref::<StringArray>().unwrap();
                let labels = batch.column(3).as_any().downcast_ref::<StringArray>().unwrap();
                let depths = batch.column(4).as_any().downcast_ref::<UInt32Array>().unwrap();
                let remaining = batch.column(5).as_any().downcast_ref::<Int32Array>().unwrap();
                let scores = batch.column(6).as_any().downcast_ref::<Float64Array>().unwrap();

                for i in 0..batch.num_rows() {
                    records.push(ProofStateRecord {
                        theorem_name: names.value(i).to_string(),
                        state_pp: states.value(i).to_string(),
                        label: labels.value(i).to_string(),
                        depth_from_root: depths.value(i),
                        remaining_depth: remaining.value(i),
                        ebm_score: scores.value(i),
                    });
                }
            }
        }

        Self { records }
    }

    /// Split into positive and negative records, indexed by theorem.
    pub fn build_index(&self) -> ContrastiveIndex {
        ContrastiveIndex::build(&self.records)
    }
}

impl Dataset<ProofStateRecord> for TrajectoryDataset {
    fn get(&self, index: usize) -> Option<ProofStateRecord> {
        self.records.get(index).cloned()
    }
    fn len(&self) -> usize {
        self.records.len()
    }
}
```

### 5.2 Contrastive Batch Construction (Negative Mining)

This is the critical data pipeline. Each batch contains (positive, [negatives]) tuples with a careful mix of hard, medium, and easy negatives.

```rust
// crates/ebm/src/training/data.rs (continued)
use burn::data::dataloader::batcher::Batcher;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Index for efficient contrastive sampling.
pub struct ContrastiveIndex {
    pos_by_theorem: HashMap<String, Vec<usize>>,
    neg_by_theorem: HashMap<String, Vec<usize>>,
    all_negatives: Vec<usize>,
    eligible_theorems: Vec<String>,
}

impl ContrastiveIndex {
    pub fn build(records: &[ProofStateRecord]) -> Self {
        let mut pos_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();
        let mut neg_by_theorem: HashMap<String, Vec<usize>> = HashMap::new();
        let mut all_negatives = Vec::new();

        for (i, rec) in records.iter().enumerate() {
            if rec.label == "positive" {
                pos_by_theorem.entry(rec.theorem_name.clone()).or_default().push(i);
            } else {
                neg_by_theorem.entry(rec.theorem_name.clone()).or_default().push(i);
                all_negatives.push(i);
            }
        }

        let eligible_theorems: Vec<String> = pos_by_theorem.keys()
            .filter(|t| neg_by_theorem.contains_key(*t))
            .cloned()
            .collect();

        Self { pos_by_theorem, neg_by_theorem, all_negatives, eligible_theorems }
    }
}

/// A single contrastive training example.
#[derive(Clone, Debug)]
pub struct ContrastiveExample {
    pub positive: ProofStateRecord,
    pub negatives: Vec<ProofStateRecord>,
}

/// A tokenized, padded batch ready for the model.
#[derive(Clone, Debug)]
pub struct ContrastiveBatch<B: Backend> {
    pub pos_input_ids: Tensor<B, 2, Int>,
    pub pos_attention_mask: Tensor<B, 2, Int>,
    pub neg_input_ids: Tensor<B, 3, Int>,      // (batch, K, seq_len)
    pub neg_attention_mask: Tensor<B, 3, Int>,
    pub remaining_depth: Tensor<B, 1>,          // (batch,) for depth regression
}

/// Batcher that constructs contrastive batches with negative mining.
pub struct ContrastiveBatcher<B: Backend> {
    records: Vec<ProofStateRecord>,
    index: ContrastiveIndex,
    tokenizer: tokenizers::Tokenizer,
    k_negatives: usize,
    max_seq_len: usize,
    hard_ratio: f64,     // 0.5 = 50% hard negatives
    medium_ratio: f64,   // 0.3 = 30% medium negatives
    device: B::Device,
}

impl<B: Backend> ContrastiveBatcher<B> {
    /// Sample one contrastive example: 1 positive + K negatives.
    fn sample_example(&self, rng: &mut impl rand::Rng) -> ContrastiveExample {
        let theorem = self.index.eligible_theorems.choose(rng).unwrap();
        let pos_indices = &self.index.pos_by_theorem[theorem];
        let pos_idx = *pos_indices.choose(rng).unwrap();
        let positive = self.records[pos_idx].clone();

        let mut negatives = Vec::with_capacity(self.k_negatives);
        let n_hard = (self.k_negatives as f64 * self.hard_ratio) as usize;
        let n_medium = (self.k_negatives as f64 * self.medium_ratio) as usize;
        let n_easy = self.k_negatives - n_hard - n_medium;

        // Hard: dead-end states from SAME theorem
        if let Some(theorem_negs) = self.index.neg_by_theorem.get(theorem) {
            for _ in 0..n_hard {
                let idx = *theorem_negs.choose(rng).unwrap();
                negatives.push(self.records[idx].clone());
            }
        }

        // Medium: off-path positives from same theorem (used as negatives)
        let siblings: Vec<usize> = pos_indices.iter()
            .copied()
            .filter(|&i| i != pos_idx)
            .collect();
        for _ in 0..n_medium {
            let pool = if siblings.is_empty() {
                self.index.neg_by_theorem.get(theorem).map(|v| v.as_slice()).unwrap_or(&[])
            } else {
                &siblings
            };
            if let Some(&idx) = pool.choose(rng) {
                negatives.push(self.records[idx].clone());
            }
        }

        // Easy: random states from other theorems
        for _ in 0..n_easy {
            if let Some(&idx) = self.index.all_negatives.choose(rng) {
                negatives.push(self.records[idx].clone());
            }
        }

        // Pad if we didn't get enough negatives
        while negatives.len() < self.k_negatives {
            if let Some(&idx) = self.index.all_negatives.choose(rng) {
                negatives.push(self.records[idx].clone());
            }
        }

        ContrastiveExample { positive, negatives }
    }
}

impl<B: Backend> Batcher<ContrastiveExample, ContrastiveBatch<B>> for ContrastiveBatcher<B> {
    fn batch(&self, examples: Vec<ContrastiveExample>) -> ContrastiveBatch<B> {
        let batch_size = examples.len();

        // Tokenize all positive states
        let pos_texts: Vec<&str> = examples.iter().map(|e| e.positive.state_pp.as_str()).collect();
        let (pos_ids, pos_mask) = tokenize_and_pad(&self.tokenizer, &pos_texts, self.max_seq_len, &self.device);

        // Tokenize all negative states: (batch_size * K) texts
        let neg_texts: Vec<&str> = examples.iter()
            .flat_map(|e| e.negatives.iter().map(|n| n.state_pp.as_str()))
            .collect();
        let (neg_ids_flat, neg_mask_flat) = tokenize_and_pad(
            &self.tokenizer, &neg_texts, self.max_seq_len, &self.device
        );

        let seq_len = neg_ids_flat.dims()[1];
        let neg_ids = neg_ids_flat.reshape([batch_size, self.k_negatives, seq_len]);
        let neg_mask = neg_mask_flat.reshape([batch_size, self.k_negatives, seq_len]);

        // Remaining depth for positive states
        let depths: Vec<f64> = examples.iter()
            .map(|e| e.positive.remaining_depth as f64)
            .collect();
        let remaining_depth = Tensor::from_floats(depths.as_slice(), &self.device);

        ContrastiveBatch {
            pos_input_ids: pos_ids,
            pos_attention_mask: pos_mask,
            neg_input_ids: neg_ids,
            neg_attention_mask: neg_mask,
            remaining_depth,
        }
    }
}

fn tokenize_and_pad<B: Backend>(
    tokenizer: &tokenizers::Tokenizer,
    texts: &[&str],
    max_len: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let encodings: Vec<_> = texts.iter()
        .map(|t| tokenizer.encode(*t, true).unwrap())
        .collect();

    let actual_max = encodings.iter()
        .map(|e| e.get_ids().len().min(max_len))
        .max()
        .unwrap_or(1);

    let n = texts.len();
    let mut ids = vec![0i64; n * actual_max];
    let mut mask = vec![0i64; n * actual_max];

    for (i, enc) in encodings.iter().enumerate() {
        let token_ids = enc.get_ids();
        let len = token_ids.len().min(actual_max);
        for j in 0..len {
            ids[i * actual_max + j] = token_ids[j] as i64;
            mask[i * actual_max + j] = 1;
        }
    }

    (
        Tensor::from_data(burn::tensor::TensorData::new(ids, [n, actual_max]), device),
        Tensor::from_data(burn::tensor::TensorData::new(mask, [n, actual_max]), device),
    )
}
```

---

## 6. EBM Training Loop in burn-rs

### 6.1 Training Loop (Revised for Shared Backbone)

With the shared backbone served by SGLang (frozen), only the energy head trains in burn-rs. This means a single AdamW optimizer with a single learning rate — no grouped optimizer workaround needed.

```rust
// crates/ebm/src/training/trainer.rs
use burn::prelude::*;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;

#[derive(Config, Debug)]
pub struct EBMTrainingConfig {
    pub lr: f64,                     // 1e-4 (single LR — only energy head trains)
    pub weight_decay: f64,           // 0.01
    pub max_grad_norm: f64,          // 1.0
    pub total_steps: usize,         // 50_000
    pub warmup_steps: usize,        // 1_000
    pub depth_loss_weight: f64,     // 0.3
    pub log_interval: usize,        // 50
    pub checkpoint_interval: usize, // 5_000
    pub k_negatives: usize,         // 4
    pub batch_size: usize,          // 32
}

pub struct EBMTrainer<B: AutodiffBackend> {
    config: EBMTrainingConfig,
    optimizer: burn::optim::AdamW<B>,
    device: B::Device,
}

impl<B: AutodiffBackend> EBMTrainer<B> {
    pub fn new(config: EBMTrainingConfig, device: &B::Device) -> Self {
        let optim_config = AdamWConfig::new()
            .with_weight_decay(config.weight_decay)
            .with_grad_clipping(
                Some(GradientClippingConfig::Norm(config.max_grad_norm))
            );

        Self {
            config: config.clone(),
            optimizer: optim_config.init(),
            device: device.clone(),
        }
    }

    /// Run the full training loop.
    pub fn train(
        &mut self,
        mut model: EnergyHead<B>,
        encoder: &InferenceHandle,      // SGLang client, frozen, used as encoder
        dataset: &TrajectoryDataset,
    ) -> EnergyHead<B> {
        let index = dataset.build_index();
        let mut rng = rand::thread_rng();

        let mut running_loss = 0.0;
        let mut running_gap = 0.0;

        for step in 0..self.config.total_steps {
            let lr = self.lr_schedule(step);

            // Step 1: Encode all states with the frozen 7B model (SGLang, no gradients)
            let batch = self.sample_contrastive_batch(dataset, &index, &mut rng);

            let pos_embeddings: Vec<Vec<f32>> = batch.pos_states.iter()
                .map(|s| encoder.encode_only(s))
                .collect();

            let neg_embeddings: Vec<Vec<f32>> = batch.neg_states.iter()
                .flat_map(|negs| negs.iter().map(|s| encoder.encode_only(s)))
                .collect();

            // Step 2: Convert to burn tensors
            let pos_tensor = embeddings_to_burn_tensor::<B>(&pos_embeddings, &self.device);
            let neg_tensor = embeddings_to_burn_tensor::<B>(&neg_embeddings, &self.device);
            let [bs, _] = pos_tensor.dims();
            let k = self.config.k_negatives;

            // Step 3: Forward through energy head (burn-rs, with gradients)
            let pos_energy = model.forward(pos_tensor);
            let neg_energy = model.forward(neg_tensor).reshape([bs, k]);

            // Step 4: Loss computation
            let loss_contrastive = info_nce_loss(pos_energy.clone(), neg_energy.clone());
            let loss_depth = depth_regression_loss(
                pos_energy.clone(),
                batch.remaining_depth.clone(),
            );
            let loss = loss_contrastive.clone()
                + loss_depth.clone() * self.config.depth_loss_weight;

            // Step 5: Backward through energy head only
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &model);
            model = self.optimizer.step(lr, model, grads_params);

            // Metrics
            let gap = neg_energy.clone().mean().into_scalar().to_f64()
                     - pos_energy.clone().mean().into_scalar().to_f64();
            running_loss += loss.clone().into_scalar().to_f64();
            running_gap += gap;

            if (step + 1) % self.config.log_interval == 0 {
                let avg_loss = running_loss / self.config.log_interval as f64;
                let avg_gap = running_gap / self.config.log_interval as f64;
                tracing::info!(
                    step = step + 1,
                    loss = format!("{avg_loss:.4}"),
                    energy_gap = format!("{avg_gap:.4}"),
                    lr = format!("{lr:.2e}"),
                );
                running_loss = 0.0;
                running_gap = 0.0;
            }

            if (step + 1) % self.config.checkpoint_interval == 0 {
                self.save_checkpoint(&model, step + 1);
            }
        }

        model
    }

    /// Linear warmup + cosine annealing schedule.
    fn lr_schedule(&self, step: usize) -> f64 {
        let base = self.config.lr;
        if step < self.config.warmup_steps {
            base * (step + 1) as f64 / self.config.warmup_steps as f64
        } else {
            let progress = (step - self.config.warmup_steps) as f64
                / (self.config.total_steps - self.config.warmup_steps) as f64;
            base * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos())
        }
    }

    fn save_checkpoint(&self, model: &EnergyHead<B>, step: usize) {
        let path = format!("checkpoints/ebm/step_{step}");
        model.save_file(&path, &burn::record::DefaultFileRecorder::new())
            .expect("Failed to save checkpoint");
        tracing::info!(step, path, "Saved checkpoint");
    }
}
```

### 6.2 Loss Functions

```rust
// crates/ebm/src/training/loss.rs
use burn::prelude::*;

/// InfoNCE loss: positive should have lowest energy (highest score).
///
/// L = -log( exp(-E(x+)) / (exp(-E(x+)) + Σ exp(-E(xi-))) )
///
/// This is cross-entropy where the "correct class" is the positive at index 0.
pub fn info_nce_loss<B: Backend>(
    pos_energy: Tensor<B, 1>,     // (batch,) — energy of positive states
    neg_energies: Tensor<B, 2>,   // (batch, K) — energies of K negatives
) -> Tensor<B, 1> {
    // Logits = -energy (higher logit = more likely)
    let pos_logits = pos_energy.neg().unsqueeze_dim(1);    // (B, 1)
    let neg_logits = neg_energies.neg();                    // (B, K)
    let logits = Tensor::cat(vec![pos_logits, neg_logits], 1); // (B, K+1)

    let [batch_size, _] = logits.dims();

    // Labels: the positive is at index 0 for every row
    let labels = Tensor::<B, 1, Int>::zeros([batch_size], &logits.device());

    // Cross-entropy
    burn::nn::loss::CrossEntropyLossConfig::new()
        .init(&logits.device())
        .forward(logits, labels)
}

/// Depth regression loss: energy should correlate with remaining proof depth.
/// States closer to QED (small remaining_depth) → lower energy.
/// Only applied to positive states (remaining_depth >= 0).
pub fn depth_regression_loss<B: Backend>(
    energy: Tensor<B, 1>,            // (batch,)
    remaining_depth: Tensor<B, 1>,   // (batch,) — -1 for unknown
) -> Tensor<B, 1> {
    // Mask: only include states where remaining_depth is known
    let mask = remaining_depth.clone().greater_equal_elem(0.0);
    let mask_f = mask.float();
    let count = mask_f.clone().sum();

    // Avoid division by zero if no valid depths in this batch
    let has_valid = count.clone().greater_elem(0.0);

    // Normalize remaining_depth to [0, 1]
    let rd = remaining_depth.clone() * mask_f.clone();
    let max_rd = rd.clone().max().clamp_min(1.0);
    let rd_norm = rd / max_rd;

    // Target: energy should equal normalized remaining depth
    // (deeper remaining = higher energy = less promising)
    let target = rd_norm;
    let diff = (energy - target) * mask_f;
    let mse = diff.clone().powf_scalar(2.0).sum() / count.clamp_min(1.0);

    // Return 0 if no valid depths, otherwise MSE
    mse * has_valid.float()
}
```

### 6.3 Training Metrics with Health Checks

```rust
// crates/ebm/src/training/metrics.rs

/// Metrics tracked during EBM training.
#[derive(Debug, Clone)]
pub struct EBMMetrics {
    /// Total loss (contrastive + depth regression)
    pub loss: f64,
    /// InfoNCE contrastive loss
    pub contrastive_loss: f64,
    /// Depth regression loss
    pub depth_loss: f64,
    /// Mean energy gap: mean(neg_energy) - mean(pos_energy)
    /// Should increase during training.
    pub energy_gap: f64,
    /// Mean energy of positive states (should decrease)
    pub pos_energy_mean: f64,
    /// Mean energy of negative states (should increase)
    pub neg_energy_mean: f64,
    /// Rank accuracy: fraction where positive has lowest energy in its group
    pub rank_accuracy: f64,
    /// Energy standard deviation (watch for collapse → 0)
    pub energy_std: f64,
    /// Spearman correlation between energy and remaining depth
    pub depth_correlation: f64,
}

impl EBMMetrics {
    /// Check for common training pathologies.
    pub fn health_check(&self) -> Vec<String> {
        let mut warnings = Vec::new();
        if self.energy_std < 0.1 {
            warnings.push(
                "⚠️  Energy std < 0.1: possible mode collapse. \
                 Increase temperature or LR.".into()
            );
        }
        if self.energy_gap < 0.0 {
            warnings.push(
                "⚠️  Negative energy gap: model scoring negatives \
                 lower than positives!".into()
            );
        }
        if self.rank_accuracy > 0.95 && self.contrastive_loss > 0.5 {
            warnings.push(
                "⚠️  High rank accuracy but high loss: negatives \
                 may be too easy.".into()
            );
        }
        warnings
    }
}
```

---

## 7. Lean REPL Client with Worker Recycling

### 7.0 Pantograph Setup (Bundled Submodule)

Pantograph is bundled as a git submodule at `vendor/Pantograph/` (pinned to commit `d047b1d`, v0.3.11).

```bash
# One-time setup
./scripts/setup_pantograph.sh
# Or manually:
git submodule update --init vendor/Pantograph
cd vendor/Pantograph && lake build
```

**Auto-discovery:** `discover_pantograph()` in `crates/lean-repl/src/types.rs` finds Pantograph automatically:
1. `PANTOGRAPH_PROJECT` env var (if set)
2. `vendor/Pantograph/` relative to workspace root (default — no env var needed)

`LeanPoolConfig::with_bundled_pantograph()` creates a config with auto-discovered path + sensible defaults:

```rust
let config = LeanPoolConfig::with_bundled_pantograph()
    .expect("Pantograph not found — run scripts/setup_pantograph.sh first");
let pool = LeanPool::new(config).await?;
```

The `build.rs` in `crates/lean-repl/` emits `LEAN_REPL_MANIFEST_DIR` at compile time so the runtime code can locate the workspace root.

### 7.1 Worker Pool and ProofHandle Pattern

Pantograph `stateId` values are process-local. The pool must never release a worker between
`start_proof` and `run_tactic` — a different worker won't have that state. The `ProofHandle`
pattern solves this: `pool.start_proof()` returns a handle that holds the worker.

```rust
// crates/lean-repl/src/pool.rs — key types

/// Pool manages N workers with semaphore-based concurrency.
pub struct LeanPool { /* workers: Mutex<Vec<LeanWorker>>, semaphore, config */ }

/// Handle to an in-progress proof. Holds a worker for the proof's lifetime.
/// Worker returned to pool on drop.
pub struct ProofHandle<'a> { guard: WorkerGuard<'a>, initial_state: ProofState }
/// Owned variant for tokio::spawn (holds Arc<LeanPool>).
pub struct ProofHandleOwned { guard: WorkerGuardOwned, initial_state: ProofState }

/// RAII guard for raw worker checkout.
pub struct WorkerGuard<'a> { worker: Option<LeanWorker>, pool_workers: &'a Mutex<Vec<LeanWorker>>, _permit: SemaphorePermit<'a> }
/// Owned variant (holds Arc<LeanPool>, OwnedSemaphorePermit).
pub struct WorkerGuardOwned { worker: Option<LeanWorker>, pool: Arc<LeanPool>, _permit: OwnedSemaphorePermit }

impl LeanPool {
    /// Start proof by expression → ProofHandle (worker held for proof lifetime).
    pub async fn start_proof(&self, expr: &str) -> Result<ProofHandle<'_>, LeanError>;
    /// Owned variant for tokio::spawn.
    pub async fn start_proof_owned(self: &Arc<Self>, expr: &str) -> Result<ProofHandleOwned, LeanError>;
    /// Start proof by Mathlib theorem name (copyFrom) → ProofHandle.
    pub async fn start_proof_by_name(&self, name: &str) -> Result<ProofHandle<'_>, LeanError>;
    /// Owned variant for tokio::spawn.
    pub async fn start_proof_by_name_owned(self: &Arc<Self>, name: &str) -> Result<ProofHandleOwned, LeanError>;
    /// Raw worker checkout for advanced use (e.g., multiple proofs on one worker).
    pub async fn checkout(&self) -> Result<WorkerGuard<'_>, LeanError>;
    pub async fn checkout_owned(self: &Arc<Self>) -> Result<WorkerGuardOwned, LeanError>;
}

impl ProofHandle<'_> {
    pub fn state_id(&self) -> u64;
    pub fn initial_state(&self) -> &ProofState;
    /// Apply tactic — guaranteed same worker that started this proof.
    pub async fn run_tactic(&mut self, state_id: u64, goal_id: Option<u64>, tactic: &str)
        -> Result<TacticResult, LeanError>;
    pub fn worker(&mut self) -> &mut LeanWorker;
}
// ProofHandleOwned has identical methods.
```

Usage:

```rust
// Basic proof:
let mut proof = pool.start_proof("forall (n : Nat), n = n").await?;
let sid = proof.state_id();
let r = proof.run_tactic(sid, None, "intro n").await?;
// ... more tactics on `proof` ...
drop(proof); // worker returned to pool

// Concurrent proofs via tokio::spawn:
let pool = Arc::new(pool);
tokio::spawn({
    let pool = pool.clone();
    async move {
        let mut proof = pool.start_proof_owned("forall (n : Nat), n = n").await.unwrap();
        // ... tactics ...
    }
});
```

There is **no standalone `pool.run_tactic()`** — it was removed because it's always wrong
with >1 worker (stateId from worker A sent to worker B).

Worker recycling (1000 requests / 30 min TTL) happens at checkout time. Workers recycle
transparently between proofs. `LeanWorker` internals:

```rust
// crates/lean-repl/src/worker.rs

pub struct LeanWorker {
    child: tokio::process::Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    requests_handled: u64,
    started_at: std::time::Instant,
    config: LeanPoolConfig,
}

impl LeanWorker {
    pub fn needs_recycling(&self) -> bool {
        self.requests_handled >= self.config.max_requests_per_worker
            || self.started_at.elapsed().as_secs() >= self.config.max_lifetime_secs
    }

    pub async fn recycle(&mut self) -> Result<(), LeanError> {
        let _ = self.child.kill().await;
        let _ = self.child.wait().await;
        // Spawn fresh process, reset counters, consume "ready." line
        // ...
    }
}
```

### 7.2 Pantograph Protocol (Critical Details)

The protocol layer is in `crates/lean-repl/src/protocol.rs` (serde types) and
`crates/lean-repl/src/worker.rs` (communication). Custom `Serialize`/`Deserialize`
implementations handle Pantograph's dot-notation commands and camelCase fields.

```rust
// crates/lean-repl/src/worker.rs — send_line (simplified)

impl LeanWorker {
    async fn send_line(&mut self, json: &str) -> Result<String, LeanError> {
        // CRITICAL: Pantograph expects exactly one JSON object per line.
        // Missing \n will hang the process indefinitely.
        self.stdin.write_all(json.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;

        // Read response with timeout
        let mut response_line = String::new();
        let timeout = std::time::Duration::from_secs(self.config.tactic_timeout_secs);

        let read_result = tokio::time::timeout(
            timeout,
            self.stdout.read_line(&mut response_line),
        ).await;

        match read_result {
            Ok(Ok(0)) => Err(LeanError::ProcessDied),
            Ok(Ok(_)) => {
                self.requests_handled += 1;
                Ok(response_line)
            }
            Ok(Err(e)) => Err(LeanError::Io(e)),
            Err(_) => {
                // Timeout — kill and recycle this worker
                self.recycle().await?;
                Err(LeanError::Timeout(self.config.tactic_timeout_secs))
            }
        }
    }
}
```

```rust
// crates/lean-repl/src/protocol.rs — request/response types (simplified)

// PantographRequest: custom Serialize (dot-notation "cmd", camelCase fields)
pub enum PantographRequest {
    GoalStart { expr: String },
    // → {"cmd": "goal.start", "payload": {"expr": "..."}}
    GoalTactic { state_id: u64, goal_id: Option<u64>, tactic: String },
    // → {"cmd": "goal.tactic", "payload": {"stateId": 0, "goalId": 0, "tactic": "..."}}
}

// PantographResponse: custom Deserialize (checks "error" key first)
pub enum PantographResponse {
    GoalStarted(GoalStartResult),
    TacticResult(TacticResultRaw),
    Error(ErrorResult),
}
```

---

## 8. Search Engine

### 8.1 Best-First Search

Per expansion (supports batched expansion of multiple nodes):

1. Pop highest-scored node(s) from priority queue (`batch_expansion_size`)
2. Batch-generate candidates via SGLang (`generate_candidates_batch`) — single HTTP request
   for all `states × n` prompts; SGLang RadixAttention caches shared prefixes
3. Inject probe tactics (deduped against LLM candidates)
4. Apply candidates in Lean, collect successful child states
5. **Deferred batch scoring**: all successful children scored in one `score_batch()` call
   - Single HTTP encode batch to SGLang → single MLP forward pass
   - Falls back to `ebm_score = 0.0` on error (search continues)
6. Push scored children onto priority queue: `α × log_prob + β × ebm_score`

After proof found, `harvest_siblings` expands proof-path ancestors for hard negatives, also using batch scoring.

### 8.2 Latency Budget per Expansion

With 600-node search budget and SGLang inference server:

- 600 expansions × 1 batched EBM encode call (~8 child states) each
- Batch encode via SGLang `/encode`: ~5-15ms per batch of 8
- Total EBM inference: 600 × ~10ms = **~6 seconds** per theorem
- Lean REPL calls dominate: 600 × 8 = 4800 applications, ~5-50ms each

### 8.3 Noise Injection for Iteration 0

```rust
// In trajectory collection for iteration 0:
// For 30% of theorems, run an additional high-temperature search
// to generate diverse failure modes for EBM training
if iteration == 0 && rng.gen::<f64>() < config.noise_fraction {
    let noisy_result = search_engine
        .with_temperature(config.noise_temperature)  // T=1.2
        .search(theorem)
        .await;
    // All states from noisy search are labeled as negatives
    // (even "successful" proofs from noisy search may be flukes)
    for node in &noisy_result.all_nodes {
        writer.record(TrajectoryRecord {
            label: "negative".into(),
            ..extract_record(node, theorem)
        });
    }
}
```

---

## 9. Expert Iteration Orchestration

```bash
#!/bin/bash
# scripts/run_iteration.sh
set -euo pipefail
ITER=${1:-0}
PREV=$((ITER - 1))

echo "=== EXPERT ITERATION $ITER ==="

if [ "$ITER" -eq 0 ]; then
    # One-time: trace Mathlib (no GPU needed)
    echo "--- Tracing Mathlib ---"
    python python/data/trace_mathlib.py

    # Phase 1: LLM fine-tuning (Python)
    echo "--- Training LLM (Python) ---"
    python python/training/train_llm.py \
        --data data/tactic_pairs/train.jsonl \
        --output checkpoints/llm/iter_0 --epochs 3
    python python/training/export_llm.py \
        --checkpoint checkpoints/llm/iter_0 \
        --output models/llm/iter_0

    # Phase 2: Search with LLM only (Rust, no EBM)
    echo "--- Initial trajectory collection (Rust, LLM only) ---"
    cargo run --release -p prover-core -- search \
        --llm-path models/llm/iter_0 \
        --no-ebm \
        --theorems data/theorem_index.json \
        --output trajectories/iter_0.parquet

    # Phase 2b: Noise injection for hard negatives
    echo "--- Noise injection run ---"
    cargo run --release -p prover-core -- search \
        --llm-path models/llm/iter_0 \
        --no-ebm \
        --temperature 1.2 \
        --noise-fraction 0.3 \
        --output trajectories/iter_0_noisy.parquet

    # Phase 3: EBM training (Rust, burn-rs!)
    echo "--- Training EBM (Rust/burn-rs) ---"
    cargo run --release -p prover-core -- train-ebm \
        --trajectories trajectories/iter_0.parquet,trajectories/iter_0_noisy.parquet \
        --output models/ebm/iter_0 \
        --steps 50000
else
    # Retrain LLM (Python)
    python python/training/train_llm.py \
        --data data/tactic_pairs/train.jsonl \
        --extra-data "trajectories/iter_*.parquet" \
        --output checkpoints/llm/iter_${ITER} \
        --base checkpoints/llm/iter_${PREV} \
        --epochs 1 --lr $(python -c "print(2e-4 * 0.5 ** $ITER)")
    python python/training/export_llm.py \
        --checkpoint checkpoints/llm/iter_${ITER} \
        --output models/llm/iter_${ITER}

    # Retrain EBM (Rust/burn-rs)
    cargo run --release -p prover-core -- train-ebm \
        --trajectories "trajectories/iter_*.parquet" \
        --output models/ebm/iter_${ITER} \
        --steps 50000 \
        --resume-from models/ebm/iter_${PREV}

    # Search with LLM + EBM (Rust)
    cargo run --release -p prover-core -- search \
        --llm-path models/llm/iter_${ITER} \
        --ebm-path models/ebm/iter_${ITER} \
        --theorems data/theorem_index.json \
        --output trajectories/iter_${ITER}.parquet
fi

# Evaluate
cargo run --release -p prover-core -- eval \
    --llm-path models/llm/iter_${ITER} \
    --ebm-path models/ebm/iter_${ITER} \
    --budgets 100,300,600

echo "=== ITERATION $ITER COMPLETE ==="
```

---

## 10. PR Development Strategy

### 10.1 Recommended PR Order (Revised)

```
PR #6 (Mean Pooling)         — Week 1    — Tiny, gets you familiar with contributing
PR #2 (InfoNCE Loss)         — Week 2    — Small, self-contained, high value
PR #4 (Warmup Scheduler)     — Week 3    — Small, clear scope
PR #1 (SpectralNorm)         — Week 4-5  — Medium, needs buffer discussion
PR #5 (Parquet Dataset)      — Week 4-5  — Medium, useful utility
PR #3 (Parameter Groups)     — Deprioritized — Not needed with shared backbone
```

### 10.2 PR Contribution Checklist (burn-rs standards)

Each PR should include:

- [ ] Implementation in the appropriate `burn-*` subcrate
- [ ] `Config` struct with `#[derive(Config)]`
- [ ] `Module` derive where applicable
- [ ] Unit tests (burn uses `#[cfg(test)]` modules)
- [ ] Integration test with at least 2 backends (ndarray + tch)
- [ ] Doc comments with examples
- [ ] Entry in the relevant module's `mod.rs` exports
- [ ] Example in the `examples/` directory (for larger PRs)
- [ ] Updated `CHANGELOG.md`

### 10.3 Testing Strategy for PRs

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    type TestBackend = NdArray<f32>;

    #[test]
    fn spectral_norm_reduces_largest_singular_value() {
        let device = Default::default();
        let layer = SpectralNormLinearConfig::new(64, 32).init::<TestBackend>(&device);
        let input = Tensor::<TestBackend, 2>::random(
            [4, 64], Distribution::Normal(0.0, 1.0), &device
        );
        let output = layer.forward(input.clone());

        // The spectral norm of the effective weight should be ≈ 1.0
        // (within tolerance due to power iteration approximation)
        let w_effective = /* extract normed weight */;
        let sigma = compute_spectral_norm(w_effective);
        assert!((sigma - 1.0).abs() < 0.1, "Spectral norm should be ≈ 1.0, got {sigma}");
    }

    #[test]
    fn info_nce_perfect_separation() {
        let device = Default::default();
        // Positive energy = -10 (very low), negative energies = 10 (very high)
        let pos = Tensor::<TestBackend, 1>::from_floats([-10.0, -10.0], &device);
        let neg = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0], [10.0, 10.0]], &device);
        let loss = info_nce_loss(pos, neg);
        // With perfect separation, loss should be very close to 0
        assert!(loss.into_scalar() < 0.01);
    }

    #[test]
    fn info_nce_random_gives_high_loss() {
        let device = Default::default();
        let pos = Tensor::<TestBackend, 1>::from_floats([0.0, 0.0], &device);
        let neg = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [0.0, 0.0]], &device);
        let loss = info_nce_loss(pos, neg);
        // With no separation, loss should be ≈ log(K+1) = log(3) ≈ 1.1
        assert!((loss.into_scalar() - (3.0_f64).ln()).abs() < 0.1);
    }
}
```

---

## 11. Solo Developer Timeline (Claude Code-Assisted)

Calibrated for one person developing on a MacBook (M-series, 32-64GB), deploying to cloud GPUs. Claude Code handles ~70% of code generation. You review, debug, and make architectural decisions.

### What the MacBook Handles

- Rust compilation and unit testing (fast — Rust compiles well on ARM)
- burn-rs with `burn-ndarray` backend (CPU, for testing EBM forward/backward)
- SGLang on CPU or small GPU (slow inference, but enough to test tactic generation on 1 theorem)
- Lean 4 + Pantograph locally (test REPL protocol)
- Python data scripts, tokenizer training
- Full integration test of the search loop on CPU (slow but functional)

### What Requires Cloud GPUs

- LLM fine-tuning (8× A100, hours)
- Trajectory collection / search (4-8× A100, hours to days)
- EBM training with real data (1-4× A100, hours)
- 7B inference at useful throughput (1× A100 minimum)

### Phase Breakdown

| Phase | Days | Key Deliverable | What Claude Code Writes | What You Debug |
|-------|------|----------------|------------------------|----------------|
| **0: Setup** | 1 | Everything compiles | Cargo workspace, all crate stubs | Dependency resolution |
| **1: Lean REPL** | 5 | Apply tactics programmatically | Worker pool, protocol, recycling | Pantograph protocol quirks, hangs |
| **2: LLM via SGLang** | 6 | Core search loop running | SGLang HTTP client, tactic generation, encode endpoint | Server connectivity, response parsing |
| **3: Search engine** | 5 | LLM-only prover with trajectory collection | Priority queue, Parquet writer, search loop | Lean integration bugs, crash recovery |
| **4: EBM in burn-rs** | 10 | Value-guided search improving solve rate | All model code, training loop, losses, metrics | Tensor shape mismatches, training instability |
| **5: Expert iteration** | 16 | 5 iterations, benchmark results | Iteration scripts, evaluation harness | Mostly waiting for GPUs |
| **6: burn-rs PRs** | 10 | Upstream contributions | PR code, tests, docs | Review cycles with maintainers |
| **Total** | **~53 days** | | | |

That's roughly **2.5 months full-time** or **3.5-4 months at 70%**.

The first month (Phases 0-3) is almost entirely on your MacBook — no cloud spend until the search loop works.

---

## 12. Cost Analysis

### Cloud GPU Pricing (Spot Instances)

| Provider | GPU | Spot Price |
|----------|-----|-----------|
| Lambda Labs | A100 80GB | $1.29/hr |
| RunPod | A100 80GB | ~$1.64/hr |
| Vast.ai | A100 80GB | ~$1.10/hr |

Estimated at **$1.50/GPU-hour** (spot A100).

### Revised Compute Estimates (Shared 7B Backbone)

| Phase | GPUs | Duration | GPU-hours | Cost |
|-------|------|----------|-----------|------|
| Data prep (Python, one-time) | 0 | 4h | 0 | $0 |
| LLM fine-tuning (per iter) | 8× A100 | 5h | 40 | $60 |
| **EBM training (burn-rs, per iter)** | **1× A100** | **6h** | **6** | **$9** |
| Search / trajectories (per iter) | 8× A100 | 15-18h | 140 | $210 |
| Noise injection (iter 0 only) | 4× A100 | 4h | 16 | $24 |
| Evaluation (per iter) | 1× A100 | 2h | 2 | $3 |
| **Per iteration total** | | | **~210** | **~$315** |
| **5 iterations** | | | **~1,080** | **~$1,620** |
| **Contingency (15%)** | | | **~160** | **~$243** |
| **Grand total compute** | | | **~1,240** | **~$1,863** |

Non-GPU costs: ~$150 (Cloud CPU for Lean workers, storage, bandwidth)

**Total project cost: ~$2,000**

EBM training in burn-rs should be comparable to PyTorch speed when using the `burn-tch` backend (same CUDA kernels). The overhead is in the custom training loop, not in tensor ops.

### Cost Optimization Strategies

- **Use spot instances aggressively.** Search runs checkpoint every N theorems — restart on preemption.
- **Start small.** Iteration 0 on 5K theorems, not 75K. Scale up if architecture works.
- **Use A10G for search.** 7B fits in 24GB with int8 quantization. A10G spot is ~$0.50/hr.

| Scenario | Estimated Cost |
|----------|---------------|
| Full plan (5 iter, 75K theorems) | ~$2,000 |
| Lean start (3 iter, 5K→25K→75K) | ~$900 |
| Minimal viable (2 iter, 5K theorems, A10G) | ~$400 |

---

## 13. Risk Mitigation

| Risk | Severity | Mitigation |
|------|----------|------------|
| burn-rs ONNX import can't handle complex transformers | N/A | **Eliminated** — shared backbone served by SGLang, not ONNX |
| SGLang can't serve DeepSeek-Prover-V2-7B | Low | SGLang natively supports DeepSeek. Tested and working. |
| `GradientsParams` doesn't expose parameter names for grouped LR | N/A | **Eliminated** — only energy head trains, single optimizer |
| SpectralNorm's u/v vectors need to persist but aren't `Param` | Medium | **Option C** (random reinit) for our code. **Param + detach** for upstream PR. Propose `Buffer` concept. |
| burn-rs training is slower than PyTorch | Low | Unlikely with `burn-tch` backend (same CUDA kernels). If it happens, profile with `nsys` and optimize the data pipeline (often the bottleneck). |
| Contrastive batch sampling is slow in Rust | Low | Pre-build the contrastive index. Sampling from `Vec<usize>` with `rand::seq::SliceRandom` is extremely fast. |
| burn-rs API changes between versions | Low | Pin to a specific burn version in `Cargo.toml`. Submit PRs against the same version. |
| Lean 4 processes OOM during long search | **Mitigated** | Worker recycling (1000 req / 30 min TTL). Implemented and tested in Phase 1. |
| 7B encoder too slow for EBM scoring | Medium | Start with shared backbone. Fall back to dedicated 1.3B if budget > 1000 nodes. Long-term: distill to 125M. |
| EBM doesn't improve search | Medium | LLM-only search is the baseline. If EBM hurts, fall back. Check energy gap, rank accuracy, negative difficulty. |
| Pantograph protocol undocumented | **Mitigated** | Pantograph bundled as submodule (pinned d047b1d). Protocol implemented and tested (10 integration tests passing, including concurrent multi-step state isolation). |
| Spot instance preemption | Medium | Checkpoint aggressively: every 5000 steps EBM, every epoch LLM, every N theorems search. |

### Decision Tree: What To Do When Things Break

```
Search loop doesn't hit >10 iter/sec?
├── Bottleneck is LLM inference → SGLang quantization (AWQ/GPTQ), or increase concurrency
├── Bottleneck is Lean REPL → increase worker count, reduce timeout
└── Bottleneck is Lean startup → pre-warm workers, reuse environments

EBM doesn't improve solve rate?
├── Check energy gap is positive (negatives > positives)
├── Check rank accuracy > 60% on validation set
├── Try more hard negatives (increase hard_ratio to 0.7)
└── If still flat → representations may be too compressed. Try unfreezing
    last 2 encoder layers (requires per-param-group LR — revisit PR #3)

7B encoder too slow for EBM scoring?
├── Budget ≤ 600 nodes → acceptable, keep shared backbone
├── Budget > 1000 → switch to dedicated 1.3B (configs/models.toml)
└── Long-term → distill 7B to 125M encoder in burn-rs
```

---

## 14. Extensions

### Short-term

- **Online EBM fine-tuning during search:** Since training and inference are both in burn-rs, do lightweight gradient updates on the EBM *within* a search iteration (a few steps on newly-discovered dead-end states). Uniquely enabled by having training in Rust.
- **MCTS with learned value:** Use EBM scores as value estimates in PUCT formula.
- **Int8 quantization for EBM inference:** `burn-tch` supports LibTorch's dynamic int8 quantization. ~2× speedup, <1% quality loss.

### Medium-term

- **Native burn-rs transformer encoder:** Implement the full encoder in burn-rs instead of relying on SGLang. Removes HTTP round-trip latency and gives full control over fine-tuning the encoder during EBM training. Major upstream contribution.
- **Distributed EBM training:** burn-rs doesn't have native DDP yet. Implementing data-parallel training would be a flagship PR.
- **Custom Lean tokenizer + small encoder:** Train BPE tokenizer on Mathlib proof states (8192 vocab, preserving `∀`, `⊢`, `→` etc.), then train a 125M encoder from scratch in burn-rs. Better compression, fully Rust.

### Long-term

- **Train everything in Rust:** LLM fine-tuning in burn-rs or candle (both support training), EBM in burn-rs, search in tokio. Zero Python dependency. The dream but premature — Python's HuggingFace ecosystem for LLM LoRA is still too convenient to replicate.

---

## 15. First Week Checklist

- [x] Create Cargo workspace, all crate stubs, `cargo check` passes
- [x] Install Lean 4 + elan
- [x] Bundle Pantograph as git submodule (`vendor/Pantograph/`, pinned d047b1d)
- [x] Build Pantograph (`./scripts/setup_pantograph.sh`)
- [x] Write `LeanWorker::new()` and `send_command()` with `\n` termination
- [x] Test: apply tactics to simple theorems, read back results
- [x] Write unit + integration tests for the REPL protocol (20 unit, 10 integration, 2 doc)
- [x] Auto-discovery: `LeanPoolConfig::with_bundled_pantograph()` — no env var needed
- [x] ProofHandle pattern: state ID routing correctness with concurrent multi-step proofs
- [x] Download DeepSeek-Prover-V2-7B weights
- [x] Verify SGLang serves the model correctly

Phases 0-1 are complete. The search loop milestone (Phase 2 end) is the true gate — everything else is incremental.
