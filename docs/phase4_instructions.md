# Phase 4: EBM in burn-rs — Claude Code Instructions

Step-by-step prompts for `claude` CLI. Phases 1-3 are complete.

Phase 4 builds the `ebm` crate: SpectralNorm layers, the energy head MLP, contrastive training with InfoNCE loss, the training loop, and integration with the search engine so the EBM actually guides proof search.

This is the largest phase (10 days in the plan). It's the core burn-rs learning goal of the project.

## Prerequisites

- Phase 3 complete: search pipeline produces Parquet trajectory files
- You have at least one trajectory file from a real search run (even a small one from the 10 test theorems). If not, generate one first:
  ```bash
  MODEL_PATH=./models/deepseek-prover-v2-7b \
  LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
    cargo run -p prover-core -- search \
      --llm-path ./models/deepseek-prover-v2-7b \
      --theorems data/test_theorems.json \
      --output trajectories/test_iter0.parquet \
      --no-ebm
  ```
- Verify the trajectory file has both positive and negative labels. If no theorems were proved, all labels will be negative — you'll need at least some positives for contrastive training. Solve this by ensuring at least 2-3 easy theorems are proved, or by manually creating a small synthetic trajectory file for testing.

## Background: The Training Data Flow

```
Phase 3 produced:
  trajectories/iter_0.parquet
    → Contains TrajectoryRecords with: state_pp, label, depth, llm_log_prob

Phase 4 training pipeline:
  1. Read Parquet → Vec<ProofStateRecord>
  2. Build ContrastiveIndex (pos/neg by theorem)
  3. For each training step:
     a. Sample batch: 1 positive + K negatives per example
     b. Encode all states: TacticGenerator.encode_only(state) → Vec<f32>  [candle, frozen]
     c. Convert Vec<f32> → burn Tensor
     d. Forward through EnergyHead → scalar energies  [burn-rs, autodiff]
     e. Compute InfoNCE + depth regression loss
     f. Backward + optimizer step (only energy head params)
  4. Save checkpoint

At inference (during search):
  - For each child state: encode_only() → energy_head.forward() → score
  - Combined with LLM log-prob for priority queue ordering
```

The encoder (7B candle model) is FROZEN. Gradients only flow through the energy head (~5M params). This is why a single AdamW optimizer with one learning rate is sufficient.

---

## Prompt 4.1 — SpectralNormLinear module

```
Implement crates/ebm/src/model/spectral_norm.rs.

Read docs/spindle_final_plan.md Section 4.1 for the full reference implementation.

We're using Option C: random reinit of the u/v vectors at each forward pass with 5 power iterations. This avoids the burn-rs buffer persistence problem entirely.

SpectralNormLinearConfig:
  - d_input: usize
  - d_output: usize
  - n_power_iterations: usize (default 5)
  - eps: f64 (default 1e-12)
  - bias: bool (default true)

SpectralNormLinear<B: Backend>:
  - weight: Param<Tensor<B, 2>>  — shape (d_output, d_input), Kaiming init
  - bias: Option<Param<Tensor<B, 1>>>  — shape (d_output,), zero init
  - n_power_iterations: usize
  - eps: f64

  Must derive Module and Debug.

forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2>:
  1. Get weight value: self.weight.val()  — shape (d_out, d_in)
  2. Random init u (d_out,) and v (d_in,) from Normal(0, 1) on same device as input
  3. Power iteration loop (n_power_iterations times):
     v = W^T @ u / ‖W^T @ u‖
     u = W @ v / ‖W @ v‖
  4. Spectral norm: σ = u^T @ W @ v (scalar)
  5. Normalized weight: W_normed = W / σ
  6. Output = input @ W_normed^T + bias

Careful with tensor shapes in matmul — burn requires explicit unsqueeze/squeeze for vector-matrix multiplication. The plan code uses unsqueeze_dim(1) to make column vectors.

IMPORTANT: The power iteration must NOT be part of the autodiff graph for u and v (they're not parameters). But the weight normalization W/σ MUST be differentiable — gradients need to flow through the division into the weight. Since u and v are created fresh (not Param), burn's autodiff won't track them — only the weight (which is a Param) gets gradients. This is exactly what we want.

Add thorough unit tests using burn::backend::NdArray as TestBackend:

1. test_output_shape — input (4, 64) → output (4, 32) for a 64→32 layer
2. test_spectral_norm_bounds — after forward, compute the actual spectral norm
   of the effective weight matrix (output = input @ W_normed^T, so extract W_normed).
   Verify it's approximately 1.0 (within 0.15 tolerance due to random init convergence).
   Do this by: creating a (d_in, d_in) identity-like probe and measuring output norms.
3. test_no_bias — create with bias=false, verify output matches manual matmul
4. test_deterministic_weight — two forward passes with same input should give
   approximately the same output (despite random u/v init) because 5 iterations
   should converge the spectral norm estimate. Tolerance: ~1% relative error.
5. test_gradient_flows — use AutodiffBackend, compute output.sum().backward(),
   verify weight.grad() is not all zeros
```

### Prompt 4.2 — EnergyHead model

```
Implement crates/ebm/src/model/energy_head.rs.

Read docs/spindle_final_plan.md Section 4.2.

EnergyHeadConfig:
  - d_encoder: usize  — input dim from encoder (4096 for DeepSeek-7B)
  - d_hidden1: usize (default 512)
  - d_hidden2: usize (default 256)
  - dropout: f64 (default 0.1)

EnergyHead<B: Backend>:
  - sn_linear1: SpectralNormLinear<B>  — d_encoder → d_hidden1
  - sn_linear2: SpectralNormLinear<B>  — d_hidden1 → d_hidden2
  - sn_linear3: SpectralNormLinear<B>  — d_hidden2 → 1, NO bias
  - dropout1: burn::nn::Dropout
  - dropout2: burn::nn::Dropout
  - log_temperature: Param<Tensor<B, 1>>  — learnable temperature, init to 0 (so exp(0)=1)

  Must derive Module and Debug.

forward(&self, h: Tensor<B, 2>) -> Tensor<B, 1>:
  - h shape: (batch, d_encoder)
  - x = sn_linear1(h) → SiLU → dropout1
  - x = sn_linear2(x) → SiLU → dropout2
  - raw_energy = sn_linear3(x).squeeze(1)  → (batch,)
  - temperature = log_temperature.val().exp()  → scalar
  - return raw_energy / temperature

Use burn::tensor::activation::silu for activation.

Add unit tests:

1. test_forward_shape — input (8, 4096) → output (8,) scalar energies
2. test_small_model — create with d_encoder=32, d_hidden1=16, d_hidden2=8
   Forward a batch of 4 → verify shape (4,)
3. test_different_inputs_different_energies — two different input tensors
   should produce different energy values
4. test_gradient_flows_through_all_layers — use Autodiff backend,
   forward → sum → backward, check that sn_linear1.weight.grad(),
   sn_linear2.weight.grad(), sn_linear3.weight.grad(), and
   log_temperature.grad() are all non-zero
5. test_temperature_scaling — set log_temperature to ln(2), verify
   outputs are roughly halved compared to log_temperature=0
6. test_parameter_count — verify total param count is approximately:
   4096*512 + 512 + 512*256 + 256 + 256*1 + 1 ≈ 2.2M
   (Our model is ~5M total with biases)
```

### Prompt 4.3 — Loss functions: InfoNCE + depth regression

```
Implement crates/ebm/src/training/loss.rs.

Read docs/spindle_final_plan.md Section 6.2 for the reference implementations.

Two loss functions:

1. info_nce_loss<B: Backend>(pos_energy: Tensor<B, 1>, neg_energies: Tensor<B, 2>) -> Tensor<B, 1>

   pos_energy: (batch,) — energies of positive (provable) states
   neg_energies: (batch, K) — energies of K negative states per example

   Lower energy = more provable. So we want positive energy < negative energy.

   Implementation:
   - Logits = -energy (negate so higher = "more likely correct class")
   - pos_logits: (batch, 1) = pos_energy.neg().unsqueeze_dim(1)
   - neg_logits: (batch, K) = neg_energies.neg()
   - logits: (batch, K+1) = cat([pos_logits, neg_logits], dim=1)
   - labels: (batch,) = zeros (the positive is at column 0)
   - return cross_entropy(logits, labels)

   Check burn's CrossEntropyLoss API carefully — it may expect different input shapes
   or conventions than PyTorch. The logits should be (batch, num_classes) and labels
   should be (batch,) with integer class indices.

2. depth_regression_loss<B: Backend>(energy: Tensor<B, 1>, remaining_depth: Tensor<B, 1>) -> Tensor<B, 1>

   energy: (batch,) — energies of positive states
   remaining_depth: (batch,) — steps remaining to QED, -1 if unknown

   States closer to QED (small remaining_depth) should have lower energy.

   Implementation:
   - Mask out entries where remaining_depth == -1
   - Normalize remaining_depth to [0, 1] within the batch
   - MSE between energy and normalized depth (both scalars per sample)
   - Return masked mean

   Handle edge cases: all depths unknown (return 0), single valid depth.

Unit tests:

1. test_info_nce_perfect_separation
   - pos_energy = [-10, -10], neg_energies = [[10, 10], [10, 10]]
   - Loss should be ≈ 0 (positive is clearly lowest energy)

2. test_info_nce_no_separation
   - pos_energy = [0, 0], neg_energies = [[0, 0], [0, 0]]
   - Loss should be ≈ ln(K+1) = ln(3) ≈ 1.099

3. test_info_nce_wrong_separation
   - pos_energy = [10, 10], neg_energies = [[-10, -10], [-10, -10]]
   - Loss should be high (positive has higher energy than negatives — bad!)

4. test_info_nce_gradient_direction
   - Use Autodiff backend
   - Create pos_energy requiring grad, neg_energy requiring grad
   - After backward: positive energy gradient should push it DOWN,
     negative energy gradient should push it UP

5. test_depth_loss_perfect
   - energy = [0.0, 0.5, 1.0], remaining_depth = [0, 5, 10]
   - After normalization: targets = [0, 0.5, 1.0]
   - Energy already matches targets → loss ≈ 0

6. test_depth_loss_with_unknowns
   - energy = [0.5, 0.5, 0.5], remaining_depth = [5, -1, 10]
   - Only indices 0 and 2 should contribute. Index 1 (depth=-1) masked out.

7. test_depth_loss_all_unknown
   - remaining_depth = [-1, -1, -1]
   - Loss should be exactly 0 (no valid entries)
```

### Prompt 4.4 — Training data: contrastive sampling

```
Implement crates/ebm/src/training/data.rs — the contrastive data pipeline.

This reads trajectory Parquet files (from Phase 3) and constructs contrastive training batches.

Read docs/spindle_final_plan.md Section 5.1 and 5.2 for reference code.

IMPORTANT: In the shared backbone architecture, the encoder is candle (not burn). So the
training data pipeline does NOT tokenize — it stores raw proof state strings. Tokenization
and encoding happen in the training loop when we call encoder.encode_only(). This is different
from a typical NLP pipeline where the batcher tokenizes.

Types needed:

1. ProofStateRecord (Clone, Debug):
   - theorem_name: String
   - state_pp: String  — the raw proof state text
   - label: String     — "positive" or "negative" (matching TrajectoryLabel)
   - depth_from_root: u32
   - remaining_depth: i32  — -1 if unknown
   - llm_log_prob: f64

2. ContrastiveIndex:
   - pos_by_theorem: HashMap<String, Vec<usize>>
   - neg_by_theorem: HashMap<String, Vec<usize>>
   - all_negatives: Vec<usize>
   - eligible_theorems: Vec<String> — theorems with BOTH positives and negatives

   fn build(records: &[ProofStateRecord]) -> Self

3. ContrastiveSample (what the training loop consumes):
   - positive: ProofStateRecord
   - negatives: Vec<ProofStateRecord>  — exactly K negatives
   - remaining_depth: i32  — from the positive

4. ContrastiveSampler:
   - records: Vec<ProofStateRecord>
   - index: ContrastiveIndex
   - k_negatives: usize
   - hard_ratio: f64 (default 0.5)
   - medium_ratio: f64 (default 0.3)
   - easy_ratio: f64 (computed: 1 - hard - medium = 0.2)

   fn from_trajectory_reader(path: &Path, k_negatives: usize) -> Result<Self>
     Read Parquet, build index.

   fn sample(&self, rng: &mut impl Rng) -> ContrastiveSample
     Pick a random eligible theorem.
     Pick a random positive from that theorem.
     Fill negatives with the hard/medium/easy mix:
     - Hard (50%): dead-end states from SAME theorem
     - Medium (30%): off-path siblings from same theorem (other positives used as negatives)
     - Easy (20%): random states from OTHER theorems
     Pad with random negatives if not enough in a category.

   fn sample_batch(&self, batch_size: usize, rng: &mut impl Rng) -> Vec<ContrastiveSample>
     Sample batch_size independent examples.

5. fn load_records_from_parquet(path: &Path) -> Result<Vec<ProofStateRecord>>
   Read the trajectory Parquet file and convert to ProofStateRecords.
   Map TrajectoryRecord fields to ProofStateRecord fields.
   Filter: only include records where label is "positive" or "negative" (skip "unknown").

Unit tests:

1. test_build_index — create 10 records (5 pos, 5 neg across 2 theorems),
   verify index has correct counts
2. test_eligible_theorems — a theorem with only positives should NOT be eligible
3. test_sample_returns_correct_k — sample with k=4, verify exactly 4 negatives
4. test_hard_negatives_from_same_theorem — with hard_ratio=1.0, all negatives
   should have the same theorem_name as the positive
5. test_easy_negatives_from_other_theorems — with easy_ratio=1.0 (hard=0, medium=0),
   negatives should be from different theorems
6. test_sample_batch_size — sample_batch(8) returns exactly 8 samples
```

### Prompt 4.5 — Training metrics and health checks

```
Implement crates/ebm/src/training/metrics.rs.

Read docs/spindle_final_plan.md Section 6.3.

EBMMetrics struct (Debug, Clone):
  - loss: f64                — total loss
  - contrastive_loss: f64    — InfoNCE component
  - depth_loss: f64          — depth regression component
  - energy_gap: f64          — mean(neg_energy) - mean(pos_energy). Should be POSITIVE and growing.
  - pos_energy_mean: f64     — mean energy of positive states (should decrease)
  - neg_energy_mean: f64     — mean energy of negative states (should increase or stay high)
  - rank_accuracy: f64       — fraction of examples where positive has lowest energy in its group
  - energy_std: f64          — std deviation of all energies (watch for collapse toward 0)

Methods:

fn compute<B: Backend>(
    pos_energy: &Tensor<B, 1>,
    neg_energies: &Tensor<B, 2>,
    contrastive_loss: f64,
    depth_loss: f64,
    total_loss: f64,
) -> Self
  - Compute all metrics from the tensors
  - For rank_accuracy: for each row, check if pos_energy[i] < min(neg_energies[i, :])

fn health_check(&self) -> Vec<String>
  Return warnings for:
  - energy_std < 0.1 → "possible mode collapse"
  - energy_gap < 0.0 → "model scoring negatives lower than positives"
  - rank_accuracy > 0.95 && contrastive_loss > 0.5 → "negatives may be too easy"
  - pos_energy_mean > 0 && neg_energy_mean < 0 → "energy polarity inverted"
  - loss is NaN or Inf → "training diverged"

fn display(&self) -> String
  Pretty-print all metrics in a single line for logging:
  "loss=0.4321 gap=1.23 rank=0.78 pos_e=-0.45 neg_e=0.78 std=0.82"

Also implement a MetricsHistory struct:
  - history: Vec<(usize, EBMMetrics)>  — (step, metrics) pairs

  fn push(&mut self, step: usize, metrics: EBMMetrics)
  fn last(&self) -> Option<&EBMMetrics>
  fn is_improving(&self, window: usize) -> bool
    Check if energy_gap is trending upward over the last `window` entries.
    Useful for early stopping decisions.

Unit tests:
1. test_compute_perfect — all positives lower → rank_accuracy = 1.0, gap > 0
2. test_compute_random — equal energies → rank_accuracy ≈ 0 (positive rarely wins by chance)
3. test_health_check_mode_collapse — energy_std = 0.01 → triggers warning
4. test_health_check_healthy — normal metrics → no warnings
```

### Prompt 4.6 — Tensor bridge: Vec<f32> ↔ burn Tensor

```
Implement crates/ebm/src/model/bridge.rs — utility functions to convert between
the candle encoder's output (Vec<f32>) and burn tensors.

Read docs/spindle_final_plan.md Section 4.5.

Functions:

1. fn embeddings_to_burn_tensor<B: Backend>(
       embeddings: &[Vec<f32>],
       device: &B::Device,
   ) -> Tensor<B, 2>
   - Input: batch of embeddings, each Vec<f32> of length d_model
   - Output: burn Tensor shape (batch_size, d_model)
   - Flatten all vecs into a single Vec<f32>, then Tensor::from_data

2. fn embedding_to_burn_tensor<B: Backend>(
       embedding: &[f32],
       device: &B::Device,
   ) -> Tensor<B, 2>
   - Single embedding → (1, d_model) tensor

3. fn burn_tensor_to_vec<B: Backend>(
       tensor: Tensor<B, 1>,
   ) -> Vec<f64>
   - Extract scalar values from a 1D burn tensor
   - Used for reading energy scores back into Rust

4. fn burn_tensor_to_f64<B: Backend>(
       tensor: Tensor<B, 1>,
   ) -> f64
   - Extract a single scalar (for loss values)

Unit tests:
- Round-trip: Vec<f32> → burn tensor → back to Vec<f32>
- Shape validation: batch of 4 embeddings of dim 64 → (4, 64) tensor
- Empty batch handling (0 embeddings) — should this error or return empty tensor?
```

### Prompt 4.7 — Training loop

```
Implement crates/ebm/src/training/trainer.rs — the core EBM training loop.

Read docs/spindle_final_plan.md Section 6.1 carefully. This is the full training loop.

EBMTrainingConfig (Config, Debug, Clone, Deserialize):
  - lr: f64 (default 1e-4)
  - weight_decay: f64 (default 0.01)
  - max_grad_norm: f64 (default 1.0)
  - total_steps: usize (default 50_000)
  - warmup_steps: usize (default 1_000)
  - depth_loss_weight: f64 (default 0.3)
  - log_interval: usize (default 50)
  - checkpoint_interval: usize (default 5_000)
  - k_negatives: usize (default 4)
  - batch_size: usize (default 32)
  - checkpoint_dir: PathBuf (default "checkpoints/ebm")

EBMTrainer — NOT generic over backend. Concrete over the backend we use.
But the training method IS generic so we can test with NdArray.

pub fn train<B: AutodiffBackend>(
    config: &EBMTrainingConfig,
    mut model: EnergyHead<B>,
    encoder: &policy::TacticGenerator,  // candle, frozen
    sampler: &ContrastiveSampler,
    device: &B::Device,
) -> Result<EnergyHead<B>>

Algorithm:
  1. Create AdamW optimizer:
     AdamWConfig::new()
       .with_weight_decay(config.weight_decay)
       .with_grad_clipping(Some(GradientClippingConfig::Norm(config.max_grad_norm)))
       .init()

  2. Create MetricsHistory

  3. For step in 0..config.total_steps:
     a. Compute LR: linear warmup then cosine annealing
        if step < warmup_steps: lr = base_lr * (step+1) / warmup_steps
        else: progress = (step - warmup) / (total - warmup)
              lr = base_lr * 0.5 * (1 + cos(π * progress))

     b. Sample contrastive batch:
        let samples = sampler.sample_batch(config.batch_size, &mut rng);

     c. Encode all states with frozen encoder (this is the slow part):
        For each sample:
          pos_embedding = encoder.encode_only(&sample.positive.state_pp)?
          neg_embeddings = sample.negatives.iter()
            .map(|n| encoder.encode_only(&n.state_pp))
            .collect()?

     d. Convert to burn tensors:
        pos_tensor: (batch, d_model)
        neg_tensor: (batch * K, d_model)

     e. Forward through energy head:
        pos_energy = model.forward(pos_tensor)  → (batch,)
        neg_energy = model.forward(neg_tensor)   → (batch * K,)
        neg_energy = neg_energy.reshape([batch, K])

     f. Compute losses:
        loss_contrastive = info_nce_loss(pos_energy, neg_energy)
        loss_depth = depth_regression_loss(pos_energy, remaining_depths)
        total_loss = loss_contrastive + config.depth_loss_weight * loss_depth

     g. Backward + step:
        let grads = total_loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(lr, model, grads_params);

     h. Compute and log metrics (every log_interval steps):
        let metrics = EBMMetrics::compute(...)
        let warnings = metrics.health_check()
        if !warnings.is_empty() { tracing::warn!(...) }
        tracing::info!(step, "{}", metrics.display())

     i. Save checkpoint (every checkpoint_interval steps):
        model.save_file(path, &DefaultFileRecorder::new())?

  4. Save final model
  5. Return model

Handle the burn-rs API carefully:
- AdamW::step() takes (lr, model, grads) and returns the updated model.
  The model is CONSUMED and returned — burn uses an ownership-based update pattern.
- GradientsParams::from_grads() needs both the gradient set and a reference to the model
  to know which parameters to update.
- loss.backward() returns a gradient set. This consumes the loss tensor.
- Make sure to .clone() any tensors you need for both loss computation and metrics.

Also implement:

fn resume_from_checkpoint<B: Backend>(
    path: &Path,
    config: &EnergyHeadConfig,
    device: &B::Device,
) -> Result<EnergyHead<B>>
  Load a previously saved energy head checkpoint.

No unit tests for the full training loop here (too expensive). We'll test it end-to-end
in prompt 4.10. But add a test for:
- test_lr_schedule: verify warmup phase, peak, and cosine decay values at specific steps
```

### Prompt 4.8 — EBM inference for search integration

```
Implement crates/ebm/src/inference.rs — batch scoring for use during search.

This module provides the ValueScorer trait implementation so the search engine can use
the EBM to score proof states.

Read docs/spindle_final_plan.md Section 4.6 for the EBMValueFunction.

EBMScorer struct:
  - energy_head: EnergyHead<B>  — loaded from checkpoint, inference mode (no dropout)
  - encoder: Arc<policy::TacticGenerator>  — shared reference to the candle model
  - device: B::Device

Methods:

1. fn load(
       ebm_checkpoint: &Path,
       energy_head_config: &EnergyHeadConfig,
       encoder: Arc<policy::TacticGenerator>,
       device: &B::Device,
   ) -> Result<Self>
   - Load energy head from checkpoint file
   - Set to eval mode (disable dropout): model.set_require_grad(false)
     or use burn's inference mode if available

2. fn score_state(&self, proof_state: &str) -> f64
   - Encode: encoder.encode_only(proof_state) → Vec<f32>
   - Convert: embedding_to_burn_tensor → (1, d_model)
   - Forward: energy_head.forward() → (1,) energy
   - Return: -energy (higher = more provable)

3. fn score_states(&self, proof_states: &[&str]) -> Vec<f64>
   - Batch version: encode all, stack, forward, negate
   - More efficient than calling score_state N times because
     the energy head forward is batched

Implement the search crate's ValueScorer trait for EBMScorer:

  impl ValueScorer for EBMScorer<B> {
      fn score(&self, proof_state: &str) -> f64 {
          self.score_state(proof_state)
      }
  }

The Backend type parameter is tricky here. For inference, we want a NON-autodiff backend
(no gradient tracking overhead). If the search crate's ValueScorer trait is not generic
over Backend, we may need to use a concrete type or a trait object.

Options:
A. Make ValueScorer generic: trait ValueScorer<B: Backend>
B. Use a concrete backend: EBMScorer<burn_tch::TchBackend<f32>>
C. Use dynamic dispatch: Box<dyn ValueScorer>
D. Erase the backend with a wrapper that stores Box<dyn Fn(&str) -> f64>

Option D is simplest for the search integration:

  pub struct EBMValueFn {
      score_fn: Box<dyn Fn(&str) -> f64 + Send + Sync>,
  }

  impl EBMValueFn {
      pub fn new<B: Backend>(scorer: EBMScorer<B>) -> Self
      where B: 'static {
          Self {
              score_fn: Box::new(move |state| scorer.score_state(state))
          }
      }
  }

  impl ValueScorer for EBMValueFn {
      fn score(&self, proof_state: &str) -> f64 {
          (self.score_fn)(proof_state)
      }
  }

Add unit tests:
1. test_score_state_returns_finite — verify output is not NaN/Inf
2. test_different_states_different_scores — two different inputs → different scores
3. test_score_states_matches_individual — batch scoring matches sequential scoring
```

### Prompt 4.9 — EBM crate: lib.rs, mod.rs files, and encoder backend

```
Wire up the entire ebm crate structure:

crates/ebm/src/lib.rs:
  pub mod model;
  pub mod training;
  pub mod inference;

  pub use model::energy_head::{EnergyHead, EnergyHeadConfig};
  pub use model::spectral_norm::{SpectralNormLinear, SpectralNormLinearConfig};
  pub use inference::{EBMScorer, EBMValueFn};
  pub use training::trainer::EBMTrainingConfig;
  pub use training::data::ContrastiveSampler;

crates/ebm/src/model/mod.rs:
  pub mod spectral_norm;
  pub mod energy_head;
  pub mod encoder;
  pub mod bridge;

crates/ebm/src/training/mod.rs:
  pub mod loss;
  pub mod data;
  pub mod trainer;
  pub mod metrics;

Implement crates/ebm/src/model/encoder.rs:

  EncoderBackend enum (for future flexibility, but currently only SharedPolicy is used):

  pub enum EncoderBackend {
      SharedPolicy {
          generator: Arc<policy::TacticGenerator>,
      },
  }

  impl EncoderBackend {
      pub fn encode(&self, proof_state: &str) -> Result<Vec<f32>> {
          match self {
              Self::SharedPolicy { generator } => {
                  let embedding = generator.encode_only(proof_state)?;
                  Ok(embedding.data)
              }
          }
      }

      pub fn encode_batch(&self, proof_states: &[&str]) -> Result<Vec<Vec<f32>>> {
          proof_states.iter()
              .map(|s| self.encode(s))
              .collect()
      }

      pub fn hidden_dim(&self) -> usize {
          match self {
              Self::SharedPolicy { generator } => generator.hidden_size(),
          }
      }
  }

Update the ebm Cargo.toml to depend on:
  - burn (workspace) with features: train, autodiff, metrics, dataset
  - burn-ndarray (workspace) — for testing
  - policy (path = "../policy") — for encoder integration
  - trajectory (path = "../trajectory") — for reading training data
  - tokenizers, serde, serde_json, anyhow, tracing, rand (workspace)

Run: cargo check -p ebm
Run: cargo test -p ebm
Fix any compilation errors. All unit tests from prompts 4.1-4.5 should pass.
```

### Prompt 4.10 — End-to-end training test

```
Create crates/ebm/tests/integration.rs — test the full training pipeline.

This is the most important test for Phase 4. It validates that gradients flow correctly,
the loss decreases, and the energy gap increases.

test_training_loop_synthetic (NO external deps, fast):

1. Create synthetic training data (no Parquet needed):
   - 20 "positive" ProofStateRecords: short state strings like "⊢ True", "n : Nat ⊢ n = n"
   - 40 "negative" ProofStateRecords: longer/different state strings
   - Assign to 4 theorems (5 pos + 10 neg each)
   - Build a ContrastiveSampler from these

2. Create a mock encoder that returns fixed embeddings:
   Since we can't load the 7B model in a fast test, create a MockEncoder
   that returns deterministic embeddings:
   - Positive states → embeddings clustered around [1, 0, 0, ..., 0]
   - Negative states → embeddings clustered around [0, 1, 0, ..., 0]
   Use d_model = 64 (not 4096) for speed.

3. Create EnergyHead with d_encoder=64, d_hidden1=32, d_hidden2=16

4. Run training for 200 steps with:
   - lr = 1e-3 (higher than production for fast convergence)
   - batch_size = 4
   - k_negatives = 2
   - log_interval = 50

5. Verify:
   - Final loss < initial loss
   - Final energy_gap > 0 (negatives scored higher than positives)
   - Final rank_accuracy > 0.6 (better than random)
   - No health check warnings at the end
   - No NaN/Inf in any metrics

This test should run in < 5 seconds on CPU with burn-ndarray.

test_checkpoint_save_load (fast):

1. Create a small EnergyHead (d_encoder=64)
2. Save to a temp file
3. Load from the file
4. Forward the same input through both → verify identical outputs

test_training_with_real_data (requires model + trajectories, #[ignore]):

1. Load TacticGenerator from MODEL_PATH
2. Load trajectory records from a real Parquet file
3. Build ContrastiveSampler
4. Create EnergyHead with d_encoder = generator.hidden_size()
5. Train for 100 steps
6. Verify loss decreases
7. Print final metrics

Run:
  cargo test -p ebm                    # fast synthetic tests
  MODEL_PATH=... cargo test -p ebm -- --ignored   # real model test
```

### Prompt 4.11 — Wire EBM into search + add train-ebm CLI command

```
Now connect everything: make the search engine use the EBM, and add the train-ebm
CLI subcommand.

1. Update crates/prover-core/src/main.rs — add TrainEbm subcommand:

   TrainEbm {
       #[arg(long, value_delimiter = ',')]
       trajectories: Vec<PathBuf>,

       #[arg(long)]
       output: PathBuf,

       #[arg(long)]
       resume_from: Option<PathBuf>,

       #[arg(long, default_value = "50000")]
       steps: usize,

       #[arg(long)]
       llm_path: PathBuf,  // needed for the encoder

       // Override training hyperparameters
       #[arg(long, default_value = "1e-4")]
       lr: f64,

       #[arg(long, default_value = "32")]
       batch_size: usize,

       #[arg(long, default_value = "4")]
       k_negatives: usize,
   }

2. Implement the train-ebm handler in pipeline.rs:
   - Load TacticGenerator (as encoder)
   - Load trajectory records from all input files
   - Print dataset summary (num records, pos/neg split, num theorems)
   - Build ContrastiveSampler
   - Create or resume EnergyHead
   - Run training loop
   - Save final model
   - Print final metrics

3. Update the Search command handler to accept --ebm-path:
   - If ebm_path is Some and !no_ebm:
     - Load EnergyHead checkpoint
     - Create EBMScorer with shared TacticGenerator reference
     - Wrap in EBMValueFn
     - Pass as Some(&value_fn) to search_engine.search()
   - If no_ebm or ebm_path is None:
     - Pass None to search_engine.search()

4. Update prover-core Cargo.toml to depend on the ebm crate.

5. Test the full pipeline:
   a. Run search without EBM → produces trajectories
   b. Run train-ebm on those trajectories → produces EBM checkpoint
   c. Run search WITH EBM → verify it uses EBM scores

   The test can be a shell script or an integration test. Even if the EBM doesn't
   improve results (it's barely trained on toy data), verify the pipeline runs.

6. cargo check --workspace
   cargo clippy --workspace
```

### Prompt 4.12 — Update CLAUDE.md

```
Update CLAUDE.md:

1. Mark Phases 0-3 as complete: [x]
2. Mark Phase 4 as complete: [x]
3. Set "Current Phase" to Phase 5

4. Add "Phase 4 Results" section:
   - Does the EnergyHead forward pass work? Shape in, shape out?
   - Does training converge on synthetic data? (loss decrease, gap increase)
   - Does training converge on real trajectory data?
   - What backend was used for testing (NdArray/Tch)?
   - Any burn-rs API surprises or workarounds needed?
   - Checkpoint size on disk

5. Update "Cross-Crate Integration Notes":
   - How to train EBM: cargo run -p prover-core -- train-ebm --trajectories ... --llm-path ... --output ...
   - How to run search with EBM: cargo run -p prover-core -- search --ebm-path ...
   - The embedding flow: proof_state → TacticGenerator.encode_only() → Vec<f32> → burn Tensor → EnergyHead → scalar
   - EBMScorer wraps this for the search engine's ValueScorer trait

6. Note any performance observations:
   - Training throughput: steps/sec on CPU? On GPU?
   - Inference throughput: states scored/sec?
   - What dominates training time: encoding (candle) or forward+backward (burn)?
```

---

## Verification Checklist

```bash
# All crates compile
cargo check --workspace

# EBM unit tests pass (no model needed, fast)
cargo test -p ebm

# No clippy warnings
cargo clippy --workspace

# Full training test with real data
MODEL_PATH=./models/deepseek-prover-v2-7b \
  cargo test -p ebm -- --ignored --nocapture

# Train EBM on real trajectories
MODEL_PATH=./models/deepseek-prover-v2-7b \
  cargo run -p prover-core -- train-ebm \
    --trajectories trajectories/test_iter0.parquet \
    --llm-path ./models/deepseek-prover-v2-7b \
    --output models/ebm/test \
    --steps 200

# Search WITH EBM
MODEL_PATH=./models/deepseek-prover-v2-7b \
LEAN_ENV_PATH=/path/to/mathlib4/.lake/build \
  cargo run -p prover-core -- search \
    --llm-path ./models/deepseek-prover-v2-7b \
    --ebm-path models/ebm/test \
    --theorems data/test_theorems.json \
    --output trajectories/test_with_ebm.parquet \
    --limit 3
```

### Success Criteria

1. **SpectralNorm** constrains spectral norm to ≈1.0 (unit test passes)
2. **EnergyHead** forward pass: (batch, 4096) → (batch,) scalar energies
3. **InfoNCE loss** gradient pushes positive energy down, negative energy up
4. **Training converges** on synthetic data: loss decreases, energy gap increases, rank accuracy > 0.6
5. **Checkpoint save/load** round-trips correctly
6. **Search engine** accepts optional EBM scorer and uses it for node scoring
7. **Full pipeline** works: search → trajectories → train-ebm → search with EBM

---

## Troubleshooting

### burn-rs "type mismatch" or "Backend does not implement AutodiffBackend"
- Training requires `AutodiffBackend`. Use `burn::backend::Autodiff<NdArray>` for CPU testing
  or `burn::backend::Autodiff<TchBackend<f32>>` for GPU.
- Inference does NOT need autodiff. Use the inner backend directly.
- The `EnergyHead<Autodiff<B>>` from training needs to be converted to `EnergyHead<B>`
  for inference. Use `model.valid()` if burn supports it, or save/load the checkpoint.

### "no method named `backward`"
- `.backward()` is only available on tensors from an AutodiffBackend.
- Make sure the training function is generic over `B: AutodiffBackend`, not just `B: Backend`.

### SpectralNorm output varies between calls
- Expected! Option C uses random u/v init each call. With 5 iterations, the variance
  should be small (<1% on output). If it's large, increase n_power_iterations to 10.

### InfoNCE loss is NaN
- Check for numerical overflow in exp(). If energies are very large (>50), the exp will overflow.
- The log_temperature in EnergyHead should prevent this — it scales down raw energy.
- Also check: is pos_energy being negated before exp? The formula uses -energy as logits.

### Training loss doesn't decrease
- Check that gradients are non-zero: after backward(), inspect model parameter grads
- Check learning rate schedule: is warmup working? Is the LR too low?
- Check negative mining: if all negatives are "easy" (from other theorems), the model learns
  nothing useful. Increase hard_ratio.
- Check that the encoder actually produces different embeddings for different states.
  If encode_only() returns the same thing for everything, the EBM can't learn.

### Burn checkpoint format issues
- burn uses its own serialization format (MessagePack or similar).
- save_file / load_file should handle this. Use DefaultFileRecorder.
- If you get version mismatch errors, ensure the same burn version for save and load.

### "expected struct EnergyHead<Autodiff<NdArray>>, found EnergyHead<NdArray>"
- Training produces a model with the Autodiff wrapper. For inference, you need to strip it.
- Save the autodiff model, then load it with the non-autodiff backend.
- Or use `model.valid()` to get the inner model (check burn API).
