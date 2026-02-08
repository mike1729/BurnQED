# Understanding Energy-Based Models for burn-qed

A self-contained guide that takes you from zero EBM knowledge to confident Phase 4 execution. Read sequentially — each section builds on the previous.

Estimated time: 3-4 hours of reading this guide + 2-4 hours with the linked papers.

---

## Part 1: The Core Intuition

### 1.1 What is an Energy-Based Model?

Forget neural networks for a moment. Think about physics.

A ball on a hilly landscape rolls downhill. It settles in valleys (low energy) and avoids hilltops (high energy). If you know the landscape, you can predict where the ball will end up: wherever energy is lowest.

An Energy-Based Model (EBM) does exactly this, but with data instead of balls:

- **Input**: some data point x (in our case, a proof state like `n : Nat ⊢ n + 0 = n`)
- **Output**: a single scalar E(x) — the "energy" of that data point
- **Interpretation**: low energy = "good" data, high energy = "bad" data

That's it. An EBM is just a function that assigns a number to each input. The function is parameterized by a neural network, and we train it so that:

- **Provable proof states** get low energy (they're in the "valleys")
- **Dead-end proof states** get high energy (they're on the "hilltops")

### 1.2 How is this different from a classifier?

A classifier would say "this proof state is 78% likely to be provable." An EBM says "this proof state has energy -2.3."

The difference matters:

**Classifier**: outputs a probability. Must sum to 1 across all classes. This is restrictive — it forces the model to commit to a distribution, even when it's uncertain.

**EBM**: outputs an unconstrained scalar. No normalization required. The model can say "this state has energy -2.3 and that state has energy -1.7" without worrying about whether the numbers "add up." This is more flexible and more stable to train when the space of possible states is enormous (which it is for proof search — there are infinitely many possible proof states).

The formal connection: an EBM *implicitly* defines a probability distribution via the Boltzmann distribution:

```
p(x) = exp(-E(x)) / Z
```

where Z = ∫ exp(-E(x)) dx is the "partition function" (a normalizing constant that makes probabilities sum to 1). But we **never compute Z**. It's intractable. Instead, we only ever compare energies: "is E(x₁) < E(x₂)?" This makes EBMs practical even when the data space is huge.

### 1.3 Why use an EBM for theorem proving?

In proof search, at each step we have a proof state and we need to decide which branch to explore next. We already have an LLM that generates candidate tactics with log-probabilities. Why add an EBM?

The LLM tells you: "given this proof state, how likely is each tactic?"
The EBM tells you: "is this proof state itself promising? Is it worth expanding?"

These are different questions:

- The LLM is a **policy** (what action to take)
- The EBM is a **value function** (how good is this position)

This is exactly the AlphaGo architecture: policy network + value network. The policy says "this move looks good," the value says "this board position is winning." Together they're much stronger than either alone.

In our combined score: `α × log_prob + β × ebm_score`

- `log_prob` comes from the LLM (policy): "this tactic is likely to be correct"
- `ebm_score` comes from the EBM (value): "the resulting proof state is promising"

The EBM helps with **search efficiency**: instead of blindly following the LLM's suggestions (which might lead to dead ends that look plausible), the EBM can recognize that a proof state is a dead end *before* we waste nodes exploring it.

---

## Part 2: The Mathematics

### 2.1 Energy function

Our energy function is a neural network:

```
E_θ(x) = EnergyHead(Encoder(x))
```

Where:
- `x` is a proof state string (e.g., `"n : Nat ⊢ n + 0 = n"`)
- `Encoder(x)` is the frozen 7B DeepSeek model. It reads the text and produces a 4096-dimensional vector that captures the "meaning" of the proof state. This is the `encode_only()` function from Phase 2.
- `EnergyHead` is a small MLP (4096 → 512 → 256 → 1) that maps the encoding to a scalar energy. **This is the only part we train.**

Why freeze the encoder? The 7B model has 7 billion parameters. The energy head has ~5 million. Training 7B parameters requires enormous compute and risks catastrophic forgetting. Training 5M parameters is fast and safe. The 7B encoder already produces excellent representations of mathematical text — we just need to learn a simple function on top of those representations.

### 2.2 Contrastive learning: how to train without labels like "energy should be -2.3"

We don't know the "correct" energy for any proof state. We only know relative ordering: proof states on the path to a successful proof (positives) should have lower energy than dead-end states (negatives).

This is **contrastive learning**: we don't learn absolute values, we learn to rank.

Training data comes from Phase 3 search trajectories:
- **Positive states**: nodes on the path from the initial goal to QED (proof found)
- **Negative states**: nodes that were explored but didn't lead to a proof

Each training example is: (one positive state, K negative states). We want:

```
E(positive) < E(negative₁)
E(positive) < E(negative₂)
...
E(positive) < E(negativeₖ)
```

### 2.3 InfoNCE Loss: making contrastive learning work

InfoNCE (Noise-Contrastive Estimation) is the standard loss function for contrastive learning. It was introduced by van den Oord et al. (2018) for representation learning and is also the loss used in CLIP.

**The formula:**

```
L = -log( exp(-E(x⁺)) / (exp(-E(x⁺)) + Σᵢ exp(-E(xᵢ⁻))) )
```

Let's unpack this:

1. We negate the energies because lower energy = better, but softmax expects higher = better.
   So `-E(x)` acts as a "logit" (unnormalized log-probability).

2. The fraction inside the log is a softmax: "what fraction of the total probability mass
   belongs to the positive example?"

3. If the positive has much lower energy than all negatives:
   - exp(-E(x⁺)) is large
   - exp(-E(xᵢ⁻)) are all small
   - The fraction ≈ 1
   - log(1) = 0
   - Loss ≈ 0 ✓

4. If the positive has similar or higher energy than negatives:
   - The fraction is small
   - log(small number) is very negative
   - Loss is large (bad) ✓

**This is literally cross-entropy classification.** We have K+1 "classes" (1 positive + K negatives), and the "correct class" is index 0 (the positive). The logits are the negated energies. So InfoNCE is just:

```python
logits = torch.cat([-pos_energy.unsqueeze(1), -neg_energies], dim=1)  # (batch, K+1)
labels = torch.zeros(batch_size, dtype=torch.long)  # correct class = 0
loss = F.cross_entropy(logits, labels)
```

That's the entire implementation. The burn-rs version uses `CrossEntropyLoss` the same way.

**Why this works:** The gradient of InfoNCE pushes the positive energy down and the negative energies up, with force proportional to how "confused" the model currently is. If the model already ranks correctly, the gradient is small (no unnecessary updates). If the model is wrong, the gradient is large (strong correction).

### 2.4 Depth regression: a bonus signal

In addition to contrastive ranking, we add a secondary loss: for positive states (on the proof path), we know how many steps remain until QED. This is `remaining_depth`.

We want the energy to roughly correlate with remaining depth:
- States close to QED (remaining_depth = 1): low energy
- States far from QED (remaining_depth = 15): higher energy

This is a simple MSE regression:

```
L_depth = MSE(E(x), normalized_remaining_depth(x))
```

We normalize remaining_depth to [0, 1] within each batch so the scale matches.

This secondary signal helps the EBM learn a smooth gradient — not just "positive vs. negative" but "how close to done." It's weighted by `depth_loss_weight = 0.3` (less important than the main contrastive loss).

### 2.5 Total loss

```
L_total = L_InfoNCE + 0.3 × L_depth
```

Both losses push the model to assign lower energy to more-provable states.

---

## Part 3: Spectral Normalization — Why and How

### 3.1 The problem: training instability in EBMs

EBMs are notoriously hard to train. The core issue: there's no constraint on the energy function's output range. The model can learn to output E(positive) = -10000 and E(negative) = +10000. This seems fine (big gap!) but causes problems:

1. **Gradient explosion**: when energies are extreme, exp(-E) either overflows (NaN) or underflows (zero gradient). Training breaks.

2. **Mode collapse**: the energy function collapses to outputting the same value for everything (energy_std → 0). The model "gives up" on distinguishing states.

3. **Lipschitz instability**: small changes in the input cause huge changes in the energy. The model becomes oversensitive and doesn't generalize.

### 3.2 The solution: constrain the Lipschitz constant

A function f is **L-Lipschitz** if for all inputs x, y:

```
|f(x) - f(y)| ≤ L × |x - y|
```

In plain English: the output can't change faster than L times the input change. This prevents the energy from being too "spiky" — it must vary smoothly.

For a linear layer y = Wx, the Lipschitz constant equals the **spectral norm** of W — its largest singular value σ₁(W).

If we normalize W by dividing by σ₁(W):

```
W_normalized = W / σ₁(W)
```

Then the normalized layer has Lipschitz constant exactly 1. The output can change at most as fast as the input changes.

For a multi-layer network, the overall Lipschitz constant is the product of per-layer constants. If every layer has Lipschitz constant 1, the whole network has Lipschitz constant 1. This guarantees stable, well-behaved energy outputs.

### 3.3 Computing the spectral norm: power iteration

Computing the exact largest singular value of a matrix is expensive (SVD is O(n³)). Instead, we use **power iteration**, which approximates it iteratively.

Given weight matrix W of shape (d_out, d_in):

```
Initialize random vectors:
  u ∈ ℝ^{d_out}  (random normal)
  v ∈ ℝ^{d_in}   (random normal)

Repeat n_iterations times:
  v = W^T u / ‖W^T u‖    (normalize)
  u = W v / ‖W v‖         (normalize)

Spectral norm ≈ σ₁ = u^T W v   (scalar)
```

After convergence, u and v are the left and right singular vectors corresponding to the largest singular value. 5 iterations is usually enough for a good approximation.

### 3.4 Option C: random reinit per forward

In PyTorch, spectral norm implementations typically store u and v as persistent buffers and update them incrementally (1 iteration per forward pass, relying on momentum across calls). This works because PyTorch buffers persist across forward passes.

burn-rs doesn't have a `Buffer` concept (yet — we may PR one). So we use **Option C from the plan**: reinitialize u and v randomly at each forward pass and run 5 power iterations from scratch. This converges because:

- 5 power iterations from random init gives σ₁ accurate to ~1% for typical weight matrices
- The slight variance between calls doesn't matter — what matters is that the weight is approximately normalized, preventing energy blow-up

The key insight: spectral norm doesn't need to be exact. It's a regularizer. Getting σ₁ within 10% is enough to prevent training instability. 5 iterations from random init easily achieves this.

### 3.5 The full SpectralNormLinear forward pass

```rust
fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    let W = self.weight.val();  // (d_out, d_in)

    // Estimate spectral norm via power iteration
    let mut u = random_normal(d_out);
    let mut v = random_normal(d_in);
    for _ in 0..5 {
        v = normalize(W^T @ u);
        u = normalize(W @ v);
    }
    let sigma = u^T @ W @ v;  // scalar ≈ σ₁(W)

    // Normalize weight
    let W_normed = W / sigma;

    // Standard linear layer with normalized weight
    return input @ W_normed^T + bias;
}
```

Crucially: the division `W / sigma` IS part of the autodiff graph (gradients flow through it back to W). But u and v are NOT parameters — they're created fresh each call and discarded. This is correct: we want to train W, and the spectral norm constraint is applied as a transformation of W during the forward pass.

---

## Part 4: The Energy Head Architecture

### 4.1 MLP structure

```
Input: h ∈ ℝ^4096  (pooled encoder output)
  ↓
SpectralNormLinear(4096, 512) → SiLU → Dropout(0.1)
  ↓
SpectralNormLinear(512, 256) → SiLU → Dropout(0.1)
  ↓
SpectralNormLinear(256, 1, no bias)
  ↓
÷ temperature
  ↓
Output: E(x) ∈ ℝ  (scalar energy)
```

Every linear layer uses spectral normalization. This gives the whole network Lipschitz constant ≤ 1, ensuring stable energy outputs.

### 4.2 SiLU activation

SiLU (Sigmoid Linear Unit), also called "swish":

```
SiLU(x) = x × sigmoid(x) = x × (1 / (1 + exp(-x)))
```

It's a smooth, non-monotonic activation that works well in modern architectures. Used in LLaMA, DeepSeek, and many recent models. Unlike ReLU, it has no "dead neuron" problem.

SiLU is 1-Lipschitz (its derivative is bounded by ~1.1), so it doesn't violate our Lipschitz constraint.

### 4.3 Learnable temperature

The output is divided by a learnable temperature τ = exp(log_τ):

```
E(x) = raw_energy / τ
```

We parameterize it as `log_temperature` (initialized to 0, so τ starts at 1) for two reasons:

1. **Unconstrained optimization**: log_τ can be any real number, but τ = exp(log_τ) is always positive. No need for constrained optimization.

2. **Sharpness control**: the temperature scales the energy. If the model learns τ < 1, it sharpens the energy landscape (bigger gaps between states). If τ > 1, it smooths it (smaller gaps). The model learns the right sharpness automatically.

The temperature has 1 parameter. It's trivial computationally but important for training stability.

### 4.4 Parameter count

```
Layer 1: 4096 × 512 + 512 = 2,097,664
Layer 2: 512 × 256 + 256  = 131,328
Layer 3: 256 × 1           = 256
Temperature:                = 1
Total:                      ≈ 2.23M parameters
```

This is tiny compared to the 7B encoder. Training it is fast — minutes on a GPU, not hours.

### 4.5 Why this architecture?

**Why 3 layers?** Two hidden layers with decreasing width (512 → 256) give enough capacity to learn a nonlinear energy landscape while being small enough to train quickly. One layer would be too linear. Four+ layers add complexity without clear benefit for this task.

**Why spectral norm on every layer?** The product of Lipschitz constants gives the network's overall Lipschitz constant. If any layer isn't normalized, the product can blow up. Normalizing all layers ensures the whole network is well-behaved.

**Why no bias on the last layer?** The energy's absolute value doesn't matter (we only compare energies). A bias on the last layer would just shift all energies by a constant, which cancels out in the InfoNCE softmax. Removing it slightly simplifies the model.

---

## Part 5: Negative Mining — Making Training Data Effective

### 5.1 Why negative selection matters

InfoNCE loss with K negatives is equivalent to a (K+1)-way classification problem. If the negatives are too easy (obviously different from the positive), the model gets perfect accuracy without learning anything useful. It's like giving a student a test where the wrong answers are obviously absurd — they pass without understanding the material.

Good negatives should be **hard**: states that look similar to the positive but are actually dead ends.

### 5.2 The three types of negatives

For a positive state from theorem T at depth d on the proof path:

**Hard negatives (50%)**: dead-end states from the SAME theorem T.
- These are states where the prover tried a tactic on T but it led nowhere.
- They share the same theorem context and similar structure.
- The model must learn subtle differences between promising and dead-end states.
- Example: positive = `n : Nat ⊢ n + 0 = n`, hard negative = `n : Nat, h : n = n ⊢ 0 = 0`
  (a state reached by a wrong application of hypothesis)

**Medium negatives (30%)**: other positive states from the same theorem (siblings).
- These are on the proof path but at a different depth or branch.
- They're useful because the model should learn ordering: states closer to QED should
  have lower energy than states further from QED.
- Slightly paradoxical: a "positive" state used as a negative. But in contrastive learning,
  "negative" just means "this specific item is not the one we're ranking highest right now."

**Easy negatives (20%)**: states from DIFFERENT theorems.
- These are from entirely different proof contexts.
- They provide baseline separation: the model should at minimum distinguish between
  different theorem contexts.
- These are easy to classify and provide stable gradients early in training.

### 5.3 Why the 50/30/20 split?

Too many hard negatives → training is too difficult early on, model doesn't learn.
Too many easy negatives → model plateaus quickly, doesn't learn fine distinctions.
The 50/30/20 split provides a curriculum: the easy negatives give the model initial signal, the medium ones refine it, and the hard ones push for maximum discrimination.

This is informed by the contrastive learning literature (e.g., MoCo, SimCLR). The exact ratios aren't critical — anything from 40/30/30 to 60/20/20 would likely work. The important thing is having a mix.

---

## Part 6: The Training Loop in Detail

### 6.1 Step by step

For each training step:

```
1. SAMPLE: Pick batch_size contrastive examples
   - Each example: 1 positive + K negatives (K=4 default)
   - Using the hard/medium/easy mix from Section 5

2. ENCODE (slow, frozen, candle):
   For each of the batch_size × (1 + K) proof state strings:
     embedding = TacticGenerator.encode_only(state)  →  Vec<f32>, dim 4096
   This is the bottleneck — each encode_only() call runs a 7B model forward pass.
   batch_size=32, K=4: that's 32 × 5 = 160 encoder calls per step.

3. CONVERT:
   Pack all Vec<f32> embeddings into burn tensors:
     pos_tensor: (batch_size, 4096)
     neg_tensor: (batch_size × K, 4096)

4. FORWARD (fast, trainable, burn-rs):
   pos_energy = energy_head.forward(pos_tensor)  →  (batch_size,)
   neg_energy = energy_head.forward(neg_tensor)  →  (batch_size × K,)
   neg_energy = neg_energy.reshape(batch_size, K) →  (batch_size, K)

5. LOSS:
   L_contrastive = InfoNCE(pos_energy, neg_energy)
   L_depth = depth_regression(pos_energy, remaining_depths)
   L_total = L_contrastive + 0.3 × L_depth

6. BACKWARD (burn-rs autodiff):
   grads = L_total.backward()
   Only the energy head's ~5M parameters get gradients.
   The encoder embeddings are detached (just Vec<f32> converted to tensors).

7. OPTIMIZE:
   model = optimizer.step(lr, model, grads)
   AdamW with weight decay 0.01, gradient clipping at norm 1.0.
```

### 6.2 Learning rate schedule

Linear warmup for 1000 steps, then cosine annealing:

```
Step 0:      lr = 0           (warmup start)
Step 500:    lr = 0.5 × 1e-4  (warmup midpoint)
Step 1000:   lr = 1e-4        (warmup end = peak)
Step 25000:  lr = 0.5 × 1e-4  (cosine midpoint)
Step 50000:  lr ≈ 0           (cosine end)
```

Warmup prevents early instability (large random gradients at initialization). Cosine annealing provides smooth decay without the sudden drops of step schedules.

### 6.3 What to watch during training

**Energy gap** = mean(negative energy) - mean(positive energy)
- Should be positive (negatives have higher energy than positives)
- Should increase during training
- If it goes negative → model is learning backwards. Check your loss sign convention.

**Rank accuracy** = fraction of examples where positive has lowest energy in its group
- Random baseline: 1/(K+1) = 0.2 for K=4
- Good training: steadily increases toward 0.7-0.9
- If it plateaus at 0.5 → negatives may be too easy

**Energy std** = standard deviation of all energy outputs
- Should be moderate (0.5-5.0)
- If it drops below 0.1 → mode collapse (model outputting same energy for everything)
- If it exceeds 50 → energy explosion (increase weight decay or check for NaN)

**Loss trend** = should decrease
- InfoNCE with K=4: random baseline loss = ln(5) ≈ 1.61
- Good convergence: loss drops to 0.3-0.8 range
- If loss is stuck near 1.61 → model isn't learning (check gradients, LR)

---

## Part 7: The candle ↔ burn Bridge

### 7.1 Why two frameworks?

- **candle**: loads and runs the 7B LLM. It handles model loading, tokenization, and transformer forward passes. It's optimized for inference of large models.
- **burn-rs**: trains the energy head. It provides autodiff (automatic differentiation), optimizers, and a training loop. It's designed for training.

The 7B model is frozen — it never trains. So we use candle for it (better inference performance). The energy head needs gradients — so we use burn-rs (better training support).

The bridge between them is simple: candle produces `Vec<f32>` embeddings, which we convert to burn `Tensor<B, 2>` via `Tensor::from_data()`. This is a memory copy, not a GPU transfer (unless the devices differ), and takes microseconds.

### 7.2 Why not just use one framework?

You could, but:
- burn-rs can't easily load HuggingFace safetensors models (candle is built for this)
- candle doesn't have a full training loop with optimizers and autodiff (burn-rs is built for this)

Using each framework for its strength is cleaner than fighting either one.

### 7.3 Gradient flow across the bridge

This is critical to understand: **gradients do NOT flow through the bridge.**

```
                candle (frozen)          bridge        burn-rs (trainable)
proof_state  →  [7B transformer]  →  Vec<f32>  →  [burn Tensor]  →  [EnergyHead]  →  energy
                                                                          ↑
                                                      gradients stop here at the bridge
                                      gradients flow: energy → EnergyHead params
```

When burn-rs computes `loss.backward()`, it traces gradients back through the EnergyHead's parameters (weight, bias, log_temperature) but stops at the input tensor. The input tensor was created from raw floats — it has no gradient history from candle. This is exactly what we want: the encoder is frozen.

In ML terminology, this is equivalent to `.detach()` in PyTorch. But we get it for free because the frameworks don't share a computation graph.

---

## Part 8: How the EBM Integrates with Search

### 8.1 During search (inference)

```
For each expansion of a search node:

1. LLM generates 32 candidate tactics with log-probabilities
2. Filter to top 8 by log-prob (beam_width)
3. Apply each tactic in Lean → get child proof states
4. For each successful child state:
   a. Encode: encoder.encode_only(child_state) → Vec<f32>
   b. Score: energy_head.forward(embedding) → energy
   c. Compute score: -energy (negate: lower energy = higher score)
   d. Combined priority: α × log_prob + β × (-energy)
5. Push children onto priority queue ordered by combined priority
```

The EBM adds one encode_only() + one energy_head.forward() per child state. The energy_head forward is ~0.1ms (tiny MLP). The encode_only() is ~50ms on GPU (7B model forward pass). So the EBM adds ~50ms per child state.

With beam_width=8 and 600 expansions: 600 × 8 × 50ms = 240 seconds of EBM overhead. This is significant but acceptable within the 600-second timeout.

### 8.2 The score combination

```
priority = α × log_prob + β × ebm_score
         = 0.5 × log_prob + 0.5 × (-energy)
```

α and β control the balance:
- α=1, β=0: pure LLM (ignore EBM entirely)
- α=0, β=1: pure EBM (ignore LLM probabilities)
- α=0.5, β=0.5: equal weighting (our default)

In practice, you might want to tune these. If the EBM is poorly trained, set β low (0.1-0.2). If the LLM is weak but the EBM is good, increase β.

---

## Part 9: Reading List

### Essential (read these — 2 hours total)

1. **LeCun (2006) — "A Tutorial on Energy-Based Learning"**
   http://yann.lecun.com/exdb/publis/pdf/lecun-06.pdf
   THE foundational reference. Sections 1-4 are essential. Skip the detailed architectures in later sections. Focus on: what is an energy function, how contrastive methods work, why EBMs are different from probabilistic models.
   *Read time: 1 hour (first 20 pages)*

2. **Oord et al. (2018) — "Representation Learning with Contrastive Predictive Coding" (CPC)**
   https://arxiv.org/abs/1807.03748
   Introduces InfoNCE loss. Sections 2.1-2.3 are the key parts. The loss function you'll implement in Phase 4 comes directly from Equation 4 of this paper.
   *Read time: 30 minutes (Sections 1-3)*

3. **Miyato et al. (2018) — "Spectral Normalization for Generative Adversarial Networks"**
   https://arxiv.org/abs/1802.05957
   Introduces spectral norm for stabilizing training. Sections 2-3 cover the theory and the power iteration algorithm. The practical algorithm is in Algorithm 1 (just 6 lines).
   *Read time: 30 minutes (Sections 1-3)*

### Highly recommended

4. **Chen et al. (2020) — "A Simple Framework for Contrastive Learning" (SimCLR)**
   https://arxiv.org/abs/2002.05709
   Shows how InfoNCE + hard negatives work in practice. The temperature scaling trick (Section 3) directly informs our learnable temperature.
   *Read time: 20 minutes (skim Sections 1-3)*

5. **Silver et al. (2017) — "Mastering the game of Go without human knowledge" (AlphaGo Zero)**
   https://www.nature.com/articles/nature24270
   Our architecture mirrors AlphaGo's: shared backbone + policy head + value head. The value network in AlphaGo is trained to predict game outcomes (win/loss), which is analogous to our EBM predicting proof feasibility.
   *Read time: 20 minutes (skim for architecture, not RL details)*

### Useful context (skim if time permits)

6. **He et al. (2020) — "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo)**
   https://arxiv.org/abs/1911.05722
   Discusses hard negative mining strategies. Our hard/medium/easy negative mix is informed by MoCo's insights about negative quality.
   *Read time: 15 minutes*

7. **Du & Mordatch (2019) — "Implicit Generation and Modeling with Energy-Based Models"**
   https://arxiv.org/abs/1903.08689
   Modern take on EBMs with neural networks. Good for understanding why spectral norm and other stabilization techniques are necessary.
   *Read time: 15 minutes (skim)*

### Not required but interesting

8. **Lample et al. (2022) — "HyperTree Proof Search for Neural Theorem Proving"**
   https://arxiv.org/abs/2205.11491
   Uses MCTS with a learned value function for theorem proving. Different from our approach (they use a critic network, not an EBM) but the same core idea of combining policy + value for search.

### Recent (2024-2025) — the current landscape

These papers show where the field is right now. Read them after the essentials.

9. **Carbone (2024) — "Hitchhiker's guide to Energy-Based Models"**
   https://arxiv.org/abs/2406.13661
   Comprehensive 2024 survey connecting EBMs to GANs, VAEs, normalizing flows, and statistical physics. The best single reference for understanding where EBMs fit in the modern generative model landscape. Sections on training methods are directly relevant — covers contrastive divergence, score matching, noise-contrastive estimation, and why InfoNCE is preferred for discriminative EBMs like ours.
   *Read time: 45 min (skim for breadth, deep-read the training methods section)*

10. **Blondel et al. (2025) — "Autoregressive Language Models are Secretly Energy-Based Models"**
    https://arxiv.org/abs/2512.15605
    Published January 2026. Establishes a formal bijection between ARMs and EBMs via the soft Bellman equation. Directly relevant to our architecture: it theoretically justifies why using the same LLM backbone for both policy (autoregressive tactic generation) and value (EBM energy scoring) makes mathematical sense — they're two views of the same model. The "lookahead capability" analysis explains why our encode_only() embeddings carry value information even though the model was trained for next-token prediction.
    *Read time: 30 min (Sections 1-3, skip the distillation proofs)*

11. **Yang et al. (2025) — "CARTS: Diversified Tactic Calibration and Bias-Resistant Tree Search" (ICLR 2025)**
    https://openreview.net/forum?id=VQwI055flA
    The most directly relevant paper to burn-qed. CARTS trains a value function for Lean theorem proving and identifies exactly the problems we'll face: **false negatives** (states labeled as dead ends that were actually provable — we just didn't find the proof), **label imbalance** (far more negatives than positives), and **domain gap** (training data comes from earlier, weaker search). Their solution: preference modeling + a bias-adjustment term. If our EBM shows the problems they describe, their fixes apply directly.
    *Read time: 40 min (essential for understanding value function failure modes in theorem proving)*

12. **Xin et al. (2024) — "DeepSeek-Prover-V1.5" (ICLR 2025)**
    https://arxiv.org/abs/2408.08152
    The direct predecessor of our base model. Introduces RMaxTS (MCTS with intrinsic-reward exploration for proof search) and RLPAF (RL from proof assistant feedback). Section 4 on RMaxTS is the closest published system to what we're building — policy model generates tactics, tree search explores, proof assistant provides feedback. Key difference: they use intrinsic reward (novelty), we use learned EBM energy. Understanding their search budget and evaluation methodology helps calibrate our expectations.
    *Read time: 30 min (Sections 3-5)*

13. **Xin et al. (2025) — "DeepSeek-Prover-V2"**
    https://arxiv.org/abs/2504.21801
    The 671B model achieves 88.9% on miniF2F. Key insight for us: they DON'T use a learned value function at all. Instead, they use chain-of-thought subgoal decomposition from a huge model + expert iteration. The 7B version (our base model) is much weaker without the 671B teacher. This frames our project: we're adding a learned value function (EBM) to the 7B model to partially compensate for not having a 671B teacher. Understanding their cold-start pipeline helps explain why the 7B model's representations are good enough for our encoder.
    *Read time: 20 min (Sections 1-3, skim the RL details)*

14. **The Harmonic Team (2025) — "Aristotle: IMO-level Automated Theorem Proving"**
    https://arxiv.org/abs/2510.01346
    State-of-the-art system that solved 5/6 IMO 2025 problems. Uses MCTS with a learned value function in the spirit of Expert Iteration and AlphaZero, proving this architecture works at the highest level. Their value function is a critic (probability of proving) trained with binary labels from search outcomes — essentially the same signal as our EBM, just different parameterization. Validates the core architecture choice of burn-qed.
    *Read time: 20 min (Section 2 on proof search architecture)*

15. **Kripner — "nanoproof" (open-source AlphaProof reimplementation)**
    https://github.com/Kripner/nanoproof
    Open-source Python implementation of HyperTree Proof Search + expert iteration. Achieves 38.5% on miniF2F. Useful as a reference implementation — you can see concretely how the policy/value training loop works, how MCTS interacts with Lean, and how data flows between search and training. Our burn-qed is the same idea but with the value function in Rust/burn-rs instead of Python/PyTorch.
    *Read time: browse the README and key source files (1 hour)*

16. **Zhang et al. (2024) — "ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search" (NeurIPS 2024)**
    https://arxiv.org/abs/2406.03816
    Uses MCTS with a Process Reward Model (PRM) for step-level evaluation in math reasoning. The PRM plays the same role as our EBM: evaluating intermediate states, not just final outcomes. Their iterative self-training loop (search → collect data → train PRM → search again) is exactly our expert iteration. Key difference: they work with informal math (text), we work with formal proofs (Lean).
    *Read time: 20 min (Sections 1-3)*

### How these map to burn-qed design decisions

| Our decision | Validated by |
|-------------|-------------|
| Policy + value with shared backbone | Blondel 2025 (theoretical), Aristotle 2025 (empirical), AlphaZero |
| Contrastive EBM (not classifier) for value | Carbone 2024 survey (EBMs more stable for ranking) |
| Expert iteration loop | nanoproof, DeepSeek-Prover-V1.5, ReST-MCTS* |
| Hard negative mining | CARTS 2025 (shows value function fails without it) |
| Spectral norm for stability | Carbone 2024 (covers EBM training failure modes) |
| 7B model as frozen encoder | DeepSeek-Prover-V2 (7B model has good representations despite weaker search) |

---

## Part 10: Self-Check Questions

Before starting Phase 4, you should be able to answer these:

### Conceptual

1. Why does lower energy mean "better" in an EBM? (Convention. The Boltzmann distribution p(x) ∝ exp(-E(x)) assigns higher probability to lower energy.)

2. Why can't we just use a classifier (provable vs. not provable)? (The space of proof states is essentially infinite. We don't need calibrated probabilities — we need relative ordering. EBMs give ranking without normalization.)

3. Why do we freeze the encoder? (7B parameters vs. 5M. Training efficiency, catastrophic forgetting prevention, and we can share the encoder with the policy model.)

4. What happens if we DON'T use spectral normalization? (Energy values can grow unbounded → gradient explosion or mode collapse → training fails.)

### Mathematical

5. What's the InfoNCE loss when the model perfectly separates positive from negatives? (≈ 0)

6. What's the InfoNCE loss when the model outputs identical energy for everything? (ln(K+1))

7. Why does power iteration with 5 steps from random init converge? (The gap between the first and second singular values determines convergence rate. For typical weight matrices, this gap is large enough that 5 iterations give <1% error.)

8. What's the gradient of InfoNCE with respect to positive energy? (It pushes positive energy DOWN, with magnitude proportional to how much probability mass the negatives have.)

### Implementation

9. Where do gradients flow in our system? (Only through the EnergyHead parameters. The encoder embeddings are detached at the candle→burn bridge.)

10. What burn-rs backend do we use for training vs. inference? (Training: Autodiff<B>. Inference: B directly. The model must be converted between them.)

11. What's the overhead of spectral norm in forward pass? (5 matrix-vector multiplications per linear layer per call. Negligible for our small MLP.)

12. How many encoder calls happen per training step? (batch_size × (1 + K). With batch=32, K=4: 160 calls. This is the bottleneck.)

### Situational awareness (from the recent papers)

13. What are the three value-function failure modes that CARTS identifies? (False negatives from incomplete search, label imbalance between pos/neg, and domain gap between training and test distributions.)

14. Why does Blondel 2025 say our shared-backbone architecture is theoretically sound? (ARMs and EBMs are bijections in function space — the policy model's hidden states already implicitly encode energy/value information.)

15. How does our approach differ from DeepSeek-Prover-V2's? (They use chain-of-thought from a 671B model instead of a learned value function. We add a small trainable EBM to the 7B model to compensate for not having the 671B teacher.)

16. What does nanoproof's 38.5% on miniF2F tell us about our expected baseline? (A Python implementation of the same policy+value architecture achieves ~38%. Our Rust implementation with the same architecture should be in a similar range, with any improvement coming from better engineering or training.)

---

## Part 11: Mapping Concepts to Phase 4 Prompts

| Concept | Where you'll implement it |
|---------|--------------------------|
| Power iteration for σ₁(W) | Prompt 4.1 — `SpectralNormLinear.forward()` |
| Energy head MLP | Prompt 4.2 — `EnergyHead` struct and forward |
| InfoNCE loss | Prompt 4.3 — `info_nce_loss()` |
| Depth regression | Prompt 4.3 — `depth_regression_loss()` |
| Hard/medium/easy negatives | Prompt 4.4 — `ContrastiveSampler.sample()` |
| Training health monitoring | Prompt 4.5 — `EBMMetrics.health_check()` |
| candle→burn tensor bridge | Prompt 4.6 — `embeddings_to_burn_tensor()` |
| Training loop with LR schedule | Prompt 4.7 — `train()` function |
| Inference scoring for search | Prompt 4.8 — `EBMScorer` |
| Full pipeline test | Prompt 4.10 — integration test |
| Wiring into search engine | Prompt 4.11 — `ValueScorer` impl |

Each prompt is self-contained but builds on the previous. The concepts from this guide map directly to the code you'll write.

---

## Part 12: Common Misconceptions

**"The EBM needs to output a probability."**
No. It outputs an unconstrained scalar. We never compute the partition function Z.

**"The EBM needs to be trained on the same data as the LLM."**
No. The LLM trains on (state, tactic) pairs. The EBM trains on (positive state, negative states) contrasts from search trajectories. Completely different data.

**"Spectral normalization makes the model weaker."**
It constrains the model, but the constraint is beneficial. Without it, the model can achieve arbitrarily low loss by amplifying energy differences — which looks good on the training objective but doesn't generalize. Spectral norm forces the model to learn meaningful features rather than just scaling up outputs.

**"The encoder must be differentiable for EBM training."**
No. We use the encoder's outputs (embeddings) as fixed inputs to the energy head. Gradients only flow through the energy head. This is why we can use candle (no autodiff) for the encoder and burn-rs (autodiff) for the energy head.

**"More negatives per example = always better."**
Not exactly. More negatives (K) make the InfoNCE bound tighter, but they also slow training (more encoder calls). And with too many negatives, the easy ones dominate and the model doesn't learn from the hard ones. K=4 is a practical sweet spot.

**"The EBM will definitely improve search performance."**
Not guaranteed. If the LLM representations don't capture proof structure well, or if the training data is too small/noisy, the EBM may not help. That's why we have the lean_start validation (Phase 5 Prompt 5.7) and the α/β tuning knobs. The EBM is a bet, not a certainty.
