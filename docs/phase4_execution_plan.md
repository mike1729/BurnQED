# Phase 4: EBM in burn-rs — Execution Plan

**Goal:** Implement the Energy-Based Model that scores proof states to guide best-first search.

## Context

Phase 4 builds the Energy-Based Model that scores proof states to guide best-first search. This is the largest phase (12 prompts in `docs/phase4_instructions.md`) and the core burn-rs learning goal. All code lives in `crates/ebm/` (currently all stubs). The EBM is a SpectralNorm MLP (~5M params) that trains on trajectory data from Phase 3. Only the energy head trains in burn-rs; the 7B encoder is frozen in candle.

```
DeepSeek-7B (candle, FROZEN) --> encode_only() --> Vec<f32>
                                                     |
                                                     v
                               EnergyHead (burn-rs, TRAINABLE)
                               SpectralNorm MLP: 4096 -> 512 -> 256 -> 1
                               Output: scalar energy (lower = more provable)
```

## Dependency Graph (10 parts)

```
Part 1 (SpectralNormLinear)
   |
Part 2 (EnergyHead)
   |
Parts 3, 4, 5, 6 (bridge, loss, data, metrics -- parallel, no inter-deps)
   |
Part 7 (Training Loop -- depends on 2-6)
   |
Part 8 (Inference + ValueScorer)
   |
Part 9 (CLI wiring)
   |
Part 10 (E2E tests + docs)
```

## Parts Overview

| Part | Files | Summary |
|------|-------|---------|
| 1 | `spectral_norm.rs` | SpectralNormLinear with power iteration, 5 unit tests |
| 2 | `energy_head.rs` | 3-layer SiLU MLP with dropout + learnable temperature (6 tests) |
| 3 | `bridge.rs`, `encoder.rs` | candle Vec<f32> <-> burn Tensor, EncoderBackend enum (7 tests) |
| 4 | `loss.rs` | InfoNCE contrastive + depth regression loss (7 tests) |
| 5 | `data.rs` | Parquet loader, ContrastiveIndex, negative mining sampler (7 tests) |
| 6 | `metrics.rs` | EBMMetrics, health checks, MetricsHistory (7 tests) |
| 7 | `trainer.rs` | Training loop: AdamW, warmup+cosine LR, checkpoints (4 tests) |
| 8 | `inference.rs` | EBMScorer, EBMValueFn implementing search::ValueScorer (4 tests) |
| 9 | `main.rs`, `pipeline.rs` | train-ebm CLI, --ebm-path flag for search (2 tests) |
| 10 | `tests/integration.rs`, `CLAUDE.md` | E2E synthetic training, checkpoint roundtrip (4 tests) |
