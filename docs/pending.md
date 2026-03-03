# Pending Plans

## Add Goedel-Prover-V2-8B Support

**Goal**: Run baseline PutnamBench evaluation with [Goedel-Prover-V2-8B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-8B) (Qwen3-8B-based, 83% miniF2F Pass@32) using existing hybrid whole-proof search — same way we run DeepSeek.

**Key difference**: DeepSeek uses custom chat tokens (`<｜User｜>`, `<｜Assistant｜>`) + assistant priming (`example := by\n  `). Goedel-V2 uses Qwen3 chat template (`<|im_start|>user`, `<|im_end|>`) and was trained for whole-proof generation (no assistant priming needed). Hidden size is 4096 for both — EBM unchanged.

### Changes

#### 1. `PromptFormat` enum — `crates/policy/src/prompt.rs`

```rust
pub enum PromptFormat {
    DeepSeekProver,  // existing behavior
    GoedelV2,        // Qwen3 chat template
}
```

Methods: `from_str()`, `format_prompt(proof_state)`, `stop_tokens()`, `whole_proof_stop_tokens()`.

- **DeepSeekProver**: `<｜begin▁of▁sentence｜><｜User｜>{msg}<｜Assistant｜>```lean4\n...\nexample := by\n  `. Stop: `["```", "\n\n"]` / `["```"]`.
- **GoedelV2**: `<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`. No priming. Stop: `["<|im_end|>"]`.

#### 2. Thread through SglangClient — `crates/policy/src/sglang.rs`

- Add `prompt_format: PromptFormat` to `SglangConfig`
- Replace hardcoded `format_prompt()` (~line 949) and stop tokens in `generate_candidates()` (~line 280) and `generate_whole_proofs()` (~line 563)

#### 3. Config + CLI — `crates/search/src/config.rs`, `crates/prover-core/src/main.rs`

- Add `prompt_format: Option<String>` to search config (default `"deepseek-prover"`)
- Add `--prompt-format` CLI arg

#### 4. Config files

- `configs/models.toml` — add `prompt_format = "deepseek-prover"` to `[llm]`
- `configs/models_goedel.toml` — new file with Goedel-V2 defaults

#### 5. Scripts

- `scripts/download_goedel_v2.sh` — `huggingface-cli download Goedel-LM/Goedel-Prover-V2-8B`
- `scripts/run_putnam_eval.sh` — add `PROMPT_FORMAT` env var, pass `--prompt-format`

### Usage

```bash
# Download
bash scripts/download_goedel_v2.sh

# Launch SGLang
MODEL_PATH=data/models/base/goedel-prover-v2-8b QUANTIZATION=bfloat16 bash scripts/start_sglang.sh

# Eval
PROMPT_FORMAT=goedel-v2 bash scripts/run_putnam_eval.sh
```

### Files summary

| File | Change |
|------|--------|
| `crates/policy/src/prompt.rs` | Add `PromptFormat` enum |
| `crates/policy/src/sglang.rs` | Use `PromptFormat` for prompts + stop tokens |
| `crates/search/src/config.rs` | Add `prompt_format` field |
| `crates/prover-core/src/main.rs` | Add `--prompt-format` CLI arg |
| `configs/models.toml` | Add `prompt_format` default |
| `configs/models_goedel.toml` | New config |
| `scripts/run_putnam_eval.sh` | Pass `--prompt-format` |
| `scripts/download_goedel_v2.sh` | New download script |

### Not changed

EBM head, encode server, Python training, search engine (`engine.rs`), Lean REPL, tactic extraction — all model-agnostic.
