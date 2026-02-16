//! Best-first proof search engine.
//!
//! Wires together lean-repl (Phase 1) and policy (Phase 2) to search for
//! proofs of Lean 4 theorems. Uses trait-based abstraction so the search
//! algorithm can be tested with mocks (no Lean, no LLM).
//!
//! # Key types
//!
//! - [`SearchEngine`] — the main search driver
//! - [`SearchConfig`] — configuration loaded from TOML
//! - [`SearchNode`] / [`ScoredNode`] — tree nodes and priority queue entries
//! - [`PolicyProvider`] / [`ValueScorer`] — traits for tactic generation and scoring.
//!   `ValueScorer` supports `score_batch()` for deferred batch EBM scoring.
//! - [`ProofEnvironment`] / [`TacticRunner`] — traits for Lean interaction
//! - [`InferencePolicyProvider`] — SGLang-backed policy adapter
//! - [`CachedPolicy`] — LRU cache wrapper with batch-aware miss grouping and hit/miss stats

pub mod adapters;
pub mod batcher;
pub mod cache;
pub mod config;
pub mod encode_batcher;
pub mod engine;
pub mod mocks;
pub mod node;

pub use adapters::{CachingEncoder, InferencePolicyProvider};
pub use batcher::GlobalBatcher;
pub use cache::CachedPolicy;
pub use encode_batcher::{BatchEncoder, GlobalEncodeBatcher};
pub use config::SearchConfig;
pub use engine::{
    PolicyProvider, ProofEnvironment, SearchEngine, SearchError, TacticRunner, ValueScorer,
};
pub use trajectory::SearchStats;
pub use node::{extract_proof_path, extract_tactic_sequence, ScoredNode, SearchNode};
