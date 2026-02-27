//! Hybrid whole-proof search engine with replay trie.
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
//! - [`PolicyProvider`] / [`ValueScorer`] — traits for proof generation and scoring.
//!   `ValueScorer` supports `score_batch()` for deferred batch EBM scoring.
//! - [`ProofEnvironment`] / [`TacticRunner`] — traits for Lean interaction
//! - [`InferencePolicyProvider`] — SGLang-backed policy adapter

pub mod adapters;
pub mod config;
pub mod engine;
pub mod mocks;
pub mod node;
pub mod trie;

pub use adapters::{CachingEncoder, InferencePolicyProvider, NullPolicyProvider};
pub use config::SearchConfig;
pub use engine::{
    PolicyProvider, ProofEnvironment, SearchEngine, SearchError, TacticRunner, ValueScorer,
};
pub use trajectory::SearchStats;
pub use node::{extract_proof_path, extract_tactic_sequence, ScoredNode, SearchNode};
