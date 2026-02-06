//! Async Lean 4 REPL client using the Pantograph JSON protocol.
//!
//! This crate provides a worker pool for interacting with Lean 4 via
//! Pantograph child processes. It handles spawning, communication,
//! timeout recovery, and automatic worker recycling.
//!
//! # Quick Start
//!
//! The easiest way to get started is with the bundled Pantograph submodule:
//!
//! ```rust,no_run
//! use lean_repl::{LeanPool, LeanPoolConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LeanPoolConfig::with_bundled_pantograph()
//!     .expect("Pantograph not found — run scripts/setup_pantograph.sh first");
//!
//! let pool = LeanPool::new(config).await?;
//! let mut proof = pool.start_proof("∀ (n : Nat), n = n").await?;
//! let sid = proof.state_id();
//! let result = proof.run_tactic(sid, None, "intro n").await?;
//! # Ok(())
//! # }
//! ```
//!
//! You can also configure the paths manually:
//!
//! ```rust,no_run
//! use lean_repl::{LeanPool, LeanPoolConfig};
//! use std::path::PathBuf;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = LeanPoolConfig {
//!     num_workers: 4,
//!     max_requests_per_worker: 1000,
//!     max_lifetime_secs: 1800,
//!     tactic_timeout_secs: 30,
//!     pantograph_path: PathBuf::from("lake"),
//!     lean_env_path: PathBuf::from("/path/to/Pantograph"),
//!     imports: vec!["Init".to_string()],
//! };
//!
//! let pool = LeanPool::new(config).await?;
//! let mut proof = pool.start_proof("∀ (n : Nat), n = n").await?;
//! let sid = proof.state_id();
//! let result = proof.run_tactic(sid, None, "intro n").await?;
//! # Ok(())
//! # }
//! ```

pub mod pool;
pub mod protocol;
pub mod session;
pub mod types;
pub mod worker;

pub use pool::{LeanPool, ProofHandle, ProofHandleOwned, WorkerGuard, WorkerGuardOwned};
pub use protocol::{PantographRequest, PantographResponse};
pub use session::ProofSession;
pub use types::{discover_pantograph, Goal, LeanError, LeanPoolConfig, ProofState, TacticResult};
pub use worker::LeanWorker;
