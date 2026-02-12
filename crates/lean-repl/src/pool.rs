use std::sync::{Arc, Mutex};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, SemaphorePermit};

use crate::types::{LeanError, LeanPoolConfig, ProofState, TacticResult};
use crate::worker::LeanWorker;

/// Pool of Lean workers with semaphore-based concurrency control.
///
/// The pool maintains a free list of workers and uses a semaphore to
/// limit the number of concurrent Lean process interactions. Workers
/// are automatically recycled when they exceed their request limit
/// or lifetime.
pub struct LeanPool {
    workers: Mutex<Vec<LeanWorker>>,
    semaphore: Arc<Semaphore>,
    config: LeanPoolConfig,
}

impl LeanPool {
    /// Create a new pool, spawning `config.num_workers` Pantograph processes.
    pub async fn new(config: LeanPoolConfig) -> Result<Self, LeanError> {
        if config.num_workers == 0 {
            return Err(LeanError::Protocol(
                "num_workers must be at least 1".into(),
            ));
        }

        let mut workers = Vec::with_capacity(config.num_workers);
        for _ in 0..config.num_workers {
            workers.push(LeanWorker::spawn(&config).await?);
        }

        let num = config.num_workers;
        tracing::info!(num_workers = num, "Lean pool initialized");

        Ok(Self {
            workers: Mutex::new(workers),
            semaphore: Arc::new(Semaphore::new(num)),
            config,
        })
    }

    /// Acquire a worker from the pool, recycling if needed.
    ///
    /// Blocks until a worker is available (semaphore-controlled).
    async fn acquire(&self) -> Result<(LeanWorker, SemaphorePermit<'_>), LeanError> {
        let permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| LeanError::Protocol("Semaphore closed".into()))?;

        let mut worker = {
            let mut workers = self.workers.lock().unwrap();
            workers
                .pop()
                .ok_or_else(|| LeanError::Protocol("No workers available despite permit".into()))?
        };

        // Recycle if the worker has exceeded its limits
        if worker.needs_recycling() {
            tracing::debug!(
                requests = worker.requests_handled(),
                "Recycling worker on checkout"
            );
            worker.recycle().await?;
        }

        Ok((worker, permit))
    }

    /// Acquire a worker using an owned semaphore permit (for `'static` use).
    async fn acquire_owned(
        self: &Arc<Self>,
    ) -> Result<(LeanWorker, OwnedSemaphorePermit), LeanError> {
        let permit = self
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| LeanError::Protocol("Semaphore closed".into()))?;

        let mut worker = {
            let mut workers = self.workers.lock().unwrap();
            workers
                .pop()
                .ok_or_else(|| LeanError::Protocol("No workers available despite permit".into()))?
        };

        if worker.needs_recycling() {
            tracing::debug!(
                requests = worker.requests_handled(),
                "Recycling worker on checkout (owned)"
            );
            worker.recycle().await?;
        }

        Ok((worker, permit))
    }

    /// Return a worker to the free list (synchronous, used from Drop).
    fn return_worker_sync(workers: &Mutex<Vec<LeanWorker>>, worker: LeanWorker) {
        workers.lock().unwrap().push(worker);
    }

    /// Start a new proof and return a [`ProofHandle`] that holds the worker
    /// for the lifetime of the proof attempt.
    ///
    /// All tactic applications on the returned handle are guaranteed to use
    /// the same Pantograph process, so state IDs remain valid.
    pub async fn start_proof(&self, expr: &str) -> Result<ProofHandle<'_>, LeanError> {
        let mut guard = self.checkout().await?;
        let state = guard.worker().start_proof(expr).await?;
        Ok(ProofHandle {
            guard,
            initial_state: state,
        })
    }

    /// Start a new proof (owned variant for spawned tasks with `Arc<LeanPool>`).
    ///
    /// Returns a [`ProofHandleOwned`] that is `'static` and can be sent
    /// across `tokio::spawn` boundaries.
    pub async fn start_proof_owned(
        self: &Arc<Self>,
        expr: &str,
    ) -> Result<ProofHandleOwned, LeanError> {
        let mut guard = self.checkout_owned().await?;
        let state = guard.worker().start_proof(expr).await?;
        Ok(ProofHandleOwned {
            guard,
            initial_state: state,
        })
    }

    /// Start a new proof by looking up a theorem name, returning a [`ProofHandle`].
    ///
    /// Uses Pantograph's `copyFrom` to resolve a fully-qualified theorem name.
    pub async fn start_proof_by_name(&self, name: &str) -> Result<ProofHandle<'_>, LeanError> {
        let mut guard = self.checkout().await?;
        let state = guard.worker().start_proof_by_name(name).await?;
        Ok(ProofHandle {
            guard,
            initial_state: state,
        })
    }

    /// Start a new proof by theorem name (owned variant for spawned tasks).
    ///
    /// Returns a [`ProofHandleOwned`] that is `'static` and can be sent
    /// across `tokio::spawn` boundaries.
    pub async fn start_proof_by_name_owned(
        self: &Arc<Self>,
        name: &str,
    ) -> Result<ProofHandleOwned, LeanError> {
        let mut guard = self.checkout_owned().await?;
        let state = guard.worker().start_proof_by_name(name).await?;
        Ok(ProofHandleOwned {
            guard,
            initial_state: state,
        })
    }

    /// Check out a worker for exclusive use across multiple operations.
    ///
    /// State IDs are scoped to a single Pantograph process. When running
    /// multi-step proofs (start_proof → apply_tactic → apply_tactic),
    /// you MUST use the same worker for all steps. This method lets you
    /// hold a worker for the duration of a proof attempt.
    ///
    /// The returned `WorkerGuard` releases the worker back to the pool
    /// when dropped.
    pub async fn checkout(&self) -> Result<WorkerGuard<'_>, LeanError> {
        let (worker, permit) = self.acquire().await?;
        Ok(WorkerGuard {
            worker: Some(worker),
            pool_workers: &self.workers,
            _permit: permit,
        })
    }

    /// Check out a worker (owned variant for spawned tasks with `Arc<LeanPool>`).
    ///
    /// Returns a [`WorkerGuardOwned`] that is `'static` and can be sent
    /// across `tokio::spawn` boundaries.
    pub async fn checkout_owned(
        self: &Arc<Self>,
    ) -> Result<WorkerGuardOwned, LeanError> {
        let (worker, permit) = self.acquire_owned().await?;
        Ok(WorkerGuardOwned {
            worker: Some(worker),
            pool: Arc::clone(self),
            _permit: permit,
        })
    }

    /// Number of workers currently available (not checked out).
    pub fn available_workers(&self) -> usize {
        self.semaphore.available_permits()
    }

    /// Total number of workers in the pool.
    pub fn num_workers(&self) -> usize {
        self.config.num_workers
    }

    /// Get a reference to the pool configuration.
    pub fn config(&self) -> &LeanPoolConfig {
        &self.config
    }

    /// Shut down the pool, killing all worker processes.
    pub async fn shutdown(&self) {
        let mut workers = self.workers.lock().unwrap();
        for worker in workers.iter_mut() {
            worker.shutdown().await;
        }
        workers.clear();
        tracing::info!("Lean pool shut down");
    }
}

// ---------------------------------------------------------------------------
// WorkerGuard (borrowed — borrows &LeanPool)
// ---------------------------------------------------------------------------

/// RAII guard that holds a worker checked out from the pool.
///
/// The worker is returned to the pool when this guard is dropped.
/// Use this to run multi-step proofs where state IDs must remain
/// valid across operations.
pub struct WorkerGuard<'a> {
    worker: Option<LeanWorker>,
    pool_workers: &'a Mutex<Vec<LeanWorker>>,
    _permit: SemaphorePermit<'a>,
}

impl<'a> WorkerGuard<'a> {
    /// Get a mutable reference to the underlying worker.
    pub fn worker(&mut self) -> &mut LeanWorker {
        self.worker.as_mut().expect("worker already taken")
    }
}

impl<'a> Drop for WorkerGuard<'a> {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            LeanPool::return_worker_sync(self.pool_workers, worker);
        }
    }
}

// ---------------------------------------------------------------------------
// WorkerGuardOwned ('static — holds Arc<LeanPool>)
// ---------------------------------------------------------------------------

/// Owned RAII guard that holds a worker checked out from the pool.
///
/// Like [`WorkerGuard`] but `'static`, so it can be sent across
/// `tokio::spawn` boundaries when you have an `Arc<LeanPool>`.
pub struct WorkerGuardOwned {
    worker: Option<LeanWorker>,
    pool: Arc<LeanPool>,
    _permit: OwnedSemaphorePermit,
}

impl WorkerGuardOwned {
    /// Get a mutable reference to the underlying worker.
    pub fn worker(&mut self) -> &mut LeanWorker {
        self.worker.as_mut().expect("worker already taken")
    }
}

impl Drop for WorkerGuardOwned {
    fn drop(&mut self) {
        if let Some(worker) = self.worker.take() {
            LeanPool::return_worker_sync(&self.pool.workers, worker);
        }
    }
}

// ---------------------------------------------------------------------------
// ProofHandle (borrowed — borrows &LeanPool)
// ---------------------------------------------------------------------------

/// Handle to an in-progress proof. Holds a worker for the proof's lifetime.
///
/// The worker is returned to the pool when the handle is dropped.
/// All tactic applications go to the same Pantograph process,
/// ensuring state IDs remain valid.
pub struct ProofHandle<'a> {
    guard: WorkerGuard<'a>,
    initial_state: ProofState,
}

impl<'a> ProofHandle<'a> {
    /// The initial proof state returned by `goal.start`.
    pub fn initial_state(&self) -> &ProofState {
        &self.initial_state
    }

    /// The initial state ID (convenience for `initial_state().state_id`).
    pub fn state_id(&self) -> u64 {
        self.initial_state.state_id
    }

    /// Apply a tactic — guaranteed to use the same worker that started this proof.
    pub async fn run_tactic(
        &mut self,
        state_id: u64,
        goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<TacticResult, LeanError> {
        self.guard.worker().apply_tactic(state_id, goal_id, tactic).await
    }

    /// Access the underlying worker for advanced use.
    pub fn worker(&mut self) -> &mut LeanWorker {
        self.guard.worker()
    }
}

// ---------------------------------------------------------------------------
// ProofHandleOwned ('static — holds Arc<LeanPool>)
// ---------------------------------------------------------------------------

/// Owned handle to an in-progress proof.
///
/// Like [`ProofHandle`] but `'static`, so it can be sent across
/// `tokio::spawn` boundaries when you have an `Arc<LeanPool>`.
pub struct ProofHandleOwned {
    guard: WorkerGuardOwned,
    initial_state: ProofState,
}

impl ProofHandleOwned {
    /// The initial proof state returned by `goal.start`.
    pub fn initial_state(&self) -> &ProofState {
        &self.initial_state
    }

    /// The initial state ID (convenience for `initial_state().state_id`).
    pub fn state_id(&self) -> u64 {
        self.initial_state.state_id
    }

    /// Apply a tactic — guaranteed to use the same worker that started this proof.
    pub async fn run_tactic(
        &mut self,
        state_id: u64,
        goal_id: Option<u64>,
        tactic: &str,
    ) -> Result<TacticResult, LeanError> {
        self.guard.worker().apply_tactic(state_id, goal_id, tactic).await
    }

    /// Access the underlying worker for advanced use.
    pub fn worker(&mut self) -> &mut LeanWorker {
        self.guard.worker()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn semaphore_limits_concurrency() {
        let sem = tokio::sync::Semaphore::new(4);
        assert_eq!(sem.available_permits(), 4);
    }
}
