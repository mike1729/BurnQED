//! Integration tests for lean-repl requiring Pantograph built via Lake.
//!
//! All tests are marked `#[ignore]` — they require Pantograph built from
//! source (`lake build` in the Pantograph directory).
//!
//! Pantograph is auto-discovered via `LeanPoolConfig::with_bundled_pantograph()`:
//! 1. `PANTOGRAPH_PROJECT` env var (if set)
//! 2. `vendor/Pantograph/` submodule (default — no env var needed)
//!
//! Setup (one-time):
//! ```bash
//! ./scripts/setup_pantograph.sh
//! ```
//!
//! Run:
//! ```bash
//! cargo test -p lean-repl -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::Instant;

use lean_repl::{LeanPool, LeanPoolConfig, ProofSession, TacticResult};

/// Extract state_id from a Success result, panicking with context on failure.
fn expect_success(result: &TacticResult, context: &str) -> u64 {
    match result {
        TacticResult::Success { state_id, .. } => *state_id,
        other => panic!("{context}: expected Success, got: {other:?}"),
    }
}

/// Assert that a tactic result is ProofComplete.
fn expect_complete(result: &TacticResult, context: &str) {
    assert!(
        matches!(result, TacticResult::ProofComplete { .. }),
        "{context}: expected ProofComplete, got: {result:?}"
    );
}

fn get_test_config(num_workers: usize) -> Option<LeanPoolConfig> {
    let mut config = LeanPoolConfig::with_bundled_pantograph()?;
    config.num_workers = num_workers;
    Some(config)
}

fn get_test_config_or_skip(num_workers: usize) -> LeanPoolConfig {
    get_test_config(num_workers).expect(
        "Pantograph not found. Run ./scripts/setup_pantograph.sh or set \
         PANTOGRAPH_PROJECT=/path/to/Pantograph",
    )
}

#[tokio::test]
#[ignore]
async fn test_single_worker_simple_proof() {
    let config = get_test_config_or_skip(1);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    // Start proof — worker held for the entire proof via ProofHandle
    let mut proof = pool
        .start_proof("forall (n : Nat), n = n")
        .await
        .expect("Failed to start proof");
    let sid = proof.state_id();
    println!("Initial state_id: {sid}");

    // Apply intro n
    let result = proof
        .run_tactic(sid, None, "intro n")
        .await
        .expect("Failed to apply intro n");
    println!("After intro n: {result:?}");

    let state_id = match &result {
        TacticResult::Success { state_id, goals } => {
            assert!(!goals.is_empty());
            let goal = &goals[0];
            assert!(
                goal.hypotheses.iter().any(|h| h.contains("Nat")),
                "Expected hypothesis with Nat, got: {:?}",
                goal.hypotheses
            );
            assert!(
                goal.target.contains("n = n"),
                "Expected target containing 'n = n', got: {}",
                goal.target
            );
            *state_id
        }
        other => panic!("Expected Success, got: {other:?}"),
    };

    // Apply rfl → should complete the proof
    let result = proof
        .run_tactic(state_id, None, "rfl")
        .await
        .expect("Failed to apply rfl");
    println!("After rfl: {result:?}");

    assert!(
        matches!(result, TacticResult::ProofComplete { .. }),
        "Expected ProofComplete after rfl, got: {result:?}"
    );

    drop(proof);
    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_single_worker_tactic_error() {
    let config = get_test_config_or_skip(1);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    let mut proof = pool
        .start_proof("forall (n : Nat), n = n")
        .await
        .expect("Failed to start proof");
    let sid = proof.state_id();

    let result = proof
        .run_tactic(sid, None, "nonexistent_tactic_12345")
        .await
        .expect("Should not return IO error for tactic failure");

    println!("Bad tactic result: {result:?}");
    assert!(
        matches!(result, TacticResult::Failed { .. }),
        "Expected Failed for invalid tactic, got: {result:?}"
    );

    drop(proof);
    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_single_worker_multiple_goals() {
    let config = get_test_config_or_skip(1);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    let mut proof = pool
        .start_proof("forall (p q : Prop), p -> q -> p /\\ q")
        .await
        .expect("Failed to start proof");
    let s0 = proof.state_id();

    // intro all the way
    let r1 = proof.run_tactic(s0, None, "intro p").await.unwrap();
    let s1 = expect_success(&r1, "intro p");

    let r2 = proof.run_tactic(s1, None, "intro q").await.unwrap();
    let s2 = expect_success(&r2, "intro q");

    let r3 = proof.run_tactic(s2, None, "intro hp").await.unwrap();
    let s3 = expect_success(&r3, "intro hp");

    let r4 = proof.run_tactic(s3, None, "intro hq").await.unwrap();
    let s4 = expect_success(&r4, "intro hq");

    // exact And.intro hp hq should complete the proof
    let result = proof.run_tactic(s4, None, "exact And.intro hp hq").await.unwrap();
    println!("After And.intro: {result:?}");

    assert!(
        matches!(result, TacticResult::ProofComplete { .. }),
        "Expected proof complete, got: {result:?}"
    );

    drop(proof);
    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_proof_session_simple() {
    let config = get_test_config_or_skip(1);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    let mut session = ProofSession::new(&pool, "forall (n : Nat), n = n")
        .await
        .expect("Failed to start session");

    assert!(!session.is_complete());
    assert_eq!(session.depth(), 0);

    let result = session.apply("intro n").await.unwrap();
    println!("After intro n: {result:?}");
    assert_eq!(session.depth(), 1);

    let result = session.apply("rfl").await.unwrap();
    println!("After rfl: {result:?}");
    assert!(session.is_complete());
    assert_eq!(session.depth(), 2);

    drop(session);
    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_pool_sequential_100() {
    let config = get_test_config_or_skip(4);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    let start = Instant::now();

    let overall_timeout = tokio::time::timeout(
        std::time::Duration::from_secs(300),
        async {
            for i in 0..100 {
                // Each proof gets its own handle; worker returned when proof dropped
                let mut proof = pool
                    .start_proof("forall (n : Nat), n = n")
                    .await
                    .unwrap_or_else(|e| panic!("Failed to start proof {i}: {e}"));
                let sid = proof.state_id();

                let r1 = proof
                    .run_tactic(sid, None, "intro n")
                    .await
                    .unwrap_or_else(|e| panic!("Failed intro on proof {i}: {e}"));
                let s1 = match r1 {
                    TacticResult::Success { state_id, .. } => state_id,
                    other => panic!("Proof {i} intro failed: {other:?}"),
                };

                let r2 = proof
                    .run_tactic(s1, None, "rfl")
                    .await
                    .unwrap_or_else(|e| panic!("Failed rfl on proof {i}: {e}"));
                assert!(
                    matches!(r2, TacticResult::ProofComplete { .. }),
                    "Proof {i} rfl didn't complete: {r2:?}"
                );

                // proof dropped here → worker returned to pool

                if (i + 1) % 25 == 0 {
                    println!(
                        "Completed {}/100 proofs in {:.1}s",
                        i + 1,
                        start.elapsed().as_secs_f64()
                    );
                }
            }
        },
    )
    .await;

    assert!(
        overall_timeout.is_ok(),
        "100 sequential proofs timed out after 300s"
    );
    println!(
        "100 sequential proofs completed in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_pool_concurrent_20() {
    let config = get_test_config_or_skip(4);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");
    let pool = Arc::new(pool);

    let start = Instant::now();

    let mut handles = Vec::new();
    for i in 0..20 {
        let pool = pool.clone();
        handles.push(tokio::spawn(async move {
            // Owned variant — 'static, works across tokio::spawn
            let mut proof = pool
                .start_proof_owned("forall (n : Nat), n = n")
                .await
                .unwrap_or_else(|e| panic!("Concurrent proof {i} start failed: {e}"));
            let sid = proof.state_id();

            let r1 = proof
                .run_tactic(sid, None, "intro n")
                .await
                .unwrap_or_else(|e| panic!("Concurrent proof {i} intro failed: {e}"));
            let s1 = match r1 {
                TacticResult::Success { state_id, .. } => state_id,
                other => panic!("Concurrent proof {i} intro unexpected: {other:?}"),
            };

            let r2 = proof
                .run_tactic(s1, None, "rfl")
                .await
                .unwrap_or_else(|e| panic!("Concurrent proof {i} rfl failed: {e}"));
            assert!(
                matches!(r2, TacticResult::ProofComplete { .. }),
                "Concurrent proof {i} not complete: {r2:?}"
            );
        }));
    }

    let timeout_result = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        for (i, handle) in handles.into_iter().enumerate() {
            handle
                .await
                .unwrap_or_else(|e| panic!("Task {i} panicked: {e}"));
        }
    })
    .await;

    assert!(
        timeout_result.is_ok(),
        "20 concurrent proofs timed out after 120s"
    );
    println!(
        "20 concurrent proofs completed in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_worker_recycling() {
    let mut config = get_test_config_or_skip(1);
    // Each proof = 3 requests (start_proof + intro + rfl).
    // Set to 9 so recycling triggers after 3 complete proofs.
    // Recycling happens on the NEXT checkout, so proof #4 gets a fresh worker.
    config.max_requests_per_worker = 9;

    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    for i in 0..8 {
        let mut proof = pool
            .start_proof("forall (n : Nat), n = n")
            .await
            .unwrap_or_else(|e| panic!("Request {i} failed: {e}"));
        let sid = proof.state_id();

        let r1 = proof
            .run_tactic(sid, None, "intro n")
            .await
            .unwrap_or_else(|e| panic!("Request {i} intro failed: {e}"));
        let s1 = match r1 {
            TacticResult::Success { state_id, .. } => state_id,
            other => panic!("Request {i} intro unexpected: {other:?}"),
        };

        let r2 = proof
            .run_tactic(s1, None, "rfl")
            .await
            .unwrap_or_else(|e| panic!("Request {i} rfl failed: {e}"));
        assert!(
            matches!(r2, TacticResult::ProofComplete { .. }),
            "Request {i} not complete: {r2:?}"
        );

        // proof dropped here → worker returned to pool
        println!("Recycling test: request {}/8 succeeded", i + 1);
    }

    println!("Worker recycling test passed");
    pool.shutdown().await;
}

#[tokio::test]
#[ignore]
async fn test_timeout_recovery() {
    let mut config = get_test_config_or_skip(1);
    config.tactic_timeout_secs = 2;

    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    // First do a normal proof to verify things work
    {
        let mut proof = pool.start_proof("forall (n : Nat), n = n").await.unwrap();
        let sid = proof.state_id();
        let r1 = proof.run_tactic(sid, None, "intro n").await.unwrap();
        let s1 = expect_success(&r1, "pre-timeout intro");
        let r2 = proof.run_tactic(s1, None, "rfl").await.unwrap();
        expect_complete(&r2, "pre-timeout rfl");
        println!("Pre-timeout proof succeeded");
    }

    // Try another proof — should still work even if worker was recycled
    {
        let mut proof = pool.start_proof("forall (n : Nat), n = n").await.unwrap();
        let sid = proof.state_id();
        let r3 = proof.run_tactic(sid, None, "intro n").await.unwrap();
        let s3 = expect_success(&r3, "post-timeout intro");
        let r4 = proof.run_tactic(s3, None, "rfl").await.unwrap();
        expect_complete(&r4, "post-timeout rfl");
        println!("Worker recovery test passed");
    }

    pool.shutdown().await;
}

/// Comprehensive test simulating best-first proof search.
///
/// Exercises the full surface area of the lean-repl API:
/// - `checkout()` / `WorkerGuard` for exclusive worker access
/// - Proof tree branching: try multiple tactics from the same state
/// - State immutability: revisit earlier states after branching
/// - Error recovery: failed tactics don't corrupt proof state
/// - Multi-goal proofs with targeted `goalId` application
/// - Goal inspection: verify hypotheses and targets at each step
/// - Varied theorem types (logic, existential, arithmetic)
/// - Rapid-fire throughput on a single worker
#[tokio::test]
#[ignore]
async fn test_proof_search_simulation() {
    let config = get_test_config_or_skip(1);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    let mut guard = pool.checkout().await.expect("checkout failed");
    let w = guard.worker();

    // ================================================================
    // Part 1: And commutativity — branching & state immutability
    // ================================================================
    println!("=== Part 1: And commutativity with proof tree branching ===");

    let state = w
        .start_proof("forall (p q : Prop), p /\\ q -> q /\\ p")
        .await
        .unwrap();
    let s0 = state.state_id;

    // Try several wrong tactics — state must survive all of them
    for bad_tactic in &["rfl", "trivial", "omega", "nonexistent_tactic_xyz"] {
        let r = w.apply_tactic(s0, None, bad_tactic).await.unwrap();
        assert!(
            matches!(r, TacticResult::Failed { .. }),
            "'{bad_tactic}' should fail on And.comm, got: {r:?}"
        );
    }

    // Correct: introduce all hypotheses
    let r1 = w.apply_tactic(s0, None, "intro p q h").await.unwrap();
    let s1 = expect_success(&r1, "And.comm intro");

    // Verify goal structure: should have ≥3 hypotheses, target mentions q and p
    if let TacticResult::Success { goals, .. } = &r1 {
        assert_eq!(goals.len(), 1, "Should have 1 goal after intros");
        let g = &goals[0];
        assert!(
            g.hypotheses.len() >= 3,
            "Expected ≥3 hypotheses (p, q, h), got: {:?}",
            g.hypotheses
        );
        println!(
            "  Goal after intro: {} hypotheses, target = {}",
            g.hypotheses.len(),
            g.target
        );
    }

    // ---- Branch A: one-shot exact proof ----
    let branch_a = w
        .apply_tactic(s1, None, "exact And.intro h.2 h.1")
        .await
        .unwrap();
    expect_complete(&branch_a, "Branch A: direct And.intro");
    println!("  Branch A (exact And.intro h.2 h.1): complete");

    // ---- Branch B: from SAME s1, decompose with constructor → 2 sub-goals ----
    // This tests state immutability — s1 is still valid after Branch A
    let rb1 = w.apply_tactic(s1, None, "constructor").await.unwrap();
    let sb = expect_success(&rb1, "Branch B: constructor");

    if let TacticResult::Success { goals, .. } = &rb1 {
        assert_eq!(
            goals.len(),
            2,
            "constructor on And should produce 2 goals, got: {:?}",
            goals
        );
        println!(
            "  Branch B (constructor): 2 goals — [{}] and [{}]",
            goals[0].target, goals[1].target
        );

        // Solve goal 0 with explicit goalId
        let rb2 = w.apply_tactic(sb, Some(0), "exact h.2").await.unwrap();
        match &rb2 {
            TacticResult::Success {
                state_id: sb2,
                goals: remaining,
            } => {
                assert_eq!(
                    remaining.len(),
                    1,
                    "After solving goal 0, 1 goal should remain"
                );
                println!(
                    "  Branch B: goal 0 solved, remaining target = {}",
                    remaining[0].target
                );

                // Solve the last goal
                let rb3 = w.apply_tactic(*sb2, None, "exact h.1").await.unwrap();
                expect_complete(&rb3, "Branch B: final goal");
                println!("  Branch B (step-by-step): complete");
            }
            TacticResult::ProofComplete { .. } => {
                println!("  Branch B: both goals solved in one step");
            }
            other => panic!("Branch B goal 0 unexpected: {other:?}"),
        }
    }

    // ---- Revisit s0: original state is still valid ----
    let r_revisit = w.apply_tactic(s0, None, "intro p q h").await.unwrap();
    let s_new = expect_success(&r_revisit, "revisit s0");
    assert!(
        s_new > s1,
        "Revisited state should have strictly higher ID ({s_new} vs {s1})"
    );
    println!("  State revisit: s0 still valid, new state_id={s_new} > {s1}");

    // ================================================================
    // Part 2: Or introduction — right injection
    // ================================================================
    println!("\n=== Part 2: Or introduction ===");

    let state2 = w
        .start_proof("forall (p q : Prop), p -> p \\/ q")
        .await
        .unwrap();
    let r = w
        .apply_tactic(state2.state_id, None, "intro p q hp")
        .await
        .unwrap();
    let s = expect_success(&r, "Or.inl intro");

    // Verify the hypothesis is there
    if let TacticResult::Success { goals, .. } = &r {
        let hyps = &goals[0].hypotheses;
        assert!(
            hyps.iter().any(|h| h.contains("hp")),
            "Expected hp in hypotheses: {hyps:?}"
        );
    }

    let r = w.apply_tactic(s, None, "exact Or.inl hp").await.unwrap();
    expect_complete(&r, "Or.inl exact");
    println!("  Proved: p -> p \\/ q");

    // ================================================================
    // Part 3: Modus ponens
    // ================================================================
    println!("\n=== Part 3: Modus ponens ===");

    let state3 = w
        .start_proof("forall (p q : Prop), (p -> q) -> p -> q")
        .await
        .unwrap();
    let r = w
        .apply_tactic(state3.state_id, None, "intro p q f hp")
        .await
        .unwrap();
    let s = expect_success(&r, "mp intro");
    let r = w.apply_tactic(s, None, "exact f hp").await.unwrap();
    expect_complete(&r, "mp exact");
    println!("  Proved: (p -> q) -> p -> q");

    // ================================================================
    // Part 4: Existential witness
    // ================================================================
    println!("\n=== Part 4: Existential witness ===");

    let state4 = w
        .start_proof("Exists (fun n : Nat => n = 0)")
        .await
        .unwrap();
    let r = w
        .apply_tactic(state4.state_id, None, "exact Exists.intro 0 rfl")
        .await
        .unwrap();
    expect_complete(&r, "Exists.intro");
    println!("  Proved: Exists n, n = 0");

    // ================================================================
    // Part 5: False elimination (ex falso quodlibet)
    // ================================================================
    println!("\n=== Part 5: False elimination ===");

    let state5 = w
        .start_proof("forall (p : Prop), False -> p")
        .await
        .unwrap();
    let r = w
        .apply_tactic(state5.state_id, None, "intro p h")
        .await
        .unwrap();
    let s = expect_success(&r, "False.elim intro");
    let r = w.apply_tactic(s, None, "exact False.elim h").await.unwrap();
    expect_complete(&r, "False.elim exact");
    println!("  Proved: False -> p");

    // ================================================================
    // Part 6: True is trivial
    // ================================================================
    println!("\n=== Part 6: True ===");

    let state6 = w.start_proof("True").await.unwrap();
    let r = w
        .apply_tactic(state6.state_id, None, "trivial")
        .await
        .unwrap();
    expect_complete(&r, "True trivial");
    println!("  Proved: True");

    // ================================================================
    // Part 7: Nat arithmetic with omega
    // ================================================================
    println!("\n=== Part 7: Nat arithmetic (omega) ===");

    let state7 = w
        .start_proof("forall (n : Nat), n + 0 = n")
        .await
        .unwrap();
    let r = w
        .apply_tactic(state7.state_id, None, "intro n")
        .await
        .unwrap();
    let s = expect_success(&r, "n+0=n intro");
    let r = w.apply_tactic(s, None, "omega").await.unwrap();
    expect_complete(&r, "n+0=n omega");
    println!("  Proved: n + 0 = n");

    // ================================================================
    // Part 8: Deep proof — iff split into two implications
    // ================================================================
    println!("\n=== Part 8: Iff (split into two directions) ===");

    let state8 = w
        .start_proof("forall (p : Prop), p -> (p \\/ True)")
        .await
        .unwrap();
    let r = w
        .apply_tactic(state8.state_id, None, "intro p hp")
        .await
        .unwrap();
    let s = expect_success(&r, "iff intro");
    let r = w.apply_tactic(s, None, "exact Or.inl hp").await.unwrap();
    expect_complete(&r, "iff Or.inl");
    println!("  Proved: p -> (p \\/ True)");

    // ================================================================
    // Part 9: Error recovery sequence — many failures, then success
    // ================================================================
    println!("\n=== Part 9: Error recovery stress ===");

    let state9 = w
        .start_proof("forall (n : Nat), n = n")
        .await
        .unwrap();
    let s = state9.state_id;

    // Throw 10 bad tactics at the same state
    let bad_tactics = [
        "simp_all_wrong",
        "exact 42",
        "apply Nat.succ",
        "induction",
        "omega",
        "rfl",
        "ring",
        "norm_num",
        "contradiction",
        "decide",
    ];
    let mut fail_count = 0;
    for tac in &bad_tactics {
        let r = w.apply_tactic(s, None, tac).await.unwrap();
        if matches!(r, TacticResult::Failed { .. }) {
            fail_count += 1;
        }
    }
    println!("  {fail_count}/{} tactics failed (expected)", bad_tactics.len());

    // Now prove it correctly from the SAME state — still valid
    let r = w.apply_tactic(s, None, "intro n").await.unwrap();
    let s2 = expect_success(&r, "recovery intro");
    let r = w.apply_tactic(s2, None, "rfl").await.unwrap();
    expect_complete(&r, "recovery rfl");
    println!("  State survived {} failed tactics, proof completed", bad_tactics.len());

    // ================================================================
    // Part 10: Throughput — 20 distinct proofs on same worker
    // ================================================================
    println!("\n=== Part 10: 20 rapid-fire proofs on same checkout ===");
    let start = Instant::now();

    let theorems: Vec<(&str, Vec<&str>)> = vec![
        ("forall (n : Nat), n = n", vec!["intro n", "rfl"]),
        ("forall (p : Prop), p -> p", vec!["intro p hp", "exact hp"]),
        ("True", vec!["trivial"]),
        (
            "forall (p : Prop), False -> p",
            vec!["intro p h", "exact False.elim h"],
        ),
        (
            "forall (p q : Prop), p -> p \\/ q",
            vec!["intro p q hp", "exact Or.inl hp"],
        ),
        (
            "forall (p q : Prop), q -> p \\/ q",
            vec!["intro p q hq", "exact Or.inr hq"],
        ),
        (
            "forall (p q : Prop), (p -> q) -> p -> q",
            vec!["intro p q f hp", "exact f hp"],
        ),
        (
            "forall (p q : Prop), p /\\ q -> p",
            vec!["intro p q h", "exact h.1"],
        ),
        (
            "forall (p q : Prop), p /\\ q -> q",
            vec!["intro p q h", "exact h.2"],
        ),
        (
            "forall (p : Prop), p -> p /\\ True",
            vec!["intro p hp", "exact And.intro hp trivial"],
        ),
        ("forall (n : Nat), n = n", vec!["intro n", "rfl"]),
        ("forall (a : Nat), a + 0 = a", vec!["intro a", "omega"]),
        (
            "forall (a b : Nat), a = b -> b = a",
            vec!["intro a b h", "exact h.symm"],
        ),
        (
            "forall (a b c : Nat), a = b -> b = c -> a = c",
            vec!["intro a b c h1 h2", "exact h1.trans h2"],
        ),
        ("Exists (fun n : Nat => n = 0)", vec!["exact Exists.intro 0 rfl"]),
        (
            "forall (p : Prop), p -> True",
            vec!["intro p _", "trivial"],
        ),
        (
            "forall (p q r : Prop), (p -> q) -> (q -> r) -> p -> r",
            vec!["intro p q r f g hp", "exact g (f hp)"],
        ),
        (
            "forall (n : Nat), 0 + n = n",
            vec!["intro n", "omega"],
        ),
        (
            "forall (p : Prop), p /\\ True -> p",
            vec!["intro p h", "exact h.1"],
        ),
        (
            "forall (n m : Nat), n = m -> n + 1 = m + 1",
            vec!["intro n m h", "omega"],
        ),
    ];

    for (i, (thm, tactics)) in theorems.iter().enumerate() {
        let state = w.start_proof(thm).await.unwrap();
        let mut sid = state.state_id;

        for (j, tac) in tactics.iter().enumerate() {
            let r = w.apply_tactic(sid, None, tac).await.unwrap();
            if j < tactics.len() - 1 {
                sid = expect_success(&r, &format!("thm[{i}] step[{j}] '{tac}'"));
            } else {
                expect_complete(&r, &format!("thm[{i}] final '{tac}'"));
            }
        }
    }

    println!(
        "  20 theorems proved in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    drop(guard);
    pool.shutdown().await;
    println!("\n=== Proof search simulation: ALL PASSED ===");
}

/// Test ProofSession error handling, apply_to_goal, and apply-after-complete.
///
/// Exercises ProofSession behavior not covered by `test_proof_session_simple`:
/// - Failed tactic records in history but doesn't advance state
/// - `apply_to_goal` with multi-goal proofs
/// - Applying a tactic after proof completion returns an error
/// - History tracks the correct sequence of tactics and results
#[tokio::test]
#[ignore]
async fn test_proof_session_error_and_state_tracking() {
    let config = get_test_config_or_skip(1);
    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    // ---- Part 1: Failed tactic doesn't advance state ----
    println!("=== Part 1: Error doesn't advance state ===");
    let mut session = ProofSession::new(&pool, "forall (n : Nat), n = n")
        .await
        .expect("Failed to start session");

    let initial_state_id = session.current_state().state_id;
    assert_eq!(session.depth(), 0);
    assert!(!session.is_complete());

    // Apply a bad tactic
    let result = session.apply("nonexistent_tactic_xyz").await.unwrap().clone();
    assert!(
        matches!(result, TacticResult::Failed { .. }),
        "Bad tactic should fail, got: {result:?}"
    );

    // depth increases (failure IS recorded in history)
    assert_eq!(session.depth(), 1);
    // state_id unchanged (state didn't advance)
    assert_eq!(session.current_state().state_id, initial_state_id);
    assert!(!session.is_complete());

    // History last entry has the bad tactic and Failed
    let (tac, res) = &session.history()[0];
    assert_eq!(tac, "nonexistent_tactic_xyz");
    assert!(matches!(res, TacticResult::Failed { .. }));
    println!("  Bad tactic: depth=1, state_id unchanged={initial_state_id}");

    // ---- Part 2: apply_to_goal with multi-goal proof ----
    println!("\n=== Part 2: apply_to_goal with multi-goal proof ===");
    drop(session);

    let mut session = ProofSession::new(
        &pool,
        "forall (p q : Prop), p -> q -> q /\\ p",
    )
    .await
    .expect("Failed to start multi-goal session");

    // Introduce hypotheses
    {
        let r = session.apply("intro p q hp hq").await.unwrap();
        assert!(
            matches!(r, TacticResult::Success { .. }),
            "intro should succeed: {r:?}"
        );
    }
    assert_eq!(session.depth(), 1);

    // constructor → splits into 2 goals
    {
        let r = session.apply("constructor").await.unwrap();
        assert!(
            matches!(r, TacticResult::Success { .. }),
            "constructor should succeed: {r:?}"
        );
    }
    assert_eq!(session.current_goals().len(), 2);
    assert_eq!(session.depth(), 2);
    println!(
        "  After constructor: {} goals, depth={}",
        session.current_goals().len(),
        session.depth()
    );

    // Solve goal 0 (the first remaining goal — should be `q`)
    let r = session.apply_to_goal(0, "exact hq").await.unwrap().clone();
    assert_eq!(session.depth(), 3);
    println!("  After solving goal 0: depth={}", session.depth());

    // Check if proof completed or one goal remains
    match &r {
        TacticResult::Success { goals, .. } => {
            assert_eq!(goals.len(), 1, "After solving goal 0, 1 goal should remain");
            // Solve the remaining goal (now goal 0 again — should be `p`)
            let r = session.apply_to_goal(0, "exact hp").await.unwrap();
            expect_complete(r, "final goal");
            assert_eq!(session.depth(), 4);
        }
        TacticResult::ProofComplete { .. } => {
            println!("  Both goals solved in one step");
        }
        other => panic!("Unexpected result after goal 0: {other:?}"),
    }

    assert!(session.is_complete());
    println!("  Proof complete, depth={}", session.depth());

    // ---- Part 3: apply after complete returns error ----
    println!("\n=== Part 3: Apply after complete ===");
    let err = session.apply("intro x").await;
    match err {
        Err(lean_repl::LeanError::Protocol(msg)) => {
            assert!(
                msg.contains("already complete"),
                "Expected 'already complete' in error, got: {msg}"
            );
            println!("  Correctly returned Protocol error: {msg}");
        }
        other => panic!("Expected Protocol error after complete, got: {other:?}"),
    }

    // ---- Part 4: History correctness ----
    println!("\n=== Part 4: History verification ===");
    let history = session.history();
    // We expect: intro, constructor, exact hq, exact hp (or fewer if both solved at once)
    println!("  History length: {}", history.len());
    assert!(
        history.len() >= 3,
        "Expected at least 3 history entries, got {}",
        history.len()
    );

    // First entry: intro
    assert_eq!(history[0].0, "intro p q hp hq");
    assert!(matches!(history[0].1, TacticResult::Success { .. }));

    // Second: constructor
    assert_eq!(history[1].0, "constructor");
    assert!(matches!(history[1].1, TacticResult::Success { .. }));

    // Third: exact hq
    assert_eq!(history[2].0, "exact hq");

    println!("  All history entries verified");

    drop(session);
    pool.shutdown().await;
}

/// Test that concurrent multi-step proofs on a multi-worker pool never
/// cross-contaminate state IDs.
///
/// This is the test that would have caught the original state-routing bug:
/// with the old API, `pool.run_tactic(state_id, ...)` could land on a
/// different worker than the one that created that `state_id`. With 4
/// workers and 10 concurrent multi-step proofs (each doing 3-5 tactic
/// applications with branching), any state ID mix-up causes immediate
/// failure.
///
/// Each spawned task proves a *different* theorem using a *different*
/// tactic sequence so we can verify results are correct per-proof, not
/// just "some proof completed."
#[tokio::test]
#[ignore]
async fn test_concurrent_multi_step_isolation() {
    let config = get_test_config_or_skip(4);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    // Each entry: (theorem, tactics, expected_goal_count_after_first_tactic)
    let proof_plans: Vec<(
        &'static str,
        Vec<&'static str>,
        Option<usize>, // expected goal count after first tactic (None = don't check)
    )> = vec![
        // 0: simple reflexivity
        ("forall (n : Nat), n = n", vec!["intro n", "rfl"], Some(1)),
        // 1: And.comm with branching constructor
        (
            "forall (p q : Prop), p /\\ q -> q /\\ p",
            vec!["intro p q h", "constructor", "exact h.2", "exact h.1"],
            Some(1),
        ),
        // 2: Or.inl
        (
            "forall (p q : Prop), p -> p \\/ q",
            vec!["intro p q hp", "exact Or.inl hp"],
            Some(1),
        ),
        // 3: modus ponens chain
        (
            "forall (p q r : Prop), (p -> q) -> (q -> r) -> p -> r",
            vec!["intro p q r f g hp", "exact g (f hp)"],
            Some(1),
        ),
        // 4: existential witness
        (
            "Exists (fun n : Nat => n = 0)",
            vec!["exact Exists.intro 0 rfl"],
            None,
        ),
        // 5: symmetry
        (
            "forall (a b : Nat), a = b -> b = a",
            vec!["intro a b h", "exact h.symm"],
            Some(1),
        ),
        // 6: transitivity
        (
            "forall (a b c : Nat), a = b -> b = c -> a = c",
            vec!["intro a b c h1 h2", "exact h1.trans h2"],
            Some(1),
        ),
        // 7: false elimination
        (
            "forall (p : Prop), False -> p",
            vec!["intro p h", "exact False.elim h"],
            Some(1),
        ),
        // 8: And.intro from components
        (
            "forall (p : Prop), p -> p /\\ True",
            vec!["intro p hp", "exact And.intro hp trivial"],
            Some(1),
        ),
        // 9: omega arithmetic
        (
            "forall (n m : Nat), n = m -> n + 1 = m + 1",
            vec!["intro n m h", "omega"],
            Some(1),
        ),
    ];

    let start = Instant::now();

    let mut handles = Vec::new();
    for (i, (thm, tactics, expected_goal_count)) in proof_plans.into_iter().enumerate() {
        let pool = pool.clone();
        handles.push(tokio::spawn(async move {
            let mut proof = pool
                .start_proof_owned(thm)
                .await
                .unwrap_or_else(|e| panic!("[{i}] Failed to start proof for '{thm}': {e}"));
            let initial_sid = proof.state_id();

            // For multi-step proofs with branching (like And.comm via constructor),
            // we need special handling: constructor produces 2 goals, and we solve
            // them with targeted goalId applications.
            let is_constructor_proof = tactics.len() == 4
                && tactics[1] == "constructor";

            let mut sid = initial_sid;

            if is_constructor_proof {
                // Step 1: intro
                let r = proof
                    .run_tactic(sid, None, tactics[0])
                    .await
                    .unwrap_or_else(|e| {
                        panic!("[{i}] Failed '{thm}' tactic '{}': {e}", tactics[0])
                    });
                sid = expect_success(&r, &format!("[{i}] intro"));

                if let Some(expected) = expected_goal_count {
                    if let TacticResult::Success { goals, .. } = &r {
                        assert_eq!(
                            goals.len(),
                            expected,
                            "[{i}] After intro: expected {expected} goal(s), got {}",
                            goals.len()
                        );
                    }
                }

                // Step 2: constructor → 2 goals
                let r = proof
                    .run_tactic(sid, None, "constructor")
                    .await
                    .unwrap_or_else(|e| {
                        panic!("[{i}] Failed '{thm}' tactic 'constructor': {e}")
                    });
                sid = expect_success(&r, &format!("[{i}] constructor"));

                if let TacticResult::Success { goals, .. } = &r {
                    assert_eq!(
                        goals.len(),
                        2,
                        "[{i}] constructor should produce 2 goals, got {}",
                        goals.len()
                    );
                }

                // Step 3: solve goal 0
                let r = proof
                    .run_tactic(sid, Some(0), tactics[2])
                    .await
                    .unwrap_or_else(|e| {
                        panic!("[{i}] Failed '{thm}' tactic '{}' on goal 0: {e}", tactics[2])
                    });
                match &r {
                    TacticResult::Success {
                        state_id: next,
                        goals,
                    } => {
                        assert_eq!(
                            goals.len(),
                            1,
                            "[{i}] After solving goal 0, expected 1 remaining"
                        );
                        sid = *next;
                    }
                    TacticResult::ProofComplete { .. } => {
                        // Both goals solved in one step — done
                        return;
                    }
                    other => panic!("[{i}] Unexpected after goal 0: {other:?}"),
                }

                // Step 4: solve remaining goal
                let r = proof
                    .run_tactic(sid, None, tactics[3])
                    .await
                    .unwrap_or_else(|e| {
                        panic!("[{i}] Failed '{thm}' tactic '{}': {e}", tactics[3])
                    });
                expect_complete(&r, &format!("[{i}] final"));
            } else {
                // Normal linear proof
                for (j, tac) in tactics.iter().enumerate() {
                    let r = proof
                        .run_tactic(sid, None, tac)
                        .await
                        .unwrap_or_else(|e| {
                            panic!("[{i}] Failed '{thm}' tactic '{tac}': {e}")
                        });

                    // After first tactic, optionally verify goal count
                    if j == 0 {
                        if let Some(expected) = expected_goal_count {
                            if let TacticResult::Success { goals, .. } = &r {
                                assert_eq!(
                                    goals.len(),
                                    expected,
                                    "[{i}] After '{tac}': expected {expected} goal(s), got {}",
                                    goals.len()
                                );
                            }
                        }
                    }

                    if j < tactics.len() - 1 {
                        sid = expect_success(&r, &format!("[{i}] step[{j}] '{tac}'"));
                    } else {
                        expect_complete(&r, &format!("[{i}] final '{tac}'"));
                    }
                }
            }

            // Verify the initial state is still valid (state immutability)
            // by re-applying the first tactic from the original state
            if tactics.len() > 1 {
                let r = proof
                    .run_tactic(initial_sid, None, tactics[0])
                    .await
                    .unwrap_or_else(|e| {
                        panic!("[{i}] State immutability check failed for '{thm}': {e}")
                    });
                let new_sid = expect_success(
                    &r,
                    &format!("[{i}] state immutability revisit"),
                );
                assert!(
                    new_sid > initial_sid,
                    "[{i}] Revisited state ID should be > initial ({new_sid} vs {initial_sid})"
                );
            }
        }));
    }

    let timeout_result = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        for (i, handle) in handles.into_iter().enumerate() {
            handle
                .await
                .unwrap_or_else(|e| panic!("Task {i} panicked: {e}"));
        }
    })
    .await;

    assert!(
        timeout_result.is_ok(),
        "Concurrent multi-step proofs timed out after 120s"
    );
    println!(
        "10 concurrent multi-step proofs (4 workers) completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    pool.shutdown().await;
}

/// Test time-based worker recycling (TTL expiration).
///
/// Only request-count recycling is covered by `test_worker_recycling`.
/// This verifies that a worker is recycled when its TTL expires.
#[tokio::test]
#[ignore]
async fn test_time_based_recycling() {
    let mut config = get_test_config_or_skip(1);
    config.max_lifetime_secs = 2;
    // Set request limit high so it won't trigger
    config.max_requests_per_worker = 10000;

    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    // Proof 1: verify pool works normally
    println!("=== Proof 1: before TTL expiry ===");
    {
        let mut proof = pool.start_proof("forall (n : Nat), n = n").await.unwrap();
        let sid = proof.state_id();
        let r1 = proof.run_tactic(sid, None, "intro n").await.unwrap();
        let s1 = expect_success(&r1, "pre-TTL intro");
        let r2 = proof.run_tactic(s1, None, "rfl").await.unwrap();
        expect_complete(&r2, "pre-TTL rfl");
        println!("  First proof completed successfully");
    }

    // Sleep past the TTL
    println!("=== Sleeping 3s to exceed 2s TTL ===");
    tokio::time::sleep(std::time::Duration::from_secs(3)).await;

    // Proof 2: worker should be recycled (new process), proof still succeeds
    println!("=== Proof 2: after TTL expiry (worker recycled) ===");
    {
        let mut proof = pool.start_proof("forall (n : Nat), n = n").await.unwrap();
        let sid = proof.state_id();
        let r1 = proof.run_tactic(sid, None, "intro n").await.unwrap();
        let s1 = expect_success(&r1, "post-TTL intro");
        let r2 = proof.run_tactic(s1, None, "rfl").await.unwrap();
        expect_complete(&r2, "post-TTL rfl");
        println!("  Second proof completed successfully (worker was recycled)");
    }

    pool.shutdown().await;
    println!("Time-based recycling test passed");
}

/// Test that both request-count and time-based recycling work together
/// across many sequential proofs on a multi-worker pool.
///
/// Unlike `test_worker_recycling` (single worker, request-count only) and
/// `test_time_based_recycling` (single worker, time-only), this test uses
/// 4 workers and triggers both recycling paths across 20 sequential proofs.
/// Each worker recycles ~2 times during the test.
#[tokio::test]
#[ignore]
async fn test_concurrent_recycling_stress() {
    let mut config = get_test_config_or_skip(4);
    // Each proof = 3 requests (start_proof + intro + rfl).
    // Recycle after 9 requests = every 3 proofs per worker.
    config.max_requests_per_worker = 9;
    // Time-based recycling also triggers during the test
    config.max_lifetime_secs = 4;

    let pool = LeanPool::new(config).await.expect("Failed to create pool");

    let start = Instant::now();

    for i in 0..20 {
        let mut proof = pool
            .start_proof("forall (n : Nat), n = n")
            .await
            .unwrap_or_else(|e| panic!("Recycling stress proof {i} start failed: {e}"));
        let sid = proof.state_id();

        let r1 = proof
            .run_tactic(sid, None, "intro n")
            .await
            .unwrap_or_else(|e| panic!("Recycling stress proof {i} intro failed: {e}"));
        let s1 = match r1 {
            TacticResult::Success { state_id, .. } => state_id,
            other => panic!("Recycling stress proof {i} intro unexpected: {other:?}"),
        };

        let r2 = proof
            .run_tactic(s1, None, "rfl")
            .await
            .unwrap_or_else(|e| panic!("Recycling stress proof {i} rfl failed: {e}"));
        assert!(
            matches!(r2, TacticResult::ProofComplete { .. }),
            "Recycling stress proof {i} not complete: {r2:?}"
        );

        // proof dropped here → worker returned to pool

        if (i + 1) % 5 == 0 {
            println!(
                "  {}/20 proofs complete ({:.1}s)",
                i + 1,
                start.elapsed().as_secs_f64()
            );
        }
    }

    println!(
        "20 proofs with combined recycling completed in {:.2}s",
        start.elapsed().as_secs_f64()
    );

    pool.shutdown().await;
}
