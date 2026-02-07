//! Integration tests for the search crate using real Lean (Pantograph) + MockPolicy.
//!
//! These tests exercise the adapter layer (`Arc<LeanPool>` as `ProofEnvironment`,
//! `ProofHandleOwned` as `TacticRunner`) with canned tactics from `MockPolicy`.
//! No LLM model is needed.
//!
//! Run with: `cargo test -p search -- --ignored --test-threads=1`

use std::sync::Arc;

use lean_repl::{LeanPool, LeanPoolConfig};
use search::mocks::{make_tactic, MockPolicy};
use search::{SearchConfig, SearchEngine};
use trajectory::{SearchResult, TrajectoryLabel};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn get_lean_config(num_workers: usize) -> LeanPoolConfig {
    let mut config = LeanPoolConfig::with_bundled_pantograph().expect(
        "Pantograph not found. Run ./scripts/setup_pantograph.sh or set \
         PANTOGRAPH_PROJECT=/path/to/Pantograph",
    );
    config.num_workers = num_workers;
    // Allow extra time for worker initialization (Lean startup is slow)
    config.tactic_timeout_secs = 60;
    config
}

fn expect_proved(result: &SearchResult) {
    assert!(result.proved, "Expected proof to be found");
    assert!(
        !result.proof_tactics.is_empty(),
        "Proved result should have non-empty proof_tactics"
    );
}

fn expect_not_proved(result: &SearchResult) {
    assert!(!result.proved, "Expected proof NOT to be found");
    assert!(
        result.proof_tactics.is_empty(),
        "Unproved result should have empty proof_tactics"
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// One-step proof: `True` proved by `trivial`.
/// Verifies adapter wiring, trajectory record structure.
#[tokio::test]
#[ignore]
async fn test_search_one_step_proof_via_lean() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let mut policy = MockPolicy::new();
    policy.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

    let engine = SearchEngine::new(SearchConfig::default());
    let result = engine
        .search_one(&pool, &policy, None, "true_test", "True")
        .await
        .expect("search_one failed");

    expect_proved(&result);
    assert_eq!(result.proof_tactics, vec!["trivial"]);

    // Trajectory: root + terminal = 2 records
    assert_eq!(result.all_records.len(), 2, "Expected 2 trajectory records (root + terminal)");

    let root = &result.all_records[0];
    assert!(root.parent_state_id.is_none(), "Root should have no parent");
    assert_eq!(root.tactic_applied, "", "Root tactic should be empty");
    assert_eq!(root.depth_from_root, 0);
    assert!(!root.is_proof_complete);

    let terminal = &result.all_records[1];
    assert!(terminal.is_proof_complete, "Terminal should be proof-complete");
    assert_eq!(terminal.tactic_applied, "trivial");
    assert_eq!(terminal.depth_from_root, 1);
}

/// Two-step proof: `∀ (n : Nat), n = n` proved by `intro n` then `rfl`.
/// Uses `add_contains_response` for the second step since Lean output formatting
/// may vary.
#[tokio::test]
#[ignore]
async fn test_search_two_step_proof_via_lean() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let mut policy = MockPolicy::new();
    policy.add_response(
        "⊢ ∀ (n : Nat), n = n",
        vec![make_tactic("intro n", -0.3)],
    );
    // Use contains match: real Lean output might be "n : Nat\n⊢ n = n" or similar
    policy.add_contains_response("n = n", vec![make_tactic("rfl", -0.1)]);

    let engine = SearchEngine::new(SearchConfig::default());
    let result = engine
        .search_one(&pool, &policy, None, "nat_refl", "∀ (n : Nat), n = n")
        .await
        .expect("search_one failed");

    expect_proved(&result);
    assert_eq!(result.proof_tactics, vec!["intro n", "rfl"]);

    // Trajectory: root + intro_n state + terminal = 3 records
    assert_eq!(result.all_records.len(), 3, "Expected 3 trajectory records");

    // Verify parent chain
    let root = &result.all_records[0];
    let middle = &result.all_records[1];
    let terminal = &result.all_records[2];

    assert!(root.parent_state_id.is_none());
    assert_eq!(middle.parent_state_id, Some(root.state_id));
    assert_eq!(terminal.parent_state_id, Some(middle.state_id));
}

/// Policy returns a bad tactic first, then a good one. Search should survive
/// the failure and find the proof.
#[tokio::test]
#[ignore]
async fn test_search_survives_tactic_failures() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let mut policy = MockPolicy::new();
    policy.add_response(
        "⊢ True",
        vec![
            make_tactic("nonexistent_tactic", -5.0),
            make_tactic("trivial", -0.1),
        ],
    );

    let engine = SearchEngine::new(SearchConfig::default());
    let result = engine
        .search_one(&pool, &policy, None, "resilient", "True")
        .await
        .expect("search_one failed");

    expect_proved(&result);
    assert_eq!(result.proof_tactics, vec!["trivial"]);
    // nodes_expanded may be 0: the engine finds the proof during the first
    // node's candidate loop but increments the counter *after* that loop,
    // so an early return on ProofComplete skips the increment.
    assert!(result.total_states >= 2, "Should have at least root + terminal states");
}

/// All tactics are invalid → search exhausts budget without finding proof.
#[tokio::test]
#[ignore]
async fn test_search_unproved_with_bad_tactics() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let policy = MockPolicy::with_default(vec![make_tactic("definitely_wrong", -1.0)]);

    let search_config = SearchConfig {
        max_nodes: 5,
        ..SearchConfig::default()
    };
    let engine = SearchEngine::new(search_config);
    let result = engine
        .search_one(&pool, &policy, None, "impossible", "True")
        .await
        .expect("search_one failed");

    expect_not_proved(&result);
    assert!(
        !result.all_records.is_empty(),
        "Should have at least the root record"
    );
}

/// Detailed trajectory record verification for a two-step proof.
/// Checks parent chain, depth, labels, and completeness flags.
#[tokio::test]
#[ignore]
async fn test_search_trajectory_records_from_real_proof() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let mut policy = MockPolicy::new();
    policy.add_response(
        "⊢ ∀ (n : Nat), n = n",
        vec![make_tactic("intro n", -0.3)],
    );
    policy.add_contains_response("n = n", vec![make_tactic("rfl", -0.1)]);

    let engine = SearchEngine::new(SearchConfig::default());
    let result = engine
        .search_one(&pool, &policy, None, "nat_refl", "∀ (n : Nat), n = n")
        .await
        .expect("search_one failed");

    expect_proved(&result);
    assert_eq!(result.all_records.len(), 3);

    // Root record
    let root = &result.all_records[0];
    assert_eq!(root.theorem_name, "nat_refl");
    assert!(root.parent_state_id.is_none());
    assert_eq!(root.tactic_applied, "");
    assert_eq!(root.depth_from_root, 0);
    assert!(!root.is_proof_complete);
    assert_eq!(root.label, TrajectoryLabel::Unknown);

    // Middle record (after "intro n")
    let middle = &result.all_records[1];
    assert_eq!(middle.theorem_name, "nat_refl");
    assert_eq!(middle.parent_state_id, Some(root.state_id));
    assert_eq!(middle.tactic_applied, "intro n");
    assert_eq!(middle.depth_from_root, 1);
    assert!(!middle.is_proof_complete);
    assert_eq!(middle.label, TrajectoryLabel::Unknown);
    // state_pp should contain the real Lean proof state
    assert!(
        middle.state_pp.contains("n"),
        "Middle state should mention 'n': got {:?}",
        middle.state_pp
    );

    // Terminal record (after "rfl")
    let terminal = &result.all_records[2];
    assert_eq!(terminal.theorem_name, "nat_refl");
    assert_eq!(terminal.parent_state_id, Some(middle.state_id));
    assert_eq!(terminal.tactic_applied, "rfl");
    assert_eq!(terminal.depth_from_root, 2);
    assert!(terminal.is_proof_complete);
    assert_eq!(terminal.label, TrajectoryLabel::Unknown);
}

/// Run two proofs concurrently on separate workers, verifying no state ID
/// cross-contamination.
#[tokio::test]
#[ignore]
async fn test_search_concurrent_two_proofs() {
    let config = get_lean_config(2);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    // Policy for "True"
    let mut policy1 = MockPolicy::new();
    policy1.add_response("⊢ True", vec![make_tactic("trivial", -0.1)]);

    // Policy for "∀ (n : Nat), n = n"
    let mut policy2 = MockPolicy::new();
    policy2.add_response(
        "⊢ ∀ (n : Nat), n = n",
        vec![make_tactic("intro n", -0.3)],
    );
    policy2.add_contains_response("n = n", vec![make_tactic("rfl", -0.1)]);

    let engine1 = SearchEngine::new(SearchConfig::default());
    let engine2 = SearchEngine::new(SearchConfig::default());

    let pool1 = Arc::clone(&pool);
    let pool2 = Arc::clone(&pool);

    let (result1, result2) = tokio::join!(
        async {
            engine1
                .search_one(&pool1, &policy1, None, "true_test", "True")
                .await
                .expect("search 1 failed")
        },
        async {
            engine2
                .search_one(&pool2, &policy2, None, "nat_refl", "∀ (n : Nat), n = n")
                .await
                .expect("search 2 failed")
        },
    );

    expect_proved(&result1);
    assert_eq!(result1.proof_tactics, vec!["trivial"]);

    expect_proved(&result2);
    assert_eq!(result2.proof_tactics, vec!["intro n", "rfl"]);

    // Verify independence: theorem names should be distinct
    assert_eq!(result1.theorem_name, "true_test");
    assert_eq!(result2.theorem_name, "nat_refl");
}

/// Set a short timeout with a high node budget and wrong tactics.
/// The search should exit within the timeout window, not run for minutes.
#[tokio::test]
#[ignore]
async fn test_search_timeout_exits_early() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    // 8 wrong tactics per state — each forces Lean to process and fail,
    // consuming wall-clock time.
    let policy = MockPolicy::with_default(vec![
        make_tactic("simp", -1.0),
        make_tactic("ring", -1.1),
        make_tactic("omega", -1.2),
        make_tactic("linarith", -1.3),
        make_tactic("norm_num", -1.4),
        make_tactic("decide", -1.5),
        make_tactic("assumption", -1.6),
        make_tactic("contradiction", -1.7),
    ]);

    let search_config = SearchConfig {
        timeout_per_theorem: 3,
        max_nodes: 1000, // high budget so timeout is the binding constraint
        ..SearchConfig::default()
    };
    let engine = SearchEngine::new(search_config);
    let result = engine
        .search_one(
            &pool,
            &policy,
            None,
            "timeout_test",
            "∀ (a b c : Nat), a + b + c = c + b + a",
        )
        .await
        .expect("search_one failed");

    expect_not_proved(&result);
    assert!(
        result.wall_time_ms >= 2000,
        "Search should have run for at least 2s, got {}ms",
        result.wall_time_ms
    );
    assert!(
        result.wall_time_ms < 15000,
        "Timeout should have kicked in before 15s, got {}ms",
        result.wall_time_ms
    );
    assert!(
        !result.all_records.is_empty(),
        "Should have at least the root record"
    );
}

/// Batch test of 5 arithmetic theorems, all 2-step (intro + omega).
/// Verifies that `omega` can close standard Nat arithmetic goals.
#[tokio::test]
#[ignore]
async fn test_search_medium_arithmetic_batch_via_omega() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let theorems = [
        ("nat_add_zero", "∀ (n : Nat), n + 0 = n", "intro n"),
        ("zero_add_nat", "∀ (n : Nat), 0 + n = n", "intro n"),
        ("nat_add_comm", "∀ (a b : Nat), a + b = b + a", "intro a b"),
        ("nat_mul_one", "∀ (n : Nat), n * 1 = n", "intro n"),
        ("nat_zero_le", "∀ (n : Nat), 0 ≤ n", "intro n"),
    ];

    for (name, statement, intro) in &theorems {
        let mut policy = MockPolicy::new();
        policy.add_response(
            &format!("⊢ {statement}"),
            vec![make_tactic(intro, -0.3)],
        );
        policy.add_contains_response("⊢", vec![make_tactic("omega", -0.5)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&pool, &policy, None, name, statement)
            .await
            .unwrap_or_else(|e| panic!("search_one failed for {name}: {e}"));

        expect_proved(&result);
        assert_eq!(
            result.proof_tactics,
            vec![intro.to_string(), "omega".to_string()],
            "Wrong proof tactics for {name}"
        );
        assert_eq!(
            result.all_records.len(),
            3,
            "Expected 3 trajectory records for {name} (root + intro + terminal)"
        );
        let terminal = result.all_records.last().unwrap();
        assert!(terminal.is_proof_complete, "Terminal should be proof-complete for {name}");
    }
}

/// 4-step multi-goal proof of `∀ (p q : Prop), p ∧ q → q ∧ p`.
/// Steps: intro p q h → constructor → exact h.2 → exact h.1
/// Exercises multi-goal states and goal_id routing.
#[tokio::test]
#[ignore]
async fn test_search_and_comm_multi_goal_four_step() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let mut policy = MockPolicy::new();
    // Step 1: exact match on root goal
    policy.add_response(
        "⊢ ∀ (p q : Prop), p ∧ q → q ∧ p",
        vec![make_tactic("intro p q h", -0.3)],
    );
    // Step 2: after intro, state contains "⊢ q ∧ p" — split into two goals
    // Note: "q ∧ p" appears in the goal but NOT as "q ∧ p" in hypothesis (hypothesis has "p ∧ q")
    policy.add_contains_response("q ∧ p", vec![make_tactic("constructor", -0.4)]);
    // Step 3: after constructor, first goal is "⊢ q" — exact h.2 closes it
    policy.add_contains_response("⊢ q", vec![make_tactic("exact h.2", -0.3)]);
    // Step 4: remaining goal is "⊢ p" — exact h.1 closes it
    policy.add_contains_response("⊢ p", vec![make_tactic("exact h.1", -0.3)]);

    let engine = SearchEngine::new(SearchConfig::default());
    let result = engine
        .search_one(
            &pool,
            &policy,
            None,
            "and_comm",
            "∀ (p q : Prop), p ∧ q → q ∧ p",
        )
        .await
        .expect("search_one failed");

    expect_proved(&result);
    assert_eq!(
        result.proof_tactics,
        vec!["intro p q h", "constructor", "exact h.2", "exact h.1"]
    );
    // root + 4 tactic applications = 5 records
    assert_eq!(result.all_records.len(), 5, "Expected 5 trajectory records");
    // Verify depth progression
    for (i, record) in result.all_records.iter().enumerate() {
        assert_eq!(
            record.depth_from_root, i as u32,
            "Record {i} should have depth {i}"
        );
    }
    assert_eq!(result.max_depth_reached, 4);
}

/// Batch test of 3 logic theorems using term-mode tactics (exact h.symm, etc.).
/// Each theorem is 2-step: intro + exact <term>.
#[tokio::test]
#[ignore]
async fn test_search_medium_logic_batch() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    // eq_symm: ∀ (a b : Nat), a = b → b = a
    {
        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ ∀ (a b : Nat), a = b → b = a",
            vec![make_tactic("intro a b h", -0.3)],
        );
        policy.add_contains_response("b = a", vec![make_tactic("exact h.symm", -0.3)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(&pool, &policy, None, "eq_symm", "∀ (a b : Nat), a = b → b = a")
            .await
            .expect("search_one failed for eq_symm");

        expect_proved(&result);
        assert_eq!(result.proof_tactics, vec!["intro a b h", "exact h.symm"]);
    }

    // eq_trans: ∀ (a b c : Nat), a = b → b = c → a = c
    {
        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ ∀ (a b c : Nat), a = b → b = c → a = c",
            vec![make_tactic("intro a b c h1 h2", -0.3)],
        );
        policy.add_contains_response("a = c", vec![make_tactic("exact h1.trans h2", -0.3)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(
                &pool,
                &policy,
                None,
                "eq_trans",
                "∀ (a b c : Nat), a = b → b = c → a = c",
            )
            .await
            .expect("search_one failed for eq_trans");

        expect_proved(&result);
        assert_eq!(
            result.proof_tactics,
            vec!["intro a b c h1 h2", "exact h1.trans h2"]
        );
    }

    // modus_ponens: ∀ (p q : Prop), (p → q) → p → q
    {
        let mut policy = MockPolicy::new();
        policy.add_response(
            "⊢ ∀ (p q : Prop), (p → q) → p → q",
            vec![make_tactic("intro p q f hp", -0.3)],
        );
        policy.add_contains_response("hp", vec![make_tactic("exact f hp", -0.3)]);

        let engine = SearchEngine::new(SearchConfig::default());
        let result = engine
            .search_one(
                &pool,
                &policy,
                None,
                "modus_ponens",
                "∀ (p q : Prop), (p → q) → p → q",
            )
            .await
            .expect("search_one failed for modus_ponens");

        expect_proved(&result);
        assert_eq!(result.proof_tactics, vec!["intro p q f hp", "exact f hp"]);
    }
}

/// Tests search resilience with backtracking: `∀ (a b c : Nat), a + b + c = c + b + a`
/// where MockPolicy returns wrong tactics before the correct one (omega).
#[tokio::test]
#[ignore]
async fn test_search_hard_arithmetic_with_backtracking() {
    let config = get_lean_config(1);
    let pool = Arc::new(LeanPool::new(config).await.expect("Failed to create pool"));

    let mut policy = MockPolicy::new();
    // Root: intro + rfl (rfl will fail on this non-trivial equality)
    policy.add_response(
        "⊢ ∀ (a b c : Nat), a + b + c = c + b + a",
        vec![
            make_tactic("intro a b c", -0.3),
            make_tactic("rfl", -5.0),
        ],
    );
    // After intro: rfl fails, then a bogus tactic, then omega succeeds
    policy.add_contains_response(
        "a + b + c",
        vec![
            make_tactic("rfl", -1.0),
            make_tactic("exact Nat.zero_le 0", -2.0),
            make_tactic("omega", -3.0),
        ],
    );
    // Fallback for any unexpected states
    policy.add_contains_response("⊢", vec![make_tactic("omega", -4.0)]);

    let engine = SearchEngine::new(SearchConfig::default());
    let result = engine
        .search_one(
            &pool,
            &policy,
            None,
            "nat_add_assoc_comm",
            "∀ (a b c : Nat), a + b + c = c + b + a",
        )
        .await
        .expect("search_one failed");

    expect_proved(&result);
    assert_eq!(result.proof_tactics[0], "intro a b c");
    assert!(
        result.stats.total_tactic_failures >= 1,
        "Should have at least 1 failed tactic, got {}",
        result.stats.total_tactic_failures
    );
    assert!(
        result.total_states >= 3,
        "Should have at least 3 states (root + intro + omega terminal), got {}",
        result.total_states
    );
}
