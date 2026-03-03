//! On-the-fly tactic rewriting for Lean version compatibility.
//!
//! DeepSeek-Prover-V2-7B was trained on Lean ~4.8 / Mathlib ~4.8 data.
//! When running against Lean 4.27 / Mathlib 4.27, many generated tactics
//! reference identifiers that have been renamed. This module intercepts
//! tactics before they reach Pantograph and applies known renames.
//!
//! The rewrite rules come from:
//! - Goedel migration (python/data/goedel_migration/fix_renames.py)
//! - Goedel migration (python/data/goedel_migration/fix_rewrites.py)
//! - Mathlib4 deprecation changelog

use std::borrow::Cow;

use regex::Regex;
use std::sync::LazyLock;

// ---------------------------------------------------------------------------
// Word-boundary-aware string replacement
// ---------------------------------------------------------------------------

/// Check if a char is a "word" character for Lean identifier boundaries.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Check if a char is a word char OR a Unicode subscript digit (₀₁₂…₉).
/// Used for the "after" boundary to avoid rewriting `le_div_iff₀` again.
fn is_word_or_subscript(c: char) -> bool {
    is_word_char(c) || ('₀'..='₉').contains(&c)
}

/// Replace `old` with `new` only at word boundaries, where the character
/// after the match is NOT a subscript digit (prevents double-renaming
/// of `le_div_iff₀`). Returns `None` if no replacement was made.
fn replace_word(text: &str, old: &str, new: &str) -> Option<String> {
    let mut result = String::with_capacity(text.len() + 16);
    let mut i = 0;
    let mut changed = false;

    while i < text.len() {
        if text[i..].starts_with(old) {
            // Check word boundary BEFORE
            let before_ok = if i == 0 {
                true
            } else {
                // Safe: we only enter this branch when i > 0
                !is_word_char(text[..i].chars().next_back().unwrap())
            };

            let after_pos = i + old.len();

            // Check word boundary AFTER — also reject subscript digits
            let after_ok = if after_pos >= text.len() {
                true
            } else {
                !is_word_or_subscript(text[after_pos..].chars().next().unwrap())
            };

            if before_ok && after_ok {
                result.push_str(new);
                i = after_pos;
                changed = true;
                continue;
            }
        }

        let c = text[i..].chars().next().unwrap();
        result.push(c);
        i += c.len_utf8();
    }

    if changed {
        Some(result)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Rename tables
// ---------------------------------------------------------------------------

/// (old_name, new_name) pairs for word-boundary replacement.
/// From Goedel migration: fix_renames.py + fix_rewrites.py + Mathlib changelog.
const WORD_RENAMES: &[(&str, &str)] = &[
    // Division lemma renames (₀ suffix added in Mathlib 4.27)
    ("le_div_iff", "le_div_iff₀"),
    ("div_le_div_iff", "div_le_div_iff₀"),
    ("div_le_iff", "div_le_iff₀"),
    ("div_lt_div_iff", "div_lt_div_iff₀"),
    ("div_lt_iff", "div_lt_iff₀"),
    ("lt_div_iff", "lt_div_iff₀"),
    // Ring/group lemma renames
    ("add_left_neg", "neg_add_cancel"),
    ("add_right_neg", "add_neg_cancel"),
    // Logic renames
    ("true_and_iff", "true_and"),
    ("and_true_iff", "and_true"),
    ("false_and_iff", "false_and"),
    ("and_false_iff", "and_false"),
    ("true_or_iff", "true_or"),
    ("or_true_iff", "or_true"),
    ("false_or_iff", "false_or"),
    ("or_false_iff", "or_false"),
];

/// Regex-based pattern replacements (for notation changes).
static PATTERN_RULES: LazyLock<Vec<(Regex, &'static str)>> = LazyLock::new(|| {
    vec![
        // BigOperator notation: ∑ x in S → ∑ x ∈ S
        (Regex::new(r"(∑\s+\w+)\s+in\s+").unwrap(), "${1} ∈ "),
        (Regex::new(r"(∏\s+\w+)\s+in\s+").unwrap(), "${1} ∈ "),
    ]
});

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Rewrite a tactic string, applying all known Mathlib v4.8→v4.27 renames.
///
/// Returns `Cow::Borrowed` if no rules matched (zero allocation in the
/// common case). Returns `Cow::Owned` with the rewritten string otherwise.
pub fn rewrite_tactic(tactic: &str) -> Cow<'_, str> {
    let mut owned: Option<String> = None;

    // Phase 1: word-boundary renames
    for &(old, new) in WORD_RENAMES {
        let text = owned.as_deref().unwrap_or(tactic);
        if let Some(replaced) = replace_word(text, old, new) {
            owned = Some(replaced);
        }
    }

    // Phase 2: regex pattern replacements
    for (re, replacement) in PATTERN_RULES.iter() {
        let text = owned.as_deref().unwrap_or(tactic);
        if re.is_match(text) {
            let replaced = re.replace_all(text, *replacement);
            owned = Some(replaced.into_owned());
        }
    }

    match owned {
        Some(s) => Cow::Owned(s),
        None => Cow::Borrowed(tactic),
    }
}

/// Extract lemma-like identifiers from a tactic string.
///
/// Returns a comma-separated list of identifiers that look like Mathlib
/// lemma references (contain a dot or start with uppercase). Used to build
/// the version-mismatch translation vocabulary from failed tactics.
pub fn extract_lemma_names(tactic: &str) -> String {
    static LEMMA_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"\b([A-Z]\w*(?:\.\w+)+|\b\w+\.\w+(?:\.\w+)*)").unwrap());

    let mut seen = std::collections::HashSet::new();
    let mut names = Vec::new();
    for cap in LEMMA_RE.captures_iter(tactic) {
        let name = cap.get(0).unwrap().as_str();
        if seen.insert(name) {
            names.push(name);
        }
    }
    names.join(",")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- replace_word unit tests ---

    #[test]
    fn test_replace_word_basic() {
        assert_eq!(
            replace_word("exact le_div_iff h", "le_div_iff", "le_div_iff₀"),
            Some("exact le_div_iff₀ h".into())
        );
    }

    #[test]
    fn test_replace_word_no_match_substring() {
        // le_div_iff_of_pos: underscore after "iff" is a word char → no boundary
        assert_eq!(
            replace_word("exact le_div_iff_of_pos h", "le_div_iff", "le_div_iff₀"),
            None
        );
    }

    #[test]
    fn test_replace_word_no_double_rename() {
        // Already has ₀ suffix → subscript after match → skip
        assert_eq!(
            replace_word("exact le_div_iff₀ h", "le_div_iff", "le_div_iff₀"),
            None
        );
    }

    #[test]
    fn test_replace_word_end_of_string() {
        assert_eq!(
            replace_word("rw [le_div_iff]", "le_div_iff", "le_div_iff₀"),
            Some("rw [le_div_iff₀]".into())
        );
    }

    #[test]
    fn test_replace_word_start_of_string() {
        assert_eq!(
            replace_word("le_div_iff h", "le_div_iff", "le_div_iff₀"),
            Some("le_div_iff₀ h".into())
        );
    }

    // --- rewrite_tactic integration tests ---

    #[test]
    fn test_no_rewrite_passthrough() {
        let tactic = "intro h";
        let result = rewrite_tactic(tactic);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "intro h");
    }

    #[test]
    fn test_le_div_iff_rename() {
        assert_eq!(rewrite_tactic("exact le_div_iff h"), "exact le_div_iff₀ h");
        assert_eq!(
            rewrite_tactic("rw [le_div_iff h, mul_comm]"),
            "rw [le_div_iff₀ h, mul_comm]"
        );
    }

    #[test]
    fn test_le_div_iff_already_renamed() {
        let result = rewrite_tactic("exact le_div_iff₀ h");
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "exact le_div_iff₀ h");
    }

    #[test]
    fn test_div_le_iff_rename() {
        assert_eq!(rewrite_tactic("exact div_le_iff h"), "exact div_le_iff₀ h");
        assert_eq!(
            rewrite_tactic("exact div_le_div_iff ha hb"),
            "exact div_le_div_iff₀ ha hb"
        );
    }

    #[test]
    fn test_div_lt_renames() {
        assert_eq!(rewrite_tactic("rw [div_lt_iff h]"), "rw [div_lt_iff₀ h]");
        assert_eq!(rewrite_tactic("rw [lt_div_iff h]"), "rw [lt_div_iff₀ h]");
        assert_eq!(
            rewrite_tactic("rw [div_lt_div_iff ha hb]"),
            "rw [div_lt_div_iff₀ ha hb]"
        );
    }

    #[test]
    fn test_add_neg_renames() {
        assert_eq!(
            rewrite_tactic("rw [add_left_neg]"),
            "rw [neg_add_cancel]"
        );
        assert_eq!(
            rewrite_tactic("simp [add_right_neg]"),
            "simp [add_neg_cancel]"
        );
    }

    #[test]
    fn test_and_iff_renames() {
        assert_eq!(rewrite_tactic("simp [true_and_iff]"), "simp [true_and]");
        assert_eq!(rewrite_tactic("rw [and_true_iff]"), "rw [and_true]");
    }

    #[test]
    fn test_bigoperator_sum_in() {
        assert_eq!(
            rewrite_tactic("have h := ∑ i in Finset.range n, f i"),
            "have h := ∑ i ∈ Finset.range n, f i"
        );
    }

    #[test]
    fn test_bigoperator_prod_in() {
        assert_eq!(
            rewrite_tactic("have h := ∏ k in s, g k"),
            "have h := ∏ k ∈ s, g k"
        );
    }

    #[test]
    fn test_multiple_rewrites_one_tactic() {
        assert_eq!(
            rewrite_tactic("simp only [le_div_iff h, add_left_neg, true_and_iff]"),
            "simp only [le_div_iff₀ h, neg_add_cancel, true_and]"
        );
    }

    #[test]
    fn test_no_false_positive_on_longer_name() {
        // le_div_iff_of_pos should NOT be rewritten
        assert_eq!(
            rewrite_tactic("exact le_div_iff_of_pos h"),
            "exact le_div_iff_of_pos h"
        );
    }

    // --- extract_lemma_names tests ---

    #[test]
    fn test_extract_lemma_names_qualified() {
        let names = extract_lemma_names("rw [Nat.add_comm, Finset.sum_bij]");
        assert!(names.contains("Nat.add_comm"));
        assert!(names.contains("Finset.sum_bij"));
    }

    #[test]
    fn test_extract_lemma_names_empty_for_simple_tactic() {
        assert_eq!(extract_lemma_names("intro h"), "");
        assert_eq!(extract_lemma_names("simp"), "");
    }

    #[test]
    fn test_extract_lemma_names_dedup() {
        let names = extract_lemma_names("simp [Nat.add_comm, Nat.add_comm]");
        assert_eq!(names.split(',').count(), 1);
    }

    // --- Known translation cases from Goedel migration & Putnam ---

    #[test]
    fn test_goedel_division_rewrites() {
        // From fix_renames.py: these are the most common failures
        assert_eq!(
            rewrite_tactic("rw [le_div_iff (by positivity)]"),
            "rw [le_div_iff₀ (by positivity)]"
        );
        assert_eq!(
            rewrite_tactic("rw [div_le_div_iff (by positivity) (by positivity)]"),
            "rw [div_le_div_iff₀ (by positivity) (by positivity)]"
        );
    }

    #[test]
    fn test_goedel_lt_division_rewrites() {
        // From fix_rewrites.py: lt variants
        assert_eq!(
            rewrite_tactic("rw [div_lt_div_iff (by nlinarith) (by nlinarith)]"),
            "rw [div_lt_div_iff₀ (by nlinarith) (by nlinarith)]"
        );
        assert_eq!(
            rewrite_tactic("rw [lt_div_iff (by positivity)]"),
            "rw [lt_div_iff₀ (by positivity)]"
        );
        assert_eq!(
            rewrite_tactic("rw [div_lt_iff (by positivity)]"),
            "rw [div_lt_iff₀ (by positivity)]"
        );
    }

    #[test]
    fn test_goedel_neg_cancel_renames() {
        // From fix_renames.py
        assert_eq!(
            rewrite_tactic("simp only [add_left_neg, zero_mul]"),
            "simp only [neg_add_cancel, zero_mul]"
        );
        assert_eq!(
            rewrite_tactic("rw [add_right_neg]"),
            "rw [add_neg_cancel]"
        );
    }

    #[test]
    fn test_bigoperator_in_have_chain() {
        // DeepSeek generates have-chains with old BigOperator notation
        assert_eq!(
            rewrite_tactic(
                "have h₁ : ∑ i in Finset.range n, (f i) ^ 2 ≤ n := by sorry"
            ),
            "have h₁ : ∑ i ∈ Finset.range n, (f i) ^ 2 ≤ n := by sorry"
        );
    }

    #[test]
    fn test_simp_list_with_renamed_lemma() {
        // Typical: simp list references old lemma names
        assert_eq!(
            rewrite_tactic("simp only [div_le_iff, mul_comm, true_and_iff]"),
            "simp only [div_le_iff₀, mul_comm, true_and]"
        );
    }

    #[test]
    fn test_complex_rw_chain() {
        assert_eq!(
            rewrite_tactic("rw [div_le_div_iff (by positivity) (by positivity), ← sub_nonneg]"),
            "rw [div_le_div_iff₀ (by positivity) (by positivity), ← sub_nonneg]"
        );
    }

    #[test]
    fn test_common_tactics_unchanged() {
        for tactic in &[
            "simp",
            "ring",
            "omega",
            "norm_num",
            "linarith",
            "exact h",
            "apply mul_comm",
            "intro a b c",
            "constructor",
            "have h : n > 0 := by omega",
            "rw [Nat.add_comm, Nat.mul_comm]",
        ] {
            let result = rewrite_tactic(tactic);
            assert!(
                matches!(result, Cow::Borrowed(_)),
                "Unexpected rewrite of common tactic: {tactic}"
            );
        }
    }
}
