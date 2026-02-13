//! Prompt formatting and tactic extraction utilities.
//!
//! These functions handle the translation between Lean 4 proof states and
//! the chat prompt format expected by DeepSeek-Prover-V2, as well as
//! extracting usable tactics from model output.

/// Format a proof state into a chat message for the model.
///
/// Uses tactic-state comment format matching DeepSeek-Prover-V1.5/V2 training data.
pub fn format_tactic_message(proof_state: &str) -> String {
    format!(
        "Complete the following Lean 4 code:\n\n\
         ```lean4\n\
         /- tactic state:\n\
         {proof_state}\n\
         -/\n\
         ```"
    )
}

/// Extract the first valid tactic from model output.
///
/// Handles:
/// - Raw tactic text (`"intro h"`)
/// - Code-fenced output (`` ```lean4\nintro h\n``` ``)
/// - Comment lines (`"-- We introduce h\nintro h"`)
/// - Full theorem declarations (`"theorem X : T := by\n  tactic"` -> `"tactic"`)
/// - Inline proof (`"theorem X := by trivial"` -> `"trivial"`)
pub fn extract_first_tactic(raw: &str) -> String {
    let text = raw.trim();
    // Strip code fence if present
    let text = if text.starts_with("```") {
        text.lines()
            .skip(1) // skip ```lean4
            .take_while(|l| !l.starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        text.to_string()
    };
    // Take the first non-empty, non-comment, non-declaration line
    text.lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty() && !l.starts_with("--") && !l.starts_with("/-"))
        .find_map(|l| {
            // Skip theorem/lemma/example declarations
            if l.starts_with("theorem ") || l.starts_with("lemma ") || l.starts_with("example ") {
                // Check for inline tactic after "by": "theorem X := by trivial"
                if let Some(by_pos) = l.rfind(" by ") {
                    let after_by = l[by_pos + 4..].trim();
                    if !after_by.is_empty() {
                        return Some(after_by.to_string());
                    }
                }
                // Declaration ends with "by" — tactic is on the next line
                None
            } else {
                Some(l.to_string())
            }
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_tactic_message() {
        let state = "n : Nat\n\u{22a2} n + 0 = n";
        let msg = format_tactic_message(state);
        assert!(msg.contains("Complete the following Lean 4 code:"));
        assert!(msg.contains("tactic state:"));
        assert!(msg.contains(state));
        assert!(msg.contains("```lean4"));
    }

    #[test]
    fn test_format_tactic_message_empty() {
        let msg = format_tactic_message("");
        assert!(msg.contains("tactic state:"));
    }

    #[test]
    fn test_extract_first_tactic_raw() {
        assert_eq!(extract_first_tactic("intro h"), "intro h");
        assert_eq!(extract_first_tactic("  simp  "), "simp");
    }

    #[test]
    fn test_extract_first_tactic_code_fence() {
        let raw = "```lean4\nintro h\nexact h\n```";
        assert_eq!(extract_first_tactic(raw), "intro h");
    }

    #[test]
    fn test_extract_first_tactic_with_comments() {
        let raw = "-- We introduce h\nintro h\nexact h";
        assert_eq!(extract_first_tactic(raw), "intro h");
    }

    #[test]
    fn test_extract_first_tactic_empty() {
        assert_eq!(extract_first_tactic(""), "");
        assert_eq!(extract_first_tactic("  \n  "), "");
    }

    #[test]
    fn test_extract_first_tactic_block_comment() {
        let raw = "/- some reasoning -/\nomega";
        assert_eq!(extract_first_tactic(raw), "omega");
    }

    #[test]
    fn test_extract_first_tactic_theorem_declaration() {
        let raw = "```lean4\ntheorem proof_of_true : True := by\n  trivial\n```";
        assert_eq!(extract_first_tactic(raw), "trivial");
    }

    #[test]
    fn test_extract_first_tactic_theorem_inline_by() {
        let raw = "theorem foo : True := by trivial";
        assert_eq!(extract_first_tactic(raw), "trivial");
    }

    #[test]
    fn test_extract_first_tactic_theorem_multi_step() {
        let raw = "```lean4\ntheorem and_comm : P \u{2227} Q \u{2192} Q \u{2227} P := by\n  intro \u{27e8}hp, hq\u{27e9}\n  exact \u{27e8}hq, hp\u{27e9}\n```";
        assert_eq!(extract_first_tactic(raw), "intro \u{27e8}hp, hq\u{27e9}");
    }

    #[test]
    fn test_extract_first_tactic_lemma_declaration() {
        let raw = "lemma foo : True := by\n  simp";
        assert_eq!(extract_first_tactic(raw), "simp");
    }

    #[test]
    fn test_extract_first_tactic_theorem_only_declaration() {
        let raw = "theorem foo : True := by";
        assert_eq!(extract_first_tactic(raw), "");
    }
}
