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

/// Extract the first complete tactic from model output.
///
/// With the `example := by\n  ` prompt prefix, the model generates tactic
/// code directly. This function:
/// 1. Strips code fences, leading comments, and focus dots
/// 2. Uses indentation to find the first complete tactic (may be multi-line)
/// 3. A new independent tactic starts when a non-empty, non-comment line
///    returns to the same or lesser indentation as the first tactic line
///
/// Examples:
/// - `"intro h\nexact h"` → `"intro h"` (two independent tactics)
/// - `"simp [a,\n  b, c]"` → `"simp [a,\n  b, c]"` (one multi-line tactic)
/// - `"have h := by\n  exact foo\nring"` → `"have h := by\n  exact foo"` (one tactic)
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

    let mut lines: Vec<&str> = text.lines().collect();

    // Drop leading empty/comment lines
    while let Some(first) = lines.first() {
        let trimmed = first.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with("/-") {
            lines.remove(0);
        } else {
            break;
        }
    }

    if lines.is_empty() {
        return String::new();
    }

    // Strip leading focus dot (· or ·) from first line if present
    let first = lines[0].trim();
    if let Some(rest) = first.strip_prefix("· ").or_else(|| first.strip_prefix("· ")) {
        lines[0] = rest;
    } else if first == "·" || first == "·" {
        lines.remove(0);
        // Re-drop leading empty lines after removing dot
        while let Some(f) = lines.first() {
            if f.trim().is_empty() { lines.remove(0); } else { break; }
        }
    }

    if lines.is_empty() {
        return String::new();
    }

    // Determine base indentation from first tactic line
    let base_indent = lines[0].len() - lines[0].trim_start().len();

    // Collect lines belonging to the first tactic:
    // continuation lines are indented more than base, or are comments
    let mut result_lines = vec![lines[0]];
    for line in &lines[1..] {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break; // blank line ends the tactic
        }
        if trimmed.starts_with("--") || trimmed.starts_with("/-") {
            // inline comment — include as part of current tactic
            result_lines.push(line);
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent > base_indent {
            // continuation of current tactic
            result_lines.push(line);
        } else {
            // same or lesser indent = new tactic, stop
            break;
        }
    }

    result_lines.join("\n").trim_end().to_string()
}

/// Extract all valid tactic lines from model output.
///
/// Like [`extract_first_tactic`] but returns all tactic lines, enabling
/// multi-tactic chain walking for negatives generation. Code fences,
/// comments, and theorem/lemma declarations are stripped.
pub fn extract_all_tactics(raw: &str) -> Vec<String> {
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

    let mut tactics = Vec::new();
    let mut past_declaration = false;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with("/-") {
            continue;
        }
        if trimmed.starts_with("theorem ")
            || trimmed.starts_with("lemma ")
            || trimmed.starts_with("example ")
        {
            // Check for inline tactic after "by"
            if let Some(by_pos) = trimmed.rfind(" by ") {
                let after_by = trimmed[by_pos + 4..].trim();
                if !after_by.is_empty() {
                    tactics.push(after_by.to_string());
                }
            }
            past_declaration = true;
            continue;
        }
        // Skip block comment closers
        if trimmed.starts_with("-/") {
            continue;
        }
        tactics.push(trimmed.to_string());
    }

    // If we saw a declaration but no tactics after it, return empty
    if past_declaration && tactics.is_empty() {
        return Vec::new();
    }

    tactics
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
    fn test_extract_first_tactic_multiline_continuation() {
        // Multi-line tactic with indented continuation
        let raw = "simp_all [lemma1, lemma2,\n  lemma3, lemma4]";
        assert_eq!(extract_first_tactic(raw), "simp_all [lemma1, lemma2,\n  lemma3, lemma4]");
    }

    #[test]
    fn test_extract_first_tactic_have_by() {
        // have ... := by includes indented body, stops at next tactic
        let raw = "have h := by\n  exact foo\nring";
        assert_eq!(extract_first_tactic(raw), "have h := by\n  exact foo");
    }

    #[test]
    fn test_extract_first_tactic_two_independent() {
        // Two independent tactics at same indent — take only first
        let raw = "intro h\nexact h";
        assert_eq!(extract_first_tactic(raw), "intro h");
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
    fn test_extract_first_tactic_induction_block() {
        // induction with match arms — all indented deeper
        let raw = "induction n with\n  | zero => rfl\n  | succ n ih => simp";
        assert_eq!(extract_first_tactic(raw), "induction n with\n  | zero => rfl\n  | succ n ih => simp");
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
    fn test_extract_first_tactic_focus_dot() {
        // Leading focus dot stripped, then first tactic only
        let raw = "\u{b7} intro h\nexact h";
        assert_eq!(extract_first_tactic(raw), "intro h");
    }

    #[test]
    fn test_extract_first_tactic_focus_dot_only() {
        let raw = "\u{b7}\nintro h";
        assert_eq!(extract_first_tactic(raw), "intro h");
    }

    #[test]
    fn test_extract_all_tactics_raw() {
        assert_eq!(extract_all_tactics("intro h"), vec!["intro h"]);
        assert_eq!(
            extract_all_tactics("intro h\nexact h"),
            vec!["intro h", "exact h"]
        );
    }

    #[test]
    fn test_extract_all_tactics_code_fence() {
        let raw = "```lean4\nintro h\nsimp\nring\n```";
        assert_eq!(
            extract_all_tactics(raw),
            vec!["intro h", "simp", "ring"]
        );
    }

    #[test]
    fn test_extract_all_tactics_declaration() {
        let raw = "```lean4\ntheorem and_comm : P \u{2227} Q \u{2192} Q \u{2227} P := by\n  intro \u{27e8}hp, hq\u{27e9}\n  exact \u{27e8}hq, hp\u{27e9}\n```";
        assert_eq!(
            extract_all_tactics(raw),
            vec!["intro \u{27e8}hp, hq\u{27e9}", "exact \u{27e8}hq, hp\u{27e9}"]
        );
    }

    #[test]
    fn test_extract_all_tactics_with_comments() {
        let raw = "-- reasoning\nintro n\n-- apply lemma\nsimp\nring";
        assert_eq!(
            extract_all_tactics(raw),
            vec!["intro n", "simp", "ring"]
        );
    }

    #[test]
    fn test_extract_all_tactics_inline_by() {
        let raw = "theorem foo : True := by trivial";
        assert_eq!(extract_all_tactics(raw), vec!["trivial"]);
    }

    #[test]
    fn test_extract_all_tactics_empty() {
        assert!(extract_all_tactics("").is_empty());
        assert!(extract_all_tactics("theorem foo : True := by").is_empty());
    }
}
