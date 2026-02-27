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
/// Takes the first line, then continues onto subsequent lines only if
/// brackets are unclosed (`[`, `(`, `{`). This handles multi-line
/// `simp [a,\n  b]` while stopping before independent tactics like
/// `intro h\nexact h`.
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
        while let Some(f) = lines.first() {
            if f.trim().is_empty() { lines.remove(0); } else { break; }
        }
    }

    if lines.is_empty() {
        return String::new();
    }

    // Take first line, then continue only while brackets are unclosed
    let mut result = lines[0].trim().to_string();
    let mut depth: i32 = bracket_depth(&result);

    for line in &lines[1..] {
        if depth <= 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        result.push('\n');
        result.push_str(trimmed);
        depth += bracket_depth(trimmed);
    }

    result.trim().to_string()
}

/// Count net bracket depth change for a line.
/// Positive = more opens than closes.
fn bracket_depth(s: &str) -> i32 {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut prev = '\0';
    for c in s.chars() {
        if c == '"' && prev != '\\' {
            in_string = !in_string;
        }
        if !in_string {
            match c {
                '[' | '(' | '{' => depth += 1,
                ']' | ')' | '}' => depth -= 1,
                _ => {}
            }
        }
        prev = c;
    }
    depth
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
    fn test_extract_first_tactic_multiline_bracket() {
        // Unclosed bracket continues to next line
        let raw = "simp_all [lemma1, lemma2,\n  lemma3, lemma4]";
        assert_eq!(extract_first_tactic(raw), "simp_all [lemma1, lemma2,\nlemma3, lemma4]");
    }

    #[test]
    fn test_extract_first_tactic_have_by() {
        // have with no unclosed bracket — first line only
        let raw = "have h := by\n  exact foo\nring";
        assert_eq!(extract_first_tactic(raw), "have h := by");
    }

    #[test]
    fn test_extract_first_tactic_two_independent() {
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
    fn test_extract_first_tactic_nested_brackets() {
        // Multiple bracket types
        let raw = "rw [show (a + b) = c from\n  by ring]";
        assert_eq!(extract_first_tactic(raw), "rw [show (a + b) = c from\nby ring]");
    }

    #[test]
    fn test_extract_first_tactic_closed_bracket_single_line() {
        // Brackets closed on same line — no continuation
        let raw = "simp [lemma1, lemma2]\nexact h";
        assert_eq!(extract_first_tactic(raw), "simp [lemma1, lemma2]");
    }

    #[test]
    fn test_extract_first_tactic_intro_then_multi() {
        // Real pattern: intro then continuation tactics — take only intro
        let raw = "intro z h₀\n   subst h₀\n   norm_num";
        assert_eq!(extract_first_tactic(raw), "intro z h₀");
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
