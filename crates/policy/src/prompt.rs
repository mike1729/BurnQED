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

/// Extract the first complete tactic from model output using indentation.
///
/// Takes the first non-empty/non-comment line, then continues onto subsequent
/// lines that are **more indented** than the first line. This captures a tactic
/// together with its indented sub-block (e.g. `have h := by\n  nlinarith`) while
/// stopping before sibling tactics at the same indent level.
///
/// DeepSeek-Prover-V2 generates multi-step proof chains where each top-level
/// tactic is at the same indent and sub-tactic bodies are indented deeper:
/// ```text
///  have h₃ : x ≥ 0 := by     ← indent 1 (first tactic)
///      nlinarith                ← indent 5 (body, included)
///    have h₄ : y ≥ 0 := by    ← indent 3 (sibling, STOP)
/// ```
pub fn extract_first_tactic(raw: &str) -> String {
    let text = raw.trim_end();
    // Strip code fence if present
    let text = if text.trim_start().starts_with("```") {
        text.lines()
            .skip(1) // skip ```lean4
            .take_while(|l| !l.trim_start().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        text.to_string()
    };

    let lines: Vec<&str> = text.lines().collect();

    // Find first non-empty, non-comment line
    let mut start = 0;
    while start < lines.len() {
        let trimmed = lines[start].trim();
        if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with("/-") {
            start += 1;
        } else {
            break;
        }
    }
    if start >= lines.len() {
        return String::new();
    }

    // Strip leading focus dot (· or ·)
    let first_line = lines[start];
    let first_trimmed = first_line.trim();
    let first_line = if let Some(rest) = first_trimmed.strip_prefix("· ").or_else(|| first_trimmed.strip_prefix("· ")) {
        rest
    } else if first_trimmed == "·" || first_trimmed == "·" {
        // Bare focus dot — skip to next non-empty line
        start += 1;
        while start < lines.len() && lines[start].trim().is_empty() {
            start += 1;
        }
        if start >= lines.len() {
            return String::new();
        }
        lines[start].trim()
    } else {
        first_trimmed
    };

    // Determine the indent of the first line (in the original text)
    let base_indent = indent_of(lines[start]);

    let mut result = first_line.to_string();

    // Decide whether to include continuation lines:
    // 1. If the first line ends with a block opener (by, where, do, =>),
    //    include indented body lines.
    // 2. If the first line has unclosed brackets ([{, include lines until
    //    brackets balance.
    // 3. Otherwise, return just the first line — no continuation.
    let ends_with_block_opener = {
        let t = first_line.trim_end();
        t.ends_with(" by")
            || t.ends_with(" where")
            || t.ends_with(" do")
            || t.ends_with(" =>")
            || t == "by"
    };

    if ends_with_block_opener {
        // Include indented body lines using body_indent heuristic
        let mut body_indent = None;
        for line in &lines[start + 1..] {
            if line.trim().is_empty() {
                break;
            }
            let li = indent_of(line);
            if li <= base_indent {
                break;
            }
            if body_indent.is_none() {
                body_indent = Some(li);
            } else if li < body_indent.unwrap() {
                break;
            }
            result.push('\n');
            result.push_str(line.trim());
        }
    } else if bracket_depth(first_line) > 0 {
        // Unclosed brackets — continue until balanced
        let mut depth = bracket_depth(first_line);
        for line in &lines[start + 1..] {
            if line.trim().is_empty() {
                break;
            }
            result.push('\n');
            result.push_str(line.trim());
            depth += bracket_depth(line);
            if depth <= 0 {
                break;
            }
        }
    }
    // else: simple tactic, first line only

    result.trim().to_string()
}

/// Count leading whitespace characters in a line.
fn indent_of(line: &str) -> usize {
    line.len() - line.trim_start().len()
}

/// Net bracket depth change for a line: +1 for each opener, -1 for each closer.
fn bracket_depth(line: &str) -> i32 {
    let mut depth = 0i32;
    for ch in line.chars() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
    }
    depth
}

/// Extract all tactics from model output as structured multi-line blocks.
///
/// Indentation-based splitting into tactic units for whole-proof replay.
/// Lines at base indent start new tactics; deeper lines are continuations
/// when the parent ends with a block opener (`by`/`where`/`do`/`=>`) or
/// has unclosed brackets.
///
/// Special cases:
/// - **`calc` blocks**: `calc` starts a block. Lines starting with `_` are
///   continuations regardless of indent. Consumed until a non-`_` line at base indent.
/// - **`conv =>` / `match` / block openers**: consume indented body.
/// - **General heuristic**: any line starting with `_` is treated as continuation.
///
/// Code fences, comments, and theorem/lemma declarations are stripped.
pub fn extract_all_tactics_structured(raw: &str) -> Vec<String> {
    let text = raw.trim_end();
    // Strip code fence if present (preserve indentation of inner lines)
    let text = if text.trim_start().starts_with("```") {
        text.lines()
            .skip(1)
            .take_while(|l| !l.trim_start().starts_with("```"))
            .collect::<Vec<_>>()
            .join("\n")
    } else {
        text.to_string()
    };

    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut tactics: Vec<String> = Vec::new();
    let mut i = 0;
    let mut past_declaration = false;

    while i < lines.len() {
        let trimmed = lines[i].trim();

        // Skip empty lines and comments
        if trimmed.is_empty()
            || trimmed.starts_with("--")
            || trimmed.starts_with("/-")
            || trimmed.starts_with("-/")
        {
            i += 1;
            continue;
        }

        // Skip declarations (theorem/lemma/example headers)
        if trimmed.starts_with("theorem ")
            || trimmed.starts_with("lemma ")
            || trimmed.starts_with("example ")
        {
            past_declaration = true;
            i += 1;
            continue;
        }

        let line_indent = indent_of(lines[i]);
        let first_line = trimmed;

        // Check if this is a calc block
        if first_line.starts_with("calc") {
            let mut block = vec![first_line.to_string()];
            i += 1;
            while i < lines.len() {
                let next_trimmed = lines[i].trim();
                if next_trimmed.is_empty() {
                    i += 1;
                    continue;
                }
                let next_indent = indent_of(lines[i]);
                if next_indent > line_indent || next_trimmed.starts_with('_') {
                    block.push(next_trimmed.to_string());
                    i += 1;
                } else {
                    break;
                }
            }
            tactics.push(block.join("\n"));
            continue;
        }

        // Check if this line ends with a block opener
        let ends_with_block_opener = is_block_opener(first_line);

        // Check for match expression
        let is_match = first_line.starts_with("match ");

        if ends_with_block_opener || is_match {
            // Consume body lines indented deeper than this opener
            let mut body_lines: Vec<String> = Vec::new();
            i += 1;
            while i < lines.len() {
                let next_trimmed = lines[i].trim();
                if next_trimmed.is_empty() {
                    // Peek ahead: if more indented body follows, skip blank line
                    let mut peek = i + 1;
                    while peek < lines.len() && lines[peek].trim().is_empty() {
                        peek += 1;
                    }
                    if peek < lines.len() && indent_of(lines[peek]) > line_indent {
                        i += 1;
                        continue;
                    }
                    break;
                }
                let next_indent = indent_of(lines[i]);
                if next_indent > line_indent
                    || next_trimmed.starts_with('|')
                    || next_trimmed.starts_with('_')
                {
                    body_lines.push(next_trimmed.to_string());
                    i += 1;
                } else {
                    break;
                }
            }
            tactics.push(build_block_tactic(first_line, &body_lines));
            continue;
        }

        // Check for unclosed brackets
        if bracket_depth(first_line) > 0 {
            let mut block = first_line.to_string();
            let mut depth = bracket_depth(first_line);
            i += 1;
            while i < lines.len() && depth > 0 {
                let next_trimmed = lines[i].trim();
                if next_trimmed.is_empty() {
                    break;
                }
                block.push('\n');
                block.push_str(next_trimmed);
                depth += bracket_depth(next_trimmed);
                i += 1;
            }
            tactics.push(block);
            continue;
        }

        // Simple single-line tactic
        tactics.push(first_line.to_string());
        i += 1;
    }

    // If we saw a declaration but no tactics after it, return empty
    if past_declaration && tactics.is_empty() {
        return Vec::new();
    }

    // Filter out tactics that can never work as standalone Lean tactics.
    // These arise from model output artifacts (dangling combinators, focus dots).
    tactics.retain(|t| !should_skip_tactic(t));

    tactics
}

/// Tactics that should be silently dropped from extracted proof sequences.
///
/// Dangling `<;>` combinators and focus dots (`·`) arise when the model
/// generates multi-tactic combinator chains that get split into individual
/// lines. These always fail in Pantograph as standalone tactics.
fn should_skip_tactic(tactic: &str) -> bool {
    let trimmed = tactic.trim();
    trimmed.is_empty()
        || trimmed == "<;>"
        || trimmed.starts_with("<;> ")
        || trimmed == "·"
        || trimmed == "."
}

/// Check whether a tactic line ends with a block opener keyword.
fn is_block_opener(line: &str) -> bool {
    let t = line.trim_end();
    t.ends_with(" by")
        || t.ends_with(" where")
        || t.ends_with(" do")
        || t.ends_with(" =>")
        || t.ends_with(":= by")
        || t == "by"
}

/// Build a tactic string from an opener line and its body lines.
///
/// For `by` blocks with a single-line body, collapse inline:
///   `have h := by` + `[nlinarith]` → `have h := by nlinarith`
/// For multi-line bodies, indent with 2 spaces so Lean parses the block:
///   `have h := by` + `[nlinarith, exact h]` → `have h := by\n  nlinarith\n  exact h`
fn build_block_tactic(opener: &str, body: &[String]) -> String {
    if body.is_empty() {
        return opener.to_string();
    }

    let is_by_block = {
        let t = opener.trim_end();
        t.ends_with(" by") || t.ends_with(":= by") || t == "by"
    };

    if is_by_block && body.len() == 1 {
        // Single-line by body → collapse inline
        return format!("{} {}", opener, body[0]);
    }

    // Multi-line body → indent each line under the opener
    let mut result = opener.to_string();
    for line in body {
        result.push('\n');
        result.push_str("  ");
        result.push_str(line);
    }
    result
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
    fn test_extract_first_tactic_multiline_indented() {
        // Indented continuation included
        let raw = "simp_all [lemma1, lemma2,\n  lemma3, lemma4]";
        assert_eq!(extract_first_tactic(raw), "simp_all [lemma1, lemma2,\nlemma3, lemma4]");
    }

    #[test]
    fn test_extract_first_tactic_have_by_with_body() {
        // have + indented body captured; sibling at same indent excluded
        let raw = " have h := by\n     exact foo\n   ring";
        assert_eq!(extract_first_tactic(raw), "have h := by\nexact foo");
    }

    #[test]
    fn test_extract_first_tactic_two_independent() {
        // Same indent → only first line
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
    fn test_extract_first_tactic_nested_indented() {
        // Deeper indentation captured
        let raw = "rw [show (a + b) = c from\n  by ring]";
        assert_eq!(extract_first_tactic(raw), "rw [show (a + b) = c from\nby ring]");
    }

    #[test]
    fn test_extract_first_tactic_same_indent_stop() {
        // Same indent = separate tactic, not captured
        let raw = "simp [lemma1, lemma2]\nexact h";
        assert_eq!(extract_first_tactic(raw), "simp [lemma1, lemma2]");
    }

    #[test]
    fn test_extract_first_tactic_deepseek_real_output() {
        // Real DeepSeek output: have + indented body, then sibling at indent 3
        let raw = " have h₃ : 0 < x := by\n     nlinarith\n   have h₄ : x ≤ 80 := by\n     nlinarith\n   interval_cases x <;> omega";
        assert_eq!(extract_first_tactic(raw), "have h₃ : 0 < x := by\nnlinarith");
    }

    #[test]
    fn test_extract_first_tactic_single_line_have() {
        // Inline body after "by" — no block opener at end, so first line only
        let raw = " have h₃ : 0 < x := by linarith\n   interval_cases x <;> omega";
        assert_eq!(extract_first_tactic(raw), "have h₃ : 0 < x := by linarith");
    }

    #[test]
    fn test_extract_first_tactic_intro_chain() {
        // intro is not a block opener — only first line, siblings excluded
        let raw = " intro u v S h₀ h₁ h₂\n   have h₃ : u = 34 := by\n     omega";
        assert_eq!(extract_first_tactic(raw), "intro u v S h₀ h₁ h₂");
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
        // Leading focus dot stripped, same indent → first line only
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

    // ------ extract_all_tactics_structured tests ------

    #[test]
    fn test_structured_simple_chain() {
        let raw = "intro h\nexact h";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["intro h", "exact h"]
        );
    }

    #[test]
    fn test_structured_multiline_have() {
        // have with by block opener → single-line body collapsed inline
        let raw = "  intro h\n  have h\u{2082} := by\n    nlinarith\n  exact h\u{2082}";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["intro h", "have h\u{2082} := by nlinarith", "exact h\u{2082}"]
        );
    }

    #[test]
    fn test_structured_calc_block() {
        let raw = "  calc x + y\n    = y + x := by ring\n  _ = z := by simp";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["calc x + y\n= y + x := by ring\n_ = z := by simp"]
        );
    }

    #[test]
    fn test_structured_conv_block() {
        // conv => with single-line body gets 2-space indent (not a `by` block)
        let raw = "  conv =>\n    rw [foo]\n  simp";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["conv =>\n  rw [foo]", "simp"]
        );
    }

    #[test]
    fn test_structured_match_arms() {
        // match arms preserved with 2-space indent
        let raw = "  match h with\n  | inl a => exact a\n  | inr b => exact b";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["match h with\n  | inl a => exact a\n  | inr b => exact b"]
        );
    }

    #[test]
    fn test_structured_code_fence() {
        let raw = "```lean4\nintro h\nsimp\nring\n```";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["intro h", "simp", "ring"]
        );
    }

    #[test]
    fn test_structured_declaration() {
        let raw = "theorem foo : True := by\n  trivial";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["trivial"]
        );
    }

    #[test]
    fn test_structured_comments() {
        let raw = "-- reason\nintro h\n-- step\nsimp";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["intro h", "simp"]
        );
    }

    #[test]
    fn test_structured_unclosed_brackets() {
        let raw = "simp [lemma1, lemma2,\n  lemma3, lemma4]";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["simp [lemma1, lemma2,\nlemma3, lemma4]"]
        );
    }

    #[test]
    fn test_structured_underscore_continuation() {
        // _ lines are continuations of calc-like blocks
        let raw = "  calc 1 + 1 = 2 := by norm_num\n  _ = 2 := by rfl";
        assert_eq!(
            extract_all_tactics_structured(raw),
            vec!["calc 1 + 1 = 2 := by norm_num\n_ = 2 := by rfl"]
        );
    }

    #[test]
    fn test_structured_empty() {
        assert!(extract_all_tactics_structured("").is_empty());
        assert!(extract_all_tactics_structured("  \n  ").is_empty());
    }

    #[test]
    fn test_structured_single_tactic() {
        assert_eq!(
            extract_all_tactics_structured("omega"),
            vec!["omega"]
        );
    }

    #[test]
    fn test_structured_deepseek_real_output() {
        // Real DeepSeek output: have + single-line body → collapsed inline
        let raw = "  have h\u{2083} : 0 < x := by\n    nlinarith\n  have h\u{2084} : x \u{2264} 80 := by\n    nlinarith\n  interval_cases x <;> omega";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], "have h\u{2083} : 0 < x := by nlinarith");
        assert_eq!(result[1], "have h\u{2084} : x \u{2264} 80 := by nlinarith");
        assert_eq!(result[2], "interval_cases x <;> omega");
    }

    #[test]
    fn test_structured_prompt_continuation_indent() {
        // Model output from prompt ending with `example := by\n  `:
        // First line at indent 0, sibling tactics at indent 2, bodies at indent 4.
        // The splitter must treat each indent-2 line as a new tactic, not skip it.
        let raw = "intro x y h\u{2080} h\u{2081} h\u{2082}\n  have h\u{2083} : 0 < x := by\n    linarith\n  have h\u{2084} : x \u{2264} 80 := by\n    nlinarith\n  interval_cases x <;> omega";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "intro x y h\u{2080} h\u{2081} h\u{2082}");
        assert_eq!(result[1], "have h\u{2083} : 0 < x := by linarith");
        assert_eq!(result[2], "have h\u{2084} : x \u{2264} 80 := by nlinarith");
        assert_eq!(result[3], "interval_cases x <;> omega");
    }

    #[test]
    fn test_structured_flat_inline_by() {
        // All tactics at indent 0, by bodies inline — should split cleanly
        let raw = "intro x y h\u{2080} h\u{2081} h\u{2082}\nhave h\u{2083} : 0 < x := by linarith\nhave h\u{2084} : x \u{2264} 80 := by nlinarith\ninterval_cases x <;> omega";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "intro x y h\u{2080} h\u{2081} h\u{2082}");
        assert_eq!(result[1], "have h\u{2083} : 0 < x := by linarith");
        assert_eq!(result[2], "have h\u{2084} : x \u{2264} 80 := by nlinarith");
        assert_eq!(result[3], "interval_cases x <;> omega");
    }

    #[test]
    fn test_structured_have_wrapper_whole_proof() {
        // Model wraps entire proof in `have : (goal) := by ...` at the root
        let raw = "have h : P := by\n  intro x\n  exact x\nexact h";
        let result = extract_all_tactics_structured(raw);
        // The have block consumes indented body; exact h is a separate tactic
        assert_eq!(result.len(), 2);
        assert!(result[0].starts_with("have h : P := by"));
        assert_eq!(result[1], "exact h");
    }

    #[test]
    fn test_structured_multi_line_by_body() {
        // by block with multiple body lines → indented, NOT collapsed
        let raw = "  have h := by\n    rw [foo]\n    simp\n  exact h";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "have h := by\n  rw [foo]\n  simp");
        assert_eq!(result[1], "exact h");
    }

    #[test]
    fn test_should_skip_tactic() {
        assert!(should_skip_tactic("<;>"));
        assert!(should_skip_tactic("<;> omega"));
        assert!(should_skip_tactic("·"));
        assert!(should_skip_tactic("."));
        assert!(should_skip_tactic(""));
        assert!(should_skip_tactic("  "));
        // These should NOT be skipped
        assert!(!should_skip_tactic("simp"));
        assert!(!should_skip_tactic("interval_cases x <;> omega"));
        assert!(!should_skip_tactic("have h := by linarith"));
    }

    #[test]
    fn test_structured_filters_dangling_combinators() {
        // Model output with dangling <;> after tactic splitting
        let raw = "interval_cases x\n<;> omega\n<;>\nsimp";
        let result = extract_all_tactics_structured(raw);
        // <;> omega and <;> should be filtered out
        assert_eq!(result, vec!["interval_cases x", "simp"]);
    }

    #[test]
    fn test_structured_filters_focus_dot() {
        let raw = "·\nintro h\n·\nexact h";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec!["intro h", "exact h"]);
    }
}
