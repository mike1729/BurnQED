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
    extract_all_tactics_impl(raw, true)
}

/// Internal implementation with decomposition control.
///
/// When `decompose` is true, typed `have`/`let` blocks with multi-line bodies
/// are split into a bare declaration + individual body tactics.
/// When false, all blocks are kept as multi-line strings (for recursive body
/// extraction — only the outermost block is decomposed).
fn extract_all_tactics_impl(raw: &str, decompose: bool) -> Vec<String> {
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
                    body_lines.push(lines[i].to_string());
                    i += 1;
                } else {
                    break;
                }
            }
            // Decompose typed have/let blocks with multi-line bodies
            // into a bare declaration (creates subgoal) + individual body tactics.
            // Only the outermost block is decomposed (decompose=true from public API).
            // Inner blocks are kept as multi-line strings to reduce goal debt.
            if decompose && body_lines.len() > 1 {
                if let Some(stripped) = strip_by_to_bare_have(first_line) {
                    tactics.push(stripped);
                    let body_text = body_lines.join("\n");
                    // Recurse with decompose=false: inner blocks stay as multi-line
                    let body_tactics = extract_all_tactics_impl(&body_text, false);
                    tactics.extend(body_tactics);
                    continue;
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

        // Join `<;>` combinator chains into a single compound tactic.
        // Lines ending with `<;>` continue onto the next line. Standalone
        // `<;>` lines (the model sometimes puts `<;>` on its own line)
        // are joined as a prefix to the next continuation.
        //
        // Example input (each at same indent):
        //   rcases b with (_ | _ | _ | _) <;>
        //   simp_all [pow_one] <;>
        //   norm_num at * <;>
        //   linarith
        //
        // Becomes one tactic:
        //   rcases b with (_ | _ | _ | _) <;> simp_all [pow_one] <;> norm_num at * <;> linarith
        if ends_with_semicolon_combinator(first_line) {
            let mut chain = first_line.to_string();
            i += 1;
            while i < lines.len() {
                let next_trimmed = lines[i].trim();
                if next_trimmed.is_empty() {
                    break;
                }
                // Standalone `<;>` line — absorb and continue
                if next_trimmed == "<;>" {
                    chain.push_str(" <;>");
                    i += 1;
                    continue;
                }
                // Continuation at same or deeper indent
                let next_indent = indent_of(lines[i]);
                if next_indent >= line_indent {
                    chain.push(' ');
                    chain.push_str(next_trimmed);
                    i += 1;
                    if !ends_with_semicolon_combinator(next_trimmed) {
                        break; // Chain ended
                    }
                } else {
                    break;
                }
            }
            tactics.push(chain);
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
/// Focus dots (`·`) arise when the model generates focused sub-blocks.
/// Standalone `<;>` lines that weren't absorbed by chain joining are also
/// dropped. Note: `<;>` at the END of a real tactic is handled by the
/// chain-joining logic, so only truly dangling `<;>` lines reach here.
fn should_skip_tactic(tactic: &str) -> bool {
    let trimmed = tactic.trim();
    trimmed.is_empty()
        || trimmed == "<;>"
        || trimmed.starts_with("<;> ")
        || trimmed == "·"
        || trimmed == "."
}

/// Check whether a tactic line ends with the `<;>` combinator.
///
/// This indicates the tactic continues on the next line and should be
/// joined into a single compound tactic during extraction.
fn ends_with_semicolon_combinator(line: &str) -> bool {
    let t = line.trim();
    t.ends_with("<;>") && t != "<;>"
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

/// Strip `:= by` or trailing `by` from a typed `have`/`let` line,
/// returning the bare tactic that creates a subgoal in Lean 4.
///
/// `have h : T := by`  → `Some("have h : T")`
/// `let x : T := by`   → `Some("let x : T")`
/// `have h := by`       → `None` (no type annotation, can't decompose)
/// `conv => by`          → `None` (not a have/let)
///
/// Returns `None` when decomposition isn't safe (no type annotation or
/// not a have/let — we can't create a typed subgoal without a type).
fn strip_by_to_bare_have(line: &str) -> Option<String> {
    let t = line.trim();

    // Only decompose have/let — these create subgoals in tactic mode
    if !t.starts_with("have ") && !t.starts_with("let ") {
        return None;
    }

    // Strip trailing ":= by" or " by"
    let core = if t.ends_with(":= by") {
        &t[..t.len() - 5]
    } else if t.ends_with(" by") {
        &t[..t.len() - 3]
    } else {
        return None;
    };

    let core = core.trim_end();

    // Must have a type annotation (contains " : ")
    // "have h : T" ✓ → can create subgoal
    // "have h"     ✗ → can't infer type without body
    if !core.contains(" : ") {
        return None;
    }

    Some(core.to_string())
}

/// Build a tactic string from an opener line and its raw (untrimmed) body lines.
///
/// For `by` blocks with a single-line body, collapse inline:
///   `have h := by` + `    nlinarith` → `have h := by nlinarith`
/// For multi-line bodies, preserve relative indentation so nested blocks
/// survive Lean's indentation-sensitive parser:
///   ```text
///   have h := by       ← opener
///       rw [foo]       ← raw body, indent 8
///       simp           ← raw body, indent 8
///   →  have h := by
///        rw [foo]      ← 2 + (8-8) = 2 spaces
///        simp          ← 2 + (8-8) = 2 spaces
///   ```
/// Nested blocks get deeper relative indent preserved:
///   ```text
///   have h := by             ← opener
///       have h₂ := by       ← raw body, indent 8 (base)
///           nlinarith        ← raw body, indent 12
///       exact h₂             ← raw body, indent 8
///   →  have h := by
///        have h₂ := by      ← 2 + 0 = 2 spaces
///            nlinarith       ← 2 + 4 = 6 spaces
///        exact h₂            ← 2 + 0 = 2 spaces
///   ```
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
        return format!("{} {}", opener, body[0].trim());
    }

    // Find the base indentation from the first non-empty body line
    let base_indent = body
        .iter()
        .find(|l| !l.trim().is_empty())
        .map(|l| indent_of(l))
        .unwrap_or(0);

    // Multi-line body → preserve relative indentation
    let mut result = opener.to_string();
    for line in body {
        if line.trim().is_empty() {
            continue;
        }
        let current_indent = indent_of(line);
        let relative = current_indent.saturating_sub(base_indent);
        result.push('\n');
        // 2-space base indent + relative nesting from original source
        for _ in 0..(2 + relative) {
            result.push(' ');
        }
        result.push_str(line.trim());
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
        // Model wraps entire proof in `have : (goal) := by ...` at the root.
        // Typed have with multi-line body → decomposed.
        let raw = "have h : P := by\n  intro x\n  exact x\nexact h";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec!["have h : P", "intro x", "exact x", "exact h"]);
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
    fn test_structured_nested_by_blocks_decomposed() {
        // Only the outermost typed have is decomposed.
        // Inner blocks stay as multi-line strings (reduces goal debt).
        let raw = "    have h\u{2083} : u = 34 := by\n      have h\u{2083}\u{2081} : 34 \u{2208} S := by\n        rw [h\u{2080}]\n        norm_num [Nat.ModEq]\n      exact h\u{2081}.unique h\u{2083}\u{2082}\n    simp";
        let result = extract_all_tactics_structured(raw);
        // Outer decomposed, inner kept as multi-line block
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "have h\u{2083} : u = 34");
        assert_eq!(result[1], "have h\u{2083}\u{2081} : 34 \u{2208} S := by\n  rw [h\u{2080}]\n  norm_num [Nat.ModEq]");
        assert_eq!(result[2], "exact h\u{2081}.unique h\u{2083}\u{2082}");
        assert_eq!(result[3], "simp");
    }

    #[test]
    fn test_structured_deeply_nested_untyped_not_decomposed() {
        // Untyped have blocks (no " : ") can't be decomposed — no type for subgoal.
        // They fall through to build_block_tactic as multi-line blocks.
        let raw = "  have a := by\n    have b := by\n      have c := by\n        omega\n      exact c\n    exact b\n  exact a";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0],
            "have a := by\n  have b := by\n    have c := by\n      omega\n    exact c\n  exact b"
        );
        assert_eq!(result[1], "exact a");
    }

    #[test]
    fn test_structured_typed_have_decomposed_basic() {
        // Basic typed have with 2 body lines → decomposed
        let raw = "  have h : T := by\n    rw [foo]\n    simp\n  exact h";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec!["have h : T", "rw [foo]", "simp", "exact h"]);
    }

    #[test]
    fn test_strip_by_to_bare_have() {
        // Typed have → stripped
        assert_eq!(
            strip_by_to_bare_have("have h : T := by"),
            Some("have h : T".to_string())
        );
        assert_eq!(
            strip_by_to_bare_have("have h₃ : u = 34 := by"),
            Some("have h\u{2083} : u = 34".to_string())
        );
        // let also works
        assert_eq!(
            strip_by_to_bare_have("let x : Nat := by"),
            Some("let x : Nat".to_string())
        );
        // "have h : T by" (no :=) also stripped
        assert_eq!(
            strip_by_to_bare_have("have h : T by"),
            Some("have h : T".to_string())
        );
        // Untyped have → None (can't create subgoal without type)
        assert_eq!(strip_by_to_bare_have("have h := by"), None);
        // Not a have/let → None
        assert_eq!(strip_by_to_bare_have("conv => by"), None);
        assert_eq!(strip_by_to_bare_have("simp only [foo] by"), None);
        // No "by" at end → None
        assert_eq!(strip_by_to_bare_have("have h : T := foo"), None);
    }

    #[test]
    fn test_decompose_does_not_affect_single_body() {
        // Typed have with single body line → still inlined (not decomposed)
        let raw = "  have h : T := by\n    omega\n  exact h";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec!["have h : T := by omega", "exact h"]);
    }

    #[test]
    fn test_decompose_does_not_affect_conv() {
        // conv blocks are not have/let, so not decomposed
        let raw = "  conv =>\n    rw [foo]\n    simp\n  exact h";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 2);
        assert!(result[0].starts_with("conv =>"));
        assert_eq!(result[1], "exact h");
    }

    #[test]
    fn test_decompose_real_is_least_proof() {
        // Real model output for the IsLeast problem.
        // Outer have decomposed, inner blocks kept as multi-line.
        let raw = "  have h\u{2083} : u = 34 := by\n    have h\u{2083}\u{2081} : 34 \u{2208} S := by\n      rw [h\u{2080}]\n      norm_num [Nat.ModEq]\n    have h\u{2083}\u{2082} : IsLeast S 34 := by\n      exact \u{27e8}h\u{2083}\u{2081}, fun x hx => by omega\u{27e9}\n    exact h\u{2081}.unique h\u{2083}\u{2082}\n  exact h\u{2083}";
        let result = extract_all_tactics_structured(raw);
        // Outer decomposed → bare "have h₃ : u = 34"
        assert_eq!(result[0], "have h\u{2083} : u = 34");
        // Inner h₃₁ has multi-line body → kept as multi-line block
        assert!(result[1].starts_with("have h\u{2083}\u{2081} : 34 \u{2208} S := by"));
        assert!(result[1].contains("rw [h\u{2080}]"));
        assert!(result[1].contains("norm_num [Nat.ModEq]"));
        // Inner h₃₂ has single-line body → inlined
        assert!(result[2].starts_with("have h\u{2083}\u{2082} : IsLeast S 34 := by"));
        assert_eq!(result[3], "exact h\u{2081}.unique h\u{2083}\u{2082}");
        assert_eq!(result[4], "exact h\u{2083}");
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

    // ------ <;> chain joining tests ------

    #[test]
    fn test_structured_semicolon_chain_basic() {
        // Multi-line <;> chain should be joined into a single tactic
        let raw = "  rcases b with (_ | _ | _ | _) <;>\n  simp_all [pow_one] <;>\n  norm_num at * <;>\n  linarith";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec![
            "rcases b with (_ | _ | _ | _) <;> simp_all [pow_one] <;> norm_num at * <;> linarith"
        ]);
    }

    #[test]
    fn test_structured_semicolon_chain_with_preceding_tactics() {
        // Tactics before the chain should stay separate
        let raw = "intro b F h₀ h₁\n  simp only [h₀, h₁] at *\n  norm_num at h₁\n  rcases b with (_ | _ | _ | _) <;>\n  simp_all [pow_one, pow_two, pow_three] <;>\n  norm_num at * <;>\n  linarith";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "intro b F h₀ h₁");
        assert_eq!(result[1], "simp only [h₀, h₁] at *");
        assert_eq!(result[2], "norm_num at h₁");
        assert_eq!(result[3], "rcases b with (_ | _ | _ | _) <;> simp_all [pow_one, pow_two, pow_three] <;> norm_num at * <;> linarith");
    }

    #[test]
    fn test_structured_semicolon_chain_with_standalone_separator() {
        // Standalone <;> line between parts of the chain
        let raw = "  norm_num at h₁\n  <;>\n  rcases b with (_ | _ | _ | _) <;>\n  linarith";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], "norm_num at h₁");
        // <;> on its own line is filtered; rcases chain is joined
        assert_eq!(result[1], "rcases b with (_ | _ | _ | _) <;> linarith");
    }

    #[test]
    fn test_structured_semicolon_chain_single_line_preserved() {
        // Already on one line — no change needed
        let raw = "interval_cases x <;> omega";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec!["interval_cases x <;> omega"]);
    }

    #[test]
    fn test_structured_semicolon_chain_real_proof() {
        // Real DeepSeek output for mathd_algebra_59
        let raw = "intro b F h₀ h₁\n  simp only [h₀, h₁] at *\n  norm_num at h₁\n  <;>\n  rcases b with (_ | _ | _ | _) <;>\n  simp_all [pow_one, pow_two, pow_three] <;>\n  norm_num at * <;>\n  linarith";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result.len(), 4);
        assert_eq!(result[0], "intro b F h₀ h₁");
        assert_eq!(result[1], "simp only [h₀, h₁] at *");
        assert_eq!(result[2], "norm_num at h₁");
        // <;> standalone was filtered, rcases chain joined
        assert_eq!(result[3], "rcases b with (_ | _ | _ | _) <;> simp_all [pow_one, pow_two, pow_three] <;> norm_num at * <;> linarith");
    }

    #[test]
    fn test_structured_semicolon_chain_with_try() {
        // Chain with (try ...) subexpressions containing <;>
        let raw = "  rcases n with (_ | _ | _) <;>\n  (try norm_num) <;>\n  (try linarith) <;>\n  nlinarith";
        let result = extract_all_tactics_structured(raw);
        assert_eq!(result, vec![
            "rcases n with (_ | _ | _) <;> (try norm_num) <;> (try linarith) <;> nlinarith"
        ]);
    }
}
