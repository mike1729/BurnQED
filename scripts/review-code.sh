#!/bin/bash
# Gather code changes and invoke Gemini for code review.
# Usage: bash scripts/review-code.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DIFF=$(git diff --cached 2>/dev/null)
if [ -z "$DIFF" ]; then DIFF=$(git diff 2>/dev/null); fi
if [ -z "$DIFF" ]; then DIFF=$(git diff HEAD~1 2>/dev/null); fi
if [ -z "$DIFF" ]; then echo "No changes detected — nothing to review."; exit 0; fi

CHANGED_FILES=$(echo "$DIFF" | grep '^diff --git' | sed 's|diff --git a/||;s| b/.*||')

FILE_CONTENTS=""
for f in $CHANGED_FILES; do
  if [ -f "$f" ]; then
    FILE_CONTENTS="$FILE_CONTENTS
=== FILE: $f ===
$(cat "$f")
"
  fi
done

PROJECT_INFO=""
for pf in package.json requirements.txt Cargo.toml go.mod lakefile.lean lean-toolchain lakemanifest.json pyproject.toml setup.py setup.cfg Gemfile build.gradle pom.xml CMakeLists.txt Makefile flake.nix .tool-versions; do
  if [ -f "$pf" ]; then
    PROJECT_INFO="$PROJECT_INFO
--- $pf ---
$(head -80 "$pf")"
  fi
done
if [ -z "$PROJECT_INFO" ]; then PROJECT_INFO="No recognized project files found"; fi

PLAN=$(cat .claude/plan.md 2>/dev/null || echo "No plan file found")
LINT_CONFIG=$(cat .eslintrc.json 2>/dev/null || cat rustfmt.toml 2>/dev/null || cat clippy.toml 2>/dev/null || cat pyproject.toml 2>/dev/null || echo "No lint config found")
TEST_FILES=$(find . -type f \( -name '*.test.*' -o -name '*.spec.*' -o -name 'test_*' -o -name '*_test.*' -o -name '*Test.*' \) -not -path '*/node_modules/*' -not -path '*/.lake/*' -not -path '*/target/*' -not -path '*/__pycache__/*' 2>/dev/null | head -20)

TMPFILE=$(mktemp /tmp/gemini_prompt.XXXXXX)
cat > "$TMPFILE" <<GEMINI_PROMPT
You are a senior software engineer performing a thorough code review. You have high standards but give actionable, specific feedback. Adapt your review to the detected language and ecosystem.

## PROJECT CONTEXT

### Project Files
$PROJECT_INFO

### Lint / Style Config
$LINT_CONFIG

### Existing Test Files
$TEST_FILES

### Original Plan (if available)
$PLAN

## CHANGES TO REVIEW

### Diff
$DIFF

### Full Content of Changed Files
$FILE_CONTENTS

## YOUR TASK

Review these changes across ALL of the following dimensions:

1. Correctness & Logic — off-by-one, null refs, race conditions, type errors, ownership issues
2. Code Quality — naming, DRY, dead code, idiomatic usage, consistency with codebase
3. Performance — unnecessary allocations, O(n^2) patterns, missing caching
4. Error Handling — missing error handling, swallowed errors, unclear messages
5. Security — injection, hardcoded secrets, missing validation, insecure defaults
6. Testing — adequate coverage, missing edge cases, assertion quality
7. Plan Compliance — do changes match the plan (if provided)?
8. API & Interface Design — breaking changes, naming consistency, type annotations

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "verdict": "APPROVE" or "REQUEST_CHANGES" or "NEEDS_DISCUSSION",
  "confidence": "high" or "medium" or "low",
  "summary": "one-line overall assessment",
  "detected_language": "language detected",
  "issues": [
    {
      "severity": "error" or "warning" or "suggestion",
      "category": "correctness|quality|performance|error-handling|security|testing|plan-compliance|api-design",
      "file": "path/to/file",
      "line": 0,
      "description": "what the problem is",
      "suggestion": "specific fix or improvement",
      "code_example": "corrected code snippet if applicable, otherwise null"
    }
  ],
  "missing_tests": [
    {"file": "path", "description": "what test is missing", "test_outline": "what it should do"}
  ],
  "plan_deviations": ["any differences from the plan"],
  "praise": ["things done well"],
  "overall_quality": "excellent" or "good" or "acceptable" or "needs-work" or "poor"
}
GEMINI_PROMPT

gemini -m gemini-3-pro-preview < "$TMPFILE"
rm -f "$TMPFILE"
