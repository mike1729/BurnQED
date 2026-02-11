#!/bin/bash
# Gather project context and invoke Gemini to review a plan.
# Usage: bash scripts/review-plan.sh [plan-file]
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Find plan file
PLAN_FILE="${1:-}"
if [ -z "$PLAN_FILE" ]; then
  if [ -f .claude/plan.md ]; then
    PLAN_FILE=".claude/plan.md"
  elif [ -d .claude/plans ]; then
    PLAN_FILE=$(ls -t .claude/plans/*.md 2>/dev/null | head -1)
  fi
fi

if [ -z "$PLAN_FILE" ] || [ ! -f "$PLAN_FILE" ]; then
  echo "ERROR: No plan file found. Write the plan first."
  echo "Checked: .claude/plan.md and .claude/plans/*.md"
  exit 1
fi

echo "Reviewing plan: $PLAN_FILE" >&2

PLAN=$(cat "$PLAN_FILE")

PROJECT_INFO=""
for pf in package.json requirements.txt Cargo.toml go.mod lakefile.lean lean-toolchain lakemanifest.json pyproject.toml setup.py setup.cfg Gemfile build.gradle pom.xml CMakeLists.txt Makefile flake.nix .tool-versions; do
  if [ -f "$pf" ]; then
    PROJECT_INFO="$PROJECT_INFO
--- $pf ---
$(head -80 "$pf")"
  fi
done
if [ -z "$PROJECT_INFO" ]; then PROJECT_INFO="No recognized project files found"; fi

README=$(cat README.md 2>/dev/null || echo "No README found")
STRUCTURE=$(find . -type f -not -path '*/node_modules/*' -not -path '*/.git/*' -not -path '*/dist/*' -not -path '*/.lake/*' -not -path '*/target/*' -not -path '*/__pycache__/*' -not -path '*/.build/*' | head -80)

TMPFILE=$(mktemp /tmp/gemini_prompt.XXXXXX)
cat > "$TMPFILE" <<GEMINI_PROMPT
You are a senior software architect reviewing a plan before implementation begins.

## PROJECT CONTEXT

### Project Files
$PROJECT_INFO

### Project Description
$README

### File Structure
$STRUCTURE

## PLAN TO REVIEW
$PLAN

## YOUR TASK

Evaluate this plan thoroughly. Consider:

1. Feasibility — Can this realistically be implemented as described? Are the steps concrete enough?
2. Completeness — Are there missing requirements, unhandled edge cases, or gaps in the approach?
3. Architecture Fit — Does the plan align with existing codebase patterns, conventions, and ecosystem?
4. Risk Assessment — What could go wrong? What are the unknowns? Are there breaking changes?
5. Simplification — Can the approach be simpler? Is it over-engineered?
6. Security — Any security implications (auth, input validation, data exposure)?
7. Testing — Is the testing strategy adequate? What test cases are missing?
8. Ordering — Are the implementation steps in the right order? Are there dependency chains?

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "verdict": "APPROVE" or "NEEDS_REVISION",
  "confidence": "high" or "medium" or "low",
  "estimated_complexity": "low" or "medium" or "high",
  "summary": "one-line overall assessment",
  "concerns": [
    {
      "severity": "high" or "medium" or "low",
      "category": "feasibility|completeness|architecture|risk|security|testing",
      "description": "what the concern is",
      "suggestion": "how to address it"
    }
  ],
  "missing_requirements": ["anything the plan forgot"],
  "suggested_plan_additions": ["specific items to add to the plan"],
  "alternative_approaches": ["simpler or better ways, if any"],
  "implementation_order_issues": ["any steps that should be reordered"]
}
GEMINI_PROMPT

gemini -m gemini-3-pro-preview < "$TMPFILE"
rm -f "$TMPFILE"
