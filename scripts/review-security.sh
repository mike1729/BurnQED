#!/bin/bash
# Gather code changes and invoke Gemini for security audit.
# Usage: bash scripts/review-security.sh
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

DIFF=$(git diff --cached 2>/dev/null || git diff 2>/dev/null || git diff HEAD~1 2>/dev/null)
if [ -z "$DIFF" ]; then echo "No changes detected."; exit 0; fi

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

ENV_EXAMPLE=$(cat .env.example 2>/dev/null || echo "No .env.example found")
DOCKER=$(cat Dockerfile 2>/dev/null || cat docker-compose.yml 2>/dev/null || echo "No Docker config found")
AUTH_FILES=$(find . -type f \( -name '*auth*' -o -name '*middleware*' -o -name '*permission*' -o -name '*guard*' -o -name '*security*' \) -not -path '*/node_modules/*' -not -path '*/.git/*' -not -path '*/.lake/*' -not -path '*/target/*' 2>/dev/null | head -10)

TMPFILE=$(mktemp /tmp/gemini_prompt.XXXXXX)
cat > "$TMPFILE" <<GEMINI_PROMPT
You are a senior application security engineer performing a focused security audit. Think like an attacker. Adapt analysis to the detected language and ecosystem.

## PROJECT CONTEXT

### Project Files
$PROJECT_INFO

### Environment Config Template
$ENV_EXAMPLE

### Docker / Deployment Config
$DOCKER

### Existing Auth/Security Files
$AUTH_FILES

## CODE TO AUDIT

### Diff
$DIFF

### Full Content of Changed Files
$FILE_CONTENTS

## AUDIT CHECKLIST

1. Injection — SQL, XSS, command, path traversal, template, deserialization
2. Authentication & Session — password hashing, brute force, JWT, OAuth
3. Authorization — broken access control, privilege escalation, missing checks
4. Data Exposure — secrets in logs/source, verbose errors, missing encryption
5. Input Validation — type/length/format, file upload, ReDoS, unsafe deserialization
6. Dependencies — known CVEs, permissive versions, unnecessary packages
7. Configuration — debug mode, CORS, security headers, Docker privileges
8. Business Logic — race conditions, rate limiting, idempotency, TOCTOU
9. Memory Safety — unsafe blocks, unchecked indexing, buffer issues (if applicable)

Respond ONLY with this JSON (no markdown fences, no extra text):
{
  "verdict": "PASS" or "FAIL" or "WARN",
  "risk_level": "critical" or "high" or "medium" or "low" or "none",
  "summary": "one-line security assessment",
  "detected_language": "language detected",
  "vulnerabilities": [
    {
      "severity": "critical" or "high" or "medium" or "low",
      "category": "injection|auth|authorization|data-exposure|input-validation|dependencies|config|business-logic|memory-safety",
      "owasp": "relevant OWASP category if applicable",
      "file": "path/to/file",
      "line": 0,
      "description": "what the vulnerability is and how exploitable",
      "impact": "what an attacker could achieve",
      "remediation": "specific fix with code example",
      "references": ["CWE or doc link"]
    }
  ],
  "positive_findings": ["security best practices observed"],
  "recommendations": ["general hardening suggestions"]
}
GEMINI_PROMPT

gemini -m gemini-3-pro-preview < "$TMPFILE"
rm -f "$TMPFILE"
