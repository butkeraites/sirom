#!/usr/bin/env bash
# PreToolUse(Bash) guard for the release flow.
#
# Blocks a `git commit` that stages a version change in pyproject.toml
# (a changed `version = "..."` line) without also staging CHANGELOG.md, so a
# version bump always carries its changelog entry in the same commit.
# See memory: version-bump-includes-changelog.
#
# Fails open: any unexpected error allows the commit (a guardrail should never
# wedge the workflow).

cmd=$(jq -r '.tool_input.command // ""' 2>/dev/null)

# Only react to commits; anything else proceeds untouched.
case "$cmd" in
  *"git commit"*) ;;
  *) exit 0 ;;
esac

version_changed=$(git diff --cached -U0 -- pyproject.toml 2>/dev/null | grep -E '^\+version = ')
changelog_staged=$(git diff --cached --name-only 2>/dev/null | grep -E '(^|/)CHANGELOG\.md$')

if [ -n "$version_changed" ] && [ -z "$changelog_staged" ]; then
  printf '%s\n' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"deny","permissionDecisionReason":"Version bump must include the CHANGELOG.md entry in the same commit (see memory: version-bump-includes-changelog)."}}'
fi
exit 0
