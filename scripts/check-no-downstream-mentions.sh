#!/usr/bin/env bash
# emc2p is meant to be fully generic -- no downstream project's own name
# or domain concepts belong in its code, tests, or docs (see
# emc2p/mcp_server.py's own module docstring: comparisons like "a caller
# like X computes it from its own Y" should describe the shape of what a
# downstream project does, never name one). docs/manifest/history.yaml is
# the one deliberate exception: a non-actionable historical record, which
# by its own nature sometimes needs to name what a past design decision
# was actually a reaction to.
#
# Scoped to newly staged lines (git diff --cached), not the whole tree:
# core.hooksPath wasn't applied automatically in any session until this
# hook's own SessionStart wiring was added, so this check had never
# actually run against real commits, and the tree had already
# accumulated pre-existing mentions elsewhere by the time it was first
# enabled for real. A whole-tree scan (the original git grep --cached
# version) would fail every future commit on that pre-existing debt
# regardless of what it touches; cleaning that debt up is separate work,
# not something this hook should block on indefinitely.
#
# Usage: check-no-downstream-mentions.sh --staged
set -euo pipefail

if [ "${1:-}" != "--staged" ]; then
  echo "usage: check-no-downstream-mentions.sh --staged" >&2
  exit 2
fi

# Broad enough to catch "story-sim", "story_sim", "storysim", and (as a
# substring) "story-simulator"/"story_simulator" in any casing.
pattern='story[_-]?sim'

allowed_path="docs/manifest/history.yaml"
self_path="scripts/check-no-downstream-mentions.sh"

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

flagged=""
while IFS= read -r f; do
  [[ -z "$f" || "$f" == "$allowed_path" || "$f" == "$self_path" ]] && continue
  added=$(git diff --cached -- "$f" | grep -E '^\+' | grep -vE '^\+\+\+' | grep -iE "$pattern" || true)
  if [[ -n "$added" ]]; then
    flagged+="$f:"$'\n'"$added"$'\n'
  fi
done < <(git diff --cached --name-only 2>/dev/null || true)

if [[ -n "$flagged" ]]; then
  echo "Found a newly staged mention of the downstream project this check exists to keep out of emc2p:"
  echo
  echo "$flagged"
  echo "emc2p is meant to be usable by any downstream project, so its own code/tests/docs"
  echo "shouldn't name one specifically -- rephrase generically instead (e.g. \"a downstream"
  echo "project's own X\" rather than naming it). The one exception is $allowed_path,"
  echo "a non-actionable historical record where naming what a past decision was a reaction"
  echo "to is often unavoidable -- move the mention there instead if that's genuinely what"
  echo "this is."
  exit 1
fi

exit 0
