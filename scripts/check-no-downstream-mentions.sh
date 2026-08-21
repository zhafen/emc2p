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

matches=$(git grep --cached --line-number --ignore-case --extended-regexp "$pattern" \
  -- . ":(exclude)$allowed_path" ":(exclude)$self_path" \
  2>/dev/null || true)

if [ -n "$matches" ]; then
  echo "Found a mention of the downstream project this check exists to keep out of emc2p:"
  echo
  echo "$matches"
  echo
  echo "emc2p is meant to be usable by any downstream project, so its own code/tests/docs"
  echo "shouldn't name one specifically -- rephrase generically instead (e.g. \"a downstream"
  echo "project's own X\" rather than naming it). The one exception is $allowed_path,"
  echo "a non-actionable historical record where naming what a past decision was a reaction"
  echo "to is often unavoidable -- move the mention there instead if that's genuinely what"
  echo "this is."
  exit 1
fi

exit 0
