#!/bin/bash
set -euo pipefail

# .git/config is per-clone and doesn't survive a fresh container -- point
# git at the repo-tracked hooks dir on every boot so .githooks/* run.
git config core.hooksPath .githooks
