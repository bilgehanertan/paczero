#!/bin/bash
# setup.sh — clone the DP-AggZO baseline repository pinned to the commit
# the paper's in-house DP-cliff reproductions were run against.
#
#  Run this script once after cloning, before invoking
# any script under scripts/baselines/.
#
# if the target directory already exists at the pinned commit,
# the script exits silently. Re-running is safe.

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$BASE_DIR/baselines/dp-aggzo"
UPSTREAM_URL="https://github.com/erguteb/dp-aggzo.git"
PINNED_COMMIT="99f64026cf744be804da3dedfe025a6e199df82f"

mkdir -p "$BASE_DIR/baselines"

if [[ -d "$TARGET/.git" ]]; then
  current=$(git -C "$TARGET" rev-parse HEAD 2>/dev/null || echo "")
  if [[ "$current" == "$PINNED_COMMIT" ]]; then
    echo "[setup] baselines/dp-aggzo already at pinned commit; nothing to do."
    exit 0
  fi
  echo "[setup] baselines/dp-aggzo exists at $current; resetting to $PINNED_COMMIT ..."
  git -C "$TARGET" fetch origin
  git -C "$TARGET" checkout "$PINNED_COMMIT"
  echo "[setup] reset to pinned commit."
  exit 0
fi

echo "[setup] cloning DP-AggZO baseline from $UPSTREAM_URL ..."
git clone "$UPSTREAM_URL" "$TARGET"
git -C "$TARGET" checkout "$PINNED_COMMIT"
echo "[setup] dp-aggzo cloned at commit $PINNED_COMMIT"
echo "[setup] done. scripts/baselines/* are now runnable."
