#!/usr/bin/env bash
# Build a runnable, self-contained copy of the model code exactly as it was used for one
# experiment: the canonical model tree in ./model plus that experiment's as-run
# overrides (experiments/<exp>/model_overrides/, mirroring paths under ./model).
#
# Usage: tools/assemble_experiment.sh experiments/<exp_dir> <work_dir>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
EXP="$(cd "$1" && pwd)"; WORK="$2"
[ -e "$WORK" ] && { echo "refusing to overwrite existing $WORK" >&2; exit 1; }
mkdir -p "$WORK"
cp -r "$REPO/model/." "$WORK/"
if [ -d "$EXP/model_overrides" ]; then
  cp -r "$EXP/model_overrides/." "$WORK/"
  echo "Applied overrides:"; (cd "$EXP/model_overrides" && find . -type f | sed 's|^\./|  |')
else
  echo "No overrides: experiment uses ./model unchanged."
fi
# experiment drivers / job scripts next to the model tree
find "$EXP" -maxdepth 1 -type f -exec cp {} "$WORK/" \;
[ -d "$EXP/slurm" ] && cp -r "$EXP/slurm" "$WORK/"
echo "Assembled $(basename "$EXP") -> $WORK"
