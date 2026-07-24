#!/usr/bin/env bash
# Run bakeoff for multiple run numbers.
# Usage: ./scripts/run_multi_bakeoff.sh <prompt_version> <start_run> <end_run>
set -euo pipefail
cd "$(dirname "$0")/.."

PROMPT_VERSION="${1:?prompt version required}"
START_RUN="${2:?start run required}"
END_RUN="${3:?end run required}"

for run in $(seq "$START_RUN" "$END_RUN"); do
  echo "############ BAKEOFF ${PROMPT_VERSION} run${run} ############"
  ./scripts/run_model_bakeoff.sh "$PROMPT_VERSION" "$run"
done

python3 -m scripts.summarize_runs --prompt-version "$PROMPT_VERSION" --runs "$START_RUN-$END_RUN"
echo "Multi-run done."
