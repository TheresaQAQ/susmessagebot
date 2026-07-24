#!/usr/bin/env bash
# Run 100-sample eval across candidate free SiliconFlow models.
# Usage:
#   ./run_model_bakeoff.sh
#   ./run_model_bakeoff.sh v2_zh_balanced 1
#   PROMPT_VERSION=v2_zh_balanced RUN=1 ./run_model_bakeoff.sh
set -euo pipefail
cd "$(dirname "$0")"

PROMPT_VERSION="${1:-${PROMPT_VERSION:-v2_zh_balanced}}"
RUN="${2:-${RUN:-1}}"

MODELS=(
  "Qwen/Qwen3-8B"
  "Qwen/Qwen3.5-9B"
  "THUDM/GLM-4-9B-0414"
  "Qwen/Qwen2.5-7B-Instruct"
)

OUT_DIR="eval_results/prompt_${PROMPT_VERSION}/run${RUN}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/bakeoff.log"

echo "Prompt=$PROMPT_VERSION Run=$RUN Out=$OUT_DIR" | tee "$LOG"

for model in "${MODELS[@]}"; do
  echo "======== START $model ========" | tee -a "$LOG"
  python eval_accuracy.py \
    --prompt-version "$PROMPT_VERSION" \
    --run "$RUN" \
    --model "$model" \
    --sleep 0.2 \
    2>&1 | tee -a "$LOG"
  echo "======== DONE $model ========" | tee -a "$LOG"
done

echo "All done. See $OUT_DIR/comparison.md and eval_results/INDEX.md" | tee -a "$LOG"
