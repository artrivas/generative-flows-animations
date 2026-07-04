#!/usr/bin/env bash
# Train all 8 score models (4 distributions x {eps, v}) to real convergence.
#
# Intended to run in Colab on a T4 GPU (see colab_train_score.ipynb) or any
# machine with a CUDA GPU -- NOT locally in this environment (no GPU here,
# see docs/technical.md, paso 5b). train_score.py's device is agnostic
# (cuda if available, else cpu), so this script runs unmodified either way.
#
# Usage (from lab3-generative-2d/):
#   bash scripts/train_all_score_models.sh

set -euo pipefail

DISTRIBUTIONS=("eight_gaussians" "two_moons" "checkerboard" "pinwheel")
PARAMS=("eps" "v")
SEED=0
STEPS=30000
BATCH_SIZE=512
LR=1e-3
OUTPUT_DIR="checkpoints"

for dist in "${DISTRIBUTIONS[@]}"; do
  for param in "${PARAMS[@]}"; do
    echo "=== Training distribution=${dist} param=${param} ==="
    python -m src.training.train_score \
      --distribution "${dist}" \
      --param "${param}" \
      --steps "${STEPS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --seed "${SEED}" \
      --log_every 1000 \
      --output-dir "${OUTPUT_DIR}"
  done
done

echo "=== All 8 runs complete. Checkpoints in ${OUTPUT_DIR}/ ==="
