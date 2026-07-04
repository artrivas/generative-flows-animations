#!/usr/bin/env bash
# Train all 4 Flow Matching velocity models to real convergence.
#
# Intended to run in Colab on a T4 GPU or any CUDA machine. The trainer uses
# cuda when available and falls back to CPU without code changes.
#
# Usage (from lab3-generative-2d/):
#   bash scripts/train_all_flow_matching.sh

set -euo pipefail

DISTRIBUTIONS=("eight_gaussians" "two_moons" "checkerboard" "pinwheel")
SEED=0
STEPS=30000
BATCH_SIZE=512
LR=1e-3
OUTPUT_DIR="checkpoints"

for dist in "${DISTRIBUTIONS[@]}"; do
  echo "=== Training Flow Matching distribution=${dist} ==="
  python -m src.training.train_flow_matching \
    --distribution "${dist}" \
    --steps "${STEPS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --log_every 1000 \
    --output-dir "${OUTPUT_DIR}"
done

echo "=== All 4 Flow Matching runs complete. Checkpoints in ${OUTPUT_DIR}/ ==="
