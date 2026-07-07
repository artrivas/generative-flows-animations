#!/usr/bin/env bash
# Full from-scratch retrain of all 12 checkpoints (4 distributions x 3
# parametrizations: eps-prediction, v-prediction, flow matching), run on the
# REMOTE GPU SERVER ONLY -- do not run this locally.
#
# Hyperparameters below were read directly from the existing checkpoints'
# saved config (torch.load(...)['config']) and are IDENTICAL across all 12
# of them: steps=4000, batch_size=512, lr=1e-3, seed=0, hidden_dim=128,
# n_layers=4, time_emb_dim=32, t_min=1e-5 (eps/v only), base_std=1.0 (flow
# matching only). This script reproduces that exact recipe for every
# distribution/parametrization so every checkpoint in checkpoints/ is
# consistent with current code.
#
# This is NOT required by any audit finding -- the v_to_score fix and the
# animation fixes made earlier do not need any of these checkpoints
# retrained. Only run this if full-repo retraining-for-consistency was
# explicitly requested.
#
# After each run finishes, copy the resulting checkpoints/*.pt,
# checkpoints/*_losses.csv, and outputs/sanity_checks/*_loss_curve.png back
# into the local repo (they will overwrite the existing checkpoints of the
# same name).

set -euo pipefail

COMMON_SCORE_ARGS="--steps 4000 --batch_size 512 --lr 1e-3 --seed 0 --hidden_dim 128 --n_layers 4 --time_emb_dim 32 --t_min 1e-5 --output-dir checkpoints"
COMMON_FLOW_ARGS="--steps 4000 --batch_size 512 --lr 1e-3 --seed 0 --hidden_dim 128 --n_layers 4 --time_emb_dim 32 --base_std 1.0 --output-dir checkpoints"

DISTRIBUTIONS="eight_gaussians two_moons checkerboard pinwheel"

# --- eps-prediction (4 runs) -----------------------------------------------
# Expected output: checkpoints/<distribution>_eps_seed0.pt (+ _losses.csv)

for dist in $DISTRIBUTIONS; do
    echo "=== eps-prediction: $dist -> checkpoints/${dist}_eps_seed0.pt ==="
    python -m src.training.train_score --distribution "$dist" --param eps $COMMON_SCORE_ARGS
done

# --- v-prediction (4 runs) -------------------------------------------------
# Expected output: checkpoints/<distribution>_v_seed0.pt (+ _losses.csv)

for dist in $DISTRIBUTIONS; do
    echo "=== v-prediction: $dist -> checkpoints/${dist}_v_seed0.pt ==="
    python -m src.training.train_score --distribution "$dist" --param v $COMMON_SCORE_ARGS
done

# --- flow matching (4 runs) -------------------------------------------------
# Expected output: checkpoints/<distribution>_flow_matching_seed0.pt (+ _losses.csv)

for dist in $DISTRIBUTIONS; do
    echo "=== flow matching: $dist -> checkpoints/${dist}_flow_matching_seed0.pt ==="
    python -m src.training.train_flow_matching --distribution "$dist" $COMMON_FLOW_ARGS
done

echo "All 12 runs complete."
