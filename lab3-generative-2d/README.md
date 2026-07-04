# Lab 3 - Generative 2D Visualizer

Visual lab for 2D diffusion models and Flow Matching. It includes synthetic
datasets, VP/VE/sub-VP forward processes, denoising score models, reverse-time
SDE sampling, Probability Flow ODE sampling, Flow Matching velocity models, and
the eight required animations.

## Install

From a clean clone:

```bash
cd lab3-generative-2d
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

All commands below assume they are run from `lab3-generative-2d/`.

## Main CLI

Train a score model:

```bash
python scripts/main.py train --kind score --distribution eight_gaussians --model eps_pred --steps 2000
```

Train a Flow Matching model:

```bash
python scripts/main.py train --kind flow_matching --distribution eight_gaussians --steps 2000
```

Generate samples:

```bash
python scripts/main.py sample --distribution eight_gaussians --model eps_pred --sampler pf_ode --steps 250 --seed 0
python scripts/main.py sample --distribution eight_gaussians --model flow_matching --sampler flow_matching_ode --steps 250 --seed 0
```

Use a config:

```bash
python scripts/main.py train --config configs/train_score_eight_gaussians_eps.yaml
python scripts/main.py train --config configs/train_flow_matching_eight_gaussians.yaml
```

## Available Names

Distributions:

| Name | Type |
|---|---|
| `eight_gaussians` | multimodal |
| `two_moons` | curved support |
| `checkerboard` | disconnected regions |
| `pinwheel` | complex geometry |
| `two_spirals` | extra curved/disconnected geometry |

Models:

| Name | Checkpoint pattern |
|---|---|
| `eps_pred` | `checkpoints/<distribution>_eps_seed0.pt` |
| `v_pred` | `checkpoints/<distribution>_v_seed0.pt` |
| `flow_matching` | `checkpoints/<distribution>_flow_matching_seed0.pt` |

Samplers:

| Name | Model |
|---|---|
| `reverse_sde` | `eps_pred` or `v_pred` |
| `pf_ode` | `eps_pred` or `v_pred` |
| `flow_matching_ode` | `flow_matching` |

## Regenerate Animations

```bash
python scripts/main.py animate --which forward_comparison --distribution eight_gaussians
python scripts/main.py animate --which density_evolution --distribution eight_gaussians
python scripts/main.py animate --which forward_trajectories --distribution eight_gaussians
python scripts/main.py animate --which score_field --distribution eight_gaussians --model eps_pred
python scripts/main.py animate --which reverse_sde --distribution eight_gaussians --model eps_pred --steps 250
python scripts/main.py animate --which pf_ode --distribution eight_gaussians --model eps_pred --steps 250
python scripts/main.py animate --which flow_matching --distribution eight_gaussians --steps 250
python scripts/main.py animate --which steps_comparison --distribution eight_gaussians
```

Videos are written to `outputs/videos/`; keyframe PNGs and validation plots are
written to `outputs/sanity_checks/`.

## Validation Scripts

```bash
python scripts/check_distributions.py
python scripts/validate_forward_process.py
python scripts/validate_integrators.py
python scripts/validate_score_field.py --checkpoint checkpoints/eight_gaussians_eps_seed0.pt
python scripts/validate_samplers.py --checkpoint checkpoints/eight_gaussians_eps_seed0.pt --steps 500
python scripts/validate_velocity_field.py --checkpoint checkpoints/eight_gaussians_flow_matching_seed0.pt
```

`scripts/train_all_score_models.sh` contains the 4 distributions x 2 score
parametrizations run list for Colab/GPU. `scripts/train_all_flow_matching.sh`
contains the 4 Flow Matching runs. Both trainers use `cuda` when available and
fall back to CPU.

## Repo Structure

```text
lab3-generative-2d/
  configs/              example train/sample configs
  references/NOTES.md   equations, citations, and implementation map
  src/data/             synthetic 2D distributions and registry
  src/forward_process/  VP, VE, sub-VP SDEs and closed-form kernels
  src/integrators/      Euler, Heun, Euler-Maruyama
  src/models/           denoiser and velocity-field MLPs
  src/training/         score and Flow Matching training loops
  src/sampling/         reverse-SDE, PF-ODE, and Flow Matching samplers
  src/viz/              animation modules
  scripts/              CLI, training launchers, validation scripts
  checkpoints/          model checkpoints
  outputs/videos/       generated MP4 animations
  outputs/sanity_checks/generated validation plots and keyframes
```

## Notes

The score checkpoints for the 4 distributions x `{eps,v}` are expected under
`checkpoints/`. The included Flow Matching checkpoint is a short CPU validation
run for `eight_gaussians`; use `scripts/train_all_flow_matching.sh` in Colab for
full convergence across all four distributions.
