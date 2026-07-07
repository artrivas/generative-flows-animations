"""
Structure-sensitive evaluation for pinwheel checkpoints.

Complements scripts/validate_samplers.py (energy distance only) with the
angle-aware metrics in src/eval/pinwheel_structure.py, since energy distance
is documented (docs/technical.md §9.4) as blind to pinwheel collapsing into
an isotropic blob. Reused across every pinwheel checkpoint candidate so
results are directly comparable.

Usage:
    python scripts/evaluate_pinwheel_structure.py --checkpoint checkpoints/pinwheel_eps_seed0.pt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.registry import get_distribution
from src.eval.pinwheel_structure import (
    DEFAULT_RADIAL_TOL,
    angular_histogram,
    angular_histogram_distance,
    arm_assignment_accuracy,
    estimate_chance_level,
)
from src.sampling import load_score_wrapper, sample_pf_ode, sample_reverse_sde
from src.sampling.utils import default_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/pinwheel_eps_seed0.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--ode_method", choices=["euler", "heun"], default="heun")
    p.add_argument("--axis_limit", type=float, default=6.5)
    return p.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray) -> float:
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    return float(2 * torch.cdist(x_t, y_t).mean() - torch.cdist(x_t, x_t).mean() - torch.cdist(y_t, y_t).mean())


def main():
    args = parse_args()
    device = default_device()
    checkpoint_path = resolve_project_path(args.checkpoint)
    wrapper, process, config = load_score_wrapper(checkpoint_path, device=device)
    dist_name, param = config["distribution"], config["param"]
    if dist_name != "pinwheel":
        print(f"WARNING: checkpoint distribution is {dist_name!r}, not 'pinwheel' -- "
              f"arm-based metrics assume the pinwheel geometry and will not be meaningful.")

    sample_fn = get_distribution(dist_name)
    real = sample_fn(args.n_samples, seed=args.seed + 101)

    reverse = sample_reverse_sde(
        wrapper, process, n_samples=args.n_samples, n_steps=args.steps, seed=args.seed, device=device,
    ).detach().cpu().numpy()
    pf = sample_pf_ode(
        wrapper, process, n_samples=args.n_samples, n_steps=args.steps, seed=args.seed,
        method=args.ode_method, device=device,
    ).detach().cpu().numpy()

    real_acc, _ = arm_assignment_accuracy(real)
    reverse_acc, reverse_resid = arm_assignment_accuracy(reverse)
    pf_acc, pf_resid = arm_assignment_accuracy(pf)

    reverse_energy = energy_distance_multivariate(reverse, real)
    pf_energy = energy_distance_multivariate(pf, real)
    reverse_ang_dist = angular_histogram_distance(reverse, real)
    pf_ang_dist = angular_histogram_distance(pf, real)

    chance_level = estimate_chance_level()
    print(f"checkpoint={checkpoint_path.name}  distribution={dist_name}  param={param}")
    print(f"chance-level arm accuracy (isotropic baseline) = {chance_level:.3f}")
    print(f"real data arm accuracy (calibration, should be near 1.0) = {real_acc:.3f}")
    print("-" * 70)
    print(f"{'sampler':<14}{'energy_dist':>14}{'arm_accuracy':>16}{'ang_hist_dist':>16}")
    print(f"{'reverse-SDE':<14}{reverse_energy:>14.4f}{reverse_acc:>16.3f}{reverse_ang_dist:>16.3f}")
    print(f"{'PF-ODE':<14}{pf_energy:>14.4f}{pf_acc:>16.3f}{pf_ang_dist:>16.3f}")

    out_dir = PROJECT_ROOT / "outputs" / "sanity_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pinwheel_structure_{checkpoint_path.stem}.png"

    fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), dpi=130)
    panels = [
        ("reverse-SDE", reverse, reverse_resid, reverse_acc, reverse_energy),
        ("PF-ODE", pf, pf_resid, pf_acc, pf_energy),
    ]
    for row, (label, samples, resid, acc, energy) in enumerate(panels):
        ax_scatter = axes[row, 0]
        passed = resid < DEFAULT_RADIAL_TOL
        ax_scatter.scatter(real[:, 0], real[:, 1], s=4, alpha=0.15, color="#808080", linewidths=0, label="real")
        ax_scatter.scatter(samples[~passed, 0], samples[~passed, 1], s=5, alpha=0.5, color="#C44E52", linewidths=0, label="off-arm")
        ax_scatter.scatter(samples[passed, 0], samples[passed, 1], s=5, alpha=0.5, color="#55A868", linewidths=0, label="on-arm")
        ax_scatter.set_xlim(-args.axis_limit, args.axis_limit)
        ax_scatter.set_ylim(-args.axis_limit, args.axis_limit)
        ax_scatter.set_aspect("equal")
        ax_scatter.set_title(f"{label}  arm_acc={acc:.3f}  energy={energy:.4f}", fontsize=11)
        ax_scatter.legend(fontsize=7, loc="upper right", markerscale=2)
        ax_scatter.grid(True, linewidth=0.3, alpha=0.35)

        ax_hist = axes[row, 1]
        bins = np.linspace(-np.pi, np.pi, 37)
        centers = (bins[:-1] + bins[1:]) / 2
        ax_hist.plot(centers, angular_histogram(real), color="#4C72B0", label="real", linewidth=1.5)
        ax_hist.plot(centers, angular_histogram(samples), color="#C44E52", label=label, linewidth=1.5)
        ax_hist.set_xlabel("theta (rad)")
        ax_hist.set_ylabel("density")
        ax_hist.set_title(f"angular histogram  (TV dist={angular_histogram_distance(samples, real):.3f})", fontsize=11)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, linewidth=0.3, alpha=0.35)

    fig.suptitle(f"Pinwheel structure diagnostics — {checkpoint_path.name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
