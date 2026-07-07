"""
Post-fix demonstration: v-prediction vs eps-prediction sample quality,
side by side, for the same distribution/sampler/step budget.

Generated after fixing the v_to_score bug in src/models/denoiser.py
(score = -(alpha*v + std*x_t)/std, one power of std, not squared).

Usage:
    python scripts/compare_v_vs_eps.py --distribution eight_gaussians
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
from src.sampling import load_score_wrapper, sample_pf_ode


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray) -> float:
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    xy = torch.cdist(x_t, y_t).mean()
    xx = torch.cdist(x_t, x_t).mean()
    yy = torch.cdist(y_t, y_t).mean()
    return float(2 * xy - xx - yy)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--distributions", nargs="+", default=["eight_gaussians", "two_moons"])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--axis_limit", type=float, default=6.5)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")

    n_dist = len(args.distributions)
    fig, axes = plt.subplots(n_dist, 3, figsize=(15.5, 5.2 * n_dist), dpi=130)
    if n_dist == 1:
        axes = axes[None, :]

    for row, dist_name in enumerate(args.distributions):
        real = get_distribution(dist_name)(args.n_samples, seed=args.seed + 101)

        eps_ckpt = PROJECT_ROOT / "checkpoints" / f"{dist_name}_eps_seed0.pt"
        v_ckpt = PROJECT_ROOT / "checkpoints" / f"{dist_name}_v_seed0.pt"

        wrapper_eps, process_eps, _ = load_score_wrapper(eps_ckpt, device=device)
        wrapper_v, process_v, _ = load_score_wrapper(v_ckpt, device=device)

        eps_samples = sample_pf_ode(
            wrapper_eps, process_eps, n_samples=args.n_samples, n_steps=args.steps,
            seed=args.seed, method="heun", device=device,
        ).detach().cpu().numpy()
        v_samples = sample_pf_ode(
            wrapper_v, process_v, n_samples=args.n_samples, n_steps=args.steps,
            seed=args.seed, method="heun", device=device,
        ).detach().cpu().numpy()

        eps_energy = energy_distance_multivariate(eps_samples, real)
        v_energy = energy_distance_multivariate(v_samples, real)

        panels = [
            ("real data", real, "#4C72B0"),
            (f"eps-pred (PF-ODE)\nenergy={eps_energy:.4f}", eps_samples, "#C44E52"),
            (f"v-pred (PF-ODE, fixed)\nenergy={v_energy:.4f}", v_samples, "#55A868"),
        ]
        for ax, (title, pts, color) in zip(axes[row], panels):
            ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.45, color=color, linewidths=0)
            ax.set_xlim(-args.axis_limit, args.axis_limit)
            ax.set_ylim(-args.axis_limit, args.axis_limit)
            ax.set_aspect("equal")
            ax.set_title(title, fontsize=11)
            ax.grid(True, linewidth=0.3, alpha=0.35)
        axes[row][0].set_ylabel(dist_name, fontsize=12)

        print(f"{dist_name}: eps_energy={eps_energy:.6f} v_energy={v_energy:.6f}")

    fig.suptitle("v-pred vs eps-pred sample quality (post-fix, PF-ODE)", fontsize=14)
    fig.tight_layout()
    out_path = PROJECT_ROOT / "outputs" / "sanity_checks" / "v_vs_eps_comparison.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
