"""
Validate step-7 samplers visually and with a lightweight distance metric.

Usage:
    python scripts/validate_samplers.py --checkpoint checkpoints/eight_gaussians_eps_seed0.pt
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
from src.sampling import load_score_wrapper, sample_pf_ode, sample_reverse_sde
from src.sampling.utils import default_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/eight_gaussians_eps_seed0.pt")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_metric", type=int, default=1000)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--ode_method", choices=["euler", "heun"], default="heun")
    p.add_argument("--axis_limit", type=float, default=6.5)
    return p.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray) -> float:
    """
    Multivariate energy distance:
        2 E||X-Y|| - E||X-X'|| - E||Y-Y'||
    computed with empirical pairwise distances.
    """
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    xy = torch.cdist(x_t, y_t).mean()
    xx = torch.cdist(x_t, x_t).mean()
    yy = torch.cdist(y_t, y_t).mean()
    return float(2 * xy - xx - yy)


def subsample_for_metric(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=n, replace=False)
    return x[idx]


def main():
    args = parse_args()
    device = default_device()
    checkpoint_path = resolve_project_path(args.checkpoint)
    wrapper, process, config = load_score_wrapper(checkpoint_path, device=device)

    dist_name = config["distribution"]
    param = config["param"]
    sample_fn = get_distribution(dist_name)
    real = sample_fn(args.n_samples, seed=args.seed + 101)

    print(f"device={device} checkpoint={checkpoint_path}")
    print(f"distribution={dist_name} param={param} steps={args.steps} seed={args.seed}")

    reverse = sample_reverse_sde(
        wrapper,
        process,
        n_samples=args.n_samples,
        n_steps=args.steps,
        seed=args.seed,
        device=device,
    ).detach().cpu().numpy()
    pf = sample_pf_ode(
        wrapper,
        process,
        n_samples=args.n_samples,
        n_steps=args.steps,
        seed=args.seed,
        method=args.ode_method,
        device=device,
    ).detach().cpu().numpy()

    real_m = subsample_for_metric(real, args.n_metric, args.seed + 1)
    reverse_m = subsample_for_metric(reverse, args.n_metric, args.seed + 2)
    pf_m = subsample_for_metric(pf, args.n_metric, args.seed + 3)
    reverse_energy = energy_distance_multivariate(reverse_m, real_m)
    pf_energy = energy_distance_multivariate(pf_m, real_m)

    out_dir = PROJECT_ROOT / "outputs" / "sanity_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sampler_comparison_{dist_name}_{param}.png"

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), dpi=130)
    panels = [
        ("real data", real, None),
        (f"reverse-SDE\nenergy={reverse_energy:.4f}", reverse, "#C44E52"),
        (f"PF-ODE ({args.ode_method})\nenergy={pf_energy:.4f}", pf, "#55A868"),
    ]
    for ax, (title, pts, color) in zip(axes, panels):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=5,
            alpha=0.45,
            color=color or "#4C72B0",
            linewidths=0,
        )
        ax.set_xlim(-args.axis_limit, args.axis_limit)
        ax.set_ylim(-args.axis_limit, args.axis_limit)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11)
        ax.grid(True, linewidth=0.3, alpha=0.35)

    fig.suptitle(f"Sampler validation - {dist_name}/{param}, N={args.steps}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")
    print(f"energy_distance(reverse_sde, real) = {reverse_energy:.6f}")
    print(f"energy_distance(pf_ode, real)      = {pf_energy:.6f}")


if __name__ == "__main__":
    main()
