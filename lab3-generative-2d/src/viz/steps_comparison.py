"""
Animation/Figure 8: compare final sample quality as integration steps N vary.

Creates a 2 x len(N) grid:
    row 1: Probability Flow ODE
    row 2: Flow Matching ODE

The PNG is the main inspection artifact. A short static MP4 with the same
grid is also saved so outputs/videos contains an artifact for animation 8.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.registry import get_distribution
from src.sampling import (
    load_score_wrapper,
    load_velocity_model,
    sample_flow_matching_ode,
    sample_pf_ode,
)
from src.sampling.utils import default_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--score_checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "eight_gaussians_eps_seed0.pt"))
    p.add_argument("--flow_checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "eight_gaussians_flow_matching_seed0.pt"))
    p.add_argument("--steps", default="10,25,50,100,250")
    p.add_argument("--n_samples", type=int, default=1200)
    p.add_argument("--n_metric", type=int, default=600)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--pf_method", choices=["euler", "heun"], default="heun")
    p.add_argument("--fm_method", choices=["euler", "heun"], default="heun")
    p.add_argument("--axis_limit", type=float, default=6.5)
    return p.parse_args()


def parse_steps(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values or any(v <= 0 for v in values):
        raise ValueError("--steps must contain positive integers")
    return values


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def energy_distance_multivariate(x: np.ndarray, y: np.ndarray) -> float:
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()
    return float(2 * torch.cdist(x_t, y_t).mean() - torch.cdist(x_t, x_t).mean() - torch.cdist(y_t, y_t).mean())


def subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=n, replace=False)]


def validate_samples(label: str, samples: np.ndarray, axis_limit: float):
    if not np.isfinite(samples).all():
        raise RuntimeError(f"{label} produced NaN or Inf values")
    max_abs = float(np.max(np.abs(samples)))
    print(f"{label}: max_abs={max_abs:.3f}")
    if max_abs > axis_limit * 5:
        raise RuntimeError(f"{label} exploded numerically: max_abs={max_abs:.3f}")


def quality_warning(name: str, metrics: list[float]):
    tolerance = 0.005
    stable_or_improves = all(metrics[i + 1] <= metrics[i] + tolerance for i in range(len(metrics) - 1))
    if stable_or_improves:
        print(f"{name}: energy distance is monotone or stable within +/-{tolerance}.")
    else:
        print(f"WARNING: {name} energy distance worsens beyond +/-{tolerance}; inspect the grid visually.")


def main():
    args = parse_args()
    step_counts = parse_steps(args.steps)
    device = default_device()

    score_wrapper, score_process, score_config = load_score_wrapper(resolve_project_path(args.score_checkpoint), device=device)
    velocity_model, flow_config = load_velocity_model(resolve_project_path(args.flow_checkpoint), device=device)
    dist_name = flow_config["distribution"]
    if score_config["distribution"] != dist_name:
        print(f"WARNING: score checkpoint distribution={score_config['distribution']} but flow checkpoint distribution={dist_name}")

    sample_fn = get_distribution(dist_name)
    real = sample_fn(args.n_samples, seed=args.seed + 1000)
    real_m = subsample(real, args.n_metric, args.seed)
    base_std = flow_config.get("base_std", 1.0)

    pf_samples = []
    fm_samples = []
    pf_metrics = []
    fm_metrics = []

    for n_steps in step_counts:
        pf = sample_pf_ode(
            score_wrapper,
            score_process,
            n_samples=args.n_samples,
            n_steps=n_steps,
            seed=args.seed,
            method=args.pf_method,
            device=device,
        ).detach().cpu().numpy()
        fm = sample_flow_matching_ode(
            velocity_model,
            n_samples=args.n_samples,
            n_steps=n_steps,
            seed=args.seed,
            method=args.fm_method,
            base_std=base_std,
            device=device,
        ).detach().cpu().numpy()

        validate_samples(f"PF-ODE N={n_steps}", pf, args.axis_limit)
        validate_samples(f"Flow Matching N={n_steps}", fm, args.axis_limit)

        pf_samples.append(pf)
        fm_samples.append(fm)
        pf_metrics.append(energy_distance_multivariate(subsample(pf, args.n_metric, args.seed + 123), real_m))
        fm_metrics.append(energy_distance_multivariate(subsample(fm, args.n_metric, args.seed + 456), real_m))

    quality_warning("PF-ODE", pf_metrics)
    quality_warning("Flow Matching", fm_metrics)

    fig, axes = plt.subplots(2, len(step_counts), figsize=(3.25 * len(step_counts), 6.8), dpi=130)
    rows = [
        ("PF-ODE", pf_samples, pf_metrics, "#55A868"),
        ("Flow Matching", fm_samples, fm_metrics, "#C44E52"),
    ]
    for row_idx, (row_name, samples_by_n, metrics, color) in enumerate(rows):
        for col_idx, (n_steps, samples, metric) in enumerate(zip(step_counts, samples_by_n, metrics)):
            ax = axes[row_idx, col_idx]
            ax.scatter(samples[:, 0], samples[:, 1], s=4, alpha=0.42, color=color, linewidths=0)
            ax.set_xlim(-args.axis_limit, args.axis_limit)
            ax.set_ylim(-args.axis_limit, args.axis_limit)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"N={n_steps}\nenergy={metric:.3f}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=11)

    fig.suptitle(f"Step-count comparison - {dist_name}", fontsize=14)
    fig.tight_layout()

    sanity_dir = PROJECT_ROOT / "outputs" / "sanity_checks"
    video_dir = PROJECT_ROOT / "outputs" / "videos"
    sanity_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    png_path = sanity_dir / f"steps_comparison_{dist_name}.png"
    mp4_path = video_dir / "steps_comparison.mp4"
    fig.savefig(png_path, dpi=130, bbox_inches="tight")

    def update(_frame):
        return []

    anim = animation.FuncAnimation(fig, update, frames=48, blit=False)
    anim.save(mp4_path, writer="ffmpeg", fps=24, dpi=130)
    plt.close(fig)

    print(f"Saved {png_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved {mp4_path.relative_to(PROJECT_ROOT)}")
    print("PF-ODE energy by N:", dict(zip(step_counts, [round(v, 6) for v in pf_metrics])))
    print("Flow Matching energy by N:", dict(zip(step_counts, [round(v, 6) for v in fm_metrics])))


if __name__ == "__main__":
    main()
