"""
Qualitative validation for a trained Flow Matching velocity field.

Usage:
    python scripts/validate_velocity_field.py --checkpoint checkpoints/eight_gaussians_flow_matching_seed0.pt
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
from src.models.velocity_field import VelocityField

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--t", type=float, default=0.5)
    p.add_argument("--grid_size", type=int, default=22)
    p.add_argument("--grid_limit", type=float, default=6.0)
    p.add_argument("--n_particles", type=int, default=1200)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main():
    args = parse_args()
    ckpt_path = resolve_project_path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    config = ckpt["config"]

    model = VelocityField(
        data_dim=2,
        time_emb_dim=config["time_emb_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dist_name = config["distribution"]
    sample_fn = get_distribution(dist_name)
    x_data = torch.from_numpy(sample_fn(args.n_particles, seed=args.seed)).float().to(DEVICE)
    torch.manual_seed(args.seed)
    x_base = torch.randn_like(x_data) * config.get("base_std", 1.0)
    t_particles = torch.full((args.n_particles, 1), args.t, device=DEVICE)
    x_interp = (1.0 - t_particles) * x_base + t_particles * x_data

    xs = np.linspace(-args.grid_limit, args.grid_limit, args.grid_size)
    ys = np.linspace(-args.grid_limit, args.grid_limit, args.grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.from_numpy(np.stack([XX.ravel(), YY.ravel()], axis=1)).float().to(DEVICE)
    t_grid = torch.full((grid.shape[0],), args.t, device=DEVICE)

    with torch.no_grad():
        velocity = model(grid, t_grid).cpu().numpy()
    U = velocity[:, 0].reshape(XX.shape)
    V = velocity[:, 1].reshape(XX.shape)
    mag = np.sqrt(U ** 2 + V ** 2)
    U_norm = U / (mag + 1e-8)
    V_norm = V / (mag + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=130)
    pts = x_interp.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.25, color="#4C72B0", label=f"interpolated particles (t={args.t})")
    q = ax.quiver(XX, YY, U_norm, V_norm, mag, cmap="viridis", scale=25, width=0.0035)
    fig.colorbar(q, ax=ax, label="|velocity|", shrink=0.8)
    ax.set_xlim(-args.grid_limit, args.grid_limit)
    ax.set_ylim(-args.grid_limit, args.grid_limit)
    ax.set_aspect("equal")
    ax.set_title(f"Flow Matching velocity field - {dist_name}, t={args.t}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "outputs" / "sanity_checks" / f"velocity_field_{dist_name}_flow_matching.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
