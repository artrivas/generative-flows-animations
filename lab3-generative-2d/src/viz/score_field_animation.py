"""
Animation 4: evolution of the learned score field as t sweeps 0 -> 1.

Loads a trained checkpoint and evaluates its score (via DenoiserWrapper,
using the exact eps/v->score conversion from NOTES.md §5/§6) on a FIXED
spatial grid at each t. Particles are overlaid at their x_t position,
using the same fixed-per-particle-noise trick as the other animations for
a smooth, non-flickering overlay.

Expected qualitative behavior: near t=0 the field should be clearly
structured around the data modes; near t=1 it should look close to a
uniform "pull toward the origin" field, since eps/v-prediction at t~1 must
approximate x_t itself (the process has erased the data signal by then) --
score = -eps/std ~ -x_t/std.

Usage:
    python -m src.viz.score_field_animation --checkpoint checkpoints/eight_gaussians_eps_seed0.pt
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
from src.forward_process import VPSDE
from src.models.denoiser import Denoiser, DenoiserWrapper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYFRAME_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"
OUT_DIR = PROJECT_ROOT / "outputs" / "videos"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "checkpoints" / "eight_gaussians_eps_seed0.pt"))
    p.add_argument("--n_frames", type=int, default=100)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--grid_size", type=int, default=22)
    p.add_argument("--grid_limit", type=float, default=6.0)
    p.add_argument("--n_particles", type=int, default=800)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--t_min", type=float, default=0.02)
    return p.parse_args()


def load_wrapper(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    config = ckpt["config"]
    model = Denoiser(
        data_dim=2,
        time_emb_dim=config["time_emb_dim"],
        hidden_dim=config["hidden_dim"],
        n_layers=config["n_layers"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    process = VPSDE()
    wrapper = DenoiserWrapper(model, process, param=config["param"])
    return wrapper, process, config


def precompute_frames(args, wrapper, process, config):
    torch.manual_seed(args.seed)
    sample_fn = get_distribution(config["distribution"])
    x0_np = sample_fn(args.n_particles, seed=args.seed)
    x0 = torch.from_numpy(x0_np).float().to(DEVICE)
    z = torch.randn_like(x0)

    xs = np.linspace(-args.grid_limit, args.grid_limit, args.grid_size)
    ys = np.linspace(-args.grid_limit, args.grid_limit, args.grid_size)
    XX, YY = np.meshgrid(xs, ys)
    grid = torch.from_numpy(np.stack([XX.ravel(), YY.ravel()], axis=1)).float().to(DEVICE)

    t_values = np.linspace(args.t_min, 1.0, args.n_frames)

    all_U, all_V, all_mag, all_particles = [], [], [], []
    with torch.no_grad():
        for t_val in t_values:
            t_grid = torch.full((grid.shape[0],), float(t_val), device=DEVICE)
            score = wrapper.score(grid, t_grid).cpu().numpy()
            U = score[:, 0].reshape(XX.shape)
            V = score[:, 1].reshape(XX.shape)
            mag = np.sqrt(U ** 2 + V ** 2)
            all_U.append(U)
            all_V.append(V)
            all_mag.append(mag)

            t_particles = torch.full((args.n_particles,), float(t_val), device=DEVICE)
            mean, std = process.marginal_prob(x0, t_particles)
            all_particles.append((mean + std * z).cpu().numpy())

    return XX, YY, t_values, all_U, all_V, all_mag, all_particles


def save_keyframes(dist_name, param, args, XX, YY, t_values, all_U, all_V, all_mag, all_particles, vmax_global):
    KEYFRAME_DIR.mkdir(parents=True, exist_ok=True)
    key_indices = [0, len(t_values) // 2, len(t_values) - 1]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), dpi=110)
    for ax, idx in zip(axes, key_indices):
        Un = all_U[idx] / (all_mag[idx] + 1e-8)
        Vn = all_V[idx] / (all_mag[idx] + 1e-8)
        q = ax.quiver(XX, YY, Un, Vn, all_mag[idx], cmap="viridis", scale=25, width=0.004, clim=(0, vmax_global))
        pts = all_particles[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.35, color="white", edgecolor="black", linewidth=0.15)
        ax.set_xlim(-args.grid_limit, args.grid_limit)
        ax.set_ylim(-args.grid_limit, args.grid_limit)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t_values[idx]:.2f}", fontsize=12)

    fig.colorbar(q, ax=axes, label="|score|", shrink=0.8)
    fig.suptitle(f"score_field_animation — {dist_name}/{param} — key frames", fontsize=13)
    kf_path = KEYFRAME_DIR / f"score_field_animation_{dist_name}_{param}_keyframes.png"
    fig.savefig(kf_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved keyframes to {kf_path.relative_to(PROJECT_ROOT)}")


def main():
    args = parse_args()
    wrapper, process, config = load_wrapper(args.checkpoint)
    dist_name, param = config["distribution"], config["param"]

    XX, YY, t_values, all_U, all_V, all_mag, all_particles = precompute_frames(args, wrapper, process, config)
    vmax_global = float(np.percentile(np.stack(all_mag), 99))

    save_keyframes(dist_name, param, args, XX, YY, t_values, all_U, all_V, all_mag, all_particles, vmax_global)

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    Un0 = all_U[0] / (all_mag[0] + 1e-8)
    Vn0 = all_V[0] / (all_mag[0] + 1e-8)
    q = ax.quiver(XX, YY, Un0, Vn0, all_mag[0], cmap="viridis", scale=25, width=0.004, clim=(0, vmax_global))
    scat = ax.scatter(all_particles[0][:, 0], all_particles[0][:, 1], s=4, alpha=0.35,
                       color="white", edgecolor="black", linewidth=0.15)
    ax.set_xlim(-args.grid_limit, args.grid_limit)
    ax.set_ylim(-args.grid_limit, args.grid_limit)
    ax.set_aspect("equal")
    title = ax.set_title("")

    def update(frame_idx):
        Un = all_U[frame_idx] / (all_mag[frame_idx] + 1e-8)
        Vn = all_V[frame_idx] / (all_mag[frame_idx] + 1e-8)
        q.set_UVC(Un, Vn, all_mag[frame_idx])
        scat.set_offsets(all_particles[frame_idx])
        title.set_text(f"Score field — {dist_name}/{param} — t = {t_values[frame_idx]:.2f}")
        return [q, scat, title]

    anim = animation.FuncAnimation(fig, update, frames=args.n_frames, blit=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "score_field.mp4"
    anim.save(out_path, writer="ffmpeg", fps=args.fps, dpi=120)
    plt.close(fig)
    print(f"Saved {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
