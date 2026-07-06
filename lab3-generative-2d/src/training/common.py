"""
Shared checkpoint-saving / loss-logging / loss-plotting helper.

Factored out of train_score.py and train_flow_matching.py (post-hoc review,
step 6) -- both scripts duplicated this ~40-line block almost verbatim.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def save_checkpoint_and_plot(
    model: torch.nn.Module,
    config: dict,
    loss_history: list,
    ckpt_name: str,
    out_dir: Path,
    project_root: Path,
    title: str,
) -> None:
    """Save {ckpt_name}.pt, {ckpt_name}_losses.csv, and a loss-curve PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{ckpt_name}.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    log_path = out_dir / f"{ckpt_name}_losses.csv"
    np.savetxt(log_path, np.array(loss_history), delimiter=",", header="loss", comments="")
    print(f"Saved loss log to {log_path}")

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    ax.plot(loss_history, linewidth=0.7, alpha=0.5, label="raw")
    window = max(1, len(loss_history) // 50)
    if len(loss_history) >= window:
        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(loss_history)), smoothed, linewidth=1.5,
                 label=f"smoothed (w={window})")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    plot_path = project_root / "outputs" / "sanity_checks" / f"{ckpt_name}_loss_curve.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss curve to {plot_path}")
