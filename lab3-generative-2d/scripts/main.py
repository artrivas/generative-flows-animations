"""
Unified command-line entry point for Lab 3.

Examples:
    python scripts/main.py train --kind score --distribution eight_gaussians --model eps_pred --steps 2000
    python scripts/main.py sample --distribution eight_gaussians --model eps_pred --sampler pf_ode --steps 250
    python scripts/main.py animate --which flow_matching --distribution eight_gaussians --steps 250
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.registry import REGISTRY as DISTRIBUTION_REGISTRY
from src.data.registry import get_distribution
from src.sampling import (
    load_score_wrapper,
    load_velocity_model,
    sample_flow_matching_ode,
    sample_pf_ode,
    sample_reverse_sde,
)
from src.sampling.utils import default_device

MODEL_NAMES = {
    "eps_pred": "eps",
    "v_pred": "v",
    "flow_matching": "flow_matching",
}

SAMPLER_NAMES = {
    "reverse_sde",
    "pf_ode",
    "flow_matching_ode",
}

ANIMATION_MODULES = {
    "forward_comparison": "src.viz.forward_comparison",
    "density_evolution": "src.viz.density_evolution",
    "forward_trajectories": "src.viz.forward_trajectories",
    "score_field": "src.viz.score_field_animation",
    "reverse_sde": "src.viz.reverse_sde_generation",
    "pf_ode": "src.viz.pf_ode_generation",
    "flow_matching": "src.viz.flow_matching_animation",
    "steps_comparison": "src.viz.steps_comparison",
}


def resolve_project_path(path_text: str | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_score_checkpoint(distribution: str, model: str, seed: int = 0) -> Path:
    if model not in {"eps_pred", "v_pred"}:
        raise ValueError("default_score_checkpoint requires eps_pred or v_pred")
    param = MODEL_NAMES[model]
    return PROJECT_ROOT / "checkpoints" / f"{distribution}_{param}_seed{seed}.pt"


def default_flow_checkpoint(distribution: str, seed: int = 0) -> Path:
    return PROJECT_ROOT / "checkpoints" / f"{distribution}_flow_matching_seed{seed}.pt"


def add_if_present(cmd: list[str], flag: str, value):
    if value is not None:
        cmd.extend([flag, str(value)])


def load_config(path_text: str | None) -> dict:
    if not path_text:
        return {}
    path = resolve_project_path(path_text)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    try:
        from omegaconf import OmegaConf
    except Exception as exc:
        raise RuntimeError("YAML configs require omegaconf; install requirements.txt first") from exc
    return dict(OmegaConf.to_container(OmegaConf.load(path), resolve=True))


def cmd_train(args):
    cfg = load_config(args.config)
    kind = args.kind or cfg.get("kind", "score")
    distribution = args.distribution or cfg.get("distribution", "eight_gaussians")
    steps = args.steps if args.steps is not None else cfg.get("steps")
    epochs = args.epochs if args.epochs is not None else cfg.get("epochs")
    batch_size = args.batch_size if args.batch_size is not None else cfg.get("batch_size")
    lr = args.lr if args.lr is not None else cfg.get("lr")
    seed = args.seed if args.seed is not None else cfg.get("seed", 0)
    output_dir = args.output_dir or cfg.get("output_dir", "checkpoints")

    if kind == "score":
        model = args.model or cfg.get("model", "eps_pred")
        if model not in {"eps_pred", "v_pred"}:
            raise ValueError("score training requires --model eps_pred or --model v_pred")
        module = "src.training.train_score"
        cmd = [sys.executable, "-m", module, "--distribution", distribution, "--param", MODEL_NAMES[model]]
    elif kind == "flow_matching":
        module = "src.training.train_flow_matching"
        cmd = [sys.executable, "-m", module, "--distribution", distribution]
    else:
        raise ValueError("--kind must be score or flow_matching")

    add_if_present(cmd, "--steps", steps)
    add_if_present(cmd, "--epochs", epochs)
    add_if_present(cmd, "--batch_size", batch_size)
    add_if_present(cmd, "--lr", lr)
    add_if_present(cmd, "--seed", seed)
    add_if_present(cmd, "--output-dir", output_dir)

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def cmd_sample(args):
    if args.distribution not in DISTRIBUTION_REGISTRY:
        available = ", ".join(sorted(DISTRIBUTION_REGISTRY))
        raise ValueError(f"Unknown distribution {args.distribution!r}. Available: {available}")
    if args.model not in MODEL_NAMES:
        raise ValueError(f"Unknown model {args.model!r}")
    if args.sampler not in SAMPLER_NAMES:
        raise ValueError(f"Unknown sampler {args.sampler!r}")

    device = default_device()
    checkpoint = resolve_project_path(args.checkpoint)

    if args.sampler in {"reverse_sde", "pf_ode"}:
        if args.model == "flow_matching":
            raise ValueError("reverse_sde and pf_ode require eps_pred or v_pred score checkpoints")
        checkpoint = checkpoint or default_score_checkpoint(args.distribution, args.model, args.checkpoint_seed)
        wrapper, process, config = load_score_wrapper(checkpoint, device=device)
        if args.sampler == "reverse_sde":
            generated = sample_reverse_sde(wrapper, process, args.n_samples, args.steps, args.seed, device=device)
        else:
            generated = sample_pf_ode(wrapper, process, args.n_samples, args.steps, args.seed, args.method, device=device)
        model_label = f"{config['param']}_pred"
    else:
        if args.model != "flow_matching":
            raise ValueError("flow_matching_ode requires --model flow_matching")
        checkpoint = checkpoint or default_flow_checkpoint(args.distribution, args.checkpoint_seed)
        model, config = load_velocity_model(checkpoint, device=device)
        generated = sample_flow_matching_ode(
            model,
            n_samples=args.n_samples,
            n_steps=args.steps,
            seed=args.seed,
            method=args.method,
            base_std=config.get("base_std", 1.0),
            device=device,
        )
        model_label = "flow_matching"

    samples = generated.detach().cpu().numpy()
    out_dir = resolve_project_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npy_path = out_dir / f"samples_{args.distribution}_{args.sampler}_{model_label}_N{args.steps}_seed{args.seed}.npy"
    np.save(npy_path, samples)

    real = get_distribution(args.distribution)(args.n_samples, seed=args.seed + 99)
    plot_dir = PROJECT_ROOT / "outputs" / "sanity_checks"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / f"sample_{args.distribution}_{args.sampler}_{model_label}_N{args.steps}.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=130)
    for ax, title, pts, color in [
        (axes[0], "real data", real, "#4C72B0"),
        (axes[1], f"{args.sampler} samples", samples, "#C44E52"),
    ]:
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.45, color=color, linewidths=0)
        ax.set_xlim(-args.axis_limit, args.axis_limit)
        ax.set_ylim(-args.axis_limit, args.axis_limit)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.grid(True, linewidth=0.3, alpha=0.35)
    fig.suptitle(f"{args.distribution} - {model_label} - {args.sampler} - N={args.steps}")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved samples to {npy_path.relative_to(PROJECT_ROOT)}")
    print(f"Saved plot to {plot_path.relative_to(PROJECT_ROOT)}")


def cmd_animate(args):
    module = ANIMATION_MODULES[args.which]
    cmd = [sys.executable, "-m", module]

    score_checkpoint = args.score_checkpoint
    flow_checkpoint = args.flow_checkpoint
    if score_checkpoint is None and args.distribution:
        score_model = args.model if args.model in {"eps_pred", "v_pred"} else "eps_pred"
        score_checkpoint = str(default_score_checkpoint(args.distribution, score_model, args.checkpoint_seed))
    if flow_checkpoint is None and args.distribution:
        flow_checkpoint = str(default_flow_checkpoint(args.distribution, args.checkpoint_seed))

    if args.which == "score_field":
        cmd.extend(["--checkpoint", score_checkpoint])
    elif args.which in {"reverse_sde", "pf_ode"}:
        cmd.extend(["--checkpoint", score_checkpoint, "--steps", str(args.steps), "--seed", str(args.seed)])
    elif args.which == "flow_matching":
        cmd.extend(["--checkpoint", flow_checkpoint, "--score_checkpoint", score_checkpoint, "--steps", str(args.steps), "--seed", str(args.seed)])
    elif args.which == "steps_comparison":
        cmd.extend(["--score_checkpoint", score_checkpoint, "--flow_checkpoint", flow_checkpoint, "--seed", str(args.seed)])
    elif args.which == "forward_comparison":
        cmd.extend(["--distribution", args.distribution, "--seed", str(args.seed)])
    elif args.which == "density_evolution":
        cmd.extend(["--distribution", args.distribution, "--seed", str(args.seed)])
    elif args.which == "forward_trajectories":
        cmd.extend(["--distribution", args.distribution, "--seed", str(args.seed)])

    add_if_present(cmd, "--n_particles", args.n_particles)
    add_if_present(cmd, "--fps", args.fps)
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def build_parser():
    p = argparse.ArgumentParser(description="Lab 3 unified entry point")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train")
    train.add_argument("--config")
    train.add_argument("--kind", choices=["score", "flow_matching"])
    train.add_argument("--distribution")
    train.add_argument("--model", choices=list(MODEL_NAMES))
    train.add_argument("--steps", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--batch_size", type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--seed", type=int)
    train.add_argument("--output-dir", dest="output_dir")
    train.set_defaults(func=cmd_train)

    sample = sub.add_parser("sample")
    sample.add_argument("--distribution", required=True, choices=sorted(DISTRIBUTION_REGISTRY))
    sample.add_argument("--model", required=True, choices=list(MODEL_NAMES))
    sample.add_argument("--sampler", required=True, choices=sorted(SAMPLER_NAMES))
    sample.add_argument("--steps", type=int, default=250)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--checkpoint_seed", type=int, default=0)
    sample.add_argument("--checkpoint")
    sample.add_argument("--method", choices=["euler", "heun"], default="heun")
    sample.add_argument("--n_samples", type=int, default=2000)
    sample.add_argument("--axis_limit", type=float, default=6.5)
    sample.add_argument("--output-dir", dest="output_dir", default="outputs/samples")
    sample.set_defaults(func=cmd_sample)

    animate = sub.add_parser("animate")
    animate.add_argument("--which", required=True, choices=sorted(ANIMATION_MODULES))
    animate.add_argument("--distribution", default="eight_gaussians", choices=sorted(DISTRIBUTION_REGISTRY))
    animate.add_argument("--model", default="eps_pred", choices=list(MODEL_NAMES))
    animate.add_argument("--steps", type=int, default=250)
    animate.add_argument("--seed", type=int, default=23)
    animate.add_argument("--checkpoint_seed", type=int, default=0)
    animate.add_argument("--score_checkpoint")
    animate.add_argument("--flow_checkpoint")
    animate.add_argument("--n_particles", type=int)
    animate.add_argument("--fps", type=int)
    animate.set_defaults(func=cmd_animate)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
