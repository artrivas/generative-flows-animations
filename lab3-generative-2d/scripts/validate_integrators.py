"""
Validate the generic integrators (src/integrators) against the VP SDE's
known closed-form kernel (marginal_prob, validated in step 2).

1. Euler & Heun (deterministic): integrate dx/dt = drift(x,t) (no noise
   term) from a fixed x0 to t=1, for several step counts N. The pure-drift
   ODE's EXACT solution is x(t) = alpha(t) x0 (VP's drift is linear), which
   is exactly the mean marginal_prob reports -- so the analytic target is
   not an approximation. Plot error vs dt on log-log axes; Heun's slope
   (empirical order) must exceed Euler's (order 2 vs order 1).

2. Euler-Maruyama (stochastic): integrate the full SDE (drift + diffusion)
   for N_TRAJECTORIES independent paths from the same x0, and compare the
   empirical variance at t=1 against the analytic variance from
   marginal_prob.

Methodological note (learned in step 2): relative error is normalized
against a FIXED scale -- ||x0|| for the deterministic convergence test,
the analytic variance itself (here close to 1, safely away from zero) for
the stochastic test -- rather than against a quantity that can legitimately
vanish (e.g. the VP mean as t -> 1).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.forward_process import VPSDE
from src.integrators import euler_maruyama_step, euler_step, heun_step

SEED = 42
X0_POINT = torch.tensor([2.5, -1.8])
STEP_COUNTS = [10, 50, 200, 1000]
N_TRAJECTORIES = 10_000
EM_N_STEPS = 1000
VAR_ERR_THRESHOLD = 0.05  # 5%: MC std-error alone at N=10000 is ~1.4%

OUT_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


# ---------------------------------------------------------------------------
# Part 1: deterministic convergence (Euler vs Heun)
# ---------------------------------------------------------------------------

def integrate_deterministic(step_fn, drift_fn, x0_batch: torch.Tensor, n_steps: int) -> torch.Tensor:
    dt = 1.0 / n_steps
    x = x0_batch.clone()
    t = torch.zeros(x0_batch.shape[0])
    for _ in range(n_steps):
        x = step_fn(x, t, dt, drift_fn)
        t = t + dt
    return x


def run_convergence_study(process, x0_batch, x0_scale):
    mean_true, _ = process.marginal_prob(x0_batch, torch.tensor([1.0]))
    mean_true = mean_true.squeeze(0)

    rows = []
    for n_steps in STEP_COUNTS:
        dt = 1.0 / n_steps
        x_euler = integrate_deterministic(euler_step, process.drift, x0_batch, n_steps).squeeze(0)
        x_heun = integrate_deterministic(heun_step, process.drift, x0_batch, n_steps).squeeze(0)

        err_euler_abs = torch.linalg.norm(x_euler - mean_true).item()
        err_heun_abs = torch.linalg.norm(x_heun - mean_true).item()

        rows.append({
            "n_steps": n_steps, "dt": dt,
            "err_euler_abs": err_euler_abs, "err_euler_rel": err_euler_abs / x0_scale,
            "err_heun_abs": err_heun_abs, "err_heun_rel": err_heun_abs / x0_scale,
        })
    return rows, mean_true


def fit_order(dts, errs):
    """log(err) = order * log(dt) + const -- slope is the empirical convergence order."""
    log_dt = np.log(dts)
    log_err = np.log(errs)
    slope, intercept = np.polyfit(log_dt, log_err, 1)
    return slope, intercept


def plot_convergence(rows, order_euler, order_heun):
    dts = [r["dt"] for r in rows]
    err_euler = [r["err_euler_abs"] for r in rows]
    err_heun = [r["err_heun_abs"] for r in rows]

    fig, ax = plt.subplots(figsize=(6, 5.5), dpi=130)
    ax.loglog(dts, err_euler, "o-", label=f"Euler (order ≈ {order_euler:.2f})", color="#C44E52")
    ax.loglog(dts, err_heun, "s-", label=f"Heun (order ≈ {order_heun:.2f})", color="#4C72B0")

    # reference slope-1 and slope-2 guide lines anchored at the largest dt point
    dt_ref = dts[0]
    ax.loglog(dts, [err_euler[0] * (dt / dt_ref) ** 1 for dt in dts],
              "--", color="#C44E52", alpha=0.4, label="slope 1 (reference)")
    ax.loglog(dts, [err_heun[0] * (dt / dt_ref) ** 2 for dt in dts],
              "--", color="#4C72B0", alpha=0.4, label="slope 2 (reference)")

    ax.set_xlabel("dt")
    ax.set_ylabel("||x_integrated(t=1) - x_analytic(t=1)||")
    ax.set_title("Euler vs Heun convergence (VP drift, deterministic)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", linewidth=0.3, alpha=0.5)
    fig.tight_layout()

    path = OUT_DIR / "integrator_convergence.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_ROOT)}")


# ---------------------------------------------------------------------------
# Part 2: Euler-Maruyama variance check
# ---------------------------------------------------------------------------

def run_em_variance_check(process, x0_point: torch.Tensor):
    torch.manual_seed(SEED)
    dt = 1.0 / EM_N_STEPS
    x = x0_point.unsqueeze(0).repeat(N_TRAJECTORIES, 1)
    t = torch.zeros(N_TRAJECTORIES)
    for _ in range(EM_N_STEPS):
        x = euler_maruyama_step(x, t, dt, process.drift, process.diffusion)
        t = t + dt

    emp_mean = x.mean(dim=0)
    emp_var = x.var(dim=0, unbiased=False)

    x0_batch = x0_point.unsqueeze(0)
    mean_true, std_true = process.marginal_prob(x0_batch, torch.tensor([1.0]))
    mean_true = mean_true.squeeze(0)
    var_true = (std_true.squeeze(0) ** 2).expand(2)

    err_var_abs = (emp_var - var_true).abs()
    err_var_rel = err_var_abs / var_true.clamp_min(1e-8)

    return {
        "emp_mean": emp_mean, "mean_true": mean_true,
        "emp_var": emp_var, "var_true": var_true,
        "err_var_abs": err_var_abs, "err_var_rel": err_var_rel,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    process = VPSDE()
    x0_batch = X0_POINT.unsqueeze(0)
    x0_scale = torch.linalg.norm(X0_POINT).item()

    # --- Part 1 ---
    print("=" * 70)
    print("Part 1: Euler vs Heun deterministic convergence (VP drift)")
    print("=" * 70)
    rows, mean_true = run_convergence_study(process, x0_batch, x0_scale)

    header = f"{'N':>6}{'dt':>10}{'err_euler_abs':>16}{'err_euler_rel':>16}{'err_heun_abs':>16}{'err_heun_rel':>16}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['n_steps']:>6}{r['dt']:>10.4f}"
              f"{r['err_euler_abs']:>16.6f}{r['err_euler_rel']*100:>15.3f}%"
              f"{r['err_heun_abs']:>16.6f}{r['err_heun_rel']*100:>15.3f}%")

    dts = [r["dt"] for r in rows]
    order_euler, _ = fit_order(dts, [r["err_euler_abs"] for r in rows])
    order_heun, _ = fit_order(dts, [r["err_heun_abs"] for r in rows])

    print(f"\nAnalytic target x(t=1) = {mean_true.tolist()}  (||x0||={x0_scale:.4f})")
    print(f"Fitted empirical order — Euler: {order_euler:.3f}  (expected ≈ 1)")
    print(f"Fitted empirical order — Heun:  {order_heun:.3f}  (expected ≈ 2)")

    plot_convergence(rows, order_euler, order_heun)

    if not (order_heun > order_euler):
        print("\n*** VALIDATION FAILED: Heun's order is NOT greater than Euler's. ***")
        print("    Check the k1/k2 averaging in heun_step, or the Euler step itself.")
        sys.exit(1)

    # --- Part 2 ---
    print("\n" + "=" * 70)
    print(f"Part 2: Euler-Maruyama variance check ({N_TRAJECTORIES} trajectories, "
          f"{EM_N_STEPS} steps)")
    print("=" * 70)
    em = run_em_variance_check(process, X0_POINT)

    print(f"emp_mean:  {em['emp_mean'].tolist()}")
    print(f"true_mean: {em['mean_true'].tolist()}")
    print(f"emp_var:   {em['emp_var'].tolist()}")
    print(f"true_var:  {em['var_true'].tolist()}")
    print(f"err_var_abs: {em['err_var_abs'].tolist()}")
    print(f"err_var_rel: {[f'{v*100:.3f}%' for v in em['err_var_rel'].tolist()]}")

    max_err_var_rel = em["err_var_rel"].max().item()
    if max_err_var_rel > VAR_ERR_THRESHOLD:
        print(f"\n*** VALIDATION FAILED: variance relative error "
              f"{max_err_var_rel:.2%} exceeds threshold {VAR_ERR_THRESHOLD:.0%}. ***")
        print("    Check the sqrt(dt) factor in euler_maruyama_step's noise term.")
        sys.exit(1)

    print(f"\nAll checks passed: Heun order ({order_heun:.2f}) > Euler order "
          f"({order_euler:.2f}); EM variance error ({max_err_var_rel:.2%}) "
          f"<= {VAR_ERR_THRESHOLD:.0%}.")


if __name__ == "__main__":
    main()
