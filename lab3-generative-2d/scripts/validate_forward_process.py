"""
Validate the closed-form marginal_prob() kernels for VP, VE, and sub-VP.

For a fixed x0 and several t values, draws N=50000 noise realizations and
forms x_t = mean + std * noise directly from marginal_prob() (the closed
form kernel) -- it does NOT integrate the SDE step by step. This checks
that the tensor shapes/broadcasting/arithmetic inside marginal_prob() and
the reparameterization sampling are internally consistent: the empirical
mean/std of the 50000 draws must match the analytic (mean, std) that
produced them, up to Monte Carlo error.

Scope note: this is an implementation sanity check (catches shape bugs,
sign errors, std-vs-variance mixups), not an independent re-derivation of
the SDE solution via Euler-Maruyama path simulation. That cross-check
belongs to a later step (solver validation).

Usage:
    python scripts/validate_forward_process.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.forward_process import VESDE, VPSDE, SubVPSDE

N_SAMPLES = 50_000
T_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]
X0_POINT = torch.tensor([2.5, -1.8])
REL_ERR_THRESHOLD = 0.02
SEED = 123

PROCESSES = {
    "VP": VPSDE(),
    "VE": VESDE(),
    "subVP": SubVPSDE(),
}

OUT_DIR = PROJECT_ROOT / "outputs" / "sanity_checks"


def relative_error(empirical: torch.Tensor, predicted: torch.Tensor) -> float:
    num = torch.linalg.norm(empirical - predicted).item()
    den = torch.linalg.norm(predicted).item()
    return num / den if den > 1e-12 else num


def scale_normalized_error(empirical: torch.Tensor, predicted: torch.Tensor, scale: float) -> float:
    """
    Error normalized by a fixed reference scale (||x0||) instead of by
    ||predicted||. Needed for the mean: as t -> 1, alpha(t) -> 0 and
    pred_mean = alpha(t) x0 -> 0 by design (the process erases x0), so
    dividing by ||pred_mean|| blows up relative error even when the
    absolute error is within Monte Carlo noise. Dividing by ||x0|| (fixed,
    nonzero) keeps the metric well-conditioned across all t.
    """
    num = torch.linalg.norm(empirical - predicted).item()
    return num / scale if scale > 1e-12 else num


def main() -> None:
    torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x0 = X0_POINT.unsqueeze(0)  # (1, 2)
    x0_scale = torch.linalg.norm(X0_POINT).item()
    results = []
    failures = []

    for proc_name, process in PROCESSES.items():
        for t_val in T_VALUES:
            t = torch.tensor([t_val])

            mean, std = process.marginal_prob(x0, t)  # mean (1,2), std (1,1)
            noise = torch.randn(N_SAMPLES, 2)
            x_t = mean + std * noise                  # (N, 2)

            emp_mean = x_t.mean(dim=0)
            emp_std = x_t.std(dim=0, unbiased=False)

            pred_mean = mean.squeeze(0)
            pred_std = std.squeeze(0).expand(2)

            err_mean_rel = relative_error(emp_mean, pred_mean)
            err_mean_x0 = scale_normalized_error(emp_mean, pred_mean, x0_scale)
            err_std = relative_error(emp_std, pred_std)
            err_max = max(err_mean_x0, err_std)

            results.append((proc_name, t_val, err_mean_rel, err_mean_x0, err_std,
                             err_max, pred_mean.tolist(), pred_std[0].item()))
            if err_max > REL_ERR_THRESHOLD:
                failures.append((proc_name, t_val, err_mean_x0, err_std))

    # ---- print + save table -------------------------------------------------
    lines = []
    header = (f"{'process':<8}{'t':>6}{'pred_std':>11}"
              f"{'err_mean/||mean||':>19}{'err_mean/||x0||':>17}{'err_std':>10}{'status':>9}")
    sep = "-" * len(header)
    lines.append(header)
    lines.append(sep)
    print(header)
    print(sep)
    for proc_name, t_val, err_mean_rel, err_mean_x0, err_std, err_max, pred_mean, pred_std in results:
        status = "OK" if err_max <= REL_ERR_THRESHOLD else "FAIL"
        row = (f"{proc_name:<8}{t_val:>6.1f}{pred_std:>11.4f}"
               f"{err_mean_rel*100:>18.3f}%{err_mean_x0*100:>16.3f}%"
               f"{err_std*100:>9.3f}%{status:>9}")
        lines.append(row)
        print(row)

    table_path = OUT_DIR / "forward_process_validation.txt"
    table_path.write_text("\n".join(lines) + "\n")
    print(f"\nSaved table to {table_path.relative_to(PROJECT_ROOT)}")

    if failures:
        print("\n*** VALIDATION FAILED ***")
        for proc_name, t_val, err_mean_x0, err_std in failures:
            print(
                f"  {proc_name} at t={t_val}: "
                f"err_mean/||x0||={err_mean_x0:.4f} err_std={err_std:.4f} "
                f"(threshold {REL_ERR_THRESHOLD:.2%})"
            )
        sys.exit(1)

    print(f"\nAll {len(results)} (process, t) combinations passed "
          f"(relative error <= {REL_ERR_THRESHOLD:.0%}).")


if __name__ == "__main__":
    main()
