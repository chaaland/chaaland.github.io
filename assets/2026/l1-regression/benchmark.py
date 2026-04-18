"""
Benchmark wall-clock time of four L1 regression solvers.

Fair comparison strategy
------------------------
Comparing by "number of iterations" is misleading because each method's
iteration has a different computational cost:

  - Coordinate descent / golden section: O(d · N · gss_iters) per outer iter
  - Coordinate descent / knot scan:       O(d · N²)            per outer iter  ← O(N²) bottleneck
  - Coordinate descent / weighted median: O(d · N log N)        per outer iter
  - IRLS:                                 O(N · d²)             per outer iter  (dense solve)

Methods also target different solution quality: coordinate descent can stall
before reaching the true LAD minimum on non-smooth objectives.

We therefore compare on **wall-clock time to reach a fixed suboptimality
threshold** ε relative to the exact LP solution:

    (loss(β) - loss*) / loss*  <  tol

The exact reference loss* is computed once per problem instance via a
scipy.optimize.linprog reformulation of the LAD problem as a linear program.

Usage
-----
    python benchmark.py

Output
------
A table of median wall-clock times (ms) over `N_SEEDS` random data
realisations at each (N, method) combination.  Cells where the method
failed to reach `TOL` within `MAX_ITERS` outer iterations are marked "DNF".
"""

import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import linprog

# ──────────────────────────────────────────────
# Algorithm implementations (self-contained)
# ──────────────────────────────────────────────


def _golden_section_1d(x_col: np.ndarray, residuals: np.ndarray, n_gss: int = 50) -> float:
    """Approximate 1-D LAD minimizer via golden-section search."""
    nz = x_col != 0
    knots = residuals[nz] / x_col[nz]
    a, b = knots.min() - 1e-6, knots.max() + 1e-6

    phi = (np.sqrt(5) - 1) / 2  # ≈ 0.618
    x1 = b - phi * (b - a)
    x2 = a + phi * (b - a)

    def f(beta: float) -> float:
        return float(np.mean(np.abs(x_col * beta - residuals)))

    f1, f2 = f(x1), f(x2)
    for _ in range(n_gss):
        if f1 < f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - phi * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + phi * (b - a)
            f2 = f(x2)
    return (a + b) / 2.0


def _knot_scan_1d(x_col: np.ndarray, residuals: np.ndarray) -> float:
    """Exact 1-D LAD minimizer by scanning all N knots — O(N²) total."""
    nz = x_col != 0
    knots = residuals[nz] / x_col[nz]
    best_val, best_knot = np.inf, knots[0]
    for z in knots:
        val = float(np.mean(np.abs(x_col * z - residuals)))
        if val < best_val:
            best_val, best_knot = val, z
    return best_knot


def _weighted_median_1d(x_col: np.ndarray, residuals: np.ndarray) -> float:
    """Exact 1-D LAD minimizer via weighted median — O(N log N) total."""
    nz = x_col != 0
    knots = residuals[nz] / x_col[nz]
    weights = np.abs(x_col[nz])

    order = np.argsort(knots)
    knots_s = knots[order]
    weights_s = weights[order]

    cumw = np.cumsum(weights_s)
    idx = np.searchsorted(cumw, weights_s.sum() / 2.0)
    return float(knots_s[min(idx, len(knots_s) - 1)])


def _cd_one_pass(X: np.ndarray, y: np.ndarray, beta: np.ndarray, b: float, solver) -> tuple:
    """One full coordinate-descent pass: update each β_k then the bias."""
    d = X.shape[1]
    for k in range(d):
        r = y - b - X @ beta + X[:, k] * beta[k]
        beta[k] = solver(X[:, k], r)
    b = float(np.median(y - X @ beta))
    return beta, b


def lad_loss(X: np.ndarray, y: np.ndarray, beta: np.ndarray, b: float) -> float:
    return float(np.mean(np.abs(X @ beta + b - y)))


# ──────────────────────────────────────────────
# Reference solution via LP
# ──────────────────────────────────────────────


def lad_lp_reference(X: np.ndarray, y: np.ndarray) -> float:
    """Solve the LAD problem exactly as a linear program.

    Reformulation:
        min   (1/N) Σ tᵢ
        s.t.  tᵢ ≥  (Xᵢ θ − yᵢ)
              tᵢ ≥ −(Xᵢ θ − yᵢ)
              tᵢ ≥ 0

    Variables: θ ∈ R^(d+1) (unconstrained), t ∈ R^N (≥ 0).
    """
    n, d = X.shape
    X_aug = np.column_stack([X, np.ones(n)])  # (n, d+1)
    p = d + 1  # number of θ variables

    # Objective: 0 for θ, 1/N for each tᵢ
    c = np.concatenate([np.zeros(p), np.ones(n) / n])

    # Inequality constraints A_ub @ x ≤ b_ub
    # Constraint 1: Xᵢ θ − tᵢ ≤  yᵢ   →  [X_aug | -I] θ ≤ y
    # Constraint 2: −Xᵢ θ − tᵢ ≤ −yᵢ  →  [-X_aug | -I] θ ≤ -y
    I_n = np.eye(n)
    A_ub = np.block(
        [
            [X_aug, -I_n],
            [-X_aug, -I_n],
        ]
    )
    b_ub = np.concatenate([y, -y])

    # θ is free (unbounded), tᵢ ≥ 0
    bounds = [(None, None)] * p + [(0, None)] * n

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")

    return float(result.fun)


# ──────────────────────────────────────────────
# Benchmarking harness
# ──────────────────────────────────────────────


@dataclass
class BenchResult:
    method: str
    N: int
    seed: int
    elapsed_s: float  # wall-clock seconds to reach tolerance (or budget)
    converged: bool  # True iff relative suboptimality < TOL
    final_gap: float  # (loss - loss*) / loss*  at termination
    n_outer_iters: int  # number of outer iterations taken


def _run_cd(
    X: np.ndarray,
    y: np.ndarray,
    loss_ref: float,
    solver,
    tol: float,
    max_iters: int,
) -> tuple[float, bool, float, int]:
    """Run coordinate descent with the given 1-D solver."""
    n, d = X.shape
    beta = np.zeros(d)
    b = float(np.median(y))

    t0 = time.perf_counter()
    iters = 0
    for iters in range(1, max_iters + 1):
        beta, b = _cd_one_pass(X, y, beta, b, solver)
        gap = (lad_loss(X, y, beta, b) - loss_ref) / max(abs(loss_ref), 1e-10)
        if gap < tol:
            break
    elapsed = time.perf_counter() - t0
    converged = gap < tol
    return elapsed, converged, gap, iters


def _run_irls(
    X: np.ndarray,
    y: np.ndarray,
    loss_ref: float,
    tol: float,
    max_iters: int,
    eps: float = 1e-6,
) -> tuple[float, bool, float, int]:
    """Run IRLS."""
    n, d = X.shape
    X_aug = np.column_stack([X, np.ones(n)])
    theta = np.zeros(d + 1)
    theta[d] = float(np.median(y))

    t0 = time.perf_counter()
    iters = 0
    gap = np.inf
    for iters in range(1, max_iters + 1):
        residuals = y - X_aug @ theta
        w = 1.0 / np.maximum(np.abs(residuals), eps)
        Xw = X_aug * w[:, np.newaxis]
        theta = np.linalg.solve(Xw.T @ X_aug, Xw.T @ y)
        beta, b = theta[:-1], theta[-1]
        gap = (lad_loss(X, y, beta, b) - loss_ref) / max(abs(loss_ref), 1e-10)
        if gap < tol:
            break
    elapsed = time.perf_counter() - t0
    converged = gap < tol
    return elapsed, converged, gap, iters


METHODS = {
    "coord_gss": lambda X, y, loss_ref, tol, max_iters: _run_cd(X, y, loss_ref, _golden_section_1d, tol, max_iters),
    "coord_knot": lambda X, y, loss_ref, tol, max_iters: _run_cd(X, y, loss_ref, _knot_scan_1d, tol, max_iters),
    "coord_wmed": lambda X, y, loss_ref, tol, max_iters: _run_cd(X, y, loss_ref, _weighted_median_1d, tol, max_iters),
    "irls": lambda X, y, loss_ref, tol, max_iters: _run_irls(X, y, loss_ref, tol, max_iters),
}


def _fmt_ms(seconds: float) -> str:
    """Format a duration as a human-readable string."""
    ms = seconds * 1000
    if ms < 1:
        return f"{ms * 1000:.1f}µs"
    if ms < 1_000:
        return f"{ms:.1f}ms"
    return f"{seconds:.2f}s"


_CSV_FIELDS = ["method", "N", "seed", "elapsed_ms", "converged", "final_gap", "n_outer_iters"]


def _result_to_row(r: BenchResult) -> dict:
    return {
        "method": r.method,
        "N": r.N,
        "seed": r.seed,
        "elapsed_ms": "" if np.isnan(r.elapsed_s) else f"{r.elapsed_s * 1000:.3f}",
        "converged": r.converged,
        "final_gap": "" if np.isnan(r.final_gap) else f"{r.final_gap:.6f}",
        "n_outer_iters": r.n_outer_iters,
    }


def run_benchmark(
    n_params: int = 50,
    n_values: list[int] | None = None,
    n_seeds: int = 10,
    tol: float = 1e-3,  # relative suboptimality threshold
    max_iters: int = 200,  # outer iteration budget
    skip_knot_above: Optional[int] = 5_000,  # skip O(N²) knot scan for large N
    csv_path: Optional[str] = None,  # stream results to this file as they complete
) -> list[BenchResult]:
    """Run the full benchmark grid.

    Parameters
    ----------
    n_params:
        Fixed number of features d (bias excluded).
    n_values:
        List of dataset sizes N to benchmark. Defaults to a log-spaced sweep.
    n_seeds:
        Number of random data realisations per (N, method) cell.
    tol:
        Convergence criterion: (loss − loss*) / loss* < tol.
    max_iters:
        Maximum number of outer iterations before declaring DNF.
    skip_knot_above:
        Skip the O(N²) knot-scan method for N > this value to avoid very
        long runtimes; set to None to always include it.
    csv_path:
        If given, stream each result row to this CSV file as it completes so
        that partial results survive an interrupted run.
    """
    import csv

    if n_values is None:
        n_values = [100, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000]

    results: list[BenchResult] = []

    n_methods = len(METHODS)
    seed_w = len(str(n_seeds))
    method_col = max(len(m) for m in METHODS) + 2

    total_cells = len(n_values) * n_seeds * (n_methods + 1)  # +1 for lp_ref per seed
    completed = 0

    csv_file = open(csv_path, "w", newline="") if csv_path else None
    csv_writer = None
    if csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDS)
        csv_writer.writeheader()
        csv_file.flush()
        print(f"Streaming results to {csv_path}")

    def _append(result: BenchResult) -> None:
        results.append(result)
        if csv_writer:
            csv_writer.writerow(_result_to_row(result))
            csv_file.flush()

    try:
        for N in n_values:
            print(f"N = {N:,d}")
            for seed in range(n_seeds):
                rng = np.random.default_rng(seed)
                X = rng.standard_normal((N, n_params))
                beta_true = rng.standard_normal(n_params)
                b_true = rng.standard_normal()
                noise = rng.laplace(scale=0.5, size=N)  # heavy-tailed noise suits LAD
                y = X @ beta_true + b_true + noise

                seed_prefix = f"  s{seed + 1:>{seed_w}}/{n_seeds}"

                # Reference: exact LP solution
                label = f"{'lp_ref':<{method_col}}"
                print(f"{seed_prefix}  {label}", end="", flush=True)
                t0 = time.perf_counter()
                loss_ref = lad_lp_reference(X, y)
                lp_elapsed = time.perf_counter() - t0
                completed += 1
                pct = 100 * completed / total_cells
                print(f"  {_fmt_ms(lp_elapsed):>8}   [{pct:5.1f}%]")

                for method_name, run_fn in METHODS.items():
                    label = f"{method_name:<{method_col}}"
                    print(f"{seed_prefix}  {label}", end="", flush=True)

                    if skip_knot_above is not None and method_name == "coord_knot" and N > skip_knot_above:
                        completed += 1
                        pct = 100 * completed / total_cells
                        print(f"  {'SKIP':>8}   [{pct:5.1f}%]")
                        _append(
                            BenchResult(
                                method=method_name,
                                N=N,
                                seed=seed,
                                elapsed_s=float("nan"),
                                converged=False,
                                final_gap=float("nan"),
                                n_outer_iters=0,
                            )
                        )
                        continue

                    elapsed, converged, gap, iters = run_fn(X, y, loss_ref, tol, max_iters)
                    completed += 1
                    pct = 100 * completed / total_cells
                    status = "✓" if converged else f"DNF gap={gap:.2e}"
                    print(f"  {_fmt_ms(elapsed):>8}  {status}   [{pct:5.1f}%]")
                    _append(
                        BenchResult(
                            method=method_name,
                            N=N,
                            seed=seed,
                            elapsed_s=elapsed,
                            converged=converged,
                            final_gap=gap,
                            n_outer_iters=iters,
                        )
                    )
            print()
    finally:
        if csv_file:
            csv_file.close()

    return results


def summarize(results: list[BenchResult]) -> None:
    """Print a summary table: median wall-clock time (ms) per (N, method)."""
    import statistics

    n_values = sorted({r.N for r in results})
    methods = list(METHODS.keys())

    # Header
    col_w = 14
    header = f"{'N':>8}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print("Median time (ms) to reach (loss − loss*)/loss* < tol")
    print("DNF = did not converge within iteration budget")
    print("SKIP = skipped (N too large for O(N²) method)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for N in n_values:
        row = f"{N:>8,d}"
        for method in methods:
            cell_results = [r for r in results if r.N == N and r.method == method]
            # Separate converged runs from non-converged
            converged_times = [r.elapsed_s * 1000 for r in cell_results if r.converged]
            all_nan = all(np.isnan(r.elapsed_s) for r in cell_results)

            if all_nan:
                cell = "SKIP"
            elif not converged_times:
                # None converged — show median time with DNF marker
                finite_times = [r.elapsed_s * 1000 for r in cell_results if not np.isnan(r.elapsed_s)]
                med = statistics.median(finite_times) if finite_times else float("nan")
                cell = f"DNF ({med:.1f})" if not np.isnan(med) else "DNF"
            else:
                med = statistics.median(converged_times)
                conv_rate = len(converged_times) / len(cell_results)
                # Flag partial convergence
                flag = "" if conv_rate == 1.0 else f"*{conv_rate:.0%}"
                cell = f"{med:.1f}{flag}"
            row += f"{cell:>{col_w}}"
        print(row)

    print("-" * len(header))
    print("* = fraction of seeds that converged within budget\n")


def plot_scaling(results: list[BenchResult], path: str = "benchmark_scaling.png") -> None:
    """Plot median wall-clock time vs N for each method on a log-log scale.

    Converged runs only.  IQR band shows spread across seeds.  Methods that
    never converged at a given N are omitted from that point rather than
    plotted as zero or infinity.
    """
    import statistics

    import matplotlib.pyplot as plt

    methods = list(METHODS.keys())
    n_values = sorted({r.N for r in results})

    colors = {"coord_gss": "tab:blue", "coord_knot": "tab:orange", "coord_wmed": "tab:green", "irls": "tab:red"}
    labels = {
        "coord_gss": "Coord. descent (golden section)",
        "coord_knot": "Coord. descent (knot scan)",
        "coord_wmed": "Coord. descent (weighted median)",
        "irls": "IRLS",
    }

    fig, ax = plt.subplots(figsize=(7, 4))

    for method in methods:
        xs, medians, lo, hi = [], [], [], []
        for N in n_values:
            times = [r.elapsed_s * 1000 for r in results if r.N == N and r.method == method and r.converged]
            if not times:
                continue
            xs.append(N)
            med = statistics.median(times)
            medians.append(med)
            lo.append(med - min(times))
            hi.append(max(times) - med)

        if not xs:
            continue

        c = colors[method]
        ax.plot(xs, medians, marker="o", color=c, label=labels[method], linewidth=2, markersize=5)
        ax.fill_between(
            xs,
            [m - err for m, err in zip(medians, lo)],
            [m + err for m, err in zip(medians, hi)],
            color=c,
            alpha=0.15,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N (number of data points)")
    ax.set_ylabel("Wall-clock time (ms)")
    ax.set_title("L1 regression solver scaling (converged runs only)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Scaling plot saved to `{path}`")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=int, default=10, help="number of features (default 10)")
    parser.add_argument("--seeds", type=int, default=5, help="random seeds per cell (default 5)")
    parser.add_argument("--tol", type=float, default=1e-3, help="relative suboptimality tolerance (default 1e-3)")
    parser.add_argument("--max-iters", type=int, default=200, help="outer iteration budget (default 200)")
    parser.add_argument(
        "--no-skip-knot", action="store_true", help="do not skip knot-scan for large N (may be very slow)"
    )
    parser.add_argument("--csv", type=str, default="benchmark_results.csv", help="path for raw CSV output")
    parser.add_argument("--plot", type=str, default="images/benchmark-scaling.png", help="path for scaling plot output")
    args = parser.parse_args()

    print("Benchmarking L1 regression solvers")
    print(f"  d={args.d} features, {args.seeds} seeds/cell")
    print(f"  convergence tol = {args.tol} (relative suboptimality)")
    print(f"  max outer iterations = {args.max_iters}")
    print()

    skip_above = None if args.no_skip_knot else 5_000

    results = run_benchmark(
        n_params=args.d,
        n_seeds=args.seeds,
        tol=args.tol,
        max_iters=args.max_iters,
        skip_knot_above=skip_above,
        csv_path=args.csv,
    )

    summarize(results)
    plot_scaling(results, path=args.plot)
