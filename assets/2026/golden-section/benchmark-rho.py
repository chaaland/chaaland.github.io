"""Benchmark golden-section search (rho=1/phi, 1 eval/iter) against
equally-spaced interval shrinking (rho=2/3, 2 evals/iter) as a function
of N, the number of terms in the least-absolute-deviations objective.

Writes results to benchmark_rho.csv for plotting elsewhere.
"""

import csv
import time
from pathlib import Path

import numpy as np

OUTPUT_PATH = Path("benchmark_rho.csv")
TRACE_OUTPUT_PATH = Path("benchmark_rho_trace.csv")

TOL = 1e-4
RHO_GOLDEN = (-1 + 5**0.5) / 2
RHO_TWO_THIRDS = 2 / 3


def make_obj_fn(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.integers(low=1, high=25, size=n)
    y = rng.integers(low=-25, high=25, size=n)

    def obj_fn(beta):
        return np.mean(np.abs(beta * x - y))

    knots = y / x
    return obj_fn, float(knots.min()), float(knots.max())


def golden_section_minimize(obj_fn, a, b, tol):
    rho = RHO_GOLDEN
    L = b - a
    x1 = b - rho * L
    x2 = a + rho * L

    f_x1 = obj_fn(x1)
    f_x2 = obj_fn(x2)
    n_evals = 2

    n_steps = 0
    while L > tol:
        n_steps += 1
        if f_x1 < f_x2:
            b = x2
            L = b - a
            x2, f_x2 = x1, f_x1
            x1 = b - rho * L
            f_x1 = obj_fn(x1)
        else:
            a = x1
            L = b - a
            x1, f_x1 = x2, f_x2
            x2 = a + rho * L
            f_x2 = obj_fn(x2)
        n_evals += 1

    return n_steps, n_evals


def interval_shrinking_minimize(obj_fn, a, b, rho, tol):
    L = b - a
    x1 = b - rho * L
    x2 = a + rho * L

    f_x1 = obj_fn(x1)
    f_x2 = obj_fn(x2)
    n_evals = 2

    n_steps = 0
    while L > tol:
        n_steps += 1
        if f_x1 < f_x2:
            b = x2
        else:
            a = x1
        L = b - a

        x1 = b - rho * L
        x2 = a + rho * L
        f_x1 = obj_fn(x1)
        f_x2 = obj_fn(x2)
        n_evals += 2

    return n_steps, n_evals


def golden_section_trace(obj_fn, a, b, tol):
    rho = RHO_GOLDEN
    L = b - a
    x1 = b - rho * L
    x2 = a + rho * L

    f_x1 = obj_fn(x1)
    f_x2 = obj_fn(x2)
    n_evals = 2
    trace = [(n_evals, L)]

    while L > tol:
        if f_x1 < f_x2:
            b = x2
            L = b - a
            x2, f_x2 = x1, f_x1
            x1 = b - rho * L
            f_x1 = obj_fn(x1)
        else:
            a = x1
            L = b - a
            x1, f_x1 = x2, f_x2
            x2 = a + rho * L
            f_x2 = obj_fn(x2)
        n_evals += 1
        trace.append((n_evals, L))

    return trace


def interval_shrinking_trace(obj_fn, a, b, rho, tol):
    L = b - a
    x1 = b - rho * L
    x2 = a + rho * L

    f_x1 = obj_fn(x1)
    f_x2 = obj_fn(x2)
    n_evals = 2
    trace = [(n_evals, L)]

    while L > tol:
        if f_x1 < f_x2:
            b = x2
        else:
            a = x1
        L = b - a

        x1 = b - rho * L
        x2 = a + rho * L
        f_x1 = obj_fn(x1)
        f_x2 = obj_fn(x2)
        n_evals += 2
        trace.append((n_evals, L))

    return trace


def run_trace():
    obj_fn, a, b = make_obj_fn(n=1000)

    rows = []
    for n_evals, width in golden_section_trace(obj_fn, a, b, tol=TOL):
        rows.append({"method": "golden", "n_evals": n_evals, "width": width})
    for n_evals, width in interval_shrinking_trace(obj_fn, a, b, rho=RHO_TWO_THIRDS, tol=TOL):
        rows.append({"method": "two_thirds", "n_evals": n_evals, "width": width})

    return rows


def write_trace_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "n_evals", "width"])
        writer.writeheader()
        writer.writerows(rows)


def time_convergence(minimize_fn, obj_fn, a, b, n_repeats=20, **kwargs):
    best_time = np.inf
    for _ in range(n_repeats):
        start = time.perf_counter()
        n_steps, n_evals = minimize_fn(obj_fn, a, b, tol=TOL, **kwargs)
        best_time = min(best_time, time.perf_counter() - start)
    return best_time, n_steps, n_evals


def run_sweep(n_values, n_repeats=20):
    rows = []

    for n in n_values:
        obj_fn, a, b = make_obj_fn(n)

        t, steps, evals = time_convergence(
            golden_section_minimize, obj_fn, a, b, n_repeats=n_repeats
        )
        rows.append({"n": n, "method": "golden", "time_s": t, "n_steps": steps, "n_evals": evals})

        t, steps, evals = time_convergence(
            interval_shrinking_minimize,
            obj_fn,
            a,
            b,
            n_repeats=n_repeats,
            rho=RHO_TWO_THIRDS,
        )
        rows.append({"n": n, "method": "two_thirds", "time_s": t, "n_steps": steps, "n_evals": evals})

    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "method", "time_s", "n_steps", "n_evals"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    n_values = np.unique(np.logspace(4, 8, num=15, dtype=int))
    rows = run_sweep(n_values, n_repeats=5)
    write_csv(rows, OUTPUT_PATH)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")

    trace_rows = run_trace()
    write_trace_csv(trace_rows, TRACE_OUTPUT_PATH)
    print(f"Wrote {len(trace_rows)} rows to {TRACE_OUTPUT_PATH}")
