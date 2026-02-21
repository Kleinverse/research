#!/usr/bin/env python3
"""Experiments for bounded LCC paper.

Usage:
    python exp_bounded.py                # both tables, 20 trials
    python exp_bounded.py --table 1      # Table 1 only
    python exp_bounded.py --table 2      # Table 2 only
    python exp_bounded.py --trials 200   # full paper runs
"""

import argparse, warnings, numpy as np
from sklearn.exceptions import ConvergenceWarning
from icalcc import ICALCC

try:
    from gpuicalcc import GPUICALCC
    _GPU = True
except ImportError:
    _GPU = False


def amari(W, A):
    P = np.abs(W @ A)
    n = P.shape[0]
    return ((P / P.max(1, keepdims=True)).sum()
            + (P / P.max(0, keepdims=True)).sum()
            - 2 * n) / (2 * n * (n - 1))


def make_est(K, d, max_iter, seed):
    cls = GPUICALCC if _GPU else ICALCC
    return cls(n_components=d, K=K, max_iter=max_iter, random_state=seed)


def sources(name, d, N, rng, alpha=None):
    if name == "uniform":    S = rng.uniform(-1, 1, (N, d))
    elif name == "beta25":   S = rng.beta(2, 5, (N, d))
    elif name == "gamma":    S = rng.gamma(alpha, 1, (N, d))
    elif name == "laplace":  S = rng.laplace(size=(N, d))
    elif name == "t5":       S = rng.standard_t(5, (N, d))
    elif name == "exp":      S = rng.exponential(1, (N, d))
    elif name == "mixed":
        S = np.hstack([rng.uniform(-1, 1, (N, d // 2)),
                       rng.laplace(size=(N, d - d // 2))])
    else: raise ValueError(name)
    return (S - S.mean(0)) / S.std(0)


def bench(dist, d, N, K, trials, alpha=None, max_iter=300):
    ais, fails = [], 0
    for t in range(trials):
        rng = np.random.RandomState(t)
        S = sources(dist, d, N, rng, alpha)
        A = rng.randn(d, d)
        est = make_est(K, d, max_iter, t)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            est.fit(S @ A.T)
        conv = getattr(est, "converged_", not any(
            issubclass(x.category, ConvergenceWarning) for x in w))
        if not conv: fails += 1
        ais.append(amari(est.components_, A))
    return np.mean(ais) * 100, fails


LABELS = {"tanh": "tanh", "exp": "exp",
          "ltanh": "LCC-tanh", "lexp": "LCC-exp",
          6: "LCC(6)", 8: "LCC(8)"}


def table1(trials=20, d=4, N=5000):
    print(f"\nTable 1: Bounded LCC (d={d}, N={N}, {trials} trials)")
    dists = [("Uniform", "uniform", None, -1.2),
             ("Beta(2,5)", "beta25", None, -0.1),
             ("Gamma(8)", "gamma", 8, 0.75),
             ("Gamma(2)", "gamma", 2, 3.0),
             ("Laplace", "laplace", None, 3.0),
             ("Student-t5", "t5", None, 6.0),
             ("Exponential", "exp", None, 6.0),
             ("Mixed", "mixed", None, None)]
    methods = ["tanh", "exp", "ltanh", "lexp"]

    print(f"{'Dist':<14}{'k4':>5}", end="")
    for m in methods: print(f"{LABELS[m]:>10}", end="")
    print()
    for name, dist, alpha, k4 in dists:
        k4s = "mixed" if k4 is None else f"{k4}"
        print(f"{name:<14}{k4s:>5}", end="")
        vals = {}
        for m in methods:
            v, f = bench(dist, d, N, m, trials, alpha)
            vals[m] = v
            star = "*" if f else ""
            print(f"{v:>9.2f}{star}", end="")
        print(f"  <- {LABELS[min(vals, key=vals.get)]}")


def table2(trials=200, d=4, N=10000):
    print(f"\nTable 2: Gamma scan (d={d}, N={N}, {trials} trials)")
    alphas = [2, 5, 8, 12, 20, 50]
    methods = ["tanh", "exp", 6, 8]

    print(f"{'a':>5}{'k3':>6}{'k4':>6}", end="")
    for m in methods: print(f"{LABELS[m]:>10}", end="")
    print()
    for a in alphas:
        print(f"{a:>5}{2/a**.5:>6.2f}{6/a:>6.1f}", end="")
        vals = {}
        for m in methods:
            v, f = bench("gamma", d, N, m, trials, a, max_iter=500)
            vals[m] = v
            star = "*" if f else ""
            print(f"{v:>9.2f}{star}", end="")
        print(f"  <- {LABELS[min(vals, key=vals.get)]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table", choices=["1", "2", "all"], default="all")
    p.add_argument("--trials", type=int, default=20)
    args = p.parse_args()
    print(f"GPU: {'yes' if _GPU else 'no'}")
    if args.table in ("1", "all"): table1(args.trials)
    if args.table in ("2", "all"): table2(args.trials)
