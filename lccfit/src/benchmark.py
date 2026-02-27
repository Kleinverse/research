#!/usr/bin/env python3
"""
benchmark_three_routes.py
=========================
Benchmark three Maclaurin approximation routes for log-CES utility:

  Route 1 -- LCC       : weighted moments -> cumulants (Faa di Bruno, algebraic O(1))
                          -> u_hat = mu_1 + sum kappa_r / r! * theta^{r-1}
  Route 2 -- CtrdM     : weighted moments -> u_hat using central moments as
                          Maclaurin coefficients (biased for k >= 4)
                          -> u_hat = mu_1 + sum mu_r / r! * theta^{r-1}
  Route 3 -- Numerical : finite-difference derivatives of the CGF
                          K(theta) = log sum_i w_i exp(theta x_i) at theta=0
                          -> kappa_r = K^(r)(0) numerically
                          -> u_hat = mu_1 + sum kappa_r / r! * theta^{r-1}

LCC and Numerical produce the same Maclaurin coefficients (both recover
true cumulants); CtrdM is biased at k >= 4.  The benchmark isolates the
computational cost: LCC uses O(1) algebraic post-processing, Numerical
evaluates K at (2*order+1) theta values at O(n) each.

Usage
-----
    python benchmark_three_routes.py -i imports_2024.csv [--order 6] [--h 1e-4]

Dependencies
------------
    numpy, pandas, torch (CUDA recommended)
"""

import argparse
import gc
import sys
import time
from math import factorial

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Device / precision
# ---------------------------------------------------------------------------
DEVICE  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE   = torch.float64
THETA   = 0.5
ORDER   = 6          # Maclaurin expansion order
H_FD    = 1e-4       # finite-difference step for numerical route
N_TRIALS = 10
T_MC    = 10_000     # MC cells per distribution

print(f"Device : {DEVICE}", flush=True)
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name()}", flush=True)


# ===========================================================================
# Data loading
# ===========================================================================

def load_cells(path: str, min_varieties: int = 5) -> list:
    print(f"Loading {path}...", file=sys.stderr)
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    cols = set(df.columns)

    val_col, is_monthly = None, False
    for vc in ['gen_val', 'con_val']:
        if vc in cols: val_col = vc; break
    if val_col is None:
        for vc in ['gen_val_mo', 'con_val_mo']:
            if vc in cols: val_col = vc; is_monthly = True; break
    if val_col is None:
        for vc in ['gen_val_yr', 'con_val_yr']:
            if vc in cols:
                val_col = vc
                if 'month' in cols: df = df[df['month'] == 12].copy()
                break

    qty_col = None
    for qc in ['qty1', 'gen_qy1_mo', 'con_qy1_mo',
               'gen_qy1_yr', 'con_qy1_yr', 'qty']:
        if qc in cols: qty_col = qc; break

    if val_col is None or qty_col is None:
        print(f"  Columns found: {sorted(cols)}", file=sys.stderr)
        print("  Input does not contain unit-level variety data; skipping Part 1.",
              file=sys.stderr)
        return None

    out = pd.DataFrame()
    out['commodity']    = df['commodity'].astype(str).str.zfill(10)
    out['year']         = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    out['country']      = df['cty_code'].astype(str)
    out['import_value'] = pd.to_numeric(df[val_col], errors='coerce')
    out['quantity']     = pd.to_numeric(df[qty_col], errors='coerce')
    out = out.dropna(subset=['import_value', 'quantity', 'year'])
    out = out[(out['quantity'] > 0) & (out['import_value'] > 0)]

    if is_monthly:
        out = out.groupby(['commodity', 'year', 'country'], as_index=False).agg(
            {'import_value': 'sum', 'quantity': 'sum'})

    out['_uv']    = out['import_value'] / out['quantity']
    out           = out[(out['_uv'] > 0) & np.isfinite(out['_uv'])].copy()
    out['_x']     = np.log(out['_uv'])
    out['_share'] = (out.groupby(['commodity', 'year'])['import_value']
                       .transform(lambda v: v / v.sum()))

    cell_list = []
    for _, grp in out.groupby(['commodity', 'year']):
        if len(grp) < min_varieties: continue
        cell_list.append((grp['_x'].values, grp['_share'].values))

    ns = [len(c[0]) for c in cell_list]
    print(f"  {len(cell_list)} cells -- varieties: "
          f"min={min(ns)}, median={int(np.median(ns))}, max={max(ns)}",
          file=sys.stderr)
    return cell_list


def pad_cells(cell_list: list) -> tuple:
    M, C = max(len(c[0]) for c in cell_list), len(cell_list)
    X    = np.zeros((C, M), dtype=np.float64)
    W    = np.zeros((C, M), dtype=np.float64)
    mask = np.zeros((C, M), dtype=np.float64)
    for i, (x, w) in enumerate(cell_list):
        n = len(x)
        X[i, :n] = x; W[i, :n] = w; mask[i, :n] = 1.0
    return (torch.tensor(X,    dtype=DTYPE, device=DEVICE),
            torch.tensor(W,    dtype=DTYPE, device=DEVICE),
            torch.tensor(mask, dtype=DTYPE, device=DEVICE))


# ===========================================================================
# Shared Step 1: weighted central moments  O(Kn)
# ===========================================================================

def compute_moments(X: torch.Tensor, W: torch.Tensor,
                    mask: torch.Tensor, order: int) -> list:
    """
    Returns [mu_1, mu_2, ..., mu_order] as [C] tensors.
    mu_1 = weighted mean; mu_r = weighted r-th central moment for r >= 2.
    """
    mu1 = (W * X * mask).sum(dim=1)                     # [C]
    C_  = (X - mu1.unsqueeze(1)) * mask                 # [C, M] centred
    moments = [None, mu1]
    for r in range(2, order + 1):
        moments.append((W * C_ ** r).sum(dim=1))
    return moments                                       # indexed 1..order


# ===========================================================================
# Route 1 -- LCC: algebraic moment-to-cumulant inversion  O(1)
# ===========================================================================

def lcc_route(moments: list, theta: float, order: int) -> torch.Tensor:
    """
    Faà di Bruno inversion of central moments -> cumulants, then
    assembles u_hat = mu_1 + sum_{r=2}^{order} kappa_r / r! * theta^{r-1}.

    Hardcoded for order <= 8.
    """
    m = moments          # m[r] is the r-th (central) moment tensor
    kp = [None] * (order + 1)
    kp[1] = m[1]
    if order >= 2: kp[2] = m[2]
    if order >= 3: kp[3] = m[3]
    if order >= 4: kp[4] = m[4] - 3*m[2]**2
    if order >= 5: kp[5] = m[5] - 10*m[3]*m[2]
    if order >= 6: kp[6] = m[6] - 15*m[4]*m[2] - 10*m[3]**2 + 30*m[2]**3
    if order >= 7: kp[7] = (m[7] - 21*m[5]*m[2] - 35*m[4]*m[3]
                             + 210*m[3]*m[2]**2)
    if order >= 8: kp[8] = (m[8] - 28*m[6]*m[2] - 56*m[5]*m[3]
                             - 35*m[4]**2 + 420*m[4]*m[2]**2
                             + 560*m[3]**2*m[2] - 630*m[2]**4)

    u = m[1].clone()
    for r in range(2, order + 1):
        u = u + (theta ** (r - 1)) / factorial(r) * kp[r]
    return u


# ===========================================================================
# Route 2 -- CtrdM: central moments as Maclaurin coefficients  O(1)
# ===========================================================================

def ctrdm_route(moments: list, theta: float, order: int) -> torch.Tensor:
    """
    Uses mu_r / r! directly as Maclaurin coefficients (biased for r >= 4).
    """
    u = moments[1].clone()
    for r in range(2, order + 1):
        u = u + (theta ** (r - 1)) / factorial(r) * moments[r]
    return u


# ===========================================================================
# Route 3 -- Numerical: finite-difference derivatives of the CGF  O(order * n)
#
# K(theta) = log sum_i w_i exp(theta x_i)
# K^(r)(0) = r-th cumulant  (same values as LCC, computed numerically)
#
# Central-difference stencil coefficients c_{r,j} such that:
#   K^(r)(0) ≈  (1 / h^r)  sum_j  c_{r,j} * K(j*h)
#
# Points j range over a symmetric window; O(h^2) accuracy.
# ===========================================================================

# Standard central-difference coefficients for derivatives 1..8
# Dict maps order r -> {offset j: coefficient c_{r,j}}
_CD_COEFFS = {
    1: {-1: -1/2,   1: 1/2},
    2: {-1:  1,     0: -2,      1:  1},
    3: {-2: -1/2,  -1:  1,      1: -1,     2:  1/2},
    4: {-2:  1,    -1: -4,      0:  6,     1: -4,    2:  1},
    5: {-3: -1/2,  -2:  2,     -1: -5/2,   1:  5/2,  2: -2,   3:  1/2},
    6: {-3:  1,    -2: -6,     -1: 15,     0: -20,   1: 15,   2: -6,   3:  1},
    7: {-4: -1/2,  -3:  3,     -2: -7,    -1:  7,    1: -7,   2:  7,
         3: -3,     4:  1/2},
    8: {-4:  1,    -3: -8,     -2: 28,    -1: -56,   0: 70,   1: -56,
         2: 28,     3: -8,      4:  1},
}


def numerical_route(X: torch.Tensor, W: torch.Tensor, mask: torch.Tensor,
                    theta: float, order: int, h: float = 1e-4) -> torch.Tensor:
    """
    Numerical Maclaurin via finite differences of K(theta).

    For each required derivative order r, evaluates K at offsets j*h
    and applies the central-difference stencil.  Total cost: O(order * n).
    Assembles u_hat identically to LCC (same cumulant values, higher cost).
    """
    # log_w[i] = log(w_i) for valid entries, -inf otherwise
    log_w = (torch.log(W.clamp(min=1e-300)) * mask
             + (mask - 1) * 1e15)         # [C, M]

    def K(th: float) -> torch.Tensor:
        """CGF value for all cells at a single theta.  [C]"""
        return torch.logsumexp(th * X + log_w, dim=1)

    # Cache K evaluations -- stencil for order r uses offsets up to ceil(r/2)
    max_offset = (order + 1) // 2 + 1
    K_cache: dict[int, torch.Tensor] = {}
    for j in range(-max_offset, max_offset + 1):
        K_cache[j] = K(j * h)

    # Recover cumulants via finite differences
    kp = [None] * (order + 1)
    kp[1] = K_cache[0]          # K'(0) ≈ K(0)/0 -- actually K(0) = log 1 = 0;
                                 # mu_1 = K'(0), computed properly below via r=1
    coeffs_1 = _CD_COEFFS[1]
    kp[1] = sum(c * K_cache[j] for j, c in coeffs_1.items()) / h

    for r in range(2, order + 1):
        coeffs = _CD_COEFFS[r]
        kp[r] = sum(c * K_cache[j] for j, c in coeffs.items()) / (h ** r)

    # Assemble u_hat
    u = kp[1].clone()
    for r in range(2, order + 1):
        u = u + (theta ** (r - 1)) / factorial(r) * kp[r]
    return u


# ===========================================================================
# Full pipeline wrappers (include Step 1 for end-to-end timing)
# ===========================================================================

def run_lcc(X, W, mask, order=None):
    m = compute_moments(X, W, mask, order)
    lcc_route(m, THETA, order)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()


def run_ctrdm(X, W, mask, order=None):
    m = compute_moments(X, W, mask, order)
    ctrdm_route(m, THETA, order)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()


def run_numerical(X, W, mask, order=None):
    if order is None: order = ORDER
    numerical_route(X, W, mask, THETA, order, H_FD)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()


# ===========================================================================
# Timing utilities
# ===========================================================================

def _clear():
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache(); torch.cuda.synchronize()


def time_fn(fn, n_trials=N_TRIALS, min_reps=100):
    _clear(); fn()
    t0 = time.perf_counter(); fn()
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    t_one = time.perf_counter() - t0
    reps = max(min_reps, int(0.5 / max(t_one, 1e-9)))

    times = []
    for _ in range(n_trials):
        _clear()
        t0 = time.perf_counter()
        for _ in range(reps): fn()
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) / reps)
    a = np.array(times)
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(n_trials))


def _fmt(t):
    if t >= 1.0:   return f"{t:.3f} s"
    if t >= 1e-3:  return f"{t*1e3:.3f} ms"
    return               f"{t*1e6:.3f} us"


# ===========================================================================
# Synthesized distributions  (matching paper Table)
# ===========================================================================

def _skewnormal(rng, n, alpha=5):
    u1, u2 = rng.randn(n), rng.randn(n)
    return np.where(u2 <= alpha * u1, u1, -u1)


DISTRIBUTIONS = {
    'Normal(0,1)':      lambda rng, n: rng.normal(0, 1, n),
    'Pareto(7)+1':      lambda rng, n: (1 - rng.rand(n)) ** (-1/7) + 1,
    'Lognormal(0,0.5)': lambda rng, n: rng.lognormal(0, 0.5, n),
    'Mixture':          lambda rng, n: np.where(
                            rng.random(n) < 0.3,
                            rng.normal(0, 1, n),
                            rng.normal(3, 0.5, n)),
    'Skew-normal(5)':   lambda rng, n: _skewnormal(rng, n),
    'Uniform(0,3)':     lambda rng, n: rng.uniform(0, 3, n),
}


# ===========================================================================
# MC benchmark on synthesized data
# ===========================================================================

def run_mc_benchmark(n_varieties: int, T: int, order: int, h_fd: float) -> None:
    rng    = np.random.RandomState(42)
    W_np   = np.full((T, n_varieties), 1.0 / n_varieties, dtype=np.float64)
    mask_np = np.ones((T, n_varieties), dtype=np.float64)
    W_t    = torch.tensor(W_np,    dtype=DTYPE, device=DEVICE)
    mask_t = torch.tensor(mask_np, dtype=DTYPE, device=DEVICE)

    hdr = (f"  {'Distribution':>22}  {'LCC':>10}  {'CtrdM':>10}  "
           f"{'Numerical':>10}  {'LCC/Num':>8}  {'CtrdM/Num':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for name, sampler in DISTRIBUTIONS.items():
        X_np = np.stack([sampler(rng, n_varieties) for _ in range(T)])
        X_t  = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)

        mn_l, _ = time_fn(lambda: run_lcc(X_t, W_t, mask_t, order))
        mn_c, _ = time_fn(lambda: run_ctrdm(X_t, W_t, mask_t, order))
        mn_n, _ = time_fn(lambda: run_numerical(X_t, W_t, mask_t, order))

        rl = mn_l / max(mn_n, 1e-15)
        rc = mn_c / max(mn_n, 1e-15)
        print(f"  {name:>22}  {_fmt(mn_l):>10}  {_fmt(mn_c):>10}  "
              f"{_fmt(mn_n):>10}  {rl:>7.2f}x  {rc:>9.2f}x")
    sys.stdout.flush()


# ===========================================================================
# Main
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description='Benchmark LCC vs CtrdM vs Numerical (GPU)')
    p.add_argument('-i', '--input',       default=None)
    p.add_argument('--min-varieties',     type=int,   default=5)
    p.add_argument('--order',             type=int,   default=6)
    p.add_argument('--h',                 type=float, default=1e-4)
    args = p.parse_args()

    order = args.order
    h_fd  = args.h

    # Part 1: Real HS10 data (optional)
    # ------------------------------------------------------------------
    if args.input:
        cell_list = load_cells(args.input, args.min_varieties)
        if cell_list is None:
            print("Input lacks unit-level data; skipping Part 1.")
        else:
            C                      = len(cell_list)
            X_gpu, W_gpu, mask_gpu = pad_cells(cell_list)
            print(f"  Padded tensor: {list(X_gpu.shape)}\n", file=sys.stderr)

            print(f"\n{'='*80}")
            print(f"  Part 1: Real HS10 data  ({C} cells, order={order})")
            print(f"{'='*80}")

            mn_s, se_s = time_fn(
                lambda: compute_moments(X_gpu, W_gpu, mask_gpu, order))
            print(f"\n  Step 1 (shared moments):  {_fmt(mn_s)}  ({_fmt(mn_s/C)} / cell)"
                  f"  s.e.={_fmt(se_s)}")

            m_pre = compute_moments(X_gpu, W_gpu, mask_gpu, order)
            mn_l2, _ = time_fn(lambda: lcc_route(m_pre, THETA, order))
            mn_c2, _ = time_fn(lambda: ctrdm_route(m_pre, THETA, order))
            mn_n2, _ = time_fn(
                lambda: numerical_route(X_gpu, W_gpu, mask_gpu, THETA, order, h_fd))
            print(f"\n  Step 2 only:")
            print(f"    LCC       (algebraic cumulants): {_fmt(mn_l2)}  ({_fmt(mn_l2/C)} / cell)")
            print(f"    CtrdM     (central moments):     {_fmt(mn_c2)}  ({_fmt(mn_c2/C)} / cell)")
            print(f"    Numerical (finite diff CGF):     {_fmt(mn_n2)}  ({_fmt(mn_n2/C)} / cell)")

            print(f"\n  Full pipeline:")
            print(f"  {'Route':>30}  {'total':>10}  {'s.e.':>10}  "
                  f"{'per cell':>10}  {'rel. Numerical':>15}")
            print("  " + "-" * 80)

            mn_lf, se_lf = time_fn(lambda: run_lcc(X_gpu, W_gpu, mask_gpu, order))
            mn_cf, se_cf = time_fn(lambda: run_ctrdm(X_gpu, W_gpu, mask_gpu, order))
            mn_nf, se_nf = time_fn(lambda: run_numerical(X_gpu, W_gpu, mask_gpu, order))
            for label, mn, se in [('LCC', mn_lf, se_lf),
                                   ('CtrdM', mn_cf, se_cf),
                                   ('Numerical', mn_nf, se_nf)]:
                rel = mn / max(mn_nf, 1e-15)
                print(f"  {label:>30}  {_fmt(mn):>10}  ({_fmt(se):>8})  "
                      f"{_fmt(mn/C):>10}  {rel:>8.3f}x")
    else:
        print("No -i input provided; skipping Part 1.")

        # ------------------------------------------------------------------
    # Part 2: Synthesized distributions
    # ------------------------------------------------------------------
    for n_var in [10, 50, 100, 500, 1_000]:
        print(f"\n{'='*80}")
        print(f"  Part 2: n={n_var} varieties, T={T_MC} cells, "
              f"theta={THETA}, order={order}")
        print(f"{'='*80}")
        run_mc_benchmark(n_var, T_MC, order, h_fd)

    print()


if __name__ == '__main__':
    main()
