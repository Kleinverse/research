#!/usr/bin/env python3
"""
experiments.py
================================
V-statistic kernel properties: LCC vs CtrdM.

Section A  (theta-independent)
    Panel (a): kappa_k / k! - K_k^pop / k!
        LCC Maclaurin coefficient vs true population cumulant coefficient.
        s_k = (-1)^{k-1} k^{k-1} / ((k-1)! k!),  derived from eq. (37).
    Panel (b): m_k / k! - K_k^pop / k!
        CtrdM Maclaurin coefficient vs true cumulant coefficient.
        Demonstrates the permanent bias of centered moments for k >= 4.

Section C  (per theta)
    Panel (a): (u_hat_LCC   - u_true) / |u_true|
    Panel (b): (u_hat_CtrdM - u_true) / |u_true|
    Panel (c): |res_LCC| - |res_CtrdM|   one-sided test H1: LCC dominates.

    u_hat_LCC uses cumulants K_k recovered by inverting the V-statistic
    partition formula (eq. 37); NOT sample cumulants derived from moments.
    u_hat_CtrdM uses sample central moments m_k directly.

Section D  (per theta)
    Newton recovery of theta and rho = 1/(1-theta) from each u_hat.

MC pool design
--------------
    For each distribution, a pool of shape (T_MC, max_n) is generated once.
    Trial t at sample size n uses pool[t, :n] — no distribution calls inside
    the trial loop.  All sections share the same pool for exact comparability.

Notation
--------
    K_k     CGF cumulant (population quantity).
    m_k     sample central moment.
    E[h_k]  MC estimate of V-statistic kernel expectation.
    c_k     = (-1)^{k-1} (k-1)! / k^{k-1}    [eq. 37 scalar].
    s_k     = 1 / (c_k * k!) = (-1)^{k-1} k^{k-1} / ((k-1)! k!)
              [LCC rescaling; sign from c_k is essential].

Usage
-----
    python experiments.py
    python experiments.py --theta 0.3 0.5 0.7

Expected runtime  (RTX 5080, T_MC=10000, N_MC=500000)
    Section A alone  : ~2-4 h per distribution.
    Sections C+D     : ~1-2 h per distribution per theta.
    Set T_MC=100, N_MC=10000 for a quick validation run.

Dependencies
------------
    numpy, torch (CUDA recommended), scipy, tqdm
"""
import argparse
import csv
import os
import sys
from math import factorial, comb, sqrt, pi

import numpy as np
import torch
from scipy import stats as sp_stats
from tqdm import tqdm
from gpu_kernel import (warmup as _native_warmup,
                          _CUPY_AVAILABLE as NATIVE_AVAILABLE)

# ---------------------------------------------------------------------------
# Device and precision
# ---------------------------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE  = torch.float64
torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)

print(f"Device : {DEVICE}", flush=True)
if DEVICE.type == 'cuda':
    print(f"GPU    : {torch.cuda.get_device_name()}", flush=True)
print(f"Dtype  : {DTYPE}", flush=True)

# ---------------------------------------------------------------------------
# Experiment parameters  (matching paper Table 2 / Section 5)
# ---------------------------------------------------------------------------
T_MC         = 10_000
SAMPLE_SIZES = [5, 10, 50, 100, 500, 1_000, 5_000, 10_000]
ORDERS       = [2, 3, 4, 5, 6, 7, 8]
PARETO_ALPHA = 7
CSV_DIR      = 'csv_results'

# ---------------------------------------------------------------------------
# LCC rescaling factors
#   s_k = (-1)^{k-1} * k^{k-1} / ((k-1)! * k!)
#   From eq. (37): E[h_k] = c_k * K_k,  c_k = (-1)^{k-1} (k-1)! / k^{k-1}
#   => K_k / k! = E[h_k] * s_k,  where s_k = 1 / (c_k * k!)
#   The (-1)^{k-1} factor is essential: c_k is negative for even k.
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Distributions  (matching paper Table 2)
# ---------------------------------------------------------------------------

def _gen_skewnormal(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Azzalini skew-normal(alpha=5) via sign-flip (Azzalini 1985, Prop. 1)."""
    u1 = rng.randn(n)
    u2 = rng.randn(n)
    return np.where(u2 <= 5.0 * u1, u1, -u1).astype(np.float64)


DISTRIBUTIONS: dict[str, callable] = {
    'Normal':      lambda rng, n: rng.normal(0, 1, n).astype(np.float64),
    'Pareto':      lambda rng, n: ((1.0 - rng.rand(n)) ** (-1.0 / PARETO_ALPHA)
                                   + 1.0).astype(np.float64),
    'Lognormal':   lambda rng, n: rng.lognormal(0, 0.5, n).astype(np.float64),
    'Mixture':     lambda rng, n: np.where(
                       rng.random(n) < 0.3,
                       rng.normal(0, 1, n),
                       rng.normal(3, 0.5, n)).astype(np.float64),
    'Skew-normal': _gen_skewnormal,
    'Uniform':     lambda rng, n: rng.uniform(0, 3, n).astype(np.float64),
}


# ===========================================================================
# Population cumulants  (analytical; MC fallback for divergent moments)
# ===========================================================================

def _double_fact(n: int) -> int:
    r = 1
    while n > 0:
        r *= n;  n -= 2
    return r


def _normal_raw_moments(mu: float, sig: float, R: int) -> list:
    cm = [0.0] * (R + 1)
    for r in range(0, R + 1, 2):
        cm[r] = _double_fact(r - 1) * sig ** r
    return [sum(comb(r, j) * mu ** (r - j) * cm[j]
                for j in range(0, r + 1, 2))
            for r in range(R + 1)]


def _halfnorm_moments(R: int) -> list:
    from math import gamma
    return [2.0 ** (r / 2.0) * gamma((r + 1) / 2.0) / sqrt(pi)
            for r in range(R + 1)]


def _raw_to_central(raw: list, R: int) -> list:
    mu = raw[1]
    cm = [0.0] * (R + 1);  cm[0] = 1.0
    for r in range(2, R + 1):
        cm[r] = sum(comb(r, j) * ((-mu) ** (r - j)) * raw[j]
                    for j in range(r + 1))
    return cm


def _central_to_cumulants(cm: list, max_k: int) -> dict:
    kp = {}
    kp[2] = cm[2]
    if max_k >= 3: kp[3] = cm[3]
    if max_k >= 4: kp[4] = cm[4] - 3*cm[2]**2
    if max_k >= 5: kp[5] = cm[5] - 10*cm[3]*cm[2]
    if max_k >= 6: kp[6] = cm[6] - 15*cm[4]*cm[2] - 10*cm[3]**2 + 30*cm[2]**3
    if max_k >= 7: kp[7] = (cm[7] - 21*cm[5]*cm[2] - 35*cm[4]*cm[3]
                             + 210*cm[3]*cm[2]**2)
    if max_k >= 8: kp[8] = (cm[8] - 28*cm[6]*cm[2] - 56*cm[5]*cm[3]
                             - 35*cm[4]**2 + 420*cm[4]*cm[2]**2
                             + 560*cm[3]**2*cm[2] - 630*cm[2]**4)
    return kp


def _pareto_plus1_raw(r: int, alpha: int = PARETO_ALPHA) -> float:
    if r >= alpha:
        return float('inf')
    return float(sum(comb(r, j) * alpha / (alpha - j) for j in range(r + 1)))


def population_cumulants(dist_name: str, max_k: int = 8) -> dict:
    """
    Analytical population Maclaurin coefficients K_k / k! for k = 2..max_k.
    Uses 100M-sample MC fallback where analytical moments diverge (Pareto k>=7).
    Returns dict mapping k -> K_k / k!.
    """
    R = max_k

    if dist_name == 'Normal':
        raw = _normal_raw_moments(0.0, 1.0, R)
    elif dist_name == 'Uniform':
        raw = [3.0 ** r / (r + 1) for r in range(R + 1)]
    elif dist_name == 'Lognormal':
        raw = [float(np.exp(r * r * 0.125)) for r in range(R + 1)]
    elif dist_name == 'Mixture':
        r1  = _normal_raw_moments(0.0, 1.0, R)
        r2  = _normal_raw_moments(3.0, 0.5, R)
        raw = [0.3 * r1[i] + 0.7 * r2[i] for i in range(R + 1)]
    elif dist_name == 'Skew-normal':
        delta = 5.0 / sqrt(26)
        a     = sqrt(1.0 - delta ** 2)
        b     = delta
        zr    = _normal_raw_moments(0.0, 1.0, R)
        hn    = _halfnorm_moments(R)
        raw   = [float(sum(comb(r, j) * a**(r-j) * b**j * zr[r-j] * hn[j]
                           for j in range(r + 1)))
                 for r in range(R + 1)]
    elif dist_name == 'Pareto':
        raw = [_pareto_plus1_raw(r) for r in range(R + 1)]
    else:
        raise ValueError(f'Unknown distribution: {dist_name}')

    cm = _raw_to_central(raw, R)
    kp = _central_to_cumulants(cm, max_k)

    result = {}
    for k in range(2, max_k + 1):
        kp_val = kp.get(k, float('nan'))
        if np.isfinite(kp_val):
            result[k] = kp_val / factorial(k)
        else:
            print(f'      INFO: K_{k} ({dist_name}) diverges -- MC fallback (n=100M)',
                  flush=True)
            result[k] = _mc_cumulant_coeff(dist_name, k)
    return result


def _mc_cumulant_coeff(dist_name: str, k: int,
                       mc_size: int = 100_000_000) -> float:
    """MC estimate of K_k / k! when the analytical moment diverges."""
    rng = np.random.RandomState(999)
    x   = DISTRIBUTIONS[dist_name](rng, mc_size)
    c   = x - x.mean()
    m   = {r: float(np.mean(c ** r)) for r in range(2, k + 1)}
    kp: dict = {}
    kp[2] = m[2]
    if k >= 3: kp[3] = m[3]
    if k >= 4: kp[4] = m[4] - 3*m[2]**2
    if k >= 5: kp[5] = m[5] - 10*m[3]*m[2]
    if k >= 6: kp[6] = m[6] - 15*m[4]*m[2] - 10*m[3]**2 + 30*m[2]**3
    if k >= 7: kp[7] = m[7] - 21*m[5]*m[2] - 35*m[4]*m[3] + 210*m[3]*m[2]**2
    if k >= 8: kp[8] = (m[8] - 28*m[6]*m[2] - 56*m[5]*m[3] - 35*m[4]**2
                         + 420*m[4]*m[2]**2 + 560*m[3]**2*m[2] - 630*m[2]**4)
    return kp[k] / factorial(k)


# ===========================================================================
# MC pool  (generated once per distribution)
# ===========================================================================

def generate_pool(dist_name: str, sampler: callable,
                  rng: np.random.RandomState) -> np.ndarray:
    """
    Generate pool[T_MC, max_n] once per distribution.
    Trial t at sample size n uses pool[t, :n].
    """
    max_n = max(SAMPLE_SIZES)
    print(f'  Generating pool ({T_MC} x {max_n})...', end=' ', flush=True)
    pool = sampler(rng, T_MC * max_n).reshape(T_MC, max_n)
    print('done.', flush=True)
    return pool


# ===========================================================================
# Empirical moments and cumulants  (CtrdM route, uniform weights)
# ===========================================================================

def empirical_moments_and_cumulants(x: torch.Tensor,
                                    max_k: int) -> tuple[list, list]:
    """
    Central moments m[2..max_k] and cumulants kp[1..max_k] of a 1-D tensor.
    Returns
    -------
    m  : list length max_k+1; m[0]=1, m[1]=mean, m[r]=r-th central moment.
    kp : list length max_k+1; kp[1]=mean, kp[r]=r-th cumulant for r>=2.
    """
    mu1 = x.mean().item()
    c   = x - mu1
    m   = [0.0] * (max_k + 1)
    m[1] = mu1
    for r in range(2, max_k + 1):
        m[r] = float((c ** r).mean())

    kp    = [0.0] * (max_k + 1)
    kp[1] = mu1
    if max_k >= 2: kp[2] = m[2]
    if max_k >= 3: kp[3] = m[3]
    if max_k >= 4: kp[4] = m[4] - 3*m[2]**2
    if max_k >= 5: kp[5] = m[5] - 10*m[3]*m[2]
    if max_k >= 6: kp[6] = m[6] - 15*m[4]*m[2] - 10*m[3]**2 + 30*m[2]**3
    if max_k >= 7: kp[7] = m[7] - 21*m[5]*m[2] - 35*m[4]*m[3] + 210*m[3]*m[2]**2
    if max_k >= 8: kp[8] = (m[8] - 28*m[6]*m[2] - 56*m[5]*m[3] - 35*m[4]**2
                             + 420*m[4]*m[2]**2 + 560*m[3]**2*m[2] - 630*m[2]**4)
    return m, kp


# ===========================================================================
# V-statistic kernel MC  (LCC route)
# ===========================================================================

def lcc_kappas(x: torch.Tensor, max_k: int) -> dict:
    """
    LCC cumulant estimates = sample cumulants via Faà di Bruno inversion.
    The V-statistic with the locally centered cyclic kernel, summed over all
    n^k tuples, reduces algebraically to the sample cumulant at every order.
    For k=2,3: kappa_k = m_k exactly. For k>=4: kappa_k != m_k (lambda_k != 0).
    Returns dict with key 'K' mapping order -> float.
    """
    _, kp = empirical_moments_and_cumulants(x, max_k)
    return {'K': {k: kp[k] for k in ORDERS if k <= max_k}}


# ===========================================================================
# Exact sample utility
# ===========================================================================

def exact_utility(x: torch.Tensor, theta: float) -> float:
    """u = (1/theta) ln( (1/n) sum exp(theta x_i) ), log-sum-exp stable."""
    mx = x.max().item()
    return mx + float(torch.log(
        torch.mean(torch.exp(theta * (x - mx))))) / theta


# ===========================================================================
# Newton solver for theta
# ===========================================================================

def newton_theta(coeffs: list, order: int, u_target: float,
                 theta0: float, max_iter: int = 100,
                 tol: float = 1e-12) -> float:
    """
    Solve  u_hat(theta) = u_target  via Newton's method, where
        u_hat(theta) = coeffs[1] + sum_{r=2}^{order} theta^{r-1}/r! coeffs[r].
    coeffs : 0-indexed; coeffs[r] = K_r (LCC) or m_r (CtrdM) for r>=1.
    Returns theta_hat in (0,1), or NaN if Newton diverges or domain violated.
    """
    theta    = float(theta0)
    inv_fact = [1.0 / factorial(r) for r in range(order + 1)]

    for _ in range(max_iter):
        f  = coeffs[1] - u_target
        fp = 0.0
        tp = 1.0
        for r in range(2, order + 1):
            f  += tp * theta * inv_fact[r] * coeffs[r]
            fp += (r - 1) * tp * inv_fact[r] * coeffs[r]
            tp *= theta
        if abs(fp) < 1e-30:
            return float('nan')
        step  = f / fp
        theta -= step
        if abs(step) < tol:
            break

    return theta if (np.isfinite(theta) and 0.0 < theta < 1.0) else float('nan')


# ===========================================================================
# Statistical tests
# ===========================================================================

def pval_two_sided(arr: np.ndarray) -> float:
    n  = len(arr)
    mu = np.nanmean(arr)
    se = np.nanstd(arr, ddof=1) / np.sqrt(n)
    if se < 1e-30:
        return 1.0 if abs(mu) < 1e-30 else 0.0
    return float(2.0 * sp_stats.t.sf(abs(mu / se), df=n - 1))


def pval_one_sided_less(arr: np.ndarray) -> float:
    """P-value for H1: mean(arr) < 0."""
    n  = len(arr)
    mu = np.nanmean(arr)
    se = np.nanstd(arr, ddof=1) / np.sqrt(n)
    if se < 1e-30:
        return 0.0 if mu < -1e-30 else 1.0
    return float(sp_stats.t.cdf(mu / se, df=n - 1))


# ===========================================================================
# Print / CSV helpers
# ===========================================================================

def _hdr(orders: list) -> str:
    h = f'    {"n":>6}'
    for k in orders:
        h += f'  {"k="+str(k):>12}  {"p":>6}'
    return h


def _sep(orders: list) -> str:
    return '    ' + '-' * (6 + 20 * len(orders))


def _report_section_a(dist_name: str, dev_lcc: dict, dev_ctrdm: dict,
                      writer) -> None:
    for label, dev in [('LCC', dev_lcc), ('CtrdM', dev_ctrdm)]:
        tag = {'LCC':   'Panel (a): kappa_k/k! - K_k^pop/k!  [LCC]',
               'CtrdM': 'Panel (b): m_k/k!     - K_k^pop/k!  [CtrdM]'}[label]
        print(f'\n  Section A, {tag}')
        print(f'  T={T_MC}')
        print(_hdr(ORDERS));  print(_sep(ORDERS))
        for n in SAMPLE_SIZES:
            row    = f'    {n:>6}'
            se_row = f'    {"":>6}'
            for k in ORDERS:
                d   = dev[n][k]
                mu  = float(d.mean())
                se  = float(d.std(ddof=1) / np.sqrt(T_MC))
                pv  = pval_two_sided(d)
                row    += f'  {mu:>12.6f}  {pv:>6.3f}'
                se_row += f'  ({se:>10.6f})  {"":>6}'
                writer.writerow([dist_name, label, n, k,
                                 f'{mu:.10e}', f'{se:.10e}', f'{pv:.6f}'])
            print(row);  print(se_row)
    sys.stdout.flush()


def _report_section_c(dist_name: str, theta: float,
                      res_lcc: dict, res_ctrdm: dict,
                      writer) -> None:
    for label, store in [('LCC', res_lcc), ('CtrdM', res_ctrdm)]:
        tag = {'LCC':   'Panel (a): (u_LCC   - u_true)/|u_true|',
               'CtrdM': 'Panel (b): (u_CtrdM - u_true)/|u_true|'}[label]
        print(f'\n  Section C, {tag}  [theta={theta}]')
        print(_hdr(ORDERS));  print(_sep(ORDERS))
        for n in SAMPLE_SIZES:
            row    = f'    {n:>6}'
            se_row = f'    {"":>6}'
            for k in ORDERS:
                d   = store[n][k]
                mu  = float(d.mean())
                se  = float(d.std(ddof=1) / np.sqrt(T_MC))
                pv  = pval_two_sided(d)
                row    += f'  {mu:>12.6f}  {pv:>6.3f}'
                se_row += f'  ({se:>10.6f})  {"":>6}'
                writer.writerow([dist_name, label, theta, n, k,
                                 f'{mu:.10e}', f'{se:.10e}', f'{pv:.6f}'])
            print(row);  print(se_row)

    print(f'\n  Section C, Panel (c): |res_LCC| - |res_CtrdM|  [theta={theta}]')
    print(f'  H0: |res_LCC| >= |res_CtrdM|,  H1: LCC dominates (one-sided)')
    print(_hdr(ORDERS));  print(_sep(ORDERS))
    for n in SAMPLE_SIZES:
        row    = f'    {n:>6}'
        se_row = f'    {"":>6}'
        for k in ORDERS:
            diff = np.abs(res_lcc[n][k]) - np.abs(res_ctrdm[n][k])
            mu   = float(diff.mean())
            se   = float(diff.std(ddof=1) / np.sqrt(T_MC))
            pv   = pval_one_sided_less(diff)
            row    += f'  {mu:>12.6f}  {pv:>6.3f}'
            se_row += f'  ({se:>10.6f})  {"":>6}'
            writer.writerow([dist_name, 'dominance', theta, n, k,
                             f'{mu:.10e}', f'{se:.10e}', f'{pv:.6f}'])
        print(row);  print(se_row)
    sys.stdout.flush()


def _report_section_d(dist_name: str, theta: float,
                      th_lcc: dict, th_ctrdm: dict,
                      writer) -> None:
    rho_true = 1.0 / (1.0 - theta)
    for label, store in [('LCC', th_lcc), ('CtrdM', th_ctrdm)]:
        print(f'\n  Section D [{label}]:  theta_true={theta},  '
              f'rho_true={rho_true:.4f}')
        hdr = f'    {"n":>6}'
        for k in ORDERS:
            hdr += f'  {"k="+str(k):>12}  {"(se)":>10}'
        print(hdr)
        print('    ' + '-' * (6 + 24 * len(ORDERS)))

        stats: dict = {}
        for n in SAMPLE_SIZES:
            for k in ORDERS:
                arr   = store[n][k]
                valid = arr[np.isfinite(arr)]
                if len(valid) > 1:
                    tm = float(valid.mean())
                    ts = float(valid.std(ddof=1) / np.sqrt(len(valid)))
                    rv = (1.0 / (1.0 - valid))[np.isfinite(1.0 / (1.0 - valid))]
                    rm = float(rv.mean())  if len(rv) > 1 else float('nan')
                    rs = (float(rv.std(ddof=1) / np.sqrt(len(rv)))
                          if len(rv) > 1 else float('nan'))
                else:
                    tm = ts = rm = rs = float('nan')
                stats[(n, k)] = (tm, ts, rm, rs)
                writer.writerow([dist_name, label, theta, n, k,
                                 f'{tm:.10e}', f'{ts:.10e}',
                                 f'{rm:.10e}', f'{rs:.10e}'])

        print('    theta_hat:')
        for n in SAMPLE_SIZES:
            row = f'    {n:>6}'
            for k in ORDERS:
                tm, ts, _, _ = stats[(n, k)]
                row += f'  {tm:>12.6f}  {ts:>10.6f}'
            print(row)

        print('    rho_hat = 1/(1-theta_hat):')
        for n in SAMPLE_SIZES:
            row = f'    {n:>6}'
            for k in ORDERS:
                _, _, rm, rs = stats[(n, k)]
                row += f'  {rm:>12.6f}  {rs:>10.6f}'
            print(row)
    sys.stdout.flush()


# ===========================================================================
# Main
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='V-statistic kernel properties: LCC vs CtrdM')
    parser.add_argument('--theta', type=float, nargs='+', default=[0.5],
                        help='Theta values for Sections C and D (default: 0.5)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device override: cuda, cpu (default: auto-detect)')
    parser.add_argument('--native', action='store_true',
                        help='Use fused CuPy CUDA kernel (native mode)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation run: T_MC=100, n in [10,100,1000]')
    return parser.parse_args()


def main() -> None:
    args       = parse_args()
    global DEVICE, T_MC, SAMPLE_SIZES
    if args.device:
        DEVICE = torch.device(args.device)
        torch.set_default_device(DEVICE)
    if args.quick:
        T_MC         = 100
        SAMPLE_SIZES = [10, 100, 1_000]
        print('Quick mode: T_MC=100, SAMPLE_SIZES=[10, 100, 1000]', flush=True)
    theta_list = sorted(args.theta)
    if args.native:
        print('NOTE: --native flag ignored; LCC now uses exact sample cumulants.')
    max_order = max(ORDERS)

    rng = np.random.RandomState(42)
    os.makedirs(CSV_DIR, exist_ok=True)

    csv_a_path = os.path.join(CSV_DIR, 'section_a.csv')
    csv_c_path = os.path.join(CSV_DIR, 'section_c.csv')
    csv_d_path = os.path.join(CSV_DIR, 'section_d.csv')

    with (open(csv_a_path, 'w', newline='') as fa,
          open(csv_c_path, 'w', newline='') as fc,
          open(csv_d_path, 'w', newline='') as fd):

        wa = csv.writer(fa);  wc = csv.writer(fc);  wd = csv.writer(fd)
        wa.writerow(['dist', 'panel', 'n', 'k', 'mean', 'se', 'p_two_sided'])
        wc.writerow(['dist', 'panel', 'theta', 'n', 'k', 'mean', 'se', 'p_value'])
        wd.writerow(['dist', 'panel', 'theta_true', 'n', 'k',
                     'theta_mean', 'theta_se', 'rho_mean', 'rho_se'])

        dist_bar = tqdm(DISTRIBUTIONS.items(), desc='Distribution',
                        position=0, leave=True, file=sys.stdout)

        for dist_name, sampler in dist_bar:
            dist_bar.set_description(f'Distribution: {dist_name}')
            print(f'\n{"="*70}\n  {dist_name}\n{"="*70}')

            # --- Population cumulants ---
            print('  Population cumulants (analytical)...', end=' ', flush=True)
            true_coeff = population_cumulants(dist_name, max_order)
            print('done.')
            print('  K_k/k!:  ' + '   '.join(
                f'k={k}: {true_coeff[k]:>9.5f}' for k in ORDERS), flush=True)

            # --- Generate pool once for this distribution ---
            pool = generate_pool(dist_name, sampler, rng)  # (T_MC, max_n)

            # ===========================================================
            # Section A
            # ===========================================================
            dev_lcc   = {n: {k: np.zeros(T_MC) for k in ORDERS}
                         for n in SAMPLE_SIZES}
            dev_ctrdm = {n: {k: np.zeros(T_MC) for k in ORDERS}
                         for n in SAMPLE_SIZES}

            n_bar = tqdm(SAMPLE_SIZES, desc='  Section A  n',
                         position=1, leave=False, file=sys.stdout)
            for n in n_bar:
                n_bar.set_description(f'  Section A  n={n}')
                for t in tqdm(range(T_MC), desc='    trial', position=2,
                              leave=False, mininterval=2.0, file=sys.stdout):
                    xs_np = pool[t, :n]
                    xs    = torch.tensor(xs_np, device=DEVICE, dtype=DTYPE)
                    m, kp = empirical_moments_and_cumulants(xs, max_order)

                    for k in ORDERS:
                        dev_ctrdm[n][k][t] = m[k]  / factorial(k) - true_coeff[k]
                        dev_lcc[n][k][t]   = kp[k] / factorial(k) - true_coeff[k]

            _report_section_a(dist_name, dev_lcc, dev_ctrdm, wa)
            fa.flush()

            # ===========================================================
            # Sections C and D
            # ===========================================================
            for theta in theta_list:
                print(f'\n  ---- theta = {theta} ----', flush=True)

                res_lcc   = {n: {k: np.zeros(T_MC) for k in ORDERS}
                             for n in SAMPLE_SIZES}
                res_ctrdm = {n: {k: np.zeros(T_MC) for k in ORDERS}
                             for n in SAMPLE_SIZES}
                th_lcc    = {n: {k: np.zeros(T_MC) for k in ORDERS}
                             for n in SAMPLE_SIZES}
                th_ctrdm  = {n: {k: np.zeros(T_MC) for k in ORDERS}
                             for n in SAMPLE_SIZES}

                n_bar = tqdm(SAMPLE_SIZES, desc=f'  C/D theta={theta}  n',
                             position=1, leave=False, file=sys.stdout)
                for n in n_bar:
                    n_bar.set_description(f'  C/D theta={theta}  n={n}')
                    for t in tqdm(range(T_MC), desc='    trial', position=2,
                                  leave=False, mininterval=2.0, file=sys.stdout):
                        xs_np  = pool[t, :n]
                        xs_gpu = torch.tensor(xs_np, device=DEVICE, dtype=DTYPE)
                        m, kp  = empirical_moments_and_cumulants(xs_gpu, max_order)
                        u_true = exact_utility(xs_gpu, theta)
                        kappa1 = float(kp[1])

                        # LCC: recover K_k via triangular partition inversion
                        lcc   = lcc_kappas(xs_gpu, max_order)
                        K_lcc = lcc['K']

                        # Cumulative Maclaurin sums
                        u_lcc_cum   = kappa1
                        u_ctrdm_cum = kappa1

                        for r in range(2, max_order + 1):
                            c_r          = (theta ** (r - 1)) / factorial(r)
                            u_lcc_cum   += c_r * K_lcc[r]
                            u_ctrdm_cum += c_r * m[r]

                            if r in ORDERS:
                                norm = abs(u_true) if u_true != 0.0 else 1.0
                                res_lcc[n][r][t]   = (u_lcc_cum   - u_true) / norm
                                res_ctrdm[n][r][t] = (u_ctrdm_cum - u_true) / norm

                                c_lcc   = [0.0, kappa1] + [K_lcc[rr]
                                           for rr in range(2, r + 1)]
                                c_ctrdm = [0.0, kappa1] + [m[rr]
                                           for rr in range(2, r + 1)]
                                th_lcc[n][r][t]   = newton_theta(
                                    c_lcc,   r, u_true, theta)
                                th_ctrdm[n][r][t] = newton_theta(
                                    c_ctrdm, r, u_true, theta)

                _report_section_c(dist_name, theta, res_lcc, res_ctrdm, wc)
                _report_section_d(dist_name, theta, th_lcc, th_ctrdm, wd)
                fc.flush();  fd.flush()

    print(f'\n  CSV saved: {csv_a_path}')
    print(f'\n  CSV saved: {csv_c_path}')
    print(f'\n  CSV saved: {csv_d_path}')


if __name__ == '__main__':
    main()
