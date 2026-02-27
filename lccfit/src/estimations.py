#!/usr/bin/env python3
"""
estimate_theta_vstat.py  (GPU-accelerated, PyTorch)

Continuum translog estimation of sigma from US import data.
Uses V-statistic centered kernels h_k = prod(z_j - z_bar_k)
with weighted MC sampling, then inverts the partition formula
to recover CGF cumulants kappa_2 ... kappa_6, and solves for
theta at orders k = 2, 4, 6.

Pipeline:
  1. For each cell, draw k-tuples from weighted empirical distribution
  2. Evaluate kernel h_k = prod_{j=1}^k (z_j - z_bar_k)
  3. Average over trials to estimate E[h_k]
  4. Invert partition formula: kappa_k = (E[h_k] - lower) / alpha(k,k)
  5. Newton solve: ln U(theta) - kappa_1 = sum_{r=2}^{order} (theta^{r-1}/r!) kappa_r

Requires: torch (ROCm or CUDA), pandas, numpy

Usage:
  python estimation.py -i imports_2024.csv -o sigma_estimates.csv
  python estimation.py -i imports_2024.csv -o results.csv --min-varieties 8

Author: T. Saito (2026)
"""
import argparse
import sys
import time
from math import factorial
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float64  # overridden by --float32 at startup

def to_dev(x):
    return torch.tensor(x, dtype=DTYPE, device=DEVICE)


# =========================================================================
# Pad ragged cells into fixed-width tensors
# =========================================================================

def pad_cells(cell_list):
    """
    cell_list: list of (x_array, w_array) per cell
    Returns: X [C, M], W [C, M], mask [C, M]  on device
    """
    M = max(len(c[0]) for c in cell_list)
    C = len(cell_list)
    X = np.zeros((C, M), dtype=np.float64)
    W = np.zeros((C, M), dtype=np.float64)
    mask = np.zeros((C, M), dtype=np.float64)
    for i, (x, w) in enumerate(cell_list):
        n = len(x)
        X[i, :n] = x
        W[i, :n] = w
        mask[i, :n] = 1.0
    return to_dev(X), to_dev(W), to_dev(mask)


# =========================================================================
# V-statistic kernel E[h_k] via weighted MC sampling (batch over cells)
# =========================================================================

def batch_vstat_Ehk(X, W, mask, k, n_mc):
    """
    MC estimate of E[h_k] = E[ prod_{j=1}^k (z_j - z_bar_k) ]
    under the weighted empirical distribution F = sum_i w_i delta_{x_i}.

    Draws k indices per trial from Categorical(w_1, ..., w_n),
    computes the locally-centered product kernel, and averages.

    X: [C, M], W: [C, M], mask: [C, M]
    Returns: Ehk [C]
    """
    C, M = X.shape

    # Normalize weights per cell (mask out padding)
    W_safe = W * mask
    W_safe = W_safe / W_safe.sum(dim=1, keepdim=True).clamp(min=1e-30)

    # Weighted sampling: [C, n_mc * k] indices
    idx = torch.multinomial(W_safe, n_mc * k, replacement=True)
    idx = idx.reshape(C, n_mc, k)                       # [C, n_mc, k]

    # Gather x values via flat indexing
    offsets = torch.arange(C, dtype=torch.long, device=DEVICE).view(C, 1, 1) * M
    flat_idx = (offsets + idx).reshape(-1)
    z = X.reshape(-1)[flat_idx].reshape(C, n_mc, k)     # [C, n_mc, k]

    # Kernel: prod_{j=1}^k (z_j - z_bar_k)
    zbar = z.mean(dim=2, keepdim=True)                   # [C, n_mc, 1]
    prods = (z - zbar).prod(dim=2)                       # [C, n_mc]

    return prods.mean(dim=1)                             # [C]


# =========================================================================
# Partition formula inversion: E[h_k] -> CGF cumulants
# =========================================================================

def _alpha(r, k):
    """alpha(r, k) = (-1)^{r-1} (r-1)! / k^{r-1}  [eq. 37]"""
    return ((-1) ** (r - 1)) * factorial(r - 1) / (k ** (r - 1))

# Precomputed coefficients
_A22 = _alpha(2, 2)   # -1/2
_A33 = _alpha(3, 3)   #  2/9
_A44 = _alpha(4, 4)   # -3!/4^3  = -6/64  = -3/32
_A24 = _alpha(2, 4)   # -1/4
_A55 = _alpha(5, 5)   #  4!/5^4  = 24/625
_A25 = _alpha(2, 5)   # -1/5
_A35 = _alpha(3, 5)   #  2/25
_A66 = _alpha(6, 6)   # -5!/6^5  = -120/7776
_A26 = _alpha(2, 6)   # -1/6
_A46 = _alpha(4, 6)   # -3!/6^3  = -6/216  = -1/36
_A36 = _alpha(3, 6)   #  2/36


def batch_invert_cumulants(Eh2, Eh3, Eh4, Eh5, Eh6):
    """
    Invert the partition formula
        E[h_k] = alpha(k,k) kappa_k + (multi-block terms)
    to recover CGF cumulants kappa_2 ... kappa_6.
    """
    k2 = Eh2 / _A22
    k3 = Eh3 / _A33
    lower4 = 3.0 * _A24**2 * k2**2
    k4 = (Eh4 - lower4) / _A44
    lower5 = 10.0 * _A25 * _A35 * k2 * k3
    k5 = (Eh5 - lower5) / _A55
    lower6 = (15.0 * _A26 * _A46 * k2 * k4
            + 10.0 * _A36**2 * k3**2
            + 15.0 * _A26**3 * k2**3)
    k6 = (Eh6 - lower6) / _A66
    return k2, k3, k4, k5, k6


# =========================================================================
# Batch exact CES log-utility
# =========================================================================

def batch_lnU(X, W, theta, mask):
    """ln U = (1/theta) ln[ sum_i w_i exp(theta x_i) ]"""
    v = theta.unsqueeze(1) * X
    v = v + (1 - mask) * (-1e30)
    vm = v.max(dim=1, keepdim=True).values
    exp_v = W * torch.exp(v - vm) * mask
    return (vm.squeeze(1) + torch.log(exp_v.sum(dim=1).clamp(min=1e-30))) / theta


# =========================================================================
# Batch theta solver (Newton, orders 2, 4, 6)
# =========================================================================

def batch_solve_theta(X, W, mask, k1, k2, k3, k4, k5, k6,
                      order=2, max_iter=200, tol=1e-10):
    """
    Solve: ln U(theta) - k1 = sum_{r=2}^{order} (theta^{r-1}/r!) kappa_r
    via Newton iteration.
    """
    C = k1.shape[0]
    theta = 0.5 * torch.ones(C, dtype=DTYPE, device=DEVICE)

    for _ in range(max_iter):
        lnU_val = batch_lnU(X, W, theta, mask)
        z = lnU_val - k1
        t = theta
        f  = (k2/2)*t
        fp = k2/2
        if order >= 3:
            f  += (k3/6)*t**2
            fp += (k3/3)*t
        if order >= 4:
            f  += (k4/24)*t**3
            fp += (k4/8)*t**2
        if order >= 5:
            f  += (k5/120)*t**4
            fp += (k5/30)*t**3
        if order >= 6:
            f  += (k6/720)*t**5
            fp += (k6/144)*t**4
        f -= z
        fp = fp.clamp(min=1e-15)
        theta_new = theta - f / fp
        converged = (theta_new - theta).abs().max() < tol
        theta = 0.5 * theta + 0.5 * theta_new
        if converged:
            break

    theta = torch.where((theta > 0) & (theta < 1) & theta.isfinite(),
                        theta, torch.full_like(theta, float('nan')))
    return theta


# =========================================================================
# Data loading and normalization
# =========================================================================

def load_and_normalize(path):
    print(f"Loading {path}...", file=sys.stderr)
    if path.endswith('.dta'):
        df = pd.read_stata(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    print(f"  {len(df)} rows, columns: {list(df.columns)[:10]}...", file=sys.stderr)

    cols = set(df.columns)

    if 'commodity' not in cols:
        print("  No 'commodity' column found.", file=sys.stderr)
        return df, False

    val_col = None
    is_monthly = False
    for vc in ['gen_val', 'con_val']:
        if vc in cols:
            val_col = vc
            break
    if val_col is None:
        for vc in ['gen_val_mo', 'con_val_mo']:
            if vc in cols:
                val_col = vc
                is_monthly = True
                break
    if val_col is None:
        for vc in ['gen_val_yr', 'con_val_yr']:
            if vc in cols:
                val_col = vc
                if 'month' in cols:
                    df = df[df['month'] == 12].copy()
                break
    if val_col is None:
        print("  ERROR: no value column found.", file=sys.stderr)
        sys.exit(1)

    qty_col = None
    for qc in ['qty1', 'gen_qy1_mo', 'con_qy1_mo', 'gen_qy1_yr',
                'con_qy1_yr', 'qty']:
        if qc in cols:
            qty_col = qc
            break
    if qty_col is None:
        print("  ERROR: no quantity column found.", file=sys.stderr)
        sys.exit(1)

    print(f"  Detected: val={val_col}, qty={qty_col}, monthly={is_monthly}",
          file=sys.stderr)

    out = pd.DataFrame()
    out['commodity'] = df['commodity'].astype(str).str.zfill(10)
    out['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    out['country'] = df['cty_code'].astype(str)
    out['import_value'] = pd.to_numeric(df[val_col], errors='coerce')
    out['quantity'] = pd.to_numeric(df[qty_col], errors='coerce')

    out = out.dropna(subset=['import_value', 'quantity', 'year'])
    out = out[out['quantity'] > 0]
    out = out[out['import_value'] > 0]

    if is_monthly:
        out = out.groupby(['commodity', 'year', 'country'], as_index=False).agg(
            {'import_value': 'sum', 'quantity': 'sum'})
        print(f"  Aggregated monthly -> annual: {len(out)} rows", file=sys.stderr)

    return out, True


# =========================================================================
# Main
# =========================================================================


# =========================================================================
# Pre-computed path: read theta columns from run.py CSV
# =========================================================================

def run_from_precomputed(path, solve_orders):
    """
    Input is a CSV from run.py that already contains theta_O* columns.
    Reads those columns directly and adds sigma_O* columns.
    No recomputation: Newton requires raw z/w, not stored in summary CSV.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()

    theta_cols = {}
    for c in df.columns:
        if c.startswith('theta_o'):
            try:
                theta_cols[int(c[7:])] = c
            except ValueError:
                pass

    avail = sorted(theta_cols.keys())
    print(f"  Theta orders in file: {avail}", file=sys.stderr)

    missing = [o for o in solve_orders if o not in theta_cols]
    if missing:
        print(f"  ERROR: orders {missing} not in file.", file=sys.stderr)
        print(f"  Rerun: python run.py <raw.dta> --solve-orders "
              f"{' '.join(str(o) for o in sorted(set(avail+missing)))}",
              file=sys.stderr)
        sys.exit(1)

    out = df.copy()
    for order in solve_orders:
        tv = pd.to_numeric(out[theta_cols[order]], errors='coerce')
        out[f'theta_O{order}'] = tv
        out[f'sigma_O{order}'] = np.where(
            (tv > 0) & (tv < 1) & np.isfinite(tv), 1.0 / (1.0 - tv), np.nan)

    print(f"  {len(out)} rows, orders {solve_orders}", file=sys.stderr)
    return out


def main():
    p = argparse.ArgumentParser(
        description='Continuum translog sigma estimation via V-statistic '
                    'centered kernels (GPU)')
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', default='theta_results.csv')
    p.add_argument('--min-varieties', type=int, default=5)
    p.add_argument('--n-mc', type=int, default=100_000,
                   help='MC samples per cell per kernel order (default: 100000)')
    p.add_argument('--hs-filter', default=None,
                   help='Comma-separated HS10 codes')
    p.add_argument('--year-filter', default=None)
    p.add_argument('--hs-col', default=None)
    p.add_argument('--year-col', default=None)
    p.add_argument('--country-col', default=None)
    p.add_argument('--value-col', default=None)
    p.add_argument('--quantity-col', default=None)
    p.add_argument('--uv-col', default=None)
    p.add_argument('--batch-size', type=int, default=256,
                   help='Cells per GPU batch (default: 256)')
    p.add_argument('--float32', action='store_true',
                   help='Use float32 instead of float64 (faster, less precise)')
    args = p.parse_args()

    global DTYPE
    if args.float32:
        DTYPE = torch.float32
        torch.set_default_dtype(torch.float32)
        print(f"Dtype: float32", file=sys.stderr)
    else:
        print(f"Dtype: float64", file=sys.stderr)

    print(f"Device: {DEVICE}", file=sys.stderr)
    if DEVICE.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}", file=sys.stderr)

    # Detect pre-computed CSV from run.py (has theta_O* columns)
    if args.input.endswith('.csv'):
        probe = pd.read_csv(args.input, nrows=1)
        probe.columns = probe.columns.str.lower().str.strip()
        if any(c.startswith('theta_o') for c in probe.columns):
            print(f"  Detected pre-computed CSV -- reading theta columns directly.",
                  file=sys.stderr)
            solve_orders = [o for o in [2, 4, 6, 8] if o in
                            [int(c[7:]) for c in probe.columns if c.startswith('theta_o')
                             and c[7:].isdigit()]]
            out = run_from_precomputed(args.input, solve_orders)
            out.to_csv(args.output, index=False)
            print(f"Wrote {len(out)} rows to {args.output}", file=sys.stderr)
            return

    df, auto_detected = load_and_normalize(args.input)

    if auto_detected:
        hs_col, yr_col = 'commodity', 'year'
        val_col, qty_col = 'import_value', 'quantity'
    else:
        hs_col = args.hs_col or 'commodity'
        yr_col = args.year_col or 'year'
        val_col = args.value_col or 'import_value'
        qty_col = args.quantity_col or 'quantity'
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        df[qty_col] = pd.to_numeric(df[qty_col], errors='coerce')
        df = df.dropna(subset=[val_col, qty_col])
        df = df[df[qty_col] > 0]

    if args.hs_filter:
        hs_set = set(args.hs_filter.split(','))
        df = df[df[hs_col].astype(str).isin(hs_set)]
        print(f"  HS filter -> {len(df)} rows", file=sys.stderr)
    if args.year_filter:
        yr_set = set(int(y) for y in args.year_filter.split(','))
        df = df[df[yr_col].astype(int).isin(yr_set)]
        print(f"  Year filter -> {len(df)} rows", file=sys.stderr)

    # Recompute shares after all filtering so weights sum to 1 within each cell
    df = df.copy()
    df['_share'] = df.groupby([hs_col, yr_col])[val_col]\
        .transform(lambda v: v / v.sum())

    if args.uv_col and args.uv_col in df.columns:
        df['_uv'] = pd.to_numeric(df[args.uv_col], errors='coerce')
    else:
        df['_uv'] = df[val_col] / df[qty_col]
    df = df[(df['_uv'] > 0) & np.isfinite(df['_uv'])].copy()
    df['_x'] = np.log(df['_uv'])

    # Build cell list
    groups = df.groupby([hs_col, yr_col])
    cell_list = []
    cell_keys = []
    for (hs, yr), grp in groups:
        if len(grp) < args.min_varieties:
            continue
        cell_list.append((grp['_x'].values, grp['_share'].values))
        cell_keys.append((hs, yr))

    C = len(cell_list)
    print(f"  {C} cells to estimate on {DEVICE}  "
          f"(n_mc={args.n_mc} per kernel order)...", file=sys.stderr)
    if C == 0:
        print("  No cells meet min-varieties threshold.", file=sys.stderr)
        return

    all_rows = []
    BS = args.batch_size
    N_MC = args.n_mc
    t0 = time.time()

    for b_start in range(0, C, BS):
        b_end = min(b_start + BS, C)
        batch = cell_list[b_start:b_end]
        keys = cell_keys[b_start:b_end]
        BC = len(batch)

        X, W, mask = pad_cells(batch)

        # Weighted mean (kappa_1)
        k1 = (W * X * mask).sum(dim=1)

        # V-statistic kernels E[h_k] for k = 2, ..., 6
        Eh2 = batch_vstat_Ehk(X, W, mask, 2, N_MC)
        Eh3 = batch_vstat_Ehk(X, W, mask, 3, N_MC)
        Eh4 = batch_vstat_Ehk(X, W, mask, 4, N_MC)
        Eh5 = batch_vstat_Ehk(X, W, mask, 5, N_MC)
        Eh6 = batch_vstat_Ehk(X, W, mask, 6, N_MC)

        # Invert partition formula -> CGF cumulants
        k2, k3, k4, k5, k6 = batch_invert_cumulants(Eh2, Eh3, Eh4, Eh5, Eh6)

        # Solve theta at orders 2, 4, 6
        th2 = batch_solve_theta(X, W, mask, k1, k2, k3, k4, k5, k6, order=2)
        th4 = batch_solve_theta(X, W, mask, k1, k2, k3, k4, k5, k6, order=4)
        th6 = batch_solve_theta(X, W, mask, k1, k2, k3, k4, k5, k6, order=6)

        def cpu(t): return t.cpu().numpy()

        n_per_cell = [len(batch[i][0]) for i in range(BC)]

        for i in range(BC):
            hs, yr = keys[i]
            row = {
                'commodity': hs, 'year': yr, 'n': n_per_cell[i],
                'Eh2': float(cpu(Eh2)[i]),
                'Eh3': float(cpu(Eh3)[i]),
                'Eh4': float(cpu(Eh4)[i]),
                'Eh5': float(cpu(Eh5)[i]),
                'Eh6': float(cpu(Eh6)[i]),
                'kappa2': float(cpu(k2)[i]),
                'kappa3': float(cpu(k3)[i]),
                'kappa4': float(cpu(k4)[i]),
                'kappa5': float(cpu(k5)[i]),
                'kappa6': float(cpu(k6)[i]),
                'theta_O2': float(cpu(th2)[i]),
                'theta_O4': float(cpu(th4)[i]),
                'theta_O6': float(cpu(th6)[i]),
            }
            for order_tag, th_arr in [('O2', th2), ('O4', th4), ('O6', th6)]:
                tv = float(cpu(th_arr)[i])
                row[f'sigma_{order_tag}'] = (1.0 / (1.0 - tv)) if (tv > 0 and tv < 1 and np.isfinite(tv)) else np.nan
            all_rows.append(row)

        elapsed = time.time() - t0
        print(f"  Batch {b_start}-{b_end}/{C}  ({elapsed:.1f}s)",
              file=sys.stderr)

    out = pd.DataFrame(all_rows)
    id_cols = ['commodity', 'year', 'n']
    col_order = [c for c in id_cols if c in out.columns]
    col_order += [c for c in out.columns if c not in col_order]
    out[col_order].to_csv(args.output, index=False)
    print(f"Wrote {len(out)} rows to {args.output}  "
          f'({time.time()-t0:.1f}s total)', file=sys.stderr)


if __name__ == '__main__':
    main()
