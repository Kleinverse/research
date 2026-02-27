"""
lccfit.converter
================
Convert raw data into locally-centered cyclic (LCC) kernel samples.

The converter draws k-tuples from the empirical distribution (optionally
weighted), evaluates the LCC kernel h_k = prod_{j=1}^k (z_j - mu_k) for
each order, and returns an LCCData object that is ready for fitting via
statsmodels, scipy, or the lccfit Newton solver.

Researcher-facing options
-------------------------
log         : bool, default False
    Apply natural-log transform to the input data before processing.
    Useful when the raw data are prices, quantities, or unit values.

weights     : array-like or None, default None
    Expenditure shares or other non-negative weights.  When provided,
    k-tuples are drawn from Categorical(w_i / sum(w_i)).  None means
    uniform sampling.

orders      : list of int, default [2, 3, 4, 5, 6, 7, 8]
    Kernel orders to compute.  Any positive integer >= 2 is valid.

n_mc        : int, default 100_000
    Number of Monte Carlo k-tuples per order.

dtype       : 'float64' or 'float32', default 'float64'
    Numerical precision.  float32 is faster on modern GPUs.

device      : 'auto', 'cuda', 'cpu', or torch.device, default 'auto'
    'auto' uses CUDA if available, else CPU.

seed        : int or None, default None
    Random seed for reproducibility (CPU path only; GPU sampling is
    non-deterministic by default).

store_samples : bool, default True
    If True, the raw [n_mc] kernel product draws for each order are
    stored in LCCData.samples (a dict of numpy arrays).  Set to False
    to reduce memory when only E[h_k] is needed.

shared_draws : bool, default False
    If True, all orders share the same random index draws within each
    MC trial: draw max(orders) indices once per trial, then for order k
    use the first k of them.  This makes (h_2^(t), h_3^(t), ...) jointly
    coherent observations, which is required for regression-based fitting.
    When False (default), each order draws independently.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

__all__ = ['convert', 'LCCData']

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class LCCData:
    """
    Container for LCC kernel estimates derived from a single data vector.

    Attributes
    ----------
    Ehk : dict[int, float]
        Monte Carlo estimate of E[h_k] for each requested order k.
        This is the key quantity for cumulant inversion and theta fitting.

    samples : dict[int, np.ndarray] or None
        Raw kernel product draws h_k^(t), shape [n_mc], for each order k.
        None if store_samples=False was passed to convert().
        Directly usable as moment-condition observations for statsmodels GMM:
            nobs = n_mc,  n_moments = len(orders)

    n : int
        Number of observations in the original data.

    orders : list[int]
        Kernel orders that were computed.

    n_mc : int
        Number of Monte Carlo draws used per order.

    log_transformed : bool
        Whether a log transform was applied to the input.

    weighted : bool
        Whether weighted sampling was used.

    dtype : str
        Numerical precision used ('float64' or 'float32').

    device : str
        Device used ('cuda' or 'cpu').

    meta : dict
        Free-form metadata dict (e.g. commodity code, year) for the caller
        to attach any identifiers.

    Methods
    -------
    to_moments_matrix()
        Returns an [n_mc, len(orders)] numpy array suitable for passing
        directly to statsmodels GMM as the moment-conditions matrix.

    summary()
        Print a compact human-readable summary.
    """

    Ehk:             Dict[int, float]
    samples:         Optional[Dict[int, np.ndarray]]
    n:               int
    orders:          List[int]
    n_mc:            int
    log_transformed: bool
    weighted:        bool
    shared_draws:    bool
    dtype:           str
    device:          str
    meta:            Dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_moments_matrix(self) -> np.ndarray:
        """
        Return an [n_mc, len(orders)] array of kernel product draws.

        Each column corresponds to one order (in the same order as
        self.orders).  Suitable for statsmodels GMM:

            import statsmodels.api as sm
            data = lccfit.convert(z, orders=[2,4,6], n_mc=50_000)
            M = data.to_moments_matrix()   # [50000, 3]
        """
        if self.samples is None:
            raise ValueError(
                "samples were not stored (store_samples=False). "
                "Re-run convert() with store_samples=True.")
        return np.column_stack([self.samples[k] for k in self.orders])

    # ------------------------------------------------------------------
    def save(self, path: str, fmt: str = 'csv') -> None:
        """
        Save E[h_k] estimates (and optional meta) to disk.

        Parameters
        ----------
        path : str
            Output file path.  Extension is appended if absent.
        fmt : {'csv', 'parquet', 'json'}, default 'csv'
            Output format.
            - 'csv'     : flat CSV, one row per LCCData object.
            - 'parquet' : Apache Parquet (requires pyarrow or fastparquet).
            - 'json'    : JSON lines record.

        The saved row contains all meta key-value pairs, n, n_mc,
        log_transformed, weighted, dtype, device, and Ehk_{k} for each order.
        """
        import os
        import json
        import pandas as pd

        # Build flat record
        row: dict = {}
        row.update(self.meta)
        row['n']               = self.n
        row['n_mc']            = self.n_mc
        row['log_transformed'] = self.log_transformed
        row['weighted']        = self.weighted
        row['dtype']           = self.dtype
        row['device']          = self.device
        for k, v in self.Ehk.items():
            row[f'Ehk_{k}'] = v

        # Ensure correct extension
        ext_map = {'csv': '.csv', 'parquet': '.parquet', 'json': '.json', 'lcc': '.lcc'}
        if fmt not in ext_map:
            raise ValueError(f"fmt must be one of {list(ext_map)}, got {fmt!r}")
        expected_ext = ext_map[fmt]
        # Strip any existing extension before appending the correct one
        base = path
        for ext in ext_map.values():
            if base.endswith(ext):
                base = base[:-len(ext)]
                break
        path = base + expected_ext

        df = pd.DataFrame([row])

        if fmt in ('csv', 'lcc'):
            write_header = not os.path.exists(path)
            df.to_csv(path, mode='a', header=write_header, index=False)
        elif fmt == 'parquet':
            if os.path.exists(path):
                existing = pd.read_parquet(path)
                df = pd.concat([existing, df], ignore_index=True)
            df.to_parquet(path, index=False)
        elif fmt == 'json':
            with open(path, 'a') as f:
                f.write(json.dumps(row) + '\n')

    def summary(self) -> None:
        """Print a compact summary of the LCC estimates."""
        print("LCCData summary")
        print(f"  n={self.n}  n_mc={self.n_mc}  "
              f"orders={self.orders}  "
              f"dtype={self.dtype}  device={self.device}")
        print(f"  log_transformed={self.log_transformed}  "
              f"weighted={self.weighted}")
        if self.meta:
            print(f"  meta={self.meta}")
        print("  E[h_k]:")
        for k, v in self.Ehk.items():
            print(f"    k={k}: {v:+.8e}")


# ---------------------------------------------------------------------------
# Internal GPU kernel
# ---------------------------------------------------------------------------

def _mc_Ehk_gpu(
    z:       torch.Tensor,   # [n]
    w_norm:  torch.Tensor,   # [n]  (already normalised; uniform if equal)
    k:       int,
    n_mc:    int,
    store:   bool,
) -> tuple[float, Optional[np.ndarray]]:
    """
    Draw n_mc k-tuples from Categorical(w_norm), compute
    h_k^(t) = prod_{j=1}^k (z_j^(t) - mu_k^(t)), return
    (E[h_k], draws_or_None).
    """
    n = z.shape[0]

    # Sample indices: [n_mc, k]
    idx = torch.multinomial(w_norm.unsqueeze(0).expand(n_mc, -1),
                            k, replacement=True)          # [n_mc, k]

    # Gather values: [n_mc, k]
    draws = z[idx]

    # Locally-centered product kernel
    mu    = draws.mean(dim=1, keepdim=True)               # [n_mc, 1]
    prods = (draws - mu).prod(dim=1)                      # [n_mc]

    Ehk   = float(prods.mean())
    arr   = prods.cpu().numpy() if store else None
    return Ehk, arr



def _mc_Ehk_shared_gpu(
    z:       "torch.Tensor",   # [n]
    w_norm:  "torch.Tensor",   # [n]  normalised weights
    orders:  list,
    n_mc:    int,
    store:   bool,
) -> "tuple[dict, dict]":
    """
    Draw n_mc max_order-tuples ONCE, compute h_k for all orders from
    the same draw by taking the first k indices of each trial.

    Returns
    -------
    Ehk     : dict[int, float]            E[h_k] per order
    samples : dict[int, ndarray or None]  raw [n_mc] draws per order
    """
    import torch
    max_k = max(orders)

    # Single draw: [n_mc, max_k]  — shared across all orders
    idx = torch.multinomial(
        w_norm.unsqueeze(0).expand(n_mc, -1),
        max_k, replacement=True
    )

    # Gather all values at once: [n_mc, max_k]
    vals = z[idx]   # [n_mc, max_k]

    Ehk:     dict = {}
    samples: dict = {}

    for k in sorted(orders):
        sub  = vals[:, :k]                            # [n_mc, k]
        mu   = sub.mean(dim=1, keepdim=True)          # [n_mc, 1]
        prods = (sub - mu).prod(dim=1)                # [n_mc]
        Ehk[k]     = float(prods.mean())
        samples[k] = prods.cpu().numpy() if store else None

    return Ehk, samples


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert(
    data:          Union[np.ndarray, list, torch.Tensor],
    log:           bool                         = False,
    weights:       Optional[Union[np.ndarray, list]] = None,
    orders:        Sequence[int]                = (2, 3, 4, 5, 6, 7, 8),
    n_mc:          int                          = 100_000,
    dtype:         str                          = 'float64',
    device:        Union[str, torch.device]     = 'auto',
    seed:          Optional[int]                = None,
    store_samples: bool                         = True,
    shared_draws:  bool                         = False,
    meta:          Optional[dict]               = None,
) -> LCCData:
    """
    Convert raw data into LCC kernel estimates.

    Parameters
    ----------
    data : array-like, shape [n]
        Raw observations.  Can be prices, unit values, log-prices, etc.
        See the `log` parameter.

    log : bool, default False
        Apply z = log(data) before processing.  Silently drops non-positive
        values with a warning.

    weights : array-like of shape [n] or None, default None
        Non-negative sampling weights (e.g. expenditure shares).  Will be
        normalised to sum to 1.  None = uniform weights.

    orders : sequence of int, default (2, 3, 4, 5, 6, 7, 8)
        Kernel orders to compute.  Each must be >= 2.

    n_mc : int, default 100_000
        Number of Monte Carlo k-tuples per order.

    dtype : {'float64', 'float32'}, default 'float64'
        Numerical precision.  float32 roughly doubles GPU throughput.

    device : {'auto', 'cuda', 'cpu'} or torch.device, default 'auto'
        Computation device.  'auto' selects CUDA if available.

    seed : int or None, default None
        Random seed for the CPU fallback path (torch.manual_seed).
        GPU multinomial is non-deterministic by default; pass seed to
        enforce reproducibility on CPU.

    store_samples : bool, default True
        If True, store the raw [n_mc] kernel product arrays in
        LCCData.samples for use with statsmodels GMM or other fitters.
        Set False to save memory when only E[h_k] scalars are needed.

    meta : dict or None, default None
        Optional metadata attached to the returned LCCData (e.g.
        {'commodity': '0101010000', 'year': 2024}).

    Returns
    -------
    LCCData

    Examples
    --------
    >>> import numpy as np
    >>> import lccfit
    >>> z = np.random.lognormal(0, 0.5, 500)
    >>> data = lccfit.convert(z, log=True, orders=[2, 4, 6], n_mc=50_000)
    >>> data.summary()

    Weighted (trade data):
    >>> data = lccfit.convert(uv, log=True, weights=shares,
    ...                       orders=[2, 4, 6], n_mc=100_000)

    Float32 for speed:
    >>> data = lccfit.convert(z, dtype='float32', n_mc=200_000)
    """
    # ------------------------------------------------------------------
    # 0. Validate arguments
    # ------------------------------------------------------------------
    if dtype not in ('float64', 'float32'):
        raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")

    orders = list(orders)
    if any(k < 2 for k in orders):
        raise ValueError("All orders must be >= 2.")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1.")

    # ------------------------------------------------------------------
    # 1. Resolve device
    # ------------------------------------------------------------------
    if isinstance(device, torch.device):
        dev = device
    elif device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    torch_dtype = torch.float64 if dtype == 'float64' else torch.float32

    # ------------------------------------------------------------------
    # 2. Prepare data array
    # ------------------------------------------------------------------
    arr = np.asarray(data, dtype=np.float64).ravel()

    log_transformed = log
    if log:
        n_before = len(arr)
        bad = arr <= 0
        if bad.any():
            warnings.warn(
                f"log=True: dropping {bad.sum()} non-positive value(s) "
                f"out of {n_before}.",
                UserWarning, stacklevel=2)
            arr = arr[~bad]
            if weights is not None:
                weights = np.asarray(weights, dtype=np.float64).ravel()[~bad]
        arr = np.log(arr)

    n = len(arr)
    if n < max(orders):
        raise ValueError(
            f"Need at least {max(orders)} observations for order "
            f"{max(orders)}, but n={n} after preprocessing.")

    # ------------------------------------------------------------------
    # 3. Prepare weights
    # ------------------------------------------------------------------
    weighted = weights is not None
    if weighted:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if len(w) != n:
            raise ValueError(
                f"weights length ({len(w)}) must match data length ({n}).")
        if np.any(w < 0):
            raise ValueError("weights must be non-negative.")
        w_sum = w.sum()
        if w_sum <= 0:
            raise ValueError("weights must have a positive sum.")
        w = w / w_sum
    else:
        w = np.ones(n, dtype=np.float64) / n

    # ------------------------------------------------------------------
    # 4. Move to device
    # ------------------------------------------------------------------
    if seed is not None:
        torch.manual_seed(seed)

    z_t = torch.tensor(arr, dtype=torch_dtype, device=dev)
    w_t = torch.tensor(w,   dtype=torch_dtype, device=dev)

    # ------------------------------------------------------------------
    # 5. Compute E[h_k] for each order
    # ------------------------------------------------------------------
    Ehk     : Dict[int, float]               = {}
    samples : Dict[int, Optional[np.ndarray]] = {}

    if shared_draws:
        # All orders share the same random index draws per trial — required
        # for regression-based fitting (each trial produces a joint observation
        # across all orders from the same drawn k-tuple).
        Ehk, samples = _mc_Ehk_shared_gpu(z_t, w_t, list(orders), n_mc, store_samples)
    else:
        for k in orders:
            ehk_val, draws = _mc_Ehk_gpu(z_t, w_t, k, n_mc, store_samples)
            Ehk[k]     = ehk_val
            samples[k] = draws

    # ------------------------------------------------------------------
    # 6. Return LCCData
    # ------------------------------------------------------------------
    return LCCData(
        Ehk             = Ehk,
        samples         = samples if store_samples else None,
        n               = n,
        orders          = orders,
        n_mc            = n_mc,
        log_transformed = log_transformed,
        weighted        = weighted,
        shared_draws    = shared_draws,
        dtype           = dtype,
        device          = str(dev),
        meta            = dict(meta) if meta is not None else {},
    )
