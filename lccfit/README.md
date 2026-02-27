# lccfit

> [!CAUTION]
> This package is in active development. APIs may change without notice and results should be verified against the reference implementation in the research repository. Use in production is not recommended at this stage.

Python package for locally centered cyclic (LCC) kernel estimation of the continuum translog model.

> Tetsuya Saito. *Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy*. TechRxiv, 2026.
> URL: https://github.com/Kleinverse/research/lcc

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kleinverse/lccfit.git
```

> [!NOTE]
> Optionally, install in editable mode. PyPI release is planned for a future version.

```bash
cd lccfit
pip install -e .
```

Requires Python 3.10+, NumPy, PyTorch. CUDA-enabled PyTorch is recommended for large datasets.

---

## Overview

Standard second-order Maclaurin truncation of the log-MGF is asymptotically inconsistent: the bias from discarding higher-order cumulants does not vanish with sample size. The LCC kernel recovers the true cumulant at each order as a nondegenerate V-statistic, and `lccfit` provides a full pipeline from raw data to theta and sigma estimates.

---

## Modules

### `lccfit.converter`

Draws k-tuples via Monte Carlo from a single cell's empirical distribution (optionally weighted), evaluates the LCC kernel $h_k = \prod_{j=1}^k (z_j - \bar{z}_k)$ for each order, and returns an `LCCData` object ready for fitting. The Monte Carlo operates within each cell. The panel structure sits one level up: call `convert()` once per cell and pass the resulting `LCCData` objects together to `batch_fit()`.

```python
import lccfit

# Basic usage
data = lccfit.convert(z, orders=[2, 4, 6], n_mc=100_000)
data.summary()

# With log transform and weights
data = lccfit.convert(uv, log=True, weights=shares, orders=[2, 4, 6])

# For GMM fitting (shared draws required)
data = lccfit.convert(z, shared_draws=True, store_samples=True)
M = data.to_moments_matrix()   # [n_mc, len(orders)]

# Save to disk
data.save('output.csv')
data.save('output.parquet', fmt='parquet')
```

**`convert()` options**

| Parameter | Default | Description |
|---|---|---|
| `log` | `False` | Apply log transform before processing |
| `weights` | `None` | Non-negative sampling weights, normalised internally |
| `orders` | `[2..8]` | Kernel orders to compute |
| `n_mc` | `100_000` | Monte Carlo draws per order |
| `dtype` | `'float64'` | Numerical precision (`'float32'` for speed) |
| `device` | `'auto'` | `'cuda'`, `'cpu'`, or `torch.device` |
| `seed` | `None` | Random seed (CPU path only) |
| `store_samples` | `True` | Store raw draws for GMM fitting |
| `shared_draws` | `False` | Share index draws across orders (required for GMM) |
| `meta` | `None` | Free-form metadata dict attached to `LCCData` |

---

### `lccfit.fit`

Partition formula inversion and Newton solver for theta and elasticity.

```python
from lccfit.fit import fit_cell, batch_fit, reg_fit_cell

# Single cell (CPU)
result = fit_cell(z, w, Ehk=data.Ehk, orders=[2, 4, 6])
result.summary()

# Batch (GPU) — preferred for large datasets
results = batch_fit(cell_list, Ehk_list, keys, solve_orders=[2, 4, 6])

# Regression variant with Jensen-corrected kappas
result = reg_fit_cell(z, w, samples=data.samples, solve_orders=[2, 4, 6])
```

**`FitResult` fields**

| Field | Description |
|---|---|
| `kappa` | Inverted CGF cumulants $\kappa_k$ for each order |
| `theta` | Estimated power-mean parameter $\theta$ per solve order |
| `sigma` | Curvature scaling parameter $\sigma = 1/(1-\theta)$ per solve order |
| `tariff` | Optimal markup $(1-\theta)/\theta$ per solve order |
| `Ehk` | E[h_k] estimates from the converter |

---

## Full Pipeline Example

```python
import numpy as np
import lccfit
from lccfit.fit import fit_cell

z = np.random.lognormal(0, 0.5, 500)
w = np.ones(500) / 500

data   = lccfit.convert(z, orders=[2, 4, 6], n_mc=100_000)
result = fit_cell(z, w, Ehk=data.Ehk, orders=[2, 4, 6])
result.summary()
```

---

## Citation

```bibtex
@misc{saito2026powermean,
  author       = {Tetsuya Saito},
  title        = {Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy},
  year         = {2026},
  howpublished = {TechRxiv},
  url          = {https://github.com/Kleinverse/research/lcc}
}
```

---

## License

MIT License. Copyright (c) 2026 Tetsuya Saito, Kleinverse AI, Inc.
