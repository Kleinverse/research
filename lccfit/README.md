# lccfit

> [!CAUTION]
> This package is in active development. APIs may change without notice and results should be verified against the reference implementation in the research repository. Use in production is not recommended at this stage.

Python package for locally centered cyclic (LCC) kernel estimation of the power-mean model.

> Tetsuya Saito. *Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy*. TechRxiv, 2026.
> URL: https://github.com/Kleinverse/research/lcc

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kleinverse/lccfit.git
```

Optionally, install in editable mode:

```bash
cd lccfit
pip install -e .
```

> [!NOTE]
> PyPI release is planned for a future version.

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
data = lccfit.convert(x, log=True, weights=w, orders=[2, 4, 6])

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

Partition formula inversion and Newton solver for theta and sigma.

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

## Numerical Experiments

Monte Carlo design: $n \in \{5, 10, \ldots, 10{,}000\}$, $k \in \{2,\ldots,8\}$, $\theta \in \{0.1, 0.5, 0.9\}$, $T = 10{,}000$ replications. Six distributions: (i) Normal(0,1), (ii) Pareto(7)+1, (iii) Lognormal(0, 0.5), (iv) mixture normal, (v) Skew-normal($\alpha$=5), (vi) Uniform(0,3).

The heatmap below shows $\Delta_u = \log_{10}(|\hat{u}^\text{LCC} - u| / |\hat{u}^\text{CtrdM} - u|)$ across the grid. Blue cells indicate LCC dominance ($\Delta_u < 0$); red cells indicate CtrdM dominance.

![Heatmap of utility residual ratios](img/heatmap.png)

LCC achieves statistically significant advantage at $k \geq 6$ across all distributions. The apparent reversal at lower orders for heavy-tailed distributions (ii), (iii) does not survive a one-sided $t$-test, confirming no statistically supported advantage for CtrdM in any cell.

**Table: $\hat{\theta}$ at $n = 10{,}000$ (selected rows; all SE < 0.0006)**

|  | k | (i) Normal | (ii) Pareto | (iii) Lognormal | (iv) Mixture | (v) Skew-N | (vi) Uniform |
|---|---|---|---|---|---|---|---|
| **LCC** | 2 | .500 | .568 | .621 | .380 | .547 | .491 |
| | 4 | .500 | .503 | .507 | .528 | .500 | .500 |
| | 6 | .500 | .500 | .501 | .499 | .500 | .500 |
| **CtrdM** | 4 | .474 | .502 | .500 | .431 | .490 | .479 |
| | 6 | .473 | .499 | .491 | .441 | .489 | .479 |

True $\theta = 0.5$. LCC converges to the true value by $k = 6$ across all distributions. CtrdM retains a permanent downward bias that does not shrink with $k$.

---

## Estimation Results (2024 US HS10 Import Data)

Applied to 9,694 HS10 categories. $\hat{\theta}_k$ recovered by Newton inversion; $\hat{\sigma}_k = 1/(1-\hat{\theta}_k)$; implied markup $\hat{\tau}^*_k = (1-\hat{\theta}_k)/\hat{\theta}_k$.

At $k = 2$ the median implied markup diverges to 1,498% — an artifact of near-unit $\hat{\sigma}_2 \approx 1.07$ caused by second-order truncation absorbing higher-order cumulant mass. At $k = 4$ and $k = 6$ the estimates agree closely (correlation 0.989) with median markup near 184% and 183% respectively.

**Power-mean parameter $\hat{\theta}$**

| Percentile | k = 2 | k = 4 | k = 6 |
|---|---:|---:|---:|
| Valid cells (n) | 3,365 | 4,607 | 4,652 |
| % of cells | 34.7% | 47.5% | 48.0% |
| 25% | 0.029 | 0.227 | 0.227 |
| **50%** | **0.063** | **0.352** | **0.354** |
| 75% | 0.174 | 0.546 | 0.551 |
| 90% | 0.495 | 0.771 | 0.766 |

**Curvature scaling $\hat{\sigma} = 1/(1-\hat{\theta})$**

| Percentile | k = 2 | k = 4 | k = 6 |
|---|---:|---:|---:|
| 25% | 1.029 | 1.293 | 1.294 |
| **50%** | **1.067** | **1.544** | **1.547** |
| 75% | 1.210 | 2.202 | 2.226 |
| 90% | 1.981 | 4.366 | 4.268 |

**Implied markup $\hat{\tau}^* = (1-\hat{\theta})/\hat{\theta}$**

| Percentile | k = 2 | k = 4 | k = 6 |
|---|---:|---:|---:|
| 25% | 4.763 | 0.832 | 0.815 |
| **50%** | **14.981** | **1.838** | **1.829** |
| 75% | 34.078 | 3.409 | 3.400 |
| 90% | 64.549 | 5.580 | 5.572 |

---

## Benchmark

LCC and CtrdM share the same $O(kn)$ computation and diverge only in an $O(1)$ post-processing step. The Numerical route applies central-difference stencils at $2k+1$ grid points, incurring an additional $O(n)$ cost per order. Wall-clock times below are at $k = 6$, $\theta = 0.5$, $T = 10{,}000$ on an NVIDIA RTX 5080 GPU.

**Table: Wall-clock time (µs per cell)**

| $n$ | Numerical | LCC | CtrdM | LCC/Numerical |
|---|---|---|---|---|
| 10 | 710 | 241 | 210 | 0.34× |
| 50 | 968 | 509 | 478 | 0.53× |
| 100 | 1,385 | 889 | 859 | 0.64× |
| 500 | 5,265 | 4,405 | 4,372 | 0.84× |
| 1,000 | 12,338 | 9,411 | 9,379 | 0.76× |
| Real HS10 ($n \approx 11$, 9,694 cells) | 0.164 | 0.083 | 0.080 | 0.51× |

LCC incurs no meaningful penalty relative to CtrdM and runs at roughly half the cost of the Numerical route across all sample sizes.

---



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
