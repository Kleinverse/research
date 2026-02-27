# Cyclic Kernel Remedy for Asymptotic Inconsistency of Power-Mean Estimations

Numerical experiments, estimation routines, and benchmarking code accompanying the paper:

> Tetsuya Saito. *Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy*. TechRxiv, 2026.
> URL: https://github.com/Kleinverse/research/lcc

A companion paper applying the LCC kernel to independent component analysis is:

> Tetsuya Saito. *Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis*. TechRxiv, February 25, 2026.
> DOI: 10.36227/techrxiv.177203264.46969730/v1

The `lccfit` package implementing the LCC estimator is distributed separately at [github.com/Kleinverse/lccfit](https://github.com/Kleinverse/lccfit). Clone the repository:

```bash
git clone https://github.com/Kleinverse/lccfit.git
```

---

## Structure

```
research/lcc/
├── README.md
└── src/
    ├── experiments.py      LCC and CtrdM Monte Carlo evaluation
    ├── estimation.py       Theta and elasticity estimation from 2024 US HS10
    ├── benchmark.py        Wall-clock comparison of three Maclaurin routes
    └── gpu_kernel.py       Optional fused CuPy CUDA/ROCm kernel
```

---

## Requirements

- Python 3.10+
- NumPy, SciPy, pandas
- PyTorch (CUDA-enabled build recommended)
- CuPy (optional, for fused GPU kernel in `gpu_kernel.py`)
  - NVIDIA CUDA 12: `pip install cupy-cuda12x`
  - AMD ROCm 7.x+: `pip install cupy-rocm-7-0`

---

## Experiments

Evaluates LCC and CtrdM across six distribution families over a grid of
sample sizes, expansion orders $k \in \{2,\ldots,8\}$, and preference
parameters.

```bash
# Default: theta = 0.5
python src/experiments.py

# Specify theta values
python src/experiments.py --theta 0.1 0.5 0.9

# Quick validation run (T=100, n in [10, 100, 1000])
python src/experiments.py --theta 0.5 --quick
```

Results are written to `csv_results/` as `section_a.csv`, `section_c.csv`,
and `section_d.csv`.

**$\hat{\theta}$ at $n = 10{,}000$ (all SE < 0.0006)**

|  | k | (i) Normal | (ii) Pareto | (iii) Lognormal | (iv) Mixture | (v) Skew-N | (vi) Uniform |
|---|---|---|---|---|---|---|---|
| **LCC** | 2 | .500 | .568 | .621 | .380 | .547 | .491 |
| | 4 | .500 | .503 | .507 | .528 | .500 | .500 |
| | 6 | .500 | .500 | .501 | .499 | .500 | .500 |
| **CtrdM** | 4 | .474 | .502 | .500 | .431 | .490 | .479 |
| | 6 | .473 | .499 | .491 | .441 | .489 | .479 |

True $\theta = 0.5$. LCC converges to the true value by $k = 6$ across all distributions. CtrdM retains a permanent downward bias that does not shrink with $k$.

---

## Data Conversion

Converts Schott's Stata file to CSV before passing to `estimation.py`.

```python
import pandas as pd
y = '2024'
df = pd.read_stata(f'imp_detl_yearly_{y}.dta')
df.to_csv(f'imports_{y}.csv', index=False)
```

---

## Estimation

Estimates the preference parameter $\theta$ and the elasticity of substitution
from the 2024 US HS10 import data using the LCC V-statistic kernel at
expansion orders $k \in \{2, 4, 6\}$ via Newton inversion of the Maclaurin
series.

```bash
python src/estimation.py -i imports_2024.csv -o theta_results.csv

# Options
--min-varieties 5      Minimum varieties per cell (default: 5)
--n-mc 100000          MC samples per cell per kernel order (default: 100000)
--batch-size 256       Cells per GPU batch (default: 256)
--float32              Use float32 instead of float64
```

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

## Benchmarking

Compares wall-clock time across LCC, CtrdM, and the numerical
central-difference baseline at $k = 6$ and $\theta = 0.5$.

```bash
# Synthesized distributions only
python src/benchmark.py

# Include real HS10 data
python src/benchmark.py -i imports_2024.csv --order 6 --h 1e-4
```

LCC and CtrdM share the same $O(kn)$ computation and diverge only in an $O(1)$ post-processing step. The Numerical route applies central-difference stencils at $2k+1$ grid points, incurring an additional $O(n)$ cost per order. Wall-clock times below are at $k = 6$, $\theta = 0.5$, $T = 10{,}000$ on an NVIDIA RTX 5080 GPU.

**Wall-clock time (µs per cell)**

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

## Citation

```bibtex
@misc{saito2026powermean,
  author       = {Tetsuya Saito},
  title        = {Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy},
  year         = {2026},
  howpublished = {TechRxiv},
  url          = {https://github.com/Kleinverse/research/lcc}
}

@misc{saito2026ica,
  author       = {Tetsuya Saito},
  title        = {Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis},
  year         = {2026},
  month        = {February},
  howpublished = {TechRxiv},
  doi          = {10.36227/techrxiv.177203264.46969730/v1}
}
```

---

## License

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt the material for any purpose, provided appropriate credit is given to the original paper and the author.
