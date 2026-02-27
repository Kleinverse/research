# Cyclic Kernel Remedy for Asymptotic Inconsistency of Power-Mean Estimations

Numerical experiments, estimation routines, and benchmarking code accompanying the paper:

> Tetsuya Saito. *Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy*. TechRxiv, 2026.
> URL: https://github.com/Kleinverse/research/lcc

A companion paper applying the LCC kernel to independent component analysis is:

> Tetsuya Saito. *Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis*. TechRxiv, February 25, 2026.
> DOI: 10.36227/techrxiv.177203264.46969730/v1

The `lccfit` package implementing the LCC estimator is distributed separately at [github.com/Kleinverse/lccfit](https://github.com/Kleinverse/lccfit).

---

## Structure

```
research/lcc/
├── README.md
└── src/
    ├── experiments.py      LCC and CtrdM Monte Carlo evaluation
    ├── estimation.py       Theta and elasticity estimation from 2024 US HS10
    ├── benchmark.py        Wall-clock comparison of three Maclaurin routes
    ├── converter.py        HS10 preprocessing via lccfit
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
and `section_d.csv`, reproducing Figures 3a--3b and Table 3 of the paper.

---

## Data Conversion

Converts the Schott HS10 import data into `LCCData` objects carrying
E[h_k] kernel estimates and optional raw sample draws for each cell.
The converter applies log transformation and expenditure-share weighting
automatically.

```bash
python src/converter.py -i imports_2024.csv -o lcc_samples.csv

# Options
--log                  Apply log transform to unit values (default: True)
--orders 2 4 6         Kernel orders to compute (default: 2 3 4 5 6 7 8)
--n-mc 100000          MC draws per order per cell (default: 100000)
--dtype float64        Numerical precision: float64 or float32
--shared-draws         Share index draws across orders (required for GMM)
--store-samples        Store raw kernel draws for statsmodels GMM fitting
--seed 42              Random seed (CPU path only)
```

The output is a flat CSV with one row per cell containing commodity code,
year, n, and `Ehk_k` columns for each requested order, ready for
`estimation.py`.

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

Reproduces Table 4, Table 5, and Figure 4 of the paper.

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

Reproduces Table 7 of the paper.

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
