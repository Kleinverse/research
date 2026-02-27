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
