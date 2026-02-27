# Kleinverse Open Research Repository (KORR)
Experiment code and reproduction materials for research papers by Kleinverse AI, Inc.

---

## Papers

### Power-Mean: Asymptotic Inconsistency of Truncated Power-Mean and Its Cyclic Kernel Remedy
> T. Saito, "Asymptotic Inconsistency of Truncated Power-Mean and
> Its Cyclic Kernel Remedy," *TechRxiv*, 2026.

Second-order Maclaurin truncation of the log-MGF is asymptotically
inconsistent: the bias from discarding higher-order cumulants does not
vanish with sample size. The locally centered cyclic (LCC) kernel resolves
this by recovering the true cumulant at each order as a nondegenerate
V-statistic. Applied to 2024 US HS10 import data, second-order truncation
collapses the trade elasticity to near zero, while higher-order LCC
estimation recovers economically meaningful elasticities.

**Code:** [lcc/](lcc/) — requires `numpy`, `scipy`, `pandas`, `torch`; optionally `lccfit`

---

### LCC-ICA: Locally Centered Cyclic Kernels for Higher-Order ICA
> T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
> Independent Component Analysis," *TechRxiv*, 2026.

FastICA truncates the cumulant generating function at order k=4,
introducing a permanent bias bounded below by the Itakura--Saito
divergence. The locally centered cyclic (LCC) kernel eliminates
this bias through cyclic centering, yielding a nondegenerate
V-statistic at every even order. ARE analysis and experiments on
synthetic sources, grayscale image unmixing, and MUSDB18 music
stem separation confirm the predicted gains and their limits.

**Code:** [lcc/](lcc/) — requires `numpy`, `scipy`

---

### ICALCC: Locally Centered Contrast Functions for FastICA with GPU Acceleration
> T. Saito, "ICALCC: Locally Centered Contrast Functions for
> FastICA with GPU Acceleration," *TechRxiv*, 2026.

Classical FastICA contrasts evaluate each sample in isolation and
cannot adapt to distributional shape. Bounded LCC contrasts replace
the pointwise nonlinearity with a pairwise statistic, inheriting
outlier robustness while gaining sensitivity to skewness through
local centering. LCC-exp admits an exact interpretation as
maximizing Rényi entropy of order 2. A GPU-accelerated extension
using PyTorch achieves substantial speedup for bounded contrasts at
large sample sizes.

**Code:** [icalcc/](icalcc/) — requires `icalcc`, `numpy`, `scipy`;
GPU scripts additionally require `gpuicalcc`, `torch`

---

## Repository

```
research/
├── lcc/        Power-mean truncation and cyclic kernel remedy
├── icalcc/     Bounded locally centered contrasts and GPU benchmark
└── README.md
```

Each subdirectory contains the experiment code for one paper.
The core libraries are maintained separately:

- [Kleinverse/lccfit](https://github.com/Kleinverse/lccfit) — LCC estimator package
- [Kleinverse/icalcc](https://github.com/Kleinverse/icalcc) — CPU package
- [Kleinverse/gpuicalcc](https://github.com/Kleinverse/gpuicalcc) — GPU-accelerated package

---

## License

All works in this repository are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt the material for any purpose,
provided appropriate credit is given to the original paper and
author(s).
