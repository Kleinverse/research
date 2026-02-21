# Kleinverse Open Research Repository (KORR)
Experiment code and reproduction materials for research papers by
[Kleinverse AI, Inc.](https://kleinverse.io)

---

## Papers

### LCC-ICA: Locally Centered Cyclic Kernels for Higher-Order ICA
> T. Saito, "Locally Centered Cyclic Kernels for Higher-Order
> Independent Component Analysis," *TechRxiv*, 2026.  
> https://doi.org/10.36227/techrxiv.XXXXXXX

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
> https://doi.org/10.36227/techrxiv.XXXXXXX

Classical FastICA contrasts evaluate each sample in isolation and
cannot adapt to distributional shape. Bounded LCC contrasts replace
the pointwise nonlinearity with a pairwise statistic, inheriting
outlier robustness while gaining sensitivity to skewness through
local centering. LCC-exp admits an exact interpretation as
maximizing Rényi entropy of order 2. A GPU-accelerated extension
using PyTorch achieves 40–48× speedup for bounded contrasts at
N = 10⁶.

**Code:** [icalcc/](icalcc/) — requires `icalcc`, `numpy`, `scipy`;
GPU scripts additionally require `gpuicalcc`, `torch`

---

## Organization
```
research/
├── lcc/        # Locally Centered Cyclic Kernels
├── icalcc/     # Bounded Locally Centered Contrasts + GPU benchmark
└── README.md
```

Each subdirectory contains the experiment code for one paper.
The core libraries are maintained separately:
- [Kleinverse/icalcc](https://github.com/Kleinverse/icalcc) — CPU package
- [Kleinverse/gpuicalcc](https://github.com/Kleinverse/gpuicalcc) — GPU-accelerated package

---

## License
All works in this repository are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
You are free to share and adapt the material for any purpose,
provided appropriate credit is given to the original paper and
author(s).
