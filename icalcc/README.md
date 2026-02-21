# Bounded Locally Centered Contrasts for ICA

Experiment code for the paper:

> T. Saito, "ICALCC: Locally Centered Contrast Functions for FastICA with GPU Acceleration," *TechRxiv*, 2026.  
> https://www.techrxiv.org/users/869274/articles/1390891

The ICALCC package is at [Kleinverse/icalcc](https://github.com/Kleinverse/icalcc).

---

## Repository Structure

```
icalcc/
├── src/
│   ├── exp_bounded.py        # Tables III and IV: separation experiments
│   ├── bench_gpu_speed.py    # Table II: GPU speedup benchmark
│   └── verify.py             # Numerical identity check (CPU vs GPU)
└── README.md
```

---

## Requirements

```bash
pip install icalcc numpy scipy
```

GPU scripts additionally require:

```bash
pip install gpuicalcc torch
```

GPU acceleration is used automatically in `exp_bounded.py` if `gpuicalcc` is installed.

---

## Usage

```bash
# Both tables (20 trials)
python src/exp_bounded.py

# Table III only: bounded LCC vs classical
python src/exp_bounded.py --table 1

# Table IV only: Gamma(α) scan (full paper run)
python src/exp_bounded.py --table 2 --trials 200

# Table II: GPU speedup benchmark
python src/bench_gpu_speed.py --mem 12

# Verify CPU and GPU produce identical results
python src/verify.py
```

> **Note:** `bench_gpu_speed.py` runs 100 trials per (K, N) combination. At N=1M, ltanh and lexp each require approximately 100s CPU time per trial. The full benchmark takes several hours.

---

## Results

### Bounded LCC vs Classical Contrasts (d=4, N=5000, 20 trials)

Mean Amari index (×10⁻²). Lower is better. Bold indicates best per row.

| Distribution | κ₄ | logcosh | exp | LCC-tanh | LCC-exp |
|---|---|---|---|---|---|
| Uniform | −1.2 | 0.93 | 0.94 | **0.68** | 0.88 |
| Beta(2,5) | −0.1 | **6.96** | 5.06 | 30.85 | 21.64 |
| Gamma(8) | 0.75 | 3.78 | 22.43 | **2.05** | 2.58 |
| Gamma(2) | 3.0 | 2.94 | 2.38 | 1.29 | **1.19** |
| Laplace | 3.0 | **1.36** | 1.24 | 1.45 | 1.28 |
| Student-t5 | 6.0 | 2.72 | 2.89 | **2.65** | 2.92 |
| Exponential | 6.0 | 1.41 | 1.08 | 0.92 | **0.60** |
| Mixed | mixed | **1.15** | 1.10 | 1.37 | 1.38 |

Bounded contrasts lead on skewed, moderately heavy-tailed sources (Gamma, Exponential).
Classical contrasts remain competitive at extreme kurtosis (Beta, Student-t5) and
on sub-Gaussian sources (Uniform), consistent with the ARE analysis in the paper.

### Gamma(α) Scan (d=4, N=10000, 200 trials)

\* at least one convergence failure out of 20 trials.

| α | κ₃ | κ₄ | logcosh | exp | LCC(6) | LCC(8) |
|---|---|---|---|---|---|---|
| 2.0 | 1.41 | 3.0 | 1.78 | 3.83 | **1.08** | 1.46 |
| 5.0 | 0.89 | 1.2 | 4.01 | \*7.68 | **1.54** | 1.32 |
| 8.0 | 0.71 | 0.8 | 6.98 | \*21.6 | **2.10** | 1.65 |
| 12 | 0.58 | 0.5 | \*11.7 | \*25.2 | 2.64 | **2.02** |
| 20 | 0.45 | 0.3 | \*23.1 | \*36.1 | 3.81 | **2.85** |
| 50 | 0.28 | 0.1 | \*38.2 | \*41.3 | \*9.60 | **5.91** |

Classical contrasts degrade progressively as kurtosis decreases and fail entirely
near Gaussian (α ≥ 12). LCC(6) dominates at moderate kurtosis; LCC(8) takes over
at near-Gaussian (α ≥ 12). The winning order shifts with α exactly as predicted
by the asymptotic analysis in the paper.

---

## GPU Numerical Verification

Differences are at floating-point machine precision (≤ 7.44e-14), confirming numerical identity between CPU and GPU implementations.

```
K=4       max|diff|=0.00e+00  PASS
K=6       max|diff|=9.85e-15  PASS
K=8       max|diff|=7.44e-14  PASS
K=ltanh   max|diff|=1.50e-14  PASS
K=lexp    max|diff|=7.11e-15  PASS
```

---

## GPU Benchmark

RTX 5080, `gpu_mem_limit=12GB`. See [Kleinverse/gpuicalcc](https://github.com/Kleinverse/gpuicalcc) for details.

Polynomial contrasts (K=4,6,8) benefit from GPU at N≥500k, reaching up to 2.4× speedup. Bounded contrasts (ltanh, lexp) achieve 40–48× speedup across all sizes due to O(NB) pairwise computation.

<details>
<summary>Full benchmark table</summary>

| K | N | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|---|
| 4 | 1k | 0.001 | 0.002 | 0.5× |
| 4 | 10k | 0.008 | 0.010 | 0.9× |
| 4 | 100k | 0.147 | 0.147 | 1.0× |
| 4 | 500k | 0.406 | 0.398 | 1.0× |
| 4 | 1M | 0.630 | 0.576 | 1.1× |
| 6 | 1k | 0.002 | 0.005 | 0.4× |
| 6 | 10k | 0.021 | 0.027 | 0.8× |
| 6 | 100k | 0.178 | 0.177 | 1.0× |
| 6 | 500k | 0.686 | 0.395 | 1.7× |
| 6 | 1M | 1.258 | 0.672 | 1.9× |
| 8 | 1k | 0.021 | 0.079 | 0.3× |
| 8 | 10k | 0.022 | 0.037 | 0.6× |
| 8 | 100k | 0.245 | 0.196 | 1.2× |
| 8 | 500k | 0.979 | 0.409 | 2.4× |
| 8 | 1M | 1.655 | 0.720 | 2.3× |
| ltanh | 1k | 0.101 | 0.006 | 16.7× |
| ltanh | 10k | 1.087 | 0.023 | 46.4× |
| ltanh | 100k | 12.328 | 0.318 | 38.7× |
| ltanh | 500k | 53.631 | 1.173 | 45.7× |
| ltanh | 1M | 104.210 | 2.168 | 48.1× |
| lexp | 1k | 0.108 | 0.006 | 18.9× |
| lexp | 10k | 1.210 | 0.029 | 41.6× |
| lexp | 100k | 15.443 | 0.376 | 41.1× |
| lexp | 500k | 67.916 | 1.477 | 46.0× |
| lexp | 1M | 130.470 | 2.807 | 46.5× |

</details>

---

## Citation

```bibtex
@misc{saito2026icalcc,
  author    = {Saito, Tetsuya},
  title     = {{ICALCC}: Locally Centered Contrast Functions for
               {FastICA} with {GPU} Acceleration},
  year      = {2026},
  publisher = {TechRxiv},
  url       = {https://www.techrxiv.org/users/869274/articles/1390891}
}

@misc{saito2026lcc,
  author    = {Saito, Tetsuya},
  title     = {Locally Centered Cyclic Kernels for Higher-Order
               Independent Component Analysis},
  year      = {2026},
  publisher = {TechRxiv},
  doi       = {10.36227/techrxiv.XXXXXXX}
}
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
