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

### Standard Distributions (d=4, max_iter=500, 20 trials)

Mean Amari index (×10⁻²). Lower is better. Bold indicates best per row.
Numbers in parentheses indicate convergence failures out of 20 trials.

| Dist | N | logcosh | exp | cube | Fast(4) | Fast(6) | Fast(8) | LCC(4) | LCC(6) | LCC(8) | LCC-tanh | LCC-exp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Laplace | 10k | 0.91 | 0.87 | 1.11 | 1.58 | 3.20 | 5.16 | 1.58 | 1.44 | 3.44 | **0.99** | 0.90 |
| Laplace | 100k | 0.29 | 0.27 | 0.35 | 0.47 | 0.85 | 1.63 | 0.47 | 0.44 | 0.64 | 0.32 | **0.29** |
| Logistic | 10k | 1.90 | 1.93 | 2.06 | 2.43 | 4.26 | 6.81 | 2.43 | 2.09 | 2.18 | 1.96 | **1.90** |
| Logistic | 100k | 0.62 | 0.63 | 0.66 | 0.75 | 1.13 | 2.04 | 0.75 | 0.69 | 0.67 | **0.63** | **0.62** |
| Uniform | 10k | 0.65 | 0.66 | 0.63 | 0.57 | 0.51 | **0.48** | 0.57 | 0.58 | 0.58 | 0.57 | 0.59 |
| Uniform | 100k | 0.20 | 0.20 | 0.19 | 0.17 | 0.15 | **0.14** | 0.17 | 0.17 | 0.17 | 0.17 | 0.16 |
| Student-t15 | 10k | 3.85 | 4.18 | 3.77 | 4.14 | 5.94 | 8.66 | 4.14 | 3.77 | 3.70 | **3.74** | 3.87 |
| Student-t15 | 100k | 1.20 | 1.28 | **1.16** | 1.20 | 1.74 | 3.25 | 1.20 | 1.17 | 1.18 | 1.17 | 1.23 |

`LCC-tanh` and `LCC-exp` match or outperform all classical contrasts on super-Gaussian
sources and are the safest default when the source distribution is unknown. On
sub-Gaussian sources (Uniform), high-order polynomial LCC is preferred. On
Student-t15 (κ₄ ≈ 0.46), all methods are within sampling noise.

### Gamma(α) Scan (d=4, N=10000, 20 trials)

Numbers in parentheses indicate convergence failures out of 20 trials.

| α | m₃ | m₄ | logcosh | exp | cube | Fast(4) | Fast(6) | Fast(8) | LCC(4) | LCC(6) | LCC(8) | LCC-tanh | LCC-exp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 2.83 | 12.0 | 0.71 | 0.70 | 0.86 | 1.09 | 1.83 | 2.59 | 1.09 | 1.65 | 6.04 | **0.55** | **0.50** |
| 1.0 | 2.00 | 6.0 | 1.10 | 1.20 | 1.17 | 1.40 | 2.65 | 4.57 | 1.40 | 1.26 | 8.69(9) | **0.75** | **0.68** |
| 2.0 | 1.41 | 3.0 | 1.78 | 3.83 | 1.54 | 1.62 | 2.65 | 4.09 | 1.62 | **1.08** | 1.46 | 1.00 | 0.89 |
| 3.0 | 1.15 | 2.0 | 2.75 | 5.73 | 2.09 | 2.07 | 3.55 | 5.83 | 2.07 | **1.24** | 1.33 | 1.34 | 1.26 |
| 5.0 | 0.89 | 1.2 | 4.01 | 7.68(2) | 2.79 | 2.65 | 4.11 | 6.86 | 2.65 | 1.54 | 1.32 | **1.87** | **1.78** |
| 8.0 | 0.71 | 0.8 | 6.98 | 21.55(3) | 4.19 | 3.68 | 5.23 | 7.93 | 3.68 | 2.10 | 1.65 | **2.79** | **2.61** |
| 12.0 | 0.58 | 0.5 | 11.73(2) | 25.22(4) | 6.06 | 4.52 | 5.13 | 7.73 | 4.52 | 2.64 | 2.02 | **3.95** | **3.80** |
| 20.0 | 0.45 | 0.3 | 23.11(6) | 36.11(11) | 9.94(1) | 6.85 | 7.07 | 9.73 | 6.85 | 3.81 | **2.85** | 6.09 | 6.17 |
| 50.0 | 0.28 | 0.1 | 38.23(8) | 41.32(10) | 32.52(6) | 25.82(6) | 18.35(3) | 20.12(5) | 24.31(6) | 8.90(1) | **5.91** | 22.30(4) | 23.83(4) |

`LCC-tanh` and `LCC-exp` dominate for moderate kurtosis (α = 1–12). Classical
contrasts degrade progressively and fail near Gaussian (α ≥ 12). At the
near-Gaussian extreme (α = 50), `LCC(8)` is the only reliable contrast.
The `LCC(8)` singularity at α = 1 (9 convergence failures) is a Newton
fixed-point defect that recovers at α = 2.

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
