# Bounded Locally Centered Contrasts for ICA

Experiment code for the paper:

> T. Saito, "Bounded Locally Centered Contrasts for Independent Component Analysis," *TechRxiv*, 2026.  
> https://doi.org/10.36227/techrxiv.XXXXXXX

The ICALCC package is at [Kleinverse/icalcc](https://github.com/Kleinverse/icalcc).

---

## Repository Structure

```
icalcc/
├── truncation.py     # Table III: bounded LCC vs classical contrasts
├── separation.py     # Table IV: Gamma(α) scan
└── README.md
```

---

## Requirements

```bash
pip install icalcc numpy scipy
```

---

## Usage

```bash
# Bounded LCC vs classical (Table III)
python truncation.py

# Gamma(α) scan (Table IV)
python separation.py --scan
```

---

## Results

### Bounded LCC vs Classical Contrasts (d=4, N=5000, 20 trials)

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

### Gamma(α) Scan (d=4, N=10000, 200 trials)

| α | κ₃ | κ₄ | logcosh | exp | LCC(6) | LCC(8) |
|---|---|---|---|---|---|---|
| 2.0 | 1.41 | 3.0 | 1.78 | 3.83 | **1.08** | 1.46 |
| 5.0 | 0.89 | 1.2 | 4.01 | *7.68 | **1.54** | 1.32 |
| 8.0 | 0.71 | 0.8 | 6.98 | *21.6 | **2.10** | 1.65 |
| 12 | 0.58 | 0.5 | *11.7 | *25.2 | 2.64 | **2.02** |
| 20 | 0.45 | 0.3 | *23.1 | *36.1 | 3.81 | **2.85** |
| 50 | 0.28 | 0.1 | *38.2 | *41.3 | *9.60 | **5.91** |

\* at least one convergence failure out of 20 trials.

---

## GPU Benchmark

RTX 5080, `gpu_mem_limit=12GB`. See [Kleinverse/gpuicalcc](https://github.com/Kleinverse/gpuicalcc) for details.

Polynomial contrasts (K=4,6,8) benefit from GPU at N≥500k, reaching up to 2.4x speedup. Bounded contrasts (ltanh, lexp) achieve 40–48x speedup across all sizes due to O(N²) pairwise computation.

<details>
<summary>Full benchmark table</summary>

| K | N | CPU (s) | GPU (s) | Speedup |
|---|---|---|---|---|
| 4 | 1k | 0.001 | 0.002 | 0.5x |
| 4 | 10k | 0.008 | 0.010 | 0.9x |
| 4 | 100k | 0.147 | 0.147 | 1.0x |
| 4 | 500k | 0.406 | 0.398 | 1.0x |
| 4 | 1M | 0.630 | 0.576 | 1.1x |
| 6 | 1k | 0.002 | 0.005 | 0.4x |
| 6 | 10k | 0.021 | 0.027 | 0.8x |
| 6 | 100k | 0.178 | 0.177 | 1.0x |
| 6 | 500k | 0.686 | 0.395 | 1.7x |
| 6 | 1M | 1.258 | 0.672 | 1.9x |
| 8 | 1k | 0.021 | 0.079 | 0.3x |
| 8 | 10k | 0.022 | 0.037 | 0.6x |
| 8 | 100k | 0.245 | 0.196 | 1.2x |
| 8 | 500k | 0.979 | 0.409 | 2.4x |
| 8 | 1M | 1.655 | 0.720 | 2.3x |
| ltanh | 1k | 0.101 | 0.006 | 16.7x |
| ltanh | 10k | 1.087 | 0.023 | 46.4x |
| ltanh | 100k | 12.328 | 0.318 | 38.7x |
| ltanh | 500k | 53.631 | 1.173 | 45.7x |
| ltanh | 1M | 104.210 | 2.168 | 48.1x |
| lexp | 1k | 0.108 | 0.006 | 18.9x |
| lexp | 10k | 1.210 | 0.029 | 41.6x |
| lexp | 100k | 15.443 | 0.376 | 41.1x |
| lexp | 500k | 67.916 | 1.477 | 46.0x |
| lexp | 1M | 130.470 | 2.807 | 46.5x |

</details>

---

## Citation

```bibtex
@misc{saito2026bounded,
  author    = {Saito, Tetsuya},
  title     = {Bounded Locally Centered Contrasts for Independent Component Analysis},
  year      = {2026},
  publisher = {TechRxiv},
  doi       = {10.36227/techrxiv.XXXXXXX}
}
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
