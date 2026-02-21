# LCC-ICA: Locally Centered Cyclic Kernels for Higher-Order ICA

Experiment code for the paper:

> **Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis**  
> Tetsuya Saito, *TexRxiv*, 2026  
> Kleinverse AI, Inc. / Globiz Professional University

---

## Overview

FastICA targets the excess kurtosis $\kappa_4 = \mathbb{E}[y^4] - 3$, truncating the
cumulant generating function (CGF) at order $k=4$. Naive substitution of higher-order
centered moments introduces a permanent bias bounded below by the Itakura--Saito
divergence. The locally centered cyclic (LCC) kernel eliminates this bias through cyclic
centering, yielding a nondegenerate V-statistic $V_k$ at every even order.

This repository contains the experiment code reproducing all results in the paper.
The core LCC library is maintained separately at
[Kleinverse/icalcc](https://github.com/Kleinverse/icalcc).

---

## Experiments

| Script | Description |
|---|---|
| `experiments/synthetic.py` | Synthetic ICA: Laplace, Logistic, Uniform, Student-$t_{15}$, Gamma($\alpha$) |
| `experiments/image.py` | Grayscale image unmixing using scikit-image textures |
| `experiments/audio.py` | MUSDB18 music stem separation |

---

## Requirements
```bash
pip install icalcc torch torchvision scikit-image musdb numpy scipy
```

GPU acceleration via PyTorch with CUDA is supported and recommended for audio experiments.

---

## Usage
```bash
# Synthetic experiments (Tables II, III, IV)
python experiments/synthetic.py

# Image separation (Table V, Fig. 2)
python experiments/image.py

# Audio separation (Table V, Fig. 3)
# Requires MUSDB18 dataset
python experiments/audio.py --musdb_path /path/to/musdb18
```

---

## Results

| Method | Laplace (100k) | Logistic (100k) | MUSDB18 Amari |
|---|---|---|---|
| FastICA(4) | 0.72 | 1.01 | 4.01 |
| LCC(6) | **0.59** | **0.91** | **1.60** |

LCC outperforms FastICA on super-Gaussian sources of moderate kurtosis. Extreme kurtosis
marks the boundary where estimator variance overtakes the bias correction.

---

## Citation
```bibtex
@article{saito2026lcc,
  author  = {Saito, Tetsuya},
  title   = {Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis},
  journal = {TexRxiv},
  year    = {2026}
}
```

---

## License

MIT License. Copyright (c) 2026 Kleinverse AI, Inc.
