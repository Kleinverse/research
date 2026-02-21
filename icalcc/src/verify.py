#!/usr/bin/env python3
"""Verify gpuicalcc produces identical results to icalcc."""

import numpy as np
from icalcc import ICALCC
from gpuicalcc import GPUICALCC

rng = np.random.RandomState(42)
S = rng.laplace(size=(2000, 3))
A = rng.randn(3, 3)
X = S @ A.T

for K in [4, 6, 8, "ltanh", "lexp"]:
    cpu = ICALCC(n_components=3, K=K, random_state=0)
    gpu = GPUICALCC(n_components=3, K=K, device="cpu", random_state=0)
    S_cpu = cpu.fit_transform(X)
    S_gpu = gpu.fit_transform(X)
    diff = np.max(np.abs(S_cpu - S_gpu))
    tag = "PASS" if diff < 1e-10 else "FAIL"
    print(f"K={str(K):<6}  max|diff|={diff:.2e}  {tag}")
