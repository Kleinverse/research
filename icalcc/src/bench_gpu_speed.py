#!/usr/bin/env python3
"""GPU speedup benchmark: ICALCC vs GPUICALCC wall-clock time.

Usage:
    python bench_gpu_speed.py              # default 8GB limit
    python bench_gpu_speed.py --mem 12     # 12GB limit
"""

import argparse
import gc
import time
import numpy as np
import torch
from icalcc import ICALCC
from gpuicalcc import GPUICALCC

parser = argparse.ArgumentParser()
parser.add_argument("--mem", type=float, default=8,
                    help="GPU memory limit in GB (default: 8)")
args = parser.parse_args()

Ns = [1_000, 10_000, 100_000, 500_000, 1_000_000]
Ks = [4, 6, 8, "ltanh", "lexp"]
d = 4


def clear():
    gc.collect()
    torch.cuda.empty_cache()


print(f"gpu_mem_limit={args.mem}GB")
print(f"{'K':<8}{'N':>10}  {'CPU (s)':>8}  {'GPU (s)':>8}  {'Speedup':>8}")
print("-" * 52)

N_RUNS = 100

for K in Ks:
    for N in Ns:
        rng = np.random.RandomState(0)
        S = rng.laplace(size=(N, d))
        A = rng.randn(d, d)
        X = S @ A.T

        # warmup GPU
        clear()
        gpu = GPUICALCC(n_components=d, K=K, device="cuda",
                        gpu_mem_limit=args.mem,
                        random_state=0, clear_gpu=False)
        gpu.fit(X)
        clear()

        t_cpus, t_gpus = [], []
        for r in range(N_RUNS):
            cpu = ICALCC(n_components=d, K=K, random_state=r)
            t0 = time.perf_counter()
            cpu.fit(X)
            t_cpus.append(time.perf_counter() - t0)
            del cpu

            clear()
            gpu = GPUICALCC(n_components=d, K=K, device="cuda",
                            gpu_mem_limit=args.mem,
                            random_state=r, clear_gpu=False)
            t0 = time.perf_counter()
            gpu.fit(X)
            t_gpus.append(time.perf_counter() - t0)
            del gpu
            clear()

        tc = np.mean(t_cpus)
        tg = np.mean(t_gpus)
        ratio = tc / tg if tg > 0 else float("inf")
        print(f"{str(K):<8}{N:>10}  {tc:>8.3f}  {tg:>8.3f}  {ratio:>7.1f}x")
    print()
