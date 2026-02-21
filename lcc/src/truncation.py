#!/usr/bin/env python3
"""
Experiments TechRxiv February 14, 2026:
  1. Per-order truncation error (CGF vs CtrdM, Fig. 1)
  2. LCC nondegeneracy at k=2,4,6,8 (Table I)

Distributions: Uniform[-sqrt(3),sqrt(3)], Laplace(0,1/sqrt(2)),
               shifted Exponential(1), all with unit variance.
"""

import numpy as np
import math

np.random.seed(42)

# ======================================================================
# Distributions (unit variance)
# ======================================================================

class Laplace:
    name = "Laplace"
    kappa4 = 3.0

    @staticmethod
    def cgf(theta):
        return -np.log(1.0 - 0.5 * theta**2)

    @staticmethod
    def mgf(theta):
        return 1.0 / (1.0 - 0.5 * theta**2)

    @staticmethod
    def central_moments():
        return {2: 1.0, 3: 0.0, 4: 6.0, 5: 0.0,
                6: 90.0, 7: 0.0, 8: 2520.0}

    @staticmethod
    def cumulants():
        return {2: 1.0, 3: 0.0, 4: 3.0, 5: 0.0,
                6: 30.0, 7: 0.0, 8: 630.0}

    @staticmethod
    def sample(N):
        return np.random.laplace(0, 1.0 / np.sqrt(2), N)


class Uniform:
    name = "Uniform"
    kappa4 = -1.2

    @staticmethod
    def cgf(theta):
        a = np.sqrt(3)
        return np.log((np.exp(a*theta) - np.exp(-a*theta)) / (2*a*theta))

    @staticmethod
    def mgf(theta):
        a = np.sqrt(3)
        return (np.exp(a*theta) - np.exp(-a*theta)) / (2*a*theta)

    @staticmethod
    def central_moments():
        return {2: 1.0, 3: 0.0, 4: 9.0/5.0, 5: 0.0,
                6: 27.0/7.0, 7: 0.0, 8: 9.0}

    @staticmethod
    def cumulants():
        m = Uniform.central_moments()
        return {
            2: 1.0, 3: 0.0,
            4: m[4] - 3,
            5: 0.0,
            6: m[6] - 15*m[4] + 30,
            7: 0.0,
            8: m[8] - 28*m[6] - 35*m[4]**2 + 420*m[4] - 630,
        }

    @staticmethod
    def sample(N):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), N)


class Exponential:
    name = "Exponential"
    kappa4 = 6.0

    @staticmethod
    def cgf(theta):
        return -np.log(1.0 - theta) - theta

    @staticmethod
    def mgf(theta):
        return np.exp(-np.log(1.0 - theta) - theta)

    @staticmethod
    def central_moments():
        return {2: 1.0, 3: 2.0, 4: 9.0, 5: 44.0,
                6: 265.0, 7: 1854.0, 8: 14833.0}

    @staticmethod
    def cumulants():
        return {k: math.factorial(k - 1) for k in range(2, 9)}

    @staticmethod
    def sample(N):
        return np.random.exponential(1, N) - 1.0


DISTS = [Uniform, Laplace, Exponential]


# ======================================================================
# LCC V-statistics via sample moments (O(N), no tuple enumeration)
# ======================================================================

def lcc_from_moments(x):
    """Compute V_2, V_4, V_6, V_8 from sample moments.
    x is a 1-D array of N observations."""
    x2 = x * x
    m2 = np.mean(x2)
    m4 = np.mean(x2 * x2)
    m6 = np.mean(x2 * x2 * x2)
    m8 = np.mean(x2 * x2 * x2 * x2)

    V2 = -m2 / 2.0
    V4 = 21.0 * m2**2 / 64.0 - 3.0 * m4 / 64.0
    V6 = (-55.0 * m2**3 / 432.0 + 15.0 * m2 * m4 / 432.0
           - m6 / 432.0)
    V8 = (7735.0 * m2**4 / 131072.0
           - 1470.0 * m2**2 * m4 / 131072.0
           + 42.0 * m4**2 / 131072.0
           + 28.0 * m2 * m6 / 131072.0
           - m8 / 131072.0)
    return {2: V2, 4: V4, 6: V6, 8: V8}


# ======================================================================
# Experiment 1: Per-order truncation error
# ======================================================================

def exp1_truncation():
    print("=" * 70)
    print("EXP 1: Per-order truncation error at theta = 0.5")
    print("=" * 70)
    theta = 0.5

    for Dist in DISTS:
        true_u = Dist.cgf(theta) / theta
        is_limit = (Dist.mgf(theta) - 1.0) / theta
        m = Dist.central_moments()
        kappa = Dist.cumulants()

        print(f"\n  {Dist.name}  (K/t = {true_u:.6f}, "
              f"(M-1)/t = {is_limit:.6f}, D_IS/t = {is_limit-true_u:.6f})")
        print(f"    {'k':>3s}  {'CGF err':>10s}  {'CtrdM err':>10s}")

        cgf_sum, ctrdm_sum = 0.0, 0.0
        for k in range(2, 9):
            cgf_sum += kappa[k] * theta**(k-1) / math.factorial(k)
            ctrdm_sum += m[k] * theta**(k-1) / math.factorial(k)
            print(f"    {k:3d}  {abs(cgf_sum-true_u):10.6f}  "
                  f"{abs(ctrdm_sum-true_u):10.6f}")


# ======================================================================
# Experiment 2: LCC nondegeneracy
# ======================================================================

def exp2_nondegeneracy(N=10000, n_trials=200):
    print("\n" + "=" * 70)
    print(f"EXP 2: LCC nondegeneracy (N={N}, {n_trials} trials)")
    print("=" * 70)

    orders = [2, 4, 6, 8]
    print(f"\n  {'':14s}", end="")
    for k in orders:
        print(f"  {'k='+str(k):>14s}", end="")
    print()

    for Dist in DISTS:
        print(f"  {Dist.name:<14s}", end="")
        for k in orders:
            vals = np.array([lcc_from_moments(Dist.sample(N))[k]
                             for _ in range(n_trials)])
            mu = vals.mean()
            se = vals.std() / np.sqrt(n_trials)
            t = abs(mu / se) if se > 0 else np.inf
            print(f"  {mu:+.3f} ({t:3.0f})", end="")
        print()


# ======================================================================

if __name__ == "__main__":
    exp1_truncation()
    exp2_nondegeneracy(N=10000, n_trials=200)
