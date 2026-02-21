"""
Asymptotic Relative Efficiency: LCC vs FastICA
================================================
ASV formula from Tichavsky, Koldovsky & Oja (2006, IEEE TSP, Prop. 1):

    ASV(g) = [E[g(s)^2] - (E[s g(s)])^2] / [E[g'(s)] - E[s g(s)]]^2

For FastICA-k:  g(y) = y^{k-1},  g'(y) = (k-1) y^{k-2}
For LCC-k:      g(y) = h_k(y),   g'(y) -> beta_k = E[h_k'(y)]

ARE(k) = ASV_FastICA(k) / ASV_LCC(k)

Prediction: squared Amari ratio (Fast/LCC)^2 should converge to ARE.

Usage:  python exp_are.py
"""
import numpy as np


# ======================================================================
# LCC h(y) and beta, copied from exp3_ica.py
# ======================================================================

def lcc_h_beta(y, k):
    """LCC nonlinearity h(y) and beta = E[h'(y)] on whitened data."""
    y2 = y * y
    y3 = y2 * y

    if k == 4:
        return (-3.0 / 16) * y3, -9.0 / 16

    m3 = np.mean(y3)
    m4 = np.mean(y2 * y2)

    if k == 6:
        dJ3 = 145 * m3 / 1944.0
        dJ4 = 115.0 / 2592
        dJ6 = -5.0 / 7776
        y4 = y2 * y2
        h_y = dJ3 * 3 * y2 + dJ4 * 4 * y3 + dJ6 * 6 * (y4 * y)
        beta = 12 * dJ4 + 30 * dJ6 * m4
        return h_y, beta

    # k == 8
    m5 = np.mean(y2 * y3)
    m6 = np.mean(y3 * y3)
    dJ3 = -7665 * m3 / 65536.0 + 497 * m5 / 262144.0
    dJ4 = 2765 * m4 / 1048576.0 - 18795.0 / 524288
    dJ5 = 497 * m3 / 262144.0
    dJ6 = 329.0 / 524288
    dJ8 = -7.0 / 2097152
    y4 = y2 * y2
    h_y = (dJ3 * 3 * y2 + dJ4 * 4 * y3 + dJ5 * 5 * y4
           + dJ6 * 6 * (y4 * y) + dJ8 * 8 * (y4 * y3))
    beta = 12 * dJ4 + 20 * dJ5 * m3 + 30 * dJ6 * m4 + 56 * dJ8 * m6
    return h_y, beta


# ======================================================================
# ASV computation
# ======================================================================

def asv(g, gp_mean, s):
    """Tichavsky et al. ASV.  g = g(s) array, gp_mean = E[g'(s)], s = source."""
    Eg2 = np.mean(g ** 2)
    Esg = np.mean(s * g)
    denom = (gp_mean - Esg) ** 2
    if denom < 1e-30:
        return np.inf
    return (Eg2 - Esg ** 2) / denom


def fastica_asv(s, k):
    """ASV for FastICA at order k:  g(y) = y^{k-1}."""
    g = s ** (k - 1)
    gp_mean = (k - 1) * np.mean(s ** (k - 2))
    return asv(g, gp_mean, s)


def lcc_asv(s, k):
    """ASV for LCC at order k, using exact h_k(y)."""
    h, beta = lcc_h_beta(s, k)
    return asv(h, beta, s)


# ======================================================================
# Distributions (standardised)
# ======================================================================

def std(x):
    return (x - np.mean(x)) / np.std(x)

DISTS = {
    "Laplace":    lambda N: std(np.random.laplace(0, 1, N)),
    "Logistic":   lambda N: std(np.random.logistic(0, 1, N)),
    "Uniform":    lambda N: std(np.random.uniform(-1, 1, N)),
    "Exponential": lambda N: std(np.random.exponential(1, N)),
    "Gamma(2)":   lambda N: std(np.random.gamma(2, 1, N)),
    "Gamma(3)":   lambda N: std(np.random.gamma(3, 1, N)),
    "Gamma(5)":   lambda N: std(np.random.gamma(5, 1, N)),
    "Gamma(8)":   lambda N: std(np.random.gamma(8, 1, N)),
    "Student-t15": lambda N: std(np.random.standard_t(15, N)),
}


# ======================================================================
# Main
# ======================================================================

def main():
    N = 10_000_000
    np.random.seed(42)

    print("=" * 80)
    print("Asymptotic Relative Efficiency:  ARE = ASV(FastICA) / ASV(LCC)")
    print("  Tichavsky, Koldovsky & Oja (2006), Prop. 1")
    print(f"  N = {N:,} for expectation estimates")
    print("=" * 80)

    # ---- Table 1: ASV and ARE for all distributions ----
    print(f"\n  {'Source':<14s}"
          f"  {'ASV_F4':>9s} {'ASV_L4':>9s} {'ARE_4':>7s}"
          f"  {'ASV_F6':>9s} {'ASV_L6':>9s} {'ARE_6':>7s}"
          f"  {'ASV_F8':>9s} {'ASV_L8':>9s} {'ARE_8':>7s}")
    print("  " + "-" * 100)

    for name, sampler in DISTS.items():
        s = sampler(N)
        af4 = fastica_asv(s, 4)
        al4 = lcc_asv(s, 4)
        are4 = af4 / al4 if al4 > 0 else np.inf

        af6 = fastica_asv(s, 6)
        al6 = lcc_asv(s, 6)
        are6 = af6 / al6 if al6 > 0 else np.inf

        af8 = fastica_asv(s, 8)
        al8 = lcc_asv(s, 8)
        are8 = af8 / al8 if al8 > 0 else np.inf

        print(f"  {name:<14s}"
              f"  {af4:9.3f} {al4:9.3f} {are4:7.2f}"
              f"  {af6:9.3f} {al6:9.3f} {are6:7.2f}"
              f"  {af8:9.3f} {al8:9.3f} {are8:7.2f}")

    # ---- Table 2: Compare ARE with empirical Amari^2 ratio ----
    # Empirical from --ho run at N=100000, 200 trials
    # Format: (Fast-k4, LCC-k4, Fast-k6, LCC-k6, Fast-k8, LCC-k8) Amari x100
    empirical = {
        "Laplace":    (0.72, 0.63, 1.30, 0.59, 2.29, 0.96),
        "Logistic":   (1.01, 1.17, 1.67, 0.91, 3.05, 0.92),
        "Uniform":    (0.24, 0.24, 0.21, 0.24, 0.19, 0.24),
    }

    print(f"\n\n  {'Source':<14s}"
          f"  {'ARE_4':>7s} {'(A_F/A_L)^2':>11s}"
          f"  {'ARE_6':>7s} {'(A_F/A_L)^2':>11s}"
          f"  {'ARE_8':>7s} {'(A_F/A_L)^2':>11s}")
    print("  " + "-" * 80)

    for name, (f4, l4, f6, l6, f8, l8) in empirical.items():
        s = DISTS[name](N)
        are4 = fastica_asv(s, 4) / lcc_asv(s, 4)
        are6 = fastica_asv(s, 6) / lcc_asv(s, 6)
        are8 = fastica_asv(s, 8) / lcc_asv(s, 8)
        r4 = (f4 / l4) ** 2
        r6 = (f6 / l6) ** 2
        r8 = (f8 / l8) ** 2
        print(f"  {name:<14s}"
              f"  {are4:7.2f} {r4:11.2f}"
              f"  {are6:7.2f} {r6:11.2f}"
              f"  {are8:7.2f} {r8:11.2f}")

    print("\n  ARE_k = theoretical prediction")
    print("  (A_F/A_L)^2 = empirical squared Amari ratio at N=100k")
    print("  Agreement validates the ASV framework for LCC.")


if __name__ == "__main__":
    main()
