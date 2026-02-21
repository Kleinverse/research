#!/usr/bin/env python3
"""
Experiment 3 for IEEE SPL paper:
  ICA separation quality via Amari index (Table II)

Usage:
  python exp3_ica.py           # full experiment (200 trials)
  python exp3_ica.py --quick   # quick run (20 trials)
  python exp3_ica.py --ho      # include higher-order LCC (k=4,6,8)
  python exp3_ica.py --test    # run test suite
  python exp3_ica.py --scan    # scan Gamma(alpha) for LCC-k8 threshold
  python exp3_ica.py --scan --quick  # quick scan (20 trials)

Combine flags: python exp3_ica.py --quick --ho
"""

import numpy as np
import math
import sys


# ======================================================================
# Distributions (unit variance)
#
# [Paper note]:
#   Uniform: sub-Gaussian (kappa4 = -1.2), symmetric
#   Laplace: super-Gaussian (kappa4 = 3), symmetric, exponential tails
#   Student-t15: super-Gaussian (kappa4 = 0.545), symmetric, polynomial tails
#       df=15 chosen so moments up to order 14 exist (V_8 needs m_8).
#       Student-t5 (kappa4=6) FAILS because m_6, m_8 are infinite.
#   Logistic: super-Gaussian (kappa4 = 1.2), symmetric, exponential tails
#   Exponential: super-Gaussian (kappa4 = 6), ASYMMETRIC
#       LCC-k6 works well; LCC-k8 has landscape pathology from m3 cross-terms.
#
#   LCC outperforms FastICA when:
#     (1) source is super-Gaussian (V_k exploits multi-moment structure)
#     (2) moments up to order k exist (finite V_k variance)
#     (3) for k=8, symmetric sources avoid cross-term pathology
# ======================================================================

class Laplace:
    name = "Laplace"
    true_kappa4 = 3.0
    @staticmethod
    def sample(N):
        return np.random.laplace(0, 1.0 / np.sqrt(2), N)

class Uniform:
    name = "Uniform"
    true_kappa4 = -1.2
    @staticmethod
    def sample(N):
        return np.random.uniform(-np.sqrt(3), np.sqrt(3), N)

class Exponential:
    name = "Exponential"
    true_kappa4 = 6.0
    @staticmethod
    def sample(N):
        return np.random.exponential(1, N) - 1.0

class StudentT15:
    name = "Student-t15"
    true_kappa4 = 6.0/11   # 6/(df-4) = 6/11 ~ 0.545 for df=15
    @staticmethod
    def sample(N):
        # var = df/(df-2) = 15/13, standardize to unit variance
        return np.random.standard_t(15, N) / np.sqrt(15.0/13)

class Logistic:
    name = "Logistic"
    true_kappa4 = 1.2       # excess kurtosis = 6/5
    @staticmethod
    def sample(N):
        # Logistic(0,1) has var = pi^2/3, standardize
        return np.random.logistic(0, 1, N) / (np.pi / np.sqrt(3))

DISTS = [Uniform, Laplace, Exponential, StudentT15, Logistic]


# ======================================================================
# ICA utilities
# ======================================================================

def random_mixing(d):
    H = np.random.randn(d, d)
    Q, R = np.linalg.qr(H)
    return Q * np.sign(np.diag(R))


def whiten(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    C = Xc @ Xc.T / Xc.shape[1]
    vals, vecs = np.linalg.eigh(C)
    D = np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-12)))
    W = D @ vecs.T
    return W @ Xc, W


def amari_index(W_est, A_true):
    P = np.abs(W_est @ A_true)
    d = P.shape[0]
    row_max = P.max(axis=1, keepdims=True)
    col_max = P.max(axis=0, keepdims=True)
    return (np.sum(P / row_max) - d + np.sum(P / col_max) - d) / (2*d*(d-1))


# ======================================================================
# FastICA (deflation, kurtosis contrast g(u)=u^3)
# ======================================================================

def fastica_kurtosis(Z, max_iter=200, tol=1e-7):
    d, N = Z.shape
    W = np.zeros((d, d))
    for p in range(d):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            y = w @ Z
            w_new = (Z * y**3).mean(axis=1) - 3.0 * w
            for j in range(p):
                w_new -= (w_new @ W[j]) * W[j]
            w_new /= np.linalg.norm(w_new) + 1e-12
            if abs(abs(w_new @ w) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        W[p] = w
    return W


# ======================================================================
# LCC V_4 statistic (for tests)
# ======================================================================

def lcc_v4(y):
    """LCC V_4 statistic. Assumes m_2=1 (whitened)."""
    return 21.0/64.0 - 3.0*np.mean(y**4)/64.0


# ======================================================================
# JADE (Cardoso & Souloumiac, jadeR v1.8)
#
# Faithful translation of Cardoso's reference MATLAB/NumPy code.
# Expects WHITENED data Z (d x N) with identity covariance.
#
# Key structure:
#   CM is m x (m * nbcm), where nbcm = m(m+1)/2 cumulant matrices
#   are stored side by side in blocks of m columns each.
#   Ip = p, p+m, p+2m, ... selects row p from each block.
#
# Jacobi criterion (Cardoso's formulation):
#   g = [CM[p,Ip]-CM[q,Iq] ; CM[p,Iq]+CM[q,Ip]]
#   gg = g @ g^T
#   ton = gg[0,0]-gg[1,1],  toff = gg[0,1]+gg[1,0]
#   theta = 0.5 * arctan2(toff, ton + sqrt(ton^2 + toff^2))
#
# Rotation updates:
#   rows:    CM[[p,q],:] = G^T @ CM[[p,q],:]
#   columns: CM[:,[Ip,Iq]] mixed by c,s
# ======================================================================

def jade(Z):
    d, N = Z.shape
    m = d
    T = float(N)

    # X is T x m (transposed for efficiency, matching Cardoso's code)
    X = Z.T.copy()  # (N, m)

    # Estimation of cumulant matrices
    nbcm = int(m * (m + 1) / 2)
    CM = np.zeros((m, m * nbcm))
    R = np.eye(m)

    Range = np.arange(m)
    for im in range(m):
        Xim = X[:, im]
        Xijm = Xim * Xim  # element-wise square
        # Qij = (Xijm[:,None] * X).T @ X / T  -  R  -  2 * outer(R[:,im], R[:,im])
        Qij = (Xijm[:, np.newaxis] * X).T @ X / T \
              - R - 2.0 * np.outer(R[:, im], R[:, im])
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = Xim * X[:, jm]
            Qij = np.sqrt(2.0) * (Xijm[:, np.newaxis] * X).T @ X / T \
                  - np.outer(R[:, im], R[:, jm]) \
                  - np.outer(R[:, jm], R[:, im])
            CM[:, Range] = Qij
            Range = Range + m

    # Joint diagonalization
    V = np.eye(m)
    seuil = 1.0e-6 / np.sqrt(T)
    encore = True
    sweep = 0

    while encore:
        encore = False
        sweep += 1
        if sweep > 200:
            break
        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = np.arange(p, m * nbcm, m)
                Iq = np.arange(q, m * nbcm, m)

                # Givens angle computation
                # g must be (2, nbcm) so that gg = g @ g.T is (2,2)
                g = np.vstack([CM[p, Ip] - CM[q, Iq],
                               CM[p, Iq] + CM[q, Ip]])
                gg = g @ g.T
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * np.arctan2(
                    toff, ton + np.sqrt(ton * ton + toff * toff))

                if abs(theta) > seuil:
                    encore = True
                    c = np.cos(theta)
                    s = np.sin(theta)

                    # Row rotation: G^T @ CM[[p,q], :]
                    # G = [[c,-s],[s,c]], G^T = [[c,s],[-s,c]]
                    rp = CM[p, :].copy()
                    rq = CM[q, :].copy()
                    CM[p, :] = c * rp + s * rq
                    CM[q, :] = -s * rp + c * rq

                    # Column rotation
                    cp = CM[:, Ip].copy()
                    cq = CM[:, Iq].copy()
                    CM[:, Ip] = c * cp + s * cq
                    CM[:, Iq] = -s * cp + c * cq

                    # Accumulate rotation
                    vp = V[:, p].copy()
                    vq = V[:, q].copy()
                    V[:, p] = c * vp + s * vq
                    V[:, q] = -s * vp + c * vq

    return V.T


# ======================================================================
# Single-order FastICA (k=4,6,8)
#
# Contrast at order k: maximize |E[y^k]| = |m_k|
# This uses a SINGLE raw moment as the separation criterion.
# Newton fixed-point:
#   g(y) = y^{k-1},  g'(y) = (k-1)*y^{k-2}
#   w <- E[z*y^{k-1}] - (k-1)*E[y^{k-2}]*w
#
# [Paper Section III note]:
#   Fast-k4 is classical FastICA (kurtosis).
#   Fast-k6, Fast-k8 are the natural baselines: they optimize a single
#   high-order moment, isolating the effect of order from the effect
#   of the LCC kernel structure.
# ======================================================================

def fastica_single(Z, k=4, max_iter=200, tol=1e-7):
    """FastICA using single moment order k as contrast."""
    d, N = Z.shape
    W = np.zeros((d, d))
    for p in range(d):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            y = w @ Z
            ykm1 = y ** (k - 1)
            beta = (k - 1) * np.mean(y ** (k - 2))
            w_new = (Z * ykm1).mean(axis=1) - beta * w
            for j in range(p):
                w_new -= (w_new @ W[j]) * W[j]
            w_new /= np.linalg.norm(w_new) + 1e-12
            if abs(abs(w_new @ w) - 1.0) < tol:
                w = w_new
                break
            w = w_new
        W[p] = w
    return W


# ======================================================================
# LCC Single-Order FastICA (k=4,6,8) -- Newton fixed-point
#
# [Paper Section III note]:
#   Each V_k is a V-statistic with kernel prod_{i=1}^k (y_i - ybar_k),
#   NOT a cumulant. The key distinction:
#
#     Fast-k6 optimizes |m_6| (a single raw moment)
#     LCC-k6  optimizes |V_6| = |f(m_3^2, m_4, m_6)|
#
#   V_6 jointly exploits m_3^2, m_4, and m_6, extracting richer structure
#   from the k-th order kernel than a single moment can. This multi-moment
#   structure is why LCC-k6 achieves ~2x better separation than Fast-k6
#   on super-Gaussian sources.
#
#   The sum sum_k c_k V_k converges to the locally-centered MGF, recovering
#   the full distribution, but each individual V_k already provides a more
#   efficient estimator of k-th order non-Gaussianity than m_k alone.
#
#   Odd orders (k=3,5,7) are excluded because:
#   (a) V_k for odd k vanishes on symmetric distributions (no signal)
#   (b) V_5, V_7 are dominated by lower-order moment leakage
#       (e.g. V_5 ~ -28*m3/125, making it essentially a skewness proxy)
#   Even orders k=4,6,8 have clean leading terms in m_k itself.
#
# Exact V_k formulas (whitened: m1=0, m2=1):
#   V_4 = 21/64 - 3*m4/64
#   V_6 = 145*m3^2/3888 + 115*m4/2592 - 5*m6/7776 - 125/648
#   V_8 = -7665*m3^2/131072 + 497*m3*m5/262144 + 2765*m4^2/2097152
#         - 18795*m4/524288 + 329*m6/524288 - 7*m8/2097152 + 117705/1048576
#
# Gradient: dJ/dw = E[z * h(y)] where h(y) = sum_r (dV_k/dm_r)*r*y^{r-1}
# Newton: w <- E[z*h(y)] - beta*w
#   beta = E[h'(y)] = sum_r (dV_k/dm_r)*r*(r-1)*E[y^{r-2}]
# ======================================================================


def _lcc_Vk(y, k):
    """Evaluate V_k on whitened projection y (m1=0, m2=1)."""
    m3 = np.mean(y**3)
    m4 = np.mean(y**4)
    if k == 4:
        return 21.0/64 - 3*m4/64
    m5 = np.mean(y**5)
    m6 = np.mean(y**6)
    if k == 6:
        return 145*m3**2/3888 + 115*m4/2592 - 5*m6/7776 - 125.0/648
    # k == 8
    m8 = np.mean(y**8)
    return (-7665*m3**2/131072 + 497*m3*m5/262144
            + 2765*m4**2/2097152 - 18795*m4/524288
            + 329*m6/524288 - 7*m8/2097152 + 117705.0/1048576)


def _lcc_h_beta(y, k):
    """Compute h(y) and beta for single even-order LCC at order k.

    Returns (h_y, beta) where:
      h_y is an array of length N (the nonlinear function for Newton step)
      beta is a scalar (E[h'(y)] on whitened data)
    """
    y2 = y * y
    y3 = y2 * y

    if k == 4:
        # V_4 = 21/64 - 3*m4/64.  dV/dm4 = -3/64.
        # h = (-3/64)*4*y^3 = -3/16 * y^3
        # beta = 12*(-3/64) = -9/16
        return (-3.0/16) * y3, -9.0/16

    m3 = np.mean(y3)
    m4 = np.mean(y2 * y2)

    if k == 6:
        # dV/dm3 = 145*m3/1944,  dV/dm4 = 115/2592,  dV/dm6 = -5/7776
        dJ3 = 145*m3/1944.0
        dJ4 = 115.0/2592
        dJ6 = -5.0/7776
        y4 = y2 * y2
        h_y = dJ3*3*y2 + dJ4*4*y3 + dJ6*6*(y4*y)
        beta = 12*dJ4 + 30*dJ6*m4   # r=3:0 (m1=0), r=4:12*dJ4, r=6:30*dJ6*m4
        return h_y, beta

    # k == 8
    m5 = np.mean(y2 * y3)
    m6 = np.mean(y3 * y3)
    dJ3 = -7665*m3/65536.0 + 497*m5/262144.0
    dJ4 = 2765*m4/1048576.0 - 18795.0/524288
    dJ5 = 497*m3/262144.0
    dJ6 = 329.0/524288
    dJ8 = -7.0/2097152
    y4 = y2 * y2
    h_y = (dJ3*3*y2 + dJ4*4*y3 + dJ5*5*y4
           + dJ6*6*(y4*y) + dJ8*8*(y4*y3))
    beta = 12*dJ4 + 20*dJ5*m3 + 30*dJ6*m4 + 56*dJ8*m6
    return h_y, beta


def fastica_lcc(Z, k=4, max_iter=200, tol=1e-7, n_restarts=5,
                warm_W=None, damping=1.75):
    """LCC-FastICA with warm-start and damped Newton."""
    d, N = Z.shape
    W = np.zeros((d, d))

    for p in range(d):
        best_w = None
        best_Vk = -np.inf

        for restart in range(n_restarts):
            if restart == 0 and warm_W is not None:
                w = warm_W[p].copy()
                w /= np.linalg.norm(w)
            else:
                w = np.random.randn(d)
                w /= np.linalg.norm(w)

            for it in range(max_iter):
                y = w @ Z
                h_y, beta = _lcc_h_beta(y, k)

                w_newton = (Z * h_y).mean(axis=1) - beta * w
                # damped step
                w_new = (1 - damping) * w + damping * w_newton

                for j in range(p):
                    w_new -= (w_new @ W[j]) * W[j]
                w_new /= np.linalg.norm(w_new) + 1e-12

                if abs(abs(w_new @ w) - 1.0) < tol:
                    w = w_new
                    break
                w = w_new

            Vk = abs(_lcc_Vk(w @ Z, k))
            if Vk > best_Vk:
                best_Vk = Vk
                best_w = w.copy()

        W[p] = best_w
    return W

def exp3_ica(d=4, n_trials=200, ho=False):
    label = "EXP 3"
    if ho:
        label += " (with higher-order LCC)"
    print("=" * 70)
    print(f"{label}: Amari index x100 (d={d}, {n_trials} trials)")
    print("=" * 70)

    Ns = [2000, 10000, 50000]
    algos = [
        ("FastICA",  fastica_kurtosis),
        ("JADE",     jade),
        ("LCC-k4",   lambda Z: fastica_lcc(Z, k=4)),
    ]
    if ho:
        Ns = [1000, 10000, 100000]
        algos = [
            ("FastICA",  fastica_kurtosis),   # = Fast-k4
            ("JADE",     jade),
            ("LCC-k4",   lambda Z: fastica_lcc(Z, k=4)),
            ("Fast-k6",  lambda Z: fastica_single(Z, k=6)),
            ("LCC-k6",   lambda Z: fastica_lcc(Z, k=6)),
            ("Fast-k8",  lambda Z: fastica_single(Z, k=8)),
            ("LCC-k8",   lambda Z: fastica_lcc(Z, k=8)),
        ]

    for di, Dist in enumerate(DISTS):
        print(f"\n  {Dist.name}")
        print(f"  {'':12s}", end="")
        for N in Ns:
            print(f"  {'N='+str(N):>8s}", end="")
        print()

        for ai, (name, algo_fn) in enumerate(algos):
            print(f"  {name:<12s}", end="", flush=True)
            for ni, N in enumerate(Ns):
                ais = []
                for trial in range(n_trials):
                    np.random.seed(di * 1000000 + ni * 10000 + trial)
                    S = np.vstack([Dist.sample(N) for _ in range(d)])
                    A_mix = random_mixing(d)
                    X = A_mix @ S
                    Z, Ww = whiten(X)
                    np.random.seed(di*1000000 + ni*10000 + trial + (ai+1)*100000)
                    try:
                        Ws = algo_fn(Z)
                        ais.append(amari_index(Ws @ Ww, A_mix))
                    except Exception:
                        ais.append(1.0)
                print(f"  {np.mean(ais)*100:8.2f}", end="", flush=True)
            print()


# ======================================================================
# Test suite (--test)
# ======================================================================

def run_tests():
    np.random.seed(123)
    n_pass, n_fail = 0, 0

    def check(name, cond, detail=""):
        nonlocal n_pass, n_fail
        if cond:
            n_pass += 1
            print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
        else:
            n_fail += 1
            print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))

    print("=" * 70)
    print("TEST SUITE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # T1. Distribution parameters
    # ------------------------------------------------------------------
    print("\nT1. Distribution parameters (N=500000)")
    N = 500000
    for Dist, want_var, want_k4 in [
        (Uniform,     1.0, -1.2),
        (Laplace,     1.0,  3.0),
        (Exponential, 1.0,  6.0),
    ]:
        x = Dist.sample(N)
        mu = np.mean(x)
        var = np.var(x)
        m4c = np.mean((x - mu)**4)
        k4 = m4c - 3.0 * var**2
        check(f"{Dist.name} mean~0",
              abs(mu) < 0.02, f"got {mu:.4f}")
        check(f"{Dist.name} var~{want_var}",
              abs(var - want_var) < 0.02, f"got {var:.4f}")
        check(f"{Dist.name} kappa4~{want_k4}",
              abs(k4 - want_k4) < 0.2, f"got {k4:.3f}")

    # ------------------------------------------------------------------
    # T2. Whitening
    # ------------------------------------------------------------------
    print("\nT2. Whitening")
    d, N = 4, 10000
    S = np.vstack([Uniform.sample(N) for _ in range(d)])
    A = random_mixing(d)
    Z, Ww = whiten(A @ S)
    cov = Z @ Z.T / N
    check("diag(Cov) ~ 1",
          np.allclose(np.diag(cov), 1.0, atol=0.05),
          f"diag = {np.diag(cov).round(3)}")
    off = cov - np.diag(np.diag(cov))
    check("off-diag ~ 0",
          np.max(np.abs(off)) < 0.05,
          f"max = {np.max(np.abs(off)):.4f}")

    # ------------------------------------------------------------------
    # T3. m_2 = 1 on whitened projections
    # ------------------------------------------------------------------
    print("\nT3. m_2 = 1 for unit w on whitened data")
    d, N = 4, 50000
    S = np.vstack([Laplace.sample(N) for _ in range(d)])
    Z, _ = whiten(random_mixing(d) @ S)
    for trial in range(5):
        w = np.random.randn(d); w /= np.linalg.norm(w)
        m2 = np.mean((w @ Z)**2)
        check(f"trial {trial+1}",
              abs(m2 - 1.0) < 0.03, f"m_2 = {m2:.4f}")

    # ------------------------------------------------------------------
    # T4. Amari index
    # ------------------------------------------------------------------
    print("\nT4. Amari index")
    I4 = np.eye(4)
    check("Amari(I, I) = 0", abs(amari_index(I4, I4)) < 1e-10)
    P = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]], dtype=float)
    check("Amari(perm, I) = 0", abs(amari_index(P, I4)) < 1e-10)
    D = np.diag([2.0, -1.0, 0.5, 3.0])
    check("Amari(scale*perm, I) = 0", abs(amari_index(D @ P, I4)) < 1e-10)

    # ------------------------------------------------------------------
    # T5. V_4 analytic values
    # ------------------------------------------------------------------
    print("\nT5. LCC V_4")
    N = 300000
    for Dist in DISTS:
        x = Dist.sample(N)
        v4 = lcc_v4(x)
        v4_theory = 12.0/64.0 - 3.0 * Dist.true_kappa4 / 64.0
        check(f"{Dist.name}",
              abs(v4 - v4_theory) < 0.02,
              f"sample={v4:.4f}, theory={v4_theory:.4f}")

    # ------------------------------------------------------------------
    # T6. JADE cumulant matrices (independent sources -> diagonal)
    # ------------------------------------------------------------------
    print("\nT6. JADE cumulant structure")
    np.random.seed(600)
    d, N = 4, 50000
    S = np.vstack([Laplace.sample(N) for _ in range(d)])
    Z, Ww = whiten(S)
    W = jade(Z)
    ai = amari_index(W @ Ww, np.eye(d))
    check("JADE on independent sources",
          ai < 0.05, f"Amari = {ai*100:.2f}%")

    # ------------------------------------------------------------------
    # T7. JADE 2D exact recovery
    # ------------------------------------------------------------------
    print("\nT7. JADE 2D")
    np.random.seed(600)
    d, N = 2, 30000
    S = np.vstack([Uniform.sample(N), Laplace.sample(N)])
    theta = np.pi / 5
    A2 = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    Z, Ww = whiten(A2 @ S)
    W = jade(Z)
    ai = amari_index(W @ Ww, A2)
    check("JADE 2D Amari < 5%", ai < 0.05, f"Amari = {ai*100:.2f}%")

    # ------------------------------------------------------------------
    # T8. FastICA 2D
    # ------------------------------------------------------------------
    print("\nT8. FastICA 2D")
    np.random.seed(700)
    Z, Ww = whiten(A2 @ S)
    W = fastica_kurtosis(Z)
    ai = amari_index(W @ Ww, A2)
    check("FastICA 2D Amari < 2%", ai < 0.02, f"Amari = {ai*100:.2f}%")

    # ------------------------------------------------------------------
    # T9. 4D all algorithms
    # ------------------------------------------------------------------
    print("\nT9. 4D separation (30 trials)")
    np.random.seed(800)
    d, N, n_trials = 4, 10000, 30
    for Dist in DISTS:
        for name, algo_fn in [("FastICA", fastica_kurtosis),
                               ("JADE", jade),
                               ("LCC-k4", fastica_lcc)]:
            ais = []
            for _ in range(n_trials):
                S = np.vstack([Dist.sample(N) for _ in range(d)])
                A = random_mixing(d)
                Z, Ww = whiten(A @ S)
                try:
                    W = algo_fn(Z)
                    ais.append(amari_index(W @ Ww, A))
                except Exception:
                    ais.append(1.0)
            mean_ai = np.mean(ais) * 100
            check(f"{Dist.name} {name} < 10%",
                  mean_ai < 10, f"Amari = {mean_ai:.2f}%")

    # ------------------------------------------------------------------
    # T10. JADE monotonicity in N
    # ------------------------------------------------------------------
    print("\nT10. JADE monotonicity (30 trials)")
    np.random.seed(900)
    d, n_trials = 4, 30
    for Dist in DISTS:
        results = {}
        for N in [2000, 10000, 50000]:
            ais = []
            for _ in range(n_trials):
                S = np.vstack([Dist.sample(N) for _ in range(d)])
                A = random_mixing(d)
                Z, Ww = whiten(A @ S)
                try:
                    W = jade(Z)
                    ais.append(amari_index(W @ Ww, A))
                except Exception:
                    ais.append(1.0)
            results[N] = np.mean(ais) * 100
        check(f"{Dist.name} 10k < 2k",
              results[10000] < results[2000] + 0.5,
              f"{results[2000]:.2f} -> {results[10000]:.2f}")
        check(f"{Dist.name} 50k < 10k",
              results[50000] < results[10000] + 0.5,
              f"{results[10000]:.2f} -> {results[50000]:.2f}")

    # ------------------------------------------------------------------
    # T11. LCC = FastICA at k=4
    # ------------------------------------------------------------------
    print("\nT11. LCC ~ FastICA at k=4")
    np.random.seed(1000)
    d, N, n_trials = 4, 10000, 30
    for Dist in DISTS:
        diffs = []
        for _ in range(n_trials):
            S = np.vstack([Dist.sample(N) for _ in range(d)])
            A = random_mixing(d)
            Z, Ww = whiten(A @ S)
            state = np.random.get_state()
            np.random.set_state(state)
            ai_f = amari_index(fastica_kurtosis(Z) @ Ww, A)
            np.random.set_state(state)
            ai_l = amari_index(fastica_lcc(Z) @ Ww, A)
            diffs.append(abs(ai_f - ai_l))
        md = np.mean(diffs) * 100
        check(f"{Dist.name} |diff| < 0.1%",
              md < 0.1, f"mean |diff| = {md:.4f}%")

    # ------------------------------------------------------------------
    # T12. V_6 formula validation (brute-force on small sample)
    # ------------------------------------------------------------------
    print("\nT12. V_6 analytic formula")
    np.random.seed(1200)
    N = 200
    for Dist in DISTS:
        y = Dist.sample(N)
        # Brute-force V_6: average over all N^6 tuples
        # Too expensive, use MC with large count
        n_mc = 300000
        idx = np.random.randint(0, N, (n_mc, 6))
        yi = y[idx]
        ybar = yi.mean(axis=1, keepdims=True)
        v6_mc = np.mean(np.prod(yi - ybar, axis=1))
        # Analytic
        m3 = np.mean(y**3)
        m4 = np.mean(y**4)
        m6 = np.mean(y**6)
        m2 = np.mean(y**2)
        m1 = np.mean(y)
        # Full formula (NOT whitened, use actual moments)
        v6_fm = (155*m1**6/324 - 155*m1**4*m2/108 + 55*m1**3*m3/162
                 + 85*m1**2*m2**2/72 - 35*m1**2*m4/648
                 - 65*m1*m2*m3/162 + 5*m1*np.mean(y**5)/1296
                 - 125*m2**3/648 + 115*m2*m4/2592
                 + 145*m3**2/3888 - 5*m6/7776)
        check(f"{Dist.name} V_6 MC~formula",
              abs(v6_mc - v6_fm) < 0.015,
              f"MC={v6_mc:.5f}, formula={v6_fm:.5f}")

    # ------------------------------------------------------------------
    # T13. LCC single-order 2D convergence (k=4,6,8)
    # ------------------------------------------------------------------
    print("\nT13. LCC single-order 2D")
    np.random.seed(1300)
    d, N = 2, 20000
    S = np.vstack([Exponential.sample(N), Laplace.sample(N)])
    theta13 = np.pi / 5
    A13 = np.array([[np.cos(theta13), -np.sin(theta13)],
                     [np.sin(theta13),  np.cos(theta13)]])
    Z, Ww = whiten(A13 @ S)
    for kk in [4, 6, 8]:
        np.random.seed(1300 + kk)
        W = fastica_lcc(Z, k=kk)
        ai = amari_index(W @ Ww, A13)
        check(f"LCC k={kk} Amari < 10%", ai < 0.10, f"Amari = {ai*100:.2f}%")

    # ------------------------------------------------------------------
    # T14. FastICA single-order 2D convergence (k=4,6,8)
    # ------------------------------------------------------------------
    print("\nT14. FastICA single-order 2D")
    for kk in [4, 6, 8]:
        np.random.seed(1400 + kk)
        W = fastica_single(Z, k=kk)
        ai = amari_index(W @ Ww, A13)
        check(f"Fast k={kk} Amari < 10%", ai < 0.10, f"Amari = {ai*100:.2f}%")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    total = n_pass + n_fail
    print(f"RESULTS: {n_pass}/{total} passed, {n_fail} failed")
    if n_fail == 0:
        print("ALL TESTS PASSED")
    print("=" * 70)
    return n_fail == 0


# ======================================================================

def scan_gamma_k8(n_trials=20, N=100000, d=4):
    """Scan Gamma(alpha) shape parameter to find where LCC-k8 works.

    Gamma(alpha) generalises Exponential (alpha=1).
    As alpha grows, skewness = 2/sqrt(alpha) -> 0 and m3 cross-terms
    in V_8 weaken. This scan finds the threshold alpha where LCC-k8
    starts to outperform Fast-k8.
    """
    print("=" * 70)
    print(f"SCAN: Gamma(alpha) shape vs LCC-k8 (d={d}, N={N}, {n_trials} trials)")
    print("=" * 70)
    print(f"  {'alpha':>6s}  {'skew':>6s}  {'kurt':>6s}"
          f"  {'FICA_4':>8s}  {'LCC_4':>8s}  {'FICA_6':>8s}  {'LCC_6':>8s}"
          f"  {'FICA_8':>8s}  {'LCC_8':>8s}")

    #for alpha in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 50.0]:
    for alpha in [1.0]:
        # Gamma(alpha): mean=alpha, var=alpha, skew=2/sqrt(alpha)
        # Standardise to zero mean, unit variance
        skew = 2.0 / np.sqrt(alpha)
        kurt = 6.0 / alpha  # excess kurtosis

        def sample_gamma(N, a=alpha):
            x = np.random.gamma(a, 1.0, N)
            return (x - a) / np.sqrt(a)

        algos = [
            ("FICA_4",  fastica_kurtosis),
            ("LCC_4",   lambda Z: fastica_lcc(Z, k=4)),
            ("FICA_6",  lambda Z: fastica_single(Z, k=6)),
            ("LCC_6",   lambda Z: fastica_lcc(Z, k=6)),
            ("FICA_8",  lambda Z: fastica_single(Z, k=8)),
            ("LCC_8",   lambda Z: fastica_lcc(Z, k=8)),
        ]

        results = {name: [] for name, _ in algos}
        for trial in range(n_trials):
            np.random.seed(trial * 1000)
            S = np.vstack([sample_gamma(N) for _ in range(d)])
            A_mix = random_mixing(d)
            Z, Ww = whiten(A_mix @ S)
            for ai, (name, fn) in enumerate(algos):
                np.random.seed(trial * 1000 + (ai+1) * 100)
                try:
                    W = fn(Z)
                    results[name].append(amari_index(W @ Ww, A_mix))
                except Exception:
                    results[name].append(1.0)

        vals = [np.mean(results[name]) * 100 for name, _ in algos]
        print(f"  {alpha:6.1f}  {skew:6.2f}  {kurt:6.2f}"
              + "".join(f"  {v:8.2f}" for v in vals))


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    if "--scan" in sys.argv:
        n = 20 if "--quick" in sys.argv else 200
        scan_gamma_k8(n_trials=n)
        sys.exit(0)
    n = 20 if "--quick" in sys.argv else 200
    ho = "--ho" in sys.argv
    exp3_ica(d=4, n_trials=n, ho=ho)
