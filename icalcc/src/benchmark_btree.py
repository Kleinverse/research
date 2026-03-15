"""benchmark_btree.py -- Correctness and speed benchmark.

Compares: ICALCC (B500), ICALCC (exact), ICALCCBTree (numpy), ICALCCBTreeNumba.

Sections
--------
1. CORRECTNESS       -- btree numpy and numba vs exact O(N^2) pairwise.
2. KERNEL SPEED      -- raw g/g' timing across N.
3. FULL ICA SPEED    -- fit_transform timing across N.
4. COMPONENTS AGREEMENT -- identical-seed components_ comparison.

Run:
    python benchmark_btree.py
    python benchmark_btree.py --skip-ica --threshold 4.0

Requirements: icalcc, icalcc_btree, icalcc_btree_numba (same dir or installed).
"""

import time
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from icalcc import ICALCC, _lcc_bounded_h_gprime
    from icalcc_btree import ICALCCBTree, _btree_h_gprime
except ImportError as e:
    sys.exit(f"Import error: {e}\nEnsure icalcc.py and icalcc_btree.py are on PYTHONPATH.")

try:
    from icalcc_btree_numba import ICALCCBTreeNumba, _numba_h_gprime, _NUMBA_AVAILABLE
except ImportError:
    ICALCCBTreeNumba = None
    _numba_h_gprime  = None
    _NUMBA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exact_bounded(y, G):
    """Full pairwise O(N^2) reference: batch_size=N."""
    return _lcc_bounded_h_gprime(y, G=G, batch_size=len(y))


def _make_problem(N, n_components, seed):
    """Synthetic ICA problem: Laplace sources, random mixing."""
    rng = np.random.RandomState(seed)
    S = rng.laplace(size=(n_components, N))
    A = rng.randn(n_components, n_components)
    return (A @ S).T          # shape (N, n_components)


def _time_fn(fn, n_reps):
    """Return median wall-clock time in milliseconds over n_reps calls."""
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# 1. Correctness check
# ---------------------------------------------------------------------------

def correctness_check(N=2000, seed=42, threshold=5.0):
    print("=" * 72)
    print("CORRECTNESS CHECK")
    print(f"  N={N}, threshold={threshold}")
    print("=" * 72)

    rng = np.random.RandomState(seed)
    y = rng.randn(N).astype(np.float64)

    hdr = (f"{'Kernel':<10} {'Impl':<8} {'Stat':<6} "
           f"{'MaxAbsErr_g':>13} {'MaxAbsErr_gp':>13} "
           f"{'RelErr_g':>11} {'RelErr_gp':>11}")
    print(hdr)
    print("-" * len(hdr))

    all_passed = True
    for G in ("tanh", "exp"):
        gy_ref, gpy_ref = _exact_bounded(y, G)
        kernel_name = "l" + G

        impls = [("numpy", lambda G=G: _btree_h_gprime(y, G=G, threshold=threshold))]
        if _NUMBA_AVAILABLE:
            impls.append(("numba", lambda G=G: _numba_h_gprime(y, G=G, threshold=threshold)))

        for label, fn in impls:
            gy_bt, gpy_bt = fn()
            abs_g  = np.abs(gy_bt  - gy_ref)
            abs_gp = np.abs(gpy_bt - gpy_ref)
            rel_g  = abs_g.max()  / (np.abs(gy_ref).max()  + 1e-15)
            rel_gp = abs_gp.max() / (np.abs(gpy_ref).max() + 1e-15)

            ok   = rel_g < 1e-3 and rel_gp < 1e-3
            flag = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False

            print(f"{kernel_name:<10} {label:<8} {flag:<6} "
                  f"{abs_g.max():>13.3e} {abs_gp.max():>13.3e} "
                  f"{rel_g:>11.3e} {rel_gp:>11.3e}")

    for G in ("tanh", "exp"):
        print()
        print(f"  Threshold sensitivity for l{G} (RelErr_g, numpy btree):")
        print(f"  {'thresh':>8} {'MaxAbsErr_g':>14} {'RelErr_g':>12}")
        gy_ref, _ = _exact_bounded(y, G)
        for thresh in (3.0, 4.0, 5.0, 6.0):
            gy_bt, _ = _btree_h_gprime(y, G=G, threshold=thresh)
            ae = np.abs(gy_bt - gy_ref).max()
            re = ae / (np.abs(gy_ref).max() + 1e-15)
            print(f"  {thresh:>8.1f} {ae:>14.3e} {re:>12.3e}")

    print()
    print(f"  Status: {'ALL PASSED' if all_passed else 'SOME FAILED -- inspect above'}")
    print()
    return all_passed


# ---------------------------------------------------------------------------
# 2. Kernel-level speed
# ---------------------------------------------------------------------------

def kernel_speed(N_list=(500, 1000, 2000, 5000, 10000), n_reps=5, seed=0, threshold=5.0):
    print("=" * 72)
    print("KERNEL-LEVEL SPEED  (g and g' only, ms, median of n_reps)")
    print(f"  n_reps={n_reps}, threshold={threshold}")
    print(f"  Numba available: {_NUMBA_AVAILABLE}")
    print("=" * 72)

    rng = np.random.RandomState(seed)

    if _NUMBA_AVAILABLE:
        print("  Warming up Numba JIT...")
        ICALCCBTreeNumba.warmup(n=256, threshold=threshold)
        print("  Done.\n")

    for G in ("tanh", "exp"):
        kernel_name = "l" + G
        hdr = (f"  {'N':>8} {'B500(ms)':>10} {'exact(ms)':>11} "
               f"{'numpy(ms)':>11} {'numba(ms)':>11} "
               f"{'sp_B500':>9} {'sp_exact':>9} {'sp_numpy':>9}")
        print(f"\n  Kernel: {kernel_name}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for N in N_list:
            y = rng.randn(N).astype(np.float64)

            t_b500  = _time_fn(lambda: _lcc_bounded_h_gprime(y, G=G, batch_size=500), n_reps)
            t_exact = _time_fn(lambda: _lcc_bounded_h_gprime(y, G=G, batch_size=N),   n_reps)
            t_numpy = _time_fn(lambda: _btree_h_gprime(y, G=G, threshold=threshold),   n_reps)

            if _NUMBA_AVAILABLE:
                t_numba  = _time_fn(lambda: _numba_h_gprime(y, G=G, threshold=threshold), n_reps)
                sp_b500  = t_b500  / t_numba
                sp_exact = t_exact / t_numba
                sp_numpy = t_numpy / t_numba
                numba_str = f"{t_numba:>11.2f}"
            else:
                sp_b500  = t_b500  / t_numpy
                sp_exact = t_exact / t_numpy
                sp_numpy = 1.0
                numba_str = f"{'N/A':>11}"

            print(f"  {N:>8d} {t_b500:>10.2f} {t_exact:>11.2f} "
                  f"{t_numpy:>11.2f} {numba_str} "
                  f"{sp_b500:>8.2f}x {sp_exact:>8.2f}x {sp_numpy:>8.2f}x")

    print()


# ---------------------------------------------------------------------------
# 3. Full ICA speed
# ---------------------------------------------------------------------------

def ica_speed(
    N_list=(1000, 3000, 5000, 10000),
    n_components=4,
    n_reps=3,
    seed=1,
    threshold=5.0,
):
    print("=" * 72)
    print(f"FULL ICA SPEED  (fit_transform, n_components={n_components}, K=ltanh)")
    print(f"  n_reps={n_reps}, threshold={threshold}")
    print(f"  Numba available: {_NUMBA_AVAILABLE}")
    print("=" * 72)

    if _NUMBA_AVAILABLE:
        print("  Warming up Numba JIT...")
        ICALCCBTreeNumba.warmup(n=256, threshold=threshold)
        print("  Done.\n")

    hdr = (f"{'N':>8} {'B500(s)':>9} {'exact(s)':>10} "
           f"{'numpy(s)':>10} {'numba(s)':>10} "
           f"{'sp_B500':>9} {'sp_exact':>9} {'sp_numpy':>9} "
           f"{'it_orig':>8} {'it_bt':>7}")
    print(hdr)
    print("-" * len(hdr))

    def _run_ica(cls, N, extra_fun_args=None):
        times = []
        iters = None
        kw = {"threshold": threshold} if cls not in (ICALCC,) else {}
        for rep in range(n_reps):
            est = cls(n_components=n_components, K='ltanh',
                      max_iter=200, random_state=rep, **kw)
            if extra_fun_args is not None:
                est.fun_args = extra_fun_args
            t0 = time.perf_counter()
            est.fit_transform(_make_problem(N, n_components, seed))
            times.append(time.perf_counter() - t0)
            if iters is None:
                iters = est.n_iter_
        return float(np.median(times)), iters

    for N in N_list:
        t_b500,  it_orig = _run_ica(ICALCC, N)
        t_exact, _       = _run_ica(ICALCC, N, extra_fun_args=dict(G='tanh', batch_size=N))
        t_numpy, it_bt   = _run_ica(ICALCCBTree, N)

        if _NUMBA_AVAILABLE:
            t_numba, _ = _run_ica(ICALCCBTreeNumba, N)
            sp_b500    = t_b500  / t_numba
            sp_exact   = t_exact / t_numba
            sp_numpy   = t_numpy / t_numba
            numba_str  = f"{t_numba:>10.4f}"
        else:
            sp_b500   = t_b500  / t_numpy
            sp_exact  = t_exact / t_numpy
            sp_numpy  = 1.0
            numba_str = f"{'N/A':>10}"

        print(f"{N:>8d} {t_b500:>9.4f} {t_exact:>10.4f} "
              f"{t_numpy:>10.4f} {numba_str} "
              f"{sp_b500:>8.2f}x {sp_exact:>8.2f}x {sp_numpy:>8.2f}x "
              f"{it_orig:>8d} {it_bt:>7d}")

    print()


# ---------------------------------------------------------------------------
# 4. Components agreement
# ---------------------------------------------------------------------------

def components_agreement(N=5000, n_components=3, seed=7, threshold=5.0):
    """Verify btree (numpy and numba) match exact pairwise reference.

    Both run with identical random_state=0 so row order is deterministic.
    Only per-row sign flips (ICA ambiguity) are corrected before comparison.
    """
    print("=" * 72)
    print("COMPONENTS AGREEMENT  (identical seed, vs exact pairwise)")
    print(f"  N={N}, n_components={n_components}, threshold={threshold}")
    print("=" * 72)

    X = _make_problem(N, n_components, seed)
    all_passed = True

    for K in ('ltanh', 'lexp'):
        G = 'tanh' if K == 'ltanh' else 'exp'

        # Reference: exact O(N^2) pairwise.
        est_ref = ICALCC(n_components=n_components, K=K,
                         max_iter=300, random_state=0)
        est_ref.fun_args = dict(G=G, batch_size=N)
        S_ref = est_ref.fit_transform(X)
        W_ref = est_ref.components_

        impls = [("numpy", ICALCCBTree)]
        if _NUMBA_AVAILABLE:
            impls.append(("numba", ICALCCBTreeNumba))

        for label, cls in impls:
            est = cls(n_components=n_components, K=K,
                      threshold=threshold, max_iter=300, random_state=0)
            S_bt = est.fit_transform(X)
            W_bt = est.components_

            signs     = np.sign(np.sum(W_ref * W_bt, axis=1, keepdims=True))
            W_aligned = W_bt * signs
            S_aligned = S_bt * signs.squeeze()

            max_err = np.abs(W_ref - W_aligned).max()
            rel_err = max_err / (np.abs(W_ref).max() + 1e-15)
            src_err = np.abs(S_ref - S_aligned).max()

            ok   = rel_err < 1e-2
            flag = "PASS" if ok else "FAIL"
            if not ok:
                all_passed = False

            print(f"  K={K:<8} impl={label:<6} {flag}  "
                  f"W_max={max_err:.3e}  W_rel={rel_err:.3e}  "
                  f"S_max={src_err:.3e}  "
                  f"iters_ref={est_ref.n_iter_}  iters={est.n_iter_}  "
                  f"conv={est.converged_}")

    print()
    print(f"  Status: {'ALL PASSED' if all_passed else 'SOME FAILED -- inspect above'}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark ICALCCBTree vs ICALCC")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--skip-kernel",      action="store_true")
    parser.add_argument("--skip-ica",         action="store_true")
    parser.add_argument("--skip-quality",     action="store_true")
    parser.add_argument("--threshold",        type=float, default=5.0)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    if not args.skip_correctness:
        correctness_check(N=2000, seed=args.seed, threshold=args.threshold)

    if not args.skip_kernel:
        kernel_speed(
            N_list=(500, 1000, 2000, 5000, 10000),
            n_reps=5,
            seed=args.seed,
            threshold=args.threshold,
        )

    if not args.skip_ica:
        ica_speed(
            N_list=(1000, 3000, 5000, 10000),
            n_components=4,
            n_reps=3,
            seed=args.seed,
            threshold=args.threshold,
        )

    if not args.skip_quality:
        components_agreement(
            N=5000,
            n_components=3,
            seed=args.seed,
            threshold=args.threshold,
        )
