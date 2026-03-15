"""icalcc_btree_numba.py -- Numba-accelerated binary-tree bounded LCC contrast.

Extends icalcc_btree with @njit(parallel=True) kernels for ltanh and lexp.
The outer loop over N is parallelized with numba.prange; the inner window
loop is compiled to native code eliminating all Python overhead.

Falls back to the pure-numpy btree (icalcc_btree) when Numba is not installed,
so the module is always importable.

Usage
-----
    from icalcc_btree_numba import ICALCCBTreeNumba

    est = ICALCCBTreeNumba(n_components=4, K='ltanh', random_state=0)
    S_hat = est.fit_transform(X)

    # Force warm-up (JIT compile) before benchmarking:
    ICALCCBTreeNumba.warmup(n=256)
"""

import numpy as np

try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False

from icalcc_btree import ICALCCBTree, _btree_contrast, _btree_h_gprime


# ---------------------------------------------------------------------------
# Numba kernels  (compiled only when Numba is available)
# ---------------------------------------------------------------------------

if _NUMBA_AVAILABLE:

    @njit(parallel=True, cache=True)
    def _numba_ltanh(y: np.ndarray, ys: np.ndarray, threshold: float):
        """Binary-tree ltanh kernel, JIT-compiled with prange parallelism.

        Parameters
        ----------
        y  : ndarray (N,)  -- original (unsorted) whitened projection.
        ys : ndarray (N,)  -- sorted copy of y (the binary tree).
        threshold : float  -- active window half-width.

        Returns
        -------
        gy, gpy : ndarray (N,) each.
        """
        N   = len(y)
        gy  = np.empty(N, dtype=np.float64)
        gpy = np.empty(N, dtype=np.float64)

        for i in prange(N):
            yi = y[i]
            lo = np.searchsorted(ys, yi - threshold)   # left boundary
            hi = np.searchsorted(ys, yi + threshold)   # right boundary

            near_g  = 0.0
            near_gp = 0.0
            for j in range(lo, hi):
                d = yi - ys[j]
                t = np.tanh(d)
                near_g  += t
                near_gp += 1.0 - t * t          # sech^2

            left_sat  = lo          # contributes +1 each
            right_sat = N - hi      # contributes -1 each

            gy[i]  = (near_g + left_sat - right_sat) / N
            gpy[i] = near_gp / N

        return gy, gpy


    @njit(parallel=True, cache=True)
    def _numba_lexp(y: np.ndarray, ys: np.ndarray, threshold: float):
        """Binary-tree lexp kernel, JIT-compiled with prange parallelism.

        Parameters
        ----------
        y  : ndarray (N,)  -- original (unsorted) whitened projection.
        ys : ndarray (N,)  -- sorted copy of y (the binary tree).
        threshold : float  -- active window half-width (tails negligible).

        Returns
        -------
        gy, gpy : ndarray (N,) each.
        """
        N   = len(y)
        gy  = np.empty(N, dtype=np.float64)
        gpy = np.empty(N, dtype=np.float64)

        for i in prange(N):
            yi = y[i]
            lo = np.searchsorted(ys, yi - threshold)
            hi = np.searchsorted(ys, yi + threshold)

            sum_g  = 0.0
            sum_gp = 0.0
            for j in range(lo, hi):
                d  = yi - ys[j]
                e  = np.exp(-0.5 * d * d)
                sum_g  += d * e
                sum_gp += (1.0 - d * d) * e

            # Tails: d * exp(-d^2/2) -> 0 at |d| >= threshold >= 4.
            gy[i]  = sum_g  / N
            gpy[i] = sum_gp / N

        return gy, gpy

else:
    # Stubs so the module body does not error at import time.
    _numba_ltanh = None
    _numba_lexp  = None


# ---------------------------------------------------------------------------
# Python-level dispatch (handles sort and 2-D / 1-D interface)
# ---------------------------------------------------------------------------

def _numba_h_gprime(
    y: np.ndarray,
    G: str = "tanh",
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated bounded LCC nonlinearity.

    Delegates to the JIT kernel if Numba is available, otherwise falls
    back to the pure-numpy btree implementation.

    Parameters
    ----------
    y         : ndarray (N,)   whitened projection.
    G         : {'tanh', 'exp'}
    threshold : float

    Returns
    -------
    gy, gpy : ndarray (N,) each.
    """
    if not _NUMBA_AVAILABLE:
        return _btree_h_gprime(y, G=G, threshold=threshold)

    ys = np.sort(y)                 # O(N log N); stays in NumPy, fine.
    y  = np.ascontiguousarray(y, dtype=np.float64)
    ys = np.ascontiguousarray(ys,  dtype=np.float64)

    if G == "tanh":
        return _numba_ltanh(y, ys, threshold)
    elif G == "exp":
        return _numba_lexp(y, ys, threshold)
    else:
        raise ValueError(f"G must be 'tanh' or 'exp', got {G!r}")


def _numba_contrast(
    x: np.ndarray,
    G: str = "tanh",
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba btree contrast callable for sklearn FastICA.

    Handles deflation mode (1-D input shape (N,)) and parallel mode
    (2-D input shape (p, N)).
    """
    if x.ndim == 1:
        return _numba_h_gprime(x, G=G, threshold=threshold)

    p, N    = x.shape
    gY      = np.empty_like(x)
    gpy_out = np.empty(p, dtype=x.dtype)
    for i in range(p):
        gi, gpi    = _numba_h_gprime(x[i], G=G, threshold=threshold)
        gY[i]      = gi
        gpy_out[i] = gpi.mean()
    return gY, gpy_out


# ---------------------------------------------------------------------------
# Extended class
# ---------------------------------------------------------------------------

class ICALCCBTreeNumba(ICALCCBTree):
    """ICALCCBTree with Numba-JIT parallel kernels for ltanh / lexp.

    Drop-in replacement for ICALCCBTree. When Numba is installed the
    bounded kernels are compiled with @njit(parallel=True); the outer
    loop over N uses numba.prange, and the inner window loop is native
    machine code. Falls back to ICALCCBTree (pure-numpy) when Numba is
    not available.

    Parameters
    ----------
    n_components : int or None, default=None
    K            : valid ICALCC K value, default=6
    threshold    : float, default=5.0
                   Active window half-width (same meaning as ICALCCBTree).
    algorithm    : {'parallel', 'deflation'}, default='parallel'
    whiten       : str or False, default='unit-variance'
    max_iter     : int, default=200
    tol          : float, default=1e-4
    w_init       : array-like or None
    whiten_solver: {'svd', 'eigh'}, default='svd'
    random_state : int, RandomState or None, default=None

    Class methods
    -------------
    warmup(n=256, threshold=5.0)
        Trigger JIT compilation on a small array before benchmarking.
        Has no effect when Numba is unavailable.

    Examples
    --------
    >>> import numpy as np
    >>> from icalcc_btree_numba import ICALCCBTreeNumba
    >>> ICALCCBTreeNumba.warmup()          # compile once
    >>> rng = np.random.RandomState(0)
    >>> X = rng.laplace(size=(2000, 4))
    >>> est = ICALCCBTreeNumba(n_components=4, K='ltanh', random_state=0)
    >>> S_hat = est.fit_transform(X)
    >>> est.converged_
    True
    """

    _BTREE_KERNELS = {'ltanh': 'tanh', 'lexp': 'exp'}

    def __init__(
        self,
        n_components=None,
        *,
        K=6,
        threshold: float = 5.0,
        algorithm="parallel",
        whiten="unit-variance",
        max_iter=200,
        tol=1e-4,
        w_init=None,
        whiten_solver="svd",
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            K=K,
            threshold=threshold,
            algorithm=algorithm,
            whiten=whiten,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            whiten_solver=whiten_solver,
            random_state=random_state,
        )
        # Override the btree (numpy) contrast with the numba contrast.
        if K in self._BTREE_KERNELS:
            self.fun      = _numba_contrast
            self.fun_args = dict(G=self._BTREE_KERNELS[K], threshold=threshold)

    @classmethod
    def warmup(cls, n: int = 256, threshold: float = 5.0) -> None:
        """Trigger JIT compilation before benchmarking.

        Runs both ltanh and lexp kernels on a small dummy array so that
        compilation time is excluded from subsequent timing measurements.

        Parameters
        ----------
        n         : int   -- dummy array size (small, just enough to compile).
        threshold : float -- must match the threshold used in benchmarks.
        """
        if not _NUMBA_AVAILABLE:
            return
        dummy = np.random.randn(n)
        _numba_h_gprime(dummy, G="tanh", threshold=threshold)
        _numba_h_gprime(dummy, G="exp",  threshold=threshold)
