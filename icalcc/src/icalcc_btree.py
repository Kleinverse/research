"""icalcc_btree.py -- Exact binary-tree bounded LCC contrast for ICALCC.

Prototype extending icalcc.ICALCC with an O(N·W) exact replacement
for the O(N·B) subsampled bounded contrast (ltanh, lexp).

For each whitened projection y (1D), the sorted array acts as the
binary tree: binary search locates the active window [lo, hi] around
each y_i where contributions are non-negligible (|y_i - y_j| <= threshold).
Saturated tails are resolved analytically rather than by dropping samples.

  g(y_i)  = (1/N) sum_j G'(y_i - y_j)
  g'(y_i) = (1/N) sum_j G''(y_i - y_j)

Tail treatment (unit-variance whitened input assumed):
  ltanh: tanh(u) -> +1 (u > threshold) / -1 (u < -threshold).
         Left-saturated tail (small y_j, large positive diff) contributes +1.
         Right-saturated tail (large y_j, large negative diff) contributes -1.
         Both counts are exact; gpy tail contribution is zero (sech^2 -> 0).
  lexp:  (diff * exp(-u^2/2)) and ((1-u^2) * exp(-u^2/2)) both -> 0.
         At threshold=5.0: exp(-12.5) < 3.7e-6 per pair; negligible.

Polynomial kernels (K = 4, 6, 8, fast4, fast6, fast8, tanh, exp, skew)
are unchanged and delegate entirely to the parent ICALCC class.

Usage
-----
    from icalcc_btree import ICALCCBTree

    est = ICALCCBTree(n_components=3, K='ltanh', random_state=0)
    S_hat = est.fit_transform(X)
    print(est.converged_, est.n_iter_)

    # polynomial kernels work identically to ICALCC
    est2 = ICALCCBTree(n_components=3, K=6, random_state=0)
    S_hat2 = est2.fit_transform(X)
"""

import numpy as np
from icalcc import ICALCC

# ---------------------------------------------------------------------------
# Core 1-D kernel
# ---------------------------------------------------------------------------

def _btree_h_gprime(
    y: np.ndarray,
    G: str = "tanh",
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact bounded LCC nonlinearity via sorted-array binary search.

    Parameters
    ----------
    y : ndarray, shape (N,)
        Whitened projection. Unit variance assumed for default threshold.
    G : {'tanh', 'exp'}
        Bounded kernel identifier ('tanh' for ltanh, 'exp' for lexp).
    threshold : float
        Active window half-width in the same units as y.

    Returns
    -------
    gy : ndarray, shape (N,)
        Nonlinearity g(y_i) = (1/N) sum_j G'(y_i - y_j).
    gpy : ndarray, shape (N,)
        Derivative g'(y_i) = (1/N) sum_j G''(y_i - y_j).

    Complexity
    ----------
    O(N log N) for the initial sort + O(N * W) for window evaluations,
    where W is the average active window width. W is bounded by the
    kernel's effective support relative to the data range.
    """
    N = len(y)
    ys = np.sort(y)  # binary tree: sorted array, O(N log N)

    gy  = np.empty(N, dtype=np.float64)
    gpy = np.empty(N, dtype=np.float64)

    if G == "tanh":
        for i in range(N):
            yi = y[i]

            # Binary search: O(log N) per sample.
            # Indices 0..lo-1:  ys[j] < yi - threshold
            #   -> diff = yi - ys[j] > threshold  -> tanh -> +1  (left sat.)
            # Indices lo..hi-1: active window, compute exactly.
            # Indices hi..N-1:  ys[j] > yi + threshold
            #   -> diff = yi - ys[j] < -threshold -> tanh -> -1  (right sat.)
            lo = np.searchsorted(ys, yi - threshold, side='left')
            hi = np.searchsorted(ys, yi + threshold, side='right')

            diff = yi - ys[lo:hi]          # shape (W,)
            t    = np.tanh(diff)

            near_g  = t.sum()
            near_gp = (1.0 - t * t).sum()  # sech^2

            left_sat  = lo        # each contributes +1
            right_sat = N - hi    # each contributes -1

            gy[i]  = (near_g + left_sat - right_sat) / N
            gpy[i] = near_gp / N            # tail sech^2 < 1.8e-4 at thresh=5

    elif G == "exp":
        for i in range(N):
            yi = y[i]
            lo = np.searchsorted(ys, yi - threshold, side='left')
            hi = np.searchsorted(ys, yi + threshold, side='right')

            diff = yi - ys[lo:hi]
            e    = np.exp(-0.5 * diff * diff)

            # Tails: diff * exp and (1-diff^2) * exp both -> 0; omit.
            gy[i]  = (diff * e).sum() / N
            gpy[i] = ((1.0 - diff * diff) * e).sum() / N

    else:
        raise ValueError(f"G must be 'tanh' or 'exp', got {G!r}")

    return gy, gpy


# ---------------------------------------------------------------------------
# Contrast callable (sklearn FastICA interface)
# ---------------------------------------------------------------------------

def _btree_contrast(
    x: np.ndarray,
    G: str = "tanh",
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary-tree bounded contrast callable for sklearn FastICA.

    Handles both deflation mode (x is 1-D, shape (N,)) and parallel
    mode (x is 2-D, shape (p, N)).
    """
    if x.ndim == 1:
        return _btree_h_gprime(x, G=G, threshold=threshold)

    p, N = x.shape
    gY      = np.empty_like(x)
    gpy_out = np.empty(p, dtype=x.dtype)
    for i in range(p):
        gi, gpi       = _btree_h_gprime(x[i], G=G, threshold=threshold)
        gY[i]         = gi
        gpy_out[i]    = gpi.mean()
    return gY, gpy_out


# ---------------------------------------------------------------------------
# Extended class
# ---------------------------------------------------------------------------

class ICALCCBTree(ICALCC):
    """ICALCC with exact binary-tree bounded contrast (ltanh / lexp).

    Replaces the subsampled O(N*B) pairwise approximation in the parent
    ICALCC with an exact O(N*W) computation using a sorted array and
    binary search. Saturated tails are counted analytically (ltanh) or
    dropped with bounded error (lexp). Polynomial kernels are unchanged.

    Parameters
    ----------
    n_components : int or None, default=None
        Number of components. Same as FastICA / ICALCC.
    K : valid ICALCC K, default=6
        Contrast selection. For K in {'ltanh', 'lexp'} the binary-tree
        kernel is activated. All other K values delegate to ICALCC.
    threshold : float, default=5.0
        Active window half-width in standard units (unit-variance input
        assumed). Controls the near-exact computation region:
          - 5.0: lexp tail error < 3.7e-6; ltanh sech^2 tail < 1.8e-4.
          - 4.0: lexp tail error < 3.4e-4; faster for large N.
        Has no effect when K is a polynomial kernel.
    algorithm : {'parallel', 'deflation'}, default='parallel'
    whiten : str or False, default='unit-variance'
    max_iter : int, default=200
    tol : float, default=1e-4
    w_init : array-like or None, default=None
    whiten_solver : {'svd', 'eigh'}, default='svd'
    random_state : int, RandomState or None, default=None

    Attributes
    ----------
    components_ : ndarray, shape (n_components, n_features)
    mixing_ : ndarray, shape (n_features, n_components)
    converged_ : bool
    n_iter_ : int

    Notes
    -----
    The Python loop over N in `_btree_h_gprime` is the primary bottleneck.
    A vectorized prefix-sum approach is planned for a future revision,
    particularly for ltanh where the tail counts reduce to cumulative
    index arithmetic and the near window can be batched over sorted blocks.
    For large N in the interim, the parent ICALCC with a tuned batch_size
    may be preferable on throughput grounds.

    Examples
    --------
    >>> import numpy as np
    >>> from icalcc_btree import ICALCCBTree
    >>> rng = np.random.RandomState(0)
    >>> S = rng.laplace(size=(3, 5000))
    >>> A = rng.randn(3, 3)
    >>> X = (A @ S).T
    >>> est = ICALCCBTree(n_components=3, K='ltanh', random_state=0)
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
        # Parent sets self.fun / self.fun_args for all K via FastICA.__init__.
        super().__init__(
            n_components=n_components,
            K=K,
            algorithm=algorithm,
            whiten=whiten,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            whiten_solver=whiten_solver,
            random_state=random_state,
        )
        self.threshold = threshold

        # Override the subsampled contrast only for bounded kernels.
        if K in self._BTREE_KERNELS:
            self.fun      = _btree_contrast
            self.fun_args = dict(G=self._BTREE_KERNELS[K], threshold=threshold)
