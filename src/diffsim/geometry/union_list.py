"""UnionList N-ary SDF union oracle (spec S4.1).

UnionList(oracles: list[SDFOracle]) implements the full SDFOracle protocol
with EXACT min over N member oracles:

    psi / classify  = elementwise min over member psi values
    distance_vector = routes each point to its argmin-|psi| member
                      (the member whose surface is closest) and returns
                      that member's (d, n_grad, ok)
    params          = concatenation of all member params (no dedup)
    near_eikonal    = False unconditionally (exact min has ridge sets at the
                      medial axis — see csg.py invariant and design note below)
    dim             = all members must agree; asserted on construction
    velocity        = inherited from SDFOracle base (returns zeros for
                      static geometry — base class covers M1)

Design decisions
----------------
1. File placement: new `union_list.py` rather than extending `csg.py`.
   The existing `Union` in csg.py is a BINARY SMOOTH BLEND (_Blend
   subclass with smoothing parameter k).  UnionList is a fundamentally
   different structure (N-ary, exact min, index-gather routing) and
   adding it to csg.py would blur the smooth-blend vs exact-min
   distinction that the csg module docstring emphasises.  A separate
   file keeps the responsibility boundary clear.

2. Tie-breaking at equidistant points: when |psi_i| == |psi_j| for
   multiple members, `np.argmin` selects the lowest index.  This is
   deterministic, stable, and easy to document in tests.  It matters
   only at the exact equidistance locus (measure zero), and lowest-index
   priority is the natural numpy convention.

3. y0 forwarding: each member's `distance_vector(pts_i, y0=y0_i)` is
   called with the per-member slice of the incoming y0.  If y0 is None
   (the default), None is forwarded.  If y0 is provided, it must be
   [N, dim] and the same slice (pts_i row indices) is passed.  Members
   that ignore y0 (e.g. TriMeshOracle, which uses BVH candidate
   selection) accept the keyword silently.

4. velocity: the SDFOracle base already defines
   `velocity(pts, t) -> zeros_like(pts)` for static geometry.
   UnionList does NOT override it — the truck is static (M1) and the
   briefing says YAGNI for non-zero velocity on the union.

5. Scalar point routing: distance_vector groups points by their argmin
   member index and calls each member once for its batch.  This is O(N)
   in classify queries and avoids N individual member calls.
"""
import numpy as np
import torch

from .oracle import SDFOracle


class UnionList(SDFOracle):
    """N-ary EXACT SDF union: psi = min(psi_0, psi_1, ..., psi_{N-1}).

    Parameters
    ----------
    oracles : list[SDFOracle]
        Non-empty list of SDFOracle instances, all with the same `dim`.

    Raises
    ------
    ValueError
        If the list is empty or member dims are inconsistent.
    """

    def __init__(self, oracles: list):
        if len(oracles) == 0:
            raise ValueError("UnionList requires at least one oracle")
        dims = [o.dim for o in oracles]
        if len(set(dims)) != 1:
            raise ValueError(
                f"All UnionList members must have the same dim; "
                f"got dims={dims}")
        self._oracles = list(oracles)
        self.dim = dims[0]
        # near_eikonal is False unconditionally: the exact min of N SDFs has
        # ridge sets at the medial axis (equidistant surface between members)
        # where |grad psi| drops discontinuously.  The eikonal shortcut
        # d = psi * n_grad is invalid near the medial axis, so Newton
        # projection is always required.  This matches the csg.py invariant:
        # "exact min/max are NOT globally eikonal => near_eikonal = False"
        # (csg.py module docstring, lines 14-16).  The _Blend(k=0) exact-min
        # Union class likewise sets near_eikonal = False at the class level.
        self.near_eikonal = False

    @property
    def params(self) -> list:
        """Concatenation of all member params (in member order)."""
        result = []
        for o in self._oracles:
            result.extend(o.params)
        return result

    def psi(self, x: torch.Tensor) -> torch.Tensor:
        """Exact min of member psi values: psi[n] = min_i psi_i(x[n]).

        Parameters
        ----------
        x : torch.Tensor [N, dim], float64
        Returns
        -------
        psi : torch.Tensor [N], float64
        """
        psis = torch.stack([o.psi(x) for o in self._oracles], dim=1)  # [N, M]
        return psis.min(dim=1).values                                   # [N]

    # --- numpy bridge overrides -------------------------------------------

    def classify(self, pts: np.ndarray) -> np.ndarray:
        """Elementwise min over member classify values (numpy, FP64)."""
        pts = np.ascontiguousarray(pts, np.float64)
        psis = np.stack([o.classify(pts) for o in self._oracles], axis=1)  # [N, M]
        return psis.min(axis=1)                                            # [N]

    def distance_vector(self, pts: np.ndarray, y0: np.ndarray = None):
        """Route each point to its argmin-|psi| member; return that member's
        (d, n_grad, ok).

        Routing rule: for each point n, the member i with the smallest |psi_i(x_n)|
        is selected (i.e. the member whose zero-set is closest).  Ties broken by
        lowest index (numpy argmin convention).

        Parameters
        ----------
        pts : np.ndarray [N, dim], float64
        y0  : np.ndarray [N, dim] or None
            Optional warm-start foot points for Newton projection.  Forwarded
            per-member for the member's point subset.  Members that ignore y0
            (e.g. TriMeshOracle) accept the keyword silently.

        Returns
        -------
        d    : np.ndarray [N, dim]  displacement foot - x
        n    : np.ndarray [N, dim]  unit normal (grad psi direction)
        ok   : np.ndarray [N] bool  convergence mask
        """
        pts = np.ascontiguousarray(pts, np.float64)
        N = len(pts)

        # Step 1: evaluate |psi| for all members to determine routing
        psi_mat = np.stack([o.classify(pts) for o in self._oracles],
                           axis=1)          # [N, M]
        idx = np.argmin(np.abs(psi_mat), axis=1)   # [N]  argmin-|psi| per point

        # Step 2: allocate output arrays
        d_out = np.empty((N, self.dim), dtype=np.float64)
        n_out = np.empty((N, self.dim), dtype=np.float64)
        ok_out = np.empty(N, dtype=bool)

        # Step 3: batch call each member for its assigned points
        for mi, oracle in enumerate(self._oracles):
            mask = (idx == mi)
            if not mask.any():
                continue
            pts_mi = pts[mask]                         # [Ni, dim]
            y0_mi = y0[mask] if y0 is not None else None
            d_mi, n_mi, ok_mi = oracle.distance_vector(pts_mi, y0=y0_mi)
            d_out[mask] = d_mi
            n_out[mask] = n_mi
            ok_out[mask] = ok_mi

        return d_out, n_out, ok_out
