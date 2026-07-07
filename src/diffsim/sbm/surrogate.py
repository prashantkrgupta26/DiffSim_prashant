"""SBM surrogate-boundary pipeline, steps 2-4 of the per-epoch geometry
pipeline (spec S4.2): lambda-criterion element classification, surrogate-face
extraction, and the frozen FP64 geometry cache (d, n, n_bar.n) at surrogate
Gauss points.

Domain-side flag: the computational domain is Omega = {psi < 0} for
domain="inside" (M1a default: physics inside the object) or {psi > 0} for
domain="outside" (M1b: flow around it). Classification counts Gauss points on
the domain side; the cached normal n = s * grad_psi/|grad_psi| with
s = +1/-1 always points OUT of Omega, so corr = n_tilde . n > 0 for a
well-oriented surrogate.

M1a invariant (spec S2.3 cancellation-node rule; general nonconforming
facets deferred to M4): every surrogate face is a WHOLE element face.
Extraction probes the 2^(dim-1) sub-face quadrants just outside each face:
all missing => surrogate face; all present => interior; mixed => the face is
partially exposed and extraction raises ValueError.

Epoch contract (spec S2.2.1): SurrogateFaces and GeometryData are immutable
between adaptation events; GeometryData is produced by ONE batched oracle
call and is always FP64.
"""
from dataclasses import dataclass
from itertools import product as iproduct
import numpy as np

from ..octree import morton
from ..octree.build import Octree
from ..octree.lookup import LeafLookup, face_offsets, face_neighbors


def _domain_sign(domain: str) -> float:
    if domain == "inside":
        return -1.0     # Omega = {psi < 0}
    if domain == "outside":
        return 1.0      # Omega = {psi > 0}
    raise ValueError(f"domain must be 'inside' or 'outside', got {domain!r}")


def classify_lambda(tree: Octree, oracle, lam: float, domain: str = "inside",
                    n1: int = 5, lipschitz_bound: float = 2.0):
    """Element classification with the PRODUCTION lambda convention
    (RatioGPSBM, Dendrite/FlowBench SBMMarker): an INTERCEPTED element is
    retained iff its domain-OUTSIDE volume fraction (1 - frac_in) <= lam.
    So lam = 1.0 retains ALL intercepted elements (production Poisson/thermal
    default; surrogate hugs Gamma from outside, Omega~ superset of Omega);
    lam = 0.5 keeps majority-inside elements (flow-past-cylinder value);
    lam = 0.0 keeps only fully-interior elements (Omega~ strictly inside).
    Fully-interior elements are always retained; fully-outside always dropped.

    The fraction is a quadrature estimate: weighted indicator sum on an
    n1^dim tensor Gauss-Legendre lattice (n1 >= 2, arbitrary order via
    leggauss) — NOT a raw point count, so frac converges to the true volume
    fraction as n1 grows.

    Two-pass narrowband: an element whose center value satisfies
    |psi(c)| > lipschitz_bound * (sqrt(dim)/2) * h cannot be cut by the zero
    set (valid for |grad psi| <= lipschitz_bound; SDF-family backends are
    ~1-Lipschitz, the default 2.0 is the safety margin) and is decided by the
    center sign alone. Only the remaining cut band pays the dense rule.
    Pass lipschitz_bound=np.inf to force dense sampling everywhere (the
    exactness test hook).

    Returns (retained Octree, frac[Nret]).
    """
    sgn = _domain_sign(domain)
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    if n1 < 2:
        raise ValueError(f"n1 must be >= 2, got {n1}")
    dim = tree.dim
    h = tree.h()

    # Pass 1: decide clearly-in/clearly-out elements from the center value.
    psi_c = oracle.classify(tree.centers())
    radius = lipschitz_bound * (np.sqrt(dim) / 2.0) * h
    decided = np.abs(psi_c) > radius
    frac = np.where(sgn * psi_c > 0.0, 1.0, 0.0)

    # Pass 2: dense weighted quadrature on the cut band only.
    band = np.where(~decided)[0]
    if len(band):
        pts, wts = np.polynomial.legendre.leggauss(n1)
        grids = np.meshgrid(*([(pts + 1.0) / 2.0] * dim), indexing="ij")
        offs = np.stack([g.ravel() for g in grids], axis=1)   # [n1^dim, dim]
        wgrids = np.meshgrid(*([wts] * dim), indexing="ij")
        wq = np.prod(np.stack([g.ravel() for g in wgrids]), axis=0)
        scale = 2.0 ** -morton.lmax(dim)
        lo = tree.anchors()[band] * scale
        hb = h[band]
        xq = (lo[:, None, :] + offs[None, :, :] * hb[:, None, None]).reshape(-1, dim)
        psi = oracle.classify(xq).reshape(len(band), len(offs))
        frac[band] = ((sgn * psi > 0.0) * wq[None, :]).sum(axis=1) / wq.sum()

    # production retention rule: interior always kept, exterior always
    # dropped, intercepted kept iff inactive fraction <= lam
    keep = (frac == 1.0) | ((frac > 0.0) & (1.0 - frac <= lam))
    ret = Octree(tree.keys[keep], tree.levels[keep], dim=dim,
                 periodic=tree.periodic)
    return ret, frac[keep]


@dataclass(frozen=True)
class SurrogateFaces:
    """Surrogate-face list, sorted by (elem, face) for determinism."""
    elem: np.ndarray   # int64 [Nf] index into the retained tree
    face: np.ndarray   # int8  [Nf] face id = 2*ax + side


def extract_surrogate(tree: Octree) -> SurrogateFaces:
    """Faces of retained elements with no retained element across them and
    not on the outer (non-periodic) domain boundary (spec S4.2 step 3).
    Raises ValueError on partially exposed faces (M1a whole-face invariant)."""
    dim = tree.dim
    lk = LeafLookup(tree)
    L = morton.lmax(dim)
    G = 1 << L
    anchors = tree.anchors()
    size = (1 << (L - tree.levels.astype(np.int64)))
    center = anchors + size[:, None] // 2
    offs = face_offsets(dim)
    per_flags = np.array(tree.periodic, bool)
    tang_combos = np.array(list(iproduct((-1, 1), repeat=dim - 1)), np.int64)

    elems, faces = [], []
    for f in range(2 * dim):
        ax = f // 2
        off = offs[f]
        tang_axes = [d for d in range(dim) if d != ax]
        # outer-boundary faces: probe leaves the box on a non-periodic axis
        base = center + off[None, :] * (size[:, None] // 2 + 1)
        outer = (((base < 0) | (base >= G)) & ~per_flags[None, :]).any(axis=1)
        # sub-face probes: shift the tangent axes by +-size/4 around base
        found = np.zeros((len(tree), len(tang_combos)), bool)
        for ci, combo in enumerate(tang_combos):
            probe = base.copy()
            for j, d in enumerate(tang_axes):
                probe[:, d] += combo[j] * (size // 4)
            found[:, ci] = lk.find(probe) >= 0
        n_found = found.sum(axis=1)
        partial = (~outer) & (n_found > 0) & (n_found < len(tang_combos))
        if partial.any():
            e = int(np.where(partial)[0][0])
            raise ValueError(
                f"partially exposed surrogate face: element {e}, face {f} "
                f"({n_found[e]}/{len(tang_combos)} sub-probes found a retained "
                "element); refine the narrowband uniformly (M1a invariant, "
                "spec S2.3 cancellation rule)")
        exposed = (~outer) & (n_found == 0)
        idx = np.where(exposed)[0]
        elems.append(idx)
        faces.append(np.full(len(idx), f, np.int8))

    elem = np.concatenate(elems)
    face = np.concatenate(faces)
    order = np.lexsort((face, elem))
    return SurrogateFaces(elem[order], face[order])


def face_gauss_points(tree: Octree, sf: SurrogateFaces, ftab) -> np.ndarray:
    """Physical coords of face Gauss points, flat [Nf*nqf, dim] in (face, q)
    order — kernels index fi*nqf + q."""
    dim = tree.dim
    scale = 2.0 ** -morton.lmax(dim)
    lo = tree.anchors()[sf.elem] * scale
    h = tree.h()[sf.elem]
    ref = ftab.xi[sf.face]                          # [Nf, nqf, dim]
    xq = lo[:, None, :] + (ref + 1.0) * 0.5 * h[:, None, None]
    return xq.reshape(-1, dim)


def p2_band(tree: Octree, sf: SurrogateFaces, n_layers: int = 2) -> np.ndarray:
    """Per-element polynomial-order marker for the SBM Neumann band (the
    local-p-refinement draft's Eq. 7): p = 2 on every surrogate-face element
    plus (n_layers - 1) NODE-adjacent rings (diagonal-inclusive — the
    draft's 'cells touching' semantics), p = 1 elsewhere. Growth from the
    actual face set guarantees the hard rule (every shifted-Neumann
    quadrature point inside p2 cells) for any n_layers >= 1.

    MEASURED (2026-07-05 overnight): ring growth by FACE neighbors is too
    thin at diagonals — the surrogate-face elements' discrete Hessians get
    pinched by the minimum-rule p1 trace constraints and the Neumann shift
    caps at L2 order ~1 (band1 0.92, band2 1.04); a thick band restores
    clean order 2 (face-band4: 1.99/2.02). Node adjacency reaches the same
    quality at smaller n_layers."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    from ..mesh.nodes import build_mesh
    p_elem = np.ones(len(tree), np.int8)
    band = set(int(e) for e in np.unique(sf.elem))
    if n_layers > 1:
        m1 = build_mesh(tree, p=1)
        conn = m1.conn
        node_elems = [[] for _ in range(len(m1.node_coords))]
        for e in range(len(tree)):
            for nd in conn[e]:
                node_elems[int(nd)].append(e)
        cur = set(band)
        for _ in range(n_layers - 1):
            nxt = set()
            for e in cur:
                for nd in conn[e]:
                    nxt.update(node_elems[int(nd)])
            cur = nxt - band
            band |= nxt
    p_elem[sorted(band)] = 2
    return p_elem


@dataclass(frozen=True)
class GeometryData:
    """Per-epoch geometry cache at surrogate-face Gauss points (spec S4.2
    step 4). All FP64, produced by one batched oracle call, frozen for the
    epoch. n points out of the computational domain; corr = n_tilde . n."""
    xq: np.ndarray     # [Nf*nqf, dim]
    d: np.ndarray      # [Nf*nqf, dim] surrogate GP -> closest point on Gamma
    n: np.ndarray      # [Nf*nqf, dim] true-boundary outward normal (out of Omega)
    corr: np.ndarray   # [Nf*nqf] area-correction factor n_tilde . n
    ok: np.ndarray     # [Nf*nqf] projection convergence mask (all True)
    domain: str

    @classmethod
    def evaluate(cls, oracle, tree: Octree, sf: SurrogateFaces, ftab,
                 domain: str = "inside",
                 warm_feet: np.ndarray = None,
                 max_fail_frac: float = 0.0) -> "GeometryData":
        sgn = -_domain_sign(domain)      # out-of-domain: +grad for "inside"
        xq = face_gauss_points(tree, sf, ftab)
        d, n_grad, ok = oracle.distance_vector(
            xq, y0=(xq + warm_feet) if warm_feet is not None else None)
        if not ok.all() and 0.0 < max_fail_frac >= (~ok).mean():
            # isolated marginal GPs (<< 1%): borrow the nearest valid
            # foot on the batch — O(h)-accurate, opt-in (hero runs);
            # strict mode (max_fail_frac=0) raises below.
            bad = np.where(~ok)[0]
            good = np.where(ok)[0]
            for i in bad:
                j = good[np.argmin(((xq[good] - xq[i]) ** 2).sum(1))]
                d[i] = d[j]
                n_grad[i] = n_grad[j]
            ok = np.ones_like(ok)
        if not ok.all():
            from ..geometry.oracle import admissibility
            raise RuntimeError(
                f"closest-point projection failed at {int((~ok).sum())} of "
                f"{len(ok)} surrogate Gauss points; admissibility report: "
                f"{admissibility(oracle, xq)}")
        n = sgn * n_grad
        ntilde = face_offsets(tree.dim).astype(np.float64)[sf.face]
        corr = np.einsum("id,id->i", np.repeat(ntilde, ftab.nqf, axis=0), n)
        return cls(np.ascontiguousarray(xq), np.ascontiguousarray(d),
                   np.ascontiguousarray(n), np.ascontiguousarray(corr),
                   ok, domain)
