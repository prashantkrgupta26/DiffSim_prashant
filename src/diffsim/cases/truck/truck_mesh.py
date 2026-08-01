"""TRUCK case mesh construction (incomplete-octree: slab carve + refine + truck carve).

Moved verbatim from tests/truck_flow.py.  Functions: _channel_box, slab_carve,
refine_region_boxes, refine_truck_band, refine_walls, refine_ground,
_face_components, flood_fill_retain, build_truck_mesh.
"""
import numpy as np
import scipy.sparse as sp

from diffsim.octree.build import build_uniform, refine_elements, Octree
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.mesh.faces import face_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.geometry.csg import Box
from diffsim.geometry.merged_trimesh import MergedTriMesh
from diffsim.sbm.surrogate import classify_lambda, extract_surrogate, GeometryData


def _channel_box(domain_min, domain_max, scale):
    """Box oracle for the channel slab in unit-cube coords.  psi<0 inside."""
    lo = np.asarray(domain_min, np.float64) * scale
    hi = np.asarray(domain_max, np.float64) * scale
    center = tuple((lo + hi) / 2.0)
    half = tuple((hi - lo) / 2.0)
    return Box(center, half)


def slab_carve(tree, channel_box):
    """Retain only cells inside the channel slab.  Returns (ret, n_intercepted).

    Dyadic slab bounds => NO cell is cut: we assert every retained cell is
    FULLY inside (frac == 1.0) and no cell is partially intercepted.  This is
    the dyadic-exactness guarantee (zero intercepted slab cells).
    """
    # lipschitz_bound=inf forces dense sampling so a straddling cell WOULD show
    # a fractional frac; dyadic bounds guarantee it does not.
    ret, frac = classify_lambda(tree, channel_box, lam=0.0, domain="inside",
                                lipschitz_bound=np.inf)
    # With lam=0.0, only fully-inside cells (frac==1) are retained; any cut cell
    # (0<frac<1) is DROPPED.  Dyadic exactness => the set of cut cells is empty,
    # which we verify by re-classifying with lam=1.0 (retains cut cells too) and
    # asserting the two retained counts are identical (no cut cells exist).
    ret_all, frac_all = classify_lambda(tree, channel_box, lam=1.0,
                                        domain="inside", lipschitz_bound=np.inf)
    n_cut = len(ret_all) - len(ret)
    assert n_cut == 0, (
        f"slab carve NOT dyadic-exact: {n_cut} intercepted (cut) slab cells; "
        f"channel bounds must land on cell boundaries.")
    assert np.all(frac == 1.0), "retained slab cells must be fully interior"
    return ret, n_cut


def refine_region_boxes(tree, regions, scale):
    """Refine cells whose center is inside each region box, up to its level."""
    for r in regions:
        lo = np.asarray(r.min_c, np.float64) * scale
        hi = np.asarray(r.max_c, np.float64) * scale
        target = int(r.refine_region_lvl)
        # iterate levels: refine cells inside the box below target level
        for _ in range(64):
            centers = tree.centers()
            lvl = tree.levels.astype(np.int64)
            inside = np.all((centers >= lo) & (centers <= hi), axis=1)
            mask = inside & (lvl < target)
            if not mask.any():
                break
            tree = refine_elements(tree, mask)
            tree = balance2to1(tree)
    return tree


def refine_truck_band(tree, merged, band_cells, refine_to):
    """Refine cells within band_cells*h of the merged truck surface, to
    refine_to.  Mirrors the C++ 'distance < RefineElementNumber x (h)' band."""
    for _ in range(64):
        centers = tree.centers()
        lvl = tree.levels.astype(np.int64)
        h = tree.h()
        cand = lvl < refine_to
        if not cand.any():
            break
        dist = np.abs(merged.classify(centers))
        mask = cand & (dist < band_cells * h)
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    return tree



def refine_walls(tree, target_lvl, band, y_max, z_max):
    """Refine cells within ``band`` of ANY channel wall (ground y=0, ceiling
    y=y_max, side walls z=0/z_max) to ``target_lvl`` — the C++
    refine_walls=true, taken fully.  The g-leg located the post-fix
    disturbance resonator in base-level cells near the ceiling above the
    truck (umax doubling at (0.34, 0.109, 0.109)): the displaced startup
    wave is under-resolved at the outer walls just as the shear layer was
    at the ground."""
    for _ in range(64):
        centers = tree.centers()
        lvl = tree.levels.astype(np.int64)
        near = ((centers[:, 1] < band) | (centers[:, 1] > y_max - band)
                | (centers[:, 2] < band) | (centers[:, 2] > z_max - band))
        mask = near & (lvl < target_lvl)
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    return tree


def refine_ground(tree, target_lvl, height):
    """Refine cells whose center lies within ``height`` of the ground (y=0)
    up to ``target_lvl`` — the C++ refine_walls heritage (Baskar: refine near
    the floor).  Resolves the near-ground shear layer (the sloped-inlet
    profile rises over y<0.0156 = TWO base-level cells unrefined — the
    u-probe wave nursery)."""
    for _ in range(64):
        centers = tree.centers()
        lvl = tree.levels.astype(np.int64)
        mask = (centers[:, 1] < height) & (lvl < target_lvl)
        if not mask.any():
            break
        tree = refine_elements(tree, mask)
        tree = balance2to1(tree)
    return tree


def _face_components(tree):
    """Cell connected components under FACE adjacency (Baskar: two cells are
    the same fluid domain only if they SHARE A FACE — corner/edge contact is
    not a flow passage).  Mirrors extract_surrogate's neighbor probe: for
    each face, sub-face quarter probes catch finer neighbors; the reverse
    direction catches coarser ones."""
    from itertools import product as _iproduct
    from diffsim.octree import morton as _morton
    from diffsim.octree.lookup import LeafLookup as _LeafLookup, \
        face_offsets as _face_offsets
    dim = tree.dim
    lk = _LeafLookup(tree)
    L = _morton.lmax(dim)
    anchors = tree.anchors()
    size = (1 << (L - tree.levels.astype(np.int64)))
    center = anchors + size[:, None] // 2
    offs = _face_offsets(dim)
    tang = np.array(list(_iproduct((-1, 1), repeat=dim - 1)), np.int64)
    n = len(tree)
    src, dst = [], []
    for f in range(2 * dim):
        ax = f // 2
        off = offs[f]
        tang_axes = [d for d in range(dim) if d != ax]
        base = center + off[None, :] * (size[:, None] // 2 + 1)
        for combo in tang:
            probe = base.copy()
            for j, d in enumerate(tang_axes):
                probe[:, d] += combo[j] * (size // 4)
            nb = lk.find(probe)
            m = nb >= 0
            src.append(np.where(m)[0])
            dst.append(nb[m])
    g = sp.coo_matrix((np.ones(sum(len(x) for x in src)),
                       (np.concatenate(src), np.concatenate(dst))),
                      shape=(n, n))
    ncomp, labels = sp.csgraph.connected_components(g, directed=False)
    return ncomp, labels


def flood_fill_retain(ret):
    """Single-fluid-domain retention (Baskar directive): drop every face-
    connected component except the largest.  The 22-body truck assembly
    encloses internal cavities (engine bay, cab, tank gaps) that the carve
    otherwise retains as isolated "fluid" pockets — each carries its own
    pressure nullspace (only one global pin exists), making the saddle
    singular (the measured underbody instability + solver floor).  Removing
    a pocket exposes no new main-domain faces (no shared face by
    definition), so the surrogate extraction is unaffected."""
    ncomp, labels = _face_components(ret)
    if ncomp <= 1:
        return ret, 0, 0
    sizes = np.bincount(labels)
    # main component by KNOWN-FLUID SEED, not cell count (final-review I-6:
    # a heavily-refined enclosed cavity could out-count the coarse outer
    # domain).  The inlet lower corner cell (min x+y+z center) is always in
    # the channel fluid.
    _cen = ret.centers()
    _seed = int(np.argmin(_cen.sum(axis=1)))
    main = int(labels[_seed])
    keepm = labels == main
    n_dropped = int((~keepm).sum())
    print(f"[truck] flood-fill: dropped {n_dropped} cells in {ncomp - 1} "
          f"enclosed pocket(s) (face-adjacency); single fluid domain = "
          f"{int(sizes[main])} cells", flush=True)
    return (Octree(ret.keys[keepm], ret.levels[keepm], dim=ret.dim,
                   periodic=ret.periodic), ncomp - 1, n_dropped)


def build_truck_mesh(cfg, base_level, region_refine=True, truck_band_to=None,
                     band_cells=3, device="cpu", merged=None,
                     bodies=None, carve_lam=1.0,
                     ground_refine_to=None, ground_band=0.0156,
                     walls_refine_to=None, carve_delta=None,
                     seal_underbody=False, seal_y=0.003, seal_boxes=None):
    """Build the incomplete-octree mesh + volumetric surrogate for the truck.

    Returns a dict: dm, mesh, cons, sf, geo, merged, scale, n_excluded,
    n_slab_cut, n_cells.
    """
    scale = cfg.domain_scale
    bodies = cfg.bodies if bodies is None else bodies
    if merged is None:
        merged = MergedTriMesh.from_bodies(bodies, cfg.config_dir, scale=scale,
                                           device=device)

    tree = build_uniform(base_level, dim=3)

    # 1. slab carve (dyadic-exact)
    cbox = _channel_box(cfg.domain_min, cfg.domain_max, scale)
    tree, n_slab_cut = slab_carve(tree, cbox)

    # 2. region refine
    if region_refine and cfg.region_refine:
        tree = refine_region_boxes(tree, cfg.region_refine, scale)

    # 2a-walls. all-walls refine (C++ refine_walls=true, in full).
    # walls_refine_to: int (single band = ground_band) or list of
    # (lvl, band) pairs — NESTED banding so the growing Re-hold boundary
    # layer (delta = sqrt(nu t)) never crosses a stacked transition: the
    # step-locked hard-solve window at steps ~35-50 across g/h/i legs is
    # the layer's shear edge straddling the band-edge interface.
    if walls_refine_to is not None:
        ymax = float(cfg.domain_max[1]) * scale
        zmax = float(cfg.domain_max[2]) * scale
        _wspec = (walls_refine_to
                  if isinstance(walls_refine_to, (list, tuple))
                  else [(int(walls_refine_to), float(ground_band))])
        for _wl, _wb in _wspec:
            n0 = len(tree)
            tree = refine_walls(tree, int(_wl), float(_wb), ymax, zmax)
            print(f"[truck] walls refine: lvl>={_wl} within {_wb} "
                  f"({n0} -> {len(tree)} cells)", flush=True)

    # 2b. ground refine (C++ refine_walls; Baskar directive)
    if ground_refine_to is not None:
        n0 = len(tree)
        tree = refine_ground(tree, int(ground_refine_to), float(ground_band))
        print(f"[truck] ground refine: lvl>={ground_refine_to} within "
              f"y<{ground_band} ({n0} -> {len(tree)} cells)", flush=True)

    # 3. truck-band refine
    if truck_band_to is not None:
        tree = refine_truck_band(tree, merged, band_cells, truck_band_to)

    n_before = len(tree)

    # 4a. distance-threshold carve (Baskar fragmentation finding): the cab
    # interior and sub-resolution front details (mirrors, grille slits,
    # cab-trailer gap ~1-3 cells at lvl 12) fragment the lam-carve into
    # salt-and-pepper retained specks — the measured chaos nursery at the
    # truck front.  carve_delta > 0 retains only cells whose CENTER is at
    # least delta*h_cell OUTSIDE the surface: thin features and interior
    # slits are absorbed into the solid (a one-sided morphological closing),
    # the staircase smooths, and flood-fill sweeps what gets sealed.
    if carve_delta is not None:
        centers = tree.centers()
        hcell = tree.h()
        psi = merged.classify(centers)
        # threshold in ABSOLUTE length (units of the FINEST band cell, not
        # per-cell h): a per-cell threshold carves fine cells deeper than
        # coarse neighbors at 2:1 interfaces -> partially exposed surrogate
        # faces (M1a violation, hit at band 13).  h_ref = min cell size.
        _h_ref = float(hcell.min())
        keepm = psi > float(carve_delta) * _h_ref
        from diffsim.octree.build import Octree as _Oc
        ret = _Oc(tree.keys[keepm], tree.levels[keepm], dim=3,
                  periodic=tree.periodic)
        print(f"[truck] delta-carve: kept {int(keepm.sum())}/{len(tree)} "
              f"(delta={carve_delta}h)", flush=True)
        _frac = None
    else:
        ret = None

    # 4. truck carve (flow AROUND the solid: domain="outside").  carve_lam
    # picks the intercepted-cell convention: 1.0 KEEPS cut cells (surrogate
    # hugs Gamma from outside — RatioGPSBM default); 0.0 REMOVES them (the
    # ThinShell paper's T~h := {T : T cap Gamma = 0} and the C++ carve).
    # u-probe forensics: with the truck touching the ground (position clips
    # y to 0.000), lam=1.0 retains PINCHED SLIVER cells along the whole
    # contact line — mostly-inside-solid cells squeezed between strong truck
    # dofs and strong ground dofs — the measured epicenter of the underbody
    # instability.  lam=0.0 removes them (paper-faithful).
    if ret is None:
        ret, _frac = classify_lambda(tree, merged, lam=float(carve_lam),
                                     domain="outside")

    # 4b. underbody seal (Baskar directive): excise the ground-clearance
    # gap under the truck footprint (an implicit skirt — sealed-underbody
    # wind-tunnel practice).  The gap flow at band-12 resolution is the
    # campaign's last instability wall (jet chaos beyond Re~1000-1500 per
    # the k/l legs); sealing removes it for bring-up; the resolved-
    # underbody physics returns with the band-13 mesh.
    if seal_underbody:
        _v = merged.verts.numpy()
        _x0, _x1 = float(_v[:, 0].min()), float(_v[:, 0].max())
        _z0, _z1 = float(_v[:, 2].min()), float(_v[:, 2].max())
        _c = ret.centers()
        _m = ((_c[:, 0] > _x0) & (_c[:, 0] < _x1)
              & (_c[:, 2] > _z0) & (_c[:, 2] < _z1)
              & (_c[:, 1] < float(seal_y)))
        n_seal = int(_m.sum())
        ret = Octree(ret.keys[~_m], ret.levels[~_m], dim=ret.dim,
                     periodic=ret.periodic)
        print(f"[truck] underbody seal: excised {n_seal} cells under "
              f"footprint x[{_x0:.4f},{_x1:.4f}] z[{_z0:.4f},{_z1:.4f}] "
              f"y<{seal_y}", flush=True)

    # 4c. box seals (gap fairings): excise cells inside arbitrary boxes —
    # e.g. the cab-trailer slot (a real aero device: gap fairing).  Each box
    # = (x0, x1, y0, y1, z0, z1) in unit coords.
    if seal_boxes:
        _c2 = ret.centers()
        for (bx0, bx1, by0, by1, bz0, bz1) in seal_boxes:
            _mb = ((_c2[:, 0] > bx0) & (_c2[:, 0] < bx1)
                   & (_c2[:, 1] > by0) & (_c2[:, 1] < by1)
                   & (_c2[:, 2] > bz0) & (_c2[:, 2] < bz1))
            print(f"[truck] box seal: excised {int(_mb.sum())} cells in "
                  f"[{bx0},{bx1}]x[{by0},{by1}]x[{bz0},{bz1}]", flush=True)
            ret = Octree(ret.keys[~_mb], ret.levels[~_mb], dim=ret.dim,
                         periodic=ret.periodic)
            _c2 = ret.centers()

    # 5. flood-fill: single fluid domain (face-adjacency components)
    ret, n_pockets, n_pocket_cells = flood_fill_retain(ret)
    n_excluded = n_before - len(ret)

    sf = extract_surrogate(ret)
    mesh = build_mesh(ret, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)
    geo = GeometryData.evaluate(merged, ret, sf, face_tables(1, 3),
                                domain="outside")
    return dict(dm=dm, mesh=mesh, cons=cons, sf=sf, geo=geo, merged=merged,
                tree=ret,
                scale=scale, n_excluded=int(n_excluded), n_slab_cut=int(n_slab_cut),
                n_cells=len(ret), n_pockets=int(n_pockets),
                n_pocket_cells=int(n_pocket_cells))
