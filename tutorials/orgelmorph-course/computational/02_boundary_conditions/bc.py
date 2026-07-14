"""OrgElMorph course - Computational C2: boundary conditions, numerically.

Importable core for the boundary-condition concept.  A boundary
condition is baked into the weak form, and the *way* it is imposed
decides what the simulation conserves.  This module runs the SAME binary
spinodal blend under the two boundary treatments the Cahn-Hilliard brick
supports, and measures what each one does to (a) mass conservation and
(b) the morphology near the wall.

  NATURAL (no-flux).  The weak form of the conserved dynamics
      Int w c_t + M Int grad w . grad mu = boundary flux
  drops its boundary term when grad mu . n = 0 on dOmega.  That is the
  no-flux / zero-Neumann condition, and it costs NOTHING to impose: you
  simply do not add a boundary integral.  Because no mass crosses the
  wall, total mass is conserved to solver tolerance -- the physics of a
  sealed box.  Every spinodal run in Physics P1 is a no-flux run.

  DIRICHLET (fixed value).  To hold the boundary composition at a
  prescribed value we REPLACE the boundary rows of the linear system
  with the identity (row i -> [.. 1 ..], rhs -> g - x_i), pinning
  (c, mu) at those nodes.  This deliberately breaks conservation: the
  pinned wall acts as a reservoir that lets material flow in or out.
  It is the treatment the manufactured-solution study (C1) needed, and
  it models a composition held fixed at a contact (an electrode, a
  substrate held at a set surface fraction).

A third kind -- a WALL FREE ENERGY (a surface term g(c) on dOmega that
makes the natural flux nonzero, i.e. a Robin-type wetting condition) --
is a genuine physical boundary condition for thin films, but it lives in
the multiphase film brick, not this binary CH brick; see the README and
the course text for where it is implemented (wodo_film / the multiphase
A2 wall term).

Everything below drives the production brick
src/diffsim/physics/cahn_hilliard.py; the boundary machinery
(dirichlet=..., gc_fn/gm_fn, and the natural default) is the brick's,
unchanged.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper

M, KAP, DT = 1.0, 5e-4, 0.02


def build_dm(level=5, p=1, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def boundary_free_nodes(mesh, cons):
    """Indices (into the free-node ordering) of the box-edge nodes."""
    coords = mesh.node_coords[cons.free_nodes]
    on = np.zeros(len(coords), bool)
    for cc in range(2):
        on |= (np.abs(coords[:, cc]) < 1e-12) | \
              (np.abs(coords[:, cc] - 1.0) < 1e-12)
    return np.where(on)[0]


def _mass(st, cf):
    """Total mass Int c dV by the solver's own Gauss quadrature."""
    v, _ = st._gp_scalar(cf)
    m = 0.0
    for pv, b in st.dm.bins.items():
        h = st.mesh.tree.h()[st.mesh.bins[pv]]
        ne = len(st.mesh.conn_of[pv])
        wq = np.tile(st.dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** 2, b["nqp"])
        m += float((wq * v[pv]).sum())
    return m


def run_bc(mode, wall=0.9, level=5, steps=60, seed=3, device="cuda:0"):
    """March a binary spinodal blend under one boundary treatment and
    record the mass history and the final field.

    mode="noflux"    : natural (zero-flux) boundaries -- the default; no
                       boundary term, mass conserved.
    mode="dirichlet" : pin (c, mu) = (wall, 0) on every box edge -- a
                       reservoir wall drawing composition to `wall`.

    The initial blend uses a FIXED seed so the two modes start from the
    identical field and any difference is the boundary's doing.  Returns
    a dict with the mass series, the final field image, and diagnostics.
    """
    dm, mesh, cons = build_dm(level, 1, device)
    if mode == "dirichlet":
        bidx = boundary_free_nodes(mesh, cons)
        st = CahnHilliardStepper(
            dm, M, KAP, DT, order=1, dirichlet=bidx,
            gc_fn=lambda x, t: np.full(len(x), wall),
            gm_fn=lambda x, t: np.zeros(len(x)))
    elif mode == "noflux":
        st = CahnHilliardStepper(dm, M, KAP, DT, order=1)
    else:
        raise ValueError(mode)

    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")

    side = int(round(np.sqrt(len(mesh.node_coords))))
    full = lambda cf: np.asarray(cons.T @ cf).reshape(side, side)
    m0 = _mass(st, st.hist[0])
    masses = [m0]
    for _ in range(steps):
        c, _ = st.step()
        masses.append(_mass(st, c))
    masses = np.asarray(masses)
    img = full(c)
    # mean composition on the four edges of the image (the boundary
    # layer): pinned to `wall` under Dirichlet, free under no-flux.
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return dict(mode=mode, wall=wall, side=side, img=img,
                masses=masses, mass_drift=float(abs(masses[-1] - m0)),
                edge_mean=float(edge.mean()), c_field=full(c),
                c_min=float(c.min()), c_max=float(c.max()),
                steps=steps, level=level)


def compare(device="cuda:0", **kw):
    """Run both boundary treatments on the identical IC and return the
    pair plus the field difference (the boundary's measured effect)."""
    nf = run_bc("noflux", device=device, **kw)
    di = run_bc("dirichlet", device=device, **kw)
    diff = float(np.abs(nf["c_field"] - di["c_field"]).max())
    return dict(noflux=nf, dirichlet=di, field_diff=diff)
