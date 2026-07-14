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


# =====================================================================
# Flux balance: d/dt Int c = -Int_dOmega J.n,  with J = -M grad mu
# ---------------------------------------------------------------------
# Mass conservation is not a slogan -- it is a measurable identity.  For
# the conserved dynamics c_t = M div(grad mu), integrating over the whole
# domain and using the divergence theorem gives
#     d/dt Int c dV = M Int_dOmega (grad mu . n) dS = -Int_dOmega J.n dS,
# i.e. the mass changes at EXACTLY the rate material crosses the wall.
# In the discrete conserved scheme the NET boundary flux and dm/dt are the
# SAME quantity by construction (the discrete divergence theorem is exact),
# so we measure the net flux directly as dm/dt from the quadrature-mass
# series -- NOT from a nodal finite difference of grad mu, which is only a
# weak (integrated) statement and does not vanish pointwise even under the
# natural no-flux BC.  The physics we then read off:
#   * NO-FLUX: the natural BC sets the boundary integral to zero, so
#     dm/dt = 0 to machine precision (mass exactly conserved).
#   * DIRICHLET: the pinned wall is a reservoir; -dm/dt is its influx,
#     which is large early and DECAYS to zero as the interior equilibrates
#     to the wall value (the reservoir stops pumping).  Watching the flux
#     shut off is the measurable signature of the balance.

def flux_balance(mode, wall=0.9, level=5, steps=60, seed=3, device="cuda:0"):
    """March one boundary treatment recording the quadrature mass each
    step; return the mass series and dm/dt = the NET boundary flux
    (exact discrete divergence theorem)."""
    dm, mesh, cons = build_dm(level, 1, device)
    if mode == "dirichlet":
        bidx = boundary_free_nodes(mesh, cons)
        st = CahnHilliardStepper(
            dm, M, KAP, DT, order=1, dirichlet=bidx,
            gc_fn=lambda x, t: np.full(len(x), wall),
            gm_fn=lambda x, t: np.zeros(len(x)))
    else:
        st = CahnHilliardStepper(dm, M, KAP, DT, order=1)
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    masses = [_mass(st, st.hist[0])]
    for _ in range(steps):
        c, _ = st.step()
        masses.append(_mass(st, c))
    masses = np.asarray(masses)
    dmdt = np.diff(masses) / DT                     # = net boundary flux
    return dict(mode=mode, t=np.arange(1, steps + 1) * DT, masses=masses,
                dmdt=dmdt, flux_abs_max=float(np.abs(dmdt).max()),
                flux_early=float(dmdt[:5].mean()),
                flux_late=float(dmdt[-5:].mean()))


# =====================================================================
# BC test matrix -- one contract, several boundary configurations
# ---------------------------------------------------------------------
# The taxonomy of boundary conditions the binary brick can express, run
# on the identical seeded blend so the numbers are comparable.

def _run_generic(bc_spec, wall=0.9, level=5, steps=60, seed=3,
                 device="cuda:0"):
    """Generic runner: bc_spec picks which free-node sets are pinned (in
    c and/or mu) and to what value.  Unpinned edges are natural (no-flux).
    """
    dm, mesh, cons = build_dm(level, 1, device)
    coords = mesh.node_coords[cons.free_nodes]
    sel = bc_spec["select"](coords)               # bool over free nodes
    idx = np.where(sel)[0]
    kw = {}
    if len(idx) and bc_spec.get("pin", True):
        kw = dict(dirichlet=idx,
                  gc_fn=lambda x, t: np.full(len(x), bc_spec.get("gc", wall)),
                  gm_fn=lambda x, t: np.full(len(x), bc_spec.get("gm", 0.0)))
    st = CahnHilliardStepper(dm, M, KAP, DT, order=1, **kw)
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    m0 = _mass(st, st.hist[0])
    for _ in range(steps):
        c, _ = st.step()
    side = int(round(np.sqrt(len(mesh.node_coords))))
    img = np.asarray(cons.T @ c).reshape(side, side)
    edge = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]])
    return dict(name=bc_spec["name"], mass_drift=float(abs(_mass(st, c) - m0)),
                edge_mean=float(edge.mean()), npin=int(len(idx)),
                c_range=(float(c.min()), float(c.max())))


def _all_edges(c):
    on = np.zeros(len(c), bool)
    for k in range(2):
        on |= (np.abs(c[:, k]) < 1e-12) | (np.abs(c[:, k] - 1.0) < 1e-12)
    return on


def _lr_edges(c):
    return (np.abs(c[:, 0]) < 1e-12) | (np.abs(c[:, 0] - 1.0) < 1e-12)


def bc_test_matrix(device="cuda:0", steps=60):
    """Run the boundary-condition taxonomy on one blend and tabulate the
    conservation / pinning signature of each -- the BC test matrix."""
    specs = [
        dict(name="natural (no-flux)", select=_all_edges, pin=False),
        dict(name="prescribed c=+0.9 (all edges)", select=_all_edges,
             gc=0.9, gm=0.0),
        dict(name="prescribed c=-0.9 (all edges)", select=_all_edges,
             gc=-0.9, gm=0.0),
        dict(name="prescribed c=0.0 (all edges)", select=_all_edges,
             gc=0.0, gm=0.0),
        dict(name="mixed: c=+0.9 on L/R, no-flux T/B", select=_lr_edges,
             gc=0.9, gm=0.0),
    ]
    return [_run_generic(s, device=device, steps=steps) for s in specs]


# =====================================================================
# Weak vs strong Dirichlet, on a tiny 1-D FE system (pure numpy)
# ---------------------------------------------------------------------
# HOW a Dirichlet value is imposed is a linear-algebra decision with
# consequences for symmetry and the Jacobian.  On a tiny -u'' = 0 problem
# (exact solution the straight line u = a + (b-a)x) we contrast:
#   (1) row replacement  -- overwrite row i with e_i; ASYMMETRIC.
#   (2) symmetric elimination -- also zero column i into the RHS; SPD kept.
#   (3) weak / penalty   -- add beta to the diagonal and beta*g to the RHS;
#        SPD kept, value imposed to O(1/beta).  As beta -> inf it matches
#        the strong value; this is the Nitsche/penalty family of WEAKLY
#        imposed boundary conditions.

def fe1d_stiffness(n):
    """1-D P1 stiffness for -u'' on [0,1], n elements (tridiagonal)."""
    h = 1.0 / n
    K = np.zeros((n + 1, n + 1))
    for e in range(n):
        ke = np.array([[1.0, -1.0], [-1.0, 1.0]]) / h
        K[e:e + 2, e:e + 2] += ke
    return K, h


def dirichlet_strong(K, b, i, g, symmetric=False):
    """Impose u_i = g.  Row replacement, optionally symmetric elimination."""
    A = K.copy(); r = b.copy().astype(float)
    if symmetric:
        r -= A[:, i] * g          # move the known column to the RHS
        A[i, :] = 0.0; A[:, i] = 0.0; A[i, i] = 1.0; r[i] = g
    else:
        A[i, :] = 0.0; A[i, i] = 1.0; r[i] = g
    return A, r


def dirichlet_penalty(K, b, i, g, beta):
    """Weakly impose u_i ~ g via a penalty: A_ii += beta, b_i += beta*g.
    Symmetric-positive-definite structure preserved (unlike row-replace)."""
    A = K.copy(); r = b.copy().astype(float)
    A[i, i] += beta; r[i] += beta * g
    return A, r


def strong_vs_weak_dirichlet(n=8, a=1.0, b_val=0.0,
                             betas=(1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6)):
    """Compare strong (row-replace / symmetric) and weak (penalty)
    Dirichlet on -u''=0, u(0)=a, u(1)=b_val (exact: u=a+(b_val-a)x).
    Returns the matrices for visualization and the penalty convergence."""
    K, h = fe1d_stiffness(n)
    rhs = np.zeros(n + 1)               # f = 0
    x = np.linspace(0, 1, n + 1)
    exact = a + (b_val - a) * x
    # strong: both ends, asymmetric row replacement then symmetric variant
    A_rr, r_rr = dirichlet_strong(K, rhs, 0, a)
    A_rr, r_rr = dirichlet_strong(A_rr, r_rr, n, b_val)
    u_rr = np.linalg.solve(A_rr, r_rr)
    A_sym, r_sym = dirichlet_strong(K, rhs, 0, a, symmetric=True)
    A_sym, r_sym = dirichlet_strong(A_sym, r_sym, n, b_val, symmetric=True)
    u_sym = np.linalg.solve(A_sym, r_sym)
    # weak: penalty sweep
    pen = []
    for beta in betas:
        A_p, r_p = dirichlet_penalty(K, rhs, 0, a, beta)
        A_p, r_p = dirichlet_penalty(A_p, r_p, n, b_val, beta)
        u_p = np.linalg.solve(A_p, r_p)
        pen.append(dict(beta=beta, u0=float(u_p[0]),
                        bc_err=float(abs(u_p[0] - a) + abs(u_p[-1] - b_val)),
                        sol_err=float(np.abs(u_p - exact).max())))
    return dict(
        n=n, x=x, exact=exact, K=K, A_rr=A_rr, A_sym=A_sym,
        u_rr=u_rr, u_sym=u_sym, penalty=pen,
        rr_symmetric=bool(np.allclose(A_rr, A_rr.T)),
        sym_symmetric=bool(np.allclose(A_sym, A_sym.T)),
        strong_err=float(np.abs(u_rr - exact).max()))
