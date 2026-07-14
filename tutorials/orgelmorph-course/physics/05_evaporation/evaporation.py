"""OrgElMorph course - Physics P5: ternary Cahn-Hilliard with SOLVENT
EVAPORATION (the drying film).

Importable core.  An organic film is cast from solution and DRIES: the
solvent leaves through the top surface, the film thins, and the solutes
concentrate until they demix.  Evaporation is therefore the CLOCK of
morphology formation -- how fast the solvent leaves sets how much time
the blend has to phase-separate before it vitrifies.

We use the (M, K)-generic brick in FILM MODE
(diffsim.physics.multiphase.MultiPhaseStepper, M=2 solutes, K=0 no
crystal, film=dict(...)).  The Landau-mapped moving frame follows the
wodo_film formulation: the computational box [0,1]x[0,1] represents a
physical strip of shrinking height h(t), the solvent flux at the top

    K = SUM_i k_e_i * avg(phi_i^top)   (>= 0, frozen per step)

drives h(t) DOWN (dh/dt = -K), and each conserved solute gains a frame-
advection term plus a surface-enrichment flux so the solute CONTENT
h * INT phi_i is conserved exactly per step (nothing evaporates but the
solvent).

THE EVAPORATION RATE (Biot number).  k_e is the nondimensional
evaporation rate of the solvent (the Biot number Bi = k_e h0 / D_s in
the wodo convention, with D_s the solute Onsager mobility; larger k_e =
more volatile solvent = faster drying).

MATCHED TERMINAL STATE (the honest comparison).  Drying RATE and final
STATE must NOT be confounded: a slow rate that only reaches phi_s=0.30 by
the horizon has a DIFFERENT dryness from a fast rate dried to phi_s=0.10,
so a bare "faster = finer" comparison mixes the two.  ``run_film`` records
the morphology metrics along the WHOLE mean-solvent-fraction trajectory
phi_s(t); the harness then reports each metric at a MATCHED phi_s (e.g.
0.30 / 0.20 / 0.10) for every rate.  Only then does "rate reshapes
morphology at fixed dryness" mean anything.

MORPHOLOGY METRICS (real lengths, not a bare "cells<1" number).  The
length scale is the first spectral moment of the LATERAL (periodic-x)
structure factor of the polymer field, expressed as a wavelength both as
a fraction of the box width AND in CELLS (= fraction * n_x) -- a resolved
length is always > 1 cell.  We report >= 2 metrics: lateral wavelength,
interfacial length, and phase contrast.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper
from diffsim.diagnostics import morphology as dgm
from diffsim.diagnostics import conservation as dcons


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    """Laterally periodic (x), non-periodic vertical (y = the drying
    direction; the top face recedes)."""
    tree = build_uniform(level, dim=2, periodic=(True, False))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _chi_aa(c_pf, c_ps, c_fs):
    """3x3 FULL-species chi (0=polymer, 1=fullerene, 2=solvent)."""
    m = np.zeros((3, 3))
    m[0, 1] = m[1, 0] = c_pf
    m[0, 2] = m[2, 0] = c_ps
    m[1, 2] = m[2, 1] = c_fs
    return m


# ----------------------------------------------------------------------
# morphology metrics (real lengths)
# ----------------------------------------------------------------------
def lateral_wavelength(g, box_width=1.0):
    """First-moment LATERAL wavelength of a demeaned periodic-x field.

    The lateral structure factor is the power of ``FFT_x(g - <g>)`` averaged
    over the (non-periodic) vertical rows.  The FFT index ``k`` counts whole
    cycles across the box, so the first spectral moment ``kbar`` is a mean
    cycles-per-box and the characteristic wavelength is ``box_width / kbar``.

    Returns ``(wl_frac, wl_cells)`` where ``wl_frac`` is the wavelength as a
    fraction of the box width and ``wl_cells = wl_frac * n_x`` is the SAME
    length in grid cells (> 1 for any resolved morphology).  This is the fix
    for the old ``_domain_scale`` which reported a bare "cells" number < 1.
    """
    g = np.asarray(g, float)
    nx = g.shape[0]
    a = g - g.mean()
    F = np.abs(np.fft.rfft(a, axis=0)) ** 2      # power along periodic x
    P = F.mean(axis=1)                           # average over y-rows
    P[0] = 0.0                                   # drop DC (demeaned)
    k = np.arange(P.shape[0])
    if P.sum() <= 0:
        return float("nan"), float("nan")
    kbar = float((k * P).sum() / P.sum())        # mean cycles per box width
    kbar = max(kbar, 1e-9)
    return float(box_width / kbar), float(nx / kbar)


def morphology_metrics(gp, box_width=1.0):
    """Multi-metric morphology readout of a polymer-field snapshot ``gp``.

    Returns a dict with the lateral wavelength (as a box-width fraction and in
    cells), the interfacial length (diffuse-interface estimate, via the shared
    diagnostics, in box-width units), and the phase contrast (field std).
    """
    nx = gp.shape[0]
    dx = box_width / nx
    wl_frac, wl_cells = lateral_wavelength(gp, box_width=box_width)
    iface = dgm.interfacial_area(gp, dx=dx)      # length (2-D)
    return {"wl_frac": wl_frac, "wl_cells": wl_cells,
            "iface": float(iface), "contrast": float(gp.std())}


# ----------------------------------------------------------------------
# node <-> grid map + fixed-domain quadrature weights
# ----------------------------------------------------------------------
def _grid_map(mesh):
    """Return ``(nx, ny, ix, iy)`` mapping full nodes onto a lateral x by
    vertical y grid (iy=0 substrate, iy=-1 top)."""
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))
    return len(xs), len(ys), ix, iy


def _fixed_domain_weights(nx, ny):
    """P1 lumped-mass (tensor-trapezoid) weights on the FIXED reference box
    [0,1]x[0,1]: uniform 1/nx in the periodic x, trapezoid in the non-periodic
    y.  ``INT phi dV_hat = sum(w * phi_grid)`` (units: box area = 1).  On a
    uniform P1 grid these equal the FE mass-matrix row sums, so the content is
    the same integral the assembly conserves."""
    wx = np.full(nx, 1.0 / nx)
    dy = 1.0 / (ny - 1)
    wy = np.full(ny, dy)
    wy[0] *= 0.5
    wy[-1] *= 0.5
    return np.outer(wx, wy)


# ----------------------------------------------------------------------
# the drying march (trajectory-aware)
# ----------------------------------------------------------------------
def run_film(dm, mesh, cons, k_e=0.4, chi=(1.5, 0.3, 0.3), N=(5.0, 5.0, 1.0),
             onsager=(0.2, 0.0, 0.2), kappa=(3e-4, 3e-4), phi0=(0.22, 0.22),
             amp=0.01, h0=1.0, dt=1e-3, dt_max=0.01, t_end=8.0, seed=7,
             phis_stop=0.08, h_min=0.12, device="cuda:0", nchecks=40,
             linsolver="cudss"):
    """March a drying ternary film at evaporation rate ``k_e`` and record the
    WHOLE trajectory: at each checkpoint the mean solvent fraction phi_s, the
    physical height h, the per-solute content INT phi_i (quadrature on the
    fixed reference domain), and the reconstructed polymer/fullerene grids (for
    metric-vs-phi_s and matched-state snapshots).  Returns a record dict."""
    ons = np.array([[onsager[0], onsager[1]],
                    [onsager[1], onsager[2]]])
    st = MultiPhaseStepper(
        dm, M=2, K=0, chi_aa=_chi_aa(*chi), N=list(N), onsager=ons,
        kappa=list(kappa), dt=dt, mob="const",
        film=dict(k_e=k_e, h0=h0), b_reg=1e-4,
        newton_tol=1e-8, newton_max=40, linsolver=linsolver)
    rng = np.random.default_rng(seed)
    icp = phi0[0] + amp * rng.standard_normal(st.nfree)
    icf = phi0[1] + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: icp, lambda x: icf])

    nx, ny, ix, iy = _grid_map(mesh)
    W = _fixed_domain_weights(nx, ny)

    def grids():
        p = np.asarray(cons.T @ st.phi(0))
        f = np.asarray(cons.T @ st.phi(1))
        gp = np.zeros((nx, ny)); gp[ix, iy] = p
        gf = np.zeros((nx, ny)); gf[ix, iy] = f
        return gp, gf

    tlog, phis_log, h_log = [], [], []
    cp_log, cf_log = [], []
    traj_gp, traj_gf = [], []

    def record():
        gp, gf = grids()
        tlog.append(st.t)
        phis_log.append(float((1.0 - gp - gf).mean()))
        h_log.append(st.h_curr)
        cp_log.append(dcons.quadrature_mass(gp.ravel(), W.ravel()))
        cf_log.append(dcons.quadrature_mass(gf.ravel(), W.ravel()))
        traj_gp.append(gp.copy()); traj_gf.append(gf.copy())

    record()
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    reason = "t_end"
    for tc in checks:
        r = st.march(t_end=tc, dt_max=dt_max, max_steps=8000,
                     dt_min=1e-9, grow_iters=25, h_min=h_min,
                     phis_stop=phis_stop)
        record()
        if r in ("phis_stop", "h_min", "dt_underflow"):
            reason = r
            break

    return dict(
        nx=nx, ny=ny, k_e=float(k_e), D_s=float(onsager[0]),
        t=np.array(tlog), phis=np.array(phis_log), h=np.array(h_log),
        content_p=np.array(cp_log), content_f=np.array(cf_log),
        traj_gp=np.array(traj_gp), traj_gf=np.array(traj_gf),
        reason=reason, t_dry=st.t, h_final=st.h_curr,
        phis_final=phis_log[-1])


# ----------------------------------------------------------------------
# backward-compatible scalar driver (used by run.py / the old figures)
# ----------------------------------------------------------------------
def run(dm, mesh, cons, k_e=0.3, chi=(1.5, 0.3, 0.3), N=(5.0, 5.0, 1.0),
        onsager=(0.2, 0.0, 0.2), kappa=(3e-4, 3e-4), phi0=(0.22, 0.22),
        amp=0.01, dt=1e-3, dt_max=0.015, t_end=8.0, seed=7,
        phis_stop=0.08, h_min=0.12, device="cuda:0", nchecks=50):
    """Thin wrapper over :func:`run_film` preserving the old return keys
    (``snaps``, ``domain_scale`` as a box-width fraction, ``contrast``) plus
    the matched-state trajectory arrays."""
    rec = run_film(dm, mesh, cons, k_e=k_e, chi=chi, N=N, onsager=onsager,
                   kappa=kappa, phi0=phi0, amp=amp, dt=dt, dt_max=dt_max,
                   t_end=t_end, seed=seed, phis_stop=phis_stop, h_min=h_min,
                   device=device, nchecks=nchecks)
    gp0, gf0 = rec["traj_gp"][0], rec["traj_gf"][0]
    gpf, gff = rec["traj_gp"][-1], rec["traj_gf"][-1]
    wl_frac, wl_cells = lateral_wavelength(gpf)
    rec.update(
        snaps={0.0: (gp0.copy(), gf0.copy(), rec["h"][0]),
               "final": (gpf.copy(), gff.copy(), rec["h_final"])},
        domain_scale=wl_frac, domain_scale_cells=wl_cells,
        contrast=float(gpf.std()))
    return rec
