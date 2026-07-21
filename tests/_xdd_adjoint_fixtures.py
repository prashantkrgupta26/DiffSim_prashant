"""Small 2-D lit XDD systems shared by the R1 adjoint gates.

Level-3 uniform bilayer in the resolvable-Debye / symmetric-transport regime
(the Block-C marchable config, ``test_xdd_run._resolvable_bilayer``) so the BDF
march reaches a lit steady state in a handful of steps on CPU or one GPU.

R0-CONSTRUCTION NOTE (deviation from the task-1 brief's fixture sketch): the
brief referenced ``diffsim.xdd.morphology.signed_distance_bilayer`` and a
``RegionMobility``-driven μ̂ field at the physical Debye length.  Neither is the
real R0 idiom — ``signed_distance_bilayer`` does not exist, and the physical
``s.lambda2`` (~2.2e-5) is un-marchable at level 3 (documented φ̂ blow-up in
``test_xdd_run._resolvable_bilayer``).  This fixture uses the *actual* proven R0
construction: a GP-coordinate signed-distance field
``dist_gp = (xq[:,1]-0.5)*_DIST_SCALE`` (matches ``test_exciton_system`` /
``test_xdd_run``), constant symmetric mobilities, ``lam2=1e-1``, ``Eg_hat=4.0``,
plus CW generation for a nondegenerate lit steady (excitons active, R̂ nonzero,
so the closure controls carry signal).
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.poisson import gauss_points
from diffsim.physics.exciton_system import (XDDSystem, bilayer_electrode_bcs)
from diffsim.physics.exciton_closures import (LangevinRecombination,
                                              OnsagerBraunDissociation,
                                              Generation)
from diffsim.xdd.params import XDDParams
from diffsim.xdd.run import log_linear_ic, _march_to_steady

# distance scale (matches test_exciton_system.py / test_xdd_run.py)
_DIST_SCALE = 4e-9


def build_small_lit_system(device, level=3, Eg_hat=4.0, mu=0.5, lam2=1e-1,
                           zeta=1e-3, want_gen=False):
    """Return (sysm, converged_lit_state) for a small resolvable-Debye bilayer.

    Resolvable-Debye + symmetric-transport config so a short BDF march reaches
    steady in a handful of steps.  CW generation at a moderate nondim peak gives
    a nondegenerate lit steady (R̂ nonzero — the closure controls have signal).

    With ``want_gen=True`` additionally returns ``(dist_gp, gen)`` so the
    illumination control can be constructed against the same generation object.
    """
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)

    xq = gauss_points(mesh, dm.tables_by_p)
    p = XDDParams()
    s = p.scales()

    # GP signed-distance field + constant symmetric coefficient fields (the
    # marchable Block-C idiom).
    dist_gp = {pv: (xq[pv][:, 1] - 0.5) * _DIST_SCALE for pv in xq}
    mu_n = {}; mu_p = {}; mu_xd = {}; mu_xa = {}; eps = {}
    for pv in xq:
        n = len(dist_gp[pv])
        mu_n[pv] = np.full(n, mu); mu_p[pv] = np.full(n, mu)
        mu_xd[pv] = np.full(n, mu); mu_xa[pv] = np.full(n, mu)
        eps[pv] = np.ones(n)

    lang = LangevinRecombination(p, strategy="sum", zeta=zeta, spatial="uniform")
    ons = OnsagerBraunDissociation(p, width=p.interface_thk)
    tau_inv = s.t0 / p.tau_x_donor
    sysm = XDDSystem(dm, lam2=lam2, eps_gp=eps,
                     mu_n_gp=mu_n, mu_p_gp=mu_p,
                     mu_xd_gp=mu_xd, mu_xa_gp=mu_xa, dist_gp=dist_gp,
                     langevin=lang, onsager=ons,
                     tau_inv_d=tau_inv, tau_inv_a=tau_inv,
                     carrier_vars="log", assembly="host")
    bilayer_electrode_bcs(sysm, mesh, cons, Eg_hat=Eg_hat, V_app_hat=0.0,
                          minority_ln=-Eg_hat)

    # CW generation at a moderate nondim peak (peak-normalised to 1).
    gen = Generation(p, profile="constant", waveform="cw")
    gd_gp = {}; ga_gp = {}
    for pv in dm.bins:
        Gd, Ga = gen.spatial(dist_gp[pv], 0.0)
        peak = max(float(np.max(np.abs(Gd))), float(np.max(np.abs(Ga))), 1e-30)
        gd_gp[pv] = Gd * (1.0 / peak); ga_gp[pv] = Ga * (1.0 / peak)
    sysm.set_generation(gd_gp, ga_gp)

    ic = log_linear_ic(mesh, Eg_hat, minority_ln=-Eg_hat)
    state, info = _march_to_steady(sysm, ic, dt0_hat=1e-6, dt_max_hat=0.1,
                                   max_steps=400, time_stepping_tol=1e-6)
    assert info["criterion_fired"], info.get("reason")
    if want_gen:
        return sysm, state, dist_gp, gen
    return sysm, state
