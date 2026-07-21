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
    # Attach the XDDParams handle so the R1 PL/TRPL QoIs can read τ̂_r via
    # _params_of(sysm) (the driver/fixture is the sanctioned owner of params).
    sysm.params = p
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


def steady_polish(sysm, state):
    """Polish `state` onto a tight root of the STEADY residual R_steady(u;p)=0.

    The transient BDF march stops on a field-CHANGE criterion at a finite final
    dt̂, so its fixed point satisfies R_steady=0 only to the march tolerance
    (~1e-6 in the free dofs).  The implicit steady adjoint linearises the exact
    steady map, so for a ≤1e-6 gradient gate the FD leg must converge the SAME
    map: a forward steady Newton solve (σ=0, no BDF history) driven to rtol
    1e-13.  This is a genuine FORWARD solve — independent of the transposed
    analytic adjoint — so the gate stays a real check, not a tautology.
    """
    from diffsim.physics.exciton_system import NDOF
    sysm.sigma = 0.0
    sysm.hist = None
    st = {f: np.asarray(state[f]).copy() for f in range(NDOF)}
    out, info = sysm.solve_newton(st, max_iter=30, rtol=1e-13)
    return out


def remarch_steady(sysm, state):
    """Re-converge the STEADY implicit problem from `state` after a parameter
    change (the IFT-consistency the steady adjoint models).

    Re-runs ``_march_to_steady`` using the current `state` as the initial
    condition with the SAME (tight) march controls as the fixture, THEN polishes
    onto the exact steady root with ``steady_polish`` — so the FD leg of the
    steady gradient gate genuinely re-solves R_steady(u;p)=0 at the mutated
    parameter (NOT a frozen-Jacobian probe, and NOT merely the loose transient
    fixed point).  Returns the new converged steady state.
    """
    st = {f: np.asarray(state[f]).copy() for f in state}
    new_state, info = _march_to_steady(sysm, st, dt0_hat=1e-6, dt_max_hat=0.1,
                                       max_steps=400, time_stepping_tol=1e-6)
    assert info["criterion_fired"], info.get("reason")
    return steady_polish(sysm, new_state)


# ══════════════════════════════════════════════════════════════════════════════
# Task 7 — the torch autograd TWIN (independent leg of the three-way check)
#
# A faithful torch reimplementation of the XDD PHYSICAL 5-field residual
# ``residual_full`` (steady, σ=0, PHYSICAL n̂/p̂ — NOT the log increment), built
# element-by-element in torch autograd from the SAME basis tables / dof layout.
# The forward map is a torch Newton solve at the operating point with the control
# a ``requires_grad=True`` leaf (``torch.linalg.solve(J, -R)`` to a tight
# residual → the implicit-function-theorem gradient); ``loss.backward()`` gives
# dJ/dp.  Because the twin re-derives A_phys (the log-unscaled, Dirichlet-clean
# physical Jacobian the analytic gradient uses) through a genuinely independent
# code path — its own torch assembly, never the R0 warp kernels or the analytic
# adjoint's transpose — it is the sole independent cross-check on the
# log-carrier-unscaling transform (Task-7 carry 1) AND the rigorous check of the
# semi-analytic carrier-µ derivative (carry 2).
# ══════════════════════════════════════════════════════════════════════════════

import math as _math

import torch as _torch

_torch.set_default_dtype(_torch.float64)

from diffsim.physics.exciton_system import IPHI, IN, IP, IXD, IXA, NDOF
from diffsim.physics.exciton_closures import (interface_mask, tanh_mask,
                                              _KB, _Q, _EPS0)


def _t(a):
    return _torch.tensor(np.asarray(a, np.float64), dtype=_torch.float64)


def _t2(a, ne, nqp):
    """GP field (flat [ne*nqp]) → torch [ne, nqp]."""
    return _torch.tensor(np.asarray(a, np.float64).reshape(ne, nqp),
                         dtype=_torch.float64)


class _BinTables:
    """Per-bin geometry + basis tables lifted into torch (built once)."""

    def __init__(self, dm, pv):
        b = dm.bins[pv]
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)  # [ne]
        tb = dm.tables_by_p[pv]
        self.conn = _torch.tensor(conn, dtype=_torch.int64)
        self.N = _t(tb.N)                       # [nqp, nbf]
        self.dN = _t(tb.dN)                     # [nqp, nbf, dim]
        self.lapN = _t(tb.lapN)                 # [nqp, nbf]
        self.w = _t(tb.w)                       # [nqp]
        self.ne, self.nbf, self.nqp = ne, nbf, nqp
        self.dim = dm.dim
        self.h = _t(h)                          # [ne]
        self.dscale = 2.0 / self.h              # [ne]
        self.jac = (0.5 * self.h) ** dm.dim     # [ne]
        self.dJxW = self.w[None, :] * self.jac[:, None]     # [ne, nqp]


class XDDSteadyTwin:
    """Torch autograd twin of the steady PHYSICAL XDD residual.

    Reimplements ``residual_full`` (σ=0, physical carriers) in torch and marches a
    dense Newton solve on the NON-DIRICHLET free dofs (Dirichlet dofs pinned to
    the BC values) with the control a leaf tensor.  Level-3 uniform → T is the
    identity and every node is free, so the reduced physical space is the nodal
    space; Dirichlet nodes are simply held fixed.
    """

    def __init__(self, sysm):
        self.sysm = sysm
        dm = sysm.dm
        self.dm = dm
        self.nn = dm.n_nodes
        self.dim = dm.dim
        self.bins = [(_BinTables(dm, pv), pv) for pv in dm.bins]
        self._shape = {pv: (B.ne, B.nqp) for B, pv in self.bins}
        # Dirichlet node/value per field (pinned dofs).
        self.dir_mask = {f: np.zeros(self.nn, bool) for f in range(NDOF)}
        self.dir_val = {f: np.zeros(self.nn) for f in range(NDOF)}
        for f, (nodes, vals) in sysm.dirichlet.items():
            self.dir_mask[f][nodes] = True
            self.dir_val[f][nodes] = vals
        self.dir_mask_t = {f: _torch.tensor(self.dir_mask[f])
                           for f in range(NDOF)}
        self.dir_val_t = {f: _t(self.dir_val[f]) for f in range(NDOF)}
        self.free_idx = {f: np.where(~self.dir_mask[f])[0] for f in range(NDOF)}
        # dist GP fields (fixed) + region eps mask base
        self.dist_gp = {pv: np.asarray(sysm.dist_gp[pv]) for pv in dm.bins}
        # base generation GP (torch, [ne, nqp])
        gd, ga = sysm._gen()
        self.gd0 = {pv: _t2(gd[pv], *self._shape[pv]) for pv in dm.bins}
        self.ga0 = {pv: _t2(ga[pv], *self._shape[pv]) for pv in dm.bins}
        # Langevin gh (uniform scalar) and Onsager k-fields precomputed as
        # torch tensors; both are STATE-independent apart from |∇φ| (Onsager),
        # which is reformed each residual eval.
        self._precompute_closures()

    # ---- closure coefficients (torch) --------------------------------------
    def _precompute_closures(self):
        s = self.sysm
        p = s.params
        sc = p.scales()
        # Langevin gh (uniform "sum" strategy): gh = zeta * gamma0 * C0^2/U0.
        # zeta is the control leaf for ClosureControl; store the gamma-part.
        self._lang_gamma_part = float(sc.gamma0 * sc.C0 ** 2 / sc.U0)
        self._lang_zeta0 = float(s.langevin.zeta)
        # Onsager: k_hat(|∇φ|) = k_base(dist) * Phi(b(|∇φ|)) * scaling; assemble
        # the dist-dependent, |∇φ|-independent pieces per bin.
        ons = s.onsager
        self._ons = {}
        for pv in self.dm.bins:
            dist = self.dist_gp[pv]
            eps_r = p.eps_D + (p.eps_A - p.eps_D) * tanh_mask(dist, ons.width)
            eps = _EPS0 * eps_r
            kT = _KB * p.T
            E_B = _Q ** 2 / (4.0 * _math.pi * eps * p.a)
            prefactor = 3.0 * sc.gamma0 / (4.0 * _math.pi * p.a ** 3)
            exp_EB = np.exp(-E_B / kT)
            mask = interface_mask(dist, p.interface_thk / 2.0)
            k_base = prefactor * exp_EB * mask * sc.t0    # × Phi(b) × scaling
            # b = C_b * E,  E = |∇φ| * phi0/x0 → b = Cb_field * |∇φ|
            Cb_field = (_Q ** 3 / (8.0 * _math.pi * eps * kT ** 2)
                        * sc.phi0 / sc.x0)
            ne, nqp = self._shape[pv]
            self._ons[pv] = dict(k_base=_t2(k_base, ne, nqp),
                                 Cb=_t2(Cb_field, ne, nqp))
        self._ons_scale_d0 = float(ons.params.ex_diss_d_scaling)
        self._ons_scale_a0 = float(ons.params.ex_diss_a_scaling)

    def _k_onsager(self, gmag, pv):
        """Onsager k_base·Phi(b) per bin (torch), pre-scaling."""
        o = self._ons[pv]
        b = o["Cb"] * gmag
        Phi = (1.0 + b + b ** 2 / 3.0 + b ** 3 / 18.0
               + b ** 4 / 180.0 + b ** 5 / 2700.0)
        return o["k_base"] * Phi

    # ---- nodal → GP interpolation (torch) ----------------------------------
    def _interp(self, field, B):
        vals = field[B.conn]                           # [ne, nbf]
        v = _torch.einsum("qa,ea->eq", B.N, vals)      # [ne, nqp]
        g = _torch.einsum("qad,ea,e->eqd", B.dN, vals, B.dscale)  # [ne,nqp,dim]
        lap = _torch.einsum("qa,ea,e->eq", B.lapN, vals, B.dscale ** 2)
        return v, g, lap

    def _tau_m(self, amag, B, mu):
        """supg·τ_M at GPs, matching _tau_gp (steady σ=0 → sig2tau=0)."""
        he = B.h[:, None]                              # [ne,1]
        uGu = 4.0 * amag ** 2 / (he ** 2)
        GG = self.dim * (2.0 / he) ** 4
        CI = 36.0
        tau = 1.0 / _torch.sqrt(uGu + CI * mu ** 2 * GG)
        return self.sysm.supg * tau

    # ---- carrier row (SUPG, conservative) — R = K@c − F  (physical) --------
    def _carrier_row(self, field, c, gradphi, mu_scalar, source_gp, sign):
        """Scatter the carrier residual R[field] for one carrier.

        mu_scalar : float or torch scalar (uniform µ̂).  sign −1 electrons / +1
        holes.  source_gp[pv] the net GP source (D̂ − R̂).  Matches
        make_xdd_carrier_Ae/be (steady σ=0): drift (aq·∇N_a)c + diffusion
        µ̂∇N_b·∇N_a + SUPG τ(U·∇N_a)·res − load (f,N_a)+τ(U·∇N_a,f)."""
        R = _torch.zeros(self.nn, dtype=_torch.float64)
        for B, pv in self.bins:
            cv, gc, lapc = self._interp(c, B)
            gphi = gradphi[pv]                          # [ne,nqp,dim]
            mu = mu_scalar
            aq = sign * mu * gphi                        # [ne,nqp,dim] drift wt
            amag = _torch.sqrt(_torch.clamp(
                _torch.sum(aq * aq, dim=2), min=0.0))    # |U|=|aq|
            tau = self._tau_m(amag, B, mu)               # [ne,nqp]
            # ∇N_a scaled: [ne,nqp,nbf,dim]
            gNa = B.dN[None] * B.dscale[:, None, None, None]
            # aq·∇N_a : [ne,nqp,nbf]
            aqgNa = _torch.einsum("eqd,eqad->eqa", aq, gNa)
            Ugw = -aqgNa                                 # U·∇N_a
            # strong residual on c (advective): res = U·∇c − µ̂ lapc  (σ=0)
            Ugc = -_torch.einsum("eqd,eqd->eq", aq, gc)  # U·∇c
            res_c = Ugc - mu * lapc                      # [ne,nqp]
            f = source_gp[pv]                            # [ne,nqp] net source
            wR = B.dJxW                                  # [ne,nqp]
            galerkin_drift = _torch.einsum("eqa,eq,eq->ea", aqgNa, cv, wR)
            galerkin_diff = mu * _torch.einsum(
                "eqad,eqd,eq->ea", gNa, gc, wR)
            supg_stiff = _torch.einsum("eq,eqa,eq,eq->ea", tau, Ugw, res_c, wR)
            load_g = _torch.einsum("qa,eq,eq->ea", B.N, f, wR)
            load_s = _torch.einsum("eqa,eq,eq->ea", Ugw, tau * f, wR)
            Re = galerkin_drift + galerkin_diff + supg_stiff - load_g - load_s
            R = R.index_add(0, B.conn.reshape(-1), Re.reshape(-1))
        return R

    # ---- exciton row (no SUPG) — R = (σ_tot M + µ̂ K)@X − F -----------------
    def _exciton_row(self, X, mu_scalar, sigtot_gp, feed_gp):
        R = _torch.zeros(self.nn, dtype=_torch.float64)
        for B, pv in self.bins:
            Xv, gX, _ = self._interp(X, B)
            mu = mu_scalar
            wR = B.dJxW
            sig = sigtot_gp[pv]                          # [ne,nqp]
            gNa = B.dN[None] * B.dscale[:, None, None, None]
            mass = _torch.einsum("eq,qa,eq->ea", sig * Xv, B.N, wR)
            diff = mu * _torch.einsum("eqad,eqd,eq->ea", gNa, gX, wR)
            load = _torch.einsum("qa,eq,eq->ea", B.N, feed_gp[pv], wR)
            Re = mass + diff - load
            R = R.index_add(0, B.conn.reshape(-1), Re.reshape(-1))
        return R

    # ---- poisson row — R = λ²ε̂K φ̂ − M(p̂−n̂) --------------------------------
    def _poisson_row(self, phi, n, p, eps_gp):
        R = _torch.zeros(self.nn, dtype=_torch.float64)
        lam2 = float(self.sysm.lam2)
        for B, pv in self.bins:
            _, gphi, _ = self._interp(phi, B)
            nv, _, _ = self._interp(n, B)
            pv_, _, _ = self._interp(p, B)
            wR = B.dJxW
            gNa = B.dN[None] * B.dscale[:, None, None, None]
            stiff = lam2 * _torch.einsum(
                "eq,eqad,eqd,eq->ea", eps_gp[pv], gNa, gphi, wR)
            src = _torch.einsum("qa,eq,eq->ea", B.N, (pv_ - nv), wR)
            Re = stiff - src
            R = R.index_add(0, B.conn.reshape(-1), Re.reshape(-1))
        return R

    # ---- full physical residual (torch) ------------------------------------
    def residual(self, U, ctrl_vals):
        """U : dict field→(nn,) torch nodal vectors (physical).  ctrl_vals a
        dict carrying the control-perturbed quantities (see _apply_control)."""
        s = self.sysm
        phi, n, p, xd, xa = (U[IPHI], U[IN], U[IP], U[IXD], U[IXA])
        eps_gp = ctrl_vals["eps_gp"]
        mu_n = ctrl_vals["mu_n"]; mu_p = ctrl_vals["mu_p"]
        mu_xd = ctrl_vals["mu_xd"]; mu_xa = ctrl_vals["mu_xa"]
        tau_inv_d = ctrl_vals["tau_inv_d"]; tau_inv_a = ctrl_vals["tau_inv_a"]
        gd = ctrl_vals["gd"]; ga = ctrl_vals["ga"]
        gh = ctrl_vals["gh"]                             # Langevin prefactor
        scale_d = ctrl_vals["scale_d"]; scale_a = ctrl_vals["scale_a"]

        R = {}
        R[IPHI] = self._poisson_row(phi, n, p, eps_gp)

        # closures at GPs
        R_hat = {}; kd = {}; ka = {}; Dhat = {}; s_carr = {}
        gradphi = {}
        for B, pv in self.bins:
            _, gphi, _ = self._interp(phi, B)
            gradphi[pv] = gphi
            nv, _, _ = self._interp(n, B)
            pvv, _, _ = self._interp(p, B)
            xdv, _, _ = self._interp(xd, B)
            xav, _, _ = self._interp(xa, B)
            gmag = _torch.sqrt(_torch.clamp(
                _torch.sum(gphi * gphi, dim=2), min=1e-300))
            R_hat[pv] = gh * nv * pvv
            kbase = self._k_onsager(gmag, pv)
            kd[pv] = kbase * scale_d
            ka[pv] = kbase * scale_a
            Dhat[pv] = kd[pv] * xdv + ka[pv] * xav
            s_carr[pv] = Dhat[pv] - R_hat[pv]

        R[IN] = self._carrier_row(IN, n, gradphi, mu_n, s_carr, -1.0)
        R[IP] = self._carrier_row(IP, p, gradphi, mu_p, s_carr, +1.0)

        sigd = {}; siga = {}; fxd = {}; fxa = {}
        for B, pv in self.bins:
            sigd[pv] = tau_inv_d + kd[pv]                # σ=0 steady
            siga[pv] = tau_inv_a + ka[pv]
            fxd[pv] = gd[pv] + R_hat[pv]
            fxa[pv] = ga[pv] + R_hat[pv]
        R[IXD] = self._exciton_row(xd, mu_xd, sigd, fxd)
        R[IXA] = self._exciton_row(xa, mu_xa, siga, fxa)
        return R

    # ---- control application: build the ctrl_vals dict from a leaf ----------
    def _ctrl_vals(self, ctrl_name, leaf):
        """Return the ctrl_vals dict with the named control set from `leaf`
        (a torch leaf scalar) and every other quantity at its base value."""
        s = self.sysm
        mu0 = {IN: float(np.mean(next(iter(s.mu_n_gp.values())))),
               IP: float(np.mean(next(iter(s.mu_p_gp.values())))),
               IXD: float(np.mean(next(iter(s.mu_xd_gp.values())))),
               IXA: float(np.mean(next(iter(s.mu_xa_gp.values()))))}
        base = dict(
            mu_n=mu0[IN], mu_p=mu0[IP], mu_xd=mu0[IXD], mu_xa=mu0[IXA],
            tau_inv_d=float(s.tau_inv_d), tau_inv_a=float(s.tau_inv_a),
            gd={pv: self.gd0[pv] for pv in s.dm.bins},
            ga={pv: self.ga0[pv] for pv in s.dm.bins},
            gh=self._lang_zeta0 * self._lang_gamma_part,
            scale_d=self._ons_scale_d0, scale_a=self._ons_scale_a0,
            eps_gp={pv: _t2(s.eps_gp[pv], *self._shape[pv])
                    for pv in s.dm.bins},
        )
        if ctrl_name == "langevin_zeta":
            base["gh"] = leaf * self._lang_gamma_part
        elif ctrl_name == "mu_n":
            base["mu_n"] = leaf * mu0[IN]                # scale multiplier
        elif ctrl_name == "mu_p":
            base["mu_p"] = leaf * mu0[IP]
        elif ctrl_name == "tau_inv_d":
            base["tau_inv_d"] = leaf
        elif ctrl_name == "tau_inv_a":
            base["tau_inv_a"] = leaf
        elif ctrl_name == "illum_scalar":
            base["gd"] = {pv: leaf * self.gd0[pv] for pv in s.dm.bins}
            base["ga"] = {pv: leaf * self.ga0[pv] for pv in s.dm.bins}
        else:
            raise ValueError(f"twin: unsupported control {ctrl_name!r}")
        return base

    # ---- solve the steady physical map, backprop dJ/dp ---------------------
    def gradient(self, state, ctrl_name, p0, qoi_fn, newton_tol=1e-11,
                 newton_max=60):
        """Return (dJ/dp, J) at control value p0 via a torch Newton solve +
        backward.  qoi_fn(U_dict)→ scalar torch differentiable observable."""
        leaf = _torch.tensor(float(p0), dtype=_torch.float64,
                             requires_grad=True)
        cv = self._ctrl_vals(ctrl_name, leaf)
        # initial guess = the analytic converged state (physical)
        U = {f: _t(np.asarray(state[f])) for f in range(NDOF)}
        # pin Dirichlet dofs
        for f in range(NDOF):
            U[f] = _torch.where(self.dir_mask_t[f], self.dir_val_t[f], U[f])
        # free dof index sets (concatenated ordering field-major)
        free = {f: _torch.tensor(self.free_idx[f]) for f in range(NDOF)}
        nfree = [len(self.free_idx[f]) for f in range(NDOF)]
        offs = np.cumsum([0] + nfree)

        def pack(Ud):
            return _torch.cat([Ud[f][free[f]] for f in range(NDOF)])

        def unpack(x, Ud):
            out = {}
            for f in range(NDOF):
                col = Ud[f].clone()
                col = col.index_copy(0, free[f], x[offs[f]:offs[f + 1]])
                out[f] = col
            return out

        x = pack(U).detach().clone().requires_grad_(False)
        for _ in range(newton_max):
            xin = x.detach().clone().requires_grad_(True)
            Ucur = unpack(xin, U)
            R = self.residual(Ucur, cv)
            rfree = _torch.cat([R[f][free[f]] for f in range(NDOF)])
            # dense Jacobian via autograd
            ntot = rfree.shape[0]
            J = _torch.zeros((ntot, ntot), dtype=_torch.float64)
            for i in range(ntot):
                gi, = _torch.autograd.grad(rfree[i], xin, retain_graph=True,
                                           create_graph=False)
                J[i] = gi
            dx = _torch.linalg.solve(J, -rfree.detach())
            x = x + dx
            if float(dx.abs().max()) < newton_tol:
                break
        # differentiable re-solve: one IFT step with the graph LIVE in the leaf.
        xin = x.detach().clone().requires_grad_(True)
        Ucur = unpack(xin, U)
        R = self.residual(Ucur, cv)
        rfree = _torch.cat([R[f][free[f]] for f in range(NDOF)])
        ntot = rfree.shape[0]
        J = _torch.zeros((ntot, ntot), dtype=_torch.float64)
        for i in range(ntot):
            gi, = _torch.autograd.grad(rfree[i], xin, retain_graph=True)
            J[i] = gi
        # implicit solution: x*(p) with r(x*,p)=0 → dx*/dp = −J^{-1} ∂r/∂p.
        # Build x_star as a differentiable function of the leaf:
        #   x_star = x_detached − J^{-1} r(x_detached, p)   (one Newton step from
        # the converged point; since r≈0 there, x_star≈x but carries dp-grad).
        r_at_conv = _torch.cat([
            self.residual(unpack(x.detach(), U), cv)[f][free[f]]
            for f in range(NDOF)])
        x_star = x.detach() - _torch.linalg.solve(J.detach(), r_at_conv)
        U_star = unpack(x_star, U)
        J_val = qoi_fn(U_star)
        g, = _torch.autograd.grad(J_val, leaf)
        return float(g), float(J_val.detach())


def _steady_current_twin_qoi(twin, sysm, state, contact="anode", h_axis=1):
    """Torch designated-current QoI Jc = Σ_wall (Kc@ĉ) as a differentiable
    function of the torch nodal state dict U — the twin's own reimplementation
    of ``SteadyCurrentQoI`` (Kc = carrier stiffness at σ=0, SUPG=0)."""
    dm = sysm.dm
    coords = dm.mesh.node_coords
    hc = coords[:, h_axis]
    lo, hi = hc.min(), hc.max()
    if contact == "anode":
        wall = np.where(np.abs(hc - lo) < 1e-9)[0]
        fld, sign = IN, -1.0
        mu0 = float(np.mean(next(iter(sysm.mu_n_gp.values()))))
    else:
        wall = np.where(np.abs(hc - hi) < 1e-9)[0]
        fld, sign = IP, +1.0
        mu0 = float(np.mean(next(iter(sysm.mu_p_gp.values()))))
    ewall = np.zeros(dm.n_nodes); ewall[wall] = 1.0
    ew = _t(ewall)

    def qoi(U):
        phi = U[IPHI]; c = U[fld]
        acc = _torch.zeros(dm.n_nodes, dtype=_torch.float64)
        for B, pv in twin.bins:
            _, gphi, _ = twin._interp(phi, B)
            cv, gc, lapc = twin._interp(c, B)
            wR = B.dJxW
            aq = sign * mu0 * gphi
            gNa = B.dN[None] * B.dscale[:, None, None, None]
            aqgNa = _torch.einsum("eqd,eqad->eqa", aq, gNa)
            drift = _torch.einsum("eqa,eq,eq->ea", aqgNa, cv, wR)
            diff = mu0 * _torch.einsum("eqad,eqd,eq->ea", gNa, gc, wR)
            Re = drift + diff                            # σ=0, SUPG=0 (Kc@ĉ)
            acc = acc.index_add(0, B.conn.reshape(-1), Re.reshape(-1))
        Jc = _torch.sum(acc * ew)
        return _torch.abs(Jc)
    return qoi


# ── control dispatch + three-way helpers (Task-7 gate battery) ────────────────

def _make_control(sysm, ctrl_name, dist_gp=None, gen=None):
    """Dispatch a control name to a live control object (steady × {closure,
    material, illumination}).  ``illum_scalar`` needs the gen handle."""
    from diffsim.xdd.adjoint import (ClosureControl, MaterialControl,
                                      IlluminationControl)
    if ctrl_name == "langevin_zeta":
        return ClosureControl(sysm, "langevin_zeta")
    if ctrl_name in ("mu_n", "mu_p", "tau_inv_d", "tau_inv_a"):
        return MaterialControl(sysm, ctrl_name)
    if ctrl_name == "illum_scalar":
        return IlluminationControl(sysm, dist_gp, gen, mode="scalar")
    raise ValueError(f"_make_control: unknown {ctrl_name!r}")


def _current_frozen_mu(sysm, st, base_mu, contact="anode", h_axis=1):
    """Designated contact current Σ_wall(Kc@ĉ) with the OBSERVABLE operator Kc
    built from ``base_mu`` (the current's µ FROZEN at base) — matches what the
    adjoint's ``SteadyCurrentQoI.dJ_dp≡0`` measures for a µ control: the
    IMPLICIT-only derivative (µ enters the residual/state, NOT the observable).
    ``base_mu`` is the construction-time µ_n (anode) / µ_p (cathode) GP dict."""
    from diffsim.physics.exciton_system import _carrier_block, IN, IP
    dm = sysm.dm
    cl = sysm._closures(st)
    coords = dm.mesh.node_coords
    hc = coords[:, h_axis]; lo, hi = hc.min(), hc.max()
    if contact == "anode":
        wall = np.where(np.abs(hc - lo) < 1e-9)[0]
        fld, sign, c_gp = IN, -1.0, cl["n_gp"]
    else:
        wall = np.where(np.abs(hc - hi) < 1e-9)[0]
        fld, sign, c_gp = IP, +1.0, cl["p_gp"]
    aq = sysm._aq(cl["gradphi"], base_mu, sign)
    Kc = _carrier_block(dm, aq, base_mu, 0.0, 0.0, 0.0)
    e = np.zeros(dm.n_nodes); e[wall] = 1.0
    Jc = float(e @ (Kc @ np.asarray(st[fld])))
    return abs(Jc)


def steady_three_way(sysm, state, ctrl_name, qoi, twin, dist_gp=None, gen=None):
    """Return (adj, twin, fd) for a steady dJ/dp of the designated-current QoI.

    adj  : XDDSteadyAdjoint gradient (analytic, log-unscaled physical A).
    twin : torch IFT autograd gradient (independent physical re-derivation).
    fd   : central FD re-solving R_steady(u;p)=0 (remarch + polish).
    """
    from diffsim.xdd.adjoint import XDDSteadyAdjoint
    ctrl = _make_control(sysm, ctrl_name, dist_gp, gen)
    adj = XDDSteadyAdjoint(sysm, [ctrl]); adj.factorize(state)
    g_adj = float(adj.gradient(state, qoi)[ctrl.name][0])

    p0 = float(ctrl.get()[0])
    twin_qoi = _steady_current_twin_qoi(twin, sysm, state,
                                        contact=qoi.contact, h_axis=qoi.h_axis)
    g_tw, _ = twin.gradient(state, ctrl_name, p0, twin_qoi)

    # FD leg RE-SOLVES R_steady(u;p)=0 at the mutated param by POLISHING from the
    # SAME base converged state (the exact IFT map the adjoint models) — not a
    # full transient re-march, whose path noise otherwise corrupts the µ-scale FD
    # (the µ remarch lands on a slightly different transient fixed point).  The µ
    # controls carry a larger step (base value 1.0 with an O(1) derivative) to
    # stay above the polish roundoff floor.
    #
    # For the carrier-µ controls the OBSERVABLE current operator Kc itself scales
    # with µ (Jny = Σ_wall K_n(µ_n)@n̂), so the TOTAL derivative carries an
    # explicit ∂Jc/∂µ term.  The adjoint's SteadyCurrentQoI.dJ_dp≡0 (pure-state
    # convention) and the twin's frozen-µ observable both measure the IMPLICIT
    # path only, so the FD leg for µ freezes Kc's µ at base too — otherwise it
    # would measure a DIFFERENT functional (total vs implicit) and the three-way
    # would compare apples to oranges.  (langevin/tau/illum have no explicit µ in
    # Kc, so their current QoI is genuinely pure-state.)
    is_mu = ctrl_name in ("mu_n", "mu_p")
    eps_rel = 1e-5 if is_mu else 1e-6
    eps = eps_rel * max(1.0, abs(p0))
    base_mu = None
    if is_mu:
        fld = "mu_n_gp" if ctrl_name == "mu_n" else "mu_p_gp"
        base_mu = {pv: getattr(sysm, fld)[pv].copy() for pv in sysm.dm.bins}

    def J_of(v):
        ctrl.set(np.array([v])); st = steady_polish(sysm, state)
        if is_mu:
            val = _current_frozen_mu(sysm, st, base_mu, contact=qoi.contact,
                                     h_axis=qoi.h_axis)
        else:
            val = qoi.value(sysm, st)
        ctrl.set(np.array([p0])); return val
    g_fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    ctrl.set(np.array([p0]))
    return g_adj, g_tw, g_fd


# ══════════════════════════════════════════════════════════════════════════════
# Task 7 — the transient torch UNROLL twin (Mode-B independent leg)
#
# Unrolls the recorded frozen-dt BDF march in torch: each step is a torch Newton
# solve of the PHYSICAL transient residual (σ=1/dt BDF1 mass + history load) with
# the control a leaf; the trajectory functional J=Σ_n j(u_n) is accumulated and
# ``loss.backward()`` gives dJ/dp.  Frozen dt / order / generation are read off
# the recorded tape (never re-adapted) — exactly Mode B's contract.
# ══════════════════════════════════════════════════════════════════════════════

class XDDTransientTwin(XDDSteadyTwin):
    """Torch unroll twin of the frozen-dt BDF transient march (physical u)."""

    def _residual_transient(self, U, ctrl_vals, sigma, hist_gp):
        """Physical transient residual: like ``residual`` but with the BDF mass
        term σ·M@u on the time-stepped rows and the history load −M@hist folded
        into each time-stepped field's source (matches step_bdf / residual_full
        with self.sigma=σ, self.hist=hist)."""
        s = self.sysm
        phi, n, p, xd, xa = (U[IPHI], U[IN], U[IP], U[IXD], U[IXA])
        eps_gp = ctrl_vals["eps_gp"]
        mu_n = ctrl_vals["mu_n"]; mu_p = ctrl_vals["mu_p"]
        mu_xd = ctrl_vals["mu_xd"]; mu_xa = ctrl_vals["mu_xa"]
        tau_inv_d = ctrl_vals["tau_inv_d"]; tau_inv_a = ctrl_vals["tau_inv_a"]
        gd = ctrl_vals["gd"]; ga = ctrl_vals["ga"]
        gh = ctrl_vals["gh"]
        scale_d = ctrl_vals["scale_d"]; scale_a = ctrl_vals["scale_a"]

        R = {}
        R[IPHI] = self._poisson_row(phi, n, p, eps_gp)   # no time term for φ̂

        R_hat = {}; kd = {}; ka = {}; s_carr = {}; gradphi = {}
        for B, pv in self.bins:
            _, gphi, _ = self._interp(phi, B)
            gradphi[pv] = gphi
            nv, _, _ = self._interp(n, B)
            pvv, _, _ = self._interp(p, B)
            xdv, _, _ = self._interp(xd, B)
            xav, _, _ = self._interp(xa, B)
            gmag = _torch.sqrt(_torch.clamp(
                _torch.sum(gphi * gphi, dim=2), min=1e-300))
            R_hat[pv] = gh * nv * pvv
            kbase = self._k_onsager(gmag, pv)
            kd[pv] = kbase * scale_d
            ka[pv] = kbase * scale_a
            # carrier source: D̂ − R̂ + hist_n  (BDF history in the load)
            s_carr[pv] = (kd[pv] * xdv + ka[pv] * xav - R_hat[pv])

        # carrier rows with σ mass + history: fold hist into source, add σ·M@c
        R[IN] = self._carrier_row_transient(
            IN, n, gradphi, mu_n, s_carr, -1.0, sigma, hist_gp[IN])
        R[IP] = self._carrier_row_transient(
            IP, p, gradphi, mu_p, s_carr, +1.0, sigma, hist_gp[IP])

        sigd = {}; siga = {}; fxd = {}; fxa = {}
        for B, pv in self.bins:
            sigd[pv] = sigma + tau_inv_d + kd[pv]
            siga[pv] = sigma + tau_inv_a + ka[pv]
            fxd[pv] = gd[pv] + R_hat[pv] + hist_gp[IXD][pv]
            fxa[pv] = ga[pv] + R_hat[pv] + hist_gp[IXA][pv]
        R[IXD] = self._exciton_row(xd, mu_xd, sigd, fxd)
        R[IXA] = self._exciton_row(xa, mu_xa, siga, fxa)
        return R

    def _carrier_row_transient(self, field, c, gradphi, mu_scalar, source_gp,
                               sign, sigma, hist_gp_f):
        """Carrier row with BDF σ mass + SUPG-consistent history load.

        source f = (D̂−R̂) + hist enters the SUPG-consistent load; the σ·M@c mass
        term uses the SUPG-augmented test too (matches _carrier_block σ M term is
        plain mass, but the be history load carries the SUPG test — mirrors
        residual_full: Kn carries σ (plain mass via kernel) and Fn carries the
        history through the SUPG-consistent be).  We add σ (N_b,N_a) mass +
        SUPG(σ) and the full source load including history."""
        R = _torch.zeros(self.nn, dtype=_torch.float64)
        sig2tau = (2.0 * sigma) ** 2
        for B, pv in self.bins:
            cv, gc, lapc = self._interp(c, B)
            gphi = gradphi[pv]
            mu = mu_scalar
            aq = sign * mu * gphi
            amag = _torch.sqrt(_torch.clamp(
                _torch.sum(aq * aq, dim=2), min=0.0))
            he = B.h[:, None]
            uGu = 4.0 * amag ** 2 / (he ** 2)
            GG = self.dim * (2.0 / he) ** 4
            tau = self.sysm.supg / _torch.sqrt(sig2tau + uGu + 36.0 * mu ** 2 * GG)
            wR = B.dJxW
            gNa = B.dN[None] * B.dscale[:, None, None, None]
            aqgNa = _torch.einsum("eqd,eqad->eqa", aq, gNa)
            Ugw = -aqgNa
            Ugc = -_torch.einsum("eqd,eqd->eq", aq, gc)
            res_c = sigma * cv + Ugc - mu * lapc
            # source: net carrier + BDF history (both through SUPG-consistent be)
            f = source_gp[pv] + hist_gp_f[pv]
            mass = _torch.einsum("eq,qa,eq->ea", sigma * cv, B.N, wR)
            galerkin_drift = _torch.einsum("eqa,eq,eq->ea", aqgNa, cv, wR)
            galerkin_diff = mu * _torch.einsum("eqad,eqd,eq->ea", gNa, gc, wR)
            supg_stiff = _torch.einsum("eq,eqa,eq,eq->ea", tau, Ugw, res_c, wR)
            load_g = _torch.einsum("qa,eq,eq->ea", B.N, f, wR)
            load_s = _torch.einsum("eqa,eq,eq->ea", Ugw, tau * f, wR)
            Re = mass + galerkin_drift + galerkin_diff + supg_stiff \
                - load_g - load_s
            R = R.index_add(0, B.conn.reshape(-1), Re.reshape(-1))
        return R

    def _hist_gp_of(self, prev_U, prev2_U, dt, order):
        """BDF history GP source (σ_BDF·u^n) per time-stepped field, as GP dicts.

        BDF1: hist = (1/dt)·u^n.  BDF2: hist = (2u^n − 0.5u^{n-1})/dt.  φ̂ has no
        time term (zeroed).  Matches step_bdf's hist_full construction."""
        hist = {}
        for f in range(NDOF):
            if f == IPHI:
                hf = _torch.zeros(self.nn, dtype=_torch.float64)
            elif order == 1 or prev2_U is None:
                hf = (1.0 / dt) * prev_U[f]
            else:
                hf = (2.0 * prev_U[f] - 0.5 * prev2_U[f]) / dt
            hist[f] = {pv: self._interp(hf, B)[0] for B, pv in self.bins}
        return hist

    def gradient_traj(self, steps, ctrl_name, p0, jstep_fn,
                      newton_tol=1e-11, newton_max=60):
        """Unroll the recorded march in torch with the control a leaf; return
        (dJ/dp, J) for J = Σ_n jstep_fn(U_n).  Each step frozen at its recorded
        dt/order/gen; the initial U_0 / prev / prev2 come from the tape."""
        leaf = _torch.tensor(float(p0), dtype=_torch.float64,
                             requires_grad=True)
        cv = self._ctrl_vals(ctrl_name, leaf)
        # illumination sets generation via the leaf; other controls keep the
        # recorded per-step gen (frozen).  For steady controls gen is constant.
        free = {f: _torch.tensor(self.free_idx[f]) for f in range(NDOF)}
        nfree = [len(self.free_idx[f]) for f in range(NDOF)]
        offs = np.cumsum([0] + nfree)

        def pack(Ud):
            return _torch.cat([Ud[f][free[f]] for f in range(NDOF)])

        def unpack(x, Ud):
            out = {}
            for f in range(NDOF):
                col = Ud[f].clone()
                col = col.index_copy(0, free[f], x[offs[f]:offs[f + 1]])
                out[f] = col
            return out

        # prev / prev2 as torch (from the first step's tape); marched forward.
        prev_U = {f: _t(np.asarray(steps[0]["prev"][f])) for f in range(NDOF)}
        prev2_U = None
        Jtot = _torch.zeros((), dtype=_torch.float64)
        for rec in steps:
            dt = float(rec["dt_hat"]); order = int(rec["order"]); sigma = 1.0 / dt
            if order == 2 and rec["prev2"] is not None and prev2_U is None:
                prev2_U = {f: _t(np.asarray(rec["prev2"][f]))
                           for f in range(NDOF)}
            hist_gp = self._hist_gp_of(prev_U, prev2_U, dt, order)
            # Dirichlet-pinned initial guess = recorded converged state
            U0 = {f: _t(np.asarray(rec["state"][f])) for f in range(NDOF)}
            for f in range(NDOF):
                U0[f] = _torch.where(self.dir_mask_t[f], self.dir_val_t[f], U0[f])
            x = pack(U0).detach().clone()
            for _ in range(newton_max):
                xin = x.detach().clone().requires_grad_(True)
                R = self._residual_transient(unpack(xin, U0), cv, sigma, hist_gp)
                rfree = _torch.cat([R[f][free[f]] for f in range(NDOF)])
                ntot = rfree.shape[0]
                Jm = _torch.zeros((ntot, ntot), dtype=_torch.float64)
                for i in range(ntot):
                    gi, = _torch.autograd.grad(rfree[i], xin, retain_graph=True)
                    Jm[i] = gi
                dx = _torch.linalg.solve(Jm, -rfree.detach())
                x = x + dx
                if float(dx.abs().max()) < newton_tol:
                    break
            # differentiable IFT step (carries dp-grad AND the history graph)
            xin = x.detach().clone().requires_grad_(True)
            R = self._residual_transient(unpack(xin, U0), cv, sigma, hist_gp)
            rfree = _torch.cat([R[f][free[f]] for f in range(NDOF)])
            ntot = rfree.shape[0]
            Jm = _torch.zeros((ntot, ntot), dtype=_torch.float64)
            for i in range(ntot):
                gi, = _torch.autograd.grad(rfree[i], xin, retain_graph=True)
                Jm[i] = gi
            r_live = _torch.cat([
                self._residual_transient(unpack(x.detach(), U0), cv, sigma,
                                         hist_gp)[f][free[f]]
                for f in range(NDOF)])
            x_star = x.detach() - _torch.linalg.solve(Jm.detach(), r_live)
            U_star = unpack(x_star, U0)
            Jtot = Jtot + jstep_fn(U_star)
            # roll history forward (keep graph so downstream steps see dp/dhist)
            prev2_U = {f: prev_U[f] for f in range(NDOF)}
            prev_U = {f: U_star[f] for f in range(NDOF)}
        g, = _torch.autograd.grad(Jtot, leaf)
        return float(g), float(Jtot.detach())


def _traj_xd_twin_qoi(twin):
    """Torch J = 0.5·‖X̂_D‖² per step (matches tests' _traj_J on IXD)."""
    def jstep(U):
        xd = U[IXD]
        return 0.5 * _torch.sum(xd * xd)
    return jstep


def transient_three_way(sysm, state, ctrl_name, twin, N=6, dist_gp=None,
                        gen=None):
    """Return (adj, twin, fd) for a transient trajectory dJ/dp (J=0.5Σ‖X̂_D‖²).

    adj  : XDDTransientAdjoint reverse-sweep gradient.
    twin : torch unroll autograd gradient (independent).
    fd   : central FD re-running the FULL forward march per perturbation.
    """
    from diffsim.xdd.run import march_with_checkpoints
    from diffsim.xdd.adjoint import XDDTransientAdjoint
    from diffsim.physics.exciton_system import IXD

    ctrl = _make_control(sysm, ctrl_name, dist_gp, gen)
    fs, steps = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                       dt_max_hat=1e-2, max_steps=N, order=1)

    def traj_J(all_steps):
        return 0.5 * sum(float(s["state"][IXD] @ s["state"][IXD])
                         for s in all_steps)

    def traj_dJdx(stp):
        dJdx = []
        for s in stp:
            seed = {f: np.zeros(sysm.dm.n_nodes) for f in range(NDOF)}
            seed[IXD] = s["state"][IXD]
            dJdx.append(np.concatenate([np.asarray(sysm.T.T @ seed[f])
                                        for f in range(NDOF)]))
        return dJdx

    adj = XDDTransientAdjoint(sysm, [ctrl], steps)
    g_adj = float(adj.gradient(traj_dJdx(steps))[ctrl.name][0])

    p0 = float(ctrl.get()[0])
    g_tw, _ = twin.gradient_traj(steps, ctrl_name, p0, _traj_xd_twin_qoi(twin))

    eps = (1e-4 if ctrl_name in ("mu_n", "mu_p") else 1e-6) * max(1.0, abs(p0))

    def J_of(v):
        ctrl.set(np.array([v]))
        _, st2 = march_with_checkpoints(sysm, state, dt0_hat=1e-4,
                                        dt_max_hat=1e-2, max_steps=N, order=1)
        val = traj_J(st2); ctrl.set(np.array([p0])); return val
    g_fd = (J_of(p0 + eps) - J_of(p0 - eps)) / (2 * eps)
    ctrl.set(np.array([p0]))
    return g_adj, g_tw, g_fd
