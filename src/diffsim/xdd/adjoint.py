"""SP-1 R1 — differentiable XDD adjoint machinery.

The Control interface (the unit of "a thing we differentiate with respect to"),
the three R1 control classes, and the two adjoint drivers (implicit steady-state
Mode A, taped frozen-dt transient Mode B) specialised to the XDD 5-field system.
Reuses the R0 assembler (exciton_system / exciton_device), the phasefield IFT
reverse idiom, the sbm transient-adjoint checkpointing lineage, and the
splu(A.T)/solve_linear transposed solve — nothing is forked.
"""
from __future__ import annotations

# ── Scaling-pathway declaration (spec §5, standing rule) ─────────────────────
# Stage residency (host | device | either):
#   adjoint assembly (∂R/∂u, ∂R/∂p)  — DEVICE. Same 5-field block system as the
#                                      R0 forward; reuse `exciton_device`
#                                      assembly. R1 introduces NO new nnz-space
#                                      arrays — the transposed operator is the
#                                      converged forward Newton Jacobian Aᵀ.
#   transposed / linear solve        — DEVICE. Workstation (2-D production):
#                                      cuDSS direct / fp32-IR / graph-captured
#                                      Krylov, exactly the merged forward path.
#                                      HERO SCALE (100M-DOF): this MUST be an
#                                      ITERATIVE path, NOT cuDSS direct. #43's
#                                      G4 established a direct-solver CAPACITY
#                                      WALL — cuDSS CANNOT factorize the hero
#                                      (15.15M dofs, 1.6B nnz) on GH200. The
#                                      transposed steady solve at hero scale
#                                      therefore takes the device-FGMRES route
#                                      (#49 direction, consistent with the P2
#                                      projection choice). Do NOT claim cuDSS
#                                      carries the adjoint to the hero.
#   Mode-B checkpoint storage        — HOST, O(steps) converged full states at
#                                      2-D production size. The step count is
#                                      BOUNDED and declared honestly:
#                                      XDD_R1_TRANSIENT_STEP_BUDGET. Long
#                                      horizons at hero scale are the binding
#                                      HOST resource and are a later rung
#                                      (binomial/Revolve NOTED, NOT built).
#   observables + sensitivity rows   — DEVICE (assembled with the forward
#                                      observables).
# 100M-DOF budget: the adjoint is the SAME sparse system as the forward, so the
# #35 device assembly, #38 ChunkedCSR, and #36 fp32-IR machinery carry it
# unchanged — R1 adds no nnz-space arrays. The item declared honestly is the
# transient checkpoint budget (host), NOT the solve; and at hero scale the
# transposed direct solve is off the table (cuDSS capacity wall, #43 G4) — the
# iterative/device-FGMRES path is the pathway.
# Deployment tiers:
#   workstation single-GPU (gpubox)  — ALL R1 gates + inverse demos (2-D scale).
#   single big node (GH200 / NVL4)   — hero rung; iterative transposed solve +
#                                      deferred long-horizon checkpointing.
#   multi-node                       — later; no R1 stage halos.
XDD_R1_TRANSIENT_STEP_BUDGET = 16      # declared bounded Mode-B step count (2-D)

from typing import Protocol, runtime_checkable

import numpy as np

import scipy.sparse as sp

from diffsim.physics.exciton_system import (
    NDOF, IPHI, IN, IP, IXD, IXA,
    _mass_block, _load_block, _carrier_block, _exciton_block, _poisson_block,
    _gp_value, _tau_gp)


def _hist_load_matrix(dm, aq_gp, mu_gp, sig2tau, supg):
    """Nodal→nodal history-LOAD operator ``Hload`` such that

        Hload @ v  ==  _load_block(aq_gp, mu_gp, _gp_value(dm, v), sig2tau, supg)

    for any full nodal vector ``v``.  Mirrors ``make_xdd_carrier_be`` VERBATIM
    (Galerkin ``N_a`` term + SUPG ``τ_M·U·∇N_a`` term, ``U = −aq`` the drift
    velocity) with the GP source expanded through the nodal→GP interpolation
    ``fq = Σ_b N_b v_b``, so the operator is EXACT — not a mass approximation.
    The Mode-B history cotangent coupling contracts with ``Hloadᵀ``.
    """
    h_all = dm.mesh.tree.h()
    rows, cols, vals = [], [], []
    for pv, b in dm.bins.items():
        conn = dm.mesh.conn_of[pv].astype(np.int64)
        ne, nbf = conn.shape
        nqp = b["nqp"]
        N = dm.tables_by_p[pv].N                 # [nqp, nbf]
        dN = dm.tables_by_p[pv].dN               # [nqp, nbf, dim]
        w = dm.tables_by_p[pv].w                 # [nqp]
        eids = dm.mesh.bins[pv]
        he = h_all[eids]                         # [ne]
        jac = (0.5 * he) ** dm.dim               # [ne]
        dscale = 2.0 / he                        # [ne]
        dJxW = w[None, :] * jac[:, None]         # [ne, nqp]
        a = aq_gp[pv].reshape(ne, nqp, dm.dim)   # drift weight at GPs
        tau = _tau_gp(dm, aq_gp, mu_gp, sig2tau, supg)[pv].reshape(ne, nqp)
        # U·∇N_a = −(aq·∇N_a)·dscale       [ne, nqp, nbf]
        Ugw = -np.einsum("eqd,qad->eqa", a, dN) * dscale[:, None, None]
        # test function  Φ_a = N_a + τ_M·Ugw_a   [ne, nqp, nbf]
        Phi = N[None, :, :] + tau[:, :, None] * Ugw
        # Ae[e,a,b] = Σ_q Φ[e,q,a]·N[q,b]·dJxW[e,q]
        Ae = np.einsum("eqa,qb,eq->eab", Phi, N, dJxW)
        rows.append(np.repeat(conn, nbf, axis=1).ravel())
        cols.append(np.tile(conn, (1, nbf)).ravel())
        vals.append(Ae.ravel())
    return sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dm.n_nodes, dm.n_nodes)).tocsr()


def _reduce_full(sysm, R_full):
    """Field-major reduce of a per-field full residual dict to (5*n_free,)."""
    return np.concatenate([np.asarray(sysm.T.T @ R_full[f])
                           for f in range(NDOF)])


@runtime_checkable
class Control(Protocol):
    name: str
    size: int
    def get(self) -> np.ndarray: ...
    def set(self, p: np.ndarray) -> None: ...
    def residual_vjp(self, sysm, state, seed_flat: np.ndarray) -> np.ndarray: ...
    def sensitivity_rows(self, sysm, state) -> np.ndarray: ...


class ClosureControl:
    """A single scalar closure parameter: Langevin prefactor (via zeta) or an
    Onsager dissociation scaling.  The A3 closures are LINEAR in these
    parameters, so the residual derivative is assembled analytically by
    re-running the R̂/k̂ contribution with the parameter-derivative GP field."""

    _SUPPORTED = ("langevin_zeta", "ex_diss_d_scaling", "ex_diss_a_scaling")

    def __init__(self, sysm, param: str):
        if param not in self._SUPPORTED:
            raise ValueError(f"ClosureControl: unknown param {param!r}")
        self.name = param
        self.size = 1
        self._sysm = sysm

    # -- read/write the live parameter on the closure objects ----------------
    def get(self) -> np.ndarray:
        s = self._sysm
        if self.name == "langevin_zeta":
            return np.array([float(s.langevin.zeta)])
        if self.name == "ex_diss_d_scaling":
            return np.array([float(s.onsager.params.ex_diss_d_scaling)])
        return np.array([float(s.onsager.params.ex_diss_a_scaling)])

    def set(self, p: np.ndarray) -> None:
        import dataclasses
        s = self._sysm
        v = float(np.asarray(p).ravel()[0])
        if self.name == "langevin_zeta":
            s.langevin = dataclasses.replace(s.langevin, zeta=v)
        else:
            s.onsager = dataclasses.replace(
                s.onsager, params=s.onsager.params.replace(**{self.name: v}))

    # -- the shared primitive: dR/dp as reduced-space rows (size, 5*n_free) --
    # -- parameter-derivative prefactor WITHOUT dividing by the live param ----
    # Both A3 closures are exactly LINEAR in their control parameter, so
    #   ∂R̂/∂ζ = R̂/ζ = (the ζ-independent prefactor)     [Langevin]
    #   ∂k̂/∂s = k̂/s = (the s-independent prefactor)      [Onsager scaling]
    # The naive R̂/ζ and k̂/s forms NaN if an optimizer drives the control to 0
    # (and LangevinRecombination even rejects ζ≤0 at construction).  DIV-BY-PARAM
    # HARDENING (Task-2 M2 carry): re-evaluate the closure with the parameter
    # factored out (set to 1) to read the prefactor directly — finite at param=0,
    # exact by linearity, and shares the closure's own code path.
    def _langevin_dRhat(self, sysm, state, cl):
        import dataclasses
        lang0 = sysm.langevin
        if float(lang0.zeta) > 1e-8:          # numerically safe: divide as before
            z = float(lang0.zeta)
            return {pv: cl["R"][pv] / z for pv in sysm.dm.bins}
        # near-zero ζ: evaluate R̂ at ζ=1 (the ζ-independent prefactor) directly.
        dm = sysm.dm
        n_gp = _gp_value(dm, state[IN]); p_gp = _gp_value(dm, state[IP])
        lang1 = dataclasses.replace(lang0, zeta=1.0)
        out = {}
        for pv in dm.bins:
            r1, _, _ = lang1(n_gp[pv], p_gp[pv], sysm.dist_gp[pv])
            out[pv] = np.broadcast_to(r1, n_gp[pv].shape).copy()
        return out

    def _onsager_dk(self, sysm, state, cl, fld_key):
        import dataclasses
        scale_key = ("ex_diss_d_scaling" if fld_key == "kd"
                     else "ex_diss_a_scaling")
        sval = float(getattr(sysm.onsager.params, scale_key))
        if abs(sval) > 1e-8:                   # numerically safe: divide as before
            return {pv: cl[fld_key][pv] / sval for pv in sysm.dm.bins}
        # near-zero scaling: evaluate k̂ at scaling=1 (the s-independent prefactor).
        dm = sysm.dm
        gmag = cl["gmag"]
        ons1 = dataclasses.replace(
            sysm.onsager,
            params=sysm.onsager.params.replace(**{scale_key: 1.0}))
        out = {}
        for pv in dm.bins:
            kd_, ka_, _ = ons1(gmag[pv], sysm.dist_gp[pv])
            k1 = kd_ if fld_key == "kd" else ka_
            out[pv] = np.broadcast_to(k1, cl[fld_key][pv].shape).copy()
        return out

    def _dR_dp_full(self, sysm, state) -> np.ndarray:
        dm = sysm.dm
        cl = sysm._closures(state)
        dR = {f: np.zeros(dm.n_nodes) for f in range(NDOF)}
        z_aq = {pv: np.zeros((len(sysm.dist_gp[pv]), dm.dim)) for pv in dm.bins}
        if self.name == "langevin_zeta":
            # R̂ is linear in zeta -> dR̂/dzeta = R̂/zeta at GPs (hardened: no
            # division by ζ when ζ→0, see _langevin_dRhat).
            dRhat = self._langevin_dRhat(sysm, state, cl)
            # carrier rows: source is (D̂ − R̂) entering via −load; exciton feed
            # +R̂.  Mirror residual_full's assembly with dRhat in place of R̂.
            fcar = {pv: -dRhat[pv] for pv in dm.bins}       # d(−R̂)/dzeta
            Fn = _load_block(dm, sysm._aq(cl["gradphi"], sysm.mu_n_gp, -1.0),
                             sysm.mu_n_gp, fcar, sysm._sig2tau(), sysm.supg)
            Fp = _load_block(dm, sysm._aq(cl["gradphi"], sysm.mu_p_gp, +1.0),
                             sysm.mu_p_gp, fcar, sysm._sig2tau(), sysm.supg)
            # residual R[IN] = Kn@n − Fn, so ∂R[IN]/∂zeta = −∂Fn/∂zeta
            dR[IN] = -Fn
            dR[IP] = -Fp
            fex = {pv: dRhat[pv] for pv in dm.bins}          # d(+R̂ feed)/dzeta
            Fxd = _load_block(dm, z_aq, sysm.mu_xd_gp, fex, 0.0, 0.0)
            Fxa = _load_block(dm, z_aq, sysm.mu_xa_gp, fex, 0.0, 0.0)
            dR[IXD] = -Fxd
            dR[IXA] = -Fxa
        else:
            # Onsager scaling: k̂_i linear in the scaling -> dk̂/ds = k̂/s.
            # Dissociation D̂ = k̂_d X̂_D + k̂_a X̂_A enters carrier source +D̂ and
            # the exciton sink (σ_tot,i carries +k̂_i) via K@X.  Only the
            # scaled species' k̂ moves.
            fld_key = "kd" if self.name == "ex_diss_d_scaling" else "ka"
            xkey = "xd_gp" if self.name == "ex_diss_d_scaling" else "xa_gp"
            xfld = IXD if self.name == "ex_diss_d_scaling" else IXA
            # k̂ linear in the scaling -> dk̂/ds = k̂/s (hardened: no division by
            # the scaling when s→0, see _onsager_dk).
            dk = self._onsager_dk(sysm, state, cl, fld_key)
            # carrier source +D̂ contribution d/ds = dk·X̂  (enters −load)
            dDs = {pv: dk[pv] * cl[xkey][pv] for pv in dm.bins}
            Fn = _load_block(dm, sysm._aq(cl["gradphi"], sysm.mu_n_gp, -1.0),
                             sysm.mu_n_gp, dDs, sysm._sig2tau(), sysm.supg)
            Fp = _load_block(dm, sysm._aq(cl["gradphi"], sysm.mu_p_gp, +1.0),
                             sysm.mu_p_gp, dDs, sysm._sig2tau(), sysm.supg)
            dR[IN] = -Fn
            dR[IP] = -Fp
            # exciton sink K@X̂ with σ_tot carrying +k̂_i: dR[xfld]/ds = M(dk·X̂)
            M_dkx = _mass_block(dm, dk) @ state[xfld]
            dR[xfld] = M_dkx
        return _reduce_full(sysm, dR)[None, :]     # (1, 5*n_free)

    def sensitivity_rows(self, sysm, state) -> np.ndarray:
        return self._dR_dp_full(sysm, state)

    def residual_vjp(self, sysm, state, seed_flat: np.ndarray) -> np.ndarray:
        return self._dR_dp_full(sysm, state) @ seed_flat


class MaterialControl:
    """A scalar material/transport parameter that enters the R0 residual terms.

    The control's scalar handle is a multiplier ``s`` about the construction
    value: for a µ field ``s`` scales the GP coefficient (``µ_gp = s·base``, so
    ``s=1`` at construction and ``∂µ_gp/∂s = base``); for a ``τ⁻¹`` scalar the
    handle is ``τ⁻¹`` itself.

    param -> term perturbed (term-to-code, R0 house style):
      "mu_n"       carrier n-row  R[IN]=K_n@n̂−F_n  (µ_n_gp ∝ s → drift+diffusion
                   +SUPG in K_n and the SUPG-consistent load F_n)
      "mu_p"       carrier p-row  R[IP]=K_p@p̂−F_p  (µ_p_gp ∝ s)
      "mu_x_donor" exciton D-row diffusion  R[IXD]=K_xd@X̂_D−F_xd  (µ_xd_gp ∝ s)
      "mu_x_acceptor" exciton A-row diffusion  (µ_xa_gp ∝ s)
      "tau_inv_d"  exciton D-row sink σ_tot,D=σ+τ⁻¹_d+k̂_d diagonal:
                   ∂R[IXD]/∂τ⁻¹_d = M X̂_D  (pure mass × field)
      "tau_inv_a"  exciton A-row sink: ∂R[IXA]/∂τ⁻¹_a = M X̂_A

    µ derivatives — INDEPENDENCE POLICY (so the FD gate is a real, non-vacuous
    check, NOT a tautology):

      * ``mu_x_donor`` / ``mu_x_acceptor`` — the exciton rows have NO SUPG, so
        ∂R/∂s is CLOSED-FORM: the exciton block is σ_tot·M + µ̂_X·K_stiff with
        the load µ-independent, hence ∂R[IX]/∂s = (∂µ̂_X/∂s)·K_stiff @ X̂ =
        _exciton_block(base, σ=0) @ X̂ (pure stiffness with µ̂→base).  This shares
        NO code path with the residual FD gate.

      * ``mu_n`` / ``mu_p`` — the carrier rows carry SUPG whose τ_M is a
        NONLINEAR function of µ.  A closed-form dτ_M/dµ would require mirroring
        the fused carrier Ae/be warp kernels term-by-term; the sanctioned
        fallback is a semi-analytic directional derivative that is DECORRELATED
        from the gate: the analytic uses an internal central step δ=1e-6 while
        the residual FD gate for µ uses eps=1e-4 (see ``MU_GATE_EPS_REL``).  The
        two differences are therefore NOT bit-identical; they agree only to
        O(δ²)+O(eps²) ~1e-8..1e-9, which is what a genuine gate looks like.
        (Task 7's three-way torch-twin is µ's rigorous independent check.)
    """

    # Residual-FD gate step for the µ (carrier) params.  Deliberately DIFFERENT
    # from the analytic internal step (1e-6, see _mu_row_residual callers) so the
    # semi-analytic carrier-µ derivative and its gate are decorrelated, not
    # bit-for-bit identical — the gate can genuinely fail.
    MU_GATE_EPS_REL = 1e-4

    _SCALARS = ("tau_inv_d", "tau_inv_a")
    _MUFIELDS = {"mu_n": "mu_n_gp", "mu_p": "mu_p_gp",
                 "mu_x_donor": "mu_xd_gp", "mu_x_acceptor": "mu_xa_gp"}
    # Permittivity: an ε scale multiplier on the Poisson-block GP field eps_gp.
    # Poisson block is λ²(ε̂ ∇N_b·∇N_a) — LINEAR in ε̂, no SUPG — so the
    # derivative is a clean closed-form Poisson block (no τ_M).  eps_A / eps_D
    # scale the acceptor / donor sub-regions (dist_gp<0 donor, ≥0 acceptor;
    # matches signed_distance_bilayer sign convention).
    _EPSFIELDS = ("eps_A", "eps_D")

    def __init__(self, sysm, param: str):
        if (param not in self._SCALARS and param not in self._MUFIELDS
                and param not in self._EPSFIELDS):
            raise ValueError(f"MaterialControl: unknown param {param!r}")
        self.name = param
        self.size = 1
        self._sysm = sysm
        if param in self._MUFIELDS:
            self._field = self._MUFIELDS[param]
            # snapshot the construction-time µ GP field as the scaling base
            self._base_field = {pv: getattr(sysm, self._field)[pv].copy()
                                for pv in sysm.dm.bins}
        elif param in self._EPSFIELDS:
            # snapshot the construction-time ε GP field + the region mask this
            # param owns (donor for eps_D, acceptor for eps_A).
            self._eps_base = {pv: sysm.eps_gp[pv].copy() for pv in sysm.dm.bins}
            is_acceptor = (param == "eps_A")
            self._eps_mask = {}
            for pv in sysm.dm.bins:
                dist = np.asarray(sysm.dist_gp[pv])
                # dist_gp may be (ngp,) or (ngp, dim); take the scalar sign field
                d0 = dist if dist.ndim == 1 else dist[..., 0]
                self._eps_mask[pv] = (d0 >= 0.0) if is_acceptor else (d0 < 0.0)

    def get(self) -> np.ndarray:
        s = self._sysm
        if self.name in self._SCALARS:
            return np.array([float(getattr(s, self.name))])
        return np.array([1.0])   # µ/ε handle = scale multiplier (1 at ctor)

    def set(self, p: np.ndarray) -> None:
        s = self._sysm
        v = float(np.asarray(p).ravel()[0])
        if self.name in self._SCALARS:
            setattr(s, self.name, v)
        elif self.name in self._MUFIELDS:
            base = self._base_field
            setattr(s, self._field,
                    {pv: v * base[pv] for pv in s.dm.bins})
        else:  # eps_A / eps_D — scale only the owned region's ε̂
            new_eps = {}
            for pv in s.dm.bins:
                e = self._eps_base[pv].copy()
                m = self._eps_mask[pv]
                e[m] = v * self._eps_base[pv][m]
                new_eps[pv] = e
            s.eps_gp = new_eps

    # -- carrier µ-row residual contribution (K@u − load) at a µ-scale multiplier --
    def _mu_row_residual(self, sysm, state, cl, scale):
        """R-contribution of the carrier (mu_n/mu_p) row at µ_gp = scale·base.

        Only the terms that actually depend on the µ field are formed; the rest
        of the residual cancels in the ∂/∂s central difference in _dR_dp_full.
        (Exciton µ and ε use closed-form derivatives — see _dR_dp_full.)
        """
        dm = sysm.dm
        base = self._base_field
        mu = {pv: scale * base[pv] for pv in dm.bins}
        sign = -1.0 if self.name == "mu_n" else +1.0
        fld = IN if self.name == "mu_n" else IP
        aq = {pv: sign * mu[pv][:, None] * cl["gradphi"][pv] for pv in dm.bins}
        K = _carrier_block(dm, aq, mu, sysm.sigma, sysm._sig2tau(), sysm.supg)
        # SUPG-consistent load carries the SAME µ-dependent aq/τ_M.
        Dhat = {pv: cl["kd"][pv] * cl["xd_gp"][pv]
                    + cl["ka"][pv] * cl["xa_gp"][pv] for pv in dm.bins}
        s_carr = {pv: Dhat[pv] - cl["R"][pv] + sysm._hist_gp(fld, pv)
                  for pv in dm.bins}
        F = _load_block(dm, aq, mu, s_carr, sysm._sig2tau(), sysm.supg)
        return fld, K @ state[fld] - F

    def _dR_dp_full(self, sysm, state) -> np.ndarray:
        dm = sysm.dm
        dR = {f: np.zeros(dm.n_nodes) for f in range(NDOF)}
        if self.name == "tau_inv_d":
            one = {pv: np.ones(len(sysm.dist_gp[pv])) for pv in dm.bins}
            dR[IXD] = _mass_block(dm, one) @ state[IXD]
        elif self.name == "tau_inv_a":
            one = {pv: np.ones(len(sysm.dist_gp[pv])) for pv in dm.bins}
            dR[IXA] = _mass_block(dm, one) @ state[IXA]
        elif self.name in ("eps_A", "eps_D"):
            # CLOSED-FORM: R[IPHI] = λ²ε̂K φ̂ − M(p̂−n̂).  ε̂ = base + s·(mask·base),
            # so ∂R[IPHI]/∂s = λ² (∂ε̂/∂s ∇N_b·∇N_a) φ̂ with ∂ε̂/∂s = mask·base.
            deps = {pv: np.where(self._eps_mask[pv], self._eps_base[pv], 0.0)
                    for pv in dm.bins}
            dKphi = _poisson_block(dm, deps, sysm.lam2)
            dR[IPHI] = dKphi @ state[IPHI]
        elif self.name in ("mu_x_donor", "mu_x_acceptor"):
            # CLOSED-FORM: exciton rows have NO SUPG.  K_xi = σ_tot·M + µ̂_X·K_stiff
            # and the load is µ-independent, so ∂R[IXi]/∂s = (∂µ̂_X/∂s)·K_stiff@X̂.
            # _exciton_block with σ_gp=0 is exactly the µ̂-weighted stiffness; with
            # µ̂→base (=∂µ̂_X/∂s) it is the closed-form derivative operator.  Shares
            # no code with the residual FD gate → non-vacuous.
            fld = IXD if self.name == "mu_x_donor" else IXA
            base = self._base_field
            zero_sig = {pv: np.zeros(len(sysm.dist_gp[pv])) for pv in dm.bins}
            dK = _exciton_block(dm, base, zero_sig)
            dR[fld] = dK @ state[fld]
        else:
            # mu_n / mu_p — carrier rows carry SUPG (τ_M nonlinear in µ).
            # Semi-analytic directional derivative, DECORRELATED from the gate:
            # analytic internal step δ=1e-6, the µ residual FD gate uses eps=1e-4
            # (MU_GATE_EPS_REL).  Not bit-identical → the gate can genuinely fail.
            cl = sysm._closures(state)
            delta = 1e-6
            fld, Rp = self._mu_row_residual(sysm, state, cl, 1.0 + delta)
            _,  Rm = self._mu_row_residual(sysm, state, cl, 1.0 - delta)
            dR[fld] = (Rp - Rm) / (2.0 * delta)
        return _reduce_full(sysm, dR)[None, :]

    def sensitivity_rows(self, sysm, state) -> np.ndarray:
        return self._dR_dp_full(sysm, state)

    def residual_vjp(self, sysm, state, seed_flat: np.ndarray) -> np.ndarray:
        return self._dR_dp_full(sysm, state) @ seed_flat


class XDDSteadyAdjoint:
    """Mode A — implicit steady-state adjoint for the XDD 5-field system.

    At a converged state R(u;p)=0, A=∂R/∂u is the Newton Jacobian (the R0
    assembler ``assemble_newton_system``).  We solve Aᵀλ=∂J/∂u ONCE and reuse
    the transpose factorization across observable seeds; the per-control
    gradient is dJ/dp = ∂J/∂p − λᵀ ∂R/∂p (the discrete IFT, phasefield idiom).

    PHYSICAL-`u` PARAMETRISATION.  The host assembler returns the Newton
    Jacobian in the *log-increment* convention for the carrier rows
    (``carrier_vars='log'``): its carrier column-blocks are right-multiplied by
    ``diag(n̂_free)`` so the increment solved for is δ(ln n̂).  But the QoI seed
    ``∂J/∂u`` and the controls' ``∂R/∂p`` are expressed w.r.t. the PHYSICAL
    fields (n̂, p̂, X̂, φ̂ — reduced with ``sysm.T.T``).  The adjoint identity
    demands one consistent parametrisation, so we UNDO the log scaling to obtain
    ``A_phys = ∂R/∂u_physical`` (right-multiply carrier columns by ``1/n̂_free``)
    and re-impose the Dirichlet identity rows (which the unscaling would perturb
    on carrier Dirichlet columns).  A_phys is what we factor and transpose.
    """

    def __init__(self, sysm, controls):
        self.sysm = sysm
        self.controls = list(controls)
        self._A = None          # A_phys (physical-u Newton Jacobian), CSR
        self._lu = None         # cached splu of A_phys.T
        self._dir_rows = None   # flat reduced indices of Dirichlet strong rows

    # -- flat reduced indices of the Dirichlet (strong identity) rows --------
    def _dirichlet_rows(self):
        if self._dir_rows is not None:
            return self._dir_rows
        sysm = self.sysm
        nf = sysm.n_free
        node_to_free = -np.ones(sysm.dm.n_nodes, np.int64)
        node_to_free[sysm.free] = np.arange(nf)
        rows = []
        for field, (nodes, _vals) in sysm.dirichlet.items():
            for nid in nodes:
                fi = node_to_free[nid]
                if fi >= 0:
                    rows.append(field * nf + fi)
        self._dir_rows = np.asarray(rows, np.int64)
        return self._dir_rows

    # -- assemble the STEADY PHYSICAL-u Jacobian at the converged state -------
    def _physical_jacobian(self, state):
        import scipy.sparse as sp
        sysm = self.sysm
        # STEADY operator: the marched state carries a stale BDF σ=1/dt and a
        # history load from the last transient step.  The implicit steady
        # adjoint linearises R_steady(u;p)=0, so reset σ=0 / hist=None before
        # assembling (else A is the transient Jacobian and the IFT is wrong).
        sysm.sigma = 0.0
        sysm.hist = None
        # Match the Newton driver: the reduce/eliminate log-scaling + Dirichlet
        # rows read sysm._current_state, so pin it to the converged state.
        sysm._current_state = state
        A, _ = sysm.assemble_newton_system(state)
        A = A.tocsr()
        if not getattr(sysm, "_log_carriers", False):
            return A
        # Undo the carrier log-scaling A_log = A_phys @ diag(scale):
        #   A_phys = A_log @ diag(1/scale), scale = n̂_free on IN/IP blocks, 1 else.
        nf = sysm.n_free
        free = sysm.free
        inv = np.ones(NDOF * nf)
        for field in (IN, IP):
            cur = np.asarray(state[field])[free]
            inv[field * nf:(field + 1) * nf] = 1.0 / cur
        A = (A @ sp.diags(inv)).tolil()
        # Re-impose Dirichlet identity rows (unscaling perturbed the diagonal
        # entry of carrier Dirichlet columns; the strong rows are u-independent).
        node_to_free = -np.ones(sysm.dm.n_nodes, np.int64)
        node_to_free[free] = np.arange(nf)
        for field, (nodes, vals) in sysm.dirichlet.items():
            for nid in nodes:
                fi = node_to_free[nid]
                if fi < 0:
                    continue
                row = field * nf + fi
                A.rows[row] = [row]
                A.data[row] = [1.0]
        return A.tocsr()

    def factorize(self, state):
        from scipy.sparse.linalg import splu
        self._A = self._physical_jacobian(state)
        self._lu = splu(self._A.T.tocsc())
        return self

    def _solve_T(self, rhs):
        if self._lu is None:
            from diffsim.sbm.adjoint import solve_adjoint
            return solve_adjoint(self._A, np.asarray(rhs, np.float64))
        return self._lu.solve(np.asarray(rhs, np.float64))

    def _dRdp_eliminated(self, control, state):
        """The control's reduced ∂R/∂p (size, 5*n_free) with the Dirichlet
        strong-row columns zeroed.  The forward assembler ELIMINATES the
        Dirichlet rows (identity rows, BC value p-independent), so their raw
        residual-derivative entries are not part of R_steady(u;p)=0 and must be
        dropped before contracting with λ (verified: the un-eliminated rows
        corrupt the gradient by ~40%)."""
        rows = np.array(control.sensitivity_rows(self.sysm, state), float)
        rows[:, self._dirichlet_rows()] = 0.0
        return rows

    def gradient(self, state, qoi) -> dict:
        if self._A is None:
            self.factorize(state)
        lam = self._solve_T(qoi.dJ_du(self.sysm, state))
        out = {}
        for c in self.controls:
            djdp = qoi.dJ_dp(self.sysm, state, c)               # (size,)
            out[c.name] = djdp - (self._dRdp_eliminated(c, state) @ lam)
        return out

    def sensitivity_rows(self, state, obs_seeds) -> np.ndarray:
        """obs_seeds: (n_obs, 5*n_free) array of ∂o/∂u.  Returns ∂o/∂p as
        (n_obs, n_p_total), p-columns ordered by self.controls.  Reuses the SAME
        factored Aᵀ across all seeds (requirement 2 / R3 Fisher rows).

        ∂o/∂p = ∂o/∂p_explicit − (∂o/∂u) A⁻¹ ∂R/∂p = −(∂R/∂p) λ_o with the
        per-seed adjoint λ_o = A⁻ᵀ (∂o/∂u)ᵀ (explicit ∂o/∂p is 0 for the
        pure-state observables this map serves)."""
        if self._A is None:
            self.factorize(state)
        obs_seeds = np.atleast_2d(np.asarray(obs_seeds, np.float64))
        n_obs = obs_seeds.shape[0]
        dRdp = [self._dRdp_eliminated(c, state) for c in self.controls]
        cols = sum(c.size for c in self.controls)
        rows = np.zeros((n_obs, cols))
        for k in range(n_obs):
            lam = self._solve_T(obs_seeds[k])
            off = 0
            for c, dr in zip(self.controls, dRdp):
                rows[k, off:off + c.size] = -(dr @ lam)
                off += c.size
        return rows


class XDDTransientAdjoint:
    """Mode B — taped transient (frozen-dt) adjoint over a checkpointed march.

    Re-targets the ``sbm/transient_adjoint.py::TransientAdjoint`` record-then-
    reverse pattern to the XDD 5-field stepper.  For a trajectory functional
    ``J = Σ_n j(u_n)`` over the ``march_with_checkpoints`` tape, the reverse
    sweep solves per step

        Aₙᵀ λₙ = (∂J/∂uₙ)  +  pendingₙ

    with ``Aₙ = ∂Rₙ/∂u`` REBUILT at the recorded state under the recorded σ /
    history / generation (dt FROZEN — the backward pass never re-adapts), then
    couples each λₙ back to the BDF history steps via

        pending[n−k][f] += (bdf_coeff/dtₙ) · (T.T @ Hload_fᵀ @ (T @ λₙ_f))

    on the TIME-STEPPED fields ``{IN, IP, IXD, IXA}`` only (φ̂ has no time term —
    ``step_bdf`` zeros ``hist_full[IPHI]``).  ``Hload_f`` is the exact history-
    load operator (SUPG-consistent for carriers).  BDF1: coeff ``1/dt`` on
    ``uⁿ``.  BDF2 (``hist=(2uⁿ−0.5uⁿ⁻¹)/dt``): coeffs ``2/dt`` on ``uⁿ`` and
    ``−0.5/dt`` on ``uⁿ⁻¹`` (read directly off ``step_bdf``).  Per-step control
    VJPs accumulate ``−λₙᵀ ∂Rₙ/∂p`` into each control's gradient.

    PHYSICAL-``u`` PARAMETRISATION.  Like Mode A, the recorded ``Aₙ`` is the
    log-increment Newton Jacobian for the carrier rows; the seeds and control
    ``∂R/∂p`` are physical, so we UNDO the carrier log-scaling and re-impose the
    Dirichlet identity rows to obtain ``A_phys = ∂R/∂u_physical``.  The Dirichlet
    strong rows are BC identities (u-independent, not physics residuals): λ is
    zeroed there before the history coupling, and the controls' ∂R/∂p Dirichlet
    columns are dropped (both the Mode-A conventions).
    """

    _TFIELDS = (IN, IP, IXD, IXA)

    def __init__(self, sysm, controls, steps):
        self.sysm = sysm
        self.controls = list(controls)
        self.steps = steps
        self._dir_rows = None

    # -- flat reduced indices of the Dirichlet (strong identity) rows --------
    def _dirichlet_rows(self):
        if self._dir_rows is not None:
            return self._dir_rows
        sysm = self.sysm
        nf = sysm.n_free
        node_to_free = -np.ones(sysm.dm.n_nodes, np.int64)
        node_to_free[sysm.free] = np.arange(nf)
        rows = []
        for field, (nodes, _vals) in sysm.dirichlet.items():
            for nid in nodes:
                fi = node_to_free[nid]
                if fi >= 0:
                    rows.append(field * nf + fi)
        self._dir_rows = np.asarray(rows, np.int64)
        return self._dir_rows

    # -- re-establish the recorded transient σ / history / generation --------
    def _restore_step(self, rec):
        """Pin sysm to the recorded step's frozen σ / BDF history / generation.

        The history is rebuilt from the recorded prev/prev2 at the recorded dt
        and BDF order — exactly ``step_bdf``'s construction, frozen (the driver
        never calls ``_dt_schedule`` on the backward pass)."""
        s = self.sysm
        dt = rec["dt_hat"]
        s.sigma = rec["sigma"]
        if rec["order"] == 1 or rec["prev2"] is None:
            hist_full = {f: (1.0 / dt) * rec["prev"][f] for f in range(NDOF)}
        else:
            hist_full = {f: (2.0 * rec["prev"][f] - 0.5 * rec["prev2"][f]) / dt
                         for f in range(NDOF)}
        hist_full[IPHI] = np.zeros(s.dm.n_nodes)
        s.hist = {f: _gp_value(s.dm, hist_full[f]) for f in range(NDOF)}
        s.set_generation(rec["gd"], rec["ga"])
        s._current_state = rec["state"]

    # -- physical-u Jacobian at the recorded (transient) step ----------------
    def _physical_jacobian(self, rec):
        sysm = self.sysm
        self._restore_step(rec)
        state = rec["state"]
        A, _ = sysm.assemble_newton_system(state)
        A = A.tocsr()
        if not getattr(sysm, "_log_carriers", False):
            return A
        nf = sysm.n_free
        free = sysm.free
        inv = np.ones(NDOF * nf)
        for field in (IN, IP):
            cur = np.asarray(state[field])[free]
            inv[field * nf:(field + 1) * nf] = 1.0 / cur
        A = (A @ sp.diags(inv)).tolil()
        node_to_free = -np.ones(sysm.dm.n_nodes, np.int64)
        node_to_free[free] = np.arange(nf)
        for field, (nodes, vals) in sysm.dirichlet.items():
            for nid in nodes:
                fi = node_to_free[nid]
                if fi < 0:
                    continue
                row = field * nf + fi
                A.rows[row] = [row]
                A.data[row] = [1.0]
        return A.tocsr()

    def _solve_T(self, A, rhs):
        from scipy.sparse.linalg import splu
        return splu(A.T.tocsc()).solve(np.asarray(rhs, np.float64))

    # -- history-load operators for the recorded step's time-stepped fields --
    def _hist_load_ops(self, rec):
        """Per-field exact history-load matrix Hload_f at the recorded state /
        frozen σ.  Carriers carry the SUPG-consistent drift/τ_M (sig2tau=(2σ)²);
        excitons are pure mass-weighted (z_aq, supg=0)."""
        s = self.sysm
        dm = s.dm
        cl = s._closures(rec["state"])
        s2t = (2.0 * s.sigma) ** 2
        aq_n = s._aq(cl["gradphi"], s.mu_n_gp, -1.0)
        aq_p = s._aq(cl["gradphi"], s.mu_p_gp, +1.0)
        z_aq = {pv: np.zeros((len(s.dist_gp[pv]), dm.dim)) for pv in dm.bins}
        return {
            IN:  _hist_load_matrix(dm, aq_n, s.mu_n_gp, s2t, s.supg),
            IP:  _hist_load_matrix(dm, aq_p, s.mu_p_gp, s2t, s.supg),
            IXD: _hist_load_matrix(dm, z_aq, s.mu_xd_gp, 0.0, 0.0),
            IXA: _hist_load_matrix(dm, z_aq, s.mu_xa_gp, 0.0, 0.0),
        }

    def _dRdp_eliminated(self, control, state):
        rows = np.array(control.sensitivity_rows(self.sysm, state), float)
        rows[:, self._dirichlet_rows()] = 0.0
        return rows

    def gradient(self, dJdx_list, qoi=None) -> dict:
        s = self.sysm
        nf = s.n_free
        T = s.T
        N = len(self.steps)
        grads = {c.name: np.zeros(c.size) for c in self.controls}
        pending = [np.zeros(NDOF * nf) for _ in range(N)]
        dir_rows = self._dirichlet_rows()
        for n in range(N - 1, -1, -1):
            rec = self.steps[n]
            A = self._physical_jacobian(rec)          # restores σ/hist/gen too
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = self._solve_T(A, rhs)
            # strong Dirichlet rows are BC identities, not physics residuals:
            # their cotangent does not propagate into ∂R/∂p or the history.
            lam[dir_rows] = 0.0
            # per-step control VJP: −λᵀ ∂Rₙ/∂p
            for c in self.controls:
                grads[c.name] -= self._dRdp_eliminated(c, rec["state"]) @ lam
            # BDF history cotangent to earlier steps (frozen dt / order)
            dt = rec["dt_hat"]
            if rec["order"] == 1 or rec["prev2"] is None:
                coeffs = [(1, 1.0 / dt)]
            else:
                coeffs = [(1, 2.0 / dt), (2, -0.5 / dt)]
            if any(n - k >= 0 for k, _ in coeffs):
                Hops = self._hist_load_ops(rec)
                for k, ck in coeffs:
                    if n - k < 0:
                        continue
                    for f in self._TFIELDS:
                        lam_f = lam[f * nf:(f + 1) * nf]
                        # (bdf_coeff)·Tᵀ Hload_fᵀ T λ_f  (physical-u space)
                        contrib = ck * (T.T @ (Hops[f].T @ (T @ lam_f)))
                        pending[n - k][f * nf:(f + 1) * nf] += np.asarray(contrib)
        return grads

    def sensitivity_rows(self, obs_seed_list) -> np.ndarray:
        """Requirement-2 transient rows: reuse ``gradient`` per observable seed
        list, stacking ∂o/∂p across controls (columns ordered by controls)."""
        cols = sum(c.size for c in self.controls)
        out = np.zeros((len(obs_seed_list), cols))
        for i, seeds in enumerate(obs_seed_list):
            g = self.gradient(seeds)
            off = 0
            for c in self.controls:
                out[i, off:off + c.size] = g[c.name]
                off += c.size
        return out


class IlluminationControl:
    """Generation control.

    ``mode='scalar'``: overall intensity (size 1) — a scalar multiplier on the
    base generation field.  ``mode='vector'``: a per-height-band piecewise
    illumination profile (size ``n_bands``) — the forward-compat requirement-1
    VECTOR control, exposing ``∂R/∂p`` as the ``(n_bands, 5·n_free)`` Jacobian.

    Generation enters ONLY the exciton rows via the exciton feed
    ``f_xi = Ĝ_i + R̂ + hist`` (load block), so the residual derivative is a pure
    exciton-row load:  ``∂R[IXD]/∂pⱼ = −_load_block(…, ∂ĝ_d/∂pⱼ, …)`` and
    likewise for IXA, with ``∂ĝ/∂pⱼ`` = base generation restricted to band ``j``
    (vector) or the whole base generation (scalar).  The load's µ_X field and the
    (zero) advection/SUPG match ``residual_full``'s exciton assembly exactly.
    """

    def __init__(self, sysm, dist_gp, gen, mode="scalar", n_bands=1, h_axis=1):
        self._sysm = sysm
        self._gen = gen
        self.name = "illumination"
        self.mode = mode
        # base generation GP fields (the current set_generation values)
        self._gd0 = {pv: sysm._gd[pv].copy() for pv in sysm.dm.bins}
        self._ga0 = {pv: sysm._ga[pv].copy() for pv in sysm.dm.bins}
        if mode == "scalar":
            self.size = 1
            self._p = np.array([1.0])
            self._masks = None
        elif mode == "vector":
            self.size = int(n_bands)
            self._p = np.ones(self.size)
            self._masks = self._band_masks(sysm, h_axis, n_bands)
        else:
            raise ValueError(f"IlluminationControl: unknown mode {mode!r}")

    def _band_masks(self, sysm, h_axis, n_bands):
        """GP nondim height per bin -> band index -> per-band boolean masks."""
        dm = sysm.dm
        coords = dm.mesh.node_coords
        hc = coords[:, h_axis]
        lo, hi = hc.min(), hc.max()
        hhat_nodal = (hc - lo) / max(hi - lo, 1e-30)
        hgp = _gp_value(dm, hhat_nodal)             # {pv: ndarray[ngp]}
        masks = {}
        edges = np.linspace(0.0, 1.0, n_bands + 1)
        for pv in dm.bins:
            b = np.clip(np.digitize(hgp[pv], edges[1:-1]), 0, n_bands - 1)
            masks[pv] = [(b == j) for j in range(n_bands)]
        return masks

    def get(self) -> np.ndarray:
        return self._p.copy()

    def set(self, p: np.ndarray) -> None:
        p = np.asarray(p, float).ravel()
        self._p = p.copy()
        dm = self._sysm.dm
        # (no-op handle-keep removed — self._sysm is already an attribute)
        if self.mode == "scalar":
            gd = {pv: p[0] * self._gd0[pv] for pv in dm.bins}
            ga = {pv: p[0] * self._ga0[pv] for pv in dm.bins}
        else:
            gd = {}; ga = {}
            for pv in dm.bins:
                w = np.zeros_like(self._gd0[pv])
                for j, m in enumerate(self._masks[pv]):
                    w[m] = p[j]
                gd[pv] = w * self._gd0[pv]
                ga[pv] = w * self._ga0[pv]
        self._sysm.set_generation(gd, ga)

    def _dR_dp_full(self, sysm, state) -> np.ndarray:
        dm = sysm.dm
        z_aq = {pv: np.zeros((len(sysm.dist_gp[pv]), dm.dim)) for pv in dm.bins}
        rows = np.zeros((self.size, NDOF * sysm.n_free))
        for j in range(self.size):
            if self.mode == "scalar":
                dgd = {pv: self._gd0[pv] for pv in dm.bins}
                dga = {pv: self._ga0[pv] for pv in dm.bins}
            else:
                dgd = {pv: np.where(self._masks[pv][j], self._gd0[pv], 0.0)
                       for pv in dm.bins}
                dga = {pv: np.where(self._masks[pv][j], self._ga0[pv], 0.0)
                       for pv in dm.bins}
            dR = {f: np.zeros(dm.n_nodes) for f in range(NDOF)}
            # R[IXi] = K_xi@X̂ − F_xi,  F_xi = _load_block(z_aq, µ_X, Ĝ_i+…, 0, 0)
            Fxd = _load_block(dm, z_aq, sysm.mu_xd_gp, dgd, 0.0, 0.0)
            Fxa = _load_block(dm, z_aq, sysm.mu_xa_gp, dga, 0.0, 0.0)
            dR[IXD] = -Fxd
            dR[IXA] = -Fxa
            rows[j] = _reduce_full(sysm, dR)
        return rows

    def sensitivity_rows(self, sysm, state) -> np.ndarray:
        return self._dR_dp_full(sysm, state)

    def residual_vjp(self, sysm, state, seed_flat: np.ndarray) -> np.ndarray:
        return self._dR_dp_full(sysm, state) @ seed_flat
