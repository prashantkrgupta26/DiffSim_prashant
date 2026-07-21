"""SP-1 R1 — differentiable XDD adjoint machinery.

The Control interface (the unit of "a thing we differentiate with respect to"),
the three R1 control classes, and the two adjoint drivers (implicit steady-state
Mode A, taped frozen-dt transient Mode B) specialised to the XDD 5-field system.
Reuses the R0 assembler (exciton_system / exciton_device), the phasefield IFT
reverse idiom, the sbm transient-adjoint checkpointing lineage, and the
splu(A.T)/solve_linear transposed solve — nothing is forked.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from diffsim.physics.exciton_system import (
    NDOF, IPHI, IN, IP, IXD, IXA,
    _mass_block, _load_block, _carrier_block, _exciton_block, _poisson_block,
    _gp_value)


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
    def _dR_dp_full(self, sysm, state) -> np.ndarray:
        dm = sysm.dm
        cl = sysm._closures(state)
        dR = {f: np.zeros(dm.n_nodes) for f in range(NDOF)}
        z_aq = {pv: np.zeros((len(sysm.dist_gp[pv]), dm.dim)) for pv in dm.bins}
        if self.name == "langevin_zeta":
            zeta = float(sysm.langevin.zeta)
            # R̂ is linear in zeta -> dR̂/dzeta = R̂/zeta at GPs
            dRhat = {pv: cl["R"][pv] / zeta for pv in dm.bins}
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
            sval = float(self.get()[0])
            dk = {pv: cl[fld_key][pv] / sval for pv in dm.bins}
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
