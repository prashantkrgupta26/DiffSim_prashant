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
    _mass_block, _load_block)


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
