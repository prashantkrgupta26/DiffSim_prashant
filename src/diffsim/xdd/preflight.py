"""XDD preflight — scales report + physics-vs-mesh feasibility checks.

The lesson of Block E, made mechanical.  Blocks A–E repeatedly hit the same
wall: the singularly-perturbed Poisson operator λ²∇²φ̂ has a Debye boundary
layer far finer than any feasible uniform-octree element, so the BDF march
either blows φ̂ up or never fires the steady criterion (E1(ii), E2, E5, and
test_xdd_run._resolvable_bilayer all record it).  `preflight()` predicts that
failure BEFORE the march wastes wall-clock, and reports the fix (raise the mesh
level, or drop to the marchable reduced drive, or use Scharfetter–Gummel).

Five reports, all from measured Block-E evidence:

  1. **Scales report** — the nondimensionalisation table (x0, φ0, t0, U0, J0,
     λ², γ0) + the derived physical Debye length.  Informational.
  2. **Debye vs mesh** — the load-bearing check.  Debye_hat = √λ² must be
     resolved by the element size h_hat = 1/2^level.  Bands from the measured
     wall: physical λ²≈2.2e-5 (Debye_hat≈0.0047) at L3 (h=0.125) is 26×
     under-resolved → φ̂ blows to ±300, march collapses (E1(ii): 1 accepted /
     47 rejected).  Marchable λ²=1e-1 (Debye_hat≈0.316) at L3 resolves it.
  3. **Interface width vs mesh** — the tanh interface half-width must span ≥ ~1
     element or the dissociation source is a sub-grid delta (E1 dissociation
     localization).
  4. **dt0 stiffness** — the σ=1/Δt̂ BDF regularisation must dominate the fast
     recombination/dissociation rate at the first step, else Newton diverges
     cold (B4/B5 it-1 blow-up).
  5. **Memory forecast** — the 2.2 GB/M-dof model (measured E5: 2228 MiB at
     330,245 dofs, cuDSS) + cuDSS workspace headroom.

`strict=True` raises on any FAIL; otherwise the report is returned (and printed)
and the run proceeds (the caller decides).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

# Physical constants (match params.py / DDFields.h).
_Q = 1.602176634e-19
_EPS0 = 8.8541878128e-12


# ── Verdict bands (measured Block-E evidence) ─────────────────────────────────
# Debye_hat / h ratio: ≥1 element resolves the layer (marchable L3: 0.316/0.125
# = 2.5 → PASS).  Below ~1 the operator is under-resolved; physical L3 gives
# 0.0047/0.125 = 0.038 → the measured collapse.  WARN band between.
DEBYE_H_PASS = 1.0        # ≥1 element per Debye length
DEBYE_H_WARN = 0.25       # below → FAIL (physical regime lands at 0.038)

# Interface half-width vs h: ≥1 element = resolved source; <0.5 = sub-grid delta.
IFACE_H_PASS = 1.0
IFACE_H_WARN = 0.5

# dt0 stiffness: σ=1/Δt̂ vs the fastest nondim rate (recomb/dissoc/decay).
# σ ≥ rate → the BDF mass term regularises the cold Newton (B4/B5 lesson).
DT_STIFF_PASS = 1.0       # σ ≥ max_rate
DT_STIFF_WARN = 0.1       # σ ≥ 0.1·max_rate; below → FAIL (cold-start blow-up)

# Memory model: measured 2228 MiB / 330,245 dofs (E5 cuDSS) = 6.75e-3 MiB/dof.
GIB_PER_MDOF = 2228.0 / 330245.0 * 1e6 / 1024.0   # MiB per dof → scaled below
MIB_PER_DOF = 2228.0 / 330245.0                    # ≈ 6.75e-3 MiB/dof (measured)


@dataclass
class Check:
    name: str
    verdict: str            # "PASS" | "WARN" | "FAIL" | "INFO"
    detail: str
    value: Optional[float] = None


@dataclass
class PreflightReport:
    scales: dict = field(default_factory=dict)
    checks: List[Check] = field(default_factory=list)

    @property
    def worst(self) -> str:
        order = {"INFO": 0, "PASS": 1, "WARN": 2, "FAIL": 3}
        w = max((order[c.verdict] for c in self.checks), default=0)
        return {v: k for k, v in order.items()}[w]

    @property
    def failed(self) -> bool:
        return any(c.verdict == "FAIL" for c in self.checks)

    def format(self) -> str:
        lines = ["── XDD preflight ──────────────────────────────────────────"]
        s = self.scales
        lines.append("scales:")
        for k in ("x0", "phi0", "t0", "U0", "J0", "lambda2", "gamma0",
                  "debye_length_m", "debye_hat"):
            if k in s:
                lines.append(f"    {k:16s} = {s[k]:.4e}")
        lines.append("checks:")
        for c in self.checks:
            v = f" ({c.value:.4e})" if c.value is not None else ""
            lines.append(f"    [{c.verdict:4s}] {c.name:22s} {c.detail}{v}")
        lines.append(f"  → overall: {self.worst}")
        lines.append("───────────────────────────────────────────────────────────")
        return "\n".join(lines)


def preflight(
    params,
    *,
    level: int,
    lam2: Optional[float] = None,
    dt0_hat: float = 1e-8,
    max_rate_hat: Optional[float] = None,
    interface_hat: Optional[float] = None,
    device_mem_gib: Optional[float] = None,
    p: int = 1,
    strict: bool = False,
    verbose: bool = True,
) -> PreflightReport:
    """Run the XDD preflight checks for a planned run.

    Parameters
    ----------
    params : XDDParams
        Parameter set (for scales, interface_thk, height).
    level : int
        Uniform-octree refinement level → element size h_hat = 1/2^level.
    lam2 : float or None
        The λ² the run will USE (may be the reduced marchable value).  If None,
        the physical params.scales().lambda2 is used (and the check reports the
        physical-drive verdict).
    dt0_hat : float
        Planned first nondim time step.
    max_rate_hat : float or None
        Fastest nondim rate (recomb/dissoc/decay).  If None, estimated from the
        exciton decay rate t0/τ_x (a lower bound on the true stiffness).
    interface_hat : float or None
        Interface half-width in device-length units.  If None, derived from
        params.interface_thk / height.
    device_mem_gib : float or None
        Available device memory (GiB) for the memory forecast.  If None, the
        forecast is INFO-only (no headroom verdict).
    p : int
        Basis degree (nodes per edge = p·2^level + 1).
    strict : bool
        Raise RuntimeError on any FAIL.
    """
    s = params.scales()
    h_hat = 1.0 / (2 ** level)
    lam2_use = lam2 if lam2 is not None else s.lambda2
    debye_hat = math.sqrt(lam2_use)
    debye_m = debye_hat * s.x0

    rep = PreflightReport()
    rep.scales = dict(
        x0=s.x0, phi0=s.phi0, t0=s.t0, U0=s.U0, J0=s.J0,
        lambda2=lam2_use, gamma0=s.gamma0,
        debye_length_m=debye_m, debye_hat=debye_hat, h_hat=h_hat,
    )

    # (1) scales report is the .scales dict; add an INFO line.
    rep.checks.append(Check(
        "scales", "INFO",
        f"λ²={lam2_use:.3e} Debye={debye_m*1e9:.3f} nm h_hat={h_hat:.4f} "
        f"(level {level}, p{p})"))

    # (2) Debye vs mesh — THE check.
    ratio = debye_hat / h_hat
    if ratio >= DEBYE_H_PASS:
        v = "PASS"
        d = f"Debye layer resolved ({ratio:.2f} elements/Debye)"
    elif ratio >= DEBYE_H_WARN:
        v = "WARN"
        d = (f"Debye layer marginally resolved ({ratio:.2f} el/Debye); "
             f"raise level or reduce drive")
    else:
        v = "FAIL"
        d = (f"Debye layer under-resolved ({ratio:.3f} el/Debye ≪ 1) — the "
             f"march will blow φ̂ up (E1(ii) wall). Raise level, reduce λ² "
             f"(marchable regime), or use Scharfetter–Gummel")
    rep.checks.append(Check("debye_vs_mesh", v, d, value=ratio))

    # (3) interface width vs mesh.
    if interface_hat is None:
        interface_hat = 0.5 * params.interface_thk / s.x0   # tanh half-width
    iratio = interface_hat / h_hat
    if iratio >= IFACE_H_PASS:
        v = "PASS"; d = f"interface source resolved ({iratio:.2f} el/half-width)"
    elif iratio >= IFACE_H_WARN:
        v = "WARN"
        d = (f"interface source marginally resolved ({iratio:.2f} el); "
             f"dissociation localizes to few nodes")
    else:
        v = "FAIL"
        d = (f"interface source sub-grid ({iratio:.3f} el ≪ 1) — dissociation "
             f"is a mesh-delta; raise level or widen interface_thk")
    rep.checks.append(Check("interface_vs_mesh", v, d, value=iratio))

    # (4) dt0 stiffness — σ=1/Δt̂ vs the fastest nondim rate.
    if max_rate_hat is None:
        # exciton decay rate t0/τ_x as a floor (recomb/dissoc are usually larger).
        max_rate_hat = s.t0 / min(params.tau_x_donor, params.tau_x_acceptor)
    sigma = 1.0 / dt0_hat
    sratio = sigma / max_rate_hat if max_rate_hat > 0 else math.inf
    if sratio >= DT_STIFF_PASS:
        v = "PASS"; d = f"σ=1/dt0 dominates stiffness ({sratio:.2f}× fastest rate)"
    elif sratio >= DT_STIFF_WARN:
        v = "WARN"
        d = (f"σ=1/dt0 comparable to stiffness ({sratio:.2f}×); cold Newton may "
             f"need more iterations")
    else:
        v = "FAIL"
        d = (f"σ=1/dt0 ≪ stiffness ({sratio:.3f}×) — cold Newton diverges "
             f"(B4/B5 it-1 blow-up); reduce dt0")
    rep.checks.append(Check("dt0_stiffness", v, d, value=sratio))

    # (5) memory forecast — 2.2 GB/M-dof measured (E5 cuDSS) + workspace.
    nodes_edge = p * (2 ** level) + 1
    n_nodes = nodes_edge ** 2                 # 2-D
    n_dofs = 5 * n_nodes                       # 5 fields (φ, n, p, X_D, X_A)
    mem_mib = MIB_PER_DOF * n_dofs
    rep.scales["forecast_dofs"] = float(n_dofs)
    rep.scales["forecast_mem_MiB"] = float(mem_mib)
    if device_mem_gib is None:
        rep.checks.append(Check(
            "memory_forecast", "INFO",
            f"{n_dofs:,} dofs → ~{mem_mib:.0f} MiB "
            f"(2.2 GB/Mdof measured; + cuDSS workspace)", value=mem_mib))
    else:
        avail_mib = device_mem_gib * 1024.0
        frac = mem_mib / avail_mib
        if frac <= 0.6:
            v = "PASS"; d = f"{mem_mib:.0f} MiB of {avail_mib:.0f} MiB ({frac:.0%})"
        elif frac <= 0.85:
            v = "WARN"
            d = (f"{mem_mib:.0f} MiB of {avail_mib:.0f} MiB ({frac:.0%}) — tight "
                 f"with cuDSS workspace")
        else:
            v = "FAIL"
            d = (f"{mem_mib:.0f} MiB of {avail_mib:.0f} MiB ({frac:.0%}) — OOM "
                 f"risk with cuDSS workspace; reduce level or split")
        rep.checks.append(Check("memory_forecast", v, d, value=mem_mib))

    if verbose:
        print(rep.format(), flush=True)

    if strict and rep.failed:
        fails = [c for c in rep.checks if c.verdict == "FAIL"]
        raise RuntimeError(
            "XDD preflight FAILED (strict=True):\n" +
            "\n".join(f"  {c.name}: {c.detail}" for c in fails))

    return rep
