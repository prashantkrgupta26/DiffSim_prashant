"""Preflight rule checks — run BEFORE any compute, always printed and
written. Each check names its rule and grades PASS / WARN / FAIL.

Rules (calibration provenance in each function):
  interface-resolution   xi/h vs the >=4-elements rule
  ic-thermodynamics      FH curvature eigs at the IC (b_reg included)
  evaporation-dt-cap     dt0 vs the march's dh_cap/(k_e dphi) clamp
  memory-forecast        G5-ladder linear sizing model vs available
  index-width     device-assembly nnz vs the 2^31 slot range

The MEMORY MODEL is calibrated on the G5 ladder
(docs/dev/2026-07-09-blockch-preconditioner.md, RTX 6000 Ada 48 GB,
blockch_dev device-bound film marches):
    (4.12M dofs -> 10.8 GPU / 10.4 host GB),
    (15.15M dofs -> 33.4 GPU / 37.7 host GB)
  =>  GPU_GB  = 2.35 + 2.05 * Mdofs
      host_GB = 0.20 + 2.48 * Mdofs
midpoint check: 5.08M -> model 12.8/12.8 vs measured 12.7/12.8 GB.
cuDSS direct has NO calibrated model; its measured 3-D wall is
ALLOC_FAILED at 228M nnz on the 48 GB card (G3 addendum).
"""
import os
from dataclasses import dataclass, field

import numpy as np

from . import analysis

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"
XI_ELEMS_RULE = 4.0            # the >=4-elements interface rule (PASS)
# FAIL floor: the smallest interface count in the VALIDATED full-res
# Wodo campaign is fig6 n100 at 2.5 lateral elements (d_bin, wodo_nova
# 250x100, all regimes replicated) — below that there is no validated
# precedent. Measured anchor, not an invented tolerance.
XI_ELEMS_FLOOR = 2.5
INT32_CEIL = 2 ** 31           # device-assembly slot arithmetic
WIDE_THRESHOLD = int(0.9 * 2 ** 31)   # auto-switch to mixed-width CSR
CUDSS_NNZ_WALL = 228e6         # measured ALLOC wall, 48 GB card (3-D)


@dataclass
class Check:
    rule: str
    status: str
    detail: str


@dataclass
class PreflightReport:
    checks: list = field(default_factory=list)
    derived: dict = field(default_factory=dict)

    def add(self, rule, status, detail):
        self.checks.append(Check(rule, status, detail))

    @property
    def failed(self):
        return any(c.status == FAIL for c in self.checks)

    def status_of(self, rule):
        for c in self.checks:
            if c.rule == rule:
                return c.status
        return None

    def format(self, params=None, resolved=None):
        lines = ["== PREFLIGHT =="]
        if params is not None and resolved is not None:
            r = resolved
            lines += [
                "resolved parameters:",
                f"  chi=(pf,ps,fs)={tuple(params.chi)}  "
                f"N=(p,f,s)={tuple(params.N)}  Bi={params.Bi}",
                f"  blend: phi_p0={r.phi_p0:.6f} phi_f0={r.phi_f0:.6f} "
                f"phi_s0={r.phi_s0:.6f}",
                f"  kappa=({r.kappa[0]:.4e},{r.kappa[1]:.4e})  "
                f"b_reg={params.b_reg}  noise={params.noise}",
                f"  mobility={params.mobility} D=(p,f)="
                f"({r.D_pair[0]:g},{r.D_pair[1]:g}) -> frozen "
                f"M=({r.M11:.4f},{r.M22:.4f})",
                f"  mesh: level-{r.level} strip {'x'.join(str(c) for c in r.cells)}"
                f" cells, {r.nodes} nodes, {r.dofs} dofs, nnz={r.nnz}",
                f"  metric: lat_scale={r.lat_scale:.4f} "
                f"y_comp={r.y_comp:.4f}  dx_lat={r.dx_lat:.5f} "
                f"dz(h=1)={r.dz0:.5f} (units of h0)",
                f"  dt0={params.dt0:g}  linsolver={params.linsolver}  "
                f"device_assembly={params.device_assembly}",
                f"  stop: h_min={params.h_min} phis_stop={params.phis_stop}"
                f" max_steps={params.max_steps} wall_cap={params.wall_cap}",
            ]
        for k, v in self.derived.items():
            lines.append(f"  derived: {k} = {v}")
        for c in self.checks:
            lines.append(f"[{c.status}] {c.rule}: {c.detail}")
        verdict = FAIL if self.failed else (
            WARN if any(c.status == WARN for c in self.checks) else PASS)
        lines.append(f"preflight verdict: {verdict}")
        return "\n".join(lines)


class PreflightError(RuntimeError):
    """Raised by FilmRun when a strict preflight FAILs."""


# ---------------------------------------------------------------------
def _host_available_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return None


def _gpu_free_gb(device):
    try:
        import torch
        if not torch.cuda.is_available() or not device.startswith("cuda"):
            return None, None
        free, total = torch.cuda.mem_get_info(device)
        return free / 1024 ** 3, total / 1024 ** 3
    except Exception:
        return None, None


def _gpu_name(device):
    try:
        import torch
        return torch.cuda.get_device_name(device)
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------
def run_preflight(params, resolved, query_hardware=True):
    p, r = params, resolved
    rep = PreflightReport()
    Np, Nf, Ns = (float(n) for n in p.N)
    kap = max(r.kappa)

    # -- interface-resolution (>=4-elements rule) ----------------------
    # xi_early: the first interfaces the drying line forms (pseudo-
    # binary tangent just inside the spinodal); xi_final: the dried p-f
    # binary (smallest). PASS needs both >= the 4-element rule; WARN
    # between the rule and the measured validated floor XI_ELEMS_FLOOR;
    # FAIL when even the FORMING interfaces sit below the floor
    # (no validated precedent; the flight recorder watches phi
    # excursions in the WARN band).
    xi_fin = analysis.interface_width(Np, Nf, p.chi[0], kap)
    xi_early, phis_at = analysis.earliest_interface_width(
        r.phi_p0, r.phi_f0, p.chi, (Np, Nf, Ns), p.b_reg, kap)
    cross = analysis.drying_line_crossing(
        r.phi_p0, r.phi_f0, p.chi, (Np, Nf, Ns), p.b_reg)
    h_star = cross[0] if cross else None
    dz_at = (h_star if h_star is not None else 1.0) * r.dz0
    n_early = xi_early / max(r.dx_lat, dz_at) if np.isfinite(xi_early) \
        else np.inf
    n_fin = xi_fin / r.dx_lat if np.isfinite(xi_fin) else np.inf
    detail = (f"xi_early={xi_early:.4g} (at phi_s={phis_at:.2f}) = "
              f"{n_early:.1f} elems; xi_final(binary)={xi_fin:.4g} = "
              f"{n_fin:.1f} lateral elems; rule: >= {XI_ELEMS_RULE:g} "
              f"(validated floor {XI_ELEMS_FLOOR:g})")
    if n_early < XI_ELEMS_FLOOR:
        rep.add("interface-resolution", FAIL, detail
                + " -- forming interfaces below the validated floor: "
                "refine the mesh or raise kappa/eps2")
    elif min(n_early, n_fin) < XI_ELEMS_RULE:
        rep.add("interface-resolution", WARN, detail
                + " -- below the 4-element rule but inside the "
                "validated band; phi min/max monitored in flight")
    else:
        rep.add("interface-resolution", PASS, detail)

    # -- ic-thermodynamics ---------------------------------------------
    lo, hi = analysis.curvature_eigs(r.phi_p0, r.phi_f0, p.chi,
                                     (Np, Nf, Ns), p.b_reg)
    if cross:
        h_s, phis_s, _ = cross
        cross_txt = (f"drying line enters the spinodal at h~{h_s:.3f} "
                     f"(phi_s~{phis_s:.3f})")
    else:
        cross_txt = "drying line never enters the spinodal (check chi/N)"
    if lo < 0:
        rep.add("ic-thermodynamics", PASS,
                f"IC inside spinodal (curvature eigs {lo:.3f}, {hi:.3f},"
                " b_reg included): immediate bulk separation expected")
    else:
        rep.add("ic-thermodynamics", WARN,
                f"stable blend at t=0 (curvature eigs {lo:.3f}, "
                f"{hi:.3f}, b_reg included), expect no separation until"
                f" enrichment -- {cross_txt}")
    rep.derived["spinodal_crossing"] = cross_txt

    # -- evaporation-dt-cap --------------------------------------------
    # march clamps dt_eff = min(dt, dh_cap / K), K = k_e * phi_s_top;
    # at t=0 that cap is dh_cap / (Bi * phi_s0).
    cap0 = p.dh_cap / max(p.Bi * r.phi_s0, 1e-300)
    if p.dt0 <= cap0:
        rep.add("evaporation-dt-cap", PASS,
                f"dt0={p.dt0:g} <= dh_cap/(Bi*phi_s0)={cap0:.3g}")
    else:
        rep.add("evaporation-dt-cap", WARN,
                f"dt0={p.dt0:g} > dh_cap/(Bi*phi_s0)={cap0:.3g}: the "
                "march will clamp step 1 to the cap")

    # -- memory-forecast (G5 sizing model; module docstring) ------------
    mdofs = r.dofs / 1e6
    gpu_gb = 2.35 + 2.05 * mdofs
    host_gb = 0.20 + 2.48 * mdofs
    free_gb = total_gb = host_av = None
    if query_hardware:
        free_gb, total_gb = _gpu_free_gb(p.device)
        host_av = _host_available_gb()
    txt = (f"model (G5 ladder, blockch_dev): GPU {gpu_gb:.1f} GB, "
           f"host {host_gb:.1f} GB for {mdofs:.2f}M dofs")
    if free_gb is not None:
        txt += f"; available GPU {free_gb:.1f}/{total_gb:.1f} GB"
    if host_av is not None:
        txt += f", host {host_av:.1f} GB"
    status = PASS
    if free_gb is not None and gpu_gb > free_gb:
        status = FAIL
        txt += " -- forecast exceeds free GPU memory"
    elif host_av is not None and host_gb > host_av:
        status = FAIL
        txt += " -- forecast exceeds available host memory"
    elif free_gb is not None and gpu_gb > free_gb / 1.2:
        status = WARN
        txt += " -- < 20% GPU headroom over the model"
    if p.linsolver == "cudss" and p.dim == 3 and r.nnz > CUDSS_NNZ_WALL:
        big_card = free_gb is not None and free_gb > 50.0
        status = WARN if big_card else FAIL
        txt += (f"; cudss direct at nnz={r.nnz / 1e6:.0f}M is beyond "
                "the measured 228M-nnz ALLOC wall (48 GB card) -- use "
                "blockch_dev")
    rep.add("memory-forecast", status, txt)

    # -- lte-noise-incompatible (the LTE controller's noise contract) ---
    # Strong/pathwise LTE is ILL-POSED under FDT forcing: the step-
    # doubling / predictor-corrector difference of two noisy realizations
    # is dominated by the independent noise draws, not the truncation
    # error, so a tolerance has no deterministic meaning.  When the user
    # switches the LTE controller ON (adapt="lte") with noise still on we
    # REPORT it here and the controller falls back to the fixed-order BDF
    # ladder (physics.lte.lte_march; noise-valid BDF1, BDF2 being the
    # deterministic recommendation).  adapt defaults to "ladder" (OFF),
    # so existing runs never see this check (bit-parity).
    adapt = getattr(p, "adapt", "ladder")
    if adapt == "lte":
        if float(getattr(p, "noise", 0.0)) != 0.0:
            rep.add("lte-noise-incompatible", WARN,
                    f"adapt=lte requested with FDT noise={p.noise:g}: "
                    "strong-order LTE is ill-posed under stochastic "
                    "forcing -- the LTE controller will REFUSE and fall "
                    "back to the fixed-order BDF ladder (noise-valid "
                    "BDF1; BDF2 is the deterministic recommendation). "
                    "Set noise=0 for the full LTE PI(D) controller.")
        else:
            rep.add("lte-noise-incompatible", PASS,
                    "adapt=lte with noise=0: deterministic run gets the "
                    "full LTE PI(D) controller.")

    # -- index-width (P0-2: the old index-width, now an auto-
    #    switch report) --------------------------------------------------
    # Device assembly used to hard-FAIL past 2^31 nnz (the int32 slot
    # arithmetic wrapped).  Mixed-width CSR (int64 offsets/slots, int32
    # columns) removes that ceiling: "auto" flips to the wide path once
    # nnz crosses ~90% of 2^31.  A user who FORCES narrow past the
    # ceiling still FAILs loudly here (no silent wrap).
    if p.device_assembly:
        pct = 100.0 * r.nnz / INT32_CEIL
        iw = getattr(p, "index_width", "auto")
        if iw == "narrow" and r.nnz >= INT32_CEIL:
            rep.add("index-width", FAIL,
                    f"nnz={r.nnz} >= 2^31 ({pct:.0f}% of range) but "
                    "index_width='narrow' forced: the int32 slot "
                    "arithmetic would WRAP -- use 'auto' or 'wide'")
        elif iw == "wide" or (iw == "auto" and r.nnz >= WIDE_THRESHOLD):
            mode = "wide (forced)" if iw == "wide" else "wide (auto)"
            rep.add("index-width", PASS,
                    f"nnz={r.nnz} = {pct:.0f}% of int32 range -> {mode}: "
                    "mixed-width CSR (int64 offsets/slots, int32 columns) "
                    "-- past the int32 nnz ceiling, near-zero mem tax")
        else:
            rep.add("index-width", PASS,
                    f"nnz={r.nnz} = {pct:.0f}% of int32 range -> narrow "
                    "(int32 offsets/slots; auto-switch to wide at "
                    f"{100.0 * WIDE_THRESHOLD / INT32_CEIL:.0f}%)")
    return rep
