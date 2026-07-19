"""SP-1 R0 E6 — XDD preflight checks (the Block-E Debye-wall lesson, mechanical).

The preflight fires WARN/FAIL on known-bad configs and PASS on the marchable
regime.  THE gate: the physical full-drive PM6 config at a feasible CPU level
must FAIL the Debye check (that is exactly the wall E1(ii)/E2/E5 documented).
"""
from __future__ import annotations

import pytest

from diffsim.xdd.params import XDDParams
from diffsim.xdd.preflight import preflight, PreflightReport, Check


def _verdict(rep: PreflightReport, name: str) -> str:
    for c in rep.checks:
        if c.name == name:
            return c.verdict
    raise KeyError(name)


# ── (1) scales report present ─────────────────────────────────────────────────
def test_scales_report_present():
    p = XDDParams()
    rep = preflight(p, level=8, lam2=1e-1, verbose=False)
    for k in ("x0", "phi0", "t0", "U0", "J0", "lambda2", "gamma0",
              "debye_length_m", "debye_hat"):
        assert k in rep.scales, f"scales missing {k}"
    assert rep.scales["debye_length_m"] > 0


# ── (2) THE Debye gate: physical full drive at feasible CPU level FAILs ────────
def test_debye_physical_fulldrive_fails():
    """The physical PM6 config (λ²≈2.2e-5, Debye≈0.5 nm) on a feasible CPU level
    (L3) must FAIL the Debye-vs-mesh check — the documented E1(ii)/E2/E5 wall."""
    p = XDDParams()                      # physical PM6:Y6 → λ² from scales
    rep = preflight(p, level=3, lam2=None, verbose=False)  # None → physical λ²
    assert _verdict(rep, "debye_vs_mesh") == "FAIL", (
        "physical full-drive at L3 must FAIL the Debye check (the E-block wall)")
    assert rep.failed


def test_debye_marchable_passes():
    """The reduced-drive marchable regime (λ²=1e-1) at L3 resolves the Debye
    layer → PASS."""
    p = XDDParams()
    rep = preflight(p, level=3, lam2=1e-1, verbose=False)
    assert _verdict(rep, "debye_vs_mesh") == "PASS"


def test_debye_warn_band():
    """A λ² between the pass and fail thresholds lands in WARN."""
    p = XDDParams()
    # h_hat at L3 = 0.125; WARN band is Debye_hat/h in [0.25, 1.0) → Debye_hat in
    # [0.03125, 0.125) → λ² in [9.77e-4, 1.56e-2).
    rep = preflight(p, level=3, lam2=4e-3, verbose=False)
    assert _verdict(rep, "debye_vs_mesh") == "WARN"


# ── (3) strict raises on FAIL ─────────────────────────────────────────────────
def test_strict_raises_on_fail():
    p = XDDParams()
    with pytest.raises(RuntimeError, match="preflight FAILED"):
        preflight(p, level=3, lam2=None, strict=True, verbose=False)


def test_strict_ok_when_marchable():
    # Marchable Debye AND a resolved interface (interface_hat ≥ h) → no FAIL.
    p = XDDParams()
    rep = preflight(p, level=3, lam2=1e-1, interface_hat=0.2,
                    strict=True, verbose=False)
    assert not rep.failed


# ── (4) dt0 stiffness check ───────────────────────────────────────────────────
def test_dt0_stiffness_fail_on_huge_dt():
    """A huge first dt0 (σ ≪ stiffness) FAILs the stiffness check."""
    p = XDDParams()
    rep = preflight(p, level=3, lam2=1e-1, dt0_hat=1e6,
                    max_rate_hat=1e3, verbose=False)
    assert _verdict(rep, "dt0_stiffness") == "FAIL"


def test_dt0_stiffness_pass_on_small_dt():
    p = XDDParams()
    rep = preflight(p, level=3, lam2=1e-1, dt0_hat=1e-8,
                    max_rate_hat=1e3, verbose=False)
    assert _verdict(rep, "dt0_stiffness") == "PASS"


# ── (5) interface-width check ─────────────────────────────────────────────────
def test_interface_subgrid_fails():
    """A thin interface on a coarse mesh → sub-grid dissociation source → FAIL."""
    p = XDDParams(interface_thk=2e-9, height=100e-9)  # half-width 0.01 device
    rep = preflight(p, level=3, lam2=1e-1, verbose=False)  # h=0.125 → 0.08 el
    assert _verdict(rep, "interface_vs_mesh") == "FAIL"


def test_interface_resolved_passes():
    p = XDDParams()
    rep = preflight(p, level=3, lam2=1e-1, interface_hat=0.2, verbose=False)
    assert _verdict(rep, "interface_vs_mesh") == "PASS"


# ── (6) memory forecast ───────────────────────────────────────────────────────
def test_memory_forecast_info_without_device_mem():
    p = XDDParams()
    rep = preflight(p, level=8, lam2=1e-1, verbose=False)
    v = _verdict(rep, "memory_forecast")
    assert v == "INFO"
    assert rep.scales["forecast_mem_MiB"] > 0
    # level 8 = 257×257 = 66049 nodes × 5 = 330245 dofs → ~2228 MiB (E5 measured)
    assert 1800 < rep.scales["forecast_mem_MiB"] < 2600


def test_memory_forecast_oom_fail():
    p = XDDParams()
    # level 9 = 513×513 → 5×263169 ≈ 1.3M dofs → ~8.9 GB > a 4 GiB device
    rep = preflight(p, level=9, lam2=1e-1, device_mem_gib=4.0, verbose=False)
    assert _verdict(rep, "memory_forecast") == "FAIL"


# ── (7) worst-verdict + report formatting ─────────────────────────────────────
def test_worst_and_format():
    p = XDDParams()
    rep = preflight(p, level=3, lam2=None, verbose=False)
    assert rep.worst == "FAIL"
    txt = rep.format()
    assert "XDD preflight" in txt and "debye_vs_mesh" in txt
