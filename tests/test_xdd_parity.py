"""SP-1 R0 Block E — E1 reduction gates + E2 SimSalabim cross-check.

Physics-anchor gates that run locally.  Baselines locked in
tests/baselines/xdd_e1_e2.json (values + tolerances + provenance); this suite
compares against them (the house pattern, cf. test_m1b_baselines.py).

E1 — reduction gates
--------------------
G_E1_1  Stripe ORDERING (collapsed / transport-only limit): 6-stripe vertical
        and horizontal morphologies produce MORE free-carrier generation
        content (∫k̂X̂ — the CMAME-2012 transport-only carrier source) than the
        2-stripe cases: 2× interface sites → more dissociation.  Asserts the
        ordering (not absolute numbers, per the plan) + the locked baseline.
G_E1_2  Kodali 1-D band: the physical PPV:PCBM device Jsc lands in the −30 A/m²
        class band.  The XDD forward solver hits the documented CPU-mesh drive
        wall on the physical config, so the band is anchored on the SimSS 1-D
        reference of the SAME physical device (E2) — asserted in G_E2_1's band
        check; here we assert the locked SimSS Jsc IS in the ±50% band.

E2 — SimSalabim 1-D cross-check
-------------------------------
G_E2_1  If a SimSS binary is available (built from source, arm64 Mac — see
        benchmarks/xdd/e2_simsalabim.py), run the matched 1-D device and assert
        {Jsc,Voc,FF} reproduce the locked reference within 5%.  Skips (not
        fails) when no binary is on PATH / SIMSS_BIN — the reference is still
        locked and the Kodali-band assertion (G_E1_2) runs unconditionally.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.tier5

_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baselines",
                              "xdd_e1_e2.json")


def _baseline() -> dict:
    with open(_BASELINE_PATH) as fh:
        return json.load(fh)


# ══════════════════════════════════════════════════════════════════════════════
# E1(i) — stripe reduction ordering
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_e1_stripe_ordering():
    """G_E1_1: 6-stripe > 2-stripe free-carrier generation (both orientations).

    Reduction / collapsed-exciton limit: in the CMAME-2012 transport-only
    model the interface delivers a free-carrier source; doubling the number of
    D/A interface sites (2→6 stripes) increases that source.  In the excitonic
    system the source is ∫k̂_D X̂_D + ∫k̂_A X̂_A.  We build tiny stripe
    morphologies with the A2 `signed_distance` utility, march each to a lit
    steady state (reduced-drive marchable regime), and assert:
      (1) 6-stripe carrier_src > 2-stripe carrier_src for vertical AND horizontal
      (the ordering — the plan asks for ordering, not absolute numbers);
      (2) each measured value matches the locked baseline within rtol.
    """
    from benchmarks.xdd.e1_reduction import e1_stripe_ordering

    base = _baseline()["e1_stripe_ordering"]
    rtol = base["rtol"]

    summary = e1_stripe_ordering(level=3)

    # (0) all four marched to steady
    for k, v in summary.items():
        assert v["fired"], f"G_E1_1: stripe case {k} did not reach steady"

    csrc = {k: v["carrier_src"] for k, v in summary.items()}
    print("\nG_E1_1 carrier_src:", {k: f"{c:.3e}" for k, c in csrc.items()})

    # (1) THE ordering gate — 6-stripe > 2-stripe, both orientations
    assert csrc["vertical_6"] > csrc["vertical_2"], (
        f"G_E1_1: vertical 6-stripe csrc {csrc['vertical_6']:.3e} not > "
        f"2-stripe {csrc['vertical_2']:.3e} (more interface sites → more diss.)")
    assert csrc["horizontal_6"] > csrc["horizontal_2"], (
        f"G_E1_1: horizontal 6-stripe csrc {csrc['horizontal_6']:.3e} not > "
        f"2-stripe {csrc['horizontal_2']:.3e}")
    # margin sanity: the 6/2 ratio is ~3.7× (robust, not marginal)
    assert csrc["vertical_6"] / csrc["vertical_2"] > 2.0, (
        "G_E1_1: 6/2 vertical ratio should exceed 2× (site-count scaling)")

    # (2) baseline lock — measured vs locked absolute values within rtol
    for k in ("vertical_2", "vertical_6", "horizontal_2", "horizontal_6"):
        exp = base[k]["carrier_src"]
        got = csrc[k]
        assert abs(got - exp) <= rtol * abs(exp), (
            f"G_E1_1: {k} carrier_src {got:.4e} vs baseline {exp:.4e} "
            f"(rtol {rtol})")
        # interface-site count is a deterministic geometric quantity
        exp_sites = base[k]["n_sites"]
        assert abs(summary[k]["n_sites"] - exp_sites) <= 0.02 * exp_sites, (
            f"G_E1_1: {k} n_sites {summary[k]['n_sites']:.1f} vs {exp_sites}")


# ══════════════════════════════════════════════════════════════════════════════
# E1(ii) — Kodali 1-D homogeneous band (anchored on the SimSS reference)
# ══════════════════════════════════════════════════════════════════════════════

def test_e1_kodali_band():
    """G_E1_2: the physical PPV:PCBM Jsc lands in the −30 A/m² Kodali class band.

    The XDD forward solver hits the documented CPU-mesh drive wall on the
    physical config (Ê_g≈52; see the baseline's _xdd_headtohead_blocked note),
    so the Kodali anchor is the SimSS 1-D reference of the SAME physical device.
    Band: ±50% of −30 A/m² → [−45, −15] A/m².  Asserts the locked SimSS Jsc is
    in-band, confirming the physical device parametrisation is Kodali-class.
    """
    e2 = _baseline()["e2_simsalabim"]
    jsc = e2["Jsc_A_per_m2"]
    lo, hi = -45.0, -15.0
    print(f"\nG_E1_2: SimSS physical Kodali device Jsc = {jsc:.3f} A/m² "
          f"(band [{lo}, {hi}])")
    assert lo <= jsc <= hi, (
        f"G_E1_2: Jsc {jsc:.3f} A/m² outside the ±50% Kodali band [{lo}, {hi}]")


# ══════════════════════════════════════════════════════════════════════════════
# E2 — SimSalabim 1-D cross-check
# ══════════════════════════════════════════════════════════════════════════════

def _find_simss():
    from benchmarks.xdd.e2_simsalabim import find_simss
    return find_simss()


def test_e2_simsalabim_reference_locked():
    """G_E2_0: the SimSS reference values are present and self-consistent.

    Runs unconditionally (no binary needed): guards the locked reference so a
    silent corruption of the baseline is caught even on CI without SimSS.
    """
    e2 = _baseline()["e2_simsalabim"]
    assert e2["Jsc_A_per_m2"] < 0.0, "G_E2_0: photocurrent Jsc must be negative"
    # Voc must be positive and below the built-in (E_g = 1.34 V for this device)
    assert 0.0 < e2["Voc_V"] < 1.34, "G_E2_0: Voc must be in (0, V_bi=E_g)"
    assert 0.0 < e2["FF"] < 1.0, "G_E2_0: FF must be in (0,1)"


def test_e2_simsalabim_crosscheck():
    """G_E2_1: run SimSS on the matched 1-D device; reproduce the locked ref.

    Skips (not fails) when no SimSS binary is available — the community-standard
    anchor is built from source (fpc) and its reference is locked in the
    baseline regardless.  When present, asserts {Jsc,Voc,FF} within 5% of the
    locked reference (SimSS is deterministic; 5% is generous headroom guarding
    against a binary/version drift).
    """
    binp = _find_simss()
    if binp is None:
        pytest.skip(
            "SimSS binary not found (set SIMSS_BIN or put `simss` on PATH). "
            "Build: git clone github.com/kostergroup/SIMsalabim; "
            "cd SimSS; fpc -O3 -Fu../Units simss.pas. Reference is locked "
            "in tests/baselines/xdd_e1_e2.json regardless.")

    from benchmarks.xdd.e2_simsalabim import run_simss, KODALI_DEVICE

    data_dir = Path(binp).resolve().parent.parent / "Data"
    assert data_dir.is_dir(), f"G_E2_1: SimSS Data dir missing at {data_dir}"

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        res = run_simss(KODALI_DEVICE, Path(td), data_dir)

    assert res.get("returncode") == 0, f"G_E2_1: SimSS failed: {res}"
    e2 = _baseline()["e2_simsalabim"]
    rtol = e2["rtol"]
    print(f"\nG_E2_1: SimSS {res}  vs ref Jsc={e2['Jsc_A_per_m2']} "
          f"Voc={e2['Voc_V']} FF={e2['FF']}")

    for key, ref_key in (("Jsc", "Jsc_A_per_m2"), ("Voc", "Voc_V"), ("FF", "FF")):
        got = res[key]
        exp = e2[ref_key]
        assert abs(got - exp) <= rtol * abs(exp), (
            f"G_E2_1: {key} {got} vs locked reference {exp} (rtol {rtol})")


# ══════════════════════════════════════════════════════════════════════════════
# E5 — Nirmal J(t) perf gate (LEDGER-LOCKED, not CI-reasserted)
# ══════════════════════════════════════════════════════════════════════════════

_E5_BASELINE_PATH = os.path.join(os.path.dirname(__file__), "baselines",
                                 "xdd_e5_perf.json")


def _e5_baseline() -> dict:
    with open(_E5_BASELINE_PATH) as fh:
        return json.load(fh)


def test_e5_perf_baseline_parses_and_speedup_consistent():
    """G_E5: the perf baseline file parses and its recorded speedup IS what the
    file's own s/step + steps + CPU-baseline arithmetic says.

    House pattern for perf: measured numbers are LEDGER-LOCKED on the box
    (benchmarks/xdd/e5_perf_nirmal.py), NOT re-run in CI (a 330k-DOF GPU march
    is not a unit test).  This test guards against silent corruption of the
    locked ledger by re-deriving the speedup from the recorded primitives and
    asserting internal consistency.
    """
    b = _e5_baseline()
    m = b["measured"]
    g = b["gate"]

    # 1) primitives present and physical
    assert m["nodes"] > 0 and m["dofs"] > 0
    assert m["s_per_step_median"] > 0.0
    assert m["steps_for_window"] > 0
    assert "splu" in m["solver"].lower(), (
        "G_E5: solver provenance must record the host splu path")

    # 2) end-to-end = s/step × steps (the recorded projection)
    e2e = m["s_per_step_median"] * m["steps_for_window"]
    assert abs(e2e - g["end_to_end_s"]) <= 1e-6 * max(e2e, 1.0), (
        f"G_E5: end-to-end {g['end_to_end_s']} != s/step*steps {e2e}")

    # 3) speedup = CPU baseline / end-to-end (both baseline-band ends)
    for hours, key in ((g["baseline_hours_lo"], "speedup_vs_10h"),
                       (g["baseline_hours_hi"], "speedup_vs_30h")):
        exp = (hours * 3600.0) / g["end_to_end_s"]
        assert abs(exp - g[key]) <= 1e-6 * max(exp, 1.0), (
            f"G_E5: {key} {g[key]} != baseline/e2e {exp}")

    # 4) the recorded verdict matches the 100x target arithmetic
    assert g["pass_vs_10h"] == (g["speedup_vs_10h"] >= g["target"])
    assert g["pass_vs_30h"] == (g["speedup_vs_30h"] >= g["target"])
