"""Film front-end gates (diffsim.film): the four named Negi 2018 2-D
configs run end-to-end through FilmRun + RunLog, plus the preflight
and autopsy failure-path units.

NEGI VALIDATION VERDICT (measured 2026-07-10, RTX 6000 Ada, cudss
device-bound, full config resolution 125x250 = the paper-scale mesh;
each case 66-105 s, 263-304 accepted steps to phi_s < 0.05):

  HONESTLY ASSERTED (robust across all four rpm cases):
    * runs complete to the phis_stop criterion without autopsy;
    * solute content conserved: mass drift measured 2.2e-15 class,
      asserted < 1e-10 (4+ decades headroom);
    * phi stays in the healthy envelope (measured [0.030, 0.933],
      asserted the autopsy bound [-0.02, 1.02]);
    * separation initiates at the TOP: the VERTICAL onset detector
      (first order-one row-mean phi_f deviation) fires at theta = 1.0
      in every case — the less-soluble fullerene (chi_fs = 0.9 >>
      chi_ps = 0.1) enriches at the solvent-lean free surface — and
      the final film is a near-pure fullerene-top bilayer (row-mean
      phi_f 0.93 top vs 0.05 bottom).

  REPLICATION GAPS (printed, NOT asserted — recorded honestly):
    * the rpm ladder does NOT differentiate under the documented
      D_s-based nondimensionalization: Bi = 0.0019..0.0073 is
      quasi-static (evaporation Peclet K h / D ~ 1e-3), so all four
      cases give the same equilibrium bilayer; final lateral
      structure is noise-scale (max lateral std 0.001-0.007), L_c
      {6000: 0.0185, 3000: 0.0183, 1500: 0.0188, 500: 0.0186} h0
      units — flat, not monotone in rpm; the paper's 500-rpm
      bulk-onset / larger-domain contrast is absent.
    * TIME-SCALE SUSPECT, diagnostic measured: rerunning the 6000 rpm
      case at Bi x100 = 0.73 (i.e. a diffusivity scale ~100x smaller
      than D_s, or equivalently faster drying) reproduces the paper's
      phenomenology: strong surface-directed gradient (top row phi_f
      0.40 vs 0.18 bottom at h = 0.3 BEFORE lateral onset), lateral
      onset mid-film (theta = 0.46), and ARRESTED lateral domains
      that survive to dryness (max lateral std 0.15 at final; L_c at
      h=0.2 is 3.5x the base case). The parameter mapping of the
      evaporation/diffusion time-scale ratio — not the solver — is
      the replication gap. Evidence: benchmarks/data/negi2018/.
"""
import os

import numpy as np
import pytest

from diffsim.film import (FilmParams, FilmRun, run_preflight,
                          PreflightError, CONFIG_DIR)

pytestmark = pytest.mark.tier3

NEGI = ["negi2018_6000rpm", "negi2018_3000rpm", "negi2018_1500rpm",
        "negi2018_500rpm"]
NEGI_BI = {"negi2018_6000rpm": 0.0073, "negi2018_3000rpm": 0.0051,
           "negi2018_1500rpm": 0.0033, "negi2018_500rpm": 0.0019}


def _cfg(name):
    return FilmParams.from_yaml(os.path.join(CONFIG_DIR, name + ".yaml"))


# ---------------------------------------------------------------------
# parameter/serialization gates (no GPU)
# ---------------------------------------------------------------------
def test_negi_config_parameters():
    """The four named configs carry EXACTLY the spec'd material system
    (guards the single-source-of-truth promise to the cluster kits)."""
    for name in NEGI:
        p = _cfg(name)
        r = p.resolve()
        assert tuple(p.N) == (87, 5, 1)
        assert tuple(p.chi) == (1.0, 0.1, 0.9)
        assert p.Bi == NEGI_BI[name]
        assert abs(r.phi_p0 - 0.1 / 3) < 1e-12          # 1:2 blend at
        assert abs(r.phi_f0 - 0.2 / 3) < 1e-12          # 90% solvent
        assert abs(r.phi_s0 - 0.9) < 1e-12
        assert r.D_pair == (1e-3, 5e-3)                 # per-species
        # eps2 = 1e-10 J/m at h0 = 1 um, Vs = 80.7 cm3/mol, T = 300 K
        assert abs(r.kappa[0] - 3.2353e-6) < 1e-9
        assert r.cells == (125, 250) and p.Lx == 0.5
        assert p.noise > 0                              # CHC noise ON


def test_params_roundtrip(tmp_path):
    p = _cfg("negi2018_6000rpm")
    path = tmp_path / "echo.yaml"
    p.to_yaml(path)
    q = FilmParams.from_yaml(path)
    assert q.config_hash() == p.config_hash()
    assert q.resolve() == p.resolve()
    # ratio-blend and explicit-fraction blends agree
    a = FilmParams(blend_ratio="1:2", phi_s0=0.9).blend()
    b = FilmParams(phi_p0=0.1 / 3, phi_f0=0.2 / 3).blend()
    assert np.allclose(a, b)


# ---------------------------------------------------------------------
# preflight unit: a deliberately under-resolved config must FAIL with
# the interface-resolution rule (and strict mode must refuse to run)
# ---------------------------------------------------------------------
def _underresolved_params(tmp_path=None):
    # deep quench (IC inside the spinodal -> the forming interfaces
    # carry the full quench width) + tiny kappa + 8x8 mesh
    return FilmParams(
        name="underresolved", chi=(6.0, 0.8, 0.8), N=(1.0, 1.0, 1.0),
        phi_p0=0.35, phi_f0=0.35, kappa=1e-6, b_reg=0.0, Bi=0.5,
        mobility="constant", Lx=1.0, resolution=(8, 8),
        linsolver="splu", device_assembly=False, preflight="strict",
        device="cpu")


def test_preflight_underresolved_fails(tmp_path):
    p = _underresolved_params()
    report = run_preflight(p, p.resolve(), query_hardware=False)
    assert report.status_of("interface-resolution") == "FAIL"
    assert report.failed
    with pytest.raises(PreflightError, match="interface-resolution"):
        FilmRun(p).run(str(tmp_path / "out"), query_hardware=False)
    # strict abort happens BEFORE compute: only the log artifacts exist
    assert not os.path.exists(
        str(tmp_path / "out" / "underresolved_final.npz"))
    # the refusal is still fully logged
    assert os.path.exists(str(tmp_path / "out" / "runlog.txt"))


# ---------------------------------------------------------------------
# autopsy unit: force a dt-ladder collapse; the diagnosis must fire
# ---------------------------------------------------------------------
def test_autopsy_dt_collapse(tmp_path, device):
    # newton_max=1 cannot converge the first implicit solve; the single
    # reject drops dt below dt_min = dt0/2 -> dt_underflow -> autopsy
    p = _cfg("wodo2012_fig3_bi10").replace(
        name="autopsy_unit", resolution=(4, 64), Lx=4 / 64,
        newton_max=1, dt_min=5e-5, device=device,
        linsolver="cudss", device_assembly=True)
    out = str(tmp_path / "boom")
    summary = FilmRun(p).run(out)
    assert summary["reason"] == "dt_underflow"
    assert "dt-collapse-at-onset" in summary["autopsy"]
    adir = os.path.join(out, "autopsy")
    for f in ("autopsy.txt", "tail.jsonl", "config.yaml", "fields.npz"):
        assert os.path.exists(os.path.join(adir, f)), f
    with open(os.path.join(adir, "autopsy.txt")) as fh:
        text = fh.read()
    assert "DIAGNOSIS" in text and "dt-collapse-at-onset" in text


# ---------------------------------------------------------------------
# THE GATE: the four Negi 2-D cases end-to-end (module docstring for
# the measured basis; full config resolution, ~70-105 s/case measured)
# ---------------------------------------------------------------------
def test_negi_2d_validation(tmp_path, device):
    summaries = {}
    finals = {}
    for name in NEGI:
        p = _cfg(name).replace(device=device, log_every=200,
                               wall_cap=1200.0)   # safety, not a gate
        out = str(tmp_path / name)
        s = FilmRun(p).run(out)
        summaries[name] = s
        finals[name] = np.load(os.path.join(out, f"{name}_final.npz"))

    print("\n== NEGI 2-D VALIDATION TABLE (paper expectations vs "
          "measured; GAP = recorded replication gap) ==")
    print(f"{'case':22s} {'reason':10s} {'steps':>5s} {'drift':>9s} "
          f"{'phi range':>17s} {'L_c':>8s} {'ons_lat':>7s} "
          f"{'ons_vert':>8s}")
    for name in NEGI:
        s = summaries[name]
        print(f"{name:22s} {s['reason']:10s} {s['steps']:5d} "
              f"{s['mass_drift']:9.1e} "
              f"[{s['phi_min']:+.3f},{s['phi_max']:.3f}] "
              f"{s['L_c']:8.4f} {str(s['onset_theta']):>7s} "
              f"{str(s['onset_v_theta']):>8s}")

    for name, s in summaries.items():
        # completion without autopsy
        assert s["reason"] == "phis_stop", (name, s["reason"])
        assert "autopsy" not in s, (name, s.get("autopsy"))
        # solute mass conserved (measured 2.2e-15; 4+ decades headroom)
        assert s["mass_drift"] < 1e-10, (name, s["mass_drift"])
        # healthy phi envelope (the autopsy bound)
        assert s["phi_min"] > -0.02 and s["phi_max"] < 1.02, name
        # separation DID initiate, vertically surface-selected: the
        # less-soluble fullerene enriches at the free surface
        assert s["onset_v_theta"] is not None, name
        assert s["onset_v_theta"] > 0.9, (name, s["onset_v_theta"])
        # final film is a fullerene-top bilayer
        gf = finals[name]["grid_f"]
        prof = gf.mean(axis=1)
        assert prof[-5:].mean() > prof[:5].mean(), (
            name, "expected fullerene-rich top")
        assert np.isfinite(s["L_c"]) and s["L_c"] > 0, name

    # paper-expectation cross-checks: PRINTED, not asserted (measured
    # as replication gaps under the D_s-based nondimensionalization --
    # module docstring; suspects recorded in the milestone report)
    lcs = [summaries[n]["L_c"] for n in NEGI]     # 6000 -> 500 rpm
    mono = all(a <= b for a, b in zip(lcs, lcs[1:]))
    print(f"paper expectation 'L_c grows as drying slows': "
          f"{'PASS' if mono else 'GAP'} (measured {lcs})")
    lat = [summaries[n]["onset_theta"] for n in NEGI]
    print(f"paper expectation '6000 rpm lateral onset at TOP': "
          f"{'PASS' if lat[0] is not None and lat[0] > 0.8 else 'GAP'}"
          f" (measured lateral onset thetas {lat})")
