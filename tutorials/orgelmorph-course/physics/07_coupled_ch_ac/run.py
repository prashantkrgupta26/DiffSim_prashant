"""OrgElMorph course - Physics P7 driver (the file the student runs).

    python run.py                 # commensurate controls + causality + checks
    python run.py --level 6       # finer mesh (slower)

Runs, in order:
  1. THREE COMMENSURATE controls (ch_only / coupled_nochi / full) with the
     SAME crystal-footprint mask and the SAME composition-contrast metric,
     decomposing crystallization-driven demixing into its crystal-bulk and
     chi-expulsion channels (absolute contrasts + resolved-denominator
     relative; no divide-by-floor);
  2. seed sensitivity of the full-run contrast (mean +/- sd over seeds);
  3. the coupled free-energy budget of the full run (entropy + chi +
     cryst + gradients; total decreases -- Lyapunov);
  4. CAUSALITY -- (a) frozen geometry: freeze the crystal (L_psi = 0) and
     show the chi-coupling still adds demixing at fixed geometry; (b)
     kinetics: a faster L_psi crystallizes sooner and the contrast tracks
     the crystalline area.

Results -> outputs/results.json, checked against baseline.yaml
(tolerance-based).  The four-fold chi convention (p1 absolute vs r14
increment) is verified separately by test_chi_limits.py.
"""
import argparse
import json
import os
import sys

import numpy as np

import coupled as C
from coupled import chi_eff_limits

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from common import check_results as _check  # noqa: E402

SEEDS = [4, 7, 11]


def three_controls(dm, mesh, cons, device, t_end=0.4):
    """Full / coupled-no-chi / ch-only on ONE shared mask + metric.
    The full run also carries the energy budget."""
    full = C.run(dm, mesh, cons, mode="full", t_end=t_end,
                 track_energy=True, device=device)
    mask = full["psi_full"] > 0.5
    nochi = C.run(dm, mesh, cons, mode="coupled_nochi", t_end=t_end,
                  device=device)
    chonly = C.run(dm, mesh, cons, mode="ch_only", t_end=t_end,
                   device=device)
    cf = C.contrast_on_mask(full["phi0_full"], mask)
    cn = C.contrast_on_mask(nochi["phi0_full"], mask)
    cc = C.contrast_on_mask(chonly["phi0_full"], mask)
    return dict(full=full, nochi=nochi, chonly=chonly, mask=mask,
                c_full=cf, c_nochi=cn, c_chonly=cc,
                chi_channel=cf - cn, cryst_channel=cn - cc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output", default=os.path.join(_HERE, "outputs"))
    ap.add_argument("--no-check", action="store_true")
    args = ap.parse_args()
    dm, mesh, cons = C.build_mesh_dm(args.level, device=args.device)

    print(f"\n=== Coupled Cahn-Hilliard + Allen-Cahn (p1, level "
          f"{args.level}) ===")

    # 0. the four-fold chi convention (verified in test_chi_limits.py)
    lim = chi_eff_limits(1.2, 2.1, 2.6, 3.0, bulk="p1")
    print(f"\n[0] four-fold chi (p1, ABSOLUTE limits): "
          f"aa={lim['aa']:.2f} ca={lim['ca']:.2f} ac={lim['ac']:.2f} "
          f"cc={lim['cc']:.2f}  (r14 would be increments over chi_aa)")

    # 1. three commensurate controls
    tc = three_controls(dm, mesh, cons, args.device)
    print("\n[1] commensurate controls (SAME mask = full crystal footprint,"
          " SAME contrast metric)")
    print(f"  ch_only        contrast = {tc['c_chonly']:+.4f}  "
          f"(CH alone -- ~0 on the crystal footprint)")
    print(f"  coupled_nochi  contrast = {tc['c_nochi']:+.4f}  "
          f"(crystal-bulk channel; chi_ca = chi_aa)")
    print(f"  full           contrast = {tc['c_full']:+.4f}  "
          f"(adds chi-expulsion channel)")
    print(f"  crystal-bulk channel (nochi - ch_only) = {tc['cryst_channel']:+.4f}")
    print(f"  chi-expulsion channel (full - nochi)   = {tc['chi_channel']:+.4f}")
    if abs(tc["c_nochi"]) > 0.02:
        print(f"  relative (full / coupled_nochi) = "
              f"{tc['c_full'] / tc['c_nochi']:.2f} (denominator resolved)")
    print("  (ch_only denominator ~0 -> NO full/ch_only ratio reported)")

    # 2. seed sensitivity of the full contrast
    cs = []
    for s in SEEDS:
        f = C.run(dm, mesh, cons, mode="full", seed=s, device=args.device)
        cs.append(C.contrast_on_mask(f["phi0_full"], f["psi_full"] > 0.5))
    cs = np.array(cs)
    print(f"\n[2] seed sensitivity of full contrast ({len(SEEDS)} seeds): "
          f"{cs.mean():.4f} +/- {cs.std(ddof=1):.4f}")

    # 3. coupled energy budget (from the full run)
    et = tc["full"]["energy_total"]
    e_end = tc["full"]["energy"][-1]
    incr = np.diff(et)
    print("\n[3] coupled free-energy budget (full run)")
    print(f"  total F: {et[0]:.4f} -> {et[-1]:.4f}   "
          f"largest positive step = {max(0.0, incr.max()):.2e}")
    print("  end split: " + ", ".join(f"{k}={v:.4f}" for k, v in
                                       e_end.items() if k != "total"))

    # 4. causality
    # (a) frozen geometry: freeze the crystal, isolate the chi channel
    fz_full = C.run(dm, mesh, cons, mode="full", freeze_psi=True,
                    device=args.device)
    fz_nochi = C.run(dm, mesh, cons, mode="coupled_nochi", freeze_psi=True,
                     device=args.device)
    fzmask = fz_full["psi_full"] > 0.5
    fz_chi = (C.contrast_on_mask(fz_full["phi0_full"], fzmask)
              - C.contrast_on_mask(fz_nochi["phi0_full"], fzmask))
    # (b) kinetics at fixed coupling: fast vs slow L_psi
    fast = C.run(dm, mesh, cons, mode="full", L_psi=6.0, device=args.device)
    slow = C.run(dm, mesh, cons, mode="full", L_psi=1.0, device=args.device)
    c_fast = C.contrast_on_mask(fast["phi0_full"], fast["psi_full"] > 0.5)
    c_slow = C.contrast_on_mask(slow["phi0_full"], slow["psi_full"] > 0.5)
    print("\n[4] causality")
    print(f"  (a) frozen geometry (L_psi=0): chi-channel contrast increment "
          f"= {fz_chi:+.4f}  (coupling demixes at FIXED geometry)")
    print(f"  (b) kinetics @ fixed coupling: fast L_psi area "
          f"{fast['area_end']:.4f} contrast {c_fast:.4f} | slow area "
          f"{slow['area_end']:.4f} contrast {c_slow:.4f}")
    print("      (both rates reach a large contrast -> the COUPLING sets "
          "the demixing magnitude; L_psi (kinetics) sets only the timing)")

    results = {
        "chi_limits_p1": lim,
        "controls": {
            "ch_only": tc["c_chonly"], "coupled_nochi": tc["c_nochi"],
            "full": tc["c_full"], "chi_channel": tc["chi_channel"],
            "cryst_channel": tc["cryst_channel"]},
        "seed_sensitivity": {
            "mean": float(cs.mean()), "sd": float(cs.std(ddof=1)),
            "n": len(SEEDS)},
        "energy": {
            "total_start": float(et[0]), "total_end": float(et[-1]),
            "largest_positive_step": float(max(0.0, incr.max())),
            "split_end": {k: float(v) for k, v in e_end.items()}},
        "causality": {
            "frozen_chi_channel": float(fz_chi),
            "kinetics_fast_area": fast["area_end"],
            "kinetics_slow_area": slow["area_end"],
            "kinetics_fast_contrast": float(c_fast),
            "kinetics_slow_contrast": float(c_slow)},
    }
    os.makedirs(args.output, exist_ok=True)
    rpath = os.path.join(args.output, "results.json")
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
    print(f"\nresults -> {rpath}")

    if not args.no_check:
        ok, rows = _check.check_files(rpath, os.path.join(_HERE,
                                                          "baseline.yaml"))
        print("\nbaseline check:\n" + _check.format_table(rows))
        if not ok:
            print("\nBASELINE CHECK FAILED")
            raise SystemExit(2)
        print("\nbaseline check PASSED")


if __name__ == "__main__":
    main()
