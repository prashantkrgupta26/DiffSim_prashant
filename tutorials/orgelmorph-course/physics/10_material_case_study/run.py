"""OrgElMorph P10 - lightweight student driver (the "does it work" script).

Marches ONE drying film for the chosen material at one spin-speed rung and
prints a matched-dryness self-check.  For the full, tolerance-checked capstone
workflow (the ladder + mesh/time convergence + chi sensitivity, gated against
baseline.yaml) run ``run_harness.py`` instead:

    python run_harness.py --config configs/p10.yaml --mode reference \\
        --output outputs/p10 --overwrite

This driver just demonstrates the loaded-material -> drying-film -> matched
metrics path.  See EXPECTED.md for what the numbers should look like.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir,
                                                "materials")))
from loader import load_system                              # noqa: E402
from material_case import (build_mesh_dm, run_film, ke_ladder_from_biot,
                           chi_triple, classify_provenance, interp_at_phis,
                           film_metrics_vs_phis)             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="PDPP5T_PCBM")
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--rung", type=int, default=2, help="ladder index (0..3)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    sysm = load_system(args.system)
    chi = chi_triple(sysm)
    N = [float(x) for x in sysm.value("N")]
    prov = classify_provenance(sysm)
    print(f"\n=== P10 material case study: {args.system} ===")
    print(f"chi (pf, ps, fs) = {chi}   N = {N}")
    print("provenance (status / source / value-known):")
    for k, p in prov.items():
        print(f"   {k:24s} {p['status']:22s} {str(p['source_type']):10s} "
              f"known={p['value_known']}")

    rungs, window = ke_ladder_from_biot(sysm.value("biot"), 0.2)
    r = rungs[min(args.rung, len(rungs) - 1)]
    print(f"\nrung {r['label']} ({r['rpm']} rpm): k_e={r['k_e']:.3f} "
          f"Bi={r['Bi']:.2f}  (drying-rate window {window})")

    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)
    rec = run_film(dm, mesh, cons, k_e=r["k_e"], chi=chi, N=N,
                   onsager=(0.2, 0.0, 0.2), phi0=(0.20, 0.20),
                   t_end=8.0, dt=1e-3, dt_max=0.01, nchecks=32,
                   phis_stop=0.10, seed=7, device=args.device,
                   linsolver="cudss")
    wl_frac, wl_cells, iface, contrast = film_metrics_vs_phis(rec)
    tgt = [0.30, 0.20, 0.12]
    mc = interp_at_phis(rec["phis"], wl_cells, tgt)
    mk = interp_at_phis(rec["phis"], contrast, tgt)
    print(f"\nmarch: reason={rec['reason']} phis_final={rec['phis_final']:.3f} "
          f"h_final={rec['h_final']:.3f}")
    print("matched dryness   wavelength(cells)   contrast")
    for t in tgt:
        print(f"   phi_s={t:.2f}      {mc[f'{t:.2f}']:8.1f}          "
              f"{mk[f'{t:.2f}']:.3f}")
    dry = mk[f"{min(tgt):.2f}"]; wet = mk[f"{max(tgt):.2f}"]
    print(f"\ncontrast rises wet->dry: {wet:.3f} -> {dry:.3f}  "
          f"({'YES' if np.isfinite(dry) and dry > wet else 'no'})  "
          "-> the demixing DEGREE is set by dryness (the robust invariant).")


if __name__ == "__main__":
    main()
