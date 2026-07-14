"""OrgElMorph course - Physics P5 driver (the file the student runs).

    python run.py                 # matched-dryness rate comparison
    python run.py --level 6

Dries the SAME dilute ternary film at several evaporation rates, driving
every rate PAST the same dryness so morphology can be compared at a
MATCHED mean solvent fraction phi_s (not at a common time -- that would
confound drying RATE with final STATE).  Prints, per rate and per matched
dryness, the lateral domain wavelength (in cells, a real resolved length)
and the per-solute content-conservation residual.

For the full workflow (config, provenance, tolerance check, figures) use
run_harness.py; compare with EXPECTED.md."""
import argparse

import numpy as np

from evaporation import build_mesh_dm, run_film, morphology_metrics

RATES = [0.3, 0.45, 0.6]     # k_e; all dry past phi_s = 0.10
MATCHED = [0.30, 0.20, 0.10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--t_end", type=float, default=8.0)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print(f"\n=== Drying ternary film: matched-dryness rate comparison "
          f"({dm.n_nodes} nodes) ===")
    header = "  {:>10}{:>10}{:>8}{:>9}".format("k_e(Bi)", "reason", "t_dry",
                                               "|dm|/m")
    header += "".join(f"{'wl@'+f'{p:.2f}':>11}" for p in MATCHED)
    print(header)
    for k_e in RATES:
        r = run_film(dm, mesh, cons, k_e=k_e, t_end=args.t_end,
                     device=args.device)
        wl = np.array([morphology_metrics(gp)["wl_cells"]
                       for gp in r["traj_gp"]])
        phis = r["phis"]; order = np.argsort(phis)
        wlm = [np.interp(p, phis[order], wl[order]) for p in MATCHED]
        cp, h = r["content_p"], r["h"]
        drift = abs(cp[-1] * h[-1] - cp[0] * h[0]) / (cp[0] * h[0])
        Bi = k_e / 0.2       # h0=1, D_s=0.2
        cells = "".join(f"{v:>11.2f}" for v in wlm)
        print(f"  {k_e:.2f}({Bi:.1f}){r['reason']:>10}{r['t_dry']:>8.2f}"
              f"{drift:>9.0e}{cells}")
    print("\nRead these DOWN a column (matched dryness), not across: at fixed "
          "phi_s the rate barely moves the lateral wavelength -- DRYNESS, not "
          "rate, sets the domain size.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
