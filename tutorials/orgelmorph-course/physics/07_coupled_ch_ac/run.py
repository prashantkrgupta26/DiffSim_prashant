"""OrgElMorph course - Physics P7 driver (the file the student runs).

    python run.py                 # crystallization-driven demixing
    python run.py --level 6

Runs the coupled (M,K)=(2,1) system with crystallization ON and OFF and
prints the demixing purity contrast (how much the crystallizing species
concentrates), then repeats the ON case at (3,1) and (3,2) to show the
same machinery scales.  Compare with EXPECTED.md; figures via
gen_figures.py."""
import argparse

import numpy as np

from coupled import build_mesh_dm, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print("\n=== Coupled Cahn--Hilliard + Allen--Cahn ===")
    on = run(dm, mesh, cons, M=2, K=1, crystallize=True,
             device=args.device)
    off = run(dm, mesh, cons, M=2, K=1, crystallize=False,
              device=args.device)
    print(f"  (M,K)=(2,1) crystallization ON : phi0 inside crystal "
          f"{on['inside']:.3f} vs outside {on['outside']:.3f}  "
          f"-> contrast {on['contrast']:+.3f}  (area {on['area_end']:.3f})")
    print(f"  (M,K)=(2,0) crystallization OFF: pure-CH demixing "
          f"contrast {off['contrast']:+.3f}")
    amp = on["contrast"] / max(off["contrast"], 0.02)
    print(f"  pure Cahn--Hilliard barely demixes (contrast "
          f"{off['contrast']:+.3f}); crystallization amplifies it "
          f"{amp:.0f}x")

    print("  --- scaling up the ladder (crystallization ON) ---")
    for M, K in [(3, 1), (3, 2)]:
        r = run(dm, mesh, cons, M=M, K=K, crystallize=True,
                device=args.device)
        print(f"  (M,K)=({M},{K}): demixing contrast {r['contrast']:+.3f}"
              f", crystalline area {r['area_end']:.3f}")
    print("\nCrystallization expels the other components from the "
          "crystal (large positive contrast), a demixing that ordinary "
          "Cahn--Hilliard does not produce.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
