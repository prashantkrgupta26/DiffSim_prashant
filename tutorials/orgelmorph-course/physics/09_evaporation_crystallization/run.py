"""OrgElMorph course - Physics P9 driver (the file the student runs).

    python run.py                 # the full evaporation-crystallization arc
    python run.py --level 5

Runs the drying ternary film twice: seeds implanted in the WET film
(dissolve) and seeds implanted MID-DRYING (grow to a crystalline film).
Prints the drying state at implant and the terminal crystalline area --
the evaporation-INDUCED crystallization mechanism.  Compare with
EXPECTED.md; figures via gen_figures.py."""
import argparse

from arc import build_mesh_dm, run_arc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dm, mesh, cons = build_mesh_dm(args.level, device=args.device)

    print("\n=== Evaporation-induced crystallization (drying film) ===")
    wet = run_arc(dm, mesh, cons, wet=True, device=args.device)
    print(f"  WET implant  (phi_s = {wet['phis_implant']:.3f}): "
          f"crystalline area {wet['area_implant']:.4f} -> "
          f"{wet['area_final']:.4f}, psi_max {wet['psi_max']:.2f}  "
          f"(DISSOLVES)")
    dry = run_arc(dm, mesh, cons, wet=False, device=args.device)
    print(f"  DRY implant  (phi_s = {dry['phis_implant']:.3f}): "
          f"crystalline area {dry['area_implant']:.4f} -> "
          f"{dry['area_final']:.4f}, psi_max {dry['psi_max']:.2f}  "
          f"(GROWS)")
    print("\nSame seeds: below the solubility (wet) they redissolve; "
          "above it (after drying concentrates the film) they grow to a "
          "crystalline film.  Crystallization strictly AFTER solvent "
          "loss.  Compare with EXPECTED.md.")


if __name__ == "__main__":
    main()
