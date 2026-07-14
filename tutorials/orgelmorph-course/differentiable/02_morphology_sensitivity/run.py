"""OrgElMorph course - Differentiable D2 driver (the file the student runs).

    python run.py                 # dJ/dchi, dJ/dkappa of the morphology
    python run.py --steps 12      # longer rollout (more demixing)

Computes the demixing-amplitude observable J of a short Cahn-Hilliard
morphology and its sensitivities to the Flory interaction chi and the
gradient penalty kappa, by adjoint, and SELF-CHECKS each against central
finite differences.  Compare with EXPECTED.md."""
import argparse

from sensitivity import sensitivity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--chi", type=float, default=2.5)
    ap.add_argument("--kappa", type=float, default=1e-2)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    r = sensitivity(level=args.level, n_steps=args.steps, chi=args.chi,
                    kappa=args.kappa, device=args.device)
    print(f"\n=== Morphology sensitivity "
          f"({r['side']}x{r['side']} nodes, {r['n_steps']} steps, "
          f"chi={r['chi']}, kappa={r['kappa']}) ===")
    print(f"  demixing amplitude  J = 1/2 sum (c_N - c_bar)^2 = "
          f"{r['J']:.6e}   (c_bar = {r['c_bar']:.3f})\n")
    print(f"  {'sensitivity':14s} {'adjoint':>14s} {'finite diff':>14s} "
          f"{'adj/fd':>10s}")
    print(f"  {'dJ/dchi':14s} {r['dJdchi_adj']:+14.6e} "
          f"{r['dJdchi_fd']:+14.6e} {r['rel_chi']:10.2e}   "
          f"(chi up -> more demixed: sign > 0)")
    print(f"  {'dJ/dkappa':14s} {r['dJdkap_adj']:+14.6e} "
          f"{r['dJdkap_fd']:+14.6e} {r['rel_kap']:10.2e}   "
          f"(kappa up -> interfaces blur: sign < 0)")

    worst = max(r["rel_chi"], r["rel_kap"])
    print(f"\n  worst adjoint-vs-FD agreement: {worst:.2e}")
    # THE GATE: the adjoint sensitivity must match finite differences.
    assert worst < 1e-6, ("adjoint disagrees with finite diff", worst)
    assert r["dJdchi_adj"] > 0, "expected dJ/dchi > 0 (more chi -> demix)"
    assert r["dJdkap_adj"] < 0, "expected dJ/dkappa < 0 (more kappa -> blur)"
    print("  SELF-CHECK PASSED: adjoint sensitivities match FD, signs are "
          "physical.")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
