"""OrgElMorph course - Differentiable D5 driver (the file the student runs).

    python run.py                 # dJ/d{dh,Tm,dsig,eps2,L} of crystallinity

Differentiates a crystallinity metric J = 1/2 ||psi_N - target||^2
through a short coupled Cahn-Hilliard x Allen-Cahn crystallisation
rollout with respect to the thermodynamic crystallisation parameters, and
SELF-CHECKS every sensitivity against central finite differences.
Compare with EXPECTED.md."""
import argparse

from crystallinity import sensitivity

PRETTY = {"dh": "dh (latent heat)", "Tm": "Tm (melt temp)",
          "dsig": "dsig (barrier)", "eps2": "eps2 (stiffness)",
          "L": "L (AC kinetics)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    r = sensitivity(level=args.level, n_steps=args.steps,
                    device=args.device)
    print(f"\n=== D5: crystallinity sensitivity "
          f"({r['side']}x{r['side']} nodes, {r['n_steps']} steps) ===")
    print(f"  crystallinity metric J = 1/2 ||psi_N - {r['tgt_psi']}||^2 = "
          f"{r['J']:.6e}   (mean crystallinity <psi> = {r['frac']:.4f})\n")
    print(f"  {'parameter':20s} {'adjoint':>14s} {'finite diff':>14s} "
          f"{'adj/fd':>10s}")
    worst = 0.0
    for nm in r["names"]:
        a, f, rel = r["g_adj"][nm], r["g_fd"][nm], r["rel"][nm]
        worst = max(worst, rel)
        print(f"  {PRETTY[nm]:20s} {a:+14.6e} {f:+14.6e} {rel:10.2e}")

    print(f"\n  worst adjoint-vs-FD agreement: {worst:.2e}")
    # THE GATE: every crystallisation sensitivity matches finite differences.
    assert worst < 1e-6, ("crystallinity gradient disagrees with FD", worst)
    print("  SELF-CHECK PASSED: all crystallisation sensitivities match FD.")
    print("\n  (Sensitivity to the NUCLEATION NOISE amplitude is a stochastic")
    print("   object - a documented frontier - not this deterministic gate.)")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
