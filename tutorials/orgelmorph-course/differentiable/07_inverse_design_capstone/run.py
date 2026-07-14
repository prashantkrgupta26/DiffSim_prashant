"""OrgElMorph course - Differentiable D7 capstone driver (the file you run).

    python run.py                 # design a T(t) schedule to hit a target
                                  # crystalline fraction

Closes the loop: PDE-constrained design.  First FD-verifies the design
objective's schedule gradient, then optimises the temperature schedule
T(t) (bounded L-BFGS-B) to hit a target crystalline fraction, and asserts
the achieved fraction converges to the target.  Compare with EXPECTED.md."""
import argparse

from design import design, fd_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # 1. FD-verified gradient self-check
    gT_adj, gT_fd, rel = fd_check(level=args.level, NS=args.steps,
                                  device=args.device)
    print("\n=== D7 capstone: inverse-design a T(t) process route ===")
    print("  schedule-gradient self-check (adjoint vs finite diff):")
    for i, (a, f, r) in enumerate(zip(gT_adj, gT_fd, rel)):
        print(f"    dJ/dT[{i}]  adj={a:+.6e} fd={f:+.6e} rel={r:.2e}")
    assert rel.max() < 1e-6, ("design gradient disagrees with FD", rel)

    # 2. design the schedule
    r = design(level=args.level, NS=args.steps, device=args.device)
    print(f"\n  target crystalline fraction (from a cold ramp): "
          f"{r['target']:.5f}")
    print(f"  flat initial guess T=1.0 gives:                 "
          f"{r['m0']:.5f}")
    print(f"  designed schedule achieves:                     "
          f"{r['achieved']:.5f}   |diff|={r['diff']:.2e}")
    print(f"  objective J: {r['J0']:.3e} (initial) -> {r['Jend']:.3e} "
          f"(designed), in {r['n_iter']} evaluations")
    print(f"  optimised schedule T(t) = "
          + ", ".join(f"{v:.3f}" for v in r["T_opt"]))
    interior = bool((r["T_opt"] > 0.2 + 1e-6).all()
                    and (r["T_opt"] < 2.0 - 1e-6).all())
    print(f"  schedule is interior (not pinned at the bounds): {interior}")

    # THE GATE: the designed process hits the target.
    assert r["diff"] < 1e-5, ("design did not reach target", r["diff"])
    assert r["Jend"] < 1e-8 * max(r["J0"], 1e-30) or r["Jend"] < 1e-16, \
        "objective did not collapse"
    print("\n  SELF-CHECK PASSED: gradient matches FD; the designed T(t) "
          "hits the target crystalline fraction.")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
