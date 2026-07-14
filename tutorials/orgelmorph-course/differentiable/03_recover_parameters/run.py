"""OrgElMorph course - Differentiable D3 driver (the file the student runs).

    python run.py                 # recover (chi, kappa) from a morphology
    python run.py --steps 8

First FD-verifies the fitting-loss gradient (the self-check the optimiser
relies on), then recovers the ground-truth material parameters from a
target morphology by L-BFGS-B on the adjoint gradient, and asserts the
loss fell and the parameters reached truth.  Compare with EXPECTED.md."""
import argparse

from recover import recover, fd_check


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # 1. FD-verified gradient self-check (off-truth point)
    g_adj, g_fd, rel = fd_check(level=args.level, n_steps=args.steps,
                                device=args.device)
    print("\n=== D3: recover material parameters from a morphology ===")
    print("  gradient self-check (adjoint vs finite diff):")
    for nm, a, f, r in zip(("dL/dchi", "dL/dkappa"), g_adj, g_fd, rel):
        print(f"    {nm:10s} adj={a:+.6e} fd={f:+.6e} rel={r:.2e}")
    assert rel.max() < 1e-6, ("loss gradient disagrees with FD", rel)

    # 2. inverse problem: recover (chi, kappa)
    r = recover(level=args.level, n_steps=args.steps, device=args.device)
    ct, kt = r["truth"]
    cg, kg = r["guess"]
    cr, kr = r["recovered"]
    print(f"\n  target morphology: {r['side']}x{r['side']} nodes, "
          f"{r['n_steps']} steps")
    print(f"  {'param':8s} {'truth':>10s} {'guess':>10s} "
          f"{'recovered':>12s} {'|err|':>11s}")
    print(f"  {'chi':8s} {ct:10.5f} {cg:10.5f} {cr:12.6f} "
          f"{r['err_chi']:11.2e}")
    print(f"  {'kappa':8s} {kt:10.5f} {kg:10.5f} {kr:12.6f} "
          f"{r['err_kappa']:11.2e}")
    print(f"  loss: {r['loss0']:.4e} (guess) -> {r['loss_end']:.4e} "
          f"(recovered), in {r['n_iter']} evaluations")

    # THE GATE: loss collapses and both parameters reach truth.
    assert r["loss_end"] < 1e-8 * max(r["loss0"], 1e-30), "loss did not fall"
    assert r["err_chi"] < 1e-3, ("chi not recovered", r["err_chi"])
    assert r["err_kappa"] < 1e-4, ("kappa not recovered", r["err_kappa"])
    print("\n  SELF-CHECK PASSED: gradient matches FD; loss collapsed; "
          "(chi, kappa) recovered to truth.")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
