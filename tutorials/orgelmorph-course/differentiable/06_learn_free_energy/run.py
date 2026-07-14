"""OrgElMorph course - Differentiable D6 driver (the file the student runs).

    python run.py                 # learn f'(c)=c^3-c from morphology snapshots

Recovers a parametrized bulk free energy from a trajectory of morphology
snapshots.  First FD-verifies the trajectory-loss gradient, then recovers
the coefficients from clean data (to ~1e-6), and shows the noisy recovery
sharpen as more snapshots are used.  Compare with EXPECTED.md."""
import argparse

from learn_energy import learn, fd_check, TRUTH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=3)
    ap.add_argument("--nst", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    # 1. FD-verified gradient self-check
    g_adj, g_fd, rel = fd_check(level=args.level, nst=args.nst,
                                device=args.device)
    print("\n=== D6: learn the free energy f(c) from snapshots ===")
    print("  trajectory-loss gradient self-check (adjoint vs finite diff):")
    for i, (a, f, r) in enumerate(zip(g_adj, g_fd, rel)):
        print(f"    dL/da{i}  adj={a:+.6e} fd={f:+.6e} rel={r:.2e}")
    assert rel.max() < 1e-6, ("trajectory gradient disagrees with FD", rel)

    # 2. recover the energy
    r = learn(level=args.level, nst=args.nst, device=args.device)
    print(f"\n  truth      f'(c) = c^3 - c   ->  a = "
          f"[{TRUTH[0]:+.3f}, {TRUTH[1]:+.3f}]")
    print(f"  recovered (clean, {r['nst']} snaps)   a = "
          f"[{r['a_clean'][0]:+.6f}, {r['a_clean'][1]:+.6f}]")
    print(f"  |a - truth| = {r['err_clean']:.2e}   loss = {r['loss_clean']:.2e}")

    print(f"\n  noisy data (sigma={r['noise']}): recovery error vs number "
          f"of snapshots K")
    for K in r["Ks"]:
        print(f"    K = {K:2d} snapshots   |a - truth| = {r['errsK'][K]:.4f}")
    improve = r["errsK"][r["Ks"][0]] / max(r["errsK"][r["Ks"][-1]], 1e-30)
    print(f"  more data -> better: K={r['Ks'][0]} to K={r['Ks'][-1]} "
          f"improves the error {improve:.1f}x")

    # THE GATE: gradient matches FD; clean recovery exact; more data helps.
    assert r["err_clean"] < 1e-5, ("clean recovery failed", r["err_clean"])
    assert r["errsK"][r["Ks"][-1]] < 0.5 * r["errsK"][r["Ks"][0]], \
        ("more snapshots did not help", r["errsK"])
    print("\n  SELF-CHECK PASSED: gradient matches FD; f'(c)=c^3-c recovered; "
          "more snapshots sharpen the fit.")
    print("\nCompare with EXPECTED.md; render figures with gen_figures.py.")


if __name__ == "__main__":
    main()
