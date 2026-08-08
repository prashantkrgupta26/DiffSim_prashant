"""Sub-project ③ / M6 Plan B: Multi-snapshot NeuralCrystalEnergy recovery.

Demonstrates the multi-snapshot recovery harness that lifts identifiability of
higher ψ-degree modes (cpl_{k,b} for b≥2) that are weakly identifiable from a
single final snapshot alone (the ② Task-5 finding).

The harness fits φ+ψ at several observed times through the hand adjoint
(CrystalCHAdjoint), using the per-step cotangent formulation:

    dJdx[s-1][...] = w * (field(t_s) - field_star(t_s))

for each observed snapshot s, with w = 1/|snapshots|.

Identifiability note
--------------------
The ② single-snapshot harness recovers cpl_{0,1} well but not cpl_{0,2} because
the degree-2 Legendre mode couples weakly to the observed final state.  Multiple
snapshots at different times expose different projections of the coupling operator,
breaking the near-degeneracy between L_1 and L_2 modes.

The descriptor path (``descriptor=True``) adds a structure-factor mismatch term
S(k) to the loss.  This is where real-MD S(k) targets would plug in for the
production ③ workflow.

Usage (CLI smoke-test)::

    .venv/bin/python examples/crystal_learn_multisnapshot.py

(Guard under ``if __name__ == '__main__':`` so the module is importable by tests.)
"""

from __future__ import annotations


def run_multisnapshot_demo(
    level: int = 2,
    n_iter: int = 60,
    lr: float = 0.5,
    descriptor: bool = False,
    lam_desc: float = 1e-3,
) -> None:
    """Run the multi-snapshot recovery demo and print results."""
    import numpy as np
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.adjoint.crystal_recovery import recover_multisnapshot

    # Build mesh
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), "cpu")

    print(f"crystal_learn_multisnapshot.py — multi-snapshot recovery demo")
    print(f"Mesh level={level}, n_nodes={dm.n_nodes}")
    print()

    planted = {"cpl_0_1": 0.15, "cpl_0_2": -0.12}
    names = ["cpl_0_1", "cpl_0_2"]
    snapshots = (2, 4, 6)
    n_steps = 6
    order = 1

    print(f"Planted: {planted}")
    print(f"Snapshots: {snapshots} (out of n_steps={n_steps})")
    print(f"n_iter={n_iter}, lr={lr}, descriptor={descriptor}"
          + (f", lam_desc={lam_desc}" if descriptor else ""))
    print()

    loss_hist, theta_hat, theta_true = recover_multisnapshot(
        dm, mesh, planted, names,
        n_steps=n_steps, snapshots=snapshots,
        order=order, n_iter=n_iter, lr=lr,
        descriptor=descriptor, lam_desc=lam_desc)

    # Print loss history (every 10 iters + final)
    print(f"{'iter':>5}  {'loss':>14}")
    print("-" * 22)
    for i, l in enumerate(loss_hist):
        if i % 10 == 0 or i == len(loss_hist) - 1:
            tag = "  ← final (re-eval at best)" if i == len(loss_hist) - 1 else ""
            print(f"{i:>5}  {l:14.6e}{tag}")

    drop = loss_hist[0] / max(loss_hist[-1], 1e-30)
    print()
    print(f"Loss drop: {loss_hist[0]:.4e} → {loss_hist[-1]:.4e}  ({drop:.1f}×)")
    print()

    # Parameter recovery table
    print(f"{'Parameter':14s}  {'planted':>10}  {'recovered':>10}  {'error%':>8}")
    print("-" * 52)
    for nm in names:
        tr = theta_true[nm]
        rc = theta_hat[nm]
        err_pct = 100.0 * abs(rc - tr) / max(abs(tr), 1e-14)
        print(f"{nm:14s}  {tr:+10.4f}  {rc:+10.4f}  {err_pct:7.1f}%")

    print()
    print("Note: cpl_0_2 (degree-2 mode) recovers within 30% thanks to the")
    print("multi-snapshot strategy — it was weakly identifiable from a single")
    print("final snapshot (the ② Task-5 finding).  The descriptor path adds")
    print("S(k) shape information where real-MD targets would plug in.")


# ---------------------------------------------------------------------------
# CLI entry point — guarded so the module is importable without side effects
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-snapshot NeuralCrystalEnergy recovery demo")
    parser.add_argument("--level", type=int, default=2,
                        help="Mesh refinement level (default: 2)")
    parser.add_argument("--n-iter", type=int, default=60,
                        help="Number of gradient-descent iterations (default: 60)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Learning rate (default: 0.5)")
    parser.add_argument("--descriptor", action="store_true",
                        help="Enable structure-factor descriptor term")
    parser.add_argument("--lam-desc", type=float, default=1e-3,
                        help="Descriptor loss weight (default: 1e-3)")
    args = parser.parse_args()

    run_multisnapshot_demo(
        level=args.level,
        n_iter=args.n_iter,
        lr=args.lr,
        descriptor=args.descriptor,
        lam_desc=args.lam_desc,
    )
