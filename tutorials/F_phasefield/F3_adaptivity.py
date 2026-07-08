"""F3 — Phase field III: spatial adaptivity (markers, epochs, transfer).

LEARNING OUTCOME. You can (i) mark interface elements with a cheap
nodal-spread indicator, (ii) refine + 2:1-balance an octree and rebuild
the problem (an EPOCH), (iii) move the state across meshes with the M3
transfer operator P — and explain the punchline: for NESTED refinement
the transfer is EXACT, so mass drift across a re-mesh is ZERO. Measured
zero. Not 1e-12 — the same floating-point number.

WHY ADAPT. A phase-field solution is expensive exactly where it is
interesting: the interface, width w ~ sqrt(2 kappa), needs 3-4 elements
across (F1's resolution rule), while the bulk (c = +-1, mu flat) is
happy on cells 10x larger. Uniformly meshing a 3-D film at interface
resolution is how you run out of A100 memory before breakfast. The
standard answer (Wodo & Ganapathysubramanian JCP 2011 — the reference
design for this stack): track the interface band with local refinement
and re-mesh as the morphology evolves.

THE THREE MOVING PARTS.

1. MARKER. Any cheap monotone proxy for "interface passes through this
   element" works. Ours: the nodal spread max(c) - min(c) over the
   element's nodes; > 0.5 means the element straddles a chunk of the
   tanh profile (c sweeps -1 -> +1 across w). |grad c| h is the
   equivalent continuum statement (the P0/spec marker). Cheap beats
   optimal here — the marker runs every epoch.

2. EPOCH = refine + balance + rebuild. refine_elements splits marked
   cells; balance2to1 enforces the 2:1 face rule (constraints get
   hanging nodes otherwise unsupported); then mesh/constraints/device
   arrays are rebuilt from scratch. MEASURED cost: 0.03-0.05 s per
   epoch at this size — noise next to the stepping. Re-meshing "from
   scratch" sounds wasteful and is the correct engineering call at
   these costs.

3. TRANSFER. A field is nodal values TIMES basis functions; new mesh,
   new basis, so nodal values must be remapped: c_new = P c_old.
   P (src/diffsim/adaptivity/transfer.py, M3 rung 1) has one row per
   new node, built in three classes:
     - shared nodes (coordinates match): identity through the old
       constraint expansion;
     - new nodes inside old elements: FE interpolation (point-eval
       weights on the OLD basis);
     - nodes outside the old domain (moving boundaries only — never
       fires here): nearest-node O(h) fallback.
   P is an EXPLICIT sparse matrix — so P^T is mechanical, which is what
   makes adaptivity DIFFERENTIABLE (the adjoint crosses the re-mesh as
   P^T; that is M3's whole theorem, reused here for free).

THE NESTEDNESS PUNCHLINE (why the drift is exactly zero). Refinement
without coarsening makes the new FE space a SUPERSET of the old: every
old p1 basis function is exactly a linear combination of children's
(the octree children interpolate their parent's linears exactly). So
interpolation P is not an approximation — the transferred field is THE
SAME FUNCTION, merely re-expressed. Same function => same integrals =>
Int c dV unchanged to the last bit. MEASURED (benchmarks/
adaptive_ch_opener.py): mass drift +0.000e+00 at both epochs.
COARSENING breaks nestedness (the fine function leaves the coarse
space) — then plain interpolation loses mass and you need an
L2 PROJECTION (mass-matrix-weighted, conservative by construction).
That is the recorded refinement; EXPLORE 3 makes you feel why.

EXPECTED RESULTS (measured for this script, RTX 6000 Ada):
    pre-adapt: 1024 elems, mass +1.454798543757e-03
    epoch 0: 291 marked -> 1897 elems (0.03 s), mass drift +0.000e+00
    epoch 1: 406 marked -> 3235 elems (0.06 s), mass drift +0.000e+00
    the mass PRINTS IDENTICALLY to all 12 shown digits across both
    re-meshes — that is what "structural + nested" buys.
    Energy across the transfer: |dE| ~ 1e-4, NOT 1e-15 — and that is
    correct, not a bug: the transferred c is the same FUNCTION, but
    energy is evaluated by QUADRATURE, and the quartic (c^2-1)^2/4 is
    not integrated exactly by the p1 rule — parent and children sample
    it at different points, so the quadrature ERROR changes (and
    shrinks: the children's rule is more accurate). Mass is linear in
    c, integrated EXACTLY at p1 — which is why ITS drift is zero to
    the bit. One transfer, two functionals, two verdicts: know what
    your quadrature is exact for.

Run:  python tutorials/F_phasefield/F3_adaptivity.py
"""
import os
import sys
import time

import numpy as np
import warp as wp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "src"))
from diffsim.octree.build import build_uniform, refine_elements
from diffsim.octree.balance import balance2to1
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim.adaptivity.transfer import transfer_operator

DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
M, KAPPA, DT = 1.0, 5e-4, 0.02


def make_problem(tree):
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), DEVICE)
    return mesh, cons, dm


def diagnostics(st, dm, mesh, c_free):
    """(mass, energy) — one GP sweep, same quadrature as the residual."""
    v, g = st._gp_scalar(c_free)
    m = E = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** 2, b["nqp"])
        m += float((wq * v[pv]).sum())
        E += float((wq * (0.25 * (v[pv] ** 2 - 1) ** 2
                          + 0.5 * KAPPA * (g[pv] ** 2).sum(1))).sum())
    return m, E


def mark_interface(tree, mesh, cons, c_free, thresh=0.5):
    """Step 1: the marker. Nodal spread of c over each element; > thresh
    flags the interface band. Returns a bool mask over tree elements."""
    cfull = np.asarray(cons.T @ c_free)
    mark = np.zeros(len(tree), bool)
    for pv in mesh.conn_of:            # p-bins (all p1 here)
        conn = mesh.conn_of[pv]
        spread = cfull[conn].max(1) - cfull[conn].min(1)
        mark[mesh.bins[pv][spread > thresh]] = True
    return mark


def epoch(tree, mesh, cons, st, c, mu, thresh=0.5):
    """Steps 2+3: refine + balance + rebuild, then transfer (c, mu)."""
    mark = mark_interface(tree, mesh, cons, c, thresh)
    t0 = time.time()
    tree2 = balance2to1(refine_elements(tree, mark))
    mesh2, cons2, dm2 = make_problem(tree2)
    P = transfer_operator(mesh, cons, mesh2)          # [n_new_nodes, n_old_free]
    cost = time.time() - t0
    # P returns ALL-node values on the new mesh; the free vector is the
    # free-node subset (hanging nodes are constrained, not unknowns).
    c2 = (P @ c)[cons2.free_nodes]
    mu2 = (P @ mu)[cons2.free_nodes]
    st2 = CahnHilliardStepper(dm2, M, KAPPA, DT, order=1)
    st2.set_initial(lambda x: np.zeros(len(x)))
    st2.x[0::2] = c2
    st2.x[1::2] = mu2
    st2.hist = [c2.copy(), c2.copy()]   # BDF history restarts (order 1
    #   first step after a re-mesh — the standard epoch protocol)
    return tree2, mesh2, cons2, dm2, st2, c2, mu2, int(mark.sum()), cost


def main(n_epochs=2, steps_before=8, steps_per_epoch=5):
    print("=" * 70)
    print("Adaptive spinodal CH: 2 re-mesh epochs, conservation watched")
    print("=" * 70)
    tree = build_uniform(5, dim=2)
    mesh, cons, dm = make_problem(tree)
    st = CahnHilliardStepper(dm, M, KAPPA, DT, order=1)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)))
    print(f"  uniform level 5: {len(tree)} elems; developing the "
          f"pattern for {steps_before} steps...")
    for _ in range(steps_before):
        c, mu = st.step()
    m, E = diagnostics(st, dm, mesh, c)
    print(f"  pre-adapt:  elems {len(tree):6d}  mass {m:+.12e}  "
          f"E {E:.6f}")
    for ep in range(n_epochs):
        (tree, mesh, cons, dm, st, c, mu,
         nmark, cost) = epoch(tree, mesh, cons, st, c, mu)
        m2, E2 = diagnostics(st, dm, mesh, c)
        print(f"  epoch {ep}: +{nmark} marked -> {len(tree)} elems "
              f"({cost:.2f} s)")
        print(f"    after transfer: mass {m2:+.12e}  "
              f"drift {m2 - m:+.3e}  |dE| {abs(E2 - E):.1e}")
        for _ in range(steps_per_epoch):
            c, mu = st.step()
        m, E = diagnostics(st, dm, mesh, c)
        print(f"    after {steps_per_epoch} steps: mass {m:+.12e}  "
              f"E {E:.6f}  c in [{c.min():+.2f},{c.max():+.2f}]")
    print("  Drift EXACTLY zero at each epoch: nested refinement means")
    print("  P re-expresses the same function — conservation by")
    print("  NESTEDNESS, not by luck. (Coarsening would break this.)")


if __name__ == "__main__":
    main()
    print("""
EXPLORE
 1. Marker thresholds. Sweep thresh in {0.2, 0.5, 0.8}. Tabulate elems
    after 2 epochs vs the energy at the shared end time (run a uniform
    level-6 reference for truth). Where is the knee of the accuracy-
    per-element curve? Now BREAK it: thresh = 0.95 under-resolves the
    band — do you see F1's pinning failure return locally?
 2. Band width. Our marker refines only the straddling elements; Wodo
    JCP 2011 refines a BUFFER around the interface so the band survives
    several steps of motion before the next epoch. Add one ring of
    neighbors (elements sharing a node with a marked one) and measure:
    epochs needed per 20 steps, with and without the buffer.
 3. THE COARSENING QUESTION. Add de-refinement: merge sibling quads
    whose parent's nodal spread is < 0.05 (bulk cells). Transfer with
    plain interpolation P and print the mass drift — it is not zero
    anymore (the fine wiggle cannot be represented coarsely; nestedness
    is gone). Then implement the L2 projection restriction
    (solve M_coarse c_coarse = P_inj^T M_fine-ish rhs — or simply
    project element-wise) and show conservation returns. This is the
    recorded refinement in the M4 ledger; doing it earns you the right
    to coarsen in production.
 4. TEMPORAL adaptivity (preview of the Wodo JCP 2011 controller). The
    coarsening stage tolerates dt 10-100x larger than the initial
    decomposition burst. Implement step-doubling: take one step of 2*dt
    and two of dt from the same state; the difference estimates the
    local truncation error (LTE ~ C dt^3 for BDF2). Accept/reject with
    dt_new = dt (tol/LTE)^(1/3), floors and ceilings applied. Plot the
    accepted dt(t) over a full quench: it should grow ~monotonically
    through coarsening — the paper's signature figure.
""")
