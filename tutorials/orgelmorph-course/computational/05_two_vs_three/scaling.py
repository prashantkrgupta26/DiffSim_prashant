"""OrgElMorph course - Computational C5: two dimensions versus three.

Importable core for the scaling concept.  Going from 2-D to 3-D is not a
constant factor -- it changes the growth LAW.  This module measures the
change live on small meshes (real Cahn-Hilliard systems) and connects it
to the device-scale ladders from the dev notes.

What changes, and why it forces the device path + blockch of C4:

  DOFS PER LEVEL.  Each refinement multiplies the node count by 2^dim:
  x4 in 2-D, x8 in 3-D.  Four levels of refinement is 256x the work in
  2-D but 4096x in 3-D.

  STENCIL / nnz-PER-DOF.  A node couples to its 3^dim - 1 neighbors: an
  8-neighbor (9-point) stencil in 2-D, a 26-neighbor (27-point) stencil
  in 3-D.  For the mixed (c, mu) system that is ~17 nonzeros/dof in 2-D
  and ~50 in 3-D (measured below) -- so the matrix is both BIGGER (more
  dofs) and DENSER (more nnz/dof).

  int32 CEILING.  The CSR index arithmetic is 32-bit; nnz > 2^31 ~ 2.1e9
  overflows the slot arithmetic.  At (M=3, K=2) production physics the
  128x128x48 superset pattern ALREADY overflows (2.17e9 nnz, dev note
  B5), which is why the block-masked pattern exists.

  DIRECT-SOLVER FILL.  On top of all that, a direct factorization's
  fill-in is far worse in 3-D (larger bandwidth) -- the cuDSS memory
  ceiling of C4.  This is the compounding reason 3-D needs the device
  assembly path and the blockch solver.

The 2-D and tiny-3-D rows are LIVE (real captured CH Jacobians); the
device-scale rows are CITED from the dev notes (minutes/step to run).
"""
import time

import numpy as np

import diffsim.solvers.linsolve as _linsolve
from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper

INT32_MAX = 2 ** 31                     # CSR slot-arithmetic ceiling
CSR_BYTES_PER_NNZ = 12                  # 8 (float64 val) + 4 (int32 col)

# The device-scale ladder, MEASURED and cited (NOT re-run):
#   docs/dev/2026-07-13-m5-device-assembly.md (D3, slab64 nnz),
#   docs/dev/2026-07-13-blockch-mpf.md (B4 film128, B5 mk32 int32 study).
# (label, dofs, nnz, note)
CITED_LADDER = [
    ("3d_slab64 (64x64x16)", 417_792, 65_028_096,
     "largest cuDSS-stable on 48 GB"),
    ("3d_film128 (128x128x64)", 6_389_760, 1_020_000_000,
     "blockch on ONE 48 GB card; cuDSS cannot"),
    ("mk32 128x128x48 (M=3,K=2)", 8_028_160, 2_170_000_000,
     "superset nnz OVERFLOWS int32 -> block-masked pattern required"),
    ("mk32 256x256x128 (Nova)", 84_541_440, 22_800_000_000,
     "no single-card stored CSR -> matrix-free or multi-GPU"),
]


def _capture_nnz(dim, level, device="cuda:0", do_time=True):
    """Assemble one real CH Newton system at this (dim, level) and return
    (dofs, nnz, step_seconds).  Intercepts the stepper's solve to grab
    the assembled Jacobian's nnz -- the true storage the solver faces."""
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.02, order=1,
                             linsolver="cudss")
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    cap = {}
    orig = _linsolve.solve_linear

    def spy(A, b, **kw):
        if "A" not in cap:
            cap["A"] = A
        return orig(A, b, **kw)

    _linsolve.solve_linear = spy
    try:
        t0 = time.perf_counter()
        st.step()
        dt = time.perf_counter() - t0
    finally:
        _linsolve.solve_linear = orig
    return dm.n_nodes * 2, int(cap["A"].nnz), float(dt)


def scaling_row(dim, level, device="cuda:0"):
    dofs, nnz, dt = _capture_nnz(dim, level, device)
    return dict(dim=dim, level=level, dofs=dofs, nnz=nnz,
                nnz_per_dof=nnz / dofs,
                mem_mb=nnz * CSR_BYTES_PER_NNZ / 1e6, step_s=dt)


def measure_scaling(device="cuda:0", levels_2d=(5, 6, 7),
                    levels_3d=(3, 4, 5)):
    """Live 2-D and tiny-3-D scaling of the real CH system.  Returns two
    lists of per-level records plus the measured growth factors."""
    r2 = [scaling_row(2, lv, device) for lv in levels_2d]
    r3 = [scaling_row(3, lv, device) for lv in levels_3d]

    def growth(recs):
        return [recs[i + 1]["dofs"] / recs[i]["dofs"]
                for i in range(len(recs) - 1)]

    return dict(two_d=r2, three_d=r3,
                dof_growth_2d=float(np.mean(growth(r2))),
                dof_growth_3d=float(np.mean(growth(r3))),
                nnz_per_dof_2d=float(np.mean([r["nnz_per_dof"] for r in r2])),
                nnz_per_dof_3d=float(np.mean([r["nnz_per_dof"] for r in r3])))


def int32_headroom(nnz):
    """How close a given nnz is to the 32-bit CSR ceiling."""
    return nnz / INT32_MAX
