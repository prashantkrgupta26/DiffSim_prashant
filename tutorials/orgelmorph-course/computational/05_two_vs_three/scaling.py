"""OrgElMorph course - Computational C5: two dimensions versus three.

Importable core for the scaling concept.  Going from 2-D to 3-D is not a
constant factor -- it changes the growth LAW, the storage, AND the physics.
This module measures all three on small meshes (real Cahn-Hilliard systems)
and connects them to the device-scale ladders from the dev notes.

WHAT ACTUALLY CHANGES:

  1. DOFS PER LEVEL.  Each refinement multiplies the node count by 2^dim:
     x4 in 2-D, x8 in 3-D.  Four levels is 256x the work in 2-D but 4096x
     in 3-D.

  2. SPARSITY (nnz per dof).  A dof's row has one entry per (coupled node) x
     (field).  The coupled nodes are those sharing a finite ELEMENT -- for a
     degree-p mesh a (2p+1)^dim block (P1: 3^dim = 9 in 2-D, 27 in 3-D).  So
     nnz/dof = fields x (2p+1)^dim, driven by element CONNECTIVITY, basis
     ORDER (p), number of FIELDS (the (c,mu) block), CONSTRAINTS (hanging
     nodes add couplings), and DIMENSION.  It is NOT a "3^d - 1 finite-
     difference stencil" -- that undercounts (misses self + the field block).
     Measured below: ~17 nnz/dof in 2-D, ~50 in 3-D.

  3. STORAGE.  The CSR values are only part of it; see estimate_capacity.py
     for the full accounting (row pointers, index mirrors, vectors, history,
     solver fill/preconditioner, output).  The int32 CSR index arithmetic
     (an IMPLEMENTATION choice, not a law) overflows at nnz > 2^31 ~ 2.1e9;
     an int64 build or a distributed/block-masked pattern lifts that wall.

  4. PHYSICS.  At the SAME resolution a 3-D morphology is not a thicker 2-D
     one: interfaces are surfaces not curves, domains percolate differently,
     and coarsening exponents and interfacial-area budgets differ.  So 2-D is
     not "cheap 3-D" -- it answers a different question.  physics_comparison()
     measures this at equal h.

The 2-D and tiny-3-D rows are LIVE (real captured CH Jacobians); the device-
scale rows are CITED from the dev notes (minutes/step to run).
"""
import time

import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper
from diffsim.diagnostics import morphology as morph

from estimate_capacity import estimate, stencil_nodes

INT32_MAX = 2 ** 31                     # CSR slot-arithmetic ceiling
CSR_BYTES_PER_NNZ = 12                  # 8 (float64 val) + 4 (int32 col)

# The device-scale ladder, MEASURED and cited (NOT re-run):
#   docs/dev/2026-07-13-m5-device-assembly.md (D3, slab64 nnz),
#   docs/dev/2026-07-13-blockch-mpf.md (B4 film128, B5 mk32 int32 study).
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


def sparsity_breakdown(dim, p=1, fields=2):
    """Decompose nnz/dof into its FE origins (spec C5): the coupled-node
    count (2p+1)^dim times the field-block width -- not a 3^d-1 stencil."""
    coupled = stencil_nodes(dim, p)
    return dict(dim=dim, p=p, fields=fields, coupled_nodes=coupled,
                nnz_per_dof=coupled * fields,
                fd_stencil_would_say=3 ** dim - 1)


def _build(dim, level, device="cuda:0"):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    return dm, mesh, cons


def _capture_nnz(dim, level, device="cuda:0"):
    """Assemble one real CH Newton system and return (dofs, nnz, step_s)
    using the SUPPORTED capture API (`capture_system=True` -> `last_system`),
    NOT a monkeypatch."""
    dm, mesh, cons = _build(dim, level, device)
    # cuDSS (GPU direct) for the step -- far faster than splu in 3-D, and the
    # capture is independent of the solver (last_system is stored before it).
    st = CahnHilliardStepper(dm, 1.0, 5e-4, 0.02, order=1,
                             linsolver=_fast_linsolver(), capture_system=True)
    rng = np.random.default_rng(3)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    t0 = time.perf_counter()
    st.step()
    dt = time.perf_counter() - t0
    A, _ = st.last_system
    return dm.n_nodes * 2, int(A.tocsr().nnz), float(dt)


def scaling_row(dim, level, device="cuda:0"):
    dofs, nnz, dt = _capture_nnz(dim, level, device)
    return dict(dim=dim, level=level, dofs=dofs, nnz=nnz,
                nnz_per_dof=nnz / dofs,
                mem_mb=nnz * CSR_BYTES_PER_NNZ / 1e6, step_s=dt)


def measure_scaling(device="cuda:0", levels_2d=(5, 6, 7),
                    levels_3d=(3, 4, 5)):
    """Live 2-D and tiny-3-D scaling of the real CH system."""
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
    """How close a given nnz is to the 32-bit CSR ceiling (impl-specific)."""
    return nnz / INT32_MAX


def memory_accounting(dim, n, solver="blockch", idx_bytes=4):
    """Full memory breakdown at (dim, n) via estimate_capacity (spec C5:
    COMPLETE accounting, not just CSR values)."""
    return estimate(dim, n, solver=solver, idx_bytes=idx_bytes)


# --- physics at equal resolution: 2-D is not cheap 3-D ---------------

def _grid_field(st, cons, dim, side):
    """Reshape the stepper's conserved field to a (side,)*dim grid."""
    c = np.asarray(cons.T @ st.x[0::2])
    return c.reshape((side,) * dim)


def _fast_linsolver():
    """cuDSS if available (GPU direct -- verified accurate on the CH saddle in
    C4, and far faster than splu in 3-D), else splu."""
    try:
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
        return "cudss"
    except Exception:
        return "splu"


def _spinodal_to(dim, level, steps, device="cuda:0", seed=3, dt=0.01):
    dm, mesh, cons = _build(dim, level, device)
    st = CahnHilliardStepper(dm, 1.0, 5e-4, dt, order=1,
                             linsolver=_fast_linsolver())
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: 0.05 * rng.standard_normal(len(x)),
                   mu_init="consistent")
    for _ in range(steps):
        st.step()
    side = int(round(len(mesh.node_coords) ** (1.0 / dim)))
    return _grid_field(st, cons, dim, side), side


def physics_comparison(level=4, steps=30, device="cuda:0"):
    """Run a spinodal quench in 2-D and 3-D at the SAME resolution (same h)
    and measure morphology: interfacial area PER unit volume, phase
    fractions, and the peak-S(q) wavelength.  The point: at equal resolution
    the 3-D pattern is topologically different (surfaces, not curves;
    bicontinuous percolation), so 2-D is not a cheap stand-in for 3-D."""
    out = {}
    for dim in (2, 3):
        field, side = _spinodal_to(dim, level, steps, device)
        dx = 1.0 / (side - 1)
        area = morph.interfacial_area(field, dx=dx)           # length^(d-1)
        vol = 1.0                                              # unit box
        lam = morph.peak_wavelength(field, dx=dx)
        flo, fhi = morph.phase_fractions(field)
        out[dim] = dict(dim=dim, side=side, dx=dx, field=field,
                        interfacial_area=area,
                        interfacial_area_density=area / vol,
                        peak_wavelength=lam,
                        phase_frac_low=flo, phase_frac_high=fhi,
                        field_range=(float(field.min()), float(field.max())))
    return out
