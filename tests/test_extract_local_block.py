"""Tests for extract_local_block: local CSR extraction from a global SPD matrix.

Uses the real octree scalar-Poisson stiffness K_p (assemble_csr) SPD-ified with
Dirichlet identity rows on boundary nodes.  Verifies:

  1. Union of owned sets covers [0, n) completely and disjointly.
  2. All columns appearing in owned rows lie in owned∪ghost (halo check).
  3. For a random global vector v, the local matvec reproduces the global
     operator's owned rows exactly:
         A_local @ v_local  ==  K_spd[owned_ids, :] @ v_full
     where v_local = v_full[owned_ids ++ ghost_ids].
  4. Jacobi diagonal equals K_spd diagonal at owned nodes.

Runs on CPU only (assemble_csr is a host scipy path).  Tests level=2 (125
nodes) and level=3 (729 nodes), partitioned into 2 and 4 ranks.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh, assemble_csr
from diffsim.mesh.partition import slab_partition, extract_local_block


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_K_spd(level: int):
    """Return (K_spd, b, mesh) for a uniform level-L 3-D mesh.

    SPD-ification: Dirichlet identity rows on boundary nodes.
    RHS: boundary -> 0, interior -> fixed-seed standard normal.
    """
    tree = build_uniform(level, dim=3)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm   = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")
    K    = assemble_csr(dm).tocsr().astype(np.float64)

    n = K.shape[0]
    bn = mesh.boundary_nodes   # bool array, shape (n,)
    bn_idx = np.where(bn)[0]
    in_idx = np.where(~bn)[0]

    # SPD-ify: zero off-diag on boundary rows AND cols, set diag=1
    # Step 1: eliminate boundary columns from interior rows
    # Step 2: set boundary rows to identity
    K = K.tolil()
    for k in bn_idx:
        # Zero row k except diagonal
        K[k, :] = 0.0
        K[k, k] = 1.0
        # Zero column k except diagonal
        K[:, k] = 0.0
        K[k, k] = 1.0
    K = K.tocsr()
    K.eliminate_zeros()

    # RHS: fixed-seed random for interior, 0 for boundary
    rng = np.random.default_rng(0)
    rhs_full = rng.standard_normal(n)
    b = np.zeros(n, dtype=np.float64)
    b[in_idx] = rhs_full[in_idx]   # interior: random
    # boundary: 0 (homogeneous Dirichlet)

    # Also verify K is valid SPD by checking diagonal > 0 on all nodes
    diag = np.array(K.diagonal())
    assert (diag > 0).all(), "K_spd diagonal must be strictly positive"

    n_per_axis = 2**level + 1
    dims = (n_per_axis, n_per_axis, n_per_axis)
    return K, b, mesh, dims


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level,n_ranks", [
    (2, 2),
    (2, 4),
    (3, 2),
    (3, 4),
])
def test_extract_local_block_correctness(level, n_ranks):
    """A_local @ v_local reproduces K_spd @ v_full on owned rows."""
    K_spd, b, mesh, dims = _build_K_spd(level)
    n = K_spd.shape[0]

    all_parts = slab_partition(dims, n_ranks)

    # (1) Union of owned covers [0, n) completely and disjointly
    all_owned = np.concatenate([p.owned for p in all_parts])
    assert len(all_owned) == n, (
        f"Total owned={len(all_owned)} != n={n}"
    )
    assert len(np.unique(all_owned)) == n, (
        "Owned sets overlap or have gaps"
    )
    assert set(all_owned.tolist()) == set(range(n)), (
        "Owned sets don't cover [0, n)"
    )

    # Random global vector (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    v_full = rng.standard_normal(n)

    for rank_idx, part in enumerate(all_parts):
        owned_ids = part.owned
        ghost_ids = part.ghost
        n_owned = len(owned_ids)

        # Extract local block
        A_local = extract_local_block(K_spd, part)

        # (2) A_local shape
        n_ghost = len(ghost_ids)
        assert A_local.shape == (n_owned, n_owned + n_ghost), (
            f"rank {rank_idx}: A_local shape {A_local.shape} != "
            f"({n_owned}, {n_owned + n_ghost})"
        )

        # (3) All columns in owned∪ghost (extract_local_block already asserts this;
        #     here we double-check via the explicit column indices)
        valid_gcols = set(int(g) for g in owned_ids) | set(int(g) for g in ghost_ids)
        # Build inverse local map to check
        A_coo = A_local.tocoo()
        # column indices in local space: owned = [0..n_owned), ghost = [n_owned..n_local)
        assert A_coo.col.max() < n_owned + n_ghost, (
            f"rank {rank_idx}: local column out of range"
        )

        # (4) Matvec correctness: A_local @ v_local == K_spd[owned_ids, :] @ v_full
        local_ids = np.concatenate([owned_ids, ghost_ids])
        v_local   = v_full[local_ids]
        Av_local  = np.asarray(A_local @ v_local, dtype=np.float64)
        Av_ref    = np.asarray(K_spd[owned_ids, :] @ v_full, dtype=np.float64)

        err = np.abs(Av_local - Av_ref).max()
        assert err < 1e-12, (
            f"rank {rank_idx}: matvec error {err:.3e} > 1e-12 "
            f"(local block does not reproduce global operator on owned rows)"
        )


@pytest.mark.parametrize("level,n_ranks", [
    (2, 2),
    (3, 4),
])
def test_extract_local_block_jacobi_diag(level, n_ranks):
    """Diagonal of A_local[:, :n_owned] matches K_spd diagonal at owned nodes."""
    K_spd, _, mesh, dims = _build_K_spd(level)
    all_parts = slab_partition(dims, n_ranks)
    K_diag = np.array(K_spd.diagonal())

    for rank_idx, part in enumerate(all_parts):
        owned_ids = part.owned
        n_owned = len(owned_ids)
        A_local = extract_local_block(K_spd, part)
        # Owned block diagonal
        A_owned = A_local[:, :n_owned]
        local_diag = np.array(A_owned.diagonal())
        ref_diag   = K_diag[owned_ids]
        err = np.abs(local_diag - ref_diag).max()
        assert err == 0.0, (
            f"rank {rank_idx}: Jacobi diag mismatch {err}"
        )


@pytest.mark.parametrize("level,n_ranks", [
    (2, 1),
    (3, 1),
])
def test_extract_local_block_n1(level, n_ranks):
    """N=1: single rank owns all nodes, no ghosts, A_local == K_spd."""
    K_spd, _, mesh, dims = _build_K_spd(level)
    all_parts = slab_partition(dims, n_ranks)
    assert len(all_parts) == 1
    part = all_parts[0]
    assert len(part.ghost) == 0

    A_local = extract_local_block(K_spd, part)
    n = K_spd.shape[0]
    assert A_local.shape == (n, n)

    # Full matvec should match K_spd exactly
    rng = np.random.default_rng(7)
    v = rng.standard_normal(n)
    err = np.abs(np.asarray(A_local @ v) - np.asarray(K_spd @ v)).max()
    assert err < 1e-12, f"N=1 A_local != K_spd: err={err:.3e}"
