"""Unit tests: torch.sparse_csr_tensor SpMV equals scipy host matvec.

Validates Task 1 of the GPU-resident NCCL-CG driver:
  make_partitioned_spmv(..., host_spmv=False)  — torch.sparse_csr_tensor path
  make_partitioned_spmv(..., host_spmv=True)   — legacy scipy CPU path

The two paths must agree to ~1e-12 in L∞ norm on a real K_p local block
(level 3, 1 rank = identity partition so A_local_csr is the full K_spd).

Also validates the halo-exchange contract at world_size=1: with no ghosts the
padded buffer is exactly n_owned long, and exchange_halo is a no-op, so the
device and host paths produce identical results.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Module-level guard: torch is required
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch", reason="torch not installed")


# ---------------------------------------------------------------------------
# Helper: build a small real K_spd and extract one rank's local block
# ---------------------------------------------------------------------------

def _build_local_block(level: int = 3):
    """Return (A_local_csr, n_owned, n_ghost) for a 1-rank identity partition."""
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh, assemble_csr
    from diffsim.mesh.partition import slab_partition

    n_per_axis = 2**level + 1
    dims = (n_per_axis, n_per_axis, n_per_axis)

    tree = build_uniform(level, dim=3)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm   = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), "cpu")

    import numpy as np_
    import scipy.sparse as sp_
    K = assemble_csr(dm).tocsr().astype(np.float64)

    n = K.shape[0]
    bn = mesh.boundary_nodes
    bn_idx = np_.where(bn)[0]
    in_idx = np_.where(~bn)[0]

    # SPD-ify (same as build_real_K_spd)
    K = K.tolil()
    for k in bn_idx:
        K[k, :] = 0.0
        K[:, k] = 0.0
        K[k, k] = 1.0
    K = K.tocsr()
    K.eliminate_zeros()

    # 1-rank partition: every node is owned, no ghosts
    all_parts = slab_partition(dims, 1)
    part = all_parts[0]
    from diffsim.mesh.partition import extract_local_block
    A_local_csr = extract_local_block(K, part)
    n_owned = len(part.owned)
    n_ghost = len(part.ghost)
    return A_local_csr, n_owned, n_ghost


# ---------------------------------------------------------------------------
# Minimal no-op comm for world_size=1
# ---------------------------------------------------------------------------

class _SerialTorchComm:
    """Minimal serial comm: allreduce = identity, exchange_halo = no-op."""

    def allreduce_sum(self, x: float) -> float:
        return float(x)

    def exchange_halo(self, local_vec: torch.Tensor) -> None:
        # No ghosts at world_size=1; this is genuinely a no-op.
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSpMVDevice:
    """torch.sparse_csr_tensor SpMV vs scipy host matvec equivalence."""

    def test_spmv_device_vs_host_l3(self):
        """Device torch.sparse SpMV agrees with scipy to 1e-12 at level 3.

        This exercises:
         - torch.sparse_csr_tensor construction from A_local_csr on CPU device
         - torch.mv(A_t, padded) returns the owned-length result
         - No halo exchange at world_size=1 (the ghost tail has 0 entries)
         - Absolute and relative L∞ agreement < 1e-12
        """
        # Import here so the test is skipped if diffsim is not installed
        pytest.importorskip("diffsim.assembly.operators",
                            reason="diffsim not installed")

        from scripts.nccl_cg_proof import make_partitioned_spmv

        A_local_csr, n_owned, n_ghost = _build_local_block(level=3)
        assert n_ghost == 0, f"Expected 0 ghosts at world_size=1, got {n_ghost}"

        device = torch.device("cpu")
        comm = _SerialTorchComm()

        # Build both variants
        spmv_dev  = make_partitioned_spmv(A_local_csr, n_owned, n_ghost,
                                          comm, device, host_spmv=False)
        spmv_host = make_partitioned_spmv(A_local_csr, n_owned, n_ghost,
                                          comm, device, host_spmv=True)

        # Fixed-seed random input vector
        rng = np.random.default_rng(42)
        p_np = rng.standard_normal(n_owned)
        p_t  = torch.tensor(p_np, dtype=torch.float64, device=device)

        # Device path (torch.sparse_csr_tensor)
        Ap_dev  = spmv_dev(p_t.clone()).cpu().numpy()
        # Host/scipy path (reference)
        Ap_host = spmv_host(p_t.clone()).cpu().numpy()

        # Also compute reference directly with scipy (belt-and-suspenders)
        import scipy.sparse as sp_
        p_padded_np = np.concatenate([p_np, np.zeros(n_ghost)])
        Ap_scipy = np.asarray(A_local_csr @ p_padded_np, dtype=np.float64)

        # Check lengths
        assert Ap_dev.shape  == (n_owned,), f"Wrong shape: {Ap_dev.shape}"
        assert Ap_host.shape == (n_owned,), f"Wrong shape: {Ap_host.shape}"

        # Agreement checks
        scale = max(np.abs(Ap_scipy).max(), 1e-300)
        err_dev_vs_scipy  = np.abs(Ap_dev  - Ap_scipy).max() / scale
        err_host_vs_scipy = np.abs(Ap_host - Ap_scipy).max() / scale
        err_dev_vs_host   = np.abs(Ap_dev  - Ap_host).max() / scale

        assert err_dev_vs_scipy  < 1e-12, (
            f"Device torch.sparse vs scipy: rel err = {err_dev_vs_scipy:.3e} >= 1e-12"
        )
        assert err_host_vs_scipy < 1e-12, (
            f"Host scipy wrapper vs scipy:  rel err = {err_host_vs_scipy:.3e} >= 1e-12"
        )
        assert err_dev_vs_host   < 1e-12, (
            f"Device torch.sparse vs host: rel err = {err_dev_vs_host:.3e} >= 1e-12"
        )

    def test_spmv_device_output_on_device(self):
        """The device SpMV returns a tensor on the correct device."""
        pytest.importorskip("diffsim.assembly.operators",
                            reason="diffsim not installed")
        from scripts.nccl_cg_proof import make_partitioned_spmv

        A_local_csr, n_owned, n_ghost = _build_local_block(level=3)
        device = torch.device("cpu")
        comm = _SerialTorchComm()

        spmv_dev = make_partitioned_spmv(A_local_csr, n_owned, n_ghost,
                                         comm, device, host_spmv=False)

        p_t = torch.ones(n_owned, dtype=torch.float64, device=device)
        Ap  = spmv_dev(p_t)

        assert isinstance(Ap, torch.Tensor), "Expected torch.Tensor output"
        assert Ap.device.type == device.type, (
            f"Output on {Ap.device}, expected {device}"
        )
        assert Ap.dtype == torch.float64, f"Wrong dtype: {Ap.dtype}"
        assert Ap.shape == (n_owned,), f"Wrong shape: {Ap.shape}"

    def test_spmv_device_reuse_buffer(self):
        """Multiple calls to the device SpMV give consistent results (buffer reuse)."""
        pytest.importorskip("diffsim.assembly.operators",
                            reason="diffsim not installed")
        from scripts.nccl_cg_proof import make_partitioned_spmv

        A_local_csr, n_owned, n_ghost = _build_local_block(level=3)
        device = torch.device("cpu")
        comm = _SerialTorchComm()

        spmv_dev = make_partitioned_spmv(A_local_csr, n_owned, n_ghost,
                                         comm, device, host_spmv=False)

        rng = np.random.default_rng(7)
        p_np = rng.standard_normal(n_owned)
        p_t  = torch.tensor(p_np, dtype=torch.float64, device=device)

        Ap1 = spmv_dev(p_t.clone()).cpu().numpy()
        Ap2 = spmv_dev(p_t.clone()).cpu().numpy()

        assert np.array_equal(Ap1, Ap2), (
            "Device SpMV returned different results on repeated calls with same input "
            "(buffer contamination?)"
        )
