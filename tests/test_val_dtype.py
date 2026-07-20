"""Task #36 mixed-precision gates: fp32-storage + FP64 iterative refinement.

The value-dtype knob (`val_dtype="fp64"|"fp32"`) joins the DeviceNSAssembler
kernel-factory cache keys the same way #33 added index width.  The invariant
under test everywhere: the CSR VALUES buffer `vals_d` stays fp64 (the scatter
atomic-adds accumulate across elements sharing a slot — accumulation must be
fp64; "never accumulate fp32").  fp32 storage is a SEPARATE round-on-store
snapshot (`_vals_fp32`) fed ONLY to the cuDSS factorization; the FP64
iterative refinement computes its residual against the fp64 `vals_d` via the
fp64 device SpMV (device_operator), NOT the fp32 snapshot.

CPU-runnable here (Mac dev loop): the resolve unit, the round-store snapshot,
and the fp64-default-unchanged guard.  The cuDSS factorization + IR gates
(G1-G5) are CUDA-only and live in the GPU lanes.
"""
import numpy as np
import pytest
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.assembly.device_assembly import (
    DeviceNSAssembler, _resolve_val_dtype)
from diffsim.physics.poisson import gauss_points
from diffsim.errors import ConfigError

pytestmark = pytest.mark.tier2


def _setup(dim, level, device):
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), device)
    xq = gauss_points(mesh, dm.tables_by_p)
    rng = np.random.default_rng(3)
    pv = list(dm.bins)[0]
    ngp = len(xq[pv])
    aq = {pv: rng.standard_normal((ngp, dim)) * 0.5}
    dq = {pv: rng.standard_normal(ngp) * 0.1}
    fq = {pv: rng.standard_normal((ngp, dim))}
    return dm, aq, dq, fq


# ── Step 1: the knob resolves + defaults to fp64 ───────────────────────────
def test_resolve_val_dtype():
    """The knob maps to a warp float dtype; bogus input FAILs loudly.
    Default (fp64) is bit-for-bit the pre-#36 storage path."""
    assert _resolve_val_dtype("fp64") is wp.float64
    assert _resolve_val_dtype("fp32") is wp.float32
    with pytest.raises(ConfigError):
        _resolve_val_dtype("bogus")
    with pytest.raises(ConfigError):
        _resolve_val_dtype("float32")


def test_val_dtype_default_unchanged(device):
    """fp64 default: vals_d is fp64 and NO fp32 snapshot is allocated —
    the assembler is bit-for-bit the pre-#36 path (no memory tax)."""
    dm, aq, dq, fq = _setup(2, 4, device)
    asm = DeviceNSAssembler(dm)                       # default val_dtype
    assert asm._val_dtype is wp.float64
    assert asm.vals_d.dtype is wp.float64
    assert asm._vals_fp32 is None
    # a fp64 assembler's snapshot refresh is a no-op (returns cleanly)
    asm.assemble(aq, dq, fq, 0.05, 20.0)
    assert asm._vals_fp32 is None


# ── Step 2: fp32 snapshot is round-on-store; vals_d stays fp64 ─────────────
@pytest.mark.parametrize("dim,level", [(2, 4), (3, 2)])
def test_fp32_snapshot_roundstore(dim, level, device):
    """val_dtype='fp32': the CSR values accumulate in fp64 (vals_d stays
    fp64 — the accumulation contract), and the fp32 snapshot equals the
    fp64 values rounded to fp32 (round-on-store, never accumulate fp32)."""
    dm, aq, dq, fq = _setup(dim, level, device)
    asm = DeviceNSAssembler(dm, val_dtype="fp32")
    assert asm._val_dtype is wp.float32
    # THE accumulation guard: the values buffer is fp64, always.
    assert asm.vals_d.dtype is wp.float64
    assert asm._vals_fp32 is not None
    assert asm._vals_fp32.dtype is wp.float32

    asm.assemble(aq, dq, fq, 0.05, 20.0)
    asm.refresh_fp32_snapshot()
    vals64 = asm.vals_d.numpy()
    snap32 = asm._vals_fp32.numpy()
    # round-on-store: snap == fp32(vals64), exactly (a single cast).
    assert snap32.dtype == np.float32
    assert np.array_equal(snap32, vals64.astype(np.float32))


def test_fp32_snapshot_node_pattern(device):
    """The fp32 snapshot round-store also works in node-graph pattern
    mode (the film / XDD path): vals_d fp64, snapshot == fp32(vals_d)."""
    dm, aq, dq, fq = _setup(3, 2, device)
    asm = DeviceNSAssembler(dm, ndof=4, node_pattern=True, val_dtype="fp32")
    assert asm.vals_d.dtype is wp.float64
    pv, b, ne, nbf, _ = asm._bins[0]
    nl = 4 * nbf
    rng = np.random.default_rng(7)
    Ae = wp.array(rng.standard_normal((ne, nl, nl)), dtype=wp.float64,
                  device=device)
    be = wp.array(rng.standard_normal((ne, nl)), dtype=wp.float64,
                  device=device)
    asm.zero_fill()
    asm.scatter_bin(0, Ae, be)
    asm.refresh_fp32_snapshot()
    vals64 = asm.vals_d.numpy()
    snap32 = asm._vals_fp32.numpy()
    assert np.array_equal(snap32, vals64.astype(np.float32))
