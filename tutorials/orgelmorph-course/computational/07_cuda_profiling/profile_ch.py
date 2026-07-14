"""OrgElMorph course - Computational C7: CUDA profiling and memory.

Profiling answers ONE question honestly: where does a time step's time and
memory actually go?  This module profiles the production Cahn-Hilliard step
(``src/diffsim/physics/cahn_hilliard.py``) with the tools that WORK on any box
and teaches the Nsight commands for the device-level detail:

  1. STEP timing -- one complete, CUDA-synchronised warm step (the JIT-compile
     first step excluded), with the Newton-iteration count.
  2. ASYNC pitfall + kernel-launch overhead -- CUDA kernels launch
     asynchronously, so an unsynchronised timer measures the LAUNCH, not the
     WORK.  We show the per-launch host overhead and why the profiling timers
     synchronise.
  3. BANDWIDTH vs arithmetic intensity -- a streaming kernel saturates device
     memory bandwidth (near the card's peak); host<->device copies saturate the
     much slower PCIe link.  The gap is why data movement, not flops, usually
     rules.
  4. PLAN / FACTOR reuse -- a sparse LU FACTORISATION costs far more than a
     triangular SOLVE with that factor; a step that refactors every Newton
     iteration pays the factor cost repeatedly (the case for factor reuse /
     a GPU solver, Chapter C4).
  5. DEVICE-MEMORY high-water -- the driver-level footprint of the whole CH
     problem (Warp arrays + host-mirrored buffers), with an analytical
     breakdown of the dominant device buffer.

On THIS box Nsight Systems runs and captures the NVTX timeline of one step; the
kernel/memcpy-level rows need a CUPTI matching the CUDA driver (see run.py's
nsys section).  Everything numeric here is measured with CUDA-synchronised
timers and driver memory queries, so it is reproducible with or without nsys.
"""
from __future__ import annotations

import time

import numpy as np
import warp as wp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper

M, KAP = 1.0, 5e-4


# ---------------------------------------------------------------------------
# micro-benchmark kernels (must live in a real module file for Warp codegen)
# ---------------------------------------------------------------------------
@wp.kernel
def _touch(a: wp.array(dtype=wp.float64)):
    """Trivial one-flop kernel -- isolates kernel-LAUNCH overhead."""
    i = wp.tid()
    a[i] = a[i] + wp.float64(1.0)


@wp.kernel
def _stream(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
            c: wp.array(dtype=wp.float64)):
    """Bandwidth-bound triad c = a + 2b: 3 arrays touched, ~2 flops -- device
    memory bandwidth, not compute, sets its time."""
    i = wp.tid()
    c[i] = a[i] + wp.float64(2.0) * b[i]


def _sync():
    wp.synchronize()


def _driver_used_bytes():
    """Driver-level used device memory (bytes): total - free.  Captures ALL
    allocations (Warp mempool + torch), unlike torch.cuda.max_memory_allocated
    which only sees torch tensors."""
    try:
        import torch
        free, total = torch.cuda.mem_get_info()
        return int(total - free)
    except Exception:
        return None


# ---------------------------------------------------------------------------
def build_stepper(level, device="cuda:0", dt=5e-3, linsolver="splu"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    st = CahnHilliardStepper(dm, M, KAP, dt, order=1, newton_tol=1e-10,
                             linsolver=linsolver, capture_system=True)
    rng = np.random.default_rng(0)
    st.set_initial(lambda x: 0.1 * np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
                   + 0.02 * rng.standard_normal(len(x)), mu_init="consistent")
    return st, dm


# 1. one complete, synchronised warm step ----------------------------------
def step_timing(level, device="cuda:0", warm=1):
    """Time ONE complete CH step with CUDA sync, excluding the first
    JIT-compile step.  Returns the step wall time (ms), Newton iterations, and
    the (dofs, nnz) of the assembled saddle system."""
    st, dm = build_stepper(level, device)
    for _ in range(warm):
        st.step()                                    # JIT-warm the kernels
    _sync()
    t0 = time.perf_counter()
    st.step()
    _sync()
    step_ms = (time.perf_counter() - t0) * 1e3
    A, r = st.last_system
    return dict(level=level, step_ms=step_ms,
                newton_iters=int(st.last_newton["iters"]),
                dofs=int(A.shape[0]), nnz=int(A.tocsr().nnz))


# 2. async pitfall + launch overhead ---------------------------------------
def launch_overhead(device="cuda:0", n_elems=1 << 20, reps=2000):
    """Per-launch host overhead of a trivial kernel, timed WITHOUT and WITH a
    final sync.  The two are close only because many queued launches hide the
    device work -- the point is that the per-launch cost is host-side dispatch,
    and that a timer around ONE launch WITHOUT sync would measure almost
    nothing (the async pitfall)."""
    a = wp.zeros(n_elems, dtype=wp.float64, device=device)
    wp.launch(_touch, dim=1, inputs=[a], device=device)
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        wp.launch(_touch, dim=1, inputs=[a], device=device)
    t_nosync = time.perf_counter() - t0
    _sync()
    t_sync = time.perf_counter() - t0
    # the async pitfall on a SINGLE real-sized launch: unsynced vs synced
    big = wp.zeros(n_elems, dtype=wp.float64, device=device)
    _sync(); t0 = time.perf_counter()
    wp.launch(_touch, dim=n_elems, inputs=[big], device=device)
    unsynced_ms = (time.perf_counter() - t0) * 1e3
    _sync(); synced_ms = (time.perf_counter() - t0) * 1e3
    return dict(us_per_launch_nosync=t_nosync / reps * 1e6,
                us_per_launch_sync=t_sync / reps * 1e6,
                async_unsynced_ms=unsynced_ms, async_synced_ms=synced_ms,
                async_ratio=synced_ms / max(unsynced_ms, 1e-9))


# 3. bandwidth vs arithmetic intensity -------------------------------------
def bandwidth(device="cuda:0", n_elems=1 << 20, reps=200):
    """Achieved DEVICE memory bandwidth (streaming triad) vs host<->device PCIe
    copy bandwidth.  The device triad moves 3 arrays and does ~2 flops, so its
    arithmetic intensity is ~1/12 flop/byte -- deep in the memory-bound regime;
    it saturates near the card's peak.  The host copies saturate the far slower
    PCIe link."""
    a = wp.zeros(n_elems, dtype=wp.float64, device=device)
    b = wp.zeros(n_elems, dtype=wp.float64, device=device)
    c = wp.zeros(n_elems, dtype=wp.float64, device=device)
    wp.launch(_stream, dim=n_elems, inputs=[a, b, c], device=device)
    _sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        wp.launch(_stream, dim=n_elems, inputs=[a, b, c], device=device)
    _sync()
    dt = (time.perf_counter() - t0) / reps
    bytes_moved = 3 * n_elems * 8               # 2 read + 1 write, fp64
    flops = 2 * n_elems                         # one add + one multiply
    dev_gbs = bytes_moved / dt / 1e9
    ai = flops / bytes_moved                    # arithmetic intensity

    copies = []
    try:
        import torch
        for sz in (1 << 18, 1 << 21, 1 << 24):
            h = torch.empty(sz, dtype=torch.float64, pin_memory=True)
            d = torch.empty(sz, dtype=torch.float64, device="cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(50):
                d.copy_(h, non_blocking=True)
            torch.cuda.synchronize(); h2d = (time.perf_counter() - t0) / 50
            t0 = time.perf_counter()
            for _ in range(50):
                h.copy_(d, non_blocking=True)
            torch.cuda.synchronize(); d2h = (time.perf_counter() - t0) / 50
            nb = sz * 8
            copies.append(dict(mb=nb / 1e6, h2d_gbs=nb / h2d / 1e9,
                               d2h_gbs=nb / d2h / 1e9))
    except Exception:
        pass
    return dict(device_gbs=dev_gbs, device_ai=ai, copies=copies,
                pcie_gbs=copies[-1]["d2h_gbs"] if copies else None)


# 4. plan / factor reuse ---------------------------------------------------
def factor_reuse(level, device="cuda:0", solves=20):
    """Cost of a sparse-LU FACTORISATION vs a triangular SOLVE with that factor,
    on the exact assembled CH saddle system.  Also the per-step device->host
    copy VOLUME (the dense element-Jacobian buffer, copied every Newton
    iteration).  A step that refactors each iteration pays ``factor`` K times;
    reusing a factor (or a GPU solver, Chapter C4) is the fix."""
    import scipy.sparse.linalg as spla
    st, dm = build_stepper(level, device)
    st.step()
    A, r = st.last_system
    Ac = A.tocsc()
    t0 = time.perf_counter(); lu = spla.splu(Ac); factor_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    for _ in range(solves):
        lu.solve(r)
    solve_ms = (time.perf_counter() - t0) / solves * 1e3
    # per-step D2H volume: Ae [ne, 2nbf, 2nbf] + be [ne, 2nbf], fp64, per iter
    iters = int(st.last_newton["iters"])
    ne = st.mesh.conn_of[1].shape[0]
    nbf = st.mesh.conn_of[1].shape[1]
    ae_bytes = ne * (2 * nbf) ** 2 * 8
    be_bytes = ne * (2 * nbf) * 8
    d2h_per_step_mb = iters * (ae_bytes + be_bytes) / 1e6
    return dict(level=level, factor_ms=factor_ms, solve_ms=solve_ms,
                factor_over_solve=factor_ms / solve_ms,
                newton_iters=iters, d2h_per_step_mb=d2h_per_step_mb,
                dofs=int(Ac.shape[0]), nnz=int(Ac.nnz))


# 5. device-memory high-water ----------------------------------------------
def _analytical_footprint(ne, nbf, nqp, dim, nnodes):
    """Bytes of the dominant device-side arrays of one CH step (fp64/int64).

    Warp's caching mempool hides the pool's internal high-water from a
    driver-level ``mem_get_info`` query (which is also quantised to the pool's
    reservation granularity), so the honest, reproducible device-memory number
    is an ANALYTICAL accounting of the arrays the code actually allocates
    (spec C5's memory-accounting discipline, here on the device side)."""
    b = 2 * nbf
    parts = {
        "Ae dense elem Jacobian": ne * b * b * 8,      # [ne, 2nbf, 2nbf]
        "be elem residual": ne * b * 8,                # [ne, 2nbf]
        "connectivity": ne * nbf * 8,                  # int64
        "basis N/dN/w": (nqp * nbf + nqp * nbf * dim + nqp) * 8,
        "node coords": nnodes * dim * 8,
    }
    parts["total"] = sum(parts.values())
    return parts


def device_memory(level, device="cuda:0", subprocess_probe=True):
    """Device-memory footprint of one CH step: an analytical accounting of the
    on-device arrays (deterministic, scales with the mesh) plus, optionally, an
    ISOLATED driver-level measurement from a fresh process (the in-process delta
    reads ~0 because the mempool caches)."""
    st, dm = build_stepper(level, device)
    st.step()
    ne, nbf = st.mesh.conn_of[1].shape
    nqp = st.dm.tables_by_p[1].nqp
    parts = _analytical_footprint(ne, nbf, nqp, st.dm.dim, st.dm.n_nodes)
    ae_mb = parts["Ae dense elem Jacobian"] / 1e6
    analytical_mb = parts["total"] / 1e6

    driver_mb = None
    if subprocess_probe:
        driver_mb = _isolated_driver_footprint(level, device)
    footprint_mb = driver_mb if driver_mb else analytical_mb
    return dict(level=level, footprint_mb=footprint_mb,
                driver_mb=driver_mb, analytical_mb=analytical_mb,
                ae_buffer_mb=ae_mb, parts_mb={k: v / 1e6 for k, v in parts.items()},
                n_elements=int(ne), n_nodes=int(st.dm.n_nodes))


def _isolated_driver_footprint(level, device):
    """Run ``mem_probe.py`` in a fresh process and return its measured
    driver-level footprint (MB), or ``None`` on any failure."""
    import json
    import os
    import subprocess
    import sys
    probe = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "mem_probe.py")
    try:
        out = subprocess.run(
            [sys.executable, probe, "--device", device, "--level", str(level)],
            capture_output=True, text=True, timeout=300, env=dict(os.environ))
        for line in out.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                return float(json.loads(line)["footprint_mb"])
    except Exception:
        pass
    return None
