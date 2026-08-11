"""tests/test_chns_device.py — SP-0 Task 12: CHNS monolithic GPU device parity.

The monolithic CHNS stepper (steppers/chns.py, CHNSMonolithicStepper) launches
its coupled (u, p, phi, mu) residual + Jacobian Warp kernel on ``self.device``.
This gate proves the CUDA path is bit-for-bit faithful to the CPU path: same
mesh, same case, same initial condition, 5 BDF1 steps of the level-5 2-D bubble
rise, comparing phi/u/p on CPU vs cuda:0 with relative error <= 1e-8.

The COO triplets still return to the host (Ae.numpy()) and the linear solve
stays scipy splu on both paths — this is a device-KERNEL parity test (the
assembly runs on the GPU), not a device-solve test.

Skips cleanly when no CUDA device is visible: it is a real gate only on a GPU
box (gpubox: LD_LIBRARY_PATH=/usr/lib/wsl/lib recipe).  On the Mac it SKIPS.
"""
import os as _os
import sys as _sys

import numpy as np
import pytest

_BENCH_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "benchmarks")
)
if _BENCH_DIR not in _sys.path:
    _sys.path.insert(0, _BENCH_DIR)
_REPO_DIR = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..")
)
if _REPO_DIR not in _sys.path:
    _sys.path.insert(0, _REPO_DIR)

from chns.cases import BUBBLE_RISE_RE35_WE10  # noqa: E402


def _cuda_available():
    try:
        import warp as wp
        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _make_cpu_dm(level, dim=2):
    """A CPU DeviceMesh (device-agnostic scaffolding).  The stepper's device=
    knob re-materialises the per-bin basis tables on the target device, so one
    CPU dm drives both the CPU and the CUDA stepper."""
    import warp as wp
    wp.init()
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh

    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    return dm, mesh, cons


def _bubble_phi0(coords, Cn, xc=0.5, yc=0.35, radius=0.2):
    r = np.sqrt((coords[:, 0] - xc) ** 2 + (coords[:, 1] - yc) ** 2)
    return -np.tanh((r - radius) / (Cn * np.sqrt(2.0)))


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(),
                    reason="no CUDA device visible (runs on gpubox)")
def test_chns_monolithic_cpu_vs_cuda_parity():
    """5 steps, level-5 2-D bubble rise, CUDA kernel vs CPU kernel, rel <= 1e-8."""
    from diffsim.steppers.chns import CHNSMonolithicStepper

    level, dt, nsteps = 5, 2.5e-3, 5
    dm, mesh, cons = _make_cpu_dm(level, dim=2)
    coords = mesh.node_coords

    def _run(device):
        st = CHNSMonolithicStepper(
            dm, BUBBLE_RISE_RE35_WE10, dt=dt, Cn_override="2h",
            gravity=True, device=device)
        st.set_initial(_bubble_phi0(coords, st.Cn))
        for _ in range(nsteps):
            st.step()
        return st.phi.copy(), st.u.copy(), st.p.copy()

    phi_c, u_c, p_c = _run("cpu")
    phi_g, u_g, p_g = _run("cuda:0")

    def _rel(a, b):
        num = float(np.linalg.norm(a - b))
        den = float(np.linalg.norm(b)) or 1.0
        return num / den

    rel_phi = _rel(phi_g, phi_c)
    rel_u = _rel(u_g, u_c)
    rel_p = _rel(p_g, p_c)
    print(f"\n[device-parity] level={level} steps={nsteps}  "
          f"rel_phi={rel_phi:.3e} rel_u={rel_u:.3e} rel_p={rel_p:.3e}")
    assert rel_phi <= 1e-8, f"phi CPU-vs-CUDA rel {rel_phi:.3e} > 1e-8"
    assert rel_u <= 1e-8, f"u CPU-vs-CUDA rel {rel_u:.3e} > 1e-8"
    assert rel_p <= 1e-8, f"p CPU-vs-CUDA rel {rel_p:.3e} > 1e-8"
