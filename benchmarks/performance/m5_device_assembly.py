"""M5 D3 — multiphase device-assembly step-time table (host vs device).

One (case, assembly) combo PER PROCESS (clean GPU/host memory
accounting; warp mempools and cuDSS workspaces never shrink):

    python m5_device_assembly.py --case 2d_l6 --assembly device

Physics = the S3b-class film production config (M = 2, K = 1, r14 +
fastmode_n + Vignes + ls_drop, film k_e = 0.1, b_reg, line_search,
noise_psi 5e-3) — the S3-3D hero candidate family.  Solver cudss in
BOTH modes (host: scipy CSR upload per iterate; device: zero-copy
plan-once + refactorize), so the table isolates assembly + handoff.
3-D slabs carve the vertical axis of the periodic-lateral cube (the
wodo strip pattern): full lateral extent (periodicity preserved),
nz elements tall — "modest z-resolution first" (S3-3D hero shape).

Timings: per-phase wall clock with wp.synchronize() fences around the
assembly and solve calls (instance-attribute wrappers — no production
code touched); "other" = step total - assembly - solve (Newton
bookkeeping, ctx build incl. host noise draws + history einsum, x
updates).  First step reported separately as setup (pattern build +
kernel compile + cuDSS plan).  Memory: nvidia-smi at end of run (warp
mempool high-water class) + host ru_maxrss.
"""
import argparse
import os
import resource
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import _bench_bootstrap  # noqa

import numpy as np
import warp as wp

from diffsim import default_device
from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper

CASES = {
    "2d_l6": dict(dim=2, level=6),
    "2d_l7": dict(dim=2, level=7),
    "2d_l8": dict(dim=2, level=8),
    # 3-D: the S3b energetics at the noisy dilute IC are too brittle
    # for FIXED-dt marching (measured: dt = 1e-4 hits newton_max at
    # the first step; dt = 1e-5 converges the first step then fails a
    # later one at L5 — the production march ladder handles this, a
    # timing bench should not).  3-D rungs run dt = 1e-5 with the FDT
    # noise OFF (noise=0): per-call assembly/solve costs are
    # state-independent (same kernels, same fixed pattern), and the
    # noise-draw host cost is measured by the 2-D rungs (noise on).
    "3d_l4": dict(dim=3, level=4, dt=1e-5, noise=0.0),
    "3d_l5": dict(dim=3, level=5, dt=1e-5, noise=0.0),
    "3d_slab64": dict(dim=3, level=6, nz=16, dt=1e-5, noise=0.0),
    "3d_slab64z32": dict(dim=3, level=6, nz=32, dt=1e-5, noise=0.0),
    "3d_slab64z48": dict(dim=3, level=6, nz=48, dt=1e-5, noise=0.0),
    "3d_slab128": dict(dim=3, level=7, nz=32, dt=1e-5, noise=0.0),
    # B4: Baskar's 3-D film target — 128 x 128 x 64 elements
    # (~1.06M nodes, 6.39M dofs at ndof = 6)
    "3d_film128": dict(dim=3, level=7, nz=64, dt=1e-5, noise=0.0),
    # B5 capacity rungs ((M, K) = (3, 2), ndof = 10; --mk 32)
    "3d_slab64z32_mk32": dict(dim=3, level=6, nz=32, dt=1e-5,
                              noise=0.0),
    "3d_slab128z48_mk32": dict(dim=3, level=7, nz=48, dt=1e-5,
                               noise=0.0),
    "3d_slab128z64_mk32": dict(dim=3, level=7, nz=64, dt=1e-5,
                               noise=0.0),
    "3d_slab128z88_mk32": dict(dim=3, level=7, nz=88, dt=1e-5,
                               noise=0.0),
}


def build_case(c, device):
    dim, level = c["dim"], c["level"]
    per = (True,) * (dim - 1) + (False,)
    tree = build_uniform(level, dim=dim, periodic=per)
    if "nz" in c:                      # vertical slab carve (wodo strip)
        hc = 2.0 ** (-level)
        cen = tree.centers()
        keep = cen[:, dim - 1] < c["nz"] * hc
        tree = Octree(tree.keys[keep], tree.levels[keep], dim=dim,
                      periodic=tree.periodic)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    return DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim),
                                device)


def make_stepper(dm, assembly, dt=1e-4, noise=5e-3, linsolver="cudss",
                 mk=(2, 1), block_sparse=False):
    """S3b-class film production physics.  mk=(2, 1): the hero config
    (M = 2 active + eliminated solvent, K = 1, ndof = 6).  mk=(3, 2):
    the B5 capacity-study family (3 retained species, 2 crystallizable,
    ndof = 10) — the (2, 1) energetics extended by a third polymer
    (chi/N/Vignes rows in the same measured ranges)."""
    M, K = mk
    assert mk in ((2, 1), (3, 2)), mk
    if mk == (2, 1):
        chi_aa = np.zeros((3, 3))
        chi_aa[0, 1] = chi_aa[1, 0] = 1.0
        chi_aa[0, 2] = chi_aa[2, 0] = 0.7248
        chi_aa[1, 2] = chi_aa[2, 1] = 0.3
        chi_ca = np.zeros((3, 3))
        chi_ca[0, 1] = chi_ca[0, 2] = 1.6
        Dslf = np.array([[1e-2, 1e-3, 0.5],
                         [1e-4, 1e-4, 1e-2],
                         [1e-2, 1e-3, 1.0]])
        kw = dict(N=[65.4, 87.0, 1.0], kappa=[2e-4] * 2,
                  dsig=[8.0], dh=[-40.0], Tm=[402.0], eps2=[4e-3],
                  L_psi=[65.4])
    else:
        chi_aa = np.zeros((4, 4))
        for (i, j), v in {(0, 1): 1.0, (0, 2): 0.5, (1, 2): 0.8,
                          (0, 3): 0.7248, (1, 3): 0.3,
                          (2, 3): 0.4}.items():
            chi_aa[i, j] = chi_aa[j, i] = v
        chi_ca = np.zeros((4, 4))
        chi_ca[0, 1] = chi_ca[0, 2] = chi_ca[0, 3] = 1.6
        chi_ca[1, 0] = chi_ca[1, 2] = chi_ca[1, 3] = 1.2
        Dslf = np.array([[1e-2, 1e-3, 5e-3, 0.5],
                         [1e-4, 1e-4, 1e-3, 1e-2],
                         [1e-3, 1e-3, 1e-3, 5e-2],
                         [1e-2, 1e-3, 5e-3, 1.0]])
        kw = dict(N=[65.4, 87.0, 40.0, 1.0], kappa=[2e-4] * 3,
                  dsig=[8.0, 6.0], dh=[-40.0, -30.0],
                  Tm=[402.0, 390.0], eps2=[4e-3, 4e-3],
                  L_psi=[65.4, 40.0])
    st = MultiPhaseStepper(
        dm, M=M, K=K, chi_aa=chi_aa, chi_ac=chi_ca.T.copy(),
        chi_ca=chi_ca, mob="fastmode_n",
        D_self=Dslf, ls_drop=(1e-6, 0.97, 35.0),
        T=333.0, dt=dt, bulk="r14", b_reg=1e-3,
        noise_psi=noise,
        noise_damp=(1e-2, 0.85, 15.0), clip_psi=False,
        newton_tol=1e-8, newton_max=50, linsolver=linsolver,
        line_search=True, noise_seed=11,
        film=dict(k_e=0.1), assembly=assembly,
        block_sparse=block_sparse, **kw)
    rng = np.random.default_rng(1011)
    nf = st.nfree
    ics = [0.10 + 0.01 * rng.standard_normal(nf),
           0.05 + 0.01 * rng.standard_normal(nf)]
    if M == 3:
        ics.append(0.05 + 0.005 * rng.standard_normal(nf))
    z = lambda x: np.zeros(len(x))
    st.set_initial([(lambda v: (lambda x: v))(v) for v in ics],
                   [z] * K, [z] * K)
    return st


def _wrap(fn, acc):
    def g(*a, **k):
        wp.synchronize()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        wp.synchronize()
        acc[0] += time.perf_counter() - t0
        acc[1] += 1
        return out
    return g


def gpu_mem_mib():
    """Per-process query is unavailable under WSL2 — fall back to the
    global counter (valid: one bench process per GPU at a time)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], text=True)
        pid = os.getpid()
        for line in out.strip().splitlines():
            p, m = line.split(",")
            if int(p) == pid:
                return int(m)
    except Exception:
        pass
    try:
        dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        out = subprocess.check_output(
            ["nvidia-smi", "-i", dev, "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True)
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--assembly", required=True,
                    choices=["host", "device"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--device", default=default_device())
    ap.add_argument("--solver", default="cudss",
                    choices=["cudss", "splu", "blockch", "blockch_dev"])
    ap.add_argument("--block-sparse", action="store_true",
                    help="kron(G, blockmask) device pattern (B5)")
    args = ap.parse_args()

    case = CASES[args.case]
    dm = build_case(case, args.device)
    st = make_stepper(dm, args.assembly, dt=case.get("dt", 1e-4),
                      noise=case.get("noise", 5e-3),
                      linsolver=args.solver,
                      mk=(3, 2) if args.case.endswith("_mk32")
                      else (2, 1), block_sparse=args.block_sparse)
    ndofs = st.nfree * st.ndof
    print(f"[{args.case}/{args.assembly}/{args.solver}] elements "
          f"{len(dm.mesh.tree)} nodes {dm.n_nodes} dofs {ndofs}",
          flush=True)

    asm_acc = [0.0, 0]
    slv_acc = [0.0, 0]
    if args.assembly == "device":
        st._assemble_device = _wrap(st._assemble_device, asm_acc)
        st._solve_dev = _wrap(st._solve_dev, slv_acc)
    else:
        st._assemble_host = _wrap(st._assemble_host, asm_acc)
        st._solve = _wrap(st._solve, slv_acc)

    # setup + warmup step (pattern build, kernel compile, solver plan)
    t0 = time.perf_counter()
    try:
        st.step()
    except Exception as e:
        print(f"RESULT {args.case} {args.assembly} {args.solver} DNF-setup: "
              f"{type(e).__name__}: {e}", flush=True)
        return
    t_setup = time.perf_counter() - t0
    print(f"  setup+first step {t_setup:.2f} s "
          f"(asm {asm_acc[0]:.2f} s / {asm_acc[1]} calls, "
          f"solve {slv_acc[0]:.2f} s / {slv_acc[1]} calls)", flush=True)

    asm_acc[:] = [0.0, 0]
    slv_acc[:] = [0.0, 0]
    t0 = time.perf_counter()
    ok = True
    for _ in range(args.steps):
        try:
            st.step()
        except Exception as e:
            print(f"RESULT {args.case} {args.assembly} {args.solver} DNF-march: "
                  f"{type(e).__name__}: {e}", flush=True)
            ok = False
            break
    t_tot = time.perf_counter() - t0
    if not ok:
        return
    n = args.steps
    other = t_tot - asm_acc[0] - slv_acc[0]
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    nnz = st._asm.nnz if args.assembly == "device" else -1
    print(f"RESULT {args.case} {args.assembly} {args.solver} dofs {ndofs} "
          f"nnz {nnz} steps {n} "
          f"step {t_tot / n:.3f} s  asm {asm_acc[0] / n:.3f} s "
          f"({asm_acc[1]} calls, {asm_acc[0] / max(asm_acc[1], 1):.3f}"
          f" s/call)  solve {slv_acc[0] / n:.3f} s "
          f"({slv_acc[1]} calls)  other {other / n:.3f} s  "
          f"gpu {gpu_mem_mib()} MiB  rss {rss:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
