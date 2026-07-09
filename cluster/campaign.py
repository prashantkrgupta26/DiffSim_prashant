"""Nova arrival campaign — staged, self-time-boxing, early-exit-safe.

Usage: python cluster/campaign.py --card {a100,h200,gh200} --hours 24
Stages run in value order; each writes into results.json as it lands, so
a killed job still leaves a complete partial report.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def log(run, msg):
    print(msg, flush=True)
    with open(os.path.join(run, "report.md"), "a") as fh:
        fh.write(msg + "\n")


def save(run, results):
    with open(os.path.join(run, "results.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=str)


def stage_sanity(run, results):
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-x",
         "tests/test_sbm_poisson.py", "tests/test_ns_adjoint.py"],
        capture_output=True, text=True, cwd=ROOT, timeout=3600)
    ok = r.returncode == 0
    tail = r.stdout.strip().splitlines()[-1] if r.stdout else "?"
    results["sanity"] = {"ok": ok, "summary": tail}
    log(run, f"## sanity: {'OK' if ok else 'FAIL'} — {tail}")
    return ok


def stage_fp64_micro(run, results):
    import numpy as np
    import warp as wp
    import torch
    wp.init()
    dev = wp.get_device()
    n = 2_000_000

    @wp.kernel
    def saxpy64(a: wp.array(dtype=wp.float64),
                b: wp.array(dtype=wp.float64), c: wp.float64):
        i = wp.tid()
        b[i] = b[i] + c * a[i] * a[i] + wp.sqrt(a[i] * a[i]
                                                + wp.float64(1.0))
    a = wp.array(np.random.rand(n), dtype=wp.float64)
    b = wp.zeros(n, dtype=wp.float64)
    wp.launch(saxpy64, dim=n, inputs=[a, b, wp.float64(0.5)])
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        wp.launch(saxpy64, dim=n, inputs=[a, b, wp.float64(0.5)])
    wp.synchronize()
    t_k = (time.perf_counter() - t0) / 200
    # torch FP64 GEMM (dense throughput)
    x = torch.rand(4096, 4096, dtype=torch.float64, device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        _ = x @ x
    torch.cuda.synchronize()
    t_g = (time.perf_counter() - t0) / 10
    tflops = 2 * 4096 ** 3 / t_g / 1e12
    results["fp64"] = {"warp_kernel_ms": t_k * 1e3,
                       "gemm4096_ms": t_g * 1e3,
                       "gemm_fp64_tflops": tflops,
                       "device": str(dev)}
    log(run, f"## fp64-micro: kernel {t_k*1e3:.2f} ms/launch, "
             f"FP64 GEMM {tflops:.1f} TFLOPS  [{dev}]")
    return True


def stage_solver_table(run, results):
    sys.path.insert(0, os.path.join(ROOT, "tests"))
    import numpy as np
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    from diffsim.steppers.linearized import LinearizedMonolithicStepper
    tab = {}
    for level in (6, 7, 8):
        for solver in ("cudss", "splu"):
            try:
                tree = build_uniform(level, dim=2)
                mesh = build_mesh(tree, p=1)
                cons = build_constraints(mesh)
                dm = DeviceMesh.from_mesh(mesh, cons,
                                          basis_tables(1, dim=2), "cuda:0")
                def lid(x, t):
                    g = np.zeros((len(x), 2))
                    g[np.abs(x[:, 1] - 1.0) < 1e-12, 0] = 1.0
                    return g
                st = LinearizedMonolithicStepper(
                    dm, 0.01, 0.05,
                    f_fn=lambda x, t: np.zeros((len(x), 2)),
                    g_fn=lid, order=2, solver=solver)
                st.set_initial(lambda x: np.zeros((len(x), 2)))
                st.step()                        # warm (factorize etc.)
                t0 = time.perf_counter()
                for _ in range(5):
                    st.step()
                dt_ms = (time.perf_counter() - t0) / 5 * 1e3
                tab[f"cavity_L{level}_{solver}_ms_per_step"] = round(
                    dt_ms, 1)
                log(run, f"  cavity L{level} {solver}: "
                         f"{dt_ms:.0f} ms/step")
            except Exception as e:
                tab[f"cavity_L{level}_{solver}"] = f"FAIL {str(e)[:60]}"
                log(run, f"  cavity L{level} {solver}: FAIL {str(e)[:60]}")
    results["solver_table"] = tab
    return True


def stage_capacity(run, results):
    """Empirical cuDSS ceiling: solve a Poisson stiffness at growing
    sizes until ALLOC failure."""
    import numpy as np
    import scipy.sparse as sp
    cap = {}
    try:
        from nvmath.sparse.advanced import direct_solver
    except ImportError:
        results["capacity"] = "nvmath missing"
        return True
    for dim, levels in ((2, (10, 11, 12)), (3, (6, 7, 8))):
        for lv in levels:
            n1 = 2 ** lv
            n = n1 ** dim
            key = f"poisson_{dim}d_L{lv}_n{n}"
            if n > 40_000_000:
                cap[key] = "skipped (assembly too large for the box)"
                continue
            try:
                # dim-D Laplacian via kron (fast to build, right structure)
                T1 = sp.diags([-1, 2, -1], [-1, 0, 1],
                              shape=(n1, n1), format="csr")
                I1 = sp.identity(n1, format="csr")
                if dim == 2:
                    A = sp.kron(T1, I1) + sp.kron(I1, T1)
                else:
                    A = (sp.kron(sp.kron(T1, I1), I1)
                         + sp.kron(sp.kron(I1, T1), I1)
                         + sp.kron(sp.kron(I1, I1), T1))
                A = (A + sp.identity(n) * 1e-8).tocsr()
                b = np.random.rand(n)
                t0 = time.perf_counter()
                _ = direct_solver(A, b)
                cap[key] = f"OK {time.perf_counter()-t0:.1f}s"
                log(run, f"  capacity {key}: OK "
                         f"{time.perf_counter()-t0:.1f}s")
            except Exception as e:
                cap[key] = f"FAIL {str(e)[:80]}"
                log(run, f"  capacity {key}: FAIL {str(e)[:60]}")
                break
    results["capacity"] = cap
    return True


def stage_band_study(run, results, card):
    """The science payload: P2-P1 Neumann sphere, 3-D L4..L7 (+L8 on
    H200). See cluster/samundra_band_study.md for interpretation."""
    levels = [4, 5, 6, 7] + ([8] if card == "h200" else [])
    out = {}
    for lv in levels:
        try:
            r = subprocess.run(
                [sys.executable, "benchmarks/poisson-sbm/band_study.py", "3", str(lv)],
                capture_output=True, text=True, cwd=ROOT,
                timeout=6 * 3600)
            lines = [ln for ln in r.stdout.splitlines() if "band(3)" in ln]
            out[f"L{lv}"] = lines[-1].strip() if lines else \
                f"no result (rc={r.returncode})"
            log(run, f"  band L{lv}: {out[f'L{lv}']}")
        except Exception as e:
            out[f"L{lv}"] = f"FAIL {str(e)[:80]}"
            log(run, f"  band L{lv}: FAIL {str(e)[:60]}")
        results["band_study"] = out
        save(run, results)
    return True


def stage_hero_timing(run, results):
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "benchmarks/inverse-heroes/hero_h1_sphere_steady.py", "4", "1"],
        capture_output=True, text=True, cwd=ROOT, timeout=2 * 3600)
    dt = time.perf_counter() - t0
    tail = [ln for ln in r.stdout.splitlines()
            if "gradient check" in ln or ln.strip().startswith("0 ")]
    results["hero_timing"] = {"one_epoch_wall_s": round(dt, 1),
                              "lines": tail}
    log(run, f"## hero-timing: setup+1 epoch = {dt:.0f}s; {tail}")
    return True


STAGES = [("sanity", stage_sanity), ("fp64-micro", stage_fp64_micro),
          ("solver-table", stage_solver_table),
          ("capacity", stage_capacity), ("band-study", stage_band_study),
          ("hero-timing", stage_hero_timing)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True,
                    choices=["a100", "h200", "gh200"])
    ap.add_argument("--hours", type=float, default=24)
    args = ap.parse_args()
    stamp = time.strftime("%Y%m%d-%H%M")
    run = os.path.join(HERE, "results", f"{args.card}-{stamp}")
    os.makedirs(run, exist_ok=True)
    deadline = time.time() + args.hours * 3600 - 1800   # 30 min reserve
    results = {"card": args.card, "start": stamp}
    log(run, f"# DiffSim Nova campaign — {args.card} — {stamp}\n")
    for name, fn in STAGES:
        if time.time() > deadline:
            log(run, f"## {name}: SKIPPED (deadline)")
            continue
        log(run, f"\n## stage: {name}")
        try:
            if name == "band-study":
                fn(run, results, args.card)
            else:
                fn(run, results)
        except Exception:
            log(run, f"stage {name} CRASHED:\n"
                     f"```\n{traceback.format_exc()[-1500:]}\n```")
        save(run, results)
    log(run, "\n# campaign complete")


if __name__ == "__main__":
    main()
