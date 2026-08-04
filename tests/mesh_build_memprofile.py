"""Per-stage memory profile of the CURRENT 3-D adaptive mesh build.

Attributes peak RSS to each stage (uniform base -> band refine loop ->
2:1 balance -> shell classify -> build_mesh -> constraints -> two-sided
surrogate extraction) at a ladder of sizes, and fits GB-per-M-DOF scaling.
Informs the streaming-build spec (which stage owns the 209.7 GB wall).

    .venv/bin/python tests/mesh_build_memprofile.py

Env: CASES ("5:7,6:8,6:9" base:refine pairs), default keeps the Mac happy.
Not pytest-collected.
"""
import os
import sys
import threading
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(__file__))


class RssSampler:
    """Background peak-RSS sampler (50 ms cadence)."""

    def __init__(self):
        self.proc = psutil.Process()
        self.peak = 0
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.proc.memory_info().rss)
            time.sleep(0.05)

    def __enter__(self):
        self.peak = self.proc.memory_info().rss
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop = True
        self._t.join()
        self.peak = max(self.peak, self.proc.memory_info().rss)


def profile_case(base, refine):
    from p2r1c_thin_plate_flow_3d import _make_sheet  # noqa
    from diffsim.octree.build import build_uniform, refine_elements
    from diffsim.octree.balance import balance2to1
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.faces import face_tables
    from diffsim.sbm.surrogate import (classify_shell_intercepted,
                                       extract_two_sided_surrogate)

    sheet = _make_sheet(0.375, 0.5, 0.5, 0.125, 0.125)
    rows = []
    gb = 1024 ** 3
    base_rss = psutil.Process().memory_info().rss

    def stage(name, fn):
        with RssSampler() as s:
            t0 = time.time()
            out = fn()
            el = time.time() - t0
        rows.append((name, (s.peak - base_rss) / gb, el))
        print(f"  {name:24s} peakRSS+{(s.peak - base_rss) / gb:7.2f} GB  "
              f"{el:7.1f}s", flush=True)
        return out

    # Stage 1: the REAL production builder end-to-end (tree refine + 2:1
    # balance + classify + build_mesh + constraints all inside).
    from p2r1c_thin_plate_flow_3d import build_adaptive_plate_mesh
    amr = stage("REAL build_adaptive(all)", lambda: build_adaptive_plate_mesh(
        base, refine, sheet, band_cells=2))
    mesh, ret = amr["mesh"], amr["ret"]

    # Attribution by re-running the post-tree stages on the builder's outputs
    # (their peaks attribute those stages; total-minus-these ~ tree ops).
    mesh2 = stage("re: build_mesh", lambda: build_mesh(ret, p=1))
    stage("re: build_constraints", lambda: build_constraints(mesh2))
    ftab = face_tables(1, 3)
    stage("extract_two_sided", lambda: extract_two_sided_surrogate(ret, sheet, ftab))

    n_nodes = len(mesh.node_coords)
    mdof = n_nodes * 4 / 1e6
    total_peak = max(r[1] for r in rows)
    print(f"[case base={base} refine={refine}] n_nodes={n_nodes} "
          f"({mdof:.2f} M-DOF)  stage-peak={total_peak:.2f} GB  "
          f"=> {total_peak / max(mdof, 1e-9):.2f} GB/M-DOF", flush=True)
    return dict(base=base, refine=refine, n_nodes=n_nodes, mdof=mdof,
                rows=rows, peak_gb=total_peak)


if __name__ == "__main__":
    cases = os.environ.get("CASES", "5:7,6:8,6:9")
    results = []
    for c in cases.split(","):
        b, r = (int(x) for x in c.split(":"))
        print(f"=== case base=L{b} refine=r{r} ===", flush=True)
        results.append(profile_case(b, r))
    print("\n=== SUMMARY (per-stage peak GB by case) ===", flush=True)
    names = [r[0] for r in results[0]["rows"]]
    hdr = "stage".ljust(24) + "".join(f"  L{c['base']}/r{c['refine']}".rjust(10)
                                      for c in results)
    print(hdr, flush=True)
    for i, n in enumerate(names):
        print(n.ljust(24) + "".join(f"{c['rows'][i][1]:10.2f}"
                                    for c in results), flush=True)
    print("MEMPROFILE-OK", flush=True)
