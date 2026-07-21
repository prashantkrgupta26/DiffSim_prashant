"""Task #37: stage-level profile of the 3-D film production step
(wodo_g5_rung physics, blockch + device assembly) — names the host-side
stages behind the forensics' ~61 s/step host floor.

BEFORE numbers (pre-v1.4 data path, stage-timed verbatim copy of the
old _attempt_device): commit 181deeb of this file; measured 2026-07-20
on gpubox Ada, 128x128x48: 31.60 s/step wall, solve 73.6%, host
fraction 8.35 s/step (gp_fields 4.43 + upload 0.99 + bdf 0.50 + rng
0.38 + flux/update 0.3 + elem_kernel 1.69 device).

THIS version instruments the v1.4 device-resident path by wrapping the
factored stage methods (_bdf_time_device / _gp_eval_device /
_fill_device_system / _solve_device_blockch) with synchronized timers
+ cProfile for function-level naming.

Run (box):
  LD_LIBRARY_PATH=/usr/lib/wsl/lib .venv/bin/python \
      benchmarks/phase-field/wodo_profile_host.py --nx 128 --nz 48 \
      --device cuda:1 --warmup 2 --steps 5
"""

import os as _bos, sys as _bsys  # noqa: E402
_bsys.path.insert(0, _bos.path.dirname(_bos.path.dirname(_bos.path.abspath(__file__))))
import _bench_bootstrap  # noqa: E402,F401
import argparse
import cProfile
import io
import pstats
import time

import numpy as np
import warp as wp

from diffsim.octree.build import build_uniform, Octree
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.wodo_film import WodoFilmStepper

import wodo_fig67 as f67

NS = f67.NS
LAT_PHYS = 3.3
PHI_S0 = 0.66


class LaunchAudit:
    """Task #40 first deliverable: attribute warp launches / copies to
    (stage x kernel).  Monkeypatches wp.launch / wp.copy /
    wp.capture_launch with counting wrappers; stage labels are set by the
    ProfiledFilm wrappers (solve / gp_eval / fill / bdf / other)."""

    def __init__(self):
        self.launches = {}          # (stage, kernel-name) -> count
        self.copies = {}            # stage -> count
        self.replays = {}           # stage -> graph replays (captured path)
        self.syncs = {}             # stage -> device->host readbacks (.numpy)
        self.dev_syncs = {}         # stage -> explicit synchronize_device
        self.stage = "other"

    def install(self):
        self._launch, self._copy = wp.launch, wp.copy
        self._cap = getattr(wp, "capture_launch", None)
        # Task #42: the DURABLE lever is the COUNT OF SYNC POINTS (#40 G3).
        # A per-inner-solve rhs upload + solution `.numpy()` download +
        # each outer matvec_numpy round-trip is a host-blocking readback
        # that stalls the pipeline; count them directly (array.numpy is the
        # device->host sync) alongside explicit synchronize_device calls.
        self._np = wp.array.numpy
        self._syncdev = getattr(wp, "synchronize_device", None)

        def launch(*a, **k):
            kern = a[0] if a else k.get("kernel")
            name = getattr(kern, "key", None) or repr(kern)
            name = name.rsplit(".", 1)[-1]
            key = (self.stage, name)
            self.launches[key] = self.launches.get(key, 0) + 1
            return self._launch(*a, **k)

        def copy(*a, **k):
            self.copies[self.stage] = self.copies.get(self.stage, 0) + 1
            return self._copy(*a, **k)

        _self = self

        def arr_numpy(self, *a, **k):
            # only count device (CUDA) readbacks — host arrays don't sync
            if str(getattr(self, "device", "cpu")).startswith("cuda"):
                _self.syncs[_self.stage] = _self.syncs.get(_self.stage, 0) + 1
            return _self._np(self, *a, **k)

        wp.launch, wp.copy = launch, copy
        wp.array.numpy = arr_numpy
        if self._syncdev is not None:
            def syncdev(*a, **k):
                self.dev_syncs[self.stage] = self.dev_syncs.get(
                    self.stage, 0) + 1
                return self._syncdev(*a, **k)
            wp.synchronize_device = syncdev
        if self._cap is not None:
            def cap(*a, **k):
                self.replays[self.stage] = self.replays.get(self.stage,
                                                            0) + 1
                return self._cap(*a, **k)
            wp.capture_launch = cap

    def report(self, nsteps):
        tot = sum(self.launches.values())
        totsync = sum(self.syncs.values())
        print(f"\n== LAUNCH AUDIT ({tot} launches = "
              f"{tot / max(nsteps, 1):.0f}/step; wp.copy "
              f"{sum(self.copies.values())} = "
              f"{sum(self.copies.values()) / max(nsteps, 1):.0f}/step; "
              f"graph replays {sum(self.replays.values())}) ==")
        print(f"== SYNC AUDIT (Task #42: {totsync} device->host readbacks "
              f"= {totsync / max(nsteps, 1):.0f}/step; "
              f"synchronize_device {sum(self.dev_syncs.values())}) ==")
        for st in sorted(self.syncs, key=self.syncs.get, reverse=True):
            print(f"  stage {st:12s} {self.syncs[st]:9d} readbacks "
                  f"({self.syncs[st] / max(nsteps, 1):9.0f}/step)")
        by_stage = {}
        for (st, _), c in self.launches.items():
            by_stage[st] = by_stage.get(st, 0) + c
        for st in sorted(by_stage, key=by_stage.get, reverse=True):
            print(f"  stage {st:12s} {by_stage[st]:9d} launches "
                  f"({by_stage[st] / max(nsteps, 1):9.0f}/step)  "
                  f"copies {self.copies.get(st, 0):7d}  "
                  f"replays {self.replays.get(st, 0):7d}  "
                  f"syncs {self.syncs.get(st, 0):7d}")
            per = {n: c for (s, n), c in self.launches.items() if s == st}
            for n in sorted(per, key=per.get, reverse=True)[:12]:
                print(f"      {n:28s} {per[n]:9d}  "
                      f"({per[n] / max(nsteps, 1):9.0f}/step)")


class _TimedRNG:
    """numpy Generator proxy timing the standard_normal draws."""

    def __init__(self, rng, tt):
        self._rng, self._tt = rng, tt

    def standard_normal(self, *a, **k):
        t0 = time.time()
        r = self._rng.standard_normal(*a, **k)
        self._tt["noise_rng"] = (self._tt.get("noise_rng", 0.0)
                                 + time.time() - t0)
        return r


class ProfiledFilm(WodoFilmStepper):
    """Production stepper with synchronized stage timers around the
    factored v1.4 stage methods (no math changes)."""

    def __init__(self, *a, **k):
        self.audit = k.pop("audit", None)
        super().__init__(*a, **k)
        self.tt = {}
        self.n_it = 0
        self._nrng = _TimedRNG(self._nrng, self.tt)

    def _set_stage(self, s):
        if self.audit is not None:
            self.audit.stage = s

    def _tic(self, sync=False):
        if sync:
            wp.synchronize_device(self.dm.device)
        return time.time()

    def _acc(self, key, t0, sync=False):
        if sync:
            wp.synchronize_device(self.dm.device)
        self.tt[key] = self.tt.get(key, 0.0) + time.time() - t0

    def _bdf_time_device(self, dt):
        self._set_stage("bdf")
        t0 = self._tic(sync=True)
        r = super()._bdf_time_device(dt)
        self._acc("bdf_hist_dev", t0, sync=True)
        self._set_stage("other")
        return r

    def _gp_eval_device(self, x):
        self.n_it += 1
        self._set_stage("gp_eval")
        t0 = self._tic(sync=True)
        super()._gp_eval_device(x)
        self._acc("gp_eval_dev", t0, sync=True)
        self._set_stage("other")

    def _fill_device_system(self, *a):
        self._set_stage("fill")
        t0 = self._tic(sync=True)
        super()._fill_device_system(*a)
        self._acc("fill_kern_scat", t0, sync=True)
        self._set_stage("other")

    def _solve_device_blockch(self, asm):
        self._set_stage("solve")
        t0 = self._tic(sync=True)
        r = super()._solve_device_blockch(asm)
        self._acc("solve", t0, sync=True)
        self._set_stage("other")
        return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=128)
    ap.add_argument("--ny", type=int, default=None)
    ap.add_argument("--nz", type=int, default=48)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=1e-4)
    ap.add_argument("--linsolver", type=str, default="blockch")
    ap.add_argument("--pstats-out", type=str,
                    default="logs/wodo_profile_host.pstats")
    ap.add_argument("--no-launch-audit", action="store_true")
    args = ap.parse_args()
    ny = args.ny if args.ny is not None else args.nx
    nx, nz, device = args.nx, args.nz, args.device
    wp.init()

    level = int(np.ceil(np.log2(max(nx, ny, nz))))
    hc = 2.0 ** -level
    tree0 = build_uniform(level, dim=3)
    cen = tree0.centers()
    keep = ((cen[:, 0] < nx * hc) & (cen[:, 1] < ny * hc)
            & (cen[:, 2] < nz * hc))
    tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=3,
                  periodic=tree0.periodic)
    del tree0, cen, keep
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    ndofs = 4 * len(mesh.node_coords)
    print(f"profile mesh: {len(tree)} elements ({nx}x{ny}x{nz}), "
          f"{ndofs} dofs, device={device}", flush=True)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=3), device)

    pp0 = pf0 = (1.0 - PHI_S0) / 2.0
    D0 = PHI_S0 * 1.0 + (1.0 - PHI_S0) * 1e-3
    M11 = D0 / (1.0 / (5.0 * pp0) + 1.0 / (NS * PHI_S0))
    M22 = D0 / (1.0 / (5.0 * pf0) + 1.0 / (NS * PHI_S0))
    kap = f67.kappa_of(5)
    lat_scale = LAT_PHYS / (nx * hc)
    audit = None if args.no_launch_audit else LaunchAudit()
    st = ProfiledFilm(
        dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, NS),
        M=(M11, 0.0, M22), kappa=(kap, kap), k_e=0.3, dt=args.dt,
        lat_scale=lat_scale, linsolver=args.linsolver, noise=1e-3,
        var_mob=True, b_reg=1e-3, noise_seed=17,
        use_device_assembly=True, audit=audit)
    rng = np.random.default_rng(17)
    st.set_initial(
        lambda x: pp0 + 0.01 * rng.standard_normal(len(x)),
        lambda x: pf0 + 0.01 * rng.standard_normal(len(x)))

    steps = []

    def cb(s, K, dt, iters):
        steps.append((time.time(), dt, iters))
        print(f"  step {len(steps):3d} t={s.t:.6f} dt={dt:.2e} "
              f"newton={iters}", flush=True)

    t0 = time.time()
    st.march(h_min=0.30, phis_stop=0.05, max_steps=args.warmup,
             callback=cb)
    print(f"warmup: {args.warmup} steps in {time.time() - t0:.0f}s "
          f"(incl. compile + symbolic setup)", flush=True)

    # -- timed, profiled steps ------------------------------------------
    st.tt.clear()
    st.n_it = 0
    n0 = len(steps)
    if audit is not None:
        audit.install()          # count launches over the timed steps only
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    st.march(h_min=0.30, phis_stop=0.05, max_steps=args.steps,
             callback=cb)
    pr.disable()
    wall = time.time() - t0
    nst = len(steps) - n0
    iters = [s[2] for s in steps[n0:]]

    print(f"\n== STAGE TABLE ({nst} steps, {st.n_it} Newton iterates, "
          f"newton per step {iters}, wall {wall:.1f}s = "
          f"{wall / max(nst, 1):.2f} s/step) ==")
    tot = sum(st.tt.values())
    for k in sorted(st.tt, key=st.tt.get, reverse=True):
        v = st.tt[k]
        print(f"  {k:14s} {v:8.2f}s  {v / max(nst, 1):7.2f} s/step  "
              f"{100 * v / wall:5.1f}% of wall")
    print(f"  {'(untimed)':14s} {wall - tot:8.2f}s  "
          f"{(wall - tot) / max(nst, 1):7.2f} s/step  "
          f"{100 * (wall - tot) / wall:5.1f}% of wall")
    host = wall - st.tt.get("solve", 0.0)
    print(f"  HOST FRACTION (wall - solve): {host:.1f}s = "
          f"{host / max(nst, 1):.2f} s/step = {100 * host / wall:.1f}%")
    print(f"  gpu peak {wp.get_mempool_used_mem_high(device) / 2 ** 30:.1f} GB")
    if audit is not None:
        audit.report(nst)

    _bos.makedirs(_bos.path.dirname(args.pstats_out) or ".",
                  exist_ok=True)
    pr.dump_stats(args.pstats_out)
    for sort in ("tottime", "cumulative"):
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats(sort).print_stats(30)
        print(f"\n== cProfile top-30 by {sort} ==")
        print(s.getvalue())


if __name__ == "__main__":
    main()
