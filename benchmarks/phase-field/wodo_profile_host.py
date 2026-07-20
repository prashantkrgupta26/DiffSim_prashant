"""Task #37 first deliverable: DIRECT stage-level profile of the 3-D
film production step (wodo_g5_rung physics, blockch + device assembly)
— names the host-side stages behind the ~61 s/step host floor that the
A100/GH200 forensics could only infer by subtraction.

Two instruments, same run:
  * a ProfiledFilm subclass whose _attempt_device is a stage-timed copy
    of the production method (wp.synchronize_device brackets so async
    GPU work lands in the stage that launched it);
  * cProfile over the timed steps for function-level naming (einsum,
    standard_normal, wp.array upload, lgmres internals, .numpy pulls).

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
from diffsim.physics.wodo_film import WodoFilmStepper, make_wodo_newton

import wodo_fig67 as f67

NS = f67.NS
LAT_PHYS = 3.3
PHI_S0 = 0.66


class ProfiledFilm(WodoFilmStepper):
    """_attempt_device copied VERBATIM from wodo_film.py @ 7aa981f with
    stage timers + syncs inserted (profiling instrument only — the
    measured run is the production math, launch for launch)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.tt = {}      # stage -> cumulative seconds
        self.n_it = 0     # Newton iterates timed

    def _tic(self, sync=False):
        if sync:
            wp.synchronize_device(self.dm.device)
        return time.time()

    def _acc(self, key, t0, sync=False):
        if sync:
            wp.synchronize_device(self.dm.device)
        self.tt[key] = self.tt.get(key, 0.0) + time.time() - t0

    def _attempt_device(self, dt, K):
        if self._asm is None:
            t0 = self._tic()
            self._init_device_assembly()
            self._acc("z_asm_setup", t0, sync=True)
        asm = self._asm
        d = self.dm.device
        t0 = self._tic()
        sigma, h1_gp, h2_gp = self._bdf_time(dt)
        self._acc("bdf_hist_gp", t0)
        self._sigma = sigma
        minv = 1.0 / self.h_curr
        mlat = 1.0 / self.lat_scale
        mvert = self.y_comp * minv
        coef = K * minv * self.y_comp
        rho = self.noise * np.sqrt(2.0 / dt)
        t0 = self._tic()
        q_gp = {}
        for pv, b in self.dm.bins.items():
            ngp = len(self.mesh.conn_of[pv]) * b["nqp"]
            q_gp[pv] = (
                rho * self._nrng.standard_normal((ngp, self.dm.dim)),
                rho * self._nrng.standard_normal((ngp, self.dm.dim)))
        self._acc("noise_rng", t0)
        negi = self.mob_model == "negi"
        Drp, Drf = (self.D_ratio if (self.var_mob or negi)
                    else (-1.0, -1.0))
        mobm = 1 if negi else 0
        t0 = self._tic()
        flux_vals_d = wp.array(
            np.ascontiguousarray(-coef * self._flux_base),
            dtype=wp.float64, device=d)
        self._acc("flux_host", t0, sync=True)
        x = self.x.copy()
        bufs = {}
        for it in range(self.newton_max):
            self.n_it += 1
            t0 = self._tic()
            fields = [self._gp(x[i::4]) for i in range(4)]
            self._acc("gp_fields", t0)
            t0 = self._tic(sync=True)
            asm.zero_fill()
            self._acc("zero_fill", t0, sync=True)
            for k_bin, (pv, b, ne, nbf, gdof) in enumerate(asm._bins):
                nqp = b["nqp"]
                nl = 4 * nbf
                if k_bin not in bufs:
                    nb_cap = min(ne, max(1, (2 << 30) // (nl * nl * 8)))
                    bufs[k_bin] = (
                        nb_cap,
                        wp.zeros((nb_cap, nl, nl), dtype=wp.float64,
                                 device=d),
                        wp.zeros((nb_cap, nl), dtype=wp.float64,
                                 device=d))
                nb_cap, Ae, be = bufs[k_bin]
                kk = make_wodo_newton(nbf, nqp, self.dm.dim)
                for e0 in range(0, ne, nb_cap):
                    nb = min(nb_cap, ne - e0)
                    s0, s1 = e0 * nqp, (e0 + nb) * nqp
                    arr = lambda a_: wp.array(
                        np.ascontiguousarray(a_[s0:s1]),
                        dtype=wp.float64, device=d)
                    t0 = self._tic(sync=True)
                    ins = [
                        arr(fields[0][0][pv]), arr(fields[0][1][pv]),
                        arr(fields[1][0][pv]), arr(fields[1][1][pv]),
                        arr(fields[2][0][pv]), arr(fields[2][1][pv]),
                        arr(fields[3][0][pv]), arr(fields[3][1][pv]),
                        arr(h1_gp[pv]), arr(h2_gp[pv]),
                        arr(q_gp[pv][0]), arr(q_gp[pv][1])]
                    self._acc("upload_gp", t0, sync=True)
                    t0 = self._tic()
                    Ae.zero_()
                    be.zero_()
                    wp.launch(kk, dim=nb, inputs=[
                        b["conn"], b["h"][e0:e0 + nb], b["N"], b["dN"],
                        b["w"], *ins,
                        self.theta_wp[pv][s0:s1],
                        wp.float64(self.M11), wp.float64(self.M12),
                        wp.float64(self.M22), wp.float64(Drp),
                        wp.float64(Drf), wp.int32(mobm),
                        wp.float64(self.c12),
                        wp.float64(self.c1s), wp.float64(self.c2s),
                        wp.float64(1.0 / self.N1),
                        wp.float64(1.0 / self.N2),
                        wp.float64(1.0 / self.Ns),
                        wp.float64(self.b_reg),
                        wp.float64(self.ch2), wp.float64(self.ch3),
                        wp.float64(self.ch4),
                        wp.float64(self.kap1), wp.float64(self.kap2),
                        wp.float64(sigma), wp.float64(mlat),
                        wp.float64(mvert), wp.float64(minv),
                        wp.float64(K),
                        Ae, be], device=d)
                    self._acc("elem_kernel", t0, sync=True)
                    t0 = self._tic()
                    asm.scatter_batch(k_bin, e0, Ae, be, nb)
                    self._acc("scatter", t0, sync=True)
            t0 = self._tic()
            asm.add_matrix_values(self._flux_slots_d, flux_vals_d)
            f1 = np.asarray(self.Tc @ x[0::4])
            f2 = np.asarray(self.Tc @ x[2::4])
            Mf = coef * self.top_face_M
            load = np.concatenate(
                [np.einsum("fab,fb->fa", Mf, fv[self.top_faces]).ravel()
                 for fv in (f1, f2)])
            asm.add_rhs_values(
                self._flux_gdof_d,
                wp.array(np.ascontiguousarray(load), dtype=wp.float64,
                         device=d))
            self._acc("flux_host", t0, sync=True)
            t0 = self._tic()
            dx = (self._solve_device_blockch(asm)
                  if self.linsolver in ("blockch", "blockch_dev")
                  else self._solve_device(asm))
            self._acc("solve", t0, sync=True)
            t0 = self._tic()
            if not np.isfinite(dx).all() or np.abs(dx).max() > 1e6:
                self._acc("newton_update", t0)
                return None, it + 1, False
            x = x + dx
            conv = np.abs(dx).max() < self.newton_tol
            self._acc("newton_update", t0)
            if conv:
                return x, it + 1, True
        return x, self.newton_max, False


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
    st = ProfiledFilm(
        dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, NS),
        M=(M11, 0.0, M22), kappa=(kap, kap), k_e=0.3, dt=args.dt,
        lat_scale=lat_scale, linsolver=args.linsolver, noise=1e-3,
        var_mob=True, b_reg=1e-3, noise_seed=17,
        use_device_assembly=True)
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
    st.tt = {}
    st.n_it = 0
    n0 = len(steps)
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
