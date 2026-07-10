"""FilmRun — build mesh + WodoFilmStepper from FilmParams, march with
the Appendix-A ladder, write snapshots + the four-layer RunLog.

    from diffsim.film import FilmParams, FilmRun
    FilmRun(FilmParams.from_yaml("case.yaml")).run("out/")

CLI: python -m diffsim.film <config.yaml> [--outdir DIR] [...].

Outputs in <outdir>:
    runlog.txt / runlog.jsonl    the RunLog (runlog.py docstring)
    config.yaml                  resolved parameter echo
    <name>_h{0.90,...}.npz       field snapshots at height milestones
                                 (2-D: phi_p/phi_f grids; 3-D: nodal)
    <name>_final.npz             final fields + h + t (+ 2-D grids)
    autopsy/                     only on failure (fields, tail, DIAGNOSIS)

Solute-content accounting (flight recorder): content_i =
h_curr * Int phi_i dtheta with the integral taken by CORNER-AVERAGE
quadrature — exact for bilinear/trilinear P1 fields on tensor cells, so
the recorded drift is the true conservation identity at machine
precision (measured class 1e-15, wodo campaign).
"""
import os
import time

import numpy as np

from .params import FilmParams
from .preflight import run_preflight, PreflightError
from .runlog import RunLog, gpu_mem_gb
from . import analysis


class _IterCache(dict):
    """Solver cache that records blockch ('blockch_iters', key) writes
    so the flight recorder can attribute outer its to attempts
    (pattern: tests/test_ternary_blockprecond._RecCache)."""

    def __init__(self):
        super().__init__()
        self.iters = []

    def __setitem__(self, k, v):
        if isinstance(k, tuple) and k[0] == "blockch_iters":
            self.iters.append(v[0] if isinstance(v, tuple) else v)
        super().__setitem__(k, v)


class FilmRun:
    def __init__(self, params: FilmParams):
        self.p = params
        self.r = params.resolve()

    # -- mesh + stepper ---------------------------------------------------
    def build(self):
        import warp as wp
        from ..octree.build import build_uniform, Octree
        from ..mesh.nodes import build_mesh
        from ..mesh.constraints import build_constraints
        from ..mesh.basis import basis_tables
        from ..assembly.operators import DeviceMesh
        from ..physics.wodo_film import WodoFilmStepper

        p, r = self.p, self.r
        wp.init()
        tree0 = build_uniform(r.level, dim=p.dim)
        cen = tree0.centers()
        keep = np.ones(len(tree0), bool)
        for d, c in enumerate(r.cells):
            keep &= cen[:, d] < c * r.hc
        tree = Octree(tree0.keys[keep], tree0.levels[keep], dim=p.dim,
                      periodic=tree0.periodic)
        mesh = build_mesh(tree, p=1)
        cons = build_constraints(mesh)
        dm = DeviceMesh.from_mesh(mesh, cons,
                                  basis_tables(1, dim=p.dim), p.device)
        st = WodoFilmStepper(
            dm, chi=tuple(p.chi), N=tuple(float(n) for n in p.N),
            M=(r.M11, 0.0, r.M22), kappa=r.kappa, k_e=p.Bi, dt=p.dt0,
            newton_tol=p.newton_tol, newton_max=p.newton_max,
            lat_scale=r.lat_scale, linsolver=p.linsolver,
            noise=p.noise, noise_seed=p.noise_seed,
            var_mob=r.var_mob, D_ratio=r.D_pair, b_reg=p.b_reg,
            mob_model=("negi" if p.mobility == "negi" else "wodo"),
            f_cheb=tuple(p.f_cheb),
            use_device_assembly=p.device_assembly)
        st._solver_cache = _IterCache()
        rng = np.random.default_rng(p.ic_seed)
        st.set_initial(
            lambda x: r.phi_p0 + p.ic_noise * rng.standard_normal(len(x)),
            lambda x: r.phi_f0 + p.ic_noise * rng.standard_normal(len(x)))
        self.mesh, self.dm, self.st = mesh, dm, st
        return st

    # -- the run ------------------------------------------------------------
    def run(self, outdir, query_hardware=True):
        p, r = self.p, self.r
        report = run_preflight(p, r, query_hardware=query_hardware)
        log = RunLog(outdir, p, r, report)
        if report.failed:
            fails = [c for c in report.checks if c.status == "FAIL"]
            if p.preflight == "strict":
                log.event("aborted by strict preflight")
                log.close()
                raise PreflightError(
                    "preflight FAIL (strict): "
                    + "; ".join(f"[{c.rule}] {c.detail}" for c in fails))
            log.event("continuing past preflight FAIL "
                      "(preflight: warn): "
                      + ", ".join(c.rule for c in fails))

        st = self.build()
        rec = _Recorder(self, log)
        reason, exc = "exception", None
        t0 = time.time()
        try:
            reason = st.march(
                h_min=p.h_min, phis_stop=p.phis_stop,
                max_steps=p.max_steps, dh_cap=p.dh_cap,
                dt_min=p.dt_min, wall_cap=p.wall_cap,
                on_attempt=rec.on_attempt)
        except Exception as e:
            exc = e
        wall = time.time() - t0

        summary = rec.summary(reason, wall)
        if exc is not None or reason == "dt_underflow":
            summary["autopsy"] = [
                rule for rule, _ in log.autopsy(reason, exc, st)]
        else:
            rec.write_final_npz()
        log.final(summary)
        log.close()
        print("RESULT " + " ".join(
            f"{k}={v}" for k, v in summary.items()
            if k not in ("onset", "onset_vertical")), flush=True)
        if exc is not None:
            raise exc
        return summary


# ---------------------------------------------------------------------
class _Recorder:
    """Flight-recorder callback + snapshot writer (promoted from the
    wodo_fig67/wodo_nova Recorder2D pattern)."""

    def __init__(self, run, log):
        self.run, self.log = run, log
        p, r, mesh = run.p, run.r, run.mesh
        self.p, self.r = p, r
        self.st = run.st
        coords = mesh.node_coords
        # corner-average (trapezoid) weights: exact Int over the strip
        # for P1 tensor elements (module docstring)
        w = np.zeros(len(coords))
        for pv, conn in mesh.conn_of.items():
            vol = mesh.tree.h()[mesh.bins[pv]] ** p.dim / 2 ** p.dim
            np.add.at(w, conn.ravel(),
                      np.repeat(vol, conn.shape[1]))
        self.w_node = w
        if p.dim == 2:
            self.ix = np.round(coords[:, 0] / r.hc).astype(int)
            self.iy = np.round(coords[:, 1] / r.hc).astype(int)
            self.nxn, self.nyn = self.ix.max() + 1, self.iy.max() + 1
        self.snap_todo = sorted((float(h) for h in p.snap_h),
                                reverse=True)
        self.mass0 = None
        self.mass_drift = 0.0
        self.pmin, self.pmax = np.inf, -np.inf
        self.nacc = 0
        self.nattempt = 0
        self.onset = None            # lateral: {t, h, theta, ...}
        self.onset_v = None          # vertical stratification onset
        self.t0 = time.time()
        self._iters_seen = 0

    # -- helpers -----------------------------------------------------------
    def _full(self, comp):
        return np.asarray(self.st.Tc @ self.st.x[comp::4])

    def grid(self, full):
        g = np.zeros((self.nyn, self.nxn))
        g[self.iy, self.ix] = full
        return g

    def _content(self, full):
        return self.st.h_curr * float(self.w_node @ full)

    def _outer_and_fallback(self):
        its = self.st._solver_cache.iters \
            if hasattr(self.st._solver_cache, "iters") else []
        new = its[self._iters_seen:]
        self._iters_seen = len(its)
        if not new:
            return None, False
        return max(i % 1000 for i in new), any(i >= 1000 for i in new)

    # -- the on_attempt hook -------------------------------------------------
    def on_attempt(self, st, K, dt, iters, ok):
        self.nattempt += 1
        outer, fallback = self._outer_and_fallback()
        rec = {"step": self.nattempt, "t": st.t, "dt": dt,
               "h": st.h_curr, "accepted": bool(ok),
               "newton_its": int(iters),
               "newton_cap": self.p.newton_max,
               "outer_its": outer, "fallback": bool(fallback),
               "wall": time.time() - self.t0,
               "gpu_gb": gpu_mem_gb(self.p.device)}
        if ok:
            self.nacc += 1
            p1, p2 = self._full(0), self._full(2)
            ps = 1.0 - p1 - p2
            c1, c2 = self._content(p1), self._content(p2)
            if self.mass0 is None:
                self.mass0 = (c1, c2)
            d1 = abs(c1 - self.mass0[0]) / abs(self.mass0[0])
            d2 = abs(c2 - self.mass0[1]) / abs(self.mass0[1])
            self.mass_drift = max(self.mass_drift, d1, d2)
            self.pmin = min(self.pmin, p1.min(), p2.min(), ps.min())
            self.pmax = max(self.pmax, p1.max(), p2.max(), ps.max())
            rec.update({
                "phis": float(ps.mean()),
                "mass_drift": max(d1, d2),
                "phi_p_min": float(p1.min()), "phi_p_max": float(p1.max()),
                "phi_f_min": float(p2.min()), "phi_f_max": float(p2.max()),
                "phi_s_min": float(ps.min()), "phi_s_max": float(ps.max()),
            })
            if self.p.dim == 2:
                self._onset_check(p2, st)
            self._snap_check(p1, p2, st)
        self.log.attempt(rec)
        if ok and self.nacc % self.p.log_every == 0:
            print(f"  [{self.p.name}] step {self.nacc:6d} t={st.t:9.4f}"
                  f" h={st.h_curr:.4f} phis={rec['phis']:.3f} "
                  f"dt={dt:.2e} it={iters:2d} "
                  f"wall={rec['wall']:7.1f}s", flush=True)

    def _onset_check(self, p2_full, st):
        """Two onset detectors on phi_f (both AMP_ON = 0.1, the house
        order-one amplitude): LATERAL — first row whose lateral std
        crosses AMP_ON (lateral domain structure); VERTICAL — first
        time a row MEAN deviates from the height average by AMP_ON
        (stratification/layering). Each records the vertical position
        theta of the leading row (0 = substrate, 1 = free surface)."""
        if self.onset is not None and self.onset_v is not None:
            return
        g = self.grid(p2_full)
        if self.onset is None:
            prof = analysis.lateral_std_profile(g)
            if prof.max() > analysis.AMP_ON:
                theta = float(np.argmax(prof)) / max(len(prof) - 1, 1)
                self.onset = {"t": st.t, "h": st.h_curr, "theta": theta,
                              "max_lateral_std": float(prof.max())}
                self.log.event(
                    f"onset(lateral): std(phi_f) crossed "
                    f"{analysis.AMP_ON} at t={st.t:.4f} "
                    f"h={st.h_curr:.4f} theta={theta:.3f} "
                    "(0=substrate, 1=surface)")
        if self.onset_v is None:
            dev = g.mean(axis=1)
            dev = dev - dev.mean()
            if np.abs(dev).max() > analysis.AMP_ON:
                theta = float(np.argmax(np.abs(dev))) \
                    / max(len(dev) - 1, 1)
                self.onset_v = {
                    "t": st.t, "h": st.h_curr, "theta": theta,
                    "dev": float(dev[np.argmax(np.abs(dev))])}
                self.log.event(
                    f"onset(vertical): row-mean phi_f deviation crossed"
                    f" {analysis.AMP_ON} at t={st.t:.4f} "
                    f"h={st.h_curr:.4f} theta={theta:.3f} "
                    f"(dev={self.onset_v['dev']:+.3f}; "
                    "0=substrate, 1=surface)")

    def _snap_check(self, p1, p2, st):
        while self.snap_todo and st.h_curr <= self.snap_todo[0]:
            tag = f"{self.snap_todo.pop(0):.2f}"
            path = os.path.join(self.log.outdir,
                                f"{self.p.name}_h{tag}.npz")
            data = dict(h=st.h_curr, t=st.t)
            if self.p.dim == 2:
                data["phi_p"] = self.grid(p1)
                data["phi_f"] = self.grid(p2)
            else:
                data["phi_p"], data["phi_f"] = p1, p2
            np.savez(path, **data)
            self.log.event(f"snapshot h={tag} -> {os.path.basename(path)}")

    # -- end of run -----------------------------------------------------------
    def summary(self, reason, wall):
        st = self.st
        p1, p2 = self._full(0), self._full(2)
        ps = 1.0 - p1 - p2
        s = {"name": self.p.name, "reason": reason,
             "steps": self.nacc, "rejects": st.n_reject,
             "t": round(st.t, 6), "h": round(st.h_curr, 6),
             "phis_end": round(float(ps.mean()), 6),
             "mass_drift": float(self.mass_drift),
             "phi_min": round(float(self.pmin), 6),
             "phi_max": round(float(self.pmax), 6),
             "wall_s": round(wall, 1),
             "onset": self.onset, "onset_vertical": self.onset_v}
        if self.p.dim == 2 and self.nacc:
            s["L_c"] = round(analysis.characteristic_length(
                self.grid(p2), self.p.Lx, st.h_curr), 6)
            s["onset_theta"] = (round(self.onset["theta"], 4)
                                if self.onset else None)
            s["onset_v_theta"] = (round(self.onset_v["theta"], 4)
                                  if self.onset_v else None)
        return s

    def write_final_npz(self):
        st = self.st
        p1, p2 = self._full(0), self._full(2)
        data = dict(phi_p=p1, phi_f=p2,
                    coords=self.run.mesh.node_coords,
                    h=st.h_curr, t=st.t)
        if self.p.dim == 2:
            data["grid_p"] = self.grid(p1)
            data["grid_f"] = self.grid(p2)
        np.savez(os.path.join(self.log.outdir,
                              f"{self.p.name}_final.npz"), **data)
