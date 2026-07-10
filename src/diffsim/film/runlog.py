"""RunLog — the four-layer diagnostic log of a film run.

(a) PREFLIGHT: rule checks, printed + written before compute
    (preflight.py; stored in runlog.txt and as a jsonl record).
(b) FLIGHT RECORDER: one JSONL record per ATTEMPT (accepted and
    rejected) in runlog.jsonl -- t, dt, h, accepted, newton_its,
    blockch outer its + fallback flag, per-component solute-content
    drift vs step 0, phi min/max per component, wall clock, GPU memory
    -- plus a human-readable line per accepted step in runlog.txt.
(c) AUTOPSY on failure (exception or dt-ladder collapse): fields npz +
    last 50 jsonl records + config + a DIAGNOSIS section pattern-
    matched from the known-failure catalog (diagnose()).
(d) PROVENANCE header in every log: git SHA, config hash, seeds,
    package versions, GPU name.

DIAGNOSIS CATALOG (each rule cites its evidence base):
  dt-collapse-at-onset   dt ladder collapsed with Newton at its cap ->
                         dt0 exceeds the onset nonlinearity scale
                         (Appendix-A ladder can only rescue from a
                         solvable dt; wodo_film march contract).
  under-resolved-interface  phi outside [-0.02, 1.02] -> interface
                         thinner than the mesh (reports xi/h from
                         preflight). Bound provenance: healthy full-res
                         campaign runs stay within ~[-0.01, 1.01]
                         (wodo_nova RESULT phi_range class); 2x that
                         excursion = pathology.
  mass-drift             solute-content drift growing with step count
                         -> flux/advection inconsistency. Should NEVER
                         happen (conservation is exact by construction,
                         measured 1e-15 class): flagged as a BUG report.
  blockch-out-of-contract  >= 3 fallback-flagged solves in the last 50
                         attempts (production contract: ZERO fallbacks,
                         tests/test_ternary_blockprecond.py) -> the
                         curvature regime is outside the pairwise
                         contract (the d12/(2 sqrt(kap sigma)) ~ 1
                         boundary, docs/dev/2026-07-09-blockch-
                         preconditioner.md): use smaller dt0 or
                         splu/cudss.
  oom                    allocation failure -> G5 sizing model forecast
                         vs the actual card.
"""
import json
import os
import subprocess
import time

import numpy as np

AUTOPSY_TAIL = 50
PHI_LO, PHI_HI = -0.02, 1.02


# ---------------------------------------------------------------------
def provenance(params):
    """Layer (d): identity of this run."""
    def _git(*args):
        try:
            return subprocess.run(
                ["git", *args], capture_output=True, text=True,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                timeout=10).stdout.strip()
        except Exception:
            return "unknown"

    vers = {}
    for mod in ("numpy", "scipy", "warp", "torch", "yaml", "nvmath"):
        try:
            m = __import__(mod)
            vers[mod] = getattr(m, "__version__", "?")
        except ImportError:
            vers[mod] = "absent"
    gpu = "none"
    try:
        import torch
        if torch.cuda.is_available() and params.device.startswith("cuda"):
            gpu = torch.cuda.get_device_name(params.device)
    except Exception:
        pass
    return {
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_hash": params.config_hash(),
        "noise_seed": params.noise_seed,
        "ic_seed": params.ic_seed,
        "versions": vers,
        "gpu": gpu,
        "device": params.device,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _fmt_provenance(prov):
    v = prov["versions"]
    return "\n".join([
        "== PROVENANCE ==",
        f"git {prov['git_sha'][:12]}"
        + (" (dirty)" if prov["git_dirty"] else "")
        + f"  config {prov['config_hash']}"
        + f"  seeds noise={prov['noise_seed']} ic={prov['ic_seed']}",
        f"gpu {prov['gpu']} ({prov['device']})  {prov['timestamp']}",
        "versions " + " ".join(f"{k}={v[k]}" for k in sorted(v)),
    ])


def gpu_mem_gb(device):
    try:
        import torch
        if torch.cuda.is_available() and device.startswith("cuda"):
            free, total = torch.cuda.mem_get_info(device)
            return (total - free) / 1024 ** 3
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------
class RunLog:
    """Layers (b)-(d); layer (a) is passed in as a formatted report."""

    def __init__(self, outdir, params, resolved, preflight_report):
        self.outdir = outdir
        self.params = params
        self.resolved = resolved
        self.preflight_report = preflight_report
        os.makedirs(outdir, exist_ok=True)
        self.records = []                 # in-memory ring for autopsy
        self.prov = provenance(params)
        self.t0 = time.time()
        self._jsonl = open(os.path.join(outdir, "runlog.jsonl"), "w")
        self._txt = open(os.path.join(outdir, "runlog.txt"), "w")
        params.to_yaml(os.path.join(outdir, "config.yaml"))
        header = _fmt_provenance(self.prov)
        pf_text = preflight_report.format(params, resolved)
        for chunk in (header, pf_text):
            print(chunk, flush=True)
            self._txt.write(chunk + "\n")
        self._jsonl.write(json.dumps(
            {"type": "provenance", **self.prov}) + "\n")
        self._jsonl.write(json.dumps({
            "type": "preflight",
            "checks": [c.__dict__ for c in preflight_report.checks],
            "derived": preflight_report.derived}) + "\n")
        self._txt.write(
            "== FLIGHT RECORDER ==\n"
            "  step      t        dt      h     phis   it outer "
            "mass_drift  phi_p[min,max]    wall  gpu\n")
        self._flush()

    def _flush(self):
        self._jsonl.flush()
        self._txt.flush()

    # -- layer (b) -------------------------------------------------------
    def attempt(self, rec):
        rec = {"type": "attempt", **rec}
        self.records.append(rec)
        self._jsonl.write(json.dumps(rec) + "\n")
        if rec["accepted"]:
            o = rec.get("outer_its")
            g = rec.get("gpu_gb")
            line = (f"  {rec['step']:6d} {rec['t']:9.4f} {rec['dt']:9.3e}"
                    f" {rec['h']:6.4f} {rec['phis']:6.4f} "
                    f"{rec['newton_its']:3d} {o if o is not None else '-':>5}"
                    f" {rec['mass_drift']:10.2e} "
                    f"[{rec['phi_p_min']:+.3f},{rec['phi_p_max']:.3f}] "
                    f"{rec['wall']:7.1f}s "
                    f"{f'{g:.1f}G' if g is not None else '-'}")
            self._txt.write(line + "\n")
        else:
            self._txt.write(
                f"  {rec['step']:6d} {rec['t']:9.4f} {rec['dt']:9.3e} "
                f"REJECT newton_its={rec['newton_its']}\n")
        if len(self.records) % 50 == 0:
            self._flush()
        return rec

    def event(self, text):
        self._txt.write(text + "\n")
        self._jsonl.write(json.dumps({"type": "event", "text": text})
                          + "\n")
        self._flush()

    def final(self, summary):
        self._jsonl.write(json.dumps({"type": "final", **summary},
                                     default=str) + "\n")
        self._txt.write("== FINAL ==\n")
        for k, v in summary.items():
            self._txt.write(f"  {k} = {v}\n")
        self._flush()

    def close(self):
        self._jsonl.close()
        self._txt.close()

    # -- layer (c) -------------------------------------------------------
    def autopsy(self, reason, exc, stepper=None):
        """Dump evidence + run the diagnosis catalog. Returns the list
        of (rule, text) diagnoses."""
        adir = os.path.join(self.outdir, "autopsy")
        os.makedirs(adir, exist_ok=True)
        tail = self.records[-AUTOPSY_TAIL:]
        with open(os.path.join(adir, "tail.jsonl"), "w") as fh:
            for rec in tail:
                fh.write(json.dumps(rec) + "\n")
        self.params.to_yaml(os.path.join(adir, "config.yaml"))
        if stepper is not None:
            try:
                p1 = np.asarray(stepper.Tc @ stepper.x[0::4])
                p2 = np.asarray(stepper.Tc @ stepper.x[2::4])
                np.savez(os.path.join(adir, "fields.npz"),
                         phi_p=p1, phi_f=p2,
                         coords=stepper.mesh.node_coords,
                         h=stepper.h_curr, t=stepper.t,
                         dt=stepper.dt)
            except Exception as e:                # fields may be gone
                with open(os.path.join(adir, "fields_dump_failed.txt"),
                          "w") as fh:
                    fh.write(repr(e))
        diags = diagnose(tail, reason, exc, self.preflight_report,
                         self.resolved)
        text = ["== AUTOPSY ==",
                f"reason: {reason}",
                f"exception: {exc!r}" if exc else "exception: none",
                "== DIAGNOSIS =="]
        if diags:
            for rule, msg in diags:
                text.append(f"[{rule}] {msg}")
        else:
            text.append("no catalog match -- raw evidence attached "
                        "(tail.jsonl, fields.npz, config.yaml); this "
                        "is a NEW failure mode: file it")
        blob = "\n".join(text)
        with open(os.path.join(adir, "autopsy.txt"), "w") as fh:
            fh.write(_fmt_provenance(self.prov) + "\n" + blob + "\n")
        print(blob, flush=True)
        self._txt.write(blob + "\n")
        self._jsonl.write(json.dumps({
            "type": "autopsy", "reason": reason, "exception": repr(exc),
            "diagnosis": [{"rule": r, "text": t} for r, t in diags]})
            + "\n")
        self._flush()
        return diags


# ---------------------------------------------------------------------
def diagnose(tail, reason, exc, preflight_report, resolved):
    """The known-failure catalog (module docstring). Pure function of
    the evidence so the tests can drive it directly."""
    diags = []
    exc_text = (repr(exc) if exc else "").lower()
    accepted = [r for r in tail if r.get("accepted")]
    rejected = [r for r in tail if not r.get("accepted", True)]

    # OOM first: it usually arrives as the exception itself
    if any(s in exc_text for s in ("out of memory", "alloc",
                                   "cuda_error_out_of_memory",
                                   "memoryerror", "cusolver",
                                   "cudss_status_alloc")):
        mdofs = resolved.dofs / 1e6
        diags.append(("oom",
                      f"allocation failure at {mdofs:.2f}M dofs / "
                      f"{resolved.nnz / 1e6:.0f}M nnz; G5 sizing model "
                      f"forecast: GPU {2.35 + 2.05 * mdofs:.1f} GB, "
                      f"host {0.20 + 2.48 * mdofs:.1f} GB "
                      "(calibrated blockch_dev; cudss direct factors "
                      "cost MORE -- measured 228M-nnz ALLOC wall on "
                      "48 GB). Reduce resolution or move to "
                      "blockch_dev / a bigger card."))

    # dt-ladder collapse with Newton pinned at its cap (records carry
    # the configured cap as 'newton_cap')
    if reason == "dt_underflow" and rejected:
        maxed = [r for r in rejected
                 if r["newton_its"] >= r.get("newton_cap", 50)]
        newton_max = (max(r["newton_its"] for r in maxed)
                      if maxed else 0)
        if len(maxed) >= max(1, len(rejected) // 2):
            where = ("at onset (no prior accepted step)"
                     if not accepted else
                     f"after {accepted[-1]['step']} accepted steps "
                     f"(t={accepted[-1]['t']:.4g})")
            diags.append(("dt-collapse-at-onset",
                          f"dt ladder collapsed {where} with Newton at "
                          f"its {newton_max}-iteration cap on the "
                          "rejects: dt0 exceeds the onset nonlinearity "
                          "scale. Start smaller (dt0/100) and let the "
                          "Appendix-A ladder grow it."))

    # phi excursions beyond the healthy envelope
    exl = [r for r in accepted
           if min(r.get("phi_p_min", 0), r.get("phi_f_min", 0),
                  r.get("phi_s_min", 0)) < PHI_LO
           or max(r.get("phi_p_max", 1), r.get("phi_f_max", 1),
                  r.get("phi_s_max", 1)) > PHI_HI]
    if exl:
        c = preflight_report.status_of("interface-resolution")
        xi_txt = ""
        for chk in preflight_report.checks:
            if chk.rule == "interface-resolution":
                xi_txt = f" Preflight said [{c}]: {chk.detail}"
        diags.append(("under-resolved-interface",
                      f"phi left [{PHI_LO}, {PHI_HI}] in "
                      f"{len(exl)}/{len(accepted)} of the last accepted"
                      f" steps: interface thinner than the mesh."
                      + xi_txt))

    # conservation drift growing with steps (should never happen)
    if len(accepted) >= 10:
        d = np.array([r["mass_drift"] for r in accepted])
        if d[-1] > 1e-8 and np.all(np.diff(
                d[np.linspace(0, len(d) - 1, 5).astype(int)]) >= 0):
            diags.append(("mass-drift",
                          f"solute-content drift grew to {d[-1]:.2e} "
                          "over the tail (conservation is exact by "
                          "construction, measured 1e-15 class): "
                          "flux/advection inconsistency -- this is a "
                          "BUG, report it with this autopsy directory"))

    # repeated blockch fallbacks
    fb = sum(1 for r in tail if r.get("fallback"))
    if fb >= 3:
        diags.append(("blockch-out-of-contract",
                      f"{fb} fallback-flagged solves in the last "
                      f"{len(tail)} attempts (production contract: "
                      "zero): the curvature regime is outside the "
                      "pairwise-Schur contract (the d12/(2 sqrt(kap "
                      "sigma)) ~ 1 boundary, docs/dev/2026-07-09-"
                      "blockch-preconditioner.md). Use smaller dt0 or "
                      "linsolver splu/cudss."))
    return diags
