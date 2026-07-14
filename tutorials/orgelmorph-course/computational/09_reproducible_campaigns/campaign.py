"""OrgElMorph course - Computational C9: the reproducible-campaign runner.

A single well-verified run (Physics P1, Computational C1) is not a scientific
result -- a *campaign* is: the same immutable model evaluated across a
parameter sweep and a seed ensemble, aggregated with confidence intervals, and
reported WITHOUT silently dropping the runs that failed.  This module is the
small, real runner that turns a manifest (``manifest.yaml``) into that
campaign.

What it demonstrates (spec Phase 3 / C9):

* **Immutable resolved config.**  The manifest is loaded, validated, and
  frozen to ``campaign.resolved.yaml`` -- the re-runnable record of exactly
  what the campaign was, defaults made explicit (mirrors the per-run
  ``config.resolved.yaml`` the Phase-0 harness already writes).
* **Parameter sweeps x seed ensembles.**  The Cartesian product of the swept
  physical parameter (here the Flory--Huggins enthalpic parameter
  ``chi`` == ``fh_B``; see the chapter for why chi, not ``k_e``) with the seed
  list defines the run grid.  Each run is a real 2-D Cahn--Hilliard spinodal
  march (``physics/01`` ``run_spinodal``), fast enough to sweep locally.
* **Checkpoint / restart.**  Every run writes ``status.json`` +
  ``results.json`` into its own directory.  Re-invoking the campaign SKIPS
  runs already ``completed`` (idempotent restart) and RETRIES ones that
  ``failed`` -- so a crashed campaign resumes where it stopped, and fixing a
  manifest typo re-runs only the broken cell.
* **Failed-run recovery + NO SILENT DROP.**  A run that raises (a real
  campaign failure: a mistyped solver, an OOM, a diverged solve) is caught,
  recorded ``failed`` with its error, and the campaign CONTINUES.  A run that
  completes but violates a hard admissibility/tolerance gate is ALSO marked
  ``failed``.  Aggregation then *surfaces* every failure by id and reason and
  refuses to present a mean as if the ensemble were complete.
* **Aggregation + confidence intervals.**  Per swept value we report the
  seed-ensemble mean, sd, sem and a bootstrap 95% CI (the shared, unit-tested
  ``diffsim.diagnostics.stochastic``) -- and how many seeds actually
  contributed vs failed.
* **Exploratory vs confirmatory.**  Each sweep carries an ``intent`` tag; the
  report keeps them apart and refuses to dress an exploratory scan up as a
  confirmatory claim.

    PYTHONPATH=<repo>/src python campaign.py --manifest manifest.yaml \\
        --output outputs/demo --overwrite     # run the tiny real sweep
    PYTHONPATH=<repo>/src python campaign.py --manifest manifest.yaml \\
        --output outputs/demo                  # RESTART: skips completed
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import traceback

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
# physics/01 core (the fast 2-D CH spinodal engine) + the course commons
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir)))
sys.path.insert(0, os.path.abspath(os.path.join(
    _HERE, os.pardir, os.pardir, "physics", "01_ch_binary_energies")))

from common.provenance import RunProvenance                 # noqa: E402
from diffsim.diagnostics import stochastic as dstoch          # noqa: E402
from spinodal import run_spinodal, _lengthscales              # noqa: E402


# ---------------------------------------------------------------------------
# manifest schema (immutable resolved config)
# ---------------------------------------------------------------------------
# The manifest's ``base`` block is the frozen model; ``sweeps`` is a list of
# {name, param, values, seeds, intent}; ``deliberate_failures`` is a list of
# override dicts that are appended to the grid as injected faults (the
# no-silent-drop demonstration).  Defaults below make the resolved record
# explicit.
_BASE_DEFAULTS = {
    "engine": "ch_spinodal",
    "energy": "fh",
    "level": 5,
    "steps": 40,
    "dt": 0.02,
    "M": 1.0,
    "kappa": 5.0e-4,
    "fh_A": 0.15,
    "amp": 0.05,
    "measure_every": 5,
    "solver": "splu",
    "device": "cuda:0",
    # hard admissibility gate: a completed run must be finite and conserve
    # mass to this bound, else it is a FAILED run (surfaced, not dropped).
    "mass_drift_max": 5.0e-2,
    "observable": "L_area",
}


def resolve_manifest(path):
    """Load a manifest YAML and fill defaults -> the immutable resolved dict."""
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    base = dict(_BASE_DEFAULTS)
    base.update(raw.get("base", {}) or {})
    sweeps = raw.get("sweeps", []) or []
    if not sweeps:
        raise ValueError("manifest has no 'sweeps'")
    for sw in sweeps:
        sw.setdefault("intent", "exploratory")
        sw.setdefault("seeds", [1, 2])
        for key in ("name", "param", "values"):
            if key not in sw:
                raise ValueError(f"sweep missing required key '{key}': {sw}")
    resolved = {
        "campaign": raw.get("campaign", "unnamed"),
        "description": raw.get("description", ""),
        "base": base,
        "sweeps": sweeps,
        "deliberate_failures": raw.get("deliberate_failures", []) or [],
    }
    return resolved


def enumerate_runs(resolved):
    """Expand the resolved manifest into the flat list of run specs.

    Each run spec is a dict carrying its sweep name, intent, the swept
    parameter value, the seed, and any injected-fault overrides.  The run_id is
    deterministic (sweep.param=value.seed=s) so restart is idempotent.
    """
    runs = []
    for sw in resolved["sweeps"]:
        for val, seed in itertools.product(sw["values"], sw["seeds"]):
            runs.append({
                "run_id": f"{sw['name']}.{sw['param']}={val:g}.seed={seed}",
                "sweep": sw["name"], "intent": sw["intent"],
                "param": sw["param"], "value": float(val), "seed": int(seed),
                "injected_fault": False, "overrides": {},
            })
    for i, fault in enumerate(resolved["deliberate_failures"]):
        ov = dict(fault)
        label = ov.pop("label", f"fault{i}")
        runs.append({
            "run_id": f"fault.{label}",
            "sweep": ov.pop("sweep", "deliberate_failure"),
            "intent": "fault_injection",
            "param": ov.pop("param", "n/a"),
            "value": float(ov.pop("value", 0.0)),
            "seed": int(ov.pop("seed", 1)),
            "injected_fault": True, "overrides": ov,
        })
    return runs


# ---------------------------------------------------------------------------
# one run (real 2-D CH spinodal march) + its hard admissibility gate
# ---------------------------------------------------------------------------
def _observable(rec, name):
    """Compute the requested morphology observable from a run record."""
    end = rec["snaps"][max(rec["snaps"])]
    lam_peak, lam_fm, L_area = _lengthscales(end, rec["dx"])
    table = {"L_area": L_area, "lambda_fm": lam_fm, "lambda_peak": lam_peak}
    if name not in table:
        raise KeyError(f"unknown observable '{name}'")
    mass_drift = float(abs(rec["mass"][-1] - rec["mass"][0]))
    return {
        "value": float(table[name]), "observable": name,
        "L_area": float(L_area), "lambda_fm": float(lam_fm),
        "lambda_peak": float(lam_peak),
        "Fend": float(rec["F_total"][-1]),
        "c_min": float(end.min()), "c_max": float(end.max()),
        "mass_drift": mass_drift,
    }


def run_one(spec, base, out_dir):
    """Execute one run spec.  Returns a status dict; NEVER raises for a run
    failure (the campaign must survive a bad cell)."""
    os.makedirs(out_dir, exist_ok=True)
    p = dict(base)
    p.update(spec["overrides"])
    if not spec["injected_fault"]:
        p[spec["param"]] = spec["value"]     # the swept physical parameter
    status = {"run_id": spec["run_id"], "sweep": spec["sweep"],
              "intent": spec["intent"], "param": spec["param"],
              "value": spec["value"], "seed": spec["seed"],
              "injected_fault": spec["injected_fault"]}
    try:
        rec = run_spinodal(
            energy=p["energy"], level=p["level"], steps=p["steps"],
            dt=p["dt"], M=p["M"], kappa=p["kappa"], fh_A=p["fh_A"],
            fh_B=p["fh_B"], amp=p["amp"], seed=spec["seed"],
            device=p["device"], linsolver=p["solver"],
            measure_every=p["measure_every"])
        obs = _observable(rec, base["observable"])
        # hard admissibility / tolerance gate -> a completed-but-invalid run
        # is a FAILURE, surfaced (not a silently-averaged NaN).
        reasons = []
        if not np.isfinite(obs["value"]):
            reasons.append("non-finite observable")
        if obs["mass_drift"] > base["mass_drift_max"]:
            reasons.append(
                f"mass drift {obs['mass_drift']:.2e} > gate "
                f"{base['mass_drift_max']:.0e}")
        if reasons:
            status.update(state="failed", failure_class="validation",
                          reason="; ".join(reasons), results=obs)
        else:
            status.update(state="completed", results=obs)
    except Exception as exc:                 # a real crashed run
        status.update(state="failed", failure_class="exception",
                      reason=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    with open(os.path.join(out_dir, "status.json"), "w") as fh:
        json.dump(status, fh, indent=2, sort_keys=True)
    return status


# ---------------------------------------------------------------------------
# aggregation -- surfaces failures, never drops them
# ---------------------------------------------------------------------------
def aggregate(resolved, statuses):
    """Aggregate the campaign.  Returns a summary dict.

    Per (sweep, value) we report the seed-ensemble mean + sd + sem + bootstrap
    95% CI over the COMPLETED seeds, together with how many seeds contributed
    and how many failed -- so a partial ensemble is never presented as whole.
    Every failure is listed by id and reason; ``campaign_complete`` is False if
    any non-injected run failed.
    """
    by_state = {"completed": [], "failed": []}
    for s in statuses:
        by_state[s["state"]].append(s)

    # group completed runs by (sweep, value)
    groups = {}
    for s in by_state["completed"]:
        groups.setdefault((s["sweep"], s["value"]), []).append(s)

    sweep_meta = {sw["name"]: sw for sw in resolved["sweeps"]}
    points = []
    for (sweep, value), runs in sorted(groups.items()):
        vals = np.array([r["results"]["value"] for r in runs], float)
        n_expected = len(sweep_meta[sweep]["seeds"])
        agg = dstoch.ensemble_aggregate(vals)
        ci = (dstoch.bootstrap_ci(vals, seed=0) if vals.size >= 2
              else {"low": float(vals[0]), "high": float(vals[0]),
                    "estimate": float(vals[0]), "confidence": 0.95})
        # count failures at this exact (sweep, value)
        n_failed_here = sum(
            1 for f in by_state["failed"]
            if f["sweep"] == sweep and f["value"] == value)
        points.append({
            "sweep": sweep, "intent": sweep_meta[sweep]["intent"],
            "param": sweep_meta[sweep]["param"], "value": value,
            "mean": float(agg["mean"]), "sd": float(agg["sd"]),
            "sem": float(agg["sem"]),
            "ci_low": ci["low"], "ci_high": ci["high"],
            "n_seeds_used": int(agg["n"]),
            "n_seeds_expected": n_expected,
            "n_seeds_failed": n_failed_here,
            "ensemble_complete": (agg["n"] == n_expected),
        })

    failures = [{"run_id": f["run_id"], "sweep": f["sweep"],
                 "injected_fault": f["injected_fault"],
                 "failure_class": f.get("failure_class"),
                 "reason": f.get("reason")}
                for f in by_state["failed"]]
    real_failures = [f for f in failures if not f["injected_fault"]]

    return {
        "campaign": resolved["campaign"],
        "n_runs_total": len(statuses),
        "n_completed": len(by_state["completed"]),
        "n_failed": len(by_state["failed"]),
        "n_failed_injected": sum(1 for f in failures if f["injected_fault"]),
        "n_failed_real": len(real_failures),
        # the campaign is "complete" only if no *unexpected* run failed; the
        # injected faults are expected to fail and are still surfaced below.
        "campaign_complete": len(real_failures) == 0,
        "failures_surfaced": failures,
        "sweep_points": points,
        "intents": sorted({p["intent"] for p in points}),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_campaign(manifest_path, output, overwrite=False, only=None,
                 logger=print):
    """Run (or restart) the campaign.

    ``only`` (an int index into the enumerated run list) executes just that one
    cell -- the SLURM job-array entry point (one array task per cell).  The
    final aggregate pass is a plain call with ``only=None``: it re-enumerates
    every cell, loads each already-written ``status.json`` (completed OR
    failed), and aggregates the whole grid -- so a task that crashed on the
    cluster is surfaced, never silently dropped.
    """
    resolved = resolve_manifest(manifest_path)
    runs_dir = os.path.join(output, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    # freeze the immutable resolved manifest (the re-runnable record)
    with open(os.path.join(output, "campaign.resolved.yaml"), "w") as fh:
        yaml.safe_dump(resolved, fh, default_flow_style=False, sort_keys=True)

    all_specs = enumerate_runs(resolved)
    if only is not None:
        if not (0 <= only < len(all_specs)):
            raise IndexError(f"--only {only} out of range "
                             f"[0, {len(all_specs)})")
        specs = [all_specs[only]]
        logger(f"campaign '{resolved['campaign']}': job-array cell {only}/"
               f"{len(all_specs)} -> {specs[0]['run_id']}")
    else:
        specs = all_specs
        logger(f"campaign '{resolved['campaign']}': {len(specs)} runs "
               f"({len(resolved['sweeps'])} sweep(s) + "
               f"{len(resolved['deliberate_failures'])} injected fault(s))")

    with RunProvenance(output, {"campaign": resolved["campaign"],
                                "solver": resolved["base"]["solver"],
                                "n_runs": len(specs)}) as prov:
        statuses = []
        for spec in specs:
            out_dir = os.path.join(runs_dir, spec["run_id"])
            done = os.path.join(out_dir, "status.json")
            # checkpoint/restart: skip already-completed runs; retry failed
            if os.path.exists(done) and not overwrite:
                with open(done) as fh:
                    prev = json.load(fh)
                if prev.get("state") == "completed":
                    logger(f"  [skip] {spec['run_id']} (completed)")
                    statuses.append(prev)
                    continue
                logger(f"  [retry] {spec['run_id']} "
                       f"(was {prev.get('state')})")
            st = run_one(spec, resolved["base"], out_dir)
            tag = "OK" if st["state"] == "completed" else "FAIL"
            extra = (f" obs={st['results']['value']:.4f}"
                     if st["state"] == "completed"
                     else f" <- {st.get('reason')}")
            logger(f"  [{tag}] {spec['run_id']}{extra}")
            statuses.append(st)
        prov.set_exit("completed")

    summary = aggregate(resolved, statuses)
    with open(os.path.join(output, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    return summary


def _print_report(summary, logger=print):
    logger("\n=== campaign aggregate ===")
    logger(f"runs: {summary['n_runs_total']} total, "
           f"{summary['n_completed']} completed, "
           f"{summary['n_failed']} failed "
           f"({summary['n_failed_real']} real, "
           f"{summary['n_failed_injected']} injected).")
    logger(f"campaign_complete (no unexpected failure): "
           f"{summary['campaign_complete']}")
    logger("\nsweep points (mean +/- sd [95% CI], seeds used/expected):")
    for p in summary["sweep_points"]:
        flag = "" if p["ensemble_complete"] else "  <-- PARTIAL ENSEMBLE"
        logger(f"  [{p['intent']:>13}] {p['param']}={p['value']:g}: "
               f"{p['mean']:.4f} +/- {p['sd']:.4f} "
               f"[{p['ci_low']:.4f}, {p['ci_high']:.4f}]  "
               f"n={p['n_seeds_used']}/{p['n_seeds_expected']}{flag}")
    logger("\nfailures SURFACED (never silently dropped):")
    if not summary["failures_surfaced"]:
        logger("  (none)")
    for f in summary["failures_surfaced"]:
        kind = "injected" if f["injected_fault"] else "REAL"
        logger(f"  [{kind}/{f['failure_class']}] {f['run_id']}: "
               f"{f['reason']}")


def main():
    ap = argparse.ArgumentParser(description="C9 reproducible-campaign runner")
    ap.add_argument("--manifest", default=os.path.join(_HERE, "manifest.yaml"))
    ap.add_argument("--output", default=os.path.join(_HERE, "outputs", "demo"))
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run every cell (ignore checkpoints)")
    ap.add_argument("--only", type=int, default=None,
                    help="run just the k-th enumerated cell (SLURM job array)")
    args = ap.parse_args()
    summary = run_campaign(args.manifest, args.output, overwrite=args.overwrite,
                           only=args.only)
    _print_report(summary)
    # exit non-zero if a REAL (non-injected) run failed -- CI gate
    sys.exit(0 if summary["campaign_complete"] else 3)


if __name__ == "__main__":
    main()
