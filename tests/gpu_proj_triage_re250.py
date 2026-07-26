"""
Projection-leg triage for the corrected Re=250 2-D thin-plate config on A100.

NOT a pytest file — run directly:
    python tests/gpu_proj_triage_re250.py

Requires CUDA; asserts at startup.

Protocol:
  P0:  corrected config, driver defaults, 1000 steps.
       bounded = max|Cd| < 1e3
  If P0 bounded → P0-long: 8000 steps; report Cd_mean(t>=1.5), St, SHEDDING flag.
  If P0 diverges → knob probes (a..e), 500 steps each, table of bounded/diverged.
  Prints PROJ-TRIAGE-OK + summary at the end.
"""
import sys
import time
import numpy as np
import torch

# ── CUDA guard ────────────────────────────────────────────────────────────────
assert torch.cuda.is_available(), (
    "CUDA not available — this script must run on a GPU node (cuda:0)."
)
print(f"[triage] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

# ── local imports ─────────────────────────────────────────────────────────────
sys.path.insert(0, "src")          # works from /work/mech-ai/baskarg/DiffSim
from tests.p2r1a_thin_plate_flow import run_flow_past_projection
from diffsim.postproc.shedding import strouhal

# ── corrected Re=250 config ───────────────────────────────────────────────────
CORRECTED = dict(
    level=7,
    refine_to=9,
    wake_refine=9,
    nu=2.5e-4,
    U_inf=1.0,
    plate_xc=0.3125,
    plate_yc=0.5,
    plate_L=0.0625,
    dt=5e-4,
    pert_eps=0.03,
    pert_t_end=0.5,
    predictor_solver="cudss",
    ppe_solver="gpu_cg",
    device="cuda:0",
    device_assembly=True,
    verbose=True,
)


def _run(label, nsteps, extra=None):
    """Run projection with CORRECTED config + optional overrides; return metrics."""
    cfg = dict(**CORRECTED, nsteps=nsteps)
    if extra:
        cfg.update(extra)
    print(f"\n{'='*72}", flush=True)
    print(f"[triage] RUN {label}  nsteps={nsteps}  overrides={extra}", flush=True)
    print(f"{'='*72}", flush=True)
    t0 = time.time()
    try:
        res = run_flow_past_projection(**cfg)
    except Exception as exc:
        print(f"[triage] {label} CRASHED: {exc}", flush=True)
        return dict(label=label, status="CRASH", max_cd=np.inf, cd_last=np.nan,
                    cd_mean=np.nan, st=np.nan, cl_std=np.nan)
    elapsed = time.time() - t0
    cd = res["cd"]
    cl = res["cl"]
    max_cd = float(np.max(np.abs(cd)))
    cd_last = float(cd[-1])
    bounded = max_cd < 1e3
    status = "bounded" if bounded else "DIVERGED"
    print(f"[triage] {label} → {status}  max|Cd|={max_cd:.4g}  "
          f"Cd[-1]={cd_last:+.4f}  wall={elapsed:.1f}s", flush=True)
    row = dict(label=label, status=status, max_cd=max_cd, cd_last=cd_last,
               cd_mean=np.nan, st=np.nan, cl_std=np.nan, nsteps=nsteps)
    if bounded:
        # Post-transient tail: t >= 1.5 (same convention as monolithic validation)
        dt = cfg["dt"]
        t_arr = (np.arange(nsteps) + 1) * dt
        mask = t_arr >= 1.5
        if mask.sum() >= 4:
            row["cd_mean"] = float(np.mean(cd[mask]))
            row["cl_std"] = float(np.std(cl[mask]))
            try:
                st_val, freq = strouhal(t_arr, cl, cfg["U_inf"], cfg["plate_L"])
                row["st"] = float(st_val)
            except Exception as e:
                print(f"[triage] strouhal failed: {e}", flush=True)
        else:
            print(f"[triage] {label}: only {mask.sum()} steps at t>=1.5 "
                  "(need 4+); skipping Cd_mean/St", flush=True)
    return row


def _print_table(rows):
    hdr = f"{'Label':<28} {'Status':<10} {'max|Cd|':>12} {'Cd[-1]':>10} {'Cd_mean':>10} {'Cl_std':>10} {'St':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['label']:<28} {r['status']:<10} "
              f"{r['max_cd']:>12.4g} {r['cd_last']:>10.4f} "
              f"{r.get('cd_mean', float('nan')):>10.4f} "
              f"{r.get('cl_std', float('nan')):>10.4g} "
              f"{r.get('st', float('nan')):>8.4f}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# P0: corrected config, 1000 steps
# ─────────────────────────────────────────────────────────────────────────────
p0 = _run("P0-corrected-1k", nsteps=1000)
rows = [p0]

if p0["status"] == "bounded":
    # ── P0-long: 8000 steps ───────────────────────────────────────────────────
    p0l = _run("P0-long-8k", nsteps=8000)
    rows.append(p0l)
    SHEDDING = p0l.get("cl_std", 0.0) > 1e-3
    print(f"\n[triage] P0-LONG SHEDDING flag = {SHEDDING}  "
          f"(Cl_std={p0l.get('cl_std', float('nan')):.4g})", flush=True)
    print(f"[triage] Cd_mean={p0l.get('cd_mean', float('nan')):.4f}  "
          f"St={p0l.get('st', float('nan')):.4f}", flush=True)

else:
    # ── P0 diverged → knob probes (a..e), 500 steps each ────────────────────
    print("\n[triage] P0 DIVERGED — running knob probes (500 steps each)", flush=True)

    probes = [
        ("(a)-inner_relax0.3",     {"inner_relax": 0.3}),
        ("(b)-inner_max25",        {"inner_max": 25}),
        # (c) beta_backflow: it IS wired into the stepper as beta_backflow=1.0 inside
        #     run_flow_past_projection but is NOT surfaced as a kwarg — SKIP per spec.
        ("(d)-dt2.5e-4",           {"dt": 2.5e-4}),
        ("(e)-no_consistent_proj", {"consistent_projection": False}),
    ]
    print("[triage] NOTE: (c) beta_backflow is NOT surfaced as a kwarg of "
          "run_flow_past_projection (hard-wired to 1.0 inside) — SKIPPED per protocol.",
          flush=True)

    for label, extra in probes:
        row = _run(label, nsteps=500, extra=extra)
        rows.append(row)

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72, flush=True)
print("PROJ-TRIAGE-OK", flush=True)
print("=" * 72, flush=True)
_print_table(rows)

# One-liner headline
p0_verdict = p0["status"]
if p0["status"] == "bounded" and len(rows) >= 2:
    pl = rows[1]
    shed = pl.get("cl_std", 0.0) > 1e-3
    print(f"HEADLINE: P0 {p0_verdict} → long run: "
          f"Cd_mean={pl.get('cd_mean', float('nan')):.4f}  "
          f"St={pl.get('st', float('nan')):.4f}  "
          f"SHEDDING={shed}", flush=True)
else:
    print(f"HEADLINE: P0 {p0_verdict} — see knob-probe table above.", flush=True)

print("=" * 72, flush=True)
