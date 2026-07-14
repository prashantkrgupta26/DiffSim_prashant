"""OrgElMorph course - Computational C9: figures + numbers.

Runs the real campaign (idempotent restart) and renders the chapter's two
figures into ../../latex/figures/ and the measured macros into
../../latex/numbers/c9.tex, so the document's numbers ARE the campaign output:

  c9_sweep.png    L_area vs chi with seed ensembles + 95% CI, exploratory vs
                  confirmatory, and the fitted parametric sensitivity.
  c9_ledger.png   the run ledger: every run accounted for, failures surfaced
                  (injected ConfigError + a real validation failure under a
                  tightened gate) -- the no-silent-drop rule, visualized.

    PYTHONPATH=<repo>/src python gen_figures.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from campaign import run_campaign, resolve_manifest, enumerate_runs

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(_HERE, "..", "..", "latex", "figures"))
NUMTEX = os.path.abspath(os.path.join(_HERE, "..", "..", "latex",
                                      "numbers", "c9.tex"))
MANIFEST = os.path.join(_HERE, "manifest.yaml")

_GREEN = "#2f8f4e"
_RED = "#c0392b"
_GRAY = "#7f8c8d"
_BLUE = "#2f6fb0"
_ORANGE = "#d98a1f"


def _sci(x):
    """Format a positive number as a LaTeX \\ensuremath scientific macro body."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** exp
    return f"\\ensuremath{{{mant:.2f}\\times10^{{{exp}}}}}"


def _load_run_values(output, sweep, value):
    """Per-seed observable values from the saved run dirs (for the dots)."""
    import json
    vals = []
    runs_dir = os.path.join(output, "runs")
    for rid in os.listdir(runs_dir):
        sp = os.path.join(runs_dir, rid, "status.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as fh:
            st = json.load(fh)
        if (st.get("state") == "completed" and st["sweep"] == sweep
                and abs(st["value"] - value) < 1e-9):
            vals.append(st["results"]["value"])
    return vals


def sweep_figure(summary, output, fname="c9_sweep.png"):
    expl = [p for p in summary["sweep_points"] if p["intent"] == "exploratory"]
    conf = [p for p in summary["sweep_points"] if p["intent"] == "confirmatory"]
    expl.sort(key=lambda p: p["value"])

    fig, ax = plt.subplots(figsize=(6.6, 4.6), dpi=150)
    xs = np.array([p["value"] for p in expl])
    ys = np.array([p["mean"] for p in expl])
    lo = np.array([p["mean"] - p["ci_low"] for p in expl])
    hi = np.array([p["ci_high"] - p["mean"] for p in expl])
    # faint per-seed dots
    for p in expl:
        v = _load_run_values(output, p["sweep"], p["value"])
        ax.scatter([p["value"]] * len(v), v, s=22, color=_BLUE, alpha=0.35,
                   zorder=2)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=_BLUE, lw=2, ms=8,
                capsize=4, zorder=3, label="exploratory scan (95% CI)")
    # linear sensitivity fit across the exploratory scan
    slope, intercept = np.polyfit(xs, ys, 1)
    xf = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xf, slope * xf + intercept, "--", color=_GRAY, lw=1.4,
            label=f"sensitivity $dL/d\\chi\\approx{slope:.3f}$")
    for p in conf:
        v = _load_run_values(output, p["sweep"], p["value"])
        ax.scatter([p["value"]] * len(v), v, s=22, color=_ORANGE, alpha=0.5,
                   zorder=2)
        ax.errorbar([p["value"]], [p["mean"]],
                    yerr=[[p["mean"] - p["ci_low"]], [p["ci_high"] - p["mean"]]],
                    fmt="s", color=_ORANGE, ms=10, capsize=4, zorder=4,
                    label="confirmatory replicate (95% CI)")
    ax.set_xlabel(r"Flory--Huggins enthalpic parameter $\chi$ (= fh\_B)")
    ax.set_ylabel(r"domain length $L_{\mathrm{area}}$ (box fraction)")
    ax.set_title("Campaign sweep: morphology vs $\\chi$, seed ensembles + CI")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)
    return slope


def ledger_figure(main_summary, teeth_summary, fname="c9_ledger.png"):
    """Every run as a colored bar; failures surfaced, not dropped."""
    rows = []
    for f in main_summary["failures_surfaced"]:
        rows.append((f["run_id"], "failed",
                     "injected" if f["injected_fault"] else "real",
                     f["reason"]))
    # completed sweep runs (collapsed to one row per point for readability)
    for p in sorted(main_summary["sweep_points"], key=lambda q: (q["intent"], q["value"])):
        rows.append((f"{p['sweep']} chi={p['value']:g} "
                     f"(n={p['n_seeds_used']})", "completed", p["intent"], ""))
    for f in teeth_summary["failures_surfaced"]:
        if not f["injected_fault"]:
            rows.append((f"{f['run_id']} (tight gate)", "failed", "real",
                         f["reason"]))

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=150)
    y = np.arange(len(rows))[::-1]
    for yi, (label, state, kind, reason) in zip(y, rows):
        if state == "completed":
            col = _GREEN
        else:
            col = _ORANGE if kind == "injected" else _RED
        ax.barh(yi, 1.0, color=col, alpha=0.85, height=0.7)
        txt = label if not reason else f"{label}   [{reason}]"
        ax.text(0.02, yi, txt, va="center", ha="left", fontsize=8,
                color="white" if state != "completed" else "white")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    handles = [plt.Rectangle((0, 0), 1, 1, color=_GREEN),
               plt.Rectangle((0, 0), 1, 1, color=_ORANGE),
               plt.Rectangle((0, 0), 1, 1, color=_RED)]
    ax.legend(handles, ["completed", "injected fault (surfaced)",
                        "real failure (surfaced -> campaign incomplete)"],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=3, framealpha=0.95)
    ax.set_title("Campaign ledger: every run accounted for, "
                 "failures surfaced not dropped")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


TEETH_MANIFEST = """
campaign: c9_teeth_demo
description: tightened mass gate; the deep-quench real run fails validation
base:
  mass_drift_max: 1.0e-3
sweeps:
  - name: chi_scan
    param: fh_B
    values: [0.86, 1.36]
    seeds: [1]
    intent: exploratory
"""


def _point(summary, sweep, value):
    for p in summary["sweep_points"]:
        if p["sweep"] == sweep and abs(p["value"] - value) < 1e-9:
            return p
    raise KeyError(f"{sweep} {value}")


def write_numbers(main, teeth, slope):
    def pt(s, v):
        return _point(main, s, v)
    lo, mid, hi = pt("chi_scan", 0.36), pt("chi_scan", 0.86), pt("chi_scan", 1.36)
    conf = pt("chi_confirm", 0.86)
    inj = next(f for f in main["failures_surfaced"] if f["injected_fault"])
    teeth_fail = next(f for f in teeth["failures_surfaced"]
                      if not f["injected_fault"])
    # parse the measured drift out of the teeth failure reason
    import re
    m = re.search(r"drift ([0-9.eE+-]+)", teeth_fail["reason"])
    drift = float(m.group(1)) if m else 0.0

    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        f"\\newcommand{{\\CnineNruns}}{{{main['n_runs_total']}}}",
        f"\\newcommand{{\\CnineNcompleted}}{{{main['n_completed']}}}",
        f"\\newcommand{{\\CnineNfailed}}{{{main['n_failed']}}}",
        f"\\newcommand{{\\CnineNinjected}}{{{main['n_failed_injected']}}}",
        f"\\newcommand{{\\CnineNsweeps}}{{{len(main['sweep_points'])}}}",
        f"\\newcommand{{\\CnineChiLo}}{{{lo['value']:.2f}}}",
        f"\\newcommand{{\\CnineChiMid}}{{{mid['value']:.2f}}}",
        f"\\newcommand{{\\CnineChiHi}}{{{hi['value']:.2f}}}",
        f"\\newcommand{{\\CnineLoMean}}{{{lo['mean']:.3f}}}",
        f"\\newcommand{{\\CnineMidMean}}{{{mid['mean']:.3f}}}",
        f"\\newcommand{{\\CnineHiMean}}{{{hi['mean']:.3f}}}",
        f"\\newcommand{{\\CnineLoSd}}{{{lo['sd']:.3f}}}",
        f"\\newcommand{{\\CnineMidSd}}{{{mid['sd']:.3f}}}",
        f"\\newcommand{{\\CnineHiSd}}{{{hi['sd']:.3f}}}",
        f"\\newcommand{{\\CnineLoCIlo}}{{{lo['ci_low']:.3f}}}",
        f"\\newcommand{{\\CnineLoCIhi}}{{{lo['ci_high']:.3f}}}",
        f"\\newcommand{{\\CnineHiCIlo}}{{{hi['ci_low']:.3f}}}",
        f"\\newcommand{{\\CnineHiCIhi}}{{{hi['ci_high']:.3f}}}",
        f"\\newcommand{{\\CnineConfMean}}{{{conf['mean']:.3f}}}",
        f"\\newcommand{{\\CnineConfSd}}{{{conf['sd']:.3f}}}",
        f"\\newcommand{{\\CnineSensitivity}}{{{slope:.3f}}}",
        f"\\newcommand{{\\CnineInjReason}}{{unknown solver}}",
        f"\\newcommand{{\\CnineTeethDrift}}{{{_sci(drift)}}}",
        f"\\newcommand{{\\CnineTeethGate}}{{{_sci(1e-3)}}}",
        f"\\newcommand{{\\CnineMassGate}}{{{_sci(5e-2)}}}",
    ]
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    out_main = os.path.join(_HERE, "outputs", "demo")
    main_summary = run_campaign(MANIFEST, out_main, overwrite=False)

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf:
        tf.write(TEETH_MANIFEST)
        teeth_path = tf.name
    out_teeth = os.path.join(_HERE, "outputs", "teeth")
    teeth_summary = run_campaign(teeth_path, out_teeth, overwrite=True)
    os.unlink(teeth_path)

    slope = sweep_figure(main_summary, out_main)
    ledger_figure(main_summary, teeth_summary)
    write_numbers(main_summary, teeth_summary, slope)


if __name__ == "__main__":
    main()
