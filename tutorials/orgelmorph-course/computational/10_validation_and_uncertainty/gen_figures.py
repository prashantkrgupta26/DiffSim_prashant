"""OrgElMorph course - Computational C10: figures + numbers.

Runs the real uncertainty-budget campaign (uncertainty.build_budget) once,
persists it to outputs/budget.json, and renders the chapter's two figures into
../../latex/figures/ plus the measured macros into ../../latex/numbers/c10.tex,
so the document's numbers ARE the budget's measured output:

  c10_budget.png        the uncertainty budget: sigma by source + total, which
                        source dominates the morphology observable.
  c10_verification.png  left: solution verification (L_area vs mesh level);
                        right: the spread decomposition (param/stoch/num error
                        bars around the nominal observable).

    PYTHONPATH=<repo>/src python gen_figures.py
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uncertainty import build_budget

_HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.abspath(os.path.join(_HERE, "..", "..", "latex", "figures"))
NUMTEX = os.path.abspath(os.path.join(_HERE, "..", "..", "latex",
                                      "numbers", "c10.tex"))
OUTJSON = os.path.join(_HERE, "outputs", "budget.json")

_PARAM = "#c0392b"
_STOCH = "#2f6fb0"
_NUM = "#d98a1f"
_TOTAL = "#2f8f4e"
_COLORS = {"parameter (chi)": _PARAM, "stochastic (seed)": _STOCH,
           "numerical (mesh)": _NUM}


def _sci(x):
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** exp
    return f"\\ensuremath{{{mant:.2f}\\times10^{{{exp}}}}}"


def budget_figure(b, fname="c10_budget.png"):
    contrib = b["budget"]["contributions"]
    labels = [c["source"] for c in contrib] + ["combined (total)"]
    sig = [c["sigma"] for c in contrib] + [b["budget"]["sigma_total"]]
    cols = [_COLORS[c["source"]] for c in contrib] + [_TOTAL]
    pct = [f"{c['pct_variance']:.0f}% var" for c in contrib] + ["quadrature"]

    fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=150)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, sig, color=cols, alpha=0.88, height=0.62)
    for yi, s, p in zip(y, sig, pct):
        ax.text(s + max(sig) * 0.015, yi, f"{s:.4f}  ({p})", va="center",
                fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, max(sig) * 1.28)
    ax.set_xlabel(r"uncertainty $\sigma$ in $L_{\mathrm{area}}$ (box fraction)")
    ax.set_title(f"Uncertainty budget: {b['material'].replace('_', ':')} "
                 f"/ $L_{{\\mathrm{{area}}}}$\n"
                 f"dominant source: {b['budget']['dominant_source']} "
                 f"({b['budget']['dominance_ratio']:.1f}$\\times$ next)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def verification_figure(b, fname="c10_verification.png"):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.6, 4.2), dpi=150)

    # left: solution verification -- L_area vs mesh level
    sv = b["solution_verification"]
    lv = sorted(sv["L_by_level"])
    Ls = [sv["L_by_level"][k] for k in lv]
    sides = [2 ** k for k in lv]
    axL.plot(sides, Ls, "o-", color=_NUM, lw=2, ms=8)
    axL.axhline(sv["L_finest"], color=_NUM, ls=":", alpha=0.6,
                label=f"finest ({sides[-1]}$^2$): {sv['L_finest']:.3f}")
    axL.annotate(f"discretization\nuncertainty {sv['discretization_uncertainty']:.4f}",
                 xy=(sides[-1], Ls[-1]), xytext=(sides[0] * 1.1, Ls[-1]),
                 fontsize=8, va="center",
                 arrowprops=dict(arrowstyle="->", color=_NUM, alpha=0.7))
    axL.set_xscale("log", base=2)
    axL.set_xlabel("mesh: cells per side")
    axL.set_ylabel(r"$L_{\mathrm{area}}$ (box fraction)")
    axL.set_title("Solution verification: mesh refinement")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend(fontsize=8)

    # right: spread decomposition -- error bars around the nominal observable
    Lnom = b["L_nominal"]
    sources = b["budget"]["contributions"]
    xs = np.arange(len(sources) + 1)
    sigmas = [c["sigma"] for c in sources] + [b["budget"]["sigma_total"]]
    labels = [c["source"].split()[0] for c in sources] + ["combined"]
    cols = [_COLORS[c["source"]] for c in sources] + [_TOTAL]
    for x, s, col in zip(xs, sigmas, cols):
        axR.errorbar([x], [Lnom], yerr=[s], fmt="o", color=col, ms=8,
                     capsize=6, lw=2, elinewidth=2)
    axR.axhline(Lnom, color="k", ls="--", alpha=0.4,
                label=f"nominal $L_{{\\mathrm{{area}}}}$ = {Lnom:.3f}")
    axR.set_xticks(xs)
    axR.set_xticklabels(labels)
    axR.set_ylabel(r"$L_{\mathrm{area}}$ $\pm\,\sigma$")
    axR.set_title("Spread by source around the nominal observable")
    axR.grid(True, axis="y", alpha=0.3)
    axR.legend(fontsize=8)

    fig.suptitle("Validation & uncertainty: verify the discretization, "
                 "then budget the spread")
    fig.savefig(os.path.join(FIGDIR, fname), bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote", fname)


def write_numbers(b):
    sv = b["solution_verification"]
    mv = b["model_validation"]
    pu = b["parameter_uncertainty"]
    st = b["stochastic_variability"]
    bud = b["budget"]
    pct = {c["source"]: c["pct_variance"] for c in bud["contributions"]}
    lv = sorted(sv["L_by_level"])
    dominant_word = bud["dominant_source"].split()[0]  # "parameter"

    lines = [
        "% AUTO-GENERATED by gen_figures.py - do not edit.",
        f"\\newcommand{{\\CtenMaterial}}{{{b['material'].replace('_', ':')}}}",
        f"\\newcommand{{\\CtenChiNom}}{{{b['chi_nominal']:.2f}}}",
        f"\\newcommand{{\\CtenChiSigma}}{{{b['chi_sigma']:.2f}}}",
        f"\\newcommand{{\\CtenLnominal}}{{{b['L_nominal']:.3f}}}",
        # code verification
        f"\\newcommand{{\\CtenMassDrift}}{{{_sci(b['code_verification']['mass_drift'])}}}",
        f"\\newcommand{{\\CtenLargestInc}}{{{_sci(max(b['code_verification']['largest_positive_increment'], 0.0))}}}",
        # solution verification
        f"\\newcommand{{\\CtenLcoarse}}{{{sv['L_by_level'][lv[0]]:.3f}}}",
        f"\\newcommand{{\\CtenLmid}}{{{sv['L_by_level'][lv[1]]:.3f}}}",
        f"\\newcommand{{\\CtenLfine}}{{{sv['L_by_level'][lv[-1]]:.3f}}}",
        f"\\newcommand{{\\CtenSideCoarse}}{{{2**lv[0]}}}",
        f"\\newcommand{{\\CtenSideFine}}{{{2**lv[-1]}}}",
        f"\\newcommand{{\\CtenDiscUnc}}{{{sv['discretization_uncertainty']:.4f}}}",
        # model validation
        f"\\newcommand{{\\CtenLamStar}}{{{mv['lambda_star_predicted']:.4f}}}",
        f"\\newcommand{{\\CtenLamMeas}}{{{mv['lambda_measured']:.4f}}}",
        f"\\newcommand{{\\CtenValidPct}}{{{100*mv['relative_discrepancy']:.1f}}}",
        # parameter uncertainty
        f"\\newcommand{{\\CtenDLdchi}}{{{pu['dL_dchi']:.3f}}}",
        f"\\newcommand{{\\CtenSigmaParam}}{{{pu['sigma_param']:.4f}}}",
        # stochastic
        f"\\newcommand{{\\CtenNseeds}}{{{st['n_seeds']}}}",
        f"\\newcommand{{\\CtenSigmaStoch}}{{{st['sigma_stoch']:.4f}}}",
        # budget
        f"\\newcommand{{\\CtenSigmaNum}}{{{sv['discretization_uncertainty']:.4f}}}",
        f"\\newcommand{{\\CtenSigmaTotal}}{{{bud['sigma_total']:.4f}}}",
        f"\\newcommand{{\\CtenRelTotal}}{{{100*bud['relative_total']:.1f}}}",
        f"\\newcommand{{\\CtenParamPct}}{{{pct['parameter (chi)']:.0f}}}",
        f"\\newcommand{{\\CtenStochPct}}{{{pct['stochastic (seed)']:.0f}}}",
        f"\\newcommand{{\\CtenNumPct}}{{{pct['numerical (mesh)']:.0f}}}",
        f"\\newcommand{{\\CtenDominant}}{{{dominant_word}}}",
        f"\\newcommand{{\\CtenDomRatio}}{{{bud['dominance_ratio']:.1f}}}",
    ]
    with open(NUMTEX, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", NUMTEX)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTJSON), exist_ok=True)
    b = build_budget()
    with open(OUTJSON, "w") as fh:
        json.dump(b, fh, indent=2, sort_keys=True, default=float)
    budget_figure(b)
    verification_figure(b)
    write_numbers(b)


if __name__ == "__main__":
    main()
