"""OrgElMorph course - Physics P10: MATERIAL-SYSTEM REPRODUCTION (capstone).

Importable core.  This is the "put it all together on a real system" capstone:
we pick ONE documented blend from the materials database, load EVERY parameter
through the validated loader (never hand-copied), reproduce a published
morphology trend, and then stress it -- mesh + time convergence and a
parameter-uncertainty sweep -- to say which conclusions are ROBUST.

THE SYSTEM.  ``PDPP5T_PCBM`` (materials.yaml) -- a diketopyrrolopyrrole-
quinquethiophene : PCBM blend spin-coated from chloroform, after the
spin-coating / drying study of Negi et al.  It is the natural choice here for
two reasons: (1) it carries a CONCRETE chi triple AND degrees of polymerization
AND an explicit spin-speed -> Biot ladder (``biot``) -- literally "the real
chi / N / Biot"; (2) at these chi it demixes robustly at tutorial scale (unlike
the archetypal P3HT:PCBM, whose weak chi_pf ~ 0.86 with a large polymer N gives
almost no lateral contrast in the tutorial horizon -- an honest negative we
document).  EVERY parameter of PDPP5T_PCBM is marked ``accelerated_tutorial`` /
``fitted`` ("representative", not a measured table value); the provenance is
surfaced verbatim so the reader knows exactly which numbers are trustworthy.

THE REPRODUCED TREND.  Negi et al. vary the SPIN SPEED, which sets the drying
RATE (higher rpm -> faster solvent removal).  The database records this as a
Biot ladder Bi(rpm).  Evaporation is the CLOCK of morphology formation
(Chapter P5): faster drying gives the blend LESS time to phase-separate before
it is quenched, so the classic expectation is a FINER frozen morphology at
higher rpm.  We reproduce the rate axis by mapping the recorded Biot ladder onto
a drying-rate ladder k_e (see :func:`ke_ladder_from_biot`) and measuring the
morphology at MATCHED DRYNESS (same mean solvent fraction phi_s for every rate,
so drying RATE is never confounded with final STATE -- the P5 correction).

THE HONEST VERDICT lives in the harness: at tutorial scale the "faster = finer"
ordering is WEAK and noisy (single seed, few lateral domains), while the
resolution-robust, uncertainty-robust conclusions are (a) morphology collapses
onto the Biot number, and (b) the DEGREE of demixing (phase contrast) is set by
DRYNESS, rises as the film dries, and is nearly rate-independent at matched
dryness.  The convergence + chi-uncertainty studies confirm which of these
survive mesh/time refinement and an order-of-magnitude change in chi.

The drying-film physics itself is NOT re-implemented here: we import the
maintained P5 brick (``physics/05_evaporation/evaporation.py``,
diffsim.physics.multiphase in film mode) so this capstone exercises the SAME
production code path the course teaches.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# reuse the maintained P5 drying-film brick (single source of truth).
_HERE = os.path.dirname(os.path.abspath(__file__))
_P5 = os.path.abspath(os.path.join(_HERE, os.pardir, "05_evaporation"))
if _P5 not in sys.path:
    sys.path.insert(0, _P5)
from evaporation import (build_mesh_dm, run_film,          # noqa: E402,F401
                         morphology_metrics)


# ---------------------------------------------------------------------------
# spin-speed Biot ladder -> runnable drying-rate ladder
# ---------------------------------------------------------------------------
def ke_ladder_from_biot(biot_dict, D_s, h0=1.0, k_e_slow=0.30, k_e_fast=0.60):
    """Map the recorded spin-speed Biot ladder onto a runnable drying-rate ladder.

    ``biot_dict`` is the material's ``biot`` parameter: a mapping of spin-speed
    label -> Biot number (e.g. ``{"s6000rpm": 0.0073, ...}``).  Two facts about
    those recorded numbers govern the mapping, and both are honoured verbatim in
    the run provenance:

    * their ABSOLUTE magnitudes are illustrative and carry an ORDER-OF-MAGNITUDE
      uncertainty (per the database), so neither the absolute Biot nor even the
      exact ratio between rungs is reproducible -- only the ORDERING (higher rpm
      = faster drying) is;
    * as recorded they are ~1e-3, far too small to dry a film within the
      tutorial horizon.

    We therefore reproduce ONLY the physically robust content -- the monotone
    spin-speed -> drying-rate ordering -- by mapping the recorded rungs, in
    rank order, LINEARLY onto a moderate, resolvable drying-rate window
    ``[k_e_slow, k_e_fast]`` (the P5 drying regime, where every rung both dries
    within the horizon and steps at reasonable cost).  The window and the
    order-preserving map are recorded so nothing is silently rescaled.

    Returns ``(rungs, window)`` where ``rungs`` is a list of dicts
    ``{"label", "rpm", "biot_recorded", "k_e", "Bi"}`` sorted by increasing rpm
    (increasing drying rate) and ``window = (k_e_slow, k_e_fast)``.
    ``Bi = k_e h0 / D_s`` is the ACTUAL Biot number of the mapped run.
    """
    items = []
    for label, bi in biot_dict.items():
        rpm = _rpm_of(label)
        items.append((rpm, label, float(bi)))
    items.sort()                                   # ascending rpm == ascending Bi
    n = len(items)
    rungs = []
    for i, (rpm, label, bi) in enumerate(items):
        frac = i / (n - 1) if n > 1 else 0.0       # rank position in [0, 1]
        k_e = float(k_e_slow) + frac * (float(k_e_fast) - float(k_e_slow))
        rungs.append({"label": label, "rpm": rpm, "biot_recorded": bi,
                      "k_e": float(k_e), "Bi": float(k_e * h0 / D_s)})
    return rungs, (float(k_e_slow), float(k_e_fast))


def _rpm_of(label):
    """Parse an integer rpm out of a spin-speed label like ``s6000rpm``."""
    digits = "".join(ch for ch in label if ch.isdigit())
    return int(digits) if digits else 0


# ---------------------------------------------------------------------------
# provenance classification (the "which numbers can I trust" table)
# ---------------------------------------------------------------------------
def classify_provenance(system):
    """Summarise each parameter's provenance for the reproducibility table.

    Returns ``{param: {status, source_type, citation, uncertainty_type,
    uncertainty_value, value_known}}`` for every parameter of the loaded
    :class:`~loader.MaterialSystem`.  ``value_known`` is ``False`` for a null
    value (an explicitly-unknown parameter such as a polymer N recorded only as
    a range).  This is the honest audit the capstone is built around: it names
    which numbers are measured, which are fitted, and which are assumed.
    """
    out = {}
    for name, p in system.params.items():
        src = p.source or {}
        unc = p.uncertainty or {}
        out[name] = {
            "status": p.status,
            "source_type": src.get("type"),
            "citation": src.get("citation"),
            "uncertainty_type": unc.get("type"),
            "uncertainty_value": unc.get("value"),
            "value_known": p.value is not None,
        }
    return out


def chi_triple(system):
    """The full-species Flory chi triple ``(chi_pf, chi_ps, chi_fs)`` of a
    loaded system, read straight from the database."""
    return (float(system.value("chi_polymer_fullerene")),
            float(system.value("chi_polymer_solvent")),
            float(system.value("chi_fullerene_solvent")))


# ---------------------------------------------------------------------------
# matched-dryness interpolation (shared with P5)
# ---------------------------------------------------------------------------
def interp_at_phis(phis, series, targets):
    """Linear-interpolate a metric ``series`` sampled along the drying
    trajectory to the MATCHED mean-solvent-fraction ``targets``.

    ``phis`` decreases along the run; a target outside the sampled range yields
    NaN (reported honestly, never extrapolated).  Mirrors P5's matched-state
    reduction so the two chapters report morphology on the same de-confounded
    axis.
    """
    phis = np.asarray(phis, float)
    series = np.asarray(series, float)
    order = np.argsort(phis)
    xs, ys = phis[order], series[order]
    lo, hi = xs.min(), xs.max()
    return {f"{t:.2f}": (float(np.interp(t, xs, ys)) if lo <= t <= hi
                         else float("nan")) for t in targets}


def film_metrics_vs_phis(rec, box_width=1.0):
    """Morphology metrics (lateral wavelength as box-fraction AND cells,
    interfacial length, phase contrast) along the WHOLE phi_s(t) trajectory of a
    :func:`run_film` record, on the polymer field."""
    wl_frac, wl_cells, iface, contrast = [], [], [], []
    for gp in rec["traj_gp"]:
        m = morphology_metrics(gp, box_width=box_width)
        wl_frac.append(m["wl_frac"]); wl_cells.append(m["wl_cells"])
        iface.append(m["iface"]); contrast.append(m["contrast"])
    return (np.array(wl_frac), np.array(wl_cells),
            np.array(iface), np.array(contrast))
