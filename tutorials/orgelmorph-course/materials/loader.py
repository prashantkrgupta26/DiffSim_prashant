"""Validated materials loader for the OrgElMorph course.

Two provenance-carrying YAML databases feed the tutorials, and NEITHER is
ever hand-copied into code:

* ``materials/materials.yaml`` (schema v2) -- the full material database.
  Every parameter is a mapping ``{value, units, definition,
  nondimensionalization, status, source, uncertainty, validity, notes}``.
  Loaded with :func:`load_system` / :func:`load_crystallization`.

* ``materials/ternary_p4.yaml`` -- the lighter P4 ternary blends
  (``{value, units, definition, source, uncertainty}`` per parameter),
  loaded with the original :func:`load_blend` API (kept for P4).

The schema-v2 loader enforces an HONESTY rule: a parameter marked
``status: production`` MUST carry a non-empty ``units`` and ``definition``
and a non-null ``value`` -- a measured/fitted number with no units or no
definition is rejected loudly. A parameter marked ``status:
accelerated_tutorial`` MAY omit them (and may have a null value, e.g. an
unknown chi to be recovered), but it must SAY ``accelerated_tutorial``
explicitly -- an accelerated value can never be silently passed off as a
production one. Missing ``status`` defaults to ``production`` (strict).

    from loader import load_system, load_crystallization, load_blend
    sys = load_system("drying_demo_ternary")
    sys.value("chi")                      # [1.5, 0.3, 0.3]
    sys.record()                          # JSON-serializable provenance dict
    cry = load_crystallization("PCBM_class")
    cry.value("Tm")                       # 558.0
    b = load_blend("synthetic_demix")     # P4 ternary blend (ternary_p4.yaml)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "ternary_p4.yaml")
MATERIALS_YAML = os.path.join(os.path.dirname(__file__), "materials.yaml")

_CHI_KEYS = ("chi_12", "chi_1s", "chi_2s")

_STATUSES = ("production", "accelerated_tutorial")
_SOURCE_TYPES = ("measured", "fitted", "literature", "assumed",
                 "accelerated_tutorial", "unknown")
_UNCERTAINTY_TYPES = ("stddev", "order_of_magnitude", "range", "none",
                      "unknown")


class MaterialError(ValueError):
    """Invalid or missing material parameter."""


# ===========================================================================
# schema-v2 loader (materials.yaml): full per-parameter provenance
# ===========================================================================
@dataclass
class Parameter:
    """One validated parameter with its full provenance."""
    name: str
    value: Any
    units: Optional[str] = None
    definition: Optional[str] = None
    nondimensionalization: Optional[str] = None
    status: str = "production"
    source: dict = field(default_factory=dict)
    uncertainty: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)
    notes: str = ""

    def record(self):
        """A JSON-serializable provenance record for this parameter."""
        return {
            "value": self.value, "units": self.units,
            "definition": self.definition,
            "nondimensionalization": self.nondimensionalization,
            "status": self.status, "source": self.source,
            "uncertainty": self.uncertainty, "validity": self.validity,
            "notes": self.notes,
        }


@dataclass
class MaterialSystem:
    """A validated material system: named parameters + description/species."""
    name: str
    description: str
    species: list
    params: dict          # {param_name: Parameter}
    section: str          # "systems" or "crystallization"
    path: str

    def param(self, key) -> Parameter:
        if key not in self.params:
            raise MaterialError(
                f"system '{self.name}' has no parameter '{key}'; "
                f"available: {sorted(self.params)}")
        return self.params[key]

    def value(self, key):
        """The bare value of parameter ``key`` (may be a number or list)."""
        return self.param(key).value

    def record(self):
        """JSON-serializable provenance for the whole system (for run output).

        This is what a run.py drops into its output directory so the exact
        resolved parameter set -- values AND provenance -- is archived with
        the results (works alongside common/provenance metadata.json).
        """
        return {
            "name": self.name, "section": self.section,
            "description": self.description.strip(), "species": self.species,
            "source_file": os.path.basename(self.path),
            "parameters": {k: p.record() for k, p in self.params.items()},
        }


def _require_str(entry, key, pname, sysname):
    v = entry.get(key)
    if not isinstance(v, str) or not v.strip():
        raise MaterialError(
            f"{sysname}.{pname}: '{key}' must be a non-empty string for a "
            f"production parameter (got {v!r})")
    return v.strip()


def _parse_parameter(pname, entry, sysname):
    """Validate one raw parameter mapping into a :class:`Parameter`."""
    if not isinstance(entry, dict):
        raise MaterialError(
            f"{sysname}.{pname} must be a mapping with a 'value' key, "
            f"got {type(entry).__name__}")
    if "value" not in entry:
        raise MaterialError(f"{sysname}.{pname} is missing required 'value'")

    status = entry.get("status", "production")
    if status not in _STATUSES:
        raise MaterialError(
            f"{sysname}.{pname}: status {status!r} not in {list(_STATUSES)}")

    source = entry.get("source", {}) or {}
    if not isinstance(source, dict):
        raise MaterialError(f"{sysname}.{pname}: 'source' must be a mapping")
    stype = source.get("type")
    if stype is not None and stype not in _SOURCE_TYPES:
        raise MaterialError(
            f"{sysname}.{pname}: source.type {stype!r} not in "
            f"{list(_SOURCE_TYPES)}")

    unc = entry.get("uncertainty", {}) or {}
    if not isinstance(unc, dict):
        raise MaterialError(f"{sysname}.{pname}: 'uncertainty' must be a mapping")
    utype = unc.get("type")
    if utype is not None and utype not in _UNCERTAINTY_TYPES:
        raise MaterialError(
            f"{sysname}.{pname}: uncertainty.type {utype!r} not in "
            f"{list(_UNCERTAINTY_TYPES)}")

    value = entry["value"]
    if status == "production":
        # honesty rule: a production value needs units + definition + a value
        _require_str(entry, "units", pname, sysname)
        _require_str(entry, "definition", pname, sysname)
        if value is None:
            raise MaterialError(
                f"{sysname}.{pname}: production parameter cannot have a null "
                f"value -- mark it status: accelerated_tutorial if it is a "
                f"placeholder/unknown")

    return Parameter(
        name=pname, value=value,
        units=entry.get("units"), definition=entry.get("definition"),
        nondimensionalization=entry.get("nondimensionalization"),
        status=status, source=source, uncertainty=unc,
        validity=entry.get("validity", {}) or {},
        notes=str(entry.get("notes", "")).strip())


def _load_db(path):
    path = path or MATERIALS_YAML
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise MaterialError(f"{path}: top level must be a mapping")
    return path, data


def load_system(name, path=None, section="systems"):
    """Load and validate one system from ``materials.yaml`` (schema v2).

    Parameters
    ----------
    name : str
        Key under ``systems:`` (or ``crystallization:``) in the YAML.
    path : str, optional
        YAML path (defaults to ``materials/materials.yaml``).
    section : str
        Top-level section to read from (``systems`` or ``crystallization``).

    Returns
    -------
    MaterialSystem
    """
    path, data = _load_db(path)
    block = data.get(section, {}) or {}
    if name not in block:
        raise MaterialError(
            f"system '{name}' not in {section} of {path}; "
            f"available: {sorted(block)}")
    entry = block[name]
    raw_params = entry.get("parameters", {}) or {}
    if not raw_params:
        raise MaterialError(
            f"system '{name}' in {path} has no 'parameters' block")
    params = {k: _parse_parameter(k, v, name) for k, v in raw_params.items()}
    return MaterialSystem(
        name=name, description=str(entry.get("description", "")),
        species=list(entry.get("species", []) or []),
        params=params, section=section, path=path)


def load_crystallization(name, path=None):
    """Load a crystallization energetics set (``crystallization:`` section)."""
    return load_system(name, path=path, section="crystallization")


def available_systems(path=None, section="systems"):
    """List the system names in a section of ``materials.yaml``."""
    _, data = _load_db(path)
    return sorted((data.get(section, {}) or {}).keys())


def save_resolved_materials(systems, out_path):
    """Archive resolved material provenance next to a run's results.

    ``systems`` is one :class:`MaterialSystem` or a list of them. Writes a
    JSON file (default ``materials.resolved.json``) recording every value and
    its provenance, so the run output is self-describing (pairs with
    ``common/provenance`` metadata.json).
    """
    if isinstance(systems, MaterialSystem):
        systems = [systems]
    payload = {"materials": [s.record() for s in systems]}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
    return out_path


# ===========================================================================
# P4 ternary blends (ternary_p4.yaml): original API, kept for chapter P4
# ===========================================================================
@dataclass
class TernaryBlend:
    """A validated ternary blend: chi triple + N triple + provenance."""
    name: str
    chi: tuple           # (chi_12, chi_1s, chi_2s)
    N: tuple             # (N1, N2, Ns)
    description: str
    provenance: dict     # per-parameter {source, uncertainty, units, ...}

    @property
    def chi12(self):
        return self.chi[0]

    @property
    def chi1s(self):
        return self.chi[1]

    @property
    def chi2s(self):
        return self.chi[2]


def _num(entry, name):
    """Pull a validated numeric ``value`` out of a schema entry mapping."""
    if not isinstance(entry, dict) or "value" not in entry:
        raise MaterialError(
            f"parameter '{name}' must be a mapping with a 'value' key "
            f"(got {entry!r})")
    v = entry["value"]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise MaterialError(f"parameter '{name}': value must be a number, "
                            f"got {type(v).__name__} ({v!r})")
    return float(v)


def load_blend(name, path=None):
    """Load and validate one blend from the ternary materials YAML.

    Parameters
    ----------
    name : str
        Key under ``blends:`` in the YAML (e.g. ``synthetic_demix``).
    path : str, optional
        YAML path (defaults to ``materials/ternary_p4.yaml``).

    Returns
    -------
    TernaryBlend
    """
    path = path or DEFAULT_YAML
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    blends = data.get("blends", {})
    if name not in blends:
        raise MaterialError(
            f"blend '{name}' not in {path}; available: {sorted(blends)}")
    b = blends[name]

    chi = tuple(_num(b[k], k) for k in _CHI_KEYS if _require(b, k, name))
    for k, c in zip(_CHI_KEYS, chi):
        if not (-2.0 <= c <= 20.0):
            raise MaterialError(f"{name}.{k} = {c} outside plausible "
                                f"Flory range [-2, 20]")

    _require(b, "N", name)
    N_entry = b["N"]
    if not isinstance(N_entry, dict) or "value" not in N_entry:
        raise MaterialError(f"{name}.N must be a mapping with a 'value' list")
    Nv = N_entry["value"]
    if (not isinstance(Nv, (list, tuple)) or len(Nv) != 3
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or x <= 0 for x in Nv)):
        raise MaterialError(f"{name}.N.value must be 3 positive numbers, "
                            f"got {Nv!r}")
    N = tuple(float(x) for x in Nv)

    prov = {k: {kk: vv for kk, vv in b[k].items() if kk != "value"}
            for k in (*_CHI_KEYS, "N")}
    return TernaryBlend(name=name, chi=chi, N=N,
                        description=str(b.get("description", "")).strip(),
                        provenance=prov)


def _require(mapping, key, name):
    if key not in mapping:
        raise MaterialError(f"blend '{name}' missing required key '{key}'")
    return True


def available_blends(path=None):
    """List the blend names in the ternary YAML."""
    path = path or DEFAULT_YAML
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return sorted((data.get("blends", {})).keys())


__all__ = [
    "MaterialError", "Parameter", "MaterialSystem",
    "load_system", "load_crystallization", "available_systems",
    "save_resolved_materials",
    "TernaryBlend", "load_blend", "available_blends",
]
