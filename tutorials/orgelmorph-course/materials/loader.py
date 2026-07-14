"""Validated materials loader for the ternary Flory-Huggins tutorial (P4).

The tutorial must never hand-copy chi/N numbers into code (spec Phase 1 P4:
"materials loader + schema, no manual copying"). This module reads a
structured, provenance-carrying YAML (``materials/ternary_p4.yaml`` by
default), validates it, and returns a small ``TernaryBlend`` with the three
Flory interaction parameters and the degree-of-polymerization triple.

Each parameter in the YAML is a mapping ``{value, units, definition, source,
uncertainty}``; the loader checks presence, type and range, so a typo'd or
out-of-range material fails loudly rather than silently poisoning a run.

    from loader import load_blend
    b = load_blend("synthetic_demix")
    b.chi            # (chi_12, chi_1s, chi_2s)
    b.N              # (N1, N2, Ns)
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

DEFAULT_YAML = os.path.join(os.path.dirname(__file__), "ternary_p4.yaml")

_CHI_KEYS = ("chi_12", "chi_1s", "chi_2s")


class MaterialError(ValueError):
    """Invalid or missing material parameter."""


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
    """List the blend names in the YAML."""
    path = path or DEFAULT_YAML
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    return sorted((data.get("blends", {})).keys())


__all__ = ["TernaryBlend", "MaterialError", "load_blend", "available_blends"]
