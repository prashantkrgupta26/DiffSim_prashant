"""YAML configuration: load, validate against a schema, resolve, and record.

The YAML config is the **canonical run record** (spec Phase 0). A tutorial
declares a :class:`ConfigSchema` of typed fields (with defaults, ranges and
choices); this module loads the user's YAML, applies CLI/mode overrides,
validates, and writes a fully-resolved ``config.resolved.yaml`` (defaults +
overrides made explicit) into the run's output directory. Re-running from a
resolved config is exactly reproducible at the configuration level.

Design: a lightweight, dependency-free (PyYAML-only) schema — dataclass/
pydantic-style ergonomics without adding pydantic to the course. Missing
required keys, wrong types, out-of-range values and bad choices are rejected
with a clear, field-named message (``diffsim.errors.ConfigError``).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field as _dc_field
from typing import Any, Optional

import yaml

try:                                   # reuse the project's typed error
    from diffsim.errors import ConfigError
except Exception:                      # pragma: no cover - stand-alone fallback
    class ConfigError(ValueError):
        """Invalid configuration (fallback when diffsim isn't importable)."""


@dataclass
class Field:
    """One schema field.

    Parameters
    ----------
    type : type
        Expected Python type (``int``, ``float``, ``bool``, ``str``, ``list``,
        ``dict``). ``int`` accepts ``bool``-free integers; ``float`` accepts
        ``int``.
    required : bool
        If ``True`` the key must be present (no default is substituted).
    default : Any
        Value used when the key is absent and ``required`` is ``False``.
    choices : sequence, optional
        Allowed values.
    min, max : number, optional
        Inclusive numeric bounds.
    help : str
        One-line description (surfaced in error messages / docs).
    """
    type: type = float
    required: bool = False
    default: Any = None
    choices: Optional[tuple] = None
    min: Optional[float] = None
    max: Optional[float] = None
    help: str = ""

    def coerce(self, name, value):
        # bool must be checked before int (bool is a subclass of int)
        if self.type is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if self.type is int and isinstance(value, bool):
            raise ConfigError(f"config field '{name}': expected int, got bool")
        if not isinstance(value, self.type):
            raise ConfigError(
                f"config field '{name}': expected {self.type.__name__}, got "
                f"{type(value).__name__} ({value!r})")
        if self.choices is not None and value not in self.choices:
            raise ConfigError(
                f"config field '{name}': {value!r} not in "
                f"allowed choices {list(self.choices)}")
        if self.min is not None and value < self.min:
            raise ConfigError(
                f"config field '{name}': {value} < minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ConfigError(
                f"config field '{name}': {value} > maximum {self.max}")
        return value


@dataclass
class ConfigSchema:
    """A named set of :class:`Field` specs.

    ``modes`` and ``meta`` keys in the YAML are reserved (mode-specific
    overrides and free-form provenance notes) and are passed through without
    schema checking.
    """
    fields: dict = _dc_field(default_factory=dict)
    name: str = "config"

    def validate(self, cfg):
        """Validate + coerce a raw config dict against this schema.

        Returns a NEW dict with defaults filled and values coerced. Reserved
        keys (``modes``, ``meta``) are preserved untouched. Unknown keys raise
        (typo protection) unless the schema is empty (free-form).
        """
        out = {}
        reserved = {"modes", "meta"}
        if self.fields:
            unknown = set(cfg) - set(self.fields) - reserved
            if unknown:
                raise ConfigError(
                    f"{self.name}: unknown config key(s) "
                    f"{sorted(unknown)}; known fields are "
                    f"{sorted(self.fields)}")
        for key, spec in self.fields.items():
            if key in cfg and cfg[key] is not None:
                out[key] = spec.coerce(key, cfg[key])
            elif spec.required:
                raise ConfigError(
                    f"{self.name}: missing required config key '{key}'"
                    + (f" ({spec.help})" if spec.help else ""))
            else:
                out[key] = copy.deepcopy(spec.default)
        for key in reserved:
            if key in cfg:
                out[key] = cfg[key]
        # keep unknown keys only when schema is free-form (no fields declared)
        if not self.fields:
            out.update(cfg)
        return out


def load_yaml(path):
    """Load a YAML file into a dict (``{}`` for an empty file)."""
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config {path}: top level must be a mapping, "
                          f"got {type(data).__name__}")
    return data


def apply_mode(cfg, mode):
    """Overlay ``cfg['modes'][mode]`` onto the base config (shallow per key).

    Modes (``quick``/``reference``/``research``) let one config carry several
    cost tiers (spec: quick <2 min, reference 5-30 min). The mode block wins;
    the ``modes`` sub-dict itself is dropped from the returned config.
    """
    cfg = copy.deepcopy(cfg)
    modes = cfg.pop("modes", {}) or {}
    if mode and mode in modes:
        overlay = modes[mode] or {}
        for k, v in overlay.items():
            cfg[k] = v
    return cfg


def resolve_config(path, schema=None, overrides=None, mode=None):
    """Load, mode-overlay, override, and validate a config.

    Parameters
    ----------
    path : str
        Path to the user's YAML config.
    schema : ConfigSchema, optional
        Schema to validate against. If ``None``, the config is loaded free-form.
    overrides : dict, optional
        CLI overrides (highest precedence). ``None`` values are ignored so an
        unset CLI flag never clobbers a config value.
    mode : str, optional
        Run mode whose ``modes[mode]`` block is overlaid before overrides.

    Returns
    -------
    dict
        The fully-resolved config (defaults + mode + overrides, validated).
    """
    raw = load_yaml(path)
    raw = apply_mode(raw, mode)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                raw[k] = v
    schema = schema or ConfigSchema()
    return schema.validate(raw)


def save_resolved(cfg, path):
    """Write the resolved config to ``config.resolved.yaml`` (sorted, block).

    The written file is the canonical, re-runnable record of exactly what ran.
    """
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, default_flow_style=False, sort_keys=True)
    return path


__all__ = [
    "Field", "ConfigSchema", "ConfigError", "load_yaml", "apply_mode",
    "resolve_config", "save_resolved",
]
