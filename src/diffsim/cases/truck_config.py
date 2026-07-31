"""TRUCK case config loader (ThinShell.pdf case; libconfig-style config.txt).

Parses ONLY what the truck driver (tests/truck_flow.py) consumes; every other
top-level key is warn-and-ignored (the ignored set is asserted by the test).

Coordinate convention (VERIFIED by bounding-box inspection, see task report):
the config's physical domain is channel_mesh.max = [16, 2, 2]; the octree's
native domain is the unit cube [0,1]^3, so a single ISOTROPIC scale factor
1/16 maps physical -> unit-cube coordinates.  ``TruckConfig`` records the raw
physical values; the driver applies /domain_scale where needed.

libconfig grammar handled here (a pragmatic subset, enough for the truck case):
  key = value ;?          scalars: bool, int, float, "string"
  key = [ a, b, c ]       homogeneous arrays (numbers)
  block = { ... }         nested settings group
  list  = ( {..}, {..} )  list of groups (geometries / region_refine / boundary)
  # ... and comments (whole-line and trailing); commented-out geometry entries
  are NOT parsed (they are inside comment lines), matching the C++ active set.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Tokenizer / recursive-descent parser for the libconfig subset
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove '#' and '//' whole-line/trailing comments (not inside strings)."""
    out_lines = []
    for line in text.splitlines():
        buf = []
        i = 0
        in_str = False
        while i < len(line):
            c = line[i]
            if c == '"':
                in_str = not in_str
                buf.append(c)
                i += 1
                continue
            if not in_str and c == '#':
                break
            if not in_str and c == '/' and i + 1 < len(line) and line[i + 1] == '/':
                break
            buf.append(c)
            i += 1
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


class _Tok:
    def __init__(self, text: str):
        self.s = text
        self.i = 0
        self.n = len(text)

    def _skip_ws(self):
        while self.i < self.n and self.s[self.i] in " \t\r\n;,":
            self.i += 1

    def peek(self):
        self._skip_ws()
        if self.i >= self.n:
            return None
        return self.s[self.i]

    def next_char(self):
        self._skip_ws()
        c = self.s[self.i]
        self.i += 1
        return c

    def read_string(self):
        # assumes current char is opening quote
        assert self.s[self.i] == '"'
        self.i += 1
        start = self.i
        while self.i < self.n and self.s[self.i] != '"':
            self.i += 1
        val = self.s[start:self.i]
        self.i += 1  # closing quote
        return val

    def read_bareword(self):
        self._skip_ws()
        start = self.i
        while self.i < self.n and (self.s[self.i].isalnum()
                                   or self.s[self.i] in "_.+-eE"):
            self.i += 1
        return self.s[start:self.i]


def _parse_value(tk: _Tok) -> Any:
    c = tk.peek()
    if c == '{':
        return _parse_group(tk)
    if c == '(':
        return _parse_list(tk)
    if c == '[':
        return _parse_array(tk)
    if c == '"':
        return tk.read_string()
    word = tk.read_bareword()
    return _coerce_scalar(word)


def _coerce_scalar(word: str) -> Any:
    lw = word.lower()
    if lw == "true":
        return True
    if lw == "false":
        return False
    try:
        return int(word)
    except ValueError:
        pass
    try:
        return float(word)
    except ValueError:
        pass
    return word


def _parse_array(tk: _Tok) -> list:
    assert tk.next_char() == '['
    vals = []
    while True:
        c = tk.peek()
        if c is None:
            raise ValueError("unterminated array")
        if c == ']':
            tk.next_char()
            break
        vals.append(_parse_value(tk))
    return vals


def _parse_list(tk: _Tok) -> list:
    assert tk.next_char() == '('
    vals = []
    while True:
        c = tk.peek()
        if c is None:
            raise ValueError("unterminated list")
        if c == ')':
            tk.next_char()
            break
        vals.append(_parse_value(tk))
    return vals


def _parse_group(tk: _Tok) -> dict:
    assert tk.next_char() == '{'
    d = {}
    while True:
        c = tk.peek()
        if c is None:
            raise ValueError("unterminated group")
        if c == '}':
            tk.next_char()
            break
        key = tk.read_bareword()
        if not key:
            raise ValueError(f"expected key near offset {tk.i}")
        eq = tk.next_char()
        if eq != '=':
            raise ValueError(f"expected '=' after {key!r}, got {eq!r}")
        d[key] = _parse_value(tk)
    return d


def _parse_toplevel(text: str) -> dict:
    tk = _Tok(text)
    d = {}
    while True:
        c = tk.peek()
        if c is None:
            break
        key = tk.read_bareword()
        if not key:
            break
        eq = tk.next_char()
        if eq != '=':
            raise ValueError(f"expected '=' after {key!r}, got {eq!r}")
        d[key] = _parse_value(tk)
    return d


# ---------------------------------------------------------------------------
# Typed config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BodySpec:
    """One immersed body (STL) with its placement and per-object refine level."""
    mesh_path: str            # STL filename (relative to the config dir)
    position: tuple           # (x, y, z) translation applied to STL coords
    refine_lvl: int           # per-object octree refinement level
    is_static: bool = True


@dataclass(frozen=True)
class RegionBox:
    """An axis-aligned cube refinement region (physical coords)."""
    min_c: tuple
    max_c: tuple
    refine_region_lvl: int


@dataclass
class TruckConfig:
    """Everything the truck driver consumes from config.txt.

    All geometric quantities are in the config's PHYSICAL frame ([16,2,2]
    domain).  ``domain_scale`` is the isotropic factor mapping physical ->
    the octree unit cube (= 1 / domain_max[0], i.e. 1/16 for the truck case).
    """
    config_dir: str
    sbm_geo: str                       # "TRUCK"
    slope_near_ground: float           # inlet ramp slope (0.25)
    # channel mesh
    domain_min: tuple                  # physical [0,0,0]
    domain_max: tuple                  # physical [16,2,2]
    refine_lvl_base: int
    refine_lvl_channel_wall: int
    refine_walls: bool
    # bodies / regions
    bodies: list                       # list[BodySpec] (ACTIVE only)
    region_refine: list                # list[RegionBox]
    # penalty
    cb_f: float                        # Nitsche penalty coeff (-> alpha)
    ci_f: float
    tau_m_scale: float                 # C++ tauM_scale (NSEquation.h:530) —
                                       # direct multiplier on tauM; tauC
                                       # inherits the inverse (1/(tauM*gg))
    # Re ramp (time-based)
    re_v: tuple                        # [Re0, Re1, Re2]
    re_ramping: tuple                  # [t0, t1, t2]
    nondim_type: str
    # solver-effort Re ramp (Phase 2)
    do_re_solver_ramp: bool
    re_solver_ramp_initial: float
    re_solver_ramp_target: float
    re_solver_ramp_increment: float
    # time stepping / output
    dt_v: tuple                        # [dt0, dt1, dt2]
    total_t_v: tuple
    ns_timestepper: str
    output_interval: int
    # bookkeeping
    unknown_keys: list = field(default_factory=list)

    @property
    def domain_scale(self) -> float:
        return 1.0 / float(self.domain_max[0])


# Top-level keys the driver actively consumes (everything else -> unknown_keys).
_CONSUMED = {
    "SBMGeo", "slopeNearGround", "channel_mesh", "geometries", "region_refine",
    "boundary", "Cb_f", "Ci_f", "tauM_scale", "Re_V", "Re_ramping", "NondimensionType",
    "DoReSolverRamp", "ReSolverRampInitial", "ReSolverRampTarget",
    "ReSolverRampIncrement", "dt_V", "totalT_V", "NSTimestepper",
    "OutputInterval",
}


def _as_tuple(v, n=None):
    if not isinstance(v, (list, tuple)):
        raise ValueError(f"expected array, got {v!r}")
    t = tuple(v)
    if n is not None and len(t) != n:
        raise ValueError(f"expected length-{n} array, got {t!r}")
    return t


def load_truck_config(path: str) -> TruckConfig:
    """Parse a TRUCK case config.txt at ``path`` into a ``TruckConfig``.

    Unknown top-level keys are collected into ``.unknown_keys`` and a single
    ``UserWarning`` is emitted listing them (warn-and-ignore contract).
    """
    import os
    with open(path, "r") as fh:
        raw = fh.read()
    text = _strip_comments(raw)
    top = _parse_toplevel(text)

    config_dir = os.path.dirname(os.path.abspath(path))

    # --- channel mesh -------------------------------------------------------
    cm = top.get("channel_mesh", {})
    if not isinstance(cm, dict):
        raise ValueError("channel_mesh must be a group")
    domain_min = _as_tuple(cm.get("min", [0, 0, 0]), 3)
    domain_max = _as_tuple(cm.get("max", [16, 2, 2]), 3)

    # --- bodies (ACTIVE geometries only) ------------------------------------
    bodies = []
    for g in top.get("geometries", []) or []:
        if not isinstance(g, dict):
            continue
        bodies.append(BodySpec(
            mesh_path=str(g["mesh_path"]),
            position=_as_tuple(g.get("position", [0.0, 0.0, 0.0]), 3),
            refine_lvl=int(g.get("refine_lvl", 0)),
            is_static=bool(g.get("is_static", True)),
        ))

    # --- region refine ------------------------------------------------------
    regions = []
    for r in top.get("region_refine", []) or []:
        if not isinstance(r, dict):
            continue
        regions.append(RegionBox(
            min_c=_as_tuple(r["min_c"], 3),
            max_c=_as_tuple(r["max_c"], 3),
            refine_region_lvl=int(r["refine_region_lvl"]),
        ))

    # --- unknown keys -------------------------------------------------------
    unknown = sorted(k for k in top if k not in _CONSUMED)
    if unknown:
        warnings.warn(
            "load_truck_config: ignoring unrecognized top-level keys: "
            + ", ".join(unknown),
            UserWarning,
            stacklevel=2,
        )

    cfg = TruckConfig(
        config_dir=config_dir,
        sbm_geo=str(top.get("SBMGeo", "TRUCK")),
        slope_near_ground=float(top.get("slopeNearGround", 0.25)),
        domain_min=domain_min,
        domain_max=domain_max,
        refine_lvl_base=int(cm.get("refine_lvl_base", 7)),
        refine_lvl_channel_wall=int(cm.get("refine_lvl_channel_wall", 6)),
        refine_walls=bool(cm.get("refine_walls", True)),
        bodies=bodies,
        region_refine=regions,
        cb_f=float(top.get("Cb_f", 20.0)),
        ci_f=float(top.get("Ci_f", 36.0)),
        tau_m_scale=float(top.get("tauM_scale", 1.0)),
        re_v=_as_tuple(top.get("Re_V", [1e3, 5e3, 1e4])),
        re_ramping=_as_tuple(top.get("Re_ramping", [0, 50, 51])),
        nondim_type=str(top.get("NondimensionType", "mix_conv")),
        do_re_solver_ramp=bool(top.get("DoReSolverRamp", False)),
        re_solver_ramp_initial=float(top.get("ReSolverRampInitial", 10)),
        re_solver_ramp_target=float(top.get("ReSolverRampTarget", 1e4)),
        re_solver_ramp_increment=float(top.get("ReSolverRampIncrement", 50)),
        dt_v=_as_tuple(top.get("dt_V", [1e-7, 0.01, 5e-4])),
        total_t_v=_as_tuple(top.get("totalT_V", [50.0, 51.0, 1e4])),
        ns_timestepper=str(top.get("NSTimestepper", "BDF2")),
        output_interval=int(top.get("OutputInterval", 100)),
        unknown_keys=unknown,
    )
    return cfg
