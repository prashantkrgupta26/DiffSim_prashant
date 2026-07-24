"""FilmParams — the user-facing parameter surface of the evaporating-film
solver (WodoFilmStepper front end).

A PI/student edits FOUR numbers to run a new material system: chi, N,
the blend, and Bi. Everything else has validated defaults promoted from
the Wodo CMS-2012 replication campaign (benchmarks/phase-field/
wodo_nova.py: CHC noise 1e-3, var-mob D_ratio 1e-3, b_reg 1e-3,
dt0 1e-4, the Appendix-A dt ladder).

YAML schema (all sections optional; defaults shown by `to_yaml`):

    name: my_case
    physics:
      chi: [chi_pf, chi_ps, chi_fs]
      N:   [N_p, N_f, N_s]
      blend:                    # EITHER volume fractions directly:
        phi_p0: 0.125
        phi_f0: 0.125
      # OR a p:f ratio + starting solvent fraction:
      #   ratio: "1:2"          # polymer : fullerene (volume)
      #   phi_s0: 0.9
      kappa: 2.0e-4             # nondimensional (scalar or [kap_p, kap_f])
      # OR physical eps^2 (kappa computed; see kappa_from_eps2):
      # eps2: {value_Jm: 1.0e-10, h0_nm: 1000, Vs_cm3_mol: 80.7, T_K: 300}
      b_reg: 1.0e-3
    evaporation: {Bi: 0.4}      # Bi = k_e in units D_s = L = h0 = 1
    mobility:   {model: variable, D_p: 1.0e-3, D_f: 1.0e-3}
    domain:     {dim: 2, Lx: 2.5, resolution: [96, 48]}
    numerics:   {dt0: 1.0e-4, noise: 1.0e-3, noise_seed: 0, ic_noise: 0.01,
                 ic_seed: 1, linsolver: cudss, device_assembly: true,
                 newton_tol: 1.0e-9, newton_max: 50}
    stop:       {h_min: 0.25, phis_stop: 0.05, max_steps: 200000,
                 wall_cap: null, dt_min: 1.0e-12, dh_cap: 0.004}
    output:     {snap_h: [0.9, 0.7, 0.6, 0.5, 0.4], log_every: 25}
    preflight: strict           # strict = abort on FAIL | warn = continue

NONDIMENSIONALIZATION (kappa_from_eps2): the model is scaled by energy
density E0 = RT/V_s (FH lattice energy per solvent molar volume) and
length h0 (initial film height), so time = h0^2/D_s and Bi = k_e h0/D_s.
The gradient coefficient eps^2 [J/m] becomes
    kappa = eps^2 / (E0 * h0^2)  [dimensionless].
Consistency anchor: the committed Wodo cases use kappa = 2e-4 for their
eps^2 = 3.57e-10 J/m, which corresponds to E0*h0^2 = 1.785e-6 J/m —
at chloroform-like E0 = 3.1e7 J/m^3 that is a 240 nm film. The Negi
configs use the same formula at their own h0 = 1000 nm.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass, field

from ..device import default_device

R_GAS = 8.314462618          # J/(mol K)


def kappa_from_eps2(value_Jm, h0_nm, Vs_cm3_mol, T_K=300.0):
    """kappa = eps^2 / ((R T / V_s) h0^2) — see module docstring."""
    E0 = R_GAS * float(T_K) / (float(Vs_cm3_mol) * 1e-6)   # J/m^3
    h0 = float(h0_nm) * 1e-9                               # m
    return float(value_Jm) / (E0 * h0 * h0)


def _parse_ratio(r):
    """'a:b' or a float a/b -> (a, b) normalized weights."""
    if isinstance(r, str):
        a, b = (float(v) for v in r.split(":"))
    else:
        a, b = float(r), 1.0
    assert a > 0 and b > 0, f"bad blend ratio {r!r}"
    return a, b


@dataclass
class ResolvedFilm:
    """Derived, ready-to-build numbers (echoed by the preflight)."""
    phi_p0: float
    phi_f0: float
    phi_s0: float
    kappa: tuple                 # (kap_p, kap_f)
    M11: float
    M22: float
    var_mob: bool
    D_pair: tuple                # (D_p, D_f) ratios vs D_s
    level: int
    cells: tuple                 # lateral..., vertical LAST
    hc: float
    lat_scale: float
    y_comp: float
    nodes: int
    dofs: int
    nnz: int                     # exact tensor-grid 4-dof CSR count
    dx_lat: float                # physical lateral spacing
    dz0: float                   # physical vertical spacing at h = 1


# (yaml_section, yaml_key) -> dataclass field
_SCHEMA = [
    ("", "name", "name"), ("", "comment", "comment"),
    ("", "preflight", "preflight"), ("", "device", "device"),
    ("physics", "chi", "chi"), ("physics", "N", "N"),
    ("physics", "kappa", "kappa"), ("physics", "eps2", "eps2"),
    ("physics", "b_reg", "b_reg"), ("physics", "f_cheb", "f_cheb"),
    ("evaporation", "Bi", "Bi"),
    ("mobility", "model", "mobility"),
    ("mobility", "D_p", "D_p"), ("mobility", "D_f", "D_f"),
    ("domain", "dim", "dim"), ("domain", "Lx", "Lx"),
    ("domain", "Ly", "Ly"), ("domain", "resolution", "resolution"),
    ("domain", "p", "p"),
    ("numerics", "dt0", "dt0"), ("numerics", "tstep", "tstep"),
    ("numerics", "noise", "noise"),
    ("numerics", "noise_seed", "noise_seed"),
    ("numerics", "ic_noise", "ic_noise"),
    ("numerics", "ic_seed", "ic_seed"),
    ("numerics", "linsolver", "linsolver"),
    ("numerics", "device_assembly", "device_assembly"),
    ("numerics", "gp_residency", "gp_residency"),
    ("numerics", "newton_tol", "newton_tol"),
    ("numerics", "newton_max", "newton_max"),
    ("stop", "h_min", "h_min"), ("stop", "phis_stop", "phis_stop"),
    ("stop", "max_steps", "max_steps"), ("stop", "wall_cap", "wall_cap"),
    ("stop", "dt_min", "dt_min"), ("stop", "dh_cap", "dh_cap"),
    ("output", "snap_h", "snap_h"), ("output", "log_every", "log_every"),
]
_BLEND_KEYS = ("phi_p0", "phi_f0", "ratio", "phi_s0")


@dataclass
class FilmParams:
    name: str = "film"
    comment: str = ""
    # -- physics: THE four PI knobs -----------------------------------
    chi: tuple = (1.0, 0.3, 0.3)          # (chi_pf, chi_ps, chi_fs)
    N: tuple = (5.0, 5.0, 1.0)            # (N_p, N_f, N_s)
    phi_p0: float | None = None           # blend: fractions directly...
    phi_f0: float | None = None
    blend_ratio: str | float | None = None  # ...or p:f ratio + phi_s0
    phi_s0: float | None = None
    Bi: float = 0.4                       # evaporation Biot (= k_e)
    # -- interface energy ----------------------------------------------
    kappa: float | tuple | None = 2e-4    # nondimensional (default: the
    #   committed Wodo N=5 value), or give physical eps2 instead...
    eps2: dict | None = None              # ...(kappa_from_eps2)
    b_reg: float = 1e-3                   # footnote-2 simplex regularizer
    f_cheb: tuple = (0.0, 0.0, 0.0)       # learned delta-f' (T2..T4)
    # -- mobility -------------------------------------------------------
    mobility: str = "variable"            # constant | variable
    D_p: float = 1e-3                     # D_p / D_s
    D_f: float | None = None              # D_f / D_s (None -> = D_p)
    # -- domain ----------------------------------------------------------
    dim: int = 2
    Lx: float = 2.5                       # lateral extent, units of h0
    Ly: float | None = None               # second lateral (3-D only)
    resolution: tuple = (96, 48)          # cells; vertical axis LAST
    p: int = 1                            # basis order (retrofit G5; 1|2)
    # -- numerics --------------------------------------------------------
    dt0: float = 1e-4
    tstep: str = "bdf1"                   # retrofit G5: bdf1 | bdf2
    adapt: str = "ladder"                 # LTE ctrl: ladder(OFF) | lte
    lte_tol: float = 1e-4                 # LTE controller tolerance
    noise: float = 1e-3                   # CHC conserved-flux amplitude
    noise_seed: int = 0
    ic_noise: float = 0.01
    ic_seed: int = 1
    linsolver: str = "cudss"              # splu|cudss|blockch|blockch_dev
    device_assembly: bool = True
    gp_residency: str = "persistent"      # persistent|batch_local|auto
    #   Task #41 memory fallback: persistent (default) holds the packed
    #   GP vals/grads/hist/noise buffers resident at full mesh size;
    #   batch_local re-evaluates them per element batch into a small
    #   reused buffer (~1/nbatch the GP residency, modest recompute);
    #   auto picks batch_local when the estimated GP residency would
    #   exceed half the free VRAM.
    newton_tol: float = 1e-9
    newton_max: int = 50
    # -- stop criteria ---------------------------------------------------
    h_min: float = 0.25
    phis_stop: float = 0.05
    max_steps: int = 200000
    wall_cap: float | None = None         # seconds
    dt_min: float = 1e-12
    dh_cap: float = 0.004
    # -- output / policy --------------------------------------------------
    snap_h: tuple = (0.9, 0.7, 0.6, 0.5, 0.4)
    log_every: int = 25
    preflight: str = "strict"             # strict | warn
    # device: resolved via diffsim.default_device() (arg > DIFFSIM_DEVICE env >
    # auto-detect: cuda:0 if a CUDA device is visible, else cpu). Set explicitly
    # (e.g. from a config file or --device) to override.
    device: str = field(default_factory=default_device)

    # ------------------------------------------------------------------
    def validate(self):
        assert len(self.chi) == 3, "chi = (chi_pf, chi_ps, chi_fs)"
        assert len(self.N) == 3, "N = (N_p, N_f, N_s)"
        assert self.mobility in ("constant", "variable", "negi"), \
            self.mobility
        assert self.linsolver in ("splu", "cudss", "blockch",
                                  "blockch_dev"), self.linsolver
        assert self.gp_residency in ("persistent", "batch_local",
                                     "auto"), self.gp_residency
        assert self.dim in (2, 3), self.dim
        assert len(self.resolution) == self.dim, (
            f"resolution needs {self.dim} entries (vertical last)")
        assert self.preflight in ("strict", "warn"), self.preflight
        # retrofit G5: basis order + time scheme (the stepper is
        # basis-generic since G1 and BDF2-capable since G2)
        assert int(self.p) in (1, 2), f"p must be 1 or 2 (got {self.p})"
        assert self.tstep in ("bdf1", "bdf2"), self.tstep
        assert self.adapt in ("ladder", "lte"), self.adapt
        if self.tstep == "bdf2" and float(self.noise) != 0.0:
            raise ValueError(
                "tstep=bdf2 is deterministic-only (set noise: 0) — the "
                "FDT-noise weak order under BDF2 is out of scope (A4b)")
        if self.linsolver == "splu" and self.device_assembly:
            raise ValueError("device_assembly requires cudss/blockch* "
                             "(the device path has no splu)")
        if self.dim == 3 and self.Ly is None:
            raise ValueError("3-D domains need Ly")

    def blend(self):
        """-> (phi_p0, phi_f0, phi_s0)."""
        if self.phi_p0 is not None and self.phi_f0 is not None:
            pp, pf = float(self.phi_p0), float(self.phi_f0)
            ps = 1.0 - pp - pf
            if self.phi_s0 is not None:
                assert abs(ps - self.phi_s0) < 1e-9, (
                    "blend over-specified and inconsistent: "
                    f"1-phi_p0-phi_f0 = {ps} vs phi_s0 = {self.phi_s0}")
            return pp, pf, ps
        if self.blend_ratio is not None and self.phi_s0 is not None:
            a, b = _parse_ratio(self.blend_ratio)
            solute = 1.0 - float(self.phi_s0)
            return (solute * a / (a + b), solute * b / (a + b),
                    float(self.phi_s0))
        raise ValueError("blend under-specified: give (phi_p0, phi_f0) "
                         "or (ratio, phi_s0)")

    def kappa_pair(self):
        # a physical eps2 block, when present, WINS over kappa (which
        # always carries its default value)
        if self.eps2 is not None:
            k = kappa_from_eps2(**self.eps2)
            return (k, k)
        if self.kappa is not None:
            k = self.kappa
            return (float(k), float(k)) if np_scalar(k) else (
                float(k[0]), float(k[1]))
        raise ValueError("give physics.kappa or physics.eps2")

    def resolve(self) -> ResolvedFilm:
        self.validate()
        pp0, pf0, ps0 = self.blend()
        assert pp0 > 0 and pf0 > 0 and ps0 > 0, (pp0, pf0, ps0)
        kap = self.kappa_pair()
        Dp = float(self.D_p)
        Df = Dp if self.D_f is None else float(self.D_f)
        # frozen-mobility mapping at the initial blend (wodo_film
        # docstring v1): M_i = D(phi0)/f''_ideal,i, units D_s = 1
        Np, Nf, Ns = (float(n) for n in self.N)
        D0 = ps0 * 1.0 + Dp * pp0 + Df * pf0
        M11 = D0 / (1.0 / (Np * pp0) + 1.0 / (Ns * ps0))
        M22 = D0 / (1.0 / (Nf * pf0) + 1.0 / (Ns * ps0))
        # strip mesh: uniform level-L octree, keep cells[i] columns
        cells = tuple(int(c) for c in self.resolution)
        level = max(1, math.ceil(math.log2(max(cells))))
        hc = 2.0 ** -level
        lat_scale = self.Lx / (cells[0] * hc)
        if self.dim == 3:
            ls_y = self.Ly / (cells[1] * hc)
            if abs(ls_y - lat_scale) > 1e-12 * lat_scale:
                raise ValueError(
                    "the mapped metric carries ONE lateral scale: need "
                    f"Lx/nx == Ly/ny (got {lat_scale} vs {ls_y})")
        y_comp = cells[-1] * hc
        # retrofit G5: basis-order-generic node + CSR-pair counts on the
        # tensor Q_p grid.  1-D: p*c+1 nodes; node-pair count per axis =
        # c*(p+1)^2 - (c-1) (each element's (p+1)^2 pairs, minus the one
        # shared-vertex pair double-counted between adjacent elements);
        # d-D CSR pairs = product over axes (the adjacency graph is the
        # Cartesian product).  p = 1 reproduces c+1 nodes and 3(c+1)-2
        # pairs EXACTLY (integer-identical), so p1 forecasts are unchanged.
        pp = int(self.p)
        nodes = 1
        pairs = 1
        for c in cells:
            nodes *= pp * c + 1
            pairs *= c * (pp + 1) ** 2 - (c - 1)
        return ResolvedFilm(
            phi_p0=pp0, phi_f0=pf0, phi_s0=ps0, kappa=kap,
            M11=M11, M22=M22, var_mob=(self.mobility == "variable"),
            D_pair=(Dp, Df), level=level, cells=cells, hc=hc,
            lat_scale=lat_scale, y_comp=y_comp, nodes=nodes,
            dofs=4 * nodes, nnz=16 * pairs,
            dx_lat=hc * lat_scale, dz0=1.0 / cells[-1])

    # -- serialization --------------------------------------------------
    def to_dict(self):
        d = {}
        for sec, key, fld in _SCHEMA:
            v = getattr(self, fld)
            if isinstance(v, tuple):
                v = list(v)
            if sec:
                d.setdefault(sec, {})[key] = v
            else:
                d[key] = v
        blend = {}
        if self.phi_p0 is not None:
            blend["phi_p0"] = self.phi_p0
        if self.phi_f0 is not None:
            blend["phi_f0"] = self.phi_f0
        if self.blend_ratio is not None:
            blend["ratio"] = self.blend_ratio
        if self.phi_s0 is not None:
            blend["phi_s0"] = self.phi_s0
        d.setdefault("physics", {})["blend"] = blend
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        kw = {}
        blend = (d.get("physics") or {}).pop("blend", None) \
            if "physics" in d else None
        for sec, key, fld in _SCHEMA:
            src = d.get(sec, {}) if sec else d
            if isinstance(src, dict) and key in src:
                v = src[key]
                if isinstance(v, list):
                    v = tuple(v)
                kw[fld] = v
        if blend:
            unknown = set(blend) - set(_BLEND_KEYS)
            assert not unknown, f"unknown blend keys {unknown}"
            kw["phi_p0"] = blend.get("phi_p0")
            kw["phi_f0"] = blend.get("phi_f0")
            kw["blend_ratio"] = blend.get("ratio")
            kw["phi_s0"] = blend.get("phi_s0")
        p = cls(**kw)
        p.validate()
        return p

    def to_yaml(self, path=None):
        d = self.to_dict()
        try:
            import yaml
            text = yaml.safe_dump(d, sort_keys=False,
                                  default_flow_style=None)
        except ImportError:                       # minimal fallback
            text = json.dumps(d, indent=2)        # JSON is valid YAML
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    @classmethod
    def from_yaml(cls, path):
        with open(path) as fh:
            text = fh.read()
        try:
            import yaml
            d = yaml.safe_load(text)
        except ImportError:
            d = json.loads(text)                  # JSON-subset fallback
        return cls.from_dict(d)

    def config_hash(self):
        """Stable digest of the resolved configuration (provenance)."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def replace(self, **kw):
        return dataclasses.replace(self, **kw)


def np_scalar(v):
    try:
        len(v)
        return False
    except TypeError:
        return True
