"""SP-1 R0 Block E — E2 SimSalabim 1-D cross-check + parameter mapping.

SimSalabim (Koster group, https://github.com/kostergroup/SIMsalabim) is the
community-standard 1-D drift-diffusion solver.  On this Mac (arm64) the Python
wrapper (`pip install pySIMsalabim`) installs cleanly but ships no binary; the
`SimSS` binary is built from source with the Free Pascal compiler
(`brew install fpc; fpc -O3 -Fu../Units simss.pas`) — verified working here
(see the E-report for the exact build + run transcript).

Parameter-mapping dictionary (DELIVERABLE) — our nondim XDD device ↔ SimSS
`device_parameters.txt` conventions.  Both solve the SAME 1-D homogeneous OPV
device: single layer, uniform generation, Langevin recombination, ohmic
contacts, no traps, no ions.

  XDD (our) quantity            SimSS parameter        Mapping
  ----------------------------  ---------------------  ------------------------
  height          [m]           L                      identity
  eps_A=eps_D=eps_r             eps_r                   identity (homogeneous)
  N_C=N_V         [m^-3]        N_c                     identity (single DOS)
  mu_n            [m^2/Vs]      mu_n                     identity
  mu_p            [m^2/Vs]      mu_p                     identity
  E_g             [eV]          E_c, E_v                 E_v = E_c + E_g
  built-in V_bi = E_g/q         W_L=E_c, W_R=E_v         ohmic ⇒ V_bi = E_g/q
  Langevin γ (strategy='sum')   useLangevin=1,           γ_L = q(mu_n+mu_p)/ε
    = 2(mu_n+mu_p)q/(ε_A+ε_D)ε0   preLangevin=ζ           (SimSS uses (mu_n+mu_p);
    ×ζ                                                     our 'sum' has ε̄=(ε_A+ε_D)/2
                                                          ⇒ identical for ε_A=ε_D.
                                                          ζ = preLangevin prefactor.)
  uniform G       [m^-3 s^-1]   G_ehp, layerGen=1,       identity; genProfile=none
                                genProfile=none          (free-carrier gen, fieldDepG=0)
  traps                         N_t_*=0                  none
  ions                          N_anion=N_cation=0       none
  T               [K]           T                        identity

Because our XDD FORWARD solver hits the documented CPU-mesh drive wall on the
PHYSICAL device (Ê_g≈52, Debye≈0.44 nm ≪ any feasible CPU element — see
test_xdd_run._resolvable_bilayer and the E-report), the E2 gate LOCKS the SimSS
1-D reference for the physical Kodali-class device as the anchor (its Jsc lands
in the −30 A/m² Kodali band, cross-validating the device parametrisation), and
records the honest mesh-wall reason the XDD-vs-SimSS absolute head-to-head is a
Block-D/E (Scharfetter–Gummel / GPU-mesh) item rather than an R0-CPU one.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _bench_bootstrap  # noqa

# Resolution order for a working SimSS binary (built in the E-work scratch dir).
_SIMSS_ENV = os.environ.get("SIMSS_BIN", "")
_SIMSS_CANDIDATES = [
    _SIMSS_ENV,
    shutil.which("simss") or "",
    shutil.which("SimSS") or "",
]


def find_simss() -> str | None:
    """Return a path to a runnable SimSS binary, or None if unavailable."""
    for c in _SIMSS_CANDIDATES:
        if c and Path(c).exists() and os.access(c, os.X_OK):
            return c
    return None


# ── Matched 1-D device: physical Kodali-class PPV:PCBM values ────────────────
KODALI_DEVICE = dict(
    L=120e-9, eps_r=3.4, N_c=2.5e25, mu_n=2.5e-7, mu_p=3.0e-8,
    E_g=1.34, T=300.0, G_ehp=2.7e27, E_c=3.9, zeta=1.0,
)


def write_layer_file(path: Path, d: dict, data_dir: Path) -> None:
    """Emit a SimSS layer parameter file for the matched homogeneous device."""
    E_v = d["E_c"] + d["E_g"]
    nk = data_dir / "nk_C60_1.txt"
    path.write_text(f"""** SIMsalabim Layer parameters (E2 matched device):
** version: 5.36
**General**
L = {d['L']:.6E}
eps_r = {d['eps_r']}
E_c = {d['E_c']}
E_v = {E_v}
N_c = {d['N_c']:.6E}
N_D = 0
N_A = 0
**Mobilities**
mu_n = {d['mu_n']:.6E}
mu_p = {d['mu_p']:.6E}
mobnDep = 0
mobpDep = 0
gamma_n = 0
gamma_p = 0
**Interface-layer-to-right**
nu_int_n = 1E3
nu_int_p = 1E3
N_t_int = 0
E_t_int = 4.7
intTrapFile = none
intTrapType = -1
C_n_int = 2E-14
C_p_int = 2E-14
**Ions**
N_anion = 0E21
N_cation = 0E21
mu_anion = 1E-11
mu_cation = 1E-11
ionsMayEnter = 0
**Generation and recombination**
G_ehp = {d['G_ehp']:.6E}
layerGen = 1
nkLayer = {nk}
fieldDepG = 0
P0 = 0
a = 1E-9
thermLengDist = 2
k_f = 1E6
k_direct = 0
preLangevin = {d['zeta']}
useLangevin = 1
**Bulk trapping**
N_t_bulk = 0E20
C_n_bulk = 2E-14
C_p_bulk = 2E-14
E_t_bulk = 4.7
bulkTrapFile = none
bulkTrapType = -1
""")


def write_setup_file(path: Path, d: dict, layer_name: str, data_dir: Path) -> None:
    """Emit the SimSS simulation-setup file for the matched device."""
    E_v = d["E_c"] + d["E_g"]
    path.write_text(f"""** SimSS Simulation Setup (E2 matched device):
** version: 5.36
**General**
T = {d['T']}
**Layers**
l1 = {layer_name}
**Contacts**
leftElec = -1
W_L = {d['E_c']}
W_R = {E_v}
S_n_L = -1E-7
S_p_L = -1E-7
S_n_R = -1E-7
S_p_R = -1E-7
R_shunt = -5E3
R_series = 0
**Optics**
G_frac = 1
genProfile = none
L_TCO = 0
L_BE = 101E-9
nkSubstrate = {data_dir / 'nk_SiO2.txt'}
nkTCO = {data_dir / 'nk_ITO.txt'}
nkBE = {data_dir / 'nk_Au.txt'}
spectrum = {data_dir / 'AM15G.txt'}
lambda_min = 3.5E-7
lambda_max = 8E-7
**Numerical Parameters**
NP = 400
tolPois = 1E-6
maxDelV = 10
maxItPois = 2000
maxItSS = 2000
currDiffInt = 2
tolCurr = 1E-4
tolDens = 1E-6
couplePC = 4
minAcc = 0.5
maxAcc = 0.95
ignoreNegDens = 1
convVar = 1
failureMode = 2
grad = 4
**Voltage range**
Vdist = 1
preCond = 0
Vpre = 0
fixIons = 0
Vscan = 1
Vmin = -0.2
Vmax = {E_v - d['E_c']:.3f}
Vstep = 0.02
Vacc = 0
NJV = 100
untilVoc = 0
**User interface**
timeout = 300
pauseAtEnd = 0
autoTidy = 0
useExpData = 0
expJV = expJV.csv
fitMode = lin
fitThreshold = 0.8
JVFile = JV.dat
varFile = none
limitDigits = 0
outputRatio = 1
scParsFile = scPars.txt
logFile = log.txt
""")


def _parse_scpars(text: str) -> dict:
    """Extract Jsc [A/m2], Voc [V], FF from a SimSS scPars.txt / stdout blob."""
    out = {}
    for key, pat in (("Jsc", r"Jsc:\s*([-\d.Ee+]+)"),
                     ("Voc", r"Voc:\s*([-\d.Ee+]+)"),
                     ("FF",  r"FF:\s*([-\d.Ee+]+)")):
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def run_simss(device: dict, workdir: Path, data_dir: Path) -> dict:
    """Run SimSS on the matched device; return {Jsc, Voc, FF} (or {} if no binary).

    Parameters
    ----------
    device : dict     KODALI_DEVICE-shaped parameter dict.
    workdir : Path    scratch dir to write inputs / run in.
    data_dir : Path   SIMsalabim/Data dir (for nk + spectrum files).
    """
    binp = find_simss()
    if binp is None:
        return {}
    workdir.mkdir(parents=True, exist_ok=True)
    layer = workdir / "L1.txt"
    setup = workdir / "setup.txt"
    write_layer_file(layer, device, data_dir)
    write_setup_file(setup, device, layer.name, data_dir)
    proc = subprocess.run(
        [binp, setup.name], cwd=str(workdir),
        capture_output=True, text=True, timeout=600, input="\n")
    blob = proc.stdout + "\n" + proc.stderr
    scp = workdir / "scPars.txt"
    if scp.exists():
        blob = blob + "\n" + scp.read_text()
    res = _parse_scpars(blob)
    res["returncode"] = proc.returncode
    return res


if __name__ == "__main__":
    import json
    binp = find_simss()
    print("SimSS binary:", binp or "NOT FOUND (set SIMSS_BIN)")
    if binp:
        # data dir is next to the binary's repo: <repo>/Data
        data_dir = Path(binp).resolve().parent.parent / "Data"
        wd = Path(os.environ.get("E2_WORKDIR", "/tmp/e2_run"))
        res = run_simss(KODALI_DEVICE, wd, data_dir)
        print(json.dumps(res, indent=2))
