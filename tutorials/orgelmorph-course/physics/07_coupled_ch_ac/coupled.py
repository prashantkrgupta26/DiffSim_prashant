"""OrgElMorph course - Physics P7: coupled Cahn-Hilliard + Allen-Cahn.

Importable core.  The real morphology problem is COUPLED: composition
(Cahn-Hilliard, P1-P5) and crystallinity (Allen-Cahn, P6) evolve
together and feed back on each other.  The signature of the coupling is
CRYSTALLIZATION-DRIVEN DEMIXING: as a species crystallizes it expels the
other components from the crystal, sharpening the composition pattern
that phase separation alone would not produce.  This concept turns the
coupling on and measures that feedback, and shows the model scaling up
the (M, K) ladder: (2,1) -> (3,1) -> (3,2).

THE FOUR-FOLD chi.  With crystallinity, the interaction between two
species depends on whether each is amorphous or crystalline, so the
single Flory chi becomes FOUR matrices (multiphase docstring):

  chi_eff(i,j) = chi_aa  (amorphous i - amorphous j)
    + psi_i^2 chi_ca + psi_j^2 chi_ac + psi_i^2 psi_j^2 chi_cc   (r14),

with chi_ca = chi_ac^T by symmetry.  Making chi_ca LARGER than chi_aa
means "a crystal of species i dislikes amorphous species j MORE than two
amorphous phases dislike each other" -- so crystallizing species i
pushes j out.  That is the thermodynamic origin of crystallization-
driven demixing.

We use MultiPhaseStepper at (M, K) = (2,1) (two solutes, one
crystallizes), then (3,1) and (3,2) to show the same machinery scales.
The DEMIXING METRIC is the purity contrast: the mean composition of the
crystallizing species INSIDE the crystal (psi>0.5) minus OUTSIDE.  With
the coupling on it is large and positive; with K=0 (crystallization off)
the same blend only demixes by ordinary Cahn-Hilliard, a much smaller
contrast.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _chis(n_sp, chi_aa_val, chi_ca_val, cryst=(0,)):
    """Build the FULL-species (n_sp x n_sp) chi_aa and chi_ca.  Amorphous
    pairs interact with chi_aa_val (all off-diagonals); a crystalline
    species k (in `cryst`) interacts with every OTHER species through
    the stronger chi_ca_val (expulsion)."""
    aa = np.full((n_sp, n_sp), chi_aa_val)
    np.fill_diagonal(aa, 0.0)
    ca = np.zeros((n_sp, n_sp))
    for k in cryst:
        for j in range(n_sp):
            if j != k:
                ca[k, j] = chi_ca_val
    return aa, ca


def _disc(c, r0, w=0.03, amp=0.95):
    return lambda x: amp * 0.5 * (1.0 - np.tanh(
        (np.sqrt((x[:, 0] - c[0]) ** 2
                 + (x[:, 1] - c[1]) ** 2) - r0) / w))


def _grid_map(mesh):
    fc = mesh.node_coords
    xs = np.unique(np.round(fc[:, 0], 12))
    ys = np.unique(np.round(fc[:, 1], 12))
    ix = np.searchsorted(xs, np.round(fc[:, 0], 12))
    iy = np.searchsorted(ys, np.round(fc[:, 1], 12))
    return len(xs), len(ys), ix, iy


def run(dm, mesh, cons, M=2, K=1, crystallize=True, chi_aa=1.2,
        chi_ca=2.6, dh=-1.3, dsig=1.0, Tm=1.0, T=0.55, eps2=1e-3,
        L_psi=6.0, kappa=5e-4, phi0=None, seeds=None, r0=0.13,
        amp=0.01, dt=5e-4, t_end=0.4, dt_max=0.02, seed=4,
        mobility=0.2, device="cuda:0", nchecks=16):
    """Coupled (M, K) quench with a seeded crystal of species 0.  If
    crystallize=False the same blend runs with K=0 (no crystal terms) as
    the control.  Returns fields, the demixing purity contrast, and the
    crystalline-area history."""
    n_sp = M + 1
    if phi0 is None:
        # concentrated blend: species 0 (crystallizer) richest
        base = [0.45, 0.30, 0.15][:M]
        phi0 = base
    Kuse = K if crystallize else 0
    aa, ca = _chis(n_sp, chi_aa, chi_ca, cryst=tuple(range(min(K, M))))
    kw = {}
    if Kuse > 0:
        kw = dict(chi_ac=ca.T.copy(), chi_ca=ca,
                  dsig=[dsig] * Kuse, dh=[dh] * Kuse, Tm=[Tm] * Kuse,
                  eps2=[eps2] * Kuse, L_psi=[L_psi] * Kuse,
                  alpha_th=[0.0] * Kuse, beta_th=[0.0] * Kuse,
                  L_th=[5.0] * Kuse)
    st = MultiPhaseStepper(
        dm, M=M, K=Kuse, chi_aa=aa, N=[1.0] * n_sp,
        onsager=(mobility * np.eye(M)).tolist(), kappa=[kappa] * M, dt=dt,
        T=T, bulk="p1", newton_tol=1e-7, newton_max=50,
        linsolver="cudss", **kw)
    rng = np.random.default_rng(seed)
    phi_fns = [(lambda v: (lambda x: np.full(len(x), v)
                           + amp * rng.standard_normal(len(x))))(p)
               for p in phi0]
    if seeds is None:
        seeds = [(0.5, 0.5)]
    if Kuse > 0:
        coords = st.free_coords
        psi_list = []
        for k in range(Kuse):
            ps = np.zeros(st.nfree)
            for c in seeds:
                d = _disc(c, r0)(coords)
                ps = np.maximum(ps, d)
            psi_list.append(ps)
        st.set_initial(phi_fns,
                       [(lambda a: (lambda x: a))(ps) for ps in psi_list],
                       [(lambda x: np.zeros(len(x)))] * Kuse)
    else:
        st.set_initial(phi_fns)
    nx, ny, ix, iy = _grid_map(mesh)

    def phi_grid(i):
        g = np.zeros((nx, ny)); g[ix, iy] = np.asarray(cons.T @ st.phi(i))
        return g

    def crys_mask():
        if Kuse == 0:
            return None
        return np.asarray(cons.T @ st.psi(0)) > 0.5

    def area():
        return 0.0 if Kuse == 0 else float(np.mean(crys_mask()))

    ts, ars = [0.0], [area()]
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=1500, dt_min=1e-7)
        ts.append(st.t); ars.append(area())

    # demixing purity contrast for species 0
    phi0_full = np.asarray(cons.T @ st.phi(0))
    if Kuse > 0:
        m = crys_mask()
        inside = float(phi0_full[m].mean()) if m.any() else float("nan")
        outside = float(phi0_full[~m].mean()) if (~m).any() else float("nan")
    else:
        # no crystal: split by the phi0 field median (pure-CH demixing)
        med = np.median(phi0_full)
        inside = float(phi0_full[phi0_full >= med].mean())
        outside = float(phi0_full[phi0_full < med].mean())
    contrast = inside - outside
    return dict(M=M, K=Kuse, t=np.array(ts), area=np.array(ars),
                phi0=phi_grid(0), phi1=phi_grid(1) if M >= 2 else None,
                psi=(phi_grid_of(cons, st, ix, iy, nx, ny)
                     if Kuse > 0 else None),
                inside=inside, outside=outside, contrast=contrast,
                area_end=ars[-1], nx=nx, ny=ny,
                phi0_std=float(phi0_full.std()))


def phi_grid_of(cons, st, ix, iy, nx, ny):
    g = np.zeros((nx, ny)); g[ix, iy] = np.asarray(cons.T @ st.psi(0))
    return g
