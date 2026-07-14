"""OrgElMorph course - Physics P6: Allen-Cahn crystallization.

Importable core.  Phase separation (Cahn-Hilliard) is only half of
organic-film morphology; the other half is CRYSTALLIZATION -- a
component ordering from an amorphous melt into a crystal.  Crystallinity
is a NON-CONSERVED order parameter psi in [0,1] (crystal can be created,
unlike composition), so it obeys ALLEN-CAHN (non-conserved gradient
flow) rather than Cahn-Hilliard.  This concept isolates the
crystallization physics: a seeded crystal that GROWS below the melting
point and MELTS above it, the growth KINETICS (the Avrami / JMAK law),
and ORIENTATION markers that distinguish grains.

We use the (M, K)-generic brick with M=1 (one composition) and K=1 (one
crystallizable species): fields (phi, mu, psi, theta).  The
crystallization free energy (2310.11844 "r14" form, PCBM-class
energetics from materials.yaml) drives psi:

    dpsi/dt = -L_psi [ df/dpsi - eps2 lap psi ],
    f_cr = phi [ q(psi) dsig + p(psi) drive ],
    q(psi) = psi^2 (1-psi)^2  (double-well barrier),
    p(psi) = psi^2 (3-2psi)   (interpolation),
    drive  = dh (T/Tm - 1)    (Turnbull undercooling; NEGATIVE for
                               T < Tm, so psi -> 1 lowers f -> growth).

  * SIGN: T < Tm gives drive < 0 -> the crystal GROWS; T > Tm gives
    drive > 0 -> the crystal MELTS.  The melting point Tm is where the
    driving force changes sign.
  * KINETICS (Avrami/JMAK): with athermal nuclei placed at t=0 growing
    at a roughly constant interface velocity v, the crystalline area of
    each grain grows like (v t)^2 in 2-D before grains impinge, so the
    crystalline FRACTION follows X(t) = 1 - exp[-(k t)^n] with Avrami
    exponent n ~ 2 (2-D, pre-existing nuclei, interface-limited).  We
    fit n from the measured X(t).
  * ORIENTATION: each seed carries an orientation marker theta; the
    grain-labeling analysis (multiphase.grain_labels) separates crystals
    by theta plateaus (grain boundaries show as theta jumps).  Here
    theta is a frozen marker; its DYNAMICS (grain-boundary energy that
    stops impinging crystals) is the coupled concept P7.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, grain_labels

# PCBM-class energetics (materials.yaml crystallization.PCBM_class /
# 2310.11844 Table 1, nondimensionalized), the validated growth/melt set.
CHI_AA = np.array([[0.0, 0.7248], [0.7248, 0.0]])
CHI_CA = np.array([[0.0, 1.0836], [0.0, 0.0]])
DSIG, DH, TM, EPS2 = 2.6355, 1.3072, 558.0, 1e-3


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _stepper(dm, T, eps2=EPS2, L_psi=5.0, dt=2e-3):
    return MultiPhaseStepper(
        dm, M=1, K=1, chi_aa=CHI_AA, chi_ac=CHI_CA.T.copy(),
        chi_ca=CHI_CA, N=[5.0298, 1.0], onsager=[[0.1]], kappa=[2e-4],
        dsig=[DSIG], dh=[DH], Tm=[TM], eps2=[eps2], L_psi=[L_psi],
        alpha_th=[0.0], beta_th=[0.0], L_th=[5.0], T=T, dt=dt,
        bulk="r14", kg_delta=1e-2, p_floor=1e-6,
        newton_tol=1e-7, newton_max=60, linsolver="cudss")


def _disc(c, r0, w=0.02, amp=0.95):
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


def run_grow_melt(dm, mesh, cons, T, phi0=0.6, r0=0.15, t_end=0.5,
                  dt_max=0.02, device="cuda:0"):
    """Single seeded disc in a uniform undercooled (or superheated)
    blend.  Returns crystalline-area history vs time."""
    st = _stepper(dm, T)
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [_disc((0.5, 0.5), r0)],
                   [lambda x: np.zeros(len(x))])
    nx, ny, ix, iy = _grid_map(mesh)

    def area():
        return float(np.mean(np.asarray(cons.T @ st.psi(0)) > 0.5))

    ts, ars = [0.0], [area()]
    checks = np.linspace(t_end / 30, t_end, 30)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=1500, dt_min=1e-7)
        ts.append(st.t); ars.append(area())
    psf = np.zeros((nx, ny)); psf[ix, iy] = np.asarray(cons.T @ st.psi(0))
    return dict(t=np.array(ts), area=np.array(ars), psi=psf, T=T,
                a0=ars[0], a1=ars[-1], nx=nx, ny=ny)


def run_avrami(dm, mesh, cons, T=250.0, L_psi=11.0, n_seeds=4, r0=0.045,
               t_end=1.8, dt_max=0.02, phi0=0.62, seed=3,
               device="cuda:0"):
    """Multiple pre-placed nuclei with distinct orientation markers.
    Track crystalline fraction X(t) and fit the Avrami exponent.  Deeper
    undercooling (T well below Tm) + a larger kinetic prefactor L_psi
    grow the crystals fast enough that X sweeps the full (0, 1) range,
    so the double-log Avrami fit has a real window (a too-slow run pins
    X near 0 and the exponent is meaningless).  Returns X(t), (n, k),
    and the final psi/theta grids."""
    rng = np.random.default_rng(seed)
    # well-separated random seed centers
    ctrs, tries = [], 0
    while len(ctrs) < n_seeds and tries < 8000:
        c = rng.uniform(0.1, 0.9, size=2)
        if all(np.hypot(c[0] - q[0], c[1] - q[1]) > 0.2 for q in ctrs):
            ctrs.append(c)
        tries += 1
    st = _stepper(dm, T, L_psi=L_psi)
    coords = st.free_coords
    psi0 = np.zeros(st.nfree)
    th0 = np.zeros(st.nfree)
    for k, c in enumerate(ctrs):
        d = _disc(c, r0)(coords)
        m = d > psi0
        psi0[m] = d[m]
        r = np.hypot(coords[:, 0] - c[0], coords[:, 1] - c[1])
        th0[r < r0 + 0.06] = 0.15 + 0.7 * (k / max(n_seeds - 1, 1))
    st.set_initial([lambda x: np.full(len(x), phi0)],
                   [lambda x: psi0], [lambda x: th0])
    nx, ny, ix, iy = _grid_map(mesh)

    def frac():
        return float(np.mean(np.asarray(cons.T @ st.psi(0)) > 0.5))

    ts, X = [0.0], [frac()]
    checks = np.linspace(t_end / 40, t_end, 40)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=2000, dt_min=1e-7)
        ts.append(st.t); X.append(frac())
    ts = np.array(ts); X = np.array(X)
    n, k = _fit_avrami(ts, X)
    psf = np.zeros((nx, ny)); psf[ix, iy] = np.asarray(cons.T @ st.psi(0))
    thf = np.zeros((nx, ny)); thf[ix, iy] = np.asarray(cons.T @ st.theta(0))
    labels, sizes = grain_labels(mesh.node_coords,
                                 np.asarray(cons.T @ st.psi(0)),
                                 np.asarray(cons.T @ st.theta(0)))
    return dict(t=ts, X=X, n_avrami=n, k_avrami=k, psi=psf, theta=thf,
                nx=nx, ny=ny, n_grains=int((sizes > 0).sum()),
                n_seeds=len(ctrs), X_end=float(X[-1]))


def _fit_avrami(t, X):
    """Fit X = 1 - exp[-(k t)^n] via the double-log linearization
    ln(-ln(1-X)) = n ln t + n ln k, over the PRE-IMPINGEMENT growth
    window (0.03 < X < 0.55): the late stage (X -> 1) is impingement-
    dominated and rolls the apparent exponent down, so the clean Avrami
    slope lives in the early-to-mid range."""
    m = (X > 0.03) & (X < 0.55) & (t > 1e-6)
    if m.sum() < 3:
        return float("nan"), float("nan")
    yy = np.log(-np.log(1.0 - X[m]))
    xx = np.log(t[m])
    A = np.vstack([xx, np.ones_like(xx)]).T
    n, b = np.linalg.lstsq(A, yy, rcond=None)[0]
    k = np.exp(b / n) if n != 0 else float("nan")
    return float(n), float(k)
