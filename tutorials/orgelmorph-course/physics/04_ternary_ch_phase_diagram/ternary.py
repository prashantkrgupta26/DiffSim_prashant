"""OrgElMorph course - Physics P4: ternary Cahn-Hilliard and the Gibbs
phase diagram.

Importable core.  Real cast blends are (at least) THREE components:
two solutes -- a polymer/small-molecule donor and a fullerene acceptor
-- plus a solvent.  With two independent conserved compositions the
free-energy landscape is a SURFACE over the Gibbs triangle, and demixing
follows TIE-LINES on that triangle rather than the single axis of the
binary case.  This concept runs the coupled ternary Cahn-Hilliard system
and visualizes the composition trajectory on the ternary phase diagram
-- the natural way to read multi-component morphology.

We use the (M, K)-generic brick with M=2 solutes and K=0 (no crystal):
diffsim.physics.multiphase.MultiPhaseStepper.  Fields (phi_1, mu_1,
phi_2, mu_2) with the solvent eliminated (phi_s = 1 - phi_1 - phi_2),
Flory-Huggins exchange chemical potentials, a symmetric Onsager mobility
coupling the two solute fluxes, and gradient penalties kap_i.  (This is
the same physics as diffsim.physics.ternary_ch.TernaryCHStepper; we use
the multiphase brick so the fast cuDSS solver and the reject ladder are
available.)  The interaction matrix chi_aa[i,j] holds the pairwise
Flory chi (index 2 = the eliminated solvent): chi_aa[0,1] = solute-
solute, chi_aa[0,2]/chi_aa[1,2] = the two solute-solvent chis.

THE GIBBS TRIANGLE.  Each mesh node has a composition (phi1, phi2, phis)
with phi1+phi2+phis = 1 -- a point in the 2-simplex, drawn in an
equilateral triangle (barycentric coordinates).  The BULK average is
conserved (it never moves), but the CLOUD of local compositions starts
as a tight blob at the initial point and, as the blend demixes, spreads
along a tie-line toward the two coexisting phases.  Watching that cloud
open up IS watching ternary phase separation.
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


def _chi_aa(c12, c1s, c2s):
    """3x3 FULL-species chi (0, 1 solutes; 2 = eliminated solvent)."""
    m = np.zeros((3, 3))
    m[0, 1] = m[1, 0] = c12
    m[0, 2] = m[2, 0] = c1s
    m[1, 2] = m[2, 1] = c2s
    return m


def bary_to_xy(phi1, phi2):
    """Map (phi1, phi2, phis=1-phi1-phi2) to 2-D triangle coordinates.
    Vertices: solvent=(0,0), phi1=(1,0), phi2=(0.5, sqrt3/2)."""
    phi1 = np.asarray(phi1); phi2 = np.asarray(phi2)
    x = phi1 * 1.0 + phi2 * 0.5
    y = phi2 * (np.sqrt(3.0) / 2.0)
    return x, y


def _quad_content(st):
    """Per-solute quadrature content INT phi_i dV and the domain volume
    INT dV, computed on the SAME Gauss points the assembly integrates on
    (weights = tabulated w x geometric Jacobian, st._wJ) -- not a nodal
    mean.  Returns (content[M], volume)."""
    content = []
    for i in range(st.M):
        v, _ = st._gp(st.phi(i))
        content.append(sum(float((st._wJ[pv] * v[pv]).sum()) for pv in v))
    vol = sum(float(st._wJ[pv].sum()) for pv in st._wJ)
    return np.array(content), vol


def run(dm, mesh, cons, chi=(3.5, 1.0, 0.6), mobility=0.2,
        kappa=(6e-4, 6e-4), phi0=(0.35, 0.35), amp=0.02, dt=5e-4,
        t_end=0.6, dt_max=0.02, seed=6, N=(1.0, 1.0, 1.0),
        device="cuda:0", snap_times=None, linsolver="cudss"):
    # spinodal check: at phi0 the 2x2 exchange Hessian
    # [[1/(N1 p1)+1/(Ns ps)-2c1s, 1/(Ns ps)+c12-c1s-c2s],
    #  [., 1/(N2 p2)+1/(Ns ps)-2c2s]] must be indefinite (det<0) for the
    # blend to demix -- chi_12=3.5 at (0.35,0.35), N=(1,1,1) gives det<0
    # (see spinodal_hessian + the chapter).
    """March a ternary spinodal quench.  Returns per-node composition
    clouds at snapshot times, the (conserved) mean trajectory, and the
    per-solute QUADRATURE content series (for the conservation report).
    ``N`` is the degree-of-polymerization triple (solute1, solute2,
    solvent); unequal entries shift the spinodal (see the N-shift study)."""
    if snap_times is None:
        snap_times = [0.0, t_end / 8, t_end / 3, t_end]
    ons = mobility * np.eye(2)
    st = MultiPhaseStepper(
        dm, M=2, K=0, chi_aa=_chi_aa(*chi), N=list(N),
        onsager=ons.tolist(), kappa=list(kappa), dt=dt, bulk="p1",
        newton_tol=1e-8, newton_max=40, linsolver=linsolver)
    rng = np.random.default_rng(seed)
    ic1 = phi0[0] + amp * rng.standard_normal(st.nfree)
    ic2 = phi0[1] + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: ic1, lambda x: ic2])

    side = int(round(np.sqrt(len(mesh.node_coords))))

    def clouds():
        p1 = np.asarray(cons.T @ st.phi(0))
        p2 = np.asarray(cons.T @ st.phi(1))
        return p1, p2

    snaps = {}
    tlog, m1, m2 = [], [], []
    c1, c2 = [], []               # per-solute quadrature content series

    def record():
        p1, p2 = clouds()
        content, vol = _quad_content(st)
        tlog.append(st.t)
        m1.append(float(p1.mean())); m2.append(float(p2.mean()))
        c1.append(float(content[0])); c2.append(float(content[1]))
        return vol

    vol = record()
    if snap_times[0] <= 1e-12:
        p1, p2 = clouds()
        snaps[0.0] = (p1.copy(), p2.copy())
    pending = [s for s in snap_times if s > 1e-12]
    checks = np.linspace(t_end / 40, t_end, 40)
    for tc in checks:
        st.march(t_end=tc, dt_max=dt_max, max_steps=2000, dt_min=1e-8,
                 grow_iters=20)
        vol = record()
        while pending and st.t >= pending[0] - 1e-9:
            p1, p2 = clouds()
            snaps[pending.pop(0)] = (p1.copy(), p2.copy())
    if t_end not in snaps:
        p1, p2 = clouds()
        snaps[t_end] = (p1.copy(), p2.copy())

    p1f, p2f = clouds()
    spread0 = float(np.std(snaps[0.0][0]))
    spreadf = float(np.std(p1f))
    return dict(snaps=snaps, side=side, chi=tuple(chi), phi0=tuple(phi0),
                N=tuple(N), t=np.array(tlog), mean1=np.array(m1),
                mean2=np.array(m2), content1=np.array(c1),
                content2=np.array(c2), volume=float(vol),
                p1f=p1f, p2f=p2f, spread0=spread0, spreadf=spreadf,
                mass1_drift=abs(m1[-1] - m1[0]),
                mass2_drift=abs(m2[-1] - m2[0]))


# =====================================================================
# ANALYSIS  (pure numpy -- kept OUT of src/; importable + unit-testable)
# =====================================================================

def free_energy(phi1, phi2, chi, N):
    """Ternary Flory--Huggins free-energy density (kT/v0 units):
      f = sum_i (phi_i/N_i) ln phi_i + sum_{i<j} chi_ij phi_i phi_j,
    with phi_s = 1 - phi1 - phi2.  chi = (chi12, chi1s, chi2s);
    N = (N1, N2, Ns).  NaN where outside the open simplex."""
    phi1 = np.asarray(phi1, float); phi2 = np.asarray(phi2, float)
    phis = 1.0 - phi1 - phi2
    c12, c1s, c2s = chi
    N1, N2, Ns = N
    with np.errstate(invalid="ignore", divide="ignore"):
        S = (phi1 / N1) * np.log(phi1) + (phi2 / N2) * np.log(phi2) \
            + (phis / Ns) * np.log(phis)
        H = c12 * phi1 * phi2 + c1s * phi1 * phis + c2s * phi2 * phis
        f = S + H
    bad = (phi1 <= 0) | (phi2 <= 0) | (phis <= 0)
    return np.where(bad, np.nan, f)


def spinodal_hessian(phi1, phi2, chi, N):
    """2x2 exchange Hessian d^2 f / d phi_i d phi_j (solvent eliminated):
      H11 = 1/(N1 phi1) + 1/(Ns phis) - 2 chi_1s
      H22 = 1/(N2 phi2) + 1/(Ns phis) - 2 chi_2s
      H12 = 1/(Ns phis) + chi_12 - chi_1s - chi_2s.
    Returns (H, det, eigvals).  det<0 => saddle => the blend is inside
    the spinodal and will demix."""
    c12, c1s, c2s = chi
    N1, N2, Ns = N
    phis = 1.0 - phi1 - phi2
    inv_s = 1.0 / (Ns * phis)
    H11 = 1.0 / (N1 * phi1) + inv_s - 2.0 * c1s
    H22 = 1.0 / (N2 * phi2) + inv_s - 2.0 * c2s
    H12 = inv_s + c12 - c1s - c2s
    H = np.array([[H11, H12], [H12, H22]])
    det = H11 * H22 - H12 * H12
    return H, float(det), np.linalg.eigvalsh(H)


def spinodal_chi12_crit(phi1, phi2, chi, N):
    """Critical chi_12 at which det(H)=0 for fixed (phi, chi_1s, chi_2s,
    N): det = H11 H22 - (inv_s + chi12 - c1s - c2s)^2 = 0, so
    chi_12* = sqrt(H11 H22) - (inv_s - c1s - c2s).  A blend with
    chi_12 > chi_12* is unstable; the shift of chi_12* with N is the
    N-dependence of the spinodal at this composition."""
    c12, c1s, c2s = chi
    N1, N2, Ns = N
    phis = 1.0 - phi1 - phi2
    inv_s = 1.0 / (Ns * phis)
    H11 = 1.0 / (N1 * phi1) + inv_s - 2.0 * c1s
    H22 = 1.0 / (N2 * phi2) + inv_s - 2.0 * c2s
    prod = H11 * H22
    if prod < 0:                  # already unstable for any chi12
        return float("-inf")
    return float(np.sqrt(prod) - (inv_s - c1s - c2s))


def spinodal_maps(chi, N, n=240):
    """Grid the Gibbs triangle: return (phi1, phi2, f, det, min_eig)
    on an n x n (phi1, phi2) lattice (NaN outside the open simplex).
    det<0 marks the spinodal-unstable region; sign(min_eig) is the
    Hessian eigenvalue sign map."""
    g = np.linspace(1e-3, 1.0 - 1e-3, n)
    P1, P2 = np.meshgrid(g, g)
    phis = 1.0 - P1 - P2
    inside = phis > 1e-3
    f = free_energy(P1, P2, chi, N)
    c12, c1s, c2s = chi
    N1, N2, Ns = N
    with np.errstate(invalid="ignore", divide="ignore"):
        inv_s = 1.0 / (Ns * phis)
        H11 = 1.0 / (N1 * P1) + inv_s - 2.0 * c1s
        H22 = 1.0 / (N2 * P2) + inv_s - 2.0 * c2s
        H12 = inv_s + c12 - c1s - c2s
        det = H11 * H22 - H12 * H12
        tr = H11 + H22
        disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
        min_eig = 0.5 * (tr - disc)
    det = np.where(inside, det, np.nan)
    min_eig = np.where(inside, min_eig, np.nan)
    return P1, P2, f, det, min_eig


# ---- 2-component Gaussian mixture (EM), full 2x2 covariances ---------
def gmm2(X, iters=200, tol=1e-7, seed=0):
    """Fit a 2-component Gaussian mixture to X (n x 2) by EM.  Returns
    dict(means[2,2], covs[2,2,2], weights[2], labels[n], resp[n,2]).
    Seeded by a phi1-median split (deterministic); no sklearn."""
    X = np.asarray(X, float)
    n = len(X)
    m0 = X[X[:, 0] <= np.median(X[:, 0])].mean(0)
    m1 = X[X[:, 0] > np.median(X[:, 0])].mean(0)
    means = np.array([m0, m1])
    covs = np.array([np.cov(X.T) + 1e-6 * np.eye(2)] * 2)
    w = np.array([0.5, 0.5])
    ll_old = -np.inf
    resp = np.full((n, 2), 0.5)
    for _ in range(iters):
        # E-step
        logp = np.empty((n, 2))
        for k in range(2):
            d = X - means[k]
            ic = np.linalg.inv(covs[k])
            q = np.einsum("ni,ij,nj->n", d, ic, d)
            logdet = np.log(max(np.linalg.det(covs[k]), 1e-300))
            logp[:, k] = np.log(w[k] + 1e-300) - 0.5 * (q + logdet
                                                        + 2 * np.log(2 * np.pi))
        mx = logp.max(1, keepdims=True)
        pr = np.exp(logp - mx)
        resp = pr / pr.sum(1, keepdims=True)
        ll = float((mx.ravel() + np.log(pr.sum(1))).sum())
        # M-step
        Nk = resp.sum(0) + 1e-12
        w = Nk / n
        for k in range(2):
            means[k] = (resp[:, k:k + 1] * X).sum(0) / Nk[k]
            d = X - means[k]
            covs[k] = (resp[:, k].reshape(-1, 1, 1)
                       * np.einsum("ni,nj->nij", d, d)).sum(0) / Nk[k] \
                + 1e-6 * np.eye(2)
        if abs(ll - ll_old) < tol * max(1.0, abs(ll_old)):
            break
        ll_old = ll
    labels = resp.argmax(1)
    # order phases so phase A is the phi1-poor one (stable labelling)
    if means[0, 0] > means[1, 0]:
        means = means[::-1].copy(); covs = covs[::-1].copy()
        w = w[::-1].copy(); resp = resp[:, ::-1].copy()
        labels = 1 - labels
    return dict(means=means, covs=covs, weights=w, labels=labels, resp=resp)


def interface_mask(p1, p2, side, keep=0.5):
    """Boolean mask of BULK (low-gradient) nodes: exclude the fraction
    (1-keep) of nodes with the largest |grad phi| (interfacial points,
    whose local composition is a mixing transient, not a bulk-phase
    composition).  Gradient by finite differences on the uniform grid."""
    g1 = np.gradient(p1.reshape(side, side))
    g2 = np.gradient(p2.reshape(side, side))
    gm = np.sqrt(g1[0] ** 2 + g1[1] ** 2 + g2[0] ** 2 + g2[1] ** 2).ravel()
    thr = np.quantile(gm, keep)
    return gm <= thr


def cluster_phases(p1, p2, side, keep=0.5, seed=0):
    """Cluster the local-composition cloud (p1, p2) into the two
    coexisting phases and report endpoints + a method sensitivity.

    Method 1 (primary): 2-component GMM on ALL nodes -> per-phase MEAN,
    2x2 COVARIANCE and POPULATION fraction.
    Method 2 (sensitivity): GMM on interface-EXCLUDED (low-|grad|) nodes
    only -- the bulk-phase compositions.  The endpoint shift between the
    two is the reported clustering sensitivity."""
    X = np.column_stack([p1, p2])
    gm = gmm2(X, seed=seed)
    A, B = gm["means"][0], gm["means"][1]
    mask = interface_mask(p1, p2, side, keep=keep)
    gm2_ = gmm2(X[mask], seed=seed)
    A2, B2 = gm2_["means"][0], gm2_["means"][1]
    sens = float(max(np.linalg.norm(A - A2), np.linalg.norm(B - B2)))
    return {
        "phaseA_mean": [float(A[0]), float(A[1])],
        "phaseB_mean": [float(B[0]), float(B[1])],
        "phaseA_cov": gm["covs"][0].tolist(),
        "phaseB_cov": gm["covs"][1].tolist(),
        "population": [float(gm["weights"][0]), float(gm["weights"][1])],
        "phaseA_mean_iface_excl": [float(A2[0]), float(A2[1])],
        "phaseB_mean_iface_excl": [float(B2[0]), float(B2[1])],
        "population_iface_excl": [float(gm2_["weights"][0]),
                                  float(gm2_["weights"][1])],
        "endpoint_sensitivity": sens,
        "bulk_fraction_kept": float(mask.mean()),
    }


def predict_binodal(chi, N, mbar, seedA, seedB, lever0=0.5):
    """Predicted coexistence tie-line by the equal-chemical-potential /
    equal-osmotic (common-tangent) conditions, pinned to pass through the
    conserved mean ``mbar`` (lever rule).  Unknowns (a1,a2,b1,b2,lam);
    residuals: mu1(a)=mu1(b), mu2(a)=mu2(b), omega(a)=omega(b) with
    omega=f-mu.phi, and lam*a+(1-lam)*b=mbar.  Newton-solved with the
    simulated cluster means as the seed.  Returns endpoints + lever
    fraction, or None if it fails to converge."""
    from scipy.optimize import least_squares
    c12, c1s, c2s = chi
    N1, N2, Ns = N
    eps = 1e-4

    def _clip(p1, p2):
        p1 = min(max(p1, eps), 1 - 2 * eps)
        p2 = min(max(p2, eps), 1 - 2 * eps)
        phis = max(1.0 - p1 - p2, eps)
        return p1, p2, phis

    def mus(p1, p2):
        p1, p2, phis = _clip(p1, p2)
        mu1 = (np.log(p1) / N1 + 1.0 / N1) - (np.log(phis) / Ns + 1.0 / Ns) \
            + c12 * p2 + c1s * (phis - p1) - c2s * p2
        mu2 = (np.log(p2) / N2 + 1.0 / N2) - (np.log(phis) / Ns + 1.0 / Ns) \
            + c12 * p1 + c2s * (phis - p2) - c1s * p1
        return mu1, mu2

    def omega(p1, p2):
        p1c, p2c, phis = _clip(p1, p2)
        S = (p1c / N1) * np.log(p1c) + (p2c / N2) * np.log(p2c) \
            + (phis / Ns) * np.log(phis)
        f = S + c12 * p1c * p2c + c1s * p1c * phis + c2s * p2c * phis
        m1, m2 = mus(p1, p2)
        return f - m1 * p1c - m2 * p2c

    def F(z):
        a1, a2, b1, b2, lam = z
        mA = mus(a1, a2); mB = mus(b1, b2)
        return [mA[0] - mB[0], mA[1] - mB[1],
                omega(a1, a2) - omega(b1, b2),
                lam * a1 + (1 - lam) * b1 - mbar[0],
                lam * a2 + (1 - lam) * b2 - mbar[1]]

    z0 = [seedA[0], seedA[1], seedB[0], seedB[1], lever0]
    lo = [eps, eps, eps, eps, 0.0]
    hi = [1 - 2 * eps, 1 - 2 * eps, 1 - 2 * eps, 1 - 2 * eps, 1.0]
    z0 = [min(max(v, l), h) for v, l, h in zip(z0, lo, hi)]
    try:
        sol = least_squares(F, z0, bounds=(lo, hi), xtol=1e-12,
                            ftol=1e-12, max_nfev=2000)
    except Exception:
        return None
    a1, a2, b1, b2, lam = sol.x
    res = float(np.linalg.norm(F(sol.x)))
    ok = (res < 1e-6 and 0 < lam < 1
          and (1 - a1 - a2) > eps and (1 - b1 - b2) > eps
          and (abs(a1 - b1) + abs(a2 - b2)) > 1e-3)
    if not ok:
        return None
    return {"phaseA": [float(a1), float(a2)], "phaseB": [float(b1), float(b2)],
            "lever": float(lam), "residual": res}
