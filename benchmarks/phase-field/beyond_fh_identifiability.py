r"""Beyond-Flory-Huggins gauge-identifiability demonstration (roadmap
"Learning rung 2").

THE RECORDED M4 FINDING (m4_learn_fmix.py / m4-milestone-report.md Sec 2): when
you try to learn a free-energy correction ON TOP of Flory-Huggins, a LINEAR-in-c
term in f'(c) is EXACTLY aliased by the FH interaction chi -- the map
(chi, c_linear) slides along a null direction that cancels in mu and the
Jacobian (measured trajectory difference 5e-15).  An optimizer with that mode
free never recovers a unique answer.

THIS SCRIPT demonstrates it deterministically, fit-free, on the verified
Stack-B binary-CH adjoint, and shows the gauge-anchored head (neural_energy)
removes it:

  * UN-ANCHORED parameterisation {chi(=B), gamma_P1, gamma_P2, gamma_P3}: the
    sensitivity d(f')/dB = (1-2c) is proportional to the P1 (linear) basis, so
    the B and gamma_P1 columns of the observation-Jacobian are collinear -> the
    Gramian is (numerically) singular -> the beyond-FH correction is
    unidentifiable.  cond ~ 1/eps.
  * ANCHORED parameterisation {chi(=B), gamma_P2, gamma_P3} (what
    neural_energy.BasisCorrEnergy / NeuralCHEnergy build BY CONSTRUCTION -- the
    P0/P1 gauge modes are simply not representable): full column rank, well
    conditioned, the correction is identifiable.

Then a robust recovery: fit the anchored coefficients from a safely-interior
single trajectory and recover the truth.  A companion gate
tests/test_beyond_fh_identifiability.py asserts (i) the conditioning gap and
(ii) the anchored recovery."""
import numpy as np
import torch

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.adjoint.torch_twin import CHTwin
from diffsim.adjoint.neural_energy import BasisCorrEnergy

torch.set_default_dtype(torch.float64)

TRUTH_COEFFS = (0.30, 0.15)          # beyond-FH truth on the anchored basis
ANCHORED_DEGREES = (2, 3)            # P2, P3 -- gauge-anchored (what the head builds)
UNANCHORED_DEGREES = (1, 2, 3)       # P1, P2, P3 -- includes the aliasing linear mode


def make_dm(level, device="cpu"):
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=2), device)
    return dm, mesh


def interior_ic(coords, c_mean=0.5, amp=0.12, seed=1, clip=(0.30, 0.70)):
    """A safely-interior initial composition (kept well inside [0,1] so the FH
    logs never hit a wall — the identifiability, not a boundary artefact, is
    what we are isolating)."""
    g = np.random.default_rng(seed)
    x, y = coords[:, 0], coords[:, 1]
    f = (c_mean + amp * np.cos(np.pi * x) * np.cos(np.pi * y)
         + 0.25 * amp * np.cos(2 * np.pi * x + g.uniform(0, 1)))
    return np.clip(f, clip[0], clip[1])


def _march_obs(energy, dm, c0_np, n_steps, dt, order, snap_at):
    twin = CHTwin(dm, energy=energy, dt=dt, order=order, device="cpu")
    out = twin.march(torch.tensor(c0_np), None, torch.tensor(1.0),
                     torch.tensor(0.01), {}, n_steps)
    return torch.cat([out[k][0] for k in snap_at])


def observation_jacobian(dm, c0_np, A, B, degrees, coeffs, n_steps, dt, order,
                         snap_at, max_rows=64, seed=0):
    """Columns = d(snapshot dofs)/d{B, gamma_k...} for one trajectory, at the
    given (B, coeffs).  Returns the singular values of the (subsampled) map."""
    en = BasisCorrEnergy(A=A, B=B, degrees=degrees, coeffs=coeffs)
    en.B.requires_grad_(True)
    en.gamma.requires_grad_(True)
    obs = _march_obs(en, dm, c0_np, n_steps, dt, order, snap_at)
    npar = 1 + len(degrees)
    g = np.random.default_rng(seed)
    idx = (np.arange(obs.numel()) if obs.numel() <= max_rows
           else np.sort(g.choice(obs.numel(), max_rows, replace=False)))
    Jc = np.zeros((idx.size, npar))
    for r, i in enumerate(idx):
        gB, gG = torch.autograd.grad(obs[int(i)], [en.B, en.gamma],
                                     retain_graph=True)
        Jc[r, 0] = float(gB)
        Jc[r, 1:] = gG.detach().numpy()
    return np.linalg.svd(Jc, compute_uv=False)


def conditioning(sv):
    sv = np.asarray(sv)
    smin = sv[sv > 0].min() if np.any(sv > 0) else 0.0
    return (sv[0] / smin) if smin > 0 else np.inf


def coverage_singular_values(dm, c0_list, A, B, degrees, coeffs, n_steps, dt,
                             order, snap_at, max_rows=96, seed=0):
    """Singular values of d(snapshots)/d(gamma) — the beyond-FH coefficient
    Jacobian — with the observations stacked across ALL protocols in c0_list.
    FH (A,B) held fixed; only the (gauge-anchored) coefficients vary.  One
    narrow trajectory aliases the higher Legendre modes (near-collinear columns
    over the small composition range); composition-diverse coverage separates
    them."""
    en = BasisCorrEnergy(A=A, B=B, degrees=degrees, coeffs=coeffs)
    en.A.requires_grad_(False)
    en.B.requires_grad_(False)
    en.gamma.requires_grad_(True)
    obs = torch.cat([_march_obs(en, dm, c0, n_steps, dt, order, snap_at)
                     for c0 in c0_list])
    npar = len(degrees)
    g = np.random.default_rng(seed)
    idx = (np.arange(obs.numel()) if obs.numel() <= max_rows
           else np.sort(g.choice(obs.numel(), max_rows, replace=False)))
    Jc = np.zeros((idx.size, npar))
    for r, i in enumerate(idx):
        (gG,) = torch.autograd.grad(obs[int(i)], [en.gamma], retain_graph=True)
        Jc[r] = gG.detach().numpy()
    return np.linalg.svd(Jc, compute_uv=False)


def run_coverage_demo(level=3, n_steps=6, dt=0.01, order=1, verbose=True):
    """The composition-coverage half of rung 2: a higher-degree gauge-anchored
    basis {P2..P5} recovered from ONE narrow trajectory is ill-conditioned (the
    modes alias over the small visited composition range); composition-diverse
    protocols separate them.  Fit-free (conditioning only) — robust."""
    dm, mesh = make_dm(level)
    coords = mesh.node_coords
    snap_at = list(range(1, n_steps, 2))
    A, B = 1.0, 2.5
    degrees = (2, 3, 4, 5)
    coeffs = (0.20, 0.12, 0.08, 0.05)

    # SINGLE shallow protocol: narrow composition band around 0.5
    single = [interior_ic(coords, c_mean=0.5, amp=0.05, seed=1,
                          clip=(0.42, 0.58))]
    # COMPOSITION-DIVERSE: three means, deeper amplitude, wide interior band
    diverse = [interior_ic(coords, c_mean=cm, amp=0.18, seed=sd,
                           clip=(0.15, 0.85))
               for cm, sd in [(0.35, 2), (0.5, 3), (0.65, 4)]]

    def visited(cl):
        a = np.concatenate(cl)
        return float(a.min()), float(a.max())

    sv_s = coverage_singular_values(dm, single, A, B, degrees, coeffs, n_steps,
                                    dt, order, snap_at)
    sv_d = coverage_singular_values(dm, diverse, A, B, degrees, coeffs, n_steps,
                                    dt, order, snap_at)
    cond_s, cond_d = conditioning(sv_s), conditioning(sv_d)
    if verbose:
        v_s, v_d = visited(single), visited(diverse)
        print(f"\n  BEYOND-FH COMPOSITION-COVERAGE (level {level}, "
              f"{n_steps} steps, BDF{order}, basis P2..P5)")
        print(f"  single  visited {v_s[0]:.2f}-{v_s[1]:.2f}  "
              f"cond = {cond_s:.3e}")
        print(f"  diverse visited {v_d[0]:.2f}-{v_d[1]:.2f}  "
              f"cond = {cond_d:.3e}")
        print(f"  coverage improves conditioning {cond_s / cond_d:.1f}x")
    return dict(cond_single=cond_s, cond_diverse=cond_d)


def fit_anchored(dm, c0_np, data, A, B, n_steps, dt, order, snap_at,
                 iters=300, lr=1e-2, log=False):
    """Recover the anchored beyond-FH coefficients from one interior trajectory
    (FH frozen at (A,B)).  Well-posed, so a plain Adam recovers the truth."""
    model = BasisCorrEnergy(A=A, B=B, degrees=ANCHORED_DEGREES)
    model.A.requires_grad_(False)
    model.B.requires_grad_(False)
    twin = CHTwin(dm, energy=model, dt=dt, order=order, device="cpu")
    M, kap = torch.tensor(1.0), torch.tensor(0.01)
    opt = torch.optim.Adam([model.gamma], lr=lr)
    for it in range(iters):
        opt.zero_grad()
        out = twin.march(torch.tensor(c0_np), None, M, kap, {}, n_steps)
        loss = sum(0.5 * ((out[k][0] - data[j]) ** 2).sum()
                   for j, k in enumerate(snap_at))
        loss.backward()
        opt.step()
        if log and (it % 50 == 0 or it == iters - 1):
            print(f"    it {it:4d}  loss {float(loss.detach()):.3e}  "
                  f"gamma {model.gamma.detach().numpy()}")
    return model


def run_demo(level=3, n_steps=6, dt=0.01, order=1, fit_iters=300, verbose=True):
    dm, mesh = make_dm(level)
    coords = mesh.node_coords
    snap_at = list(range(1, n_steps, 2))
    A, B = 1.0, 2.5
    c0 = interior_ic(coords, c_mean=0.5, amp=0.12, seed=1)

    # --- conditioning: un-anchored (with P1) vs anchored (P2,P3) -----------
    sv_un = observation_jacobian(dm, c0, A, B, UNANCHORED_DEGREES,
                                 (0.0, *TRUTH_COEFFS), n_steps, dt, order,
                                 snap_at)
    sv_an = observation_jacobian(dm, c0, A, B, ANCHORED_DEGREES,
                                 TRUTH_COEFFS, n_steps, dt, order, snap_at)
    cond_un, cond_an = conditioning(sv_un), conditioning(sv_an)

    # --- recovery on the anchored (identifiable) parameterisation ----------
    truth = BasisCorrEnergy(A=A, B=B, degrees=ANCHORED_DEGREES,
                            coeffs=TRUTH_COEFFS)
    for p in truth.parameters():
        p.requires_grad_(False)
    data = [x.detach() for x in
            [_march_obs(truth, dm, c0, n_steps, dt, order, [k])
             for k in snap_at]]
    data = [d.reshape(-1) for d in data]
    model = fit_anchored(dm, c0, data, A, B, n_steps, dt, order, snap_at,
                         iters=fit_iters, log=verbose)
    g_truth = np.array(TRUTH_COEFFS)
    coeff_err = float(np.linalg.norm(model.gamma.detach().numpy() - g_truth)
                      / np.linalg.norm(g_truth))

    if verbose:
        print(f"\n  BEYOND-FH GAUGE-IDENTIFIABILITY (level {level}, "
              f"{n_steps} steps, BDF{order})")
        print(f"  un-anchored {{B,P1,P2,P3}} cond = {cond_un:.3e}   "
              f"(B and P1 aliased -> singular)")
        print(f"  anchored    {{B,P2,P3}}    cond = {cond_an:.3e}   "
              f"(gauge-anchored -> well posed)")
        print(f"  gap = {cond_un / cond_an:.2e}x")
        print(f"  anchored recovery: truth {g_truth} -> "
              f"{model.gamma.detach().numpy()}  rel_err {coeff_err:.3e}")
    return dict(cond_unanchored=cond_un, cond_anchored=cond_an,
                coeff_err=coeff_err)


if __name__ == "__main__":
    run_demo(level=4, n_steps=8, verbose=True)
