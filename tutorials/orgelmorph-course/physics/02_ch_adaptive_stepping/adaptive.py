"""OrgElMorph course - Physics P2: Cahn-Hilliard with ADAPTIVE time stepping.

Importable core for the tutorial.  We march the SAME binary spinodal
quench of P1 two ways -- with a FIXED time step, and with the
error-controlled adaptive stepper -- and compare cost (step count) and
accuracy (final morphology + free-energy history).

The physics is unchanged from P1 (conserved Model-B gradient flow of the
Ginzburg-Landau free energy F[c] = INT [ f(c) + (kap/2)|grad c|^2 ] dV);
what changes is HOW we integrate it in time.  Coarsening is a
multi-scale-in-time process: the quench is violent (interfaces form in a
few fast steps) and then coarsening slows down geometrically (domains
merge ever more rarely).  A fixed dt small enough for the quench wastes
enormous effort on the slow tail; an adaptive dt spends steps where the
solution actually changes.

THE CONTROLLER (diffsim.physics.cahn_hilliard.adaptive_march).  Classic
step-doubling local-truncation-error (LTE) control (Wodo & Ganapathy-
subramanian, JCP 230 (2011) 6037): from one state, take one step of size
dt and, separately, two steps of dt/2; the difference estimates the LTE,

    LTE ~ ||c_dt - c_{dt/2}||_2 / (2^p - 1),   p = BDF order.

Accept the (more accurate) half-step solution if LTE < tol, and propose

    dt_new = dt * clamp( safety * (tol/LTE)^{1/(p+1)}, 0.5, 2 ),

otherwise REJECT, halve dt, and retry.  dt GROWS through coarsening --
that is the property that makes long phase-field horizons affordable.

WHY BDF2 MUST BE VARIABLE-COEFFICIENT.  A constant-coefficient BDF2 is
only second-order when the step is constant.  Under an adaptive dt the
step ratio r = dt_n / dt_{n-1} varies, so the stepper rebuilds the BDF2
weights from the ACTUAL (dt_n, dt_{n-1}) every step
(cahn_hilliard.step); r = 1 reproduces the fixed-step 3/2,[2,-1/2]
scheme bit-exactly.  Cross-reference the Computational track (temporal
adaptivity) for the order verification.
"""
import numpy as np

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.cahn_hilliard import CahnHilliardStepper, adaptive_march


def build_mesh_dm(level=6, p=1, device="cuda:0"):
    """Uniform 2-D box, 2^level cells per side (level 6 = 64x64).
    Natural no-flux boundaries -> mass conserved."""
    tree = build_uniform(level, dim=2)
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


# --- total Ginzburg-Landau free energy (the Lyapunov functional) ------
# Same quadrature the solver uses; we only need the TOTAL here (P1
# already split it into bulk + interface), because P2 is about time
# integration, not the energy budget.

def free_energy(st, dm, mesh, cf):
    v, g = st._gp_scalar(cf)                 # c and grad c at Gauss pts
    F = 0.0
    for pv, b in dm.bins.items():
        h = mesh.tree.h()[mesh.bins[pv]]
        ne = len(mesh.conn_of[pv])
        wq = np.tile(dm.tables_by_p[pv].w, ne) \
            * np.repeat((h / 2.0) ** dm.dim, b["nqp"])
        fbulk = 0.25 * (v[pv] ** 2 - 1.0) ** 2         # poly double well
        fint = 0.5 * st.kappa * (g[pv] ** 2).sum(1)
        F += float((wq * (fbulk + fint)).sum())
    return F


def _new_stepper(dm, dt, order, kappa, M, seed, amp, c_avg, device):
    st = CahnHilliardStepper(dm, M, kappa, dt, order=order, energy="poly")
    rng = np.random.default_rng(seed)
    st.set_initial(lambda x: c_avg + amp * rng.standard_normal(len(x)),
                   mu_init="consistent")
    return st


def run_fixed(dm, mesh, dt, t_end, order=2, kappa=5e-4, M=1.0,
              seed=7, amp=0.05, c_avg=0.0, device="cuda:0"):
    """Fixed-dt reference march.  Returns dict with per-step time and
    total free energy, the final field, and the step count."""
    st = _new_stepper(dm, dt, order, kappa, M, seed, amp, c_avg, device)
    ts = [0.0]
    Fs = [free_energy(st, dm, mesh, st.hist[0])]
    nsteps = 0
    while st.t < t_end - 1e-12:
        st.dt = min(dt, t_end - st.t)
        c, _ = st.step()
        nsteps += 1
        ts.append(st.t)
        Fs.append(free_energy(st, dm, mesh, c))
    side = int(round(np.sqrt(len(mesh.node_coords))))
    full = np.asarray(st.cons.T @ st.x[0::2]).reshape(side, side)
    return dict(t=np.array(ts), F=np.array(Fs), nsteps=nsteps,
                field=full, side=side, dt0=dt)


def run_adaptive(dm, mesh, t_end, tol=2e-3, dt0=2e-4, dt_max=0.05,
                 order=2, kappa=8e-4, M=1.0, seed=7, amp=0.05,
                 c_avg=0.0, nchecks=60, device="cuda:0"):
    """Error-controlled march.  We call adaptive_march in short
    checkpoint windows so we can record the free energy and the current
    dt along the way (and accumulate the full accepted-dt history)."""
    st = _new_stepper(dm, dt0, order, kappa, M, seed, amp, c_avg, device)
    ts = [0.0]
    Fs = [free_energy(st, dm, mesh, st.hist[0])]
    dt_at = [st.dt]
    all_dts = []
    checks = np.linspace(t_end / nchecks, t_end, nchecks)
    for tc in checks:
        # adaptive_march advances st in place to tc, returning the
        # accepted (t, dt) pairs it took to get there.
        _, dts = adaptive_march(st, tc, tol=tol, dt_max=dt_max,
                                dt_min=1e-6)
        all_dts.extend(dts)
        ts.append(st.t)
        Fs.append(free_energy(st, dm, mesh, st.x[0::2]))
        dt_at.append(st.dt)
    side = int(round(np.sqrt(len(mesh.node_coords))))
    full = np.asarray(st.cons.T @ st.x[0::2]).reshape(side, side)
    return dict(t=np.array(ts), F=np.array(Fs), dt_at=np.array(dt_at),
                dt_hist=np.array(all_dts), nsteps=len(all_dts),
                field=full, side=side, tol=tol)


def _domain_scale(field):
    """Isotropic characteristic domain size from the first spectral
    moment of the (periodic-extended) 2-D field: smaller = finer.  A
    ROBUST, seed-independent statistic (unlike the pixelwise field)."""
    a = field - field.mean()
    F2 = np.abs(np.fft.fft2(a)) ** 2
    n = field.shape[0]
    kx = np.fft.fftfreq(n) * n
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2)
    F2[0, 0] = 0.0
    kbar = (K * F2).sum() / F2.sum()
    return float(n / max(kbar, 1e-9))


def compare(dm, mesh, t_end=0.8, dt_fixed=4e-3, tol=2e-3, order=2,
            kappa=8e-4, M=1.0, seed=7, device="cuda:0"):
    """Run the NAIVE fixed-dt integrator and the adaptive integrator to
    the same horizon.

    The point of the concept: a single fixed dt cannot serve both the
    violent spinodal quench (which here needs dt ~ 1e-6 to control the
    truncation error) and the slow coarsening tail (which tolerates
    dt ~ 5e-2).  The adaptive controller spends steps where the solution
    changes.  The result is that the naive fixed step is BOTH more
    expensive AND less accurate (its energy lags, because it
    under-resolves the quench), while the adaptive run reaches a more
    relaxed state for less work.

    The EXACT coarsened microstructure depends on the step sequence just
    as it depends on the RNG seed (P1's seed note), so we compare the
    ROBUST statistics -- free energy, domain scale, phase values -- not
    the pixelwise field."""
    fx = run_fixed(dm, mesh, dt_fixed, t_end, order=order, kappa=kappa,
                   M=M, seed=seed, device=device)
    ad = run_adaptive(dm, mesh, t_end, tol=tol, order=order, kappa=kappa,
                      M=M, seed=seed, device=device)
    dF = abs(fx["F"][-1] - ad["F"][-1])
    speedup = fx["nsteps"] / max(ad["nsteps"], 1)
    dt_min = float(ad["dt_hist"].min())
    # steps a fixed dt would need to be SAFE for the whole quench (its
    # stiffest step) -- the cost the adaptive controller avoids.
    implied_fixed = int(round(t_end / dt_min))
    return dict(fixed=fx, adaptive=ad, dF_end=dF, speedup=speedup,
                t_end=t_end, dt_fixed=dt_fixed, tol=tol, dt_min=dt_min,
                implied_fixed=implied_fixed,
                dscale_fixed=_domain_scale(fx["field"]),
                dscale_adapt=_domain_scale(ad["field"]),
                crange_fixed=(float(fx["field"].min()),
                              float(fx["field"].max())),
                crange_adapt=(float(ad["field"].min()),
                              float(ad["field"].max())))
