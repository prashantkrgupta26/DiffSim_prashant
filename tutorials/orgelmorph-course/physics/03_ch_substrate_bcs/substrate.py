"""OrgElMorph course - Physics P3: substrate / surface energy and the
Cahn-Hilliard boundary conditions.

Importable core.  Real organic films are cast on a SUBSTRATE and dry
against AIR: the two interfaces are not neutral -- one component usually
prefers a wall -- and that preference sets the VERTICAL stratification
of the morphology, which in a solar cell decides whether the right
material meets the right electrode.  This chapter adds that physics
through the BOUNDARY CONDITION and maps BOTH Cahn-Hilliard boundary
integrals to the production assembly.

THE TWO CAHN-HILLIARD BOUNDARY CONDITIONS (mixed (phi, mu) form).  The
conserved dynamics is  dphi/dt = div(Lam grad mu),  mu = f'(phi) -
kappa lap phi.  Its weak form has TWO natural boundary integrals, one
per equation:

  (a) MASS-FLUX no-flux, grad(mu).n = 0.  Integrating div(Lam grad mu)
      by parts against a test function on the PHI (mass-balance)
      equation leaves a boundary term  INT_bdy N_a (Lam grad mu).n dS.
      The natural ("do-nothing") condition DROPS that term -> no mass
      leaves the box.  In code this is the ABSENCE of any face term on
      the phi rows: mass is conserved because we never add a phi-row
      boundary integral (see run_harness for the machine-eps drift).

  (b) WALL condition, kappa grad(phi).n + f_w'(phi) = 0.  A SUBSTRATE
      SURFACE ENERGY  F_w = INT_Gamma_w f_w(phi) dS,  f_w = g phi +
      h phi^2, adds a boundary term to the MU equation: IBP of the
      -kappa lap phi term gives INT_bdy N_a kappa grad(phi).n dS, and
      the variation of F_w contributes + INT_Gamma_w N_a f_w'(phi) dS,
      so the natural condition is kappa grad(phi).n = -f_w'(phi).  In
      MultiPhaseStepper (module A2) the mu_i residual gains
      -INT_w N_a (g_i + 2 h_i phi_i) dS via the consistent face mass
      (src/diffsim/physics/multiphase.py _assemble_host, ~L2290).

Both are NATURAL (Neumann-type) conditions; NEITHER exchanges mass, so
the wall energy re-arranges material without a reservoir.  (A third
option, DIRICHLET pinning of phi/mu, DOES exchange mass; it is the
Computational-track boundary-condition chapter.)

We use the (M, K)-generic brick in its simplest form
(diffsim.physics.multiphase.MultiPhaseStepper with M=1, K=0): one
conserved composition phi in (0,1) (the eliminated "solvent" is
phi_s = 1 - phi), Flory-Huggins bulk, lateral-periodic or confined
sides, PLUS the optional substrate/air wall energy.

PREFERRED COMPOSITION.  The bounded quadratic well f_w = g phi + h phi^2
is minimized at phi* = -g/(2h).  g < 0 (with h > 0) puts phi* above the
blend mean -> the wall ENRICHES the component; a shallow well with
phi* below the mean DEPLETES it.  Keeping phi* strictly inside (0, 1)
keeps the [1e-3, 1-1e-3] admissibility projection from ever firing, so
the quadrature mass is conserved to machine precision (run_harness
reports projected_dofs = 0).  These g, h are the surface-field and its
curvature; the brick does NOT implement a contact angle, so we describe
them as the surface-energy asymmetry, not a wetting angle.
"""
import numpy as np
import scipy.sparse as sp

from diffsim.octree.build import build_uniform
from diffsim.mesh.nodes import build_mesh
from diffsim.mesh.constraints import build_constraints
from diffsim.mesh.basis import basis_tables
from diffsim.assembly.operators import DeviceMesh
from diffsim.physics.multiphase import MultiPhaseStepper, _face_mass
from diffsim.diagnostics import conservation as dcons
from diffsim.diagnostics import energy as dgen
from diffsim.diagnostics import admissibility as dadm

SUBSTRATE_FACE = (1, 0)     # axis 1 (y), side 0 (min) = the y=0 substrate
AIR_FACE = (1, 1)           # axis 1 (y), side 1 (max) = the free/air surface


def build_mesh_dm(level=6, p=1, device="cuda:0", lateral_periodic=True):
    """Uniform 2-D box, 2^level cells/side.  y is non-periodic so the
    y=0 edge is a real substrate; x is periodic (a laterally-unbounded
    cast film) unless lateral_periodic=False (confined: no-flux sides)."""
    tree = build_uniform(level, dim=2, periodic=(bool(lateral_periodic), False))
    mesh = build_mesh(tree, p=p)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(p, dim=2), device)
    return dm, mesh, cons


def _chi_aa(chi):
    """2x2 FULL-species chi (includes the eliminated solvent, index 1)."""
    m = np.zeros((2, 2))
    m[0, 1] = m[1, 0] = chi
    return m


class _ProjTrack:
    """Mixin that instruments the admissibility projection so a caller can
    tell whether it ever FIRED (spec P3: report projected_dofs +
    max_correction honestly).  The base ``_project`` clips the Newton
    iterate to [1e-3, 1-1e-3] and rescales the simplex; the committed
    state is therefore always in-bounds, so an *external* bounds check
    can never see the projection act.  We measure it at the source: the
    per-call count of phi entries the box-clip would move (accumulated
    over the whole solve) and the largest single correction.  For an
    interior-phi* well these stay 0 (the field never nears the bounds);
    only a preference that drives phi outside (0, 1) makes them nonzero,
    and that is exactly when mass gets injected."""

    def _reset_proj(self):
        self._proj_fires = 0
        self._proj_maxcorr = 0.0

    def _project(self, x):
        nd = self.ndof
        if not hasattr(self, "_proj_fires"):
            self._reset_proj()
        for i in range(self.M):
            p = x[2 * i::nd]
            corr = np.maximum(np.maximum(1e-3 - p, p - (1.0 - 1e-3)), 0.0)
            nz = int(np.count_nonzero(corr))
            if nz:
                self._proj_fires += nz
                self._proj_maxcorr = max(self._proj_maxcorr,
                                         float(corr.max()))
        return super()._project(x)


class _TrackedStepper(_ProjTrack, MultiPhaseStepper):
    """MultiPhaseStepper with projection instrumentation (single wall)."""


class _OpposingWallStepper(_ProjTrack, MultiPhaseStepper):
    """Substrate wall (base wall_face) PLUS an opposing wall on the AIR
    face.  The base class assembles the substrate-wall natural term
    (module A2); this subclass composes the SAME consistent-face-mass
    term for the air face on top of the base free-space (A, r).  Host
    assembly only (the composition hooks _assemble_host)."""

    def __init__(self, *args, wall2_g, wall2_h, wall2_face=AIR_FACE, **kw):
        super().__init__(*args, **kw)
        assert self.assembly == "host", "opposing wall needs host assembly"
        self._w2_faces, self._w2_M = _face_mass(self.dm, wall2_face)
        self._w2_g, self._w2_h = float(wall2_g), float(wall2_h)
        self._w2_face = wall2_face

    def _assemble_host(self, x, ctx):
        A, r = super()._assemble_host(x, ctx)
        nd, i = self.ndof, 0
        g, h = self._w2_g, self._w2_h
        faces, fM = self._w2_faces, self._w2_M
        nfn = faces.shape[1]
        F2 = np.zeros(self.dm.n_nodes * nd)
        fv = np.asarray(self.Tc @ x[2 * i::nd])
        gd = nd * faces + (2 * i + 1)                 # mu_i rows
        fw = g + 2.0 * h * fv[faces]
        np.add.at(F2, gd.ravel(),
                  np.einsum("fab,fb->fa", fM, fw).ravel())
        r = r + np.asarray(self.Tn.T @ F2)
        if h != 0.0:
            cd = nd * faces + 2 * i                    # phi_i cols
            K2 = sp.coo_matrix(
                ((-2.0 * h * fM).ravel(),
                 (np.repeat(gd, nfn, axis=1).ravel(),
                  np.tile(cd, (1, nfn)).ravel())),
                shape=(self.dm.n_nodes * nd,) * 2).tocsr()
            A = (A + self.Tn.T @ K2 @ self.Tn).tocsr()
        return A, r


class WallDiagnostics:
    """Quadrature-based diagnostics bound to a mesh: the true content
    INT phi dV, the full energy budget F = F_bulk + F_grad + F_wall (all
    by quadrature), the node->grid map, and the projection report.
    Reuses diffsim.diagnostics so the numbers are the shared, unit-tested
    metrics (spec P3: QUADRATURE mass, not nodal-mean-vs-nominal)."""

    def __init__(self, dm, mesh, cons):
        self.dm, self.mesh, self.cons = dm, mesh, cons
        # element Gauss weights x Jacobian per bin (the SAME wq the
        # assembly integrates with: tile(tables.w) * repeat((h/2)^dim))
        self.wJ = {}
        for pv, b in dm.bins.items():
            hh = mesh.tree.h()[mesh.bins[pv]]
            ne = len(mesh.conn_of[pv])
            self.wJ[pv] = (np.tile(dm.tables_by_p[pv].w, ne)
                           * np.repeat((hh / 2.0) ** dm.dim, b["nqp"]))
        self._w_all = np.concatenate([self.wJ[pv] for pv in self.wJ])
        # substrate + air consistent face masses (surface quadrature)
        self.sub_faces, self.sub_M = _face_mass(dm, SUBSTRATE_FACE)
        self.air_faces, self.air_M = _face_mass(dm, AIR_FACE)
        # robust node -> (ix, iy) map on the FULL grid (iy=0 = substrate)
        fc = mesh.node_coords
        self.xs = np.unique(np.round(fc[:, 0], 12))
        self.ys = np.unique(np.round(fc[:, 1], 12))
        self.ix = np.searchsorted(self.xs, np.round(fc[:, 0], 12))
        self.iy = np.searchsorted(self.ys, np.round(fc[:, 1], 12))
        self.nx, self.ny = len(self.xs), len(self.ys)

    # -- content -------------------------------------------------------
    def phi_gp(self, st, i=0):
        v, g = st._gp(st.phi(i))
        return (np.concatenate([v[pv] for pv in v]),
                np.concatenate([g[pv] for pv in g], axis=0))

    def quad_mass(self, st, i=0):
        """True content INT phi_i dV by Gauss-point quadrature."""
        v, _ = self.phi_gp(st, i)
        return dcons.quadrature_mass(v, self._w_all)

    # -- grid / profile ------------------------------------------------
    def grid(self, st, i=0):
        full = np.asarray(self.cons.T @ st.phi(i))
        g = np.zeros((self.nx, self.ny))
        g[self.ix, self.iy] = full
        return g

    def profile(self, st, i=0):
        """Lateral-mean phi per height row (phi(y))."""
        return self.grid(st, i).mean(axis=0)

    # -- wall energy by surface quadrature -----------------------------
    def _wall_energy(self, st, walls, i=0):
        full = np.asarray(self.cons.T @ st.phi(i))
        Fw = 0.0
        for faces, fM, g, h in walls:
            pf = full[faces]
            linw = np.einsum("fab->fa", fM)           # INT N_a dS
            Fw += (g * float((linw * pf).sum())
                   + h * float(np.einsum("fa,fab,fb->", pf, fM, pf)))
        return Fw

    def energy_budget(self, st, chi, kappa, walls, i=0):
        """F = F_bulk + F_grad + F_wall, each by quadrature.  walls is a
        list of (faces, fmass, g, h) surface terms.  Flory-Huggins bulk
        f = phi ln phi + (1-phi) ln(1-phi) + chi phi (1-phi)."""
        v, g = self.phi_gp(st, i)
        eps = 1e-12
        phi = np.clip(v, eps, 1.0 - eps)
        f_bulk = (phi * np.log(phi) + (1 - phi) * np.log(1 - phi)
                  + chi * phi * (1 - phi))
        F_bulk = dgen.integrate_density(f_bulk, self._w_all)
        F_grad = dgen.gradient_energy(g, self._w_all, kappa)
        F_wall = self._wall_energy(st, walls, i)
        return dgen.assemble_total(bulk=F_bulk, grad=F_grad, wall=F_wall)

    # -- admissibility -------------------------------------------------
    def projection_report(self, st, i=0, lo=1e-3, hi=1.0 - 1e-3):
        full = np.asarray(self.cons.T @ st.phi(i))
        return dadm.projection_report(full, lo, hi)


def boundary_layer_thickness(profile, ys, phi_bulk=None):
    """Enrichment boundary-layer thickness at the substrate.

    The excess e(y) = phi(y) - phi_bulk decays away from the wall; we
    report BOTH the 1/e distance (interpolated) and an exponential-fit
    decay length (least squares of log|e| vs y over the sign-consistent
    near-wall region).  Returns (delta_1e, delta_fit, excess0, e_array).
    phi_bulk defaults to the upper-half-column mean (far from the wall).
    """
    prof = np.asarray(profile, float)
    yv = np.asarray(ys, float) - float(np.min(ys))
    if phi_bulk is None:
        phi_bulk = float(prof[len(prof) // 2:].mean())
    exc = prof - phi_bulk
    e0 = float(exc[0])
    # 1/e distance
    delta_1e = np.nan
    if abs(e0) > 1e-9:
        thr = e0 / np.e
        for k in range(1, len(yv)):
            crossed = (exc[k] <= thr) if e0 > 0 else (exc[k] >= thr)
            if crossed:
                denom = exc[k] - exc[k - 1]
                frac = 0.0 if denom == 0 else (thr - exc[k - 1]) / denom
                delta_1e = float(yv[k - 1] + (yv[k] - yv[k - 1]) * frac)
                break
    # exponential fit over the near-wall, above-noise region
    delta_fit = np.nan
    if abs(e0) > 1e-9:
        mask = np.abs(exc) > 0.05 * abs(e0)
        reg = np.where(mask)[0]
        reg = reg[reg < max(2, len(yv) // 2)]
        if len(reg) >= 3:
            slope = np.polyfit(yv[reg], np.log(np.abs(exc[reg])), 1)[0]
            if slope < 0:
                delta_fit = float(-1.0 / slope)
    return delta_1e, delta_fit, e0, exc


def make_stepper(dm, wall_g=0.0, wall_h=0.0, chi=2.2, kappa=1e-3, dt=2e-4,
                 onsager=0.2, wall2_g=None, wall2_h=None,
                 newton_tol=1e-8, newton_max=30, linsolver="splu"):
    """Construct the M=1, K=0 binary-CH stepper with the substrate wall
    energy.  If wall2_g/wall2_h are given, an OPPOSING wall is added on
    the air face (uses the two-wall subclass)."""
    common = dict(dm=dm, M=1, K=0, chi_aa=_chi_aa(chi), N=[1.0, 1.0],
                  onsager=[[onsager]], kappa=[kappa], dt=dt,
                  wall_g=[wall_g], wall_h=[wall_h], wall_face=SUBSTRATE_FACE,
                  newton_tol=newton_tol, newton_max=newton_max,
                  linsolver=linsolver)
    if wall2_g is not None or wall2_h is not None:
        return _OpposingWallStepper(wall2_g=float(wall2_g or 0.0),
                                    wall2_h=float(wall2_h or 0.0),
                                    wall2_face=AIR_FACE, **common)
    return _TrackedStepper(**common)


def simulate(dm, mesh, cons, diag, wall_g=0.0, wall_h=0.0, chi=2.2,
             kappa=1e-3, phi0=0.5, amp=0.02, dt=2e-4, t_end=0.4, seed=5,
             dt_max=0.02, onsager=0.2, wall2_g=None, wall2_h=None,
             linsolver="splu", snap_times=None, record_every=0.05):
    """March a binary quench with the substrate (and optional air) wall
    energy.  Records the TRUE quadrature-mass series, the energy budget
    trajectory, field snapshots, and the final enrichment profile.
    Returns a per-case diagnostics dict."""
    if snap_times is None:
        snap_times = [0.0, t_end / 4.0, t_end]
    st = make_stepper(dm, wall_g=wall_g, wall_h=wall_h, chi=chi, kappa=kappa,
                      dt=dt, onsager=onsager, wall2_g=wall2_g, wall2_h=wall2_h,
                      linsolver=linsolver)
    rng = np.random.default_rng(seed)
    ic = phi0 + amp * rng.standard_normal(st.nfree)
    st.set_initial([lambda x: ic])
    st._reset_proj()                 # count projection firings over the run

    walls = [(diag.sub_faces, diag.sub_M, wall_g, wall_h)]
    if wall2_g is not None or wall2_h is not None:
        walls.append((diag.air_faces, diag.air_M,
                      float(wall2_g or 0.0), float(wall2_h or 0.0)))

    tlog, masses, Ftot, Fbulk, Fgrad, Fwall = [], [], [], [], [], []
    snaps = {}

    def record():
        tlog.append(st.t)
        masses.append(diag.quad_mass(st))
        bud = diag.energy_budget(st, chi, kappa, walls)
        Ftot.append(bud["total"]); Fbulk.append(bud["bulk"])
        Fgrad.append(bud["grad"]); Fwall.append(bud["wall"])

    record()
    if snap_times[0] <= 1e-12:
        snaps[0.0] = diag.grid(st)
    next_snaps = [s for s in snap_times if s > 1e-12]
    st.dt = dt
    stalls = 0
    while st.t < t_end - 1e-12:
        t_before = st.t
        st.march(t_end=min(st.t + record_every, t_end), dt_max=dt_max,
                 max_steps=4000, dt_min=1e-8)
        record()
        while next_snaps and st.t >= next_snaps[0] - 1e-9:
            snaps[next_snaps.pop(0)] = diag.grid(st)
        if st.t <= t_before + 1e-14:      # march could not advance (dt
            stalls += 1                    # underflow / stiff wall) — stop
            if stalls >= 2:
                break
    if t_end not in snaps:
        snaps[t_end] = diag.grid(st)
    proj_dofs = int(getattr(st, "_proj_fires", 0))
    max_corr = float(getattr(st, "_proj_maxcorr", 0.0))

    prof = diag.profile(st)
    delta_1e, delta_fit, e0, excess = boundary_layer_thickness(prof, diag.ys)
    grid = diag.grid(st)
    masses = np.asarray(masses)
    return dict(
        wall_g=wall_g, wall_h=wall_h, wall2_g=wall2_g, wall2_h=wall2_h,
        chi=chi, kappa=kappa, phi0=phi0,
        substrate_phi=float(grid[:, 0].mean()),
        air_phi=float(grid[:, -1].mean()),
        film_mean_phi=float(grid.mean()),
        enrichment=float(grid[:, 0].mean() - grid.mean()),
        phi_star=(float(-wall_g / (2.0 * wall_h)) if wall_h else None),
        mass0=float(masses[0]), mass_final=float(masses[-1]),
        mass_drift=float(dcons.mass_drift(masses)),
        rel_mass_drift=float(dcons.relative_mass_drift(masses)),
        projected_dofs=int(proj_dofs), max_correction=float(max_corr),
        phi_min=float(grid.min()), phi_max=float(grid.max()),
        F_bulk=float(Fbulk[-1]), F_grad=float(Fgrad[-1]),
        F_wall=float(Fwall[-1]), F_total=float(Ftot[-1]),
        F_wall0=float(Fwall[0]),
        delta_1e=float(delta_1e), delta_fit=float(delta_fit),
        bdlayer_e0=float(e0),
        # trajectories / fields for figures
        t=np.asarray(tlog), mass_series=masses,
        F_total_series=np.asarray(Ftot), F_wall_series=np.asarray(Fwall),
        F_bulk_series=np.asarray(Fbulk), F_grad_series=np.asarray(Fgrad),
        profile=prof, excess=excess, ys=np.asarray(diag.ys),
        snaps=snaps, side=diag.nx)


# -- backward-compatible thin wrapper for the student-facing run.py -----
def run(dm, mesh, cons, wall_g=0.0, wall_h=0.0, chi=2.2, kappa=1e-3,
        phi0=0.5, amp=0.02, dt=2e-4, t_end=0.4, seed=5, dt_max=0.02,
        device="cuda:0", snap_times=None, linsolver="splu"):
    """Convenience wrapper used by run.py: builds the diagnostics helper
    and returns simulate()'s record with the legacy field names kept."""
    diag = WallDiagnostics(dm, mesh, cons)
    r = simulate(dm, mesh, cons, diag, wall_g=wall_g, wall_h=wall_h, chi=chi,
                 kappa=kappa, phi0=phi0, amp=amp, dt=dt, t_end=t_end,
                 seed=seed, dt_max=dt_max, linsolver=linsolver,
                 snap_times=snap_times)
    r["wall_phi"] = np.array([r["substrate_phi"]])
    r["bulk_phi"] = np.array([r["film_mean_phi"]])
    r["yprofile"] = r["profile"]
    return r
