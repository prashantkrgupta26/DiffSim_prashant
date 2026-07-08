"""M4 track (c): Wodo & Ganapathysubramanian, Comput. Mater. Sci. 55
(2012) 113-126 — evaporating ternary film (polymer/fullerene/solvent) in
the Landau-mapped frame. Adapts TernaryCHStepper (4-dof monolithic
Newton, C1-regularized FH log).

MODEL (their Eqs. 16-23, nondimensionalized):
  theta = z / h_curr(t): computational domain FIXED (strip in [0,1]^2,
  vertical axis = last axis); physical film height h_curr(t) shrinks.
  Mapped gradient: grad~ = (d/dx, (1/h_curr) d/dtheta) — implemented by
  scaling the vertical component of BOTH the test-function gradients and
  the GP field gradients by minv = 1/h_curr (so the M grad-mu divergence
  and the kappa terms pick up 1/h^2 on the vertical block:
  anisotropic lap = d2/dx2 + (1/h^2) d2/dtheta2).

  d(phi_i)/dt + K (theta/h) d(phi_i)/dtheta = grad~ . (M_ij grad~ mu_j)
  mu_i = dfFH/dphi_i - kap_i lap~ phi_i

SIGN NOTE (load-bearing; the paper's Sec. 5.2 text has a slip): with
h' = dh/dt = -k_e avg(phi_s^top) = -K (K >= 0), the mapping z = theta h
gives phi_t|_z = phi_t|_theta - theta (h'/h) phi_theta, i.e. the LHS
advection coefficient is +K theta/h (features move UP in theta as the
top sweeps down; theta' = -z h'/h^2 = +K theta/h). The top-surface
solute balance (only solvent evaporates, J_i^air = 0) at the interface
receding with h' reads J_i^diff . n = phi_i h' => M grad mu_i . n =
+K phi_i (a natural/Neumann ENRICHMENT flux). With exactly this pair,
d/dt [ h * Int phi_i dtheta ] = 0 (physical solute content conserved,
their footnote 3) — flipping either sign leaks solute at rate 2*K*Phi.

FLORY-HUGGINS with chain lengths (f = sum phi_i/N_i ln phi_i + chi
terms, phi_s = 1 - phi_1 - phi_2 eliminated; exchange potentials):
  mu_i = (1/N_i)(ln phi_i + 1) - (1/N_s)(ln phi_s + 1)
         + chi_12 phi_j + chi_is (phi_s - phi_i) - chi_js phi_j
  d(mu_i)/d(phi_i) = (1/N_i)/phi_i + (1/N_s)/phi_s - 2 chi_is
  d(mu_i)/d(phi_j) = (1/N_s)/phi_s + chi_12 - chi_1s - chi_2s
C1-regularized log kept from the base (linear extension below 1e-4).
Their b*sum(1/phi_i) simplex regularizer (b = 1e-3) is SKIPPED in v1 —
the regularized log already supplies a growing restoring force.

v2 (b_reg > 0) RESTORES their footnote-2 term  f += b sum_i 1/phi_i:
  mu_i += b (1/phi_s^2 - 1/phi_i^2)
  d11  += 2b (1/phi_1^3 + 1/phi_s^3),  d12 += 2b / phi_s^3
(inverses floored at 1e-3 — C0, monotone restoring). NOT just
numerics for the DILUTE initial states of the 2-D campaign: at
phi_p = phi_f = 0.125 the term adds +2b/phi^3 ~= +2.05 to the p-f
exchange curvature, so the Np = 100 blend is STABLE at t = 0 (pure FH:
-0.32, i.e. instant bulk spinodal). With b = 1e-3 (their value) the
bulk quenches only upon enrichment — which arrives top-first =>
surface-directed layering, the paper's Fig 6/7 multilayer mechanism.

MOBILITY v1: constant SPD M (M12 = 0). Their composition-dependent
M_i = D(phi)/f''_ideal(phi_i) with D = sum D_i phi_i, D_p = D_f =
1e-3 D_s is v2. v1 mapping (documented, used by the Fig-3 benchmark):
freeze M at the initial composition, M0 = D(phi^0)/f''_ideal(phi^0) in
units D_s = 1, L = h0 = 1 => time unit h0^2/D_s and Biot Bi = k_e
exactly (their Eq. 33). At the 1D blend (0.2, 0.2, 0.6) this gives
M0 ~= 0.225 and an effective solvent-gradient relaxation diffusivity
M0*(d11 + d12) ~= 0.93 ~ D_s — self-consistent. v1 has no mobility
freeze-out as phi_s -> 0 (their D drops to 1e-3 D_s; noted, v2).

MOBILITY v2 (var_mob=True; M4-c Fig 6/7): their local
M_ii(phi) = D(phi) / f''_ideal,i,  D = phi_s + D_ratio (1 - phi_s),
f''_ideal,i = 1/(N_i phi_i) + 1/(N_s phi_s)  (regularized inverses),
evaluated per GP at the current Newton iterate; the Jacobian keeps M
PICARD-FROZEN (no dM/dphi blocks) — same converged solution, the
Appendix-A heuristic absorbs the odd extra iteration. MEASURED
NECESSITY: without freeze-out the frozen-M film keeps coarsening at
D ~ D0 ~ 0.75 after the solvent is gone (their D -> 0.1 at
phi_s = 0.1) and every 2-D morphology collapses to the equilibrium
bilayer before the stop criterion; the paper's 'morphology frozen'
regime needs D(phi). Langevin noise is scaled by sqrt(M_ii(phi))
in-kernel (local FDT), so fluctuations freeze out with the mobility.

TIME STEPPING: BDF1 + their Appendix-A heuristic (iters < 20 =>
dt *= 1.25; no convergence in 50 (or divergence) => dt *= 0.25, retry).
Per accepted step: h_curr -= dt * K, K frozen at t_n (also frozen over
the Newton solve). No Langevin noise (CHC term) in v1 — separation is
seeded by initial-condition noise only. No SUPG on the advection term
(cell Peclet ~ K*h_el/D_eff ~ 0.05 at Bi = 10, ny = 128; noted).

v1.1 (M4-c Fig 6/7 2-D campaign) GENERALIZED METRIC: the computational
strip may be a sub-rectangle [0, Xcomp] x [0, Ycomp] of the unit-square
octree, representing a physical domain Lx x h_curr:
  physical x = lat_scale * xi_x   (lat_scale = Lx / Xcomp, static)
  theta      = xi_y / Ycomp,  physical z = h_curr * theta
Gradient factors (per-direction ms, applied to test AND trial):
  lateral mlat = 1/lat_scale;  vertical mvert = Ycomp / h_curr.
Advection K theta (1/h) d/dtheta = K * xi_y * (1/h) * d/dxi_y — the
Ycomp factors cancel, so the kernel keeps the RAW xi_y coordinate and a
separate madv = 1/h_curr. Volume integrals carry a constant measure
ratio (Ycomp/lat_scale) vs the mapped-physical ones while the top-edge
boundary integral carries (1/lat_scale) only => the enrichment-flux
coefficient picks up Ycomp: coef = K * (1/h) * Ycomp. With that pair
the conservation identity d/dt [h * Int phi dx dtheta] = 0 is preserved
(defaults lat_scale = Ycomp = 1 reproduce v1 exactly).
Linear solve: linsolver = "splu" (v1 default) or "cudss" — nvmath
DirectSolver, plan once on the first Newton iterate, then
reset_operands(a=..) + refactorize per iterate (fixed sparsity).

v1.2 DEVICE-BOUND MARCH (use_device_assembly=True; M4): the host
COO + scipy assembly is replaced by DeviceNSAssembler(ndof=4) — the
slot-map CSR pattern is built ONCE per mesh; each Newton iterate
launches the CH kernel into Ae/be blocks and scatters them into the
device CSR values via the slot maps, then solves the zero-copy
torch-CSR through nvmath cuDSS (plan once, reset_operands +
refactorize per iterate). TOP-FACE FLUX CHOICE (documented): the
enrichment fluxes are natural-BC (Neumann) ENRICHMENTS of interior
equations, NOT strong rows, so set_strong_rows does not apply; their
O(n_topface) values are computed on host (tiny, closed-form face-mass
products) and folded into asm.vals_d / asm.F_d by a slot scatter-add
before the solve. Host path stays the default. Requires identity
constraints (uniform strips — true for every Wodo mesh). Parity gate:
trajectory agreement < 1e-11 vs the host path (tests/test_wodo_film).

v1.1 CONSERVED LANGEVIN NOISE (their CHC term, Sec. 5.3): stochastic
flux q_i per GP, residual += Int grad~(w) . q_i dV, q_i ~ N(0,1) *
noise * sqrt(2 M_ii / dt) (FDT shape; the absolute nondimensional
amplitude 'noise' is a free knob — the paper's RT/Vs normalization is
absorbed). Load-vector only (frozen over the Newton solve, redrawn per
attempt); GLOBAL solute content is conserved to machine precision by
partition of unity (sum_a grad N_a = 0). noise = 0 reproduces v1.
MEASURED NECESSITY (M4-c Fig 6/7): without it, IC noise diffusively
decays before the N=5 quench crosses the spinodal (phi_s = 0.60) and
separation inherits ONLY the vertical enrichment gradient -> spurious
all-layered morphologies; the paper's percolated N=5 regime needs the
continuous lateral re-seeding.
"""
import numpy as np
import scipy.sparse as sp
import warp as wp

from ..assembly.operators import _kernel_cache
from ..mesh.nodes import _local_offsets
from .ternary_ch import TernaryCHStepper, _rlog, _rinv


@wp.func
def _binv2(x: wp.float64) -> wp.float64:
    # floored 1/x^2 for the b-regularizer (their footnote-2 term)
    y = wp.max(x, wp.float64(1e-3))
    return wp.float64(1.0) / (y * y)


@wp.func
def _binv3(x: wp.float64) -> wp.float64:
    y = wp.max(x, wp.float64(1e-3))
    return wp.float64(1.0) / (y * y * y)


def make_wodo_newton(nbf: int, nqp: int, dim: int):
    """tch_newton + (a) chain-length FH (+ optional b-regularizer),
    (b) anisotropic metric: vertical gradients scaled by mvert =
    Ycomp/h_curr, lateral by mlat = 1/lat_scale, (c) mapped-frame
    advection +K xi_y (1/h) d(phi)/dxi_y (rows 4a+0 / 4a+2 and their
    diagonal Jacobian blocks), (d) optional local mobility
    M_ii = D(phi)/f''_ideal,i (Dr >= 0), (e) conserved Langevin flux
    q_i (load only, scaled by local sqrt(M_ii)). Vertical = dim-1."""
    key = ("wodo_newton", nbf, nqp, dim)
    if key in _kernel_cache:
        return _kernel_cache[key]
    dim_pow = float(dim)
    vax = dim - 1

    @wp.kernel(module="unique", enable_backward=False,
               module_options={"max_unroll": 0})
    def wodo_k(conn: wp.array2d(dtype=wp.int32),
               h: wp.array(dtype=wp.float64),
               Ntab: wp.array2d(dtype=wp.float64),
               dNtab: wp.array3d(dtype=wp.float64),
               wtab: wp.array(dtype=wp.float64),
               p1k: wp.array(dtype=wp.float64),
               gp1k: wp.array2d(dtype=wp.float64),
               m1k: wp.array(dtype=wp.float64),
               gm1k: wp.array2d(dtype=wp.float64),
               p2k: wp.array(dtype=wp.float64),
               gp2k: wp.array2d(dtype=wp.float64),
               m2k: wp.array(dtype=wp.float64),
               gm2k: wp.array2d(dtype=wp.float64),
               h1: wp.array(dtype=wp.float64),
               h2: wp.array(dtype=wp.float64),
               q1: wp.array2d(dtype=wp.float64),
               q2: wp.array2d(dtype=wp.float64),
               theta: wp.array(dtype=wp.float64),
               M11: wp.float64, M12: wp.float64, M22: wp.float64,
               Dr: wp.float64,
               c12: wp.float64, c1s: wp.float64, c2s: wp.float64,
               n1i: wp.float64, n2i: wp.float64, nsi: wp.float64,
               breg: wp.float64,
               kap1: wp.float64, kap2: wp.float64,
               sigma: wp.float64,
               mlat: wp.float64, mvert: wp.float64,
               madv: wp.float64, kadv: wp.float64,
               Ae: wp.array3d(dtype=wp.float64),
               be: wp.array2d(dtype=wp.float64)):
        e = wp.tid()
        he = h[e]
        jac = wp.pow(he * wp.float64(0.5), wp.float64(dim_pow))
        dscale = wp.float64(2.0) / he
        for q in range(nqp):
            dJxW = wtab[q] * jac
            gp = e * nqp + q
            p1 = p1k[gp]
            p2 = p2k[gp]
            ps = wp.float64(1.0) - p1 - p2
            mu1b = n1i * (_rlog(p1) + wp.float64(1.0)) \
                - nsi * (_rlog(ps) + wp.float64(1.0)) \
                + c12 * p2 + c1s * (ps - p1) - c2s * p2 \
                + breg * (_binv2(ps) - _binv2(p1))
            mu2b = n2i * (_rlog(p2) + wp.float64(1.0)) \
                - nsi * (_rlog(ps) + wp.float64(1.0)) \
                + c12 * p1 + c2s * (ps - p2) - c1s * p1 \
                + breg * (_binv2(ps) - _binv2(p2))
            d11 = n1i * _rinv(p1) + nsi * _rinv(ps) \
                - wp.float64(2.0) * c1s \
                + wp.float64(2.0) * breg * (_binv3(p1) + _binv3(ps))
            d22 = n2i * _rinv(p2) + nsi * _rinv(ps) \
                - wp.float64(2.0) * c2s \
                + wp.float64(2.0) * breg * (_binv3(p2) + _binv3(ps))
            d12 = nsi * _rinv(ps) + c12 - c1s - c2s \
                + wp.float64(2.0) * breg * _binv3(ps)
            # mobility: constant (Dr < 0) or their local
            # M_ii = D(phi)/f''_ideal,i (v2; Picard-frozen in Jacobian)
            M11l = M11
            M22l = M22
            if Dr >= wp.float64(0.0):
                Dloc = wp.max(ps + Dr * (p1 + p2), Dr)
                M11l = Dloc / (n1i * _rinv(p1) + nsi * _rinv(ps))
                M22l = Dloc / (n2i * _rinv(p2) + nsi * _rinv(ps))
            s1 = wp.sqrt(M11l)
            s2 = wp.sqrt(M22l)
            # advection coefficient on the RAW xi_y-derivative:
            # K * xi_y * (1/h_curr)  (Ycomp cancels; see docstring)
            adv = kadv * theta[gp] * madv
            for a in range(nbf):
                Na = Ntab[q, a]
                gM1 = wp.float64(0.0)
                gM2 = wp.float64(0.0)
                gP1 = wp.float64(0.0)
                gP2 = wp.float64(0.0)
                gQ1 = wp.float64(0.0)
                gQ2 = wp.float64(0.0)
                for dd in range(dim):
                    ms = mlat
                    if dd == vax:
                        ms = mvert
                    gNa = dNtab[q, a, dd] * dscale * ms
                    gM1 += gNa * (M11l * gm1k[gp, dd]
                                  + M12 * gm2k[gp, dd]) * ms
                    gM2 += gNa * (M12 * gm1k[gp, dd]
                                  + M22l * gm2k[gp, dd]) * ms
                    gP1 += gNa * gp1k[gp, dd] * ms
                    gP2 += gNa * gp2k[gp, dd] * ms
                    gQ1 += gNa * q1[gp, dd] * s1
                    gQ2 += gNa * q2[gp, dd] * s2
                r1 = (Na * (sigma * p1k[gp] - h1[gp]
                            + adv * gp1k[gp, vax]) + gM1 + gQ1) * dJxW
                rm1 = (Na * (m1k[gp] - mu1b)) * dJxW - kap1 * gP1 * dJxW
                r2 = (Na * (sigma * p2k[gp] - h2[gp]
                            + adv * gp2k[gp, vax]) + gM2 + gQ2) * dJxW
                rm2 = (Na * (m2k[gp] - mu2b)) * dJxW - kap2 * gP2 * dJxW
                wp.atomic_add(be, e, 4 * a + 0, -r1)
                wp.atomic_add(be, e, 4 * a + 1, -rm1)
                wp.atomic_add(be, e, 4 * a + 2, -r2)
                wp.atomic_add(be, e, 4 * a + 3, -rm2)
                for b in range(nbf):
                    Nb = Ntab[q, b]
                    lap = wp.float64(0.0)
                    for dd in range(dim):
                        ms = mlat
                        if dd == vax:
                            ms = mvert
                        lap += dNtab[q, a, dd] * dNtab[q, b, dd] \
                            * dscale * dscale * ms * ms
                    NN = Na * Nb * dJxW
                    lapw = lap * dJxW
                    advw = Na * adv * dNtab[q, b, vax] * dscale * dJxW
                    # phi1 row: d/dphi1 (dt + advection), d/dmu1, d/dmu2
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 0,
                                  sigma * NN + advw)
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 1,
                                  M11l * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 0, 4 * b + 3,
                                  M12 * lapw)
                    # mu1 row
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 0,
                                  -d11 * NN - kap1 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 2,
                                  -d12 * NN)
                    wp.atomic_add(Ae, e, 4 * a + 1, 4 * b + 1, NN)
                    # phi2 row
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 2,
                                  sigma * NN + advw)
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 3,
                                  M22l * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 2, 4 * b + 1,
                                  M12 * lapw)
                    # mu2 row
                    wp.atomic_add(Ae, e, 4 * a + 3, 4 * b + 2,
                                  -d22 * NN - kap2 * lapw)
                    wp.atomic_add(Ae, e, 4 * a + 3, 4 * b + 0,
                                  -d12 * NN)
                    wp.atomic_add(Ae, e, 4 * a + 3, 4 * b + 3, NN)

    _kernel_cache[key] = wodo_k
    return wodo_k


class WodoFilmStepper(TernaryCHStepper):
    """Landau-mapped evaporating-film stepper (linear elements, BDF1).

    Extra state: h_curr (physical film height, starts at 1), k_e
    (evaporation rate; Bi = k_e in units D_s = L = 1), chain lengths
    N = (N_1, N_2, N_s). K = k_e * avg(phi_s at the top node row) is
    frozen at t_n for each solve; h_curr -= dt * K on acceptance.
    """

    def __init__(self, dm, chi=(1.0, 0.3, 0.3), N=(5.0, 5.0, 1.0),
                 M=(0.225, 0.0, 0.225), kappa=(2e-4, 2e-4), k_e=1.0,
                 dt=1e-4, newton_tol=1e-9, newton_max=50,
                 lat_scale=1.0, linsolver="splu", noise=0.0,
                 noise_seed=0, var_mob=False, D_ratio=1e-3, b_reg=0.0,
                 use_device_assembly=False):
        super().__init__(dm, chi=chi, M=M, kappa=kappa, dt=dt, order=1,
                         newton_tol=newton_tol, newton_max=newton_max)
        assert dm.mesh.p == 1, "Wodo film v1: linear elements only"
        self.N1, self.N2, self.Ns = (float(n) for n in N)
        self.k_e = float(k_e)
        self.h_curr = 1.0
        self.n_reject = 0
        self.lat_scale = float(lat_scale)
        self.linsolver = linsolver
        self._cudss = None
        self.use_device_assembly = bool(use_device_assembly)
        self._asm = None            # DeviceNSAssembler (lazy, per mesh)
        self._cudss_dev = None      # device-CSR DirectSolver plan
        self.noise = float(noise)
        self._nrng = np.random.default_rng(noise_seed)
        self.var_mob = bool(var_mob)
        self.D_ratio = float(D_ratio)
        self.b_reg = float(b_reg)
        # skip the T4 congruence product when constraints are identity
        # (uniform strips: no hanging nodes, natural BCs only)
        n = self.Tc.shape[0]
        self._proj_identity = (
            self.Tc.shape[0] == self.Tc.shape[1]
            and (self.Tc - sp.identity(n, format="csr")).nnz == 0)
        # theta at GPs (computational vertical coordinate; static)
        self.theta_wp = {
            pv: wp.array(np.ascontiguousarray(self.xq[pv][:, dm.dim - 1]),
                         dtype=wp.float64, device=dm.device)
            for pv in self.xq}
        self._build_top_faces()

    # -- top-surface topology -------------------------------------------
    def _build_top_faces(self):
        """Top-face node lists + CONSISTENT P1 face mass matrices:
        tensor product of the 1-D edge mass le * [[1/3,1/6],[1/6,1/3]]
        over the lateral dims (dim=2: 2-node edges, the historical
        le/6 [[2,1],[1,2]]; dim=3: 4-node quad faces)."""
        coords = self.mesh.node_coords
        vax = self.dm.dim - 1
        ymax = coords[:, vax].max()
        tol = 1e-12
        self.y_comp = float(ymax)     # computational vertical extent
        self.top_nodes = np.where(coords[:, vax] > ymax - tol)[0]
        assert len(self.top_nodes) > 0
        m1 = np.array([[1.0 / 3.0, 1.0 / 6.0],
                       [1.0 / 6.0, 1.0 / 3.0]])
        faces, fmass = [], []
        for pv, conn in self.mesh.conn_of.items():
            offs = _local_offsets(pv, self.dm.dim)
            top_loc = np.where(offs[:, vax] == pv)[0]   # p=1: 2^(d-1)
            of = offs[top_loc][:, :vax]                 # in-face offsets
            Mu = np.ones((len(top_loc), len(top_loc)))
            for dd in range(vax):
                Mu *= m1[of[:, None, dd], of[None, :, dd]]
            nn = conn[:, top_loc]                       # [ne, nfn]
            on_top = np.all(coords[nn, vax] > ymax - tol, axis=1)
            h_el = self.mesh.tree.h()[self.mesh.bins[pv]]
            for e in np.where(on_top)[0]:
                faces.append(nn[e])
                fmass.append(h_el[e] ** vax * Mu)
        self.top_faces = np.asarray(faces, np.int64)     # [ntf, nfn]
        self.top_face_M = np.asarray(fmass, np.float64)  # [ntf,nfn,nfn]

    def _top_phis_avg(self, p1_free, p2_free):
        f1 = np.asarray(self.Tc @ p1_free)
        f2 = np.asarray(self.Tc @ p2_free)
        return float(np.mean(1.0 - f1[self.top_nodes] - f2[self.top_nodes]))

    # -- linear solve (per-Newton-iterate matrix; fixed sparsity) --------
    @staticmethod
    def _cudss_opts():
        from nvmath.sparse.advanced import DirectSolverOptions
        import glob as _glob
        mt = _glob.glob(
            "/home/bglab/Baskar/DiffSim/.venv/lib/python3.12/"
            "site-packages/nvidia/cu12/lib/libcudss_mtlayer_gomp.so*")
        return DirectSolverOptions(multithreading_lib=mt[0]) if mt \
            else None

    def _solve(self, A, r):
        if self.linsolver == "cudss":
            from nvmath.sparse.advanced import DirectSolver
            b = np.ascontiguousarray(r, np.float64)
            try:
                if self._cudss is None:
                    self._cudss = DirectSolver(A, b,
                                               options=self._cudss_opts())
                    self._cudss.plan()
                else:
                    self._cudss.reset_operands(a=A, b=b)
                self._cudss.factorize()
                return np.asarray(self._cudss.solve())
            except Exception:
                # bad state (singular factor / pattern mismatch): drop the
                # plan and signal divergence to the dt heuristic
                self._cudss = None
                return np.full(A.shape[0], np.nan)
        from scipy.sparse.linalg import splu
        return splu(A.tocsc()).solve(r)

    def _solve_device(self, asm):
        """cuDSS on the device-resident torch CSR. STABLE operands:
        the CSR values tensor is a zero-copy dlpack view of
        asm.vals_d and b a persistent device buffer refreshed from
        asm.F_d — the scatter updates values IN PLACE, so the plan is
        made once (fixed sparsity) and each iterate only refactorizes
        (nvmath invalidates the plan if operand buffers change)."""
        import torch
        from nvmath.sparse.advanced import DirectSolver
        try:
            if self._cudss_dev is None:
                A_t, F_t = asm.device_csr()
                self._F_view = F_t              # zero-copy over asm.F_d
                self._b_t = torch.empty_like(F_t)
                self._b_t.copy_(self._F_view)
                self._cudss_dev = DirectSolver(A_t, self._b_t,
                                               options=self._cudss_opts())
                self._cudss_dev.plan()
            else:
                self._b_t.copy_(self._F_view)
            self._cudss_dev.factorize()
            return np.asarray(self._cudss_dev.solve().cpu())
        except Exception:
            self._cudss_dev = None
            return np.full(asm.Nfull, np.nan)

    # -- one implicit solve at frozen (h_curr, K); does NOT commit ------
    def _attempt(self, dt, K):
        if self.use_device_assembly:
            return self._attempt_device(dt, K)
        d = self.dm.device
        sigma = 1.0 / dt
        v1, _ = self._gp(self.hist[0][0])
        v2, _ = self._gp(self.hist[0][1])
        h1_gp = {pv: sigma * v1[pv] for pv in v1}
        h2_gp = {pv: sigma * v2[pv] for pv in v2}
        minv = 1.0 / self.h_curr
        mlat = 1.0 / self.lat_scale
        mvert = self.y_comp * minv
        # surface flux K*(1/h), mapped measure; Ycomp = boundary/volume
        # computational-measure ratio (docstring v1.1)
        coef = K * minv * self.y_comp
        # conserved Langevin flux (FDT shape), frozen over this attempt;
        # kernel multiplies by the LOCAL sqrt(M_ii)
        rho = self.noise * np.sqrt(2.0 / dt)
        q_gp = {}
        for pv, b in self.dm.bins.items():
            ngp = len(self.mesh.conn_of[pv]) * b["nqp"]
            q_gp[pv] = (
                rho * self._nrng.standard_normal((ngp, self.dm.dim)),
                rho * self._nrng.standard_normal((ngp, self.dm.dim)))
        Dr = self.D_ratio if self.var_mob else -1.0
        x = self.x.copy()
        for it in range(self.newton_max):
            fields = [self._gp(x[i::4]) for i in range(4)]
            rows, cols, vals = [], [], []
            F_full = np.zeros(self.dm.n_nodes * 4)
            for pv, b in self.dm.bins.items():
                conn = self.mesh.conn_of[pv].astype(np.int64)
                ne, nbf = conn.shape
                nqp = b["nqp"]
                arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                          dtype=wp.float64, device=d)
                Ae = wp.zeros((ne, 4 * nbf, 4 * nbf), dtype=wp.float64,
                              device=d)
                be = wp.zeros((ne, 4 * nbf), dtype=wp.float64, device=d)
                kk = make_wodo_newton(nbf, nqp, self.dm.dim)
                wp.launch(kk, dim=ne, inputs=[
                    b["conn"], b["h"], b["N"], b["dN"], b["w"],
                    arr(fields[0][0][pv]), arr(fields[0][1][pv]),
                    arr(fields[1][0][pv]), arr(fields[1][1][pv]),
                    arr(fields[2][0][pv]), arr(fields[2][1][pv]),
                    arr(fields[3][0][pv]), arr(fields[3][1][pv]),
                    arr(h1_gp[pv]), arr(h2_gp[pv]),
                    arr(q_gp[pv][0]), arr(q_gp[pv][1]),
                    self.theta_wp[pv],
                    wp.float64(self.M11), wp.float64(self.M12),
                    wp.float64(self.M22), wp.float64(Dr),
                    wp.float64(self.c12),
                    wp.float64(self.c1s), wp.float64(self.c2s),
                    wp.float64(1.0 / self.N1), wp.float64(1.0 / self.N2),
                    wp.float64(1.0 / self.Ns),
                    wp.float64(self.b_reg),
                    wp.float64(self.kap1), wp.float64(self.kap2),
                    wp.float64(sigma), wp.float64(mlat),
                    wp.float64(mvert), wp.float64(minv), wp.float64(K),
                    Ae, be], device=d)
                Aeh, beh = Ae.numpy(), be.numpy()
                gdof = (conn[:, :, None] * 4
                        + np.arange(4)[None, None, :]).reshape(ne, 4 * nbf)
                rows.append(np.repeat(gdof, 4 * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, 4 * nbf)).ravel())
                vals.append(Aeh.ravel())
                np.add.at(F_full, gdof.ravel(), beh.ravel())
            # -- top-surface enrichment load: R_i -= (K/h) Int w phi_i dS
            # (consistent P1 face mass Mf, _build_top_faces).
            # be = -R => F += +(K/h) Mf phi ; Jacobian dR/dphi = -(K/h) Mf.
            f1 = np.asarray(self.Tc @ x[0::4])
            f2 = np.asarray(self.Tc @ x[2::4])
            Mf = coef * self.top_face_M              # [ntf, nfn, nfn]
            nfn = self.top_faces.shape[1]
            for comp, fv in ((0, f1), (2, f2)):
                gd = 4 * self.top_faces + comp       # [ntf, nfn]
                np.add.at(F_full, gd.ravel(),
                          np.einsum("fab,fb->fa", Mf,
                                    fv[self.top_faces]).ravel())
                rows.append(np.repeat(gd, nfn, axis=1).ravel())
                cols.append(np.tile(gd, (1, nfn)).ravel())
                vals.append(-Mf.ravel())
            Kmat = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.dm.n_nodes * 4,) * 2).tocsr()
            if self._proj_identity:
                A, r = Kmat, F_full
            else:
                A = (self.T4.T @ Kmat @ self.T4).tocsr()
                r = np.asarray(self.T4.T @ F_full)
            dx = self._solve(A, r)
            if not np.isfinite(dx).all() or np.abs(dx).max() > 1e6:
                return None, it + 1, False               # diverged
            x = x + dx
            if np.abs(dx).max() < self.newton_tol:
                return x, it + 1, True
        return x, self.newton_max, False                 # no convergence

    # -- device-bound attempt (v1.2): slot-map scatter + cuDSS -----------
    def _init_device_assembly(self):
        """Once per mesh: slot-map CSR pattern (DeviceNSAssembler,
        ndof=4) + top-face flux slot/dof arrays. The flux entries live
        inside top-element blocks, so every (row, col) pair exists in
        the element-pattern CSR."""
        from ..assembly.device_assembly import DeviceNSAssembler
        assert self._proj_identity, (
            "device-bound film v1.2: identity constraints only "
            "(uniform strips; no hanging nodes)")
        self._asm = DeviceNSAssembler(self.dm, ndof=4)
        d = self.dm.device
        ntf, nfn = self.top_faces.shape
        rows, cols, gdofs = [], [], []
        for comp in (0, 2):
            gd = 4 * self.top_faces + comp
            rows.append(np.repeat(gd, nfn, axis=1).ravel())
            cols.append(np.tile(gd, (1, nfn)).ravel())
            gdofs.append(gd.ravel())
        slots = self._asm.csr_slots(np.concatenate(rows),
                                    np.concatenate(cols))
        self._flux_slots_d = wp.array(slots.astype(np.int32),
                                      dtype=wp.int32, device=d)
        self._flux_gdof_d = wp.array(
            np.concatenate(gdofs).astype(np.int32), dtype=wp.int32,
            device=d)
        # per-comp identical base values; scaled by -coef per attempt
        self._flux_base = np.concatenate([self.top_face_M.ravel()] * 2)

    def _attempt_device(self, dt, K):
        """One implicit solve, fully device-bound (v1.2 docstring):
        CH-kernel Ae/be blocks scatter into the device CSR via the
        slot maps; the natural-BC top-face flux (NOT strong rows —
        set_strong_rows does not apply) is a tiny host-values add into
        asm.vals_d / asm.F_d; solve = zero-copy torch CSR -> cuDSS."""
        if self._asm is None:
            self._init_device_assembly()
        asm = self._asm
        d = self.dm.device
        sigma = 1.0 / dt
        v1, _ = self._gp(self.hist[0][0])
        v2, _ = self._gp(self.hist[0][1])
        h1_gp = {pv: sigma * v1[pv] for pv in v1}
        h2_gp = {pv: sigma * v2[pv] for pv in v2}
        minv = 1.0 / self.h_curr
        mlat = 1.0 / self.lat_scale
        mvert = self.y_comp * minv
        coef = K * minv * self.y_comp
        rho = self.noise * np.sqrt(2.0 / dt)
        q_gp = {}
        for pv, b in self.dm.bins.items():
            ngp = len(self.mesh.conn_of[pv]) * b["nqp"]
            q_gp[pv] = (
                rho * self._nrng.standard_normal((ngp, self.dm.dim)),
                rho * self._nrng.standard_normal((ngp, self.dm.dim)))
        Dr = self.D_ratio if self.var_mob else -1.0
        # flux Jacobian values: frozen over the attempt (coef frozen)
        flux_vals_d = wp.array(
            np.ascontiguousarray(-coef * self._flux_base),
            dtype=wp.float64, device=d)
        x = self.x.copy()
        for it in range(self.newton_max):
            fields = [self._gp(x[i::4]) for i in range(4)]
            asm.zero_fill()
            for k_bin, (pv, b, ne, nbf, gdof) in enumerate(asm._bins):
                nqp = b["nqp"]
                arr = lambda a_: wp.array(np.ascontiguousarray(a_),
                                          dtype=wp.float64, device=d)
                Ae = wp.zeros((ne, 4 * nbf, 4 * nbf), dtype=wp.float64,
                              device=d)
                be = wp.zeros((ne, 4 * nbf), dtype=wp.float64, device=d)
                kk = make_wodo_newton(nbf, nqp, self.dm.dim)
                wp.launch(kk, dim=ne, inputs=[
                    b["conn"], b["h"], b["N"], b["dN"], b["w"],
                    arr(fields[0][0][pv]), arr(fields[0][1][pv]),
                    arr(fields[1][0][pv]), arr(fields[1][1][pv]),
                    arr(fields[2][0][pv]), arr(fields[2][1][pv]),
                    arr(fields[3][0][pv]), arr(fields[3][1][pv]),
                    arr(h1_gp[pv]), arr(h2_gp[pv]),
                    arr(q_gp[pv][0]), arr(q_gp[pv][1]),
                    self.theta_wp[pv],
                    wp.float64(self.M11), wp.float64(self.M12),
                    wp.float64(self.M22), wp.float64(Dr),
                    wp.float64(self.c12),
                    wp.float64(self.c1s), wp.float64(self.c2s),
                    wp.float64(1.0 / self.N1), wp.float64(1.0 / self.N2),
                    wp.float64(1.0 / self.Ns),
                    wp.float64(self.b_reg),
                    wp.float64(self.kap1), wp.float64(self.kap2),
                    wp.float64(sigma), wp.float64(mlat),
                    wp.float64(mvert), wp.float64(minv), wp.float64(K),
                    Ae, be], device=d)
                asm.scatter_bin(k_bin, Ae, be)
            # top-face enrichment flux (host values -> device add)
            asm.add_matrix_values(self._flux_slots_d, flux_vals_d)
            f1 = np.asarray(self.Tc @ x[0::4])
            f2 = np.asarray(self.Tc @ x[2::4])
            Mf = coef * self.top_face_M
            load = np.concatenate(
                [np.einsum("fab,fb->fa", Mf, fv[self.top_faces]).ravel()
                 for fv in (f1, f2)])
            asm.add_rhs_values(
                self._flux_gdof_d,
                wp.array(np.ascontiguousarray(load), dtype=wp.float64,
                         device=d))
            dx = self._solve_device(asm)
            if not np.isfinite(dx).all() or np.abs(dx).max() > 1e6:
                return None, it + 1, False               # diverged
            x = x + dx
            if np.abs(dx).max() < self.newton_tol:
                return x, it + 1, True
        return x, self.newton_max, False                 # no convergence

    # -- march loop with the Appendix-A dt heuristic ---------------------
    def march(self, h_min=0.42, phis_stop=0.05, max_steps=20000,
              dh_cap=0.004, dt_min=1e-12, callback=None, wall_cap=None):
        """March until avg phi_s <= phis_stop or h_curr <= h_min.
        dh_cap bounds the per-step height decrement (the h-update is
        explicit). wall_cap (seconds, optional) stops early on wall
        clock. Returns a stop-reason string."""
        import time as _time
        t_wall0 = _time.time()
        reason = "max_steps"
        for _ in range(max_steps):
            if wall_cap is not None and _time.time() - t_wall0 > wall_cap:
                reason = "wall_cap"
                break
            p1n, p2n = self.hist[0]
            phis_avg = float(np.mean(1.0 - np.asarray(self.Tc @ p1n)
                                     - np.asarray(self.Tc @ p2n)))
            if phis_avg <= phis_stop:
                reason = "phis_stop"
                break
            if self.h_curr <= h_min:
                reason = "h_min"
                break
            K = max(self.k_e * self._top_phis_avg(p1n, p2n), 0.0)
            dt_eff = min(self.dt, dh_cap / max(K, 1e-12))
            x_new, iters, ok = self._attempt(dt_eff, K)
            if not ok:
                self.n_reject += 1
                self.dt = dt_eff * 0.25                  # reject + retry
                if self.dt < dt_min:
                    reason = "dt_underflow"
                    break
                continue
            self.x = x_new
            self.hist = [(x_new[0::4].copy(), x_new[2::4].copy()),
                         self.hist[0]]
            self.t += dt_eff
            self.h_curr -= dt_eff * K
            if iters < 20:
                self.dt = dt_eff * 1.25
            else:
                self.dt = dt_eff
            if callback is not None:
                callback(self, K, dt_eff, iters)
        return reason
