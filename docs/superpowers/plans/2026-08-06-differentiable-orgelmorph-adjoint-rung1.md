# Differentiable OrgElMorph — Rung 1 (CPU M-component adjoint) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a three-way-verified discrete IFT adjoint through the M-component multi-Cahn-Hilliard BDF march, giving `dJ/dp` for design parameters (χ-matrix, N, mobility, κ) of an arbitrary morphology objective, wired into `daisy-morph`'s reserved `gradients=` knob.

**Architecture:** Reuse the proven binary pattern (`src/diffsim/adjoint/phasefield.py`): a self-contained vectorised-numpy discrete forward + Jacobian (so cotangents flow and gradients are three-way verifiable), generalized from a 2-DOF `(c, μ)` node block to a `2M`-DOF `[(φ₀,μ₀),…,(φ_{M-1},μ_{M-1})]` block, with the bulk free energy behind a `MultiEnergy` protocol so free-energy learning (M6) reuses the identical cotangent path. The reference forward achieves parity with the production `MultiPhaseStepper` (K=0) at Newton tolerance.

**Tech Stack:** Python, numpy, scipy.sparse (`splu`), torch (autograd twin), pytest. DiffSim mesh stack (`DeviceMesh`, `basis_tables`, `build_uniform`, `build_mesh`, `build_constraints`).

**Design spec:** `docs/superpowers/specs/2026-08-06-differentiable-orgelmorph-adjoint-design.md`

## Global Constraints

- **Physics scope v1:** phase-separation only (K=0). No ψ/θ. Bulk mode `"p1"`, mobility `mob="const"`.
- **Gate scope:** uniform mesh (`constraints.T == identity`), natural no-flux BCs (no Dirichlet rows), BDF1 and variable-coefficient BDF2. (Mirrors the binary gate scope.)
- **Verification is the deliverable:** every gradient is **three-way verified** — hand IFT adjoint == torch autograd twin == central finite difference. Tolerances: `adj/twin < 1e-10`, `adj/fd < 1e-6` (same locks as `tests/test_phasefield_adjoint.py`).
- **Parity target:** `diffsim.physics.multiphase.MultiPhaseStepper` at K=0, `bulk="p1"`, `mob="const"`, `linsolver="splu"`. The numpy production bulk reference is `multiphase.np_potentials` (K=0 branch).
- **DOF layout:** node-major `2M` block; global dof of (node `a`, field `f`) is `a*2M + f`, with `f = 2i` → φᵢ, `f = 2i+1` → μᵢ (i = 0..M-1). Solvent `φ_s = 1 − Σφᵢ` eliminated (species index M).
- **House rules:** measure-then-lock; commit-per-green; findings-log honesty; mobility closure stays NAMED (`const`); Gramian-before-compute on the inverse-design demo; no dyadic feature dims in gates.
- **Interior-field gates:** keep test ICs interior (φᵢ ∈ ~[0.05, 0.9], Σφ < 0.95) so the production log regularisation (`np_rlog` eps=1e-4) is inactive and derivatives are the exact logs.
- **Branch:** DiffSim work on `diff-orgelmorph` (already created off `master`). daisy-morph work on a `feat/gradients` branch in `../DAISY/daisy-morph`.
- **Rung scope:** this plan is **rung 1 only** (CPU, reference-grade ≤128²). Rung 2 (GPU 256×128) and the mean-φ initial-condition gradient are separate follow-on plans named in the spec.

---

### Task 1: `MultiEnergy` protocol + `FHMultiEnergy` (M-component bulk free energy)

The bulk free energy behind a protocol, with analytic φ- and parameter-derivatives, unit-gated against complex-step differentiation *before* it touches a mesh. At K=0 the exchange potential is exactly `multiphase.np_potentials`'s K=0 branch.

**Files:**
- Create: `src/diffsim/adjoint/multiphase.py` (this task adds the energy classes only)
- Test: `tests/test_multiphase_adjoint.py` (new; energy unit tests only in this task)

**Interfaces:**
- Consumes: `diffsim.physics.multiphase.np_potentials` (parity oracle in tests only).
- Produces:
  - `class FHMultiEnergy(chi, N, breg=0.0)` where `chi` is `(M+1, M+1)` symmetric ndarray, `N` is length-`M+1` ndarray.
    - `.M -> int`
    - `.param_names -> tuple[str]` = `("chi_0_1", "chi_0_2", …, "N_0", …, "N_M")` (upper-triangle χ pairs incl. solvent + per-species N)
    - `.mu(phis) -> list[np.ndarray]` length M — exchange potentials μᵢ = ∂f/∂φᵢ − ∂f/∂φ_s.
    - `.dmu_dphi(phis) -> list[list[np.ndarray]]` — `H[i][j]` = ∂μᵢ/∂φⱼ (M×M).
    - `.dmu_dparam(phis, name) -> list[np.ndarray]` length M — ∂μᵢ/∂(that parameter).
  - `phis` is a list of M arrays (Gauss-point or nodal values); all methods broadcast elementwise.

**Reference formulas** (K=0, `ps = 1 − Σφ`, `Ninv = 1/N`; matches `np_potentials` and `ternary_ch.py`):

```
mu_i      = Ninv_i (ln φ_i + 1) − Ninv_M (ln ps + 1)
            + Σ_{l≠i} φ̂_l χ[i,l] − Σ_{l≠M} φ̂_l χ[M,l]        (φ̂_M ≡ ps, l over 0..M)
∂μ_i/∂φ_j = δ_ij (Ninv_i/φ_i) + Ninv_M/ps
            + (χ[i,j] if i≠j else 0) − χ[i,M] − χ[M,j]
∂μ_i/∂χ[a,b] = (φ̂_b·[i==a] + φ̂_a·[i==b]) − (φ̂_b·[M==a] + φ̂_a·[M==b])   (a<b)
∂μ_i/∂N_j    = δ_ij · (−1/N_i²)(ln φ_i + 1)   for retained j;
∂μ_i/∂N_M    = −(−1/N_M²)(ln ps + 1) = (1/N_M²)(ln ps + 1)
```
(Sanity vs `ternary_ch`: M=2, N=1 → `∂μ₁/∂φ₁ = 1/φ₁ + 1/ps − 2χ₁ₛ`, `∂μ₁/∂φ₂ = 1/ps + χ₁₂ − χ₁ₛ − χ₂ₛ`. ✓)

- [ ] **Step 1: Write the failing test — μ parity vs `np_potentials` (ternary + quaternary)**

```python
# tests/test_multiphase_adjoint.py
import numpy as np
import pytest
from diffsim.physics.multiphase import np_potentials

pytestmark = pytest.mark.ad

def _rng_phis(M, n, seed):
    r = np.random.default_rng(seed)
    x = 0.1 + 0.15 * r.random((M, n))          # interior, Σ < 0.9
    return [x[i] for i in range(M)]

@pytest.mark.parametrize("M", [2, 3])
def test_energy_mu_parity(M):
    from diffsim.adjoint.multiphase import FHMultiEnergy
    chi = np.full((M + 1, M + 1), 1.0); np.fill_diagonal(chi, 0.0)
    chi = 0.5 * (chi + chi.T)
    N = np.ones(M + 1)
    phis = _rng_phis(M, 50, seed=1)
    en = FHMultiEnergy(chi, N)
    got = en.mu(phis)
    pars = dict(chi_aa=chi, chi_ac=np.zeros_like(chi), chi_ca=np.zeros_like(chi),
                chi_cc=np.zeros_like(chi), Ninv=1.0 / N, dsig=[], drive=[],
                breg=0.0, bulk="p1")
    ref, _ = np_potentials(phis, [], pars)
    for i in range(M):
        assert np.allclose(got[i], ref[i], atol=1e-12, rtol=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_energy_mu_parity -q`
Expected: FAIL — `ModuleNotFoundError`/`ImportError: cannot import name 'FHMultiEnergy'`.

- [ ] **Step 3: Implement `MultiEnergy` + `FHMultiEnergy`**

```python
# src/diffsim/adjoint/multiphase.py
r"""M-component phase-separation adjoint (K=0): discrete IFT adjoint through the
multi-Cahn-Hilliard BDF march, the M-generic sibling of adjoint/phasefield.py.
Parity target: physics/multiphase.MultiPhaseStepper (K=0, bulk='p1', const mob).
Bulk energy sits behind the MultiEnergy protocol so free-energy learning (M6)
reuses the identical cotangent path."""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


class MultiEnergy:
    """Protocol: bulk free energy exposing exchange potentials, their phi-
    Jacobian, and per-parameter cotangents.  Implementations: FHMultiEnergy
    (parametric FH, rung 1); a neural/extended-FH energy (M6) is a drop-in."""
    M = 0
    param_names = ()
    def mu(self, phis): raise NotImplementedError
    def dmu_dphi(self, phis): raise NotImplementedError
    def dmu_dparam(self, phis, name): raise NotImplementedError


class FHMultiEnergy(MultiEnergy):
    def __init__(self, chi, N, breg=0.0):
        self.chi = np.asarray(chi, np.float64)
        self.N = np.asarray(N, np.float64)
        self.M = self.chi.shape[0] - 1
        self.Ninv = 1.0 / self.N
        self.breg = float(breg)
        pairs = [f"chi_{a}_{b}" for a in range(self.M + 1)
                 for b in range(a + 1, self.M + 1)]
        self.param_names = tuple(pairs) + tuple(f"N_{i}"
                                                for i in range(self.M + 1))

    def _ps(self, phis):
        return 1.0 - sum(phis)

    def mu(self, phis):
        M, chi, Ninv = self.M, self.chi, self.Ninv
        ps = self._ps(phis)
        phi_of = lambda l: phis[l] if l < M else ps
        def S(m):
            return sum(phi_of(l) * chi[m, l] for l in range(M + 1) if l != m)
        out = []
        for i in range(M):
            mu = (Ninv[i] * (np.log(phis[i]) + 1.0)
                  - Ninv[M] * (np.log(ps) + 1.0) + S(i) - S(M))
            if self.breg:
                b2 = lambda x: 1.0 / np.maximum(x, 1e-3) ** 2
                mu = mu + self.breg * (b2(ps) - b2(phis[i]))
            out.append(mu)
        return out

    def dmu_dphi(self, phis):
        M, chi, Ninv = self.M, self.chi, self.Ninv
        ps = self._ps(phis)
        H = [[None] * M for _ in range(M)]
        for i in range(M):
            for j in range(M):
                term = Ninv[M] / ps + (chi[i, j] if i != j else 0.0) \
                    - chi[i, M] - chi[M, j]
                if i == j:
                    term = term + Ninv[i] / phis[i]
                if self.breg:
                    db2 = lambda x: -2.0 / np.maximum(x, 1e-3) ** 3
                    # d/dphi_j of breg(b2(ps)-b2(phi_i)); dps/dphi_j = -1
                    add = self.breg * (-db2(ps))
                    if i == j:
                        add = add - self.breg * db2(phis[i])
                    term = term + add
                H[i][j] = term * np.ones_like(phis[0]) if np.isscalar(term) \
                    else term
        return H

    def dmu_dparam(self, phis, name):
        M = self.M
        ps = self._ps(phis)
        phi_of = lambda l: phis[l] if l < M else ps
        z = [np.zeros_like(phis[0]) for _ in range(M)]
        if name.startswith("chi_"):
            _, a, b = name.split("_"); a, b = int(a), int(b)
            for i in range(M):
                val = (phi_of(b) if i == a else 0.0) \
                    + (phi_of(a) if i == b else 0.0) \
                    - (phi_of(b) if M == a else 0.0) \
                    - (phi_of(a) if M == b else 0.0)
                z[i] = val * np.ones_like(phis[0]) if np.isscalar(val) else val
            return z
        if name.startswith("N_"):
            j = int(name.split("_")[1])
            Nj = self.N[j]
            if j < M:
                z[j] = (-1.0 / Nj ** 2) * (np.log(phis[j]) + 1.0)
            else:  # solvent N_M
                for i in range(M):
                    z[i] = (1.0 / Nj ** 2) * (np.log(ps) + 1.0)
            return z
        raise KeyError(name)
```

- [ ] **Step 4: Run μ-parity test — expect PASS**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_energy_mu_parity -q`
Expected: PASS (both M=2 and M=3).

- [ ] **Step 5: Write + run the complex-step gate for all analytic derivatives**

```python
def _cstep_dmu_dphi(en, phis, j, i):
    h = 1e-30
    p = [x.astype(complex) for x in phis]
    p[j] = p[j] + 1j * h
    return en.mu(p)[i].imag / h

@pytest.mark.parametrize("M", [2, 3])
def test_energy_derivs_complex_step(M):
    from diffsim.adjoint.multiphase import FHMultiEnergy
    r = np.random.default_rng(3)
    chi = 0.3 + 0.5 * r.random((M + 1, M + 1)); chi = 0.5 * (chi + chi.T)
    np.fill_diagonal(chi, 0.0)
    N = 1.0 + r.random(M + 1)
    phis = _rng_phis(M, 20, seed=4)
    en = FHMultiEnergy(chi, N)
    H = en.dmu_dphi(phis)
    for i in range(M):
        for j in range(M):
            assert np.allclose(H[i][j], _cstep_dmu_dphi(en, phis, j, i),
                               atol=1e-9, rtol=1e-7)
    for name in en.param_names:
        an = en.dmu_dparam(phis, name)
        # central FD of mu wrt the scalar parameter
        for i in range(M):
            eps = 1e-6
            if name.startswith("chi_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                c1 = chi.copy(); c1[a, b] += eps; c1[b, a] += eps
                c2 = chi.copy(); c2[a, b] -= eps; c2[b, a] -= eps
                fd = (FHMultiEnergy(c1, N).mu(phis)[i]
                      - FHMultiEnergy(c2, N).mu(phis)[i]) / (2 * eps)
            else:
                jj = int(name.split("_")[1])
                n1 = N.copy(); n1[jj] += eps
                n2 = N.copy(); n2[jj] -= eps
                fd = (FHMultiEnergy(chi, n1).mu(phis)[i]
                      - FHMultiEnergy(chi, n2).mu(phis)[i]) / (2 * eps)
            assert np.allclose(an[i], fd, atol=1e-6, rtol=1e-5), (name, i)
```

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_energy_derivs_complex_step -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/diffsim/adjoint/multiphase.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): FHMultiEnergy + MultiEnergy protocol (mu, dmu_dphi, dmu_dparam), complex-step gated"
```

---

### Task 2: `MultiCHDiscrete` (numpy M-component residual + Jacobian + mass matrix)

The mesh operator: the M-generic mirror of `phasefield.CHDiscrete`. Same basis-table pull and scatter, but `2M` fields per node, the M×M `Onsager` mobility in the φ-rows, per-species κ in the μ-rows, and the bulk Jacobian block from `FHMultiEnergy.dmu_dphi`.

**Files:**
- Modify: `src/diffsim/adjoint/multiphase.py`
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: `FHMultiEnergy` (Task 1); a `DeviceMesh` built uniform, p=1.
- Produces:
  - `class MultiCHDiscrete(dm, M)`
    - `.M`, `.nn`, `.ndof = 2*M*nn`, `.bins` (per-bin basis cache like `CHDiscrete`).
    - `.interp(field_nodal) -> list per bin of (val[ne,nqp], grad[ne,nqp,dim])`
    - `.mass_matrix() -> csr (nn, nn)`
    - `.assemble(phis_free, mus_free, hist_gp, params, want_jac=True) -> (R[ndof], J or None)` where `phis_free`/`mus_free` are lists of M length-`nn` arrays; `hist_gp` is a list (per species) of per-bin `[ne,nqp]` BDF history loads; `params = dict(onsager=(M,M) ndarray, kappa=len-M, energy=FHMultiEnergy, sigma=float)`.
    - `.dR_dparam(phis_free, mus_free, params, name) -> ndarray[ndof]`

**Residual (K=0), per species i:**
```
R_{φ_i} = ∫ N (σ φ_i − hist_i) + Σ_j Λ_ij ∫ ∇N·∇μ_j
R_{μ_i} = ∫ N (μ_i − energy.mu_i(φ)) − κ_i ∫ ∇N·∇φ_i
```
Jacobian element block (2M×2M per node-pair), nonzero sub-blocks:
```
∂R_{φ_i}/∂φ_i = σ·NN
∂R_{φ_i}/∂μ_j = Λ_ij·LL
∂R_{μ_i}/∂φ_j = −(dmu_dphi[i][j] at GP)·NN  − (κ_i·LL if i==j else 0)
∂R_{μ_i}/∂μ_i = NN
```
where `NN[e,a,b] = ∫ dJxW N_a N_b`, `LL[e,a,b] = ∫ dJxW dscale² ∇N_a·∇N_b`, and the `dmu_dphi` term is quadrature-weighted like `FPP` in `CHDiscrete`.

- [ ] **Step 1: Write the failing test — element Jacobian vs complex-step of the residual (M=2, one small mesh)**

```python
def _dm(level, dim=2):
    from diffsim.octree.build import build_uniform
    from diffsim.mesh.nodes import build_mesh
    from diffsim.mesh.constraints import build_constraints
    from diffsim.mesh.basis import basis_tables
    from diffsim.assembly.operators import DeviceMesh
    tree = build_uniform(level, dim=dim)
    mesh = build_mesh(tree, p=1)
    cons = build_constraints(mesh)
    dm = DeviceMesh.from_mesh(mesh, cons, basis_tables(1, dim=dim), "cpu")
    return dm, mesh

def test_discrete_jacobian_complex_step():
    from diffsim.adjoint.multiphase import MultiCHDiscrete, FHMultiEnergy
    M = 2
    dm, mesh = _dm(2)
    op = MultiCHDiscrete(dm, M)
    r = np.random.default_rng(7)
    chi = 0.3 + 0.4 * r.random((M + 1, M + 1)); chi = 0.5 * (chi + chi.T)
    np.fill_diagonal(chi, 0.0)
    en = FHMultiEnergy(chi, np.ones(M + 1))
    nn = dm.n_nodes
    phis = [0.25 + 0.05 * r.random(nn) for _ in range(M)]
    mus = [0.05 * r.random(nn) for _ in range(M)]
    onsager = np.array([[1.0, 0.2], [0.2, 0.8]])
    params = dict(onsager=onsager, kappa=[0.01, 0.01], energy=en, sigma=13.7)
    hist = [[np.zeros_like(B["dJxW"]) for B in op.bins] for _ in range(M)]
    R, J = op.assemble(phis, mus, hist, params, want_jac=True)
    # complex-step one column of J: perturb dof k, imag(R)/h == J[:,k]
    def col(k):
        h = 1e-30
        pf = [x.astype(complex) for x in phis]
        mf = [x.astype(complex) for x in mus]
        node, fld = divmod(k, 2 * M)
        (pf if fld % 2 == 0 else mf)[fld // 2][node] += 1j * h
        Rc, _ = op.assemble(pf, mf, hist, params, want_jac=False)
        return Rc.imag / h
    for k in r.integers(0, op.ndof, size=12):
        assert np.allclose(np.asarray(J[:, k].todense()).ravel(), col(int(k)),
                           atol=1e-8, rtol=1e-6), k
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: MultiCHDiscrete`).

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_discrete_jacobian_complex_step -q`

- [ ] **Step 3: Implement `MultiCHDiscrete`** (append to `src/diffsim/adjoint/multiphase.py`)

```python
class MultiCHDiscrete:
    def __init__(self, dm, M):
        T = dm.constraints.T
        assert T.shape[0] == T.shape[1] and (abs(T - sp.eye(T.shape[0])).nnz
                                             == 0), \
            "adjoint gate assumes constraints.T == identity (uniform mesh)"
        self.dm = dm
        self.M = int(M)
        self.blk = 2 * self.M
        self.dim = dm.dim
        self.nn = dm.n_nodes
        self.ndof = self.blk * self.nn
        self.bins = []
        for pv, b in dm.bins.items():
            tb = dm.tables_by_p[pv]
            conn = dm.mesh.conn_of[pv].astype(np.int64)
            h = np.asarray(dm.mesh.tree.h()[dm.mesh.bins[pv]], np.float64)
            N = np.asarray(tb.N, np.float64)
            dN = np.asarray(tb.dN, np.float64)
            w = np.asarray(tb.w, np.float64)
            ne, nbf = conn.shape
            dscale = 2.0 / h
            jac = (h / 2.0) ** self.dim
            dJxW = w[None, :] * jac[:, None]
            gdof = (conn[:, :, None] * self.blk
                    + np.arange(self.blk)[None, None, :]).reshape(
                        ne, self.blk * nbf)
            self.bins.append(dict(conn=conn, N=N, dN=dN, ne=ne, nbf=nbf,
                                  nqp=N.shape[0], dscale=dscale, dJxW=dJxW,
                                  gdof=gdof))
        self._mass = None

    def interp(self, field):
        out = []
        for B in self.bins:
            vals = field[B["conn"]]
            v = np.einsum("qa,ea->eq", B["N"], vals)
            g = np.einsum("qad,ea,e->eqd", B["dN"], vals, B["dscale"])
            out.append((v, g))
        return out

    def mass_matrix(self):
        if self._mass is not None:
            return self._mass
        rows, cols, vals = [], [], []
        for B in self.bins:
            NN = np.einsum("eq,qa,qb->eab", B["dJxW"], B["N"], B["N"])
            conn, nbf = B["conn"], B["nbf"]
            rows.append(np.repeat(conn, nbf, axis=1).ravel())
            cols.append(np.tile(conn, (1, nbf)).ravel())
            vals.append(NN.ravel())
        self._mass = sp.coo_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.nn, self.nn)).tocsr()
        return self._mass

    def assemble(self, phis, mus, hist_gp, params, want_jac=True):
        M, blk = self.M, self.blk
        Lam = np.asarray(params["onsager"], np.float64)
        kap = params["kappa"]
        energy = params["energy"]
        sigma = params["sigma"]
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        R = np.zeros(self.ndof)
        rows, cols, vals = [], [], []
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            gphi = [pi[i][bi][1] for i in range(M)]
            gmu = [mi[i][bi][1] for i in range(M)]
            mu_gp = [mi[i][bi][0] for i in range(M)]
            muref = energy.mu(phi_gp)                 # list M, [ne,nqp]
            NN = np.einsum("eq,qa,qb->eab", dJxW, N, N)
            LL = np.einsum("eq,qad,qbd->eab",
                           dJxW * dscale[:, None] ** 2, dN, dN)
            ne, nbf = B["ne"], B["nbf"]
            Ae = np.zeros((ne, blk * nbf, blk * nbf)) if want_jac else None
            H = energy.dmu_dphi(phi_gp) if want_jac else None
            for i in range(M):
                # R_phi_i
                flux = np.zeros_like(gmu[0])
                for j in range(M):
                    flux = flux + Lam[i, j] * gmu[j]
                gN_flux = np.einsum("qad,e,eqd->eqa", dN, dscale, flux)
                Rphi = np.einsum("eq,qa->ea",
                                 dJxW * (sigma * phi_gp[i] - hist_gp[i][bi]),
                                 N) + np.einsum("eq,eqa->ea", dJxW, gN_flux)
                np.add.at(R, B["gdof"][:, 2 * i::blk].ravel(), Rphi.ravel())
                # R_mu_i
                gN_gphi = np.einsum("qad,e,eqd->eqa", dN, dscale, gphi[i])
                Rmu = np.einsum("eq,qa->ea",
                                dJxW * (mu_gp[i] - muref[i]), N) \
                    - kap[i] * np.einsum("eq,eqa->ea", dJxW, gN_gphi)
                np.add.at(R, B["gdof"][:, 2 * i + 1::blk].ravel(), Rmu.ravel())
                if not want_jac:
                    continue
                # Jacobian sub-blocks (row φ_i, μ_i; cols φ_j, μ_j)
                Ae[:, 2 * i::blk, 2 * i::blk] += sigma * NN          # dRφi/dφi
                for j in range(M):
                    Ae[:, 2 * i::blk, 2 * j + 1::blk] += Lam[i, j] * LL  # dRφi/dμj
                    FPP = np.einsum("eq,qa,qb->eab", dJxW * H[i][j], N, N)
                    Ae[:, 2 * i + 1::blk, 2 * j::blk] += -FPP        # dRμi/dφj
                Ae[:, 2 * i + 1::blk, 2 * i::blk] += -kap[i] * LL    # +κ term
                Ae[:, 2 * i + 1::blk, 2 * i + 1::blk] += NN          # dRμi/dμi
            if want_jac:
                gdof = B["gdof"]
                rows.append(np.repeat(gdof, blk * nbf, axis=1).ravel())
                cols.append(np.tile(gdof, (1, blk * nbf)).ravel())
                vals.append(Ae.ravel())
        J = None
        if want_jac:
            J = sp.coo_matrix(
                (np.concatenate(vals),
                 (np.concatenate(rows), np.concatenate(cols))),
                shape=(self.ndof, self.ndof)).tocsr()
        return R, J

    def dR_dparam(self, phis, mus, params, name):
        M, blk = self.M, self.blk
        energy = params["energy"]
        pi = [self.interp(phis[i]) for i in range(M)]
        mi = [self.interp(mus[i]) for i in range(M)]
        out = np.zeros(self.ndof)
        for bi, B in enumerate(self.bins):
            dJxW, N, dN, dscale = B["dJxW"], B["N"], B["dN"], B["dscale"]
            phi_gp = [pi[i][bi][0] for i in range(M)]
            if name.startswith("onsager_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                gmu_b = mi[b][bi][1]
                gN = np.einsum("qad,e,eqd->eqa", dN, dscale, gmu_b)
                Rphi = np.einsum("eq,eqa->ea", dJxW, gN)
                np.add.at(out, B["gdof"][:, 2 * a::blk].ravel(), Rphi.ravel())
            elif name.startswith("kappa_"):
                i = int(name.split("_")[1])
                gN = np.einsum("qad,e,eqd->eqa", dN, dscale, pi[i][bi][1])
                Rmu = -np.einsum("eq,eqa->ea", dJxW, gN)
                np.add.at(out, B["gdof"][:, 2 * i + 1::blk].ravel(),
                          Rmu.ravel())
            else:  # bulk-energy param routed through the energy object
                dmu = energy.dmu_dparam(phi_gp, name)
                for i in range(M):
                    Rmu = -np.einsum("eq,qa->ea", dJxW * dmu[i], N)
                    np.add.at(out, B["gdof"][:, 2 * i + 1::blk].ravel(),
                              Rmu.ravel())
        return out
```

- [ ] **Step 4: Run the Jacobian complex-step test — expect PASS**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_discrete_jacobian_complex_step -q`
Expected: PASS.

- [ ] **Step 5: Add + run a `dR_dparam` complex-step test** (mirror Step 1's `col`, but perturb each named parameter and compare `dR_dparam` to `imag(R)/h`; include `onsager_0_1`, `kappa_0`, `chi_0_1`, `N_0`).

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py -k dR_dparam -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/diffsim/adjoint/multiphase.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): MultiCHDiscrete residual+Jacobian+mass (2M block), complex-step gated"
```

---

### Task 3: `MultiCHForward` (Newton BDF march) + forward-parity gate vs `MultiPhaseStepper`

The forward driver, M-generic sibling of `phasefield.CHForward`, recording per-step converged state + BDF coefficients for the adjoint. The gate is bit-consistency with the production `MultiPhaseStepper` (K=0) — this is what makes the adjoint differentiate the *real* brick.

**Files:**
- Modify: `src/diffsim/adjoint/multiphase.py`
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: `MultiCHDiscrete`, `FHMultiEnergy`.
- Produces:
  - `class MultiCHForward(dm, energy, onsager, kappa, dt=1e-2, order=1, newton_tol=1e-12, newton_max=30)` with `.M`, `.op` (`MultiCHDiscrete`), `.steps` (list of per-step dicts: `phis`(list M), `mus`(list M), `x`(packed ndof), `sigma`, `ch`(list), `dt`, `params`).
    - `.set_initial(phi0_list)` — φ init lists (M arrays); μ init 0; history seeded.
    - `.step(record=True) -> (phis, mus)`
    - `.run(n_steps) -> list`

Reuse the BDF coefficient logic and `_hist_gp` from `phasefield.CHForward` verbatim, but loop history over all M φ-fields. Newton uses `splu(J.tocsc()).solve(-R)` and updates each field's nodal slice from `dx[2*i::blk]` (φᵢ) and `dx[2*i+1::blk]` (μᵢ).

- [ ] **Step 1: Write the failing forward-parity test (ternary, BDF1, 3 steps)**

```python
def test_forward_parity_ternary():
    from diffsim.adjoint.multiphase import MultiCHForward, FHMultiEnergy
    from diffsim.physics.multiphase import MultiPhaseStepper
    M = 2
    dm, mesh = _dm(3)
    coords = mesh.node_coords
    chi = np.zeros((M + 1, M + 1)); chi[0, 1] = chi[1, 0] = 2.5
    chi[0, 2] = chi[2, 0] = 1.0; chi[1, 2] = chi[2, 1] = 0.8
    N = np.ones(M + 1)
    onsager = np.eye(M)
    kappa = [1e-2, 1e-2]
    dt = 1e-2
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.30 + 0.05 * cc, 0.30 + 0.05 * cc]
    # production
    st = MultiPhaseStepper(dm, M=M, K=0, chi_aa=chi, N=N.tolist(),
                           onsager=onsager.tolist(), kappa=kappa, dt=dt,
                           bulk="p1", newton_tol=1e-12, newton_max=40,
                           linsolver="splu", tstep="bdf1")
    st.set_initial([(lambda x, v=p: v) for p in phi0])
    for _ in range(3):
        st.step()
    ref = [np.asarray(st.phi(i)) for i in range(M)]
    # reference numpy forward
    fwd = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=onsager,
                         kappa=kappa, dt=dt, order=1)
    fwd.set_initial(phi0)
    fwd.run(3)
    for i in range(M):
        assert np.allclose(fwd.steps[-1]["phis"][i], ref[i],
                           atol=1e-9, rtol=1e-7), i
```

> **Parity note for the implementer:** `MultiPhaseStepper.set_initial` seeds fields on the free-node layout and `st.phi(i)` returns the full nodal field; confirm the reference forward uses the identical node ordering (it does — both build off the same `dm`). If `st.phi(i)` differs in a constant, check the `b_reg` default (should be 0) and that `tstep="bdf1"` matches `order=1`. If parity is off only at machine-eps × mesh, relax `atol` to `1e-8` and record the measured value in the test docstring (measure-then-lock).

- [ ] **Step 2: Run — expect FAIL** (`ImportError: MultiCHForward`).

- [ ] **Step 3: Implement `MultiCHForward`** (append; mirror `CHForward`, packing/unpacking M fields; `_bdf`/`_hist_gp` copied and generalized to per-species history).

- [ ] **Step 4: Run parity test — expect PASS.**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_forward_parity_ternary -q`

- [ ] **Step 5: Add a quaternary (M=3) parity smoke test at level 2, 2 steps; PASS.**

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat(orgelmorph-adj): MultiCHForward BDF march + forward-parity gate vs MultiPhaseStepper (K=0)"
```

---

### Task 4: `MultiCHAdjoint` (reverse-sweep IFT adjoint)

The reverse sweep, M-generic sibling of `phasefield.CHAdjoint`. History cotangent now couples **all M** φ-fields (each carries a `−(ch_k/dt)·Mass` block on its φ-rows).

**Files:**
- Modify: `src/diffsim/adjoint/multiphase.py`
- Test: `tests/test_multiphase_adjoint.py` (gradient assembled below; three-way in Task 5)

**Interfaces:**
- Consumes: `MultiCHForward`.
- Produces:
  - `class MultiCHAdjoint(fwd)` with `.gradient(dJdx_list, param_names) -> {name: float}` where `dJdx_list[n]` is a length-`ndof` node-major cotangent `∂j/∂xₙ`, `param_names` a subset of `energy.param_names ∪ {onsager_i_j} ∪ {kappa_i}`.

```python
class MultiCHAdjoint:
    def __init__(self, fwd):
        self.fwd = fwd
        self.op = fwd.op

    def gradient(self, dJdx_list, param_names):
        op = self.op
        blk, M = op.blk, op.M
        steps = self.fwd.steps
        Ns = len(steps)
        Mass = op.mass_matrix()
        grads = {nm: 0.0 for nm in param_names}
        pending = [np.zeros(op.ndof) for _ in range(Ns)]
        zero_hist = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                     for _ in range(M)]
        for n in range(Ns - 1, -1, -1):
            rec = steps[n]
            _, J = op.assemble(rec["phis"], rec["mus"], zero_hist,
                               rec["params"], want_jac=True)
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = splu(J.T.tocsc()).solve(rhs)
            for nm in param_names:
                dRdp = op.dR_dparam(rec["phis"], rec["mus"], rec["params"], nm)
                grads[nm] -= float(lam @ dRdp)
            ch, dt = rec["ch"], rec["dt"]
            for k, cc in enumerate(ch):
                kn = n - (k + 1)
                if kn < 0:
                    continue
                for i in range(M):
                    lam_phi_i = lam[2 * i::blk]
                    hc = (cc / dt) * (Mass @ lam_phi_i)
                    pending[kn][2 * i::blk] += hc
        return grads
```

- [ ] **Step 1: Write a failing test** — assert `MultiCHAdjoint(fwd).gradient(...)` runs and returns a dict with the requested keys and finite values for a 3-step ternary FH march with `J = 0.5 Σᵢ|φᵢ,N − 0.5|²` (cotangent only on the last step's φ-rows). This locks the interface before the three-way gate.

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement `MultiCHAdjoint`** (code above).

- [ ] **Step 4: Run — expect PASS** (keys present, finite).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(orgelmorph-adj): MultiCHAdjoint reverse-sweep IFT adjoint (M-field history coupling)"
```

---

### Task 5: `MultiCHTwin` + the three-way gradient gate (the milestone)

The torch autograd twin (independent reference) and the non-negotiable three-way gate: `adj == twin == fd` for every parameter, ternary and quaternary, BDF1 and BDF2.

**Files:**
- Modify: `src/diffsim/adjoint/torch_twin.py` (add `MultiCHTwin`)
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: `dm`; `FHMultiEnergy` formulas re-expressed in torch.
- Produces:
  - `class MultiCHTwin(dm, M, dt=1e-2, order=1, device="cpu")` with `.march(phi0_list, chi_t, N_t, onsager_t, kappa_t, n_steps) -> list of packed torch tensors [ndof]` (per committed step), where `chi_t` is an `(M+1,M+1)` tensor (symmetric, requires_grad on the upper-tri leaves), `N_t` length `M+1`, `onsager_t` `(M,M)`, `kappa_t` length M. Marched with `torch.linalg.solve` inside an unclamped Newton loop driven to tight residual (autograd-through-convergence = the IFT gradient).

Implementation mirrors the existing `CHTwin` (`torch.set_default_dtype(float64)`, basis tables as torch tensors, node-major `2M` packing, FH `mu` in torch). Keep ICs interior so no log clamp is needed.

- [ ] **Step 1: Write the failing three-way test (ternary, BDF1, 3 steps)**

```python
def _three_way_multi(dm, coords, M, order, n_steps, dt=0.01):
    import torch
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    from diffsim.adjoint.torch_twin import MultiCHTwin
    nn = dm.n_nodes
    blk = 2 * M
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = np.ones(M + 1)
    ons0 = np.eye(M)
    kap0 = [0.01] * M
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    phi0 = [0.30 + 0.05 * cc for _ in range(M)]
    tgt = 0.30
    names = (["chi_0_1", "chi_0_2"] + [f"N_{i}" for i in range(M + 1)]
             + [f"onsager_0_0", "onsager_0_1"] + ["kappa_0"])

    def run(chi, N, ons, kap, record=False):
        fwd = MultiCHForward(dm, FHMultiEnergy(chi, N), onsager=ons,
                             kappa=list(kap), dt=dt, order=order)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        xs = [fwd.steps[-1]["phis"][i] for i in range(M)]
        J = 0.5 * float(sum(((x - tgt) ** 2).sum() for x in xs))
        return (fwd, J) if record else J

    fwd, _ = run(chi0, N0, ons0, kap0, record=True)
    adj = MultiCHAdjoint(fwd)
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - tgt
    g_adj = adj.gradient(dJdx, names)

    def fd(name):
        eps = 1e-6
        c, N, o, k = chi0.copy(), N0.copy(), ons0.copy(), list(kap0)
        cH, cL = (c.copy(), c.copy())
        # perturb the right leaf
        def bump(sign):
            cc_, NN_, oo_, kk_ = c.copy(), N.copy(), o.copy(), list(k)
            if name.startswith("chi_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                cc_[a, b] += sign * eps; cc_[b, a] += sign * eps
            elif name.startswith("N_"):
                NN_[int(name.split("_")[1])] += sign * eps
            elif name.startswith("onsager_"):
                _, a, b = name.split("_"); a, b = int(a), int(b)
                oo_[a, b] += sign * eps
            else:
                kk_[int(name.split("_")[1])] += sign * eps
            return run(cc_, NN_, oo_, kk_)
        return (bump(+1) - bump(-1)) / (2 * eps)
    g_fd = {nm: fd(nm) for nm in names}

    twin = MultiCHTwin(dm, M, dt=dt, order=order, device="cpu")
    chi_t = torch.tensor(chi0, requires_grad=False)
    # build differentiable leaves for the requested params (see twin API)
    g_tw = twin.grads(phi0, chi0, N0, ons0, kap0, n_steps, names, tgt)
    return {nm: (g_adj[nm], g_tw[nm], g_fd[nm]) for nm in names}

def test_three_way_ternary_bdf1():
    dm, mesh = _dm(3)
    res = _three_way_multi(dm, mesh.node_coords, M=2, order=1, n_steps=3)
    for p, (a, t, f) in res.items():
        r_t = abs(a - t) / max(abs(t), 1e-14)
        r_f = abs(a - f) / max(abs(f), 1e-14)
        print(f"{p:12s} adj={a:+.6e} twin={t:+.6e} fd={f:+.6e} "
              f"adj/twin={r_t:.2e} adj/fd={r_f:.2e}")
        assert r_t < 1e-10, (p, a, t)
        assert r_f < 1e-6, (p, a, f)
```

> **Implementer note:** expose a convenience `MultiCHTwin.grads(phi0, chi, N, onsager, kappa, n_steps, names, target)` that builds the requested leaves with `requires_grad=True`, marches, forms `loss = 0.5 Σᵢ‖φᵢ,N − target‖²`, calls `.backward()`, and returns `{name: leaf.grad}`. This keeps the leaf/`backward` bookkeeping inside the twin (mirrors how the binary test drives `CHTwin.march`).

- [ ] **Step 2: Run — expect FAIL** (`ImportError: MultiCHTwin`).

- [ ] **Step 3: Implement `MultiCHTwin`** in `torch_twin.py` (mirror `CHTwin`; M-field packing; FH `mu` + gradient/mobility/κ terms in torch; unclamped Newton to `1e-12`).

- [ ] **Step 4: Run — expect PASS** (all params `adj/twin<1e-10`, `adj/fd<1e-6`). Record measured ratios in the test docstring (measure-then-lock).

- [ ] **Step 5: Add `test_three_way_quaternary_bdf1` (M=3, level 2, 2 steps) and `test_three_way_ternary_bdf2` (order=2, 4 steps). PASS.**

- [ ] **Step 6: Export symbols + commit**

Add to `src/diffsim/adjoint/__init__.py`:
```python
from .multiphase import (MultiEnergy, FHMultiEnergy, MultiCHDiscrete,
                         MultiCHForward, MultiCHAdjoint)
```
and extend `__all__`.

```bash
git add -A && git commit -m "feat(orgelmorph-adj): MultiCHTwin + THREE-WAY gate (chi/N/mobility/kappa, ternary+quaternary, BDF1/2) — M-component adjoint verified"
```

---

### Task 6: `daisy-morph` `gradients=` wiring (separate repo)

Replace the reserved `NotImplementedError` with a call to the verified CPU engine, returning `dJ/dp` on the `SimResult`. Runs the reference differentiable forward on CPU at reference resolution; FD fallback retained; limitation documented.

**Files (in `../DAISY/daisy-morph`):**
- Modify: `src/daisy_morph/core.py:126-132` (the `gradients` block) and the `_run_forward`/return path
- Modify: `src/daisy_morph/result.py` (add `.gradients` field)
- Create: `src/daisy_morph/gradients.py` (thin adapter: parse gradient groups → engine param names, build the engine, run, map back)
- Test: `tests/test_gradients.py`
- Docs: `AGENTS.md`, `README.md` (`gradients=` now returns; reference-resolution limitation)

**Interfaces:**
- Consumes: `diffsim.adjoint.multiphase.{MultiCHForward, MultiCHAdjoint, FHMultiEnergy}`.
- Produces: `simulate(..., gradients=["chi","mobility",...], objective=<callable|None>)` returns a `SimResult` with `r.gradients` = `{group: value}`. Default objective = interfacial (gradient) energy.

- [ ] **Step 1: Branch + failing test**

```bash
cd ../DAISY/daisy-morph && git checkout -b feat/gradients
```
```python
# tests/test_gradients.py
import numpy as np
from daisy_morph import simulate

def test_chi_gradient_matches_fd():
    kw = dict(phi=(0.30, 0.30), chi=(2.5, 1.0, 0.8), level=4, seed=0,
              device="cpu", t_end=0.05, dt=5e-4, tstep="bdf1",
              stop_interface_frac=None)
    r = simulate(**kw, gradients=["chi"])
    assert "chi" in r.gradients
    g = r.gradients["chi"]                       # dJ/dchi12 for the default obj
    # central FD on chi12 via two plain forwards + the same objective
    def obj_of(chi12):
        rr = simulate(**{**kw, "chi": (chi12, 1.0, 0.8)})
        return rr.health["interface_energy"][-1]
    fd = (obj_of(2.5 + 1e-4) - obj_of(2.5 - 1e-4)) / (2e-4)
    assert abs(g["chi_0_1"] - fd) / max(abs(fd), 1e-12) < 1e-3
```

- [ ] **Step 2: Run — expect FAIL** (`NotImplementedError`).

- [ ] **Step 3: Implement `gradients.py` adapter + wire `core.py`.** Map groups → engine params (`"chi"` → all `chi_i_j`, `"mobility"` → all `onsager_i_j`, `"kappa"` → all `kappa_i`; `"phi0"` raises `NotImplementedError("phi0 gradient is the rung-1 fast-follow")`). Build `FHMultiEnergy`/`MultiCHForward` from the same parsed inputs `_run_forward` uses, march on CPU, seed the objective's `∂j/∂φ` on the final step, call `MultiCHAdjoint.gradient`. Replace lines 126-132 to dispatch here instead of raising.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Add `SimResult.gradients` (default `None`); update `GRADIENT_GROUPS` docstring; document the reference-resolution limitation in `AGENTS.md`/`README.md`. Run full CPU suite:**

Run: `python -m pytest tests/ -q`
Expected: PASS (GPU tests auto-skip).

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: wire gradients= to DiffSim M-component adjoint (CPU reference-grade); FD fallback retained"
```

---

### Task 7: Inverse-design demo + Gramian pre-check

The executable proof the gradients are usable: gradient descent on the χ-matrix to hit a target interfacial-energy, preceded by a Gramian/identifiability pre-check (house rule: Gramian-before-compute).

**Files (in `../DAISY/daisy-morph`):**
- Create: `examples/inverse_design_chi.py`
- Test: `tests/test_inverse_design.py`

**Interfaces:**
- Consumes: `simulate(..., gradients=["chi"])`.
- Produces: `inverse_design_chi(target_ie, chi0, steps, lr) -> (chi_final, history)` and a `gramian_check(gradients) -> dict` reporting the design-Jacobian Gram spectrum (near-null directions = unidentifiable χ combinations, per the `neural_energy.py` gauge lesson).

- [ ] **Step 1: Failing test** — from `chi0=(3.0,1.0,0.6)`, 8 gradient steps toward a target interfacial energy produced by `chi=(3.5,1.0,0.6)`; assert the objective decreases monotonically for ≥6 of 8 steps and the final gap < 30% of the initial gap (loose, CPU level-4).

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement** `gramian_check` (assemble the per-parameter gradient vectors, form `G = AᵀA`, report eigenvalues + condition number; warn on near-null directions) and the descent loop.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: inverse-design-chi demo + Gramian identifiability pre-check (gradient-based BO proof)"
```

---

## Follow-on plans (named, not in this plan)

- **Rung 2 — GPU production adjoint (256×128).** On-device `Jᵀ`/cuDSS reusing the production Jacobian; committed-state trajectory checkpointing; bump the daisy-morph DiffSim pin through the 6 GPU integration tests. Its own spec section §7.
- **Mean-φ initial-condition gradient (fast-follow).** Cotangent into `x₀` then `∂x₀/∂φ_mean` under a fixed noise seed; wire the `"phi0"` gradient group.

## Self-Review

- **Spec coverage:** §Architecture→Tasks 1-5; §Parameter derivatives→Tasks 1-2 (table); §Objective interface→Task 5 objective + Task 6 default; §Verification gate→Tasks 1,2,3,5; §Module layout→Tasks 1-5 + `__init__` (Task 5 Step 6); §daisy-morph wiring→Task 6; §Inverse-design demo + Gramian→Task 7; §9 MultiEnergy protocol→Task 1 (`MultiEnergy` base, χ/N through the object; mobility/κ engine-level in Task 2). Rung 2 + mean-φ explicitly deferred (spec §7). No gaps.
- **Placeholder scan:** all code steps carry real code; formulas given explicitly (energy derivatives derived + ternary-cross-checked); the two "mirror `CHForward`/`CHTwin`" steps (3.3, 5.3) reference an exact existing file to copy structure from and specify the M-generalization delta — acceptable since the sibling is fully shown in the codebase, not a placeholder.
- **Type consistency:** `phis`/`mus` are length-M lists of length-`nn` arrays throughout; `param_names` scheme (`chi_a_b`, `N_i`, `onsager_a_b`, `kappa_i`) is identical across `FHMultiEnergy` (Task 1), `MultiCHDiscrete.dR_dparam` (Task 2), `MultiCHAdjoint.gradient` (Task 4), the three-way test (Task 5), and the daisy-morph group→param mapping (Task 6). `blk = 2*M`, DOF `a*blk + f` consistent. `steps[n]["phis"]` produced in Task 3, consumed in Tasks 4-5.
