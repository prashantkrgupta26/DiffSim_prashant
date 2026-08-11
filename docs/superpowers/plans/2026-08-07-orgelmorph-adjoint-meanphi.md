# Differentiable OrgElMorph — mean-φ initial-condition gradient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the mean-composition (initial-condition) gradient `dJ/dφ_mean_i` to the M-component adjoint — the `phi0` gradient group that currently raises `NotImplementedError` — three-way verified, on CPU.

**Architecture:** The mean-φ gradient is a distinct mechanism from the residual-parameter path, and it is *already latent* in the reverse sweep. The BDF history couples each committed step back to earlier states; at the start of the trajectory those couplings land on the initial condition x₀, and the current adjoint **drops them** (`if kn < 0: continue`, `multiphase.py:468`). Accumulating those dropped contributions instead of discarding them yields `dJ/dx₀` exactly. Because the initial field is `ics_i = φ_mean_i + amp·noise` under a fixed seed, `∂ics_i[node]/∂φ_mean_i = 1` for every node, so `dJ/dφ_mean_i = Σ_nodes (dJ/dx₀)_{φ_i}`. No new forward machinery — one accumulator in the existing history loop.

**Tech Stack:** numpy/scipy (hand adjoint), torch (autograd twin), pytest. Pure CPU — no GPU, no rung-2 backend involved. This is the fast-follow named in the design spec §7 and the deliverables checklist.

## Global Constraints

- **Three-way gate is the house-rule acceptance:** hand IFT adjoint == torch twin == central finite differences, per parameter, for a ternary and one quaternary case, BDF1 and BDF2. Tolerances copied from the existing gate: `adj/twin < 1e-10`, `adj/fd < 1e-6`.
- **No regression:** the existing three-way gate for χ/N/mobility/κ (`tests/test_multiphase_adjoint.py`) and the daisy-morph gradient suite must stay green. The `phi0` change is additive.
- **Naming convention:** the mean-φ design parameter for retained species `i` (0..M-1) is `phi0_i`. The eliminated solvent has no independent mean (φ_s = 1 − Σφ), so there is no `phi0_M`.
- **Mean-φ math:** `dJ/dφ_mean_i = Σ_nodes phi0_cot_i`, where `phi0_cot_i` accumulates `(ch_k/dt)·(Mass @ λ_{φ_i})` for every history term whose target step index `kn = n−(k+1)` is `< 0` (i.e. lands on the initial condition). Same sign/scaling as the existing history cotangent (`multiphase.py:471`) — no extra sign flip; the three-way gate is the proof.
- **Reference the existing template:** the three-way helper `_three_way_multi` and `_check_three_way` in `tests/test_multiphase_adjoint.py` are the pattern to mirror. Do not invent a new objective — reuse `J = 0.5 Σ_i ‖φ_i,N − tgt‖²` with cotangent `(φ_i,N − tgt)` on the last step's φ-rows.
- **Run tests with** `.venv/bin/pytest` on Mac (no bare `python`; no `timeout` on macOS). daisy-morph runs from its own dir with its own `.venv`.
- **Branch:** DiffSim `diff-orgelmorph`; daisy-morph `feat/gradients`. Commit-per-green.

---

## File Structure

**DiffSim (branch `diff-orgelmorph`):**
- Modify `src/diffsim/adjoint/multiphase.py` — `MultiCHAdjoint.gradient` accumulates the `kn<0` history cotangent into a per-species `phi0_cot` and returns `phi0_i` gradients; residual-parameter names are handled exactly as before.
- Modify `src/diffsim/adjoint/torch_twin.py` — `MultiCHTwin.grads` accepts `phi0_i` names, making a scalar per-species mean-offset leaf added to the initial field.
- Modify `tests/test_multiphase_adjoint.py` — Task 1 hand-vs-FD test, Task 2 twin-vs-FD test, Task 3 three-way `phi0` gate.

**daisy-morph (branch `feat/gradients`):**
- Modify `src/daisy_morph/gradients.py` — `_group_names` returns the `phi0_i` names instead of raising.
- Modify `src/daisy_morph/tests/test_gradients.py` — `phi0` now returns; FD cross-check.
- Modify `tests/test_core_cpu.py` — replace `test_gradients_phi0_still_reserved` (asserts raise) with a test asserting `phi0` now returns gradients.
- Modify `README.md` / `AGENTS.md` — update the gradients row (phi0 live, no longer fast-follow-pending).

---

## Task 1: Engine — accumulate the initial-condition (mean-φ) cotangent

**Files:**
- Modify: `src/diffsim/adjoint/multiphase.py:439-473` (`MultiCHAdjoint.gradient`)
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: `MultiCHForward` (records `steps`), `MultiCHDiscrete.mass_matrix()`, `.nn`, `.blk`, `.M`.
- Produces: `MultiCHAdjoint.gradient(dJdx_list, param_names)` now accepts names of the form `phi0_i` (i in 0..M−1). For those it returns `dJ/dφ_mean_i`; residual names (`chi_*`, `N_*`, `onsager_*`, `kappa_*`) behave exactly as before.

- [ ] **Step 1: Write the failing test** (hand adjoint vs central FD — no twin yet)

Add to `tests/test_multiphase_adjoint.py` (reuse the module's existing `_dm` helper and imports):

```python
def _meanphi_adj_vs_fd(dm, coords, M, order, n_steps, dt=0.01):
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    nn = dm.n_nodes
    blk = 2 * M
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    base = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = [f"phi0_{i}" for i in range(M)]

    def run(phi0, record=False):
        fwd = MultiCHForward(dm, FHMultiEnergy(chi0, N0), onsager=ons0,
                             kappa=list(kap0), dt=dt, order=order)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        xs = [fwd.steps[-1]["phis"][i] for i in range(M)]
        J = 0.5 * float(sum(((x - tgt) ** 2).sum() for x in xs))
        return (fwd, J) if record else J

    fwd, _ = run(base, record=True)
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - tgt
    g_adj = MultiCHAdjoint(fwd).gradient(dJdx, names)

    eps = 1e-6
    g_fd = {}
    for i in range(M):
        hi = [base[j] + (eps if j == i else 0.0) for j in range(M)]
        lo = [base[j] - (eps if j == i else 0.0) for j in range(M)]
        g_fd[f"phi0_{i}"] = (run(hi) - run(lo)) / (2 * eps)
    return {nm: (g_adj[nm], g_fd[nm]) for nm in names}


def test_meanphi_adj_vs_fd_ternary_bdf1():
    dm, mesh = _dm(3)
    res = _meanphi_adj_vs_fd(dm, mesh.node_coords, M=2, order=1, n_steps=3)
    for p, (a, f) in res.items():
        rel = abs(a - f) / max(abs(f), 1e-14)
        print(f"meanphi-bdf1 {p} adj={a:+.6e} fd={f:+.6e} rel={rel:.2e}")
        assert rel < 1e-6, (p, a, f)


def test_meanphi_adj_vs_fd_ternary_bdf2():
    dm, mesh = _dm(3)
    res = _meanphi_adj_vs_fd(dm, mesh.node_coords, M=2, order=2, n_steps=4)
    for p, (a, f) in res.items():
        rel = abs(a - f) / max(abs(f), 1e-14)
        assert rel < 1e-6, (p, a, f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_meanphi_adj_vs_fd_ternary_bdf1 -v`
Expected: FAIL — `MultiCHAdjoint.gradient` currently initializes `grads = {nm: 0.0 for nm in param_names}` and then calls `op.dR_dparam(..., "phi0_0")`, which raises `KeyError('phi0_0')` (or the grad stays 0.0 and the assertion fails). Either way, not the FD value.

- [ ] **Step 3: Write minimal implementation**

Replace `MultiCHAdjoint.gradient` (`src/diffsim/adjoint/multiphase.py:439-473`) with:

```python
    def gradient(self, dJdx_list, param_names):
        """dJdx_list[n] = dj/dx_n as a length-ndof node-major vector.
        Returns {name: dJ/dname}.  Residual params (chi/N/onsager/kappa) use
        the reverse-sweep dR/dp; names of the form 'phi0_i' return the
        mean-composition (initial-condition) gradient dJ/dphi_mean_i, formed
        from the history cotangent that lands on the initial state x0."""
        op = self.op
        blk, M = op.blk, op.M
        steps = self.fwd.steps
        Ns = len(steps)
        Mass = op.mass_matrix()
        res_names = [nm for nm in param_names if not nm.startswith("phi0_")]
        phi0_names = [nm for nm in param_names if nm.startswith("phi0_")]
        grads = {nm: 0.0 for nm in param_names}
        pending = [np.zeros(op.ndof) for _ in range(Ns)]
        # cotangent that accumulates onto the initial condition x0 (the history
        # terms whose target step index kn < 0). Per retained phi-species.
        phi0_cot = [np.zeros(op.nn) for _ in range(M)] if phi0_names else None
        # J = dR/dx is independent of the BDF history load, so zero history
        # rebuilds the Jacobian at the converged iterate.
        zero_hist = [[np.zeros_like(B["dJxW"]) for B in op.bins]
                     for _ in range(M)]
        for n in range(Ns - 1, -1, -1):
            rec = steps[n]
            _, J = op.assemble(rec["phis"], rec["mus"], zero_hist,
                               rec["params"], want_jac=True)
            rhs = np.asarray(dJdx_list[n], np.float64) + pending[n]
            lam = self.fwd.backend.solve_T(J, rhs)
            for nm in res_names:
                dRdp = op.dR_dparam(rec["phis"], rec["mus"], rec["params"], nm)
                grads[nm] -= float(lam @ dRdp)
            # history cotangent: R_n depends on phi_i,{n-k} only through
            # -(ch_k/dt) Mass on the phi_i-block -> +(ch_k/dt) Mass lam_phi_i
            # onto the phi_i-columns of the earlier step (kn>=0), or onto the
            # initial-condition accumulator (kn<0 -> x0).
            ch, dt = rec["ch"], rec["dt"]
            for k, cc in enumerate(ch):
                kn = n - (k + 1)
                for i in range(M):
                    hc = (cc / dt) * (Mass @ lam[2 * i::blk])
                    if kn < 0:
                        if phi0_cot is not None:
                            phi0_cot[i] += hc
                    else:
                        pending[kn][2 * i::blk] += hc
        # dJ/dphi_mean_i = sum_nodes (dJ/dx0)_{phi_i}, since a uniform mean
        # shift adds 1 to every nodal value of the initial phi_i field.
        for nm in phi0_names:
            i = int(nm.split("_")[1])
            grads[nm] = float(phi0_cot[i].sum())
        return grads
```

(This preserves the residual-parameter path bit-for-bit — `res_names` is the old `param_names` for any non-`phi0` call, and the `kn<0` branch used to be `continue`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py -k "meanphi or three_way or adjoint_interface" -v`
Expected: the two new `meanphi_adj_vs_fd` tests PASS (rel < 1e-6); the existing `three_way` and `adjoint_interface` tests still PASS (no regression to the residual path).

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/multiphase.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): mean-phi initial-condition gradient (x0 history cotangent), adj==FD gated"
```

---

## Task 2: Twin — mean-offset leaf for the initial field

**Files:**
- Modify: `src/diffsim/adjoint/torch_twin.py:471-526` (`MultiCHTwin.grads`)
- Test: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: `MultiCHTwin.march` (unchanged).
- Produces: `MultiCHTwin.grads(phi0_list, chi, N, onsager, kappa, n_steps, names, target)` now accepts `phi0_i` names. For each, it creates a scalar leaf `off_i` (init 0.0) and marches with `phi_i = phi0_list[i] + off_i`; the returned grad is `dloss/doff_i`.

- [ ] **Step 1: Write the failing test** (twin vs central FD, isolating the twin path)

Add to `tests/test_multiphase_adjoint.py`:

```python
def test_meanphi_twin_vs_fd_ternary_bdf1():
    from diffsim.adjoint.torch_twin import MultiCHTwin
    import numpy as np
    dm, mesh = _dm(3)
    coords = mesh.node_coords
    M, order, n_steps, dt = 2, 1, 3, 0.01
    chi0 = np.zeros((M + 1, M + 1))
    chi0[0, 1] = chi0[1, 0] = 2.5
    chi0[0, 2] = chi0[2, 0] = 1.0
    chi0[1, 2] = chi0[2, 1] = 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)
    ons0 = np.eye(M)
    kap0 = [0.01, 0.02]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    base = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = [f"phi0_{i}" for i in range(M)]
    twin = MultiCHTwin(dm, M, dt=dt, order=order, device="cpu")
    g_tw = twin.grads(base, chi0, N0, ons0, kap0, n_steps, names, tgt)

    import torch
    def loss_of(phi0):
        out = twin.march(
            [torch.tensor(np.asarray(p)) for p in phi0],
            torch.tensor(chi0), torch.tensor(N0), torch.tensor(ons0),
            [torch.tensor(k) for k in kap0], n_steps)
        xN = out[-1]
        return 0.5 * float(sum(((xN[2 * i::2 * M] - tgt) ** 2).sum()
                               for i in range(M)))
    eps = 1e-6
    for i in range(M):
        hi = [base[j] + (eps if j == i else 0.0) for j in range(M)]
        lo = [base[j] - (eps if j == i else 0.0) for j in range(M)]
        fd = (loss_of(hi) - loss_of(lo)) / (2 * eps)
        assert abs(g_tw[f"phi0_{i}"] - fd) / max(abs(fd), 1e-14) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_meanphi_twin_vs_fd_ternary_bdf1 -v`
Expected: FAIL — `MultiCHTwin.grads` ignores `phi0_i` names (the leaf-building block only recognizes `chi_`/`N_`/`onsager_`/`kappa`), so `leaves["phi0_0"]` is never created and a `KeyError` is raised at the return `{nm: float(leaves[nm].grad) ...}`.

- [ ] **Step 3: Write minimal implementation**

In `MultiCHTwin.grads` (`src/diffsim/adjoint/torch_twin.py:471`), (a) recognize `phi0_i` when building leaves, and (b) add the offset to the initial field. Edit the leaf-building loop to skip `phi0_` in the residual-leaf branch and create an offset leaf; then edit the `phis = [...]` construction:

Replace the leaf-building loop (currently lines ~481-490) so `phi0_` names get a scalar offset leaf:

```python
        leaves = {}
        phi0_off = {}
        for nm in names:
            if nm.startswith("phi0_"):
                i = int(nm.split("_")[1])
                leaves[nm] = torch.tensor(0.0, dtype=torch.float64,
                                          device=self.dev, requires_grad=True)
                phi0_off[i] = leaves[nm]
                continue
            base = (chi0 if nm.startswith("chi_") else
                    N0 if nm.startswith("N_") else
                    ons0 if nm.startswith("onsager_") else kap0)
            if nm.startswith("chi_") or nm.startswith("onsager_"):
                _, a, b = nm.split("_"); v = float(base[int(a), int(b)])
            else:
                v = float(base[int(nm.split("_")[1])])
            leaves[nm] = torch.tensor(v, dtype=torch.float64,
                                      device=self.dev, requires_grad=True)
```

Then replace the `phis = [...]` line (currently ~516-517) with an offset-aware build:

```python
        phis = []
        for i in range(M):
            p = torch.tensor(np.asarray(phi0_list[i]), dtype=torch.float64,
                             device=self.dev)
            if i in phi0_off:
                p = p + phi0_off[i]
            phis.append(p)
```

(The rest of `grads` — `march`, `loss`, `loss.backward()`, `return {nm: float(leaves[nm].grad) ...}` — is unchanged and now returns `dloss/doff_i` for the `phi0_i` names.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py::test_meanphi_twin_vs_fd_ternary_bdf1 -v`
Expected: PASS (twin == FD to < 1e-6). Also run `-k "three_way"` to confirm the residual-param twin path is unregressed.

- [ ] **Step 5: Commit**

```bash
git add src/diffsim/adjoint/torch_twin.py tests/test_multiphase_adjoint.py
git commit -m "feat(orgelmorph-adj): MultiCHTwin mean-phi offset leaf (twin==FD for phi0 group)"
```

---

## Task 3: Three-way gate for the mean-φ gradient

**Files:**
- Modify: `tests/test_multiphase_adjoint.py`

**Interfaces:**
- Consumes: Task 1 (hand adjoint `phi0_i`) + Task 2 (twin `phi0_i`).
- Produces: `_three_way_meanphi(dm, coords, M, order, n_steps)` returning `{name: (adj, twin, fd)}`, and three test functions gating ternary BDF1/BDF2 + quaternary BDF1 through `_check_three_way`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_multiphase_adjoint.py` (reuse the existing `_check_three_way`):

```python
def _three_way_meanphi(dm, coords, M, order, n_steps, dt=0.01):
    from diffsim.adjoint.multiphase import (MultiCHForward, MultiCHAdjoint,
                                            FHMultiEnergy)
    from diffsim.adjoint.torch_twin import MultiCHTwin
    nn = dm.n_nodes
    blk = 2 * M
    chi0 = np.zeros((M + 1, M + 1))
    for a in range(M + 1):
        for b in range(a + 1, M + 1):
            chi0[a, b] = chi0[b, a] = 2.5 if (a, b) == (0, 1) else 1.0
    N0 = 1.0 + 0.3 * np.arange(M + 1)
    ons0 = np.eye(M) + 0.1 * (np.ones((M, M)) - np.eye(M))
    kap0 = [0.01 * (i + 1) for i in range(M)]
    cc = np.cos(np.pi * coords[:, 0]) * np.cos(np.pi * coords[:, 1])
    base = [0.28 + 0.05 * cc for _ in range(M)]
    tgt = 0.28
    names = [f"phi0_{i}" for i in range(M)]

    def run(phi0, record=False):
        fwd = MultiCHForward(dm, FHMultiEnergy(chi0, N0), onsager=ons0,
                             kappa=list(kap0), dt=dt, order=order)
        fwd.set_initial(phi0)
        fwd.run(n_steps)
        xs = [fwd.steps[-1]["phis"][i] for i in range(M)]
        J = 0.5 * float(sum(((x - tgt) ** 2).sum() for x in xs))
        return (fwd, J) if record else J

    fwd, _ = run(base, record=True)
    dJdx = [np.zeros(blk * nn) for _ in range(n_steps)]
    for i in range(M):
        dJdx[-1][2 * i::blk] = fwd.steps[-1]["phis"][i] - tgt
    g_adj = MultiCHAdjoint(fwd).gradient(dJdx, names)

    eps = 1e-6
    g_fd = {}
    for i in range(M):
        hi = [base[j] + (eps if j == i else 0.0) for j in range(M)]
        lo = [base[j] - (eps if j == i else 0.0) for j in range(M)]
        g_fd[f"phi0_{i}"] = (run(hi) - run(lo)) / (2 * eps)

    twin = MultiCHTwin(dm, M, dt=dt, order=order, device="cpu")
    g_tw = twin.grads(base, chi0, N0, ons0, kap0, n_steps, names, tgt)
    return {nm: (g_adj[nm], g_tw[nm], g_fd[nm]) for nm in names}


def test_three_way_meanphi_ternary_bdf1(device):
    dm, mesh = _dm(3)
    res = _three_way_meanphi(dm, mesh.node_coords, M=2, order=1, n_steps=3)
    _check_three_way(res, "MP-T-bdf1")


def test_three_way_meanphi_ternary_bdf2(device):
    dm, mesh = _dm(3)
    res = _three_way_meanphi(dm, mesh.node_coords, M=2, order=2, n_steps=4)
    _check_three_way(res, "MP-T-bdf2")


def test_three_way_meanphi_quaternary_bdf1(device):
    dm, mesh = _dm(2)
    res = _three_way_meanphi(dm, mesh.node_coords, M=3, order=1, n_steps=2)
    _check_three_way(res, "MP-Q-bdf1")
```

- [ ] **Step 2: Run tests to verify state**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py -k "three_way_meanphi" -v`
Expected: PASS (all three). If they were run before Tasks 1-2 they would fail; here they should pass immediately since the engine + twin already support `phi0_i`. If any `adj/twin` exceeds 1e-10 or `adj/fd` exceeds 1e-6, STOP — the sign or scaling of the `phi0_cot` accumulation is wrong; revisit Task 1's `kn<0` branch.

- [ ] **Step 3: Run the full adjoint suite (no-regression)**

Run: `.venv/bin/pytest tests/test_multiphase_adjoint.py -q`
Expected: all green (the pre-existing χ/N/mobility/κ three-way gate, the rung-2 backend tests, and the new mean-φ tests), GPU-gated cuDSS test SKIPPED.

- [ ] **Step 4: Commit**

```bash
git add tests/test_multiphase_adjoint.py
git commit -m "test(orgelmorph-adj): three-way gate for mean-phi gradient (ternary bdf1/bdf2, quaternary bdf1)"
```

---

## Task 4: daisy-morph — wire the `phi0` gradient group

**Files:**
- Modify: `src/daisy_morph/gradients.py:91-102` (`_group_names`)
- Modify: `tests/test_gradients.py`, `tests/test_core_cpu.py`
- Modify: `README.md`, `AGENTS.md`

**Interfaces:**
- Consumes: DiffSim `MultiCHAdjoint.gradient` with `phi0_i` names (Task 1). daisy-morph's `.venv` installs `diffsim` editable against the DiffSim checkout, so Task 1 is already importable.
- Produces: `simulate(..., gradients=["phi0"])` returns `r.gradients["phi0"] = {"phi0_0": dJ/dφ_mean_0, ...}`; the `phi0` group no longer raises.

- [ ] **Step 1: Write the failing test**

In `daisy-morph/tests/test_gradients.py`, add a returns+FD cross-check (mirror the existing chi test in that file for the harness):

```python
def test_phi0_gradient_returns_and_matches_fd():
    import numpy as np
    from daisy_morph.core import simulate
    from daisy_morph import gradients as g
    kw = dict(level=4, aspect=1, periodic=False, mobility=1.0, kappa=1e-2,
              dt=5e-4, tstep="bdf1", t_end=0.01, seed=0, amp=0.02)
    # simulate MUST use the same seed+amp as the FD objective_value calls, or
    # the two run on different noise realizations and the check is meaningless.
    r = simulate(phi=(0.35, 0.35), chi=(3.5, 1.0, 0.6), device="cpu",
                 level=4, periodic=False, t_end=0.01, dt=5e-4, tstep="bdf1",
                 seed=0, amp=0.02, mobility=1.0, kappa=1e-2,
                 gradients=["phi0"])
    assert r.gradients is not None
    assert "phi0_0" in r.gradients["phi0"] and "phi0_1" in r.gradients["phi0"]
    # central-FD cross-check on phi_mean_0 via the objective_value entry point
    M = 2
    N_arr = np.ones(M + 1)
    chi_m = np.zeros((M + 1, M + 1))
    chi_m[0, 1] = chi_m[1, 0] = 3.5
    chi_m[0, 2] = chi_m[2, 0] = 1.0
    chi_m[1, 2] = chi_m[2, 1] = 0.6
    eps = 1e-6
    hi = g.objective_value(M, np.array([0.35 + eps, 0.35]), chi_m, N_arr, **kw)
    lo = g.objective_value(M, np.array([0.35 - eps, 0.35]), chi_m, N_arr, **kw)
    fd = (hi - lo) / (2 * eps)
    adj = r.gradients["phi0"]["phi0_0"]
    assert abs(adj - fd) / max(abs(fd), 1e-12) < 1e-5
```

> Note: confirm `objective_value`'s signature in `gradients.py` — it is `objective_value(M, phi0, chi_m, N_arr, **kw)` and forwards `**kw` to `_forward`, whose keyword-only args are `level, aspect, periodic, mobility, kappa, dt, tstep, t_end, seed, amp`. Pass exactly those in `kw` (as above). If a name differs, match the real signature rather than the sketch.

- [ ] **Step 2: Run test to verify it fails**

Run (daisy-morph dir): `.venv/bin/pytest tests/test_gradients.py::test_phi0_gradient_returns_and_matches_fd -v`
Expected: FAIL with `NotImplementedError: phi0 (mean-composition) gradient is the rung-1 fast-follow ...` from `_group_names`.

- [ ] **Step 3: Write minimal implementation**

In `src/daisy_morph/gradients.py`, replace the `phi0` branch of `_group_names` (lines 98-101):

```python
    if group == "phi0":
        return [f"phi0_{i}" for i in range(M)]
```

- [ ] **Step 4: Update the obsolete reserved test**

In `daisy-morph/tests/test_core_cpu.py`, replace `test_gradients_phi0_still_reserved` (which asserts `phi0` raises) with:

```python
def test_gradients_phi0_now_returns(monkeypatch):
    # phi0 (mean-composition) gradient is live (three-way verified engine).
    monkeypatch.setattr(doc, "_GPU_OK", True)
    r = simulate(phi=(0.35, 0.35), chi=(3.5, 1.0, 0.6), device="cpu",
                 level=4, periodic=False, t_end=0.01, dt=5e-4, tstep="bdf1",
                 gradients=["phi0"])
    assert r.gradients is not None and "phi0_0" in r.gradients["phi0"]
```

- [ ] **Step 5: Run the daisy-morph gradient + core tests**

Run (daisy-morph dir): `.venv/bin/pytest tests/test_gradients.py tests/test_core_cpu.py -q`
Expected: green — the new `phi0` returns+FD test passes, the updated `test_gradients_phi0_now_returns` passes, the GPU-gated gradient test SKIPS, and nothing else regresses.

- [ ] **Step 6: Update docs**

In `README.md` and `AGENTS.md`, update the gradients row: `phi0` (mean-composition) is now live and three-way verified — remove the "fast-follow pending" / "phi0 still reserved" wording. Keep it to the one row/line each file already devotes to gradient groups.

- [ ] **Step 7: Commit** (daisy-morph, branch `feat/gradients`)

```bash
git add src/daisy_morph/gradients.py tests/test_gradients.py tests/test_core_cpu.py README.md AGENTS.md
git commit -m "feat: wire phi0 (mean-composition) gradient group to the verified engine"
```

---

## Self-Review

**Spec coverage (design spec §7 + deliverables checklist "mean-φ initial-condition gradient (fast-follow)"):**
- "Cotangent into x₀ then ∂x₀/∂φ_mean (uniform shift of the φ-field mean under a fixed noise seed)" → Task 1 (`phi0_cot` accumulation of the `kn<0` history cotangent; `.sum()` implements the uniform-shift chain). ✓
- "a distinct mechanism from the residual-parameter path, isolated and added once the engine is verified" → the `phi0` path is separate from `dR_dparam`; residual path untouched. ✓
- Three-way verified (house rule) → Task 3 (ternary BDF1/BDF2 + quaternary BDF1), Task 1 (adj vs FD early), Task 2 (twin vs FD). ✓
- `daisy-morph` `phi0` group live → Task 4. ✓

**Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N" — every step has real code or a real command. The one indirection (confirm `objective_value` signature) points at the exact existing function and its keyword-only args.

**Type consistency:** `phi0_i` naming is identical across engine (`gradient`), twin (`grads`), `_group_names`, and every test. `MultiCHAdjoint.gradient` uses `self.fwd.backend.solve_T` (consistent with the rung-2 refactor already on `diff-orgelmorph`). `phi0_cot[i]` is length `op.nn`; `Mass @ lam[2*i::blk]` is length `nn`; `.sum()` yields the scalar `dJ/dφ_mean_i`. The objective `J = 0.5 Σ‖φ_i,N − tgt‖²` and its cotangent `(φ_i,N − tgt)` on the last step's φ-rows match the existing `_three_way_multi` template exactly.
```
