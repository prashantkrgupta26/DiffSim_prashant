# Sub-project ②: `NeuralCrystalEnergy` — non-parametric coupled free-energy learning (K>0)

**Date:** 2026-08-08
**Branch (to be):** `feat/neural-crystal-energy` (off `master` after ① `feat/crystallization-adjoint` merges)
**Depends on:** ① crystallization adjoint engine (Tasks 1–5) merged — in particular the `CrystalEnergy` protocol, `CrystalCHForward`/`CrystalCHAdjoint`, and `CrystalCHTwin`.

## Goal

Learn the coupled crystallization free energy `f(φ, ψ)` from φ+ψ trajectories, in a
**general non-parametric form**, through the exact ψ-coupled adjoint built in ①.

Sub-project ① proved the gradients through the ψ-coupled BDF march are exact
(three-way gated) for the *parametric* `AdditiveCrystalEnergy`. ② replaces that
energy with a **learnable** one that drops into the same `CrystalEnergy` protocol —
**zero engine rework** — and then *recovers* it by trajectory-matching. It is the
K>0 analogue of M6 Plan A (`BasisMultiEnergy` free-energy learning), lifted one
level to the coupled energy.

## Functional form

```
f(φ, ψ) = f_base(φ)  +  Σ_{k∈K} φ_k · h_k(ψ_k)
```

- **`f_base(φ)` = the M6 `BasisMultiEnergy` energy** (Flory–Huggins + gauge-anchored
  shifted-Legendre correction to each exchange potential μ_i). The pure-φ energy is
  therefore *also* learnable, reusing M6 verbatim. `BasisMultiEnergy` exposes
  `mu` (= ∂f_base/∂φ_i) and `dmu_dphi` (= ∂²f_base/∂φ_i∂φ_j); the protocol needs only
  derivatives, never `f` itself.
- **`h_k(ψ)` = Σ_{b∈deg_ψ} c_{k,b} · L_b(û_k)`**, with `û_k = 2ψ_k − 1` mapping the
  crystallinity domain [0,1] to the shifted-Legendre domain [−1,1]. `L_b` are the
  same closed-form shifted-Legendre polynomials as M6 (`_legendre_np`).
- **`deg_ψ ⊆ {1, 2, 3, …}` — the constant mode b=0 is excluded** (gauge; see below).
- The coupling is **linear in φ_k**: `h_k` is a free function of ψ_k only. (Full 2-D
  `g_k(φ_k, ψ_k)` is out of scope — ②b.)

### Learnable parameters

```
param_names = BasisMultiEnergy.param_names                       # chi_a_b, N_i, basis_i_k
            + ( f"cpl_{k}_{b}"  for k in crystallizable for b in deg_ψ )
```

## Protocol quantities (all analytic, complex-safe)

Let `k` range over `crystallizable`, `j` index the ψ-fields (`k = crystallizable[j]`).
`L_b`, `L_b'`, `L_b''` are the shifted-Legendre value and derivatives at `û_k`; the
`ψ`-chain factors are `dû/dψ = 2`, `d²û/dψ² = 0`.

| quantity | value |
|---|---|
| `dfdphi_i` | `μ_i^base` + (`h_k(ψ_k)` if `i == k ∈ K` else 0) |
| `dfdpsi_k` | `φ_k · h_k'(ψ_k)`,  `h_k'(ψ) = 2 Σ_b c_{k,b} L_b'(û_k)` |
| `d2fdphidphi[i][j]` | `BasisMultiEnergy.dmu_dphi[i][j]` (coupling contributes 0 — linear in φ, ψ-only h) |
| `d2fdphidpsi[i][j]` | `h_k'(ψ_k)` if `i == k`, else 0 (diagonal in k) |
| `d2fdpsidpsi[j][j']` | `φ_k · h_k''(ψ_k)` if `j == j'`, else 0;  `h_k'' = 4 Σ_b c_{k,b} L_b''(û_k)` |
| `dfdphi_dparam(name)` | base names → `BasisMultiEnergy.dmu_dparam`; `cpl_{k}_{b}` → `z[k] = L_b(û_k)` |
| `dfdpsi_dparam(name)` | `cpl_{k}_{b}` → `z[j] = 2 φ_k L_b'(û_k)`; base names → 0 |

This requires a **second derivative** of the Legendre polynomials (`L_b''`), which M6
does not use. Task 1 adds a `_legendre2_np` (or extends `_legendre_np` to return
`L_b''`) alongside the existing complex-safe closed forms.

## Gauge analysis (why b=0 is dropped)

- **φ (conserved).** The `{1, φ_i}` per-species gauge is already removed inside
  `BasisMultiEnergy` (Legendre modes k≥2 are L2-orthogonal to `{1, φ_i}`).
- **ψ (non-conserved).** Adding a constant to a coupling function, `h_k → h_k + c`,
  shifts `dfdphi_k` by the constant `c` — i.e. a constant added to the exchange
  potential μ_k, which is exactly the **species-k `{1}` conservation gauge on φ_k**
  and therefore invisible to the conserved φ dynamics. `dfdpsi_k = φ_k h_k'` is
  unchanged (h_k' kills the constant). Hence **c_{k,0} is provably unidentifiable**
  and excluded. The linear mode b=1 is identifiable (it makes μ_k vary with ψ_k and
  gives a nonzero AC driving force), so `deg_ψ` starts at 1.
- **Diagnostic:** a `coupling_gauge_residual(k)` method (analogue of M6’s
  `gauge_residual`) confirming that the constant projection of the coupling channel
  is the only null direction.

## Tasks (mirrors M6 Plan A, end-to-end through recovery)

1. **`NeuralCrystalEnergy` energy object** — new file
   `src/diffsim/adjoint/neural_crystal.py` (kept separate from the engine, as
   `neural_multiphase.py` is from `multiphase.py`). Composes a `BasisMultiEnergy`
   base + the linear-in-φ Legendre-in-ψ coupling. Adds `_legendre2_np`. Implements
   the full `CrystalEnergy` protocol above. **Gate:** complex-step (1e-30) unit tests
   on every protocol derivative including the `cpl_*` params, dtype-preservation
   verified (no float casts on the φ/ψ/param path).

2. **Exact-match reduction gate** vs `AdditiveCrystalEnergy` — the analogue of M6’s
   "twin Legendre == hand path, byte-identical". Set the neural coeffs to the
   Legendre projection of the additive coupling `h_k^add(ψ) = q(ψ)dσ_k + p(ψ)drive_k`
   (degrees chosen to span q, p exactly). Assert `dfdpsi`, `d2fdphidpsi`,
   `d2fdpsidpsi`, `d2fdphidphi` match to ~1e-13, and `dfdphi` matches **up to the
   per-species b=0 gauge constant** (assert the ψ-varying part matches). This both
   validates the neural form and pins the gauge statement empirically.

3. **Twin coupling extension** — teach `CrystalCHTwin` (① Task 4) the torch mirror of
   the Legendre-in-ψ coupling correction so it produces gradients for the `cpl_{k}_{b}`
   leaves. Additive-only change; existing twin behavior unchanged. **Gate:** twin
   `cpl_*` grads vs central FD < 1e-6.

4. **Three-way gate** through `CrystalCHForward`/`CrystalCHAdjoint`/`CrystalCHTwin`
   with `names = ["cpl_0_1","cpl_0_2","basis_0_2","chi_0_1","N_0","mob_m0","kappa_0","eps2_0","L_0"]`
   over ternary K=(0,) bdf1, ternary K=(0,) bdf2, and quaternary K=(0,2) bdf1
   (adj/twin < 1e-10, adj/fd < 1e-6). Confirms the *learnable* coupling’s gradients
   are exact through the coupled march.

5. **Synthetic recovery** — plant a `NeuralCrystalEnergy` with known `cpl_*` (and a
   basis correction), generate a φ+ψ trajectory, then recover the coefficients by
   trajectory-matching the φ+ψ fields through the ① hand adjoint. Assert a large loss
   drop and coefficient recovery on the identifiable directions; document which
   directions are weakly identifiable from a single final snapshot (expected, and it
   motivates the ③ multi-snapshot + descriptor design, exactly as M6 Plan A Task 5
   did for the pure-φ basis). Ship a small `examples/` driver.

## Verification strategy

Three independent legs, per the house three-way rule: hand adjoint
(`CrystalCHAdjoint`) == autograd twin (`CrystalCHTwin`) == central finite differences.
Plus the exact-match reduction gate (Task 2) anchoring the neural form to the
already-verified parametric energy, and the complex-step unit gate (Task 1) on the
analytic derivatives.

## Designed-for / out of scope

- **In:** learnable coupled `f(φ,ψ)` (linear-in-φ, non-parametric in ψ), its exact
  gradients through the ψ-coupled adjoint, and synthetic recovery.
- **Out (deliberate):**
  - **Full 2-D coupling** `g_k(φ_k, ψ_k)` (φ-dependent coupling) → ②b if data demands it.
  - **χ(ψ)** crystallinity-dependent interaction + production parity → ①b.
  - **MD φ+ψ ingest** (real data, multi-snapshot + descriptor loss) → ③ / M6 Plan B analogue.
  - **θ (orientation)** adjoint — no θ dof anywhere.
  - **GPU** cuDSS backend for the 2M+K system — separate scaling piece.

## Interfaces

```python
class NeuralCrystalEnergy(CrystalEnergy):
    def __init__(self, chi, N, crystallizable, deg_psi=(1, 2), coeffs=None,
                 dom_phi=(0.05, 0.95), breg=0.0): ...
    # CrystalEnergy protocol: dfdphi, dfdpsi, d2fdphidphi, d2fdphidpsi,
    # d2fdpsidpsi, dfdphi_dparam, dfdpsi_dparam; param_names; crystallizable; M
    # + coupling_gauge_residual(k)
```

Consumes: `BasisMultiEnergy` (`neural_multiphase.py`), the shifted-Legendre helpers,
the `_q/_p` polynomials (reduction gate only). Produces: a drop-in `CrystalEnergy` for
`CrystalCHDiscrete`/`Forward`/`Adjoint`/`Twin` — no engine change.
