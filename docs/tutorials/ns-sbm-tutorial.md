# DiffSim incompressible Navier–Stokes: a VMS + SBM tutorial

**Audience.** You know finite elements and variational multiscale (VMS)
stabilization in general, but not this codebase. This document takes you from
the incompressible Navier–Stokes weak form to *running and exploring* DiffSim's
two NS engines — the **monolithic VMS saddle solver** and the
**pressure-projection (Helmholtz–Leray) solver** — including weak imposition of
boundary conditions (Nitsche) and the **Shifted Boundary Method** (SBM).

Every math claim is derived from the code and the in-repo design/synthesis
documents, with `file:line` citations. Every "expected result" number in the
worked examples was produced by running the example on the GPU box (`gpubox`),
not invented.

**How to read this.** Sections 2–6 are the math + code walkthrough (weak forms,
VMS $\tau$'s, Nitsche, SBM). Sections 7–8 are worked examples with runnable
scripts and verified numbers. Section 9 is the knob map for exploration. All
scripts live in `examples/ns_sbm/`; the pytest that validates them against these
numbers is `tests/test_tutorial_examples.py`.

---

## 1. Introduction & scope

### The two engines

DiffSim solves the incompressible Navier–Stokes equations with **equal-order
$P_1/P_1$** (and $P_2/P_2$) collocated velocity–pressure elements
(`ndof = dim + 1` per node, node-major: $u_1,\dots,u_{\dim},p$). Equal-order
pairs violate the inf–sup (LBB) condition, so **both** engines are
residual-based-VMS stabilized (SUPG + PSPG + grad-div). The two differ in how
they solve the coupled system each time step:

| | **Monolithic** | **Pressure-projection** |
|---|---|---|
| Class | `LinearizedMonolithicStepper` | `LerayProjectionStepper` / `LeraySBMStepper` |
| Solve | ONE coupled $(u,p)$ **saddle** system per step | predictor → **SPD** pressure-Poisson → velocity correction |
| Operator | indefinite $\begin{bmatrix}F&G\\D&C\end{bmatrix}$ | one nonsymmetric + two SPD sub-solves |
| Scalability | direct (splu/cuDSS) or block-preconditioned FGMRES | the **SPD PPE admits AMG/CG** → the 100M-DOF path |
| Role here | the **oracle** (exact reference) | the **scalable** engine |
| Source | `src/diffsim/steppers/linearized.py` | `src/diffsim/steppers/leray.py`, `leray_sbm.py` |

**When to use which.** The monolithic solver is the *correctness oracle*: it is
robust and its steady state is the bar every projection run is measured against
("does the split reproduce the same-mesh monolithic?"). The projection solver is
the *scalability lever*: its pressure step is a symmetric positive-definite (SPD)
Poisson problem that AMG/CG handles at scales where the indefinite saddle
factorization runs out of memory (`leray_sbm.py:19–23`). For 3-D drag on the
current feasible meshes, the monolithic path is the working one (Section 8).

### SBM in brief

The **Shifted Boundary Method** lets an immersed body sit *off* the mesh grid.
Instead of body-fitting, we keep a grid-aligned octree and pick a **surrogate
boundary** $\tilde\Gamma$ (whole element faces that hug the true boundary
$\Gamma$). A **Taylor shift** transfers the boundary condition from $\tilde\Gamma$
to $\Gamma$ using a distance vector $d$ (surrogate GP → closest point on
$\Gamma$). When the body *is* grid-aligned, $d=0$ and SBM degenerates to
ordinary Nitsche — which is why the examples build up from $d=0$ (Sections 7.1–7.2)
to $d\neq0$ (Sections 7.3, 8).

### Prerequisites & how to run

You need the DiffSim environment (Warp, SciPy, NumPy). All examples run on CPU
for the small meshes here; a GPU box is recommended for 3-D. From the repo root:

```bash
PYTHONPATH=src:tests:examples python examples/ns_sbm/lid_driven_cavity.py
PYTHONPATH=src:tests:examples python examples/ns_sbm/flow_past_square.py
PYTHONPATH=src:tests:examples python examples/ns_sbm/flow_past_cylinder.py
PYTHONPATH=src:tests:examples python examples/ns_sbm/sphere_3d.py
```

Validate them against the numbers in this document:

```bash
PYTHONPATH=src:tests:examples python -m pytest tests/test_tutorial_examples.py -q
```

---

## 2. Governing equations

### Strong form

On a domain $\Omega$ with density $1$ and kinematic viscosity $\nu$, the
incompressible Navier–Stokes equations for velocity $u$ and pressure $p$ are
(`ns_bricks.py:11–14`):

$$
\partial_t u + (u\cdot\nabla)u = -\nabla p + \nu\,\nabla\!\cdot(\nabla u) + f
\qquad\text{(momentum)}
$$
$$
\nabla\cdot u = 0 \qquad\text{(continuity / incompressibility)}
$$

### Boundary conditions

- **Dirichlet** (inflow, walls, moving lid, no-slip body): $u = g$ on
  $\Gamma_D$. Imposed **strongly** (row replacement) or **weakly** (Nitsche,
  Section 5 / SBM, Section 6).
- **Outflow / "do-nothing"**: the natural boundary condition of the weak form.
  Integrating the viscous and pressure terms by parts drops their surface terms
  at the outflow, leaving $\nu\,\partial_n u - p\,n = 0$ (a traction-free
  condition) (`ns_bricks.py:16–17`). This is the open outlet used in the channel
  examples.

### Nondimensionalization (Reynolds number)

The examples fix a reference speed $U$ and a body length scale $D$ and set the
viscosity from the Reynolds number:

$$
\nu = \frac{U\,D}{\mathrm{Re}}.
$$

In the channel fixtures $U = U_{\text{in}} = 1$ and $D = 2\times\text{half}$ (the
obstacle side length); the cavity uses $D=L=1$ (the cavity side)
(`ladder_fixtures.py:122–124`, `260–269`). Drag is reported as
$C_d = F_x / q_{\text{ref}}$ with $q_{\text{ref}} = \tfrac12 U^2 D$ in 2-D
(`ladder_rungA_square_strong.py:70–74`) and $\tfrac12 U^2 \pi R^2$ for the sphere
(`p2r0_task10_sphere_derisk.py:94–95`).

---

## 3. Monolithic VMS formulation

### Galerkin weak form

Test momentum with $w$ and continuity with $q$. Integrate the viscous and
pressure terms by parts; drop the do-nothing surface terms (`ns_bricks.py:16–21`):

$$
\int_\Omega w\cdot\partial_t u
+ \int_\Omega w\cdot(u\cdot\nabla)u
+ \nu\int_\Omega \nabla w : \nabla u
- \int_\Omega (\nabla\cdot w)\,p
= \int_\Omega w\cdot f,
$$
$$
\int_\Omega q\,(\nabla\cdot u) = 0.
$$

### Equal-order $P_1/P_1$ is inf–sup unstable → why stabilize

Equal-order velocity/pressure spaces do **not** satisfy the LBB condition, so the
pure Galerkin saddle system is singular/oscillatory in pressure. DiffSim adds
**residual-based VMS/SUPG–PSPG** stabilization on the momentum strong residual
(`ns_bricks.py:23–27`).

### Residual-based VMS: the fine scale $u' = -\tau_M R_M$

VMS splits $u = u_h + u'$ into a resolved (coarse) scale $u_h$ and an unresolved
(fine) scale $u'$, modeled algebraically as

$$
u' = -\,\tau_M\,R_M(u_h,p_h),
$$

where $R_M$ is the momentum strong residual. Substituting $u'$ back into the weak
form and integrating the fine-scale terms produces:

- **SUPG** (streamline-upwind on the momentum test): $\tau_M\,(a\cdot\nabla w)\cdot R_M$,
- **PSPG** (pressure-stabilizing, on the continuity test): $\tau_M\,(\nabla q)\cdot R_M$ — *this is the term that makes equal-order work*,
- **grad-div / LSIC**: $\tau_C\,(\nabla\cdot w)(\nabla\cdot u)$.

(`ns_bricks.py:23–27`, `44–51`.)

### $\tau_M$, $\tau_C$ definitions (metric form)

The monolithic steppers use the **metric-tensor** $\tau$ form
(`physics/vms.py:5–12`, `38–45`). On DiffSim's axis-aligned octree cubes the
element metric is diagonal ($G=(2/h)^2 I$), so:

$$
\tau_M = \Big[\,\Big(\tfrac{2b_0}{\Delta t}\Big)^2
+ \tfrac{4|u|^2}{h^2}
+ C_I\,\nu^2\,\dim\,\big(\tfrac{2}{h}\big)^4\,\Big]^{-1/2},
\qquad C_I = 36,
$$
$$
\tau_C = \frac{1}{\tau_M\,\dim\,(2/h)^2}.
$$

The transient term $(2b_0/\Delta t)^2$ is present when `timestab=True` (it makes
the stabilized *spatial* problem depend on $\Delta t$; dropped for steady/order
studies) (`physics/vms.py:38–44`, `linearized.py:42–46`). The device function is
`tau_m_metric` (`physics/vms.py:60–68`) and `tau_c_metric`
(`physics/vms.py:70–74`); the host mirror is `tau_metric_host`
(`physics/vms.py:38–45`).

> The projection engine instead uses the **h-based** $\tau$ form (Eq. 45 of the
> Helmholtz–Leray VMS draft), `tau_hbased_host` (`physics/vms.py:48–56`). Its
> defaults ($c_1=4$, $c_2 C_I = 36\cdot16\cdot\dim$) reproduce the metric form on
> cubes *by construction*, so the two engines are comparable term-for-term on
> these meshes (`physics/vms.py:13–19`).

### The saddle system $\begin{bmatrix}F&G\\D&C\end{bmatrix}$

After linearization (below), each node block is the full $(\dim+1)\times(\dim+1)$
coupling written by the element kernel `lin_ns_Ae` (`ns_bricks.py:341–447`):

- **$F$** (velocity–velocity, `ns_bricks.py:408–411`): mass $\sigma N_a N_b$ +
  convection $N_a(a\cdot\nabla N_b + s(\nabla\!\cdot a)N_b)$ + viscous
  $\nu\,\nabla N_a\!:\!\nabla N_b$ + SUPG $\tau_M(a\cdot\nabla N_a)R_{M,b}$;
  plus grad-div $\tau_C\,\partial_i N_a\,\partial_j N_b$
  (`ns_bricks.py:413–417`).
- **$G$** (velocity–pressure, `ns_bricks.py:418–423`): the pressure gradient
  $-\int(\nabla\!\cdot w)\,p$ in integrated-by-parts form
  $-\partial_i N_a\,N_b$, plus the SUPG pressure coupling
  $\tau_M(a\cdot\nabla N_a)\,\partial_i N_b$.
- **$D$** (continuity–velocity, `ns_bricks.py:424–429`): the divergence
  $N_a\,\partial_i N_b$ plus the PSPG velocity coupling
  $\tau_M\,\partial_i N_a\,R_{M,b}$.
- **$C$** (continuity–pressure, the **PSPG C-block**, `ns_bricks.py:442–444`):
  $\tau_M\,\nabla N_a\cdot\nabla N_b$. This nonzero pressure–pressure block is
  exactly what stabilizes the equal-order pair — without PSPG the $(p,p)$ block
  is empty and the saddle is singular.

### BDF time integration

The transient term uses backward differencing (BDF1 bootstrap → BDF2). Per step,
$\sigma = b_0/\Delta t$ enters the mass term and $\tau_M$'s transient part; the
history $-(b_1 u^n + b_2 u^{n-1})/\Delta t$ becomes part of the linearized strong
residual and flows through the SUPG/PSPG consistency terms
(`linearized.py:181–226`, `ns_bricks.py:36–41`).

### Convection linearization

The nonlinear convection $(u\cdot\nabla)u$ is handled by an **advecting field**
$a$ precomputed at Gauss points:

- **Picard/Oseen** (default): $a$ = a BDF **extrapolation** of the previous
  velocities, *fine-scale-corrected* $a=\text{extrap}(u_{\text{pre}} - \tau_M
  R_{M,\text{pre}})$ (`linearized.py:47–53`, `126–155`).
- **Newton** (optional): adds the cross-term $(\delta u\cdot\nabla)a$ block
  $N_a(\nabla a)_{ij}N_b$ (`ns_bricks.py:430–441`); the RHS partner
  $(a\cdot\nabla)a$ is folded into $f_{\text{eff}}$.

The convection uses the **skew-symmetric** form $a\cdot\nabla u + s(\nabla\!\cdot
a)u$ with $s=\tfrac12$ (energy-stable, self-adjoint) (`ns_bricks.py:36–41`,
`400–401`; `physics/vms.py:20–27`). The $-\nu\,\Delta u_h$ term in the linearized
residual (`ns_bricks.py:406–407`) is identically zero at $P_1$ (Q1 diagonal
second derivatives vanish) and required at $P_2$.

### Code walkthrough — math term → code

`assemble_linear_ns(dm, aq, div_aq, fq, nu, sigma, ...)`
(`ns_bricks.py:507–562`) builds the constrained node-major saddle matrix $A$ and
RHS $b$. It launches `lin_ns_Ae` (matrix) and `lin_ns_be` (RHS) per element bin,
then constrains with the vector transfer $T_{\text{vec}} = T\otimes I_{\dim+1}$
(`ns_bricks.py:560–562`).

The RHS kernel `lin_ns_be` (`ns_bricks.py:450–504`) carries the same SUPG/PSPG
consistency on the force $f$: $(N_a + \tau_M\,a\cdot\nabla N_a)f_i$ on the
momentum rows (`ns_bricks.py:496–497`) and $\tau_M\,\partial_i N_a\,f_i$ on the
continuity (PSPG) row (`ns_bricks.py:499–501`).

`LinearizedMonolithicStepper.step()` (`linearized.py:181–245`) orchestrates one
step: compute the BDF order/coefficients, build the advecting field $a$ (with the
fine-scale correction `_corrected_gp`, `linearized.py:126–155`), assemble via
`assemble_linear_ns`, apply strong velocity Dirichlet rows + a single pressure
pin, solve, and rotate the history. The strong rows and pressure pin are set by
overwriting matrix rows to identity (`linearized.py:227–236`).

---

## 4. Pressure-projection (Helmholtz–Leray)

The projection engine implements Algorithm 1 of the Helmholtz–Leray VMS draft
(`leray.py:1–23`). Per step (with $p^\* $ = extrapolated pressure, order 1
$p^\* = \hat p^n$):

1. **Momentum predictor** (nonlinear, Picard): solve the monolithic block with
   the **pressure DOFs pinned** to $p^\*$'s nodal values, so the momentum rows
   see $\nabla p^\*$ through the coupling and $\tau$/fine-scale terms are
   recomputed per iterate (`leray.py:6–13`, `_predict` `leray.py:719–830`).
2. **PPE** (SPD pressure-Poisson for the increment $\phi = \hat p - p^\*$):
   $$(\nabla \phi,\nabla q) = \sigma\,(\hat u - \tau_M r_m(\hat u,p^\*),\ \nabla q)$$
   with the h-based $\tau_M$ (`leray.py:14–19`, `948–1106`).
3. **Velocity correction** (fine scale retained, consistent-mass L2 projection):
   $$u^{n+1} = \hat u - \tfrac{1}{\sigma}\big(\nabla\hat p - \nabla p^\*\big).$$
4. $p^\* \leftarrow \hat p$ (first-order incremental).

### Why projection scales: SPD-PPE → AMG → 100M

The pressure step is a **symmetric positive-definite** Poisson problem. Unlike
the indefinite saddle, it is the natural target for algebraic multigrid + CG,
which is the mandated path to the 100M-DOF hero (`leray_sbm.py:19–23`). This is
the whole reason projection exists alongside the monolithic oracle.

### The VMS-consistent PPE (`consistent_projection`)

Naïvely splitting the projection *breaks faithfulness*: the split's steady state
is generally **not** the monolithic steady state. The `consistent_projection`
mode (the 2026-07-23 fix, `leray.py:48–72`) is the single switch that turns on
the coherent VMS-stabilized Helmholtz–Leray set so the monolithic steady state
becomes a **fixed point** of the split. It bundles:

- **#1 PSPG-consistent PPE operator** — the coarse divergence is kept
  **collocated** with the test $q$ (NOT integrated by parts), so the split's
  discrete incompressibility matches the monolithic PSPG continuity row
  term-for-term (`leray.py:1007–1034`). The full-flux by-parts (the older
  `ppe_fine_scale` path) fabricates a spurious boundary term
  $\sigma(u_h\cdot n, q)_\Gamma$ at an open outflow (`leray.py:1015–1019`).
- **#2 fine-scale $u' = -\tau_M r_m$ in BOTH the PPE source AND the L2
  correction**, with the consistent mass $D=G^\top$ (`leray.py:53–61`).
- **#3 disjoint outflow BCs** (Baskar's rule): velocity is left free at the
  outflow (natural do-nothing $\partial_n u = 0$), while the pressure
  **correction** $p'=0$ is imposed as a Dirichlet condition on the outflow nodes
  (`pressure_outflow_nodes`) (`leray.py:62–69`). This is the authoritative fix:
  $\nabla u\cdot n=0$ in the predictor only, $p'=0$ in the PPE only.
- **#4 rotational-incremental pressure update** (Timmermans 1996):
  $p = p^\* + \phi - \nu\,\nabla\!\cdot\hat u$, valid for constant $\nu$
  (`leray.py:62`, `1114–1168`).
- **#5 P1 boundary-vorticity outflow term** and **#6 backflow (directional
  do-nothing) stabilization**, both on the outflow (`leray.py:73–118`,
  `ns_bricks.py:95–338`).

Two secular-drift cures ride with the mode (`leray.py:121–168`): the rotational
correction $q$ is pinned to zero on the outflow rows
(`rotational_pin_outflow`, `leray.py:1133–1135`) and, for immersed walls, on the
SBM/wall nodes (`rotational_pin_wall`, `leray.py:1163–1167`) — the latter is
**the drag fix** for weak-Nitsche bodies (see Section 7.2).

### Fine scale threaded through all three sub-solves

The fine-scale velocity $-\tau_M r_m$ is computed once in the PPE pass and
**stashed** (`fs_vel`, `leray.py:973`, `1004–1006`) so the same $u'$ enters the
PPE source *and* the velocity correction — the consistency that makes the
correction a true discrete projection.

### Code walkthrough — the `consistent_projection` knobs

`LerayProjectionStepper.__init__` (`leray.py:36–348`) exposes the mode and its
sub-knobs. Setting `consistent_projection=True` *overrides* `ppe_fine_scale=True`
and `pressure_update="rotational"` (`leray.py:161–169`) so a later explicit
kwarg cannot half-enable it. `step()` (`leray.py:833–946`) runs one
predictor→PPE→pressure-update **pass** (`_projection_pass`, `leray.py:948–1171`)
then the correction (`_correct_and_finish`, `leray.py:1282+`). The PPE source
assembly is the heart: `flux = -sigma * fs_vel` (fine scale, by parts) plus a
**collocated** coarse-divergence term `be0` (`leray.py:1024–1034`).

### `LeraySBMStepper` — the composed engine

`LeraySBMStepper` (`leray_sbm.py:81–160`) *composes* the base projection stepper
with an immersed-geometry oracle (it does **not** fork it). It runs the surrogate
pipeline once, builds the shifted-Nitsche face block once
(`sbm_vector_dirichlet`, `leray_sbm.py:147–152`), and injects it into the
predictor via the `extra_block`/`sbm_nodes` hook (`leray_sbm.py:378–381`). The
box inflow/walls are strong; the immersed body is weak (SBM). Public ergonomics
(`set_initial`, `step()`, `divergence_l2()`, `surrogate_traction()`) delegate to
the base.

### Monolithic (oracle) vs projection (scalable) — the standing verdict

The projection is *validated by faithfulness* to the same-mesh monolithic. In
2-D this holds end-to-end (Sections 7.1–7.3). In 3-D the monolithic is stable and
physical while the projection split's long-time transient is not yet stable at
feasible mesh (Section 8) — a documented R2 item.

---

## 5. Weak imposition — Nitsche's method

Instead of overwriting rows to set $u=g$ on $\Gamma_D$, Nitsche's method imposes
the Dirichlet condition **weakly** through consistent boundary integrals. For the
component-diagonal (Laplacian) viscous form DiffSim uses (each velocity component
gets the scalar Poisson-Nitsche terms with $\kappa\to\nu$,
`sbm/vector.py:1–14`), the added boundary terms on $\Gamma_D$ are:

$$
\underbrace{-\,\nu\,\langle \partial_n u,\ w\rangle_{\Gamma_D}}_{\text{consistency}}
\;\underbrace{-\;\nu\,\langle u-g,\ \partial_n w\rangle_{\Gamma_D}}_{\text{adjoint-consistency}}
\;+\;\underbrace{\frac{\alpha\nu}{h}\,\langle u-g,\ w\rangle_{\Gamma_D}}_{\text{penalty}}.
$$

- **Consistency** ($-\nu\langle\partial_n u,w\rangle$): recovers the exact
  solution — the true $u$ satisfies the weak form with these terms.
- **Adjoint-consistency** ($-\nu\langle u-g,\partial_n w\rangle$): symmetrizes
  the bilinear form (optimal $L^2$ convergence) and vanishes when $u=g$.
- **Penalty** ($\tfrac{\alpha\nu}{h}\langle u-g,w\rangle$): coercivity; the
  scale $\alpha\nu/h$ ($\alpha$ = dimensionless penalty, $h$ = element size)
  balances the consistency term. Too small → the condition leaks; too large →
  ill-conditioning. The examples use $\alpha=10$
  (`ladder_rungB_square_nitsche.py:69`).

**Sign conventions.** DiffSim's assembly is the one-sided Laplacian form
($-N_a\,\partial_n N_b$, i.e. `-Na*gnb` in the scalar kernel referenced at
`sbm/vector.py:492–498`), consistent with the one-sided viscous traction the drag
integral reads (Section 6).

### Code — `sbm_vector_dirichlet`

`sbm_vector_dirichlet(dm, sf, geo, g_fn, nu, ndof, alpha=10.0, ...)`
(`sbm/vector.py:23–82`) assembles the vector Nitsche block over full node-major
DOFs. It **reuses the verified scalar Poisson-Nitsche face kernels**
(`make_sbm_dirichlet_Ae/_be`) and places each scalar block at node-major position
`node*ndof + c` for component `c < dim` (`sbm/vector.py:73–78`) — no new device
code, the scalar patch-exactness carries over verbatim. The boundary data is
evaluated at the *mapped* (true-boundary) points $g(\text{xq}+d)$
(`sbm/vector.py:34`) — this is where the SBM shift $d$ enters (Section 6). A
penalty-only variant `sbm_vector_penalty` (`sbm/vector.py:85–141`) is used to
re-pin the wall trace after the projection correction (`leray.py:1287–1294`).

---

## 6. The Shifted Boundary Method (SBM)

### Surrogate vs true boundary

The true immersed boundary $\Gamma$ (level set of an oracle, e.g. a `Box` or
`Sphere`) generally cuts through octree cells. SBM does **not** body-fit. Instead
it classifies cells and keeps a **surrogate boundary** $\tilde\Gamma$ made of
*whole* element faces (`sbm/surrogate.py:14–21`):

- `classify_lambda(tree, oracle, lam, domain)` (`sbm/surrogate.py:40–100`) marks
  a cut cell as *retained* iff its domain-**outside** volume fraction
  $\le \lambda$. So `lam=0.0` keeps only fully-interior cells (surrogate strictly
  inside $\Omega$, and — if the body is cell-aligned — the surrogate coincides
  with $\Gamma$ exactly, $d=0$); `lam=0.5` is the flow-past value
  (`sbm/surrogate.py:44–50`).
- `extract_surrogate(tree)` (`sbm/surrogate.py:110–157`) returns the surrogate
  faces: faces of retained cells with no retained neighbour across them and not
  on the outer domain boundary. Whole-face invariant: partially-exposed faces
  raise.

### The Taylor shift $S = N + \nabla N\cdot d$

Let $d$ be the vector from a surrogate Gauss point to its closest point on the
true boundary $\Gamma$. The **shifted** trial/test operator is a first-order
Taylor extrapolation from the surrogate to the true boundary:

$$
S\,N_a \;=\; N_a + (\nabla N_a)\cdot d.
$$

The Dirichlet data is imposed at the mapped point $y = \text{xq} + d$
(`sbm/vector.py:34`), and the shifted trial value $Su = u + (\nabla u)\cdot d$
(`leray_sbm.py:299–302`). When $d=0$ (cell-aligned body), $S N_a = N_a$ and SBM
is exactly standard Nitsche.

### Surrogate normal $\bar n$, distance $d$, area correction `corr`

`GeometryData.evaluate` (`sbm/surrogate.py:222–253`) produces the frozen FP64
per-epoch geometry cache at surrogate Gauss points:

- **`d`** — the shift vector (surrogate GP → closest point on $\Gamma$), from the
  oracle's `distance_vector` (`sbm/surrogate.py:229`).
- **`n`** — the *true-boundary* outward normal $n = \pm\nabla\psi/|\nabla\psi|$,
  always pointing **out of** the computational domain $\Omega$
  (`sbm/surrogate.py:11–13`, `248`).
- **`corr`** — the **area correction** $\text{corr} = \tilde n\cdot n$, where
  $\tilde n$ is the surrogate face's grid-aligned unit normal
  (`sbm/surrogate.py:249–250`). It converts a surrogate-face measure to the true
  boundary measure ($\text{corr}=1$ when $\tilde\Gamma=\Gamma$).

### Shifted no-penetration

In the projection factoring, the continuity constraint at the body becomes a
**shifted no-penetration** condition on the PPE: subtract the surrogate
shifted-normal velocity $\tilde n\cdot(\hat u + (\nabla\hat u)\cdot d)$ so the
corrected field carries no flow through the body
(`leray_sbm.py:245–305`, the `_t6_nopenetration_flux` hook). By default the PPE's
natural surrogate BC is homogeneous Neumann $\nabla\phi\cdot\hat n = 0$ (Suresh
Remark 3.9), which already preserves the predictor's blockage
(`leray_sbm.py:383–415`).

### Force / drag on the immersed body

The drag reads the traction on the surrogate faces, area-corrected to the true
boundary. `surrogate_traction(dm, sf, geo, x_all, nu, ndof)`
(`sbm/vector.py:562–605`) computes

$$
F = \oint_{\tilde\Gamma} \big(p\,n - \nu\,(\nabla u)^\top n\big)\ \text{corr}\ \mathrm{d}\tilde\Gamma,
$$

with the crucial **orientation contract**: `geo.n` is domain-outward (= *into*
the obstacle for exterior flow), so the obstacle-outward normal is
$\hat n = -\text{geo.n}$ and the integrand is $+p\,\text{geo.n} - \nu(\nabla
u)^\top\text{geo.n}$ (`sbm/vector.py:562–604`; the comment records $C_d=-2.85\to
+2.85$ before/after the flip). There are two integration approaches: this
surrogate-boundary + area-correction form (approach 2, used throughout), and a
field-extrapolation-to-true-boundary form `sbm_consistent_flux`
(`sbm/vector.py:454–559`) reproducing the NSHT_SBM production drag (pressure +
viscous of the field Taylor-shifted to $\Gamma$, no penalty). At $d=0$ both
reduce to the raw body-fitted traction.

---

## 7. Worked 2-D examples

Each example marches BOTH engines on the SAME mesh and compares them; the bar is
the **same-mesh monolithic** (the box-free oracle), not a literature value (the
confined unit-box domain inflates the drag; both engines see the same
confinement). All numbers below were produced on `gpubox` (CPU/splu) and are
re-verified by `tests/test_tutorial_examples.py`.

### 7.1 Lid-driven cavity (Ghia)

**Geometry / fixture.** Closed unit square, no obstacle. Top lid ($y=1$) slides
at $U=1$; the other three walls are no-slip. Level-4 mesh = $16\times16$ cells,
$P_1/P_1$. Enclosed flow, so BOTH engines pin one pressure DOF (free-node 0).
Build: `build_cavity` in `examples/ns_sbm/lid_driven_cavity.py`.

**Stepper setup.** Projection: `LerayProjectionStepper(..., order=1,
picard_iters=1)` (order-1 BDF pseudo-time to steady). Monolithic:
`LinearizedMonolithicStepper(..., order=1)`. Both use the same `lid_g` Dirichlet
trace.

**Run.**
```bash
PYTHONPATH=src:tests:examples python examples/ns_sbm/lid_driven_cavity.py 4 100 200
```

**Expected result** (verified: level 4, Re=100, dt=0.05, 200 steps):

| station $y$ | projection $u$ | monolithic $u$ | Ghia $u$ |
|---|---|---|---|
| 0.9766 | +0.8376 | +0.8529 | +0.84123 |
| 0.5000 | −0.1894 | −0.1462 | −0.20581 |

$$
\max|\text{proj}-\text{mono}| = 0.0486,\quad
\max|\text{proj}-\text{Ghia}| = 0.0164,\quad
\max|\text{mono}-\text{Ghia}| = 0.0620.
$$

The projection tracks the monolithic on the same mesh, and both track Ghia's
$129^2$ table within the coarse-mesh tolerance. (Note: the equal-order VMS
pointwise $\|\nabla\!\cdot u\|$ is *not* driven to zero — it is finite/bounded
(~2 here); the scheme controls the *weak* divergence, and $\|\nabla\!\cdot u\|$ is
the wrong steady-state gate, `ladder_rung0_cavity.py:189–193`.)

**Knobs to explore.** `LEVEL` (5–6 tightens toward Ghia, slower); `RE` (100 or
400; 400 needs more steps); `NSTEPS`.

### 7.2 Flow past a square (SBM $d=0$; strong AND weak Nitsche)

**Geometry / fixture.** A cell-aligned square (`half = 0.125 = k/2^{\text{level}}`)
carved from a unit-box channel with `lam=0.0`, so the surrogate faces coincide
with the true box: $d_{\max}=0$, $\text{corr}=1$ — **standard Nitsche**, no
shift. Strong inflow ($x=0$, $u=U_{\text{in}}$) + lateral walls; free outflow
($x=1$). Build: `build_square_channel_2d` (`ladder_fixtures.py:207–219`).

**Two no-slip modes** (`examples/ns_sbm/flow_past_square.py`):
- **strong** — obstacle nodes join the strong Dirichlet set (row replacement).
- **weak** — no-slip imposed by the Nitsche block (`sbm_vector_dirichlet`);
  obstacle nodes stay free (skip the strong overwrite).

The projection runs `consistent_projection` mode; the weak path additionally uses
the rotational **wall pin** (`rot_pin_wall=True`) — *the drag fix* for weak
bodies (`leray.py:1136–1167`). Drag from `surrogate_traction` (exact body-fitted
at $d=0$).

**Run.**
```bash
PYTHONPATH=src:tests:examples python examples/ns_sbm/flow_past_square.py weak 4 40
```

**Expected result** (verified: level 4, Re=40, dt=0.02, weak Nitsche):

| | projection | monolithic | rel-diff |
|---|---|---|---|
| $C_d$ | **+1.3529** | **+1.3941** | 2.96% |
| mean $|u|$ | 0.9057 | 0.8550 | 5.93% |

The weak-Nitsche projection matches the same-mesh weak-Nitsche monolithic in both
drag and velocity — the Nitsche layer preserves faithfulness on the working
projection (rung B verdict).

**Note on the "strong" base split.** The base single-pass projection on an *open*
outflow (without the consistent-projection machinery) is pressure-unstable and
can blow up — a known rung-A finding (`ladder_rungA_square_strong.py:150–163`).
The validated, robust path is the weak `consistent_projection` one shown here.

**Knobs.** `MODE` (weak/strong); `ALPHA` (0 removes the penalty → the obstacle
stops being felt: an anti-vacuity check); `RE` (100 sheds).

### 7.3 Flow past a body with a genuine SBM shift ($d\neq0$, $C_d$)

**Geometry / fixture.** The SAME square, but its center is offset by a sub-cell
fraction (`offset=0.05`), so the grid-aligned surrogate no longer coincides with
the true `Box` face: $0 < d_{\max} < h$. Now the shift is genuinely active — the
Taylor $(\nabla N_a)\cdot d$ term in the Nitsche block AND the area correction
`geo.corr` do real work (`ladder_rungC_square_shift.py:1–52`). Build:
`build_square_channel_2d(..., offset=0.05)`.

**Stepper setup.** *Identical* to 7.2's weak path — the shift is fixture **data**
(`geo.d`, `geo.corr`), not solver code. The monolithic oracle carries the SAME
shifted geometry, so it is the same-mesh-**with-shift** oracle.

**Run.**
```bash
PYTHONPATH=src:tests:examples python examples/ns_sbm/flow_past_cylinder.py 4 40
```

**Expected result** (verified: level 4, Re=40, offset=0.05):

$$
d_{\max}=0.0500\ \ (0<d_{\max}<h=0.0625,\ d_{\max}/h=0.80)
$$

| | projection | monolithic (true shift) | rel-diff |
|---|---|---|---|
| $C_d$ | **+1.5407** | **+1.5457** | **0.32%** |
| mean $|u|$ | 0.9090 | 0.8527 | 6.60% |

The consistent-projection split *with the genuine SBM shift active* matches the
same-mesh-with-shift monolithic to 0.3% in drag — **projection + SBM works
end-to-end in 2-D**.

**Anti-vacuity (the shift is load-bearing).** Run with `--zero-shift` to set
`geo.d=0`, `geo.corr=1` while keeping everything else fixed; the match against the
TRUE shifted oracle breaks — proof the Taylor term and area correction are doing
real work (`ladder_rungC_square_shift.py:37–49`).

**Knobs.** `OFFSET` (0 → back to body-fitted; must stay $<h$); `LEVEL`; `RE`.

---

## 8. Worked 3-D example — SBM sphere (the hero regime)

**Geometry / fixture.** An immersed `Sphere(center=(0.35,0.5,0.5), R=0.12)` carved
from a unit-box octree channel (`lam=0.5`), `ndof=4`, strong inflow ($x=0$) +
lateral walls ($y,z$), free outflow ($x=1$), weak no-slip on the sphere. The
sphere is not grid-aligned, so the SBM shift $d\neq0$ is genuinely active in 3-D.
Build: `build_sphere_3d` (`p2r0_task10_sphere_derisk.py:52–77`). At level 4:
`n_free = 4907`, 64 surrogate faces, $D/h = 3.84$.

**The honest 3-D verdict** (documented,
`tests/baselines/p2r0_task10_sphere.json`):

- **MONOLITHIC 3-D SBM-NS** — the **working** drag path. Stable and physical: it
  recovers from the startup transient to a positive steady drag,
  $$C_d = +0.381\quad(\text{Re}=100,\ \alpha=10,\ \text{level }4),$$
  reproducing the M1b sphere lock. This is the hero result for 3-D drag. It
  scales via GPU direct (`solver="cudss"`) or block-preconditioned FGMRES past
  the host-splu wall (~level 5 / 143k DOF)
  (`p2r0_task10_sphere_derisk.py:139–281`).
- **PROJECTION + volumetric SBM** (`LeraySBMStepper`) — the composition is
  **de-risked** (finite, axisymmetric $|C_{\text{lat}}|\ll|C_d|$, BDF2 engages,
  and the PPE-space weak divergence is machine-zero,
  `test_p2r0_projection_sbm.py:1130–1244`), **but** its long-time drag transient
  is unstable at feasible 3-D mesh (the coupled iteration's $C_d$ diverges
  monotonically while staying finite — a documented NEEDS_CONTEXT / R2 item,
  `test_p2r0_projection_sbm.py:1250–1264`). So in 3-D we validate the pipeline
  invariants and take drag from the monolithic.

The confined unit-box domain inflates $C_d$ above the unconfined literature value
(~0.6–0.7 at Re=300); both engines see the same confinement, so the bar is
faithfulness, not the literature number.

**Run.**
```bash
# hero drag path (monolithic; level-4 splu is ~minutes)
PYTHONPATH=src:tests:examples python examples/ns_sbm/sphere_3d.py 4 100 10
# additionally check the projection composition invariants (short window)
PYTHONPATH=src:tests:examples python examples/ns_sbm/sphere_3d.py 3 100 10 --pipeline
```

**Expected result.** MONOLITHIC steady $C_d = +0.381$ (positive, physical). The
`--pipeline` leg reports `finite=True`, `BDF2 engaged=True`, and an axisymmetric
traction ($|C_{\text{lat}}|/|C_d|$ at machine-noise level).

**Solver path.** Host splu here (small). To scale up, build the fixture on a CUDA
device and route the monolithic solve through `solver="cudss"` (GPU direct) or
`solver="blockamgx"` (block-preconditioned FGMRES: Cahouet–Chabard Schur +
AMG-on-$F$), both wired in `monolithic_cd`
(`p2r0_task10_sphere_derisk.py:179–281`).

**Knobs.** `LEVEL` (4 is the de-risk resolution); `RE`; `ALPHA` (10 is the
monolithic-stable value here).

---

## 9. Exploration guide — the knob map

### The knob map

| Knob | Where | Effect |
|---|---|---|
| `consistent_projection` | `LerayProjectionStepper` / `LeraySBMStepper` | turns on the coherent VMS-Helmholtz-Leray set (#1–#6); makes the monolithic steady state a fixed point of the split. Default OFF = bit-for-bit base split (`leray.py:48–72`). |
| `pressure_update` | projection steppers | `"standard"` ($p=p^\*+\phi$) / `"rotational"` (Timmermans $-\nu q$) / `"chorin"` (non-incremental). `consistent_projection` forces `"rotational"` (`leray.py:191–205`). |
| `rotational_pin_wall` | projection steppers | pin the rotational $-\nu q$ correction to 0 on the immersed-wall nodes — **the drag fix** for weak bodies (`leray.py:1136–1167`). |
| `pressure_outflow_nodes` | projection steppers | disjoint outflow BC: $p'=0$ Dirichlet on outflow (external flow) vs. node-0 pin (enclosed) (`leray.py:216–232`). |
| `solver` | all steppers | `"splu"` (host direct) / `"cudss"` (GPU direct) / `"fused"` (device BiCGStab) / `"amgx"`/`"blockamgx"` (AMG). |
| `order` | all steppers | BDF order (1 or 2; BDF1 bootstrap → BDF2 at $t\ge1.5\Delta t$). |
| `picard_iters` | projection steppers | predictor Picard iterations (drives the predictor↔PPE coupling). |
| `alpha` | SBM steppers | Nitsche penalty scale ($\alpha\nu/h$); 0 removes the penalty (anti-vacuity). |
| `beta_backflow` | SBM steppers | outflow/surrogate directional-do-nothing backflow stabilization. |
| `Re` / `level` | fixtures | Reynolds number / mesh refinement. |

### Sweeps

- **Re sweep**: Re=40 (steady) → Re=100 (vortex shedding; the driver extracts a
  Strouhal number from the lift signal, `ladder_rungA_square_strong.py:308–327`).
- **Mesh (level) sweep**: refine to tighten toward Ghia / a converged $C_d$;
  watch the splu wall (~level 5 in 3-D) and switch to `cudss`/AMG.
- **$\alpha$ sweep**: the sphere de-risk sweeps $\alpha\in\{10,20,50,100\}$ to
  find the stable-physical regime (`p2r0_task10_sphere_derisk.py:296–357`).

### Reading diagnostics

- `st.divergence_l2()` — the pointwise $\|\nabla\!\cdot u\|$ sentinel. For the
  equal-order VMS scheme this is NOT driven to zero; require it *finite/bounded*,
  not tiny (`ladder_rung0_cavity.py:189–193`, `linearized.py:397–409`).
- The **PPE-space solenoidality identity**
  $\|\sigma B^\top\hat u - K_p\phi\|\approx0$ is the right divergence gate — it is
  machine-zero at the fixed point (`test_p2r0_projection_sbm.py:1199–1235`).
- `surrogate_traction` → $C_d = F_x/q_{\text{ref}}$, $C_l = F_y/q_{\text{ref}}$;
  a small $|C_{\text{lat}}|/|C_d|$ confirms an axisymmetric 3-D wake.

---

## 10. References

**In-repo derivation & verdict documents**
- `docs/dev/2026-07-23-projection-ladder-FINAL-synthesis.md` — the projection
  validation ladder capstone (rungs 0→C, the end-to-end 2-D story).
- `docs/dev/specs/2026-07-23-consistent-projection-fix-design.md` — the
  `consistent_projection` fix (#1–#6) design.
- `tests/baselines/p2r0_task10_sphere.json` — the 3-D sphere de-risk record
  (monolithic stable $C_d=+0.381$; projection-split instability finding).

**Group papers referenced in the code**
- ns_projection_vms (Helmholtz–Leray VMS draft) — Algorithm 1, Eqs. 44a–c, 45,
  47–49, Remark 2.3 (the projection engine; `leray.py:1–23`, `48–72`).
- Timmermans (1996) — rotational-incremental pressure update (`leray.py:1114`).
- Bazilevs et al. (CMAME 2009), Esmaily-Moghadam et al. (Comput. Mech. 2011) —
  directional-do-nothing backflow stabilization (`ns_bricks.py:95–115`).
- Pacheco, Schussnig, Steinbach, Fries (IJNME 2021, arXiv:2411.02100) — P1
  boundary-vorticity outflow term (`ns_bricks.py:175–338`).
- Suresh (octree-SBM pressure-projection paper) — Remark 3.9 homogeneous-Neumann
  surrogate BC (`leray_sbm.py:383–415`).
- Dokken/Karniadakis et al. — Nitsche wall pressure-traction / KIO consistent
  pressure Neumann BC (`sbm/vector.py:144–194`, `197–336`).

**Code map**
- Engines: `src/diffsim/api/ns_bricks.py` (`assemble_linear_ns`, $\tau$, PSPG),
  `src/diffsim/steppers/linearized.py` (monolithic),
  `src/diffsim/steppers/leray.py` + `leray_sbm.py` (projection + SBM).
- SBM: `src/diffsim/sbm/surrogate.py` (`classify_lambda`, `extract_surrogate`,
  `GeometryData`), `src/diffsim/sbm/vector.py` (`sbm_vector_dirichlet`,
  `surrogate_traction`).
- VMS $\tau$: `src/diffsim/physics/vms.py`.
- Examples: `examples/ns_sbm/{lid_driven_cavity,flow_past_square,flow_past_cylinder,sphere_3d}.py`.
- Tests: `tests/test_tutorial_examples.py`.
