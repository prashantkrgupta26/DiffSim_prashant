# Mac brainstorming handoff — full program context

**Snapshot: 2026-07-15 ~23:20 CDT.** Supersedes
`docs/dev/2026-07-12-continuity-handoff.md` (still accurate on gpubox hazards,
Sec 6 there). Purpose: Baskar is moving the *brainstorming/driving seat* to his
Mac. This file carries the program state that is **not derivable from the code
alone** — rulings, open decisions, what's blocked on whom, and the live
frontier — so a fresh Claude Code session on the Mac starts with full context.

**How to use this on the Mac:** clone `github.com/BaskarGS/diffsim`, open a
session in the repo root, and say *"read docs/dev/2026-07-15-mac-brainstorm-handoff.md
and let's brainstorm."* Everything below is in git; the caveats in Sec 7 list
what is not.

---

## 0. What DiffSim is (one paragraph)

A GPU-native differentiable octree-SBM finite-element framework (NVIDIA Warp).
Octree meshing + surrogate-boundary immersed FEM + a differentiable adjoint
stack, applied to incompressible flow, heat/mass transport, and phase-field
(Cahn–Hilliard / Allen–Cahn) morphology evolution in drying organic-electronic
films. The scientific arc: *forward solver → replication of published
experiments → inverse problems (recover geometry / recover thermodynamics)*.
Production is GPU-only by program rule.

## 1. Program map — what is done

All of these are **gated** (a test or benchmark measures them) and pushed.
Headline numbers are measured and locked in `tests/baselines/*.json` or the
findings ledgers in `docs/dev/`.

| Milestone | State | Headline |
|---|---|---|
| **M0 / M0.5** — octree foundations, k-generic refactor | ✅ | 2:1 balance, hanging constraints, k = 2/3/4 |
| **M1a** — SBM foundations | ✅ | 3-D asymptotic 2nd order to L7 on Nova (1.97/1.94/2.03); dyadic-radius halo rule |
| **M1b** — incompressible NS | ✅ | cavity Re 100/1000; cylinder Re 20 **C_d = 1.352**; sphere Re 300 |
| **M1c** — adjoints + neural-SDF geometry | ✅ | sphere steady **1.8e-3**, transient **2.6e-4** |
| **M1d** — device migration | ✅ | cuDSS **300×** over host splu; H1 hero epoch **44.3×** |
| **M2** — heat/mass + closures + p2-NS | ✅ | de Vahl Davis **0.05 % / 0.02 %**; unified penalty law α ~ Pe × p² |
| **M3** rungs 1–2 — differentiable adaptivity | ✅ | **the bunny headline**: cell-scale GENIE ear-edit recovered from transient flow, err **1.23e-3** |
| **M4** — differentiable phase-field + learned thermo | ✅ | Wodo CMS-2012 films, **all 14 cases in ~11 min on one A100** (~100× the 2012 cost); **first learned free energy** (FH χ's + k_e to **1e-7**) |
| **M5 "OrgElMorph"** — coupled phase separation + crystallization | ✅ | S0–S4 green; the (M,K) factory is **config-only** (quaternary needed zero code changes) |
| **Learning rung 2** — beyond-Flory–Huggins | ✅ (just landed) | see Sec 2 |

Supporting infrastructure that also landed: the **blockch CH preconditioner**
(G1–G5 — full-res Wodo 3-D, 230×230×70, **15.15 M dofs on one RTX 6000 Ada**
where the paper used 256 CPUs); the **film front-end** (`diffsim.film`,
FilmParams + 13 named YAML configs + 4-layer RunLog); the **Negi-2018
spin-coating replication**; the **178-pp OrgElMorph course** + docs site; CI
(CPU gates green, GPU gate staged).

**83 test files.** Milestone reports live at `docs/dev/m{1,1d,2,4}-milestone-report.md`.

## 2. The live frontier — beyond-Flory–Huggins (task #59, COMPLETE)

The roadmap's *"Learning rung 2: instrument-space observables (S(q,t), film
height h(t), PSF) + composition-diverse protocols → genuinely beyond-Flory–Huggins
recovery."* Finished and pushed on 2026-07-14/15 in four verified layers.

**The science.** M4 learned the *parametric* FH interaction and, in doing so,
mapped a hard gauge structure. A correction to f'(c) learned on top of FH has
two structurally unidentifiable modes:

- **T0 (constant in f')** shifts μ by a constant → invisible to conserved
  c-dynamics. No data fixes it, ever.
- **T1 (linear in f')** integrates to a *quadratic* free energy already spanned
  by the FH χ term. Concretely **∂f'/∂B = (1 − 2c) is exactly proportional to
  the Legendre P₁ mode** — χ and any linear correction are exactly aliased (M4
  measured the trajectory difference along this mode at **5e-15**).

So a beyond-FH correction must live orthogonal to {1, c} — cubic-and-up in f.
That is the whole design premise of what was built.

**What was built** (`src/diffsim/adjoint/neural_energy.py`): two energy heads
exposing an `.fp(c)/.fpp(c)` contract that `adjoint/torch_twin.CHTwin` consumes:

- `NeuralCHEnergy` — FH + a small MLP correction, **L2-projected orthogonal to
  {1, c} every evaluation** (projection coefficients differentiable in the
  weights, so the optimizer never sees the aliased directions). Analytic
  input-derivative → f''(c) closed form, no nested autograd inside Newton.
- `BasisCorrEnergy` — FH + shifted-Legendre degrees k ≥ 2, gauge-anchored
  **exactly** by orthogonality. The interpretable sibling; a handful of
  coefficients whose Jacobian conditioning *is* a clean identifiability number.

**The four measured results:**

| Layer | Result |
|---|---|
| Gauge identifiability | un-anchored {χ, P₁, P₂, P₃} cond **1.4e16** (singular) vs anchored {χ, P₂, P₃} cond **1.8e1** — a **7.6e14×** gap |
| Differentiable S(q) | `diagnostics/structure_factor_torch.py`; numpy parity ~1e-16, S(q)-loss gradient through the twin vs FD **1.8e-11** |
| Composition coverage | one narrow trajectory cond ~2e2–5e2 vs composition-diverse ~14–18 → **15–30×** better conditioned |
| **End-to-end recovery** | truth [0.20, 0.12, 0.08, 0.05] on {P₂..P₅}: single narrow trajectory **fails** (coeff_err 0.89), composition-diverse **recovers** to **4.1e-5** — a **~2.2e4×** gap |

**The load-bearing insight:** gauge anchoring is what makes learning a free-energy
*functional* on top of FH well-posed. Without it the beyond-FH term is not merely
hard to fit — it is *unidentifiable*. And the Tikhonov penalty is science, not a
numerical hack: where data leaves a direction unconstrained, the penalty pulls it
to a **wrong** answer (the identifiability failure made visible) rather than
wandering the composition out of [0,1] into an FH-log NaN.

Files: `adjoint/neural_energy.py`, `diagnostics/structure_factor_torch.py`,
`benchmarks/phase-field/beyond_fh_identifiability.py`; gated by
`tests/test_{neural_energy,structure_factor_torch,beyond_fh_identifiability}.py`
(14 tests). Dev note: `docs/dev/2026-07-14-beyond-fh-learned-thermo.md`.

**Recorded further frontiers (optional, none started):**
1. **h(t) as a differentiable observable** — already a forward state
   (`WodoFilmStepper.h_curr`); needs the taped film path.
2. **Port the head to the production ternary film** (`physics/wodo_film.py`).
   Today the differentiable adjoint (Stack B) covers binary CH + coupled CH×AC;
   the ternary film uses FD Gauss–Newton (`m4_learn_fmix.py`). The MLP head is
   the general drop-in once the film has a taped path (or FD-GN on the anchored
   coefficients as an interim).
3. **k_e on the IFT adjoint sweep** (the one processing param still FD-only).
4. **Fit against a real differentiable S(q,t) loss** rather than real-space
   snapshots — this is the one that most directly earns the "instrument-space"
   claim in the roadmap sentence.

## 3. The brainstorm surface — open questions worth your Mac time

This is the part that isn't in the code. Roughly ordered by how much a decision
from you unblocks.

### 3a. Open rulings I've been waiting on (`docs/theory/flow_film_formulation_p2.md`)

Track B = **M6 flow/film** cannot start without these:

- **RB1.** CAC-vs-Landau-frame coexistence as designed (two film modes)?
- **RB2.** Constitutive v1 (generalized Newtonian) as the M6 entry, v2
  viscoelastic deferred until the Gu speed-sweep demands it?
- **RB3.** Gu-2017 as Track B's headline validation campaign (with a
  parameter-estimation fallback for unknown isoindigo params)?

### 3b. Papers — the highest-leverage brainstorm, and Mac-native

Three papers ratified 2026-07-09; outlines v1 live in `papers/` (**gitignored,
local-only** — tarball `paper-outlines-2026-07-09.tar.gz` was copied to the Mac;
see Sec 7).

| Paper | Status | Blocked on |
|---|---|---|
| **(1) Framework + inverse-design heroes** | skeleton + beats; 2 number-lookup FLAGS | your review of the skeleton |
| **(2) Phase-field replication + learned free energy** | skeleton + beats; 100×-quote + ledger-order FLAGS | your review — **and this is where beyond-FH (Sec 2) now belongs** |
| **(3) p2 band study** | skeleton + beats + explicit REMAINING WORK list | Samundra wraps up |

The identifiability/continuation methods story was folded into paper 1 (the
six-run bunny ladder), not a fourth paper. **Next review level = you approve the
skeletons (titles, venue, claim ledgers, section purposes); then level 3 =
per-paragraph topic sentences, then prose.** Outline protocol: claim ledger →
section beats → topic sentences; measured numbers only, `% OUTLINE FLAG` for
gaps, never a guess.

**Open question worth deciding:** paper 2's claim ledger predates beyond-FH.
The rung-2 result (gauge structure + the 2.2e4× recovery gap) is arguably the
paper's *strongest* claim and may deserve to reframe the ledger rather than be
appended to it. That's a genuine editorial call, and exactly the kind of thing
to hash out on the Mac.

### 3c. Roadmap "Forward" (`docs/dev/roadmap.md`)

- ~~Learning rung 2~~ — **done** (Sec 2). *The roadmap file's Forward list still
  shows this as open; it wants a refresh, which is a 2-minute job whenever you
  want it.*
- **CH block preconditioner for full-resolution 3-D films** (15.2 M dofs).
  G5 already marches this on one card; the open item is the **FH-3D inner-Krylov
  stall at L6 quench onset** (recorded, needs a dumped-system lab).
- **PNP electrokinetics** (production DendrIon physics); crystallization (η fields).
- **Space-time (k = 4), multi-GPU, THB refinement.**

### 3d. Structure-to-property (`docs/theory/structure_property_p3.md`)

- **SP-1** — excitonic drift-diffusion port (OPV; port of the CPU code in
  `previous_codes/`; validation targets exist as shipped test cases → port first).
- **SP-2** — conjugate ionic-electronic transport (OMIEC/OECT; a new build to
  spec) on shared infrastructure (the morphology-input contract is common).
  Linkage between the two simulators deliberately deferred.

### 3e. Recorded-but-unbuilt

- **G4b species-coupled Schur** — honest boundary found: χ₁₂ = 6 deep quench is
  structurally beyond *any* pairwise preconditioner (dropped d12 > kept d11;
  even exact pair-solves need 127–161 outer iterations).
- **M3 rung 3** (relaxed classification) — research brief at
  `docs/theory/n6_relaxed_classification_brief.md`.
- Remaining **critical-evaluation** P2/P3 items (`CriticalEvaluations/`, gitignored).
- The **H200 3-D stretch case** (M4's one pending external).

## 4. Waiting on you (externals, unchanged unless noted)

1. **Nova — submit the 3 cleared kits.** `cluster/wodo_campaign/{a100_negi3d,
   a100_wodo3d_hero,a100_s3d_pilot}.sbatch`. Sequence: on Nova
   `git fetch && git reset --hard origin/master`; fill `FIXME_PARTITION` from
   `sinfo -o "%P %G %N"`; `sbatch` each. Expected walls/memory are documented in
   each script header (A100-80 ~120–160 s/step at ~15–16 M dofs).
   *Note: `a100_negi3d.sbatch` carries `--no-strict` deliberately — all 4 rpm
   cases fail the interface-resolution preflight (~2.2 elements vs the 2.5
   validated floor) and would otherwise **abort before marching**. 4-element
   resolution would need ~227 lateral cells (~52 M dofs), over even 80 GB.*
2. **GPU CI.** The hardened `ci.yml` **is pushed** (`dd62272` — the venv-scoped
   commit landed, so the Workflows-scope blocker is resolved). Two things remain:
   - the self-hosted runner is **not running** (it was on a session-scoped
     `nohup`, PID 38284, now dead). Durable fix:
     `cd ~/actions-runner && sudo ./svc.sh install && sudo ./svc.sh start`
     (needs your password).
   - set repo variable **`ENABLE_GPU_CI = true`** (Settings → Secrets and
     variables → Actions → Variables) to activate the gpu-smoke job.
3. **🔐 Rotate the GitHub PAT.** The fine-grained token in `~/.git-credentials`
   was pasted into a chat transcript and I've flagged it repeatedly. Rotate at
   github.com/settings/tokens?type=beta. **Do this before anything else on this
   list.**
4. **RB1–RB3 rulings** (Sec 3a) and the **paper-skeleton review** (Sec 3b).

## 5. Standing rules (the memory reconstruction — read this)

The persistent memory folder does **not** travel with git (Sec 7). These are
the rulings it encodes:

**Program-level (yours):**
- Production runs are **GPU-only**; device→host transfers only for output/checkpointing.
- User-facing equation authoring stays **Integrands-style** (Taly/Dendrite/DiffPack
  heritage) — spec §3 is a hard promise, not a preference.
- **Pedagogical weak-form documentation on every brick** (state the weak form,
  map term-to-code); interactive onboarding site with a graded FEM sequence.
- **NS advection forms:** nonlinear monolithic at **s = 1/2** (skew-symmetric);
  linearized monolithic at s = 1/2 or s = 0; **s = 1 avoided** in
  Dirichlet-dominated problems.
- **Pressure projection:** THE key document is the Helmholtz-Leray VMS draft
  (`ns_projection_vms_paper` in `MyPapers/`), **Algorithm 1 — nonlinear Newton
  predictor** (this *overrides* spec S5.1's Oseen-linearized sketch), no tau_C,
  h-based tau_m, fine-scale retained in PPE RHS and velocity update.
- **NeuralSDF:** DiffSim does **not** train INRs — trained INRs are *provided
  inputs*; all dev/testing uses an analytic proxy with GENIE structure.
- Cross-check implementations against **group production code**, not just papers
  — conventions differ (github.com/baskargroup/{Dendrite, DendrIon,
  FlowBench_NSHT, Flow-Bench-Dendrite}; bitbucket.org/baskargroup/{dendrite-ibm,
  proteus}).
- **All new developments basis-function agnostic** (basis arrays only from the
  factory tabulation, generic face quadrature) **and working under BDF1 + BDF2**;
  BDF2 deterministic-only (noise + BDF2 weak order is out of scope — the LTE
  noise-fallback to BDF1 was ratified correct). Cross-matrix gate per feature
  (feature × {basis, tstep}).

**Process (how I work for you):**
- Every gate tolerance locked from **measured** values with **≥2× headroom**;
  never bit-identical on GPU. Parity tolerances treated as distributions
  (repeats, tail lock).
- **Honest-verdict ledgers**: gaps recorded with mechanism, not hidden.
- **Commit-per-green** with measured tables. **Agents never push** — the
  supervisor verifies and pushes. Authorship = Baskar (+ Claude co-author line).
- **Status updates always carry the current time.**
- Report at phase boundaries.

**Ops lessons (hard-won):**
- **Never edit a `.py` while a run that imports it is in flight** — Warp compiles
  kernel source from disk lazily; you get phantom errors.
- Tee long runs to files; **never pipe through `head`/`tail`** (SIGPIPE kills the
  producer under `set -o pipefail`).
- `pkill` with a char-class pattern (`[R]unner`) to avoid self-match.
- cuDSS mt-layer **leaks a thread per plan** (multiphase uses plain
  `DirectSolverOptions`; wodo_film 3-D still on the mt layer — bounded,
  ~1 plan/case). Explicitly `.free()` discarded cuDSS plans.
- **cuDSS diverges on the indefinite polynomial CH saddle** (no pivoting) but is
  fine on FH; `splu` is the CH default.
- **Diff constructed objects first**: when two "identical" code paths diverge,
  mechanically diff the object `__dict__` *before* trajectory forensics (this is
  how the S3b `Tm=1` missing-`**kw`-splat bug was found).

## 6. gpubox facts (relevant when you drive it remotely)

- **2× RTX 6000 Ada 48 GB, 62 GB host RAM, WSL2.** Shared box, no root;
  Mojdeh's conda at `/home/bglab/Mojdeh/ENTER` is the venv's base interpreter.
- **⚠️ Use `.venv/bin/python`, never bare `python`** — after any process restart
  bare `python` resolves to base conda (3.12.4, **no warp**). The venv has warp
  1.14.0. This has bitten every session.
- tectonic (for the LaTeX course): `/home/bglab/Mojdeh/ENTER/envs/orgtex/bin/tectonic`.
- **Intermittent GPU pathology**, two modes measured 2026-07-12: (a) clock-governor
  pinning at 210–450 MHz under load; (b) fresh processes 10–100× slow *regardless*
  of reported clocks. Both intermittent; healthy windows measured the same day
  (2685 MHz sustained; 31.2 burn-launches/s is the baseline — `gpu_burn.py cuda:0 15`
  is the probe). Host reboot is the presumed cure if persistent.
- **Recommended pattern for Mac driving:** keep GPU sessions on the gpubox inside
  **tmux**, attach from the Mac over ssh — preserves GPU access, project memory,
  and session history. Mac-side Claude for papers/theory/planning/brainstorming;
  box-side for solver/GPU work. Say the word and I'll set up the tmux config.

## 7. What exists ONLY on the gpubox (not in git)

- **`~/.claude/projects/-home-bglab-Baskar-DiffSim/memory/`** — the persistent
  memory (10 files). Sec 5 above is the reconstruction. If you want it verbatim
  on the Mac, copy the folder — but note Claude Code keys the path by project
  directory, so on the Mac it lands under a different slug (e.g.
  `-Users-<you>-…-DiffSim`); copy the *contents* into the Mac's project memory
  folder, not the folder name.
- **`papers/`** — the three outlines (gitignored). Tarball
  `paper-outlines-2026-07-09.tar.gz` is at the repo root and was copied to the
  Mac 2026-07-09; it predates beyond-FH. **Decision still open (from the papers
  README): tar-and-copy vs a private branch/repo for git-based sync.** If you
  want to brainstorm papers on the Mac, this is worth settling first.
- **`MyPapers/`** — anchor PDFs (you have the sources).
- **`previous_codes/`** — the excitonic drift-diffusion reference for SP-1.
- **`CriticalEvaluations/`** — the external critical-evaluation tracks.
- `/home/bglab/Baskar/s3_renders_L6/` — README hero frames (38 npz + pngs).
- Session scratchpads under `/tmp/claude-1000/…` — disposable.

## 8. Key ledgers to read on resume (all in-repo)

- `docs/dev/roadmap.md` — the milestone spine (Forward list needs a rung-2 refresh).
- `docs/dev/2026-07-14-beyond-fh-learned-thermo.md` — **the live frontier**.
- `docs/dev/m4-milestone-report.md` — the learned-free-energy story + §2 gauge structure.
- `docs/dev/2026-07-09-blockch-preconditioner.md` — the G1–G5 ladder + five measured design laws.
- `docs/dev/2026-07-12-m5-s3-evaporation.md` — S3, incl. the knife-edge matrix + the retraction.
- `docs/dev/2026-07-10-film-frontend-negi-validation.md` — the film front-end + Negi replication.
- `docs/dev/specs/2026-07-10-m5-orgelmorph.md` — the M5 spec.
- `docs/theory/{crystallization_formulation_p1,flow_film_formulation_p2,structure_property_p3}.md`
  — the three forward-looking memos (p2 holds RB1–RB3).
- `docs/dev/2026-07-12-continuity-handoff.md` — the predecessor to this file.
