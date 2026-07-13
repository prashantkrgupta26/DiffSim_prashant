# M5 D-track — multiphase FULL DEVICE-SIDE ASSEMBLY

Base commit 8149edb; branch master.  The flagged performance gap: the
multiphase `assemble()` computed element Ae/be ON DEVICE but finalized
on HOST (Ae/be `.numpy()` pull -> scipy COO -> `tocsr` -> the
`Tn.T @ K @ Tn` constraint triple products; py-spy showed marches in
`_coo_to_compressed`).  This note records the port to the wodo_film
v1.2 slot-map device path (DeviceNSAssembler), the parity gates, and
the measured host-vs-device step-time table — the enabler for the
S3-3D hero runs (Nova A100-80).

## 1. Design (what shipped)

`MultiPhaseStepper(..., assembly="host"|"device")` — constructor flag,
default "host" (existing path bit-identical; the flag makes the modes
bit-class-parity testable).  Device mode:

- PATTERN once per mesh: `DeviceNSAssembler(dm, ndof=2M+2K)` — the
  element-graph CSR superset; (M, K)-generic through ndof and
  basis-generic through nbf/nqp (p = 1 and p = 2 gated).  Auto
  node-graph pattern above 2e8 dof-pair entries (G5 rung c — the 3-D
  slabs take this path).
- PER ITERATE, all on device: x upload -> `gp_multifield` GP eval
  (new assembly/gp_field.py kernel: [n_nodes, ndof] -> vals
  [ngp, ndof] + grads [ngp, ndof, dim]; the `_pack_fields` mirror —
  this removes BOTH the per-iterate host einsums and the [ngp, ndof,
  dim+1] uploads) -> the SAME `make_mpf_newton` kernel launched in
  element batches (Ae transient capped ~2 GB, wodo G5 pattern) ->
  `scatter_batch` atomicAdd into the device CSR values/rhs.
- Per-ATTEMPT inputs (hist combine incl. BDF2, FDT/CHC noise draws,
  MMS sources, T field) stay host-computed in `_attempt_ctx` (code
  motion from `_attempt`, RNG order preserved) and upload ONCE per
  attempt; zero buffers cached for inactive inputs.
- CONSTRAINTS: every multiphase production mesh has IDENTITY
  constraints — uniform 2-D/3-D, p = 1/2, and ALL periodic meshes
  (verified: `build_mesh` bakes periodic seams into the connectivity;
  `build_constraints` returns T == I for them).  Asserted in
  `_init_device_assembly`; adapted/hanging-node meshes stay on
  assembly="host" (the constraint-aware weighted scatter exists in
  DeviceNSAssembler — NS-proven — but the multiphase face terms and
  the node-graph pattern are not wired through weights; recorded
  frontier).  The wodo device path uses the same identity contract.
- FACE TERMS (A2 wall, S3a top flux): natural-BC enrichments whose
  entries live inside boundary element blocks (pattern superset
  covers them).  O(surface) values host-computed per iterate and
  slot-scattered (`csr_slots` once per mesh) — the wodo v1.2
  documented hybrid.  Measured cost: in the 2-D L7 film bench the
  whole "other" bucket (which contains it) is < 10% of step time.
- DIRICHLET: `set_strong_rows` plan once; per-iterate b_vals
  = g - x (the host Newton-increment form) via the new
  `DeviceNSAssembler.apply_strong_rows` (extracted from assemble(),
  code motion).
- SOLVE (D2): linsolver="cudss" -> zero-copy torch-CSR (dlpack over
  vals_d), STABLE operands, plan ONCE + refactorize per iterate;
  PLAIN `DirectSolverOptions` (no mt layer — the gomp thread-leak
  finding, 16390 ledger Sec 9) and explicit `.free()` on teardown
  (the discarded-plan double-free finding).  linsolver="splu" pulls
  the fixed-pattern CSR (parity-gate solver).  blockch is NOT wired
  for multiphase in either mode (pre-existing scope: the AC blocks
  need the per-psi extension of the "pairs" meta — module docstring).
- BONUS FIX (gated): the host `T^T K T` prunes exact zeros, so its
  nnz FLAPS when noise switches coupling blocks on (the S2-era cuDSS
  plan-flapping: measured 884736 -> 1179648 nnz).  The device pattern
  is fixed by the mesh graph — nnz never changes; the cuDSS plan is
  built exactly once per mesh (regression-gated:
  `test_d2_march_parity_and_plan_stability` asserts plan count == 1
  across a noisy march).

Newton loop semantics in device mode mirror the host exactly (trust
clamp, projection, line-search ladder, convergence on the applied
increment).  One documented difference: the device buffers hold the
LAST evaluated line-search trial, so a non-monotone backtrack
re-assembles at the accepted point (same accepted iterate, one extra
assembly — host keeps the trial matrices instead).

## 2. D1 gates (tests/test_multiphase_device.py) — GREEN

Config matrix: (M, K) in {(1,0), (2,1), (3,2)}, p in {1, 2}, film
on/off, wall on/off, aniso + T-field + D(T), Dirichlet strong rows,
FDT psi + CHC phi noise, KWC vs frozen theta, BDF1 vs BDF2, periodic
{none, both-axes, lateral-only}.

- (i) assembled-matrix parity at the SAME Newton iterate (state
  synced host -> device, RNG streams aligned; canonicalized sparse
  difference).  MEASURED across all 9 configs: rel max|dA|
  2.6e-17..4.1e-16, rel max|dr| 1.6e-15..4.3e-15 (pure FP ordering:
  device GP eval + atomic scatter vs host einsum + COO dedup).
  LOCK 1e-13 (>= 23x).  The superset-vs-pruned pattern is real and
  visible: e.g. r14+fastmode_n nnz 42362 host vs 82944 device.
- (ii) one-step solution parity (fixed dt, splu both): MEASURED
  max|dx| 2.2e-16..3.6e-15 vs device-repeat atomics spread up to
  2.7e-15 (same class).  LOCK 1e-12.

## 3. D2 gates — GREEN

- March parity host(splu) vs device(cudss) on an S1-class seeded
  disc + FDT noise config (L5 periodic, r14 + fastmode_n + Vignes +
  chi_ca solubility): psi-area 0.123047 identical across host /
  device / device-repeat; psi_max |h-d| 6.7e-16; phi-mean 2.8e-17
  (conservation).  Locks 5e-3/1e-12/5e-3 (distribution-class
  headroom — FP-chaos house rule).
- cuDSS plan stability: plan count == 1 over the noisy march, nnz
  331776 fixed (the S2 flapping regression gate).
- Representative device-forced subset (stated per the D2 contract):
  the 9-config matrix-parity + 5-config one-step gates + the S1-class
  noisy march above run the multiphase physics surface (film, wall,
  aniso, T-field, Dirichlet, noise, BDF2, KWC/frozen, p2) in device
  mode end-to-end; the full suites stay on the default host mode.

## 4. D3 — measured step-time table

(benchmarks/performance/m5_device_assembly.py; one combo per process;
cudss both modes — isolates assembly + handoff; S3b-class film
production physics (M = 2, K = 1, ndof = 6, r14 + fastmode_n +
Vignes + film + line_search); RTX 6000 Ada 48 GB, WSL2 — note the
ledger's clock-governor caveat on absolute walls.)

All rows: 3 timed steps after the setup step; per-step averages;
"asm s/call" = per assembly invocation (1 + Newton iterations + line
search trials); 3-D slabs = periodic-lateral cube carved to nz
elements tall.  2-D rows: dt 1e-4, noise_psi 5e-3.  3-D rows: dt
1e-5, noise OFF (the fixed-dt bench cannot ride the production
reject ladder the noisy 3-D quench needs — measured DNFs recorded in
the bench docstring; per-call assembly/solve costs are
state-independent).  gpu = nvidia-smi global (one bench process per
GPU; n/m = not measured, the per-process query predates the WSL2
fallback); rss = host high-water.

| case | dofs | mode | step s | asm s/step | asm s/call | solve s/step | other | gpu MiB | rss GB |
|---|---|---|---|---|---|---|---|---|---|
| 2d_l6 (64^2)  |  24,960 | host   |  0.841 |  0.724 | 0.103 | 0.097 | 0.020 | n/m | 1.0 |
|               |         | device |  0.116 |  0.011 | 0.002 | 0.091 | 0.014 | n/m | 1.1 |
| 2d_l7 (128^2) |  99,072 | host   |  4.471 |  3.750 | 0.511 | 0.570 | 0.151 | n/m | 1.6 |
|               |         | device |  0.389 |  0.018 | 0.002 | 0.338 | 0.033 | n/m | 1.3 |
| 2d_l8 (256^2) | 394,752 | host   | 17.956 | 15.069 | 2.153 | 2.510 | 0.377 | n/m | 3.3 |
|               |         | device |  2.116 |  0.096 | 0.014 | 1.780 | 0.240 | n/m | 3.5 |
| 3d_l4 (16^3)* |  26,112 | host   |  3.059 |  2.508 | 0.502 | 0.445 | 0.106 |  1,197 | 1.5 |
|               |         | device |  0.790 |  0.032 | 0.006 | 0.678 | 0.081 |  1,477 | 1.3 |
| 3d_l5 (32^3)  | 202,752 | host   | 38.580 | 25.278 | 5.056 | 12.942 | 0.360 |  6,937 | 5.5 |
|               |         | device | 28.314 |  0.224 | 0.045 | 27.815 | 0.275 | 12,203 | 6.0 |
| 3d_slab64 (64x64x16) | 417,792 | host | 76.868 | 49.725 | 9.945 | 26.623 | 0.520 | 13,051 | 10.4 |
|               |         | device | 71.740 |  0.449 | 0.090 | 70.868 | 0.424 | 24,517 | 11.6 |
| 3d_slab64z32 (64x64x32) | 811,776 | device | CEILING (see below) | | | | | 48,213 | |

(* 3d_l4 measured with noise ON, dt 1e-5 — the one 3-D rung where
the fixed-dt noisy march converged; kept as measured.)

3d_slab64z32 (812k dofs): cuDSS fills the card to 48.2 GB and enters
the hybrid-memory crawl — the first factorization had not returned
after 16+ minutes (vs ~18 s/call at slab64) and the run was stopped.
SIDE FINDING (WSL2): the crawl at the dxg ceiling starved the shared
GPU paravirt layer (`dxgkio_make_resident: Ioctl failed: -12` in
dmesg) and KILLED an unrelated pytest process on the OTHER card — do
not co-schedule anything with a ceiling-class cuDSS factorization on
this box.  slab64z48 / slab128 skipped (strictly worse on 48 GB).

STEP SPEEDUPS (host -> device): 2-D 7.3x / 11.5x / 8.5x (L6/L7/L8);
3-D 3.9x (L4), 1.36x (L5), 1.07x (slab64).  ASSEMBLY per call: 52x
(L6 2-D) to 255x (L7 2-D), 84-112x in 3-D — the flagged
host-finalization gap is closed; the march is SOLVER-BOUND everywhere
on the device path.

MEASURED NUANCE (3-D direct-solver fill penalty): the device CSR is
the fixed element-graph SUPERSET (every ndof x ndof node-pair block
dense), while the host T^T K T hands cuDSS the numerically pruned
pattern.  In 2-D the superset factorization is CHEAPER per call than
the host round trip (L8: 0.30 vs 0.42 s/call).  In 3-D the fill-in
from the structurally-zero couplings (mainly the near-empty theta
rows/cols) is superlinear: at 3d_l5 the device solve is 6.95 vs 3.24
s/call, at slab64 17.7 vs 6.66 — the pattern penalty exceeds the
upload savings and the net step win compresses to 1.36x / 1.07x.  FOLLOW-UP RECORDED (Sec 6): pattern =
kron(G, blockmask) with the compile-time (M, K, mob, theta) block
sparsity mask — nnz stays attempt-invariant (plan stability
untouched) while the structural zeros drop (~24/36 density at
(M, K) = (2, 1) frozen-theta).

## 5. S3-3D hero candidate config (recorded; kit-building is the
supervisor's)

Shape: the film slab — periodic-lateral cube carved to nz elements
(benchmarks/performance/m5_device_assembly.py `build_case`), film
mode k_e = 0.1, 2 active + eliminated solvent, K = 1
(M = 2, ndof = 6), the S3b production energetics (bench
`make_stepper` verbatim: r14 + fastmode_n + Vignes softened floors +
ls_drop (1e-6, 0.97, 35), chi_ca 1.6 solubility, b_reg 1e-3,
noise_psi 5e-3 + damp, line_search, cudss, assembly="device").

- 48 GB card (this box): LARGEST STABLE = 3d_slab64 (64 x 64 x 16,
  417,792 dofs, nnz 65,028,096).  Measured walls: 71.7 s/step
  (solve-bound: cuDSS 17.7 s/call, assembly 0.09 s/call), 24.5 GB
  GPU, 11.6 GB host.  Setup (pattern + compile + plan) 124 s.
- A100-80 (Nova) candidate: 3d_slab64z32 (64 x 64 x 32, 811,776
  dofs) — 2x the dofs of the measured 24.5 GB case; the 80 GB pool
  clears the 48 GB crawl ceiling with margin.  Expected solve-bound
  walls: several minutes/step at cuDSS scaling unless the blockmask
  pattern (Sec 6) or the blockch route lands first — measure before
  committing a campaign.
- dt ladder note: the noisy 3-D quench needs the production march
  ladder (fixed dt 1e-4 diverges at the first step, 1e-5 fails later
  steps at L5 — measured in the bench DNF trail).

## 5b. D4 verdict — NO default flip (recorded with measurements)

Performance alone justifies a flip (device never slower: 7.3-11.5x
2-D steps, 1.07-3.9x 3-D, assembly 52-255x/call, parity machine-
class, plan flapping cured).  The flip fails on GATE CLASS, not
physics: the existing suites contain BIT-CLASS determinism gates
that the host path satisfies only because host assembly is fully
deterministic (one thread per element writes its own Ae block; the
COO -> CSR fold is ordered).  The device scatter is cross-element
atomicAdd — scheduling-dependent rounding by design (the house
FP-chaos position).  MEASURED (quiet box, forced device):
test_a1_dt_hook's inert-ratio lock |r - 1| < 1e-12 sees 2.43e-12
deviation and FLAKES 5/10 (physics correct: active ratio 0.3660 vs
Arrhenius 0.3657); the full-suite forced-device probe failed exactly
that gate and nothing else in the portion that ran (first 20 items,
two runs — one contended, one quiet-killed).  Same gate class:
a3_delta_zero "exact regression", the a4b bit-identical regressions,
the S0 fixed-dt 1e-13 lock.

Path to a future flip (its own pass, per-gate measured-then-locked):
(i) relock the bit-class gates as device-aware distribution locks —
or pin assembly="host" inside those gates explicitly (they gate
FORMULATION identities, which the host mode legitimately checks
bit-wise); (ii) make the default assembly=None AUTO: "device" on
identity-constraint meshes, "host" otherwise (a bare flip would turn
the identity assert into a crash for adapted-mesh users).  Until
then: production configs opt in with assembly="device" (the hero-kit
setting, Sec 5); the default stays "host".

## 6. Frontiers

- Weighted (hanging-node) constraints on the device path: fold the
  face-term slots and the node-graph pattern through the constraint
  weights (DeviceNSAssembler's weighted scatter is NS-proven; the
  multiphase wiring is the open part).  Until then adapted meshes
  use assembly="host" (asserted with a clear message).
- blockch pairs + AC blocks (pre-existing follow-up) would replace
  cuDSS above its 3-D memory ceiling; the device CSR is already
  zero-copy consumable by blockch_dev (wodo G5 route).
- TemperatureField (A1 segregated scalar solve) stays host splu —
  measured small next to the multiphase solve; migrate only if it
  shows in a 3-D T-field profile.
- march()-level host work that remains in device mode: `_attempt_ctx`
  (noise draws, history einsum via `_pack_fields(self.hist)` once per
  attempt) and the Newton x-updates.  Visible as "other" in the D3
  table; the next migration targets if they ever dominate.
