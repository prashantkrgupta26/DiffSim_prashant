# Continuity handoff — restarting Claude off the gpubox

Snapshot: 2026-07-12 ~17:30 CDT. Purpose: if the WSL2 gpubox becomes
unavailable, this document lets Baskar restart a Claude Code session
on the Mac (or any machine) with full program context. It is the
supervisor-session state that is NOT derivable from the code alone.
Refreshed at push milestones; trust the newest git version.

## 0. Bootstrap on a new machine

1. Install Claude Code; `git clone github.com/BaskarGS/diffsim` and
   open a session in the repo root.
2. Point the session at THIS file first ("read
   docs/dev/2026-07-12-continuity-handoff.md and resume").
3. Re-create the persistent memory (Sec 5) — on the gpubox it lives
   at `~/.claude/projects/-home-bglab-Baskar-DiffSim/memory/`; if the
   box is reachable, copy that folder verbatim; otherwise Sec 5 is
   the reconstruction.
4. GPU-dependent work needs a CUDA box (production is GPU-only by
   program rule); Mac sessions are for papers/theory/planning and for
   driving Nova. The gpubox: 2x RTX 6000 Ada 48 GB, 62 GB host RAM,
   WSL2 (see Sec 6 hazards).

## 1. Program map (M-milestones)

- M0-M4 (bricks, NS/VMS, heat, cahn_hilliard incl. FH energy,
  ternary_ch, wodo_film + film front-end + RunLog, blockch
  preconditioner G1-G5): DONE, pushed, dev notes under docs/dev/.
- M5 "OrgElMorph" (multi-CH x multi-AC crystallization,
  src/diffsim/physics/multiphase.py): S0/S1/S2 DONE + pushed; the
  (a)-pack (T-field A1, substrate A2, anisotropy A3) + A4 (quadratic
  basis, BDF2) DONE + pushed (fd2e3e7); S3a film frame DONE + pushed
  (645d10c); S3b = the ONLY open M5 item (Sec 2); S4 (quaternary,
  configs-only by design) queued after.
- Theory memos for ratified plans: docs/theory/
  crystallization_formulation_p1.md (M5), flow_film_formulation_p2.md
  (Track A done; Track B = M6 flow/film, rulings RB1-RB3 still open
  with Baskar), structure_property_p3.md (SP-1 excitonic port + SP-2
  ionic-electronic OECT — distinct simulators, linkage deferred).
- External critical evaluation (local-only folder, gitignored):
  Tracks A/B/C queued — CI, assert->exceptions/SolveResult, README
  claims, packaging. Not started.

## 2. Exact in-flight state (as of this snapshot)

- S3 COMPLETE and pushed (8149edb, 2026-07-12 ~21:10 CDT): S3b
  gates green at the fate-robust config (r0=0.2, t_implant=12.5,
  terminal-state locks; the whole knife-edge saga resolved — a
  missing **kw splat had built the gate stepper with Tm=1; ledger
  Sec 2.4 has the retraction + surviving findings + lesson).
- IN FLIGHT: the device-assembly port agent (queue item 2 below),
  launched ~21:15 CDT on stages D1 (pattern+scatter parity) -> D2
  (cuDSS/blockch handoff, nnz-stability) -> D3 (host-vs-device
  step-time table to the 48GB 3-D limit = the Nova hero-kit basis)
  -> D4 (default flip if unambiguous). Its dev note:
  docs/dev/2026-07-13-m5-device-assembly.md. If it died: check
  git log for its commit-per-green chain and the dev note for the
  last verified stage.

## 3. Task queue (order ratified by Baskar)

1. Close S3b (above) -> push -> S3 complete.
2. Device-side assembly port for multiphase (Baskar-flagged perf
   gap): element Ae/be already on GPU; sparse finalization is host
   scipy COO->CSR + constraint triple products. Port to the
   wodo_film v1.2 slot-map path (ndof-generic kron(G, ones(ndof))
   pattern, in-kernel slot scatter into device CSR, zero-copy
   cuDSS/blockch). Gates: parity vs host path, before/after
   step-time table 2-D and 3-D. Its 3-D scale validation doubles as
   the S3-3D hero kit for Nova A100-80 (Baskar's 80 GB idea).
3. S4 quaternary demonstrations (2 active + 2 solvents; 3 active +
   1 solvent) — configs + gates only, the (M,K) factory needs no
   code changes.
4. M0-M5 retrofit audit: basis-order agnostic everywhere + BDF2
   where relevant (standing rule, Sec 5).
5. README hero images from /home/bglab/Baskar/s3_renders_L6 (38 npz
   + pngs + config.json; also still in the session scratchpad).
   Baskar wants "very visually pleasing" film-drying sequence
   (phase separation -> nucleation -> impinged grains as h shrinks);
   swappable-image slots in the landing README.
6. S3c front-end integration (FilmParams species list, RunLog
   crystallization preflight rules) — recorded handoff in the S3
   ledger Sec 4b.
7. Track B (M6: CAC film-air interface + flow + rheology) after
   Baskar's RB1-RB3 rulings; SP-1/SP-2 structure-property builds;
   critical-evaluation tracks.

## 4. Nova state

- Two campaign kits pushed and CLEARED for submission (Baskar was
  submitting ~16:00 CDT 2026-07-12): cluster/wodo_campaign/
  a100_negi3d.sbatch + a100_wodo3d_hero.sbatch. Sequence: git fetch
  && git reset --hard origin/master on Nova; fill FIXME_PARTITION
  from sinfo; sbatch both. Expected walls/memory documented in the
  script headers (A100-80 estimates ~120-160 s/step at ~15-16M dofs).
- S3-3D hero kit: NOT yet built — deliberately waiting on the
  device-assembly port (queue item 2).

## 5. Standing rules + memory reconstruction (Baskar's rulings)

- GPU-only production; Integrands API promise; pedagogical
  weak-form documentation on all bricks (state the weak form, map
  term-to-code); s=1/2 NS forms. (diffsim-program-preferences)
- ALL new developments basis-function agnostic (basis arrays only
  from the factory tabulation, generic face quadrature) AND working
  under BDF1 + BDF2; BDF2 deterministic-only (noise+BDF2 weak order
  out of scope). Retroactive to M0-M5 (queue item 4). Cross-matrix
  gate per feature (feature x {basis, tstep}).
- Every gate tolerance locked from MEASURED values with >=2x
  headroom; parity tolerances treated as distributions (repeats,
  tail lock). Honest-verdict ledgers: gaps recorded with mechanism.
- Commit-per-green with measured tables; agents never push — the
  supervisor verifies and pushes. All authorship = Baskar (+Claude
  co-author line).
- Status updates to Baskar always carry the current time.
- Ops lessons: tee long runs to files, never pipe through
  head/tail (SIGPIPE); never edit .py files while a run that
  imports them is in flight (Warp compiles kernel source from disk
  lazily); pkill with a char-class pattern to avoid self-match;
  cuDSS mt-layer leaks a thread per plan (multiphase uses plain
  DirectSolverOptions; wodo_film 3-D still on mt layer — bounded,
  ~1 plan/case); explicitly .free() discarded cuDSS plans.
- Baskar plans to move his Claude driving seat to the Mac
  (tmux-on-gpubox + ssh attach recommended when asked).

## 6. gpubox hazards (WSL2)

- Intermittent GPU pathology, TWO modes measured 2026-07-12:
  (a) clock-governor pinning at 210-450 MHz under load;
  (b) fresh processes 10-100x slow REGARDLESS of reported clocks
  (profiles: warp DtoH + cuDSS). Both intermittent; healthy windows
  measured same day (2685 MHz sustained, 31.2 burn-launches/s
  baseline — scratchpad/gpu_burn.py cuda:0 15 is the probe).
  Host reboot is the presumed cure if persistent.
- Shared box (Mojdeh's conda python at /home/bglab/Mojdeh/ENTER is
  the venv's base interpreter). No root.

## 7. What exists ONLY on the gpubox (not in git)

- ~/.claude/projects/-home-bglab-Baskar-DiffSim/memory/ — the
  persistent memory (Sec 5 reconstructs the essentials).
- /home/bglab/Baskar/s3_renders_L6/ — the README hero frames
  (durable copy; ALSO copy to Mac when convenient).
- Session scratchpad (/tmp/claude-1000/...): campaign scripts
  (s3b_seeded.py, s3b_film.py etc.), gpu_burn.py, gate logs —
  disposable but useful; the important protocols are documented in
  the S3 ledger and this file.
- Gitignored local folders: MyPapers/ (anchor PDFs — Baskar has the
  sources), previous_codes/ (excitonic_drift_diffusion reference),
  CriticalEvaluations/, papers/ (outlines; tarball copied to Mac
  2026-07-09).

## 8. Key ledgers to read on resume (in-repo)

docs/dev/2026-07-12-m5-s3-evaporation.md (S3, incl. the knife-edge
matrix), 2026-07-12-m5-apack.md, 2026-07-10-m5-s2-16390-replication.md
(incl. Sec 9 solver findings), 2026-07-09-blockch-preconditioner.md,
2026-07-10-film-frontend-negi-validation.md, docs/dev/specs/
2026-07-10-m5-orgelmorph.md, and the three docs/theory/ memos.
