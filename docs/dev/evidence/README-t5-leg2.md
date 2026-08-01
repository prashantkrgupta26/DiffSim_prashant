# T5 gate-leg — surviving primary evidence (leg 2)

`t5-leg2-gate.log` is a byte-exact copy of the surviving cluster log for the
T5 truck gate-leg, **leg 2**, captured on nova (GH200 node nova24-gh-1) under
hold 11783474 on 2026-07-30.

## Provenance

- nova source: `/work/mech-ai/baskarg/DiffSim/cluster/results/t5-gate.log`
- nova preserved copy: `/work/mech-ai/baskarg/DiffSim/cluster/results/t5-leg2-preserved.log`
- md5 (all three identical): `4c30ae7ed3bbcae8867382e2e80207ea`
- 70 lines; 24 step rows (steps 0–23).

## Why this is the primary record

Leg 1's live-observed trajectory was **destroyed by a log-hygiene defect**:
`cluster/t5_gate_run.sh` teed with `tee "$LOG"` (no `-a`, fixed filename), so
relaunching leg 2 truncated the leg-1 log. Leg 2 is the **same code** and
independently replicates the behavior. It is the log that survives and is
therefore the citable primary evidence. See the incident record in
`.superpowers/sdd/progress.md` (T5 REVIEW + T5 INCIDENT RESOLUTION) and the
campaign doc `docs/dev/2026-07-30-truck-campaign.md` (§ T5).

## What the log shows (verified)

- Header: GATE LEG base=7 band=12 nsteps=400 viz_interval=20 ckpt=100
  equilibrate=False assembly=device; WARP 1.15.0; GH200 480GB.
- Step 0: `t=514.8s` (mesh build + one-time NS-kernel JIT), rss=283.9G.
- Steps 1–23: ~7–13 s/step (device CSR handoff + warm-start), iters 0–5.
- rss=283.9G held flat; nvidia-smi 65–70 GiB.
- `cd_react=-0.00000` while the startup pressure response is developing;
  onset at **step 23** (`cd_react=+0.00022`), marked `*** FIRST CONTACT ***`.
- `cd_surr` is the known startup surrogate pathology (grows to ~ -2.9e4);
  `cd_react` is the canonical observable.

Leg 2 was killed by the controller stop-order at step 23 (no completion
summary; that is expected, not a failure).
