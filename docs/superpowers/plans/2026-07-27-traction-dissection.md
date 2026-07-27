# Traction Dissection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the consistent Nitsche reaction term-by-term (consistency/adjoint/penalty/backflow, partition-exact), explain the 2.3× bare-σ·n excess, and resolve the reaction's 30% literature deficit via a resolution+blockage GPU campaign — ending in a data-driven canonical-force-observable decision.

**Architecture:** Per the approved spec (`docs/superpowers/specs/2026-07-27-traction-dissection-design.md`): (1) `return_terms=` flag on `sbm_vector_dirichlet_twosided` gated by a 1e-14 partition test; (2) `reaction_terms=` in `run_flow_past` splitting the LD-5 arbiter per term (1e-12 march-level partition gate) + the bare-σ·n-vs-consistency bridge; (3) new probe `tests/gpu_traction_dissect.py` running legs D1–D4 (r9/r10/r11 resolution ladder + L/32 blockage) with flushed per-leg output.

**Tech Stack:** existing SBM assembly kernels (`src/diffsim/sbm/vector.py`), the LD-5 reaction machinery in `tests/p2r1a_thin_plate_flow.py`, `gpu_leakdrag_discriminator.py` shared pieces, gpubox toolkit.

## Global Constraints

- Branch `traction-dissection` off current master; repo `/Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim`; bash starts with the repo `cd`; python `.venv/bin/python`; box via `gpubox-run.sh`/`gpubox-poll.sh` only; never edit .py mid-run.
- ALL default paths bit-for-bit: `return_terms=False`, `reaction_terms=False` (parity tests mandatory). Existing gates stay green: test_p2r0_projection_sbm (the one-sided SBM path shares vector.py!), test_p2r1a_*, test_leakdrag_probe_cpu, test_p2r1c_*.
- Partition gates are BINDING: assembly Σterms==monolithic atol 1e-14 (uniform + adaptive tiny meshes); march Σ f_x_t == f_x_total 1e-12 per step. A coarser split is allowed ONLY with the partition gate passing and honest naming.
- Campaign numbers in octree units (the unit-system rule); dt=5e-4 default, D3 fallback dt=2.5e-4/16000 recorded if needed; per-leg prints flush=True.
- `git add` explicit paths only; commit trailer `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

### Task 1: `return_terms` per-term Nitsche assembly + partition gate

**Files:**
- Read FIRST: `src/diffsim/sbm/vector.py` — `sbm_vector_dirichlet` and `sbm_vector_dirichlet_twosided` (the kernel structure determines whether a clean 4-way split is possible; the two-sided function sums two one-sided calls, so the split likely lives in the ONE-SIDED assembly / its kernel `make_sbm_dirichlet_Ae/be`)
- Modify: `src/diffsim/sbm/vector.py`
- Test: `tests/test_sbm_term_partition.py` (create)

**Interfaces:**
- Consumes: existing kernels.
- Produces: `sbm_vector_dirichlet_twosided(..., return_terms=False)`; when True returns `(A, b, terms)` where `terms` is a dict name→(A_t, b_t) whose sum equals (A, b); term names from {"consistency","adjoint","penalty","backflow"} or an honestly-named coarser partition.

- [ ] **Step 1: Write the failing partition test**

```python
# tests/test_sbm_term_partition.py
"""Partition gates for per-term Nitsche assembly (spec 2026-07-27)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))

def _tiny_two_sided(refine_to=None):
    from p2r1a_thin_plate_flow import _build_shell
    fx = _build_shell(4, 5.0/16.0, 0.5, 1.0/16.0, refine_to=refine_to)
    return fx

def _assemble(fx, return_terms, a_face=None):
    from diffsim.sbm.vector import sbm_vector_dirichlet_twosided
    noslip = lambda y: np.zeros((len(y), 2))
    kw = dict(alpha=50.0)
    # VERIFY-FIRST: read the real signature for the advecting-field kwarg
    # names (a_face_plus/a_face_minus) and the beta_backflow default; pass
    # a_face to BOTH sides when provided.
    if a_face is not None:
        kw.update(a_face_plus=a_face, a_face_minus=a_face, beta_backflow=1.0)
    return sbm_vector_dirichlet_twosided(
        fx["dm"], fx["sfp"], fx["gp"], fx["sfm"], fx["gm"],
        noslip, 0.004, 3, return_terms=return_terms, **kw)

def test_partition_exact_uniform():
    fx = _tiny_two_sided()
    A, b = _assemble(fx, False)
    A2, b2, terms = _assemble(fx, True)
    assert (A2 - A).nnz == 0 or abs(A2 - A).max() < 1e-14
    assert np.allclose(b2, b, atol=1e-14)
    As = sum(t[0] for t in terms.values())
    bs = sum(t[1] for t in terms.values())
    assert abs(As - A).max() < 1e-14, "term matrices do not partition A"
    assert np.allclose(bs, b, atol=1e-14), "term rhs do not partition b"
    assert len(terms) >= 3, f"split too coarse: {list(terms)}"

def test_partition_exact_adaptive():
    fx = _tiny_two_sided(refine_to=6)
    A, b = _assemble(fx, False)
    _, _, terms = _assemble(fx, True)
    As = sum(t[0] for t in terms.values())
    bs = sum(t[1] for t in terms.values())
    assert abs(As - A).max() < 1e-14
    assert np.allclose(bs, b, atol=1e-14)

def test_default_false_bit_identical():
    fx = _tiny_two_sided()
    A1, b1 = _assemble(fx, False)
    A2, b2 = _assemble(fx, False)
    assert (A1 - A2).nnz == 0 and np.array_equal(b1, b2)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_sbm_term_partition.py -q`
Expected: FAIL — unexpected keyword `return_terms`.

- [ ] **Step 3: Implement**

Read the one-sided assembly kernel: identify how consistency (−⟨σ(u)·ñ, w̃⟩-type), adjoint (−⟨u−g, σ(w)·ñ⟩-type), penalty (α/h⟨u−g, w̃⟩) and backflow terms enter Ae/be. Implementation options (choose per the kernel's structure, gate decides): (a) term-mask parameters on the kernel (run it N times, one term enabled each — simplest, N× assembly cost acceptable at diagnostic scale); (b) dedicated per-term mini-kernels. Thread `return_terms` through `sbm_vector_dirichlet` → `_twosided` (sum per-term across sides). Backflow term = zero-shaped block when no advecting field.

- [ ] **Step 4: Run partition + regression gates**

Run: `cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && .venv/bin/python -m pytest tests/test_sbm_term_partition.py tests/test_p2r0_projection_sbm.py tests/test_p2r1a_thin_plate_flow.py -q`
Expected: all green (p2r0 shares vector.py — MUST stay bit-for-bit).

- [ ] **Step 5: Commit**

```bash
cd /Users/baskarg/Dropbox/work/Projects/ClaudeCode/DiffSim && git add src/diffsim/sbm/vector.py tests/test_sbm_term_partition.py && git commit -m "$(cat <<'EOF'
feat(sbm): per-term Nitsche assembly (return_terms) with 1e-14 partition gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: term-split reaction in the march + σ·n bridge

**Files:**
- Read: the LD-5 reaction code in `tests/p2r1a_thin_plate_flow.py` (reaction_sets machinery, host + device paths)
- Modify: `tests/p2r1a_thin_plate_flow.py`
- Test: `tests/test_leakdrag_probe_cpu.py` (extend)

**Interfaces:**
- Consumes: Task 1's `return_terms`; LD-5's `reaction_sets`/w-vectors.
- Produces: `run_flow_past(..., reaction_terms=False)` — requires `reaction_sets`; returns `reaction_terms_hist` [nsteps, nterms] + `reaction_terms_names`; per-step partition assert Σ==total at 1e-12. The per-term blocks A_t are geometry-static like Af_c: assemble ONCE with the driver's α (values), reuse per step; backflow term follows the driver's per-step backflow handling if active (read how the driver currently handles per-step Af updates — mirror exactly; if the driver's Af is static in this march, the backflow term is static zero and say so in a comment).

- [ ] **Step 1: Failing test** — extend the CPU gate: `reaction_terms=True` on the tiny config → hist shape [nsteps, nterms], names present, per-step Σ==total to 1e-12, default-False parity (bit-for-bit cd + no new keys). Write it, see it fail (unexpected kwarg).
- [ ] **Step 2: Implement** per the Produces block. The σ·n bridge: in the probe (Task 3), not here — the driver only exposes the per-term history.
- [ ] **Step 3: Gates:** `.venv/bin/python -m pytest tests/test_leakdrag_probe_cpu.py tests/test_p2r1a_thin_plate_flow.py -q` all green.
- [ ] **Step 4: Commit** — `feat(p2r1a): per-term consistent-reaction history (reaction_terms)` + trailer; explicit paths.

---

### Task 3: dissection probe + CPU gate

**Files:**
- Create: `tests/gpu_traction_dissect.py` (imports shared pieces from `gpu_leakdrag_discriminator` — reaction-set builder etc.; does NOT modify it)
- Test: `tests/test_traction_dissect_cpu.py` (create)

**Interfaces:**
- Consumes: Tasks 1-2; `run_discriminator`-style config constants.
- Produces: `run_dissect_leg(tag, alpha=50.0, nsteps=8000, refine_to=9, wake_refine=9, plate_L_inv=16, dt=5e-4, ...) -> dict` with `cd_rxn_total, cd_rxn_terms (dict name->float), cd_surr, bridge_ratio (=cd_surr / consistency-term-cd), St, dt_used`; `__main__` runs legs D1(r9) D2(r10) D3(r11; dt fallback 2.5e-4/16000 on ConvergenceError or non-finite — record `dt_used`) D4(L/32@r10-matched: plate_L_inv=32, refine_to=10) with per-leg `flush=True` prints, final table + `TRACTDISSECT-OK`.
- Per-leg npz: `results/tractdissect_<tag>.npz` (t, cd, cl, reaction_terms_hist, names).
- CPU gate: one tiny leg (level=4, uniform, 4 steps, splu/host) — finite everything, partition holds, bridge_ratio finite.

Steps: TDD (gate RED → probe → GREEN); local regression `.venv/bin/python -m pytest tests/test_traction_dissect_cpu.py tests/test_sbm_term_partition.py -q`; commit `feat(p2r1a): traction-dissection probe (term table + resolution/blockage legs)` + trailer.

---

### Task 4: GPU campaign + recording + observable decision prep

**Files:**
- Modify: `docs/dev/thinshell-gpu-runbook.md`, `docs/dev/p2r1a-thin-plate-runbook.md`

Steps:
- [ ] Sync box (idle-check first), launch `bash scripts/remote/gpubox-run.sh ".venv/bin/python tests/gpu_traction_dissect.py" tractdissect`; poll (~3-4h; D3 is the long leg — if a leg exceeds 2h without its flush line, investigate before killing).
- [ ] Record: dated "Traction dissection" sections in both runbooks — the per-leg term table (which Nitsche term carries the force), the bridge ratio (bare σ·n vs consistency term — the 2.3× explanation), the resolution trend r9→r11 with the spec's decision rule applied verbatim (monotone-toward-3.36 ⇒ resolution; else formulation-escalation), the blockage direction vs the CV trend, `dt_used` per leg.
- [ ] The OBSERVABLE DECISION per the spec: if resolution-convergent ⇒ write the follow-up task stub (canonical-observable adoption) into the runbook follow-ups; else an escalation paragraph for Baskar. Do NOT implement the adoption in this plan.
- [ ] Commit runbooks only; ledger.

---

## Self-Review Against Spec

1. ✅ return_terms + 1e-14 partition (uniform+adaptive) — Task 1 (coarser-split escape hatch honestly gated).
2. ✅ reaction_terms march split + 1e-12 per-step partition — Task 2; σ·n bridge in the probe — Task 3.
3. ✅ Legs D1-D4 incl. D3 dt fallback + flush-per-leg (the buffering lesson) — Task 3/4.
4. ✅ Decision rule verbatim in the recording step; adoption deferred — Task 4.
5. ✅ p2r0 one-sided regression named in Task 1's gates (vector.py shared).
6. Placeholder scan: verify-first notes name their oracles (vector.py signature, driver's Af handling); no TBDs. Type check: `run_dissect_leg` outputs match Task 4's recording needs; term names flow from Task 1 through Task 3.
