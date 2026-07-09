# phase-field — Cahn–Hilliard, Allen–Cahn, evaporating films

The M4 phase-field stack: conserved (Cahn–Hilliard) and non-conserved
(Allen–Cahn) dynamics, ternary solvent-evaporation films, and the learned-
thermodynamics demonstration. All device-bound (slot-map scatter + zero-copy
cuDSS).

| Script | What it demonstrates | Result |
|---|---|---|
| `adaptive_ch_opener.py` | interface-band re-mesh + M3 state transfer | zero mass drift across re-carve |
| `wodo_fig3.py` | 1-D evaporating film (Wodo CMS-2012 Fig 3) | first-run drying front, surface-initiated separation |
| `wodo_fig67.py` | 2-D morphology, Figs 6 + 7 | 3/3 gates (A: 2.2 → 1.5e6 percolated→multilayer) |
| `wodo_nova.py` | the full Nova campaign runner (figs 3–7) | **all 14 cases at 250×100 in ~11 min on one A100** |
| `m4_learn_fmix.py` | learn Flory–Huggins parameters from a film trajectory | χ_pf, χ_ps, χ_fs, k_e recovered to 1e-7 in 147 s |

## Run

```bash
python benchmarks/phase-field/wodo_nova.py --list           # case table (no compute)
python benchmarks/phase-field/wodo_fig67.py                 # local 2-D replication
python benchmarks/phase-field/m4_learn_fmix.py              # the learned free energy
```

Cluster campaign kit (A100 / H200 sbatch + case table):
`cluster/wodo_campaign/`. Output fields land in `benchmarks/data/wodo_nova/`
(gitignored runtime data); the committed campaign evidence and logs are in
`docs/dev/nova-mirrors/wodo_campaign/`.

## The physics ledger

Three model completions were **measured-necessary** for the Wodo regimes:
conserved (CHC) noise, composition-dependent mobility with freeze-out, and the
b/φ regularizer (which is physics, not a numerical crutch — it drives the
dilute-blend surface-first separation). The evaporation flux is an external
clock invisible to LTE control, so the stepper carries an explicit dt-cap
`dt ≤ tol·h_surf/(k_e·Δφ)`. Full story: `docs/projects/phase-field.md` and
`docs/dev/m4-milestone-report.md`.
