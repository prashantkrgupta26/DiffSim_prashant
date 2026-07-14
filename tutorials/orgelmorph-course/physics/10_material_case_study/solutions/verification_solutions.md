# P10 — verification solutions

These are the checks a correct submission must pass; the harness computes them
and `baseline.yaml` gates them in `--mode reference`.

## 1. Loading + provenance (the audit)

- `materials.resolved.json` exists and records every PDPP5T_PCBM parameter with
  its `status`, `source`, and `uncertainty`.
- All parameters resolve to `status: accelerated_tutorial`, `source.type:
  fitted` (representative), and χ_pf carries `uncertainty.type:
  order_of_magnitude` (value 0.5). No value is silently upgraded to "measured".

## 2. Reproduced trend (matched dryness)

- All `n_rungs` rungs exit `phis_stop` with φ_s monotone (`all_dried`,
  `all_monotone_drying` both true).
- Bi spans [Bi_min, Bi_max] = [1.5, 3.0].
- Phase contrast rises wet → dry on the slow AND fast rung
  (`contrast_rises_slow`, `contrast_rises_fast` true).
- Contrast at matched dryness is nearly rate-independent
  (`contrast_rate_spread_mid` a few %).
- The wavelength collapses onto Bi (`bi_collapse_wl_spread_mid` modest).
- `finer_ordering_mid` may be **either** true or false — the point is that it is
  *not* a robust discriminator; the submission must say so rather than assert a
  clean trend.

## 3. Convergence

- `mesh_contrast_rel_change` and `time_contrast_rel_change` are both small
  (a few %): the demixing-degree conclusion survives refinement.
- The box-fraction wavelength (`mesh_wl_frac_rel_change`) moves more — the
  expected signature of its noisier character.

## 4. Sensitivity

- `sens_contrast_monotone_in_chi` is true: contrast increases with χ_pf across
  the order-of-magnitude band (`sens_contrast_rel_span` sizeable), as
  Flory–Huggins predicts.

## 5. Conservation

- `balance_fullerene_rel_max < 1e-9` (machine precision — the hard gate).
- `balance_polymer_rel_max` is looser (~1e-4) and *reported*, with the
  surface-layer / large-N explanation; it must not be hidden.

## The verdict a correct submission states

Robust: demixing degree is set by dryness (rises wet→dry, rate-independent) and
by χ_pf (monotone); morphology collapses onto Bi. Not robust at tutorial scale:
the "faster = finer" wavelength ordering (needs seed ensembles + a larger box).
