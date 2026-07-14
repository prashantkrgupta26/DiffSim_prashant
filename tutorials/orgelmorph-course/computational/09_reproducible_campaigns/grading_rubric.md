# C9 — Grading rubric

**Deliverable.** A student-authored campaign: a `manifest.yaml` sweeping a
parameter of their choice over ≥2 values × ≥2 seeds on the CH spinodal engine,
run to a `summary.json`, plus a short write-up.

| criterion | points | what full marks looks like |
|-----------|--------|----------------------------|
| Immutable resolved config | 15 | `campaign.resolved.yaml` written; re-running from it reproduces the grid; no hand-edited constants |
| Sweep × seed ensemble | 15 | deterministic run ids; Cartesian grid correct; each cell its own dir + `status.json` |
| Checkpoint / restart | 15 | restart skips completed and retries failed; idempotent (shown in the log) |
| No silent drop | 20 | an injected fault is caught and surfaced; a tolerance-violating run is marked failed; aggregate reports both with reasons |
| Aggregation + CI | 15 | mean + sd + bootstrap 95% CI per point via `diagnostics.stochastic`; seeds-used vs expected reported |
| Exploratory vs confirmatory | 10 | the two intents kept separate; the headline number comes from the confirmatory replicate |
| Provenance + write-up | 10 | `metadata.json` present; the write-up states which failure the campaign surfaced and why the mean is (in)complete |

**Automatic fail.** Any aggregation that computes a mean while dropping a
failed/NaN run without surfacing it (the one rule the chapter exists to teach).
