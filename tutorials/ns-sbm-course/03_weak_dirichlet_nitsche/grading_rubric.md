# Grading rubric — 03 Weak Dirichlet / Nitsche

*The course rubric (`../ASSESSMENT.md`) specialised to this chapter.*

| Component | Weight | 03 specifics — full marks |
|---|---:|---|
| **Scientific correctness & interpretation** | 30% | Correct three Nitsche terms and the role of each; $d=0$ framed as "SBM shift off"; the confinement caveat on absolute $C_d$ understood; does **not** compare the drag to the unconfined literature value as if it were the bar. |
| **Numerical verification** | 20% | Faithfulness: weak-Nitsche projection matches the same-mesh weak-Nitsche monolithic in $C_d$ and mean $|u|$, rel-diffs reported. |
| **Reproducibility** | 15% | Resolved config + provenance; drag figure regenerated; invariants reproduce. |
| **Software & CUDA** | 10% | Weak `consistent_projection` path used (stable); `splu`; run does not blow up; clean `exit_reason`. |
| **Failure diagnosis** | 10% | `--alpha 0` run: the obstacle stops being felt ($C_d\to0$); mechanism (penalty removed) explained. |
| **Exploration & research bridge** | 10% | $\alpha$ sweep plotted (leak vs conditioning); a paragraph bridging $d=0$ Nitsche to the SBM shift. |
| **Communication** | 5% | Labeled axes/units; rel-diffs quoted; honest about confinement and coarse-mesh strong-vs-weak differences. |

## Automatic zero-credit triggers (flag, don't fail silently)

- Reporting the absolute $C_d$ as "wrong" against the unconfined literature
  value without noting confinement.
- Claiming $\alpha=0$ is "just a weaker penalty" (it removes enforcement).
- Hand-copied numbers that do not match `results.json`.
- Loosening `baseline.yaml` without a documented reason.

## Partial-credit guidance

- Three Nitsche terms listed but roles not explained → cap Scientific at
  half.
- Faithfulness rel-diffs reported but the anti-vacuity ($\alpha=0$) not run
  → half of Failure diagnosis.
