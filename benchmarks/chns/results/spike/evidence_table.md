# SP-0 Task 8 — coupling-decision spike evidence

BUBBLE_RISE_RE35_WE10, level 6 (64x64), Cn_override=2h. Steppers compared to EACH OTHER (no reference-curve comparison — Task 9).

| stepper | ratio | steps (run/full) | capped | survived | wall/step mean (s) | wall/step p95 (s) | Newton mean/max | clamp total | max\|drift\| | rise-vel peak | circ final |
|---|---|---|---|---|---|---|---|---|---|---|---|
| staggered | 10 | 2400/2400 | no | YES | 0.430 | 0.447 | 4.00/4 | 0 | 5.04e-06 | 0.0307 | 1.3132 |
| monolithic | 10 | 496/2400 | yes | YES | 2.055 | 2.071 | 3.01/4 | 0 | 1.11e-15 | 0.0276 | 1.1845 |
| staggered | 100 | 2400/2400 | no | YES | 0.432 | 0.449 | 4.00/4 | 0 | 5.41e-06 | 0.0376 | 1.2805 |
| monolithic | 100 | 499/2400 | yes | YES | 2.054 | 2.071 | 3.01/4 | 0 | 8.88e-16 | 0.0335 | 1.1750 |
| staggered | 1000 | 50/2400 | yes | YES | 0.436 | 0.456 | 4.00/4 | 0 | 4.24e-06 | 0.0353 | 1.1057 |
| monolithic | 1000 | 50/2400 | yes | YES | 2.136 | 3.010 | 3.08/4 | 0 | 7.77e-16 | 0.0328 | 1.1061 |

## Centroid agreement (staggered vs monolithic)

Both downsampled centroid_y series are linearly interpolated onto the COMMON time support [0, min(t_max)] (the two runs cover different spans when one is capped or dies); max|delta| and the value of each at the common t_max are reported.

| ratio | common t_max | max\|Δcentroid_y\| | centroid @ common t_max (stag / mono) |
|---|---|---|---|
| 10 | 1.2400 | 1.228e-03 | 0.24271 / 0.24172 |
| 100 | 1.2475 | 1.431e-03 | 0.24908 / 0.24765 |
| 1000 | 0.1250 | 5.207e-05 | 0.25278 / 0.25273 |

