# D6 — expected results (self-check)

Running `python run.py` (defaults: 9×9 nodes, polynomial-basis bulk
energy $f'(c)=a_0c^3+a_1c$, $M=1$, $\kappa=0.005$, $dt=0.005$, 10
snapshots, BDF2) reproduces the recovery below. The ground truth is the
double well $f'(c)=c^3-c$, i.e. $a=[1,-1]$.

**Gradient self-check** (trajectory loss, adjoint vs finite diff, off-truth
$a=[0.7,-1.3]$):

| gradient | adjoint | finite diff | rel |
|---|---|---|---|
| $dL/da_0$ | $-3.966004\mathrm{e}{+0}$ | $-3.966004\mathrm{e}{+0}$ | $5.4\mathrm{e}{-10}$ |
| $dL/da_1$ | $-4.831506\mathrm{e}{+1}$ | $-4.831506\mathrm{e}{+1}$ | $3.2\mathrm{e}{-11}$ |

**Clean recovery** (all 10 snapshots):
`a = [+1.000000, -1.000000]`, $|a-\text{truth}|=1.4\mathrm{e}{-11}$,
loss $6.6\mathrm{e}{-21}$.

**Noisy recovery** ($\sigma=0.01$, single seed as in `run.py`): the error
falls sharply from $K=1$ snapshot to several, e.g. $|a-\text{truth}|
\approx 0.18$ at $K=1$ vs. $\lesssim 0.01$ by $K\ge3$ — more than 20×
better. (The document figure averages this over 12 noise seeds for a
smooth trend; single runs fluctuate.)

**What must be true regardless of hardware:**

- **The trajectory-loss gradient matches finite differences** (`rel <
  1e-6`; measured $\sim10^{-10}$) — the FD-verified gate.
- **Clean data recovers $c^3-c$ essentially exactly** — the snapshots were
  produced by a member of the basis, so a perfect fit exists.
- **More snapshots beat noise.** Each snapshot is an independent
  constraint on the same two coefficients; averaging over more of them
  suppresses the $\sigma$-noise. A single snapshot ($K=1$) is
  ill-conditioned; a trajectory is not.
- **A constant term in $f'$ is unrecoverable** by construction — it shifts
  $\mu$ by a constant and leaves the conserved $c$-dynamics invariant, so
  morphology data cannot see it. (This is why the basis excludes it.)

Exact noisy numbers depend on the RNG and BLAS; the qualitative law (clean
= exact, noisy improves with $K$) must hold.
