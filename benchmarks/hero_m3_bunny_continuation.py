"""M3 HEADLINE ATTEMPT: cell-scale bunny GENIE edit recovered from the
transient drag history via EPOCH CONTINUATION (rung 2, gated d64ddcd).
Target = fixed data at alpha*; each continuation epoch carves at its
anchor and optimizes within the trust region."""
import sys
import numpy as np

sys.path.insert(0, "tests")
sys.path.insert(0, "benchmarks")
import hero_d_bunny_drag as hd
from diffsim.adaptivity.continuation import continuation_recover
from diffsim.geometry.provided_inr import ProvidedINROracle, extract_modes
import torch

LEVEL = 5
TRUST = 0.35 / 2 ** LEVEL

# modes (as in hero_d main)
o0 = ProvidedINROracle.from_genie_json(hd.BUNNY_JSON, head=0)
g = np.linspace(0.05, 0.95, 40)
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
with torch.no_grad():
    v = o0.psi(torch.tensor(P)).numpy()
band = P[np.abs(v) < 0.04]
rng = np.random.default_rng(7)
band = band[rng.permutation(len(band))[:900]]
band += rng.uniform(-0.01, 0.01, band.shape)
V, evals, stab = extract_modes(o0, band, k=hd.K,
                               check_pts01=band[::-1].copy())
Vpack = (V, o0._Vscale.numpy())
print(f"[modes] stability {stab:.3f}", flush=True)

alpha_star = np.array([0.03, -0.005, 0.0, 0.0])

# ---- target DATA (its own carve at alpha*)
hd._EPOCH.clear()
_, _, target = hd.run_transient(alpha_star, Vpack)
print(f"[setup] target drag series ready; alpha*={alpha_star}",
      flush=True)


def epoch_fn(alpha):
    hd._EPOCH.clear()
    hd.build_epoch(Vpack, np.asarray(alpha, float))
    return dict(anchor=np.asarray(alpha, float).copy())


def resid_fn(E, alpha):
    _, _, series = hd.run_transient(np.asarray(alpha, float), Vpack)
    return series - target


def jac_fn(E, alpha, r0):
    eps = 1e-4
    J = np.zeros((len(r0), hd.K))
    for k in range(hd.K):
        a2 = np.asarray(alpha, float).copy(); a2[k] += eps
        _, _, s2 = hd.run_transient(a2, Vpack)
        J[:, k] = (s2 - target - r0) / eps
    return J


a_fin, hist = continuation_recover(epoch_fn, resid_fn, jac_fn,
                                   np.zeros(hd.K), TRUST, n_epochs=6,
                                   inner_iters=3, verbose=True)
err = np.linalg.norm(a_fin - alpha_star)
print(f"[result] recovered {np.round(a_fin, 5)} vs {alpha_star}",
      flush=True)
print(f"[result] err = {err:.4e} (bar 0.009 = 30%|a*|)", flush=True)
