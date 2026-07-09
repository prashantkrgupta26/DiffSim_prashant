"""ProvidedINROracle — wraps a trained SIREN checkpoint (spec N2/N3).

DiffSim does not train INRs (program boundary N0). This wrapper consumes
checkpoints as produced by the group's GENIE pipeline (assets/sdf/):
sine layers sin(w0 (Wx+b)), skip concatenation [h, x]/sqrt(2) at given
layers (the /sqrt2 is the reference siren.ts convention and was decisive
in evaluating both provided models), linear final head.

Domain contract: the INR lives on [-1,1]^dim; the DiffSim pipeline lives
on [0,1]^dim. The wrapper maps x_inr = 2x - 1 INTERNALLY — Newton/IFT,
distances and gradients all emerge in pipeline coordinates automatically
through the composed torch graph.

GENIE edit structure: psi(x; alpha) = h(x)^T (theta_L + V alpha) + b with
V the thick-band Gram eigenmodes (extract_modes) — geometry affine in
alpha, same contract as AnalyticINRProxy.

Per-model conventions are EXPLICIT constructor state (w0 schedule, skip
set): the two provided checkpoints differ (bunny json stores w0=30
per-layer; the sphere .pt was numerically determined to run at w0=1.0 —
origin-centered isotropic zero level set only under that setting; gate 6
re-validates on every run)."""
import json

import numpy as np
import torch

from .oracle import SDFOracle


class ProvidedINROracle(SDFOracle):
    near_eikonal = False

    _Vscale = None

    def __init__(self, layers, final_w, final_b, w0s, skip_at, dim=3,
                 head=0, modes=None, window_center=0.0, window_half=1.0):
        """layers: list of (W [out,in], b [out]) numpy; w0s per layer;
        skip_at: set of layer indices receiving [h, x]/sqrt2; head: row of
        the final weight to use (GENIE multi-head); modes: optional
        [hidden, k] eigenmode matrix enabling alpha edits."""
        self.dim = dim
        self._Ws = [torch.tensor(np.asarray(W, np.float64)) for W, _ in layers]
        self._Bs = [torch.tensor(np.asarray(b, np.float64)) for _, b in layers]
        self._w0s = [float(w) for w in w0s]
        self._skip = set(skip_at)
        fw = np.asarray(final_w, np.float64)
        fb = np.asarray(final_b, np.float64)
        self._theta0 = torch.tensor(fw[head])          # [hidden]
        self._fbias = float(fb[head])
        # computational window: pipeline [0,1]^dim maps onto the INR-frame
        # sub-box window_center +- window_half (choosing the window around
        # the geometry is standard immersed practice; it also sets the
        # feature's resolved size — pick it so the geometry spans >= ~8
        # elements at the working level, else interior surrogate fragments
        # appear near the field's central critical point where the
        # projection is genuinely ill-posed [measured on the sphere
        # checkpoint at L4 full-frame]).
        self._wc = np.asarray(window_center, np.float64) * np.ones(dim)
        self._wh = float(window_half)
        self._V = (torch.tensor(np.asarray(modes, np.float64))
                   if modes is not None else None)
        k = 0 if self._V is None else self._V.shape[1]
        self.alpha = torch.zeros(k, dtype=torch.float64)

    # ---- loaders ---------------------------------------------------------
    @classmethod
    def from_state_dict(cls, path, n_layers, w0, skip_at=(4,), head=0,
                        **kw):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        layers = [(sd[f"layers.{i}.weight"].numpy(),
                   sd[f"layers.{i}.bias"].numpy()) for i in range(n_layers)]
        return cls(layers, sd["final_linear.weight"].numpy(),
                   sd["final_linear.bias"].numpy(), [w0] * n_layers,
                   skip_at, head=head, **kw)

    @classmethod
    def from_genie_json(cls, path, head=0, **kw):
        j = json.load(open(path))
        layers = [(L["weight"], L["bias"]) for L in j["layers"]]
        w0s = [float(L["w0"]) for L in j["layers"]]
        return cls(layers, j["final"]["weight"], j["final"]["bias"], w0s,
                   set(j["meta"]["skip_in"]), head=head, **kw)

    # ---- SIREN forward ----------------------------------------------------
    def features(self, x01: torch.Tensor) -> torch.Tensor:
        """Penultimate features h(x) for x in PIPELINE coords [0,1]^dim."""
        x = (2.0 * x01 - 1.0) * self._wh \
            + torch.tensor(self._wc)           # domain contract + window
        h = x
        for i, (W, B) in enumerate(zip(self._Ws, self._Bs)):
            if i in self._skip:
                h = torch.cat([h, x], dim=1) / np.sqrt(2.0)
            h = torch.sin(self._w0s[i] * (h @ W.T + B))
        return h

    @property
    def params(self):
        return [self.alpha]

    def psi(self, x01: torch.Tensor) -> torch.Tensor:
        theta = self._theta0
        if self._V is not None:
            a = self.alpha
            if getattr(self, "_Vscale", None) is not None:
                a = a / self._Vscale
            theta = theta + self._V @ a
        return self.features(x01) @ theta + self._fbias


def extract_modes(oracle: ProvidedINROracle, band_pts01: np.ndarray,
                  k: int = 16, check_pts01: np.ndarray = None):
    """Gram deformation modes (GENIE computeTopKBasis, spec N3): H over
    THICK-BAND samples (pipeline coords), eigh(G = H^T H), top-k. Returns
    (V [hidden,k], evals, stability) and installs V on the oracle.
    stability: min |cos| principal-angle diagonostic vs an independent
    sample draw when check_pts01 given (paper's reproducibility rule)."""
    with torch.no_grad():
        band_t = torch.tensor(np.asarray(band_pts01, np.float64))
        H = oracle.features(band_t)
        G = H.T @ H
        evals, evecs = torch.linalg.eigh(G)
        idx = torch.argsort(evals, descending=True)[:k]
        V = evecs[:, idx]
        # NORMALIZE to surface-displacement units: unit alpha_k moves the
        # zero level set by ~unit length (delta_x ~ delta_psi/|grad psi|).
        # Raw Gram eigenvectors have RMS feature responses O(1-10): a
        # "small-looking" alpha of 0.01 then moves the surface MULTIPLE
        # CELLS (measured: 2.5 cells -> J barriers, false basins in
        # recovery problems). In these units alpha is grid-intuitive.
        x_g = band_t.clone().requires_grad_(True)
        with torch.enable_grad():
            psi_g = oracle.psi(x_g)
            (gpsi,) = torch.autograd.grad(psi_g.sum(), x_g)
        gnorm = gpsi.norm(dim=1).clamp_min(1e-12).mean()
        # displacement scales kept SEPARATE: V stays orthonormal (the
        # stability metric and any subspace math depend on it); psi()
        # applies theta = theta0 + V @ (alpha / scale) so alpha is in
        # surface-displacement units
        scale = torch.ones(V.shape[1], dtype=torch.float64)
        for k_ in range(V.shape[1]):
            scale[k_] = (((H @ V[:, k_]) ** 2).mean().sqrt()
                         / gnorm).clamp_min(1e-30)
        stability = None
        if check_pts01 is not None:
            H2 = oracle.features(torch.tensor(
                np.asarray(check_pts01, np.float64)))
            e2, v2 = torch.linalg.eigh(H2.T @ H2)
            V2 = v2[:, torch.argsort(e2, descending=True)[:k]]
            # principal-angle cosines between the two mode subspaces
            s = torch.linalg.svdvals(V.T @ V2)
            stability = float(s.min())
    oracle._V = V
    oracle._Vscale = scale
    oracle.alpha = torch.zeros(k, dtype=torch.float64)
    return V.numpy(), evals[idx].numpy(), stability
