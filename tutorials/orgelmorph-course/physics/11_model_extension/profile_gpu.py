"""P11 - GPU profile of the vitrification brick's Newton step.

Measures the per-Newton-step cost (analytic-Jacobian assembly + dense solve +
residual) of ``vitrification.py`` at increasing grid size N, on the GPU (and CPU
for reference), plus the peak device memory.  Writes the measured timings to
``<run-dir>/profile.json`` so ``gen_figures.py`` renders the profile figure and
number macros FROM saved data (never hand-copied).

    PYTHONPATH=<repo>/src python profile_gpu.py --run-dir outputs/p11

This is the contributor-workflow "profile" step (spec P11 / C8): a new mechanism
ships with a measured cost, not a hand-waved one.
"""
import argparse
import json
import math
import os
import time

import torch

import vitrification as V

HERE = os.path.dirname(os.path.abspath(__file__))


def _time_step(N, device, kappa=3e-4, M0=1.0, phi_g=0.7, w=0.03, dt=1e-3,
               nrep=20, nwarm=5):
    """Median wall time (ms) of one full Newton step (residual + analytic
    Jacobian assembly + dense solve) at grid size ``N`` on ``device``."""
    h = 1.0 / N
    x = torch.arange(N, dtype=V.DTYPE, device=device) * h
    phi = 0.5 + 0.18 * torch.sin(2 * math.pi * x) + 0.02 * torch.cos(8 * math.pi * x)
    phi_old = phi.clone()
    cuda = device.type == "cuda"

    def one():
        R = V.residual(phi, phi_old, dt, kappa, M0, phi_g, w, h)
        J = V.jacobian(phi, dt, kappa, M0, phi_g, w, h)
        _ = torch.linalg.solve(J, R)

    for _ in range(nwarm):
        one()
    if cuda:
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    ts = []
    for _ in range(nrep):
        t0 = time.perf_counter()
        one()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    peak_mb = (torch.cuda.max_memory_allocated(device) / 1e6) if cuda else 0.0
    ts.sort()
    return ts[len(ts) // 2], peak_mb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=os.path.join(HERE, "outputs", "p11"))
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = (torch.device(args.device)
           if (args.device.startswith("cuda") and torch.cuda.is_available())
           else torch.device("cpu"))
    Ns = [64, 128, 256, 512, 1024]

    gpu_ms, gpu_mem, cpu_ms = [], [], []
    for N in Ns:
        ms, mem = _time_step(N, dev)
        gpu_ms.append(ms); gpu_mem.append(mem)
        cms, _ = _time_step(N, torch.device("cpu"))
        cpu_ms.append(cms)
        print(f"N={N:5d}  {dev.type} {ms:8.3f} ms  peak {mem:8.2f} MB   "
              f"cpu {cms:8.3f} ms   speedup {cms/ms:5.2f}x")

    payload = {"device": str(dev), "N": Ns, "gpu_ms": gpu_ms,
               "gpu_peak_mb": gpu_mem, "cpu_ms": cpu_ms,
               "speedup_maxN": cpu_ms[-1] / gpu_ms[-1]}
    out = os.path.join(args.run_dir, "profile.json")
    os.makedirs(args.run_dir, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
