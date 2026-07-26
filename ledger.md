## Task A1 - MMS Convergence Test
**Date:** July 26, 2026  
**Script:** `tutorials/A_foundations/A1_mms_convergence.py`  
**Hardware:** NVIDIA RTX 2000 Ada Generation (16 GiB, CUDA Toolkit 12.9, Driver 12.8, Warp 1.15.0)  

### Observed Convergence Rates:
- **p=1 (Linear Elements):**
  - Errors: `[1.606e-03, 4.015e-04, 1.004e-04]`
  - Convergence Orders: **`2.00`**, **`2.00`** ($\mathcal{O}(h^2)$)
- **p=2 (Quadratic Elements):**
  - Errors: `[2.049e-04, 2.572e-05, 3.218e-06]`
  - Convergence Orders: **`2.99`**, **`3.00`** ($\mathcal{O}(h^3)$)

### Generated Figures:
- `tutorials/A_foundations/figures/convergence_p1.png`
- `tutorials/A_foundations/figures/convergence_p2.png`

### Notes / Findings:
- Linear ($p=1$) and Quadratic ($p=2$) elements achieved the theoretical $L_2$ convergence rates of $\mathcal{O}(h^{p+1})$.
- CUDA kernel compilation and device table assembly executed cleanly with cached module load times ~1ms on GPU.