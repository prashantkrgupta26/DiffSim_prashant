## Task A1 - MMS Convergence Test
**Date:** July 26, 2026  
**Script:** `tutorials/A_foundations/A1_mms_convergence.py`  
**Hardware:** NVIDIA RTX 2000 Ada Generation (16 GiB, CUDA Toolkit 12.9, Driver 12.8, Warp 1.15.0)  

### Observed Convergence Rates:
- **p=1 (Linear Elements):**
  - Errors: `[1.302e-03, 3.255e-04, 8.138e-05]`
  - Convergence Orders: **`2.00`**, **`2.00`** ($\mathcal{O}(h^2)$)
- **p=2 (Quadratic Elements - Patch Test for $u^* = x^2 + y^2$):**
  - Errors: `[1.985e-15, 5.943e-15, 2.943e-14]`
  - Convergence Orders: **Machine Precision Zero** ($\sim 10^{-15}$, exact discrete representation)

### Generated Figures:
- `tutorials/A_foundations/figures/convergence_p1.png`
- `tutorials/A_foundations/figures/convergence_p2.png`

### Notes / Findings:
- **Linear ($p=1$)**: Achieved exact theoretical $L_2$ convergence rate of $\mathcal{O}(h^2)$ (`2.00`, `2.00`).
- **Quadratic ($p=2$)**: Passed patch test for $u^* = x^2 + y^2$ with $L_2$ errors at machine precision zero ($\approx 10^{-15}$), since exact quadratic solution lies in the discrete $p=2$ space.
- CUDA kernel compilation and device table assembly executed cleanly with cached module load times ~1ms on NVIDIA RTX 2000 Ada Generation GPU.