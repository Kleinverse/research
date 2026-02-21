# Kleinverse Research Repository

Experiment code and reproduction materials for research papers by Kleinverse AI, Inc.

---

## Papers

### LCC-ICA: Locally Centered Cyclic Kernels for Higher-Order ICA

> T. Saito, "Locally Centered Cyclic Kernels for Higher-Order Independent Component Analysis," *TechRxiv*, 2026.  
> https://doi.org/10.36227/techrxiv.XXXXXXX

FastICA truncates the cumulant generating function at order $k=4$, introducing a permanent bias bounded below by the Itakura--Saito divergence. The locally centered cyclic (LCC) kernel eliminates this bias through cyclic centering, yielding a nondegenerate V-statistic at every even order. ARE analysis and experiments on synthetic sources, grayscale image unmixing, and MUSDB18 music stem separation confirm the predicted gains and their limits.

**Code:** [lcc/](lcc/) — requires `numpy`, `scipy`

---

## Organization

Each subdirectory contains the experiment code for one paper.
