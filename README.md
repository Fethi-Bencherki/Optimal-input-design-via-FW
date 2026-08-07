# Optimal Input Design via Frank–Wolfe

Jupyter Notebook examples associated with the paper **“Optimal Input Design via Frank–Wolfe”** by Fethi Bencherki, Bruce Lee, Nikolai Matni, and Anders Rantzer. The public arXiv link will be added here when the paper is posted.

This repository contains the numerical experiments illustrating the paper’s Frank–Wolfe framework for optimal input design in finite-horizon linear systems, including covariance design with known dynamics, experiment design with unknown dynamics, and adaptive online LQR balancing exploration and exploitation.

## Repository Contents

- 📘 **Experiment A — Budget-Constrained Input Design**  
  [`Experiment A - organized.ipynb`](<Experiment A - organized.ipynb>)  
  Applies Frank–Wolfe to a known finite-horizon linear system. The experiment
  solves the covariance-design problem for several budget values and plots the
  objective across Frank–Wolfe iterations.

- 📗 **Experiment B — Adaptive System Identification**  
  [`Experiment B - organized.ipynb`](<Experiment B - organized.ipynb>)  
  Repeatedly estimates an unknown linear system and designs the next
  experiment using the current estimate. It compares Frank–Wolfe with
  certainty equivalence, naive exploration, and a frequency-based periodic
  input baseline under a common covariance budget.

- 📙 **Experiment C — Adaptive Online LQR**  
  [`Experiment C - organized.ipynb`](<Experiment C - organized.ipynb>)  
  Studies control-oriented experiment design in an online LQR problem. It
  compares Frank–Wolfe implicit dual control with certainty-equivalent naive
  exploration and sampling-based gain perturbation, reporting cumulative
  regret over multiple trials.

Each notebook is self-contained and includes short documentation blocks before
the corresponding code cells. The core NumPy/SciPy implementations are kept
separate from the optional Numba acceleration cells.

## Installation

Optional packages for acceleration and figure export are:

```bash
python -m pip install numba tikzplotlib
```

- **Numba** accelerates the specialized two-state, one-input Riccati and
  covariance calculations. If Numba is not installed, the notebooks
  automatically use the complete NumPy/SciPy reference implementations.
- **tikzplotlib** exports Matplotlib figures as TikZ/PGFPlots `.tex` files.
  It is required by the export command in Experiment A and is handled as an
  optional dependency in Experiments B and C.

## Compared Methods

| Experiment | Methods |
|---|---|
| A | Frank–Wolfe covariance design for several energy budgets |
| B | Frank–Wolfe, certainty equivalence, naive exploration, and frequency-based periodic inputs |
| C | Frank–Wolfe implicit dual control, certainty-equivalent naive exploration, and sampling-based gain perturbation baseline |

The competing methods in a Monte Carlo trial use shared process-noise
realizations where applicable, allowing paired comparisons under the same
random disturbances.

## Figure Export

Running the complete notebooks generates the following TikZ/PGFPlots files:

| Experiment | Generated file |
|---|---|
| A | `fw_objective_vs_beta.tex` |
| B | `tikz_figures/system_id_error_mc.tex` |
| B | `tikz_figures/system_id_cost_mc.tex` |
| C | `tikz_figures/experiment_c_cumulative_regret.tex` |


## Implementation Notes

- The LMO combines a generalized finite-horizon Riccati recursion with a
  scalar search for the budget multiplier.
- The multiplier search checks strict Riccati admissibility and the covariance
  budget simultaneously.
- A scale-aware upper bracket initializes the search, with bracket expansion
  retained as numerical protection.
- A minimum-energy policy provides a nonzero fallback if the requested budget
  is infeasible or a numerical search cannot certify another policy.
- The accelerated implementations preserve the public function interfaces and
  automatically fall back to the reference implementations for unsupported
  dimensions or when Numba is unavailable.
- Small floating-point differences between the accelerated and reference paths
  can occur because matrix operations are evaluated in a different order.

## Reproducibility

Random seeds and experiment parameters are specified directly in the final
cells of the notebooks. To reproduce a reported figure, run the notebook from
top to bottom without changing these values.

For a paper or archival reference, link to a tagged GitHub release or a
permanent commit rather than the repository’s moving default branch.

## Citation

If you use this code, please cite the associated paper:

```bibtex
@misc{bencherki2026optimal,
  title  = {Optimal Input Design via Frank--Wolfe},
  author = {Bencherki, Fethi and Lee, Bruce and Matni, Nikolai and Rantzer, Anders},
  year   = {2026},
  note   = {Preprint}
}
```

Once the paper is publicly available, replace the preprint note with its arXiv
identifier, DOI, or final publication information.
