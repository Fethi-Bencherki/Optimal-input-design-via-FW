# Optimal Input Design via Frank–Wolfe

Jupyter Notebook examples associated with the paper **“Optimal Input Design via Frank–Wolfe”** by Fethi Bencherki, Bruce Lee, Nikolai Matni, and Anders Rantzer. The public arXiv link will be added here when the paper is posted.

This repository contains the numerical experiments illustrating the paper’s Frank–Wolfe framework for optimal input design in finite-horizon linear systems, including covariance design with known dynamics, experiment design with unknown dynamics, and adaptive online LQR balancing exploration and exploitation.

## Repository Contents

- 📘 **Experiment A — Budget-Constrained Input Design**  
  [`Experiment A.ipynb`](<Experiment A.ipynb>)  
  Applies Frank–Wolfe to a known finite-horizon linear system. The experiment
  solves the covariance-design problem for several budget values and plots the
  objective across Frank–Wolfe iterations.

- 📗 **Experiment B — Adaptive System Identification**  
  [`Experiment B.ipynb`](<Experiment B.ipynb>)  
  Repeatedly estimates an unknown linear system and designs the next
  experiment using the current estimate. It compares Frank–Wolfe with
  certainty equivalence, naive exploration, and a frequency-based periodic
  input baseline under a common covariance budget.

- 📙 **Experiment C — Adaptive Online LQR**  
  [`Experiment C.ipynb`](<Experiment C.ipynb>)  
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


## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{bencherki2026optimal,
  title  = {Optimal Input Design via Frank--Wolfe},
  author = {Bencherki, Fethi and Lee, Bruce and Matni, Nikolai and Rantzer, Anders},
  journal={arXiv preprint arXiv:},
  year   = {2026},
}
```

