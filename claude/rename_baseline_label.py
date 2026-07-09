#!/usr/bin/env python3
"""Rename the display label of the periodic-input baseline in Experiment B.ipynb.

Changes the matplotlib legend label 'Wagenmaker-style' -> 'Frequency-based' in:
  * plot_single_seed_results (errors plot)
  * plot_objective_cost_results (costs plot)
  * the method_specs list used by the Monte-Carlo tikz-exporting plot function

It deliberately does NOT touch:
  * the dict key 'wagenmaker_style' (internal identifier),
  * comments/docstrings crediting Wagenmaker--Jamieson (method provenance).

The notebook is edited as raw text so cell outputs/metadata stay byte-identical.
"""

NOTEBOOK = "/Users/brucelee/overleaf_docs/frankwolfelqr/experiments/Experiment B.ipynb"

REPLACEMENTS = [
    (
        "plt.semilogy(episodes, errors_wagenmaker, marker='o', label='Wagenmaker-style')",
        "plt.semilogy(episodes, errors_wagenmaker, marker='o', label='Frequency-based')",
    ),
    (
        "plt.semilogy(episodes, costs_wagenmaker, marker='o', label='Wagenmaker-style')",
        "plt.semilogy(episodes, costs_wagenmaker, marker='o', label='Frequency-based')",
    ),
    (
        "('wagenmaker_style', 'Wagenmaker-style'),",
        "('wagenmaker_style', 'Frequency-based'),",
    ),
]


def main():
    with open(NOTEBOOK, encoding="utf-8") as fh:
        text = fh.read()

    for old, new in REPLACEMENTS:
        count = text.count(old)
        assert count == 1, f"expected exactly 1 occurrence, found {count}: {old!r}"
        text = text.replace(old, new)

    with open(NOTEBOOK, "w", encoding="utf-8") as fh:
        fh.write(text)

    # sanity: valid JSON afterwards
    import json

    with open(NOTEBOOK, encoding="utf-8") as fh:
        json.load(fh)
    print("OK: 3 label strings replaced; notebook is still valid JSON.")


if __name__ == "__main__":
    main()
