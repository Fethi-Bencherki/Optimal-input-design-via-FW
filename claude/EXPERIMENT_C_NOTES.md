# Experiment C (adaptive online LQR): what changed and why

Notes for coauthors. Prepared 2026-07-09 after debugging and re-running the
experiment behind Figure 2 of root4.tex. Everything referenced here lives in
`experiments/claude/`; the original notebook was not modified.

## Summary

The FW method's intermittent failures were caused by two implementation bugs,
not by hyperparameter sensitivity. With those fixed (plus four smaller
correctness fixes aligning the code with the paper), the method is stable
across 40 consecutive seeds with zero solver failures, and Figure 2 was
regenerated from the new run at an 8cm x 5cm aspect.

## The two failure-causing bugs

1. **Zero-gain sentinel in the LMO bisection.** The dual bisection used the
   bracket `[lam_min, lam_min + 1e6]`. Early in learning the information
   gradient can be enormous (weak prior), making `lam_min ~ 1e10`, so the
   bracket is relatively tiny, the budget target is never met, and the code
   returned zero feedback gains. That is an open-loop episode on an unstable
   plant. Fix: the paper's own construction (Section IV) with the Lemma 3(iv)
   bracket, the certificate-based stopping rule, and a minimum-energy fallback
   policy that never returns zero gains.
2. **Unguarded infinite-horizon DARE for the budget beta.** On
   (near-)unstabilizable early estimates the value iteration silently returns
   garbage, inflating beta by orders of magnitude, which then licenses
   destabilizing exploration (observed closed-loop spectral radius 4.26,
   regret ~1e90). Fix: finite-horizon certainty-equivalent LQR gains for the
   beta-defining comparison policy, which are always well posed.

Failure chain on seed 17 with the old code: bad initial estimate -> bug 1
gives an open-loop episode -> estimate worsens -> bug 2 inflates beta ->
unstable exploration -> cumulative regret ~1e95.

## Four smaller correctness fixes (aligning code with the paper)

- Interval weights `(tau_m - tau_{m-1})` -> `(tau_{m+1} - tau_m)` in the
  penalty terms of eq. (19). The old convention paired an interval with data
  that is not yet available at its start (this was also fixed in the paper).
- Stage-cost masks used inclusive `t <= tau_m`; the paper (and the notebook's
  own budget computation) use exclusive `t < tau_m`.
- FW mixture sampling was uniform over the search-point policies; the paper's
  Remark 2 requires `p_j = 2(j+1)/(M(M+1))`.
- The LMO's covariance propagation was initialized from Sigma_w rather than
  the current state second moment used everywhere else.

Note: the surviving "Experiment C.ipynb" is a T=400 two-method debug state
(the T=5000 three-method version that produced the original figure appears
lost). The protocol was restored method-symmetrically: T=5000, stochastic
rollouts with common random numbers across methods, regret measured against
the realized cost of the true-model LQR on the same noise, initial probe
scale 0.3 (chosen by the pre-registered rule "smallest probe in {1.0, 0.3,
0.1} with no catastrophic divergence for any method"; at the debug value 0.01
all three methods diverge). The sampling baseline was reconstructed from the
paper text (30 perturbations of the CE gains, best under objective (19),
perturbation scale 0.1, its best-performing setting).

## Hyperparameter changes (FW method only, baselines untouched)

| knob | old | new |
|---|---|---|
| LMO bisection bracket | `[lam_min, lam_min+1e6]`, zero-gain sentinel | Lemma 3(iv) bracket, certificate stop, min-energy fallback |
| beta comparison policy | unguarded DARE, noise 1.0/(t+t0) | finite-horizon CE-LQR, noise 0.1/(t+t0) (matches the actual naive baseline) |
| FW policy sampling | uniform | `2(j+1)/(M(M+1))` |
| planning posterior | raw Gram | Gram + 1e-3 I (regularized Gram, as in the paper) |
| admissibility test | `S_t >= -1e-9` | `S_t > 1e-9` (strict, Lemma 3) |
| FW iterations M | 400 | 400 (verified insensitive vs 100) |

## Final results (mc_final.npz, source of Figure 2)

Seeds 17-56 (40 consecutive seeds, no selection), T=5000, 20 episodes,
70% percentile bands. Final cumulative regret:

| method | mean | median | [p15, p85] |
|---|---|---|---|
| FW | 18.63 | 8.96 | [3.00, 21.98] |
| Sampling | 18.80 | 10.26 | [3.60, 20.25] |
| CE + Naive | 22.80 | 14.11 | [7.21, 25.57] |

FW has lower final regret than naive on 36/40 seeds and than sampling on
31/40 seeds, with zero LMO failures or fallbacks across all 800 episodes.
Honest caveat: the advantage over the (reconstructed) sampling baseline is
consistent but modest; the advantage over naive exploration is clear.

## Reproducing

One command: `./reproduce.sh` (uses /opt/miniconda3/envs/cyberrunner/bin/python;
requires numpy + numba). It reruns the 40-seed study into `mc_final.npz` and
regenerates the paper's Figure 2 file in place. The exact configuration and
seed list are also stored inside `mc_final.npz` (`config_json`, `seeds`).
Per-seed runs are deterministic given the seed. Supporting evidence for the
tuning choices is in the `sens_*.npz`, `samp_sweep_*.npz`, and
`mc_*probe*.npz` files, and `PROGRESS.md` is the full debugging log.
