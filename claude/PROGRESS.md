# Experiment C (adaptive online LQR) — STATUS: DONE

Figure regenerated in place:
`/Users/brucelee/overleaf_docs/frankwolfelqr/adaptive_online_lqr_regret_simplified_sampling_band70.tex`
(width=8cm, height=5cm => height:width = 5:8; legend FW / CE + Naive /
Sampling; mean lines + 70% percentile bands; 40 seeds). Verified by compiling
root4.tex under -jobname=expCcheck (2 passes, 0 errors) and inspecting the
rendered Figure-2 page; expCcheck.* aux files deleted. root4.tex untouched.

## Final Monte Carlo (mc_final.npz)

- Protocol: T=5000, K_episodes=20, tau_k = round(T*(k/20)^1.7), system
  A=[[1.2,1],[0,1]], B=[[0],[1]], sigma_w=0.3, Q=R=I; shared 50-step probe
  with sigma_probe=0.3 (see below); STOCHASTIC rollouts with process noise
  shared across methods (common random numbers); regret per step = realized
  cost minus realized cost of the true-model LQR K* on the SAME noise
  ("relative to the true-model LQR controller", cancels shared noise-luck).
- Seeds 17..56 (trial+17 convention, 40 contiguous seeds, no selection).
- Results (final cumulative regret): FW mean 18.63 / median 8.96,
  band15-85 [3.00, 21.98]; Sampling 18.80 / 10.26 [3.60, 20.25];
  CE + Naive 22.80 / 14.11 [7.21, 25.57]. Per-seed: FW < Naive on 36/40,
  FW < Sampling on 31/40. FW diagnostics over 800 episodes: 0 LMO failures,
  0 fallbacks, budget constraint never active (Sec. V-A large-beta regime).
- Reproduce: see reproduce.sh (runner expc_run.py, config stored inside the
  .npz as config_json).

## Bugs found in the notebook implementation (fixed in expc_core.py; flags
reproduce old behavior; validated bit-exact vs notebook before fixing)

- F1 interval weights: (tau_m - tau_{m-1}) [old non-causal] ->
  (tau_{m+1} - tau_m) per eq. (19), in the FW stage costs AND the beta
  objective.
- F2 moment off-by-one: inclusive t <= tau_m -> exclusive {m: t < tau_m},
  Sigma_m = sum_{t=tau_k}^{tau_m-1} (notebook was internally inconsistent:
  beta already used exclusive).
- F3 FW mixture sampling: uniform -> p_j = 2(j+1)/(M(M+1)) (Remark 2).
- F4 LMO covariance propagation started from Sigma_w -> from the current
  state second moment (consistent with beta + rollout).
- F5 zero-gain sentinel: stage-II bracket [lam_min, lam_min+1e6] is
  RELATIVELY tiny when lam_min ~ 1e10 (huge info gradient from weak prior);
  budget never met -> returned K=0 -> open-loop UNSTABLE episode. Fixed:
  paper-faithful folded bisection (_lmo_paper) with Lemma 3(iv) bracket
  bar-lambda = (beta+b0)/(beta-b0)*||M||/mu, admissible-at-zero shortcut,
  certificate early stop, minimum-energy fallback (never zero gains).
- F6 beta via unguarded DARE value iteration silently diverges on
  (near-)unstabilizable estimates -> beta explodes -> destabilizing
  over-exploration. Fixed: finite-horizon CE-LQR gains for the
  beta-defining comparison policy.
- NOT a bug: no spurious (1+lambda) factor in the notebook state we
  received (stage weight M + lambda*W2 with W2 = blkdiag(Q,R)).
- Protocol restorations (method-symmetric, debug residue in the surviving
  notebook, which was a T=400 2-method debug state; the T=5000 3-method
  version that made the figure is lost): (i) probe sigma_probe 0.01 -> 0.3
  (at 0.01 the initial B-estimate has error std ~5.7/entry and ALL methods
  diverge to 1e72+; pre-registered rule "smallest probe in {1.0,0.3,0.1}
  with zero catastrophic divergence for ANY method" picked 0.3; 0.1
  diverges for all methods, 1.0 trivializes learning); (ii) stochastic
  rollouts ("bands over noise realizations", caption); (iii) realized-K*
  regret comparator ("relative to the true-model LQR controller", caption).
  The Sampling baseline was reconstructed from the paper text (perturb CE
  gains, best under objective (19)): N=30 candidates + K_ce, sigma_K=0.1
  (0.05 indistinguishable, 0.2 diverges — kept its best working setting).

## Failure diagnosis (seed 17, old code, old probe; from fw_diag)

ep0: est_err 3.55 -> lam_min 1.76e10 -> F5 bracket too small -> K=0
open-loop unstable episode; ep1: beta 5.8e6 (garbage posterior, F6) -> LMO
accepts near-lambda_crit policy, true closed-loop spectral radius 4.26 ->
regret 8e90; ep2-3 pay the blown-up state covariance decay (1e95). Root
causes: F5 + F6 + uninformative probe; F1-F4 are correctness fixes.

## Hyperparameters old -> new (FW method only; baselines untouched)

- LMO bisection: two-stage, bracket lam_min+1e6, zero-gain sentinel ->
  folded single bisection, Lemma 3(iv) bracket, min-energy fallback (F5).
- beta comparison policy: DARE (unguarded) + noise 1.0/(t+t0) ->
  finite-horizon CE-LQR + noise 0.1/(t+t0) (matches the ACTUAL Naive
  baseline's exploration scale; old code inconsistently used 1.0 for beta
  while Naive plays 0.1).
- FW mixture sampling: uniform -> p_j = 2(j+1)/(M(M+1)).
- Lambda (planning posterior): raw Gram -> Gram + 1e-3*I ridge
  ("regularized Gram", paper).
- s_tol admissibility: >= -1e-9 -> > 1e-9 (strict S_t > 0, Lemma 3).
- M_fw: 400 -> 400 (unchanged; verified insensitive vs 100).
- Conventions F1, F2, F4 as above.

## Files

- expc_core.py           implementation (all methods, flags old/new)
- expc_run.py            MC runner (saves npz with config + curves + diags)
- expc_make_figure.py    pgfplots generator (templates original style)
- expc_validate.py       validation vs notebook_reference.py (all passed)
- notebook_reference.py  verbatim notebook code (validation only)
- mc_final.npz           FINAL 40-seed stochastic run (figure source)
- mc_tuned.npz, mc_stoch_probe{0.1,0.3,1.0}.npz, mc_tuned_probe{0.1,0.3}.npz,
  samp_sweep_*.npz, sens_*.npz   tuning/diagnosis evidence
- log_old_probe001.txt, log_old_probe1.txt  old-code evidence (runs killed
  on coordinator instruction after harvesting; npz never written)
- reproduce.sh           one-command reproduction of mc_final + figure
