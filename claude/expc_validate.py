"""Validate expc_core kernels against the notebook's own implementations.

Checks (notebook conventions, i.e. notebook_config flags):
 1. _sweep_gains  vs notebook solve_lqr_with_noise_feasibility (gains, feasibility)
 2. _cov_budget   vs notebook compute_cost_and_covariance      (budget, moments)
 3. full LMO (old mode) vs notebook numba solve_subproblem     (moments, gains)
 4. make_costs (old flags) vs notebook make_costs_fun          (stage weights)
 5. objective_eq19 (old flags) vs notebook compute_objective_tau0
"""
import numpy as np
import expc_core as core
import notebook_reference as ref

rng = np.random.default_rng(0)
A = np.array([[1.2, 1.0], [0.0, 1.0]])
B = np.array([[0.0], [1.0]])
n, m = 2, 1
nxu = 3
Sigma_w = 0.09 * np.eye(2)
W2 = np.eye(3)
T_loc = 60

# random indefinite stage weights of the LMO form: W_base + negative info part
W_base = np.eye(3)
M_seq = np.empty((T_loc, 3, 3))
X = rng.normal(size=(3, 3))
neg = -(X @ X.T) * 0.05
for t in range(T_loc):
    M_seq[t] = W_base + neg * (1.0 + 0.5 * np.sin(t / 7.0))

lam = 0.8
cfg_old = core.notebook_config()

# --- 1) Riccati sweep -------------------------------------------------------
K1, feas1 = core._sweep_gains(A, B, M_seq, W2, lam, cfg_old.s_tol)
K2, _, _, _, feas2 = ref.solve_lqr_with_noise_feasibility(
    A, B, M_seq + lam * np.broadcast_to(W2, M_seq.shape), Sigma_w)
print("1) sweep: gain err", np.max(np.abs(K1 - K2)), " feas", feas1, feas2)
assert feas1 == feas2 and np.max(np.abs(K1 - K2)) < 1e-10

# --- 2) covariance / budget --------------------------------------------------
bud1, X1 = core._cov_budget(A, B, K1, Sigma_w, Sigma_w, W2)
bud2, X2 = ref.compute_cost_and_covariance(A, B, K2, Sigma_w, T_loc, W2)
print("2) cov: budget err", abs(bud1 - bud2), " X err", np.max(np.abs(X1 - X2)))
assert abs(bud1 - bud2) < 1e-8 and np.max(np.abs(X1 - X2)) < 1e-10

# --- 3) full LMO in old mode vs notebook numba solve_subproblem --------------
tau_abs = np.array([10, 25, 40, 70], dtype=int)  # tau_k=10 ... T=70
tau_k = 10
beta = 40.0
L1, Kop1, diag = core.solve_lmo(A, B, M_seq, W2, Sigma_w, Sigma_w, beta,
                                tau_abs, tau_k, cfg_old)
L2, Kop2 = ref.solve_subproblem(A, B, M_seq, tau_abs, tau_k, 70, W2, beta,
                                Sigma_w)
print("3) LMO: moment err", np.max(np.abs(L1 - L2)),
      " gain err", np.max(np.abs(Kop1 - Kop2)), " diag", diag)
assert np.max(np.abs(L1 - L2)) < 1e-5 and np.max(np.abs(Kop1 - Kop2)) < 1e-5

# --- 4) stage-cost construction ----------------------------------------------
T = 70
tau = np.array([0, 10, 25, 40, 70])
k = 1
H_hat = np.eye(n * nxu)
D_base = core.partial_trace_H(H_hat, Sigma_w, n, nxu)
Inv_stack = np.stack([np.linalg.inv(np.eye(3) + rng.normal(size=(3, 3)) @ np.eye(3) * 0
                                    + np.diag([2.0, 3.0, 4.0]) * (i + 1))
                      for i in range(2)])
M1 = core.make_costs(T, tau, k, W_base, Inv_stack, D_base, cfg_old)
M2 = ref.make_costs_fun(T=T, tau=tau, W_base=W_base, Inv_stack=Inv_stack,
                        H_hat=H_hat, Sigma_w=Sigma_w, k=k)
print("4) make_costs: err", np.max(np.abs(M1 - M2)))
assert np.max(np.abs(M1 - M2)) < 1e-12

# --- 5) objective (19) --------------------------------------------------------
S_list, _ = core.track_covariances_noisy(A, B, A, B, Sigma_w, Sigma_w,
                                         60, 10, 1.0, np.eye(2), np.eye(1))
tau_rel = tau[k:] - tau[k]
K_info = max(len(tau_rel) - 2, 0)
v1 = core.objective_eq19(S_list, tau_rel, W_base, D_base, 3 * np.eye(3), cfg_old)
v2 = ref.compute_objective_tau0(S_list, tau_rel, K_info, np.eye(2), np.eye(1),
                                H_hat, 3 * np.eye(3), Sigma_w)
print("5) objective: ", v1, v2, " err", abs(v1 - v2))
assert abs(v1 - v2) < 1e-8

# and the noisy tracker itself vs the notebook's
S_ref = ref.track_cumulative_covariances_naive_policy(
    A, B, A, B, Sigma_w, Sigma_w, 60, 1, 10, np.eye(2), np.eye(1))
err = max(np.max(np.abs(a - b)) for a, b in zip(S_list, S_ref))
print("   tracker err", err)
assert err < 1e-8

print("ALL VALIDATION CHECKS PASSED")
