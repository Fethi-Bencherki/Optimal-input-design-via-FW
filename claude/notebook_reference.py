"""Verbatim code cells extracted from experiments/Experiment C.ipynb.
Used only as a validation reference for expc_core.py. Do not edit.
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#                               NumPy Utilities
# =============================================================================
def block_diag_np(*arrays):
    """Construct a block-diagonal matrix using NumPy."""
    shapes = np.array([a.shape for a in arrays])
    rows = int(shapes[:, 0].sum())
    cols = int(shapes[:, 1].sum())
    out = np.zeros((rows, cols), dtype=arrays[0].dtype)
    r, c = 0, 0
    for a in arrays:
        rr, cc = a.shape
        out[r:r+rr, c:c+cc] = a
        r += rr
        c += cc
    return out


# =========================================================================================
#                  infinite horizon LQR solver
# =========================================================================================

def dare_solve(A, B, Q, R, tol=1e-10, max_iters=10_000):
    """
    Solve the discrete-time algebraic Riccati equation (DARE) via value iteration.
    Converges if (A, B) stabilizable and (A, Q^0.5) detectable.
    """
    def next_P(P):
        BtPB = B.T @ P @ B
        BtPA = B.T @ P @ A
        K = np.linalg.solve(R + BtPB, BtPA)
        Pn = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        return 0.5 * (Pn + Pn.T)

    P = Q.copy()
    P_prev = P + 10 * tol * np.eye(Q.shape[0], dtype=Q.dtype)
    i = 0
    while np.linalg.norm(P - P_prev, ord='fro') > tol and i < max_iters:
        P_next = next_P(P)
        P_prev = P
        P = P_next
        i += 1

    BtPB = B.T @ P @ B
    BtPA = B.T @ P @ A
    K = np.linalg.solve(R + BtPB, BtPA)
    return P, K

# =============================================================================
#                       finite horizon LQR solver with Feasibility flags
# =============================================================================

def solve_lqr_with_noise_feasibility(A, B, M_seq, Sigma_w, x0=None):
    """
    Finite-horizon LQR with backward Riccati sweep, process noise, and feasibility checks.
    M_seq[t] = [[Q_t, S_t],[S_t^T, R_t]]
    """
    T, n, m = M_seq.shape[0], A.shape[0], B.shape[1]

    P_next = np.zeros((n, n), dtype=A.dtype)
    feasible = True
    K_rev = []
    P_rev = []
    c_rev = []

    for M_t in M_seq[::-1]:
        Q, S, R = M_t[:n, :n], M_t[:n, n:], M_t[n:, n:]

        BtPB = B.T @ P_next @ B
        BtPA = B.T @ P_next @ A
        R_tilde = R + BtPB

        eig_min = np.min(np.linalg.eigvalsh(R_tilde))
        is_step_feasible = eig_min >= -1e-9
        feasible = bool(feasible and is_step_feasible)

        if is_step_feasible:
            K_t = np.linalg.solve(R_tilde, BtPA + S.T)
        else:
            K_t = np.zeros((m, n), dtype=A.dtype)

        AK = A - B @ K_t
        P_t = Q + K_t.T @ R @ K_t - (S @ K_t + K_t.T @ S.T) + AK.T @ P_next @ AK 
        P_t = 0.5 * (P_t + P_t.T)
        c_t = np.trace(P_next @ Sigma_w)

        K_rev.append(K_t)
        P_rev.append(P_t)
        c_rev.append(c_t)
        P_next = P_t

    P_T = np.zeros((n, n), dtype=A.dtype)
    K_list = np.asarray(K_rev[::-1], dtype=A.dtype)
    P_list = np.concatenate([np.asarray(P_rev[::-1], dtype=A.dtype), P_T[np.newaxis]], axis=0)
    c_list = np.concatenate([np.asarray(c_rev[::-1], dtype=A.dtype), np.zeros((1,), dtype=A.dtype)], axis=0)

    J_star = None if x0 is None else x0 @ P_list[0] @ x0 + c_list[0]
    return K_list, P_list, c_list, J_star, feasible




# =============================================================================
#                      Cost and Covariance Computation
# =============================================================================

def compute_cost_and_covariance(A, B, K_list, Sigma_w, T, W):
    """Compute cumulative LQR cost and covariance trajectory."""
    n = A.shape[0]

    Sigma_x = Sigma_w.copy()
    Sigma_z_seq = []
    for K in K_list:
        A_cl = A - B @ K
        Sigma_u = K @ Sigma_x @ K.T
        Sigma_xu = -Sigma_x @ K.T
        Sigma_z = np.block([[Sigma_x, Sigma_xu],
                            [Sigma_xu.T, Sigma_u]])
        Sigma_x_next = A_cl @ Sigma_x @ A_cl.T + Sigma_w
        Sigma_x = 0.5 * (Sigma_x_next + Sigma_x_next.T)
        Sigma_z_seq.append(Sigma_z)

    Sigma_z_seq = np.asarray(Sigma_z_seq, dtype=A.dtype)
    sum_cov = np.sum(Sigma_z_seq, axis=0)
    total_cost = np.trace(W @ sum_cov)
    return total_cost, Sigma_z_seq




# ============================================================================================
#                         One Frank-Wolfe LTV instant - the unspecialized version 
# ============================================================================================
def solve_subproblem(A, B, M_seq, tau, tau_k, tau_K, G, beta, Sigma_w,
                     lambda_min=0.0, lambda_max=50.0, tol=1e-3, max_iter=20):
    """
    Solves the FW linear minimization oracle by bisection over lambda.

    Convention:
        - M_seq has length T_local = T - tau_k.
        - tau is the remaining absolute grid: [tau_k, tau_{k+1}, ..., T].
        - Returned Lambda_tilde_list contains moments at tau_{k+1}, ..., T.
        - Inclusive convention for nonterminal update times:
              G_m = sum_{t=tau_k}^{tau_m} E[z_t z_t^T].
        - Terminal T is clipped to final controlled stage T-1.
    """
    # ============================================================
    # Stage I: find smallest lambda >= 0 giving a well-posed LQR
    # ============================================================
    eig_M = np.linalg.eigvalsh(M_seq)
    min_eig_M_per_t = np.min(eig_M, axis=1)
    max_neg_eig_M = np.max(-min_eig_M_per_t)

    min_eig_G = np.min(np.linalg.eigvalsh(G))
    safe_min_eig_G = np.maximum(min_eig_G, 1e-8)

    # lambda must be nonnegative
    lam_upper = max(0.0, float(max_neg_eig_M) / float(safe_min_eig_G))

    eps_inner = 1e-7
    max_iters_inner = 5000

    lam_lo = 0.0
    lam_hi = 10e3
    #lam_hi = float(lam_upper)

    # If lam_upper is zero, lambda=0 may already be feasible.
    # If not, expand the bracket until feasible.
    _, _, _, _, feasible_zero = solve_lqr_with_noise_feasibility(
        A, B, M_seq, Sigma_w
    )

    if feasible_zero:
        lambda_min = 0.0
    else:
        if lam_hi <= 0.0:
            lam_hi = 1.0

        i = 0
        while i < max_iters_inner:
            M_tilde_seq = M_seq + lam_hi * np.broadcast_to(G, M_seq.shape)
            _, _, _, _, feasible_hi = solve_lqr_with_noise_feasibility(
                A, B, M_tilde_seq, Sigma_w
            )

            if feasible_hi:
                break

            lam_hi *= 2.0
            i += 1

        # Bisection between infeasible lam_lo and feasible lam_hi
        i = 0
        while (lam_hi - lam_lo > eps_inner) and (i < max_iters_inner):
            lam = 0.5 * (lam_lo + lam_hi)
            M_tilde_seq = M_seq + lam * np.broadcast_to(G, M_seq.shape)

            _, _, _, _, feasible = solve_lqr_with_noise_feasibility(
                A, B, M_tilde_seq, Sigma_w
            )

            if feasible:
                lam_hi = lam
            else:
                lam_lo = lam

            i += 1

        lambda_min = lam_hi

    # ============================================================
    # Stage II: bisection on the budget
    # ============================================================
    lambda_max = lambda_min + 10e5
    max_iters_outer = 8000

    T_local = M_seq.shape[0]
    n, m = A.shape[0], B.shape[1]
    nxu = n + m

    K_opt = np.zeros((T_local, m, n), dtype=A.dtype)
    X_seq_opt = np.zeros((T_local, nxu, nxu), dtype=A.dtype)

    lmin = float(lambda_min)
    lmax = float(lambda_max)
    it = 0

    while (lmax - lmin > eps_inner) and (it < max_iters_outer):
        lmid = 0.5 * (lmin + lmax)

        M_tilde_seq = M_seq + lmid * np.broadcast_to(G, M_seq.shape)

        K_seq, _, _, _, _ = solve_lqr_with_noise_feasibility(
            A, B, M_tilde_seq, Sigma_w
        )

        cost, X_seq = compute_cost_and_covariance(
            A, B, K_seq, Sigma_w, T_local, G
        )

        if cost > beta:
            lmin = lmid
        else:
            lmax = lmid
            K_opt = K_seq
            X_seq_opt = X_seq

        it += 1

    # ============================================================
    # Build cumulative moments at tau_{k+1}, ..., T
    # ============================================================
    X_prefix = np.cumsum(X_seq_opt, axis=0)

    tau = np.asarray(tau)
    j_offsets = (tau - tau_k).astype(int)

    # Drop tau_k itself. Keep tau_{k+1}, ..., T.
    j_gather = j_offsets[1:]

    # Terminal endpoint T maps to the final controlled stage T-1.
    j_gather = np.clip(j_gather, 0, T_local - 1)

    Lambda_tilde_list = X_prefix[j_gather, :, :]

    return Lambda_tilde_list, K_opt

# ==========================================================================
#                           Main Frank-Wolfe function 
# ==========================================================================

def frank_wolfe(A, B, Lambda, beta, Sigma_w, M_steps,
                make_costs_fn, solve_subproblem_fn,
                W_base, H_hat, tau, k, T):
    """
    Frank-Wolfe with list-based storage.

    X_current stores moments at:
        tau[k+1], tau[k+2], ..., tau[-1]

    The last element is the terminal moment at T.
    The information penalty uses X_current[:-1].
    """
    K_terminal = tau.shape[0] - 1
    nxu = W_base.shape[0]
    L = K_terminal - k

    X_current = [
        np.zeros((nxu, nxu), dtype=W_base.dtype)
        for _ in range(L)
    ]

    # No regularization
    Lambda_k = 0.5 * (Lambda + Lambda.T)

    policies = []

    for j in range(M_steps):
        if L > 1:
            X_arr = np.stack(X_current[:-1], axis=0)
            X_arr = 0.5 * (X_arr + np.swapaxes(X_arr, -1, -2))

            A_stack = X_arr + Lambda_k[None, :, :]

            Inv_stack = np.asarray([
                np.linalg.inv(A_m) for A_m in A_stack
            ])
            Inv_stack = 0.5 * (Inv_stack + np.swapaxes(Inv_stack, -1, -2))
        else:
            Inv_stack = np.zeros((0, nxu, nxu), dtype=W_base.dtype)

        M_seq = make_costs_fn(
            T=T,
            tau=tau,
            W_base=W_base,
            Inv_stack=Inv_stack,
            H_hat=H_hat,
            Sigma_w=Sigma_w,
            k=k
        )

        tau_k_int = int(tau[k])
        tau_K_int = int(tau[-1])
        tau_remaining_abs = tau[k:]

        S_k_val, K_seq = solve_subproblem_fn(
            A, B,
            M_seq,
            tau_remaining_abs,
            tau_k_int,
            tau_K_int,
            W_base,
            beta,
            Sigma_w
        )

        gamma = 2.0 / (j + 2.0)

        X_next = []
        for X, S in zip(X_current, S_k_val):
            Y = (1.0 - gamma) * X + gamma * S
            X_next.append(0.5 * (Y + Y.T))
        X_current = X_next

        policies.append(K_seq)

    return X_current, policies

# =====================================================================================
#                           This functions constructs LTV cost matrices for FW instant
# =====================================================================================


def make_costs_fun(T, tau, W_base, Inv_stack, H_hat, Sigma_w, k):
    """
    Construct the FW-linearized stage-cost matrices M_t for t in [tau[k], T).

    Implements

        M_t = W_base + sum_{m=k+1}^K 1{t <= tau_m} W_m^(i),

    where

        W_m^(i) =
            - (Lambda_k + tilde_Lambda_m^(i))^{-1}
              D_m
              (Lambda_k + tilde_Lambda_m^(i))^{-1},

        D_m = (tau_m - tau_{m-1}) * Tr_{Sigma_w}(H_hat).

    Convention:
        tau = [tau_0, ..., tau_K, tau_{K+1}=T]
        information terms use m = k+1, ..., K
        terminal time tau_{K+1}=T is excluded from information penalty
        controlled stages are t = tau[k], ..., T-1
    """
    tau = np.asarray(tau)
    Sigma_w = np.atleast_2d(Sigma_w)

    n = Sigma_w.shape[0]
    nxu = W_base.shape[0]

    # tau = [tau_0, ..., tau_K, tau_{K+1}=T]
    K_info_terminal = tau.shape[0] - 2
    L = K_info_terminal - k
    tau_k = int(tau[k])

    # Compute D_base = Tr_{Sigma_w}(H_hat), shape (nxu, nxu)
    H4 = H_hat.reshape(n, nxu, n, nxu)
    D_base = np.tensordot(Sigma_w, H4, axes=([0, 1], [0, 2]))
    D_base = 0.5 * (D_base + D_base.T)

    # Information update indices m = k+1, ..., K
    m_idx = np.arange(k + 1, K_info_terminal + 1, dtype=np.int32)

    # If no information terms remain, just return W_base over the remaining horizon
    times = np.arange(tau_k, T)
    Nt = times.shape[0]

    Wb = 0.5 * (W_base + W_base.T)
    base = np.broadcast_to(Wb, (Nt, nxu, nxu))

    if L <= 0:
        return base.copy()

    # D_m = (tau_m - tau_{m-1}) * D_base
    deltas = (tau[m_idx] - tau[m_idx - 1]).astype(W_base.dtype)

    # W_m^(i) = - Inv_m D_m Inv_m
    W_info_stack = []
    for Inv_m, delta in zip(Inv_stack, deltas):
        X = Inv_m @ (delta * D_base) @ Inv_m
        W_info_stack.append(-0.5 * (X + X.T))

    W_info_stack = np.asarray(W_info_stack, dtype=W_base.dtype)

    # tau_m values for m = k+1, ..., K
    tau_m_arr = tau[m_idx]

    # Inclusive convention: contribution appears when t <= tau_m
    mask = (times[:, None] <= tau_m_arr[None, :]).astype(W_base.dtype)

    sum_part = (
        mask[:, :, None, None] * W_info_stack[None, :, :, :]
    ).sum(axis=1)

    M_seq = base + sum_part
    M_seq = 0.5 * (M_seq + np.swapaxes(M_seq, -1, -2))

    return M_seq

# =============================================================================
#                           Rollout to update the prior for FW
# =============================================================================

def rollout_combined(
    K_seq,              # shape (T, m, n) or list of (m,n)
    A, B,               # (n,n), (n,m)
    Sigma_w,            # (n,n)  process-noise covariance
    Sigma_x0,           # (n,n)  state covariance at start of segment
    tau_k, tau_k_plus_1,
    Q_lqr=None, R_lqr=None
):
    """
    Realization-independent covariance rollout for:
        x_{t+1} = (A - B K_t) x_t + w_t,   E[w_t]=0,  Cov(w_t)=Sigma_w

    What it computes (no means, no trajectories):
      - Stage state covariance:      Zx_t         (n,n)
      - Stage *joint* second moment: Zz_t = E[z_t z_t^T], z_t=[x_t; u_t] with u_t=-K_t x_t (n+m, n+m)
      - Per-step stage cost:         c_t = tr(blkdiag(Q,R) @ Zz_t)  (if Q,R provided)
      - Episode sums:
          cov_sum   = Σ_t Zz_t
          cross_cov = Σ_t E[x_{t+1} z_t^T] = Σ_t F_t Zx_t [I, -K_t]^T, with F_t=A-BK_t

    Returns:
        cov_sum       : (n+m, n+m)
        cross_cov_sum : (n,   n+m)
        x_cov_final   : (n, n)     # state covariance at end of segment (for chaining)
        c_stage       : (T,)       # per-step costs (zeros if Q,R not given)
        Zz_stack      : (T, n+m, n+m)  # per-step joint second moments
    """
    T = int(tau_k_plus_1 - tau_k)
    n = A.shape[0]
    m = B.shape[1]
    dtype = Sigma_x0.dtype

    # Slice the gain segment
    if isinstance(K_seq, (list, tuple)):
        K_segment = K_seq[:T]
    else:
        K_segment = K_seq[:T, ...]  # (T, m, n)

    # Cost weight (optional)
    use_costs = (Q_lqr is not None) and (R_lqr is not None)
    if use_costs:
        W_base = np.block([
            [np.asarray(Q_lqr, dtype=dtype), np.zeros((n, m), dtype=dtype)],
            [np.zeros((m, n), dtype=dtype), np.asarray(R_lqr, dtype=dtype)],
        ])
    else:
        W_base = np.zeros((n + m, n + m), dtype=dtype)

    I_n = np.eye(n, dtype=dtype)

    cov_sum   = np.zeros((n + m, n + m), dtype=dtype)
    cross_sum = np.zeros((n,     n + m), dtype=dtype)
    Zx_t = np.asarray(Sigma_x0, dtype=dtype).copy()
    Zz_stack = []
    c_stage = []

    for K_t in K_segment:
        # ensure K_t has shape (m,n)
        K_t = np.asarray(K_t, dtype=dtype).reshape(m, n)

        # u = -K x
        U_cov  = K_t @ Zx_t @ K_t.T           # (m,m)
        XU_cov = -Zx_t @ K_t.T                # (n,m)
        Zz_t   = np.block([[Zx_t,    XU_cov],
                           [XU_cov.T, U_cov ]])  # (n+m, n+m)

        # Accumulate joint moment and cross term
        cov_sum   = cov_sum + Zz_t
        F_t       = A - B @ K_t               # (n,n)
        H_t       = np.vstack([I_n, -K_t])   # (n+m, n)
        cross_sum = cross_sum + F_t @ Zx_t @ H_t.T   # (n, n+m)

        # Cost for this stage
        c_t = np.einsum('ab,ba->', W_base, Zz_t) if use_costs else np.array(0., dtype=dtype)

        # Propagate state covariance
        Zx_next = F_t @ Zx_t @ F_t.T + np.asarray(Sigma_w, dtype=dtype)
        Zx_t = Zx_next
        Zz_stack.append(Zz_t)
        c_stage.append(c_t)

    Zx_T = Zx_t
    Zz_stack = np.asarray(Zz_stack, dtype=dtype)
    c_stage = np.asarray(c_stage, dtype=dtype)

    return cov_sum, cross_sum, Zx_T, c_stage, Zz_stack
# =============================================================================
#                          Beta Computation Helpers
# =============================================================================

def track_cumulative_covariances_naive_policy(
    A, B, A_rollout, B_rollout, Sigma_w, Sigma_x0, T, m, t0, Q=None, R=None, K=None
):
    """
    Uses an LQR gain synthesized on (A, B) if K is not provided.
    Covariance propagation uses the rollout dynamics (A_rollout, B_rollout).

    Returns:
        S_list: list of cumulative covariances S_t in R^{(n+m) x (n+m)}
        for t=1,...,T, where

            S_t = sum_{s=1}^t E[z_s z_s^T],
            z_s = [x_s; u_s].

    For beta computation, pass (A_hat, B_hat) for both synthesis and rollout.
    For true naive evaluation, pass (A_hat, B_hat) for synthesis and
    (A_true, B_true) for rollout.
    """
    n = A_rollout.shape[0]

    if K is None:
        assert Q is not None and R is not None, "Provide (Q, R) to synthesize K when K is None."
        P, K = dare_solve(A, B, Q, R)

    Z = Sigma_x0
    S_list = []

    for t in range(1, T + 1):
        Sigma_eta = (1/ (t + t0)) * np.eye(m, dtype=Sigma_w.dtype)

        U_cov = K @ Z @ K.T + Sigma_eta
        XU_cov = -Z @ K.T

        Z_joint = np.block([
            [Z,        XU_cov],
            [XU_cov.T, U_cov]
        ])

        S = Z_joint if t == 1 else S_list[-1] + Z_joint
        S_list.append(S)

        Acl_rollout = A_rollout - B_rollout @ K
        Z = Acl_rollout @ Z @ Acl_rollout.T + Sigma_w + B_rollout @ Sigma_eta @ B_rollout.T

    return S_list

def compute_objective_tau0(S_list, tau, K, Q, R, H_hat, Lambda0, Sigma_w):
    W = block_diag_np(Q, R)

    # Full remaining-horizon LQR cost over [tau_k, T]
    S_T = S_list[-1]
    LQR_cost = np.trace(W @ S_T)

    penalty = 0.0

    # tau is relative to the current tau_k, e.g.
    # [0, 10, 20, 30, 40, 50]
    #
    # K is the number of nonterminal update times.
    # Example: K=4 corresponds to update times 10,20,30,40.
    for m in range(1, K + 1):
        tau_m = int(tau[m])
        tau_m_prev = int(tau[m - 1])

        # Data accumulated up to update time tau_m.
        # This convention uses data from local stages 0,...,tau_m-1.
        S_tau_m = S_list[tau_m - 1]

        inside = Lambda0 + S_tau_m
        inside = 0.5 * (inside + inside.T)

        kron_term = np.kron(np.linalg.inv(inside), Sigma_w)

        penalty += (tau_m - tau_m_prev) * np.trace(H_hat @ kron_term)

    return LQR_cost + penalty
# ===========================================================================
#                     H(\hat theta)  ---  Hessian computation 
# ============================================================================

# ----------------------------
# Discrete-time Lyapunov solver
# ----------------------------
def solve_discrete_lyapunov_np(A, Q, max_iter=1000, tol=1e-8):
    P = Q
    for _ in range(max_iter):
        P_next = A.T @ P @ A + Q
        if np.linalg.norm(P_next - P) < tol:
            break
        P = P_next
    return P

# -------------------------------------
# Cost function with fixed A_hat, B_hat
# ------------------------------------
def lqr_cost_fixed_theta(K, A_hat, B_hat, Q, R, Sigma_w, T):
    A_cl = A_hat - B_hat @ K      # K can be (m,n)
    Q_bar = Q + K.T @ R @ K
    P = solve_discrete_lyapunov_np(A_cl.T, Q_bar)
    
    # Handle scalar T
    if np.ndim(T) == 0:
        return T * np.trace(P @ Sigma_w)
    else:
        return np.trace(T @ P @ Sigma_w)

# --------------------------------------
# Riccati iteration to compute optimal K
# --------------------------------------
def riccati_iteration(A, B, Q, R, max_iter=1000, tol=1e-8):
    P = Q.copy()
    for _ in range(max_iter):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.linalg.norm(P_next - P) < tol:
            P = P_next
            break
        P = P_next
    # Final optimal gain
    K_opt = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K_opt


def _finite_difference_jacobian(fun, x, eps=1e-5):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(fun(x), dtype=float).reshape(-1)
    J = np.zeros((f0.size, x.size), dtype=float)
    for i in range(x.size):
        dx = np.zeros_like(x)
        dx[i] = eps
        fp = np.asarray(fun(x + dx), dtype=float).reshape(-1)
        fm = np.asarray(fun(x - dx), dtype=float).reshape(-1)
        J[:, i] = (fp - fm) / (2.0 * eps)
    return J


def _finite_difference_hessian(fun, x, eps=1e-4):
    x = np.asarray(x, dtype=float)
    d = x.size
    H = np.zeros((d, d), dtype=float)
    f0 = float(fun(x))
    for i in range(d):
        ei = np.zeros_like(x)
        ei[i] = eps
        fpi = float(fun(x + ei))
        fmi = float(fun(x - ei))
        H[i, i] = (fpi - 2.0 * f0 + fmi) / (eps**2)
        for j in range(i + 1, d):
            ej = np.zeros_like(x)
            ej[j] = eps
            fpp = float(fun(x + ei + ej))
            fpm = float(fun(x + ei - ej))
            fmp = float(fun(x - ei + ej))
            fmm = float(fun(x - ei - ej))
            Hij = (fpp - fpm - fmp + fmm) / (4.0 * eps**2)
            H[i, j] = Hij
            H[j, i] = Hij
    return H

# ----------------------------
# Hessian of CE LQR cost
# ----------------------------
def hessian_CE(A_hat, B_hat, Q, R, Sigma_w, T):
    n, m = B_hat.shape[0], B_hat.shape[1]

    # Step 1: compute K_hat
    _, K_hat = riccati_iteration(A_hat, B_hat, Q, R)

    # Step 2: Hessian of J w.r.t K (A_hat, B_hat fixed)
    # K_hat kept as (m,n) shape
    K_hat_vec = K_hat.flatten()
    H_K = _finite_difference_hessian(
        lambda K_vec: lqr_cost_fixed_theta(K_vec.reshape(m, n), A_hat, B_hat, Q, R, Sigma_w, T),
        K_hat_vec,
    )

    # Step 3: Jacobian of K w.r.t theta = vec(A,B)
    def K_fn(theta_vec):
        A = theta_vec[:n*n].reshape(n, n)
        B = theta_vec[n*n:].reshape(n, m)
        _, K_theta = riccati_iteration(A, B, Q, R)
        return K_theta.flatten()

    theta_hat_vec = np.concatenate([A_hat.flatten(), B_hat.flatten()])
    J_K_theta = _finite_difference_jacobian(K_fn, theta_hat_vec)  # shape (m*n, d)

    # Step 4: Chain-rule Hessian
    H_CE = J_K_theta.T @ H_K @ J_K_theta
    return H_CE


def _naive_policy_covariances_with_final(
    A, B, A_rollout, B_rollout, Sigma_w, Sigma_x0, T, m, t0, Q=None, R=None, K=None
):
    """
    Same covariance recursion as track_cumulative_covariances_naive_policy,
    but also returns the final state covariance so the Naive baseline can
    maintain its own independent state-covariance path across episodes.
    """
    n = A_rollout.shape[0]

    if K is None:
        assert Q is not None and R is not None, "Provide (Q, R) to synthesize K when K is None."
        P, K = dare_solve(A, B, Q, R)

    Z = np.asarray(Sigma_x0, dtype=Sigma_w.dtype).copy()
    S_list = []

    for t in range(1, T + 1):
        Sigma_eta = (0.1 / (t + t0)) * np.eye(m, dtype=Sigma_w.dtype)

        U_cov = K @ Z @ K.T + Sigma_eta
        XU_cov = -Z @ K.T

        Z_joint = np.block([
            [Z,        XU_cov],
            [XU_cov.T, U_cov]
        ])

        S = Z_joint if t == 1 else S_list[-1] + Z_joint
        S_list.append(S)

        Acl_rollout = A_rollout - B_rollout @ K
        Z = Acl_rollout @ Z @ Acl_rollout.T + Sigma_w + B_rollout @ Sigma_eta @ B_rollout.T
        Z = 0.5 * (Z + Z.T)

    return S_list, Z


# ============================================================================
# Scalar-input specialized acceleration for solve_subproblem
# ============================================================================
# This cell preserves the same bisection logic, tolerances, lambda range,
# Riccati sign, and moment indexing as the NumPy solve_subproblem above. It only
# specializes the hot n=2, m=1 linear algebra to avoid thousands of tiny
# np.linalg calls. For other dimensions, it falls back to the NumPy version.

try:
    import numba as nb
    _SCALAR_NUMBA_AVAILABLE = True
except Exception:
    nb = None
    _SCALAR_NUMBA_AVAILABLE = False

_solve_subproblem_numpy_reference = solve_subproblem

if _SCALAR_NUMBA_AVAILABLE:

    @nb.njit
    def _lqr_feasible_scalar_n2m1(A, B, M_seq, G, lam):
        T = M_seq.shape[0]
        a00 = A[0, 0]; a01 = A[0, 1]; a10 = A[1, 0]; a11 = A[1, 1]
        b0 = B[0, 0]; b1 = B[1, 0]

        p00 = 0.0
        p01 = 0.0
        p11 = 0.0
        feasible = True

        for tt in range(T - 1, -1, -1):
            q00 = M_seq[tt, 0, 0] + lam * G[0, 0]
            q01 = M_seq[tt, 0, 1] + lam * G[0, 1]
            q10 = M_seq[tt, 1, 0] + lam * G[1, 0]
            q11 = M_seq[tt, 1, 1] + lam * G[1, 1]
            s0  = M_seq[tt, 0, 2] + lam * G[0, 2]
            s1  = M_seq[tt, 1, 2] + lam * G[1, 2]
            r   = M_seq[tt, 2, 2] + lam * G[2, 2]

            pb0 = p00 * b0 + p01 * b1
            pb1 = p01 * b0 + p11 * b1
            bt_pb = b0 * pb0 + b1 * pb1

            btpa0 = b0 * (p00 * a00 + p01 * a10) + b1 * (p01 * a00 + p11 * a10)
            btpa1 = b0 * (p00 * a01 + p01 * a11) + b1 * (p01 * a01 + p11 * a11)

            rtilde = r + bt_pb
            step_feasible = rtilde >= -1.0e-9
            if not step_feasible:
                feasible = False
                k0 = 0.0
                k1 = 0.0
            else:
                k0 = (btpa0 + s0) / rtilde
                k1 = (btpa1 + s1) / rtilde

            ac00 = a00 - b0 * k0
            ac01 = a01 - b0 * k1
            ac10 = a10 - b1 * k0
            ac11 = a11 - b1 * k1

            # AK.T @ P_next @ AK
            qd00 = ac00 * (p00 * ac00 + p01 * ac10) + ac10 * (p01 * ac00 + p11 * ac10)
            qd01 = ac00 * (p00 * ac01 + p01 * ac11) + ac10 * (p01 * ac01 + p11 * ac11)
            qd10 = ac01 * (p00 * ac00 + p01 * ac10) + ac11 * (p01 * ac00 + p11 * ac10)
            qd11 = ac01 * (p00 * ac01 + p01 * ac11) + ac11 * (p01 * ac01 + p11 * ac11)

            # Correct sign for u = -Kx, same as active NumPy implementation.
            pt00 = q00 + r * k0 * k0 - (s0 * k0 + k0 * s0) + qd00
            pt01 = q01 + r * k0 * k1 - (s0 * k1 + k0 * s1) + qd01
            pt10 = q10 + r * k1 * k0 - (s1 * k0 + k1 * s0) + qd10
            pt11 = q11 + r * k1 * k1 - (s1 * k1 + k1 * s1) + qd11

            p00 = pt00
            p01 = 0.5 * (pt01 + pt10)
            p11 = pt11

        return feasible


    @nb.njit
    def _lqr_gain_scalar_n2m1(A, B, M_seq, G, lam):
        T = M_seq.shape[0]
        K_seq = np.zeros((T, 1, 2), dtype=np.float64)

        a00 = A[0, 0]; a01 = A[0, 1]; a10 = A[1, 0]; a11 = A[1, 1]
        b0 = B[0, 0]; b1 = B[1, 0]

        p00 = 0.0
        p01 = 0.0
        p11 = 0.0
        feasible = True

        for tt in range(T - 1, -1, -1):
            q00 = M_seq[tt, 0, 0] + lam * G[0, 0]
            q01 = M_seq[tt, 0, 1] + lam * G[0, 1]
            q10 = M_seq[tt, 1, 0] + lam * G[1, 0]
            q11 = M_seq[tt, 1, 1] + lam * G[1, 1]
            s0  = M_seq[tt, 0, 2] + lam * G[0, 2]
            s1  = M_seq[tt, 1, 2] + lam * G[1, 2]
            r   = M_seq[tt, 2, 2] + lam * G[2, 2]

            pb0 = p00 * b0 + p01 * b1
            pb1 = p01 * b0 + p11 * b1
            bt_pb = b0 * pb0 + b1 * pb1

            btpa0 = b0 * (p00 * a00 + p01 * a10) + b1 * (p01 * a00 + p11 * a10)
            btpa1 = b0 * (p00 * a01 + p01 * a11) + b1 * (p01 * a01 + p11 * a11)

            rtilde = r + bt_pb
            step_feasible = rtilde >= -1.0e-9
            if not step_feasible:
                feasible = False
                k0 = 0.0
                k1 = 0.0
            else:
                k0 = (btpa0 + s0) / rtilde
                k1 = (btpa1 + s1) / rtilde

            K_seq[tt, 0, 0] = k0
            K_seq[tt, 0, 1] = k1

            ac00 = a00 - b0 * k0
            ac01 = a01 - b0 * k1
            ac10 = a10 - b1 * k0
            ac11 = a11 - b1 * k1

            qd00 = ac00 * (p00 * ac00 + p01 * ac10) + ac10 * (p01 * ac00 + p11 * ac10)
            qd01 = ac00 * (p00 * ac01 + p01 * ac11) + ac10 * (p01 * ac01 + p11 * ac11)
            qd10 = ac01 * (p00 * ac00 + p01 * ac10) + ac11 * (p01 * ac00 + p11 * ac10)
            qd11 = ac01 * (p00 * ac01 + p01 * ac11) + ac11 * (p01 * ac01 + p11 * ac11)

            pt00 = q00 + r * k0 * k0 - (s0 * k0 + k0 * s0) + qd00
            pt01 = q01 + r * k0 * k1 - (s0 * k1 + k0 * s1) + qd01
            pt10 = q10 + r * k1 * k0 - (s1 * k0 + k1 * s0) + qd10
            pt11 = q11 + r * k1 * k1 - (s1 * k1 + k1 * s1) + qd11

            p00 = pt00
            p01 = 0.5 * (pt01 + pt10)
            p11 = pt11

        return K_seq, feasible


    @nb.njit
    def _cost_and_xseq_scalar_n2m1(A, B, K_seq, Sigma_w, G):
        T = K_seq.shape[0]
        X_seq = np.zeros((T, 3, 3), dtype=np.float64)
        sum_cov = np.zeros((3, 3), dtype=np.float64)

        a00 = A[0, 0]; a01 = A[0, 1]; a10 = A[1, 0]; a11 = A[1, 1]
        b0 = B[0, 0]; b1 = B[1, 0]

        sx00 = Sigma_w[0, 0]
        sx01 = Sigma_w[0, 1]
        sx10 = Sigma_w[1, 0]
        sx11 = Sigma_w[1, 1]

        for tt in range(T):
            k0 = K_seq[tt, 0, 0]
            k1 = K_seq[tt, 0, 1]

            xu0 = -(sx00 * k0 + sx01 * k1)
            xu1 = -(sx10 * k0 + sx11 * k1)
            uu = k0 * (sx00 * k0 + sx01 * k1) + k1 * (sx10 * k0 + sx11 * k1)

            X_seq[tt, 0, 0] = sx00
            X_seq[tt, 0, 1] = sx01
            X_seq[tt, 0, 2] = xu0
            X_seq[tt, 1, 0] = sx10
            X_seq[tt, 1, 1] = sx11
            X_seq[tt, 1, 2] = xu1
            X_seq[tt, 2, 0] = xu0
            X_seq[tt, 2, 1] = xu1
            X_seq[tt, 2, 2] = uu

            for ii in range(3):
                for jj in range(3):
                    sum_cov[ii, jj] += X_seq[tt, ii, jj]

            f00 = a00 - b0 * k0
            f01 = a01 - b0 * k1
            f10 = a10 - b1 * k0
            f11 = a11 - b1 * k1

            n00 = f00 * (sx00 * f00 + sx01 * f01) + f01 * (sx10 * f00 + sx11 * f01) + Sigma_w[0, 0]
            n01 = f00 * (sx00 * f10 + sx01 * f11) + f01 * (sx10 * f10 + sx11 * f11) + Sigma_w[0, 1]
            n10 = f10 * (sx00 * f00 + sx01 * f01) + f11 * (sx10 * f00 + sx11 * f01) + Sigma_w[1, 0]
            n11 = f10 * (sx00 * f10 + sx01 * f11) + f11 * (sx10 * f10 + sx11 * f11) + Sigma_w[1, 1]

            sx00 = n00
            sx01 = 0.5 * (n01 + n10)
            sx10 = sx01
            sx11 = n11

        total_cost = 0.0
        for ii in range(3):
            for jj in range(3):
                total_cost += G[ii, jj] * sum_cov[jj, ii]

        return total_cost, X_seq


    @nb.njit
    def _solve_subproblem_scalar_core(A, B, M_seq, tau, tau_k, G, beta, Sigma_w, lam_upper):
        eps_inner = 1.0e-7
        max_iters_inner = 5000
        max_iters_outer = 5000

        lam_lo = 0.0
        lam_hi = lam_upper

        feasible_zero = _lqr_feasible_scalar_n2m1(A, B, M_seq, G, 0.0)
        if feasible_zero:
            lambda_min = 0.0
        else:
            if lam_hi <= 0.0:
                lam_hi = 1.0
            i = 0
            while i < max_iters_inner:
                feasible_hi = _lqr_feasible_scalar_n2m1(A, B, M_seq, G, lam_hi)
                if feasible_hi:
                    break
                lam_hi *= 2.0
                i += 1

            i = 0
            while (lam_hi - lam_lo > eps_inner) and (i < max_iters_inner):
                lam = 0.5 * (lam_lo + lam_hi)
                feasible = _lqr_feasible_scalar_n2m1(A, B, M_seq, G, lam)
                if feasible:
                    lam_hi = lam
                else:
                    lam_lo = lam
                i += 1
            lambda_min = lam_hi

        lambda_max = lambda_min + 10.0e5
        T_local = M_seq.shape[0]
        K_opt = np.zeros((T_local, 1, 2), dtype=np.float64)
        X_seq_opt = np.zeros((T_local, 3, 3), dtype=np.float64)

        lmin = lambda_min
        lmax = lambda_max
        it = 0
        while (lmax - lmin > eps_inner) and (it < max_iters_outer):
            lmid = 0.5 * (lmin + lmax)
            K_seq, feasible = _lqr_gain_scalar_n2m1(A, B, M_seq, G, lmid)
            cost, X_seq = _cost_and_xseq_scalar_n2m1(A, B, K_seq, Sigma_w, G)

            if cost > beta:
                lmin = lmid
            else:
                lmax = lmid
                K_opt = K_seq
                X_seq_opt = X_seq
            it += 1

        X_prefix = np.zeros((T_local, 3, 3), dtype=np.float64)
        for tt in range(T_local):
            if tt == 0:
                for ii in range(3):
                    for jj in range(3):
                        X_prefix[tt, ii, jj] = X_seq_opt[tt, ii, jj]
            else:
                for ii in range(3):
                    for jj in range(3):
                        X_prefix[tt, ii, jj] = X_prefix[tt - 1, ii, jj] + X_seq_opt[tt, ii, jj]

        Lout = tau.shape[0] - 1
        Lambda_tilde_list = np.zeros((Lout, 3, 3), dtype=np.float64)
        for rr in range(Lout):
            j = int(tau[rr + 1] - tau_k)
            if j < 0:
                j = 0
            if j > T_local - 1:
                j = T_local - 1
            for ii in range(3):
                for jj in range(3):
                    Lambda_tilde_list[rr, ii, jj] = X_prefix[j, ii, jj]

        return Lambda_tilde_list, K_opt


def solve_subproblem(A, B, M_seq, tau, tau_k, tau_K, G, beta, Sigma_w,
                     lambda_min=0.0, lambda_max=500.0, tol=1e-4, max_iter=5000):
    """Scalar-input fast override; falls back to the NumPy reference otherwise."""
    if (_SCALAR_NUMBA_AVAILABLE and A.shape == (2, 2) and B.shape == (2, 1)
            and M_seq.shape[1:] == (3, 3) and G.shape == (3, 3)
            and Sigma_w.shape == (2, 2)):
        A_c = np.ascontiguousarray(A, dtype=np.float64)
        B_c = np.ascontiguousarray(B, dtype=np.float64)
        M_c = np.ascontiguousarray(M_seq, dtype=np.float64)
        G_c = np.ascontiguousarray(G, dtype=np.float64)
        Sw_c = np.ascontiguousarray(Sigma_w, dtype=np.float64)
        tau_c = np.ascontiguousarray(np.asarray(tau, dtype=np.int64))

        # Same lambda upper-bound computation as the NumPy implementation.
        eig_M = np.linalg.eigvalsh(M_c)
        min_eig_M_per_t = np.min(eig_M, axis=1)
        max_neg_eig_M = np.max(-min_eig_M_per_t)
        min_eig_G = np.min(np.linalg.eigvalsh(G_c))
        safe_min_eig_G = np.maximum(min_eig_G, 1e-8)
        #lam_upper = max(0.0, float(max_neg_eig_M) / float(safe_min_eig_G))
        lam_upper = max(0, 0.1)

        return _solve_subproblem_scalar_core(
            A_c, B_c, M_c, tau_c, int(tau_k), G_c, float(beta), Sw_c, float(lam_upper)
        )

    return _solve_subproblem_numpy_reference(
        A, B, M_seq, tau, tau_k, tau_K, G, beta, Sigma_w,
        lambda_min=lambda_min, lambda_max=lambda_max, tol=tol, max_iter=max_iter
    )

    
print("Scalar-input specialized solve_subproblem override installed." if _SCALAR_NUMBA_AVAILABLE else "Numba unavailable; using NumPy solve_subproblem.")




def main_single_run(seed=1, run_naive=True):
    
    # ======================================================================
    # True system & setup parameters
    # ======================================================================
    rng = np.random.default_rng(seed)
    #A_true = np.array([[1.0, 1.0],
     #                  [0.0, 0.9]], dtype=np.float64)
    A_true = np.array([[1.2, 1.0],
                       [0.0, 1.0]], dtype=np.float64)
    B_true = np.array([[0.0],
                       [1.0]], dtype=np.float64)
    n, m = B_true.shape

    noise_std = 0.3
    Sigma_w = (noise_std**2) * np.eye(n, dtype=np.float64)
    x0 = rng.normal(size=(n,)).astype(np.float64)

    T, K_episodes = 400, 20
    M = 400
    Q_lqr = np.eye(n, dtype=np.float64)
    R_lqr = np.eye(m, dtype=np.float64)
    W_base = block_diag_np(Q_lqr, R_lqr)
    G = np.eye(n + m, dtype=np.float64)

    # old \tau interval ------------------------------------------------
    #tau = np.linspace(0, T, K_episodes + 1, dtype=int)

    # new \tau interval -------------------------------------------------
    # More frequent updates early, less frequent updates later, since explorations are emphasized early on
    alpha = 1.7  # larger alpha => more front-loaded updates
    
    grid = np.linspace(0.0, 1.0, K_episodes + 1)
    tau = np.round(T * grid**alpha).astype(int)
    
    # Ensure valid, strictly increasing update times.
    tau[0] = 0
    tau[-1] = T
    tau = np.maximum.accumulate(tau)
    
    # Fix possible duplicates caused by rounding.
    for i in range(1, len(tau)):
        if tau[i] <= tau[i - 1]:
            tau[i] = tau[i - 1] + 1
    
    tau[-1] = T


    # ======================================================================
    # Optimal average per-step baseline: T J_star
    # ======================================================================
    _, K_star = riccati_iteration(A_true, B_true, Q_lqr, R_lqr)
    Sigma_x0_initial = np.outer(x0, x0).astype(np.float64)
    Acl_star = A_true - B_true @ K_star
    Sigma_star = solve_discrete_lyapunov_np(Acl_star.T, Sigma_w)
    J_star = float(np.trace((Q_lqr + K_star.T @ R_lqr @ K_star) @ Sigma_star))

    c_opt_t = np.full(T, J_star, dtype=np.float64)
    cum_opt = np.arange(1, T + 1, dtype=np.float64) * J_star

    # ======================================================================
    # Initial priors
    # ======================================================================
    probe_horizon = 50
    sigma_probe = 0.01
    K_stab = np.zeros((m, n))

    def rollout_with_probe(K_stab, x0, T, sigma_p, rng_probe):
        x = x0.copy()
        xs = []
        xus = []
        for _t in range(T):
            eta = sigma_p * rng_probe.normal(size=(m,)).astype(x.dtype)
            u = -K_stab @ x + eta
            noise_std_prior =0.4
            w = noise_std_prior * rng_probe.normal(size=(n,)).astype(x.dtype)
            x_next = A_true @ x + B_true @ u + w
            xu = np.concatenate([x, u])
            xs.append(x_next)
            xus.append(xu)
            x = x_next
        xs = np.asarray(xs, dtype=x0.dtype)
        xus = np.asarray(xus, dtype=x0.dtype)
        cov = (xus.T @ xus) / T
        cross_cov = (xs.T @ xus) / T
        return cov, cross_cov

    rng_probe = np.random.default_rng(1 + seed)
    S_xu_init, S_xxu_init = rollout_with_probe(K_stab, x0, probe_horizon, sigma_probe, rng_probe)
    AB_hat_init = S_xxu_init @ np.linalg.pinv(S_xu_init)
    A_hat_init, B_hat_init = AB_hat_init[:, :n], AB_hat_init[:, n:]
    
    # ======================================================================
    # Frank-Wolfe approach 
    # ======================================================================
    S_xu, S_xxu = S_xu_init.copy(), S_xxu_init.copy()
    A_hat, B_hat = A_hat_init.copy(), B_hat_init.copy()
    Sigma_x0_FW = Sigma_x0_initial.copy()
    regret_steps_FW = []

    for k in range(K_episodes):
        H_horizon = int(T - int(tau[k]))
        H_hat = np.eye(n * (n + m), dtype=np.float64)
        

        S_list_beta = track_cumulative_covariances_naive_policy(
            A_hat, B_hat,          # synthesize K on estimates
            A_hat, B_hat,          # predicted rollout under estimated dynamics
            Sigma_w, Sigma_x0_FW,
            H_horizon, m,
            int(tau[k]),
            Q_lqr, R_lqr
        )

        tau_remaining = tau[k:] - tau[k]
        K_info = max(len(tau_remaining) - 2, 0)

        beta = compute_objective_tau0(
            S_list_beta,
            tau_remaining,
            K_info,
            Q_lqr, R_lqr, H_hat, S_xu, Sigma_w
        )
        Lambda = S_xu
        X_opt, policies = frank_wolfe(
            A=A_hat, B=B_hat, Lambda=Lambda, beta=beta, Sigma_w=Sigma_w, M_steps=M,
            make_costs_fn=make_costs_fun, solve_subproblem_fn=solve_subproblem,
            W_base=W_base, H_hat=H_hat, tau=tau, k=k, T=T
        )

        # Preserve existing policy-selection choice from this notebook: uniform over FW search policies.
        L = len(policies)
        pick_rng = np.random.default_rng(seed * 12 + k)
        j = int(pick_rng.integers(low=1, high=L + 1))
        K_seq = policies[j - 1]

        ep_len = int(tau[k + 1]) - int(tau[k])
        cov_sum, cross_cov, Sigma_x0_FW, c_FW_k, Z_stack = rollout_combined(
            K_seq[0:ep_len],
            A_true, B_true, Sigma_w, Sigma_x0_FW,
            int(tau[k]), int(tau[k + 1]),
            Q_lqr=Q_lqr, R_lqr=R_lqr
        )

        c_opt_slice = c_opt_t[int(tau[k]):int(tau[k + 1])]
        regret_steps_FW.append(c_FW_k - c_opt_slice)

        S_xu = S_xu + cov_sum
        S_xxu = S_xxu + cross_cov
        AB_hat = S_xxu @ np.linalg.pinv(S_xu)
        A_hat, B_hat = AB_hat[:, :n], AB_hat[:, n:]

    regret_steps_FW = np.concatenate(regret_steps_FW, axis=0)
    cum_regret_FW = np.cumsum(regret_steps_FW)
    total_regret_FW = float(cum_regret_FW[-1])
    print("episodic_total_regret (FW):", total_regret_FW)

    # ======================================================================
    # CE with naive explorations approach
    # ======================================================================
    cum_naive_regret = None
    if run_naive:
        S_xu_naive, S_xxu_naive = S_xu_init.copy(), S_xxu_init.copy()
        A_hat_naive, B_hat_naive = A_hat_init.copy(), B_hat_init.copy()
        Sigma_x0_naive = Sigma_x0_initial.copy()

        c_naive_t = np.zeros((T,), dtype=np.float64)

        for k in range(K_episodes):
            ep_len = int(tau[k + 1] - tau[k])

            S_naive_list_k, Sigma_x0_naive = _naive_policy_covariances_with_final(
                A_hat_naive, B_hat_naive,
                A_true, B_true,
                Sigma_w, Sigma_x0_naive,
                ep_len, m,
                int(tau[k]),
                Q_lqr, R_lqr
            )

            S_naive_stack_k = np.stack(S_naive_list_k, axis=0)
            S_naive_prev_k = np.concatenate([np.zeros_like(S_naive_stack_k[:1]), S_naive_stack_k[:-1]], axis=0)
            Z_naive_k = S_naive_stack_k - S_naive_prev_k
            c_naive_k = np.einsum('ab,tba->t', W_base, Z_naive_k)
            c_naive_t[int(tau[k]):int(tau[k + 1])] = c_naive_k

            Sigma_zz_ep = S_naive_stack_k[-1]
            S_xu_naive = S_xu_naive + Sigma_zz_ep

            # Preserve the existing Naive prior update style from the notebook.
            A_concat_true = np.concatenate([A_true, B_true], axis=1)
            S_xxu_naive = S_xxu_naive + A_concat_true @ Sigma_zz_ep
            AB_hat_naive = S_xxu_naive @ np.linalg.pinv(S_xu_naive)
            A_hat_naive, B_hat_naive = AB_hat_naive[:, :n], AB_hat_naive[:, n:]

        cum_naive = np.cumsum(c_naive_t)
        cum_naive_regret = cum_naive - cum_opt

    return float(cum_regret_FW[-1]), cum_regret_FW, cum_naive_regret


def _run_one_trial_for_mc(trial):
    """
    One Monte Carlo trial wrapper.
    """
    R_fw, cum_regret_FW, cum_naive_regret = main_single_run(
        seed=trial + 17, # 2 is good
        run_naive=True
    )
    
    if cum_naive_regret is None:
        raise RuntimeError("Naive regret is None; ensure run_naive=True.")
    return R_fw, cum_regret_FW, cum_naive_regret


def main_multiple_runs(num_trials=10, parallel=True, max_workers=None):
    """
    Monte Carlo wrapper with optional parallel execution.
    Curves:
      - FW mean + 10--90 percentile band
      - Naive mean + 10--90 percentile band
    """
    import numpy as np
    trial_indices = list(range(num_trials))

    if parallel and num_trials > 1:
        import os
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        if max_workers is None:
            max_workers = min(num_trials, os.cpu_count() or 1)

        print(f"Running {num_trials} Monte Carlo trials in parallel with max_workers={max_workers}.")

        try:
            ctx = mp.get_context("fork")
        except Exception:
            ctx = None

        try:
            if ctx is not None:
                with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
                    results = list(executor.map(_run_one_trial_for_mc, trial_indices))
            else:
                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(_run_one_trial_for_mc, trial_indices))
        except Exception as e:
            print("Parallel execution failed; falling back to serial execution.")
            print("Reason:", repr(e))
            results = []
            for trial in trial_indices:
                print(f"Trial {trial + 1}/{num_trials}")
                results.append(_run_one_trial_for_mc(trial))
    else:
        results = []
        for trial in trial_indices:
            print(f"Trial {trial + 1}/{num_trials}")
            results.append(_run_one_trial_for_mc(trial))

    total_regrets_fw = []
    total_regrets_naive = []
    cum_regret_trials_fw = []
    cum_regret_trials_naive = []

    for trial, (R_fw, cum_regret_FW, cum_naive_regret) in enumerate(results):
        print(f"Finished trial {trial + 1}/{num_trials}")

        total_regrets_fw.append(float(R_fw))
        cum_regret_trials_fw.append(cum_regret_FW)

        total_regrets_naive.append(float(cum_naive_regret[-1]))
        cum_regret_trials_naive.append(cum_naive_regret)


    total_regrets_fw = np.asarray(total_regrets_fw)
    total_regrets_naive = np.asarray(total_regrets_naive)

    cum_regret_trials_fw = np.stack(cum_regret_trials_fw, axis=0)
    cum_regret_trials_naive = np.stack(cum_regret_trials_naive, axis=0)

    Rt_fw_mean = np.mean(cum_regret_trials_fw, axis=0)
    Rt_naive_mean = np.mean(cum_regret_trials_naive, axis=0)

    fw_p10 = np.percentile(cum_regret_trials_fw, 10, axis=0)
    fw_p90 = np.percentile(cum_regret_trials_fw, 90, axis=0)

    naive_p10 = np.percentile(cum_regret_trials_naive, 10, axis=0)
    naive_p90 = np.percentile(cum_regret_trials_naive, 90, axis=0)


    avg_regret_fw = float(np.mean(total_regrets_fw))
    std_regret_fw = float(np.std(total_regrets_fw))
    avg_regret_naive = float(np.mean(total_regrets_naive))
    std_regret_naive = float(np.std(total_regrets_naive))

    import matplotlib.pyplot as plt

    Tlen = Rt_fw_mean.shape[0]
    t_grid = np.arange(1, Tlen + 1)

    fig = plt.figure()

    fw_line, = plt.plot(t_grid, Rt_fw_mean, linewidth=2.75, label="FW mean cumulative regret")
    plt.fill_between(t_grid, fw_p10, fw_p90, alpha=0.20, color=fw_line.get_color(), label="FW 10--90 percentile")

    naive_line, = plt.plot(t_grid, Rt_naive_mean, linewidth=2.75, label="Naive mean cumulative regret")
    plt.fill_between(t_grid, naive_p10, naive_p90, alpha=0.20, color=naive_line.get_color(), label="Naive 10--90 percentile")


    plt.xlabel("time $t$")
    plt.ylabel("cumulative regret $R_t$ vs optimal (LQR cost)")
    plt.title("FW vs Naive cumulative regret: mean and 10--90 percentile bands")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()

    print(f"FW       average_total_regret: {avg_regret_fw:.4f}  std: {std_regret_fw:.4f}")
    print(f"Naive    average_total_regret: {avg_regret_naive:.4f}  std: {std_regret_naive:.4f}")

    return (
        (avg_regret_fw, std_regret_fw),
        (avg_regret_naive, std_regret_naive),
    )


if __name__ == "__main__":
    main_multiple_runs(num_trials=1, parallel=True)


