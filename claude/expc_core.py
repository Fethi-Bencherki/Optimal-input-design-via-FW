"""
Experiment C -- adaptive online LQR (root4.tex Sec. V-D "Adaptive online LQR"
and Sec. VI-C).  Extracted from experiments/"Experiment C.ipynb" and audited
against the paper's formulation.

Protocol (preserved from the notebook / paper figure):
  x_{t+1} = A x_t + B u_t + w_t,  A = [[1.2, 1.0], [0.0, 1.0]], B = [[0], [1]],
  Sigma_w = 0.3^2 I, Q = I, R = I, T = 5000, K_episodes = 20 with a
  front-loaded update grid tau_k = round(T * (k/20)^1.7).
  A 50-step open-loop probe (sigma_probe = 0.01, prior noise std 0.4) provides
  the initial least-squares estimate and Gram matrix (divided by the probe
  length, as in the notebook).  Per-step costs and data updates use expected
  (covariance-propagated) quantities under the true dynamics; regret is
  measured against T * J_star of the true-model LQR controller.

Correctness fixes vs. the notebook (each has a flag so the old behavior can be
reproduced for diagnosis; "paper" = root4.tex conventions):
  F1 interval weights   W_m = (tau_{m+1} - tau_m) * Tr_{Sigma_w}(H)   [eq. (19)]
                        old: (tau_m - tau_{m-1})  (non-causal pairing).
  F2 moment convention  Sigma_m^pi = sum_{t=tau_k}^{tau_m - 1}, stage weights on
                        {m : t < tau_m}; old: inclusive t <= tau_m (off by one,
                        also inconsistent with the notebook's own beta term).
  F3 mixture sampling   p_j = 2(j+1)/(M(M+1))  [Remark 2]; old: uniform.
  F4 LMO covariance     propagation starts from the current state second moment
                        (consistent with beta and the true rollout); old: Sigma_w.
  F5 LMO robustness     stage-II bracket seeded at the generalized bar-lambda of
                        Lemma 3(iv) with a *feasible* fallback policy; old code
                        started from lambda_min + 1e6 and returned ZERO gains if
                        the budget was never met (zero gains on an unstable
                        plant -> covariance blow-up).
  F6 beta guard         beta is evaluated with finite-horizon CE-LQR gains
                        (always well posed) instead of an unguarded DARE value
                        iteration that silently returns garbage on
                        (nearly) unstabilizable estimates, inflating beta.

The "Sampling" baseline (paper Sec. VI-C: "locally perturbs the
certainty-equivalent gains and selects the best candidate under the objective
(19)") is reconstructed here; it was not present in the surviving notebook.
"""

import numpy as np

try:
    import numba as nb
    NUMBA = True
except Exception:  # pragma: no cover
    nb = None
    NUMBA = False


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """All protocol + hyperparameters. Defaults = tuned settings."""

    def __init__(self, **kw):
        # ---------------- protocol (do not change: shared by all methods) ---
        self.T = 5000
        self.K_episodes = 20
        self.tau_alpha = 1.7
        self.noise_std = 0.3
        self.probe_horizon = 50
        self.sigma_probe = 0.01
        self.noise_std_prior = 0.4
        # ---------------- FW hyperparameters ---------------------------------
        self.M_fw = 100                # FW iterations per replanning
        self.fw_sampling = "paper"     # "paper": p_j=2(j+1)/(M(M+1)); "uniform"
        self.beta_explore_scale = 0.1  # sigma_eta^2 = scale/(t+t0) in the
        #                                beta-defining naive comparison policy
        #                                (old notebook: 1.0; actual Naive
        #                                baseline uses 0.1 -> consistent now)
        self.beta_finite_horizon_ce = True   # F6 (False = old DARE)
        self.ridge_lambda = 1e-3       # ridge added to Lambda for FW planning
        #                                (paper: "regularized Gram matrix")
        # ---------------- paper-convention flags ----------------------------
        self.paper_interval_weights = True   # F1
        self.exclusive_moments = True        # F2
        self.lmo_start_state_cov = True      # F4
        self.robust_lmo = True               # F5
        self.s_tol = 1e-9                    # S_t(lambda) > s_tol admissibility
        self.stochastic = True         # realized rollouts ("percentile bands
        #                                over noise realizations", paper VI-C);
        #                                False = expected-covariance accounting
        #                                (surviving notebook's debug state)
        # ---------------- baselines (unchanged tuning) ----------------------
        self.naive_explore_scale = 0.1  # sigma_eta^2 = 0.1/(t+t0)  (notebook)
        self.samp_n_candidates = 30
        self.samp_sigma = 0.1
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown config key {k}")
            setattr(self, k, v)

    def as_dict(self):
        return {k: getattr(self, k) for k in vars(self)}


def notebook_config(**kw):
    """Configuration reproducing the notebook's conventions (pre-fix)."""
    base = dict(
        M_fw=400, fw_sampling="uniform", beta_explore_scale=1.0,
        beta_finite_horizon_ce=False, ridge_lambda=0.0,
        paper_interval_weights=False, exclusive_moments=False,
        lmo_start_state_cov=False, robust_lmo=False, s_tol=-1e-9,
        stochastic=False,
    )
    base.update(kw)
    return Config(**base)


# =============================================================================
# Shared linear-algebra helpers
# =============================================================================

def block_diag_np(*arrays):
    shapes = np.array([a.shape for a in arrays])
    out = np.zeros((int(shapes[:, 0].sum()), int(shapes[:, 1].sum())),
                   dtype=arrays[0].dtype)
    r = c = 0
    for a in arrays:
        rr, cc = a.shape
        out[r:r + rr, c:c + cc] = a
        r += rr
        c += cc
    return out


def dare_solve(A, B, Q, R, tol=1e-10, max_iters=10_000):
    """DARE via value iteration (notebook).  Returns (P, K, converged)."""
    P = Q.copy()
    P_prev = P + 10 * tol * np.eye(Q.shape[0])
    i = 0
    while np.linalg.norm(P - P_prev, ord="fro") > tol and i < max_iters:
        K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        Pn = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        P_prev, P = P, 0.5 * (Pn + Pn.T)
        i += 1
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return P, K, i < max_iters


def solve_discrete_lyapunov_np(A, Q, max_iter=1000, tol=1e-8):
    P = Q
    for _ in range(max_iter):
        P_next = A.T @ P @ A + Q
        if np.linalg.norm(P_next - P) < tol:
            break
        P = P_next
    return P


def finite_horizon_lqr_gains(A, B, Q, R, T):
    """Time-varying finite-horizon LQR gains (always well posed for psd cost)."""
    n, m = B.shape
    P = np.zeros((n, n))
    K_rev = []
    for _ in range(T):
        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)
        P = Q + A.T @ P @ A - (A.T @ P @ B) @ K
        P = 0.5 * (P + P.T)
        K_rev.append(K)
    return np.asarray(K_rev[::-1])


# =============================================================================
# LMO: generalized LQ + bisection over lambda (paper Sec. IV)
# n = 2, m = 1 scalar-specialized numba kernels (port of the notebook kernels,
# with the F4/F5 fixes behind flags).
# =============================================================================

if NUMBA:

    @nb.njit(cache=True)
    def _sweep_gains(A, B, M_seq, W2, lam, s_tol):
        """Backward Riccati (eq. 13) for stage weight M_t + lam*W2.

        Returns (K_seq, feasible) where feasible <=> S_t(lambda) > s_tol
        for all t.  Cross terms handled; u_t = -K_t x_t.
        """
        T = M_seq.shape[0]
        K_seq = np.zeros((T, 1, 2), dtype=np.float64)
        a00 = A[0, 0]; a01 = A[0, 1]; a10 = A[1, 0]; a11 = A[1, 1]
        b0 = B[0, 0]; b1 = B[1, 0]
        p00 = 0.0; p01 = 0.0; p11 = 0.0
        feasible = True
        for tt in range(T - 1, -1, -1):
            q00 = M_seq[tt, 0, 0] + lam * W2[0, 0]
            q01 = M_seq[tt, 0, 1] + lam * W2[0, 1]
            q11 = M_seq[tt, 1, 1] + lam * W2[1, 1]
            s0 = M_seq[tt, 0, 2] + lam * W2[0, 2]
            s1 = M_seq[tt, 1, 2] + lam * W2[1, 2]
            r = M_seq[tt, 2, 2] + lam * W2[2, 2]

            pb0 = p00 * b0 + p01 * b1
            pb1 = p01 * b0 + p11 * b1
            bt_pb = b0 * pb0 + b1 * pb1
            btpa0 = b0 * (p00 * a00 + p01 * a10) + b1 * (p01 * a00 + p11 * a10)
            btpa1 = b0 * (p00 * a01 + p01 * a11) + b1 * (p01 * a01 + p11 * a11)

            rtilde = r + bt_pb
            if rtilde > s_tol:
                k0 = (btpa0 + s0) / rtilde
                k1 = (btpa1 + s1) / rtilde
            else:
                feasible = False
                k0 = 0.0
                k1 = 0.0
            K_seq[tt, 0, 0] = k0
            K_seq[tt, 0, 1] = k1

            ac00 = a00 - b0 * k0; ac01 = a01 - b0 * k1
            ac10 = a10 - b1 * k0; ac11 = a11 - b1 * k1
            qd00 = ac00 * (p00 * ac00 + p01 * ac10) + ac10 * (p01 * ac00 + p11 * ac10)
            qd01 = ac00 * (p00 * ac01 + p01 * ac11) + ac10 * (p01 * ac01 + p11 * ac11)
            qd10 = ac01 * (p00 * ac00 + p01 * ac10) + ac11 * (p01 * ac00 + p11 * ac10)
            qd11 = ac01 * (p00 * ac01 + p01 * ac11) + ac11 * (p01 * ac01 + p11 * ac11)

            pt00 = q00 + r * k0 * k0 - 2.0 * s0 * k0 + qd00
            pt01 = q01 + r * k0 * k1 - (s0 * k1 + k0 * s1) + qd01
            pt10 = q01 + r * k1 * k0 - (s1 * k0 + k1 * s0) + qd10
            pt11 = q11 + r * k1 * k1 - 2.0 * s1 * k1 + qd11

            p00 = pt00
            p01 = 0.5 * (pt01 + pt10)
            p11 = pt11
        return K_seq, feasible

    @nb.njit(cache=True)
    def _cov_budget(A, B, K_seq, Sigma_w, Sigma_x0, W2):
        """Closed-loop second moments; returns (budget=Tr(W2*sum), X_seq)."""
        T = K_seq.shape[0]
        X_seq = np.zeros((T, 3, 3), dtype=np.float64)
        a00 = A[0, 0]; a01 = A[0, 1]; a10 = A[1, 0]; a11 = A[1, 1]
        b0 = B[0, 0]; b1 = B[1, 0]
        sx00 = Sigma_x0[0, 0]; sx01 = Sigma_x0[0, 1]; sx11 = Sigma_x0[1, 1]
        budget = 0.0
        for tt in range(T):
            k0 = K_seq[tt, 0, 0]
            k1 = K_seq[tt, 0, 1]
            xu0 = -(sx00 * k0 + sx01 * k1)
            xu1 = -(sx01 * k0 + sx11 * k1)
            uu = k0 * (sx00 * k0 + sx01 * k1) + k1 * (sx01 * k0 + sx11 * k1)
            X_seq[tt, 0, 0] = sx00; X_seq[tt, 0, 1] = sx01; X_seq[tt, 0, 2] = xu0
            X_seq[tt, 1, 0] = sx01; X_seq[tt, 1, 1] = sx11; X_seq[tt, 1, 2] = xu1
            X_seq[tt, 2, 0] = xu0;  X_seq[tt, 2, 1] = xu1;  X_seq[tt, 2, 2] = uu
            for ii in range(3):
                for jj in range(3):
                    budget += W2[ii, jj] * X_seq[tt, jj, ii]
            f00 = a00 - b0 * k0; f01 = a01 - b0 * k1
            f10 = a10 - b1 * k0; f11 = a11 - b1 * k1
            n00 = f00 * (sx00 * f00 + sx01 * f01) + f01 * (sx01 * f00 + sx11 * f01) + Sigma_w[0, 0]
            n01 = f00 * (sx00 * f10 + sx01 * f11) + f01 * (sx01 * f10 + sx11 * f11) + Sigma_w[0, 1]
            n11 = f10 * (sx00 * f10 + sx01 * f11) + f11 * (sx01 * f10 + sx11 * f11) + Sigma_w[1, 1]
            sx00 = n00; sx01 = n01; sx11 = n11
        return budget, X_seq

    @nb.njit(cache=True)
    def _lmo_core(A, B, M_seq, W2, Sigma_w, Sigma_x0, beta, tau_off,
                  exclusive, s_tol, robust, max_M2, mu_W2):
        """Two-stage bisection LMO.

        Stage I: smallest admissible lambda (S_t > s_tol for all t).
        Stage II: bisection on the budget map b(lambda) <= beta, bracket per
        Lemma 3(iv) (robust=True) or [lambda_min, lambda_min + 1e6] with a
        zero-gain sentinel (robust=False, notebook behavior).

        Returns (Lambda_list, K_opt, diag) with
        diag = [lam_min, lam_star, budget_star, found, fallback, b0].
        """
        T_local = M_seq.shape[0]
        eps_inner = 1e-7
        max_iters_inner = 5000

        # ---------------- Stage I ----------------
        _, feas0 = _sweep_gains(A, B, M_seq, W2, 0.0, s_tol)
        if feas0:
            lam_min = 0.0
        else:
            lam_hi = 0.1
            i = 0
            while i < max_iters_inner:
                _, fh = _sweep_gains(A, B, M_seq, W2, lam_hi, s_tol)
                if fh:
                    break
                lam_hi *= 2.0
                i += 1
            lam_lo = 0.0
            i = 0
            while (lam_hi - lam_lo > eps_inner) and (i < max_iters_inner):
                lam = 0.5 * (lam_lo + lam_hi)
                _, f = _sweep_gains(A, B, M_seq, W2, lam, s_tol)
                if f:
                    lam_hi = lam
                else:
                    lam_lo = lam
                i += 1
            lam_min = lam_hi

        # ---------------- Stage II ----------------
        K_opt = np.zeros((T_local, 1, 2), dtype=np.float64)
        X_opt = np.zeros((T_local, 3, 3), dtype=np.float64)
        bud_opt = np.inf
        found = 0.0
        fallback = 0.0
        b0 = np.nan

        if robust:
            # minimum-energy value b0 (Assumption 1(ii)); stage weight = W2
            M_energy = np.zeros((T_local, 3, 3), dtype=np.float64)
            for tt in range(T_local):
                for ii in range(3):
                    for jj in range(3):
                        M_energy[tt, ii, jj] = W2[ii, jj]
            K_e, _ = _sweep_gains(A, B, M_energy, W2, 0.0, s_tol)
            b0, X_e = _cov_budget(A, B, K_e, Sigma_w, Sigma_x0, W2)

            if beta <= b0:
                # infeasible budget: return the minimum-energy policy
                X_pref = X_e.copy()
                for tt in range(1, T_local):
                    X_pref[tt] += X_pref[tt - 1]
                Lout = tau_off.shape[0]
                Lambda_list = np.zeros((Lout, 3, 3), dtype=np.float64)
                for rr in range(Lout):
                    j = tau_off[rr] - 1 if exclusive else tau_off[rr]
                    if j < 0:
                        j = 0
                    if j > T_local - 1:
                        j = T_local - 1
                    Lambda_list[rr] = X_pref[j]
                diag = np.array([lam_min, np.inf, b0, 0.0, 2.0, b0])
                return Lambda_list, K_e, diag

            # generalized Lemma 3(iv) bracket (valid for indefinite M via
            # Tr(M Sigma) <= ||M||_2 Tr(Sigma)):
            lam_bar = (beta + b0) / (beta - b0) * max_M2 / mu_W2
            lam_hi2 = lam_bar * 1.05
            if lam_hi2 < lam_min * 1.05 + 1e-6:
                lam_hi2 = lam_min * 1.05 + 1e-6
            g = 0
            while g < 60:
                K_h, fh = _sweep_gains(A, B, M_seq, W2, lam_hi2, s_tol)
                bh, Xh = _cov_budget(A, B, K_h, Sigma_w, Sigma_x0, W2)
                if fh and bh <= beta:
                    K_opt = K_h
                    X_opt = Xh
                    bud_opt = bh
                    found = 1.0
                    break
                lam_hi2 *= 2.0
                g += 1
            if found == 0.0:
                fallback = 1.0  # bracket search failed: fall back to the
                K_opt = K_e     # (always feasible) minimum-energy policy
                X_opt = X_e
                bud_opt = b0
        else:
            lam_hi2 = lam_min + 1.0e6

        lmin = lam_min
        lmax = lam_hi2
        it = 0
        while (lmax - lmin > eps_inner) and (it < 200):
            lmid = 0.5 * (lmin + lmax)
            K_m, fm = _sweep_gains(A, B, M_seq, W2, lmid, s_tol)
            bm, Xm = _cov_budget(A, B, K_m, Sigma_w, Sigma_x0, W2)
            if (not fm) or (bm > beta):
                lmin = lmid
            else:
                lmax = lmid
                K_opt = K_m
                X_opt = Xm
                bud_opt = bm
                found = 1.0
            it += 1

        X_pref = X_opt.copy()
        for tt in range(1, T_local):
            X_pref[tt] += X_pref[tt - 1]
        Lout = tau_off.shape[0]
        Lambda_list = np.zeros((Lout, 3, 3), dtype=np.float64)
        for rr in range(Lout):
            j = tau_off[rr] - 1 if exclusive else tau_off[rr]
            if j < 0:
                j = 0
            if j > T_local - 1:
                j = T_local - 1
            Lambda_list[rr] = X_pref[j]
        diag = np.array([lam_min, lmax, bud_opt, found, fallback,
                         b0 if robust else np.nan])
        return Lambda_list, K_opt, diag


if NUMBA:

    @nb.njit(cache=True)
    def _emit_moments(X_opt, tau_off, exclusive):
        T_local = X_opt.shape[0]
        X_pref = X_opt.copy()
        for tt in range(1, T_local):
            X_pref[tt] += X_pref[tt - 1]
        Lout = tau_off.shape[0]
        Lambda_list = np.zeros((Lout, 3, 3), dtype=np.float64)
        for rr in range(Lout):
            j = tau_off[rr] - 1 if exclusive else tau_off[rr]
            if j < 0:
                j = 0
            if j > T_local - 1:
                j = T_local - 1
            Lambda_list[rr] = X_pref[j]
        return Lambda_list

    @nb.njit(cache=True)
    def _lmo_paper(A, B, M_seq, W2, Sigma_w, Sigma_x0, beta, tau_off,
                   exclusive, s_tol, max_M_norm, mu_W2):
        """Paper-faithful LMO (Sec. IV 'Bisection'): a single bisection on
        [0, bar_lambda] in which an inadmissible Riccati sweep OR
        b(lambda) > beta moves lambda_lo, otherwise lambda_hi.  bar_lambda is
        the generalized Lemma 3(iv) bracket (doubling as numerical safety);
        stops early once b(lambda_hi) is within 1e-6 rel. of beta (certificate
        lambda_hi*(beta - b) negligible).  Never returns zero gains: if the
        budget is infeasible (beta <= b0) it returns the minimum-energy
        policy.  Returns (Lambda_list, K_opt, diag) with
        diag = [lam_lo, lam_star, budget_star, found, fallback, b0]."""
        T_local = M_seq.shape[0]

        # minimum-energy policy / value b0 (Assumption 1(ii))
        M_energy = np.empty((T_local, 3, 3), dtype=np.float64)
        for tt in range(T_local):
            for ii in range(3):
                for jj in range(3):
                    M_energy[tt, ii, jj] = W2[ii, jj]
        K_e, _ = _sweep_gains(A, B, M_energy, W2, 0.0, s_tol)
        b0, X_e = _cov_budget(A, B, K_e, Sigma_w, Sigma_x0, W2)
        if beta <= b0:
            diag = np.array([np.nan, np.inf, b0, 0.0, 2.0, b0])
            return _emit_moments(X_e, tau_off, exclusive), K_e, diag

        # lambda = 0 admissible with slack budget -> exact optimum at lambda=0
        K0, f0 = _sweep_gains(A, B, M_seq, W2, 0.0, s_tol)
        if f0:
            bz, Xz = _cov_budget(A, B, K0, Sigma_w, Sigma_x0, W2)
            if bz <= beta:
                diag = np.array([0.0, 0.0, bz, 1.0, 0.0, b0])
                return _emit_moments(Xz, tau_off, exclusive), K0, diag

        # upper bracket
        lam_hi = (beta + b0) / (beta - b0) * max_M_norm / mu_W2
        if lam_hi < 1e-8:
            lam_hi = 1e-8
        found = 0.0
        bud_opt = b0
        K_opt = K_e
        X_opt = X_e
        g = 0
        while g < 80:
            K_h, fh = _sweep_gains(A, B, M_seq, W2, lam_hi, s_tol)
            if fh:
                bh, Xh = _cov_budget(A, B, K_h, Sigma_w, Sigma_x0, W2)
                if bh <= beta:
                    K_opt = K_h
                    X_opt = Xh
                    bud_opt = bh
                    found = 1.0
                    break
            lam_hi *= 2.0
            g += 1
        if found == 0.0:
            diag = np.array([np.nan, np.inf, b0, 0.0, 1.0, b0])
            return _emit_moments(X_e, tau_off, exclusive), K_e, diag

        # folded bisection (paper Sec. IV)
        lam_lo = 0.0
        it = 0
        while (lam_hi - lam_lo) > max(1e-12, 1e-8 * lam_hi) and it < 100:
            lmid = 0.5 * (lam_lo + lam_hi)
            K_m, fm = _sweep_gains(A, B, M_seq, W2, lmid, s_tol)
            if not fm:
                lam_lo = lmid
                it += 1
                continue
            bm, Xm = _cov_budget(A, B, K_m, Sigma_w, Sigma_x0, W2)
            if bm > beta:
                lam_lo = lmid
            else:
                lam_hi = lmid
                K_opt = K_m
                X_opt = Xm
                bud_opt = bm
                if bm >= beta * (1.0 - 1e-8):
                    break
            it += 1
        diag = np.array([lam_lo, lam_hi, bud_opt, 1.0, 0.0, b0])
        return _emit_moments(X_opt, tau_off, exclusive), K_opt, diag

    @nb.njit(cache=True)
    def _static_gain_moments(A, B, K_batch, Sigma_w, Sigma_x0, tau_off, T_loc):
        """For each static gain K_c (u = -K_c x, no added noise), propagate
        second moments for T_loc steps under (A, B) and return the cumulative
        z-moment S at the local offsets tau_off (S over stages 0..off-1) plus
        the final total S_T.  Output shape (C, L+1, 3, 3), last slot = S_T.
        Used by the Sampling baseline's objective-(19) evaluation."""
        C = K_batch.shape[0]
        L = tau_off.shape[0]
        out = np.zeros((C, L + 1, 3, 3), dtype=np.float64)
        a00 = A[0, 0]; a01 = A[0, 1]; a10 = A[1, 0]; a11 = A[1, 1]
        b0 = B[0, 0]; b1 = B[1, 0]
        for c in range(C):
            k0 = K_batch[c, 0, 0]
            k1 = K_batch[c, 0, 1]
            f00 = a00 - b0 * k0; f01 = a01 - b0 * k1
            f10 = a10 - b1 * k0; f11 = a11 - b1 * k1
            sx00 = Sigma_x0[0, 0]; sx01 = Sigma_x0[0, 1]; sx11 = Sigma_x0[1, 1]
            s = np.zeros((3, 3), dtype=np.float64)
            ptr = 0
            for tt in range(T_loc):
                while ptr < L and tau_off[ptr] == tt:
                    out[c, ptr] = s
                    ptr += 1
                xu0 = -(sx00 * k0 + sx01 * k1)
                xu1 = -(sx01 * k0 + sx11 * k1)
                uu = k0 * (sx00 * k0 + sx01 * k1) + k1 * (sx01 * k0 + sx11 * k1)
                s[0, 0] += sx00; s[0, 1] += sx01; s[0, 2] += xu0
                s[1, 0] += sx01; s[1, 1] += sx11; s[1, 2] += xu1
                s[2, 0] += xu0;  s[2, 1] += xu1;  s[2, 2] += uu
                n00 = f00 * (sx00 * f00 + sx01 * f01) + f01 * (sx01 * f00 + sx11 * f01) + Sigma_w[0, 0]
                n01 = f00 * (sx00 * f10 + sx01 * f11) + f01 * (sx01 * f10 + sx11 * f11) + Sigma_w[0, 1]
                n11 = f10 * (sx00 * f10 + sx01 * f11) + f11 * (sx01 * f10 + sx11 * f11) + Sigma_w[1, 1]
                sx00 = n00; sx01 = n01; sx11 = n11
            while ptr < L and tau_off[ptr] == T_loc:
                out[c, ptr] = s
                ptr += 1
            out[c, L] = s
        return out


def sampling_objectives(A_hat, B_hat, K_batch, Sigma_w, Sigma_x0, tau_rel,
                        W_base, D_base, Lambda0, cfg):
    """Objective (19) for a batch of static candidate gains under the
    estimated model (numba fast path; mirrors objective_eq19 with the
    track_covariances_noisy(explore_scale=0) tracker)."""
    T_loc = int(tau_rel[-1])
    K_info = max(len(tau_rel) - 2, 0)
    tau_off = np.asarray(tau_rel[1:K_info + 1], dtype=np.int64)  # exclusive
    S = _static_gain_moments(np.ascontiguousarray(A_hat),
                             np.ascontiguousarray(B_hat),
                             np.ascontiguousarray(K_batch),
                             np.ascontiguousarray(Sigma_w),
                             np.ascontiguousarray(Sigma_x0),
                             tau_off, T_loc)
    C = K_batch.shape[0]
    vals = np.empty(C)
    for c in range(C):
        lqr = float(np.trace(W_base @ S[c, -1]))
        pen = 0.0
        for i in range(K_info):
            mm = i + 1
            if cfg.paper_interval_weights:
                delta = int(tau_rel[mm + 1]) - int(tau_rel[mm])
            else:
                delta = int(tau_rel[mm]) - int(tau_rel[mm - 1])
            inside = Lambda0 + S[c, i]
            inside = 0.5 * (inside + inside.T)
            pen += delta * float(np.trace(D_base @ np.linalg.inv(inside)))
        vals[c] = lqr + pen
    return vals


def solve_lmo(A, B, M_seq, W2, Sigma_w, Sigma_x0, beta, tau_abs, tau_k, cfg):
    """Python wrapper: LMO for the remaining horizon.

    tau_abs: absolute remaining grid [tau_k, tau_{k+1}, ..., T]; moments are
    returned at tau_{k+1}, ..., T.
    """
    if not NUMBA:
        raise RuntimeError(
            "numba is required for the T=5000 protocol; run with "
            "/opt/miniconda3/envs/cyberrunner/bin/python")
    tau_off = (np.asarray(tau_abs[1:], dtype=np.int64) - int(tau_k))
    # Frobenius norm upper-bounds the spectral norm; the bracket only needs
    # an upper bound on ||M_t||_2 (Lemma 3(iv) holds for any lambda >= bar).
    max_M2 = float(np.sqrt(np.max(np.einsum("tij,tij->t", M_seq, M_seq))))
    mu_W2 = float(np.min(np.linalg.eigvalsh(W2)))
    Sx0 = Sigma_x0 if cfg.lmo_start_state_cov else Sigma_w
    if cfg.robust_lmo:
        Lambda_list, K_opt, diag = _lmo_paper(
            np.ascontiguousarray(A), np.ascontiguousarray(B),
            np.ascontiguousarray(M_seq), np.ascontiguousarray(W2),
            np.ascontiguousarray(Sigma_w), np.ascontiguousarray(Sx0),
            float(beta), tau_off, bool(cfg.exclusive_moments),
            float(cfg.s_tol), max_M2, mu_W2)
    else:
        Lambda_list, K_opt, diag = _lmo_core(
            np.ascontiguousarray(A), np.ascontiguousarray(B),
            np.ascontiguousarray(M_seq), np.ascontiguousarray(W2),
            np.ascontiguousarray(Sigma_w), np.ascontiguousarray(Sx0),
            float(beta), tau_off, bool(cfg.exclusive_moments),
            float(cfg.s_tol), bool(cfg.robust_lmo), max_M2, mu_W2)
    return Lambda_list, K_opt, diag


# =============================================================================
# FW stage-cost construction (paper Sec. V-B/V-D)
# =============================================================================

def partial_trace_H(H_hat, Sigma_w, n, nxu):
    """[Tr_{Sigma_w}(H)]_{ij} = Tr(Sigma_w [H]_{(i,j)})  (paper Sec. V-D)."""
    H4 = H_hat.reshape(n, nxu, n, nxu)
    D = np.tensordot(Sigma_w, H4, axes=([0, 1], [0, 2]))
    return 0.5 * (D + D.T)


def make_costs(T, tau, k, W_base, Inv_stack, D_base, cfg):
    """Stage weights M_t = blkdiag(Q,R) + sum_{m in info(t)} M_m^{(i)} for
    t = tau_k, ..., T-1, with M_m^{(i)} = -Inv_m W_m Inv_m and
    W_m = Delta_m * D_base.

    F1: Delta_m = tau_{m+1} - tau_m (paper) vs old tau_m - tau_{m-1}.
    F2: info(t) = {m : t < tau_m} (paper) vs old {m : t <= tau_m}.
    """
    tau = np.asarray(tau)
    nxu = W_base.shape[0]
    K_info_terminal = tau.shape[0] - 2  # last nonterminal update index
    L = K_info_terminal - k
    tau_k = int(tau[k])
    times = np.arange(tau_k, T)
    Wb = 0.5 * (W_base + W_base.T)
    base = np.broadcast_to(Wb, (times.shape[0], nxu, nxu))
    if L <= 0:
        return base.copy()
    m_idx = np.arange(k + 1, K_info_terminal + 1)
    if cfg.paper_interval_weights:
        deltas = (tau[m_idx + 1] - tau[m_idx]).astype(np.float64)
    else:
        deltas = (tau[m_idx] - tau[m_idx - 1]).astype(np.float64)
    W_info = np.empty((L, nxu, nxu))
    for i, (Inv_m, d) in enumerate(zip(Inv_stack, deltas)):
        X = Inv_m @ (d * D_base) @ Inv_m
        W_info[i] = -0.5 * (X + X.T)
    tau_m_arr = tau[m_idx]
    if cfg.exclusive_moments:
        mask = (times[:, None] < tau_m_arr[None, :]).astype(np.float64)
    else:
        mask = (times[:, None] <= tau_m_arr[None, :]).astype(np.float64)
    M_seq = base + np.einsum("tm,mij->tij", mask, W_info)
    return 0.5 * (M_seq + np.swapaxes(M_seq, -1, -2))


# =============================================================================
# Frank-Wolfe over the remaining horizon
# =============================================================================

def frank_wolfe(A, B, Lambda, beta, Sigma_w, Sigma_x0, W_base, W2, D_base,
                tau, k, T, cfg):
    """Returns (policies, weights, ep_diag).  policies[j] is the LMO gain
    sequence of FW iteration j; weights are the mixture probabilities."""
    K_terminal = tau.shape[0] - 1
    nxu = W_base.shape[0]
    L = K_terminal - k
    X_current = [np.zeros((nxu, nxu)) for _ in range(L)]
    Lambda_k = 0.5 * (Lambda + Lambda.T) + cfg.ridge_lambda * np.eye(nxu)

    tau_remaining = np.asarray(tau[k:], dtype=np.int64)
    policies = []
    diags = []

    # The LMO is iterate-independent when no information terms remain.
    M_steps = cfg.M_fw if L > 1 else 1

    for j in range(M_steps):
        if L > 1:
            X_arr = np.stack(X_current[:-1], axis=0)
            X_arr = 0.5 * (X_arr + np.swapaxes(X_arr, -1, -2))
            A_stack = X_arr + Lambda_k[None, :, :]
            Inv_stack = np.stack([np.linalg.inv(Am) for Am in A_stack])
            Inv_stack = 0.5 * (Inv_stack + np.swapaxes(Inv_stack, -1, -2))
        else:
            Inv_stack = np.zeros((0, nxu, nxu))

        M_seq = make_costs(T, tau, k, W_base, Inv_stack, D_base, cfg)
        S_list, K_seq, diag = solve_lmo(
            A, B, M_seq, W2, Sigma_w, Sigma_x0, beta,
            tau_remaining, int(tau[k]), cfg)

        gamma = 2.0 / (j + 2.0)
        X_current = [0.5 * ((1 - gamma) * X + gamma * S
                            + ((1 - gamma) * X + gamma * S).T)
                     for X, S in zip(X_current, S_list)]
        policies.append(K_seq)
        diags.append(diag)

    M = len(policies)
    if cfg.fw_sampling == "paper" and M > 1:
        weights = 2.0 * (np.arange(M) + 1.0) / (M * (M + 1.0))
    else:
        weights = np.full(M, 1.0 / M)
    return policies, weights, np.asarray(diags)


# =============================================================================
# Expected rollouts / covariance trackers
# =============================================================================

def rollout_expected(K_seq, A, B, Sigma_w, Sigma_x0, ep_len, W_base):
    """Expected-covariance rollout of u_t = -K_t x_t on (A, B) (notebook's
    rollout_combined).  Returns (cov_sum, cross_sum, Sigma_x_final, c_stage)."""
    n, m = B.shape
    I_n = np.eye(n)
    cov_sum = np.zeros((n + m, n + m))
    cross_sum = np.zeros((n, n + m))
    Zx = Sigma_x0.copy()
    c_stage = np.zeros(ep_len)
    for t in range(ep_len):
        K_t = np.asarray(K_seq[t]).reshape(m, n)
        U = K_t @ Zx @ K_t.T
        XU = -Zx @ K_t.T
        Zz = np.block([[Zx, XU], [XU.T, U]])
        cov_sum += Zz
        F = A - B @ K_t
        Hs = np.vstack([I_n, -K_t])
        cross_sum += F @ Zx @ Hs.T
        c_stage[t] = np.einsum("ab,ba->", W_base, Zz)
        Zx = F @ Zx @ F.T + Sigma_w
        Zx = 0.5 * (Zx + Zx.T)
    return cov_sum, cross_sum, Zx, c_stage


def track_covariances_noisy(A_syn, B_syn, A_roll, B_roll, Sigma_w, Sigma_x0,
                            T, t0, explore_scale, Q, R, K=None, K_seq=None):
    """Cumulative z-moments of u_t = -K x_t + eta_t with
    Var(eta_t) = explore_scale/(t + t0) I, propagated under (A_roll, B_roll).
    K synthesized on (A_syn, B_syn) by DARE if not given; K_seq (time varying)
    takes precedence.  Returns (S_list, Sigma_x_final).  (notebook's
    track_cumulative_covariances_naive_policy / _naive_policy_covariances...)
    """
    n, m = B_roll.shape
    if K_seq is None and K is None:
        _, K, _ = dare_solve(A_syn, B_syn, Q, R)
    Z = np.asarray(Sigma_x0, dtype=np.float64).copy()
    S_list = []
    S = np.zeros((n + m, n + m))
    for t in range(1, T + 1):
        Kt = np.asarray(K_seq[t - 1]).reshape(m, n) if K_seq is not None else K
        Sigma_eta = (explore_scale / (t + t0)) * np.eye(m)
        U = Kt @ Z @ Kt.T + Sigma_eta
        XU = -Z @ Kt.T
        Zj = np.block([[Z, XU], [XU.T, U]])
        S = S + Zj
        S_list.append(S.copy())
        Acl = A_roll - B_roll @ Kt
        Z = Acl @ Z @ Acl.T + Sigma_w + B_roll @ Sigma_eta @ B_roll.T
        Z = 0.5 * (Z + Z.T)
    return S_list, Z


def objective_eq19(S_list, tau_rel, W_base, D_base, Lambda0, cfg):
    """Objective (19) value of a policy with cumulative moments S_list
    (S_list[t-1] = sum of first t stage moments), remaining grid tau_rel
    = [0, tau_{k+1}-tau_k, ..., T-tau_k].  LQR term + information penalty
    with weights Delta_m (F1 flag).  Uses the partial-trace identity
    Tr(H (P kron Sigma_w)) = Tr(Tr_{Sigma_w}(H) P)."""
    S_T = S_list[-1]
    lqr_cost = float(np.trace(W_base @ S_T))
    penalty = 0.0
    K_info = max(len(tau_rel) - 2, 0)
    for mm in range(1, K_info + 1):
        tau_m = int(tau_rel[mm])
        if cfg.paper_interval_weights:
            delta = int(tau_rel[mm + 1]) - tau_m
        else:
            delta = tau_m - int(tau_rel[mm - 1])
        S_tau_m = S_list[tau_m - 1]
        inside = Lambda0 + S_tau_m
        inside = 0.5 * (inside + inside.T)
        penalty += delta * float(np.trace(D_base @ np.linalg.inv(inside)))
    return lqr_cost + penalty



def rollout_real(K_seq, A, B, x, W_noise, W_base, eta_seq=None):
    """Realized rollout u_t = -K_t x_t (+ eta_t) on (A, B) with given noise
    rows W_noise (ep_len, n).  Returns (S_zz, S_xz, x_final, stage_costs)."""
    n = A.shape[0]
    m = B.shape[1] if B.ndim == 2 else 1
    ep_len = W_noise.shape[0]
    S_zz = np.zeros((n + m, n + m))
    S_xz = np.zeros((n, n + m))
    costs = np.zeros(ep_len)
    for t in range(ep_len):
        K_t = np.asarray(K_seq[t]).reshape(m, n)
        u = -K_t @ x
        if eta_seq is not None:
            u = u + eta_seq[t]
        z = np.concatenate([x, u])
        costs[t] = z @ W_base @ z
        x_next = A @ x + B @ u + W_noise[t]
        S_zz += np.outer(z, z)
        S_xz += np.outer(x_next, z)
        x = x_next
    return S_zz, S_xz, x, costs


def _safe_ls_update(S_xu, S_xxu, cov_sum, cross_sum, A_hat, B_hat, n):
    """Accumulate expected data and re-estimate; on non-finite data or a
    failed pinv (numerical blow-up), keep the previous estimate so the run
    records the blown-up regret honestly instead of crashing."""
    if np.all(np.isfinite(cov_sum)) and np.all(np.isfinite(cross_sum)):
        S_xu = S_xu + cov_sum
        S_xxu = S_xxu + cross_sum
        try:
            AB = S_xxu @ np.linalg.pinv(S_xu)
            if np.all(np.isfinite(AB)):
                A_hat, B_hat = AB[:, :n], AB[:, n:]
        except np.linalg.LinAlgError:
            pass
    return S_xu, S_xxu, A_hat, B_hat


# =============================================================================
# Single-seed experiment (all three methods)
# =============================================================================

def build_tau(T, K_episodes, alpha):
    grid = np.linspace(0.0, 1.0, K_episodes + 1)
    tau = np.round(T * grid ** alpha).astype(int)
    tau[0] = 0
    tau[-1] = T
    tau = np.maximum.accumulate(tau)
    for i in range(1, len(tau)):
        if tau[i] <= tau[i - 1]:
            tau[i] = tau[i - 1] + 1
    tau[-1] = T
    return tau


def run_single_seed(seed, cfg, methods=("fw", "naive", "sampling")):
    rng = np.random.default_rng(seed)
    A_true = np.array([[1.2, 1.0], [0.0, 1.0]])
    B_true = np.array([[0.0], [1.0]])
    n, m = B_true.shape
    nxu = n + m
    Sigma_w = (cfg.noise_std ** 2) * np.eye(n)
    x0 = rng.normal(size=(n,))

    T, K_episodes = cfg.T, cfg.K_episodes
    Q_lqr = np.eye(n)
    R_lqr = np.eye(m)
    W_base = block_diag_np(Q_lqr, R_lqr)
    W2 = W_base.copy()          # budget weight = blkdiag(Q,R) (LQR energy)
    tau = build_tau(T, K_episodes, cfg.tau_alpha)

    # optimal steady-state cost of the true-model LQR controller
    P_star, K_star, _ = dare_solve(A_true, B_true, Q_lqr, R_lqr)
    Acl_star = A_true - B_true @ K_star
    Sigma_star = solve_discrete_lyapunov_np(Acl_star.T, Sigma_w)
    J_star = float(np.trace((Q_lqr + K_star.T @ R_lqr @ K_star) @ Sigma_star))
    c_opt_t = np.full(T, J_star)

    # ---------------- initial probe (shared by all methods) -----------------
    rng_probe = np.random.default_rng(1 + seed)
    x = x0.copy()
    xs, xus = [], []
    for _t in range(cfg.probe_horizon):
        eta = cfg.sigma_probe * rng_probe.normal(size=(m,))
        u = eta  # K_stab = 0
        w = cfg.noise_std_prior * rng_probe.normal(size=(n,))
        x_next = A_true @ x + B_true @ u + w
        xus.append(np.concatenate([x, u]))
        xs.append(x_next)
        x = x_next
    xs = np.asarray(xs)
    xus = np.asarray(xus)
    S_xu_init = (xus.T @ xus) / cfg.probe_horizon
    S_xxu_init = (xs.T @ xus) / cfg.probe_horizon
    AB0 = S_xxu_init @ np.linalg.pinv(S_xu_init)
    A_hat0, B_hat0 = AB0[:, :n], AB0[:, n:]
    Sigma_x0_initial = np.outer(x0, x0)

    H_hat = np.eye(n * nxu)     # identity surrogate for the model-task Hessian
    D_base = partial_trace_H(H_hat, Sigma_w, n, nxu)

    # shared process-noise realizations (common random numbers across methods)
    W_noise_all = cfg.noise_std * np.random.default_rng(
        seed * 100003 + 9).normal(size=(T, n))

    if cfg.stochastic:
        # comparator: the true-model LQR controller K* on the SAME noise
        # realization ("regret relative to the true-model LQR controller");
        # cancels the common O(sqrt(T)) noise-luck component of the cost.
        K_star_seq = np.repeat(K_star[None, :, :], T, axis=0)
        _, _, _, c_opt_t = rollout_real(K_star_seq, A_true, B_true,
                                        x0.copy(), W_noise_all, W_base)

    out = {"seed": seed, "J_star": J_star, "tau": tau}

    # ======================= FW dual-control method ==========================
    if "fw" in methods:
        S_xu, S_xxu = S_xu_init.copy(), S_xxu_init.copy()
        A_hat, B_hat = A_hat0.copy(), B_hat0.copy()
        Sigma_x0 = Sigma_x0_initial.copy()
        x_cur = x0.copy()
        regret_steps = []
        ep_diag = []

        for k in range(K_episodes):
            if cfg.stochastic:
                Sigma_x0 = np.outer(x_cur, x_cur)
            H_horizon = int(T - tau[k])
            # ---- budget beta: objective of the CE + exploration comparison
            if cfg.beta_finite_horizon_ce:
                K_ce_seq = finite_horizon_lqr_gains(A_hat, B_hat, Q_lqr, R_lqr,
                                                    H_horizon)
                S_list_beta, _ = track_covariances_noisy(
                    A_hat, B_hat, A_hat, B_hat, Sigma_w, Sigma_x0,
                    H_horizon, int(tau[k]), cfg.beta_explore_scale,
                    Q_lqr, R_lqr, K_seq=K_ce_seq)
            else:
                S_list_beta, _ = track_covariances_noisy(
                    A_hat, B_hat, A_hat, B_hat, Sigma_w, Sigma_x0,
                    H_horizon, int(tau[k]), cfg.beta_explore_scale,
                    Q_lqr, R_lqr)
            tau_rel = tau[k:] - tau[k]
            beta = objective_eq19(S_list_beta, tau_rel, W_base, D_base,
                                  S_xu, cfg)

            policies, weights, fw_diags = frank_wolfe(
                A_hat, B_hat, S_xu, beta, Sigma_w, Sigma_x0,
                W_base, W2, D_base, tau, k, T, cfg)

            pick_rng = np.random.default_rng(seed * 12 + k)
            j = int(pick_rng.choice(len(policies), p=weights))
            K_seq = policies[j]

            ep_len = int(tau[k + 1]) - int(tau[k])
            if cfg.stochastic:
                cov_sum, cross_sum, x_cur, c_k = rollout_real(
                    K_seq, A_true, B_true, x_cur,
                    W_noise_all[tau[k]:tau[k + 1]], W_base)
            else:
                cov_sum, cross_sum, Sigma_x0, c_k = rollout_expected(
                    K_seq, A_true, B_true, Sigma_w, Sigma_x0, ep_len, W_base)
            regret_steps.append(c_k - c_opt_t[tau[k]:tau[k + 1]])

            Kmax = float(np.max(np.abs(K_seq[:ep_len])))
            rho = max(float(np.max(np.abs(np.linalg.eigvals(
                A_true - B_true @ np.asarray(K_seq[t]).reshape(m, n)))))
                for t in range(min(ep_len, 50)))
            ep_diag.append(dict(
                k=k, beta=float(beta), picked=j,
                lam_min=float(fw_diags[-1][0]), lam_star=float(fw_diags[-1][1]),
                budget=float(fw_diags[-1][2]), found=float(fw_diags[-1][3]),
                fallback=float(np.sum(fw_diags[:, 4])),
                est_err=float(np.linalg.norm(A_hat - A_true) +
                              np.linalg.norm(B_hat - B_true)),
                Kmax=Kmax, rho_true_cl=rho,
                ep_regret=float(np.sum(regret_steps[-1]))))

            S_xu, S_xxu, A_hat, B_hat = _safe_ls_update(
                S_xu, S_xxu, cov_sum, cross_sum, A_hat, B_hat, n)

        out["fw"] = np.cumsum(np.concatenate(regret_steps))
        out["fw_diag"] = ep_diag

    # ======================= CE + Naive baseline =============================
    if "naive" in methods:
        S_xu, S_xxu = S_xu_init.copy(), S_xxu_init.copy()
        A_hat, B_hat = A_hat0.copy(), B_hat0.copy()
        Sigma_x0 = Sigma_x0_initial.copy()
        c_t = np.zeros(T)
        AB_true = np.concatenate([A_true, B_true], axis=1)
        x_cur = x0.copy()
        for k in range(K_episodes):
            ep_len = int(tau[k + 1] - tau[k])
            if cfg.stochastic:
                _, K_nv, _ = dare_solve(A_hat, B_hat, Q_lqr, R_lqr)
                rng_eta = np.random.default_rng(seed * 31 + 2000 + k)
                t_loc = np.arange(1, ep_len + 1)
                eta_seq = (np.sqrt(cfg.naive_explore_scale /
                                   (t_loc + int(tau[k])))[:, None]
                           * rng_eta.normal(size=(ep_len, m)))
                K_seq_nv = np.repeat(K_nv[None, :, :], ep_len, axis=0)
                cov_sum, cross_sum, x_cur, c_k = rollout_real(
                    K_seq_nv, A_true, B_true, x_cur,
                    W_noise_all[tau[k]:tau[k + 1]], W_base, eta_seq=eta_seq)
                c_t[tau[k]:tau[k + 1]] = c_k
                S_xu, S_xxu, A_hat, B_hat = _safe_ls_update(
                    S_xu, S_xxu, cov_sum, cross_sum, A_hat, B_hat, n)
            else:
                S_list, Sigma_x0 = track_covariances_noisy(
                    A_hat, B_hat, A_true, B_true, Sigma_w, Sigma_x0,
                    ep_len, int(tau[k]), cfg.naive_explore_scale, Q_lqr, R_lqr)
                S_stack = np.stack(S_list)
                Z_k = S_stack - np.concatenate(
                    [np.zeros_like(S_stack[:1]), S_stack[:-1]], axis=0)
                c_t[tau[k]:tau[k + 1]] = np.einsum("ab,tba->t", W_base, Z_k)
                Sigma_zz_ep = S_stack[-1]
                S_xu, S_xxu, A_hat, B_hat = _safe_ls_update(
                    S_xu, S_xxu, Sigma_zz_ep, AB_true @ Sigma_zz_ep,
                    A_hat, B_hat, n)
        out["naive"] = np.cumsum(c_t - c_opt_t)

    # ======================= Sampling baseline ===============================
    if "sampling" in methods:
        S_xu, S_xxu = S_xu_init.copy(), S_xxu_init.copy()
        A_hat, B_hat = A_hat0.copy(), B_hat0.copy()
        Sigma_x0 = Sigma_x0_initial.copy()
        c_t = np.zeros(T)
        AB_true = np.concatenate([A_true, B_true], axis=1)
        x_cur = x0.copy()
        for k in range(K_episodes):
            if cfg.stochastic:
                Sigma_x0 = np.outer(x_cur, x_cur)
            H_horizon = int(T - tau[k])
            ep_len = int(tau[k + 1] - tau[k])
            tau_rel = tau[k:] - tau[k]
            samp_rng = np.random.default_rng(seed * 977 + 31 * k + 5)
            _, K_ce, _ = dare_solve(A_hat, B_hat, Q_lqr, R_lqr)
            candidates = np.stack(
                [K_ce] + [K_ce + cfg.samp_sigma * samp_rng.normal(size=K_ce.shape)
                          for _ in range(cfg.samp_n_candidates)])
            if NUMBA:
                vals = sampling_objectives(A_hat, B_hat, candidates, Sigma_w,
                                           Sigma_x0, tau_rel, W_base, D_base,
                                           S_xu, cfg)
            else:
                vals = []
                for K_c in candidates:
                    S_list, _ = track_covariances_noisy(
                        A_hat, B_hat, A_hat, B_hat, Sigma_w, Sigma_x0,
                        H_horizon, int(tau[k]), 0.0, Q_lqr, R_lqr, K=K_c)
                    vals.append(objective_eq19(S_list, tau_rel, W_base,
                                               D_base, S_xu, cfg))
                vals = np.asarray(vals)
            best_K = candidates[int(np.argmin(vals))]
            K_seq = np.repeat(best_K[None, :, :], ep_len, axis=0)
            if cfg.stochastic:
                cov_sum, cross_sum, x_cur, c_k = rollout_real(
                    K_seq, A_true, B_true, x_cur,
                    W_noise_all[tau[k]:tau[k + 1]], W_base)
            else:
                cov_sum, cross_sum, Sigma_x0, c_k = rollout_expected(
                    K_seq, A_true, B_true, Sigma_w, Sigma_x0, ep_len, W_base)
            c_t[tau[k]:tau[k + 1]] = c_k
            S_xu, S_xxu, A_hat, B_hat = _safe_ls_update(
                S_xu, S_xxu, cov_sum, cross_sum, A_hat, B_hat, n)
        out["sampling"] = np.cumsum(c_t - c_opt_t)

    return out
