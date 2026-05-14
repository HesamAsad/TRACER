"""
Toy linear experiment (updated): adds right-preconditioned GD to exactly match closed forms
in 1 step for FT and SD, while keeping plain GD to show slow convergence under ill-conditioning.

Preconditioning:
  - FT:  W <- W - ((W C - Y X^T) @ C_plus), with gamma=1  => equals closed-form FT in 1 step.
  - SD:  W <- W - (( (1+λ) W C - (Y X^T + λ W0 C) ) @ (C_plus/(1+λ))), gamma=1 => equals closed-form SD.

Everything else matches the previous script (data model, closed forms, plots).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any

# ----------------------------- Utilities ----------------------------- #

def set_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def rand_orth(rng: np.random.Generator, d: int, r: int) -> np.ndarray:
    A = rng.standard_normal((d, r))
    Q, _ = np.linalg.qr(A, mode='reduced')
    return Q[:, :r]

def projector_and_pinvC_from_X(X: np.ndarray, rcond: float = 1e-15) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns P (data-space projector onto range(X)), C_plus, U_r, cvals.
    Uses a very small rcond to avoid truncating directions (for exact equalities).
    """
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    tol = rcond * (s.max() if s.size > 0 else 1.0)
    r = int(np.sum(s > tol))
    if r == 0:
        raise ValueError("Rank(X) = 0 under tolerance; adjust rcond.")
    U_r = U[:, :r]
    s_r = s[:r]
    P = U_r @ U_r.T
    C_plus = (U_r / (s_r**2)) @ U_r.T
    cvals = s_r**2
    return P, C_plus, U_r, cvals

def frob(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord='fro'))

def rel_frob_err(A: np.ndarray, B: np.ndarray, eps: float = 1e-12) -> float:
    return frob(A - B) / (frob(B) + eps)

def centered_targets(n: int) -> np.ndarray:
    H = np.eye(n) - np.ones((n, n)) / n
    return H

# ----------------------------- Data generation ----------------------------- #

def make_toy_data(
    seed: int = 0,
    n: int = 128,
    d: int = 32,
    d_t: int = 32,
    p: int = 16,
    rank: int = 12,
    spectrum_decay: float = 3.0,
    img_noise: float = 0.02,
    txt_noise: float = 0.02,
) -> Dict[str, np.ndarray]:
    rng = set_seed(seed)
    r = min(rank, d, d_t, n)

    U_img = rand_orth(rng, d, r)
    U_txt = rand_orth(rng, d_t, r)

    s_vals = np.logspace(0.0, -spectrum_decay, r)
    S_img = np.diag(s_vals)
    S_txt = np.diag(s_vals)

    G = rng.standard_normal((r, n))

    X  = U_img @ S_img @ G + img_noise * rng.standard_normal((d, n))
    XT = U_txt @ S_txt @ G + txt_noise * rng.standard_normal((d_t, n))

    W0 = rng.standard_normal((p, d))
    WT = rng.standard_normal((p, d_t))

    H = centered_targets(n)
    Y = WT @ XT @ H  # p x n

    P, C_plus, U_r, cvals = projector_and_pinvC_from_X(X, rcond=1e-15)
    return dict(X=X, XT=XT, W0=W0, WT=WT, Y=Y, H=H, P=P, C_plus=C_plus, U_r=U_r, cvals=cvals)

# ----------------------------- Closed forms ----------------------------- #

def closed_form_FT(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, P: np.ndarray, C_plus: np.ndarray) -> np.ndarray:
    d = X.shape[0]
    I = np.eye(d)
    return W0 @ (I - P) + (Y @ X.T) @ C_plus

def closed_form_L2(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[0]
    C = X @ X.T
    K = (Y @ X.T) + lam * W0
    A = C + lam * np.eye(d)
    W_T = np.linalg.solve(A.T, K.T)
    return W_T.T

def closed_form_SD(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float, P: np.ndarray, C_plus: np.ndarray) -> np.ndarray:
    d = X.shape[0]
    I = np.eye(d)
    alpha = 1.0 / (1.0 + lam)
    return W0 @ (I - alpha * P) + alpha * (Y @ X.T) @ C_plus

# ----------------------------- Plain GD (as before) ----------------------------- #

def lipschitz_constants(C_eigmax: float, lam_l2: float, lam_sd: float) -> Dict[str, float]:
    return dict(FT=C_eigmax, L2=C_eigmax + lam_l2, SD=(1.0 + lam_sd) * C_eigmax)

def gd_FT(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, steps: int, gamma: float, W_star: np.ndarray = None):
    W = W0.copy()
    errs = np.zeros(steps)
    for t in range(steps):
        grad = (W @ X - Y) @ X.T
        W -= gamma * grad
        if W_star is not None:
            errs[t] = rel_frob_err(W, W_star)
    return W, errs

def gd_L2(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float, steps: int, gamma: float, W_star: np.ndarray = None):
    W = W0.copy()
    errs = np.zeros(steps)
    for t in range(steps):
        grad = (W @ X - Y) @ X.T + lam * (W - W0)
        W -= gamma * grad
        if W_star is not None:
            errs[t] = rel_frob_err(W, W_star)
    return W, errs

def gd_SD(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float, steps: int, gamma: float, W_star: np.ndarray = None):
    W = W0.copy()
    errs = np.zeros(steps)
    C = X @ X.T
    B = (Y @ X.T) + lam * (W0 @ C)
    for t in range(steps):
        grad = (1.0 + lam) * (W @ C) - B
        W -= gamma * grad
        if W_star is not None:
            errs[t] = rel_frob_err(W, W_star)
    return W, errs

# ----------------------------- NEW: Right-preconditioned GD ----------------------------- #

def gd_FT_precond(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, C_plus: np.ndarray, steps: int = 1, gamma: float = 1.0):
    """
    W <- W - gamma * ((W C - Y X^T) @ C_plus)
    For gamma=1 and steps=1, equals the closed-form FT exactly.
    """
    W = W0.copy()
    C = X @ X.T
    GX = (Y @ X.T)
    for _ in range(steps):
        grad = (W @ C - GX) @ C_plus
        W -= gamma * grad
    return W

def gd_SD_precond(W0: np.ndarray, X: np.ndarray, Y: np.ndarray, lam: float, C_plus: np.ndarray, steps: int = 1, gamma: float = 1.0):
    """
    W <- W - gamma * ( ((1+lam) W C - (Y X^T + lam W0 C)) @ ((1/(1+lam)) C_plus) )
    For gamma=1 and steps=1, equals the closed-form SD exactly.
    """
    W = W0.copy()
    C = X @ X.T
    B = (Y @ X.T) + lam * (W0 @ C)
    M = C_plus / (1.0 + lam)
    for _ in range(steps):
        grad = ((1.0 + lam) * (W @ C) - B) @ M
        W -= gamma * grad
    return W

# ----------------------------- Diagnostics ----------------------------- #

def verify_orthogonality(W: np.ndarray, W0: np.ndarray, P: np.ndarray) -> float:
    d = P.shape[0]
    I = np.eye(d)
    num = frob((W - W0) @ (I - P))
    den = frob(W0 @ (I - P)) + 1e-12
    return num / den

def spectral_shrinkage_L2(W0, W_L2, W_star, U_r, cvals, lam):
    r = U_r.shape[1]
    pred, obs = [], []
    for i in range(r):
        e_i = U_r[:, [i]]
        ci = float(cvals[i])
        lhs = frob((W_L2 - W_star) @ e_i)
        rhs = (lam / (lam + ci)) * frob((W0 - W_star) @ e_i)
        obs.append(lhs); pred.append(rhs)
    pred, obs = np.array(pred), np.array(obs)
    return dict(
        mean_abs_diff=float(np.mean(np.abs(obs - pred))),
        max_abs_diff=float(np.max(np.abs(obs - pred))),
        corr=float(np.corrcoef(pred, obs)[0,1]) if len(pred) > 1 else 1.0
    )

def spectral_shrinkage_SD(W0, W_SD, W_star, U_r, lam):
    r = U_r.shape[1]
    pred, obs = [], []
    fac = 1.0 / (1.0 + lam)
    for i in range(r):
        e_i = U_r[:, [i]]
        lhs = frob((W_SD - W0) @ e_i)
        rhs = fac * frob((W_star - W0) @ e_i)
        obs.append(lhs); pred.append(rhs)
    pred, obs = np.array(pred), np.array(obs)
    return dict(
        mean_abs_diff=float(np.mean(np.abs(obs - pred))),
        max_abs_diff=float(np.max(np.abs(obs - pred))),
        corr=float(np.corrcoef(pred, obs)[0,1]) if len(pred) > 1 else 1.0
    )

# ----------------------------- Run & summarize ----------------------------- #

def run_single_seed(seed: int, lam_l2: float, lam_sd: float, steps_plain: int = 1500, step_factor: float = 0.9) -> Dict[str, Any]:
    data = make_toy_data(seed=seed)
    X, W0, Y = data["X"], data["W0"], data["Y"]
    P, C_plus, U_r, cvals = data["P"], data["C_plus"], data["U_r"], data["cvals"]

    # Closed forms
    W_star = (Y @ X.T) @ C_plus
    W_FT_cf = closed_form_FT(W0, X, Y, P, C_plus)
    W_L2_cf = closed_form_L2(W0, X, Y, lam=lam_l2)
    W_SD_cf = closed_form_SD(W0, X, Y, lam=lam_sd, P=P, C_plus=C_plus)

    # Plain GD step sizes
    C_eigmax = float(cvals.max())
    Ls = lipschitz_constants(C_eigmax, lam_l2=lam_l2, lam_sd=lam_sd)
    gamma_ft = step_factor / Ls["FT"]
    gamma_l2 = step_factor / Ls["L2"]
    gamma_sd = step_factor / Ls["SD"]

    # Plain GD trajectories
    W_FT_gd, errs_ft = gd_FT(W0, X, Y, steps=steps_plain, gamma=gamma_ft, W_star=W_FT_cf)
    W_L2_gd, errs_l2 = gd_L2(W0, X, Y, lam=lam_l2, steps=steps_plain, gamma=gamma_l2, W_star=W_L2_cf)
    W_SD_gd, errs_sd = gd_SD(W0, X, Y, lam=lam_sd, steps=steps_plain, gamma=gamma_sd, W_star=W_SD_cf)

    # Preconditioned GD (1 step, gamma=1)
    W_FT_pgd = gd_FT_precond(W0, X, Y, C_plus=C_plus, steps=1, gamma=1.0)
    W_SD_pgd = gd_SD_precond(W0, X, Y, lam=lam_sd, C_plus=C_plus, steps=1, gamma=1.0)

    # Metrics
    out = dict(
        seed=seed,
        fin_ft_plain=rel_frob_err(W_FT_gd, W_FT_cf),
        fin_l2_plain=rel_frob_err(W_L2_gd, W_L2_cf),
        fin_sd_plain=rel_frob_err(W_SD_gd, W_SD_cf),
        fin_ft_pgd=rel_frob_err(W_FT_pgd, W_FT_cf),
        fin_sd_pgd=rel_frob_err(W_SD_pgd, W_SD_cf),
        ortho_FT=verify_orthogonality(W_FT_cf, W0, P),
        ortho_L2=verify_orthogonality(W_L2_cf, W0, P),
        ortho_SD=verify_orthogonality(W_SD_cf, W0, P),
        L2_spec=spectral_shrinkage_L2(W0, W_L2_cf, W_star, U_r, cvals, lam=lam_l2),
        SD_spec=spectral_shrinkage_SD(W0, W_SD_cf, W_star, U_r, lam=lam_sd),
        errs_ft=errs_ft, errs_l2=errs_l2, errs_sd=errs_sd
    )
    return out

def aggregate_over_seeds(seeds, lam_l2: float, lam_sd: float, steps_plain: int = 1500):
    per_seed = {}
    rows = []
    for s in seeds:
        o = run_single_seed(s, lam_l2=lam_l2, lam_sd=lam_sd, steps_plain=steps_plain)
        per_seed[s] = o
        rows.append(dict(
            seed=s,
            rel_err_FT_plain=o["fin_ft_plain"],
            rel_err_L2_plain=o["fin_l2_plain"],
            rel_err_SD_plain=o["fin_sd_plain"],
            rel_err_FT_pgd=o["fin_ft_pgd"],
            rel_err_SD_pgd=o["fin_sd_pgd"],
            ortho_FT=o["ortho_FT"],
            ortho_L2=o["ortho_L2"],
            ortho_SD=o["ortho_SD"],
            L2_spec_mean_abs=o["L2_spec"]["mean_abs_diff"],
            L2_spec_corr=o["L2_spec"]["corr"],
            SD_spec_mean_abs=o["SD_spec"]["mean_abs_diff"],
            SD_spec_corr=o["SD_spec"]["corr"],
        ))
    return pd.DataFrame(rows), per_seed

def plot_convergence(per_seed, rep_seed: int):
    o = per_seed[rep_seed]
    plt.figure(figsize=(6,4))
    plt.plot(o["errs_ft"], label="FT (plain GD → CF)")
    plt.plot(o["errs_l2"], label="L2-SP (plain GD → CF)")
    plt.plot(o["errs_sd"], label="SD (plain GD → CF)")
    plt.yscale("log"); plt.xlabel("Iteration"); plt.ylabel("Relative Frobenius error")
    plt.title(f"Convergence to closed forms (seed={rep_seed})")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    seeds = [0,1,2,3,4]
    lam_l2, lam_sd = 1.0, 1.0
    steps_plain = 1500

    df, per_seed = aggregate_over_seeds(seeds, lam_l2=lam_l2, lam_sd=lam_sd, steps_plain=steps_plain)

    print("\n=== Summary across seeds (plain GD and preconditioned GD) ===")
    with pd.option_context('display.precision', 3):
        print(df)

    print("\nAverages:")
    with pd.option_context('display.precision', 3):
        print(df.mean(numeric_only=True))

    # Representative convergence plot (plain GD only)
    plot_convergence(per_seed, rep_seed=seeds[0])

    # Save CSV
    df.to_csv("toy_linear_closed_forms_results_precond.csv", index=False)
    print("\nSaved: toy_linear_closed_forms_results_precond.csv")
