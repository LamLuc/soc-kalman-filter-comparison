# Author: Luc Lam Thanh
# Affiliation: Industrial University of Ho Chi Minh City (IUH)
#              Bosch Global Software Technologies Vietnam
# ORCID: 0009-0005-2382-3696

"""Batched SOC estimators (rows = independent runs). All use NOMINAL cell parameters.
x = [SOC, V1, V2]; i > 0 discharge; y = OCV(SOC) - V1 - V2 - R0 i."""
import numpy as np
from model import ocv, docv, ocv_inv, NOM

P0_DEF = np.diag([0.05, 1e-4, 1e-4])

def _AB(p=NOM):
    a1, a2 = np.exp(-1 / p['tau1']), np.exp(-1 / p['tau2'])
    return np.diag([1.0, a1, a2]), np.array([-1 / (3600 * p['Q']), p['R1'] * (1 - a1), p['R2'] * (1 - a2)])

def cc(I, soc0):
    N, n = I.shape
    s = np.empty((N, n)); s[:, 0] = soc0
    s[:, 1:] = soc0[:, None] - np.cumsum(I[:, :-1], axis=1) / (3600 * NOM['Q'])
    return s

def ocv_init(I, Y):
    """SOC from first sample, assuming the pack starts at rest (V1=V2=0)."""
    return ocv_inv(Y[:, 0] + NOM['R0'] * I[:, 0])

def cc_ocv(I, Y, soc0_unused=None):
    return cc(I, ocv_init(I, Y))

def _update(x, P, H, innov, R):
    PH = np.einsum('nij,nj->ni', P, H)
    S = np.einsum('ni,ni->n', H, PH) + R
    K = PH / S[:, None]
    x = x + K * innov[:, None]
    P = P - np.einsum('ni,nj->nij', K, PH)
    return x, 0.5 * (P + np.transpose(P, (0, 2, 1))), K, S

def ekf(I, Y, soc0, q=1e-9, r_mv=10.0, P0=P0_DEF, adapt=None):
    """adapt = (lam, qmin, qmax): innovation-based adaptive Q_soc (Mohamed & Schwarz 1999 style):
    Q_soc,k = clip(K_soc^2 * C_v,k, qmin, qmax), C_v = exponentially-weighted innovation variance."""
    N, n = I.shape; A, B = _AB(); R = (r_mv * 1e-3) ** 2
    x = np.zeros((N, 3)); x[:, 0] = soc0
    P = np.broadcast_to(P0, (N, 3, 3)).copy()
    Qd = np.zeros((N, 3, 3)); Qd[:, 0, 0] = q; Qd[:, 1, 1] = Qd[:, 2, 2] = 1e-6
    Cv = np.full(N, R); est = np.empty((N, n))
    for k in range(n):
        if k:
            x = x @ A.T + I[:, k - 1, None] * B
            P = A @ P @ A.T + Qd
        H = np.stack([docv(x[:, 0]), -np.ones(N), -np.ones(N)], 1)
        innov = Y[:, k] - (ocv(x[:, 0]) - x[:, 1] - x[:, 2] - NOM['R0'] * I[:, k])
        x, P, K, S = _update(x, P, H, innov, R)
        if adapt is not None:
            lam, qmin, qmax = adapt
            Cv = lam * Cv + (1 - lam) * innov**2
            Qd[:, 0, 0] = np.clip(K[:, 0] ** 2 * Cv, qmin, qmax)
        est[:, k] = x[:, 0]
    return est

def ukf(I, Y, soc0, q=1e-9, r_mv=10.0, alpha=1.0, beta=2.0, kappa=0.0, P0=P0_DEF):
    """Linear KF time update (state equation is linear); unscented measurement update.
    Sigma points' SOC is used through ocv(), whose domain is clipped to [0,1]."""
    N, n = I.shape; A, B = _AB(); R = (r_mv * 1e-3) ** 2; L = 3
    lam = alpha**2 * (L + kappa) - L
    Wm = np.full(2 * L + 1, 1 / (2 * (L + lam))); Wc = Wm.copy()
    Wm[0] = lam / (L + lam); Wc[0] = Wm[0] + 1 - alpha**2 + beta
    x = np.zeros((N, 3)); x[:, 0] = soc0
    P = np.broadcast_to(P0, (N, 3, 3)).copy(); Qd = np.diag([q, 1e-6, 1e-6]); est = np.empty((N, n))
    for k in range(n):
        if k:
            x = x @ A.T + I[:, k - 1, None] * B
            P = A @ P @ A.T + Qd
        Sc = np.linalg.cholesky((L + lam) * P + 1e-14 * np.eye(3))
        cols = np.transpose(Sc, (0, 2, 1))
        X = np.concatenate([x[:, None], x[:, None] + cols, x[:, None] - cols], 1)
        Yc = ocv(X[:, :, 0]) - X[:, :, 1] - X[:, :, 2] - NOM['R0'] * I[:, k, None]
        ym = Yc @ Wm; dy = Yc - ym[:, None]; dx = X - x[:, None]
        Pyy = dy**2 @ Wc + R
        Pxy = np.einsum('s,nsi,ns->ni', Wc, dx, dy)
        K = Pxy / Pyy[:, None]
        x = x + K * (Y[:, k] - ym)[:, None]
        P = P - np.einsum('ni,nj->nij', K, K) * Pyy[:, None, None]
        P = 0.5 * (P + np.transpose(P, (0, 2, 1)))
        est[:, k] = x[:, 0]
    return est

def jekf(I, Y, soc0, q=1e-9, r_mv=10.0, q_r0=1e-10, q_c=1e-10, P0=P0_DEF):
    """Joint EKF: x = [SOC, V1, V2, R0, c] with c = Q_nom/Q_true; R0 and c follow random walks.
    Serves the same purpose as Plett's dual EKF (online R0 and capacity) in a single filter."""
    N, n = I.shape; R = (r_mv * 1e-3) ** 2; p = NOM
    a1, a2 = np.exp(-1 / p['tau1']), np.exp(-1 / p['tau2']); g = 1 / (3600 * p['Q'])
    x = np.zeros((N, 5)); x[:, 0] = soc0; x[:, 3] = p['R0']; x[:, 4] = 1.0
    P = np.zeros((N, 5, 5)); P[:, :3, :3] = P0; P[:, 3, 3] = (0.5 * p['R0']) ** 2; P[:, 4, 4] = 0.15**2
    Qd = np.diag([q, 1e-6, 1e-6, q_r0, q_c]); est = np.empty((N, n))
    F = np.broadcast_to(np.diag([1.0, a1, a2, 1.0, 1.0]), (N, 5, 5)).copy()
    for k in range(n):
        if k:
            i = I[:, k - 1]
            F[:, 0, 4] = -g * i
            x = np.stack([x[:, 0] - g * x[:, 4] * i, a1 * x[:, 1] + p['R1'] * (1 - a1) * i,
                          a2 * x[:, 2] + p['R2'] * (1 - a2) * i, x[:, 3], x[:, 4]], 1)
            P = F @ P @ np.transpose(F, (0, 2, 1)) + Qd
        i = I[:, k]
        H = np.stack([docv(x[:, 0]), -np.ones(N), -np.ones(N), -i, np.zeros(N)], 1)
        innov = Y[:, k] - (ocv(x[:, 0]) - x[:, 1] - x[:, 2] - x[:, 3] * i)
        x, P, _, _ = _update(x, P, H, innov, R)
        x[:, 3] = np.clip(x[:, 3], 1e-3, 0.1); x[:, 4] = np.clip(x[:, 4], 0.7, 1.6)
        est[:, k] = x[:, 0]
    return est

def metrics(est, S, valid, warm=300):
    """Per-run RMSE, max|e| (percentage points) over valid samples after `warm` s."""
    e = (est - S) * 100
    m = valid.copy(); m[:, :warm] = False
    cnt = m.sum(1)
    if (cnt == 0).any():
        raise ValueError('evaluation window shorter than warm-up for some runs')
    rmse = np.sqrt((np.where(m, e, 0) ** 2).sum(1) / cnt)
    mx = np.where(m, np.abs(e), 0).max(1)
    return rmse, mx

def tconv(est, S, thr):
    """Time [s] after the last sample with |e| > thr (pct points). inf if it never settles."""
    e = np.abs(est - S) * 100 > thr
    out = np.zeros(e.shape[0])
    for j, r in enumerate(e):
        idx = np.flatnonzero(r)
        out[j] = 0.0 if idx.size == 0 else (np.inf if idx[-1] == r.size - 1 else idx[-1] + 1)
    return out
