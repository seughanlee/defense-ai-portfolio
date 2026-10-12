"""Motion models, Kalman filter and IMM estimator.

State: [px, vx, py, vy] (4D) or [px, vx, py, vy, omega] (5D, turn-rate augmented).
"""
import numpy as np


def make_H(n):
    H = np.zeros((2, n))
    H[0, 0] = H[1, 2] = 1.0
    return H


def _block_diag(*blocks):
    n = sum(b.shape[0] for b in blocks)
    out = np.zeros((n, n))
    i = 0
    for b in blocks:
        k = b.shape[0]
        out[i:i + k, i:i + k] = b
        i += k
    return out


def _accel_noise(T, sigma_a):
    """Discrete white-noise acceleration covariance for [px, vx, py, vy]."""
    g = np.array([[T**2 / 2], [T]])
    Q1 = sigma_a**2 * (g @ g.T)
    return _block_diag(Q1, Q1)


def ct_matrix(omega, T):
    """Transition matrix of the coordinated-turn model for a known turn rate."""
    if abs(omega) < 1e-9:
        F1 = np.array([[1.0, T], [0, 1.0]])
        return _block_diag(F1, F1)
    s, c, omc = np.sin(omega * T), np.cos(omega * T), _one_minus_cos(omega * T)
    return np.array([
        [1, s / omega, 0, -omc / omega],
        [0, c, 0, -s],
        [0, omc / omega, 1, s / omega],
        [0, s, 0, c],
    ])


def _one_minus_cos(a):
    """1 - cos(a) without cancellation for small a (turn rate near zero)."""
    return 2 * np.sin(a / 2) ** 2


class LinearModel:
    def __init__(self, F, Q):
        self.F, self.Q = F, Q

    def predict(self, x, P):
        return self.F @ x, self.F @ P @ self.F.T + self.Q


def cv_model(T, sigma_a, dim=4, sigma_w_floor=1e-3):
    """Constant velocity. With dim=5 the turn-rate state is reset to zero each step."""
    F, Q = ct_matrix(0.0, T), _accel_noise(T, sigma_a)
    if dim == 5:
        F = _block_diag(F, np.zeros((1, 1)))
        Q = _block_diag(Q, np.array([[sigma_w_floor**2]]))
    return LinearModel(F, Q)


def ct_fixed_model(T, omega, sigma_a):
    """Coordinated turn with a fixed, assumed turn rate (linear)."""
    return LinearModel(ct_matrix(omega, T), _accel_noise(T, sigma_a))


class CTEKFModel:
    """Nearly coordinated turn with unknown turn rate, propagated by EKF linearization."""

    def __init__(self, T, sigma_a, sigma_w):
        self.T = T
        self.Q = _block_diag(_accel_noise(T, sigma_a), np.array([[(sigma_w * T)**2]]))

    def predict(self, x, P):
        x_pred, F = self.transition(x)
        return x_pred, F @ P @ F.T + self.Q

    def transition(self, x):
        """Return f(x) and its Jacobian."""
        T, w = self.T, x[4]
        _, vx, _, vy, _ = x
        F4 = ct_matrix(w, T)
        if abs(w) < 1e-9:
            d = np.array([-T**2 * vy / 2, -T * vy, T**2 * vx / 2, T * vx])
        else:
            s, c, omc = np.sin(w * T), np.cos(w * T), _one_minus_cos(w * T)
            d = np.array([
                (T * c * vx - T * s * vy) / w - (s * vx - omc * vy) / w**2,
                -T * s * vx - T * c * vy,
                (T * s * vx + T * c * vy) / w - (omc * vx + s * vy) / w**2,
                T * c * vx - T * s * vy,
            ])
        F = np.eye(5)
        F[:4, :4] = F4
        F[:4, 4] = d
        return np.append(F4 @ x[:4], w), F


def kf_update(x, P, z, R):
    """Update with Joseph-form covariance; returns measurement likelihood too."""
    H = make_H(len(x))
    y = z - H @ x
    S = H @ P @ H.T + R
    S_inv = np.linalg.inv(S)
    K = P @ H.T @ S_inv
    x = x + K @ y
    I_KH = np.eye(len(x)) - K @ H
    P = I_KH @ P @ I_KH.T + K @ R @ K.T
    lik = np.exp(-0.5 * y @ S_inv @ y) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
    return x, P, max(lik, 1e-300)


class KF:
    def __init__(self, model, x0, P0):
        self.model = model
        self.x, self.P = x0.copy(), P0.copy()

    def step(self, z, R):
        self.x, self.P = self.model.predict(self.x, self.P)
        self.x, self.P, _ = kf_update(self.x, self.P, z, R)
        return self.x, self.P


def make_pi(M, p_first, p_others):
    """Markov transition matrix: model 0 stays with p_first, others with p_others,
    leaving probability split equally among the remaining models."""
    Pi = np.empty((M, M))
    for i in range(M):
        p = p_first if i == 0 else p_others
        Pi[i] = (1 - p) / (M - 1)
        Pi[i, i] = p
    return Pi


class IMM:
    """Interacting Multiple Model estimator. All models share the same state dimension."""

    def __init__(self, models, Pi, x0, P0, mu0=None):
        M = len(models)
        self.models = models
        self.Pi = np.asarray(Pi, float)
        self.mu = np.asarray(mu0, float) if mu0 is not None else np.r_[0.9, np.full(M - 1, 0.1 / (M - 1))]
        self.xs = [x0.copy() for _ in models]
        self.Ps = [P0.copy() for _ in models]

    def step(self, z, R):
        M = len(self.models)

        # 1) Mixing: w[i, j] = P(model i at k-1 | model j at k)
        c = self.Pi.T @ self.mu
        w = self.Pi * self.mu[:, None] / c[None, :]
        x_mix, P_mix = [], []
        for j in range(M):
            xj = sum(w[i, j] * self.xs[i] for i in range(M))
            Pj = sum(w[i, j] * (self.Ps[i] + np.outer(self.xs[i] - xj, self.xs[i] - xj))
                     for i in range(M))
            x_mix.append(xj)
            P_mix.append(Pj)

        # 2) Mode-matched filtering
        lik = np.zeros(M)
        for j, model in enumerate(self.models):
            x, P = model.predict(x_mix[j], P_mix[j])
            self.xs[j], self.Ps[j], lik[j] = kf_update(x, P, z, R)

        # 3) Mode probability update
        self.mu = c * lik
        self.mu /= self.mu.sum()

        # 4) Combination
        x = sum(self.mu[j] * self.xs[j] for j in range(M))
        P = sum(self.mu[j] * (self.Ps[j] + np.outer(self.xs[j] - x, self.xs[j] - x))
                for j in range(M))
        return x, P
