"""Target trajectory and 2D radar measurement simulation."""
import numpy as np

G = 9.81
SPEED = 250.0  # m/s

# (duration [s], turn rate [rad/s]) - straight / 2g left turn / straight / 3g right turn / straight
SCENARIO = [
    (60, 0.0),
    (20, 2 * G / SPEED),
    (40, 0.0),
    (15, -3 * G / SPEED),
    (45, 0.0),
]


def ct_step(x, omega, T):
    """Exact coordinated-turn propagation. State: [px, vx, py, vy]."""
    px, vx, py, vy = x
    if abs(omega) < 1e-9:
        return np.array([px + vx * T, vx, py + vy * T, vy])
    s, c = np.sin(omega * T), np.cos(omega * T)
    return np.array([
        px + (s * vx - (1 - c) * vy) / omega,
        c * vx - s * vy,
        py + ((1 - c) * vx + s * vy) / omega,
        s * vx + c * vy,
    ])


def turn_scenario(g_left, g_right):
    """Same timeline as SCENARIO with different turn load factors."""
    return [(60, 0.0), (20, g_left * G / SPEED), (40, 0.0), (15, -g_right * G / SPEED), (45, 0.0)]


def simulate_truth(T=1.0, scenario=SCENARIO):
    """Return true states (N+1, 4) and maneuver flags (N+1,)."""
    x = np.array([80e3, -SPEED, 30e3, 0.0])
    states, maneuvering = [x], [False]
    for duration, omega in scenario:
        for _ in range(int(round(duration / T))):
            x = ct_step(x, omega, T)
            states.append(x)
            maneuvering.append(omega != 0.0)
    return np.array(states), np.array(maneuvering)


def measure(truth, sigma_r, sigma_az, rng):
    """Radar at origin measuring range and azimuth with Gaussian noise."""
    n = len(truth)
    r = np.hypot(truth[:, 0], truth[:, 2]) + rng.normal(0, sigma_r, n)
    az = np.arctan2(truth[:, 2], truth[:, 0]) + rng.normal(0, sigma_az, n)
    return np.column_stack([r, az])


def polar_to_cart(z, sigma_r, sigma_az):
    """Converted measurement and its linearized covariance."""
    r, az = z
    c, s = np.cos(az), np.sin(az)
    pos = np.array([r * c, r * s])
    J = np.array([[c, -r * s], [s, r * c]])
    R = J @ np.diag([sigma_r**2, sigma_az**2]) @ J.T
    return pos, R
