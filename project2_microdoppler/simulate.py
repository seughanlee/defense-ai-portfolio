"""Physics-based micro-Doppler signal simulator.

Rotor blades use the rigid-rod blade model (V. C. Chen, "The Micro-Doppler Effect in Radar"):
each blade of length L returns L * sinc(x) * exp(-j x), with
x = (4 pi / lambda) * (L / 2) * cos(beta) * cos(Omega t + phi_k).
Bird wings use the same rod integral with an oscillating flap angle instead of rotation.
"""
import numpy as np

from config import DWELL, LAMBDA, N_SAMPLES, PRF, TIP_SPEED_MAX

T_AXIS = np.arange(N_SAMPLES) / PRF
K = 4 * np.pi / LAMBDA


def _sinc(x):
    return np.sinc(x / np.pi)   # numpy sinc is sin(pi x)/(pi x)


def rotor_return(L, omega, n_blades, phi0, cos_beta, t=T_AXIS):
    k = np.arange(n_blades)[:, None]
    x = K * (L / 2) * cos_beta * np.cos(omega * t[None, :] + phi0 + 2 * np.pi * k / n_blades)
    return (_sinc(x) * np.exp(-1j * x)).sum(0)


def wing_return(L, f_flap, amp, theta0, cos_beta, t=T_AXIS):
    theta = theta0 + amp * np.sin(2 * np.pi * f_flap * t)
    x = K * (L / 2) * cos_beta * np.sin(theta)
    return _sinc(x) * np.exp(-1j * x)


def body_phase(v_radial, rng, t=T_AXIS):
    """Bulk motion: Doppler 2 v / lambda (positive = approaching)."""
    return np.exp(1j * (2 * np.pi * (2 * v_radial / LAMBDA) * t + rng.uniform(0, 2 * np.pi)))


def _capped_omega(L, rev_hz):
    return min(2 * np.pi * rev_hz, TIP_SPEED_MAX / L)


def _at_power(component, ratio_db):
    """Scale a micro-motion component to ratio_db relative to the unit-power body return."""
    return component * np.sqrt(10 ** (ratio_db / 10) / np.mean(np.abs(component) ** 2))


def time_axis(dwell=DWELL):
    return np.arange(int(round(PRF * dwell))) / PRF


def multicopter(rng, n_rotors, t=T_AXIS):
    L = rng.uniform(0.08, 0.20)
    omega = _capped_omega(L, rng.uniform(60, 130))
    n_blades = 3 if rng.random() < 0.2 else 2
    cos_beta = np.cos(np.deg2rad(rng.uniform(0, 30)))
    mbr = rng.uniform(-12, 0)
    rotors = sum(rotor_return(L, omega * (1 + rng.normal(0, 0.05)), n_blades,
                              rng.uniform(0, 2 * np.pi), cos_beta, t) for _ in range(n_rotors))
    v = rng.uniform(-15, 15)
    s = (1.0 + _at_power(rotors, mbr)) * body_phase(v, rng, t)
    return s, dict(v_radial=v, tip_speed=L * omega, mbr=mbr)


def fixed_wing_uav(rng, t=T_AXIS):
    L = rng.uniform(0.12, 0.35)
    omega = _capped_omega(L, rng.uniform(70, 150))
    n_blades = int(rng.choice([2, 3]))
    cos_beta = rng.uniform(0.2, 1.0)        # propeller disc is vertical: depends on aspect
    mbr = rng.uniform(-12, -2)
    v = rng.uniform(-40, 40)
    prop = rotor_return(L, omega, n_blades, rng.uniform(0, 2 * np.pi), cos_beta, t)
    s = (1.0 + _at_power(prop, mbr)) * body_phase(v, rng, t)
    return s, dict(v_radial=v, tip_speed=L * omega, mbr=mbr)


def helicopter(rng, t=T_AXIS):
    L = rng.uniform(4.0, 8.0)
    tip = rng.uniform(180, 230)
    n_blades = int(rng.choice([2, 3, 4, 5]))
    cos_beta = np.cos(np.deg2rad(rng.uniform(0, 30)))
    mbr = rng.uniform(-8, 2)
    main = rotor_return(L, tip / L, n_blades, rng.uniform(0, 2 * np.pi), cos_beta, t)

    L_t = rng.uniform(0.5, 1.2)
    tip_t = rng.uniform(180, 230)
    n_t = int(rng.choice([2, 4]))
    tail = rotor_return(L_t, tip_t / L_t, n_t, rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 1.0), t)

    v = rng.uniform(-60, 60)
    s = (1.0 + _at_power(main, mbr) + _at_power(tail, mbr + rng.uniform(-10, -3))) * body_phase(v, rng, t)
    return s, dict(v_radial=v, tip_speed=tip, mbr=mbr)


def bird(rng, t=T_AXIS):
    L = rng.uniform(0.15, 0.5)
    f_flap = rng.uniform(3, 12)
    amp = np.deg2rad(rng.uniform(30, 60))
    theta0 = np.deg2rad(rng.uniform(-10, 10))
    cos_beta = rng.uniform(0.3, 1.0)
    mbr = rng.uniform(-10, 0)
    wings = wing_return(L, f_flap, amp, theta0, cos_beta, t)
    bob = rng.uniform(0.005, 0.03) * np.sin(2 * np.pi * f_flap * t + np.pi)   # body heave [m]
    v = rng.uniform(-20, 20)
    s = (1.0 + _at_power(wings, mbr)) * np.exp(-1j * K * bob) * body_phase(v, rng, t)
    return s, dict(v_radial=v, tip_speed=L * amp * 2 * np.pi * f_flap, mbr=mbr)


GENERATORS = {
    "quadcopter": lambda rng, t=T_AXIS: multicopter(rng, 4, t),
    "hexacopter": lambda rng, t=T_AXIS: multicopter(rng, 6, t),
    "octocopter": lambda rng, t=T_AXIS: multicopter(rng, 8, t),
    "fixed_wing_uav": fixed_wing_uav,
    "helicopter": helicopter,
    "bird": bird,
}


def simulate(target, snr_db, rng, dwell=DWELL):
    """Unit-power target return plus complex white noise at the requested per-sample SNR.
    dwell (observation time, s) defaults to the v1 setting; the random draws do not depend on it."""
    t = T_AXIS if dwell == DWELL else time_axis(dwell)
    s, meta = GENERATORS[target](rng, t)
    s = s / np.sqrt(np.mean(np.abs(s) ** 2))
    n = len(t)
    noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    return s + noise * 10 ** (-snr_db / 20), meta
