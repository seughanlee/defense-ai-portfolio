"""Shared experiment setup: constants, filter configurations, Monte Carlo loop, metrics."""
import numpy as np

from filters import IMM, KF, CTEKFModel, ct_fixed_model, cv_model, make_pi
from simulate import SCENARIO, measure, polar_to_cart, simulate_truth

T = 1.0
SIGMA_R = 30.0                  # m
SIGMA_AZ = np.deg2rad(0.2)      # rad
SIGMA_A_LOW, SIGMA_A_HIGH = 1.0, 30.0   # m/s^2

# v2 maneuver models (chosen with sweep.py on tuning seed 100, not the evaluation seed)
OMEGA_FIXED = 0.1               # rad/s, ~2.5g at 250 m/s
SIGMA_A_CT_FIXED = 2.0          # absorbs mismatch between assumed and true turn rate
SIGMA_A_CT_EKF = 5.0
SIGMA_W = 0.05                  # rad/s^2, turn-rate random walk
SIGMA_W0 = 0.1                  # rad/s, initial turn-rate std
P_CV, P_MAN = 0.95, 0.95        # CT-EKF IMM model persistence probabilities
P_MAN_FIXED = 0.90              # fixed-CT IMM (transition sweep covered CT-EKF only)

ONSET_WINDOW = 5                # steps after a turn starts
WARMUP = 10                     # steps excluded from peak metrics (initial convergence)


def two_point_init(z0, z1, R1):
    x0 = np.array([z1[0], (z1[0] - z0[0]) / T, z1[1], (z1[1] - z0[1]) / T])
    P0 = np.zeros((4, 4))
    idx_p, idx_v = [0, 2], [1, 3]
    P0[np.ix_(idx_p, idx_p)] = R1
    P0[np.ix_(idx_v, idx_v)] = 2 * R1 / T**2
    P0[np.ix_(idx_p, idx_v)] = R1 / T
    P0[np.ix_(idx_v, idx_p)] = R1 / T
    return x0, P0


def augment(x0, P0):
    """Add a zero-mean turn-rate state."""
    P5 = np.zeros((5, 5))
    P5[:4, :4] = P0
    P5[4, 4] = SIGMA_W0**2
    return np.append(x0, 0.0), P5


def filter_specs(p_cv=P_CV, p_man=P_MAN, sigma_w=SIGMA_W, sigma_a_ct_ekf=SIGMA_A_CT_EKF,
                 omega_fixed=OMEGA_FIXED, sigma_a_ct_fixed=SIGMA_A_CT_FIXED):
    """(name, factory(x0, P0)) pairs. Factories receive the 4D two-point initialization."""
    return [
        ("CV-KF (high q)",
         lambda x, P: KF(cv_model(T, SIGMA_A_HIGH), x, P)),
        ("IMM CV+CV (v1)",
         lambda x, P: IMM([cv_model(T, SIGMA_A_LOW), cv_model(T, SIGMA_A_HIGH)],
                          make_pi(2, 0.95, 0.90), x, P)),
        ("IMM CV+CT fixed",
         lambda x, P: IMM([cv_model(T, SIGMA_A_LOW),
                           ct_fixed_model(T, omega_fixed, sigma_a_ct_fixed),
                           ct_fixed_model(T, -omega_fixed, sigma_a_ct_fixed)],
                          make_pi(3, P_CV, P_MAN_FIXED), x, P)),
        ("IMM CV+CT-EKF",
         lambda x, P: IMM([cv_model(T, SIGMA_A_LOW, dim=5), CTEKFModel(T, sigma_a_ct_ekf, sigma_w)],
                          make_pi(2, p_cv, p_man), *augment(x, P))),
        # High-q CV as a catch-all for maneuvers outside the CT model's envelope (not re-tuned)
        ("IMM CV+CT-EKF+CVhigh",
         lambda x, P: IMM([cv_model(T, SIGMA_A_LOW, dim=5), CTEKFModel(T, sigma_a_ct_ekf, sigma_w),
                           cv_model(T, SIGMA_A_HIGH, dim=5)],
                          make_pi(3, p_cv, p_man), *augment(x, P))),
    ]


def run_once(specs, rng, scenario=SCENARIO):
    truth, maneuvering = simulate_truth(T, scenario)
    zs = measure(truth, SIGMA_R, SIGMA_AZ, rng)
    conv = [polar_to_cart(z, SIGMA_R, SIGMA_AZ) for z in zs]
    x0, P0 = two_point_init(conv[0][0], conv[1][0], conv[1][1])
    filters = [factory(x0, P0) for _, factory in specs]

    n = len(truth) - 2
    est = np.zeros((len(filters), n, 4))
    nees = np.zeros((len(filters), n))
    mu_man = np.full((len(filters), n), np.nan)
    for k in range(2, len(truth)):
        z, R = conv[k]
        for f_idx, f in enumerate(filters):
            x, P = f.step(z, R)
            x, P = x[:4], P[:4, :4]          # evaluate position/velocity only
            e = truth[k] - x
            est[f_idx, k - 2] = x
            nees[f_idx, k - 2] = e @ np.linalg.solve(P, e)
            if isinstance(f, IMM):
                mu_man[f_idx, k - 2] = 1.0 - f.mu[0]

    meas_pos = np.array([c[0] for c in conv[2:]])
    return truth[2:], maneuvering[2:], meas_pos, est, nees, mu_man


def run_mc(specs, runs, seed, scenario=SCENARIO):
    rng = np.random.default_rng(seed)
    pos_se, vel_se, nees_all, mu_all, meas_se = [], [], [], [], []
    for r in range(runs):
        truth, man, meas_pos, est, nees, mu_man = run_once(specs, rng, scenario)
        pos_se.append(((est[:, :, [0, 2]] - truth[None, :, [0, 2]]) ** 2).sum(-1))
        vel_se.append(((est[:, :, [1, 3]] - truth[None, :, [1, 3]]) ** 2).sum(-1))
        meas_se.append(((meas_pos - truth[:, [0, 2]]) ** 2).sum(-1))
        nees_all.append(nees)
        mu_all.append(mu_man)
        if r == 0:
            sample = (truth, meas_pos, est)
    return dict(names=[n for n, _ in specs], runs=runs, seed=seed, man=man,
                t=np.arange(2, 2 + len(man)) * T,
                pos_se=np.array(pos_se), vel_se=np.array(vel_se),   # (runs, filters, time)
                nees=np.array(nees_all), mu=np.array(mu_all), meas_se=np.array(meas_se),
                sample=sample)


def onset_mask(man):
    mask = np.zeros_like(man)
    for k in np.flatnonzero(np.diff(man.astype(int)) == 1) + 1:
        mask[k:k + ONSET_WINDOW] = True
    return mask


def chi2_interval(dof, z=1.959964):
    """Wilson-Hilferty approximation of the two-sided 95% chi-square interval."""
    a = 2 / (9 * dof)
    return tuple(dof * (1 - a + s * z * np.sqrt(a)) ** 3 for s in (-1, 1))


def summarize(res):
    man, onset = res["man"], onset_mask(res["man"])
    avg_nees_t = res["nees"].mean(0)
    rows = []
    for i, name in enumerate(res["names"]):
        p = res["pos_se"][:, i]
        rows.append(dict(
            name=name,
            pos=np.sqrt(p.mean()),
            straight=np.sqrt(p[:, ~man].mean()),
            turn=np.sqrt(p[:, man].mean()),
            onset=np.sqrt(p[:, onset].mean()),
            peak=np.sqrt(p.mean(0))[WARMUP:].max(),
            vel=np.sqrt(res["vel_se"][:, i].mean()),
            nees=avg_nees_t[i].mean(),
            nees_peak=avg_nees_t[i][WARMUP:].max(),
        ))
    return rows
