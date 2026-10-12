"""Sanity checks for the motion models. Run: python test_filters.py"""
import numpy as np

from filters import CTEKFModel, IMM, KF, ct_matrix, cv_model, make_pi
from simulate import ct_step


def test_ct_matrix_matches_simulator():
    x = np.array([1000.0, -250.0, 500.0, 40.0])
    for w in [0.0, 0.08, -0.12]:
        assert np.allclose(ct_matrix(w, 1.0) @ x, ct_step(x, w, 1.0))


def test_ct_ekf_jacobian_finite_difference():
    model = CTEKFModel(T=1.0, sigma_a=0.0, sigma_w=0.0)
    f = lambda v: model.transition(v)[0]
    eps = 1e-6
    for w in [0.08, -0.12, 0.0]:
        x = np.array([1000.0, -250.0, 500.0, 40.0, w])
        J_num = np.column_stack([(f(x + eps * e) - f(x - eps * e)) / (2 * eps) for e in np.eye(5)])
        assert np.allclose(J_num, model.transition(x)[1], atol=1e-4), w


def test_imm_reduces_to_kf_with_identical_models():
    rng = np.random.default_rng(0)
    model = cv_model(1.0, 5.0)
    x0, P0 = np.array([0.0, 10.0, 0.0, 0.0]), np.eye(4) * 100
    kf = KF(model, x0, P0)
    imm = IMM([model, model], make_pi(2, 0.9, 0.9), x0, P0)
    for _ in range(20):
        z, R = rng.normal(size=2) * 10, np.eye(2) * 100
        assert np.allclose(kf.step(z, R)[0], imm.step(z, R)[0])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
