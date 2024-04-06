import numpy as np

from sdprlayer.ro_problems import RealProblem, ToyProblem
from sdprlayer.ro_tuner import run_calibration


def test_toy_outer(noise=0, verbose=False):
    prob = ToyProblem(n_landmarks=5, n_positions=1, d=2, noise=noise)
    constraints = prob.get_constraints()
    decimal = 2 if noise == 0 else 1
    biases = run_calibration(prob, constraints, verbose=verbose)
    np.testing.assert_almost_equal(
        biases,
        prob.biases,
        decimal=decimal,
        err_msg="bias did not converge to ground truth",
    )


def test_ro_outer(noise=0, verbose=False, plots=False):
    prob = RealProblem(n_positions=5, d=2, n_landmarks=4, noise=noise)
    constraints = prob.get_constraints()
    prob.generate_biases()
    decimal = 2 if noise == 0 else 1
    biases = run_calibration(
        prob,
        constraints,
        verbose=verbose,
        init_noise=1e-2,
        plots=plots,
    )
    np.testing.assert_almost_equal(
        biases,
        prob.biases,
        decimal=decimal,
        err_msg="bias did not converge to ground truth",
    )


if __name__ == "__main__":
    test_toy_outer(noise=0, verbose=True)
    test_toy_outer(noise=1e-3, verbose=True)
