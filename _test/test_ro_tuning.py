import numpy as np

from ro_certs.problem import Reg
from sdprlayer.ro_problems import RealProblem, ToyProblem
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration

# N_CALIB = None  # calibrates all
N_CALIB = 1  # calibrates only first (all others are set to zero)

SEED = 0
INIT_NOISE = 0.1


def test_toy_outer(noise=0, verbose=False, plots=False):
    np.random.seed(SEED)
    prob = ToyProblem(n_anchors=5, n_positions=1, d=2, noise=noise, n_calib=N_CALIB)
    constraints = prob.get_constraints()
    decimal = 2 if noise == 0 else 1
    biases = run_calibration(
        prob, constraints, verbose=verbose, plots=plots, init_noise=INIT_NOISE
    )
    np.testing.assert_almost_equal(
        biases,
        prob.biases[: prob.n_calib],
        decimal=decimal,
        err_msg="bias did not converge to ground truth",
    )


def test_ro_outer(noise=0, verbose=False, plots=False):
    np.random.seed(SEED + 1)
    prob = RealProblem(
        n_positions=3,
        d=2,
        n_anchors=4,
        noise=noise,
        n_calib=N_CALIB,
        reg=Reg.CONSTANT_VELOCITY,
    )
    # prob.plot()
    constraints = prob.get_constraints()
    prob.generate_biases()

    options["adam"]["lr"] = 1e-2
    biases = run_calibration(
        prob,
        constraints,
        verbose=verbose,
        plots=plots,
        init_noise=INIT_NOISE,
        options=options,
    )
    decimal = 2 if noise == 0 else 1
    np.testing.assert_almost_equal(
        biases,
        prob.biases_gt[: prob.n_calib],
        decimal=decimal,
        err_msg="bias did not converge to ground truth",
    )


if __name__ == "__main__":
    # test_toy_outer(noise=0, verbose=True, plots=True)
    # test_toy_outer(noise=1e-2, verbose=True, plots=True)

    test_ro_outer(noise=0, verbose=True, plots=True)
    test_ro_outer(noise=1e-2, verbose=True, plots=True)
