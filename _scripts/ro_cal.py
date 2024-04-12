import matplotlib.pylab as plt
import numpy as np

from ro_certs.problem import Reg
from sdprlayer.ro_problems import get_dataset_problem
from sdprlayer.ro_tuner import calibrate_for_bias_grid, calibrate_for_sigma_acc_est_grid
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration

SEED = 1


def generate_results(prob_small):
    constraints = prob_small.get_constraints()
    options["adam"]["lr"] = 1e-2

    # some sanity checks
    loss_raw = prob_small.get_range_cost(prob_small.trajectory, biases=None)
    loss_gt = prob_small.get_range_cost(
        prob_small.trajectory, biases=prob_small.biases_gt
    )
    assert loss_gt <= loss_raw

    biases = run_calibration(
        prob_small,
        constraints,
        verbose=True,
        plots=True,
        init_noise=0.1,
        options=options,
    )
    print("done")


if __name__ == "__main__":
    np.random.seed(SEED)
    prob_small = get_dataset_problem(
        n_positions=5, use_gt=False, reg=Reg.CONSTANT_VELOCITY, downsample_mode="middle"
    )

    # sigma_acc_est = calibrate_for_sigma_acc_est_grid(prob_small, plot=True, solver="GN")
    # print("best sigma:", sigma_acc_est)

    prob_small.set_sigma_acc_est(20.0)
    sigma_acc_est = calibrate_for_sigma_acc_est_grid(
        prob_small, plot=True, solver="SDPR", gridsize=0
    )

    # bias_est = calibrate_for_bias_grid(prob_small, plot=False, gridsize=0, solver="GN")
    # bias_est = calibrate_for_bias_grid(prob_small, plot=False, gridsize=0, solver="SDPR")
    bias_est = calibrate_for_bias_grid(
        prob_small, plot=True, gridsize=1e-1, solver="GN"
    )
    # bias_est = calibrate_for_bias_grid(prob_small, plot=True, solver="SDPR")

    generate_results(prob_small)
    print("done")
