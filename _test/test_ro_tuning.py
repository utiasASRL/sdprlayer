from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import torch

from ro_certs.gauss_newton import gauss_newton
from ro_certs.problem import Reg
from sdprlayer import SDPRLayer
from sdprlayer.ro_problems import RealProblem, ToyProblem, get_dataset_problem
from sdprlayer.ro_tuner import calibrate_for_bias_grid
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration, unbias_problem

# N_CALIB = None  # calibrates all
N_CALIB = 1  # calibrates only first (all others are set to zero)

SEED = 1
INIT_NOISE = 0.1


def test_toy_outer(noise=0, verbose=False, plots=False):
    np.random.seed(SEED)
    prob = ToyProblem(n_anchors=5, n_positions=1, d=2, noise=noise, n_calib=N_CALIB)
    constraints = prob.get_constraints()
    decimal = 2 if noise == 0 else 1
    # options["adam"]["lr"] = 1e-2
    biases = run_calibration(
        prob,
        constraints,
        verbose=verbose,
        plots=plots,
        init_noise=INIT_NOISE,
        options=options,
    )
    np.testing.assert_almost_equal(
        biases,
        prob.biases_gt[: prob.n_calib],
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
    prob.add_bias()

    options["adam"]["lr"] = 1e-2
    options["grad_norm_tol"] = 1e-5
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


def test_ro_bias_calib(gridsize=1e-3):
    """Verify the cost landscape on a grid of bias values."""
    np.random.seed(SEED)

    n_positions = 5
    prob1 = RealProblem(
        n_positions=n_positions,
        d=2,
        n_anchors=4,
        noise=0,
        n_calib=N_CALIB,
        reg=Reg.CONSTANT_VELOCITY,
    )
    prob2 = deepcopy(prob1)

    prob1.add_bias()
    prob1.title = "first non-zero"

    biases = np.random.uniform(low=-0.1, high=0.1, size=prob2.K)
    prob2.add_bias(biases=biases)
    prob2.title = "all non-zero"

    prob3 = get_dataset_problem(n_positions=n_positions, use_gt=True, gt_noise=0)
    prob3.title = "real-world noise-0"

    prob4 = get_dataset_problem(n_positions=n_positions, use_gt=True, gt_noise=1e-2)
    prob4.title = "real-world noise-1e-2"

    prob5 = get_dataset_problem(n_positions=n_positions, use_gt=True, gt_noise=1e-1)
    prob5.title = "real-world noise-1e-1"

    prob6 = get_dataset_problem(n_positions=n_positions, use_gt=False)
    prob6.title = "real-world"

    for prob in [prob6]:  # [prob1, prob2, prob3, prob4, prob5, prob6]:
        # for prob in [prob1, prob2, prob3]:
        # fig, ax = prob.plot(show=False)
        # ax.set_title(prob.title)
        best_bias = calibrate_for_bias_grid(prob, gridsize, plot=True)
        plt.show(block=False)

        if abs(best_bias - prob.biases_gt[0]) > gridsize + 1e-10:
            print(f"failed for '{prob.title}'")
        else:
            print(f"{prob.title} passed.")
    return


def test_local_vs_sdp():
    """Make sure that Q and the cost returned by the (biased) Gauss Newton solver are the same."""
    # TODO(FD) this doesn't currently speed up the solver.
    # X_est = np.outer(x_est, x_est)
    # x_warm = np.zeros(len(constraints) + 1)  # in Rn
    # s_warm = X_est[np.triu_indices(X_est.shape[0])]  # in Rm
    # y_warm = np.zeros_like(s_warm)  # Rm
    # options["solver_args"]["warm_start"] = [x_warm, y_warm, s_warm]

    p = [1.0]
    for reg in [Reg.NONE, Reg.ZERO_VELOCITY, Reg.CONSTANT_VELOCITY]:
        for downsample_mode in ["first", "middle", "uniform"]:
            np.random.seed(SEED)
            prob = get_dataset_problem(
                n_positions=5,
                use_gt=False,
                reg=reg,
                downsample_mode=downsample_mode,
                add_bias=False,
            )
            assert isinstance(prob, RealProblem)  # for debugging only
            # Remove the bias from distance measurements.
            prob_unbiased = unbias_problem(prob, p)
            theta_est, info = gauss_newton(prob_unbiased.theta, prob_unbiased)

            # Here, we remove the bias in Q.
            x_est = prob.get_x(theta=theta_est)
            Q = prob.build_data_mat(torch.tensor(p), verbose=True)

            fig, ax = plt.subplots()
            im = ax.matshow(Q.detach().numpy())
            plt.colorbar(im)

            cost_est = x_est.T @ Q.detach().numpy() @ x_est

            assert abs(info["cost"] - cost_est) < 1e-8

            constraints = prob.get_constraints()
            optlayer = SDPRLayer(
                n_vars=constraints[0].shape[0], constraints=constraints
            )

            # from cert_tools.sdp_solvers import adjust_Q
            # Q_new, scale, offset = adjust_Q(Q)
            # X, x = optlayer(Q_new, solver_args=options["solver_args"])
            # cost_sdp = torch.trace(Q_new @ X) * scale + offset
            solver_args = options["solver_args"]
            X, x = optlayer(Q, solver_args=solver_args)
            cost_sdp = torch.trace(Q @ X)

            if not optlayer.check_rank(X):
                print(f"{reg}, {downsample_mode}, rank not one")

            err = abs(cost_sdp - cost_est).item() / cost_sdp.item()
            if err > 1e-3:
                print(f"{reg}, {downsample_mode}, error: {err}")

            theta_sdp = x.reshape((prob.N, prob.get_dim() + 1))[:, : prob.get_dim()]
            try:
                np.testing.assert_almost_equal(theta_sdp, theta_est, decimal=3)
            except AssertionError:
                print(f"{reg}, {downsample_mode}, positions:")
                print(theta_est, theta_sdp)


if __name__ == "__main__":
    test_local_vs_sdp()

    # test_ro_bias_calib()

    test_toy_outer(noise=0, verbose=True, plots=True)
    test_toy_outer(noise=1e-2, verbose=True, plots=True)

    test_ro_outer(noise=0, verbose=True, plots=True)
    test_ro_outer(noise=1e-3, verbose=True, plots=True)
