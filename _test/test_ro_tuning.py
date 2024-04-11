from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import torch

from ro_certs.problem import Problem, Reg
from sdprlayer import SDPRLayer
from sdprlayer.ro_problems import RealProblem, ToyProblem
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration

# N_CALIB = None  # calibrates all
N_CALIB = 1  # calibrates only first (all others are set to zero)

SEED = 1
INIT_NOISE = 0.1


def get_dataset_problem(n_positions, use_gt=True, gt_noise=0):
    prob_data = Problem.init_from_dataset(
        fname="trial1",
        accumulate=True,
        use_gt=use_gt,
        regularization=Reg.NONE,
        gt_noise=gt_noise,
    )
    if use_gt and gt_noise == 0:
        assert prob_data.calculate_noise_level() == 0
    prob_large = RealProblem.init_from_prob(prob_data, n_calib=1)
    prob_large.DOWNSAMPLE_MODE = "uniform"

    prob = prob_large.get_downsampled_version(number=n_positions)
    if use_gt and gt_noise == 0:
        assert prob.calculate_noise_level() == 0
    prob.add_bias()
    return prob


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

    for prob in [prob1, prob2, prob3, prob4, prob5, prob6]:
        # for prob in [prob1, prob2, prob3]:
        fig, ax = prob.plot(show=False)
        ax.set_title(prob.title)
        # bias_grid = np.arange(-1, 1, step=gridsize)
        # assert bias_grid[0] <= prob.biases_gt[0] <= bias_grid[-1]
        bias_gt = prob.biases_gt[0]
        bias_grid = np.arange(
            bias_gt - 5 * gridsize, bias_gt + 5 * gridsize, step=gridsize
        )
        loss_values = []

        constraints = prob.get_constraints()
        optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

        # Define loss
        def gen_loss(p, **kwargs):
            X, x = optlayer(prob.build_data_mat(p), solver_args=options["solver_args"])
            test = True
            if optlayer.check_rank(X, errors="ignore"):
                # If X is not rank 1 then its structure also isn't right.
                test = False

            positions = prob.get_positions(x, test=test)
            loss = torch.norm(positions - torch.tensor(prob.trajectory))
            return loss, positions

        for b in bias_grid:
            # Set up polynomial parameter tensor
            loss, __ = gen_loss(torch.tensor([b]))
            loss_values.append(loss)

        best_bias = bias_grid[np.argmin(loss_values)]

        fig, ax = plt.subplots()
        ax.scatter(bias_grid, loss_values)
        ax.axvline(best_bias)
        ax.set_xlabel("bias")
        ax.set_ylabel("loss")
        ax.set_title(prob.title)
        plt.show(block=False)

        if abs(best_bias - prob.biases_gt[0]) > gridsize + 1e-10:
            print(f"failed for '{prob.title}'")
        else:
            print(f"{prob.title} passed.")
    return


if __name__ == "__main__":
    test_ro_bias_calib()

    test_toy_outer(noise=0, verbose=True, plots=True)
    test_toy_outer(noise=1e-2, verbose=True, plots=True)

    test_ro_outer(noise=0, verbose=True, plots=True)
    test_ro_outer(noise=1e-3, verbose=True, plots=True)
