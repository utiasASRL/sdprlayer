import matplotlib.pylab as plt

from ro_certs.problem import Problem, Reg
from sdprlayer.ro_problems import RealProblem
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration


def generate_results():
    # this one converges after ca. 150 steps.
    prob_dataset = Problem.init_from_dataset(
        fname="trial1",
        accumulate=True,
        use_gt=True,
        gt_noise=1e-2,
        regularization=Reg.NONE,
    )
    # this one converges after ca. 120 steps.
    prob_dataset = Problem.init_from_dataset(
        fname="trial1",
        accumulate=True,
        use_gt=False,
        # gt_noise=1e-2,
        regularization=Reg.NONE,
    )

    prob = RealProblem.init_from_prob(prob_dataset, n_calib=1)
    # prob.generate_biases()

    prob_small = prob.get_downsampled_version(number=5, method="uniform")
    prob_small.biases_gt = prob_small.get_biases(prob_small.D_noisy_sq, squared=True)
    constraints = prob_small.get_constraints()
    options["adam"]["lr"] = 1e-2

    fig, ax = prob_small.plot(show=False)
    print("noise level:", prob_small.calculate_noise_level())

    # some sanity checks
    loss_raw = prob_small.get_range_cost(prob_small.trajectory, biases=None)
    loss_gt = prob_small.get_range_cost(
        prob_small.trajectory, biases=prob_small.biases_gt
    )
    assert loss_gt <= loss_raw

    # plt.matshow(prob_small.D_noisy_sq)

    options["grad_norm_tol"] = 1e-6
    biases = run_calibration(
        prob_small,
        constraints,
        verbose=True,
        plots=True,
        init_noise=0.1,
        options=options,
    )


if __name__ == "__main__":
    generate_results()
