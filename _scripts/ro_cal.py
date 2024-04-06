from ro_certs.problem import Problem, Reg
from sdprlayer.ro_problems import RealProblem
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration


def generate_results():
    prob_dataset = Problem.init_from_dataset(
        fname="trial1", accumulate=True, use_gt=False, regularization=Reg.NONE
    )
    prob = RealProblem.init_from_prob(prob_dataset, n_calib=1)
    # prob.generate_biases()

    prob_small = prob.get_downsampled_version(number=4, method="keep-first")
    prob_small.biases = prob_small.get_biases(prob_small.D_noisy_sq, squared=True)
    constraints = prob_small.get_constraints()
    options["adam"]["lr"] = 1e-2

    import matplotlib.pylab as plt

    plt.matshow(prob_small.D_noisy_sq)

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
