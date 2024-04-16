import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from ro_certs.problem import Reg
from ro_certs.utils.plotting_tools import savefig
from sdprlayer.ro_problems import RealProblem, get_dataset_problem
from sdprlayer.ro_tuner import calibrate_for_bias_grid, calibrate_for_sigma_acc_est_grid
from sdprlayer.ro_tuner import options_default as options
from sdprlayer.ro_tuner import run_calibration
from sdprlayer.sdprlayer import SDPRLayer

SEED = 1


def get_biases(prob_small):
    constraints = prob_small.get_constraints()
    options["adam"]["lr"] = 1e-2

    # some sanity checks
    loss_raw = prob_small.get_range_cost(prob_small.trajectory, biases=None)
    loss_gt = prob_small.get_range_cost(
        prob_small.trajectory, biases=prob_small.biases_gt
    )
    assert loss_gt <= loss_raw

    biases, info = run_calibration(
        prob_small,
        constraints,
        verbose=True,
        plots=True,
        options=options,
    )
    return biases, info


def generate_results(prob, biases, appendix=""):
    fig, ax = plt.subplots()
    plt.show(block=False)

    df_data = []
    for i, prob_sub in enumerate(prob.iterate(batch_size=6)):
        assert isinstance(prob_sub, RealProblem)
        constraints = prob_sub.get_constraints()
        optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

        # run localization without removing biases
        Q = prob_sub.build_data_mat(biases_est=None, sigma_acc_est=None)
        X, x1 = optlayer(Q, solver_args=options["solver_args"])

        p1 = prob.get_positions(x1, test=True)
        loss1 = torch.norm(p1 - torch.tensor(prob_sub.trajectory)).item()
        ax.scatter(*p1[:, :2].T, color="C0")
        df_data.append(
            {
                "calibrated": False,
                "loss": loss1,
                "index": i,
                "start time": prob_sub.times[0],
                "X": X.detach().numpy(),
                "estimate": p1.detach().numpy(),
                "gt": prob.trajectory,
            }
        )

        # run localization with biases
        Q = prob_sub.build_data_mat(biases_est=torch.tensor(biases), sigma_acc_est=None)
        X, x2 = optlayer(Q, solver_args=options["solver_args"])

        p2 = prob.get_positions(x2, test=True)
        loss2 = torch.norm(p2 - torch.tensor(prob_sub.trajectory)).item()
        ax.scatter(*p2[:, :2].T, color="C1")
        ax.scatter(*prob_sub.trajectory[:, :2].T, color="k")
        df_data.append(
            {
                "calibrated": True,
                "loss": loss2,
                "index": i,
                "start time": prob_sub.times[0],
                "X": X.detach().numpy(),
                "estimate": p2.detach().numpy(),
                "gt": prob_sub.trajectory,
            }
        )
        print(f"treated until time {prob_sub.times[-1]:.1f}s / {prob.times[-1]:.1f}s")

    ax.scatter([], [], color="C0", label="calib")
    ax.scatter([], [], color="C1", label="no calib")
    ax.legend()
    fname = f"_plots/positions-{appendix}.png"
    savefig(fig, fname)

    df = pd.DataFrame(df_data)
    fig, ax = plt.subplots()
    sns.lineplot(
        df, hue="calibrated", x="start time", y="loss", ax=ax, hue_order=[False, True]
    )
    means = df.groupby("calibrated").loss.median()
    ax.axhline(means[False], color="C0")
    ax.axhline(means[True], color="C1")
    savefig(fig, f"_plots/errors-{appendix}.png")

    fname = f"_plots/df-{appendix}.pkl"
    df.to_pickle(fname)
    print("Saved data as", fname)
    return df


def prepare():
    np.random.seed(SEED)
    prob_small = get_dataset_problem(
        n_positions=5, use_gt=False, reg=Reg.NONE, downsample_mode="middle"
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
    # bias_est = calibrate_for_bias_grid(prob_smal l, plot=True, solver="SDPR")

    biases = get_biases(prob_small)
    print("done")
    return biases


def setup():
    np.random.seed(SEED)
    prob_small = get_dataset_problem(
        n_positions=5,
        use_gt=False,
        reg=reg,
        downsample_mode=bias_method,
        add_bias=False,
    )
    p = torch.tensor([1])
    Q = prob_small.build_data_mat(torch.tensor(p), verbose=True)
    constraints = prob_small.get_constraints()
    optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

    solver_args = options["solver_args"]
    X, x = optlayer(Q, solver_args=solver_args)
    optlayer.check_rank(X)


if __name__ == "__main__":
    reg = Reg.NONE
    for bias_method in ["uniform", "middle", "gt"]:  # "first"
        np.random.seed(SEED)
        prob_all = get_dataset_problem(
            n_positions=None, use_gt=False, reg=reg, add_bias=False
        )
        if bias_method == "gt":
            biases = prob_all.biases_gt[: prob_all.n_calib]
        else:
            prob_small = get_dataset_problem(
                n_positions=5,
                use_gt=False,
                reg=reg,
                downsample_mode=bias_method,
                add_bias=False,
            )
            # prob_small.set_sigma_acc_est(20.0)
            biases, info = get_biases(prob_small)

        print(f"estimated: {biases[:prob_all.n_calib]}")
        print(f"real:      {prob_all.biases_gt[:prob_all.n_calib]}")

        appendix = f"{bias_method}-{prob_all.regularization}"
        generate_results(prob_all, biases, appendix=appendix)
