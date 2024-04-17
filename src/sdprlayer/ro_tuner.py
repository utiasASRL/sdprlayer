from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import torch

from ro_certs.gauss_newton import gauss_newton
from sdprlayer.ro_problems import RealProblem
from sdprlayer.sdprlayer import SDPRLayer

options_default = dict(
    max_iter=200,
    grad_norm_tol=1e-4,
    adam=dict(lr=1e-4),
    init_mode="gt",
    init_noise=1e-2,
    solver_args=dict(
        eps=1e-6,
        solve_method="SCS",
        max_iters=int(1e5),
        acceleration_lookback=-int(1e4),
    ),
)


def plot_losses(sigma_grid, loss_values, cost_values, best_sigma, param="sigma"):
    fig, ax = plt.subplots()
    ax.scatter(sigma_grid, loss_values, color="C0", label="position error")
    ax2 = ax.twinx()
    ax2.scatter(sigma_grid, cost_values, color="C1", label="distance error")
    ax.axvline(best_sigma, color="C1", label=f"{param} estimate")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(param)
    ax.set_ylabel("outer loss", color="C0")
    ax2.set_ylabel("distance error", color="C1")
    lims = ax.get_ylim()
    ax.set_ylim(max(lims[0], 1e-3), min(lims[1], 1e3))
    return fig, ax


def unbias_problem(prob, p):
    """Regenrate prob_new.D_noisy_sq, where the biases in p have been removed."""
    prob_new = deepcopy(prob)
    biases = deepcopy(prob_new.biases_gt)
    biases[: prob_new.n_calib] = p
    prob_new.generate_D_biased(biases=-biases)
    prob_new.D_noisy_sq = prob_new.D_biased**2
    return prob_new


def update_cost_plot(ax, idx, loss):
    ax.scatter(idx, loss, color="k", s=1)


def update_error_plot(ax, idx, err):
    ax.scatter(idx, err, color="k", s=1)


def update_bias_plot(ax, idx, errors, labels=False):
    for i, b in enumerate(errors):
        label = f"b{i}" if labels else None
        ax.scatter(idx, b, color=f"C{i}", s=1, label=label)


def update_pos_plot(ax, pos):
    if pos.shape[1] >= 1:
        ax.plot(*pos[:2, :], marker="o")
    else:
        ax.scatter(*pos[:2, :], s=1)


def setup_error_plots(target_cost=None, target_error=None):
    fig, (ax_cost, ax_error, ax_bias) = plt.subplots(1, 3)
    fig.set_size_inches(10, 5)
    ax_cost.set_yscale("log")
    ax_bias.set_yscale("log")
    ax_error.set_yscale("log")
    ax_cost.set_title("outer loss")
    ax_error.set_title("position error")
    ax_bias.set_title("bias error")
    if target_cost is not None:
        ax_cost.axhline(target_cost, color="k", label=f"target: {target_cost:.2e}")
    if target_error is not None:
        ax_error.axhline(target_error, color="k", label=f"target: {target_error:.2e}")
    return fig, ax_cost, ax_error, ax_bias


def setup_position_plot(prob):
    fig_pos, ax_pos = plt.subplots()
    if prob.trajectory.shape[0] > 1:
        ax_pos.plot(*prob.trajectory[:, :2].T, color="k", marker="o")
    else:
        ax_pos.scatter(*prob.trajectory[:, :2].T, color="k")
    ax_pos.scatter(*prob.anchors[:, : prob.d].T, marker="x", color="k")
    return fig_pos, ax_pos


# Define loss
def gen_loss(
    bias, prob: RealProblem, optlayer: SDPRLayer, solver: str, options=options_default
):
    if solver == "SDPR":
        # TODO(FD) setting sigma_acc_est to None on purpose because it's
        # not implemented yet. It's therefore not differentaible! Still
        # good enough for testing / calibrating for now.
        Q = prob.build_data_mat(
            biases_est=bias,
            sigma_acc_est=None,
            verbose=options["solver_args"]["verbose"],
        )
        X, x = optlayer(Q, solver_args=options["solver_args"])
        cost = torch.trace(Q @ X)

        test = True  # weather or not to test structure of x
        if not optlayer.check_rank(X):
            test = False  # not expected to pass test anyways
        positions = prob.get_positions(x, test=test)

        # x_hat, info = rank_project(X, p=1)
        # if x_hat[0, 0] < 0:
        #     x_hat *= -1
        # x_hat = x_hat[1:]
        # positions_hat = prob.get_positions(x_hat, test=False)
        loss = torch.norm(positions - torch.tensor(prob.trajectory))
    elif solver == "GN":

        # first, remove the bias from the measurements.
        if bias is not None:
            prob_new = unbias_problem(prob, bias)
        else:
            prob_new = prob
        theta_est, info = gauss_newton(prob_new.theta, prob=prob_new, verbose=False)
        if info["success"]:
            cost = info["cost"]
            positions = theta_est[:, : prob.d]
            loss = np.linalg.norm(positions - prob_new.trajectory)
        else:
            loss = np.nan
            cost = np.nan
            positions = None
    else:
        raise ValueError(solver)
    return cost, loss, positions


def calibrate_for_bias_grid(
    prob: RealProblem, gridsize=1e-3, plot=False, solver="SDPR"
):
    def gen_loss_here(p):
        options = deepcopy(options_default)
        options["solver_args"]["verbose"] = True
        cost, error, positions = gen_loss(
            p, prob, optlayer, solver=solver, options=options
        )
        try:
            return cost.item(), error.item(), positions
        except:
            return cost, error, positions

    constraints = prob.get_constraints()
    optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

    bias_gt = prob.biases_gt[0]
    if gridsize == 0:
        bias_grid = [bias_gt]
    else:
        bias_grid = np.arange(
            bias_gt - 10 * gridsize, bias_gt + 10 * gridsize, step=gridsize
        )
    error_values = []
    cost_values = []

    if plot:
        fig, ax = prob.plot(show=False)

    for i, b in enumerate(bias_grid):
        cost, error, positions_est = gen_loss_here(torch.tensor([b]))

        if plot:
            prob.plot_estimates(
                [positions_est],
                labels=[f"$b$ {b:.2f}"],
                ax=ax,
                color="C0",
                alpha=0.1 + 0.9 * i / len(bias_grid),
                show=False,
            )

        error_values.append(error)
        cost_values.append(cost)
    best_bias = bias_grid[np.argmin(error_values)]

    if plot:
        fig, ax = plot_losses(
            bias_grid, error_values, cost_values, best_bias, param="bias"
        )
        ax.axvline(bias_gt, color="C2", label="bias gt")
        ax.set_xscale("linear")
    return best_bias


def calibrate_for_sigma_acc_est_grid(
    prob: RealProblem, plot=False, solver="GN", gridsize=0.1
):
    def gen_loss_here(p):
        options = deepcopy(options_default)
        options["solver_args"]["verbose"] = True
        cost, error, positions = gen_loss(
            p, prob, optlayer, solver=solver, options=options
        )
        try:
            return cost.item(), error.item(), positions
        except:
            return cost, error, positions

    if gridsize == 0:
        sigma_grid = [prob.SIGMA_ACC_EST]
    else:
        sigma_grid = np.logspace(-2, 2, 20)

    constraints = prob.get_constraints()
    optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

    loss_values = []
    cost_values = []
    if plot:
        fig, ax = prob.plot(show=False)

    for i, s in enumerate(sigma_grid):
        # Set up polynomial parameter tensor
        prob.set_sigma_acc_est(s)
        loss, cost, positions_est = gen_loss_here(None)
        if plot and (positions_est is not None):
            prob.plot_estimates(
                [positions_est],
                labels=[None],
                ax=ax,
                color="C0",
                alpha=0.1 + 0.9 * i / len(sigma_grid),
                show=False,
            )
        loss_values.append(loss)
        cost_values.append(cost)

    best_sigma = sigma_grid[np.nanargmin(loss_values)]
    if plot and gridsize > 0:
        prob.set_sigma_acc_est(s)
        cost, error, positions_est = gen_loss(bias=None)
        prob.plot_estimates(
            [positions_est],
            labels=["best estimate"],
            ax=ax,
            color="C1",
            show=False,
        )

    if plot:
        fig, ax = plot_losses(
            sigma_grid, loss_values, cost_values, best_sigma, param="sigma"
        )
    return best_sigma


def run_calibration(
    prob,
    constraints,
    verbose=False,
    plots=False,
    options=options_default,
    appendix="",
):
    """Make sure that we converge to the (almost) perfect biases when using
    (almost) perfect distances.
    """

    def gen_loss_here(bias, verbose=False):
        options["solver_args"]["verbose"] = verbose
        cost, error, positions = gen_loss(
            bias,
            prob,
            optlayer,
            solver="SDPR",
            options=options,
        )
        return cost, error, positions

    optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

    # Set up parameter tensor
    if options["init_mode"] == "gt":
        bias_init = prob.biases_gt[: prob.n_calib] + np.random.normal(
            scale=options["init_noise"], loc=0, size=prob.n_calib
        )
    elif options["init_mode"] == "zero":
        bias_init = np.zeros(prob.n_calib)
    costs = []
    converged = False

    if prob.BIAS_MODE == "set_to_zero":
        biases = np.zeros(prob.K)
        biases[: prob.n_calib] = prob.biases_gt[: prob.n_calib]
    elif prob.BIAS_MODE == "set_to_gt":
        biases = deepcopy(prob.biases_gt)
    else:
        raise ValueError(prob.BIAS_MODE)
    Q = prob.build_data_mat(biases_est=torch.tensor(prob.biases_gt[: prob.n_calib]))
    x = prob.get_x()
    cost2 = x.T @ Q.detach().numpy() @ x
    try:
        cost1 = prob.get_range_cost(trajectory=prob.trajectory, biases=biases)
    except NotImplementedError:
        cost1 = cost2
    if abs(cost1 - cost2) > 1e-10:
        print("Warning: costs are not the same! This may lead to faulty behavior")
        print(
            f"cost1: {cost1:.4e}, cost2: {cost2:.4e}, error: {abs(cost1 - cost2):.4e}"
        )

    starting_cost, starting_error, _ = gen_loss_here(
        torch.tensor(bias_init), verbose=True
    )
    target_cost, target_error, _ = gen_loss_here(
        torch.tensor(prob.biases_gt[: prob.n_calib]), verbose=True
    )
    # starting_loss = 1e-3
    # target_loss = 1e-3
    if target_cost > starting_cost:
        print("Warning: target is worse than init.")
    print("target biases:", prob.biases_gt)
    print("target loss:", target_cost)
    if plots:
        fig, ax_cost, ax_error, ax_bias = setup_error_plots(target_cost, target_error)
        fig_pos, ax_pos = setup_position_plot(prob)
        update_cost_plot(ax_cost, 0, starting_cost)
        update_error_plot(ax_error, 0, starting_error)
        update_bias_plot(
            ax_bias, 0, np.abs(bias_init - prob.biases_gt[: prob.n_calib]), labels=True
        )
        plt.show(block=False)

    p = torch.tensor(bias_init, requires_grad=True)
    opt = torch.optim.Adam(params=[p], **options["adam"])
    for n_iter in range(1, options["max_iter"]):
        opt.zero_grad()

        # forward pass
        cost, error, positions = gen_loss_here(p, verbose=False)
        costs.append(cost)

        # compute gradient
        cost.backward(retain_graph=True)

        # perform one optimizer step
        opt.step()

        # print new values etc.
        biases_est = p.detach().numpy()
        errors = np.abs(biases_est - prob.biases_gt[: prob.n_calib])

        if plots:
            update_bias_plot(ax_bias, n_iter, errors)
            update_cost_plot(ax_cost, n_iter, cost.item())
            update_error_plot(ax_error, n_iter, error.item())
            update_pos_plot(ax_pos, positions[:, : prob.d].detach().numpy().T)

        # check if gradient is close to zero
        p_grad_norm = p.grad.norm(p=torch.inf)
        if verbose:
            errors_str = ",".join([f"{e:.2e}" for e in errors])
            print(f"{n_iter}: errors: {errors_str}\t bias0:{biases_est[0]:.2e}", end="")
            print(f"\tgrad norm: {p_grad_norm:.2e}\tloss: {costs[-1]:.2e}")
        if p_grad_norm < options["grad_norm_tol"]:
            msg = f"converged in grad after {n_iter} iterations."
            converged = True
            break
    if not converged:
        msg = f"did not converge in {n_iter} iterations"

    if plots and (appendix != ""):
        # timestamp = int(time.time())
        fname = f"_plots/convergence_{appendix}.png"
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.1, transparent=False, dpi=100
        )
        print(f"saved as {fname}")
    info = {"success": converged, "msg": msg}
    return biases_est, info
