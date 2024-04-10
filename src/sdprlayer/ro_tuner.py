import time

import matplotlib.pylab as plt
import numpy as np
import torch

from sdprlayer.sdprlayer import SDPRLayer

options_default = dict(
    max_iter=2000,
    grad_norm_tol=1e-7,
    adam=dict(lr=1e-4),
    solver_args=dict(eps=1e-8, solve_method="SCS"),
)


def update_loss_plot(ax, idx, loss):
    ax.scatter(idx, loss, color="k", s=1)


def update_err_plot(ax, idx, errors, labels=False):
    for i, b in enumerate(errors):
        label = f"b{i}" if labels else None
        ax.scatter(idx, b, color=f"C{i}", s=1, label=label)


def update_pos_plot(ax, pos):
    if pos.shape[1] >= 1:
        ax.plot(*pos[:2, :], marker="o")
    else:
        ax.scatter(*pos[:2, :], s=1)


def setup_error_plots(target_loss):
    fig, (ax_loss, ax_err) = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)
    ax_loss.set_yscale("log")
    ax_err.set_yscale("log")
    ax_loss.set_title("position error")
    ax_err.set_title("bias error")
    ax_loss.axhline(target_loss, color="k", label=f"target: {target_loss:.2e}")
    return fig, ax_loss, ax_err


def setup_position_plot(prob):
    fig_pos, ax_pos = plt.subplots()
    if prob.trajectory.shape[0] > 1:
        ax_pos.plot(*prob.trajectory[:, : prob.d].T, color="k", marker="o")
    else:
        ax_pos.scatter(*prob.trajectory[:, : prob.d].T, color="k")
    ax_pos.scatter(*prob.anchors[:, : prob.d].T, marker="x", color="k")
    return fig_pos, ax_pos


def run_calibration(
    prob,
    constraints,
    verbose=False,
    init_noise=1e-3,
    plots=False,
    options=options_default,
):
    """Make sure that we converge to the (almost) perfect biases when using
    (almost) perfect distances.
    """
    optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

    # Set up parameter tensor
    bias_init = prob.biases_gt[: prob.n_calib] + np.random.normal(
        scale=init_noise, loc=0, size=prob.n_calib
    )

    def gen_loss(p, **kwargs):
        X, x = optlayer(prob.build_data_mat(p), solver_args=options["solver_args"])
        test = True  # weather or nto to test structure of x. If it's not rank 1 this doesn't usually pass.
        if optlayer.check_rank(X):
            test = False

        positions = prob.get_positions(x, test=test)
        loss = torch.norm(positions - torch.tensor(prob.trajectory))
        return loss, positions

    losses = []
    converged = False

    if prob.BIAS_MODE == "set_to_zero":
        biases = np.zeros(prob.K)
        biases[: prob.n_calib] = prob.biases_gt[: prob.n_calib]
    elif prob.BIAS_MODE == "set_to_gt":
        biases = prob.biases_gt
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
        print("Warning: costs are not the same! This will lead to faulty behavior")

    starting_loss = gen_loss(torch.tensor(bias_init))[0].item()
    target_loss = gen_loss(torch.tensor(prob.biases_gt[: prob.n_calib]))[0].item()
    assert target_loss < starting_loss, "target is worse than init."
    print("target biases:", prob.biases_gt)
    print("target loss:", target_loss)
    if plots:
        fig, ax_loss, ax_err = setup_error_plots(target_loss)
        fig_pos, ax_pos = setup_position_plot(prob)
        update_loss_plot(ax_loss, 0, starting_loss)
        update_err_plot(
            ax_err, 0, np.abs(bias_init - prob.biases_gt[: prob.n_calib]), labels=True
        )
        plt.show(block=False)

    p = torch.tensor(bias_init, requires_grad=True)
    opt = torch.optim.Adam(params=[p], **options["adam"])
    for n_iter in range(1, options["max_iter"]):
        opt.zero_grad()

        # forward pass
        loss, positions = gen_loss(p)
        losses.append(loss.item())

        # compute gradient
        loss.backward(retain_graph=True)

        # check if gradient is close to zero
        p_grad_norm = p.grad.norm(p=torch.inf)
        if p_grad_norm < options["grad_norm_tol"]:
            msg = f"converged in grad after {n_iter} iterations."
            converged = True
        else:
            # perform one optimizer step
            opt.step()

        # print new values etc.
        biases_est = p.detach().numpy()
        errors = np.abs(biases_est - prob.biases_gt[: prob.n_calib])
        if verbose and ((n_iter < 20) or (n_iter % 10 == 0) or converged):
            errors_str = ",".join([f"{e:.2e}" for e in errors])
            print(
                f"{n_iter}: errors: {errors_str}\tgrad norm: {p_grad_norm:.2e}\tloss: {losses[-1]:.2e}"
            )

        if plots and ((n_iter < 20) or (n_iter % 10 == 0)):
            update_err_plot(ax_err, n_iter, errors)
            update_loss_plot(ax_loss, n_iter, loss.item())
            update_pos_plot(ax_pos, positions[:, : prob.d].detach().numpy().T)

        if converged:
            break
    if not converged:
        msg = f"did not converge in {n_iter} iterations"

    if plots:
        timestamp = int(time.time())
        fname = f"_plots/fig_err_{timestamp}.png"
        fig.savefig(
            fname, bbox_inches="tight", pad_inches=0.1, transparent=False, dpi=100
        )
        print(f"saved as {fname}")
    return biases_est
