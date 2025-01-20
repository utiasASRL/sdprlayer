import os

import matplotlib.cm as cm
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch
from pandas import DataFrame, read_pickle

from sdprlayers import PolyMinLayer, SDPRLayer, SDPRLayerMosek, polyval
from sdprlayers.utils.plot_tools import savefig

torch.set_default_dtype(torch.float64)


class Poly6Example:
    def __init__(self, p_vals=None):
        prob_data = self.get_prob_data(p_vals)
        self.p_vals = prob_data["p_vals"]
        self.constraints = prob_data["constraints"]

    def get_prob_data(self, p_vals=None):
        # Define polynomial
        if p_vals is None:
            p_vals = np.array(
                [
                    5.0000 + 5.0,
                    1.3167 * 2,
                    -1.4481 * 3,
                    0 * 4,
                    0.2685 * 3,
                    -0.0667 * 2,
                    0.0389,
                ]
            )
        # Define Constraints
        constraints = []
        A = sp.csc_array((4, 4))  # x^2 = x*x
        A[2, 0] = 1 / 2
        A[0, 2] = 1 / 2
        A[1, 1] = -1
        constraints += [A]
        A = sp.csc_array((4, 4))  # x^3 = x^2*x
        A[3, 0] = 1
        A[0, 3] = 1
        A[1, 2] = -1
        A[2, 1] = -1
        constraints += [A]
        A = sp.csc_array((4, 4))  # x^3*x = x^2*x^2
        A[3, 1] = 1 / 2
        A[1, 3] = 1 / 2
        A[2, 2] = -1
        constraints += [A]

        # Candidate solution
        x_cand = np.array([[1.0000, -1.4871, 2.2115, -3.2888]]).T

        return dict(p_vals=p_vals, constraints=constraints, x_cand=x_cand)

    def tune_poly_sdpr(
        self,
        x_min_targ=-1.0,
        p_val_targ=5,
        max_iter=2000,
        lr=1e-3,
        verbose=False,
        diff_qcqp=True,
    ):
        # Create SDPR Layer
        optlayer = SDPRLayer(
            n_vars=4, constraints=self.constraints, use_dual=False, diff_qcqp=diff_qcqp
        )

        # Set up polynomial parameter tensor
        p = torch.tensor(self.p_vals, requires_grad=True)

        # Define outer optimization loss
        def gen_loss(poly, **kwargs):
            sdp_solver_args = {"eps": 1e-9}
            sol, x = optlayer(build_data_mat(poly), solver_args=sdp_solver_args)
            if diff_qcqp:
                x_min = x[1, 0]
            else:
                x_min = (sol[1, 0] + sol[0, 1]) / 2
            loss = 1 / 2 * (x_min - x_min_targ) ** 2
            loss += 1 / 2 * (polyval(poly, x_min) - p_val_targ) ** 2

            return loss, sol

        # Define Optimizer
        opt = torch.optim.Adam(params=[p], lr=lr)
        # Execute iterations
        info = []
        n_iter = 0
        loss_val = np.inf
        # Outer loop
        while loss_val > 1e-4 and n_iter < max_iter:
            # Update Loss
            opt.zero_grad()
            loss, sol = gen_loss(p)
            # run optimizer
            loss.backward()
            opt.step()
            # Store info
            loss_val = loss.item()
            x_min = ((sol[1, 0] + sol[0, 1]) / 2).detach().numpy()
            info.append(dict(loss=loss_val, x_min=x_min, poly=p.detach().tolist()))
            n_iter += 1
            if verbose:
                print(f"n_iter:\t{n_iter}\tmin:\t{x_min}\tloss:\t{loss_val}")
                # Check the rank of the solution
                X_new = sol.detach().numpy()
                evals_new = np.sort(np.linalg.eigvalsh(X_new))[::-1]
                evr_new = evals_new[0] / evals_new[1]
                print(f"Eigenvalue Ratio:\t{evr_new}")
        # Return updated polynomial and information
        info = DataFrame(info)
        return p.detach().numpy(), info

    def tune_poly_torch(
        self,
        x_min_targ=-1,
        p_val_targ=5,
        max_iter=2000,
        x_init=-2.0,
        lr=1e-3,
        verbose=False,
        verbose_inner=False,
    ):
        # Set up polynomial parameter tensor
        p = torch.tensor(self.p_vals, requires_grad=True)
        # Inner optimization params
        opt_kwargs_inner = {
            "optimizer": "sgd",
            "lr": 1e-2,
            "grad_sq_tol": 1e-10,
            "max_iter": 10000,
            "verbose": verbose_inner,
        }
        # Define poly opt layer
        layer = PolyMinLayer(opt_params=opt_kwargs_inner)

        # Define outer optimization loss
        def get_loss_outer(poly):
            x_min = layer.forward(poly, x_init)
            loss = 1 / 2 * (x_min - x_min_targ) ** 2
            loss += 1 / 2 * (polyval(poly, x_min) - p_val_targ) ** 2
            return loss, x_min

        # Define Optimizer
        opt = torch.optim.Adam(params=[p], lr=lr)
        # Execute iterations
        info = []
        n_iter = 0
        loss_val = np.inf
        # Outer loop
        while loss_val > 1e-4 and n_iter < max_iter:
            # Update Loss
            opt.zero_grad()
            loss, sol = get_loss_outer(p)
            # run optimizer
            loss.backward()
            opt.step()
            # Store info
            loss_val = np.abs(loss.item())
            x_min = sol.detach().numpy()
            info.append(dict(loss=loss_val, x_min=x_min, poly=p.detach().tolist()))
            n_iter += 1
            if verbose:
                print(f"n_iter:\t{n_iter}\tmin:\t{x_min}\tloss:\t{loss_val}")
            # Start next iteration at current minimum
            x_init = x_min
        # Return updated polynomial
        info = DataFrame(info)
        return p.detach().numpy(), info


def build_data_mat(p):
    Q_tch = torch.zeros((4, 4), dtype=torch.double)
    Q_tch[0, 0] = p[0]
    Q_tch[[1, 0], [0, 1]] = p[1] / 2
    Q_tch[[2, 1, 0], [0, 1, 2]] = p[2] / 3
    Q_tch[[3, 2, 1, 0], [0, 1, 2, 3]] = p[3] / 4
    Q_tch[[3, 2, 1], [1, 2, 3]] = p[4] / 3
    Q_tch[[3, 2], [2, 3]] = p[5] / 2
    Q_tch[3, 3] = p[6]

    return Q_tch


def plot_polynomial(p_vals):
    x = np.linspace(-2.5, 2.5, 100)
    y = np.polyval(p_vals[::-1], x)
    plt.plot(x, y)


def test_rank_maximization():
    """This test checks if its possible increase the rank of the SDP relaxation
    using the returned gradients from the implicit differentiation."""
    np.random.seed(2)
    # Initialize problem
    prob = Poly6Example()
    data = prob.get_prob_data()
    constraints = data["constraints"]

    # Create SDPR Layer
    mosek_params = {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
        "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
    }
    optlayer = SDPRLayerMosek(
        n_vars=4, constraints=constraints[:-1], mosek_params=mosek_params
    )

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # arguments for sdp solver

    # Define loss to tighten the problem
    def gen_loss(p, tighten=True):
        Q = build_data_mat(p)
        Q.retain_grad()
        X, x = optlayer(Q)
        # loss = -torch.trace(X)
        if tighten:
            # loss = torch.norm(S[1:])
            loss = torch.trace(X)
        else:
            loss = -torch.trace(X)
        return loss, X

    # Define Optimizer
    opt = torch.optim.Adam(params=[p], lr=1e-1)
    # Execute iterations
    losses = []
    max_iter = 500
    n_iter = 0
    tight = False
    data = []
    while n_iter < max_iter:
        # Update Loss
        opt.zero_grad()
        loss, X = gen_loss(p, tighten=(not tight))
        # run optimizer
        loss.backward()
        opt.step()
        loss_val = np.abs(loss.item())
        losses.append(loss_val)
        n_iter += 1
        # Compute Rank
        tight, ER = SDPRLayer.check_tightness(X)
        # Get eigenvalues
        e_vals = torch.linalg.svdvals(X)

        # Display iteration information
        print(f"iter:{n_iter}\tloss:\t{losses[-1]},\tER:{ER}\tTight: {tight}")
        data += [
            {
                "ER": ER,
                "tight": tight,
                "e1": e_vals[0].detach().numpy(),
                "e2": e_vals[1].detach().numpy(),
                "loss": loss_val,
            }
        ]

    df = DataFrame(data)
    fig, axs = plt.subplots(3, 1)
    # Map 'tight' boolean values to colors
    # Define a colormap: True to one color, False to another
    colors = df["tight"].map({True: "green", False: "red"})

    # Plot 'loss' using colors based on 'tight'
    axs[0].scatter(df.index, df["loss"], c=colors)
    axs[0].set_yscale("log")
    axs[0].set_title("Loss vs. Iteration")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")

    # Example plot for eigenvalues (or any other values you wish to plot)
    axs[1].semilogy(df["e1"], label="e1")
    axs[1].semilogy(df["e2"], label="e2")
    axs[1].set_title("Eigenvalues vs. Iteration")
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Eigenvalue")
    axs[1].legend()

    # Example plot for ER values
    axs[2].semilogy(df["ER"], color="blue")
    axs[2].set_title("ER vs. Iteration")
    axs[2].set_xlabel("Iteration")
    axs[2].set_ylabel("ER")

    plt.tight_layout()
    plt.show()


# Run functions
def run_analysis(n_iters=5000, x_targ=1.7, p_targ=7.3, lr=1e-2):
    # Define problem
    prob = Poly6Example()
    # Plot polynomial
    plt.figure()
    plot_polynomial(prob.p_vals)
    plt.ylim([-10, 30])
    # Tune polynomial - sdpr
    p_opt_sdpr, info_sdpr = prob.tune_poly_sdpr(
        x_min_targ=x_targ, p_val_targ=p_targ, max_iter=n_iters, lr=lr, verbose=True
    )
    plot_polynomial(p_opt_sdpr)
    # tune polynomial - torch - Global min
    p_opt_th, info_th1 = prob.tune_poly_torch(
        x_min_targ=x_targ,
        p_val_targ=p_targ,
        max_iter=n_iters,
        x_init=-2.0,
        lr=lr,
        verbose=True,
    )
    plot_polynomial(p_opt_th)
    # tune polynomial - torch - local min
    p_opt_th_loc, info_th2 = prob.tune_poly_torch(
        x_min_targ=x_targ,
        p_val_targ=p_targ,
        max_iter=n_iters,
        x_init=2.0,
        lr=lr,
        verbose=True,
    )
    plot_polynomial(p_opt_th_loc)
    plt.plot(x_targ, p_targ, "k+")
    plt.legend(
        ["Init", "SDPR", "Torch (init=-2)", "Torch (init=-2)", "Target Glob Min"]
    )
    plt.title(f"Optimized Polynomial - {n_iters} iterations")
    plt.show()

    # save data
    folder = "_results"
    info_sdpr.to_pickle(os.path.join(folder, "poly6_sdpr.pkl"))
    info_th1.to_pickle(os.path.join(folder, "poly6_th1.pkl"))
    info_th2.to_pickle(os.path.join(folder, "poly6_th2.pkl"))


# POST PROCESSING
def plot_poly_evolution(ax, filename="poly6_sdpr.pkl", step=None, N_pts=10):
    # Load data
    folder = "_results"
    filename = os.path.join(folder, filename)
    info = read_pickle(filename)
    polys = np.array(info["poly"].values)
    xmins = info["x_min"].values
    losses = info["loss"].values
    n_iters = len(losses)
    if step is None:
        step = n_iters // (N_pts - 1)

    # Plot polynomial evolution
    cmap = cm.get_cmap("coolwarm")
    pvals = []
    for ind in range(0, n_iters, step):
        if ind == 0 or ind == n_iters:
            alpha = 1
        else:
            alpha = 0.5
        color = cmap(ind / n_iters)
        x = np.linspace(-2.5, 2.5, 100)
        y = np.polyval(polys[ind][::-1], x)
        ax.plot(x, y, color=color, alpha=alpha)
        # store min location
        pvals += [np.polyval(polys[ind][::-1], xmins[ind])]
        ax.scatter(
            xmins[ind],
            pvals[-1],
            color="black",
            marker=".",
        )  # plot points
    ax.set_ylim([0, 13])
    ax.set_xlabel("x-values")
    # plot minimas
    x_vals = xmins[::step]
    pvals = np.stack(pvals)
    ax.plot(x_vals, pvals, "k:", alpha=0.5, linewidth=1)  # plot lines

    x_targ = 1.7
    p_targ = 7.3
    ax.plot(x_targ, p_targ, "k+", markersize=20)

    print(f"{filename}\tfinal loss:\t{losses[-1]}\tnum_iters:\t{n_iters}")


def plot_final_polys():
    x_targ = 1.7
    p_targ = 7.3
    # Load data
    folder = "_results"
    filename = os.path.join(folder, "poly6_sdpr.pkl")
    info_sdpr = read_pickle(filename)
    p_opt_sdpr = np.array(info_sdpr["poly"].values)[-1]
    filename = os.path.join(folder, "poly6_th1.pkl")
    info_th1 = read_pickle(filename)
    p_opt_th1 = np.array(info_th1["poly"].values)[-1]
    filename = os.path.join(folder, "poly6_th2.pkl")
    info_th2 = read_pickle(filename)
    p_opt_th2 = np.array(info_th2["poly"].values)[-1]

    # Plot
    prob = Poly6Example()
    fig = plt.figure()
    plot_polynomial(prob.p_vals)
    plot_polynomial(p_opt_sdpr)
    plot_polynomial(p_opt_th1)
    plot_polynomial(p_opt_th2)
    plt.plot(x_targ, p_targ, "k+")
    plt.ylim([0, 13])
    plt.legend(
        ["Init", "SDPR", "Torch (init=-2)", "Torch (init=-2)", "Target Glob Min"]
    )
    plt.title(f"Converged Polynomials")
    plt.show()

    return fig


def plot_all(save=False, make_title=False):
    step = 20
    N_pts = 30

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    plot_poly_evolution(axs[0], filename="poly6_th2.pkl", step=20, N_pts=N_pts)
    plot_poly_evolution(axs[1], filename="poly6_th1.pkl", N_pts=N_pts)
    plot_poly_evolution(axs[2], filename="poly6_sdpr.pkl", N_pts=N_pts)
    if make_title:
        axs[0].set_title("Gradient Descent (init at x=2)")
        axs[1].set_title("Gradient Descent (init at x=-2)")
        axs[2].set_title("SDPRLayers (ours)")
    else:
        axs[0].set_title(" ")
        axs[1].set_title(" ")
        axs[2].set_title(" ")
    axs[0].set_ylabel("y-values")
    plt.tight_layout()

    fig2 = plot_final_polys()
    plt.show()

    if save:
        savefig(fig, "poly_opt_evo")
        savefig(fig2, "poly_opt_final")


if __name__ == "__main__":
    # test_poly_torch()
    run_analysis()
    # plot_all(save=True)
    # test_rank_maximization()
