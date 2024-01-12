import os

import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
from pandas import DataFrame, read_pickle
import scipy.sparse as sp
import torch

from sdprlayer import SDPRLayer
from sdprlayer import PolyMinLayer, polyval

torch.set_default_dtype(torch.float64)


class Poly6Example:
    def __init__(self, p_vals=None):
        prob_data = self.get_prob_data(p_vals)
        self.p_vals = prob_data["p_vals"]
        self.constraints = prob_data["constraints"]

    def get_prob_data(self, p_vals):
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
        A = sp.csc_array((4, 4))  # w^2 = 1
        A[0, 0] = 1
        constraints += [(A, 1.0)]
        A = sp.csc_array((4, 4))  # x^2 = x*x
        A[2, 0] = 1 / 2
        A[0, 2] = 1 / 2
        A[1, 1] = -1
        constraints += [(A, 0.0)]
        A = sp.csc_array((4, 4))  # x^3 = x^2*x
        A[3, 0] = 1
        A[0, 3] = 1
        A[1, 2] = -1
        A[2, 1] = -1
        constraints += [(A, 0.0)]
        A = sp.csc_array((4, 4))  # x^3*x = x^2*x^2
        A[3, 1] = 1 / 2
        A[1, 3] = 1 / 2
        A[2, 2] = -1
        constraints += [(A, 0.0)]

        # Candidate solution
        x_cand = np.array([[1.0000, -1.4871, 2.2115, -3.2888]]).T

        return dict(p_vals=p_vals, constraints=constraints, x_cand=x_cand)

    def tune_poly_sdpr(
        self, x_min_targ=-1.0, p_val_targ=5, max_iter=2000, lr=1e-3, verbose=False
    ):
        # Create SDPR Layer
        optlayer = SDPRLayer(n_vars=4, constraints=self.constraints)

        # Set up polynomial parameter tensor
        p = torch.tensor(self.p_vals, requires_grad=True)

        # Define outer optimization loss
        def gen_loss(poly, **kwargs):
            sdp_solver_args = {"eps": 1e-9}
            (sol,) = optlayer(build_data_mat(poly), solver_args=sdp_solver_args)
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
    folder = "/home/cho/research/sdpr_backprop/_scripts/outputs"
    info_sdpr.to_pickle(os.path.join(folder, "poly6_sdpr.pkl"))
    info_th1.to_pickle(os.path.join(folder, "poly6_th1.pkl"))
    info_th2.to_pickle(os.path.join(folder, "poly6_th2.pkl"))


# POST PROCESSING
def plot_poly_evolution(filename="poly6_sdpr.pkl", label="SDPR", N_pts=10):
    # Load data
    folder = "/home/cho/research/sdpr_backprop/_scripts/outputs"
    filename = os.path.join(folder, filename)
    info = read_pickle(filename)
    polys = np.array(info["poly"].values)
    xmins = info["x_min"].values
    losses = info["loss"].values
    n_iters = len(losses)
    step = n_iters // (N_pts - 1)

    # Plot polynomial evolution
    cmap = cm.get_cmap("coolwarm")
    pvals = []
    plt.figure()
    for ind in range(0, n_iters, step):
        color = cmap(ind / n_iters)
        x = np.linspace(-2.5, 2.5, 100)
        y = np.polyval(polys[ind][::-1], x)
        plt.plot(x, y, color=color, label=None)
        # store min location
        pvals += [np.polyval(polys[ind][::-1], xmins[ind])]
        plt.scatter(
            xmins[ind], pvals[-1], color=color, marker="o", alpha=0.5
        )  # plot points
    plt.ylim([0, 13])
    plt.title("Polynomial Evolution - " + label)
    # plot minimas
    x_vals = xmins[::step]
    pvals = np.stack(pvals)
    plt.plot(x_vals, pvals, "k--", alpha=0.5)  # plot lines

    # Plot motion of minimum
    plt.figure()
    plt.plot(xmins)
    plt.title("Motion of Minimum - " + label)
    plt.ylabel("Minimum")
    plt.xlabel("Iteration")


def plot_final_polys():
    x_targ = 1.7
    p_targ = 7.3
    # Load data
    folder = "/home/cho/research/sdpr_backprop/_scripts/outputs"
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
    plt.figure()
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


def plot_all():
    plot_poly_evolution(filename="poly6_th1.pkl", label="Torch Init=-2", N_pts=30)
    plot_poly_evolution(filename="poly6_th2.pkl", label="Torch Init=2", N_pts=30)
    plot_poly_evolution(filename="poly6_sdpr.pkl", label="SDPR", N_pts=30)
    plot_final_polys()
    plt.show()


if __name__ == "__main__":
    # test_poly_torch()
    # run_analysis()
    plot_all()
