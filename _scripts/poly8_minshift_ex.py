import os

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


def get_prob_data():
    # Define polynomial
    p_vals = np.array(
        [5.0000, 1.3167 * 2, -1.4481 * 3, 0 * 4, 0.2685 * 3, -0.0667 * 2, 0.0389]
    )

    Constraints = []
    A = sp.csc_array((4, 4))  # w^2 = 1
    A[0, 0] = 1
    Constraints += [(A, 1.0)]
    A = sp.csc_array((4, 4))  # x^2 = x*x
    A[2, 0] = 1 / 2
    A[0, 2] = 1 / 2
    A[1, 1] = -1
    Constraints += [(A, 0.0)]
    A = sp.csc_array((4, 4))  # x^3 = x^2*x
    A[3, 0] = 1
    A[0, 3] = 1
    A[1, 2] = -1
    A[2, 1] = -1
    Constraints += [(A, 0.0)]
    A = sp.csc_array((4, 4))  # x^3*x = x^2*x^2
    A[3, 1] = 1 / 2
    A[1, 3] = 1 / 2
    A[2, 2] = -1
    Constraints += [(A, 0.0)]

    # Candidate solution
    x_cand = np.array([[1.0000, -1.4871, 2.2115, -3.2888]]).T

    # Dual optimal
    mults = -np.array([[-3.1937], [2.5759], [-0.0562], [0.8318]])

    return dict(p_vals=p_vals, Constraints=Constraints, x_cand=x_cand, opt_mults=mults)


def plot_polynomial(p_vals):
    x = np.linspace(-2.5, 2, 100)
    y = np.polyval(p_vals[::-1], x)
    plt.plot(x, y)


def test_poly8_prob():
    """The goal of this script is to shift the optimum of the polynomial
    to a different point by using backpropagtion on rank-1 SDPs"""
    # Get data from data function
    data = get_prob_data()
    # # Set up CVXPY optimization
    Q = cp.Parameter((4, 4))
    Constraints = data["Constraints"]
    # Set up cvxpy program
    n = 4
    m = len(Constraints)
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    for i, (A, b) in enumerate(Constraints):
        constraints += [cp.trace(A @ X) == b]
    objective = cp.Minimize(cp.trace(Q @ X))
    problem = cp.Problem(objective=objective, constraints=constraints)
    # m = len(Constraints)
    # y = cp.Variable(shape=(m,))
    # As, b = zip(*Constraints)
    # b = np.concatenate([np.atleast_1d(bi) for bi in b])
    # objective = cp.Maximize(b @ y)
    # LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])
    # constraint = LHS << Q
    # problem = cp.Problem(objective, [constraint])

    assert problem.is_dpp()
    # Build convex opt layer
    optlayer = CvxpyLayer(problem, parameters=[Q], variables=[X])

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Define Q tensor from polynomial parameters (there must be a better way to do this)
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

    # Get Solution to initial problem
    Q = build_data_mat(p)
    (sol,) = optlayer(Q)
    X_init = sol.detach().numpy()
    evals_init = np.sort(np.linalg.eigvalsh(X_init))[::-1]
    evr_init = evals_init[0] / evals_init[1]
    print(f"Initial Eigenvalue Ratio:\t{evr_init}")

    # get optimum and set loss to a shifted point
    x_shift = 0.5
    x_target = X_init[0, 1] + x_shift

    # Define loss
    def gen_loss():
        (sol,) = optlayer(build_data_mat(p))
        loss = 1 / 2 * (sol[1, 0] - x_target) ** 2
        return loss, sol

    # Define Optimizer
    opt = torch.optim.Adam(params=[p], lr=1e-2)
    # Execute iterations
    losses = []
    minima = []
    max_iter = 1000
    n_iter = 1
    loss_val = np.inf
    while loss_val > 1e-4 and n_iter < max_iter:
        # Update Loss
        opt.zero_grad()
        loss, sol = gen_loss()
        # run optimizer
        loss.backward(retain_graph=True)
        opt.step()
        loss_val = loss.item()
        losses.append(loss_val)
        x_min = sol.detach().numpy()[0, 1]
        n_iter += 1

        print(f"min:\t{x_min}\tloss:\t{losses[-1]}")

    print(f"ITERATIonS: \t{n_iter}")
    # Check the rank of the solution
    X_new = sol.detach().numpy()
    evals_new = np.sort(np.linalg.eigvalsh(X_new))[::-1]
    evr_new = evals_new[0] / evals_new[1]
    print(f"New Eigenvalue Ratio:\t{evr_new}")

    plt.figure()
    plot_polynomial(p_vals=data["p_vals"])
    plot_polynomial(p_vals=p.detach().numpy())
    plt.axvline(x=X_init[0, 1], color="r", linestyle="--")
    plt.axvline(x=X_new[0, 1], color="b", linestyle="--")
    plt.legend(["initial poly", "new poly", "initial argmin", "new argmin"])
    plt.show()
    print("done")


if __name__ == "__main__":
    test_poly8_prob()
