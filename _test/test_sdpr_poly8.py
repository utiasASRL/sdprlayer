import os

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from sdprlayer import SDPRLayer

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


def test_prob_sdp(display=False):
    """The goal of this script is to shift the optimum of the polynomial
    to a different point by using backpropagtion on rank-1 SDPs"""
    np.random.seed(2)
    # Get data from data function
    data = get_prob_data()
    Constraints = data["Constraints"]

    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=4, Constraints=Constraints)

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Define loss
    def gen_loss(p_val, **kwargs):
        x_target = -1
        sdp_solver_args = {"eps": 1e-9}
        (sol,) = optlayer(build_data_mat(p_val), solver_args=sdp_solver_args)
        loss = 1 / 2 * (sol[1, 0] - x_target) ** 2
        return loss, sol

    # Define Optimizer
    opt = torch.optim.Adam(params=[p], lr=1e-2)
    # Execute iterations
    losses = []
    minima = []
    max_iter = 1000
    X_init = None
    n_iter = 0
    loss_val = np.inf
    while loss_val > 1e-4 and n_iter < max_iter:
        # Update Loss
        opt.zero_grad()
        loss, sol = gen_loss(p)
        if n_iter == 0:
            X_init = sol.cpu().detach().numpy()
        # run optimizer
        loss.backward(retain_graph=True)
        opt.step()
        loss_val = loss.item()
        losses.append(loss_val)
        x_min = sol.detach().numpy()[0, 1]
        n_iter += 1
        if display:
            print(f"min:\t{x_min}\tloss:\t{losses[-1]}")
    if display:
        print(f"ITERATIonS: \t{n_iter}")
    # Check the rank of the solution
    X_new = sol.detach().numpy()
    evals_new = np.sort(np.linalg.eigvalsh(X_new))[::-1]
    evr_new = evals_new[0] / evals_new[1]
    if display:
        print(f"New Eigenvalue Ratio:\t{evr_new}")

    if display:
        plt.figure()
        plot_polynomial(p_vals=data["p_vals"])
        plot_polynomial(p_vals=p.detach().numpy())
        plt.axvline(x=X_init[0, 1], color="r", linestyle="--")
        plt.axvline(x=X_new[0, 1], color="b", linestyle="--")
        plt.legend(["initial poly", "new poly", "initial argmin", "new argmin"])
        plt.show()

    # Check that nothing has changed
    assert n_iter == 93, ValueError("Number of iterations was expected to be 93")
    np.testing.assert_almost_equal(loss_val, 9.4637779e-5, decimal=9)
    np.testing.assert_almost_equal(evr_new, 96614772541.3, decimal=1)


def test_grad_num(autograd_test=True, use_dual=True):
    """The goal of this script is to test the dual formulation of the SDPRLayer"""
    # Get data from data function
    data = get_prob_data()
    Constraints = data["Constraints"]

    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=4, Constraints=Constraints, use_dual=True)

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Define loss
    def gen_loss(p_val, **kwargs):
        x_target = -1
        (sol,) = optlayer(build_data_mat(p_val), **kwargs)
        x_val = (sol[1, 0] + sol[0, 1]) / 2
        loss = 1 / 2 * (x_val - x_target) ** 2
        return loss, sol

    # arguments for sdp solver
    sdp_solver_args = {"eps": 1e-9}

    # Check gradient w.r.t. parameter p
    if autograd_test:
        res = torch.autograd.gradcheck(
            lambda *x: gen_loss(*x, solver_args=sdp_solver_args)[0],
            [p],
            eps=1e-4,
            atol=1e-4,
            rtol=1e-3,
        )
        assert res is True

    # Manually compute and compare gradients
    stepsize = 1e-6
    # Compute Loss
    loss, sol = gen_loss(p, solver_args=sdp_solver_args)
    loss_init = loss.detach().numpy().copy()
    # Compute gradient
    loss.backward()
    grad_computed = p.grad.numpy().copy()
    # Get current parameter value
    p_val_init = p.cpu().detach().numpy().copy()
    # Compute gradients
    delta_loss = np.zeros(p_val_init.shape)
    for i in range(len(delta_loss)):
        delta_p = np.zeros(p_val_init.shape)
        delta_p[i] = stepsize
        p_val = torch.tensor(p_val_init + delta_p, requires_grad=True)
        loss, sol = gen_loss(torch.tensor(p_val), solver_args=sdp_solver_args)
        loss_curr = loss.detach().numpy().copy()
        delta_loss[i] = loss_curr - loss_init
    grad_num = delta_loss / stepsize
    # check gradients
    np.testing.assert_allclose(grad_computed, grad_num, atol=1e-3, rtol=0)


if __name__ == "__main__":
    # test_prob_sdp()
    test_grad_num()
