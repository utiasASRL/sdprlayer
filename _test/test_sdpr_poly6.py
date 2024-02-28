import os
import unittest

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch

from cert_tools.eopt_solvers import solve_eopt, opts_cut_dflt
from sdprlayer import SDPRLayer

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


class testSDPRPoly6(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(testSDPRPoly6, self).__init__(*args, **kwargs)


def get_prob_data():
    # Define polynomial
    p_vals = np.array(
        [5.0000, 1.3167 * 2, -1.4481 * 3, 0 * 4, 0.2685 * 3, -0.0667 * 2, 0.0389]
    )

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

    # Dual optimal
    mults = -np.array([[-3.1937], [2.5759], [-0.0562], [0.8318]])

    return dict(p_vals=p_vals, constraints=constraints, x_cand=x_cand, opt_mults=mults)


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


def local_solver(p: torch.Tensor, x_init=0.0):
    # Detach parameters
    p_vals = p.cpu().detach().double().numpy()
    # Simple gradient descent solver
    grad_tol = 1e-10
    max_iters = 200
    n_iter = 0
    alpha = 1e-2
    grad_sq = np.inf
    x = x_init
    while grad_sq > grad_tol and n_iter < max_iters:
        # compute polynomial gradient
        p_deriv = np.array([p * i for i, p in enumerate(p_vals)])[1:]
        grad = np.polyval(p_deriv[::-1], x)
        grad_sq = grad**2
        # Descend
        x = x - alpha * grad
    # Convert to expected vector form
    x_hat = np.array([1, x, x**2, x**3])[:, None]
    return x_hat


def certifier(Q, constraints, x_cand):
    opts = opts_cut_dflt
    method = "cuts"
    _, output = solve_eopt(
        Q=Q, Constraints=constraints, x_cand=x_cand, opts=opts, method=method
    )
    if not output["status"] == "POS_LB":
        raise ValueError("Unable to certify solution")
    # diffcp assumes the form:  H = Q - A*mult
    return output["H"], -output["mults"]


def test_run_sdp():
    """The goal of this script is to shift the optimum of the polynomial
    to a different point by using backpropagtion on rank-1 SDPs"""
    np.random.seed(2)
    # Get data from data function
    data = get_prob_data()
    constraints = data["constraints"]
    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=4, constraints=constraints, use_dual=False)

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)
    sdp_solver_args = {"eps": 1e-9, "verbose": True}
    X, x = optlayer(build_data_mat(p), solver_args=sdp_solver_args)
    x_vals = x.detach().numpy()
    np.testing.assert_allclose(x_vals[0], data["x_cand"][1:, :], rtol=1e-3, atol=1e-3)


def test_prob_sdp(display=False):
    """The goal of this script is to shift the optimum of the polynomial
    to a different point by using backpropagtion on rank-1 SDPs"""
    np.random.seed(2)
    # Get data from data function
    data = get_prob_data()
    constraints = data["constraints"]

    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=4, constraints=constraints)

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
    max_iter = 120
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
        plt.title("")

    # Assert that optimization worked as expected.
    assert n_iter < 120, ValueError("Terminated on max iterations")
    assert loss_val <= 1e-4, ValueError("Loss did not drop to expected value")
    assert np.log10(evr_new) >= 9, ValueError("Solution is not Rank-1")


def test_prob_local(display=False):
    """The goal of this script is to shift the optimum of the polynomial
    to a different point by using backpropagtion on rank-1 SDPs. Forward
    pass is performed with an iterative solver while the backward pass uses
    the computed certificate/multipliers."""
    np.random.seed(2)
    # Get data from data function
    data = get_prob_data()
    constraints = data["constraints"]

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Create SDPR Layer
    sdpr_args = dict(n_vars=4, constraints=constraints, use_dual=True)
    sdpr_args["local_solver"] = local_solver
    sdpr_args["certifier"] = certifier
    sdpr_args["local_args"] = dict(p=p, x_init=-1.5)
    optlayer = SDPRLayer(**sdpr_args)

    # Define loss
    def gen_loss(p_val):
        x_target = -1
        sdp_solver_args = {"eps": 1e-9, "solve_method": "local"}
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

    # Check convergence.
    assert n_iter < 120, ValueError("Terminated on max iterations")
    assert loss_val <= 1e-4, ValueError("Loss did not drop to expected value")


def test_grad_sdp(autograd_test=True, use_dual=True):
    """The goal of this script is to test the dual formulation of the SDPRLayer"""
    # Get data from data function
    data = get_prob_data()
    constraints = data["constraints"]

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Create SDPR Layer
    sdpr_args = dict(n_vars=4, constraints=constraints, use_dual=use_dual)
    optlayer = SDPRLayer(**sdpr_args)

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
        torch.autograd.gradcheck(
            lambda *x: gen_loss(*x, solver_args=sdp_solver_args)[0],
            [p],
            eps=1e-4,
            atol=1e-5,
            rtol=1e-6,
        )

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
        loss, sol = gen_loss(p_val, solver_args=sdp_solver_args)
        loss_curr = loss.detach().numpy().copy()
        delta_loss[i] = loss_curr - loss_init
    grad_num = delta_loss / stepsize
    # check gradients
    np.testing.assert_allclose(grad_computed, grad_num, atol=1e-3, rtol=0)


def test_grad_sdp_mosek(use_dual=True):
    """Test SDPRLayer with MOSEK as the solver"""
    # Get data from data function
    data = get_prob_data()
    constraints = data["constraints"]

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Create SDPR Layer
    sdpr_args = dict(n_vars=4, constraints=constraints, use_dual=use_dual)
    optlayer = SDPRLayer(**sdpr_args)

    # Define loss
    def gen_loss(p_val, **kwargs):
        x_target = -1
        (sol,) = optlayer(build_data_mat(p_val), **kwargs)
        x_val = (sol[1, 0] + sol[0, 1]) / 2
        loss = 1 / 2 * (x_val - x_target) ** 2
        return loss, sol

    # arguments for sdp solver
    mosek_params = {
        "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
        "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-9,
        "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-9,
        "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-9,
        "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-9,
        "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
    }
    sdp_solver_args = {
        "solve_method": "mosek",
        "mosek_params": mosek_params,
        "verbose": True,
    }

    # Check gradient w.r.t. parameter p
    torch.autograd.gradcheck(
        lambda *x: gen_loss(*x, solver_args=sdp_solver_args)[0],
        [p],
        eps=1e-4,
        atol=1e-3,
        rtol=0.0,
    )


def test_grad_local(autograd_test=True):
    """This test function compares the local version of SDPRLayer with the
    SDP version. Local refers to the fact that the forward pass uses a local
    solver and the reverse pass uses the certificate."""
    # Get data from data function
    data = get_prob_data()
    constraints = data["constraints"]

    # Set up polynomial parameter tensor
    p = torch.tensor(data["p_vals"], requires_grad=True)

    # Create SDPR Layer (SDP version)
    sdpr_args = dict(n_vars=4, constraints=constraints, use_dual=True)
    optlayer_sdp = SDPRLayer(**sdpr_args)
    # Create SDPR Layer (Local version)
    sdpr_args["local_solver"] = local_solver
    sdpr_args["certifier"] = certifier
    sdpr_args["local_args"] = dict(p=p, x_init=-1.5)
    optlayer_local = SDPRLayer(**sdpr_args)

    # Define loss
    def gen_loss_sdp(p_val, **kwargs):
        x_target = -1
        (sol,) = optlayer_sdp(build_data_mat(p_val), **kwargs)
        x_val = (sol[1, 0] + sol[0, 1]) / 2
        loss = 1 / 2 * (x_val - x_target) ** 2
        return loss, sol

    def gen_loss_local(p_val, **kwargs):
        x_target = -1
        kwargs.update(dict(solver_args=dict(solve_method="local")))
        (sol,) = optlayer_local(build_data_mat(p_val), **kwargs)
        x_val = (sol[1, 0] + sol[0, 1]) / 2
        loss = 1 / 2 * (x_val - x_target) ** 2
        return loss, sol

    # arguments for sdp solver
    sdp_solver_args = {"eps": 1e-9}

    # get optimizer to zero to zero the gradients
    opt = torch.optim.SGD(params=[p], lr=1e-2)

    # SDP VERSION
    # Compute Loss
    opt.zero_grad()
    loss, sol = gen_loss_sdp(p, solver_args=sdp_solver_args)
    loss_sdp = loss.detach().numpy().copy()
    # Compute gradient
    loss.backward()
    grad_sdp = p.grad.numpy().copy()
    # LOCAL SOLVER VERSION
    opt.zero_grad()
    loss, sol = gen_loss_local(p, solver_args=sdp_solver_args)
    loss_local = loss.detach().numpy().copy()
    # Compute gradient
    loss.backward()
    grad_local = p.grad.numpy().copy()

    # Check gradient w.r.t. parameter p
    if autograd_test:
        res = torch.autograd.gradcheck(
            lambda *x: gen_loss_local(*x, solver_args=sdp_solver_args)[0],
            [p],
            eps=1e-5,
            atol=1e-7,
            rtol=1e-3,
        )
        assert res is True
    # Compare with SDP version.
    np.testing.assert_allclose(loss_local, loss_sdp, atol=2e-7, rtol=0)
    np.testing.assert_allclose(grad_local, grad_sdp, atol=2e-6, rtol=0)


if __name__ == "__main__":
    # test_prob_sdp()
    # test_prob_local(display=True)
    # test_grad_sdp()
    # test_grad_sdp_mosek()
    # test_grad_local()
    test_run_sdp()
