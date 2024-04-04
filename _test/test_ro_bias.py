import os

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch
from cert_tools.eopt_solvers import opts_cut_dflt, solve_eopt

from sdprlayer import SDPRLayer

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


class RangeOnlyProblem(object):
    NOISE = 1e-3

    def __init__(self, n_landmarks, n_positions, d, noise=NOISE):
        self.n_landmarks = n_landmarks
        self.n_positions = n_positions
        self.d = d

        self.landmarks = np.random.uniform(low=-5, high=5, size=(n_landmarks, d))
        self.biases = np.random.uniform(size=n_landmarks)  # between 0 and 1
        self.positions = np.zeros((n_positions, d))
        self.gt_distances = (
            np.linalg.norm(
                self.landmarks[None, :, :] - self.positions[:, None, :], axis=2
            )
            + self.biases[None, :]
        )  # n_positions x n_landmarks distance matrix
        self.distances = self.gt_distances + np.random.normal(
            scale=noise, loc=0, size=self.gt_distances.shape
        )

    def build_data_mat(self, biases):
        Q_tch = torch.zeros((self.d + 2, self.d + 2), dtype=torch.double)

        alphas = (torch.Tensor(self.distances) - biases) ** 2 - torch.Tensor(
            np.linalg.norm(self.landmarks, axis=1) ** 2
        )
        # Structure of Q:
        # sum_i (
        # [ alpha_i**2  2*alpha_i*a_i'  -alpha_i ] 1
        # [    *        4*a_i*a_i'      -2*a_i   ] d
        # [    *             *             1     ] 1
        # )
        landmarks_tensor = torch.Tensor(self.landmarks)
        Q_tch[0, 0] = torch.sum(alphas**2)
        a = 2 * torch.sum(alphas * landmarks_tensor.T, axis=1)
        Q_tch[0, 1 : self.d + 1] = a
        Q_tch[1 : self.d + 1, 0] = a
        Q_tch[0, self.d + 1] = -torch.sum(alphas)
        Q_tch[1 : self.d + 1, 1 : self.d + 1] = torch.sum(
            landmarks_tensor[:, :, None] @ landmarks_tensor[:, None, :],
            axis=0,
        )  # (K x d x 1) @ (K x 1 x d) = (K x d x d)
        Q_tch[1 : self.d + 1, self.d + 1] = torch.sum(landmarks_tensor, axis=0)
        Q_tch[self.d + 1, self.d + 1] = self.n_landmarks
        return Q_tch

    def plot_problem(self):
        fig, ax = plt.subplots()
        ax.scatter(*self.landmarks[:, :2].T)
        ax.scatter(*self.positions[:, :2].T)


def get_prob_data(noise=1e-3):
    n_landmarks = 5
    n_positions = 1
    d = 2
    # Define landmarks
    np.random.seed(0)
    prob = RangeOnlyProblem(
        n_landmarks=n_landmarks, n_positions=n_positions, d=d, noise=noise
    )

    # homogenizing constraint not included here as it is added by the layer
    constraints = []
    # x is of shape [h, x', z]
    A = sp.csc_array((d + 2, d + 2))  # z = ||x||^2
    for i in range(d):
        A[i + 1, i + 1] = 1
    A[0, -1] = -0.5
    A[-1, 0] = -0.5
    constraints += [A]
    return prob, constraints


def local_solver(p: torch.Tensor, x_init=0.0):
    # Detach parameters
    biasess = p.cpu().detach().double().numpy()
    # Simple gradient descent solver
    grad_tol = 1e-10
    max_iters = 200
    n_iter = 0
    alpha = 1e-2
    grad_sq = np.inf
    x = x_init
    while grad_sq > grad_tol and n_iter < max_iters:
        # compute polynomial gradient
        p_deriv = np.array([p * i for i, p in enumerate(biasess)])[1:]
        grad = np.polyval(p_deriv[::-1], x)
        grad_sq = grad**2
        # Descend
        x = x - alpha * grad
    # Convert to expected vector form
    x_hat = np.array([1, x, x**2, x**3])[:, None]
    return x_hat


def certifier(objective, constraints, x_cand):
    opts = opts_cut_dflt
    method = "cuts"
    _, output = solve_eopt(
        Q=objective, Constraints=constraints, x_cand=x_cand, opts=opts, method=method
    )
    if not output["status"] == "POS_LB":
        raise ValueError("Unable to certify solution")
    # diffcp assumes the form:  H = Q - A*mult
    return output["H"], -output["mults"]


def test_inner():
    """Make sure we get the correct position when using the ground truth biases."""
    from cert_tools.linalg_tools import rank_project

    np.random.seed(2)
    # Get data from data function
    prob, constraints = get_prob_data(noise=0)
    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=prob.d + 2, constraints=constraints, use_dual=False)

    # Use perfect biases.
    values = prob.biases
    p = torch.tensor(values, requires_grad=True)
    sdp_solver_args = {"eps": 1e-9, "verbose": True}
    X, x_select = optlayer(prob.build_data_mat(p), solver_args=sdp_solver_args)
    x_select = x_select.detach().numpy().flatten()
    positions_est = x_select[: prob.d]

    # testing
    x_project, info = rank_project(X.detach().numpy(), 1)
    x_project = x_project.flatten()
    assert info["error eigs"] < 1e-8, "not rank 1!"
    assert abs(x_project[0] - 1) < 1e-10, f"not homogeneous! {x_project[0]}!=1"
    np.testing.assert_almost_equal(
        x_select,
        x_project[1:],
        decimal=6,
        err_msg="selection not equal to projection!",
    )
    np.testing.assert_almost_equal(
        positions_est,
        prob.positions.flatten(),
        decimal=6,
        err_msg="did not converge to ground truth",
    )


def test_outer(display=False):
    """Make sure that we converge to the correct biases when using ground truth distances."""
    np.random.seed(2)
    # Get data from data function
    prob, constraints = get_prob_data(noise=0)

    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=4, constraints=constraints)

    # Set up polynomial parameter tensor
    values = prob.biases + np.random.normal(scale=1e-3, loc=0, size=prob.biases.shape)
    p = torch.tensor(values, requires_grad=True)

    # Define loss
    def gen_loss(values, **kwargs):
        sdp_solver_args = {"eps": 1e-9}
        X, x = optlayer(prob.build_data_mat(values), solver_args=sdp_solver_args)
        loss = torch.sum(
            (
                x[1 : 1 + prob.d].reshape((prob.n_positions, prob.d))
                - torch.Tensor(prob.positions)
            )
            ** 2
        )
        return loss, X

    # Define Optimizer
    opt = torch.optim.Adam(params=[p], lr=1e-2)
    # Execute iterations
    losses = []
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
        print(f"iterations: \t{n_iter}")
    # Check the rank of the solution
    X_new = sol.detach().numpy()
    evals_new = np.sort(np.linalg.eigvalsh(X_new))[::-1]
    evr_new = evals_new[0] / evals_new[1]
    if display:
        print(f"new EVR:\t{evr_new}")

    # Assert that optimization worked as expected.
    assert n_iter < 120, "Terminated on max iterations"
    assert loss_val <= 1e-4, "Loss did not drop to expected value"
    assert np.log10(evr_new) >= 9, "Solution is not Rank-1"


if __name__ == "__main__":
    test_inner()
    test_outer()
