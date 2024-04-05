import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch
from cert_tools.linalg_tools import rank_project

from ro_certs.problem import Reg
from ro_certs.sdp_setup import get_A_list, get_Q_matrix, get_R_matrix
from sdprlayer import SDPRLayer
from sdprlayer.ro_tuner import RealProblem, ToyProblem

torch.set_default_dtype(torch.float64)


def get_ro_data(noise=0, reg=Reg.CONSTANT_VELOCITY) -> RealProblem:
    n_landmarks = 5
    n_positions = 4
    d = 2
    np.random.seed(0)
    prob = RealProblem(
        n_positions=n_positions, d=d, n_landmarks=n_landmarks, noise=noise, reg=reg
    )

    A_0, constraints = get_A_list(prob)
    return prob, constraints


def get_toy_data(noise=0) -> ToyProblem:
    n_landmarks = 5
    n_positions = 1
    d = 2
    # Define landmarks
    np.random.seed(0)
    prob = ToyProblem(
        n_landmarks=n_landmarks, n_positions=n_positions, d=d, noise=noise
    )

    # homogenizing constraint not included here as it is added by the layer
    constraints = []
    # x is of shape [h, x', z]
    A = sp.csc_array((prob.d + 2, prob.d + 2))  # z = ||x||^2
    for i in range(prob.d):
        A[i + 1, i + 1] = 1
    A[0, -1] = -0.5
    A[-1, 0] = -0.5
    constraints += [A]
    return prob, constraints


def test_inner(prob, constraints, decimal, verbose=False):
    """Make sure we get the (almost) perfect position when using ground truth biases."""
    assert isinstance(prob, RealProblem)
    optlayer = SDPRLayer(
        objective=None,
        n_vars=constraints[0].shape[0],
        constraints=constraints,
        use_dual=False,
    )

    # Use perfect biases.
    values = prob.biases
    p = torch.tensor(values, requires_grad=True)

    sdp_solver_args = {"eps": 1e-9, "verbose": verbose}
    X, x_select = optlayer(prob.build_data_mat(p), solver_args=sdp_solver_args)
    x_select = x_select.detach().numpy().flatten()
    positions_est = prob.get_positions(x_select)

    # testing
    x_project, info = rank_project(X.detach().numpy(), 1)
    x_project = x_project.flatten()
    assert info["error eigs"] < 1e-7, "not rank 1!"
    assert abs(x_project[0] - 1) < 1e-7, f"not homogeneous! {x_project[0]}!=1"
    np.testing.assert_almost_equal(
        x_select,
        x_project[1:],
        decimal=6,
        err_msg="selection not equal to projection!",
    )
    np.testing.assert_almost_equal(
        positions_est.flatten(),
        prob.positions.flatten(),
        decimal=decimal,
        err_msg="did not converge to ground truth",
    )


def test_toy_inner(noise=0, verbose=False):
    prob, constraints = get_toy_data(noise=noise)
    decimal = 6 if noise == 0 else abs(round(np.log10(noise)))
    test_inner(prob, constraints, decimal=decimal, verbose=verbose)


def test_ro_inner(noise=0, verbose=False):
    prob, constraints = get_ro_data(noise=noise)
    prob.generate_biases()
    decimal = 6 if noise == 0 else abs(round(np.log10(noise)))
    test_inner(prob, constraints, decimal=decimal, verbose=verbose)


def test_Q_zeronoise():
    # test Q for toy data
    prob, constraints = get_toy_data(noise=0)
    assert isinstance(prob, ToyProblem)

    p = torch.tensor(prob.biases, requires_grad=True)
    Q = prob.build_data_mat(p).detach().numpy()
    x_gt = prob.get_x()
    err = abs(x_gt.T @ Q @ x_gt)
    assert err < 1e-10, f"error not zero: {err}"

    # test Q for ro data
    prob, constraints = get_ro_data(noise=0, reg=Reg.NONE)
    assert isinstance(prob, RealProblem)
    x_gt = prob.get_x()

    # try with zero bias
    biases = np.zeros(prob.K)
    prob.generate_biases(biases=biases)
    p = torch.tensor(prob.biases, requires_grad=True)

    # even non-corrected matrix should have zero error!
    Q_old = get_Q_matrix(prob)
    err1 = abs(x_gt.T @ Q_old @ x_gt)
    assert err1 < 1e-10, f"error not zero: {err1}"

    # nonzero bias
    prob.generate_biases(biases=None)
    p = torch.tensor(prob.biases, requires_grad=True)

    # does not correct for bias
    Q_old = get_Q_matrix(prob)
    err2 = abs(x_gt.T @ Q_old @ x_gt)

    # corrects for bias
    Q = prob.build_data_mat(p).detach().numpy()
    err3 = abs(x_gt.T @ Q @ x_gt)
    assert err3 <= err2
    assert err3 < 1e-10, f"error not zero: {err3}"


def test_outer(prob, constraints, decimal, verbose=False):
    """Make sure that we converge to the (almost) perfect biases when using
    (almost) perfect distances.
    """
    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=4, constraints=constraints)

    # Set up polynomial parameter tensor
    values = prob.biases + np.random.normal(scale=1e-3, loc=0, size=prob.biases.shape)
    p = torch.tensor(values, requires_grad=True)

    # Define loss
    def gen_loss(values, **kwargs):
        sdp_solver_args = {"eps": 1e-9}
        X, x = optlayer(prob.build_data_mat(values), solver_args=sdp_solver_args)
        loss = torch.sum((prob.get_positions(x) - torch.Tensor(prob.positions)) ** 2)
        return loss, X

    opt = torch.optim.Adam(params=[p], lr=1e-3)

    # Execute iterations
    losses = []
    max_iter = 1000
    grad_norm_tol = 1e-7
    loss_val = np.inf
    p_grad_norm = np.inf
    converged = False
    for n_iter in range(max_iter):
        # Update Loss
        opt.zero_grad()
        loss, sol = gen_loss(p)

        # run optimizer
        loss.backward(retain_graph=True)
        p_grad_norm = p.grad.norm(p=2)
        biases = p.detach().numpy().round(3)
        if p_grad_norm < grad_norm_tol:
            msg = f"converged in grad after {n_iter} iterations."
            converged = True
            break
        opt.step()
        loss_val = loss.item()
        losses.append(loss_val)
        n_iter += 1
        if verbose and ((n_iter % 10 == 0) or converged):
            print(
                f"{n_iter}: biases: {biases}\tgrad norm: {p_grad_norm:.2e}\tloss: {losses[-1]:.2e}"
            )
    if not converged:
        msg = f"did not converge in {n_iter} iterations"
    print(msg)

    np.testing.assert_almost_equal(
        biases,
        prob.biases,
        decimal=decimal,
        err_msg="bias did not converge to ground truth",
    )


def test_toy_outer(noise=0, verbose=False):
    prob, constraints = get_toy_data(noise=noise)
    decimal = 2 if noise == 0 else 1
    test_outer(prob, constraints, decimal=decimal, verbose=verbose)


def test_ro_outer(noise=0, verbose=False):
    prob, constraints = get_ro_data(noise=noise)
    prob.generate_biases()
    decimal = 2 if noise == 0 else 1
    test_outer(prob, constraints, noise=decimal, erbose=verbose)


def test_ro_Q_matrices(noise=0):
    np.random.seed(1)
    prob, __ = get_ro_data(noise=noise)
    # matrices without biases
    R = get_R_matrix(prob)
    Q = R + get_Q_matrix(prob)

    # check that zero bias gives the same result.
    prob.generate_biases(biases=np.zeros(prob.K))

    biases = torch.Tensor(prob.biases)
    Q_new = prob.build_data_mat(biases).detach().numpy()
    np.testing.assert_allclose(Q_new, Q.toarray(), atol=1e-6)

    # create problem with biases
    prob.generate_biases()

    # create matrix where we remove the biases
    biases = torch.Tensor(prob.biases)
    Q_new = prob.build_data_mat(biases).detach().numpy()
    np.testing.assert_allclose(Q_new, Q.toarray())


if __name__ == "__main__":
    test_Q_zeronoise()
    # test_toy_inner(noise=0)
    # test_toy_outer(noise=0, verbose=True)

    # test_toy_inner(noise=1e-3)
    # test_toy_outer(noise=1e-3, verbose=True)

    # test_ro_Q_matrices()

    # test_ro_inner(noise=0)
    # test_ro_outer(noise=0)

    # test_ro_inner(noise=1e-3)
    # test_ro_outer(noise=1e-3)
