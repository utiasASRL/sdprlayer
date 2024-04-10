import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch
from cert_tools.linalg_tools import rank_project

from ro_certs.problem import Reg
from sdprlayer import SDPRLayer
from sdprlayer.ro_problems import RealProblem, ToyProblem

torch.set_default_dtype(torch.float64)

EPS_SDP = 1e-7
SEED = 0


def test_inner(prob, constraints, decimal, verbose=False):
    """Make sure we get the (almost) perfect position when using ground truth biases."""
    optlayer = SDPRLayer(
        objective=None,
        n_vars=constraints[0].shape[0],
        constraints=constraints,
        use_dual=False,
    )

    # Use perfect biases.
    values = prob.biases
    p = torch.tensor(values, requires_grad=True)

    sdp_solver_args = {"eps": EPS_SDP, "verbose": verbose}
    X, x_select = optlayer(prob.build_data_mat(p), solver_args=sdp_solver_args)
    x_select = x_select.detach().numpy().flatten()
    positions_est = prob.get_positions(x_select)

    # testing
    x_project, info = rank_project(X.detach().numpy(), 1)
    x_project = x_project.flatten()
    assert info["error eigs"] < 1e-6, "not rank 1!"
    assert abs(x_project[0] - 1) < 1e-7, f"not homogeneous! {x_project[0]}!=1"
    np.testing.assert_almost_equal(
        x_select,
        x_project[1:],
        decimal=6,
        err_msg="selection not equal to projection!",
    )
    return positions_est


def test_toy_inner(noise=0, verbose=False):
    np.random.seed(SEED)
    prob = ToyProblem(n_anchors=5, n_positions=1, d=2, noise=noise)
    constraints = prob.get_constraints()

    decimal = 6 if noise == 0 else abs(round(np.log10(noise)))
    positions_est = test_inner(prob, constraints, decimal=decimal, verbose=True)
    np.testing.assert_almost_equal(
        positions_est.flatten(),
        prob.positions.flatten(),
        decimal=decimal,
        err_msg="did not converge to ground truth",
    )


def test_ro_inner(noise=0, verbose=False):
    np.random.seed(SEED)
    for reg in [Reg.NONE, Reg.CONSTANT_VELOCITY, Reg.ZERO_VELOCITY]:
        prob = RealProblem(n_positions=5, d=2, n_anchors=4, noise=noise, reg=reg)
        constraints = prob.get_constraints()
        prob.generate_biases()

        decimal = 6 if noise == 0 else 2
        positions_est = test_inner(prob, constraints, decimal=decimal, verbose=verbose)

        # below doesn't need to be equal because of the GP prior.
        err = np.linalg.norm(positions_est - prob.positions)
        if reg == Reg.NONE:
            print(
                f"error for {reg} (has to be close to zero (noise={noise:.0e})): {err:.4f}"
            )
            np.testing.assert_almost_equal(
                positions_est.flatten(),
                prob.positions.flatten(),
                decimal=decimal,
                err_msg="did not converge to ground truth",
            )
        else:
            print(f"error for {reg} (doesn't have to be close to zero): {err:.4f}")


def test_Q_zeronoise():
    # test Q for toy data
    np.random.seed(SEED)
    prob = ToyProblem(n_anchors=5, n_positions=1, d=2, noise=0)
    constraints = prob.get_constraints()

    p = torch.tensor(prob.biases, requires_grad=True)
    Q = prob.build_data_mat(p).detach().numpy()
    x_gt = prob.get_x()
    err = abs(x_gt.T @ Q @ x_gt)
    assert err < 1e-10, f"error not zero: {err}"

    # test Q for ro data
    np.random.seed(SEED)
    prob = RealProblem(n_positions=5, d=2, n_anchors=4, noise=0, reg=Reg.NONE)
    x_gt = prob.get_x()

    # try with zero bias
    biases = np.zeros(prob.K)
    prob.generate_biases(biases=biases)
    p = torch.tensor(prob.biases, requires_grad=True)

    # even non-corrected matrix should have zero error!
    Q_old = prob.get_Q_matrix()
    err1 = abs(x_gt.T @ Q_old @ x_gt)
    assert err1 < 1e-10, f"error not zero: {err1}"

    # nonzero bias
    prob.generate_biases()
    p = torch.tensor(prob.biases, requires_grad=True)

    # does not correct for bias
    Q_old = prob.get_Q_matrix()
    err2 = abs(x_gt.T @ Q_old @ x_gt)

    # corrects for bias
    Q = prob.build_data_mat(p).detach().numpy()
    err3 = abs(x_gt.T @ Q @ x_gt)
    assert err3 <= err2
    assert err3 < 1e-10, f"error not zero: {err3}"


def test_ro_Q_matrices(noise=0):
    np.random.seed(SEED)
    prob = RealProblem(n_positions=5, d=2, n_anchors=4, noise=noise)

    # matrices without biases
    R = prob.get_R_matrix()
    Q = R + prob.get_Q_matrix()

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

    test_toy_inner(noise=0)
    test_toy_inner(noise=1e-3)

    test_ro_Q_matrices()

    test_ro_inner(noise=0)
    test_ro_inner(noise=1e-3)
