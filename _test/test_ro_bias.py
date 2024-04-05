import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch
from cert_tools.linalg_tools import rank_project

from ro_certs.problem import Problem, Reg
from ro_certs.sdp_setup import get_A_list, get_Q_matrix, get_R_matrix
from sdprlayer import SDPRLayer

torch.set_default_dtype(torch.float64)

N_LANDMARKS = 5
N_POSITIONS = 1
D = 2


class RangeOnlyProblem(object):
    NOISE = 1e-3

    def __init__(self, n_landmarks, n_positions, d, noise=NOISE):
        self.n_landmarks = n_landmarks
        self.n_positions = n_positions
        self.d = d

        self.landmarks = np.random.uniform(low=-5, high=5, size=(n_landmarks, d))
        self.biases = 0.1 * np.arange(n_landmarks)  # easier for debuggin
        # self.biases = np.random.uniform(size=n_landmarks)  # between 0 and 1
        self.positions = np.zeros((n_positions, d))
        self.gt_distances = np.linalg.norm(
            self.landmarks[None, :, :] - self.positions[:, None, :], axis=2
        )
        # n_positions x n_landmarks distance matrix
        self.biased_distances = self.gt_distances + self.biases[None, :]
        self.biased_distances = self.biased_distances + np.random.normal(
            scale=noise, loc=0, size=self.gt_distances.shape
        )

    def build_data_mat(self, biases):
        """build simple data matrix with bias
        Structure of Q:
        sum_i (
        [ alpha_i**2  2*alpha_i*a_i'  -alpha_i ] 1
        [    *        4*a_i*a_i'      -2*a_i   ] d
        [    *             *             1     ] 1
        )
        where alpha is:
        alpha = (d - b)**2 - a**2 /// d**2 - a**2  # -2*b*d + b**2
        """
        Q_tch = torch.zeros((self.d + 2, self.d + 2), dtype=torch.double)

        landmarks_tensor = torch.Tensor(self.landmarks)  # K x d
        alphas = (
            torch.Tensor(self.biased_distances) - biases
        ) ** 2 - landmarks_tensor.norm(p=2, dim=1) ** 2
        Q_tch[0, 0] = torch.sum(alphas**2)
        a = 2 * torch.sum(alphas * landmarks_tensor.T, axis=1)
        Q_tch[0, 1 : self.d + 1] = a
        Q_tch[1 : self.d + 1, 0] = a
        Q_tch[0, self.d + 1] = -torch.sum(alphas)
        Q_tch[self.d + 1, 0] = -torch.sum(alphas)
        Q_tch[1 : self.d + 1, 1 : self.d + 1] = 4 * torch.sum(
            landmarks_tensor[:, :, None] @ landmarks_tensor[:, None, :],
            axis=0,
        )  # (K x d x 1) @ (K x 1 x d) = (K x d x d)
        Q_tch[1 : self.d + 1, self.d + 1] = -2 * torch.sum(landmarks_tensor, axis=0)
        Q_tch[self.d + 1, 1 : self.d + 1] = -2 * torch.sum(landmarks_tensor, axis=0)
        Q_tch[self.d + 1, self.d + 1] = self.n_landmarks
        return Q_tch

    def plot_problem(self):
        fig, ax = plt.subplots()
        ax.scatter(*self.landmarks[:, :2].T)
        ax.scatter(*self.positions[:, :2].T)


def get_prob_data(noise=1e-3):
    # Define landmarks
    np.random.seed(0)
    prob = RangeOnlyProblem(
        n_landmarks=N_LANDMARKS, n_positions=N_POSITIONS, d=D, noise=noise
    )

    # homogenizing constraint not included here as it is added by the layer
    constraints = []
    # x is of shape [h, x', z]
    A = sp.csc_array((D + 2, D + 2))  # z = ||x||^2
    for i in range(D):
        A[i + 1, i + 1] = 1
    A[0, -1] = -0.5
    A[-1, 0] = -0.5
    constraints += [A]
    return prob, constraints


def test_toy_inner(noise=0, verbose=False):
    """Make sure we get the (almost) perfect position when using ground truth biases."""

    np.random.seed(2)
    # Get data from data function
    prob, constraints = get_prob_data(noise=noise)
    # Create SDPR Layer
    optlayer = SDPRLayer(
        objective=None, n_vars=prob.d + 2, constraints=constraints, use_dual=False
    )

    # Use perfect biases.
    values = prob.biases
    p = torch.tensor(values, requires_grad=True)

    if noise == 0:
        Q = prob.build_data_mat(p).detach().numpy()
        x_gt = np.hstack([1.0, np.zeros(prob.d + 1)])
        assert abs(x_gt.T @ Q @ x_gt) < 1e-10

    sdp_solver_args = {"eps": 1e-9, "verbose": verbose}
    X, x_select = optlayer(prob.build_data_mat(p), solver_args=sdp_solver_args)
    x_select = x_select.detach().numpy().flatten()
    positions_est = x_select[: prob.d]

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
    decimal = 6 if noise == 0 else abs(round(np.log10(noise)))
    np.testing.assert_almost_equal(
        positions_est,
        prob.positions.flatten(),
        decimal=decimal,
        err_msg="did not converge to ground truth",
    )


def test_toy_outer(noise=0, verbose=False):
    """Make sure that we converge to the (almost) perfect biases when using
    (almost) perfect distances.
    """
    np.random.seed(2)
    # Get data from data function
    prob, constraints = get_prob_data(noise=noise)

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

    decimal = 3 if noise == 0 else 1
    np.testing.assert_almost_equal(
        biases,
        prob.biases,
        decimal=decimal,
        err_msg="bias did not converge to ground truth",
    )


def add_biases_to_Q(Q_torch: torch.Tensor, prob: Problem, biases: torch.Tensor):
    """Change Q to be parametrized with unknon biases.

    In this example, we use a weighted norm with weights Sn

    The new strcuture of each Q block should be:
    hn  [ alphan.T @ Sn @ alphan  2*alphan.T @ Sn @ An -alphan.T @ Sn @ 1 ] 1
    xn  [    *                      4*An.T @ Sn @ An   -2 @ An.T @ Sn @ 1 ] d
    zn  [    *                *                            1.t @ Sn @ 1   ] 1
    where:
    alphan = (dn - bn)**2 - an**2, inR(K)
         [a1.T]
    An = [... ]  inR(K x d)
         [aK.T]
    So we just need to change the structure of the last column/row of Q!

    Taking into account the correct order etc, the new row should have the structure:
     x1                                z1          ...   xN   zN    h
    [2*alpha1.T @ S1 @ An     alpha1.T @ S1 @ 1   ...   ...  ...  sum_n alphan.T @ Sn @ alphan]
    """
    distances = torch.Tensor(np.sqrt(prob.D_noisy_sq))
    landmarks = torch.Tensor(prob.anchors)  # n_landmarks x d
    S_all = torch.Tensor(prob.Sig_inv)
    E = np.sum(prob.W)

    # we want alpha to be n_landmarks x n_positions
    alpha = (distances - biases[:, None]) ** 2 - (
        torch.norm(landmarks, p=2, dim=1) ** 2
    )[:, None]

    Q_torch[-1, -1] = 0
    dim = prob.get_dim() + 1
    for n in range(prob.N):
        nnz = prob.W[:, n] != 0
        S = S_all[nnz, :][:, nnz]
        # xi element
        new_xi = 2 * alpha[nnz, n].T @ S @ landmarks[nnz, :] / E
        Q_torch[-1, n * dim : n * dim + prob.d] = new_xi
        Q_torch[n * dim : n * dim + prob.d, -1] = new_xi
        # zi element
        new_zi = -torch.sum(alpha[nnz, n].T @ S) / E
        Q_torch[-1, (n + 1) * dim - 1 : (n + 1) * dim] = new_zi
        Q_torch[(n + 1) * dim - 1 : (n + 1) * dim, -1] = new_zi
        # h element
        Q_torch[-1, -1] += alpha[nnz, n].T @ S @ alpha[nnz, n] / E
    return Q_torch


def add_sigma_to_R(R, prob, sigma):
    raise NotImplementedError("parametrizing sigma is not implemented yet.")


def build_data_mat(prob, biases=None, sigma_acc_est=None):
    # build data matrix, introducing unknown biases.
    R_old = torch.Tensor(get_R_matrix(prob).toarray())
    Q_old = torch.Tensor(get_Q_matrix(prob).toarray())
    if biases is not None:
        Q_new = add_biases_to_Q(Q_old, prob, biases)
        return Q_new + R_old
    elif sigma_acc_est is not None:
        R_new = add_sigma_to_R(R_old)
        return Q_old + R_new
    else:
        raise ValueError("must either give biases or sigma_acc_est")


def test_ro_certs(noise=0):
    np.random.seed(1)
    n_landmarks = 5
    n_positions = 4
    d = 2
    prob = Problem(
        N=n_positions,
        d=d,
        K=n_landmarks,
        regularization=Reg.CONSTANT_VELOCITY,
    )
    prob.generate_random(sigma_dist_real=noise)
    # matrices without biases
    R = get_R_matrix(prob)
    Q = R + get_Q_matrix(prob)

    # check that zero bias gives the same result.
    biases = np.zeros(n_landmarks)
    prob.add_biases(biases)
    prob.D_noisy_sq = prob.D_biased**2

    biases = torch.Tensor(biases)
    Q_new = build_data_mat(prob, biases).detach().numpy()
    np.testing.assert_allclose(Q_new, Q.toarray(), atol=1e-6)

    # create problem with biases
    biases = np.random.uniform(size=n_landmarks)
    prob.add_biases(biases)
    prob.D_noisy_sq = prob.D_biased**2

    # create matrix where we remove the biases
    biases = torch.Tensor(biases)
    Q_new = build_data_mat(prob, biases).detach().numpy()
    np.testing.assert_allclose(Q_new, Q.toarray())

    # TODO: continue here
    # A_0, A_list = get_A_list(prob)


if __name__ == "__main__":
    # test_toy_inner(noise=0)
    # test_toy_outer(noise=0, verbose=True)

    # test_toy_inner(noise=1e-3)
    # test_toy_outer(noise=1e-3, verbose=True)

    test_ro_certs()
