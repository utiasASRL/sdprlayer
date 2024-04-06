from copy import deepcopy

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch

from ro_certs.problem import Problem, Reg
from ro_certs.sdp_setup import get_A_list, get_Q_matrix, get_R_matrix


class RealProblem(Problem):

    @staticmethod
    def init_from_prob(prob: Problem, n_calib=None):
        real_prob = RealProblem(
            n_anchors=prob.K,
            n_positions=prob.N,
            d=prob.d,
            reg=prob.regularization,
            n_calib=n_calib,
            real=True,
        )
        real_prob.anchors = prob.anchors
        real_prob.D_noisy_sq = prob.D_noisy_sq
        real_prob.W = prob.W
        real_prob.trajectory = prob.trajectory
        real_prob.times = prob.times
        real_prob.biases = real_prob.get_biases(real_prob.D_noisy_sq, squared=True)
        return real_prob

    def get_downsampled_version(self, number=3, method="keep-first"):
        other = deepcopy(self)
        if method == "keep-first":
            keep_idx = np.arange(number)
        elif method == "keep-last":
            keep_idx = np.range(other.N - number, other.N)
        elif method == "keep-uniform":
            keep_idx = np.linspace(0, other.N - 1, number).astype(int)
            keep_idx = np.unique(keep_idx)
        else:
            raise ValueError(f"Unknown method {method}")
        assert len(keep_idx) == number
        other.N = number
        other.D_noisy_sq = self.D_noisy_sq[:, keep_idx]
        other.W = self.W[:, keep_idx]
        other.trajectory = self.trajectory[keep_idx, :]
        other.times = self.times[keep_idx]
        self.trajectory = None
        return other

    def __init__(
        self,
        n_anchors,
        n_positions,
        d,
        noise=0,
        reg=Reg.CONSTANT_VELOCITY,
        n_calib=None,
        real=False,
    ):
        super().__init__(
            K=n_anchors,
            N=n_positions,
            d=d,
            regularization=reg,
        )
        self.n_calib = n_calib if n_calib else n_anchors
        if not real:
            self.generate_random(sigma_dist_real=noise)

    def generate_biases(self, biases=None):
        if biases is None:
            biases = np.zeros(self.K)
            biases[: self.n_calib] = np.random.uniform(size=self.n_calib) * 0.1
        self.add_biases(biases)
        self.D_noisy_sq = self.D_biased**2
        self.biases = biases

    def get_x(self):
        z = np.linalg.norm(self.trajectory, axis=1) ** 2
        return np.hstack(
            [
                [1.0],
                np.hstack([self.theta[:, : self.get_dim()], z[:, None]]).flatten(),
            ]
        )

    def get_positions(self, x):
        dim = self.get_dim() + 1
        theta = x.reshape((-1, dim))  # each row contains x_i, v_i, z_i
        # make sure z == ||x||^2
        if isinstance(x, np.ndarray):
            np.testing.assert_almost_equal(
                np.linalg.norm(theta[:, : self.d], axis=1) ** 2, theta[:, -1], decimal=4
            )
        return theta[:, : self.d]

    def add_biases_to_Q(self, Q_torch: torch.Tensor, biases: torch.Tensor):
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
                  h                          x1                         z1          ...   xN   zN
        [sum_n(alphan.T @ Sn @ alphan) 2*alpha1.T @ S1 @ An     alpha1.T @ S1 @ 1   ...   ...  ...]
        """

        hom_pos = 0  # position of homogeneous variable. Always 0 here! Used to be -1.
        distances = torch.tensor(np.sqrt(self.D_noisy_sq))
        anchors = torch.tensor(self.anchors)  # n_landmarks x d
        S_all = torch.tensor(self.Sig_inv)
        E = np.sum(self.W)

        # we want alpha to be n_landmarks x n_positions
        alpha = (distances - biases[:, None]) ** 2 - (
            torch.norm(anchors, p=2, dim=1) ** 2
        )[:, None]

        Q_torch[hom_pos, hom_pos] = 0

        # make sure we start filling at 1 if the first column is the hom. variable
        s = 1 if hom_pos == 0 else 0

        dim = self.get_dim() + 1
        # TODO(FD) could probably do this without forloop.
        for n in range(self.N):
            nnz = self.W[:, n] != 0
            S = S_all[nnz, :][:, nnz]
            # xi element
            new_xi = 2 * alpha[nnz, n] @ S @ anchors[nnz, :] / E
            Q_torch[hom_pos, s + n * dim : s + n * dim + self.d] = new_xi
            Q_torch[s + n * dim : s + n * dim + self.d, hom_pos] = new_xi
            # zi element
            new_zi = -torch.sum(alpha[nnz, n] @ S) / E
            Q_torch[hom_pos, s + (n + 1) * dim - 1 : s + (n + 1) * dim] = new_zi
            Q_torch[s + (n + 1) * dim - 1 : s + (n + 1) * dim, hom_pos] = new_zi
            # h element
            Q_torch[hom_pos, hom_pos] += alpha[nnz, n] @ S @ alpha[nnz, n] / E
        return Q_torch

    def add_sigma_to_R(self, R, sigma):
        raise NotImplementedError("parametrizing sigma is not implemented yet.")

    def build_data_mat(self, biases_est=None, sigma_acc_est=None):
        # build data matrix, introducing biases or sigma parameters.
        R_old = torch.tensor(self.get_R_matrix().toarray())
        Q_old = torch.tensor(self.get_Q_matrix().toarray())
        if biases_est is not None:
            assert len(biases_est) == self.n_calib
            biases = torch.zeros(self.K)
            biases[: self.n_calib] = biases_est
            Q_new = self.add_biases_to_Q(Q_old, biases)
            return Q_new + R_old
        elif sigma_acc_est is not None:
            R_new = self.add_sigma_to_R(R_old, sigma_acc_est)
            return Q_old + R_new

        else:
            raise ValueError("must either give biases or sigma_acc_est")

    def get_Q_matrix(self):
        return get_Q_matrix(self, hom_position="first")

    def get_R_matrix(self):
        return get_R_matrix(self, hom_position="first")

    def get_constraints(self):
        A_0, A_list = get_A_list(self, hom_position="first")
        return A_list


class ToyProblem(object):
    NOISE = 1e-3

    def __init__(self, n_anchors, n_positions, d, noise=NOISE, n_calib=None):
        assert n_positions == 1, "more than 1 not supported yet"
        self.n_anchors = n_anchors
        self.n_positions = n_positions
        self.n_calib = n_calib if n_calib else n_anchors
        self.d = d

        self.anchors = np.random.uniform(low=-5, high=5, size=(n_anchors, d))

        self.biases = np.zeros(self.n_anchors)
        self.biases[: self.n_calib] = 0.1 * np.arange(1, self.n_calib + 1)
        # self.biases[: self.n_calib] = np.random.uniform(size=n_calib)  # between 0 and 1

        # self.trajectory = np.random.uniform(low=-1, high=1, size=(n_positions, d))
        self.trajectory = np.mean(self.anchors, axis=0)[None, :]

        self.gt_distances = np.linalg.norm(
            self.anchors[None, :, :] - self.trajectory[:, None, :], axis=2
        )
        # n_positions x n_anchors distance matrix
        self.biased_distances = self.gt_distances + self.biases[None, :]
        self.biased_distances = self.biased_distances + np.random.normal(
            scale=noise, loc=0, size=self.gt_distances.shape
        )

    def get_x(self):
        return np.hstack(
            [1.0, self.trajectory.flatten(), np.linalg.norm(self.trajectory) ** 2]
        )

    def get_positions(self, x):
        return x[: self.d].reshape((self.n_positions, self.d))

    def build_data_mat(self, biases_est):
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
        assert isinstance(biases_est, torch.Tensor)
        assert len(biases_est) == self.n_calib
        biases = torch.zeros(self.n_anchors)
        biases[: self.n_calib] = biases_est

        Q_tch = torch.zeros((self.d + 2, self.d + 2))
        anchors_tensor = torch.tensor(self.anchors)  # K x d
        alphas = (
            torch.tensor(self.biased_distances) - biases
        ) ** 2 - anchors_tensor.norm(p=2, dim=1) ** 2
        Q_tch[0, 0] = torch.sum(alphas**2)
        a = 2 * torch.sum(alphas * anchors_tensor.mT, axis=1)
        Q_tch[0, 1 : self.d + 1] = a
        Q_tch[1 : self.d + 1, 0] = a
        Q_tch[0, self.d + 1] = -torch.sum(alphas)
        Q_tch[self.d + 1, 0] = -torch.sum(alphas)
        Q_tch[1 : self.d + 1, 1 : self.d + 1] = 4 * torch.sum(
            anchors_tensor[:, :, None] @ anchors_tensor[:, None, :],
            axis=0,
        )  # (K x d x 1) @ (K x 1 x d) = (K x d x d)
        Q_tch[1 : self.d + 1, self.d + 1] = -2 * torch.sum(anchors_tensor, axis=0)
        Q_tch[self.d + 1, 1 : self.d + 1] = -2 * torch.sum(anchors_tensor, axis=0)
        Q_tch[self.d + 1, self.d + 1] = self.n_anchors
        return Q_tch

    def get_constraints(self):
        # x is of shape [h, x', z]
        A = sp.lil_array((self.d + 2, self.d + 2))  # z = ||x||^2
        for i in range(self.d):
            A[i + 1, i + 1] = 1
        A[0, -1] = -0.5
        A[-1, 0] = -0.5
        return [A]

    def plot_problem(self):
        fig, ax = plt.subplots()
        ax.scatter(*self.landmarks[:, :2].T)
        ax.scatter(*self.trajectory[:, :2].T)
