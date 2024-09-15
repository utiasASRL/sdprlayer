import numpy as np
import torch
import torch.nn as nn
from poly_matrix import PolyMatrix
from torch.profiler import record_function

from sdprlayer import SDPRLayer
from sdprlayer.utils.lie_algebra import se3_inv, se3_log


class FundMatSDPBlock(nn.Module):
    """
    Compute the fundamental matrix relating two images using the
    Semidefinite Programming Relaxation (SDPR)Layer.
    """

    def __init__(self):
        """
        Initialize the FundMatSDPBlock class.

        Args:
            T_s_v (torch.tensor): 4x4 transformation matrix providing the transform from the vehicle frame to the
                                  sensor frame.
        """
        super(FundMatSDPBlock, self).__init__()

        # Generate constraints
        constraints = (
            self.get_nullspace_constraint()
            + self.get_epipole_constraints()
            + self.get_fund_norm_constraint()
        )

        # Initialize SDPRLayer
        self.sdprlayer = SDPRLayer(n_vars=13, constraints=constraints)

        tol = 1e-12
        self.mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": tol * 1e-2,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        }

    def forward(
        self,
        keypoints_src,
        keypoints_trg,
        weights,
        inv_cov_weights=None,
        verbose=False,
    ):
        """
        Compute the optimal fundamental matrix relating source and target frame. This
            matrix minimizes the following cost function:
            C(F) = sum( x_trg^T F x_src )
            where x_trg is a target keypoint drawn from keypoints_trg and x_src is a source keypoint
            drawn from keypoints_src

        Args:
            keypoints_src (torch,tensor, Bx3xN): 2D (homogenized) point coordinates of keypoints from source frame.
            keypoints_trg (torch,tensor, Bx3xN): 2D (homogenized) point coordinates of keypoints from target frame.
            weights (torch.tensor, Bx1xN): weights in range (0, 1) associated with the matched source and target
                                           points.

        Returns:
            F_trg_src (torch.tensor, Bx3x3): Fundamental matrix relating source and target frame.
            e_src (torch.tensor, Bx3x1): Epipole corresponding to the fundamental matrix.
        """
        batch_size, _, n_points = keypoints_src.size()

        # Construct objective function
        with record_function("SDPR: Build Cost Matrix"):
            Qs, scale, offset = self.get_obj_matrix_vec(
                keypoints_src, keypoints_trg, weights, inv_cov_weights
            )
        # Set up solver parameters
        # solver_args = {
        #     "solve_method": "SCS",
        #     "eps": 1e-7,
        #     "normalize": True,
        #     "max_iters": 100000,
        #     "verbose": False,
        # }
        solver_args = {
            "solve_method": "mosek",
            "mosek_params": self.mosek_params,
            "verbose": verbose,
        }

        # TODO Rework this section to retrieve optimum
        # Run layer
        with record_function("SDPR: Run Optimization"):
            X, x = self.sdprlayer(Qs, solver_args=solver_args)
        # Extract solution
        t_trg_src_intrg = x[:, 9:]
        R_trg_src = torch.reshape(x[:, 0:9], (-1, 3, 3)).transpose(-1, -2)
        t_src_trg_intrg = -t_trg_src_intrg
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(keypoints_src)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(keypoints_src)  # Bx1x1
        trans_cols = torch.cat([t_src_trg_intrg, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([R_trg_src, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4

        # Convert from sensor to vehicle frame
        T_s_v = self.T_s_v.expand(batch_size, 4, 4).cuda()
        T_trg_src = se3_inv(T_s_v).bmm(T_trg_src).bmm(T_s_v)

        return T_trg_src

    @staticmethod
    def get_obj_matrix_vec(
        keypoints_src,
        keypoints_trg,
        weights,
        inv_cov_weights=None,
        scale_offset=True,
    ):
        """Compute the QCQP (Quadratically Constrained Quadratic Program) objective matrix
        based on the given 2D keypoints from source and target frames, and their corresponding weights.

         Args:
            keypoints_src (torch.Tensor): A tensor of shape (N_batch, 3, N) representing the
                                             2D coordinates of keypoints in the source frame.
                                             N_batch is the batch size and N is the number of keypoints.
            keypoints_trg (torch.Tensor): A tensor of shape (N_batch, 3, N) representing the
                                             2D coordinates of keypoints in the target frame.
                                             N_batch is the batch size and N is the number of keypoints.
            weights (torch.Tensor): A tensor of shape (N_batch, 1, N) representing the weights
                                    corresponding to each keypoint.
        Returns:
            _type_: _description_
        """
        B = keypoints_src.shape[0]  # Batch dimension
        N = keypoints_src.shape[2]  # Number of points
        # Indices
        h = 0
        c = slice(1, 10)
        t = slice(10, 13)
        # relabel and dehomogenize
        m = keypoints_src[:, :3, :]
        y = keypoints_trg[:, :3, :]
        Q_n = torch.zeros(B, N, 13, 13).cuda()
        # world frame keypoint vector outer product
        M = torch.einsum("bin,bjn->bnij", m, m)  # BxNx3x3
        if inv_cov_weights is not None:
            W = inv_cov_weights  # BxNx3x3
        else:
            # Weight with identity if no weights are provided
            W = torch.eye(3, 3).cuda().expand(B, N, -1, -1)
        # diagonal elements
        Q_n[:, :, c, c] = kron(M, W)  # BxNx9x9
        Q_n[:, :, t, t] = W  # BxNx3x3
        Q_n[:, :, h, h] = torch.einsum("bin,bnij,bjn->bn", y, W, y)  # BxN
        # Off Diagonals
        m_ = m.transpose(-1, -2).unsqueeze(3)  # BxNx3x1
        Wy = torch.einsum("bnij,bjn->bni", W, y).unsqueeze(3)  # BxNx3x1
        Q_n[:, :, c, t] = -kron(m_, W)  # BxNx9x3
        Q_n[:, :, t, c] = Q_n[:, :, c, t].transpose(-1, -2)  # BxNx3x9
        Q_n[:, :, c, h] = -kron(m_, Wy).squeeze(-1)  # BxNx9
        Q_n[:, :, h, c] = Q_n[:, :, c, h]  # BxNx9
        Q_n[:, :, t, h] = Wy.squeeze(-1)  # BxNx3
        Q_n[:, :, h, t] = Q_n[:, :, t, h]  # Bx3xN
        # Scale by weights
        weights = weights.squeeze(1)
        Q = torch.einsum("bnij,bn->bij", Q_n, weights)
        # NOTE: operations below are to improve optimization conditioning for solver
        # remove constant offset
        if scale_offset:
            offsets = Q[:, 0, 0].clone()
            Q[:, 0, 0] = torch.zeros(B).cuda()
            # rescale
            scales = torch.norm(Q, p="fro")
            Q = Q / torch.norm(Q, p="fro")
        else:
            scales, offsets = None, None
        return Q, scales, offsets

    @staticmethod
    def get_obj_matrix(keypoints_src, keypoints_trg, weights, inv_cov_weights=None):
        """
        Compute the QCQP (Quadratically Constrained Quadratic Program) objective matrix
        based on the given 3D keypoints from source and target frames, and their corresponding weights.
        NOTE: This function is here only for debugging. It is not used in the forward pass.
              This function is currently not vectorized and iterates over each batch and each keypoint.

        Args:
            keypoints_src (torch.Tensor): A tensor of shape (N_batch, 4, N) representing the
                                             3D coordinates of keypoints in the source frame.
                                             N_batch is the batch size and N is the number of keypoints.
            keypoints_trg (torch.Tensor): A tensor of shape (N_batch, 4, N) representing the
                                             3D coordinates of keypoints in the target frame.
                                             N_batch is the batch size and N is the number of keypoints.
            weights (torch.Tensor): A tensor of shape (N_batch, 1, N) representing the weights
                                    corresponding to each keypoint.

        Returns:
            list: A list of tensors representing the QCQP objective matrices for each batch.
        """
        # Get batch dimension
        N_batch = keypoints_src.shape[0]
        # Indices
        h = [0]
        c = slice(1, 10)
        t = slice(10, 13)
        Q_batch = []
        scales = torch.zeros(N_batch).cuda()
        offsets = torch.zeros(N_batch).cuda()
        for b in range(N_batch):
            Q_es = []
            for i in range(keypoints_trg.shape[-1]):
                if inv_cov_weights is None:
                    W_ij = torch.eye(3).cuda()
                else:
                    W_ij = inv_cov_weights[b, i, :, :]
                m_j0_0 = keypoints_src[b, :3, [i]]
                y_ji_i = keypoints_trg[b, :3, [i]]
                # Define matrix
                Q_e = torch.zeros(13, 13).cuda()
                # Diagonals
                Q_e[c, c] = kron(m_j0_0 @ m_j0_0.T, W_ij)
                Q_e[t, t] = W_ij
                Q_e[h, h] = y_ji_i.T @ W_ij @ y_ji_i
                # Off Diagonals
                Q_e[c, t] = -kron(m_j0_0, W_ij)
                Q_e[t, c] = Q_e[c, t].T
                Q_e[c, h] = -kron(m_j0_0, W_ij @ y_ji_i)
                Q_e[h, c] = Q_e[c, h].T
                Q_e[t, h] = W_ij @ y_ji_i
                Q_e[h, t] = Q_e[t, h].T

                # Add to running list of measurements
                Q_es += [Q_e]
            # Combine objective
            Q_es = torch.stack(Q_es)
            Q = torch.einsum("nij,n->ij", Q_es, weights[b, 0, :])
            # remove constant offset
            offsets[b] = Q[0, 0].clone()
            Q[0, 0] = 0.0
            # Rescale
            scales[b] = torch.norm(Q, p="fro")
            Q = Q / torch.norm(Q, p="fro")
            # Add to running list of batched data matrices
            Q_batch += [Q]

        return torch.stack(Q_batch), scales, offsets

    @staticmethod
    def get_nullspace_constraint():
        """Generate 3 constraints that force the source epipole to be in the nullspace of the fundamental
        matrix."""
        # labels
        h = "h"
        F = "F"
        e = "e"
        variables = {F: 9, e: 3, h: 1}
        constraints = []
        for i in range(3):
            A = PolyMatrix()
            a = np.zeros((3, 1))
            a[i, :] = 1.0
            A[e, F] = kron(np.eye(3), a.T)
            constraints += [A.get_matrix(variables)]
        return constraints

    @staticmethod
    def get_epipole_constraints():
        """Generate constraint that epipole is homogeneous"""
        h = "h"
        F = "F"
        e = "e"
        variables = {F: 9, e: 3, h: 1}
        A = PolyMatrix()
        # Constrain the epipolar line to have unit length
        A[e, e] = np.eye(3)
        A[h, h] = -1.0
        constraints = [A.get_matrix(variables)]

        return constraints

    @staticmethod
    def get_fund_norm_constraint():
        """Constrain the Frobenius norm of the fundamental matrix to be one"""
        h = "h"
        F = "F"
        e = "e"
        variables = {F: 9, e: 3, h: 1}
        A = PolyMatrix()
        # Constrain third vector element of e to be 1
        A[F, F] = np.eye(9)
        A[h, h] = -1.0
        constraints = [A.get_matrix(variables)]

        return constraints

    @staticmethod
    def plot_points(s_in, t_in, w_in):
        """purely for debug"""
        import matplotlib.pyplot as plt

        s = s_in.cpu().detach().numpy()
        t = t_in.cpu().detach().numpy()
        w = w_in.cpu().detach().numpy()
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            s[0, 0, :],
            s[0, 1, :],
            s[0, 2, :],
            marker="*",
            color="g",
        )
        ax.scatter3D(
            t[0, 0, :],
            t[0, 1, :],
            t[0, 2, :],
            marker="*",
            color="b",
        )
        ax.scatter3D(
            0.0,
            0.0,
            0.0,
            marker="*",
            color="r",
        )
        return ax


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    prod = A[..., :, None, :, None] * B[..., None, :, None, :]
    other_dims = tuple(A.shape[:-2])
    return prod.reshape(
        *other_dims, A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]
    )
