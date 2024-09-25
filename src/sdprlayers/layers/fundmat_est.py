import numpy as np
import torch
import torch.nn as nn
from poly_matrix import PolyMatrix
from torch.profiler import record_function

from sdprlayers.layers.sdprlayer import SDPRLayer
from sdprlayers.utils.lie_algebra import se3_inv, se3_log


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
            + self.get_epi_norm_constraints()
            + self.get_fund_norm_constraint()
        )

        # Initialize SDPRLayer
        self.sdprlayer = SDPRLayer(n_vars=13, constraints=constraints)

        tol = 1e-10
        self.mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": tol * 1e-2,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        }

    def get_device(module):
        """
        Get the device of a PyTorch nn.Module.

        Args:
            module (torch.nn.Module): The module to check.

        Returns:
            torch.device: The device on which the module is located.
        """
        return next(module.parameters()).device

    def forward(
        self,
        keypoints_src,
        keypoints_trg,
        weights,
        verbose=False,
        rescale=True,
        reg=0.0,
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
                keypoints_src, keypoints_trg, weights, scale_offset=rescale, reg=reg
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
        fund_mat = torch.reshape(x[:, :9, :], (-1, 3, 3)).mT
        epipole = x[:, 9:12, :]

        return fund_mat, epipole

    @staticmethod
    def get_obj_matrix_vec(
        keypoints_src,
        keypoints_trg,
        weights,
        scale_offset=True,
        reg=0.0,
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
        f = slice(0, 9)
        e = slice(9, 12)
        h = 12
        # Form data matrix from points and weights
        # NOTE: if s_k, t_k, w_k are the kth source, target and weights, this einsum computes:
        #      sum_k  ( w_k * s_k @ t_k^T )      on each batch dim
        weights_sqz = torch.squeeze(weights, 1)
        X = torch.einsum(
            "bik,bjk,bk->bij",
            keypoints_src[:, :3, :],
            keypoints_trg[:, :3, :],
            weights_sqz,
        )
        # Vectorized (transposed) data matrix
        vecXt = torch.reshape(X, (-1, 9, 1)).squeeze(2)
        # Form cost matrix
        Q = torch.zeros(B, 13, 13, device=keypoints_src.device)
        Q[:, f, h] = vecXt / 2
        Q[:, h, f] = vecXt / 2

        # Add regularization
        R = torch.tensor(reg) * torch.eye(Q.size(1))[None, :, :].expand(Q.size())
        Q += R
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
    def get_epi_norm_constraints():
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


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    prod = A[..., :, None, :, None] * B[..., None, :, None, :]
    other_dims = tuple(A.shape[:-2])
    return prod.reshape(
        *other_dims, A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]
    )
