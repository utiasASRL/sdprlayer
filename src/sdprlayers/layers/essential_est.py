import pickle

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from poly_matrix import PolyMatrix
from torch.profiler import record_function

from sdprlayers.layers.sdprlayer import SDPRLayer
from sdprlayers.utils.lie_algebra import se3_inv, se3_log


class EssentialSDPBlock(nn.Module):
    """
    Compute the essential matrix relating two images using the
    Semidefinite Programming Relaxation (SDPR)Layer. Inputs to the layer need to be normalized 2D points.

    SDP relaxation is based on:
    "An Efficient Solution to Non-Minimal Case Essential Matrix Estimation" by Ji Zhao
    """

    def __init__(self):
        """
        Initialize the FundMatSDPBlock class.

        Args:
            T_s_v (torch.tensor): 4x4 transformation matrix providing the transform from the vehicle frame to the
                                  sensor frame.
        """
        super(EssentialSDPBlock, self).__init__()
        # Define constraint dict
        self.var_dict = {"h": 1, "e1": 3, "e2": 3, "e3": 3, "t": 3}
        self.size = 13

        # Generate constraints
        constraints = self.get_t_norm_constraint()
        constraints += self.get_tcross_constraints()
        # constraints += self.get_tE_constraints()

        # Initialize SDPRLayer
        self.sdprlayer = SDPRLayer(n_vars=13, use_dual=False, constraints=constraints)

        tol = 1e-8
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

    @staticmethod
    def load_constr_from_file(filename):
        """Load learned constraints from file"""
        print("Loading Constraints from " + filename)
        with open(filename, "rb") as handle:
            data = pickle.load(handle)
        return data["constraints"]

    def forward(
        self,
        keypoints_src,
        keypoints_trg,
        weights,
        verbose=False,
        rescale=True,
    ):
        """
        Compute the optimal fundamental matrix relating source and target frame. This
            matrix minimizes the following cost function:
            C(E) = sum( x_trg^T E x_src )
            where x_trg is a target keypoint drawn from keypoints_trg and x_src is a source keypoint drawn from keypoints_src. It is assumed that these keypoints are normalized.

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
                keypoints_src, keypoints_trg, weights, scale_offset=rescale
            )
        # Set up solver parameters
        # solver_args = {
        #     "solve_method": "SCS",
        #     "eps": 1e-7,
        #     "normalize": True,
        #     "max_iters": 100000,
        #     "verbose": True,
        # }
        # solver_args = {
        #     "solve_method": "mosek",
        #     "mosek_params": self.mosek_params,
        #     "verbose": verbose,
        # }
        # NOTE: Clarabel exploits chordal sparsity which is key to making sure that the solution is rank-1
        tol = 1e-4
        solver_args = dict(
            solve_method="Clarabel",
            verbose=True,
            tol_gap_abs=tol,
            tol_gap_rel=tol,
            tol_feas=tol,
            tol_infeas_abs=tol,
            tol_infeas_rel=tol,
            tol_ktratio=tol / 100,
            chordal_decomposition_enable=True,
            chordal_decomposition_complete_dual=True,
        )

        # TODO Rework this section to retrieve optimum
        # Run layer
        with record_function("SDPR: Run Optimization"):
            X, x = self.sdprlayer(Qs, solver_args=solver_args)
        # Extract solution
        E_mats = torch.reshape(x[:, 1:10, :], (-1, 3, 3))
        trans_vecs = x[:, -3:, :]

        return E_mats, trans_vecs, X

    @staticmethod
    def get_obj_matrix_vec(
        keypoints_src,
        keypoints_trg,
        weights,
        scale_offset=True,
    ):
        """Compute the QCQP (Quadratically Constrained Quadratic Program) objective matrix based on the given 2D keypoints from source and target frames, and their corresponding weights.

         Args:
            keypoints_src (torch.Tensor): A tensor of shape (N_batch, 3, N) representing the 2D normalized coordinates of keypoints in the source frame. N_batch is the batch size and N is the number of keypoints.
            keypoints_trg (torch.Tensor): A tensor of shape (N_batch, 3, N) representing the 2D normalized coordinates of keypoints in the target frame. N_batch is the batch size and N is the number of keypoints.
            weights (torch.Tensor): A tensor of shape (N_batch, 1, N) representing the weights corresponding to each keypoint.
        Returns:
            _type_: _description_
        """
        B = keypoints_src.shape[0]  # Batch dimension
        N = keypoints_src.shape[2]  # Number of points
        # Indices
        e = slice(1, 10)
        # Form data matrix from points and weights
        # NOTE: if s_k, t_k, w_k are the kth source, target and weights, the cost matrix is given by
        #      sum_k  w_k * (s_k KRON t_k )(s_k KRON t_k )^T      on each batch dim
        weights_sqz = torch.squeeze(weights, 1)
        # Form [vecX]_k = kron(a,b) = vec( a @ b^T ) = vec(X), for row major vectorization
        X = torch.einsum("bik, bjk->bijk", keypoints_src, keypoints_trg)
        vecX = X.reshape((B, 9, N))
        # Q = sum ( w_k vecX_k vecX_k.T )
        Q_ff = torch.einsum("bik,bjk,bk->bij", vecX, vecX, weights_sqz)

        # Form cost matrix
        Q = torch.zeros(B, 13, 13, device=keypoints_src.device)
        Q[:, e, e] = Q_ff

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

    def get_t_norm_constraint(self):
        """Generate unit norm constraint on the translation vector."""
        h = "h"
        t = "t"
        variables = self.var_dict
        A = PolyMatrix()
        # Constrain the third element of the epipole to be unity
        A[t, t] = np.eye(3)
        A[h, h] = -1.0
        constraints = [A.get_matrix(variables)]
        return constraints

    def get_tcross_constraints(self):
        """Generate constraints corresponding to:
        E E^T = t^x t^xT
        """
        h = "h"
        e1 = "e1"
        e2 = "e2"
        e3 = "e3"
        t = "t"
        constraints = []
        # Diagonal terms
        A = PolyMatrix()
        A[e1, e1] = np.eye(3)
        A[t, t] = -np.diag(np.array([0, 1, 1]))
        constraints.append(A)

        A = PolyMatrix()
        A[e2, e2] = np.eye(3)
        A[t, t] = -np.diag(np.array([1, 0, 1]))
        constraints.append(A)

        A = PolyMatrix()
        A[e3, e3] = np.eye(3)
        A[t, t] = -np.diag(np.array([1, 1, 0]))
        constraints.append(A)
        # Off-diagonal (1,2)
        A = PolyMatrix()
        A[e1, e2] = np.eye(3) / 2
        mat = np.zeros((3, 3))
        mat[0, 1] = 0.5
        mat[1, 0] = 0.5
        A[t, t] = mat
        constraints.append(A)
        # Off-diagonal (1,3)
        A = PolyMatrix()
        A[e1, e3] = np.eye(3) / 2
        mat = np.zeros((3, 3))
        mat[0, 2] = 0.5
        mat[2, 0] = 0.5
        A[t, t] = mat
        constraints.append(A)
        # Off-diagonal (2,3)
        A = PolyMatrix()
        A[e2, e3] = np.eye(3) / 2
        mat = np.zeros((3, 3))
        mat[1, 2] = 0.5
        mat[2, 1] = 0.5
        A[t, t] = mat
        constraints.append(A)

        # Convert polymatrices to sparse matrices
        constraints = [A.get_matrix(self.var_dict) for A in constraints]

        return constraints

    def get_tE_constraints(self):
        """Generate constraints corresponding to E^T t = 0"""
        h = "h"
        e1 = "e1"
        e2 = "e2"
        e3 = "e3"
        t = "t"

        constraints = []
        for i in range(3):
            A = PolyMatrix()
            A[t, e1] = np.array([[1, 0, 0]]).T @ np.eye(3)[[i], :] / 2
            A[t, e2] = np.array([[0, 1, 0]]).T @ np.eye(3)[[i], :] / 2
            A[t, e3] = np.array([[0, 0, 1]]).T @ np.eye(3)[[i], :] / 2
            constraints.append(A)

        # Convert polymatrices to sparse matrices
        constraints = [A.get_matrix(self.var_dict) for A in constraints]

        return constraints


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    prod = A[..., :, None, :, None] * B[..., None, :, None, :]
    other_dims = tuple(A.shape[:-2])
    return prod.reshape(
        *other_dims, A.shape[-2] * B.shape[-2], A.shape[-1] * B.shape[-1]
    )
