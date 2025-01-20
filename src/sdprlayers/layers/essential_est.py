import pickle

import numpy as np
import scipy as sp
import torch
import torch.nn as nn
from cert_tools import HomQCQP
from cert_tools.sparse_solvers import solve_dsdp
from diffcp import cones
from kornia.geometry import motion_from_essential_choose_solution
from poly_matrix import PolyMatrix
from torch.profiler import record_function

from sdprlayers.layers.sdprlayer import SDPRLayer
from sdprlayers.utils.lie_algebra import so3_wedge


class SDPEssMatEst(nn.Module):
    """
    Compute the essential matrix relating two images using the
    Semidefinite Programming Relaxation (SDPR)Layer. Inputs to the layer need to be normalized 2D points.

    SDP relaxation is based on:
    "An Efficient Solution to Non-Minimal Case Essential Matrix Estimation" by Ji Zhao

    Essential matrix is defined by the equations:

    point_target^T @ E @ point_source = 0

    E = [t_st_t]^x R_ts

    where

    point_target = R_ts @ point_source + t_st_t

    """

    def __init__(self, K_source=None, K_target=None, tol=1e-10, **kwargs):
        """
        Initialize the FundMatSDPBlock class.
        Intrinsic matrix inputs are used solely to determine the best essential matrix for the given
        Args:
            tol = tolerance used for the SDP Solver
            K_source = Intrinsic matrix of the source image points
            K_target = Intrinsic matrix of the target image points
            kwargs = Key word arguments passed to the SDPRLayer class.
        """
        super(SDPEssMatEst, self).__init__()
        # Define constraint dict
        self.var_dict = {"h": 1, "e1": 3, "e2": 3, "e3": 3, "t": 3}
        self.size = 13
        # Intrinsic Matrices
        self.K_source = K_source
        self.K_target = K_target

        # Generate constraints
        constraints = self.get_t_norm_constraint()
        constraints += self.get_tcross_constraints()

        # Initialize SDPRLayer
        self.sdprlayer = SDPRLayer(
            n_vars=13, use_dual=True, constraints=constraints, redun_list=[], **kwargs
        )

        self.tol = tol
        self.mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": tol * 1e-2,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        }
        # Initialize HomQCQP class for sparse decomposition
        self.homQCQP = HomQCQP(homog_var="h")
        # Get Example cost matrix
        vec = torch.tensor([[[1, 1, 1]]]).mT
        wt = torch.ones(1, 1, 1)
        Q, _, _ = self.get_obj_matrix_vec(vec, vec, wt)
        self.homQCQP.C = PolyMatrix.init_from_sparse(
            Q[0].numpy(), var_dict=self.var_dict, symmetric=True
        )[0]
        # Get Constraints
        self.homQCQP.As = []
        for A in constraints:
            self.homQCQP.As.append(
                PolyMatrix.init_from_sparse(A, var_dict=self.var_dict, symmetric=True)[
                    0
                ]
            )
        # Perform clique decomposition
        self.homQCQP.clique_decomposition()

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
        choose_soln=True,
    ):
        """
        Compute the optimal essential matrix relating source and target frame. This
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

        # Construct objective function
        # with record_function("SDPR: Build Cost Matrix"):
        Qs, scale, offset = self.get_obj_matrix_vec(
            keypoints_src, keypoints_trg, weights, scale_offset=rescale
        )

        # Solve decomposed SDP
        ext_vars_list = []
        for Q in Qs:
            # Overwrite stored cost
            self.homQCQP.C = PolyMatrix.init_from_sparse(
                Q.detach().numpy(), var_dict=self.var_dict, symmetric=True
            )[0]
            # Run solve
            cliques, info = solve_dsdp(
                self.homQCQP, form="dual", tol=self.tol, verbose=verbose
            )
            # Recover primal solution
            Y, ranks, factor_dict = self.homQCQP.get_mr_completion(
                cliques, var_list=list(self.var_dict.keys()), rank_tol=1e5
            )
            S = Y @ Y.T
            # Recover dual solution
            H = self.homQCQP.get_dual_matrix(
                info["dual"], var_list=self.var_dict
            ).toarray()
            mults = np.array(info["mults"])

            # DEBUG
            # C = Q.detach().numpy()
            # constraints = self.sdprlayer.constr_list + [self.sdprlayer.A_0]
            # A_bar = []
            # H2 = C.copy()
            # for i, A in enumerate(constraints):
            #     H2 += A * mults[i]
            #     A_bar.append(A @ Y)
            # A_bar = np.hstack(A_bar)
            # ls_sol = np.linalg.lstsq(A_bar, -C @ Y)
            # mults_2 = ls_sol[0]

            # Add solution to list to pass to layer
            ext_vars_list.append(
                dict(
                    x=mults,
                    y=cones.vec_symm(S),
                    s=cones.vec_symm(H),
                )
            )

        # Solver set to external to bypass cvxpy forward solve
        solver_args = dict(solve_method="external", ext_vars_list=ext_vars_list)
        # call sdprlayer
        with record_function("SDPR: Run Optimization"):
            X, x = self.sdprlayer(Qs, solver_args=solver_args)
        # Solution rank
        rank = np.max(ranks)

        # Extract solution
        E_mats = torch.reshape(x[:, 1:10, :], (-1, 3, 3))

        if choose_soln:
            # There is ambiguity in the solution at this point due to the fact that there are 10 possible essential matrices for a given set of keypoint correspondences.
            # To deal with this we compute the "best" rotation and translation using the kornia library
            if self.K_source is None:
                K_source = torch.eye(3).expand(E_mats.shape[0], -1, -1)
            else:
                K_source = self.K_source

            if self.K_target is None:
                K_target = torch.eye(3).expand(E_mats.shape[0], -1, -1)
            else:
                K_target = self.K_target
            # Choose the rotation and translation that best represent the keypoints
            # Returns R_ts and t_ts_s
            Rs, ts, points_3d = motion_from_essential_choose_solution(
                E_mats,
                K_target,
                K_source,
                keypoints_trg[:, :2, :].mT,
                keypoints_src[:, :2, :].mT,
            )
            # reconstruct Essential matrices
            Es = E_mats
        else:
            # NOTE this does not necessarily get the solution with the most points in
            # front of the camera.
            # Get the solution from the optimization
            ts = x[:, 10:]
            Es = E_mats
            # Get a valid Rotation matrix
            U, Sigma, V = torch.linalg.svd(Es)
            W = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
            Rs = U @ W @ V
            # Make sure the frame is right handed
            Rs = Rs * torch.linalg.det(Rs)

        return Es, Rs, ts, X, rank

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
        #      sum_k  w_k * (t_k KRON s_k )(t_k KRON s_k )^T      on each batch dim
        weights_sqz = torch.squeeze(weights, 1)
        # Form [vecX]_k = kron(a,b) = vec( a @ b^T ) = vec(X), for row major vectorization
        X = torch.einsum("bik, bjk->bijk", keypoints_trg, keypoints_src)
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
