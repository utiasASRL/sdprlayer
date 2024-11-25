import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from cert_tools import HomQCQP
from cert_tools.sparse_solvers import solve_dsdp
from poly_matrix import PolyMatrix

import sdprlayers.utils.fund_mat_utils as utils
from sdprlayers import FundMatSDPBlock
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import so3_wedge

matplotlib.use("TkAgg")


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestFundMat(unittest.TestCase):

    def __init__(self, *args, formulation="fro_norm", constraint_file=None, **kwargs):
        super(TestFundMat, self).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        self.device = "cuda:0"
        # Store formulation and constraint file
        self.formulation = formulation
        # Set seed
        set_seed(0)
        batch_size = 1
        n_points = 50
        # Set up test problem
        # NOTE t_ts is the vector from the source to the target frame expressed in the source frame
        # NOTE R_ts is the rotation that maps vectors in the source frame to vectors in the target frame
        # NOTE trailing "s" indicates that a variable is batched
        t_tss, R_tss, key_ss = utils.get_gt_setup(
            N_map=n_points, N_batch=batch_size, traj_type="clusters"
        )
        # HACK Simplified transform, lateral shift
        # R_tss = np.eye(3)[None, :, :]
        # t_tss = np.array([[10.0, 0.0, 0.0]]).T[None, :, :]
        # Transforms from source to target
        t_tss = torch.tensor(t_tss)
        R_tss = torch.tensor(R_tss)
        # Keypoints (3D) defined in source frame
        key_ss = torch.tensor(key_ss)[None, :, :].expand(batch_size, -1, -1)
        # Keypoints in target frame
        key_ts = torch.bmm(R_tss, key_ss - t_tss)
        # homogenize coordinates
        trg_coords = torch.concat(
            [key_ts, torch.ones(batch_size, 1, key_ss.size(2))], dim=1
        )
        src_coords = torch.concat(
            [key_ss, torch.ones(batch_size, 1, key_ss.size(2))], dim=1
        )

        # Define Camera
        camera = CameraModel(400, 600, 10.0, 0.0, 0.0)
        # camera = CameraModel(1.0, 1.0, 0.0, 0.0, 0.0)

        # Apply camera to get image points
        self.src_img_pts = camera.camera_model(src_coords)
        self.trg_img_pts = camera.camera_model(trg_coords)
        # Store values
        self.keypoints_3D_src = src_coords
        self.keypoints_3D_trg = trg_coords

        # Generate Scalar Weights
        self.weights = (
            torch.ones(self.keypoints_3D_src.size(0), 1, self.keypoints_3D_src.size(2))
            / n_points
        )
        self.camera = camera

        # Construct Essential Matrix
        # NOTE: E = R_ts t_ts^ = R_ts(-R_ts t_ts)^ = (t_st)^ R_ts
        Es = R_tss.bmm(so3_wedge(t_tss))
        check = (
            self.keypoints_3D_trg[0, :3, [0]].cpu().mT
            @ Es[0]
            @ self.keypoints_3D_src[0, :3, [0]].cpu()
        )
        np.testing.assert_allclose(check, 0.0, atol=1e-12)
        # Get intrinsic camera mat
        K_inv = torch.linalg.inv(camera.K)
        K_invs = K_inv.expand(batch_size, 3, 3)
        # Construct (Normalized) Fundamental matrices
        Fs_unnorm = K_invs.mT.bmm(Es).bmm(K_invs)
        if formulation == "fro_norm":
            norm = torch.linalg.norm(Fs_unnorm, keepdim=True)
        elif formulation == "unity_elem":
            norm = torch.abs(Fs_unnorm[:, 2, 2])
        if norm > 0:
            self.Fs = Fs_unnorm / norm
        else:
            self.Fs = Fs_unnorm
        # Construct epipoles
        u, s, vh = torch.linalg.svd(self.Fs)
        self.es = vh[:, [2], :].mT
        if not formulation == "fro_norm":
            eps_norm = torch.abs(self.es[:, 2, 0])
            assert torch.all(eps_norm > 1e-9), ValueError(
                "Epipoles cannot be at infinity for this formulation."
            )
            self.es = self.es / self.es[:, 2, 0]
        # Construct solution vectors
        self.sol = torch.cat(
            [
                torch.ones((batch_size, 1, 1)),
                torch.reshape(self.Fs.mT, (-1, 9, 1)),
                self.es,
            ],
            dim=1,
        )
        # Initialize layer
        self.layer = FundMatSDPBlock(self.formulation, constraint_file=constraint_file)

    def test_constraints(self):
        """Test that the constraints characterize the fundamental matrix and epipole."""

        # Get constraints and test
        constraints = self.layer.sdprlayer.constr_list
        viol = np.zeros((len(constraints)))
        sol = np.array(t.sol[0])
        for i, A in enumerate(constraints):
            viol[i] = (sol.T @ A @ sol)[0, 0]
            np.testing.assert_allclose(
                viol[i], 0.0, atol=1e-8, err_msg=f"Constraint {i} has violation"
            )

    def test_cost_matrix_nonoise(self):
        """Test the objective matrix with no noise"""
        self.test_cost_matrix(sigma_val=0.0)

    def test_cost_matrix(self, sigma_val=5.0):
        """Test the objective matrix with no noise"""
        # Sizes
        B = t.src_img_pts.size(0)
        N = t.src_img_pts.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = t.trg_img_pts
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = t.src_img_pts
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Compute actual cost at ground truth solution
        cost_true = np.zeros(B)
        for b in range(B):
            src = srcs[b].cpu().numpy()
            trg = trgs[b].cpu().numpy()
            F = t.Fs[b].cpu().numpy()
            for i in range(N):
                cost_true[b] += (
                    t.weights[b, :, i] * (trg[:, [i]].T @ F @ src[:, [i]]) ** 2
                )

        # Construct objective matrix
        Q, _, _ = FundMatSDPBlock.get_obj_matrix_vec(
            srcs, trgs, t.weights, scale_offset=False
        )
        # Check that matrix does the same thing
        cost_mat = t.sol.mT.bmm(Q.bmm(t.sol))[:, 0, 0]
        np.testing.assert_allclose(
            cost_mat,
            cost_true,
            atol=1e-12,
            err_msg="Matrix cost not equal to true cost",
        )

    def test_feasibility(self):
        """Test that the sdpr localization properly estimates the target
        transformation under no noise condition"""

        # zero the weights and run the problem
        wts = t.weights * torch.tensor(0.0)
        Fs_est, es_est = self.layer(
            t.src_img_pts, t.trg_img_pts, wts, rescale=False, verbose=True
        )
        # Check properties of solution
        print("done")

    def test_nuclear_norm(self):
        """Test optimization under nuclear norm"""

        # zero the weights and run the problem
        wts = t.weights * torch.tensor(0.0)
        Fs_est, es_est, X = self.layer(
            t.src_img_pts, t.trg_img_pts, wts, rescale=False, reg=1e3, verbose=True
        )
        # Check properties of solution
        u, s, vh = np.linalg.svd(X[0])
        plt.semilogy(s, ".")
        plt.title("singular values under nuclear norm")
        plt.show()

    def test_sdpr_forward_nonoise(self):
        """Test that the sdpr localization properly estimates the target
        transformation under no noise condition"""

        self.test_sdpr_forward(sigma_val=0.0)

    def test_sdpr_forward(self, sigma_val=5.0):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Sizes
        B = t.src_img_pts.size(0)
        N = t.src_img_pts.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = t.trg_img_pts
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = t.src_img_pts
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        Fs_est, es_est, X = self.layer(
            srcs,
            trgs,
            t.weights,
            verbose=True,
            rescale=False,
            reg=100.0,
        )
        # Check Solution Rank
        u, s, v = np.linalg.svd(X[0])
        plt.semilogy(s, ".")
        plt.show()

    def test_sparse_solve(self):
        problem = HomQCQP(homog_var="h")
        var_sizes = self.layer.var_dict
        problem.var_sizes = var_sizes
        # Get cost function
        trgs = t.trg_img_pts
        srcs = t.src_img_pts
        C_torch, _, _ = self.layer.get_obj_matrix_vec(
            srcs, trgs, self.weights, scale_offset=False
        )
        C = C_torch[0].numpy()
        problem.C = PolyMatrix.init_from_sparse(C, var_sizes, symmetric=True)
        # Get constraints
        problem.As = []
        for A in self.layer.sdprlayer.constr_list:
            pmat = PolyMatrix.init_from_sparse(A, var_sizes, symmetric=True)
            problem.As.append(pmat)
        # decompose
        problem.clique_decomposition()
        # Solve
        X_list, info = solve_dsdp(problem, form="dual", verbose=True, tol=1e-10)
        Y, ranks, factor_dict = problem.get_mr_completion(
            X_list, var_list=list(var_sizes.keys())
        )
        assert Y.shape[1] == 1, "solution not rank-1"


if __name__ == "__main__":

    # Frobenius Norm Formulation
    # t = TestFundMat()
    # Unity element constraint
    t = TestFundMat(formulation="unity_elem")

    # t.test_constraints()
    # t.test_cost_matrix_nonoise()
    # t.test_cost_matrix()
    # t.test_feasibility()
    # t.test_nuclear_norm()
    # t.test_sdpr_forward_nonoise()
    # t.test_sdpr_forward()
    t.test_sparse_solve()
