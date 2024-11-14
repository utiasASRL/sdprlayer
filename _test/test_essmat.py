import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from cert_tools import HomQCQP
from cert_tools.sparse_solvers import solve_dsdp
from poly_matrix import PolyMatrix

import sdprlayers.utils.fund_mat_utils as utils
from sdprlayers import EssentialSDPBlock
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import so3_wedge

# matplotlib.use("TkAgg")


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestEssMat(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEssMat, self).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        self.device = "cuda:0"
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
        # t_tss = np.array([[1.0, 0.1, 0.0]]).T[None, :, :]
        # Transforms from source to target
        t_tss = torch.tensor(t_tss)
        R_tss = torch.tensor(R_tss)
        R_sts = R_tss.mT
        self.R_tss = R_tss
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

        # Apply camera to get image points
        src_img_pts = camera.camera_model(src_coords)
        trg_img_pts = camera.camera_model(trg_coords)
        # Get inverse intrinsic camera mat
        K_inv = torch.linalg.inv(camera.K)
        K_invs = K_inv.expand(batch_size, 3, 3)
        # Store normalized image coordinates
        self.keypoints_src = K_invs.bmm(src_img_pts[:, :3, :])
        self.keypoints_trg = K_invs.bmm(trg_img_pts[:, :3, :])

        # Generate Scalar Weights
        self.weights = (
            torch.ones(self.keypoints_src.size(0), 1, self.keypoints_src.size(2))
            / n_points
        )
        self.camera = camera

        # Normalize the translations
        t_norm = torch.norm(t_tss)
        self.t_tss = t_tss / t_norm
        # Construct Essential Matrix
        self.Es = so3_wedge(self.t_tss).bmm(R_sts)
        check = (
            self.keypoints_src[0, :3, [0]].cpu().mT
            @ self.Es[0]
            @ self.keypoints_trg[0, :3, [0]].cpu()
        )
        np.testing.assert_allclose(check, 0.0, atol=1e-12)

        # Construct solution vectors
        self.sol = torch.cat(
            [
                torch.ones((batch_size, 1, 1)),
                torch.reshape(self.Es, (-1, 9, 1)),  # row-major vectorization
                self.t_tss,
            ],
            dim=1,
        )
        # Initialize layer
        self.layer = EssentialSDPBlock()

    def test_constraints(self):
        """Test that the constraints characterize the fundamental matrix and epipole."""

        # Get constraints and test
        constraints = self.layer.sdprlayer.constr_list
        viol = np.zeros((len(constraints)))
        sol = np.array(t.sol[0])
        for i, A in enumerate(constraints):
            viol[i] = (sol.T @ A @ sol)[0, 0]
            np.testing.assert_allclose(
                viol[i], 0.0, atol=1e-8, err_msg=f"Constraint {i+1} has violation"
            )

    def test_cost_matrix_nonoise(self):
        """Test the objective matrix with no noise"""
        self.test_cost_matrix(sigma_val=0.0)

    def test_cost_matrix(self, sigma_val=5.0):
        """Test the objective matrix with no noise"""
        # Sizes
        B = t.keypoints_src.size(0)
        N = t.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = t.keypoints_trg
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = t.keypoints_src
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Compute actual cost at ground truth solution
        cost_true = np.zeros(B)
        for b in range(B):
            src = srcs[b].cpu().numpy()
            trg = trgs[b].cpu().numpy()
            E = t.Es[b].cpu().numpy()
            for i in range(N):
                cost_true[b] += (
                    t.weights[b, :, i] * (src[:, [i]].T @ E @ trg[:, [i]]) ** 2
                )

        # Construct objective matrix
        Q, _, _ = EssentialSDPBlock.get_obj_matrix_vec(
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
        E_mats, t_vecs, X = self.layer(
            t.keypoints_src, t.keypoints_trg, wts, rescale=False, verbose=True
        )
        # Check properties of solution
        print("done")

    def test_sdpr_forward_nonoise(self, plot=False):
        """Test that the sdpr localization properly estimates the target
        transformation under no noise condition"""

        self.test_sdpr_forward(sigma_val=0.0, plot=plot)

    def test_sdpr_forward(self, sigma_val=5.0, plot=False):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Sizes
        B = t.keypoints_src.size(0)
        N = t.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = t.keypoints_trg
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = t.keypoints_src
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        Es_est, ts_est, X, rank = self.layer(
            srcs,
            trgs,
            t.weights,
            verbose=True,
            rescale=False,
        )
        # Check Solution Rank
        if plot:
            u, s, v = np.linalg.svd(X[0])
            plt.semilogy(s, ".")
            plt.show()

        # Check Rank
        assert rank == 1, ValueError("Rank of solution is not 1")

        # Check that estimate matches actual
        E_est = Es_est[0].numpy()
        t_est = ts_est[0].numpy()
        E_gt = self.Es[0].numpy()
        t_gt = self.t_tss[0].numpy()

        # If no noise, check that we are close to solution
        if sigma_val == 0.0:
            # Check sign ambiguity
            if np.linalg.norm(E_gt - E_est) > np.linalg.norm(E_gt + E_est):
                E_est = -E_est
            if np.linalg.norm(t_est - E_est) > np.linalg.norm(t_est + E_est):
                t_est = -t_est
            np.testing.assert_allclose(t_est, t_gt, atol=1e-4)
            np.testing.assert_allclose(E_est, E_gt, atol=1e-4)
        else:
            # Otherwise, check that the cost is better than the ground truth cost
            b = 0
            src = srcs[b].cpu().numpy()
            trg = trgs[b].cpu().numpy()
            E = t.Es[b].cpu().numpy()
            cost_gt = 0.0
            cost_est = 0.0
            for i in range(N):
                cost_gt += (
                    self.weights[b, :, i] * (src[:, [i]].T @ E_gt @ trg[:, [i]]) ** 2
                )
                cost_est += (
                    self.weights[b, :, i] * (src[:, [i]].T @ E_est @ trg[:, [i]]) ** 2
                )
            assert cost_est < cost_gt, ValueError(
                "Estimate cost not lower than ground truth cost"
            )


if __name__ == "__main__":
    # Unity element constraint
    t = TestEssMat()

    # t.test_constraints()
    # t.test_cost_matrix_nonoise()
    # t.test_cost_matrix()
    # t.test_feasibility()
    # t.test_sdpr_forward_nonoise()
    t.test_sdpr_forward(sigma_val=200 / 800)
