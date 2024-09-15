import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

import sdprlayer.utils.fund_mat_utils as utils
from sdprlayer.fundmat_est import FundMatSDPBlock
from sdprlayer.utils.camera_model import CameraModel
from sdprlayer.utils.lie_algebra import so3_wedge

matplotlib.use("TkAgg")


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestFundMat(unittest.TestCase):

    def __init__(t, *args, **kwargs):
        super(TestFundMat, t).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        t.device = "cuda:0"
        # Set seed
        set_seed(0)
        batch_size = 1
        # Set up test problem
        # NOTE t_ts is the vector from the source to the target frame expressed in the source frame
        # NOTE R_ts is the rotation that maps vectors in the source frame to vectors in the target frame
        # NOTE trailing "s" indicates that a variable is batched
        t_tss, R_tss, key_ss = utils.get_gt_setup(
            N_map=50, N_batch=batch_size, traj_type="circle"
        )

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
        camera = CameraModel(484.5, 484.5, 0.0, 0.0, 0.0).cuda()
        # Apply camera to get image points
        t.src_img_pts = camera.camera_model(src_coords).cuda()
        t.trg_img_pts = camera.camera_model(trg_coords).cuda()
        # Store values
        t.keypoints_3D_src = src_coords.cuda()
        t.keypoints_3D_trg = trg_coords.cuda()

        # Generate Scalar Weights
        t.weights = torch.ones(
            t.keypoints_3D_src.size(0), 1, t.keypoints_3D_src.size(2)
        ).cuda()
        t.camera = camera

        # Construct Essential Matrix
        Es = so3_wedge(t_tss).bmm(R_tss)
        # Get intrinsic camera mat
        K_inv = torch.linalg.inv(camera.K)
        K_invs = K_inv.expand(batch_size, 3, 3)
        # Construct (Normalized) Fundamental matrices
        Fs_unnorm = K_invs.mT.bmm(Es).bmm(K_invs)
        norm = torch.linalg.norm(Fs_unnorm, keepdim=True)
        t.Fs = Fs_unnorm / norm
        # Construct epipoles
        u, s, vh = torch.linalg.svd(t.Fs)
        t.es = vh[:, [2], :].mT
        # Construct solution vectors
        t.sol = torch.cat(
            [
                torch.reshape(t.Fs.mT, (-1, 9, 1)),
                t.es,
                torch.ones((batch_size, 1, 1)),
            ],
            dim=1,
        )

    def test_constraints(self):
        """Test that the constraints characterize the fundamental matrix and epipole."""

        # init fundamental matrix estimator
        fund_est = FundMatSDPBlock()
        constraints = fund_est.sdprlayer.constr_list
        viol = np.zeros((len(constraints)))
        sol = np.array(t.sol[0])
        for i, A in enumerate(constraints):
            viol[i] = sol.T @ A @ sol
            np.testing.assert_allclose(
                viol[i], 0.0, atol=1e-12, err_msg=f"Constraint {i} has violation"
            )

    def test_cost_matrix(self):
        """Test the cost matrix"""
        pass

    # def test_sdpr_forward(t):
    #     """Test that the sdpr localization properly estimates the target
    #     transformation"""

    #     # Instantiate
    #     fundmat_est = FundMatSDPBlock()
    #     F, e = fundmat_est(
    #         t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, verbose=True
    #     )

    #     # TODO Check difference
    #     np.testing.assert_allclose(F_est, F_actual, atol=5e-7)


if __name__ == "__main__":
    t = TestFundMat()
    t.test_constraints()
