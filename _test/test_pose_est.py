import unittest

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame

from sdprlayers import LieOptPoseEstimator, SDPPoseEstimator, SVDPoseEstimator
from sdprlayers.utils.keypoint_tools import get_inv_cov_weights
from sdprlayers.utils.lie_algebra import se3_exp, se3_inv, se3_log
from sdprlayers.utils.plot_tools import plot_ellipsoid
from sdprlayers.utils.pose_est_grad_compare import (
    get_experiment_data,
    get_soln_and_jac,
    process_grad_data,
)
from sdprlayers.utils.stereo_camera_model import StereoCameraModel
from sdprlayers.utils.stereo_tuner import get_gt_setup

# matplotlib.use("TkAgg")


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestLocalize(unittest.TestCase):
    def __init__(t, *args, **kwargs):
        super(TestLocalize, t).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        t.device = "cuda:0"
        # Set seed
        set_seed(0)
        batch_size = 1
        # Set up test problem
        r_v0s, C_v0s, r_ls = get_gt_setup(
            N_map=50, N_batch=batch_size, traj_type="circle"
        )
        r_v0s = torch.tensor(r_v0s)
        C_v0s = torch.tensor(C_v0s)
        r_ls = torch.tensor(r_ls)[None, :, :].expand(batch_size, -1, -1)
        # Define Stereo Camera
        stereo_cam = StereoCameraModel(0.0, 0.0, 484.5, 0.24).cuda()
        # Frame tranform from vehicle to camera (sensor)
        pert = 0.01
        xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]])
        T_s_v = se3_exp(xi_pert)[0]

        # Generate image coordinates (in vehicle frame)
        cam_coords_v = torch.bmm(C_v0s, r_ls - r_v0s)
        cam_coords_v = torch.concat(
            [cam_coords_v, torch.ones(batch_size, 1, r_ls.size(2))], dim=1
        )

        # Source coords in vehicle frame
        src_coords_v = torch.concat(
            [r_ls, torch.ones(batch_size, 1, r_ls.size(2))], dim=1
        )
        # Map to camera frame
        cam_coords = T_s_v[None, :, :].bmm(cam_coords_v)
        src_coords = T_s_v[None, :, :].bmm(src_coords_v)
        # Create transformation matrix
        zeros = torch.zeros(batch_size, 1, 3).type_as(r_v0s)  # Bx1x3
        one = torch.ones(batch_size, 1, 1).type_as(r_v0s)  # Bx1x1
        r_0v_v = -C_v0s.bmm(r_v0s)
        trans_cols = torch.cat([r_0v_v, one], dim=1)  # Bx4x1
        rot_cols = torch.cat([C_v0s, zeros], dim=1)  # Bx4x3
        T_trg_src = torch.cat([rot_cols, trans_cols], dim=2)  # Bx4x4
        # Store values
        t.keypoints_3D_src = src_coords.cuda()
        t.keypoints_3D_trg = cam_coords.cuda()
        t.T_trg_src = T_trg_src
        t.stereo_cam = stereo_cam
        # Generate Scalar Weights
        t.weights = torch.ones(
            t.keypoints_3D_src.size(0), 1, t.keypoints_3D_src.size(2)
        ).cuda()
        t.stereo_cam = stereo_cam
        t.T_s_v = T_s_v.cuda()

    def test_svd_forward(t):
        """Test that the SVD Block properly estimates the target transformation"""

        # Instantiate
        svd_block = SVDPoseEstimator(t.T_s_v)
        # Run forward with data

        T_trg_src = svd_block(t.keypoints_3D_src, t.keypoints_3D_trg, t.weights)
        # Check that
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-12)

    def test_sdpr_forward(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Instantiate
        sdpr_block = SDPPoseEstimator(t.T_s_v)
        T_trg_src = sdpr_block(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, verbose=True
        )
        # Check that
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=5e-7)

    def test_sdpr_mat_weight_cost(t):
        """Test that the sdpr localization properly estimates the target
        transformation"""
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights > 0
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )
        # Instantiate
        sdpr_block = SDPPoseEstimator(t.T_s_v)
        Q, scales, offsets = sdpr_block.get_obj_matrix(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )
        Q_vec, scales_vec, offsets_vec = sdpr_block.get_obj_matrix_vec(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )
        # Check that
        np.testing.assert_allclose(Q.cpu().numpy(), Q_vec.cpu().numpy(), atol=1e-15)

    def test_sdpr_mat_weight_forward(t):
        """Test that the sdpr localization properly estimates the target
        transformation. Use matrix weights."""
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights > 0
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )

        # Instantiate
        sdpr_block = SDPPoseEstimator(t.T_s_v)
        T_trg_src = sdpr_block(
            t.keypoints_3D_src,
            t.keypoints_3D_trg,
            t.weights,
            inv_cov_weights,
            verbose=True,
        )

        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-6)

    def test_grad_Q(self):
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights > 0
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )
        # Instantiate
        sdpr_block = SDPPoseEstimator(t.T_s_v)

        Q, _, _ = sdpr_block.get_obj_matrix_vec(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, inv_cov_weights
        )
        Q.requires_grad_(True)

        tol = 1e-12
        mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": tol,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": tol * 1e-2,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": tol,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": tol,
        }

        solver_args = {
            "solve_method": "mosek",
            "mosek_params": mosek_params,
            "verbose": True,
        }

        def forward_mat(Q_mat):
            X, x = sdpr_block.sdprlayer(Q_mat, solver_args=solver_args)
            return x

        torch.autograd.gradcheck(
            func=forward_mat, eps=1e-5, atol=1e-10, rtol=1e-2, inputs=[Q]
        )

    def test_lieopt_forward(t):
        """Test that the local localization properly estimates the target
        transformation"""

        # Instantiate
        lie_block = LieOptPoseEstimator(t.T_s_v, 1, 50)
        # lie_block.to(t.device)
        lie_block.cuda()
        # Run with ground truth initialization
        T_trg_src = lie_block(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, t.T_trg_src.cuda()
        )
        # Check that the difference is small
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-8)

        # Define perturbation
        pert = 0.1
        xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]])
        T_pert = se3_exp(xi_pert)
        T_init = T_pert.bmm(t.T_trg_src)
        # Run with perturbed starting point
        T_trg_src = lie_block(
            t.keypoints_3D_src, t.keypoints_3D_trg, t.weights, T_init.cuda()
        )
        # Check that the difference is small
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-10)

    def test_lieopt_mat_weight_forward(t):
        """Test that the lie algebra localization properly estimates the target
        transformation. Use matrix weights."""
        # Get matrix weights - assuming 0.5 pixel std dev
        valid = t.weights > 0
        inv_cov_weights, cov = get_inv_cov_weights(
            t.keypoints_3D_trg, valid, t.stereo_cam
        )

        # Instantiate
        lie_block = LieOptPoseEstimator(t.T_s_v, 1, 50)
        lie_block.to(t.device)
        # Test with ground truth initialization
        T_trg_src = lie_block(
            t.keypoints_3D_src,
            t.keypoints_3D_trg,
            t.weights,
            t.T_trg_src.cuda(),
            inv_cov_weights,
            verbose=True,
        )
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-7)
        # Define perturbation
        pert = 0.5
        xi_pert = torch.tensor([[pert, pert, pert, pert, pert, pert]])
        T_pert = se3_exp(xi_pert)
        T_init = T_pert.bmm(t.T_trg_src)
        # Test with perturbed starting point
        T_trg_src = lie_block(
            t.keypoints_3D_src,
            t.keypoints_3D_trg,
            t.weights,
            T_init.cuda(),
            inv_cov_weights,
            verbose=True,
        )
        # Check that the difference is small
        diff = se3_log(se3_inv(T_trg_src.cpu()).bmm(t.T_trg_src)).numpy()
        np.testing.assert_allclose(diff, np.zeros((1, 6)), atol=1e-8)

    def test_inv_cov_numerical(t, plot=False):
        N_pts = 1000
        # get random point
        pt = torch.tensor([3.0, 3.0, 3.0, 1.0])[None, :, None].cuda()
        # Convert to pixel space
        pixel_gt = t.stereo_cam.camera_model(pt)
        # Generate noise in pixel space
        noise_pxl = torch.randn(1, 4, N_pts) * t.stereo_cam.sigma
        pixel_noisy = pixel_gt.expand(1, -1, N_pts) + noise_pxl.cuda()
        # Convert back to 3D
        disparity = pixel_noisy[0, 0, :] - pixel_noisy[0, 2, :]
        pt_noisy = t.get_cam_points(pixel_noisy, disparity, t.stereo_cam.Q)
        noise_pt = (pt_noisy - torch.mean(pt_noisy, dim=2, keepdim=True))[:, :3, :]
        cov = torch.matmul(noise_pt, noise_pt.transpose(1, 2)) / (N_pts - 1)
        # Compute inverse covariance
        valid = (torch.ones(1, 1, 1) > 0).cuda()
        W, cov_cam = get_inv_cov_weights(pt, valid, t.stereo_cam)
        # Move back to cpu
        cov_cam = cov_cam[0, 0].cpu().detach().numpy()
        cov = cov[0].cpu().detach().numpy()
        pt = pt.cpu().detach().numpy()
        noise_pt = noise_pt.cpu().detach().numpy()
        if plot:
            # Plot covariance
            plt.figure()
            ax = plt.axes(projection="3d")
            # plot measurements in camera frame
            plot_ellipsoid(np.zeros((3, 1)), cov_cam, ax=ax, color="r")
            plot_ellipsoid(np.zeros((3, 1)), cov, ax=ax, color="b")
            ax.scatter3D(
                noise_pt[0, 0, :],
                noise_pt[0, 1, :],
                noise_pt[0, 2, :],
                marker=".",
                color="black",
                alpha=0.5,
            )
            plt.show()

        # Compare covariances
        np.testing.assert_allclose(cov_cam, cov, atol=1e-4)

    @staticmethod
    def get_cam_points(img_coords, point_disparities, Q):

        batch_size, _, num_points = img_coords.size()
        point_disparities = point_disparities.reshape(
            batch_size, 1, num_points
        )  # Bx1xN

        # Create the [ul, vl, d, 1] vector
        ones = torch.ones(batch_size, num_points).type_as(point_disparities)
        uvd1_pixel_coords = torch.stack(
            (
                img_coords[:, 0, :],
                img_coords[:, 1, :],
                point_disparities[:, 0, :],
                ones,
            ),
            dim=1,
        )  # Bx4xN

        # [X, Y, Z, d]^T = Q * [ul, vl, d, 1]^T
        Q_b = Q.expand(batch_size, 4, 4).cuda()
        cam_coords = Q_b.bmm(uvd1_pixel_coords)  # Bx4xN

        # [x, y, z, 1]^T = (1/d) * [X, Y, Z, d]^T
        inv_disparity = 1.0 / point_disparities  # Elementwise division
        cam_coords = cam_coords * inv_disparity  # Elementwise multiplication

        return cam_coords

    def test_inv_cov_weights(t):
        import matplotlib

        matplotlib.use("TkAgg")

        valid = t.weights > 0
        # test "valid" masking
        valid[0, 0, 0] = torch.tensor(0)
        W, cov_cam = get_inv_cov_weights(t.keypoints_3D_trg, valid, t.stereo_cam)
        id = torch.eye(3).cuda()
        assert torch.all(
            W[0, 0, :, :] == torch.zeros((3, 3)).cuda()
        ), "Invalid mask not working"
        assert W.size() == (1, 50, 3, 3), "Size Incorrect"

    def test_grads(t):
        """Test Gradients generated by SDPRLayers for scalar-weighted pose estimation problem.
        Test for different gradient generation schemes."""

        # Different methods for computing gradients:
        estimator_list = [
            "sdpr-sdp",
            "sdpr-cift",
            "sdpr-is",
        ]
        tols = {
            "sdpr-sdp": 2e-4,
            "sdpr-cift": 5e-5,
            "sdpr-is": 5e-5,
        }
        # Generate and process grad data
        df = t.generate_grad_data(experiment="stereo-sclwt")
        jac_diffs = process_grad_data(
            df=df, experiment="stereo-sclwt", return_diffs=True
        )
        # Loop over estimators
        for estimator in estimator_list:
            # Test that the infinity norm of the jacobian is below an acceptable level.
            np.testing.assert_allclose(
                jac_diffs[estimator],
                np.zeros(jac_diffs[estimator].shape),
                atol=tols[estimator],
                err_msg=f"Error too large for {estimator}",
            )

    def generate_grad_data(
        t, experiment="stereo-sclwt", n_batch=50, n_points=30, noise_std=0.5, **opts
    ):
        """A function for comparing gradients generated by SDPRLayers for the pose estimation problem"""
        # parameters

        # Set seeds for reproducability
        torch.manual_seed(0)
        np.random.seed(0)

        # Generate experiment data
        points_s, points_t, weights, T_t_s_gt, mat_wts = get_experiment_data(
            n_batch=n_batch,
            n_points=n_points,
            noise_std=noise_std,
            experiment=experiment,
        )

        # Assess estimators
        estimator_list = [
            "svd",
            "sdpr-sdp",
            "sdpr-cift",
            "sdpr-is",
        ]

        data_dicts = []
        for iEst, estimator in enumerate(estimator_list):
            T_t_s_est, jacobians, time_f, time_b = get_soln_and_jac(
                estimator, points_t, points_s, weights, mat_wts, T_t_s_gt, **opts
            )
            # Compute distance from ground truth value
            xi_err = se3_log(se3_inv(T_t_s_est) @ T_t_s_gt)

            # Store data
            data_dict = dict(
                estimator=estimator,
                T_t_s_est=T_t_s_est,
                jacobians=jacobians,
                xi_err=xi_err,
                time_f=time_f,
                time_b=time_b,
            )
            data_dicts.append(data_dict)
        df = DataFrame(data_dicts)

        return df


if __name__ == "__main__":
    t = TestLocalize()
    # t.test_sdpr_mat_weight_forward()
    # t.test_sdpr_mat_weight_cost()
    # t.test_sdpr_forward()
    # t.test_inv_cov_weights()
    # t.test_inv_cov_numerical(plot=True)
    # t.test_svd_forward()
    # t.test_lieopt_forward()
    # t.test_lieopt_mat_weight_forward()
    # t.test_grad_Q()
    t.test_grads()
