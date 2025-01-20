import time
import unittest

import kornia.geometry.epipolar as epi
import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import findEssentialMat
from pandas import DataFrame
from tqdm import tqdm

import sdprlayers.utils.ess_mat_utils as utils
from sdprlayers import SDPEssMatEst
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import se3_exp, so3_wedge
from sdprlayers.utils.plot_tools import plot_map, plot_poses


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestEssMat(unittest.TestCase):

    def __init__(self, *args, n_batch=1, n_points=50, tol=1e-12, **kwargs):
        super(TestEssMat, self).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        self.device = "cuda:0"
        # Set seed
        set_seed(0)
        # Set up test problem
        # NOTE ts_ts_s is the vector from the source to the target frame expressed in the source frame
        # NOTE Rs_ts is the rotation that maps vectors in the source frame to vectors in the target frame
        ts_ts_s, Rs_ts, key_ss = utils.get_gt_setup(
            N_map=n_points,
            N_batch=n_batch,
            traj_type="clusters",
            offs=np.array([[0, 0, 3]]).T,
        )
        # Transforms from source to target
        Rs_ts = torch.tensor(Rs_ts)
        ts_ts_s = torch.tensor(ts_ts_s)
        ts_st_t = Rs_ts.bmm(-ts_ts_s)
        # store rotation solution
        self.Rs_ts = Rs_ts

        # Keypoints (3D) defined in source frame
        key_ss = torch.tensor(key_ss)[None, :, :].expand(n_batch, -1, -1)
        # Keypoints in target frame
        key_ts = Rs_ts.bmm(key_ss) + ts_st_t
        # homogenize coordinates
        trg_coords = torch.concat(
            [key_ts, torch.ones(n_batch, 1, key_ss.size(2))], dim=1
        )
        src_coords = torch.concat(
            [key_ss, torch.ones(n_batch, 1, key_ss.size(2))], dim=1
        )

        # Define Camera
        # camera = CameraModel(800, 800, 0.0, 0.0, 0.0)
        camera = CameraModel(1, 1, 0.0, 0.0, 0.0)

        # Apply camera to get image points
        src_img_pts = camera.camera_model(src_coords)
        trg_img_pts = camera.camera_model(trg_coords)
        # Get inverse intrinsic camera mat
        K_inv = torch.linalg.inv(camera.K)
        K_invs = K_inv.expand(n_batch, 3, 3)
        # Store normalized image coordinates
        self.keypoints_src = K_invs.bmm(src_img_pts)
        self.keypoints_trg = K_invs.bmm(trg_img_pts)

        # Generate Scalar Weights
        self.weights = torch.ones(
            self.keypoints_src.size(0), 1, self.keypoints_src.size(2)
        )
        self.camera = camera

        # Normalize the translations
        t_norm = torch.norm(ts_st_t, dim=1, keepdim=True)
        self.ts_st_t = ts_st_t / t_norm
        # Construct Essential Matrix
        self.Es = so3_wedge(self.ts_st_t[:, :, 0]).bmm(self.Rs_ts)

        # Check that the matrix makes sense
        check = 0.0
        for n in range(n_batch):
            check += (
                self.keypoints_trg[0, :, [n]].mT
                @ self.Es[0]
                @ self.keypoints_src[0, :, [n]]
            )

        np.testing.assert_allclose(check, 0.0, atol=1e-12)

        # Construct solution vectors
        self.sol = torch.cat(
            [
                torch.ones((n_batch, 1, 1)),
                torch.reshape(self.Es, (-1, 9, 1)),  # row-major vectorization
                self.ts_st_t,
            ],
            dim=1,
        )
        K_batch = self.camera.K.expand(1, -1, -1)
        # Initialize layer
        self.layer = SDPEssMatEst(tol=tol, K_source=K_batch, K_target=K_batch)

    def test_constraints(self, sol=None):
        """Test that the constraints characterize the fundamental matrix and epipole."""
        if sol is None:
            sol = np.array(self.sol[0])

        # Get constraints and test
        constraints = self.layer.sdprlayer.constr_list
        viol = np.zeros((len(constraints)))
        for i, A in enumerate(constraints):
            viol[i] = (sol.T @ A @ sol)[0, 0]
            np.testing.assert_allclose(
                viol[i], 0.0, atol=1e-8, err_msg=f"Constraint {i+1} has violation"
            )

    def test_cost_matrix_nonoise(self):
        """Test the objective matrix with no noise"""
        self.test_cost_matrix(sigma_val=0.0)

    def test_cost_matrix(self, sigma_val=10 / 800):
        """Test the objective matrix with no noise"""
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        # trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        # srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Compute actual cost at ground truth solution
        cost_true = np.zeros(B)
        for b in range(B):
            src = srcs[b].cpu().numpy()
            trg = trgs[b].cpu().numpy()
            E = self.Es[b].cpu().numpy()
            for i in range(N):
                cost_true[b] += (
                    self.weights[b, :, i] * (trg[:, [i]].T @ E @ src[:, [i]]) ** 2
                )

        # Construct objective matrix - (No scaling)
        Q, scale, offs = SDPEssMatEst.get_obj_matrix_vec(
            srcs, trgs, self.weights, scale_offset=False
        )

        for b in range(B):
            Q_list = []
            rows = []
            for n in range(N):
                # Compute Kronecker product
                row = (trgs[b, :, [n]] @ srcs[[b], :, n]).reshape(1, -1)
                Q_list.append(row.mT @ row)
                rows.append(row)
            Q_test = sum(Q_list)
            rows = torch.cat(rows, 0)
            np.testing.assert_allclose(
                Q[b][1:10][:, 1:10].numpy(),
                Q_test.numpy(),
                atol=1e-12,
                err_msg="Cost matrix does not match cost function",
            )

        # Check that matrix does the same thing
        cost_mat = self.sol.mT.bmm(Q.bmm(self.sol))[:, 0, 0]
        np.testing.assert_allclose(
            cost_mat,
            cost_true,
            atol=1e-12,
            err_msg="Matrix cost not equal to true cost",
        )

        # Construct objective matrix - (with scaling)
        Q, scale, offs = SDPEssMatEst.get_obj_matrix_vec(
            srcs, trgs, self.weights, scale_offset=True
        )

        # Check that matrix does the same thing
        cost_mat = self.sol.mT.bmm(Q.bmm(self.sol))[:, 0, 0] * scale + offs
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
        wts = self.weights * torch.tensor(0.0)
        E_mats, t_vecs, X, rank = self.layer(
            self.keypoints_src, self.keypoints_trg, wts, rescale=False, verbose=True
        )

    def test_sdpr_forward_nonoise(self, plot=False):
        """Test that the sdpr localization properly estimates the target
        transformation under no noise condition"""

        self.test_sdpr_forward(sigma_val=0.0, plot=plot)

    def test_sdpr_forward(self, sigma_val=1e-2, plot=False):
        """Test that the sdpr localization properly estimates the target
        transformation"""

        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        Es_est, Rs_est, ts_est, X, rank = self.layer(
            srcs,
            trgs,
            self.weights,
            verbose=True,
            rescale=True,
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
        t_gt = self.ts_st_t[0].numpy()

        # If no noise, check that we are close to solution
        if sigma_val == 0.0:
            # Check sign ambiguity (only required when no noise)
            if np.linalg.norm(E_gt - E_est) > np.linalg.norm(E_gt + E_est):
                E_est = -E_est
            if np.linalg.norm(t_gt - t_est) > np.linalg.norm(t_gt + t_est):
                t_est = -t_est
            np.testing.assert_allclose(E_est, E_gt, atol=1e-5)
            np.testing.assert_allclose(t_est, t_gt, atol=1e-5)
        else:
            # Otherwise, check that the cost is better than the ground truth cost
            b = 0
            src = srcs[b].cpu().numpy()
            trg = trgs[b].cpu().numpy()
            E_gt = self.Es[b].cpu().numpy()
            cost_gt = 0.0
            cost_est = 0.0
            for i in range(N):
                cost_gt += (
                    self.weights[b, :, i] * (trg[:, [i]].T @ E_gt @ src[:, [i]]) ** 2
                )
                cost_est += (
                    self.weights[b, :, i] * (trg[:, [i]].T @ E_est @ src[:, [i]]) ** 2
                )
            assert cost_est < cost_gt, ValueError(
                "Estimate cost not lower than ground truth cost"
            )

    def test_layer_backward(self, sigma_val=0):
        """Test backpropagation through layer

        Args:
            sigma_val (float, optional): _description_. Defaults to 5.0.
        """
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        def layer_wrapper(new_src, new_trg, new_wt, out="E"):
            # Homogenize input image tensor
            one = torch.tensor([[1]])
            new_trg_h = torch.cat((new_trg, one), dim=1).unsqueeze(2)
            new_src_h = torch.cat((new_src, one), dim=1).unsqueeze(2)
            new_wt = new_wt.unsqueeze(2)
            # Stack with other measurements + weights
            new_srcs = torch.cat([srcs[:, :, :-1], new_src_h], dim=2)
            new_trgs = torch.cat([trgs[:, :, :-1], new_trg_h], dim=2)
            new_weights = torch.cat([self.weights[:, :, :-1], new_wt], dim=2)
            Es_est, ts_est, X, rank = self.layer(
                new_srcs,
                new_trgs,
                new_weights,
                rescale=False,
                verbose=False,
            )

            assert rank == 1, ValueError("Solution is not rank 1")
            if out == "E":
                return Es_est
            elif out == "t":
                return ts_est
            else:
                return None

        # Test backward gradient. We only modify the last image points and weight
        inputs = (
            srcs[:, :2, -1].clone().detach().requires_grad_(False),
            trgs[:, :2, -1].clone().detach().requires_grad_(False),
            self.weights[:, :2, -1].clone().detach().requires_grad_(False),
        )

        # source-essential gradient
        inputs[0].requires_grad_(True)
        eps = 1e-3
        atol = 1e-5
        rtol = 5e-2
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="E"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        # source-translation gradient
        eps = 1e-3
        atol = 1e-5
        rtol = 5e-2
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="t"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        inputs[0].requires_grad_(False)

        # target-essential gradient
        inputs[1].requires_grad_(True)
        eps = 1e-3
        atol = 1e-5
        rtol = 5e-2
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="E"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        # target-translation gradient
        eps = 1e-3
        atol = 1e-5
        rtol = 1e-2
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="t"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        inputs[1].requires_grad_(False)

        # weight-essential gradient
        inputs[2].requires_grad_(True)
        eps = 1e-3
        atol = 1e-5
        rtol = 1e-1
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="E"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        # weight-translation gradient
        eps = 1e-3
        atol = 1e-5
        rtol = 1e-1
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="t"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        inputs[2].requires_grad_(False)

    def test_cost_backward(self, sigma_val=0):
        """Test backpropagation through layer

        Args:
            sigma_val (float, optional): _description_. Defaults to 5.0.
        """
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        def cost_wrapper(new_src, new_trg, new_wt):
            # Homogenize input image tensor
            one = torch.tensor([[[1]]])
            new_trg_h = torch.cat((new_trg, one), dim=1)
            new_src_h = torch.cat((new_src, one), dim=1)
            # Stack with other measurements + weights
            new_srcs = torch.cat([srcs, new_src_h], dim=2)
            new_trgs = torch.cat([trgs, new_trg_h], dim=2)
            new_weights = torch.cat([self.weights, new_wt], dim=2)
            Q, _, _ = self.layer.get_obj_matrix_vec(
                new_srcs,
                new_trgs,
                new_weights,
                scale_offset=False,
            )
            return Q

        # Test backward
        inputs = (
            torch.tensor(
                np.random.randn(1, 2, 1), requires_grad=False, dtype=torch.float64
            ),
            torch.tensor(
                np.random.randn(1, 2, 1), requires_grad=False, dtype=torch.float64
            ),
            torch.tensor(np.array([[[1]]]), requires_grad=True, dtype=torch.float64),
        )
        eps = 1e-6
        atol = 1e-10
        rtol = 1e-5
        torch.autograd.gradcheck(
            lambda *x: cost_wrapper(*x), inputs=inputs, eps=eps, atol=atol, rtol=rtol
        )

    def test_kornia_solution(self, sigma_val=0.0):
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Get essential matrix, and extract rotation, translation
        Es, ts, Rs = utils.get_kornia_solution(
            srcs,
            trgs,
            self.weights,
            self.camera.K.unsqueeze(0),
        )

        # Test values
        if sigma_val == 0.0:
            for b in range(B):
                if np.linalg.norm(self.Es[b] - Es[b]) > np.linalg.norm(
                    self.Es[b] + Es[b]
                ):
                    Es[b] = -Es[b]
                np.testing.assert_allclose(
                    Es[b].numpy(),
                    self.Es[b].numpy(),
                    atol=1e-10,
                    err_msg="Kornia solution not correct",
                )
        else:
            # Otherwise, check that the cost is better than the ground truth cost
            for b in range(B):
                src = srcs[b].cpu().numpy()
                trg = trgs[b].cpu().numpy()
                E_gt = self.Es[b].cpu().numpy()
                E_est = Es[b].detach().numpy()
                cost_gt = 0.0
                cost_est = 0.0
                for i in range(N):
                    cost_gt += (
                        self.weights[b, :, i]
                        * (trg[:, [i]].T @ E_gt @ src[:, [i]]) ** 2
                    )
                    cost_est += (
                        self.weights[b, :, i]
                        * (trg[:, [i]].T @ E_est @ src[:, [i]]) ** 2
                    )
                assert cost_est < cost_gt, ValueError(
                    "Estimate cost not lower than ground truth cost"
                )

    def test_kornia_backward(self, sigma_val=0.0):
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Wrapper for gradients
        def ess_wrapper(new_src, new_trg, new_wt, out="E"):
            # Stack with other measurements + weights
            new_src = new_src.unsqueeze(2)
            new_trg = new_trg.unsqueeze(2)
            new_srcs = torch.cat([srcs[:, :2, :-1], new_src], dim=2).mT
            new_trgs = torch.cat([trgs[:, :2, :-1], new_trg], dim=2).mT
            new_weights = torch.cat([self.weights[:, 0, :-1], new_wt], dim=1)

            Es, ts, Rs = utils.get_kornia_solution(
                new_srcs,
                new_trgs,
                new_weights,
                self.camera.K.unsqueeze(0),
                self.Es,
            )

            if out == "E":
                return Es
            elif out == "t":
                return ts
            else:
                return None

        # Test backward gradient. We only modify the last image points and weight
        inputs = (
            srcs[:, :2, -1].clone().detach().requires_grad_(False),
            trgs[:, :2, -1].clone().detach().requires_grad_(False),
            self.weights[:, :2, -1].clone().detach().requires_grad_(False),
        )

        # source-essential gradient
        inputs[0].requires_grad_(True)
        eps = 1e-4
        atol = 1e-10
        rtol = 1e-10
        torch.autograd.gradcheck(
            lambda *x: ess_wrapper(*x, out="E"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )

    def compare_with_kornia(self, sigma_val=0.0):
        """Compare gradients between Kornia (local solve) and SDP solver"""
        # Estimators
        estimator_list = ["sdpr-sdp", "sdpr-is", "sdpr-cift", "kornia"]
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        srcs = self.keypoints_src.clone()
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)
        # Batchify Intrinsic matrix
        K = self.camera.K[None, :, :]

        # Get opencv solution for comparison
        srcs_cv = srcs[0, :2].numpy().T[::-1, :]
        trgs_cv = trgs[0, :2].numpy().T[::-1, :]
        E1 = findEssentialMat(
            points1=srcs_cv,
            points2=trgs_cv,
            cameraMatrix=np.eye(3),
        )[0]
        _, s, _ = np.linalg.svd(E1)
        E_cv = torch.tensor(-E1[None, ...] / s[0])
        cost_cv = utils.compute_cost(srcs, trgs, self.weights, E_cv[None, ...])

        # Assess estimators
        data_dicts = []
        for estimator in estimator_list:
            Es_est, jacobians, time_f, time_b = utils.get_soln_and_jac(
                estimator, trgs, srcs, self.weights, K
            )
            # Compute distance from ground truth value (flip sign if needed)
            err1 = torch.norm(Es_est - self.Es)
            err2 = torch.norm(Es_est + self.Es)
            est_err_norms = torch.stack([err1, err2], dim=-1)
            # est_err_norms = torch.min(errs, dim=-1)

            # Compute cost
            costs = utils.compute_cost(srcs, trgs, self.weights, Es_est)
            # Store data
            data_dict = dict(
                estimator=estimator,
                Es_est=Es_est,
                costs=costs,
                jacobians=jacobians,
                est_err_norms=est_err_norms,
                time_f=time_f,
                time_b=time_b,
            )
            data_dicts.append(data_dict)
        df = DataFrame(data_dicts)

        # Plot Jacobian differences
        fig, ax = plt.subplots(3, 1)
        ax[0].matshow(df["jacobians"][0][0][0])
        ax[0].set_title("SDPR-SDP Jacobian")
        ax[1].matshow(df["jacobians"][1][0][0])
        ax[1].set_title("SDPR-IS Jacobian")
        ax[2].matshow(df["jacobians"][3][0][0])
        ax[2].set_title("Kornia Jacobian")
        plt.show()
        # Store to file
        fname = "_results/ess_mat/grad_comp.pkl"
        df.to_pickle(fname)
        return fname


if __name__ == "__main__":
    # Unity element constraint
    t = TestEssMat(n_points=15, n_batch=1, tol=1e-12)

    # t.test_constraints()
    # t.test_cost_matrix_nonoise()
    # t.test_cost_matrix()
    # t.test_feasibility()
    # t.test_sdpr_forward_nonoise()
    # t.test_sdpr_forward(sigma_val=10 / 800)
    # t.test_cost_backward()
    # t.test_layer_backward(sigma_val=0 / 800)
    # t.test_kornia_solution(sigma_val=10 / 800)
    # t.test_kornia_solution(sigma_val=0 / 800)
    # t.test_kornia_backward()

    # Gradient Comparison
    t.compare_with_kornia(sigma_val=0 / 800)
