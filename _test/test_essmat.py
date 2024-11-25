import unittest

import kornia.geometry.epipolar as epi
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

    def __init__(self, *args, n_points=50, tol=1e-12, **kwargs):
        super(TestEssMat, self).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        self.device = "cuda:0"
        # Set seed
        set_seed(0)
        batch_size = 1
        # Set up test problem
        # NOTE ts_ts_s is the vector from the source to the target frame expressed in the source frame
        # NOTE Rs_ts is the rotation that maps vectors in the source frame to vectors in the target frame
        ts_ts_s, Rs_ts, key_ss = utils.get_gt_setup(
            N_map=n_points, N_batch=batch_size, traj_type="clusters"
        )
        # Transforms from source to target
        Rs_ts = torch.tensor(Rs_ts)
        ts_ts_s = torch.tensor(ts_ts_s)
        ts_st_t = Rs_ts.bmm(-ts_ts_s)
        # store rotation solution
        self.Rs_ts = Rs_ts

        # Keypoints (3D) defined in source frame
        key_ss = torch.tensor(key_ss)[None, :, :].expand(batch_size, -1, -1)
        # Keypoints in target frame
        key_ts = Rs_ts.bmm(key_ss - ts_ts_s)
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
        self.weights = torch.ones(
            self.keypoints_src.size(0), 1, self.keypoints_src.size(2)
        )
        self.camera = camera

        # Normalize the translations
        t_norm = torch.norm(ts_st_t)
        self.ts_st_t = ts_st_t / t_norm
        # Construct Essential Matrix
        self.Es = so3_wedge(self.ts_st_t).bmm(self.Rs_ts)
        check = (
            self.keypoints_trg[0, :3, [0]].mT
            @ self.Es[0]
            @ self.keypoints_src[0, :3, [0]]
        )
        np.testing.assert_allclose(check, 0.0, atol=1e-12)

        # Construct solution vectors
        self.sol = torch.cat(
            [
                torch.ones((batch_size, 1, 1)),
                torch.reshape(self.Es, (-1, 9, 1)),  # row-major vectorization
                self.ts_st_t,
            ],
            dim=1,
        )
        K_batch = self.camera.K.expand(1, -1, -1)
        # Initialize layer
        self.layer = EssentialSDPBlock(tol=tol, K_source=K_batch, K_target=K_batch)

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
                    t.weights[b, :, i] * (trg[:, [i]].T @ E @ src[:, [i]]) ** 2
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
        E_mats, t_vecs, X, rank = self.layer(
            t.keypoints_src, t.keypoints_trg, wts, rescale=False, verbose=True
        )

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
        t_gt = self.ts_st_t[0].numpy()

        # If no noise, check that we are close to solution
        if sigma_val == 0.0:
            # Check sign ambiguity
            if np.linalg.norm(E_gt - E_est) > np.linalg.norm(E_gt + E_est):
                E_est = -E_est
            if np.linalg.norm(t_est - t_est) > np.linalg.norm(t_est + t_est):
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
        B = t.keypoints_src.size(0)
        N = t.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = t.keypoints_trg
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = t.keypoints_src
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
            new_weights = torch.cat([t.weights[:, :, :-1], new_wt], dim=2)
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
            t.weights[:, :2, -1].clone().detach().requires_grad_(False),
        )

        # source-essential gradient
        inputs[0].requires_grad_(True)
        eps = 1e-4
        atol = 1e-5
        rtol = 1e-1
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="E"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        # source-translation gradient
        eps = 1e-4
        atol = 1e-5
        rtol = 1e-1
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
        rtol = 1e-1
        torch.autograd.gradcheck(
            lambda *x: layer_wrapper(*x, out="E"),
            inputs=inputs,
            eps=eps,
            atol=atol,
            rtol=rtol,
        )
        # target-translation gradient
        eps = 1e-4
        atol = 1e-5
        rtol = 1e-1
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
        eps = 1e-2
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
        B = t.keypoints_src.size(0)
        N = t.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = t.keypoints_trg
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = t.keypoints_src
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        def cost_wrapper(new_src, new_trg, new_wt):
            # Homogenize input image tensor
            one = torch.tensor([[[1]]])
            new_trg_h = torch.cat((new_trg, one), dim=1)
            new_src_h = torch.cat((new_src, one), dim=1)
            # Stack with other measurements + weights
            new_srcs = torch.cat([srcs, new_src_h], dim=2)
            new_trgs = torch.cat([trgs, new_trg_h], dim=2)
            new_weights = torch.cat([t.weights, new_wt], dim=2)
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
        trgs = self.keypoints_trg
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Get essential matrix, and extract rotation, translation
        Es, ts, Rs = get_kornia_solution(
            srcs[:, :2, :].mT,
            trgs[:, :2, :].mT,
            self.weights[:, 0, :],
            self.camera.K.unsqueeze_(0),
            self.Es,
        )

        # Test values
        np.testing.assert_allclose(
            Es.numpy(),
            self.Es.numpy(),
            atol=1e-10,
            err_msg="Kornia solution not correct",
        )

    def test_kornia_backward(self, sigma_val=0.0):
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Wrapper for gradients
        def ess_wrapper(new_src, new_trg, new_wt, out="E"):
            # Stack with other measurements + weights
            new_src = new_src.unsqueeze(2)
            new_trg = new_trg.unsqueeze(2)
            new_srcs = torch.cat([srcs[:, :2, :-1], new_src], dim=2).mT
            new_trgs = torch.cat([trgs[:, :2, :-1], new_trg], dim=2).mT
            new_weights = torch.cat([t.weights[:, 0, :-1], new_wt], dim=1)

            Es, ts, Rs = get_kornia_solution(
                new_srcs,
                new_trgs,
                new_weights,
                self.camera.K.unsqueeze_(0),
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
            t.weights[:, :2, -1].clone().detach().requires_grad_(False),
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


def get_kornia_solution(srcs, trgs, wts, K, Es_gt):
    Es_kornia = epi.find_essential(srcs, trgs, wts)
    # find the best of the 10 kornia solutions
    vals = torch.linalg.matrix_norm(Es_gt - Es_kornia)
    ind = torch.argmin(vals)
    Es_kornia = Es_kornia[:, ind, :, :]
    # Decompose solution
    Rs, ts, points = epi.motion_from_essential_choose_solution(
        Es_kornia,
        K,
        K,
        srcs,
        trgs,
    )
    Es = so3_wedge(ts).bmm(Rs)

    return Es, ts, Rs


if __name__ == "__main__":
    # Unity element constraint
    t = TestEssMat(n_points=20, tol=1e-12)

    # t.test_constraints()
    # t.test_cost_matrix_nonoise()
    # t.test_cost_matrix()
    # t.test_feasibility()
    # t.test_sdpr_forward_nonoise()
    # t.test_sdpr_forward(sigma_val=10 / 800)
    # t.test_cost_backward()
    # t.test_layer_backward(sigma_val=0 / 800)
    t.test_kornia_solution()
    # t.test_kornia_backward()
    # t.compare_with_kornia(sigma_val=0 / 800)
