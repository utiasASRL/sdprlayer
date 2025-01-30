import time
import unittest

import kornia.geometry.epipolar as epi
import matplotlib.pyplot as plt
import numpy as np
import torch
from cv2 import findEssentialMat
from pandas import DataFrame

import sdprlayers.utils.ess_mat_utils as utils
from sdprlayers import SDPEssMatEst
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import se3_exp, so3_exp, so3_log, so3_wedge
from sdprlayers.utils.plot_tools import plot_map, plot_poses


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class MultipleErrors(Exception):
    def __init__(self, errors):
        self.errors = errors
        super().__init__(self._format_errors())

    def _format_errors(self):
        return "\n".join(f"{i+1}. {str(e)}" for i, e in enumerate(self.errors))


class TestEssMat(unittest.TestCase):

    def __init__(
        self, *args, n_batch=10, n_points=200, tol=1e-12, seed=1, plot=False, **kwargs
    ):
        super(TestEssMat, self).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)
        torch.autograd.set_detect_anomaly(True)
        self.device = "cuda:0"
        # Set seed
        set_seed(seed)
        # Set up test problem
        # NOTE ts_ts_s is the vector from the source to the target frame expressed in the source frame
        # NOTE Rs_ts is the rotation that maps vectors in the source frame to vectors in the target frame
        dist = 4
        ts_ts_s, Rs_ts, keys_3d_s = utils.get_gt_setup(
            N_map=n_points,
            N_batch=n_batch,
            traj_type="clusters",
            offs=dist,
            lm_bound=dist * 0.7,
        )
        if plot:
            ax = plot_map(keys_3d_s)
            plot_poses(Rs_ts, ts_ts_s, ax)
            ax.set_box_aspect([1, 1, 1])
        # Define Camera
        # self.camera = CameraModel(800, 800, 0.0, 0.0, 0.0)
        self.camera = CameraModel(1, 1, 0.0, 0.0, 0.0)

        # Transforms from source to target
        self.xi = so3_log(torch.tensor(Rs_ts))
        self.xi.requires_grad_(True)
        self.Rs_ts = so3_exp(self.xi)
        ts_ts_s = torch.tensor(ts_ts_s)
        ts_st_t = self.Rs_ts.bmm(-ts_ts_s)
        # Keep track of gradients for translation
        ts_st_t.requires_grad_(True)

        # Keypoints (3D) defined in source frame
        keys_3d_s = torch.tensor(keys_3d_s)[None, :, :].expand(n_batch, -1, -1)
        self.keys_3d_s = keys_3d_s
        # Homogenize coords
        keys_3dh_s = torch.concat(
            [keys_3d_s, torch.ones(n_batch, 1, keys_3d_s.size(2))], dim=1
        )
        # Apply camera to get image points
        src_img_pts = self.camera.camera_model(keys_3dh_s)
        # Get inverse intrinsic camera mat
        K_inv = torch.linalg.inv(self.camera.K)
        K_invs = K_inv.expand(n_batch, 3, 3)
        # Store normalized image coordinates
        self.keypoints_src = K_invs.bmm(src_img_pts)
        # Map to target points
        self.keypoints_trg = self.map_src_to_trg(
            keys_3d_s=keys_3d_s, Rs_ts=self.Rs_ts, ts_st_t=ts_st_t
        )
        # Check the keypoints to ensure good conditioning
        if np.any(np.abs(self.keypoints_src.detach().numpy()) > 1e2):
            raise ValueError("Source keypoints are not well conditioned")
        if np.any(np.abs(self.keypoints_trg.detach().numpy()) > 1e2):
            raise ValueError("Source target are not well conditioned")

        # Visualize feature tracks
        if plot:
            self.plot_feature_tracks()
        # Generate Scalar Weights
        self.weights = torch.ones(
            self.keypoints_src.size(0), 1, self.keypoints_src.size(2)
        )

        # Normalize the translations
        t_norm = torch.norm(ts_st_t, dim=1, keepdim=True)
        self.ts_st_t_norm = ts_st_t / t_norm
        self.ts_st_t_unnorm = ts_st_t
        # Construct Essential Matrix
        self.Es = self.get_essential(self.ts_st_t_norm, self.xi)

        # Check that the matrix makes sense
        check = 0.0
        for n in range(n_batch):
            check += (
                self.keypoints_trg[0, :, [n]].mT
                @ self.Es[0]
                @ self.keypoints_src[0, :, [n]]
            )

        np.testing.assert_allclose(check.detach(), 0.0, atol=1e-12)

        # Construct solution vectors
        self.sol = torch.cat(
            [
                torch.ones((n_batch, 1, 1)),
                torch.reshape(self.Es, (-1, 9, 1)),  # row-major vectorization
                self.ts_st_t_norm,
            ],
            dim=1,
        )
        K_batch = self.camera.K.expand(1, -1, -1)
        # Initialize layer
        self.layer = SDPEssMatEst(tol=tol, K_source=K_batch, K_target=K_batch)

    def plot_feature_tracks(self):
        # Detach keypoints from computation graph and convert to numpy
        keypoints_src_np = self.keypoints_src.detach().numpy()
        keypoints_trg_np = self.keypoints_trg.detach().numpy()

        # Plot feature tracks
        n_batch = keypoints_src_np.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_batch)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        axes = axes.flatten()  # Flatten the grid to easily iterate over

        for i in range(n_batch):
            ax = axes[i]
            for j in range(keypoints_src_np.shape[2]):
                ax.plot(
                    [keypoints_src_np[i, 0, j], keypoints_trg_np[i, 0, j]],
                    [keypoints_src_np[i, 1, j], keypoints_trg_np[i, 1, j]],
                    "k-",  # Line connecting source and target
                )
                ax.plot(
                    keypoints_src_np[i, 0, j],
                    keypoints_src_np[i, 1, j],
                    "bo",  # Source keypoint in blue
                )
                ax.plot(
                    keypoints_trg_np[i, 0, j],
                    keypoints_trg_np[i, 1, j],
                    "ro",  # Target keypoint in red
                )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Feature Tracks for Batch {i+1}")

        # Hide any unused subplots
        for i in range(n_batch, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()

    def get_essential(self, ts, xi):
        """Return essential matrix associated with a translation and Lie algebra representation of a rotation matrix."""
        return so3_wedge(ts[..., 0]) @ so3_exp(xi)

    def map_src_to_trg(self, keys_3d_s, Rs_ts, ts_st_t):
        """Maps 3D source keypoints to 2D target keypoints."""
        n_batch = self.keys_3d_s.shape[0]
        # Keypoints in target frame
        keys_3d_t = Rs_ts.bmm(keys_3d_s) + ts_st_t
        # homogenize coordinates
        trg_coords = torch.concat(
            [keys_3d_t, torch.ones(n_batch, 1, keys_3d_s.size(2))], dim=1
        )
        # Map through camera model
        trg_img_pts = self.camera.camera_model(trg_coords)
        # Normalize camera coords
        K_inv = torch.linalg.inv(self.camera.K)
        K_invs = K_inv.expand(n_batch, 3, 3)
        keys_2d_t = K_invs.bmm(trg_img_pts)
        return keys_2d_t

    def test_constraints(self, sol=None):
        """Test that the constraints characterize the fundamental matrix and epipole."""
        if sol is None:
            sol = self.sol[0].detach()

        # Get constraints and test
        constraints = self.layer.sdprlayer.constr_list
        viol = np.zeros((len(constraints)))
        for i, A in enumerate(constraints):
            viol[i] = (sol.T @ A.toarray() @ sol)[0, 0]
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
        trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        srcs = self.keypoints_src.clone()
        srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Compute actual cost at ground truth solution
        cost_true = np.zeros(B)
        for b in range(B):
            src = srcs[b].detach().numpy()
            trg = trgs[b].detach().numpy()
            E = self.Es[b].detach().numpy()
            for i in range(N):
                cost_true[b] += (
                    self.weights[b, :, i] * (trg[:, [i]].T @ E @ src[:, [i]]) ** 2
                )
        cost_true = cost_true / N

        # Construct objective matrix - (No scaling)
        Q, scale, offs = SDPEssMatEst.get_obj_matrix_vec(
            srcs, trgs, self.weights, scale_offset=False, regularize=False
        )

        for b in range(B):
            Q_list = []
            rows = []
            for n in range(N):
                # Compute Kronecker product
                row = (trgs[b, :, [n]] @ srcs[[b], :, n]).reshape(1, -1)
                Q_list.append(row.mT @ row)
                rows.append(row)
            Q_test = sum(Q_list) / N
            rows = torch.cat(rows, 0)
            np.testing.assert_allclose(
                Q[b][1:10][:, 1:10].detach(),
                Q_test.detach(),
                atol=1e-12,
                err_msg="Cost matrix does not match cost function",
            )

        # Check that matrix does the same thing
        cost_mat = self.sol.mT.bmm(Q.bmm(self.sol))[:, 0, 0]
        np.testing.assert_allclose(
            cost_mat.detach(),
            cost_true,
            atol=1e-12,
            err_msg="Matrix cost not equal to true cost",
        )

        # Construct objective matrix - (with scaling)
        Q, scale, offs = SDPEssMatEst.get_obj_matrix_vec(
            srcs, trgs, self.weights, scale_offset=True, regularize=False
        )

        # Check that matrix does the same thing
        cost_mat = self.sol.mT.bmm(Q.bmm(self.sol))[:, 0, 0] * scale + offs
        np.testing.assert_allclose(
            cost_mat.detach(),
            cost_true,
            atol=1e-12,
            err_msg="Matrix cost not equal to true cost",
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
        E_est = Es_est[0].detach()
        t_est = ts_est[0].detach()
        E_gt = self.Es[0].detach()
        t_gt = self.ts_st_t_norm[0].detach()

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
            src = srcs[b].cpu().detach()
            trg = trgs[b].cpu().detach()
            E_gt = self.Es[b].cpu().detach()
            cost_gt = 0.0
            cost_est = 0.0
            for i in range(N):
                cost_gt += (
                    self.weights[b, :, i] * (trg[:, [i]].T @ E_gt @ src[:, [i]]) ** 2
                )
                cost_est += (
                    self.weights[b, :, i] * (trg[:, [i]].T @ E_est @ src[:, [i]]) ** 2
                )
            assert cost_est < cost_gt + 1e-5, ValueError(
                "Estimate cost not lower than ground truth cost"
            )

    def test_kornia_solution(self):
        """Tests the Kornia solution for the essential matrix
        Note: Kornia only works when the points are coordinated (i.e. no noise)"""
        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        trgs = self.keypoints_trg.clone()
        srcs = self.keypoints_src.clone()
        # sigma = torch.tensor(sigma_val)
        # trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
        # srcs[:, :2, :] += sigma * torch.randn(B, 2, N)

        # Get essential matrix, and extract rotation, translation
        Es, ts, Rs = utils.get_kornia_solution(
            srcs,
            trgs,
            self.weights,
            self.camera.K.unsqueeze(0),
        )

        # Test values
        for b in range(B):
            if torch.linalg.norm(self.Es[b] - Es[b]) > torch.linalg.norm(
                self.Es[b] + Es[b]
            ):
                Es[b] = -Es[b]
            np.testing.assert_allclose(
                Es[b].detach(),
                self.Es[b].detach(),
                atol=1e-10,
                err_msg="Kornia solution not correct",
            )

    def test_estimators_backward(
        self,
        test=True,
        sigma_val=0.0,
        plot_jacobians=False,
        save_data=False,
        test_kornia=False,
    ):
        """Compare gradients between Kornia (local solve) and SDP solver
        NOTE: Gradients are computed with respect to pose variables. Gradients wrt keypoints will not work for Kornia since it only gives the correct solution when there is no-noise (i.e. points are coordinated).
        """
        # Estimators
        if test_kornia:
            estimator_list = ["sdpr-sdp", "sdpr-is", "sdpr-cift", "kornia"]
        else:
            estimator_list = ["sdpr-sdp", "sdpr-is", "sdpr-cift"]
        tolerances = {
            "sdpr-sdp": dict(soln_atol=1e-4, jac_atol=2e-2),
            "sdpr-is": dict(soln_atol=1e-4, jac_atol=2e-2),
            "sdpr-cift": dict(soln_atol=1e-4, jac_atol=2e-2),
            "kornia": dict(soln_atol=1e-10, jac_atol=1e-10),
        }

        # Sizes
        B = self.keypoints_src.size(0)
        N = self.keypoints_src.size(2)
        # Add Noise
        sigma = torch.tensor(sigma_val)
        trgs = self.keypoints_trg.clone()
        srcs = self.keypoints_src.clone()
        if not test:
            # Only add noise if not testing.
            trgs[:, :2, :] += sigma * torch.randn(B, 2, N)
            srcs[:, :2, :] += sigma * torch.randn(B, 2, N)
        # Batchify Intrinsic matrix
        K = self.camera.K[None, :, :]
        # Jacobian input tensors
        jac_inputs = [self.ts_st_t_unnorm, self.xi]
        # Compute analytical jacobians
        jacobians_true = []
        Es_vec = self.Es.flatten(start_dim=1)
        for jac_input in jac_inputs:
            jacobians_true.append(utils.tensor_jacobian(Es_vec, jac_input))
        # Assess estimators
        data_dicts = []
        for estimator in estimator_list:
            Es_est, jacobians = utils.get_soln_and_jac(
                estimator,
                trgs,
                srcs,
                self.weights,
                K,
                jac_vars=jac_inputs,
                rescale=True,
                tol=1e-12,
            )
            # Compute distance from ground truth value (flip sign if needed)
            err1 = torch.norm(Es_est - self.Es, dim=(-1, -2))
            err2 = torch.norm(Es_est + self.Es, dim=(-1, -2))
            est_err_norm, indices = torch.min(torch.stack([err1, err2], dim=-1), dim=-1)
            est_err_norm = est_err_norm.detach()
            # The essential matrix has a sign ambiguity, so we flip signs accordingly
            for b in range(B):
                if indices[b] > 0:
                    Es_est[b] = -Es_est[b]
                    jacobians[0][b] = -jacobians[0][b]
                    jacobians[1][b] = -jacobians[1][b]
            # Compute cost
            costs = utils.compute_cost(srcs, trgs, self.weights, Es_est)
            if test:
                error_messages = []
                # Loop over test cases
                for b in range(B):
                    # Test solution
                    try:
                        np.testing.assert_allclose(
                            est_err_norm[b],
                            0,
                            atol=tolerances[estimator]["soln_atol"],
                            err_msg=f"{estimator} solution error is large",
                        )
                    except AssertionError as e:
                        error_messages.append(
                            f"Test Case {b}: {estimator} Solution Error: {str(e)}"
                        )
                    # Test jacobians
                    for i in range(len(jacobians)):
                        try:
                            np.testing.assert_allclose(
                                jacobians[i][b].detach(),
                                jacobians_true[i][b].detach(),
                                atol=tolerances[estimator]["jac_atol"],
                                err_msg=f"{estimator} jacobian error is large",
                            )
                        except AssertionError as e:
                            error_messages.append(
                                f"Test Case {b}-{i}: {estimator} Jacobian Error: {str(e)}"
                            )
            # Store data
            data_dict = dict(
                estimator=estimator,
                Es_est=Es_est,
                costs=costs,
                jacobians=jacobians,
                est_err_norms=est_err_norm,
            )
            data_dicts.append(data_dict)

        df = DataFrame(data_dicts)

        # Plot Jacobians
        if plot_jacobians:
            for ind in range(len(jac_inputs)):
                fig, ax = plt.subplots(1, len(estimator_list) + 1)
                for i in range(len(estimator_list)):
                    ax[i].matshow(df["jacobians"][i][ind][0])
                    ax[i].set_title(f"{estimator_list[i]}")
                ax[-1].matshow(jacobians_true[ind][0])
                ax[-1].set_title("Ground Truth")
            plt.show()

        # Store to file
        if save_data:
            fname = "_results/ess_mat/grad_comp.pkl"
            df.to_pickle(fname)
            return fname, df

        # Rethrow errors
        if len(error_messages) > 0:
            raise MultipleErrors(error_messages)

    def compute_analytical_jacobians(self):
        """Compute the analytical jacobians of the essential matrix with respect to the rigid body translation and rotation."""
        # Variables
        t = self.ts_st_t_norm[..., 0]
        R = self.Rs_ts
        E = self.Es
        B = E.shape[0]
        # Jacobian computation
        jac_t_b, jac_xi_b = [], []
        for b in range(B):
            jac_t, jac_xi = [], []
            for a in torch.eye(3):
                jac_t.append((so3_wedge(a) @ R[b]).reshape(9, 1))
                jac_xi.append((so3_wedge(t[b]) @ so3_wedge(a) @ R[b]).reshape(9, 1))
            jac_t_b.append(torch.cat(jac_t, dim=-1).detach())
            jac_xi_b.append(torch.cat(jac_xi, dim=-1).detach())

        jacobians = [torch.stack(jac_t_b), torch.stack(jac_xi_b)]

        return jacobians


if __name__ == "__main__":
    # Unity element constraint
    # t = TestEssMat(n_points=50, n_batch=10, tol=1e-12)
    t = TestEssMat(plot=False)

    # t.test_constraints()
    # t.test_cost_matrix_nonoise()
    # t.test_cost_matrix()
    # t.test_sdpr_forward_nonoise()
    # t.test_sdpr_forward(sigma_val=10 / 800)
    # t.test_kornia_solution()
    t.test_estimators_backward()
