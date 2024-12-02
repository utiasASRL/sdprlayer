import numpy as np
import theseus as th
import torch
import torch.nn as nn

from sdprlayers.utils.lie_algebra import se3_inv, se3_log


class LieOptPoseEstimator(nn.Module):
    """
    Compute the relative pose between the source and target frames using
    Local Lie Algebra computation, from Theseus.
    """

    def __init__(self, T_s_v, N_batch, N_map, opt_kwargs_in={}):
        """
        Initialize the SVD class.

        Args:
            T_s_v (torch.tensor): 4x4 transformation matrix providing the transform from the vehicle frame to the
                                  sensor frame.
        """
        super(LieOptPoseEstimator, self).__init__()

        # Store sensor vehicle transform
        self.register_buffer("T_s_v", T_s_v)
        self.N_batch = N_batch
        self.N_map = N_map

        # SET UP THESEUS OPTIMIZATION
        # Optimization variables
        T_trg_src = th.SE3(name="T_trg_src")
        # Auxillary (data) variables (pixel keypoints_3D_trgurements and landmarks)
        keypoints_3D_trg = th.Variable(
            torch.zeros(N_batch, 4, N_map), name="keypoints_3D_trg"
        )
        inv_cov_weights = th.Variable(
            torch.zeros(N_batch, N_map, 3, 3), name="inv_cov_weights"
        )
        keypoints_3D_src = th.Variable(
            torch.zeros(N_batch, 4, N_map), name="keypoints_3D_src"
        )
        weights = th.Variable(torch.ones(N_batch, 1, N_map), name="weights")

        objective = th.Objective()
        optim_vars = [T_trg_src]
        aux_vars = [keypoints_3D_src, keypoints_3D_trg, weights, inv_cov_weights]
        cost_function = th.AutoDiffCostFunction(
            optim_vars=optim_vars,
            dim=N_map * 3,
            err_fn=self.error_fn_vec,
            aux_vars=aux_vars,
            cost_weight=th.ScaleCostWeight(float(np.sqrt(2))),
            name="registration_cost",
        )
        objective.add(cost_function)

        # Optimization parameters
        opt_kwargs = {
            "abs_err_tolerance": 1e-9,
            "rel_err_tolerance": 1e-9,
            "max_iterations": 200,
        }
        opt_kwargs.update(opt_kwargs_in)
        # Build layer
        self.th_layer = th.TheseusLayer(th.GaussNewton(objective, **opt_kwargs))

    @staticmethod
    def error_fn_vec(optim_vars, aux_vars):
        """Error function to be optimized by Theseus. Returns weighted errors for each map point.

        Args:
            optim_vars (_type_): _description_
            aux_vars (_type_): _description_

        Returns:
            tensor : torch tensor representing a vector of errors to be minimized
        """
        # (Bx4x4)
        T_trg_src = optim_vars[0]
        # (Bx4xN), (Bx4xN), (Bx1xN),(BxNx3x3)
        keypoints_3D_src, keypoints_3D_trg, weights, inv_cov_weights = aux_vars
        # Get dimensions
        B = keypoints_3D_trg.shape[0]
        N = keypoints_3D_trg.shape[-1]
        # get error
        bottom = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(B, 1, 4).to(T_trg_src.device)
        T_trg_src_mat = torch.cat([T_trg_src.tensor, bottom], dim=1)
        errors = keypoints_3D_trg.tensor[:, :3, :] - (
            T_trg_src_mat[:, :3, :3].bmm(keypoints_3D_src.tensor[:, :3, :])
            + T_trg_src_mat[:, :3, 3].unsqueeze(-1)
        )
        # Multiply by weight matrix.
        # NOTE we want the transpose of the cholesky factor
        if inv_cov_weights is not None:
            W_half = torch.linalg.cholesky(inv_cov_weights.tensor).transpose(-2, -1)
            errors_weighted = torch.einsum("bnij,bjn->bin", W_half, errors)
            errors_weighted = errors_weighted * weights.tensor
        else:
            errors_weighted = errors * weights.tensor
        # reshape errors (N_batch, 3 * N_map)
        errors_stacked = errors_weighted.reshape(B, -1)
        return errors_stacked

    def forward(
        self,
        keypoints_3D_src,
        keypoints_3D_trg,
        weights,
        T_trg_src_init,
        inv_cov_weights=None,
        backward_mode="implicit",
        verbose=False,
    ):
        """
        Compute the pose, T_trg_src, from the source to the target frame.

        Args:
            keypoints_3D_src (torch,tensor, Bx4xN): 3D point coordinates of keypoints from source frame.
            keypoints_3D_trg (torch,tensor, Bx4xN): 3D point coordinates of keypoints from target frame.
            weights (torch.tensor, Bx1xN): weights in range (0, 1) associated with the matched source and target
                                           points.
            T_trg_src_init (torch.tensor, Bx4x4): Initial guess for the relative transform from the source to the
            inv_cov_weights (torch.tensor, BxNx3x3): Inverse Covariance Matrices defined for each point.

        Returns:
            T_trg_src (torch.tensor, Bx4x4): relative transform from the source to the target frame.
        """
        batch_size = keypoints_3D_src.size()[0]
        device = keypoints_3D_src.device
        # Get inverse covariance weights
        if inv_cov_weights is None:
            inv_cov_weights = (
                torch.eye(3).expand(batch_size, self.N_map, 3, 3).to(device)
            )
        # Set up theseus inputs
        theseus_inputs = {
            "T_trg_src": T_trg_src_init[:, :3, :],
            "keypoints_3D_src": keypoints_3D_src,
            "keypoints_3D_trg": keypoints_3D_trg,
            "weights": weights,
            "inv_cov_weights": inv_cov_weights,
        }
        # Run Forward pass
        vars_th, info = self.th_layer.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": verbose,
                "backward_mode": backward_mode,
            },
        )
        # Get variables from theseus output
        T_trg_src = vars_th["T_trg_src"]
        bottom = (
            torch.tensor([0.0, 0.0, 0.0, 1.0])
            .expand(batch_size, 1, 4)
            .to(T_trg_src.device)
        )
        T_trg_src = torch.cat([T_trg_src, bottom], dim=1)
        # Convert from sensor to vehicle frame
        T_s_v = self.T_s_v.expand(batch_size, 4, 4).to(device)
        T_trg_src = se3_inv(T_s_v).bmm(T_trg_src).bmm(T_s_v)

        return T_trg_src

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.th_layer.to(*args, **kwargs)

    def cuda(self, device=None):
        super().cuda(device=device)
        if device is None:
            device = torch.cuda.current_device()
        self.th_layer.to(f"cuda:{device}")

    @staticmethod
    def plot_points(s_in, t_in, w_in):
        """purely for debug"""
        import matplotlib.pyplot as plt

        s = s_in.cpu().detach().numpy()
        t = t_in.cpu().detach().numpy()
        w = w_in.cpu().detach().numpy()
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter3D(
            s[0, 0, :],
            s[0, 1, :],
            s[0, 2, :],
            marker="*",
            color="g",
        )
        ax.scatter3D(
            t[0, 0, :],
            t[0, 1, :],
            t[0, 2, :],
            marker="*",
            color="b",
        )
        ax.scatter3D(
            0.0,
            0.0,
            0.0,
            marker="*",
            color="r",
        )
        return ax
