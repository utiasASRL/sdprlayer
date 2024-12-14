import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame, read_pickle
from tqdm import tqdm

from _scripts.stereo_cal import get_cal_data
from sdprlayers import LieOptPoseEstimator, SDPPoseEstimator, SVDPoseEstimator
from sdprlayers.utils.lie_algebra import se3_exp, se3_inv, se3_log
from sdprlayers.utils.stereo_tuner import StereoCamera

# List of estimators
estimator_list = [
    "svd",
    "sdpr",
    "sdpr-qcqpdiff",
    "sdpr-qcqpdiff-reuse",
    "lieopt-gt",
    "lieopt-gt-unroll",
    "lieopt-rand",
]


# Function to get point clouds and ground truth
def get_point_clouds(n_points=30, n_batch=1, noise_std=0.0, precision=torch.double):
    """Generate random point cloud and its transformed version. The transformation used is also randomly generated."""
    print("Generating points...")
    # Set default torch precision
    torch.set_default_dtype(precision)
    # Define random, homogenized points
    points_s = 2 * torch.rand(n_batch, 3, n_points, dtype=precision) - 1
    points_s = torch.concatenate([points_s, torch.ones((n_batch, 1, n_points))], axis=1)
    # Define random SE(3) transform
    pert_se3 = torch.rand(n_batch, 6, dtype=precision)
    T_t_s = se3_exp(pert_se3)

    # Transform points
    points_t = (T_t_s @ points_s).clone()
    # Add noise
    noise = noise_std * torch.randn(n_batch, 3, n_points)
    noise = torch.concatenate([noise, torch.zeros((n_batch, 1, n_points))], axis=1)
    points_t = points_t + noise
    # Get weights
    weights = torch.ones((n_batch, 1, n_points), dtype=precision)
    print("...Done")
    return points_s, points_t, weights, T_t_s


def get_stereo_point_clouds(n_batch=30, noise_std=0.0, precision=torch.double):
    """Get point clouds and associated matrix weights for stereo calibration setup"""
    # Define camera
    camera = StereoCamera(
        f_u=484.5,
        f_v=484.5,
        c_u=0.0,
        c_v=0.0,
        b=0.24,
        sigma_u=noise_std,
        sigma_v=noise_std,
    )
    # Use setup from baseline calibration example
    rs_ts_s, Cs_ts, points_s, cam_meas = get_cal_data(
        radius=3,
        board_dims=[1.0, 1.0],
        N_squares=[8, 8],
        N_batch=n_batch,
        setup="cone",
        plot=False,
        cam=camera,
    )
    # Get ground truth transform
    T_top = torch.cat([Cs_ts, -Cs_ts.bmm(rs_ts_s)], dim=2)
    T_bottom = torch.tensor([[0, 0, 0, 1]]).repeat(n_batch, 1, 1)
    T_t_s = torch.cat([T_top, T_bottom], dim=1)

    # Retrieve points and weights from camera measurements
    points_t, mat_wts = camera.inverse(cam_meas)
    points_t = points_t.clone()
    # get scalar weights (unit)
    n_points = points_t.shape[2]
    weights = torch.ones((n_batch, 1, n_points), dtype=precision)
    # Add homogeneous coord
    points_s = torch.cat([points_s, torch.ones((n_batch, 1, n_points))], axis=1)
    points_t = torch.cat([points_t, torch.ones((n_batch, 1, n_points))], axis=1)

    return points_s, points_t, weights, mat_wts, T_t_s


def get_soln_and_jac(
    estimator, points_t, points_s, weights, mat_wts, T_t_s_gt, **kwargs
):
    """Apply estimator to point clouds and obtain solution and solution gradient.
    NOTE: gradients are computed sequentially using torch's grad function and then assembled into a Jacobian for each input. We also loop over the batch dimension.
    All computations are done on the CPU sequentially.

    "jacobians" output has dimensions B x (num inputs) x (output dims) x (input dims).
    """
    T_s_v = torch.eye(4)  # vehicle to sensor transform set to identity
    n_batch = points_t.shape[0]  # number of batches
    n_points = points_t.shape[2]  # number of points in the point cloud
    precision = points_t.dtype  # precision of the points
    # Create estimator module
    if estimator == "svd":
        forward = SVDPoseEstimator(T_s_v=T_s_v)
    elif estimator == "sdpr":
        forward = SDPPoseEstimator(T_s_v=T_s_v, diff_qcqp=False)
    elif estimator == "sdpr-qcqpdiff":
        forward = SDPPoseEstimator(
            T_s_v=T_s_v, diff_qcqp=True, compute_multipliers=True
        )
    elif estimator == "sdpr-qcqpdiff-reuse":
        forward = SDPPoseEstimator(
            T_s_v=T_s_v, diff_qcqp=True, compute_multipliers=False
        )
    elif estimator in ["lieopt-gt", "lieopt-gt-unroll", "lieopt-rand"]:
        forward = LieOptPoseEstimator(T_s_v=T_s_v, N_batch=1, N_map=n_points)
    elif estimator == "lieopt-rand":
        forward = LieOptPoseEstimator(T_s_v=T_s_v, N_batch=1, N_map=n_points)
    else:
        raise ValueError("Estimator not known!")

    # Manually loop through batches
    n_batch = points_t.shape[0]
    estimates, jacobians, times_f, times_b = [], [], [], []
    jacobians = []
    print(f"Running {n_batch} Tests of {n_points} points with estimator {estimator}")
    for b in tqdm(range(n_batch)):
        # Define input variables
        inputs = [
            points_s[[b], :, :].requires_grad_(True),
            points_t[[b], :, :].requires_grad_(True),
            weights[[b], :, :].requires_grad_(True),
        ]
        if not estimator == "svd" and mat_wts is not None:
            kwargs.update(dict(inv_cov_weights=mat_wts[[b], :, :, :]))

        # Apply forward pass of estimator and time the response
        Tf_0 = time.time()
        # If running Lie group optimization, set intialization for each batch
        if estimator == "lieopt-gt":
            T_t_s_init = T_t_s_gt
            kwargs.update(dict(T_trg_src_init=T_t_s_init[[b], :, :]))
        elif estimator == "lieopt-gt-unroll":
            T_t_s_init = T_t_s_gt
            kwargs.update(
                dict(T_trg_src_init=T_t_s_init[[b], :, :], backward_mode="unroll")
            )
        elif estimator == "lieopt-rand":
            # Random SE3 transform
            pert_se3 = torch.rand(n_batch, 6, dtype=precision)
            T_t_s_init = se3_exp(pert_se3)
            kwargs.update(dict(T_trg_src_init=T_t_s_init[[b], :, :]))

        estimates.append(forward(*inputs, **kwargs))
        Tf_1 = time.time()
        times_f.append(Tf_1 - Tf_0)
        # Compute Jacobian
        # NOTE: We do this by manually looping to avoid issues with vmap
        # Output gradient vectors
        grad_outputs = torch.eye(16).reshape(16, 4, 4)
        Tb_0 = time.time()
        input_jacs = [[] for i in range(len(inputs))]
        # Loop over output gradients
        for grad_output in grad_outputs:
            grads = torch.autograd.grad(
                estimates[-1][0], inputs, grad_output, retain_graph=True
            )
            for iInput in range(len(inputs)):
                input_jacs[iInput].append(grads[iInput].flatten())
        # Stack gradients into jacobian and store
        jacobians.append([torch.stack(jac) for jac in input_jacs])
        Tb_1 = time.time()
        times_b.append(Tb_1 - Tb_0)

    # Get average times
    time_f = np.mean(times_f)
    time_b = np.mean(times_b)
    # batch estimats
    estimates = torch.concat(estimates, dim=0).detach()

    return estimates, jacobians, time_f, time_b


def gen_estimator_data(n_points=30, n_batch=1, noise_std=0.0, use_mat_wts=False):
    """Compare Solutions and gradients of different estimators

    Args:
        n_points (int, optional): Number of points in cloud. Defaults to 30.
        n_batch (int, optional): Number of trials (batched). Defaults to 1.
        noise_std (float, optional): Standard deviation of noise added to points. Defaults to 0.0.
    """

    # Generate input data
    if use_mat_wts:
        points_s, points_t, weights, mat_wts, T_t_s_gt = get_stereo_point_clouds(
            n_batch=n_batch, noise_std=noise_std
        )
    else:
        points_s, points_t, weights, T_t_s_gt = get_point_clouds(
            n_points=n_points, n_batch=n_batch, noise_std=noise_std
        )
        mat_wts = None

    data_dicts = []
    for iEst, estimator in enumerate(estimator_list):
        T_t_s_est, jacobians, time_f, time_b = get_soln_and_jac(
            estimator, points_t, points_s, weights, mat_wts, T_t_s_gt
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

    # Store to file
    import time

    timestr = time.strftime("%Y%m%dT%H%M")
    if use_mat_wts:
        wt_str = "matwt_"
    else:
        wt_str = "sclwt"
    fname = "_results/grad_comp_" + wt_str + timestr + ".pkl"
    df.to_pickle(fname)
    return fname


def process_grad_data(filename="_results/grad_comp_20241202T1158.pkl"):
    """Post process gradient data from trials"""
    # Read file
    df = read_pickle(filename)

    # get values for the SVD solution
    df_svd = df[df["estimator"] == "svd"]
    iInput = 0
    jacs_svd = torch.stack(
        [jacs[iInput] for jacs in df_svd["jacobians"].values[0]]
    ).numpy()

    # Loop through other solutions and compare with svd solution
    estimators = df.estimator.to_list()
    data = []
    for estimator in estimators:
        df_2 = df[df["estimator"] == estimator]
        # Get solution error
        error = df_2["xi_err"].values[0].numpy()
        trans_err = np.sqrt(np.mean(error[:, :3] ** 2, axis=1))
        rot_err = np.sqrt(np.mean(error[:, 3:] ** 2, axis=1))
        # Get Jacobian
        jacs = torch.stack(
            [jacs[iInput] for jacs in df_2["jacobians"].values[0]]
        ).numpy()
        # Compare Jacobians
        jac_diff = np.max(np.abs(jacs - jacs_svd), axis=(1, 2))
        data.append(
            dict(
                estimator=estimator,
                jac_diff_mean=np.mean(jac_diff),
                jac_diff_std=np.std(jac_diff),
                trans_err_mean=np.mean(trans_err),
                trans_err_std=np.std(trans_err),
                rot_err_mean=np.mean(rot_err),
                rot_err_std=np.std(rot_err),
                time_forward=df_2["time_f"].values[0],
                time_backward=df_2["time_b"].values[0],
            )
        )
    df_out = DataFrame(data)
    print(df_out)


def test_jac_func(
    estimator="lieopt-rand", n_points=30, n_batch=5, noise_std=0.0, use_mat_wts=False
):
    # Get points
    if use_mat_wts:
        points_s, points_t, weights, mat_wts, T_t_s_gt = get_stereo_point_clouds(
            n_batch=n_batch, noise_std=noise_std
        )
    else:
        points_s, points_t, weights, T_t_s_gt = get_point_clouds(
            n_points=n_points, n_batch=n_batch, noise_std=noise_std
        )
        mat_wts = None
    assert estimator in estimator_list, ValueError("Estimator not recognized")
    # Test estimator
    T_t_s, jacs, time_f, time_b = get_soln_and_jac(
        estimator, points_t, points_s, weights, mat_wts, T_t_s_gt=T_t_s_gt
    )

    if use_mat_wts:
        _, jacs_true, _, _ = get_soln_and_jac(
            "lieopt-gt", points_t, points_s, weights, mat_wts, T_t_s_gt=T_t_s_gt
        )
    else:
        _, jacs_true, _, _ = get_soln_and_jac(
            "svd", points_t, points_s, weights, mat_wts, T_t_s_gt=T_t_s_gt
        )

    try:
        np.testing.assert_allclose(T_t_s, T_t_s_gt, atol=1e-6)
    except:
        if estimator == "lieopt-rand":
            print("Converged to local min")

    for b in range(n_batch):
        for i in range(2):
            np.testing.assert_allclose(jacs[b][i], jacs_true[b][i], atol=1e-7)


if __name__ == "__main__":
    # Tests
    use_mat_wts = True
    test_jac_func("svd", use_mat_wts=use_mat_wts)
    # test_jac_func("sdpr", use_mat_wts=use_mat_wts)
    # test_jac_func("sdpr-qcqpdiff", noise_std=0.0, use_mat_wts=use_mat_wts)
    # test_jac_func("sdpr-qcqpdiff-reuse", use_mat_wts=use_mat_wts)
    # test_jac_func("lieopt-gt-unroll", use_mat_wts=use_mat_wts)
    # test_jac_func("lieopt-rand", use_mat_wts=use_mat_wts)

    # # Generate data (no noise)
    # fname = gen_estimator_data(
    #     n_points=30, n_batch=100, noise_std=0.1, use_mat_wts=use_mat_wts
    # )
    # # Post process
    # process_grad_data(filename=fname)

    # Re-process existing results.
    # process_grad_data(filename="_results/grad_comp_20241202T1616.pkl")
