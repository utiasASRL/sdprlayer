import os
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
    "sdpr-sdp",
    "sdpr-cift",
    "sdpr-is",
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
    weights = torch.ones((n_batch, 1, n_points), dtype=precision) / n_points
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
    weights = torch.ones((n_batch, 1, n_points), dtype=precision) / n_points
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
    # Lie Opt Parameters:
    opt_kwargs = {
        "abs_err_tolerance": 1e-12,
        "rel_err_tolerance": 1e-12,
        "max_iterations": 200,
    }
    # Create estimator module
    if estimator == "svd":
        forward = SVDPoseEstimator(T_s_v=T_s_v)
    elif estimator == "sdpr-sdp":
        forward = SDPPoseEstimator(T_s_v=T_s_v, diff_qcqp=False)
    elif estimator == "sdpr-cift":
        forward = SDPPoseEstimator(
            T_s_v=T_s_v, diff_qcqp=True, compute_multipliers=True
        )
    elif estimator == "sdpr-is":
        forward = SDPPoseEstimator(
            T_s_v=T_s_v, diff_qcqp=True, compute_multipliers=False
        )
    elif estimator in ["lieopt-gt", "lieopt-gt-unroll", "lieopt-rand"]:
        forward = LieOptPoseEstimator(
            T_s_v=T_s_v, N_batch=1, N_map=n_points, opt_kwargs_in=opt_kwargs
        )
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

        # Adjust tolerances for SDP
        if estimator in ["sdpr-sdp", "sdpr-cift", "sdpr-is"]:
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
                "verbose": False,
            }
            kwargs.update(dict(solver_args=solver_args))

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


def get_experiment_data(n_batch, noise_std, n_points, experiment):
    if experiment == "stereo-matwt":
        points_s, points_t, weights, mat_wts, T_t_s_gt = get_stereo_point_clouds(
            n_batch=n_batch, noise_std=noise_std
        )
    elif experiment == "stereo-sclwt":
        points_s, points_t, weights, mat_wts, T_t_s_gt = get_stereo_point_clouds(
            n_batch=n_batch, noise_std=noise_std
        )
        mat_wts = None
    elif experiment == "pointcld":
        points_s, points_t, weights, T_t_s_gt = get_point_clouds(
            n_points=n_points, n_batch=n_batch, noise_std=noise_std
        )
        mat_wts = None
    else:
        raise ValueError("Experiment Unknown")
    return points_s, points_t, weights, T_t_s_gt, mat_wts


def gen_estimator_data(
    n_points=30, n_batch=1, noise_std=0.0, experiment="pointcloud", save_data=True
):
    """Compare Solutions and gradients of different estimators

    Args:
        n_points (int, optional): Number of points in cloud. Defaults to 30.
        n_batch (int, optional): Number of trials (batched). Defaults to 1.
        noise_std (float, optional): Standard deviation of noise added to points. Defaults to 0.0.
    """

    # Generate experiment data
    points_s, points_t, weights, T_t_s_gt, mat_wts = get_experiment_data(
        n_batch=n_batch, n_points=n_points, noise_std=noise_std, experiment=experiment
    )

    # Assess estimators
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

    if save_data:
        # Store to file
        import time

        fname = "_results/grad_comp_" + experiment + ".pkl"
        df.to_pickle(fname)
        return fname
    else:
        return df


def process_grad_data(
    filename=None, df=None, experiment="pointcld", return_diffs=False
):
    """Post process gradient data from trials"""
    if filename is None and df is None:
        filename = f"_results/grad_comp_{experiment}.pkl"
    if df is None:
        # Read file
        df = read_pickle(filename)

    # "correct" solution depends on experiment
    if experiment == "stereo-matwt":
        # mat weighted, lieopt more accurate
        df_true = df[df["estimator"] == "lieopt-gt-unroll"]
    else:
        # scalar weighted, svd more accurate
        df_true = df[df["estimator"] == "svd"]
    # Fix the inputs to the source frame keypoints
    iInput = 0
    jacs_true = torch.stack(
        [jacs[iInput] for jacs in df_true["jacobians"].values[0]]
    ).numpy()
    T_t_s_true = df_true["T_t_s_est"].values[0]

    # Loop through other solutions and compare with svd solution
    estimators = df.estimator.to_list()
    data = []
    jac_diffs = {}
    for estimator in estimators:
        df_2 = df[df["estimator"] == estimator]
        T_t_s_est = df_2["T_t_s_est"].values[0]
        # Get relative solution error
        error = se3_log(se3_inv(T_t_s_est) @ T_t_s_true)
        trans_err = np.sqrt(np.mean(error[:, :3].numpy() ** 2, axis=1))
        rot_err = np.sqrt(np.mean(error[:, 3:].numpy() ** 2, axis=1))
        # Get absolute solution error
        error_gt = df_2["xi_err"].values[0].numpy()
        trans_err_gt = np.sqrt(np.mean(error_gt[:, :3] ** 2, axis=1))
        rot_err_gt = np.sqrt(np.mean(error_gt[:, 3:] ** 2, axis=1))

        # Get Jacobian
        jacs = torch.stack(
            [jacs[iInput] for jacs in df_2["jacobians"].values[0]]
        ).numpy()
        # Compare Jacobians
        jac_diff = np.max(np.abs(jacs - jacs_true), axis=(1, 2)) / np.max(
            np.max(np.abs(jacs_true), axis=(1, 2))
        )
        data.append(
            dict(
                estimator=estimator,
                jac_diff_mean=np.mean(jac_diff),
                jac_diff_std=np.std(jac_diff),
                trans_err_rel_mean=np.mean(trans_err),
                rot_err_rel_mean=np.mean(rot_err),
                trans_err_abs_mean=np.mean(trans_err_gt),
                rot_err_abs_mean=np.mean(rot_err_gt),
                time_forward=df_2["time_f"].values[0],
                time_backward=df_2["time_b"].values[0],
            )
        )
        # Store jacobian differences
        jac_diffs[estimator] = jac_diff

    df_out = DataFrame(data)
    if return_diffs:
        return jac_diffs
    else:
        return df_out


def test_jac_func(
    estimator="lieopt-rand", n_points=30, n_batch=5, noise_std=0.0, use_mat_wts=False
):
    torch.manual_seed(0)
    np.random.seed(0)
    # Generate experiment data
    points_s, points_t, weights, T_t_s_gt, mat_wts = get_experiment_data(
        n_batch=n_batch, n_points=n_points, noise_std=noise_std, experiment=experiment
    )
    assert estimator in estimator_list, ValueError("Estimator not recognized")
    # Test estimator
    T_t_s, jacs, time_f, time_b = get_soln_and_jac(
        estimator, points_t, points_s, weights, mat_wts, T_t_s_gt=T_t_s_gt
    )

    if use_mat_wts:
        T_t_s_true, jacs_true, _, _ = get_soln_and_jac(
            "lieopt-gt-unroll", points_t, points_s, weights, mat_wts, T_t_s_gt=T_t_s_gt
        )
    else:
        T_t_s_true, jacs_true, _, _ = get_soln_and_jac(
            "svd", points_t, points_s, weights, mat_wts, T_t_s_gt=T_t_s_gt
        )
    if noise_std == 0.0:
        tol = 1e-10
    else:
        tol = 1e-6

    try:
        np.testing.assert_allclose(T_t_s, T_t_s_gt, atol=tol)
    except:
        if estimator == "lieopt-rand":
            print("Converged to local min")

    for b in range(n_batch):
        i = 0
        np.testing.assert_allclose(jacs[b][i], jacs_true[b][i], atol=1e-5)


def gen_latex_tables(experiment):
    """Generate latex table for experiment based on results.

    Args:
        experiment (_type_): _description_
    """

    # Process data
    df = process_grad_data(experiment=experiment)

    # Change headings
    heading_map = dict(
        estimator="Method",
        jac_diff_mean="Jac. Diff. (mean)",
        jac_diff_std="Jac. Diff. (std)",
        trans_err_rel_mean="Rel. Trans. (m)",
        rot_err_rel_mean="Rel. Rot. (rad)",
        time_backward="Backprop Time (s)",
    )
    # Create a list of columns to keep
    columns_to_keep = [col for col in df.columns if col in heading_map]
    # Drop columns that are not in the heading_map
    df = df[columns_to_keep]
    # Rename the columns
    df = df.rename(columns=heading_map)

    # Label mapping
    labelmap = {
        "svd": "SVD",
        "sdpr-sdp": "SDPR-SDP",
        "sdpr-cift": "SDPR-CIFT",
        "sdpr-is": "SDPR-IS",
        "lieopt-gt": "Theseus-GT",
        "lieopt-gt-unroll": "Theseus-GT-UR",
        "lieopt-rand": "Theseus-RND",
    }

    # Format entries
    def format_str(x):
        if not isinstance(x, str):
            return f"{x:.2E}"
        else:
            if x in labelmap:
                return labelmap[x]
            else:
                return x

    df = df.map(format_str)
    print("Experiment " + experiment)
    print(df.to_latex(index=False))


def set_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    # Seed
    set_seed()

    # Tests
    # use_mat_wts = True
    # test_jac_func("svd", use_mat_wts=use_mat_wts)
    # test_jac_func("sdpr-sdp", use_mat_wts=use_mat_wts)
    # test_jac_func("sdpr-cift", noise_std=0.5, use_mat_wts=use_mat_wts)
    # test_jac_func("sdpr-is", noise_std=0.1, use_mat_wts=use_mat_wts)
    # test_jac_func("lieopt-gt-unroll", use_mat_wts=use_mat_wts)
    # test_jac_func("lieopt-rand", use_mat_wts=use_mat_wts)

    # # Run Experiments
    # experiments = ["pointcld", "stereo-sclwt", "stereo-matwt"]
    experiments = ["stereo-sclwt", "stereo-matwt"]
    noise_levels = [0.5, 0.5]
    n_batch = 50
    for iExp, experiment in enumerate(experiments):
        noise_std = noise_levels[iExp]
        fname = gen_estimator_data(
            n_points=30, n_batch=n_batch, noise_std=1, experiment=experiment
        )
        # Post process
        process_grad_data(filename=fname, experiment=experiment)

    # Re-process existing results.
    # process_grad_data(
    #     filename="_results/grad_comp_matwt_20241214T1406.pkl", use_mat_wts=use_mat_wts
    # )

    # Print Latex Tables
    for experiment in experiments:
        gen_latex_tables(experiment=experiment)
