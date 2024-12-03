import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame, read_pickle
from tqdm import tqdm

from sdprlayers import LieOptPoseEstimator, SDPPoseEstimator, SVDPoseEstimator
from sdprlayers.utils.lie_algebra import se3_exp, se3_inv, se3_log

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
    points_t = T_t_s @ points_s
    # Add noise
    noise = noise_std * torch.randn(n_batch, 3, n_points)
    noise = torch.concatenate([noise, torch.zeros((n_batch, 1, n_points))], axis=1)
    points_t = points_t + noise
    # Get weights
    weights = torch.ones((n_batch, 1, n_points), dtype=precision)
    print("...Done")
    return points_s, points_t, weights, T_t_s


def get_soln_and_jac(estimator, points_t, points_s, weights, T_t_s_gt, **kwargs):
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
        forward = SDPPoseEstimator(T_s_v=T_s_v, diff_qcqp=True)
    elif estimator == "sdpr-qcqpdiff-reuse":
        forward = SDPPoseEstimator(T_s_v=T_s_v, diff_qcqp=True, resolve_kkt=False)
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


def gen_estimator_data(n_points=30, n_batch=1, noise_std=0.0):
    """Compare Solutions and gradients of different estimators

    Args:
        n_points (int, optional): Number of points in cloud. Defaults to 30.
        n_batch (int, optional): Number of trials (batched). Defaults to 1.
        noise_std (float, optional): Standard deviation of noise added to points. Defaults to 0.0.
    """

    # Generate input data
    points_s, points_t, weights, T_t_s_gt = get_point_clouds(
        n_points=n_points, n_batch=n_batch, noise_std=noise_std
    )

    data_dicts = []
    for iEst, estimator in enumerate(estimator_list):
        T_t_s_est, jacobians, time_f, time_b = get_soln_and_jac(
            estimator, points_t, points_s, weights, T_t_s_gt
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
    import time

    timestr = time.strftime("%Y%m%dT%H%M")
    fname = "_results/grad_comp_" + timestr + ".pkl"
    df.to_pickle(fname)
    return fname


def process_grad_data(filename="_results/grad_comp_20241202T1158.pkl"):
    """Post process gradient data from trials"""
    # Read file
    df = read_pickle(filename)

    # get values for the SVD solution
    df_svd = df[df["estimator"] == "svd"]
    iInput = 0
    err_svd = df_svd["xi_err"].values[0]
    jacs_svd = torch.stack(
        [jacs[iInput] for jacs in df_svd["jacobians"].values[0]]
    ).numpy()
    jacs_norm_svd = np.sqrt(np.einsum("bij,bij->b", jacs_svd, jacs_svd))

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
        jac_norms = np.sqrt(np.einsum("bij,bij->b", jacs, jacs))
        dot_prods = np.einsum("bij,bij->b", jacs_svd, jacs)
        cos_dist = dot_prods / jac_norms / jacs_norm_svd - 1
        mag_dist = jac_norms / jacs_norm_svd - 1
        data.append(
            dict(
                estimator=estimator,
                cos_dist_mean=np.mean(cos_dist),
                cos_dist_std=np.std(cos_dist),
                mag_dist_mean=np.mean(mag_dist),
                mag_dist_std=np.std(mag_dist),
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


def test_jac_func(estimator="lieopt-rand", n_points=30, n_batch=2, noise_std=0.0):
    # Get points
    points_s, points_t, weights, T_t_s_gt = get_point_clouds(
        n_points=n_points, n_batch=n_batch, noise_std=noise_std
    )
    assert estimator in estimator_list, ValueError("Estimator not recognized")
    # Test estimator
    T_t_s, jacobians, time_f, time_b = get_soln_and_jac(
        estimator, points_t, points_s, weights, T_t_s_gt=T_t_s_gt
    )

    try:
        np.testing.assert_allclose(T_t_s, T_t_s_gt, atol=1e-6)
    except:
        if estimator == "lieopt-rand":
            print("Converged to local min")


if __name__ == "__main__":
    # Tests
    # test_jac_func("svd")
    # test_jac_func("sdpr")
    # test_jac_func("sdpr-qcqpdiff", noise_std=0.5)
    test_jac_func("sdpr-qcqpdiff-reuse", n_batch=50)
    # test_jac_func("lieopt-gt-unroll")
    # test_jac_func("lieopt-rand")

    # # Generate data (no noise)
    # fname = gen_estimator_data(n_points=30, n_batch=100, noise_std=0.1)
    # # Post process
    # process_grad_data(filename=fname)

    # process_grad_data(filename="_results/grad_comp_20241202T1616.pkl")
