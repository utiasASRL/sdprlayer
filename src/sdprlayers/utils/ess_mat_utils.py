import time

import kornia.geometry.epipolar as epi
import matplotlib.pyplot as plt
import numpy as np
import torch
from pylgmath.so3.operations import vec2rot
from tqdm import tqdm

import sdprlayers.utils.fund_mat_utils as utils
from sdprlayers import SDPEssMatEst
from sdprlayers.utils.camera_model import CameraModel
from sdprlayers.utils.lie_algebra import se3_exp, so3_wedge
from sdprlayers.utils.plot_tools import plot_map, plot_poses


def get_gt_setup(
    traj_type="clusters",  # Trajectory format [clusters,circle]
    N_batch=1,  # Number of poses
    N_map=10,  # number of landmarks
    offs=np.array([[0, 0, 2]]).T,  # offset between poses and landmarks
    n_turns=0.1,  # (circle) number of turns around the cluster
    lm_bound=1.0,  # Bounding box of uniform landmark distribution.
):
    """Used to generate a trajectory of ground truth pose data"""

    # Ground Truth Map Points
    # Cluster at the origin
    r_l = lm_bound * 2 * (np.random.rand(3, N_map) - 0.5)
    # Ground Truth Poses
    r_p0s = []
    C_p0s = []
    if traj_type == "clusters":
        # Ground Truth Poses
        for i in range(N_batch):
            r_p0s += [0.2 * np.random.randn(3, 1)]
            aaxis = np.zeros((3, 1))
            aaxis[1, 0] = 0.5 * np.random.randn(1)[0]
            C_p0s.append(vec2rot(aaxis_ba=aaxis))
        # Offset from the origin
        r_l = r_l + offs
    elif traj_type == "circle":
        # GT poses equally spaced along n turns of a circle
        radius = np.linalg.norm(offs)
        assert radius > 0.2, "Radius of trajectory circle should be larger"
        if N_batch > 1:
            delta_phi = n_turns * 2 * np.pi / (N_batch - 1)
        else:
            delta_phi = n_turns * 2 * np.pi
        phi = delta_phi
        for i in range(N_batch):
            # Location
            r = radius * np.array([[np.cos(phi), np.sin(phi), 0]]).T
            r_p0s += [r]
            # Z Axis points at origin
            z = -r / np.linalg.norm(r)
            x = np.array([[0.0, 0.0, 1.0]]).T
            y = skew(z) @ x
            C_p0s += [np.hstack([x, y, z]).T]
            # Update angle
            phi = (phi + delta_phi) % (2 * np.pi)
    r_p0s = np.stack(r_p0s)
    C_p0s = np.stack(C_p0s)

    return r_p0s, C_p0s, r_l


def get_prob_data(camera=None, N_map=30, N_batch=1):
    # get ground truth information
    r_p0s, C_p0s, r_l = get_gt_setup(N_map=N_map, N_batch=N_batch)

    # generate measurements
    pixel_meass = []
    r_ls = []
    for i in range(N_batch):
        r_p = r_p0s[i]
        C_p0 = C_p0s[i]
        r_ls += [r_l]
        r_l_inC = C_p0 @ (r_l - r_p)
        pixel_meass += [camera.forward(r_l_inC)]
    pixel_meass = np.stack(pixel_meass)
    r_ls = np.stack(r_ls)

    # Convert to torch tensors
    r_p0s = torch.tensor(r_p0s)
    C_p0s = torch.tensor(C_p0s)
    r_ls = torch.tensor(r_ls)
    pixel_meass = torch.tensor(pixel_meass)

    return r_p0s, C_p0s, r_ls, pixel_meass


def kron(A, B):
    # kronecker workaround for matrices
    # https://github.com/pytorch/pytorch/issues/74442
    return (A[:, None, :, None] * B[None, :, None, :]).reshape(
        A.shape[0] * B.shape[0], A.shape[1] * B.shape[1]
    )


def skew(vec):
    if vec.shape == (3,):
        vec = np.expand_dims(vec, axis=1)
    assert vec.shape == (3, 1), "Input vector must have shape (3,1)"
    return np.array(
        [
            [0.0, -vec[2, 0], vec[1, 0]],
            [vec[2, 0], 0.0, -vec[0, 0]],
            [-vec[1, 0], vec[0, 0], 0.0],
        ]
    )


# def get_cv2_solution(srcs, trgs, wts, K=torch.eye(4)):


def get_kornia_solution(srcs, trgs, wts, K=torch.eye(3)):
    """Get the essential matrix using the kornia library.
    This uses Nister's 5 point algorithm."""
    # Reshape to kornia format
    srcs_krn = srcs[:, :2, :].mT
    trgs_krn = trgs[:, :2, :].mT
    wts_krn = wts[:, 0, :]
    # get essential matrix
    Es_kornia = epi.find_essential(srcs_krn, trgs_krn, wts_krn)
    # find the best of the 10 kornia solutions using the sampson distance
    dists = []
    for i in range(10):
        # Epipolar distances for all points
        point_dists = epi.sampson_epipolar_distance(
            srcs_krn, trgs_krn, Es_kornia[:, i, :, :]
        )
        # Sum distances across points
        dists.append(torch.sum(point_dists, 1))
    dists = torch.stack(dists, 1)
    # Get singular values
    _, sv, _ = torch.linalg.svd(Es_kornia)
    # Normalize Essential Matrix
    Es_kornia = Es_kornia / sv[..., [0], None].expand(1, 10, 3, 3)
    # Null space check
    dists_chkd = torch.where(sv[..., 2] < 1e-9, dists, 1000)
    # Get index of best solution.
    ind = torch.argmin(dists_chkd, 1)
    n_batch = Es_kornia.shape[0]
    Es_kornia_best = Es_kornia[torch.arange(n_batch), ind, :, :]
    # Normaliz
    # Decompose solution
    K = torch.eye(3).expand(n_batch, 3, 3)
    Rs, ts, points = epi.motion_from_essential_choose_solution(
        Es_kornia_best,
        K,
        K,
        srcs_krn,
        trgs_krn,
    )
    Es = Es_kornia_best

    return Es, ts, Rs


def get_soln_and_jac(estimator, points_t, points_s, weights, K, tol=1e-12, **kwargs):
    """Apply estimator to point clouds and obtain solution and solution gradient.
    NOTE: gradients are computed sequentially using torch's grad function and then assembled into a Jacobian for each input. We also loop over the batch dimension.
    All computations are done on the CPU sequentially.

    "jacobians" output has dimensions B x (num inputs) x (output dims) x (input dims).
    """
    n_batch = points_t.shape[0]  # number of batches
    n_points = points_t.shape[2]  # number of points in the point cloud
    precision = points_t.dtype  # precision of the points
    # Create estimator module
    if estimator == "sdpr-sdp":
        forward = SDPEssMatEst(K_source=K, K_target=K, diff_qcqp=False, tol=tol)
    elif estimator == "sdpr-cift":
        forward = SDPEssMatEst(
            K_source=K, K_target=K, diff_qcqp=True, compute_multipliers=True, tol=tol
        )
    elif estimator == "sdpr-is":
        forward = SDPEssMatEst(
            K_source=K, K_target=K, diff_qcqp=True, compute_multipliers=False, tol=tol
        )
    elif estimator == "kornia":
        forward = get_kornia_solution
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
        if estimator in "kornia":
            # Add intrinsic camera matrix for Kornia solution
            kwargs.update(dict(K=K))
        # Run Estimator
        outputs = forward(*inputs, **kwargs)
        estimates.append(outputs[0])
        Tf_1 = time.time()
        times_f.append(Tf_1 - Tf_0)
        # Compute Jacobian
        # NOTE: We do this by manually looping to avoid issues with vmap
        # Output gradient vectors
        grad_outputs = torch.eye(9).reshape(9, 3, 3)
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
    # batch estimates
    estimates = torch.concat(estimates, dim=0).detach()

    return estimates, jacobians, time_f, time_b


def compute_cost(srcs, trgs, weights, Es_est):
    costs = []
    B, _, N = srcs.shape
    for b in range(B):
        src = srcs[b].cpu().numpy()
        trg = trgs[b].cpu().numpy()
        E_est = Es_est[b].detach().numpy()
        cost = 0.0
        for i in range(N):
            cost += weights[b, :, i] * (trg[:, [i]].T @ E_est @ src[:, [i]]) ** 2
        costs.append(cost)
    return costs
