#!/bin/bash/python
# boilerplate
import numpy as np
import matplotlib.pyplot as plt
import warnings

# SDPRlayer import
from sdprlayer import SDPRLayer
import torch

# Stereo problem imports
from mwcerts.stereo_problems import (
    Localization,
    skew,
)
import spatialmath.base as sm
from pylgmath.se3.transformation import Transformation as Trans


class Camera:
    def __init__(
        c, f_u=200.0, f_v=200.0, c_u=0.0, c_v=0.0, b=0.05, sigma_u=0.5, sigma_v=0.5
    ):
        c.f_u = f_u
        c.f_v = f_v
        c.c_u = c_u
        c.c_v = c_v
        c.b = b
        c.sigma_u = sigma_u
        c.sigma_v = sigma_v

    def get_intrinsic_mat(c, M=None):
        c.M = np.array(
            [
                [c.f_u, 0.0, c.c_u, 0.0],
                [0.0, c.f_v, 0.0, c.c_v, 0.0],
                [c.f_u, 0.0, c.c_u, -c.f_u * c.b],
                [0.0, c.f_v, 0.0, c.c_v, 0.0],
            ]
        )

    def forward(c, p_inC):
        """forward camera model, points to pixels"""
        p_inC = np.hstack(p_inC)
        z = p_inC[2, :]
        x = p_inC[0, :] / z
        y = p_inC[1, :] / z
        # noise
        noise = np.random.randn(4, len(x))
        # pixel measurements
        ul = c.f_u * x + c.c_u + c.sigma_u * noise[0, :]
        vl = c.f_v * y + c.c_v + c.sigma_v * noise[1, :]
        ur = ul - c.f_u * c.b / z + c.sigma_u * noise[2, :]
        vr = vl + c.sigma_v * noise[3, :]

        return ul, vl, ur, vr

    def inverse(c, ul, vl, ur, vr):
        use_torch = torch.is_tensor(c.f_u)

        # compute disparity
        d = ul - ur
        assert all(d > 0), "Negative disparity in data"
        # If using torch then convert measurements to tensors.
        if use_torch:
            d = torch.tensor(d)
            ul = torch.tensor(ul)
            vl = torch.tensor(vl)
            Sigma = torch.zeros((3, 3))
        else:
            Sigma = np.zeros((3, 3))
        # Define pixel covariance
        Sigma[0, 0] = c.sigma_u**2
        Sigma[1, 1] = c.sigma_v**2
        Sigma[2, 2] = 2 * c.sigma_u**2
        Sigma[0, 2] = c.sigma_u**2
        Sigma[2, 0] = c.sigma_u**2

        # compute euclidean measurement coordinates
        ratio = c.b / d
        x = (ul - c.c_u) * ratio
        y = (vl - c.c_v) * ratio * c.f_u / c.f_v
        z = ratio * c.f_u
        if use_torch:
            meas = torch.vstack([x, y, z])
        else:
            meas = np.vstack([x, y, z])
        # compute weights
        weights = []
        for i in range(len(ul)):
            G = torch.zeros((3, 3))
            # Define G
            G[0, 0] = z[i] / c.f_u
            G[1, 1] = z[i] / c.f_v
            G[0, 2] = -x[i] * z[i] / c.f_u / c.b
            G[1, 2] = -y[i] * z[i] / c.f_v / c.b
            G[2, 2] = -z[i] ** 2 / c.f_u / c.b

            # Covariance matrix
            Cov = G @ Sigma @ G.T
            if torch.linalg.matrix_rank(Cov) < 3:
                warnings.warn("Covariance matrix is not full rank")
                Cov = torch.eye(3)
            if use_torch:
                W = torch.linalg.inv(Cov)

            else:
                W += np.linalg.inv(Cov)
            weights += [0.5 * (W + W.T)]

        return meas, weights


def get_gt_setup(
    traj_type="clusters",  # Trajectory format [clusters,circle]
    Np=1,  # Number of poses
    Nm=10,  # number of landmarks
    offs=np.array([[0, 0, 1]]).T,  # offset between poses and landmarks
    n_turns=0.5,  # (circle) number of turns around the cluster
    lm_bound=0.9,  # Bounding box of uniform landmark distribution.
):
    """Used to generate a trajectory of ground truth pose data"""

    # Ground Truth Map Points
    # Cluster at the origin
    r_l = lm_bound * (np.random.rand(3, Nm) - 0.5)
    # Ground Truth Poses
    r_p = []
    C_p0 = []
    if traj_type == "clusters":
        # Ground Truth Poses
        for i in range(Np):
            r_p += [0.1 * np.random.randn(3, 1)]
            C_p0 += [sm.roty(0.1 * np.random.randn(1)[0])]
        # Offset from the origin
        r_l = r_l + offs
    elif traj_type == "circle":
        # GT poses equally spaced along n turns of a circle
        radius = np.linalg.norm(offs)
        assert radius > 0.2, "Radius of trajectory circle should be larger"
        if Np > 1:
            delta_phi = n_turns * 2 * np.pi / (Np - 1)
        else:
            delta_phi = 0.0
        phi = 0.0
        for i in range(Np):
            # Location
            r = radius * np.array([[np.cos(phi), np.sin(phi), 0]]).T
            r_p += [r]
            # Z Axis points at origin
            z = -r / np.linalg.norm(r)
            x = np.array([[0.0, 0.0, 1.0]]).T
            y = skew(z) @ x
            C_p0 += [np.hstack([x, y, z]).T]
            # Update angle
            phi = (phi + delta_phi) % (2 * np.pi)

    r_l = np.expand_dims(r_l.T, axis=2)

    return r_p, C_p0, r_l


def get_prob_data(camera=Camera(), Nm=30):
    # get ground truth information
    r_p, C_p0, r_l = get_gt_setup(Nm=Nm)

    # generate measurements
    r_l_inC = [C_p0[0] @ (r_l_i - r_p[0]) for r_l_i in r_l]
    pixel_meas = camera.forward(r_l_inC)

    return r_p, C_p0, r_l, pixel_meas


def get_data_mat(cam_torch: Camera, r_l, pixel_meas):
    # Get euclidean measurements from pixels
    meas, weights = cam_torch.inverse(*pixel_meas)
    y = meas.float()
    # Indices
    h = [0]
    c = slice(1, 10)
    t = slice(10, 13)
    Qs = []
    for i in range(meas.shape[1]):
        W_ij = weights[i].contiguous()
        m_j0_0 = torch.tensor(r_l[i], dtype=torch.float32).contiguous()
        y_ji_i = y[:, [i]]
        # Define matrix
        Q_e = torch.zeros(13, 13)
        Q_e[c, c] = torch.kron(m_j0_0 @ m_j0_0.T, W_ij)
        Q_e[c, t] = -torch.kron(m_j0_0, W_ij)
        Q_e[t, c] = -torch.kron(m_j0_0, W_ij).T
        Q_e[c, h] = -torch.kron(m_j0_0, W_ij @ y_ji_i)
        Q_e[h, c] = -torch.kron(m_j0_0, W_ij @ y_ji_i).T
        Q_e[t, t] = W_ij
        Q_e[t, h] = W_ij @ y_ji_i
        Q_e[h, t] = (W_ij @ y_ji_i).T
        Q_e[h, h] = y_ji_i.T @ W_ij @ y_ji_i
        # Add to overall matrix
        Qs += [Q_e]
    Q = torch.stack(Qs)
    Q = torch.sum(Q, dim=0)
    # TODO Find out why this is required (it should not be)
    Q = (Q.T + Q) / 2
    # Rescale
    Q[0, 0] = 0.0
    Q = Q / torch.norm(Q, p="fro")

    return Q


# Loss Function
def get_loss(X, r_p_in0, C_p0):
    # Convert to tensors
    C_p0 = torch.tensor(C_p0[0], dtype=torch.float32)
    r_p_in0 = torch.tensor(r_p_in0[0], dtype=torch.float32)
    # Extract solution (assume Rank-1)
    r = X[10:, [0]]
    C_vec = X[1:10, [0]]
    C = C_vec.reshape((3, 3))
    # Define loss as difference to ground truth
    loss = torch.norm(r - r_p_in0) + torch.norm(C.T @ C_p0 - torch.eye(3), p="fro")
    return loss


def tune_stereo_params(cam_torch: Camera, r_p, C_p0, r_l, pixel_meas):
    # Define a localization class to get the constraints
    prob = Localization(r_p, C_p0, r_l)
    prob.generate_constraints()
    prob.generate_redun_constraints()
    constraints = prob.constraints + prob.constraints_r

    constraints_list = [(c.A.get_matrix(prob.var_list), c.b) for c in constraints]

    # Build Layer
    sdpr_layer = SDPRLayer(13, Constraints=constraints_list, use_dual=True)

    # Define optimizer
    params = [cam_torch.b]
    opt = torch.optim.SGD(params=params, lr=1)

    torch.autograd.set_detect_anomaly(True)

    # Optimization loop
    max_iter = 200
    tol_grad_sq = 1e-10
    grad_sq = np.inf
    n_iter = 0
    loss_stored = []
    while grad_sq > tol_grad_sq and n_iter < max_iter:
        # zero grad
        opt.zero_grad()
        # build loss
        Q = get_data_mat(cam_torch, r_l, pixel_meas)
        solver_args = {"solve_method": "mosek"}
        X = sdpr_layer(Q, solver_args=solver_args)[0]
        loss = get_loss(X, r_p, C_p0)
        loss_stored += [loss.item()]
        # Back prop and update
        loss.backward()
        grad = np.vstack([p.grad for p in params])
        grad_sq = np.sum([g**2 for g in grad])
        opt.step()
        print(f"Iter:\t{n_iter}\tLoss:\t{loss_stored[-1]}")
        n_iter += 1
