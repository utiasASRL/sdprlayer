from contextlib import AbstractContextManager
from typing import Any
import numpy as np
import torch
import unittest
import matplotlib.pyplot as plt
import sdprlayer.stereo_tuner as st
from mwcerts.stereo_problems import Localization, skew
from sdprlayer import SDPRLayer


# Define camera ground truth
cam_gt = st.Camera(
    f_u=484.5,
    f_v=484.5,
    c_u=0.0,
    c_v=0.0,
    b=0.24,
    sigma_u=0.5,
    sigma_v=0.5,
)


def get_cal_data(
    Np=3,  # Number of poses
    N_map=10,  # number of landmarks
    offs=np.array([[0, 0, 2]]).T,  # offset between poses and landmarks
    n_turns=0.2,  # (circle) number of turns around the cluster
    board_dims=np.array([0.3, 0.3]),  # width and height of calibration board
    N_squares=np.array(
        [10, 10]
    ),  # number of squares in calibration board (width, height)
    plot=False,
):
    """Generate Ground truth pose and landmark data. Also generate pixel measurements"""
    # Ground Truth Map Points
    sq_size = board_dims / N_squares
    r_l = []
    for i in range(N_squares[0]):
        for j in range(N_squares[1]):
            r_l += [np.array([[0.0, i * sq_size[0], j * sq_size[1]]]).T]
    r_l = np.hstack(r_l)
    r_l = r_l - np.mean(r_l, axis=1, keepdims=True)
    # Ground Truth Poses
    r_ps = []
    C_p0s = []

    # GT poses equally spaced along n turns of a circle
    radius = np.linalg.norm(offs)
    assert radius > 0.2, "Radius of trajectory circle should be larger"
    if Np > 1:
        delta_phi = n_turns * 2 * np.pi / (Np - 1)
        phi = 0.0
    else:
        delta_phi = n_turns
        phi = delta_phi
    for i in range(Np):
        # Location
        r = radius * np.array([[np.cos(phi), 0.0, np.sin(phi)]]).T
        r_ps += [r]
        # Z Axis points at origin
        z = -r / np.linalg.norm(r)
        y = np.array([[0.0, 1.0, 0.0]]).T
        x = -skew(z) @ y
        C_p0s += [np.hstack([x, y, z]).T]
        # Update angle
        phi = (phi + delta_phi) % (2 * np.pi)

    if plot:
        # Plot data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_poses(C_p0s, r_ps, ax=ax)
        plot_map(r_l, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        r = np.linalg.norm(offs) * 1.1
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)

    # Get pixel measurements
    r_ls = []
    pixel_meass = []
    for i in range(Np):
        # get landmarks in camera frame
        r_l_c = C_p0s[i] @ (r_l - r_ps[i])
        # get pixel measurements
        pixel_meass += [cam_gt.forward([r_l_c])]
        # repeat same landmarks for each pose
        r_ls += [np.expand_dims(r_l.T, axis=2)]

        if plot:
            plot_pixel_meas(pixel_meass[-1])

    if plot:
        plt.show()

    return r_ps, C_p0s, r_ls, pixel_meass


def plot_poses(R_cw, t_cw_w, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i in range(len(R_cw)):
        origin = t_cw_w[i]
        directions = R_cw[i].T

        for j in range(3):
            ax.quiver(*origin, *directions[:, j], color=["r", "g", "b"][j])


def plot_pixel_meas(pixel_meas, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    for i in range(len(pixel_meas)):
        u, v = pixel_meas[:2]
        ax.plot(u, v, "o", color="r")
        u, v = pixel_meas[2:]
        ax.plot(u, v, "o", color="b")
    plt.axis("equal")
    plt.xlim(-cam_gt.f_u, cam_gt.f_u)
    plt.ylim(-cam_gt.f_v, cam_gt.f_v)


def plot_map(r_l, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*r_l, ".", color="k")


def run_sdpr_cal(r_ps, C_p0s, r_ls, pixel_meass):
    # generate parameterized camera
    cam_torch = st.Camera(
        f_u=torch.tensor(cam_gt.f_u, requires_grad=True),
        f_v=torch.tensor(cam_gt.f_v, requires_grad=True),
        c_u=torch.tensor(cam_gt.c_u, requires_grad=True),
        c_v=torch.tensor(cam_gt.c_v, requires_grad=True),
        b=torch.tensor(cam_gt.b, requires_grad=True),
        sigma_u=cam_gt.sigma_u,
        sigma_v=cam_gt.sigma_v,
    )
    params = [cam_torch.b]
    # Set up optimizer
    opt = torch.optim.Adam(params, lr=1e-3)

    # Termination criteria
    term_crit = {"max_iter": 2000, "tol_grad_sq": 1e-12, "tol_loss": 1e-12}

    # Run Tuner
    iter_info = st.tune_stereo_params(
        cam_torch=cam_torch,
        params=params,
        opt=opt,
        term_crit=term_crit,
        r_p=r_ps,
        C_p0=C_p0s,
        r_l=r_ls,
        pixel_meas=pixel_meass,
        verbose=True,
    )

    print("Done")

def run_theseus_cal(r_ps, C_p0s, r_ls, pixel_meass):
    
    


if __name__ == "__main__":
    # Generate data
    r_ps, C_p0s, r_ls, pixel_meass = get_cal_data(plot=False)
    run_sdpr_cal(r_ps, C_p0s, r_ls, pixel_meass)