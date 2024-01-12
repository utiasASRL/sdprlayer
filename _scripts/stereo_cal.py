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
    b=0.7,
    sigma_u=0.5 * 0,
    sigma_v=0.5 * 0,
)

torch.set_default_dtype(torch.float64)


def get_cal_data(
    N_batch=3,  # Number of poses
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
    r_p0s = []
    C_p0s = []

    # GT poses equally spaced along n turns of a circle
    radius = np.linalg.norm(offs)
    assert radius > 0.2, "Radius of trajectory circle should be larger"
    if N_batch > 1:
        delta_phi = n_turns * 2 * np.pi / (N_batch - 1)
        phi = 0.0
    else:
        delta_phi = n_turns
        phi = delta_phi
    for i in range(N_batch):
        # Location
        r = radius * np.array([[np.cos(phi), 0.0, np.sin(phi)]]).T
        r_p0s += [r]
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
        plot_poses(C_p0s, r_p0s, ax=ax)
        plot_map(r_l, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        r = np.linalg.norm(offs) * 1.1
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)

    # Get pixel measurements
    pixel_meass = []
    r_ls = []
    for i in range(N_batch):
        r_p = r_p0s[i]
        C_p0 = C_p0s[i]
        r_ls += [r_l]
        r_l_inC = C_p0 @ (r_l - r_p)
        pixel_meass += [cam_gt.forward(r_l_inC)]
        if plot:
            plot_pixel_meas(pixel_meass[-1])
    pixel_meass = np.stack(pixel_meass)
    r_ls = np.stack(r_ls)
    r_p0s = np.stack(r_p0s)
    C_p0s = np.stack(C_p0s)

    if plot:
        plt.show()

    return r_p0s, C_p0s, r_ls, pixel_meass


def get_random_inits(radius, N_batch=3, plot=False):
    r_p0s = []
    C_p0s = []

    for i in range(N_batch):
        # random locations
        r_ = np.random.random((3, 1)) - 0.5
        r = radius * r_ / np.linalg.norm(r_)
        r_p0s += [r]
        # random orientation pointing at origin
        z = -r / np.linalg.norm(r)
        y = np.random.randn(3, 1)
        y = y - y.T @ z * z
        y = y / np.linalg.norm(y)
        x = -skew(z) @ y
        C_p0s += [np.hstack([x, y, z]).T]

    if plot:
        # Plot data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_poses(C_p0s, r_p0s, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        r = radius * 1.1
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_zlim(-r, r)
        plt.show()

    r_p0s = np.stack(r_p0s)
    C_p0s = np.stack(C_p0s)

    return r_p0s, C_p0s


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


def run_sdpr_cal(r_p0s, C_p0s, r_ls, pixel_meass):
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
        r_p=r_p0s,
        C_p0=C_p0s,
        r_l=r_ls,
        pixel_meas=pixel_meass,
        verbose=True,
    )

    print("Done")


def run_theseus_cal_b(r_p0s, C_p0s, r_ls, pixel_meass):
    # Convert to tensor
    pixel_meass = torch.tensor(pixel_meass)

    # dictionary of paramter test values
    param_dict = {
        "b": dict(
            offs=1,
            lr=5e-3,
            tol_grad_sq=1e-10,
            atol=2e-3,
            atol_nonoise=1e-5,
        ),
    }

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
    optim = "LBFGS"
    for key, tune_params in param_dict.items():
        # Add offset to torch param
        getattr(cam_torch, key).data += tune_params["offs"]
        # Define parameter and learning rate
        params = [getattr(cam_torch, key)]
        if optim == "Adam":
            opt = torch.optim.Adam(params, lr=tune_params["lr"])
        elif optim == "LBFGS":
            if key == "b":
                lr = 1e-1
            else:
                lr = 10
            opt = torch.optim.LBFGS(
                params,
                history_size=50,
                tolerance_change=1e-16,
                tolerance_grad=1e-16,
                lr=lr,
                max_iter=1,
                line_search_fn="strong_wolfe",
            )
        # Termination criteria
        term_crit = {
            "max_iter": 500,
            "tol_grad_sq": 1e-14,
            "tol_loss": 1e-10,
        }
        opt_kwargs = {
            "abs_err_tolerance": 1e-10,
            "rel_err_tolerance": 1e-8,
            "max_iterations": 100,
        }
        # Run Tuner
        iter_info = st.tune_stereo_params_theseus(
            cam_torch=cam_torch,
            params=params,
            opt=opt,
            term_crit=term_crit,
            r_p0s_gt=r_p0s,
            C_p0s_gt=C_p0s,
            r_ls=r_ls,
            pixel_meass=pixel_meass,
            verbose=True,
            opt_kwargs=opt_kwargs,
        )

        plt.figure()
        plt.plot(iter_info["loss"])
        plt.ylabel("Loss")
        plt.xlabel("Iteration")
        plt.show()


def find_local_minima(N_batch=1):
    r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(N_batch=N_batch,plot=False)
    N_map = r_ls.shape[2]
    # Convert to tensor
    pixel_meass = torch.tensor(pixel_meass)
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
    # Get theseus layer
    theseus_opts = {
            "abs_err_tolerance": 1e-10,
            "rel_err_tolerance": 1e-8,
            "max_iterations": 100,
        }
    layer = st.build_theseus_layer(N_map=N_map, N_batch=N_batch,opt_kwargs_in=theseus_opts)
    # invert the camera measurements
    meas, weights = cam_torch.inverse(pixel_meass)
    # Generate random initializations
    r_p0s_init, C_p0s_init = get_random_inits(radius=2.0, N_batch=100, plot=False)
    
    losses, C_sols, r_sols = [], [], []
    with torch.no_grad():    
        for i in range(r_p0s_init.shape[0]):
            theseus_inputs = {
                    "C_p0s": C_p0s_init[[i],:,:],
                    "r_p0s": r_p0s_init[[i],:,:],
                    "r_ls": torch.tensor(r_ls),
                    "meas": meas,
                    "weights": weights,
                }
        
            out, info = layer.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": True,
                    "backward_mode": "implicit",
                },
            )
            # TODO record optimal costs
            
            # TODO get optimal solutions
            
        

if __name__ == "__main__":
    # Generate data
    # r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(plot=False)
    # run_theseus_cal(r_p0s, C_p0s, r_ls, pixel_meass)
    # run_sdpr_cal(r_p0s, C_p0s, r_ls, pixel_meass)
    # r_p0s, C_p0s = get_random_inits(radius=2.0, N_batch=20, plot=True)
