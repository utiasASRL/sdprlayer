import os

import matplotlib.pyplot as plt
import numpy as np
from pickle import dump, load
import torch

from mwcerts.stereo_problems import skew
import sdprlayer.stereo_tuner as st


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

torch.set_default_dtype(torch.float64)


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_cal_data(
    N_batch=3,  # Number of poses
    N_map=10,  # number of landmarks
    offs=np.array([[0, 0, 2]]).T,  # offset between poses and landmarks
    n_turns=0.2,  # (circle) number of turns around the cluster
    board_dims=np.array([0.3, 0.3]),  # width and height of calibration board
    N_squares=[10, 10],  # number of squares in calibration board (width, height)
    plot=False,
):
    """Generate Ground truth pose and landmark data. Also generate pixel measurements"""
    # Ground Truth Map Points
    sq_size = board_dims / np.array(N_squares)
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
            if "color" in kwargs:
                ax.quiver(*origin, *directions[:, j], **kwargs)
            else:
                ax.quiver(
                    *origin, *directions[:, j], color=["r", "g", "b"][j], **kwargs
                )


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

    ax.plot(*r_l, ".", color="k")


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
    iter_info = st.tune_stereo_params_sdpr(
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


# Comparison scripts


def tune_baseline_theseus(local=True):
    """Comparison of baseline tunings when inner optimization starts
    from either local or global minimum"""
    # load data
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(folder, "outputs")
    prob_data = load(open(folder + "/stereo_cal_local_min_init.pkl", "rb"))
    r_p0s = prob_data["r_p0s"]
    C_p0s = prob_data["C_p0s"]
    r_ls = torch.tensor(prob_data["r_ls"])
    r_ls.unsqueeze_(0)  # add batch dimension
    pixel_meass = prob_data["pixel_meass"]
    if local:
        r_p0s_init = prob_data["r_p0s_init_l"].unsqueeze(0)
        C_p0s_init = prob_data["C_p0s_init_l"].unsqueeze(0)
    else:
        r_p0s_init = prob_data["r_p0s_init_g"].unsqueeze(0)
        C_p0s_init = prob_data["C_p0s_init_g"].unsqueeze(0)
    cam_torch = prob_data["cam_torch"]

    # Define parameter and learning rate
    params = [cam_torch.b]

    opt = torch.optim.Adam(params, lr=1e-2, betas=(0.7, 0.999))
    # Termination criteria
    term_crit = {
        "max_iter": 500,
        "tol_grad_sq": 1e-10,
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
        r_p0s_init=r_p0s_init,
        C_p0s_init=C_p0s_init,
        verbose=True,
        opt_kwargs=opt_kwargs,
    )

    return iter_info


def tune_baseline_sdpr():
    # load data
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(folder, "outputs")
    prob_data = load(open(folder + "/stereo_cal_local_min_init.pkl", "rb"))
    r_p0s = prob_data["r_p0s"]
    C_p0s = prob_data["C_p0s"]
    r_ls = prob_data["r_ls"]
    pixel_meass = prob_data["pixel_meass"]
    cam_torch = prob_data["cam_torch"]

    # Define parameter and learning rate
    params = [cam_torch.b]

    opt = torch.optim.Adam(params, lr=1e-2, betas=(0.9, 0.999))
    # Termination criteria
    term_crit = {
        "max_iter": 500,
        "tol_grad_sq": 1e-10,
        "tol_loss": 1e-10,
    }
    # Run Tuner
    iter_info = st.tune_stereo_params_sdpr(
        cam_torch=cam_torch,
        params=params,
        opt=opt,
        term_crit=term_crit,
        r_p0s_gt=r_p0s,
        C_p0s_gt=C_p0s,
        r_ls=r_ls,
        pixel_meass=pixel_meass,
        verbose=True,
    )

    return iter_info


def tune_baseline(tuner="spdr", N_batch=20):
    set_seed(0)
    # Generate data
    offs = np.array([[0, 0, 3]]).T
    r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
        offs=offs, board_dims=[0.6, 1.0], N_squares=[8, 8], N_batch=N_batch, plot=False
    )
    r_p0s = torch.tensor(r_p0s)
    C_p0s = torch.tensor(C_p0s)
    r_ls = torch.tensor(r_ls)
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

    # Define parameter to tune
    params = [cam_torch.b]
    # Define optimizer
    opt = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
    # Termination criteria
    term_crit = {
        "max_iter": 500,
        "tol_grad_sq": 1e-10,
        "tol_loss": 1e-10,
    }
    # Run Tuner
    if tuner == "spdr":
        iter_info = st.tune_stereo_params_sdpr(
            cam_torch=cam_torch,
            params=params,
            opt=opt,
            term_crit=term_crit,
            r_p0s_gt=r_p0s,
            C_p0s_gt=C_p0s,
            r_ls=r_ls,
            pixel_meass=pixel_meass,
            verbose=True,
        )
    else:
        # Generate random initializations
        radius = np.linalg.norm(offs)
        r_p0s_init, C_p0s_init = get_random_inits(
            radius=radius, N_batch=N_batch, plot=False
        )
        r_p0s_init = torch.tensor(r_p0s_init)
        C_p0s_init = torch.tensor(C_p0s_init)
        # opt parameters
        opt_kwargs = {
            "abs_err_tolerance": 1e-10,
            "rel_err_tolerance": 1e-8,
            "max_iterations": 100,
            "step_size": 0.5,
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
            r_p0s_init=r_p0s_init,
            C_p0s_init=C_p0s_init,
            verbose=True,
            opt_kwargs=opt_kwargs,
        )

    return iter_info


def compare_tune_baseline(N_batch=20):
    """Compare tuning of baseline parameters with SDPR and Theseus.
    Use actual batch of measurements"""
    info_s = tune_baseline("spdr", N_batch=N_batch)
    info_l = tune_baseline("theseus", N_batch=N_batch)

    # Plot loss
    plt.figure()
    plt.plot(info_l["loss"], label="Theseus")
    plt.plot(info_s["loss"], label="SDPR")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()

    # Plot parameter values
    plt.figure()
    plt.plot(info_l["params"], label="Theseus")
    plt.plot(info_s["params"], label="SDPR")
    plt.ylabel("Baseline Value")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
    print("Done")


def compare_tune_baseline_single():
    """Compare tuning of baseline parameters with SDPR and Theseus.
    Use a single currated inner opttimization with local and global min"""
    info_s = tune_baseline_sdpr()
    info_g = tune_baseline_theseus(local=False)
    info_l = tune_baseline_theseus(local=True)

    # Plot loss
    plt.figure()
    plt.plot(info_l["loss"], label="Local init")
    plt.plot(info_g["loss"], label="Global init")
    plt.plot(info_s["loss"], label="SDPR")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()

    # Plot parameter values
    plt.figure()
    plt.plot(info_l["params"], label="Local init")
    plt.plot(info_g["params"], label="Global init")
    plt.plot(info_s["params"], label="SDPR")
    plt.ylabel("Baseline Value")
    plt.xlabel("Iteration")
    plt.legend()
    plt.show()
    print("Done")


def find_local_minima(N_inits=100, store_data=False):
    set_seed(0)

    # Generate data
    offs = np.array([[0, 0, 3]]).T
    r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
        offs=offs, board_dims=[0.6, 1.0], N_squares=[8, 8], N_batch=1, plot=False
    )
    r_p0s = torch.tensor(r_p0s)
    C_p0s = torch.tensor(C_p0s)
    r_ls = torch.tensor(r_ls)
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
    layer = st.build_theseus_layer(
        N_map=N_map, N_batch=N_inits, opt_kwargs_in=theseus_opts
    )
    # invert the camera measurements
    meas, weights = cam_torch.inverse(pixel_meass)
    # Generate random initializations
    radius = np.linalg.norm(offs)
    r_p0s_init, C_p0s_init = get_random_inits(
        radius=radius, N_batch=N_inits, plot=False
    )
    r_p0s_init = torch.tensor(r_p0s_init)
    C_p0s_init = torch.tensor(C_p0s_init)

    with torch.no_grad():
        # Set initializations and measurements
        theseus_inputs = {
            "C_p0s": C_p0s_init,
            "r_p0s": r_p0s_init[:, :, 0],
            "r_ls": torch.tensor(r_ls),
            "meas": meas,
            "weights": weights,
        }
        # Run theseus
        out, info = layer.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": True,
                "backward_mode": "implicit",
            },
        )
        # get optimal solutions
        C_sols = out["C_p0s"]
        r_sols = out["r_p0s"]
        # record optimal costs
        losses = info.best_err.detach().numpy()
    C_sols = C_sols.detach().numpy()
    r_sols = r_sols.detach().numpy()
    # Show loss distribution
    plt.figure()
    plt.hist(losses)
    plt.show()

    # Plot final solutions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(C_sols, r_sols, ax=ax, alpha=0.2)
    plot_poses(C_p0s, r_p0s, ax=ax, color="k")
    plot_map(r_ls[0].detach().numpy(), ax=ax)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    r = radius * 1.1
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    plt.show()

    # Find local minima
    loss_min = np.min(losses)
    ind_local = np.where(np.abs(losses - loss_min) > 20)[0]
    ind_global = np.where(np.abs(losses - loss_min) < 20)[0]

    # Store data
    if store_data:
        prob_data = dict(
            cam_torch=cam_torch,
            r_p0s=r_p0s,
            C_p0s=C_p0s,
            r_ls=r_ls,
            pixel_meass=pixel_meass,
            r_p0s_init_l=r_p0s_init[ind_local[0]],
            C_p0s_init_l=C_p0s_init[ind_local[0]],
            r_p0s_init_g=r_p0s_init[ind_global[0]],
            C_p0s_init_g=C_p0s_init[ind_global[0]],
        )
        folder = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(folder, "outputs")
        dump(prob_data, open(folder + "/stereo_cal_local_min_init.pkl", "wb"))


if __name__ == "__main__":
    # Generate data
    # r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(plot=False)
    # run_theseus_cal(r_p0s, C_p0s, r_ls, pixel_meass)
    # run_sdpr_cal(r_p0s, C_p0s, r_ls, pixel_meass)
    # r_p0s, C_p0s = get_random_inits(radius=2.0, N_batch=20, plot=True)
    # find_local_minima(store_data=True)
    compare_tune_baseline()
    # compare_tune_baseline_single()
