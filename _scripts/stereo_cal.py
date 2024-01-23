import os

import matplotlib.pyplot as plt
import numpy as np
from pickle import dump, load
from spatialmath import SO3
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


def get_points_in_cone(radius, beta, N_batch):
    """Generate N random points in a cone of radius and angle beta"""
    # Generate N random azimuthal angles between 0 and 2*pi
    polar = 2 * np.pi * np.random.rand(N_batch)

    # Generate N random polar angles between 0 and beta
    azimuth = beta * (2 * np.random.rand(N_batch) - 1)

    # Z axis rotation
    Cz = SO3.Rz(azimuth)
    Cx = SO3.Rx(polar)
    C = Cx * Cz
    # get x axes
    Cs = C.A
    if isinstance(Cs, list):
        points = [c[:, [0]] * radius for c in Cs]
    else:
        points = [Cs[:, [0]] * radius]

    return points


def get_cal_data(
    N_batch=3,  # Number of poses
    N_map=10,  # number of landmarks
    radius=2,  # offset between poses and landmarks
    n_turns=0.2,  # (circle) number of turns around the cluster
    board_dims=np.array([0.3, 0.3]),  # width and height of calibration board
    N_squares=[10, 10],  # number of squares in calibration board (width, height)
    setup="circle",  # setup of GT poses
    cone_angles=(np.pi / 4, np.pi / 4),  # camera FOV (alpha), region cone (beta)
    plot=False,
    plot_pixel_meas=False,
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
    assert radius > 0.2, "Radius of trajectory circle should be larger"
    if setup == "circle":  # GT poses equally spaced along n turns of a circle
        offs = np.array([[0, 0, radius]]).T
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
    elif setup == "cone":  # Setup GT poses in a cone
        alpha, beta = cone_angles  # Note FOV divided by 2
        # Pick location
        r_p0s = get_points_in_cone(radius, beta, N_batch)
        # FOV perturbations

        C_p0s = []
        for i in range(N_batch):
            # random orientation pointing at origin
            z = -r_p0s[i] / np.linalg.norm(r_p0s[i])
            y = np.random.randn(3, 1)
            y = y - y.T @ z * z
            y = y / np.linalg.norm(y)
            x = -skew(z) @ y
            C = np.hstack([x, y, z]).T
            # Perturb orientation
            C = SO3.Rx((2 * np.random.random() - 1) * alpha) @ SO3(C)
            C_p0s += [C.A]

    if plot:
        # Plot data
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_poses(C_p0s, r_p0s, ax=ax)
        plot_map(r_l, ax=ax)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        r = np.linalg.norm(radius) * 1.1
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
        if plot_pixel_meas:
            plot_pixel_meas(pixel_meass[-1])
    pixel_meass = torch.tensor(np.stack(pixel_meass))
    r_ls = torch.tensor(np.stack(r_ls))
    C_p0s = torch.tensor(np.stack(C_p0s))
    r_p0s = torch.tensor(np.stack(r_p0s))

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


def find_local_minima(N_inits=100, store_data=False, **kwargs):
    set_seed(5)
    radius = 3
    # Generate data
    r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
        radius=radius,
        board_dims=[0.6, 1.0],
        N_squares=[8, 8],
        N_batch=1,
        plot=False,
        setup="cone",
        **kwargs,
    )
    r_p0s = torch.tensor(r_p0s)
    C_p0s = torch.tensor(C_p0s)
    r_ls = torch.tensor(r_ls)
    pixel_meass = torch.tensor(pixel_meass)
    N_map = r_ls.shape[2]
    # Convert to tensor

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
        "max_iterations": 300,
        "step_size": 0.2,
    }
    layer = st.build_theseus_layer(
        N_map=N_map, N_batch=N_inits, opt_kwargs_in=theseus_opts
    )
    # invert the camera measurements
    meas, weights = cam_torch.inverse(pixel_meass)
    # Generate random initializations
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
            "r_ls": r_ls,
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
    plt.xlabel("Loss")
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

    # Find local minima
    loss_min = np.min(losses)
    ind_local = np.where(np.abs(losses - loss_min) > 5)[0]
    ind_global = np.where(np.abs(losses - loss_min) < 5)[0]

    r_p0s_init_l = r_p0s_init[ind_local]
    C_p0s_init_l = C_p0s_init[ind_local]
    r_p0s_init_g = r_p0s_init[ind_global]
    C_p0s_init_g = C_p0s_init[ind_global]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(C_sols, r_sols, ax=ax, alpha=0.2)
    plot_poses(C_p0s, r_p0s, ax=ax, color="k")
    plot_map(r_ls[0].detach().numpy(), ax=ax)
    plot_poses(C_p0s_init_l, r_p0s_init_l, ax=ax, color="r")
    plot_poses(C_p0s_init_g, r_p0s_init_g, ax=ax, color="g")
    plt.title("Local (red) and Global (green) Initializations")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    r = radius * 1.1
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    plt.show()

    # Double check
    with torch.no_grad():
        # Set initializations and measurements
        theseus_inputs = {
            "C_p0s": C_p0s_init_l,
            "r_p0s": r_p0s_init_l[:, :, 0],
            "r_ls": r_ls,
            "meas": meas,
            "weights": weights,
        }
        # Run theseus
        assert len(r_p0s_init_l) > 0, "No local minima found"
        out, info = layer.forward(
            theseus_inputs,
            optimizer_kwargs={
                "track_best_solution": True,
                "verbose": True,
                "backward_mode": "implicit",
            },
        )
        assert torch.all(info.best_err > 120.0), "Local minima should have higher loss"

    # Store data
    if store_data:
        prob_data = dict(
            cam_torch=cam_torch,
            r_p0s=r_p0s,
            C_p0s=C_p0s,
            r_ls=r_ls,
            pixel_meass=pixel_meass,
            r_p0s_init_l=r_p0s_init_l[0],
            C_p0s_init_l=C_p0s_init_l[0],
            r_p0s_init_g=r_p0s_init_g[0],
            C_p0s_init_g=C_p0s_init_g[0],
        )
        folder = os.path.dirname(os.path.realpath(__file__))
        folder = os.path.join(folder, "outputs")
        dump(prob_data, open(folder + "/stereo_cal_local_min_init.pkl", "wb"))


# BATCH TUNING COMPARISON


def tune_baseline(
    tuner="spdr",
    opt_select="sgd",
    b_offs=0.01,
    gt_init=False,
    N_batch=20,
    radius=3,
    prob_data=(),
    term_crit={},
):
    # Get problem data
    r_p0s, C_p0s, r_ls, pixel_meass = prob_data
    N_map = r_ls.shape[2]
    # termination criteria
    default_term_crit = {
        "max_iter": 15,
        "tol_grad_sq": 1e-8,
        "tol_loss": 1e-10,
    }
    default_term_crit.update(term_crit)
    term_crit = default_term_crit

    # generate parameterized camera
    cam_torch = st.Camera(
        f_u=torch.tensor(cam_gt.f_u, requires_grad=True),
        f_v=torch.tensor(cam_gt.f_v, requires_grad=True),
        c_u=torch.tensor(cam_gt.c_u, requires_grad=True),
        c_v=torch.tensor(cam_gt.c_v, requires_grad=True),
        b=torch.tensor(cam_gt.b + b_offs, requires_grad=True),
        sigma_u=cam_gt.sigma_u,
        sigma_v=cam_gt.sigma_v,
    )

    # Define parameter to tune
    params = [cam_torch.b]
    # Define optimizer
    if opt_select == "sgd":
        opt = torch.optim.SGD(params, lr=1e-4)
    elif opt_select == "adam":
        opt = torch.optim.Adam(params, lr=1e-3)
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
        if gt_init:
            r_p0s_init = r_p0s.clone()
            C_p0s_init = C_p0s.clone()
        else:  # random init
            r_p0s_init, C_p0s_init = get_random_inits(
                radius=radius, N_batch=N_batch, plot=False
            )
            r_p0s_init = torch.tensor(r_p0s_init)
            C_p0s_init = torch.tensor(C_p0s_init)
        # opt parameters
        opt_kwargs = {
            "abs_err_tolerance": 1e-10,
            "rel_err_tolerance": 1e-8,
            "max_iterations": 500,
            "step_size": 1,
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


def compare_tune_baseline(N_batch=20, N_runs=10):
    """Compare tuning of baseline parameters with SDPR and Theseus.
    Run N_runs times with N_batch poses"""
    offset = 0.003  # offset for init baseline
    n_iters = 50  # Number of outer iterations
    opt_select = "sgd"  # optimizer to use
    # termination criteria
    term_crit = {
        "max_iter": 100,
        "tol_grad_sq": 1e-6,
        "tol_loss": 1e-10,
    }
    info_p, info_s, info_tg, info_tl = [], [], [], []
    set_seed(0)
    for i in range(N_runs):
        # Generate data
        print("__________________________________________________________")
        print(f"Run {i} of {N_runs}: Gen Data")
        radius = 3
        r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
            radius=radius,
            board_dims=[0.6, 1.0],
            N_squares=[8, 8],
            N_batch=N_batch,
            setup="cone",
            plot=False,
        )
        info_p.append(
            dict(r_p0s=r_p0s, C_p0s=C_p0s, r_map=r_ls, pixel_meass=pixel_meass)
        )
        prob_data = (r_p0s, C_p0s, r_ls, pixel_meass)
        # Run tuners
        # print(f"Run {i} of {N_runs}: SDPR")
        # info_s.append(
        #     tune_baseline(
        #         "spdr",
        #         opt_select=opt_select,
        #         b_offs=offset,
        #         N_batch=N_batch,
        #         prob_data=prob_data,
        #         term_crit=term_crit,
        #     )
        # )
        # print(f"Run {i} of {N_runs}: Theseus (gt init)")
        # info_tg.append(
        #     tune_baseline(
        #         "theseus",
        #         opt_select=opt_select,
        #         b_offs=offset,
        #         gt_init=True,
        #         N_batch=N_batch,
        #         prob_data=prob_data,
        #         term_crit=term_crit,
        #     )
        # )
        print(f"Run {i} of {N_runs}: Theseus (rand init)")
        info_tl.append(
            tune_baseline(
                "theseus",
                opt_select=opt_select,
                b_offs=offset,
                gt_init=False,
                N_batch=N_batch,
                prob_data=prob_data,
                term_crit=term_crit,
            )
        )

    # Save data
    data = dict(info_p=info_p, info_s=info_s, info_tg=info_tg, info_tl=info_tl)
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(folder, "outputs")
    offset_str = str(offset).replace(".", "p")
    dump(
        data,
        open(
            folder
            + f"/compare_tune_b{offset_str}_{opt_select}_{N_batch}btchs_{N_runs}runs.pkl",
            "wb",
        ),
    )


def compare_tune_baseline_pp(filename="compare_tune_b0p003_batch.pkl", ind=0):
    # Load data
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(folder, "outputs")

    data = load(open(folder + "/" + filename, "rb"))
    info_s = data["info_s"][ind]
    info_tl = data["info_tl"][ind]
    info_tg = data["info_tg"][ind]
    # Plot loss
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(info_tl["loss"], label="Theseus (rand init)")
    axs[0, 0].plot(info_tg["loss"], label="Theseus (gt init)")
    axs[0, 0].plot(info_s["loss"], label="SDPR")
    axs[0, 0].set_yscale("log")
    axs[0, 0].set_title("Outer Loss")
    axs[0, 0].legend()

    # process inner losses
    info_tl["loss_inner_sum"] = info_tl["loss_inner"].apply(
        lambda x: torch.sum(x).detach().numpy()
    )
    info_tg["loss_inner_sum"] = info_tg["loss_inner"].apply(
        lambda x: torch.sum(x).detach().numpy()
    )

    axs[1, 0].plot(info_tl["grad_sq"], label="Theseus (rand init)")
    axs[1, 0].plot(info_tg["grad_sq"], label="Theseus (gt init)")
    axs[1, 0].plot(info_s["grad_sq"], label="SDPR")
    axs[1, 0].set_title("Gradient Squared")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_yscale("log")
    axs[1, 0].legend()

    # Plot parameter values
    axs[0, 1].plot(info_tl["params"], label="Theseus (rand init)")
    axs[0, 1].plot(info_tg["params"], label="Theseus (gt init)")
    axs[0, 1].plot(info_s["params"], label="SDPR")
    axs[0, 1].axhline(cam_gt.b, color="k", linestyle="--", label="Actual Value")
    axs[0, 1].set_title("Baseline Error to GT")
    axs[0, 1].legend()

    # Inner loop optimization time
    axs[1, 1].plot(info_tl["time_inner"], label="Theseus (rand init)")
    axs[1, 1].plot(info_tg["time_inner"], label="Theseus (gt init)")
    axs[1, 1].plot(info_s["time_inner"], label="SDPR")
    axs[1, 1].set_yscale("log")
    axs[1, 1].legend()
    axs[1, 1].set_title("Inner Optimization Time")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_converged_vals(filename="compare_tune_b0p003_batch.pkl", ind=0):
    # Load data
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = os.path.join(folder, "outputs")

    data = load(open(folder + "/" + filename, "rb"))
    # info_s = data["info_s"][ind]
    # info_tl = data["info_tl"][ind]
    info_tg = data["info_tg"][ind]
    info_p = data["info_p"][ind]

    # Plot final solutions
    r_p0s, C_p0s = info_tg.iloc[-1]["solution"]
    r_p0s = r_p0s.detach().numpy()
    C_p0s = C_p0s.detach().numpy()
    r_p0s_gt = info_p["r_p0s"].detach().numpy()
    C_p0s_gt = info_p["C_pos"].detach().numpy()

    # Plot final solutions
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_poses(C_p0s_gt, r_p0s_gt, ax=ax, color="k")
    plot_poses(C_p0s, r_p0s, ax=ax)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    r = 3 * 1.1
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)

    plt.show()


if __name__ == "__main__":
    # Test generation of calibration data
    # r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
    #     setup="cone", cone_angles=(0.0, np.pi / 4), N_batch=100, plot=True
    # )
    # r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
    #     setup="cone", cone_angles=(np.pi / 4, 0.0), N_batch=100, plot=True
    # )
    # r_p0s, C_p0s, r_ls, pixel_meass = get_cal_data(
    #     setup="cone", cone_angles=(np.pi / 4, np.pi / 4), N_batch=100, plot=True
    # )

    # Local minimum search
    # find_local_minima(store_data=False)

    # Comparison over multiple instances (batch):
    compare_tune_baseline(N_batch=20, N_runs=1)
    # compare_tune_baseline_pp(
    #     filename="compare_tune_b0p003_sgd_20btchs_2runs.pkl", ind=0
    # )
    # plot_converged_vals(filename="compare_tune_b0p003_sgd_20btchs_1runs.pkl")
