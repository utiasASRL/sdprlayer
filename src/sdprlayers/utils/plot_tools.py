#!/usr/bin/env python
import os

import matplotlib.pyplot as plt
import numpy as np

root_dir = os.getcwd()
fig_dir = os.path.join(root_dir, "figs")


def make_dirs_safe(path):
    """Make directory of input path, if it does not exist yet."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def plot_ellipsoid(bias, cov, ax=None, color="b", stds=3, label: str = None):
    """Plot a 3D ellipsoid

    Args:
        bias (_type_): offset to center of ellipsoid
        cov (_type_): covariance matrix associated with ellipsoid
        ax (_type_, optional): axis handle. Defaults to None.
        color (str, optional): color of ellipsoid. Defaults to 'b'.
        stds (int, optional): Number of standard deviations for ellipsoid. Defaults to 1.
        label (str): Label string for plotting. Defaults to None.

    Returns:
        _type_: _description_
    """
    if ax is None:
        ax = plt.axes(projection="3d")
    # Parameterize in terms of angles
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    L = np.linalg.cholesky(cov)
    ellipsoid = (stds * L @ np.stack((x, y, z), 0).reshape(3, -1) + bias).reshape(
        3, *x.shape
    )
    surf = ax.plot_surface(
        *ellipsoid, rstride=4, cstride=4, color=color, alpha=0.25, label=label
    )
    # These lines required for legend
    # surf._edgecolors = surf._edgecolors3d
    # surf._facecolors = surf._facecolors3d
    return surf


def make_axes_transparent(ax):
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.set_axis_off()


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


def plot_map(r_l, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.plot(*r_l, "*", color="k", markersize=2, **kwargs)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return ax


def savefig(fig, name, dpi=200, verbose=True):
    name = os.path.join(fig_dir, name)
    make_dirs_safe(name)
    ext = name.split(".")[-1]
    fig.savefig(name, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=dpi)
    if verbose:
        print(f"saved plot as {name}")
