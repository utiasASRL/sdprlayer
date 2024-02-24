#!/usr/bin/env python
import os

import numpy as np
import matplotlib.pylab as plt

seaborn_kwargs = {"edgecolor": "none", "linewidth": 0}
root_dir = os.getcwd()
fig_dir = os.path.join(root_dir, "figs")


def make_dirs_safe(path):
    """Make directory of input path, if it does not exist yet."""
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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


def savefig(fig, name, dpi=200, verbose=True):
    name = os.path.join(fig_dir, name)
    make_dirs_safe(name)
    ext = name.split(".")[-1]
    fig.savefig(name, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=dpi)
    if verbose:
        print(f"saved plot as {name}")
