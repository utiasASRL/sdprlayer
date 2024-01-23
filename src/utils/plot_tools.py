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


def savefig(fig, name, verbose=True):
    name = os.path.join(fig_dir, name)
    make_dirs_safe(name)
    ext = name.split(".")[-1]
    fig.savefig(name, bbox_inches="tight", pad_inches=0.1, transparent=True, dpi=200)
    if verbose:
        print(f"saved plot as {name}")
