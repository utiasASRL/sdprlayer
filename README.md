
# SDPRLayer

# Introduction

The SDPRLayer is an optimization layer for semidefinite program relaxations (SDPR) of non-convex polynomial problems designed for certifiable optimization. It is effectively a wrapper for [CVXPYLayers](https://github.com/cvxgrp/cvxpylayers) that is built specifically for semidefinite programs (SDPs). 

When the SDP relaxation is *tight* to the original problem, the output of the layer and the gradients produced during backpropagation are *guaranteed* to be correct.

Please see our [paper](https://arxiv.org/abs/2405.19309) for more details.

# Usage

The user is responsible for converting a given polynomial problem to a *Quadratically Constrained Quadratic Problem* (QCQP). The SDPRLayer can accept the QCQP in non-homogenized form:

$$
\begin{equation}
	\begin{array}{rl}
		\min\limits_{\bm{x}} &\bm{x}^T\bm{F}_{\bm{\theta}}\bm{x} + \bm{f}_{\bm{\theta}}^T\bm{x} + f_{\bm{\theta}}\\ 
		s.t.&\bm{x}^T\bm{G}_{\bm{\theta} i}\bm{x} + \bm{g}_{\bm{\theta} i}^T\bm{x} + g_{\bm{\theta} i} = 0,\forall i \in [N_c],
	\end{array}
\end{equation}
$$

Alternatively, the user can specify the problem in homogenized form:
$$
\begin{equation}
	\begin{array}{rl}
		\min\limits_{\bm{x}} & \bm{x}^T\bm{Q}_{\bm{\theta}}\bm{x} \\
		s.t. & \bm{x}^T\bm{A}_{\bm{\theta} i}\bm{x} = 0,\forall i \in [N_c],\\
		&\bm{x}^T\bm{A}_0\bm{x} = 1,
	\end{array}
\end{equation}
$$

If the problem will be provided in non-homogenized form, the user must set the "homogenize" flag to true during initialization of the layer.

The input vectors/matrices that characterize the cost and constraints are provided to the SDPRLayer on initialization of the layer. The cost must be provided either as a 3-tuple (non-homogenized) or a matrix (homogenized). The constraints are provided as a list of either 3-tuples (non-homogenized) or matrices (homogenized).

Importantly, setting the cost input or members of the constraint list to `None` flags them as inputs that will require gradients and will change throughout tuning/training. These flagged inputs are then provided to the `forward` function of the SDPRLayer as a list of PyTorch tensors in the same order that they appeared in the initialization (cost first, then constraints in list order).

The SDPRLayer returns both the vector solution to the original problem and the matrix solution to the SDP relaxation. Both outputs are PyTorch tensors.

Please see `_test` directory for example usages. 

# Installation

The SDPRLayer relies on our PolyMatrix package as well as few modifications to the CVXPYLayers and diffcp libraries. Prior to setting up the Conda environment, clone the following repositories to the parent folder of the SDPRLayer repo:

```bash
git clone git@github.com:utiasASRL/poly_matrix.git
git clone git@github.com:holmesco/diffcp.git
git clone git@github.com:holmesco/cvxpylayers.git
```

Once these repositories have been downloaded, the environment can be created.

```bash
conda env create -f environment.yml
```

The following test functions can be used to verify the installation:

```bash
conda activate sdprlayer
pytest _test/test_sdprlayer.py
pytest _test/test_sdpr_poly6.py
pytest _test/test_stereo_tune.py
```

## Multithread Library Issue

We have noticed that there are sometimes issues with the Conda environment due to collisions between the Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp'). This can cause crashing or deadlock. If observed, defining the following environment variable can fix the issue:
```bash
export MKL_THREADING_LAYER=GNU
```

The issue is well known and other workarounds are given [here](https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md).

# Citation

If you use any of the code or ideas from this repo, please cite our work:
```txt
@article{holmesSDPRLayers2024,
  title = {{SDPRLayers}: {Certifiable Backpropagation Through Polynomial Optimization Problems} in {Robotics}},
  author = {Holmes, Connor and D{\"u}mbgen, Frederike and Barfoot, Timothy D.},
  year = {2024},
  eprint = {2405.19309},
  publisher = {arXiv},
  journal = {arXiv:2405.19309}
  url={https://arxiv.org/abs/2405.19309}
}
```