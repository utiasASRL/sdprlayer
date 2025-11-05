
# SDPRLayers


The SDPRLayer is an PyTorch-based optimization layer for semidefinite program relaxations (SDPR) of non-convex polynomial problems designed for certifiable optimization. The forward pass of the layer solves the SDP relaxation, while the backward pass uses implicit differentiation to compute gradients of the solution with respect to input parameters. When the SDP relaxation is *tight* to the original problem, the output of the layer and the gradients produced during backpropagation are *guaranteed* to be correct. Backpropagation can be computed using either the SDP relaxation itself or the QCQP formulation of the polynomial problem.

Please see our [paper](https://arxiv.org/abs/2405.19309) for more details.

# Usage

The user is responsible for converting a given polynomial problem to a *Quadratically Constrained Quadratic Problem* (QCQP) of the following form:

$$
\begin{equation}
	\begin{array}{rl}
		\min\limits_{\mathbf{x}} & \mathbf{x}^{\top}\mathbf{Q}_{\mathbf{\theta}}\mathbf{x} \\
		s.t. & \mathbf{x}^{\top}\mathbf{A}_{\mathbf{\theta} i}\mathbf{x} = 0,\quad\forall i \in [m],\\
		&\mathbf{x}^{\top}\mathbf{A}_0\mathbf{x} = 1,
	\end{array}
\end{equation}
$$

with *Semidefinite Problem (SDP)* Relaxation given by:

$$
\begin{equation}
	\begin{array}{rl}
		\min\limits_{\mathbf{X}} & \left<\mathbf{Q}_{\mathbf{\theta}},\mathbf{X}\right>\\
		s.t.&\left<\mathbf{A}_{\mathbf{\theta} i},\mathbf{X}\right>= 0, \quad \forall i \in [m],\\
		& \left<\mathbf{A}_{0},\mathbf{X}\right> = 1,\\
		& \mathbf{X} \succeq \mathbf{0}.\\
	\end{array}
\end{equation}
$$  

The SDPRLayer can accept the QCQP in homogenized or non-homogenized form. If the problem will be provided in non-homogenized form, the user must set the "homogenize" flag to true during initialization of the layer.

The matrices corresponding to the cost and constraints are provided to the `SDPRLayer` class on initialization, with the cost provided as a single (`numpy') array/matrix and the constraints provided as a list of matrices.

Importantly, setting the cost input or members of the constraint list to `None` signals to the layer that they are *parameterized* and will be provided as a list of PyTorch tensors when the `forward` function of the layer is called. This list of tensors should be provided in the same order that they appeared in the initialization (cost first, then constraints in list order) and gradients of the solution will be computed with respect to these inputs. The first dimension of the input matrices is expected to correspond to the batch dimension

The SDPRLayer always returns the matrix solution to the SDP relaxation of the QCQP. When the relaxation is tight -- i.e. its rank is equal to one (across entire batch) -- the layer also returns the *globally-optimal* vector solution to the original problem. Otherwise, the vector solution is set to `None`. Both outputs are PyTorch tensors.

## Layer Setup

The following input options are availabe when setting up the layer:

Input | Default | Description
----- | ------- | -----------
n_vars| | Size of the QCQP variable
constraints| [] | List of input constraints. List values that are set to `None` are interpreted as parameterized constraints.
objective| None | Input objective matrix
homogenize| False | Specifies whether the problem needs to be homogenized (typically not used)
use_dual | True | Specifies whether the internal disciplined convex program (DCP) is specified in the primal form of the QCQP above or its dual form. 
diff_qcqp | True | Specifies whether to differentiate via the QCQP KKT conditions or the SDP KKT conditions. This option can also be used to differentiate a local solution to the QCQP, without certifying global optimality.
compute_multipliers | False | Only used if `diff_qcqp` is set to True. Specifies whether to recompute the Lagrange multipliers and certificate for original QCQP system. If used with This option can be used to compute multipliers for an existing local solution (if certification not required).
licq_tol| 1e-7 | Tolerance used when computing the Lagrange multipliers via least-squares 
lsqr_tol| 1e-10 | Tolerance used when computing the solution to the (adjoint) differential KKT equation via LSQR (when system is not symmetric). 
minres_tol | 1e-20 | Tolerance used when computing the solution to the (adjoint) differential KKT equation via MINRES (system is symmetric and size is >300)
kkt_tol | 1e-5 |  Tolerance used when checking that the KKT conditions are satisfied. Violation of this check usually indicates that the certificate matrix is incorrect or too many redundant constraints have been removed
redun_list| [] | List of indicies corresponding to constraints that should be labelled as 'redundant' when computing Lagrange multipliers or solving the differential KKT system. **Note: Failure to properly identify the redundant constraints can lead to incorrect gradients!**


In the [paper](https://arxiv.org/abs/2405.19309), we discuss three different modes of usage for SDPRLayers: *SDPR-IS*, *SDPR-CIFT*, and *SDPR-SDP*. These modes correspond to the following options:

Mode | Settings | Default
--- | --- | ---
SDPR-IS | ```diff_qcqp=True``` ```compute_multipliers=False``` |X
SDPR-CIFT | ```diff_qcqp=True``` ```compute_multipliers=True```|
SDPR-SDP | ```diff_qcqp=False```|

## The Forward Function

The layer is called the same wasy any other PyTorch layer is called: the objective and/or constraints are provided as arguments to the forward function and the function returns the optimization solution, which is linked to the inputs in the compute graph. 

An additional keyword argument, `ext_vars_list`, provides the user with a means to solve the optimization externally and inject the solution into SDPRLayers with an external variable list. When this variable is set, the optimization routine is bypassed, but the output solution linked to the input parameters via the compute graph for automatic differentiation.

The external variable list should be structured as a list with length corresponding to the batch dimension of the input parameters. Each element of the list should be a dictionary with the primal-dual solution specified as follows:

Key | Value
---|---
`x` | List/array of Lagrange multipliers corresponding to the problem constraints.
`y` | Primal solution in half-vectorized form.
`s` | Dual solution (certificate matrix) in half-vectorized form.

The half-vectorized form can be computed using the `diffcp.cones` module.
Note that the naming convention follows the standard form conic optimization problem given in [this paper](https://web.stanford.edu/~boyd/papers/pdf/diff_cone_prog.pdf). In our framework the dual/certificate matrix is formulated as:
$$
\mathbf{H} = \mathbf{Q} + \sum\limits_{i=0}^{m}\lambda_i \mathbf{A}_i,
$$
where $\lambda_i$ are the Lagrange multipliers. The primal solution corresponds to the PSD matrix that solves the relaxation of the QCQP. To see how this can be used with external solvers, see the `SDPRLayersMosek` class defined in `sdprlayer.py`. If `compute_multipliers` is set to true, then only the primal solution need be specified (the other variables will be recomputed).

All additional keyword arguments are passed directly to CvxpyLayers, internally.

## Differentiating Local Solutions

The external variable list also provides a means to **differentiate a local solution**, $\hat{\mathbf{x}}$, to the QCQP above. To do this, one can simply define the primal solution as the outer product of the local solution with itself ($\hat{\mathbf{x}}\hat{\mathbf{x}}^{\top}$). By setting the `compute_multipliers` flag to `True`, the dual solution and Lagrange multipliers can be left as empty lists, since they will be recomputed. 

Note that in this case, the solution and its gradients are not certified to be optimal.

## Examples

Two differentiable layers have been built and tested that make use of SDPRLayers:
- **Pose Estimation**: Estimate relative pose between two clouds of 3D keypoints. The SDP formulation allows to find global solutions when matrix-weights of the errors are used, which is important for aligning point clouds built from stereo images. Problem formulation 
- **Essential Matrix Estimation**: Global estimation of the essential matrix between two clouds of 2D point correspondences. This formulation finds the globally optimal essential matrix that minimizes the "algebraic error" associated with a set of point correspondences. We follow the setup shown in [this paper](https://arxiv.org/pdf/1903.09067v3).

The test scripts in [the test section](#testing-functionality) are a good source of example usages of the pre-built layers, while the layers themselves demonstrate how to make use of SDPRLayers 

The `_scripts` directory also has the experiments used in the paper. An example of how to use the *tightening* tools is given [here](_scripts/tighten_example.ipynb). To see an example usage of the SDPRLayer in a full robotics pipeline, see [our robot localization example](https://github.com/utiasASRL/deep_learned_visual_features/tree/mat-weight-sdp-version) (note that this is not on the main branch of the repo).

# Installation

We have set up a conda environment to 

The SDPRLayer relies on our `PolyMatrix` and `certifiable tools` repos as well as modified versions of `CVXPYLayers` and `diffcp` libraries. These custom repositories have been set up as submodules in the `sdprlayers` repo. Prior to setting up the conda environment, initialize the submodules with

```
git submodule update --init --recursive
```

Once the submodules are set up, the environment can be created.

```
conda env create -f environment.yml
```
Note that the environment is set up for our local version of CUDA and may need to be modified depending on your CUDA driver. You may need to reinstall a specific version of torch:
```
pip uninstall torch torchaudio torchvision
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
```

## Testing Functionality

The following test functions can be used to verify the installation:

```
pytest _test/test_sdpr_poly6.py
pytest _test/test_sdpr_poly4.py
pytest _test/test_pose_est.py
pytest _test/test_essmat.py
```

Make sure to source the `.env` file in the root directory prior to running tests:

```
source .env
```

## Multithread Library Issue

We have noticed that there are sometimes issues with the Conda environment due to collisions between the Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp'). This can cause crashing or deadlock. If observed, defining the following environment variable can fix the issue:
```
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
