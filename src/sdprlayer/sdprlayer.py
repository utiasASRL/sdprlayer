import numpy as np
from copy import deepcopy
import scipy.sparse as sp
import torch

import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
from diffcp import cones


mosek_params_dflt = {
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
}


class SDPRLayer(CvxpyLayer):
    """
    This class represents a differentiable, semidefinite relaxation layer for
    non-convex QCQPs. The forward function call returns the (differentiable)
    solution to the QCQP.

    """

    def __init__(
        self,
        n_vars,
        constraints,
        objective=None,
        homogenize=False,
        use_dual=True,
        local_solver=None,
        local_args={},
        certifier=None,
    ):
        """Initialize the SDPRLayer class. This functions sets up the SDP relaxation
        using CVXPY, adding in parameters to be filled in later during forward function
        call. If homogenize input is True, it is assumed that the problem is in the standard
        non-convex, non-homogenized form.

        min x^T F x + fvec^T x + f
        s.t. x^T G_i x + gvec_i^T x + g_i = 0

        These matrices are then converted into homogenized form (below).
        If homogenize flag is set False then it is assumed that the problem is already in
        homogenized form:

        min x^T Q x
        s.t. x^T A_i x = 0,
             x^T A_0 x = 1,

        NOTE: The homogenization variable is assumed to be at the first position of x.
        NOTE: The homogenization constraint is assumed to come last.
        If any of the objective or constraints are meant to be parameterized, then they should
        be set to None. Otherwise, they should be set to the fixed values.

        Functionality for local_solvers and certifiers is currently not fully implemented.

        Args:
            n_vars (int): dimension of variable vector x
            constraints (list): list of constraints, either 3-tuple or matrices
            objective (tuple or array): objective function, either 3-tuple or matrix
            homogenize (boolean): defaults to False. If true, constraints and objective are converted to matrices from 3-tuples
            local_solver=None,
            local_args={},
            certifier=None,
            use_dual=True,
        """
        # Store information
        self.homogenize = homogenize
        self.constr_list = constraints
        self.use_dual = use_dual
        self.local_solver = local_solver
        self.local_args = local_args
        self.certifier = certifier
        # Add homogenization variable
        if homogenize:
            n_vars = n_vars + 1
        # parameter list (for cvxpylayers)
        # NOTE: parameters are the optimization matrices and are always
        # stored in homogenized form
        params = []
        # objective matrix
        if objective is None:
            Q = cp.Parameter((n_vars, n_vars), symmetric=True)
            params += [Q]
        elif self.homogenize:
            assert (
                type(objective) is tuple
            ), "objective input must be tuple if homogenize flag is active"
            Q = self.homog_matrix(*objective)
        else:
            Q = objective
        # Store objective matrix (for use in certifier)
        self.Q = Q

        # check constraints set to None are asssumed to be parameterized
        for iConstr in range(len(constraints)):
            if self.constr_list[iConstr] is None:  # Parameterized constraint
                self.constr_list[iConstr] = cp.Parameter(
                    (n_vars, n_vars), symmetric=True
                )
                # Add constraint to the list of parameters
                params += list(self.constr_list[iConstr])
            else:  # Fixed Constraint
                if self.homogenize:
                    assert (
                        type(self.constr_list[iConstr]) is tuple
                    ), "constraint must be list of tuples if homogenize flag is active"
                    self.constr_list[iConstr] = self.homog_matrix(
                        *self.constr_list[iConstr]
                    )
                else:  # otherwise constraint is already set up properly
                    continue
        N_constrs = len(self.constr_list)
        # homenization constraint matrix
        A_0 = sp.lil_array((n_vars, n_vars))
        A_0[0, 0] = 1.0
        # Set Standard Formulation (Homogenized SDP)
        # If using local solver then must use dual formulation to avoid
        # definition of extra slacks by CVXPY
        if use_dual or local_solver is not None:
            y = cp.Variable(shape=(N_constrs + 1,))
            rho = y[-1]
            objective = cp.Maximize(rho)
            LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(self.constr_list)])
            LHS += rho * A_0
            constraint = LHS << Q
            problem = cp.Problem(objective, [constraint])
            variables = []
            constraints_ = [constraint]
            self.H = Q - LHS
        else:
            # NOTE: CVXPY adds new constraints when canonicalizing if
            # the problem is defined using the primal form.
            X = cp.Variable((n_vars, n_vars), symmetric=True)
            constraints = [X >> 0]
            for A in self.constr_list:
                constraints += [cp.trace(A @ X) == 0.0]
            constraints += [cp.trace(A_0 @ X) == 1.0]
            objective = cp.Minimize(cp.trace(Q @ X))
            problem = cp.Problem(objective=objective, constraints=constraints)
            variables = [X]
            constraints_ = []
        assert problem.is_dpp()
        # store problem and parameters
        self.problem = problem
        self.params = params
        # Call CvxpyLayers init
        super(SDPRLayer, self).__init__(
            problem=problem,
            variables=variables,
            constraints=constraints_,
            parameters=params,
        )

    def forward(self, *params, **kwargs):
        """Solve problem (or a batch of problems) corresponding to params
        Args:
          params: a sequence of torch Tensors. If the "homogenize" flag was
                  set to true for the problem, then these tensors are assumed
                  to come in triplets that define the parameterized objective
                  and constraints. That is,
                    params = F, fvec, f, G_1, gvec_1, g_1, ... , G_m, gvec_m, g_m
                  If the homoginize flag is set to False for the problem,
                  then these tensors must be the homogenized objective and
                  constraint matrices. That is,
                    params = Q, A_1, ..., A_m
                  These Tensors can have either 2 or 3 dimenstions. If a
                  Tensor has 3 dimensions, then its first dimension is
                  interpreted as the batch size. These Tensors must all have
                  the same dtype and device.
          kwargs: key word arguments to be passed into the cvxpylayer. For
                  example, `solver_args'

        Returns:
          a list of optimal variable values, one for each CVXPY Variable
          supplied to the constructor.

        """
        # homogenize if required
        if self.homogenize:
            assert len(self.param_ids) * 3 == len(
                params
            ), "Expected 3 inputs per parameter to homogenize constraints"
            params_homog = []
            ind = 0
            while ind < len(params):
                # Unpack
                mat, vec, const = params[ind : ind + 3]
                # check dimensions
                ndims = mat.ndim
                assert (
                    ndims == vec.ndim and ndims == const.ndim
                ), "Inconsistent dimensions on inputs"
                # check batch dimension
                if ndims > 2:
                    if ind == 0:  # get batch dimension
                        N_batch = mat.shape[0]
                    assert (
                        mat.shape[0] == N_batch
                        and vec.shape[0] == N_batch
                        and const.shape[0] == N_batch
                    ), "Inconsistent batch dimesion"
                    # Homogenize
                    params_homog += [torch.vmap(self.homog_matrix)(mat, vec, const)]
                else:
                    params_homog += [self.homog_matrix(mat, vec, const)]
                # Increment index
                ind += 3
        else:  # problem already homogenized
            params_homog = params
            # Check dimensions and ensure consistency
            ndims = params_homog[0].ndim
            if ndims > 2:
                N_batch = params_homog[0].shape[0]
            else:
                N_batch = 1
            for param in params_homog:
                assert param.ndim == ndims, "Parameter dimensions inconsistent"
                if ndims > 2:
                    assert param.shape[0] == N_batch, "Inconsistent batch dimension"
                else:
                    param = param.unsqueeze(0)

        # Define new kwargs to not affect original
        kwargs_new = deepcopy(kwargs)

        # This section constructs a solution using an 'external' solver (MOSEK or
        # user-provided local solver). The solution is then injected into diffcp to
        # compute the gradients.
        if "solver_args" in kwargs and "solve_method" in kwargs["solver_args"]:
            method = kwargs["solver_args"]["solve_method"]
            # Check if we are injecting a solution
            if method == "local" or method == "mosek":
                assert self.use_dual, "Primal not implemented. Set use_dual=True"
                # TODO this loop should be set up so that we can run in parallel.
                ext_vars_list = []
                for iBatch in range(N_batch):
                    # Populate CVXPY Parameters with batch values
                    for i in range(len(self.params)):
                        self.params[i].value = params_homog[i][iBatch].detach().numpy()
                    # Solve the problem
                    if method == "mosek":
                        assert "MOSEK" in cp.installed_solvers(), "MOSEK not installed"
                        # Get parameters for mosek
                        verbose = kwargs["solver_args"].get("verbose", False)
                        mosek_params = kwargs["solver_args"].get(
                            "mosek_params", mosek_params_dflt
                        )
                        self.problem.solve(
                            solver=cp.MOSEK, verbose=verbose, mosek_params=mosek_params
                        )
                        # Solver check
                        if not self.problem.status == "optimal":
                            raise ValueError("MOSEK did not converge")
                        # Extract primal and dual variables
                        X = np.array(self.problem.constraints[0].dual_value)
                        H = np.array(self.H.value)
                        mults = self.problem.variables()[0].value
                    elif method == "local":
                        # Check if local solution methods have been defined.
                        assert (
                            self.local_solver is not None
                        ), "Local solver not defined."
                        assert self.certifier is not None, "Certifier not defined."
                        # Run Local Solver
                        x_cand = self.local_solver(**self.local_args)
                        X = x_cand @ x_cand.T
                        # Certify Local Solution
                        H, mults = self.certifier_wrapper(
                            objective=self.Q,
                            constraints=self.constr_list,
                            x_cand=x_cand,
                        )
                        # TODO Improve this check using some other library
                        min_eig = np.min(np.linalg.eigvalsh(H))
                        if min_eig < -1e-8:
                            # TODO this should queue a reinitialization
                            raise ValueError("Local solution not certified")

                    # Add to list of solutions
                    ext_vars_list += [
                        dict(
                            x=mults,
                            y=cones.vec_symm(X),
                            s=cones.vec_symm(H),
                        )
                    ]

                # Update solver arguments (copy required here)
                solver_args = dict(solve_method="external", ext_vars_list=ext_vars_list)
                if "solver_args" in kwargs_new:
                    kwargs_new["solver_args"].update(solver_args)
                else:
                    kwargs_new["solver_args"] = solver_args

        # Call cvxpylayers forward function
        res = super().forward(*params_homog, **kwargs_new)
        # If using mosek or local then overwrite the primal solution for better numerical accuracy?

        # Extract non-homogenized solution
        Xs = res[0]
        xs = self.recovery_map(Xs)
        # Adjust dimensions
        if ndims < 3:
            xs = xs.squeeze(0)
            Xs = Xs.squeeze(0)
        return Xs, xs

    def certifier_wrapper(self, objective, constraints, x_cand, **kwargs):
        """Wrapper for certifier function. This function extracts parameter
        values if necessary and then calls the (user-provided) certifier function."""
        # Get Cost
        if isinstance(objective, cp.Parameter):
            Q = objective.value
        elif isinstance(objective, np.ndarray) or sp.issparse(constraint):
            Q = objective
        else:
            raise ValueError("Objective must be a parameter or a numpy array")
        # Get constraints
        constraint_list = []
        for constraint in constraints:
            if isinstance(constraint, cp.Parameter):
                constraint_list += [(constraint.value, 0.0)]
            elif isinstance(constraint, np.ndarray) or sp.issparse(constraint):
                constraint_list += [(constraint, 0.0)]
            else:
                raise ValueError("Constraint must be a parameter or a numpy array")
        # Add homogenizing constraint
        n_vars = Q.shape[0]
        A_0 = sp.lil_array((n_vars, n_vars))
        A_0[0, 0] = 1.0
        constraint_list += [(A_0, 1.0)]

        # Call certifier
        return self.certifier(
            objective=Q,
            constraints=constraint_list,
            x_cand=x_cand,
            **kwargs,
        )

    @staticmethod
    def recovery_map(Xs, round=False):
        """Extract the rank-1 solution from the SDP matrix solution"""
        # Expand dimension if not batched.
        if Xs.ndim == 2:
            Xs = Xs.unsqueeze(0)
        # recovery vector QCQP variable
        if not round:
            extract = torch.vmap(SDPRLayer.extract_rowcol)
            xs = extract(Xs)
        else:
            # Not yet implemented, use SVD to get singular vector with max
            # singular value.
            raise NotImplementedError()
        return xs

    @staticmethod
    def homog_matrix(F, fvec, f):
        """Convert quadratic function to homogenized form (matrix)"""
        if torch.is_tensor(F) and torch.is_tensor(fvec) and torch.is_tensor(f):
            Q_left = torch.vstack([f, 0.5 * fvec])
            Q_right = torch.vstack([0.5 * fvec.T, F])
            Q = torch.hstack([Q_left, Q_right])
        else:
            Q = np.block([[f, 0.5 * fvec.T], [0.5 * fvec, F]])

        return Q

    @staticmethod
    def check_rank(X):
        # Check rank
        sorted_eigs = np.sort(np.linalg.eigvalsh(X.detach().numpy()))
        sorted_eigs = np.abs(sorted_eigs)
        assert sorted_eigs[-1] / sorted_eigs[-2] > 1e6, "X is not rank-1"

    @staticmethod
    def extract_rowcol(X):
        """Assumes that the homogenized variable corresponds to the first row/col"""
        x = (X[1:, [0]] + X[[0], 1:].T) / 2.0
        return x
