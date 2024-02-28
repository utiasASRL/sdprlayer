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
        add_homog_constr=True,
        local_solver=None,
        local_args={},
        certifier=None,
    ):
        """Initialize the SDPRLayer class. This functions sets up the SDP relaxation
        using CVXPY, adding in parameters to be filled in later during forward function
        call. If homogenize input is True, it is assumed that the problem is in the standard
        non-convex, non-homogenized form.

        min x^T F x + _f^T x + f
        s.t. x^T G_i x + _g_i^T x + g_i = 0

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
        self.constraints = constraints
        self.use_dual = use_dual
        self.local_solver = local_solver
        self.local_args = local_args
        self.certifier = certifier
        # Add homogenization variable
        if homogenize:
            n_vars = n_vars + 1
        # parameter list (for cvxpylayers)
        params = []
        # objective matrix
        if objective is None:
            Q = cp.Parameter((n_vars, n_vars), symmetric=True)
            params += [Q]
        elif self.homogenize:
            assert (
                objective is tuple
            ), "objective input must be tuple if homogenize flag is active"
            Q = self.homog_matrix(*objective)
        else:
            Q = objective

        # check constraints set to None are asssumed to be parameterized
        for iConstr in range(len(constraints)):
            if self.constraints[iConstr] is None:
                self.constraints[iConstr] = cp.Parameter(
                    (n_vars, n_vars), symmetric=True
                )
                # Add constraint to the list of parameters
                params += list(self.constraints[iConstr])
            else:
                if self.homogenize:
                    assert (
                        self.constraints[iConstr] is tuple
                    ), "constraint must be list of tuples if homogenize flag is active"
                    self.constraints[iConstr] = self.homog_matrix(
                        *self.constraints[iConstr]
                    )
                else:  # otherwise constraint is already set up properly
                    continue
        N_constrs = len(self.constraints)
        # homenization constraint matrix
        A_0 = sp.csc_array((n_vars, n_vars))
        A_0[0, 0] = 1.0
        # Set Standard Formulation (Homogenized SDP)
        # If using local solver then must use dual formulation to avoid
        # definition of extra slacks by CVXPY
        if use_dual or local_solver is not None:
            y = cp.Variable(shape=(N_constrs,))
            rho = cp.Variable(shape=(1,))  # homog constraint
            objective = cp.Maximize(rho)
            LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(self.constraints)])
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
            for A in self.constraints:
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

    def forward(self, objective=None, constraints=None, **kwargs):
        """Run (differentiable) optimization using the provided input tensors"""

        # get objective function values
        if objective is not None:
            if self.homogenize:
                assert (
                    objective is tuple
                ), "objective input must be tuple if homogenize flag is active."
                # Convert to batch
                if objective[0].ndim == 2:
                    Qs = self.homog_matrix(*objective).unsqueeze(0)
                else:  # batch
                    Qs = torch.vmap(self.homog_matrix)(*objective)
            else:
                if objective[0].ndim == 2:
                    Qs = objective.unsqueeze(0)
                else:
                    Qs = objective
        else:
            Qs = []
        # Get constraint values
        if constraints is not None:
            As = []
            for constraint in constraints:
                if self.homogenize:
                    assert (
                        constraint is tuple
                    ), "constraint must be list of tuples if homogenize flag is active"
                    if constraint[0].ndims == 2:
                        As += [self.homog_matrix(*constraint).unsqueeze(0)]
                    else:  # batch
                        As += [torch.vmap(self.homog_matrix)(*constraint)]
        else:
            As = []
        N_batch = Qs.shape[0]
        # Store parameter values
        param_vals = [Qs] + As
        # Define new kwargs to not affect original
        kwargs_new = deepcopy(kwargs)
        # If using external solver then we need to solve and then inject solution
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
                        self.params[i].value = param_vals[i][iBatch].detach().numpy()
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
                        raise NotImplementedError("Local solver not implemented")
                        # # Check if local solution methods have been defined.
                        # assert (
                        #     self.local_solver is not None
                        # ), "Local solver not defined."
                        # assert self.certifier is not None, "Certifier not defined."
                        # # Run Local Solver
                        # x_cand = self.local_solver(**self.local_args)
                        # X = x_cand @ x_cand.T
                        # # Certify Local Solution
                        # H, mults = self.certifier(
                        #     Q=Q_val, constraints=self.constraints, x_cand=x_cand
                        # )
                        # # TODO Improve this check using some other library
                        # min_eig = np.min(np.linalg.eigvalsh(H))
                        # if min_eig < -1e-8:
                        #     # TODO this should queue a reinitialization
                        #     raise ValueError("Local solution not certified")

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
        res = super().forward(*param_vals, **kwargs_new)

        # TODO If using mosek or local then overwrite the primal solution for better numerical accuracy?

        # Extract non-homogenized solution
        Xs = res[0]
        xs = self.recovery_map(Xs)
        return Xs, xs

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
            Q_left = torch.hstack([f, 0.5 * fvec]).unsqueeze(1)
            Q_right = torch.vstack([0.5 * fvec, F])
            Q = torch.hstack([Q_left, Q_right])
        else:
            Q_left = np.hstack([f, 0.5 * fvec])
            Q_right = np.vstack([0.5 * fvec, F])
            Q = np.hstack([Q_left, Q_right])
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
