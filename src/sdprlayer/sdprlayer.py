import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from diffcp import cones

mosek_params_dflt = {
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-8,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-8,
    "MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
}


class SDPRLayer(CvxpyLayer):
    def __init__(
        self,
        n_vars,
        Constraints,
        local_solver=None,
        local_args={},
        certifier=None,
        use_dual=True,
    ):
        # Store information
        self.local_solver = local_solver
        self.local_args = local_args
        self.certifier = certifier
        self.Constraints = Constraints
        self.use_dual = use_dual
        # SET UP CVXPY PROGRAM
        Q = self.init_cost_mat(n_vars)
        m = len(Constraints)
        # Dual vs Primal Formulation
        if use_dual or local_solver is not None:
            # If using local solver then must use dual formulation to avoid
            # definition of extra slacks by CVXPY
            y = cp.Variable(shape=(m,))
            As, bs = zip(*Constraints)
            b = np.concatenate([np.atleast_1d(b) for b in bs])
            objective = cp.Maximize(b @ y)
            LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])
            constraints = LHS << Q
            problem = cp.Problem(objective, [constraints])
            variables = []
            constraints_ = [constraints]
            self.H = Q - LHS
        else:
            # NOTE: CVXPY adds new constraints when canonicalizing if
            # the problem is defined using the primal form.
            X = cp.Variable((n_vars, n_vars), symmetric=True)
            constraints = [X >> 0]
            for i, (A, b) in enumerate(Constraints):
                constraints += [cp.trace(A @ X) == b]
            objective = cp.Minimize(cp.trace(Q @ X))
            problem = cp.Problem(objective=objective, constraints=constraints)

            variables = [X]
            constraints_ = []
        assert problem.is_dpp()
        # store problem and parameters
        self.problem = problem
        self.Q = Q
        # Call CvxpyLayers init
        super(SDPRLayer, self).__init__(
            problem=problem,
            variables=variables,
            constraints=constraints_,
            parameters=[Q],
        )

    def init_cost_mat(self, n_vars):
        """This function should be replaced if specific sparsity patterns
        is known.
        """
        Q = cp.Parameter((n_vars, n_vars), symmetric=True)
        return Q

    def forward(self, Q: torch.tensor, **kwargs):
        if "solver_args" in kwargs and "solve_method" in kwargs["solver_args"]:
            method = kwargs["solver_args"]["solve_method"]
            if method == "mosek":
                mosek_params = kwargs["solver_args"]
                mosek_params.pop("solve_method")
                verbose = mosek_params.pop("verbose", None)
                self.Q.value = Q.cpu().detach().double().numpy()
                self.problem.solve(
                    solver=cp.MOSEK, verbose=verbose, mosek_params=mosek_params_dflt
                )
                # Set variables to inject
                if self.use_dual:
                    X = self.problem.constraints[0].dual_value
                    H = np.array(self.H.value)
                    mults = self.problem.variables()[0].value
                else:
                    raise NotImplementedError

            elif method == "local":
                # Check if local solution methods have been defined.
                assert self.local_solver is not None, "Local solver not defined."
                assert self.certifier is not None, "Certifier not defined."
                # Run Local Solver
                x_cand = self.local_solver(**self.local_args)
                X = x_cand @ x_cand.T
                # Certify Local Solution
                Q_detach = Q.cpu().detach().double().numpy()
                H, mults = self.certifier(
                    Q=Q_detach, Constraints=self.Constraints, x_cand=x_cand
                )

            if method == "local" or method == "mosek":
                # Set variables to inject
                ext_vars = dict(
                    x=mults,
                    y=cones.vec_symm(X),
                    s=cones.vec_symm(H),
                )
                # Update solver arguments
                solver_args = dict(solve_method="external", ext_vars=ext_vars)
                if "solver_args" in kwargs:
                    kwargs["solver_args"].update(solver_args)
                else:
                    kwargs["solver_args"] = solver_args

        # Call cvxpylayers forward function
        res = super().forward(Q, **kwargs)

        return res
