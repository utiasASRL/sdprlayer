import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from diffcp import cones
from copy import deepcopy

mosek_params_dflt = {
    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-14,
    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
}


class SDPRLayer(CvxpyLayer):
    def __init__(
        self,
        n_vars,
        constraints,
        local_solver=None,
        local_args={},
        certifier=None,
        use_dual=True,
    ):
        # Store information
        self.local_solver = local_solver
        self.local_args = local_args
        self.certifier = certifier
        self.constraints = constraints
        self.use_dual = use_dual
        # SET UP CVXPY PROGRAM
        Q = cp.Parameter((n_vars, n_vars), symmetric=True)
        m = len(self.constraints)
        # Dual vs Primal Formulation
        if use_dual or local_solver is not None:
            # If using local solver then must use dual formulation to avoid
            # definition of extra slacks by CVXPY
            y = cp.Variable(shape=(m,))
            As, bs = zip(*self.constraints)
            b = np.concatenate([np.atleast_1d(b) for b in bs])
            objective = cp.Maximize(b @ y)
            LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])
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
            for i, (A, b) in enumerate(self.constraints):
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

    def forward(self, Q: torch.tensor, **kwargs):
        # Define new kwargs to not affect original
        kwargs_new = deepcopy(kwargs)
        if "solver_args" in kwargs and "solve_method" in kwargs["solver_args"]:
            method = kwargs["solver_args"]["solve_method"]
            # Check if we are injecting a solution
            if method == "local" or method == "mosek":
                # Check if Q is not batched
                Qs = Q.cpu().detach().double().numpy()
                if Qs.ndim == 2:
                    Qs = Qs.unsqueeze(0)

                ext_vars_list = []
                for i in range(Qs.shape[0]):
                    if method == "mosek":
                        assert "MOSEK" in cp.installed_solvers(), "MOSEK not installed"

                        # Get data matrix and make sure its symmetric
                        Q_val = Qs[i]
                        assert np.allclose(Q_val, Q_val.T), "Q must be symmetric"
                        self.Q.value = Q_val
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
                        assert (
                            self.use_dual
                        ), "Primal not implemented. Set use_dual=True"

                        # TODO set up batched version of local solver

                        # Run Local Solver
                        x_cand = self.local_solver(**self.local_args)
                        X = x_cand @ x_cand.T
                        # Get detach version of Q
                        Q_detach = Q.cpu().detach().double().numpy()
                        # Certify Local Solution
                        H, mults = self.certifier(
                            Q=Q_detach, constraints=self.constraints, x_cand=x_cand
                        )
                        # TODO add in a check here to make sure H is PSD

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
        res = super().forward(Q, **kwargs_new)
        # TODO If using mosek or local then overwrite the results for
        # better accuracy.
        return res
