import numpy as np
import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer


class SDPRLayer(CvxpyLayer):
    def __init__(
        self,
        n_vars,
        Constraints,
        local_solver=None,
        local_args={},
        certifier=None,
        use_dual=False,
    ):
        # Store information
        self.local_solver = local_solver
        self.local_args = local_args
        self.certifier = certifier
        self.Constraints = Constraints
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
        # Check if local solution methods have been defined.
        if self.local_solver is not None and self.certifier is not None:
            # Run Local Solver
            local_soln = self.local_solver(**self.local_args)
            # Certify Local Solution
            Q_detach = Q.cpu().detach().double().numpy()
            local_cert = self.certifier(
                Q=Q_detach, Constraints=self.Constraints, x_cand=local_soln
            )
            # Update the solver arguments
            new_solver_args = dict(
                solve_method="local_cert", local_soln=local_soln, local_cert=local_cert
            )
            if "solver_args" in kwargs:
                kwargs["solver_args"].update(new_solver_args)
            else:
                kwargs["solver_args"] = new_solver_args

        # Call cvxpylayers forward function
        res = super().forward(Q, **kwargs)

        return res
