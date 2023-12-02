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
        certifier=None,
        use_primal=True,
    ):
        # Store information
        self.local_solver = local_solver
        self.certifier = certifier
        self.Constraints = Constraints
        # SET UP CVXPY PROGRAM
        Q = self.init_cost_mat(n_vars)
        m = len(Constraints)
        # Primal vs Dual
        if use_primal:
            # NOTE: CVXPY adds new constraints when canonicalizing if
            # the problem is defined using the primal form.
            X = cp.Variable((n_vars, n_vars), symmetric=True)
            constraints = [X >> 0]
            for i, (A, b) in enumerate(Constraints):
                constraints += [cp.trace(A @ X) == b]
            objective = cp.Minimize(cp.trace(Q @ X))
            problem = cp.Problem(objective=objective, constraints=constraints)
            variables = [X]
        else:
            y = cp.Variable(shape=(m,))
            As, bs = zip()
            b = np.concatenate([np.atleast_1d(b) for b in bs])
            objective = cp.Maximize(b @ y)
            LHS = cp.sum([y[i] * Ai for (i, Ai) in enumerate(As)])
            constraint = LHS << Q
            problem = cp.Problem(objective, [constraint])
            variables = [y]
        assert problem.is_dpp()

        # Call CvxpyLayers init
        super(SDPRLayer, self).__init__(
            problem=problem,
            variables=variables,
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
            if "local_args" not in kwargs:
                raise ValueError("Local solver requires dictionary of local_args")
            # Run Local Solver
            local_soln = self.local_solver(**kwargs["local_args"])
            # Certify Local Solution
            Q_detach = Q.cpu().detach().double().numpy()
            local_cert = self.certifier(
                Q=Q_detach, Constraints=self.Constraints, x_cand=local_soln
            )
            # add solver args
            solver_args = dict(
                solve_method="local_cert", local_soln=local_soln, local_cert=local_cert
            )
            # Call cvxpylayers forward function
            res = super().forward(Q, solver_args=solver_args, **kwargs)
        else:
            # Call cvxpylayers forward function without args.
            res = super().forward(Q, **kwargs)

        return res
