from contextlib import AbstractContextManager
import unittest

import numpy as np
import numpy.random as npr
import pytest
import scipy.linalg as la
import torch


from sdprlayer import SDPRLayer

torch.set_default_dtype(torch.double)


def set_seed(x):
    npr.seed(x)
    torch.manual_seed(x)


def unvec_symm(x, dim):
    X = torch.zeros((dim, dim))
    # triu_indices gets indices of upper triangular matrix in row-major order
    col_idx, row_idx = np.triu_indices(dim)
    X[(row_idx, col_idx)] = x
    X = X + X.T
    X /= np.sqrt(2)
    X[np.diag_indices(dim)] = torch.diagonal(X) * np.sqrt(2) / 2
    return X


# Alias the homogenization function
homog_mat = SDPRLayer.homog_matrix


class TestSDPRLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSDPRLayer, self).__init__(*args, **kwargs)
        set_seed(3)
        n = 4  # Dimension of SDP (non-homogenized)
        p = 3  # Number of constraints
        vdim = np.floor(n * (n + 1) / 2).astype(int)
        # Test setup is based on solving a random generic SDPs, but with homogenization.
        # Randomly generate constraints and objective of the form
        # min <F, X> s.t. <G_i, X> = b_i for i=1,...,p
        G_vec = torch.rand(vdim, p)
        F_vec = torch.rand(vdim)
        b = torch.rand(p, 1)
        # Store objective
        objective = unvec_symm(F_vec, n)
        # Store homogenized objective
        self.objective = (objective, torch.zeros(n, 1), torch.zeros(1, 1))
        self.objective_h = homog_mat(*self.objective)
        params_all = [objective, torch.zeros(1, 1)]
        # Store constraints
        self.constraints = []
        self.constraints_h = []
        for i in range(p):
            G = unvec_symm(G_vec[:, i], n)
            g_vec = torch.zeros(n, 1)
            g = -b[i].unsqueeze(0)
            # Store constraint tuples
            self.constraints.append((G, g_vec, g))
            # Store homogenized constraints
            self.constraints_h.append(homog_mat(G, g_vec, g))
            # Store non-zero parameters
            params_all.append(G)
            params_all.append(g)
        self.params_all = params_all
        # Store dims
        self.n = n
        self.p = p
        self.vdim = vdim

    def test_homog_mat(self):
        # Torch test
        input = self.objective
        Q_h_t = homog_mat(*input)
        # Numpy test
        Q = input[0].detach().numpy()
        q_vec = input[1].detach().numpy()
        q = input[2].detach().numpy()
        Q_h_n = homog_mat(Q, q_vec, q)

        np.testing.assert_allclose(Q_h_n, Q_h_t.detach().numpy())

    def test_if_0(self):
        # Interface Test 0.1: homog external, fixed constraints, fixed objective
        # No parameters
        parameters = []
        with pytest.raises(Exception):
            self.run_interface_test(
                constraints=self.constraints_h,
                objective=self.objective_h,
                parameters=parameters,
                homogenize=False,
            )
        # Interface Test 0.2: homog internal, fixed constraints, fixed objective
        # No parameters
        with pytest.raises(Exception):
            self.run_interface_test(
                constraints=self.constraints,
                objective=self.objective,
                parameters=parameters,
                homogenize=True,
            )

    def test_if_1(self):
        # Interface Test 1: homog external, fixed constraints, param objective
        parameters = self.params_all[0:2]
        self.run_interface_test(
            constraints=self.constraints_h,
            objective=None,
            parameters=parameters,
            homogenize=False,
        )

    def test_if_2(self):
        # Interface Test 2: homog internal, fixed constraints, param objective
        parameters = self.params_all[0:2]
        self.run_interface_test(
            constraints=self.constraints,
            objective=None,
            parameters=parameters,
            homogenize=True,
        )

    def test_if_3(self):
        # Interface Test 3: homog external, param constraints, fixed objective
        parameters = self.params_all[2:]
        self.run_interface_test(
            constraints=[None] * self.p,
            objective=self.objective_h,
            parameters=parameters,
            homogenize=False,
        )

    def test_if_4(self):
        # Interface Test 4: homog internal, param constraints, fixed objective
        parameters = self.params_all[2:]
        self.run_interface_test(
            constraints=[None] * self.p,
            objective=self.objective,
            parameters=parameters,
            homogenize=True,
        )

    def test_if_5(self):
        # Interface Test 5: homog external, param constraints, param objective
        parameters = self.params_all
        self.run_interface_test(
            constraints=[None] * self.p,
            objective=None,
            parameters=parameters,
            homogenize=False,
        )

    def test_if_6(self):
        # Interface Test 6: homog external, some param constraints, fixed objective
        parameters = self.params_all[2:4]
        constraints = self.constraints_h.copy()
        constraints[0] = None
        self.run_interface_test(
            constraints=constraints,
            objective=self.objective_h,
            parameters=parameters,
            homogenize=False,
        )

    def test_slv_1(self):
        # Solver Test 1: SCS, Use Primal Formulation
        # NOTE: Dual is default and is tested by other functions
        set_seed(2)
        self.run_interface_test(
            constraints=self.constraints_h,
            objective=None,
            parameters=self.params_all[0:2],
            homogenize=False,
            use_dual=False,
        )

    def test_slv_2(self):
        # Solver Test 2: MOSEK, Dual Formulation
        mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
        }
        msk_solver_args = {
            "solve_method": "mosek",
            "mosek_params": mosek_params,
            "verbose": False,
        }
        self.run_interface_test(
            constraints=self.constraints_h,
            objective=None,
            parameters=self.params_all[0:2],
            homogenize=False,
            solver_args=msk_solver_args,
        )

    def run_interface_test(
        self, constraints, objective, parameters, homogenize, **kwargs
    ):
        """Generalized test for the interface of the SDPRLayer class."""
        # If already homogenized incease the dimension
        if homogenize:
            n = self.n
        else:
            n = self.n + 1
        # Make layer
        layer = SDPRLayer(
            n,
            objective=objective,
            constraints=constraints,
            homogenize=homogenize,
        )

        def run_layer(*params):
            # Reorganize parameters
            params_h = []
            ind = 0
            while ind < len(params):
                param_triplet = [params[ind], torch.zeros(self.n, 1), params[ind + 1]]
                if not homogenize:  # homogenize externally
                    params_h.append(homog_mat(*param_triplet))
                else:  # pass params direct, homogenize internally
                    params_h += param_triplet
                ind += 2
            # Call layer
            if "solver_args" in kwargs:
                X, x = layer(*params_h, **kwargs)
            else:
                X, x = layer(*params_h, solver_args={"eps": 1e-10})
            return X

        if len(parameters) > 0:
            # require gradients for the parameters
            for param in parameters:
                param.requires_grad = True
            # check the gradients
            torch.autograd.gradcheck(
                run_layer,
                parameters,
                eps=1e-5,
                atol=1e-5,
                rtol=1e-5,
            )
        else:
            # check the forward pass
            run_layer(*parameters)


if __name__ == "__main__":
    # unittest.main()
    # TestSDPRLayer().test_scs_compare_primaldual
    # TestSDPRLayer().test_scs_msk_compare_dual()
    # TestSDPRLayer().test_dual_mosek()
    test = TestSDPRLayer()
    test.test_slv_2()
