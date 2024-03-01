import unittest

import numpy as np
import numpy.random as npr
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
    def __init__(self, n=4, p=2, *args, **kwargs):
        # Test setup is based on solving a random generic SDPs, but with homogenization.
        super(TestSDPRLayer, self).__init__(*args, **kwargs)
        set_seed(2)
        n = 4  # Dimension of SDP (non-homogenized)
        p = 3  # Number of constraints
        vdim = np.floor(n * (n + 1) / 2).astype(int)
        # Randomly generate constraints and objective of the form
        G_vec = torch.rand(vdim, p)
        F_vec = torch.rand(vdim)
        b = torch.rand(p, 1)
        # Store objective
        self.objective = unvec_symm(F_vec, n)
        # Store homogenized objective
        self.objective_h = homog_mat(
            self.objective, torch.zeros(n, 1), torch.zeros(1, 1)
        )
        params_all = [self.objective, torch.zeros(1, 1)]
        # Store constraints
        self.constraints = []
        self.constraints_h = []
        for i in range(p):
            G = unvec_symm(G_vec[:, i], n)
            g_vec = torch.zeros(n, 1)
            g = -b[i]
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
        Q = self.objective
        q_vec = torch.rand(Q.shape[0], 1)
        q = torch.rand(1, 1)
        Q_h_t = homog_mat(Q, q_vec, q)
        # Numpy test
        Q = self.objective.detach().numpy()
        q_vec = q_vec.detach().numpy()
        q = q.detach().numpy()
        Q_h_n = homog_mat(Q, q_vec, q)

        np.testing.assert_allclose(Q_h_n, Q_h_t.detach().numpy())

    def test_if_1(self):
        # Interface Test 1: pre-homogenized, fixed constraints, param objective
        parameters = self.params_all[0:2]
        self.run_interface_test(
            constraints=self.constraints_h,
            objective=None,
            parameters=parameters,
            homogenize=False,
        )

    def test_if_1_old(self):
        # Interface Test 1: pre-homogenized, fixed constraints, param objective
        constraints = [homog_mat(*c) for c in self.constraints]
        layer = SDPRLayer(self.n + 1, constraints=constraints)

        def run_layer(C):
            obj = homog_mat(C, torch.zeros(C.shape[0], 1), torch.zeros(1, 1))
            X, x = layer(obj, solver_args={"eps": 1e-12})
            return x

        self.objective.requires_grad = True
        torch.autograd.gradcheck(
            run_layer,
            [self.objective],
            eps=1e-9,
            atol=1e-6,
            rtol=1e-6,
        )

    def run_interface_test(self, constraints, objective, parameters, homogenize):
        """Generalized test for the interface of the SDPRLayer class."""
        # If already homogenized incease the dimension
        if homogenize:
            n = self.n
        else:
            n = self.n + 1
        # Make layer
        layer = SDPRLayer(
            n, objective=objective, constraints=constraints, homogenize=homogenize
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
                    params_h.append(param_triplet)
                ind += 2
            # Call layer
            X, x = layer(*params_h, solver_args={"eps": 1e-12})
            return x

        # require gradients for the parameters
        for param in parameters:
            param.requires_grad = True
        # check the gradients
        torch.autograd.gradcheck(
            run_layer,
            parameters,
            eps=1e-9,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_primal_scs(self):
        set_seed(2)
        n = self.n
        p = self.p
        constraints = []
        for i in range(p):
            A = np.random.rand(n, n)
            A = A @ A.T
            constraints.append(A)
        layer = SDPRLayer(n, constraints=constraints, use_dual=False)
        matdim = n * (n + 1) / 2

        def run_layer(c_vec):
            C = unvec_symm(c_vec)
            layer(C, solver_args={"eps": 1e-12})

        torch.autograd.gradcheck(
            run_layer,
            [np.random.rand(matdim, 1)],
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_dual_scs(self):
        set_seed(2)
        n = self.n
        p = self.p
        constraints = []
        for i in range(p):
            A = np.random.rand(n, n)
            A = A @ A.T
            b = np.random.rand(1)
            constraints.append((A, b))
        layer = SDPRLayer(n, constraints=constraints, use_dual=True)
        C_tch = torch.randn(n, n, requires_grad=True).double()
        C_tch_symm = (C_tch + C_tch.t()) / 2
        torch.autograd.gradcheck(
            lambda *x: layer(*x, solver_args={"eps": 1e-12}),
            [C_tch_symm],
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_scs_compare_primaldual(self):
        set_seed(2)
        n = self.n
        p = self.p
        constraints = []
        for i in range(p):
            A = np.random.rand(n, n)
            A = A @ A.T
            b = np.random.rand(1)
            constraints.append((A, b))
        layer = SDPRLayer(n, constraints=constraints, use_dual=False)
        layer_d = SDPRLayer(n, constraints=constraints, use_dual=True)
        C_tch = torch.randn(n, n, requires_grad=True).double()
        C_tch_symm = (C_tch + C_tch.t()) / 2
        C_tch_symm.retain_grad()
        X = layer(C_tch_symm, solver_args={"eps": 1e-12})[0]
        X_d = layer_d(C_tch_symm, solver_args={"eps": 1e-12})[0]
        assert np.allclose(X.detach().numpy(), X_d.detach().numpy())

        # Check cost gradients
        for idx in list(zip(*np.triu_indices(n))):
            eval = torch.zeros((n, n))
            eval[idx] = 1.0
            eval[idx[::-1]] = 1.0
            X.backward(eval, retain_graph=True)
            grad = C_tch_symm.grad.detach().numpy().copy()
            C_tch_symm.grad.zero_()
            X_d.backward(eval, retain_graph=True)
            grad_d = C_tch_symm.grad.detach().numpy().copy()
            C_tch_symm.grad.zero_()
            assert np.allclose(grad, grad_d, atol=1e-7)

    def test_scs_msk_compare_dual(self):
        set_seed(2)
        n = self.n
        p = self.p
        constraints = []
        for i in range(p):
            A = np.random.rand(n, n)
            A = A @ A.T
            b = np.random.rand(1)
            constraints.append((A, b))
        layer = SDPRLayer(n, constraints=constraints, use_dual=False)
        layer_d = SDPRLayer(n, constraints=constraints, use_dual=True)
        C_tch = torch.randn(n, n, requires_grad=True).double()
        C_tch_symm = (C_tch + C_tch.t()) / 2
        C_tch_symm.retain_grad()
        X = layer(C_tch_symm, solver_args={"eps": 1e-12})[0]
        mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
        }
        msk_solver_args = {
            "solve_method": "mosek",
            "mosek_params": mosek_params,
            "verbose": False,
        }
        X_d = layer_d(C_tch_symm, solver_args=msk_solver_args)[0]
        assert np.allclose(X.detach().numpy(), X_d.detach().numpy())

        # Check cost gradients
        for idx in list(zip(*np.triu_indices(n))):
            eval = torch.zeros((n, n))
            eval[idx] = 1.0
            eval[idx[::-1]] = 1.0
            X.backward(eval, retain_graph=True)
            grad = C_tch_symm.grad.detach().numpy().copy()
            C_tch_symm.grad.zero_()
            X_d.backward(eval, retain_graph=True)
            grad_d = C_tch_symm.grad.detach().numpy().copy()
            C_tch_symm.grad.zero_()
            assert np.allclose(grad, grad_d, atol=1e-7)

    def test_dual_mosek(self):
        set_seed(2)
        n = self.n
        p = self.p
        constraints = []
        for i in range(p):
            A = np.random.rand(n, n)
            A = A @ A.T
            b = np.random.rand(1)
            constraints.append((A, b))
        layer = SDPRLayer(n, constraints=constraints, use_dual=True)
        C_tch = torch.randn(n, n, requires_grad=True).double()
        C_tch_symm = (C_tch + C_tch.T) / 2
        # arguments for sdp solver
        mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-10,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
        }
        sdp_solver_args = {
            "solve_method": "mosek",
            "mosek_params": mosek_params,
            "verbose": False,
        }

        # Check strict feasibility
        X = layer(C_tch_symm * 0.0, solver_args=sdp_solver_args)[0]
        assert np.trace(X.detach().numpy()) > 1e-3

        torch.autograd.gradcheck(
            lambda *x: layer(*x, solver_args=sdp_solver_args)[0],
            [C_tch],
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
            raise_exception=True,
        )


if __name__ == "__main__":
    # unittest.main()
    # TestSDPRLayer().test_scs_compare_primaldual
    # TestSDPRLayer().test_scs_msk_compare_dual()
    # TestSDPRLayer().test_dual_mosek()
    test = TestSDPRLayer()
    test.test_if_1()
