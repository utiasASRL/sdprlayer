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


class TestSDPRLayer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSDPRLayer, self).__init__(*args, **kwargs)
        self.n = 4
        self.p = 2

    def test_if_1(self):
        # Interface test 1: cost parameterized, constraints fixed, no batch
        set_seed(2)
        n = self.n
        p = self.p
        constraints = []
        for i in range(p):
            A = np.random.rand(n - 1, n - 1)
            A = A @ A.T
            b = np.random.rand(1, 1)
            A = la.block_diag(b, A)
            constraints.append(A)
        constraints.append(A)
        layer = SDPRLayer(n, constraints=constraints, use_dual=False)
        C_tch = torch.randn(n, n, requires_grad=True).double()
        C_tch_symm = (C_tch + C_tch.t()) / 2
        torch.autograd.gradcheck(
            lambda *x: layer(*x, solver_args={"eps": 1e-12, "verbose": True}),
            [C_tch_symm],
            eps=1e-6,
            atol=1e-5,
            rtol=1e-5,
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
        C_tch = torch.randn(n, n, requires_grad=True).double()
        C_tch_symm = (C_tch + C_tch.t()) / 2
        torch.autograd.gradcheck(
            lambda *x: layer(*x, solver_args={"eps": 1e-12}),
            [C_tch_symm],
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
    TestSDPRLayer().test_if_1()
