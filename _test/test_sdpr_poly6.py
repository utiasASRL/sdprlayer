import os
import unittest

import matplotlib.pylab as plt
import numpy as np
import scipy.sparse as sp
import torch
from diffcp import cones

from sdprlayers.layers.sdprlayer import SDPRLayer, SDPRLayerMosek

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")


class TestSDPRPoly6(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSDPRPoly6, self).__init__(*args, **kwargs)

        # Define polynomial
        p_vals = np.array(
            [5.0000, 1.3167 * 2, -1.4481 * 3, 0 * 4, 0.2685 * 3, -0.0667 * 2, 0.0389]
        )
        # homogenizing constraint not included here as it is added by the layer
        constraints = []
        A = sp.csc_array((4, 4))  # x^2 = x*x
        A[2, 0] = 1 / 2
        A[0, 2] = 1 / 2
        A[1, 1] = -1
        constraints += [A]
        A = sp.csc_array((4, 4))  # x^3 = x^2*x
        A[3, 0] = 1
        A[0, 3] = 1
        A[1, 2] = -1
        A[2, 1] = -1
        constraints += [A]
        A = sp.csc_array((4, 4))  # x^3*x = x^2*x^2 Redundant
        A[3, 1] = 1 / 2
        A[1, 3] = 1 / 2
        A[2, 2] = -1
        constraints += [A]

        # Candidate solution
        x_cand = np.array([[1.0000, -1.4871, 2.2115, -3.2888]]).T

        # Dual optimal
        mults = -np.array([[-3.1937], [2.5759], [-0.0562], [0.8318]])

        self.data = dict(
            p_vals=p_vals, constraints=constraints, x_cand=x_cand, opt_mults=mults
        )

    def test_run_sdp(self):
        """The goal of this script is to shift the optimum of the polynomial
        to a different point by using backpropagtion on rank-1 SDPs"""
        np.random.seed(2)
        # Get data from data function
        constraints = self.data["constraints"]
        # Create SDPR Layer
        optlayer = SDPRLayer(n_vars=4, constraints=constraints, use_dual=False)

        # Set up polynomial parameter tensor
        p = torch.tensor(self.data["p_vals"], requires_grad=True)
        sdp_solver_args = {"eps": 1e-9, "verbose": True}
        X, x = optlayer(build_data_mat(p), solver_args=sdp_solver_args)
        x_vals = x.detach().numpy()
        np.testing.assert_allclose(x_vals, self.data["x_cand"], rtol=1e-3, atol=1e-3)

    def test_grad_configs(self):
        """Test SDPRLayers with under different configurations"""

        for use_mosek in [False, True]:  # Mosek as solver
            for use_dual in [False, True]:  # Dual or Primal Formulation
                for diff_qcqp in [False, True]:  # Differentiate through QCQP
                    for compute_mult in [False, True]:  # Recompute Lagrange Multipliers
                        print(
                            f"Test Config: dual: {use_dual}, mosek: {use_mosek}, diff qcqp: {diff_qcqp}, recompute mults: {compute_mult}"
                        )
                        self.run_sdprlayers_grad_test(
                            use_mosek=use_mosek,
                            use_dual=use_dual,
                            diff_qcqp=diff_qcqp,
                            compute_mult=compute_mult,
                        )
                        print("PASS")

    def test_grad_remove_constraint(self):
        """Tests what happens when too many constraints are considered redundant"""
        for diff_qcqp in [False, True]:  # Differentiate through QCQP
            for compute_mult in [False, True]:  # Recompute Lagrange Multipliers
                try:
                    self.run_sdprlayers_grad_test(
                        diff_qcqp=diff_qcqp,
                        compute_mult=compute_mult,
                        redun_list=[1, 2],
                    )
                    raise ValueError("Execution should have failed")
                except:
                    pass

    def run_sdprlayers_grad_test(
        self,
        use_mosek=True,
        use_dual=True,
        diff_qcqp=True,
        compute_mult=False,
        redun_list=[2],
    ):
        """Test SDPRLayer with different configurations"""
        # Get data from data function
        constraints = self.data["constraints"]

        # Set up polynomial parameter tensor
        p = torch.tensor(self.data["p_vals"], requires_grad=True)

        # Create SDPR Layer
        sdpr_args = dict(
            n_vars=4,
            constraints=constraints,
            use_dual=use_dual,
            diff_qcqp=diff_qcqp,
            compute_multipliers=compute_mult,
            redun_list=redun_list,
        )
        if use_mosek:
            sdpr_args["mosek_params"] = {
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
                "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
                "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
                "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
                "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
                "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
            }
            optlayer = SDPRLayerMosek(**sdpr_args)
        else:
            optlayer = SDPRLayer(**sdpr_args)

        # Define loss
        # NOTE: we skip the derivative wrt p_0 since it should be identically zero. Numerical issues cause it to be different.
        p_0 = p[0]

        def gen_loss(p_val, **kwargs):
            p_vals = torch.hstack([p_0, p_val])
            X, x = optlayer(build_data_mat(p_vals), **kwargs)
            return x

        # Check gradient w.r.t. parameter p
        if not use_mosek:
            kwargs = dict(solver_args={"eps": 1e-10})
            tols = dict(eps=1e-3, atol=1e-3, rtol=1e-2)
        else:
            kwargs = {}
            tols = dict(eps=1e-3, atol=1e-3, rtol=0)
        torch.autograd.gradcheck(lambda *p: gen_loss(*p, **kwargs), [p[1:]], **tols)

    def test_ext_solution(self):
        """Test SDPRLayer Using externally computed solution"""
        # Get data from data function
        constraints = self.data["constraints"]

        # Set up polynomial parameter tensor
        p = torch.tensor(self.data["p_vals"], requires_grad=True)

        # Create SDPR Layers
        mosek_params = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 500,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
            "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
            "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-12,
            "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
        }
        # First layer just computes solution
        optlayer1 = SDPRLayerMosek(
            n_vars=4,
            constraints=constraints,
            use_dual=True,
            diff_qcqp=True,
            compute_multipliers=False,
            redun_list=[2],
            mosek_params=mosek_params,
        )
        # Second layer used to diff.
        optlayer2 = SDPRLayer(
            n_vars=4,
            constraints=constraints,
            use_dual=True,
            diff_qcqp=True,
            compute_multipliers=True,
            redun_list=[2],
        )

        # Define loss
        # NOTE: we skip the derivative wrt p_0 since it should be identically zero. Numerical issues cause it to be different.
        p_0 = p[0]

        def gen_loss(p_val, **kwargs):
            p_vals = torch.hstack([p_0, p_val])
            with torch.no_grad():
                # Compute solution
                x_vals = optlayer1(build_data_mat(p_vals), **kwargs)[1].numpy()

            # Differentiate the solution that is passed into the second layer
            vecX = cones.vec_symm(x_vals @ x_vals.T)
            ext_vars_list = [dict(x=np.zeros(4), s=np.zeros(vecX.shape), y=vecX)]
            X, x = optlayer2(build_data_mat(p_vals), ext_vars_list=ext_vars_list)

            return x

        # Check gradient w.r.t. parameter p

        kwargs = {}
        tols = dict(eps=1e-3, atol=1e-3, rtol=0)
        torch.autograd.gradcheck(lambda *p: gen_loss(*p, **kwargs), [p[1:]], **tols)

    def test_redundant_constraint(self):
        """This function tests the redundant constraint search functionality"""
        np.random.seed(2)
        # Get constraints
        constraints = self.data["constraints"]
        # Remove redundant constraint
        constraints = constraints[:-1]
        # Create SDPR Layer
        optlayer = SDPRLayer(n_vars=4, constraints=constraints, use_dual=False)

        # Set up polynomial parameter tensor
        p = torch.tensor(self.data["p_vals"], requires_grad=True)
        sdp_solver_args = {"eps": 1e-9, "verbose": True}
        X, x = optlayer(build_data_mat(p), solver_args=sdp_solver_args)
        # Check Tightness
        tight, ER = SDPRLayer.check_tightness(X)
        assert not tight, ValueError("Problem should not be tight")

        # Generate a set of samples of the feasible set
        n_samples = 10
        samples = []
        for i in range(n_samples):
            x = (np.random.rand() * 2 - 1) * 2
            samples += [np.array([1, x, x**2, x**3])]
        # Get new set of constraints
        constraints_new = SDPRLayer.find_constraints(samples)
        # Redefine layer with new constraints
        optlayer = SDPRLayer(n_vars=4, constraints=constraints_new, use_dual=False)
        X, x = optlayer(build_data_mat(p), solver_args=sdp_solver_args)
        # Check Tightness
        tight, ER_new = SDPRLayer.check_tightness(X)
        assert tight, ValueError("Problem should be tight")

    def test_tune_optimum(self, display=False):
        """The goal of this script is to shift the optimum of the polynomial
        to a different point by using backpropagtion on rank-1 SDPs"""
        np.random.seed(2)
        # Get data from data function
        constraints = self.data["constraints"]

        # Create SDPR Layer
        optlayer = SDPRLayer(n_vars=4, constraints=constraints)

        # Set up polynomial parameter tensor
        p = torch.tensor(self.data["p_vals"], requires_grad=True)

        # Define loss
        def gen_loss(p_val, **kwargs):
            x_target = -1
            sdp_solver_args = {"eps": 1e-9}
            X, x = optlayer(build_data_mat(p_val), solver_args=sdp_solver_args)
            loss = 1 / 2 * (x[1, 0] - x_target) ** 2
            return loss, X

        # Define Optimizer
        opt = torch.optim.Adam(params=[p], lr=1e-2)
        # Execute iterations
        losses = []
        minima = []
        max_iter = 120
        X_init = None
        n_iter = 0
        loss_val = np.inf
        while loss_val > 1e-4 and n_iter < max_iter:
            # Update Loss
            opt.zero_grad()
            loss, sol = gen_loss(p)
            if n_iter == 0:
                X_init = sol.cpu().detach().numpy()
            # run optimizer
            loss.backward(retain_graph=True)
            opt.step()
            loss_val = loss.item()
            losses.append(loss_val)
            x_min = sol.detach().numpy()[0, 1]
            n_iter += 1
            if display:
                print(f"min:\t{x_min}\tloss:\t{losses[-1]}")
        if display:
            print(f"ITERATIonS: \t{n_iter}")
        # Check the rank of the solution
        X_new = sol.detach().numpy()
        evals_new = np.sort(np.linalg.eigvalsh(X_new))[::-1]
        evr_new = evals_new[0] / evals_new[1]
        if display:
            print(f"New Eigenvalue Ratio:\t{evr_new}")

        if display:
            plt.figure()
            plot_polynomial(p_vals=self.data["p_vals"])
            plot_polynomial(p_vals=p.detach().numpy())
            plt.axvline(x=X_init[0, 1], color="r", linestyle="--")
            plt.axvline(x=X_new[0, 1], color="b", linestyle="--")
            plt.legend(["initial poly", "new poly", "initial argmin", "new argmin"])
            plt.show()
            plt.title("")

        # Assert that optimization worked as expected.
        assert n_iter < max_iter, ValueError("Terminated on max iterations")
        assert loss_val <= 1e-4, ValueError("Loss did not drop to expected value")
        assert np.log10(evr_new) >= 9, ValueError("Solution is not Rank-1")


def plot_polynomial(p_vals):
    x = np.linspace(-2.5, 2, 100)
    y = np.polyval(p_vals[::-1], x)
    plt.plot(x, y)


# Define Q tensor from polynomial parameters
def build_data_mat(p):
    Q_tch = torch.zeros((4, 4), dtype=torch.double)
    Q_tch[0, 0] = p[0]
    Q_tch[[1, 0], [0, 1]] = p[1] / 2
    Q_tch[[2, 1, 0], [0, 1, 2]] = p[2] / 3
    Q_tch[[3, 2, 1, 0], [0, 1, 2, 3]] = p[3] / 4
    Q_tch[[3, 2, 1], [1, 2, 3]] = p[4] / 3
    Q_tch[[3, 2], [2, 3]] = p[5] / 2
    Q_tch[3, 3] = p[6]

    return Q_tch


if __name__ == "__main__":
    test = TestSDPRPoly6()
    # test_prob_sdp(display=True)
    # test_prob_local(display=True)
    # test_redundant_constraint()
    # test.test_grad_configs()
    # test.test_grad_remove_constraint()
    test.test_ext_solution()
