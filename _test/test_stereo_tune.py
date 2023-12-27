from contextlib import AbstractContextManager
from typing import Any
import numpy as np
import torch
import unittest
import matplotlib.pyplot as plt
import sdprlayer.stereo_tuner as st
from mwcerts.stereo_problems import Localization
from sdprlayer import SDPRLayer


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestStereoTune(unittest.TestCase):
    def __init__(t, *args, **kwargs):
        super(TestStereoTune, t).__init__(*args, **kwargs)
        # Set up ground truth measurements

        # Define camera
        cam_gt = st.Camera(
            f_u=484.5,
            f_v=484.5,
            c_u=0.0,
            c_v=0.0,
            b=0.24,
            sigma_u=0.5,
            sigma_v=0.5,
        )
        t.cam_gt = cam_gt

    def test_data_matrix_no_noise(t):
        # Generate problem
        t.cam_gt.sigma_v = 0.0
        t.cam_gt.sigma_u = 0.0
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=7)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )
        # Get data matrix
        Q_torch = st.get_data_mat(cam_torch, r_l, pixel_meas)
        Q_torch = Q_torch.detach().numpy()
        # Init Localization problem
        prob = Localization(r_p, C_p0, r_l)
        # Get euclidean measurements from pixels
        meas, weights = cam_torch.inverse(*pixel_meas)
        # Check that measurements are correct (should be exact with no noise)
        meas_val = meas.detach().numpy()
        meas_gt = C_p0[0] @ (np.hstack(r_l) - r_p[0])
        np.testing.assert_allclose(meas_val, meas_gt, atol=1e-9)
        # Build meas graph for comparison
        v1 = prob.G.Vp["x0"]
        for i in range(meas.shape[1]):
            v2 = prob.G.Vm[f"m{i}"]
            meas_val = meas[:, [i]].detach().numpy().astype("float32")
            weight_val = weights[i].detach().numpy().astype("float32")
            prob.G.add_edge(v1, v2, meas_val, weight_val)
        # Generate cost matrix
        prob.generate_cost()
        Q_desired = prob.Q.get_matrix(prob.var_list).todense().astype("float32")
        Q_desired[0, 0] = 0.0
        Q_desired = Q_desired / np.linalg.norm(Q_desired, ord="fro")
        # Test
        np.testing.assert_allclose(Q_torch, Q_desired, rtol=1e-7, atol=1e-7)

    def test_data_matrix(t):
        np.random.seed(0)
        torch.manual_seed(0)
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=7)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )
        # Get data matrix
        Q_torch = st.get_data_mat(cam_torch, r_l, pixel_meas)
        Q_torch = Q_torch.detach().numpy()
        # Init Localization problem
        prob = Localization(r_p, C_p0, r_l)
        # Get euclidean measurements from pixels
        meas, weights = cam_torch.inverse(*pixel_meas)
        # Check that measurements are correct (should be exact with no noise)
        meas_val = meas.detach().numpy()
        meas_gt = C_p0[0] @ (np.hstack(r_l) - r_p[0])
        np.testing.assert_allclose(meas_val, meas_gt, atol=2e-2)
        # Build meas graph for comparison
        v1 = prob.G.Vp["x0"]
        for i in range(meas.shape[1]):
            v2 = prob.G.Vm[f"m{i}"]
            meas_val = meas[:, [i]].detach().numpy()
            weight_val = weights[i].detach().numpy()
            prob.G.add_edge(v1, v2, meas_val, weight_val)
        # Generate cost matrix
        prob.generate_cost()
        Q_desired = prob.Q.get_matrix(prob.var_list).todense()
        Q_desired[0, 0] = 0.0
        Q_desired = Q_desired / np.linalg.norm(Q_desired, ord="fro")
        # Test
        np.testing.assert_allclose(Q_torch, Q_desired, rtol=1e-7, atol=1e-7)

    def test_forward(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)

        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=50)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        # Define a localization class to get the constraints
        prob = Localization(r_p, C_p0, r_l)
        prob.generate_constraints()
        prob.generate_redun_constraints()
        constraints = prob.constraints + prob.constraints_r
        constraints_list = [(c.A.get_matrix(prob.var_list), c.b) for c in constraints]

        # Build Layer
        sdpr_layer = SDPRLayer(13, Constraints=constraints_list, use_dual=True)
        # Run Forward pass
        Q = st.get_data_mat(cam_torch, r_l, pixel_meas)
        solver_args = {"solve_method": "mosek", "verbose": True}
        X = sdpr_layer(Q, solver_args=solver_args)[0]
        # Get the solution
        X = X.detach().numpy()
        # Make sure it is rank-1
        assert np.linalg.matrix_rank(X, tol=1e-6) == 1
        # Extract solution
        r_inP = X[10:, [0]]
        C_vec = X[1:10, [0]]
        C = C_vec.reshape((3, 3), order="F")
        # Check the error
        np.testing.assert_allclose(C.T @ r_inP, r_p[0], atol=1e-3)
        np.testing.assert_allclose(C.T @ C_p0[0], np.eye(3), atol=5e-4)

    def test_grads_camera(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=5)

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # TEST CAMERA INVERSE
        # Test inverse camera gradients
        def inverse_wrapper_meas(*x):
            cam = st.Camera(*x)
            meas, weight = cam.inverse(*pixel_meas)
            return meas

        torch.autograd.gradcheck(
            inverse_wrapper_meas,
            inputs=params,
            eps=1e-5,
            atol=1e-5,
            rtol=0,
        )

        def inverse_wrapper_weight(*x):
            cam = st.Camera(*x)
            meas, weight = cam.inverse(*pixel_meas)
            return weight

        torch.autograd.gradcheck(
            inverse_wrapper_weight,
            inputs=params,
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_grads_data_mat(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=5)

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # TEST DATA MATRIX GENERATION
        def data_mat_wrapper(*x):
            cam = st.Camera(*x)
            return st.get_data_mat(cam, r_l, pixel_meas)

        torch.autograd.gradcheck(
            data_mat_wrapper,
            inputs=params,
            eps=1e-5,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_grads_optlayer(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)

        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=5)

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # TEST OPT LAYER
        # Define a localization class to get the constraints
        prob = Localization(r_p, C_p0, r_l)
        prob.generate_constraints()
        prob.generate_redun_constraints()
        constraints = prob.constraints + prob.constraints_r
        constraints_list = [(c.A.get_matrix(prob.var_list), c.b) for c in constraints]
        # Build opt layer
        sdpr_layer = SDPRLayer(13, constraints=constraints_list, use_dual=False)
        sdpr_layer_dual = SDPRLayer(13, constraints=constraints_list, use_dual=True)

        # Test gradients to matrix
        cam = st.Camera(*params)
        Q = st.get_data_mat(cam, r_l, pixel_meas)
        solver_args = {"solve_method": "SCS", "eps": 1e-9}
        torch.autograd.gradcheck(
            lambda x: sdpr_layer(x, solver_args=solver_args)[0],
            Q,
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

        # Generate solution X from parameters
        def get_X_from_params(*params, solver, verbose=True):
            cam = st.Camera(*params)
            Q = st.get_data_mat(cam, r_l, pixel_meas)
            Q.retain_grad()
            # Run Forward pass
            if solver == "mosek":
                # Run Forward pass
                mosek_params = {
                    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-13,
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
                    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-14,
                    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-13,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-13,
                }
                solver_args = {
                    "solve_method": "mosek",
                    "mosek_params": mosek_params,
                    "verbose": verbose,
                }
                X = sdpr_layer_dual(Q, solver_args=solver_args)[0]
            elif solver == "SCS":
                solver_args = {"solve_method": "SCS", "eps": 1e-9, "verbose": verbose}
                X = sdpr_layer(Q, solver_args=solver_args)[0]
            assert (
                np.linalg.matrix_rank(X.detach().numpy(), tol=1e-6) == 1
            ), "X is not rank-1"
            return X

        torch.autograd.gradcheck(
            lambda *x: get_X_from_params(*x, solver="SCS"),
            params,
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

        torch.autograd.gradcheck(
            lambda *x: get_X_from_params(*x, solver="mosek"),
            params,
            eps=1e-6,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_grad_loss(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)

        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=5)

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # Define a localization class to get the constraints
        prob = Localization(r_p, C_p0, r_l)
        prob.generate_constraints()
        prob.generate_redun_constraints()
        constraints = prob.constraints + prob.constraints_r
        constraints_list = [(c.A.get_matrix(prob.var_list), c.b) for c in constraints]
        # Build opt layer
        sdpr_layer = SDPRLayer(13, constraints=constraints_list, use_dual=False)
        sdpr_layer_dual = SDPRLayer(13, constraints=constraints_list, use_dual=True)

        # Test gradients to matrix
        cam = st.Camera(*params)
        Q = st.get_data_mat(cam, r_l, pixel_meas)
        solver_args = {"solve_method": "SCS", "eps": 1e-9}
        X = sdpr_layer(Q, solver_args=solver_args)[0]
        x = X[:, [1]]

        # define loss function based on vector solution
        def get_loss_vec(x):
            X_new = x @ x.T
            return st.get_loss_from_sol(X_new, r_p, C_p0)

        torch.autograd.gradcheck(
            get_loss_vec,
            x,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
        )

    def test_tune_gt_init(t, plot=False):
        """test the case where we start at the ground truth."""
        set_seed(0)
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        # Define optimizer
        params = [cam_torch.b]
        opt = torch.optim.SGD(params=params, lr=1e-3)

        # Run Tuner
        iter_info = st.tune_stereo_params(
            cam_torch=cam_torch,
            opt=opt,
            params=params,
            r_p=r_p,
            C_p0=C_p0,
            r_l=r_l,
            pixel_meas=pixel_meas,
            verbose=True,
        )

        if plot:
            plt.figure()
            plt.plot(iter_info["loss"])
            plt.xlabel("Iteration")
            plt.ylabel
            plt.show()

    def test_tune_baseline_no_noise(t, plot=False):
        """test the case where we start at the ground truth."""
        set_seed(0)
        # Offset to add to parameter
        offs = 0.2
        t.cam_gt.sigma_u = 0.0
        t.cam_gt.sigma_v = 0.0
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=20)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b + offs, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        # Define optimizer
        params = [cam_torch.b]
        opt = torch.optim.SGD(params=params, lr=5e-3)

        # Run Tuner
        iter_info = st.tune_stereo_params(
            cam_torch=cam_torch,
            opt=opt,
            params=params,
            r_p=r_p,
            C_p0=C_p0,
            r_l=r_l,
            pixel_meas=pixel_meas,
            verbose=True,
        )
        if plot:
            fig, axs = plt.subplots(2, 1)
            axs[0].plot(iter_info["loss"])
            axs[0].set_ylabel("Loss")
            axs[1].plot(iter_info["params"])
            axs[1].axhline(t.cam_gt.b, color="k", linestyle="--")
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel("Baseline")
            plt.show()

        np.testing.assert_allclose(cam_torch.b.detach().numpy(), t.cam_gt.b, atol=1e-3)

    def test_tune_f_no_noise(t, plot=False):
        """test the case where we start at the ground truth."""
        set_seed(0)
        # Offset to add to parameter
        offs = 100
        t.cam_gt.sigma_u = 0.0
        t.cam_gt.sigma_v = 0.0
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=20)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u + offs, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        # Define optimizer
        params = [cam_torch.f_u]
        opt = torch.optim.SGD(params=params, lr=1e4)

        # Run Tuner
        iter_info = st.tune_stereo_params(
            cam_torch=cam_torch,
            opt=opt,
            params=params,
            r_p=r_p,
            C_p0=C_p0,
            r_l=r_l,
            pixel_meas=pixel_meas,
            verbose=True,
        )
        if plot:
            fig, axs = plt.subplots(2, 1)
            axs[0].plot(iter_info["loss"])
            axs[0].set_ylabel("Loss")
            axs[1].plot(iter_info["params"])
            axs[1].axhline(t.cam_gt.f_u, color="k", linestyle="--")
            axs[1].set_xlabel("Iteration")
            axs[1].set_ylabel("Baseline")
            plt.figure()
            plt.plot(iter_info["params"], iter_info["loss"])
            plt.show()

        np.testing.assert_allclose(
            cam_torch.f_u.detach().numpy(), t.cam_gt.f_u, atol=1e-2
        )

    def test_tune_params_no_noise(t, plot=False):
        """Test offsets for all parameters. Remove all noise (weighting set to identity)"""
        set_seed(0)
        # Turn off noise
        t.cam_gt.sigma_u = 0.0
        t.cam_gt.sigma_v = 0.0

        # dictionary of paramter test values
        param_dict = {
            "f_u": dict(offs=100, lr=1e4, tol_grad_sq=1e-12, atol=1e-1),
            "f_v": dict(offs=100, lr=5e4, tol_grad_sq=1e-15, atol=1e-1),
            "c_u": dict(offs=100, lr=1e4, tol_grad_sq=1e-15, atol=1e-2),
            "c_v": dict(offs=100, lr=1e4, tol_grad_sq=1e-15, atol=1e-2),
            "b": dict(offs=0.2, lr=5e-3, tol_grad_sq=1e-11, atol=1e-5),
        }
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=20)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        for key, tune_params in param_dict.items():
            # Add offset
            getattr(cam_torch, key).data += tune_params["offs"]

            # Define optimizer
            params = [getattr(cam_torch, key)]

            # Run Tuner
            iter_info = st.tune_stereo_params(
                cam_torch=cam_torch,
                params=params,
                tune_params=tune_params,
                r_p=r_p,
                C_p0=C_p0,
                r_l=r_l,
                pixel_meas=pixel_meas,
                verbose=True,
            )
            if plot:
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(iter_info["loss"])
                axs[0].set_ylabel("Loss")
                axs[0].set_title(f"Parameter: {key}")
                axs[1].plot(iter_info["params"])
                axs[1].axhline(getattr(t.cam_gt, key), color="k", linestyle="--")
                axs[1].set_xlabel("Iteration")
                axs[1].set_ylabel("Parameter Value")
                plt.figure()
                plt.plot(iter_info["params"], iter_info["loss"])
                plt.ylabel("Loss")
                plt.xlabel("Parameter Value")
                plt.title(f"Parameter: {key}")

                plt.show()

            np.testing.assert_allclose(
                getattr(cam_torch, key).detach().numpy(),
                getattr(t.cam_gt, key),
                atol=tune_params["atol"],
            )

    def test_tune_params(t, plot=False):
        """Test offsets for all parameters. Use default noise level"""
        set_seed(0)
        # Set noise to one half pixel.
        t.cam_gt.sigma_u = 0.5
        t.cam_gt.sigma_v = 0.5

        # dictionary of paramter test values
        param_dict = {
            "f_u": dict(offs=100, lr=1e4, tol_grad_sq=1e-12, atol=5),
            "f_v": dict(offs=100, lr=5e4, tol_grad_sq=1e-15, atol=5),
            "c_u": dict(offs=100, lr=1e4, tol_grad_sq=1e-15, atol=5),
            "c_v": dict(offs=100, lr=1e4, tol_grad_sq=1e-15, atol=5),
            "b": dict(offs=0.2, lr=5e-3, tol_grad_sq=1e-10, atol=2e-3),
        }
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=20)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        for key, tune_params in param_dict.items():
            # Add offset
            getattr(cam_torch, key).data += tune_params["offs"]

            # Define optimizer
            params = [getattr(cam_torch, key)]

            # Run Tuner
            iter_info = st.tune_stereo_params(
                cam_torch=cam_torch,
                params=params,
                tune_params=tune_params,
                r_p=r_p,
                C_p0=C_p0,
                r_l=r_l,
                pixel_meas=pixel_meas,
                verbose=True,
            )
            if plot:
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(iter_info["loss"])
                axs[0].set_ylabel("Loss")
                axs[0].set_title(f"Parameter: {key}")
                axs[1].plot(iter_info["params"])
                axs[1].axhline(getattr(t.cam_gt, key), color="k", linestyle="--")
                axs[1].set_xlabel("Iteration")
                axs[1].set_ylabel("Parameter Value")
                plt.figure()
                plt.plot(iter_info["params"], iter_info["loss"])
                plt.ylabel("Loss")
                plt.xlabel("Parameter Value")
                plt.title(f"Parameter: {key}")

                plt.show()

            np.testing.assert_allclose(
                getattr(cam_torch, key).detach().numpy(),
                getattr(t.cam_gt, key),
                atol=tune_params["atol"],
            )

    def test_tune_params_unsuper(t, plot=False):
        """Test offsets for all parameters. Use default noise level"""
        set_seed(0)
        # Set noise to one half pixel.
        t.cam_gt.sigma_u = 0.5
        t.cam_gt.sigma_v = 0.5

        # dictionary of paramter test values
        param_dict = {
            "f_u": dict(offs=100, lr=1e4, tol_grad_sq=1e-12, atol=5),
            "f_v": dict(offs=100, lr=5e4, tol_grad_sq=1e-15, atol=5),
            "c_u": dict(offs=100, lr=1e4, tol_grad_sq=1e-15, atol=5),
            "c_v": dict(offs=100, lr=1e4, tol_grad_sq=1e-15, atol=5),
            "b": dict(offs=0.2, lr=5e-3, tol_grad_sq=1e-10, atol=2e-3),
        }
        # Generate problem
        r_p, C_p0, r_l, pixel_meas = st.get_prob_data(camera=t.cam_gt, Nm=20)

        # generate parameterized camera
        cam_torch = st.Camera(
            f_u=torch.tensor(t.cam_gt.f_u, requires_grad=True),
            f_v=torch.tensor(t.cam_gt.f_v, requires_grad=True),
            c_u=torch.tensor(t.cam_gt.c_u, requires_grad=True),
            c_v=torch.tensor(t.cam_gt.c_v, requires_grad=True),
            b=torch.tensor(t.cam_gt.b, requires_grad=True),
            sigma_u=t.cam_gt.sigma_u,
            sigma_v=t.cam_gt.sigma_v,
        )

        for key, tune_params in param_dict.items():
            # Add offset
            getattr(cam_torch, key).data += tune_params["offs"]

            # Define optimizer
            params = [getattr(cam_torch, key)]

            # Run Tuner
            iter_info = st.tune_stereo_params_unsuper(
                cam_torch=cam_torch,
                params=params,
                tune_params=tune_params,
                r_p=r_p,
                C_p0=C_p0,
                r_l=r_l,
                pixel_meas=pixel_meas,
                verbose=True,
            )
            if plot:
                fig, axs = plt.subplots(2, 1)
                axs[0].plot(iter_info["loss"])
                axs[0].set_ylabel("Loss")
                axs[0].set_title(f"Parameter: {key}")
                axs[1].plot(iter_info["params"])
                axs[1].axhline(getattr(t.cam_gt, key), color="k", linestyle="--")
                axs[1].set_xlabel("Iteration")
                axs[1].set_ylabel("Parameter Value")
                plt.figure()
                plt.plot(iter_info["params"], iter_info["loss"])
                plt.ylabel("Loss")
                plt.xlabel("Parameter Value")
                plt.title(f"Parameter: {key}")

                plt.show()

            np.testing.assert_allclose(
                getattr(cam_torch, key).detach().numpy(),
                getattr(t.cam_gt, key),
                atol=tune_params["atol"],
            )


if __name__ == "__main__":
    # unittest.main()
    test = TestStereoTune()
    test.test_tune_params_unsuper(plot=True)
