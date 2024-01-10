import unittest

import matplotlib.pyplot as plt
import numpy as np
import torch

from mwcerts.stereo_problems import Localization
import sdprlayer.stereo_tuner as st
from sdprlayer import SDPRLayer


def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)


class TestStereoTune(unittest.TestCase):
    def __init__(t, *args, no_noise=False, **kwargs):
        super(TestStereoTune, t).__init__(*args, **kwargs)
        # Default dtype
        torch.set_default_dtype(torch.float64)

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
        # Noise
        t.set_noise(no_noise)

    def set_noise(t, no_noise):
        t.no_noise = no_noise
        if no_noise:
            # turn off noise
            t.cam_gt.sigma_u = 0.0
            t.cam_gt.sigma_v = 0.0
        else:
            # turn on noise
            t.cam_gt.sigma_u = 0.5
            t.cam_gt.sigma_v = 0.5

    def test_data_matrix(t, N_batch=2):
        # Generate problem
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=7, N_batch=N_batch
        )

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
        pixel_meass = torch.tensor(pixel_meass)
        Q_torch = st.get_data_mat(cam_torch, r_ls, pixel_meass)
        Q_torch = Q_torch.detach().numpy()
        # Get euclidean measurements from pixels
        meas, weights = cam_torch.inverse(pixel_meass)
        # Build meas graph for comparison
        for b in range(N_batch):
            # Check that measurements are correct (should be exact with no noise)
            if t.no_noise:
                meas_val = meas[b].detach().numpy()
                meas_gt = C_p0s[b] @ (r_ls[b] - r_ps[b])
                np.testing.assert_allclose(meas_val, meas_gt, atol=1e-9)
            # Init Localization problem
            r_ls_b = [r_ls[b, :, [i]] for i in range(r_ls.shape[2])]
            prob = Localization([r_ps[b]], [C_p0s[b]], r_ls_b)
            v1 = prob.G.Vp["x0"]
            for i in range(meas.shape[-1]):
                v2 = prob.G.Vm[f"m{i}"]
                meas_val = meas[b, :, [i]].detach().numpy()
                weight_val = weights[b, i].detach().numpy()
                prob.G.add_edge(v1, v2, meas_val, weight_val)
            # Generate cost matrix
            prob.generate_cost()
            Q_desired = prob.Q.get_matrix(prob.var_list).todense()
            Q_desired[0, 0] = 0.0
            Q_desired = Q_desired / np.linalg.norm(Q_desired, ord="fro")
            Q_desired[0, 0] = 1.0
            # Test
            np.testing.assert_allclose(Q_torch[b], Q_desired, rtol=1e-7, atol=1e-7)

    def test_forward_scs(t):
        t.run_forward_sdpr(solver="SCS")

    def test_forward_mosek(t):
        t.run_forward_sdpr(solver="mosek")

    def run_forward_sdpr(t, solver="SCS", N_batch=3):
        # Generate problem
        set_seed(0)
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=50, N_batch=3
        )

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

        # get the constraints
        constraints_list = st.get_constraints(r_ps, C_p0s, r_ls)
        # Build Layer
        sdpr_layer = SDPRLayer(13, constraints=constraints_list, use_dual=True)
        # Run Forward pass
        Q = st.get_data_mat(cam_torch, r_ls, pixel_meass)
        # select solver
        if solver == "mosek":
            solver_args = {"solve_method": "mosek", "verbose": True}
        elif solver == "SCS":
            solver_args = {"solve_method": "SCS", "eps": 1e-9, "verbose": True}
        X = sdpr_layer(Q, solver_args=solver_args)[0]
        # Check solutions
        for b in range(N_batch):
            # Get the solution
            X_b = X[b].detach().numpy()
            # Make sure it is rank-1
            assert np.linalg.matrix_rank(X_b, tol=1e-6) == 1
            # Extract solution
            r_inP = X_b[10:, [0]]
            C_vec = X_b[1:10, [0]]
            C = C_vec.reshape((3, 3), order="F")
            # Check the error
            if t.no_noise:
                np.testing.assert_allclose(C.T @ r_inP, r_ps[b], atol=1e-7)
                np.testing.assert_allclose(C.T @ C_p0s[b], np.eye(3), atol=1e-7)
            else:
                np.testing.assert_allclose(C.T @ r_inP, r_ps[b], atol=5e-3)
                np.testing.assert_allclose(C.T @ C_p0s[b], np.eye(3), atol=5e-3)

    def test_fwd_pypose(t, N_map=20, N_batch=1):
        """Test forward pass of theseus layer"""
        set_seed(0)
        # Generate problem
        r_p0s, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=N_map, N_batch=N_batch
        )

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
        # Initialize module
        reg = st.MWRegistration(r_p0s, C_p0s, r_ls)
        # invert the camera measurements
        pixel_meass = torch.tensor(pixel_meass)
        meas, weights = cam_torch.inverse(pixel_meass)
        # Run optimization
        st.run_pypose_opt(reg, meas, weights)

        # Check the error
        C_p0_est = reg.T_p0s.rotation().matrix().detach().numpy()
        r_0p_p_est = reg.T_p0s.translation().detach().numpy()

        if t.no_noise:
            atol_r = 1e-7
            atol_c = 1e-7
        else:
            atol_r = 7e-3
            atol_c = 7e-3
        for b in range(N_batch):
            np.testing.assert_allclose(
                r_0p_p_est[b], (-C_p0s[b] @ r_p0s[b]).squeeze(-1), atol=atol_r
            )
            np.testing.assert_allclose(C_p0_est[b].T @ C_p0s[b], np.eye(3), atol=atol_c)

    def test_fwd_theseus(t, N_map=20, N_batch=5):
        """Test forward pass of theseus layer"""
        set_seed(0)
        # Generate problem
        r_p0s, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=N_map, N_batch=N_batch
        )

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

        # Build layer
        theseus_layer = st.build_theseus_layer(cam_torch, N_map=N_map, N_batch=N_batch)
        # invert the camera measurements
        pixel_meass = torch.tensor(pixel_meass)
        meas, weights = cam_torch.inverse(pixel_meass)
        # Run Forward pass
        r_p0s = r_p0s.squeeze(2)
        theseus_inputs = {
            "C_p0s": torch.tensor(C_p0s),
            "r_p0s": torch.tensor(r_p0s),
            "r_ls": torch.tensor(r_ls),
            "meas": meas,
            "weights": weights,
        }
        updated_inputs, info = theseus_layer.forward(
            theseus_inputs,
            optimizer_kwargs={"track_best_solution": True, "verbose": True},
        )
        C_est = updated_inputs["C_p0s"]
        r_p_est = updated_inputs["r_p0s"]

        # Check the error
        if t.no_noise:
            atol_r = 1e-7
            atol_c = 1e-7
        else:
            atol_r = 7e-3
            atol_c = 7e-3
        for b in range(N_batch):
            r_p0_est_d = r_p_est[b].detach().numpy()
            C_p0_est_d = C_est[b].detach().numpy()
            np.testing.assert_allclose(r_p0_est_d, r_p0s[b], atol=atol_r)
            np.testing.assert_allclose(C_p0_est_d.T @ C_p0s[b], np.eye(3), atol=atol_c)

    def test_grads_theseus(t, N_map=20, N_batch=1):
        """Test backward pass of theseus layer"""
        set_seed(0)
        # Generate problem
        r_p0s, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=N_map, N_batch=N_batch
        )
        r_p0s = r_p0s.squeeze(2)
        pixel_meass = torch.tensor(pixel_meass)

        # opt parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # Build layer
        theseus_layer = st.build_theseus_layer(N_map=N_map, N_batch=N_batch)

        def theseus_wrapper(*params, out="r"):
            # Use params to create camera model
            cam_torch = st.Camera(
                *params, sigma_u=t.cam_gt.sigma_u, sigma_v=t.cam_gt.sigma_v
            )
            # invert the camera measurements
            meas, weights = cam_torch.inverse(pixel_meass)
            # Run Forward pass
            theseus_inputs = {
                "C_p0s": torch.tensor(C_p0s),
                "r_p0s": torch.tensor(r_p0s),
                "r_ls": torch.tensor(r_ls),
                "meas": meas,
                "weights": weights,
            }
            updated_inputs, info = theseus_layer.forward(
                theseus_inputs,
                optimizer_kwargs={
                    "track_best_solution": True,
                    "verbose": True,
                    "backward_mode": "implicit",
                },
            )
            if out == "r":
                return updated_inputs["r_p0s"]
            elif out == "C":
                return updated_inputs["C_p0s"]

        # Test backward
        eps = 1e-8
        atol_r = 1e-4
        atol_c = 1e-4
        torch.autograd.gradcheck(
            lambda *x: theseus_wrapper(*x, out="r"),
            inputs=params,
            eps=eps,
            atol=atol_r,
        )
        torch.autograd.gradcheck(
            lambda *x: theseus_wrapper(*x, out="C"),
            inputs=params,
            eps=eps,
            atol=atol_c,
        )

    def test_grads_camera(t, N_batch=1):
        """Test gradients for camera parameters

        Args:
            t (_type_): _description_
            N_batch (int, optional): _description_. Defaults to 1.

        Returns:
            _type_: _description_
        """
        set_seed(0)
        # Generate problem
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=7, N_batch=N_batch
        )
        pixel_meass_tch = torch.tensor(pixel_meass)
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

            meas, weight = cam.inverse(pixel_meass_tch)
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
            meas, weight = cam.inverse(pixel_meass_tch)
            return weight

        torch.autograd.gradcheck(
            inverse_wrapper_weight,
            inputs=params,
            eps=1e-4,
            atol=1e-4,
            rtol=1e-4,
        )

    def test_grads_data_mat(t, N_batch=2):
        set_seed(0)
        # Generate problem
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=7, N_batch=N_batch
        )
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
            return st.get_data_mat(cam, r_ls, pixel_meass)

        torch.autograd.gradcheck(
            data_mat_wrapper,
            inputs=params,
            eps=1e-5,
            atol=1e-5,
            rtol=1e-5,
        )

    def test_grads_optlayer_scs(t):
        t.check_grads_optlayer(solver="SCS")

    def test_grads_optlayer_mosek(t):
        t.check_grads_optlayer(solver="mosek")

    def check_grads_optlayer(t, solver="SCS", N_batch=1):
        set_seed(0)
        # Generate problem
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=7, N_batch=N_batch
        )

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # TEST OPT LAYER
        constraints_list = st.get_constraints(r_ps, C_p0s, r_ls)
        # Build opt layer
        sdpr_layer = SDPRLayer(13, constraints=constraints_list, use_dual=False)
        sdpr_layer_dual = SDPRLayer(13, constraints=constraints_list, use_dual=True)

        # Generate solution X from parameters
        def get_outputs_from_params(*params, solver, out="loss", verbose=True):
            cam = st.Camera(*params)
            Q = st.get_data_mat(cam, r_ls, pixel_meass)
            # Run Forward pass
            if solver == "mosek":
                # Run Forward pass
                mosek_params = {
                    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
                    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-14,
                    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
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
            if out == "loss":
                out = st.get_loss_from_sol(X[0], r_ps[0], C_p0s[0])
            elif out == "sol":
                # Extract solution (assume Rank-1)
                r = (X[0, 10:, [0]] + X[0, [0], 10:].T) / 2.0
                C_vec = (X[0, 1:10, [0]] + X[0, [0], 1:10].T) / 2.0
                out = r, C_vec
            else:
                raise ValueError("Invalid output type")
            return out

        # Solution gradient check
        torch.autograd.gradcheck(
            lambda *x: get_outputs_from_params(*x, solver=solver, out="sol"),
            params,
            eps=1e-5,
            atol=1e-6,
        )

        # Loss gradient check
        torch.autograd.gradcheck(
            lambda *x: get_outputs_from_params(*x, solver=solver, out="loss"),
            params,
            eps=1e-5,
            atol=1e-3,
            rtol=1e-3,
        )

    def test_compare_grads(t, N_batch=1):
        set_seed(0)
        # Generate problem
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=7, N_batch=N_batch
        )

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # TEST OPT LAYER
        constraints_list = st.get_constraints(r_ps, C_p0s, r_ls)
        # Build opt layer
        sdpr_layer = SDPRLayer(13, constraints=constraints_list, use_dual=False)
        sdpr_layer_dual = SDPRLayer(13, constraints=constraints_list, use_dual=True)

        # Test gradients to matrix
        cam = st.Camera(*params)
        Q = st.get_data_mat(cam, r_ls, pixel_meass)

        # Generate solution X from parameters
        def get_X_from_params(*params, solver, verbose=True):
            cam = st.Camera(*params)
            Q = st.get_data_mat(cam, r_ls, pixel_meass)
            # Run Forward pass
            if solver == "mosek":
                # Run Forward pass
                mosek_params = {
                    "MSK_IPAR_INTPNT_MAX_ITERATIONS": 1000,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-12,
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-12,
                    "MSK_DPAR_INTPNT_CO_TOL_MU_RED": 1e-14,
                    "MSK_DPAR_INTPNT_CO_TOL_INFEAS": 1e-12,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-12,
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

        # Make sure solutions match
        X0_scs = get_X_from_params(*params, solver="SCS")[0]
        X0_msk = get_X_from_params(*params, solver="mosek")[0]
        np.testing.assert_allclose(
            X0_scs.detach().numpy(), X0_msk.detach().numpy(), atol=1e-6
        )

        # Check gradients
        opt = torch.optim.Adam(params, lr=1e-3)
        opt.zero_grad()
        loss = st.get_loss_from_sol(X0_scs, r_ps[0], C_p0s[0])
        loss.backward()
        loss0_scs = loss.detach().numpy().copy()
        grads_scs = [p.grad.detach().numpy().copy() for p in params]
        opt.zero_grad()
        loss = st.get_loss_from_sol(X0_msk, r_ps[0], C_p0s[0])
        loss.backward()
        loss0_msk = loss.detach().numpy().copy()
        grads_msk = [p.grad.detach().numpy().copy() for p in params]
        np.testing.assert_allclose(grads_scs, grads_msk, atol=1e-6)
        opt.zero_grad()
        # Check diff in loss
        np.testing.assert_allclose(loss0_msk, loss0_scs, atol=1e-6)

        # Check effect of peturbation
        params[-1].data += 1e-4
        X1_scs = get_X_from_params(*params, solver="SCS")[0]
        X1_msk = get_X_from_params(*params, solver="mosek")[0]
        np.testing.assert_allclose(
            X1_scs.detach().numpy(), X1_msk.detach().numpy(), atol=1e-6
        )

        # Check difference in loss
        loss = st.get_loss_from_sol(X1_scs, r_ps[0], C_p0s[0])
        loss1_scs = loss.detach().numpy().copy()
        loss = st.get_loss_from_sol(X1_msk, r_ps[0], C_p0s[0])
        loss1_msk = loss.detach().numpy().copy()
        np.testing.assert_allclose(
            loss1_msk - loss0_msk, loss1_scs - loss0_scs, atol=1e-6
        )

    def test_grad_loss(t):
        """Test gradients of the output loss function"""
        set_seed(0)
        # Generate problem
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(camera=t.cam_gt, N_map=7)

        # Camera parameters
        params = (
            torch.tensor(t.cam_gt.f_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.f_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_u, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.c_v, requires_grad=True, dtype=torch.float64),
            torch.tensor(t.cam_gt.b, requires_grad=True, dtype=torch.float64),
        )

        # TEST OPT LAYER
        constraints_list = st.get_constraints(r_ps, C_p0s, r_ls)
        # Build opt layer
        sdpr_layer = SDPRLayer(13, constraints=constraints_list, use_dual=False)
        sdpr_layer_dual = SDPRLayer(13, constraints=constraints_list, use_dual=True)

        # Test gradients to matrix
        cam = st.Camera(*params)
        Q = st.get_data_mat(cam, r_ls, pixel_meass)
        solver_args = {"solve_method": "SCS", "eps": 1e-9}
        X = sdpr_layer(Q, solver_args=solver_args)[0]
        x = X[0][:, [1]]

        # define loss function based on vector solution
        def get_loss_vec(x):
            X_new = x @ x.T
            return st.get_loss_from_sol(X_new, r_ps[0], C_p0s[0])

        torch.autograd.gradcheck(
            get_loss_vec,
            x,
            eps=1e-6,
            atol=1e-6,
            rtol=1e-6,
        )

    # INTEGRATION TESTS

    def test_tune_params_no_opt(t, plot=False, optim="Adam", N_map=20, N_batch=1):
        """Tune all parameters without using the optimization layer. That is,
        the ground truth is used to map the landmarks into the camera frame and
        the loss on landmark locations is used to tune the camera parameters."""
        set_seed(0)
        # Generate problems
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=N_map, N_batch=N_batch
        )
        # dictionary of paramter test values
        param_dict = {
            "f_u": dict(
                offs=50,
                atol=5,
                atol_nonoise=1e-5,
            ),
            "f_v": dict(offs=50, atol=5, atol_nonoise=1e-5),
            "c_u": dict(offs=50, atol=5, atol_nonoise=1e-5),
            "c_v": dict(offs=50, atol=5, atol_nonoise=1e-5),
            "b": dict(
                offs=0.1,
                atol=2e-3,
                atol_nonoise=2e-3,
            ),
        }

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
        # Loop through all parameters
        params = []
        for key, tune_params in param_dict.items():
            # Add offset to torch param
            getattr(cam_torch, key).data += tune_params["offs"]
            # Create optimizer
            params += [getattr(cam_torch, key)]
        if optim == "Adam":
            opt = torch.optim.Adam(params[:-1], lr=10)
            opt.add_param_group({"params": [params[-1]], "lr": 1e-1})
        elif optim == "LBFGS":
            opt = torch.optim.LBFGS(
                params,
                tolerance_change=1e-12,
                tolerance_grad=1e-12,
                lr=100,
                max_iter=1,
                line_search_fn="strong_wolfe",
            )
        elif optim == "SGD":
            opt = torch.optim.SGD(params[:-1], lr=1e-3)
            opt.add_param_group({"params": [params[-1]], "lr": 1e-4})
        # Termination criteria
        term_crit = {"max_iter": 2000, "tol_grad_sq": 1e-15, "tol_loss": 1e-12}
        # Run Tuner
        iter_info = st.tune_stereo_params_no_opt(
            cam_torch=cam_torch,
            params=params,
            opt=opt,
            term_crit=term_crit,
            r_ps=r_ps,
            C_p0s=C_p0s,
            r_ls=r_ls,
            pixel_meass=pixel_meass,
            verbose=True,
        )
        if plot:
            plt.figure()
            plt.plot(iter_info["loss"])
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.show()
        for key, tune_params in param_dict.items():
            if t.no_noise:
                np.testing.assert_allclose(
                    getattr(cam_torch, key).detach().numpy(),
                    getattr(t.cam_gt, key),
                    atol=tune_params["atol_nonoise"],
                )
            else:
                np.testing.assert_allclose(
                    getattr(cam_torch, key).detach().numpy(),
                    getattr(t.cam_gt, key),
                    atol=tune_params["atol"],
                )

    def test_tune_params_sep(t, tuner="sdpr", optim="LBFGS", plot=False):
        """Test tuning offsets on each parameter using theseus."""
        set_seed(0)
        # Generate problem
        r_p0s, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=20, N_batch=10
        )
        # Convert to tensor
        pixel_meass = torch.tensor(pixel_meass)

        # dictionary of paramter test values
        param_dict = {
            "f_u": dict(
                offs=1,
                lr=100e-2,
                tol_grad_sq=1e-15,
                atol=5,
                atol_nonoise=1e-3,
            ),
            "f_v": dict(offs=1, lr=5e-2, tol_grad_sq=1e-15, atol=5, atol_nonoise=1e-3),
            "c_u": dict(offs=1, lr=5e-2, tol_grad_sq=1e-15, atol=5, atol_nonoise=1e-3),
            "c_v": dict(offs=1, lr=5e-2, tol_grad_sq=1e-15, atol=5, atol_nonoise=1e-3),
            "b": dict(
                offs=0.01,
                lr=5e-3,
                tol_grad_sq=1e-10,
                atol=2e-3,
                atol_nonoise=1e-5,
            ),
        }

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
            # Add offset to torch param
            getattr(cam_torch, key).data += tune_params["offs"]
            # Define parameter and learning rate
            params = [getattr(cam_torch, key)]
            if optim == "Adam":
                opt = torch.optim.Adam(params, lr=tune_params["lr"])
            elif optim == "LBFGS":
                if key == "b":
                    lr = 1e-1
                else:
                    lr = 10
                opt = torch.optim.LBFGS(
                    params,
                    history_size=50,
                    tolerance_change=1e-16,
                    tolerance_grad=1e-16,
                    lr=lr,
                    max_iter=1,
                    line_search_fn="strong_wolfe",
                )
            # Termination criteria
            term_crit = {
                "max_iter": 500,
                "tol_grad_sq": 1e-14,
                "tol_loss": 1e-10,
            }
            # Run Tuner
            if tuner == "sdpr":
                # Run Tuner
                iter_info = st.tune_stereo_params_sdpr(
                    cam_torch=cam_torch,
                    params=params,
                    opt=opt,
                    term_crit=term_crit,
                    r_p0s=r_p0s,
                    C_p0s=C_p0s,
                    r_ls=r_ls,
                    pixel_meass=pixel_meass,
                    verbose=True,
                )
            elif tuner == "theseus":
                iter_info = st.tune_stereo_params_theseus(
                    cam_torch=cam_torch,
                    params=params,
                    opt=opt,
                    term_crit=term_crit,
                    r_p0s_gt=r_p0s,
                    C_p0s_gt=C_p0s,
                    r_ls=r_ls,
                    pixel_meass=pixel_meass,
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
            if t.no_noise:
                np.testing.assert_allclose(
                    getattr(cam_torch, key).detach().numpy(),
                    getattr(t.cam_gt, key),
                    atol=tune_params["atol_nonoise"],
                )
            else:
                np.testing.assert_allclose(
                    getattr(cam_torch, key).detach().numpy(),
                    getattr(t.cam_gt, key),
                    atol=tune_params["atol"],
                )

    def test_tune_params(
        t, plot=True, tuner="sdpr", optim="Adam", N_map=30, N_batch=10
    ):
        """Test offsets for all parameters simultaneously. Use default noise level"""
        set_seed(1)
        # Generate problems
        r_ps, C_p0s, r_ls, pixel_meass = st.get_prob_data(
            camera=t.cam_gt, N_map=N_map, N_batch=N_batch
        )

        # dictionary of paramter test values
        param_dict = {
            "f_u": dict(
                offs=10,
                atol=3,
                atol_nonoise=1e-2,
            ),
            "f_v": dict(offs=10, atol=3, atol_nonoise=1e-2),
            "c_u": dict(offs=10, atol=3, atol_nonoise=1e-2),
            "c_v": dict(offs=10, atol=3, atol_nonoise=1e-2),
            "b": dict(
                offs=0.1,
                atol=5e-3,
                atol_nonoise=2e-3,
            ),
        }

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
        # Loop through all parameters
        params = []
        for key, tune_params in param_dict.items():
            # Add offset to torch param
            getattr(cam_torch, key).data += tune_params["offs"]
            # Create optimizer
            params += [getattr(cam_torch, key)]
            # Set up optimizer
            if optim == "Adam":
                opt = torch.optim.Adam(params[:-1], lr=5)
                opt.add_param_group({"params": [params[-1]], "lr": 1e-1})
            elif optim == "LBFGS":
                if key == "b":
                    lr = 1
                else:
                    lr = 10
                opt = torch.optim.LBFGS(
                    params,
                    history_size=50,
                    tolerance_change=1e-16,
                    tolerance_grad=1e-16,
                    lr=lr,
                    max_iter=1,
                    line_search_fn="strong_wolfe",
                )
            elif optim == "SGD":
                opt = torch.optim.SGD(params[:-1], lr=1e-3)
                opt.add_param_group({"params": [params[-1]], "lr": 1e-4})
        # Termination criteria
        term_crit = {"max_iter": 4000, "tol_grad_sq": 1e-14, "tol_loss": 1e-12}

        # Run Tuner
        if tuner == "sdpr":
            iter_info = st.tune_stereo_params_sdpr(
                cam_torch=cam_torch,
                params=params,
                opt=opt,
                term_crit=term_crit,
                r_ps=r_ps,
                C_p0s=C_p0s,
                r_ls=r_ls,
                pixel_meass=pixel_meass,
                verbose=True,
                solver="mosek",
            )
        elif tuner == "theseus":
            iter_info = st.tune_stereo_params_theseus(
                cam_torch=cam_torch,
                params=params,
                opt=opt,
                term_crit=term_crit,
                r_p=r_ps,
                C_p0=C_p0s,
                r_l=r_ls,
                pixel_meas=pixel_meass,
                verbose=True,
            )
        if plot:
            plt.figure()
            plt.plot(iter_info["loss"])
            plt.ylabel("Loss")
            plt.xlabel("Iteration")
            plt.show()
        for key, tune_params in param_dict.items():
            if t.no_noise:
                np.testing.assert_allclose(
                    getattr(cam_torch, key).detach().numpy(),
                    getattr(t.cam_gt, key),
                    atol=tune_params["atol_nonoise"],
                )
            else:
                np.testing.assert_allclose(
                    getattr(cam_torch, key).detach().numpy(),
                    getattr(t.cam_gt, key),
                    atol=tune_params["atol"],
                )


def plot_poses(R_cw, t_cw_w, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    for i in range(len(R_cw)):
        origin = t_cw_w[i]
        directions = R_cw[i].T

        for j in range(3):
            ax.quiver(*origin, *directions[:, j], color=["r", "g", "b"][j])


def plot_map(r_l, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*r_l, ".", color="k")


if __name__ == "__main__":
    # unittest.main()
    test = TestStereoTune(no_noise=False)
    test.test_tune_params_sep(plot=False)
