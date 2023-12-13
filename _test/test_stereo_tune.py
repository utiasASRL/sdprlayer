from contextlib import AbstractContextManager
from typing import Any
import numpy as np
import torch
import unittest
import matplotlib.pyplot as plt
import sdprlayer.stereo_tuner as st
from mwcerts.stereo_problems import Localization
from sdprlayer import SDPRLayer


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

    def test_forward(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)
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
        r = X[10:, [0]]
        C_vec = X[1:10, [0]]
        C = C_vec.reshape((3, 3))
        # Check the error
        np.testing.assert_allclose(C.T @ r, r_p[0], rtol=1e-6)
        np.testing.assert_allclose(C.T @ C_p0[0], np.eye(3), rtol=1e-6)

    def test_grad(t):
        # Generate problem
        np.random.seed(0)
        torch.manual_seed(0)
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

        # Define a localization class to get the constraints
        prob = Localization(r_p, C_p0, r_l)
        prob.generate_constraints()
        prob.generate_redun_constraints()
        constraints = prob.constraints + prob.constraints_r

        constraints_list = [(c.A.get_matrix(prob.var_list), c.b) for c in constraints]

        # Build Layer
        sdpr_layer = SDPRLayer(13, Constraints=constraints_list, use_dual=True)
        # build loss

        def get_loss_from_b(b):
            cam_torch.b = b
            Q = st.get_data_mat(cam_torch, r_l, pixel_meas)
            solver_args = {"solve_method": "mosek", "verbose": True}
            X = sdpr_layer(Q, solver_args=solver_args)[0]
            loss = st.get_loss(X, r_p, C_p0)
            return loss

        res = torch.autograd.gradcheck(
            get_loss_from_b,
            [cam_torch.b],
            eps=1e-4,
            atol=1e-4,
            rtol=1e-3,
        )
        assert res is True

    def test_gt_init(t):
        """test the case where we start at the ground truth."""

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

        # Run Tuner
        st.tune_stereo_params(
            cam_torch=cam_torch, r_p=r_p, C_p0=C_p0, r_l=r_l, pixel_meas=pixel_meas
        )


if __name__ == "__main__":
    # unittest.main()
    test = TestStereoTune()
    # test.test_data_matrix_no_noise()
    test.test_data_matrix()
    # test.test_grad()
    # test.test_forward()
    # test.test_gt_init()