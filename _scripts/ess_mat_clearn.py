#!/usr/bin/env python3
import os
import pickle
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import spatialmath.base as sm

# from solvers.common import solve_sdp_cvxpy
from lifters.state_lifter import StateLifter
from poly_matrix import PolyMatrix
from pylgmath.so3.operations import hat

from sdprlayers.layers.essential_est import EssentialSDPBlock

root_dir = os.path.abspath(os.path.dirname(__file__) + "/../")
sys.path.insert(0, root_dir)
fig_folder = os.path.join(root_dir, "figs")


class EssentialLifter(StateLifter):
    EPS_SVD = 1e-10
    EPS_SPARSE = 1e-9

    def __init__(self, formulation="fro_norm"):
        # Initialize with just the standard shape of our var list
        super().__init__()
        # Update internal variable dictionary.
        self.var_dict_ = dict(h=1, e1=3, e2=3, e3=3, t=3)
        self.formulation = formulation

    def get_Q(self, noise=1e-3, output_poly=False):
        return None

    def get_p(self, parameters=None, var_subset=None) -> np.ndarray:
        """No parameters defined for this problem"""
        return [1.0]

    def get_x(self, theta=None, parameters=None, var_subset=None) -> np.ndarray:
        """Construct a random feasible vector"""
        # Random Translation
        t_ts = 50 * np.random.rand(3, 1)
        # Random Rotation - keep perturbation small so that z-axis is
        # is consistent across views
        axis = np.random.rand(3, 1)
        axis /= np.linalg.norm(axis)
        angle = 2 * np.pi * np.random.rand()
        R_ts = sm.angvec2r(angle, axis)
        # Get essential matrix
        t_ts = t_ts / np.linalg.norm(t_ts)
        E = hat(t_ts) @ R_ts.T
        vecE = np.reshape(E, (9, 1))
        # construct QCQP feasible point
        feas_pt = np.vstack([np.array([[1]]), vecE, t_ts])
        return feas_pt

    def sample_theta(self):
        """We do not need to define this for this problem."""
        pass


def gen_learned_constraints(plot=False):
    """Generate the learned set of constraints"""
    # Define lifter
    lifter = EssentialLifter()
    # Define list of known constraints
    A_known = []
    layer = EssentialSDPBlock()
    A_known += layer.get_t_norm_constraint()
    A_known += layer.get_tcross_constraints()

    # Get learned constraints
    A_learned = lifter.get_A_learned(
        A_known=A_known, var_dict=lifter.var_dict_, verbose=True
    )
    for A in A_learned:
        read_sparse_mat(A, lifter.var_dict_, show=plot)
    # Save the constraints
    print("Learned Constraints Verified...Saving to file")
    date = datetime.today().strftime("%Y-%m-%d")
    filename = "ess_saved_constraints_" + date + ".pkl"
    with open(filename, "wb") as handle:
        data = {"constraints": A_learned}
        pickle.dump(data, handle)


def load_constraints(filename="saved_constraints_2024-08-13.pkl"):
    """Load learned constraints from file"""
    print("Loading Constraints from " + filename)
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
        return data["constraints"]

    # def get_learned_constraint(iConst=1):
    #     """Load saved constraints and sort them."""
    #     # Init SLAM problem
    #     slam = init_stereo_prob(trans_frame="local")
    #     constraints = [
    #         (c.A.get_matrix(slam.var_list, output_type="csr"), c.b)
    #         for c in slam.constraints
    #     ]
    #     n_constraints = len(constraints)
    #     # Load learned constraints
    #     A_learned = load_constraints()
    #     constraints += [(A, 0.0) for A in A_learned]
    #     # Get cost matrix
    #     Q, offset, scale = slam.get_norm_cost_mat()
    #     # Solve SDP
    #     X, info = solve_sdp_mosek(
    #         Q, Constraints=constraints, adjust=(scale, offset), verbose=True
    #     )
    #     # Get and sort lagrange multipliers
    #     y_learned = np.abs(info["yvals"][n_constraints:])
    #     ind = np.argsort(y_learned)[::-1]
    #     y_learned = y_learned[ind]
    #     # Sort the constraints according to their influence
    #     A_learned = np.array(A_learned)[ind]
    #     # print different constraints
    #     # for iConst in range(5, 20):
    #     #     print("Constraint " + str(iConst))
    #     #     read_sparse_mat(A_learned[iConst], slam.var_list, show=False)

    # print("Constraint " + str(iConst))
    # newdict = convert_labels(slam.var_list)
    # ax = read_sparse_mat(A_learned[iConst], newdict, homog="w", show=True)
    # savefig(plt.gcf(), os.path.join(fig_folder, "constraint_ex.png"))


def read_sparse_mat(A, var_dict, show=False, homog="w_0"):
    """Plot a polymatrix based on a sparse representation and a variable dictionary

    Args:
        A (_type_): _description_
        var_dict (_type_): _description_
    """
    A_poly, _ = PolyMatrix.init_from_sparse(A, var_dict, symmetric=True)
    print(A_poly.get_expr(homog=homog))
    if show:
        ax = A_poly.matshow()
    else:
        ax = None
    return ax


if __name__ == "__main__":
    gen_learned_constraints()
