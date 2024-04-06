import numpy as np
import torch

from sdprlayer.sdprlayer import SDPRLayer

EPS_SDP = 1e-9


def run_calibration(prob, constraints, verbose=False, init_noise=1e-3, plots=False):
    """Make sure that we converge to the (almost) perfect biases when using
    (almost) perfect distances.
    """
    # Create SDPR Layer
    optlayer = SDPRLayer(n_vars=constraints[0].shape[0], constraints=constraints)

    # Set up polynomial parameter tensor
    values = prob.biases + np.random.normal(
        scale=init_noise, loc=0, size=prob.biases.shape
    )
    p = torch.tensor(values, requires_grad=True)

    # Define loss
    def gen_loss(values, **kwargs):
        sdp_solver_args = {"eps": EPS_SDP}
        X, x = optlayer(prob.build_data_mat(values), solver_args=sdp_solver_args)
        eigs, __ = torch._linalg_eigh(X, compute_v=False)
        evr = eigs[-1] / eigs[-2]
        assert evr > 1e6
        positions = prob.get_positions(x)
        loss = torch.norm(positions - torch.tensor(prob.positions))
        return loss, positions

    # opt = torch.optim.Adam(params=[p], lr=1e-2, eps=1e-10, weight_decay=0.2)
    opt = torch.optim.Adam(params=[p], lr=1e-3)

    # Execute iterations
    losses = []
    max_iter = 2000
    grad_norm_tol = 1e-7
    p_grad_norm = np.inf
    converged = False
    print("target biases:", prob.biases)
    print("target loss:", gen_loss(prob.biases)[0])
    if plots:
        fig, (ax_loss, ax_err) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)

        fig_pos, ax_pos = plt.subplots()
        ax_pos.scatter(*prob.positions[:, : prob.d].T, color="k")
        ax_pos.scatter(*prob.anchors[:, : prob.d].T, marker="x", color="k")
        plt.show(block=False)
    for n_iter in range(max_iter):
        # Update Loss
        opt.zero_grad()
        loss, positions = gen_loss(p)

        # run optimizer
        loss.backward(retain_graph=True)
        p_grad_norm = p.grad.norm(p=2)
        biases = p.detach().numpy().round(3)
        if p_grad_norm < grad_norm_tol:
            msg = f"converged in grad after {n_iter} iterations."
            converged = True
            break
        opt.step()
        losses.append(loss.item())
        if verbose and ((n_iter < 10) or (n_iter % 10 == 0) or converged):
            print(
                f"{n_iter}: biases: {biases}\tgrad norm: {p_grad_norm:.2e}\tloss: {losses[-1]:.2e}"
            )

        if plots:
            errors = np.abs(biases - prob.biases)
            [
                ax_err.scatter(n_iter, b, color=f"C{i}", s=1)
                for i, b in enumerate(errors)
            ]
            ax_loss.scatter(n_iter, loss.item(), color="k", s=1)
            ax_pos.scatter(*positions[:, : prob.d].detach().numpy().T)
        if n_iter % 10 == 0:
            continue
    if not converged:
        msg = f"did not converge in {n_iter} iterations"
    return biases
