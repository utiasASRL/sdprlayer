from typing import Any
import matplotlib.pylab as plt
import numpy as np
import torch
from torch.profiler import record_function


class PolyMinLayer(torch.nn.Module):
    """Layer that minimizes a polynomial and outputs the minimum value.
    Backpropagation is performed via the implicit function theorem.
    It is assumed that the polynomial has the lowest degree first."""

    def __init__(self, opt_params):
        super(PolyMinLayer, self).__init__()
        # store optimizer parameters
        self.opt_params = opt_params

    def forward(self, poly, x_init):
        return PolyMinLayerFn.apply(poly, x_init, self.opt_params)


def polyval(poly, x):
    # evaluate polynomial
    eval = torch.zeros_like(x)
    for i in range(poly.shape[0]):
        eval += poly[i] * torch.pow(x, i)
    return eval


class PolyMinLayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, poly, x_init: float, opt_params={}):
        """Minimize polynomial specified by poly with respect to x_init"""
        assert poly.ndim == 1, "Only one polynomial supported. No batching allowed"
        assert poly.shape[0] > 2, "Polynomial must be of degree at least 2"
        # update opt_params
        opt_params_default = dict(
            optimizer="lbfgs", lr=0.1, grad_sq_tol=1e-12, max_iter=100, verbose=False
        )
        opt_params_default.update(opt_params)
        opt_params = opt_params_default
        # init variable
        x_min = torch.tensor(x_init, requires_grad=True)
        # define optimizer
        if opt_params["optimizer"] == "sgd":
            opt = torch.optim.SGD(params=[x_min], lr=opt_params["lr"])
        elif opt_params["optimizer"] == "lbfgs":
            opt = torch.optim.LBFGS(params=[x_min], max_iter=1)
        # Detach the poly coefficients to avoid accidental grad
        # computation
        poly = poly.detach()

        # define closure
        def closure():
            opt.zero_grad()
            pval = polyval(poly, x_min)
            pval.backward()
            return pval

        # Inner optimization loop
        pvals = []
        grad_sq = np.inf
        n_iter = 0
        while grad_sq > opt_params["grad_sq_tol"] and n_iter < opt_params["max_iter"]:
            pvals += [opt.step(closure)]
            grad_sq = torch.sum(x_min.grad**2)
            n_iter += 1
            if opt_params["verbose"]:
                print(
                    f"n_iter:\t{n_iter}\tgrad^2:\t{grad_sq}\tmin:\t{x_min}\tloss:\t{pvals[-1]}"
                )

        # Save values for backpropagation
        ctx.save_for_backward(poly, x_min)

        return x_min

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Backpropagation via implicit function theorem"""
        # get saved tensors
        poly, x_min = ctx.saved_tensors
        # compute hessian of polynomial wrt x
        nab_xx = torch.zeros_like(x_min)
        for i in range(2, poly.shape[0]):
            nab_xx += poly[i] * torch.pow(x_min, i - 2) * i * (i - 1)
        # Compute gradient wrt p
        nab_xp = torch.zeros_like(poly)
        for i in range(1, poly.shape[0]):
            nab_xp[i] = torch.pow(x_min, i - 1) * i
        # Divide by hessian (implicit function theorem)
        grad = -nab_xp / nab_xx * grad_outputs[0]

        return grad, None, None
