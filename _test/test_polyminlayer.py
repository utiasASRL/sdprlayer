import unittest

import matplotlib.pylab as plt
import numpy as np
import torch

from sdprlayer import PolyMinLayer

torch.set_default_dtype(torch.float64)


class TestPolyMin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPolyMin, self).__init__(*args, **kwargs)
        self.poly = np.array(
            [5.0000, 1.3167 * 2, -1.4481 * 3, 0 * 4, 0.2685 * 3, -0.0667 * 2, 0.0389]
        )
        # self.poly = np.array([0.0, 0.0, 1.0])
        self.opt_params = dict(
            optimizer="sgd",
            lr=1e-3,
            grad_sq_tol=1e-10,
            max_iter=10000,
            verbose=True,
        )

    def plot_polynomial(self):
        x = np.linspace(-2.5, 2.5, 100)
        y = np.polyval(self.poly[::-1], x)
        plt.plot(x, y)

    def test_forward(self, plot=False):
        # Init values
        x_init = [-2.0, 5.0]
        poly = torch.tensor(self.poly)
        # define layer and run local optimization
        layer = PolyMinLayer(opt_params=self.opt_params)
        with torch.no_grad():
            x_min_1 = layer.forward(poly, x_init[0])
            x_min_2 = layer.forward(poly, x_init[1])

        if plot:
            plt.figure()
            self.plot_polynomial()
            plt.plot(x_min_1, np.polyval(self.poly[::-1], x_min_1), "ro")
            plt.plot(x_min_2, np.polyval(self.poly[::-1], x_min_2), "bo")
            plt.show()

        assert np.allclose(
            x_min_1, -1.4871, atol=1e-4
        ), f"Local minimum is {x_min_2} but should be -1.4871"
        assert np.allclose(
            x_min_2, 1.5996, atol=1e-4
        ), f"Local minimum is {x_min_2} but should be 1.59960"

    def test_backward(self):
        # Init values
        poly = torch.tensor(self.poly, requires_grad=True)
        # define layer and run local optimization
        layer = PolyMinLayer(opt_params=self.opt_params)

        def layer_wrapper(poly):
            return layer(poly, x_init)

        x_init = -2.0
        torch.autograd.gradcheck(layer_wrapper, inputs=[poly], eps=1e-6, atol=1e-5)
        x_init = 5.0
        torch.autograd.gradcheck(layer_wrapper, inputs=[poly], eps=1e-6, atol=1e-5)

    def test_integrated(self):
        # Init values
        poly = torch.tensor(self.poly, requires_grad=True)
        # define layer and run local optimization
        layer = PolyMinLayer(opt_params=self.opt_params)

        def loss_fcn(poly):
            x_min = layer(poly, x_init)
            loss = 1 / 2 * (x_min - 0.0) ** 2
            return loss

        x_init = -2.0
        torch.autograd.gradcheck(loss_fcn, inputs=[poly], eps=1e-6, atol=1e-5)
        x_init = 5.0
        torch.autograd.gradcheck(loss_fcn, inputs=[poly], eps=1e-6, atol=1e-5)


if __name__ == "__main__":
    test = TestPolyMin()
    # test.test_forward(plot=True)
    # test.test_backward(verbose=False)
    test.test_integrated()
