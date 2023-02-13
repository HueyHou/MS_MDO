import numpy as np
from modopt.api import Problem

class MyProblem(Problem):
    def initialize(self, ):
        self.problem_name = 'three-hump camel'
        self.options.declare('dtype', default=float)

    def setup(self):
        self.add_design_variables('x',
                                  shape=(2, ),
                                  lower=None,
                                  upper=None,
                                  equals=None,
                                  vals=np.array([.5, .5], dtype = self.options['dtype']))

        self.add_objective('f')

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', vals=None)
        self.declare_objective_hessian(of='x', wrt='x', vals=None)

    def compute_objective(self, dvs, obj):
        x1 = dvs['x'][0]
        x2 = dvs['x'][1]
        obj['f'] = 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2

    def compute_objective_gradient(self, dvs, grad):
        x1 = dvs['x'][0]
        x2 = dvs['x'][1]
        grad['x'] = np.array([
            4*x1 - 4.2*x1**3 + x1**5 + x2,
            x1 + 2*x2
        ])
    # directly write its gradient

    # def compute_objective_gradient(self, dvs, grad):
    #     h = 1e-8
    #     for i in range(2):
    #         e = np.zeros((2), dtype=complex)
    #         e[i] = 1j * h
    #         xx = np.zeros((2), dtype=complex)
    #         xx = dvs['x'] + e
    #         x1 = xx[0]
    #         x2 = xx[1]
    #         c = 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2
    #         cim = np.imag(c)
    #         grad['x'][i] = cim / h
    # # using complex-step method

    def compute_objective_hessian(self, dvs, hess):
        x1 = dvs['x'][0]
        x2 = dvs['x'][1]
        hess['x', 'x'] = np.array([[4-12.6*x1**2+5*x1, 1],
                         [1, 2]])

