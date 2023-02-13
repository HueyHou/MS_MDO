import numpy as np

from modopt.api import Problem

class Rosenbrock(Problem):
    def initialize(self, ):
        # Name your problem
        self.problem_name = 'rosenbrock'

    def setup(self):
        # Add design variables of your problem
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([.3, .3]))
        self.add_objective('f')

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', )
        self.declare_objective_hessian(wrt='x', of='x')

    def compute_objective(self, dvs, obj):
        obj['f'] = (1 - dvs['x'][0])**2 + 100 * (dvs['x'][1] - dvs['x'][0]**2)**2

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = np.array([
            -400 * dvs['x'][0] * (dvs['x'][1] - dvs['x'][0]**2) + 2 * (dvs['x'][0] - 1),
            200 * (dvs['x'][1] - dvs['x'][0]**2)
        ])

    def compute_objective_hessian(self, dvs, hess):
        hess['x'] = np.array([[2 - 400 * (dvs['x'][1] - 3 * dvs['x'][0]**2), -400 * dvs['x'][0]],
                            [-400 * dvs['x'][0], 200]])

    # def compute_objective_hvp(self, x, p):
    #     hess = np.array([[2 - 400 * (x[1] - 3 * x[0]**2), -400 * x[0]],
    #                      [-400 * x[0], 200]])
    #     hvp = np.matmul(hess, p)

    #     return hvp

