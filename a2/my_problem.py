import numpy as np
from modopt.api import Problem


class MyProblem(Problem):
    def initialize(self, ):
        # Name your problem
        self.problem_name = 'x^4'

    def setup(self):
        # Add design variables of your problem
        self.add_design_variables('x',
                                  shape=(2, ),
                                  vals=np.array([.3, .3]))
        self.add_objective('f')

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', )

    # Compute the value of the objective with given design variable values
    def compute_objective(self, dvs, obj):
        obj['f'] = np.sum(dvs['x']**4)

    def compute_objective_gradient(self, dvs, grad):
        grad['x'] = 4 * dvs['x']**3

        