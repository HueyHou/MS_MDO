import numpy as np
import time
from modopt.api import Optimizer
from numpy.linalg import *

"""
BFGS; inverse BFGS
"""

class MyOptimizer(Optimizer):
    def initialize(self):

        # Name your algorithm
        self.solver_name = 'Quasi-Newton BFGS inverse'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient

        self.options.declare('max_itr', default=1000, types=int)
        self.options.declare('opt_tol', default=1e-5, types=float)

        # Specify format of outputs available from your optimizer after each iteration
        self.default_outputs_format = {
            'itr': int,
            'obj': float,
            # for arrays from each iteration, shapes need to be declared
            'x': (float, (self.problem.nx, )),
            'opt': float,
            'time': float,
        }

        # Enable user to specify, as a list, which among the available outputs
        # need to be stored in memory and written to output files
        self.options.declare('outputs',
                             types=list,
                             default=['itr', 'obj', 'x', 'opt', 'time'])

    def solve(self):
        nx = self.problem.nx
        x = self.problem.x.get_data()
        opt_tol = self.options['opt_tol']
        max_itr = self.options['max_itr']

        obj = self.obj
        grad = self.grad

        start_time = time.time()

        # Setting intial values for initial iterates
        x_k = x * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)
        B_k = np.identity(2)
        V_k = np.linalg.inv(B_k)

        # Iteration counter
        itr = 0

        # Optimality
        opt = np.linalg.norm(g_k)

        # Initializing outputs
        self.update_outputs(itr=0,
                            x=x_k,
                            obj=f_k,
                            opt=opt,
                            time=time.time() - start_time)

        while (opt > opt_tol and itr < max_itr):
            itr_start = time.time()
            itr += 1

            # ALGORITHM STARTS HERE
            # >>>>>>>>>>>>>>>>>>>>>
            x_k0 =  x_k
            g_k0 =  grad(x_k)
            V_k0 =  V_k

            p_k  =  - np.matmul(V_k0, g_k0)
            x_k  =  x_k0 + p_k * 1.7
            f_k  =  obj(x_k)
            g_k  =  grad(x_k)
            s_k  =  x_k - x_k0
            y_k  =  g_k - g_k0

            p1 = np.matmul(s_k, np.transpose(y_k)) / np.matmul(np.transpose(s_k), y_k)
            p2 = np.matmul(y_k, np.transpose(s_k)) / np.matmul(np.transpose(y_k), s_k)
            p3 = np.matmul(s_k, np.transpose(s_k)) / np.matmul(np.transpose(y_k), s_k)

            V_k = np.matmul(np.matmul((np.identity(2) - p1), V_k0), (np.identity(2) - p2)) + p3

            opt  =  np.linalg.norm(g_k)

            # <<<<<<<<<<<<<<<<<<<
            # ALGORITHM ENDS HERE

            # Append arrays inside outputs dict with new values from the current iteration
            self.update_outputs(itr=itr,
                                x=x_k,
                                obj=f_k,
                                opt=opt,
                                time=time.time() - start_time)

        # Run post-processing for the Optimizer() base class
        self.run_post_processing()

        end_time = time.time()
        self.total_time = end_time - start_time

