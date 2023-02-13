import numpy as np
import time
from modopt.api import Optimizer
from numpy.linalg import *

"""
Newton
"""

class MyOptimizer3(Optimizer):
    def initialize(self):

        # Name your algorithm
        self.solver_name = 'Newton'

        self.obj = self.problem._compute_objective
        self.grad = self.problem._compute_objective_gradient
        self.hess = self.problem._compute_objective_hessian

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
        hess = self.hess

        start_time = time.time()

        # Setting intial values for initial iterates
        x_k = x * 1.
        f_k = obj(x_k)
        g_k = grad(x_k)
        B_k = hess(x_k)
        rho = .99
        c1 = 1e-4
        alphab = 2.

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
            x_k0 = x_k
            g_k0 = g_k
            B_k0 = B_k

            p_k   =  - solve(B_k0, g_k0)
            alpha = self.backls(rho, c1, alphab, x_k0, p_k)

            x_k = x_k + alpha * p_k
            f_k = obj(x_k)
            g_k = grad(x_k)
            opt = np.linalg.norm(g_k)
            B_k = hess(x_k)

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

    def backls(self, rho, c1, alphab, xk, pk):
        alpha = alphab
        rho = rho
        c1 = c1
        xk = xk
        pk = pk
        obj = self.obj
        grad = self.grad
        while obj(xk + alpha * pk) >= obj(xk) + c1 * alpha * np.matmul(np.transpose(grad(xk)), pk):
            alpha = rho * alpha
        
        return alpha

