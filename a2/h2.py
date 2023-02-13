from my_problem12 import MyProblem
# from rosenbrock import Rosenbrock
# from my_optimizerV import MyOptimizer
from modopt.optimization_algorithms import SteepestDescent, QuasiNewton, Newton
import matplotlib.pyplot as plt

tol = 1E-8
max_itr = 100

"""
"""
# dtype = float
dtype = complex

# prob = Rosenbrock()
# prob = MyProblem()
prob = MyProblem(dtype=dtype)

# optimizer = SteepestDescent(prob,opt_tol=tol,max_itr=max_itr,outputs=['itr', 'obj', 'x', 'opt', 'time'])
# optimizer = QuasiNewton(prob,opt_tol=tol,max_itr=max_itr)
optimizer = Newton(prob,opt_tol=tol,max_itr=max_itr)
# optimizer = MyOptimizer(prob,opt_tol=tol,max_itr=max_itr)

"""
"""
optimizer.check_first_derivatives(prob.x.get_data())
optimizer.solve()
optimizer.print_results(summary_table=True)

print(optimizer.outputs['itr'][-1])
print(optimizer.outputs['x'][-1])
print(optimizer.outputs['time'][-1])
print(optimizer.outputs['obj'][-1])
print(optimizer.outputs['opt'][-1])

plt.semilogy(optimizer.outputs['itr'],optimizer.outputs['opt'] )
plt.xlabel('iteration')
plt.ylabel('gradient norm')
plt.legend()
plt.tight_layout()
plt.show()

