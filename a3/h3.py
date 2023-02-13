from rosenbrock import Rosenbrock
from my_optimizerSD_bls import MyOptimizer1
from my_optimizerQNB_bls import MyOptimizer2
from my_optimizerN_bls import MyOptimizer3
import matplotlib.pyplot as plt

tol = 1E-5
max_itr = 100

"""
"""
prob1      = Rosenbrock()
prob2      = Rosenbrock()
prob3      = Rosenbrock()
optimizer1 = MyOptimizer1(prob1, opt_tol=tol, max_itr=max_itr)
optimizer2 = MyOptimizer2(prob2, opt_tol=tol, max_itr=max_itr)
optimizer3 = MyOptimizer3(prob3, opt_tol=tol, max_itr=max_itr)

"""
"""
optimizer1.check_first_derivatives(prob1.x.get_data())
optimizer1.solve()
optimizer1.print_results(summary_table=True)
optimizer2.check_first_derivatives(prob2.x.get_data())
optimizer2.solve()
optimizer2.print_results(summary_table=True)
optimizer3.check_first_derivatives(prob3.x.get_data())
optimizer3.solve()
optimizer3.print_results(summary_table=True)

"""
"""
plt.figure(1)
plt.plot(optimizer1.outputs['itr'],optimizer1.outputs['opt'], label = 'SD')
plt.plot(optimizer2.outputs['itr'],optimizer2.outputs['opt'], label = 'QNB')
plt.plot(optimizer3.outputs['itr'],optimizer3.outputs['opt'], label = 'N')
plt.xlabel('iteration')
plt.ylabel('gradient norm')
plt.legend()
plt.tight_layout()

plt.figure(2)
plt.plot(optimizer1.outputs['itr'],optimizer1.outputs['obj'], label = 'SD')
plt.plot(optimizer2.outputs['itr'],optimizer2.outputs['obj'], label = 'QNB')
plt.plot(optimizer3.outputs['itr'],optimizer3.outputs['obj'], label = 'N')
plt.xlabel('iteration')
plt.ylabel('objective function')
plt.legend()
plt.tight_layout()

plt.figure(3)
plt.plot(optimizer1.outputs['x'][:,0],optimizer1.outputs['x'][:,1], label = 'SD')
plt.plot(optimizer2.outputs['x'][:,0],optimizer2.outputs['x'][:,1], label = 'QNB')
plt.plot(optimizer3.outputs['x'][:,0],optimizer3.outputs['x'][:,1], label = 'N')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.tight_layout()

plt.show()
