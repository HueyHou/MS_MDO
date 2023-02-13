from rosenbrock import Rosenbrock
from my_optimizerSD_bls import MyOptimizer1
import matplotlib.pyplot as plt

tol = 1E-8
max_itr = 100

"""
"""
prob = Rosenbrock()
optimizer = MyOptimizer1(prob,opt_tol=tol,max_itr=max_itr)

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

plt.figure(1)
plt.semilogy(optimizer.outputs['itr'],optimizer.outputs['opt'] )
plt.xlabel('iteration')
plt.ylabel('gradient norm')
plt.legend()
plt.tight_layout()

# plt.figure(2)
# plt.plot(optimizer.outputs['x'][:,0], optimizer.outputs['x'][:,1])

plt.show()

