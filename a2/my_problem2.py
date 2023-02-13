import numpy as np
import math as ma
from modopt.api import Problem


class MyProblem(Problem):
    def initialize(self, ):
        # Name your problem
        self.problem_name = 'tetrahedron'
        self.options.declare('dtype', default=float)

    def setup(self):
        # Add design variables of your problem
        self.add_design_variables('x',
                                  shape=(6, ),
                                  vals=np.array([.1, .1, .1, .01, .01, .01], dtype = self.options['dtype']))
        self.add_objective('f')

    def setup_derivatives(self):
        # Declare objective gradient and its shape
        self.declare_objective_gradient(wrt='x', )
        self.declare_objective_hessian(of='x', wrt='x', vals=None)

    # Compute the value of the objective with given design variable values
    def compute_objective(self, dvs, obj):
        da = dvs['x'][0]
        db = dvs['x'][1]
        dc = dvs['x'][2]
        dal = dvs['x'][3]
        dbe = dvs['x'][4]
        dga = dvs['x'][5]

        obj['f'] = (da*db*dc*(ma.sin(2*dal)+ma.sin(2*dbe)+ma.sin(2*dga)+2*ma.cos(dal)*ma.cos(dbe)*ma.cos(dga)-2)**(1/2))/6

    def compute_objective_gradient(self, dvs, grad):
        h = 1e-8
        for i in range(6):
            e = np.zeros((6), dtype=complex)
            e[i] = 1j * h
            xx = np.zeros((6), dtype=complex)
            xx = dvs['x'] + e
            da = xx[0]
            db = xx[1]
            dc = xx[2]
            dal = xx[3]
            dbe = xx[4]
            dga = xx[5]
            c = (da*db*dc*(ma.sin(2*dal)+ma.sin(2*dbe)+ma.sin(2*dga)+2*ma.cos(dal)*ma.cos(dbe)*ma.cos(dga)-2)**(1/2))/6
            cim = np.imag(c)
            grad['x'][i] = cim / h

    def compute_objective_hessian(self, dvs, hess):    
        h = 1e-8
        for i in range(6):
            e = np.zeros((6), dtype=complex)
            e[i] = 1j * h
            for k in range(6):
                e2 = np.zeros((6), dtype=complex)
                e2[k] = 1j * h
                xx = np.zeros((6), dtype=complex)
                xx = dvs['x'] + e + e2
                da = xx[0]
                db = xx[1]
                dc = xx[2]
                dal = xx[3]
                dbe = xx[4]
                dga = xx[5]
                ch = (da*db*dc*(ma.sin(2*dal)+ma.sin(2*dbe)+ma.sin(2*dga)+2*ma.cos(dal)*ma.cos(dbe)*ma.cos(dga)-2)**(1/2))/6
                cimh = np.imag(ch)
                hess['x', 'x'][i][k] = cimh / h

    # def compute_objective_hessian(self, dvs, hess):    
    #     h = 1e-8
    #     for i in range(6):
    #         e = np.zeros((6), dtype=complex)
    #         e[i] = 1j * h
    #         for k in range(6):
    #             e2 = np.zeros((6), dtype=complex)
    #             e2[k] = 1j * h
    #             xx = np.zeros((6), dtype=complex)
    #             xx = dvs['x'] + e + e2
    #             da = xx[0]
    #             db = xx[1]
    #             dc = xx[2]
    #             dal = xx[3]
    #             dbe = xx[4]
    #             dga = xx[5]
    #             ch = (da*db*dc*(ma.sin(2*dal)+ma.sin(2*dbe)+ma.sin(2*dga)+2*ma.cos(dal)*ma.cos(dbe)*ma.cos(dga)-2)**(1/2))/6
    #             cimh = np.imag(ch)
    #             hess['x', 'x'][i][k] = cimh / h

    # def compute_objective_hessian(self, dvs, hess):    
    #     h = 1e-8
    #     for i in range(6):
    #         e = np.zeros((6), dtype=complex)
    #         e[i] = 1j * h
    #         for k in range(6):
    #             # e2 = np.zeros((6), dtype=complex)
    #             # e2[k] = 1j * h
    #             xx = np.zeros((6), dtype=complex)
    #             xx2 = np.zeros((6), dtype=complex)

    #             xx = dvs['x'] + e + h 
    #             da = xx[0]
    #             db = xx[1]
    #             dc = xx[2]
    #             dal = xx[3]
    #             dbe = xx[4]
    #             dga = xx[5]
    #             ch1 = (da*db*dc*(ma.sin(2*dal)+ma.sin(2*dbe)+ma.sin(2*dga)+2*ma.cos(dal)*ma.cos(dbe)*ma.cos(dga)-2)**(1/2))/6

    #             xx2 = dvs['x'] + e - h
    #             da = xx2[0]
    #             db = xx2[1]
    #             dc = xx2[2]
    #             dal = xx2[3]
    #             dbe = xx2[4]
    #             dga = xx2[5]
    #             ch2 = (da*db*dc*(ma.sin(2*dal)+ma.sin(2*dbe)+ma.sin(2*dga)+2*ma.cos(dal)*ma.cos(dbe)*ma.cos(dga)-2)**(1/2))/6

    #             cimh = np.imag(ch1-ch2)
    #             hess['x', 'x'][i][k] = cimh / h**2 / 2

    # def compute_objective_gradient(self, dvs, grad):
    #     # self.x.set_data(x)
    #     h = 1e-8
    #     for i in range(6):
    #         e = np.zeros((6), dtype=complex)
    #         e[i] = 1j * h
    #         xx = np.zeros((6), dtype=complex)
    #         xx = dvs['x'] + e
    #         # c = self.compute_objective(xx, self.obj)
    #         # c = self.compute_objective(xx, c)
    #         # c = {}
    #         cim = np.imag(self.compute_objective(xx + e, self.obj))
    #         grad['x'][i] = cim / h

    # def compute_objective_hessian(self, dvs, hess):    
    #     h = 1e-8
    #     for i in range(6):
    #         e = np.zeros((6), dtype=complex)
    #         e[i] = 1j * h
    #         xx = np.zeros((6), dtype=complex)
    #         xx = dvs['x'] + e
    #         ch = self.compute_objective_gradient(xx)
    #         cimh = np.imag(ch)
    #         hess['x'][i] = cimh / h

        