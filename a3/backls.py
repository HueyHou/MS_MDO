import numpy as np


def backls(rho, c1, alphab, xk, pk):
    alpha = alphab
    f0 = obj(xk)
    fk = obj(xk + alpha * pk)
    gk = grad(xk)
    while fk >= f0 + c1 * alpha * gk * pk:
        alpha = rho * alpha
    
    return alpha
















def BacktrackingLineSearch(x0, alphabar, rho, c1,pk):
    alpha_array = []
    x=x0
    alpha = alphabar
    rho = rho
    c1 = c1
    pk = pk
    f = func(x0)
    f1 = func(x0+alpha*pk)
    while f1 >= f + c1*alpha*(np.dot(gradient(x0),pk)):
        alpha = rho*alpha
        alpha_array = np.append(alpha_array,alpha)
        
    return alpha,alpha_array