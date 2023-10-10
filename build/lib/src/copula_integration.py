import numpy as np
from scipy.special import ndtri
from numpy.linalg import solve
from numpy.linalg import cholesky
from scipy import integrate
from scipy.optimize import least_squares

# The guassian  cupola

def _gaussian(x,y, kappa):
    R = cholesky([[1,kappa],[kappa,1]])
    x = ndtri([x,y])
    z = solve(R,x.T)
    log_det= np.sum(np.log(np.diag(R)))
    return np.exp(-0.5 * np.sum(  np.power(z,2) - np.power(x,2) ) - log_det)

# Ali–Mikhail–Haq cupola

def acop(x,y,kappa, normd):
    if normd == 0:
        return ( 1 + kappa * ((1+x)*(1+y)-3) + (kappa**2)*(1-x)*(1-y)) / ((1-kappa*(1-x)*(1-y))**3)
    elif normd == 1:
        return _gaussian(x,y, kappa)
    return 

# Various integral calculation functions

#Get the measure of workers in each sector given k and m

def func_y(k,m,kappa,normd):
    ans1, _ = integrate.dblquad(lambda x,y: acop(x,y,kappa,normd), 0 , 1, lambda x: m + x * k,  1)
    ans0, _ = integrate.dblquad(lambda x,y: acop(x,y,kappa,normd), 0,1 , 0, lambda x: m + x * k )
    return ans0, ans1

# Gives the mass of workers in each sector where ans0 is the high type sector. Needs the mean skill of workers in each sector and intercept value for the separation function in the high type sector

def func_mx(x0,x1,m,kappa,normd):
    out, _= integrate.dblquad(lambda x,y: acop(x,y,kappa,normd), 0,1, lambda x: np.maximum(np.minimum(m + x * x0/x1,1),0), 1)
    ans1, _ = integrate.dblquad(lambda x,y: x*acop(x,y,kappa,normd), 0 , 1, lambda x: m + x *x0/x1, 1 )
    ans0, _ = integrate.dblquad(lambda x,y: y*acop(x,y,kappa,normd), 0,1 , 0, lambda x: m + x * x0/x1 )
    return np.array([2*ans0, 2*ans1, out])


    return ans0, ans1

# Function for solving for equilibrium using least squares
def equilibrium_solve(guess,kappa,normd):
    return least_squares(lambda x: func_mx(x[0],x[1],x[2],kappa,normd)-np.array([x[0],x[1],1/2]),  guess, bounds=[[0.1,0.1,0.01],[0.9,0.9, 0.25]])
