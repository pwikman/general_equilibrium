import numpy as np
from scipy.special import ndtri
from scipy import integrate
from scipy.optimize import least_squares
from scipy.stats import norm
from src.copula_integration import acop

# Area of integration limits, up=upper limit, dw=lower limit
up = 1
dw = 0

#For comparataive statics, change sigma
σ= 1
γ=1


def beta_part(x, β=0):
    return 0-β*np.power(1-x,2)

# Determining the area of integration
def theta_det(x_h,x_l,θ,C):
    if θ==0:
        return dw
    elif θ==1:
        return pw
    meany = ( C + np.power( x_l, γ ) * np.exp( ndtri( [θ] )[0] ) ) / np.power( x_h, γ )
    out = 0 if meany<= 0 else norm.cdf( np.log( meany ) / σ )
    return out

# Determining the area of integration
def theta_det2(x_h,x_l,θ,C):
    if θ==0:
        return dw
    elif θ==1:
        return pw
    meany = ( C + np.power( x_l, γ ) * np.exp( σ*ndtri( [θ] )[0] ) ) / np.power( x_h, γ )
    out = 0 if meany<= 0 else norm.cdf( np.log( meany ) )
    return out

# Calculates the mass of workers in each sector and their allocation externality given input allocation externality and constant C
#For equilibirum first output should equal first input, same with second output, third and fourth output should be equal to zero.
def equilibrium_xs(x_h,x_l,C,kappa, normd=1):
    out, _ = integrate.dblquad(lambda x,y: acop(x,y,kappa, normd), dw, up, lambda x: theta_det(x_h,x_l,x,C), up)
    out2,  _= integrate.dblquad(lambda x,y: acop(x,y,kappa, normd), dw, up, dw, lambda x: theta_det(x_h,x_l,x,C))
    ans_l, _ = integrate.dblquad(lambda x,y: y*acop(x,y,kappa, normd), dw, up, dw, lambda x: theta_det(x_h,x_l,x,C))
    ans_h, _ = integrate.dblquad(lambda x,y: x*acop(x,y,kappa, normd), dw, up, lambda x: theta_det(x_h,x_l,x,C), up)
    return np.array([2*ans_h, 2*ans_l, out, out2])

#Least squares solver with bounds to not let the allocation externality go to wilde. Guess is a tripllet of x0,x1 and C
def equilibrium_solve(guess):
    return least_squares(lambda x: equilibrium_xs(x[0],x[1],x[2])-np.array([x[0],x[1],1/2,1/2]),  guess, bounds=[[0.1,0.1,0],[0.9,0.9, np.inf]])

# Go from wage level to worker type
def trans(w,x,sig):  
    return norm.cdf( np.log( w / np.power( x, γ ) ) / sig )


# The derivate of trans
def trans_diff(w,x,sig):
    return norm.pdf( np.log( w / np.power( x, γ ) ) / sig ) / ( sig * w  )

def mass_wage_sector(x,θ,C,deff,kappa, normd=1):
    if deff == dw:
        sig = 1
        x_0, x_1 = x
        ans, _ = integrate.quad(lambda x: acop(x,trans(θ,x_0,sig),kappa, normd), dw, theta_det2(x_0,x_1,trans(θ,x_0, sig),C))
    elif deff == up:
        sig = σ
        x_1, x_0 = x
        ans, _ = integrate.quad(lambda x: acop(trans(θ,x_0,sig),x,kappa, normd), dw, theta_det(x_0,x_1,trans(θ,x_0, sig),-C))
    out = ans * trans_diff(θ,x_0, sig)
    return out

def summer(x,θ,C,sig,kappa, normd=1):
    x_0, x_1 = x
    ans0, _ = integrate.quad(lambda x: acop(x,trans(θ,x_0,sig),kappa, normd), dw, theta_det(x_0,x_1,trans(θ,x_0,sig),C))
    if θ >= C + beta_part(x_1):
        ans1, _ = integrate.quad(lambda x: acop(trans(θ,x_1,sig),x,kappa, normd), dw, theta_det(x_1,x_0,trans(θ,x_1,sig),-C))
        return ans0* trans_diff(θ,x_0,sig) + ans1* trans_diff(θ,x_1,sig)
    else:
        return ans0 * trans_diff(θ,x_0,sig)