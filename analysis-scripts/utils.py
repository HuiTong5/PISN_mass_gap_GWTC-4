import numpy as np
from scipy import special as scs

xp = np

def Di(z):

    """
    Wrapper for the scipy implmentation of Spence's function.
    Note that we adhere to the Mathematica convention as detailed in:
    https://reference.wolfram.com/language/ref/PolyLog.html

    Inputs
    z: A (possibly complex) scalar or array

    Returns
    Array equivalent to PolyLog[2,z], as defined by Mathematica
    """

    return scs.spence(1.-z+0j)

def chi_effective_prior_from_isotropic_spins(xs, q, aMax=1.0):

    """
    Function defining the conditional priors p(chi_eff|q) corresponding to
    uniform, isotropic component spin priors from https://github.com/tcallister/effective-spin-priors/blob/main/priors.py.

    Inputs
    q: Mass ratio value (according to the convention q<1)
    aMax: Maximum allowed dimensionless component spin magnitude
    xs: Chi_effective value or values at which we wish to compute prior

    Returns:
    Array of prior values
    """

    # Ensure that `xs` is an array and take absolute value
    xs = xp.reshape(xp.abs(xs),-1)

    # Set up various piecewise cases
    pdfs = xp.ones(xs.size,dtype=complex)*(-1.)
    caseZ = (xs==0)
    caseA = (xs>0)*(xs<aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseB = (xs<aMax*(1.-q)/(1.+q))*(xs>q*aMax/(1.+q))
    caseC = (xs>aMax*(1.-q)/(1.+q))*(xs<q*aMax/(1.+q))
    caseD = (xs>aMax*(1.-q)/(1.+q))*(xs<aMax/(1.+q))*(xs>=q*aMax/(1.+q))
    caseE = (xs>aMax*(1.-q)/(1.+q))*(xs>aMax/(1.+q))*(xs<aMax)
    caseF = (xs>=aMax)

    # Select relevant effective spins
    x_A = xs[caseA]
    x_B = xs[caseB]
    x_C = xs[caseC]
    x_D = xs[caseD]
    x_E = xs[caseE]

    pdfs[caseZ] = (1.+q[caseZ])/(2.*aMax)*(2.-xp.log(q[caseZ]))

    pdfs[caseA] = (1.+q[caseA])/(4.*q[caseA]*aMax**2)*(
                    q[caseA]*aMax*(4.+2.*xp.log(aMax) - xp.log(q[caseA]**2*aMax**2 - (1.+q[caseA])**2*x_A**2))
                    - 2.*(1.+q[caseA])*x_A*xp.arctanh((1.+q[caseA])*x_A/(q[caseA]*aMax))
                    + (1.+q[caseA])*x_A*(Di(-q[caseA]*aMax/((1.+q[caseA])*x_A)) - Di(q[caseA]*aMax/((1.+q[caseA])*x_A)))
                    )

    pdfs[caseB] = (1.+q[caseB])/(4.*q[caseB]*aMax**2)*(
                    4.*q[caseB]*aMax
                    + 2.*q[caseB]*aMax*xp.log(aMax)
                    - 2.*(1.+q[caseB])*x_B*xp.arctanh(q[caseB]*aMax/((1.+q[caseB])*x_B))
                    - q[caseB]*aMax*xp.log((1.+q[caseB])**2*x_B**2 - q[caseB]**2*aMax**2)
                    + (1.+q[caseB])*x_B*(Di(-q[caseB]*aMax/((1.+q[caseB])*x_B)) - Di(q[caseB]*aMax/((1.+q[caseB])*x_B)))
                    )

    pdfs[caseC] = (1.+q[caseC])/(4.*q[caseC]*aMax**2)*(
                    2.*(1.+q[caseC])*(aMax-x_C)
                    - (1.+q[caseC])*x_C*xp.log(aMax)**2.
                    + (aMax + (1.+q[caseC])*x_C*xp.log((1.+q[caseC])*x_C))*xp.log(q[caseC]*aMax/(aMax-(1.+q[caseC])*x_C))
                    - (1.+q[caseC])*x_C*xp.log(aMax)*(2. + xp.log(q[caseC]) - xp.log(aMax-(1.+q[caseC])*x_C))
                    + q[caseC]*aMax*xp.log(aMax/(q[caseC]*aMax-(1.+q[caseC])*x_C))
                    + (1.+q[caseC])*x_C*xp.log((aMax-(1.+q[caseC])*x_C)*(q[caseC]*aMax-(1.+q[caseC])*x_C)/q[caseC])
                    + (1.+q[caseC])*x_C*(Di(1.-aMax/((1.+q[caseC])*x_C)) - Di(q[caseC]*aMax/((1.+q[caseC])*x_C)))
                    )

    pdfs[caseD] = (1.+q[caseD])/(4.*q[caseD]*aMax**2)*(
                    -x_D*xp.log(aMax)**2
                    + 2.*(1.+q[caseD])*(aMax-x_D)
                    + q[caseD]*aMax*xp.log(aMax/((1.+q[caseD])*x_D-q[caseD]*aMax))
                    + aMax*xp.log(q[caseD]*aMax/(aMax-(1.+q[caseD])*x_D))
                    - x_D*xp.log(aMax)*(2.*(1.+q[caseD]) - xp.log((1.+q[caseD])*x_D) - q[caseD]*xp.log((1.+q[caseD])*x_D/aMax))
                    + (1.+q[caseD])*x_D*xp.log((-q[caseD]*aMax+(1.+q[caseD])*x_D)*(aMax-(1.+q[caseD])*x_D)/q[caseD])
                    + (1.+q[caseD])*x_D*xp.log(aMax/((1.+q[caseD])*x_D))*xp.log((aMax-(1.+q[caseD])*x_D)/q[caseD])
                    + (1.+q[caseD])*x_D*(Di(1.-aMax/((1.+q[caseD])*x_D)) - Di(q[caseD]*aMax/((1.+q[caseD])*x_D)))
                    )

    pdfs[caseE] = (1.+q[caseE])/(4.*q[caseE]*aMax**2)*(
                    2.*(1.+q[caseE])*(aMax-x_E)
                    - (1.+q[caseE])*x_E*xp.log(aMax)**2
                    + xp.log(aMax)*(
                        aMax
                        -2.*(1.+q[caseE])*x_E
                        -(1.+q[caseE])*x_E*xp.log(q[caseE]/((1.+q[caseE])*x_E-aMax))
                        )
                    - aMax*xp.log(((1.+q[caseE])*x_E-aMax)/q[caseE])
                    + (1.+q[caseE])*x_E*xp.log(((1.+q[caseE])*x_E-aMax)*((1.+q[caseE])*x_E-q[caseE]*aMax)/q[caseE])
                    + (1.+q[caseE])*x_E*xp.log((1.+q[caseE])*x_E)*xp.log(q[caseE]*aMax/((1.+q[caseE])*x_E-aMax))
                    - q[caseE]*aMax*xp.log(((1.+q[caseE])*x_E-q[caseE]*aMax)/aMax)
                    + (1.+q[caseE])*x_E*(Di(1.-aMax/((1.+q[caseE])*x_E)) - Di(q[caseE]*aMax/((1.+q[caseE])*x_E)))
                    )

    pdfs[caseF] = 0.

    # Deal with spins on the boundary between cases
    if xp.any(pdfs==-1):
        boundary = (pdfs==-1)
        pdfs[boundary] = 0.5*(chi_effective_prior_from_isotropic_spins(q,aMax,xs[boundary]+1e-6)\
                        + chi_effective_prior_from_isotropic_spins(q,aMax,xs[boundary]-1e-6))

    return xp.real(pdfs)