import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rndm
import scipy.optimize as opt

## A fano resonance exhibits an asymmetric profile due to interference between
## the resonant and background scattering probabilities.
##
## The scattering cross section of the fano profile can be expressed as:
##                  sigma = (e + q)^2 / (e^2) + 1
##
## Where q is a phenomenological shape parameter and e is a reduced energy,
## defined by:
##                  e = 2(E - E(f)) / r
##
## E(f) is the resonant energy, and r is the width of the resonance.
##
## For comparison between various values or q, the normalised value of the Fano
## profile is defined by dividing by 1 + q^2:
##                  sigma(norm) = sigma / (1 + q^2)
##                              = (1/ (1 + q^2)) * (e + q)^2/(e^2 + 1)
##
## There are three special cases of resonance profiles:
##              1. q=0, antiresonance characteristic where a dip trough appears
##                 at the centre of resonance
##              2. 0<q<inf, asymmetric resonance with maximum peak and minimum
##                 trough
##              3. q=inf, lorentzian shape resonance, which is typically seen
##                 in oscillating systems
##
## www.demonstrations.wolfram.com/FanoResonance/


## first define the fano peak itself ##
def A_fano(E, Ef, r, q, A, offset):
    '''
    E = x values
    A = absorption
    Ef = peak
    r = peak width
    q = shape parameter
    offset = DC offset
    '''
    eps = 2 * (E - Ef) / r
    f = ((eps + q) ** 2) / ((eps **2) + 1)
    y = A * (1 - f) + offset
    return y


def T_fano(E, Ef, r, q, A, offset):
    '''
    '''
    eps = 2 * (E - EF) / r
    f = ((eps + q) ** 2) / ((eps ** 2) + 1)
    y = T * f + offset
    return y


if __name__ == '__main__':

    params = [729, 40, 0, 1, 0] # Initialise starting parameters
    RI = np.arange(1.3, 1.65, 0.1) # Refractive index range
    peak_wl = []

    fig, ax = plt.subplots()
    for i, index in enumerate(range(len(RI))):
        col = f'C{index}'

        specname = 'h_mim_15_10RI_' + str(int(10*RI[i]))
        print(specname)
        spec = np.loadtxt(specname + '.txt', skiprows=1)

        ## Perform fit
        popt, _ = opt.curve_fit(A_fano, spec[5:20,2], spec[5:20, 3], p0=params)
        print(popt)
        peak_wl.append(popt[0])

        ## Create spectrum graph
        ax.plot(spec[:,2], spec[:,3], 'o', label=str(RI[i]), color=col)
        x = np.array(range(600, 901, 1))
        ax.plot(x, A_fano(x, *params), label='Fano Fit', linestyle='--', color=col)

        ax.grid(True)
        plt.show()

        fig.clf()
        plt.close(fig)
