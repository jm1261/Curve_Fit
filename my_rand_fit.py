import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import scipy.optimize as opt

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


def T_fano(E, Ef, r, q, T, offset):
    '''
    '''
    eps = 2 * (E - Ef) / r
    f = ((eps + q) ** 2) / ((eps ** 2) + 1)
    y = T * f + offset
    return y


def f_lsq(params, x, y_meas):
    '''
    '''
    y_fano = T_fano(x, *params)

    #y_fano = A_fano(x, *params)

    res_sq = y_fano - y_meas
    return res_sq


if __name__ == '__main__':

    spectra = np.load('spectra.npy')
    _, wavs, _, _ = np.loadtxt('power_spectrum.csv', delimiter=',', skiprows=1, unpack=True)

    spec = spectra[0:,]

    guesses = [1000 * (rand(5) - 0.5) for i in range(50)]
    results = [opt.least_squares(f_lsq, g, args=(wavs, spec)) for g in guesses]

    r_params = [r.x for r in results]
    r_cost = [r.cost for r in results]

    best_index = np.argmin(r_cost)
    best_params = r_params[best_index]
    print(best_params)

    fig, ax = plt.subplots()
    ax.plot(wavs, spec, label='Raw Data')
    ax.plot(wavs, A_fano(wavs, *best_params), label='Curve Fit')
    ax.legend()
    ax.grid(True)
    plt.show()
    fig.clf()
    plt.close(fig)
