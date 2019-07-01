import numpy as np
from numpy.random import rand
import matplotlib
import scipy.optimize as opt
#matplotlib.use('Agg') # For remote use
import matplotlib.pyplot as plt

def fano(x, a, b, c, A, e):
    eps = 2*(x-a)/b
    f = (eps+c)**2/(eps**2+1)
    y = A*f+e
    return y

def f_lsq(params, x, y_meas):
    y_fan = fano(x, *params)
    ressq = y_fan-y_meas
    return ressq

save_hist = False

spectra = np.load('spectra.npy')
_,wavs,_,_ = np.loadtxt('power_spectrum.csv',delimiter=',',
    skiprows=1, unpack=True)

spec = spectra[0,:]

# guesses = [np.append(745, 100*(rand(4)-0.5)) for i in range(10)]
guesses = [1000*(rand(5)-0.5) for i in range(50)]
results = [opt.least_squares(f_lsq, g, args=(wavs, spec)) for g in guesses]
r_params = [r.x for r in results]
r_cost = [r.cost for r in results]
del results

## Find best parameters corresponding to minimum residuals
best_index = np.argmin(r_cost)
best_params = r_params[best_index]
print(best_params)

# Output
fit_fig, fit_ax = plt.subplots()
fit_ax.plot(wavs, spec, label='Raw data')
fit_ax.plot(wavs, fano(wavs, *best_params), label='Curve fit')
fit_ax.legend()
fit_ax.set_xlabel('Wavelength (nm)')
fit_ax.set_ylabel('Intensity')
#fit_fig.savefig('plot.png')
plt.show()

if save_hist:
    h_fig, h_ax = plt.subplots()
    h_ax.hist(r_cost)
    h_ax.set_xlabel('Value of cost function at the solution')
    h_ax.set_ylabel('Number of occurences')
    h_fig.savefig('histogram.png')
