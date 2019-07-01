#!/usr/bin/python3
import matplotlib       # For remote use
matplotlib.use('Agg')   # For remote use
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as ran
import scipy.optimize as opt

def fano(x, a, b, q, A, e):
    eps = 2*(x-a)/b
    f = ((eps+q)**2)/(eps**2+1)
    y = A*(1-f) + e
    return y

popt = [675, 40, 0, 0.6, 0] # Initialise starting parameters
RI = np.arange(1.3, 1.65, 0.1)
peak_wl = []

fig, ax = plt.subplots()
for i in range(len(RI)):
	col = 'C'+str(int(i))
	specname = 'h_mim_15_10RI_'+str(int(10*RI[i]))
	print(specname)
	spec = np.loadtxt(specname + '.txt', skiprows=1)

	## Perform fit
	popt, _  = opt.curve_fit(fano, spec[5:20,2], spec[5:20,3], p0=popt)
	print(popt)
	peak_wl.append(popt[0])

	## Create spectrum graph

	ax.plot(spec[:,2], spec[:,3], 'o', label=str(RI[i]), color = col)
	x = np.array(range(600, 901, 1))
	ax.plot(x, fano(x, *popt),label='Fano fit',\
		        linestyle='--', color= col)

ax.set_title('Gold NDoM with 10nm coating, varying RI')
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Normalised intensity')
ax.set_xlim(650, 850)
ax.set_ylim(0, 1)
ax.grid(True)
ax.legend(title = 'Ref. index', loc = 'best')
## plt.show()
fig.savefig('cylinder_specs.png')
fig.clf()

sens_data = np.array([[RI[i], peak_wl[i]] for i in range(len(peak_wl))])
np.savetxt('sensitivity.txt', sens_data, fmt='%.10f', delimiter='\t', \
    header = 'RI\t\tPeak wavalength (nm)')
