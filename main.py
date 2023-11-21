import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import *
from plots import *

data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)

bins = np.linspace(0, 20, 201)

theta = np.pi / 4
delta_m = 2.4e-3
length = 295

energy_vals = np.arange(0.5, 100, 0.5)

p_vals = [oscillation_probability(theta, delta_m, length, i) for i in energy_vals]

# plt.figure()
# plt.title("probability of not oscillating via equation (1)")
# plt.xlabel('energy (GeV)')
# plt.ylabel('muon probability')
# plt.plot(energy_vals, p_vals, 'r'   )
# plt.show()



