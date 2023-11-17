import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)

bins = np.linspace(0, 20, 201)

# plt.style.use("fast")
# plt.figure()
# plt.title("histogram of event energies")
# plt.ylabel("number of events")
# plt.xlabel("energy (GeV)")
# plt.hist(bins[:-1], bins, weights=data)

# plt.figure()
# plt.title("predicted")
# plt.ylabel("number of events")
# plt.xlabel("energy (GeV)")
# plt.hist(bins[:-1], bins, weights=unosc_flux_prediction)
# plt.show()


def muon_probability(theta, delta_m, length, energy):
    """Return probability that muon neutrino is observed as muon neutrino"""
    return 1 - (np.sin(2*theta)*np.sin(2*theta)*np.sin(1.267*delta_m*length/energy))


theta = np.pi / 4
delta_m = 2.4e-3
length = 295

energy_vals = np.arange(0.5, 100, 0.5)

p_vals = [muon_probability(theta, delta_m, length, i) for i in energy_vals]

plt.figure()
plt.xlabel('energy (GeV')
plt.ylabel('muon probability')
plt.plot(energy_vals, p_vals, 'r')
plt.show()



