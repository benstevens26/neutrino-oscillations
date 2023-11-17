import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)

bins = np.linspace(0, 20, 201)

plt.style.use("fast")
plt.figure()
plt.title("histogram of event energies")
plt.ylabel("number of events")
plt.xlabel("energy (GeV)")
plt.hist(bins[:-1], bins, weights=data)


plt.figure()
plt.title("predicted")
plt.ylabel("number of events")
plt.xlabel("energy (GeV)")
plt.hist(bins[:-1], bins, weights=unosc_flux_prediction)
plt.show()

