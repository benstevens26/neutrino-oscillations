import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functions import *

# data
data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)

# params
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'Arial'


def fig1():
    """Histogram of event energies"""
    bins = np.linspace(0, 20, 201)
    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("Histogram of event energies")
    plt.ylabel("Number of events")
    plt.xlabel("Energy (GeV)")
    plt.hist(bins[:-1], bins, weights=data)

    plt.show()


def fig2():
    """Histogram of predicted energies"""
    bins = np.linspace(0, 20, 201)
    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("Histogram of non-oscillating event energies")
    plt.ylabel("Number of events")
    plt.xlabel("Energy (GeV)")
    plt.hist(bins[:-1], bins, weights=unosc_flux_prediction)
    plt.show()

    plt.show()


def fig3(theta=np.pi/4, delta_m=2.4e-3, length=295):
    """Oscillation probability against energy"""
    energy_vals = np.arange(0.5, 100, 0.5)
    p_vals = [oscillation_probability(theta, delta_m, length, i) for i in energy_vals]

    fig = plt.figure(figsize=(12, 8))
    plt.title("Probability of not oscillating via equation (1)")
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Muon probability')
    plt.plot(energy_vals, p_vals, 'r'   )
    plt.show()



fig3()
