from main import *
import matplotlib as mpl
import matplotlib.pyplot as plt


# params
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.size'] = 15
mpl.rcParams['font.family'] = 'Arial'


def fig1():
    """Histogram of event energies"""
    bins = np.linspace(0, 10, 201)
    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("Histogram of event energies")
    plt.ylabel("Number of events")
    plt.xlabel("Energy (GeV)")
    plt.hist(bins[:-1], bins, weights=data)

    plt.show()


def fig2():
    """Histogram of simulated non-oscillating event energies"""
    bins = np.linspace(0, 10, 201)
    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("Histogram of simulated non-oscillating event energies")
    plt.ylabel("Number of events")
    plt.xlabel("Energy (GeV)")
    plt.hist(bins[:-1], bins, weights=unosc_flux_prediction)
    plt.show()


def fig3(e_range, theta=np.pi/4, delta_m=2.4e-3, L=295):
    """Oscillation probability against energy"""
    energy_vals = np.linspace(e_range[0], e_range[1], 100000)
    u = [theta, delta_m]

    p_vals = [oscillation_probability(u, L, i) for i in energy_vals]

    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("Probability of not oscillating via equation (1)")
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Muon probability')

    plt.plot(energy_vals, p_vals, label="${\\theta}_{23} = \\pi/4$ \n"
                                        "${\\Delta}m_{23}^{2} = 2.4$x$10^{-3}$ \n"
                                        "$L = 295$")

    plt.legend()
    plt.show()


def fig4(theta=np.pi/4, delta_m=2.4e-3, L=295):
    """Oscillated event rate prediction"""
    bins = np.linspace(0, 10, 201)
    bin_midpoints = np.arange(0.025, 10, 0.05)
    u = [theta, delta_m]

    p = [oscillation_probability(u, L, i) for i in bin_midpoints]
    lambda_i = p * unosc_flux_prediction

    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("Oscillated event rate prediction $\lambda_i (\\bf{u}\\sf)$")
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Number of events')

    plt.hist(bins[:-1], bins, weights=lambda_i, label="${\\theta}_{23} = $"+str(theta/np.pi)+"$\pi$ \n"
                                        "${\\Delta}m_{23}^{2} = $"+str(delta_m)+" \n"
                                        "$L = $"+str(L))


    plt.hist(bins[:-1], bins, weights=data, label="${\\theta}_{23} = $"+"???"+" \n"
                                        "${\\Delta}m_{23}^{2} = $"+"???"+" \n"
                                        "$L = $"+"295")
    plt.legend()
    plt.show()


def fig5(delta_m=2.4e-3):
    """Negative log likelihood against mixing angle"""

    # theta range 0 to 2pi
    theta_vals = np.linspace(0, 2*np.pi, 1000)

    # u values to calculate NPP for
    u_vals = [[i, delta_m] for i in theta_vals]

    npp_vals = [negative_log_likelihood(u=i) for i in u_vals]

    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("NLL against $\\theta_{23}$")
    plt.xlabel('$\\theta_{23}$ ($\pi$ rads)')
    plt.ylabel('Negative log likelihood')

    # normalized_npp = npp_vals / np.sqrt(np.sum([i**2 for i in npp_vals]))
    # plt.plot(theta_vals/np.pi, normalized_npp)

    plt.plot(theta_vals/np.pi, npp_vals)

    plt.show()





