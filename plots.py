import numpy as np

from main import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import latex

mpl.style.use('ben')

# params
# font
plt.rcParams['font.family'] = 'Times New Roman'



def fig1():
    """Histogram of event energies"""
    bins = np.linspace(0, 10, 201)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, 11, 1)
    yticks = np.arange(0, 24, 2)

    plt.ylabel("Events")
    plt.ylim(0, 23)
    plt.xlim(0, 10)
    plt.xlabel(r"$\nu_{\mu}$ energy (GeV)")
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.hist(bins[:-1], bins, weights=data, color='lightcoral', alpha=1, label=(r"$\nu_{\mu}$ data"))
    plt.legend(loc=(0.5, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig1.png')
    plt.show()



def fig2():
    """Predicted event energies (no oscillation)"""

    bins = np.linspace(0, 10, 201)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, 11, 1)
    yticks = np.arange(0, 200, 20)

    plt.ylabel("Events")
    plt.ylim(0, 200)
    plt.xlim(0, 10)
    plt.xlabel(r"$\nu_{\mu}$ energy (GeV)")
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.hist(bins[:-1], bins, weights=unosc_flux_prediction, color='darkturquoise', alpha=1, label=(r"$\nu_{\mu}$ expected flux"))
    plt.legend(loc=(0.5, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig2.png')
    plt.show()


def fig2_2():
    """Fig1 and Fig2"""

    bins = np.linspace(0, 10, 201)
    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, 11, 1)
    # yticks = np.arange(0, 24, 2)

    plt.ylabel("Events")
    # plt.ylim(0, 23)
    plt.xlim(0, 10)
    plt.xlabel(r"$\nu_{\mu}$ energy (GeV)")
    plt.xticks(xticks)
    # plt.yticks(yticks)
    plt.hist(bins[:-1], bins, weights=unosc_flux_prediction, color='darkturquoise', alpha=1, label=(r"$\nu_{\mu}$ expected flux"))
    plt.hist(bins[:-1], bins, weights=data, color='lightcoral', alpha=1, label=(r"$\nu_{\mu}$ data"))
    plt.legend(loc=(0.5, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig2_2.png')
    plt.show()


def fig3(e_range, theta=np.pi/4, delta_m=2.4e-3, L=295):
    """ Survival probability against muon neutrino energy

    :param e_range: [a, b] range over which to plot (GeV)
    :param theta: mixing angle (radians)
    :param delta_m: squared mass difference (eV^2)
    :param L: distance travelled by neutrino (km)
    """
    energy_vals = np.linspace(e_range[0], e_range[1], 100000)
    u = [theta, delta_m]

    p_vals = [oscillation_probability(u, L, i) for i in energy_vals]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, e_range[1]+1, (e_range[1])/10)
    yticks = np.arange(0, 1.1, 0.1)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.xlim(0, e_range[1])
    plt.ylim(0, 1)

    plt.ylabel('Oscillation probability')
    plt.xlabel(r"$\nu_{\mu}$ energy (GeV)")

    plt.plot(energy_vals, p_vals, color='lightcoral',
             label=r"${\theta}_{23}$ ="+str(theta/np.pi)+r"$\pi$"+"\n"+"${\Delta}m_{23}^{2}$ ="+str(delta_m))

    plt.legend(loc=(0.5, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig3.png')
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


def fig6(delta_m=2.4e-3):
    """NLL against mixing angle"""
    # testing minimiser on NPP data
    a = 0.12 * np.pi
    b = 0.23 * np.pi
    c = 0.44 * np.pi

    theta_min = parabolic_minimiser(negative_log_likelihood, a, b, c)
    nll_min = negative_log_likelihood(theta_min)

    # theta range 0 to 2pi
    theta_vals = np.linspace(0, np.pi/2, 5000)

    npp_vals = [negative_log_likelihood(i) for i in theta_vals]

    fig = plt.figure(figsize=(12, 8))
    plt.grid(alpha=0.3)
    plt.title("NLL against $\\theta_{23}$")
    plt.xlabel('$\\theta_{23}$ ($\pi$ rads)')
    plt.ylabel('Negative log likelihood')
    plt.plot(theta_min/np.pi, nll_min, 'ro', label='$\\theta$ ='+str(theta_min/np.pi)+"$\pi$")
    plt.plot(theta_vals/np.pi, npp_vals)
    plt.legend()
    plt.show()

def fig7():
    """Contour plot"""

    # theta and delta m values to use
    theta_vals = np.linspace(0.67, 0.9, 400)
    delta_m_vals = np.linspace(2.3e-3, 2.6e-3, 400)
    X, Y = np.meshgrid(theta_vals, delta_m_vals)


    npp_vals = [[negative_log_likelihood(u=[X[i,j], Y[i,j]]) for j in range(len(theta_vals))] for i in tqdm(range(len(theta_vals)))]

    plt.figure()
    plt.contourf(X, Y, npp_vals)
    plt.show()

    # npp_vals = [[negative_log_likelihood(u=[i,j]) for i in theta_vals] for j in delta_m_vals]
    # print([negative_log_likelihood(u=[theta_vals[0], i]) for i in delta_m_vals])
    # print(npp_vals[0])
    #
    # plt.figure()
    # plt.contour([X, Y], npp_vals)
    # plt.show()


    # u values to calculate NPP for
    # u_vals = [[i, j] for i in theta_vals for j in delta_m_vals]

    # npp_vals = [negative_log_likelihood(i) for i in u_vals]


    # print(npp_vals)
    # fig = plt.figure(figsize=(12, 8))
    # plt.grid(alpha=0.3)
    # plt.title("NLL against $\\theta_{23}$")
    # plt.xlabel('$\\theta_{23}$ ($\pi$ rads)')
    # plt.ylabel('Negative log likelihood')

    # normalized_npp = npp_vals / np.sqrt(np.sum([i**2 for i in npp_vals]))
    # plt.plot(theta_vals/np.pi, normalized_npp)
    #
    # plt.plot(theta_vals/np.pi, npp_vals)
    #
    # plt.show()

fig2_2()
fig3(e_range=[0,10])



