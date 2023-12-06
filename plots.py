import numpy as np
import scipy.misc

from main import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

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

    p_vals = [oscillation_probability(u, i) for i in energy_vals]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, e_range[1]+1, (e_range[1])/10)
    yticks = np.arange(0, 1.1, 0.1)
    plt.xlim(0, e_range[1])
    plt.ylim(0, 1)

    plt.ylabel('Oscillation probability')
    plt.xlabel(r"$\nu_{\mu}$ energy (GeV)")

    plt.plot(energy_vals, p_vals, color='royalblue',
             label=r"${\theta}_{23}$ ="+str(theta/np.pi)+r"$\pi$"+"\n"+"${\Delta}m_{23}^{2}$ ="+str(delta_m))

    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.legend(loc=(0.5, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig3.png')
    plt.show()


def fig4(theta=np.pi/4, delta_m=2.4e-3):
    """Predicted event energies (oscillated)"""
    bins = np.linspace(0, 10, 201)
    u = [theta, delta_m]
    lambda_i = get_lambda(u)

    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, 11, 1)
    yticks = np.arange(0, max(lambda_i)+5, 5)

    plt.ylabel("Events")
    plt.ylim(0, max(lambda_i))
    plt.xlim(0, 10)
    plt.xlabel(r"$\nu_{\mu}$ energy (GeV)")
    plt.xticks(xticks)
    plt.yticks(yticks)

    plt.hist(bins[:-1], bins, color='darkturquoise', weights=lambda_i, label="${\\theta}_{23} = $"+str(theta/np.pi)+"$\pi$ \n"
                                        "${\\Delta}m_{23}^{2} = $"+str(delta_m))

    plt.hist(bins[:-1], bins, color='lightcoral', weights=data, label=r"detected $\nu_{\mu}$")

    plt.legend(loc=(0.5, 0.4))
    plt.tight_layout()
    plt.savefig('figs/fig4.png')
    plt.show()


def fig5(delta_m=2.4e-3):
    """Negative log likelihood against mixing angle (fixed delta_m)"""

    theta_vals = np.linspace(0, np.pi/2, 2000)
    u_vals = [[i, delta_m] for i in theta_vals]
    nll_vals = [negative_log_likelihood(u=i) for i in u_vals]

    # u_vals_2 = [[i, 2.5e-3] for i in theta_vals]
    # npp_vals_2 = [negative_log_likelihood(u=i) for i in u_vals_2]


    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(0, 1/2 + 0.05, 0.05)
    # yticks = np.arange(0, 1., 0.1)
    plt.xlim(0, 1/2)
    # plt.ylim(0, 1)

    plt.ylabel('Negative log-likelihood')
    plt.xlabel(r"Mixing angle $\theta_{23}$ /$ \pi$")

    plt.plot(theta_vals/np.pi, nll_vals, lw=2, color='royalblue',
             label=r"${\Delta}m_{23}^{2}$ = "+str(delta_m))

    # plt.plot(theta_vals/np.pi, npp_vals_2, color='mediumorchid',
    #          label=r"${\Delta}m_{23}^{2}$ ="+str(2.5e-3))

    plt.xticks(xticks)
    # plt.yticks(yticks)
    plt.legend(loc=(0.3, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig5.png')
    plt.show()


def fig5_5(theta=np.pi/4):
    """Negative log likelihood against delta_m (fixed theta)"""

    delta_m_vals = np.linspace(1e-3, 4e-3, 2000)
    u_vals = [[theta, i] for i in delta_m_vals]
    nll_vals = [negative_log_likelihood(u=i) for i in u_vals]

    # u_vals_2 = [[i, 2.5e-3] for i in theta_vals]
    # npp_vals_2 = [negative_log_likelihood(u=i) for i in u_vals_2]


    fig = plt.figure(figsize=(8, 6), dpi=100)
    xticks = np.arange(1e-3, 4.1e-3, 0.5e-3)
    # yticks = np.arange(0, 1., 0.1)
    plt.xlim(1e-3, 4e-3)
    # plt.ylim(0, 1)

    plt.ylabel('Negative log-likelihood')
    plt.xlabel("Squared mass difference (eV^2)")

    plt.plot(delta_m_vals, nll_vals, lw=2, color='royalblue',
             label=r"$\theta_{23}$ = "+str(np.round(theta, 3)))

    # plt.plot(theta_vals/np.pi, npp_vals_2, color='mediumorchid',
    #          label=r"${\Delta}m_{23}^{2}$ ="+str(2.5e-3))

    plt.xticks(xticks)
    # plt.yticks(yticks)
    plt.legend(loc=(0.3, 0.7))
    plt.tight_layout()
    plt.savefig('figs/fig5_5.png')
    plt.show()


def fig6(delta_m=2.4e-3):
    """NLL against mixing angle with parabolic minimisation"""

    a, b, c = gen_points([np.pi/4 - 0.3, np.pi/4 + 0.3])

    theta_min, delta_m = parabolic_minimiser_nd(nll2, axis=0, x0=[np.pi/4, delta_m], xrange=0.3)

    nll_min = negative_log_likelihood(u=[theta_min, delta_m])

    print(r"$\sin{\theta_{23}}$")

    theta_lb, theta_ub = nll_error([theta_min, delta_m])
    nll_lb = negative_log_likelihood([theta_lb, delta_m])
    nll_ub = negative_log_likelihood([theta_ub, delta_m])

    # curv_err = 1/np.sqrt(curvature)  # error from curvature
    # print("theta errors from curvature")
    # print(theta_min - curv_err)
    # print(theta_min + curv_err)

    theta_vals = np.linspace(0, np.pi/2, 2000)
    u_vals = [[i, delta_m] for i in theta_vals]
    nll_vals = [negative_log_likelihood(u=i) for i in u_vals]

    fig = plt.figure(figsize=(8, 6), dpi=100)
    # xticks = np.arange(0, 1/2 + 0.05, 0.05)
    # yticks = np.arange(0, 1., 0.1)
    plt.xlim(0.15, 1/2)
    plt.ylim(min(nll_vals)-100, max(nll_vals)+50)

    plt.ylabel('Negative log-likelihood')
    plt.xlabel(r"Mixing angle $\theta_{23}$ /$ \pi$")

    plt.plot(theta_vals/np.pi, nll_vals, lw=2, color='royalblue',
             label=r"${\Delta}m_{23}^{2}$ = "+str(delta_m))

    plt.plot(theta_min/np.pi, nll_min, color="black", marker="s",
             markersize=5, linestyle="None", label=r"$\theta_{23}$ = "+str(np.round(theta_min, 4)))

    plt.plot(theta_lb/np.pi, nll_lb, color="red", marker="s",
             markersize=5, linestyle="None", label=r"$\theta_{23}^{-}$ = "+str(np.round(theta_lb, 4)))

    plt.plot(theta_ub/np.pi, nll_ub, color="red", marker="s",
             markersize=5, linestyle="None", label=r"$\theta_{23}^{+}$ = "+str(np.round(theta_ub, 4)))

    plt.vlines(theta_min/np.pi, ymin=min(nll_vals)-150, ymax=max(nll_vals)+150, color='black', linestyle='--')
    plt.vlines(theta_lb/np.pi, ymin=min(nll_vals)-150, ymax=max(nll_vals)+150, color='red', linestyle='--')
    plt.vlines(theta_ub/np.pi, ymin=min(nll_vals)-150, ymax=max(nll_vals)+150, color='red', linestyle='--')
    plt.vlines(theta_min/np.pi, ymin=min(nll_vals)-150, ymax=max(nll_vals)+150, color='red', alpha=0.2, linewidth=10)

    # plt.plot(theta_vals/np.pi, npp_vals_2, color='mediumorchid',
    #          label=r"${\Delta}m_{23}^{2}$ ="+str(2.5e-3))

    # plt.xticks(xticks)
    # plt.yticks(yticks)
    plt.legend(loc=(0.65, 0.07), fontsize=16)
    plt.tight_layout()
    plt.savefig('figs/fig6.png')
    plt.show()


def fig7():
    """2D Contour plot"""

    theta_min = 0.8080466225297152
    delta_m_min = 0.0024334819681467263
    # theta and delta m values to use
    theta_vals = np.linspace(theta_min-0.1, theta_min+0.1, 400)
    delta_m_vals = np.linspace(delta_m_min-1e-3, delta_m_min+1e-3, 400)
    X, Y = np.meshgrid(theta_vals, delta_m_vals)

    Z = [[negative_log_likelihood(u=[X[i,j], Y[i,j]])
                 for j in range(len(theta_vals))] for i in tqdm(range(len(theta_vals)))]
    nll_min = min(Z)

    Z = np.array(Z)

    plt.figure()
    plt.contourf(X, Y, Z)
    plt.show()

    ax = plt.figure().add_subplot(projection='3d')

    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
                    alpha=0.3)

    ax.scatter(theta_min, delta_m_min, nll_min, label="Minima")

    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    # ax.contour(X, Y, Z, zdir='z', offset=-100, cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

    ax.set(xlim=(theta_vals[0], theta_vals[-1]), ylim=(delta_m_vals[0], delta_m_vals[-1]), zlim=(1000, 2000),
           xlabel=r'$\theta_{23}$', ylabel=r"$\Delta m_{23}^{2}$", zlabel='Negative log-likelihood')


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


def fig8():
    """"""

    u_vals = []
    for i in range(10):
        u_min = univariate_minimiser(nll2, x0=[np.pi/4, 2.4e-3], xrange=[0.3, 1e-3], max_iter=250, tol=1e-6)
        u_vals.append(u_min)

    print(u_vals)

    nll_vals = [negative_log_likelihood(i) for i in u_vals]

    print(u_vals[nll_vals.index(min(nll_vals))])

    print(np.mean(nll_vals), np.std(nll_vals))

    # print(negative_log_likelihood([0.8088, 2.4e-3]))
    # print(negative_log_likelihood(u_min))




fig7()




