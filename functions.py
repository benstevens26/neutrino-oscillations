""" Functions module

"""

import numpy as np
from numpy import sin
from scipy.special import factorial

# unoscillated prediction
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)


def oscillation_probability(u, L, E):
    """ Return probability that muon neutrino has not oscillated

    :param u: 2-dimensional array containing mixing angle and squared mass difference
    :param L: distance travelled between muon neutrino source and detector
    :param E: energy of muon neutrino
    :return: probability that muon neutrino has not oscillated
    """
    theta = u[0]
    delta_m = u[1]

    return 1 - (sin(2*theta)**2 * sin(1.267*delta_m*L/E)**2)

def get_lambda(u, L=295):
    """ Return the oscillated event rate prediction

    :param u: list of parameters [mixing_angle, squared_mass_difference]
    :return: oscillated event rate prediction lambda_i
    """

    bin_midpoints = np.arange(0.025, 10, 0.05)
    p = [oscillation_probability(u, L, i) for i in bin_midpoints]
    return p * unosc_flux_prediction


def negative_log_likelihood(u):
    """ Return negative log likelihood for given parameters

    :param u: list of parameters [mixing_angle, squared_mass_difference]
    :param m: data of muon neutrino events
    :param lambda_i: oscillated event rate prediction
    :return: negative log likelihood of given parameters
    """
    lambda_vals = get_lambda(u)
    m = unosc_flux_prediction
    NLL = 2 * np.sum(lambda_vals - (m * np.log(lambda_vals)) + np.log(factorial(m)))

    return NLL


