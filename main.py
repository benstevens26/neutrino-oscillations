""" Main

Function: oscillation_probability: Return probability that muon neutrino has not oscillated

"""

import numpy as np
from numpy import sin
from scipy.special import factorial

# import data
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)
data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)


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


def parabolic_minimiser(function, a, b, c, tolerance=1e-6, max_iter=100):
    """ find local minimum using parabolic interpolation

    :param function: function to minimise
    :param a, b, c: x values to perform initial interpolation
    :param tolerance: tolerance to provide stopping point
    :param max_iter: maximum number of iterations
    :return: estimate of minimizer x_min
    """
    for i in range(max_iter):
        y_a, y_b, y_c = function(a), function(b), function(c)
        y_min = min([y_a, y_b, y_c])

        top = ((c*c)-(b*b))*y_a + ((a*a)-(c*c))*y_b + ((b*b)-(a*a))*y_c
        bottom = (c-b)*y_a + (a-c)*y_b + (b-a)*y_c

        d = 0.5 * (top / bottom)
        y_d = function(d)

        if abs(y_d - y_min) < tolerance:
            return d

        x_list = [a, b, c, d]
        y_list = [y_a, y_b, y_c, y_d]

        max_index = y_list.index(max(y_list))

        x_list.pop(max_index)

        a, b, c = x_list[0], x_list[1], x_list[2]

    print("Maximum Iterations Reached")
    return d


























