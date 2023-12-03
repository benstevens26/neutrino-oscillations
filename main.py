""" Main

"""

import numpy as np
from numpy import sin
from scipy.special import factorial

# import data
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)
data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)


def oscillation_probability(u, L, E):
    """ Return survival probability of muon neutrino

    :param u: list containing mixing angle and squared mass difference
    :param L: distance travelled between muon neutrino source and detector
    :param E: muon neutrino energy (GeV)
    :return: probability that muon neutrino has not oscillated
    """
    theta = u[0]
    delta_m = u[1]

    return 1 - (sin(2*theta)**2 * sin(1.267*delta_m*L/E)**2)


def get_lambda(u):
    """ Return the oscillated event rate prediction

    :param u: list of parameters [mixing_angle, squared_mass_difference]
    :return: oscillated event rate prediction lambda_i
    """
    bin_midpoints = np.arange(0.025, 10, 0.05)  # using bin midpoints as energies
    p = [oscillation_probability(u, 295, i) for i in bin_midpoints]
    return p * unosc_flux_prediction


def negative_log_likelihood(u):
    """ Return negative log likelihood for given parameters

    :param mixing_angle:
    :param squared_mass_difference:
    :param m: data of muon neutrino events
    :param lambda_i: oscillated event rate prediction
    :return: negative log likelihood of given parameters
    """
    lambda_vals = get_lambda(u)
    m = data
    nll = 2 * np.sum(lambda_vals - (m * np.log(lambda_vals)) + np.log(factorial(m)))
    return nll


def negative_log_likelihood_2(u):
    """ Return negative log likelihood for given parameters

    :param u: [mixing angle, squared mass difference]
    :param m: data of muon neutrino events
    :param lambda_i: oscillated event rate prediction
    :return: negative log likelihood of given parameters
    """
    lambda_vals = get_lambda(u)
    m = data
    nll = 2 * np.sum(lambda_vals - (m * np.log(lambda_vals)))  # without factorial term

    return nll


def parabolic_minimiser(f: callable, x1: float, x2: float, x3: float,
                        tol=1e-6, max_iter=1e6, full_output=False):
    """ Return local minima via parabolic interpolation, where possible

    :param f: function to be minimised
    :param x1: initial value 1
    :param x2: initial value 2
    :param x3: initial value 3
    :param tol: criteria to stop iteration
    :param max_iter: maximum number of iterations
    :param full_output: [x_min, f(x_min)]
    :return: x_min or result after max_iter
    """

    max_iter = int(max_iter)

    for i in range(max_iter):

        y1, y2, y3 = f(x1), f(x2), f(x3)
        x_list = [x1, x2, x3]
        y_list = [y1, y2, y3]
        y_min = min([y_list])

        top = ((x3*x3)-(x2*x2))*y1 + ((x1*x1)-(x3*x3))*y2 + ((x2*x2)-(x1*x1))*y3
        bottom = (x3-x2)*y1 + (x1-x3)*y2 + (x2-x1)*y3

        x4 = 0.5 * (top / bottom)  # minima of parabola
        y4 = f(x4)

        if y4 >= y_min:
            print("Not decreasing, returning")
            return x4

        if abs(y4 - y_min) < tol:
            print("Success in", i, "iterations")

            if full_output:
                return [x4, y4]

            return x4

        x_list.append(x4)
        y_list.append(y4)
        max_index = y_list.index(max(y_list))
        x_list.pop(max_index)

        x1, x2, x3 = x_list

    print("Maximum Iterations Reached")
    min_index = min(y_list)
    x_min = x_list[min_index]
    return x_min


























