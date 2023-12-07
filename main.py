""" Main

"""

import numpy as np
from numpy import sin
from scipy.special import factorial
from tqdm import tqdm


# import data
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)
data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)

# constant of experiment
L = 295


def oscillation_probability(u, E):
    """ Return survival probability of muon neutrino

    :param u: list of parameters [mixing_angle (rad), squared_mass_difference (eV^2)]
    :param E: muon neutrino energy (GeV)
    :return: muon neutrino survival probability
    """

    return 1 - (sin(2*u[0])**2 * sin(1.267*u[1]*L/E)**2)


def get_lambda(u):
    """ Return the oscillated event rate prediction (lambda)

    :param u: list of parameters [mixing_angle, squared_mass_difference, cross-section factor]
    :return: oscillated event rate prediction (lambda)
    """
    bin_midpoints = np.arange(0.025, 10, 0.05)  # using bin midpoints as energies
    p = [oscillation_probability(u, i) for i in bin_midpoints]

    if len(u) == 3:
        return p * unosc_flux_prediction * u[2] * bin_midpoints

    return p * unosc_flux_prediction



def negative_log_likelihood(u):
    """ Return negative log likelihood for given parameters

    :param u: list of parameters [mixing_angle, squared_mass_difference]
    :return: negative log likelihood of given parameters
    """
    lambda_vals = get_lambda(u)
    m = data
    nll = 2 * np.sum(lambda_vals - (m * np.log(lambda_vals)) + np.log(factorial(m)))
    return nll


def nll2(u):
    """ Alternative nll function, ONLY for changes in nll

    :param u: list of parameters [mixing_angle, squared_mass_difference]
    :return: negative log likelihood
    """

    lambda_vals = get_lambda(u)
    m = data
    nll = 2 * np.sum(lambda_vals - (m * np.log(lambda_vals)))
    return nll


def nll_error(u_min):
    """ Return error on nll_min

    :param u_min: u minima
    :return: (theta_lb, theta_ub)
    """

    theta_min = u_min[0]
    delta_m_min = u_min[1]
    nll_min = negative_log_likelihood(u_min)

    epsilon = 0
    while (negative_log_likelihood([theta_min + epsilon, delta_m_min]) - nll_min) < 1.0:
        epsilon += 1e-5

    theta_ub = theta_min + epsilon

    epsilon = 0
    while (negative_log_likelihood([theta_min, delta_m_min + epsilon]) - nll_min) < 1.0:
        epsilon += 1e-5

    delta_m_ub = delta_m_min + epsilon

    epsilon = 0
    while (negative_log_likelihood([theta_min - epsilon, delta_m_min]) - nll_min) < 1.0:
        epsilon += 1e-5

    theta_lb = theta_min - epsilon

    epsilon = 0
    while (negative_log_likelihood([theta_min, delta_m_min - epsilon]) - nll_min) < 1.0:
        epsilon += 1e-5

    delta_m_lb = delta_m_min - epsilon

    return theta_lb, theta_ub, delta_m_lb, delta_m_ub


def gen_points(domain):
    """Generate 3 points within the domain"""
    domain_points = np.linspace(domain[0], domain[1], 1000)
    x1 = domain_points[np.random.randint(0, 333)]
    x2 = domain_points[np.random.randint(333, 667)]
    x3 = domain_points[np.random.randint(666, 999)]
    return x1, x2, x3


def parabolic_minimiser(f, x1: float, x2: float, x3: float, args=(),
                        tol=1e-6, max_iter=1e6, full_output=False):
    """ Return local minima via parabolic interpolation, where possible

    :param f: function to be minimised
    :param x1: initial guess 1
    :param x2: initial value 2
    :param x3: initial value 3
    :param args: tuple of arguments to be passed to the function f
    :param tol: criteria to stop iteration
    :param max_iter: maximum number of iterations
    :param full_output: returns [x, f(x)] for each interpolated minima
    :return: x_min or result after max_iter
    """

    max_iter = int(max_iter)
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0")

    points = []
    x_list = []
    y_list = []

    for i in range(max_iter):

        y1, y2, y3 = f(x1, *args), f(x2, *args), f(x3, *args)
        x_list = [x1, x2, x3]
        y_list = [y1, y2, y3]
        y_min = min(y_list)
        x_min = x_list[y_list.index(y_min)]

        top = ((x3*x3)-(x2*x2))*y1 + ((x1*x1)-(x3*x3))*y2 + ((x2*x2)-(x1*x1))*y3
        bottom = (x3-x2)*y1 + (x1-x3)*y2 + (x2-x1)*y3

        x4 = 0.5 * (top / bottom)  # x4 is minima of parabola
        y4 = f(x4, *args)
        points.append([x4, y4])

        x_list.append(x4)
        y_list.append(y4)

        if x4 == x1 or x4 == x2 or x4 == x3:
            print("Degenerate case encountered, regenerating initial guesses")
            x1, x2, x3 = gen_points([x1, x3])
            continue

        if abs(x4 - x_min) < tol:
            print("Success in", i+1, "iteration(s)")

            if full_output:
                curvature = (2*y1)/((x2-x1)*(x3-x1)) - (2*y2)/((x2-x1)*(x3-x2)) + (2*y3)/((x3-x1)*(x3-x2))
                return x4, points, curvature

            return x4

        max_index = y_list.index(max(y_list))
        x_list.pop(max_index)

        x1, x2, x3 = x_list

    print(max_iter,"iterations reached")
    min_index = y_list.index(min(y_list))
    x_min = x_list[min_index]

    if full_output:
        return points

    return x_min


def parabolic_minimiser_nd(f, axis, x0, xrange: float, args=(),
                          tol=1e-6, max_iter=1e6, full_output=False):
    """ Return local minima via parabolic interpolation along a given axis

    :param f: function to be minimised
    :param axis: axis (index i) on which to minimise f(x_0, x_1, ... x_i)
    :param x0: initial guess of minima [x0_min, x1_min ... xi_min]
    :param xrange: initial range to form initial guesses
    :param args: tuple of arguments to be passed to the function f
    :param tol: criteria to stop iteration
    :param max_iter: maximum number of iterations
    :param full_output: returns [x, f(x)] for each interpolated minima
    :return: x_min or result after max_iter along axis x_i
    """

    max_iter = int(max_iter)
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0")

    points = []
    x_list = []
    y_list = []

    x1 = x0.copy()
    x2 = x0.copy()
    x3 = x0.copy()

    a, b, c = gen_points([x0[axis] - xrange/2, x0[axis] + xrange/2])

    x1[axis] = a
    x2[axis] = b
    x3[axis] = c

    initial_vals = [x1, x2, x3]

    for i in range(max_iter):
        y1, y2, y3 = f(x1, *args), f(x2, *args), f(x3, *args)
        x_list = [x1, x2, x3]
        y_list = [y1, y2, y3]
        y_min = min(y_list)
        x_min = x_list[y_list.index(y_min)]

        top = (((x3[axis]*x3[axis])-(x2[axis]*x2[axis]))*y1
               + ((x1[axis]*x1[axis])-(x3[axis]*x3[axis]))*y2
               + ((x2[axis]*x2[axis])-(x1[axis]*x1[axis]))*y3)
        bottom = ((x3[axis]-x2[axis])*y1 + (x1[axis]-x3[axis])*y2
                  + (x2[axis]-x1[axis])*y3)

        x4 = x1.copy()  # is only changing along the axis that is being minimised
        x4[axis] = 0.5 * (top / bottom)  # x4 is minima of parabola in [axis] index

        y4 = f(x4, *args)
        points.append([x4, y4])

        x_list.append(x4)
        y_list.append(y4)

        if x4[axis] == x1[axis] or x4[axis] == x2[axis] or x4[axis] == x3[axis]:
            # print("Degenerate case encountered, regenerating initial guesses")
            a, b, c = gen_points([x1[axis], x3[axis]])
            x1[axis] = a
            x2[axis] = b
            x3[axis] = c
            continue

        if abs(x4[axis] - x_min[axis]) < tol:
            # print("Success in", i+1, "iteration(s)")

            # if full_output:
            #     curvature = (2*y1)/((x2-x1)*(x3-x1)) - (2*y2)/((x2-x1)*(x3-x2)) + (2*y3)/((x3-x1)*(x3-x2))
            #     return x4, points, curvature

            return x4

        max_index = y_list.index(max(y_list))
        x_list.pop(max_index)

        x1, x2, x3 = x_list

    # print(max_iter,"iterations reached")
    min_index = y_list.index(min(y_list))
    x_min = x_list[min_index]

    if full_output:
        return points

    return x_min


def univariate_minimiser(f, x0, xrange, args=(),
                         tol=1e-6, max_iter=1e6):
    """ Return minima by successive parabolic minimisation along each axis """

    max_iter = int(max_iter)
    if max_iter <= 0:
        raise ValueError("max_iter must be greater than 0")

    for i in tqdm(range(max_iter)):
        best_min = x0.copy()

        for j in range(len(x0)):

            theta_min, delta_m = parabolic_minimiser_nd(f, axis=j, x0=x0, xrange=xrange[j],
                                                        args=args, tol=tol, max_iter=max_iter,
                                                        full_output=False)
            x0 = [theta_min, delta_m]

        if np.linalg.norm(np.asarray(x0)-np.asarray(best_min)) < tol:

            # print("success in", i, "cycles")
            return x0

    print("Max iterations reached")
    return x0


def gradient_descent(f, x0, h, alpha, args=(), tol=1e-6, max_iter=1e6, full_output=False):
    """ Return minima of f using gradient descent

    :param f: objective function to be minimised
    :param x0: initial guess [x0_1, x0_2 ... x0_n]
    :param h: width used for forward difference scheme
    :param alpha: learning rate
    :param args: arguments to be passed to the objective function
    :param tol: convergence criteria to stop iteration
    :param max_iter: maximum number of iterations
    :param full_output: return (x_min, iterations, points)
    :return: x_min
    """

    max_iter = int(max_iter)
    dims = len(x0)
    x0 = np.array(x0, dtype=np.float64)
    grad = np.zeros(dims, dtype=np.float64)
    points = []

    for i in tqdm(range(max_iter)):

        for j in range(dims):
            x0_plus_h = x0.copy()
            x0_plus_h[j] += h
            grad[j] = (f(x0_plus_h, *args) - f(x0, *args)) / h

        x1 = x0 - (alpha * grad)
        points.append(x1)

        if np.linalg.norm(x1 - x0) < tol and abs(negative_log_likelihood(x1)-negative_log_likelihood(x0)) < tol:
            if full_output:
                return x1, f(x1), i+1, points
            return x1

        x0 = x1.copy()

    return x1








































