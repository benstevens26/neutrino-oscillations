"""
NOTES
-----
This module uses an NLL fit to extract neutrino oscillation parameters
from simulated T2K data. More detail can be found in README.md

NAME
    neutrino_oscillation

FUNCTIONS
    oscillation_probability():
        Return muon neutrino survival probability
    get_lambda():
        Return oscillated rate event prediction
    negative_log_likelihood():
        Return negative log likelihood for given oscillation parameters
    nll2():
        Alternative negative log likelihood function, more efficient for
        changes in nll



"""

import numpy as np
from numpy import sin
from scipy.special import factorial
from scipy.optimize import line_search as ls
from tqdm import tqdm


# import data
unosc_flux_prediction = np.loadtxt("bs521.txt", skiprows=204)
data = np.loadtxt("bs521.txt", skiprows=2, max_rows=200)

# constant of experiment
L = 295


def oscillation_probability(u: list, E: float):
    """
    Return survival probability of muon neutrino

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
        return p * unosc_flux_prediction * abs(u[2]) * bin_midpoints

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


def gradient(f, x, h, args=()):
    """Return gradient of function f at x using a finite difference approximation"""
    dims = len(x)
    grad = np.zeros(dims, dtype=np.float64)
    for i in range(dims):
        x_plus_h = x.copy()
        x_plus_h[i] += h
        x_minus_h = x.copy()
        x_minus_h[i] -= h
        grad[i] = (f(x_plus_h, *args) - f(x_minus_h, *args)) / (2*h)
    return grad


def gradient_callable(f, h=1e-6, args=()):
    """Return gradient function of function f at x using a finite difference approximation"""
    def get_grad(x, args=args):
        grad = gradient(f, x, h=h, args=args)
        return grad

    def grad_callable(x):
        return get_grad(x)

    return grad_callable


def gradient_descent(f, x0, h=1e-8, alpha=1e-8, args=(), tol=1e-6, max_iter=1e5, full_output=False):
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

    for i in range(max_iter):

        grad = gradient(f, x0, h=h, args=args)

        x1 = x0 - (alpha * grad)
        points.append(x1)

        if full_output:
            print("Iteration:", i)
            print("Gradient:", grad)
            print("f(x):", f(x1))

        if np.linalg.norm(f(x1) - f(x0)) < tol:
            if full_output:
                return x1, f(x1), i+1, points
            return x1

        x0 = x1.copy()

    return x1


def line_search(f, x, grad, direction, args=()):
    """Return alpha value that satisfies the Armijo-Goldstein condition"""

    alpha = 1
    c = 0.1
    p = 0.5

    x_guess = x + (alpha * direction)
    f_guess = f(x_guess, *args)
    f_x = f(x, *args)

    while f_guess > (f_x + c * alpha * (grad.T @ direction)):
        alpha = p * alpha
        x_guess = x + (alpha * direction)
        f_guess = f(x_guess)

    return alpha


def update_hessian(hessian, x_change, grad_change):
    """Return updated inverse Hessian approximation via DFP method"""
    term1 = np.outer(x_change, x_change) / np.dot(grad_change, x_change)
    term2_n = (hessian @ np.outer(grad_change, grad_change) @ hessian)
    term2_d = grad_change @ hessian @ grad_change
    H = hessian + term1 - (term2_n / term2_d)
    return H


def update_hessian_2(hessian, x_change, grad_change):
    """Return updated inverse Hessian approximation via BFGS method"""
    I = np.identity(len(x_change))
    roe = 1 / (grad_change.T @ x_change)
    H = ((I - (roe * (x_change @ grad_change.T))) @ hessian @ (I - (roe * (grad_change @ x_change.T)))
         + (roe * x_change @ x_change.T))
    return H


def quasi_newton_minimiser(f, x0, args=(), h=1e-8, tol=1e-6, max_iter=1e5, full_output=False):
    """
    Return minima of f using an implementation of the BFGS method
    :param f: objective function to be minimised
    :param x0: initial guess for parameters
    :param args: extra arguments to be passed to the objective function
    :param h: for finite difference
    :param tol: convergence criteria to stop iteration
    :param max_iter: maximum number of iterations
    :param full_output: return (x_min, iterations, points)
    :return: minima of f x_min
    """
    x0 = np.array(x0, dtype=np.float64)
    dims = len(x0)
    max_iter = int(max_iter)
    points = []
    f_prime = gradient_callable(f, h=h)

    # starting with identity matrix as inverse Hessian
    I = np.identity(dims)
    H = I.copy()
    x = x0.copy()

    for i in range(max_iter):
        points.append(x)
        grad = f_prime(x)

        if np.linalg.norm(grad) < tol:  # convergence
            if full_output:
                return x, f(x, *args), i, points
            return x

        # search direction
        direction = - (H @ grad)

        # using backtracking line search with Armijo-Goldstein conditions
        alpha = line_search(f, x, grad, direction)
        x_new = x + (alpha * direction)

        if np.linalg.norm(x_new - x) < tol:
            if full_output:
                return x, f(x, *args), i, points
            return x

        grad_new = f_prime(x_new)
        x_change = x_new - x
        grad_change = grad_new - grad

        # update hessian via DFP method
        H = update_hessian(H, x_change, grad_change)
        x = x_new

    print("Quasi-Newton maximum iterations reached - did not converge")
    return x


def minimise(f, x0, x_range=None, tol=1e-6, args=(), runs=100, method: str = "quasi-newton"):
    """ Return minima of function f within x_range

    :param f: objective function to be minimised
    :param x0: initial guess for each parameter
    :param x_range: width of search range for each parameter (default: 0.5 * x0 in each parameter)
    :param args: arguments to be passed to the objective function
    :param runs: number of different restarts searching for global minima
    :param method: "gradient-descent" "quasi-newton" "univariate" default is "quasi-newton"
    :return: minima: list of minima, one from each run
    """

    if x_range is None:
        x_range = [n*0.5 for n in x0]

    x0 = np.array(x0, dtype=np.float64)
    x_range = np.array(x_range, dtype=np.float64)
    dims = len(x0)
    initial_guesses = np.zeros(dims)
    minima = []

    if method == "gradient-descent":

        for i in tqdm(range(runs)):
            for j in range(len(x0)):
                initial_guesses[j] = (x0[j] - x_range[j]/2) + (np.random.rand() * x_range[j])

            x_min = gradient_descent(f, x0=initial_guesses, args=args, tol=tol)
            minima.append(x_min)

        return minima

    if method == "quasi-newton":

        for i in tqdm(range(runs)):
            for j in range(len(x0)):
                initial_guesses[j] = (x0[j] - x_range[j]/2) + (np.random.rand() * x_range[j])

            x_min = quasi_newton_minimiser(f, x0=initial_guesses, args=args, tol=tol)
            minima.append(x_min)

        return minima

    if method == "univariate":
        raise NotImplementedError

    return

