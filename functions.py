""" Functions module

"""

import numpy as np

def oscillation_probability(theta, delta_m, length, energy):
    """Return probability that muon neutrino is observed as muon neutrino"""
    return 1 - (np.sin(2*theta)*np.sin(2*theta)*np.sin(1.267*delta_m*length/energy))