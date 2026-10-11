# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21
 
import numpy as np

""" Density_Estimation class.

    This is a base class for density estimation.
"""
class Density_Estimation:
    # Constructor
    def __init__(self, x_train: np.ndarray):
        """ Constructs a density estimation object.
            :param x_train: The sample points.
            :type x_train: numpy.ndarray
            :return: None
        """
        # Data fields
        self._x_train = x_train
    # Getters
    def get_x_train(self):
        """ Returns the sample points.
            :return: The sample points.
            :rtype: numpy.ndarray
        """
        return self._x_train