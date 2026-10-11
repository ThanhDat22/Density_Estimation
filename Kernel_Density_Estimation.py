# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-30

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-30

from Density_Estimation import Density_Estimation

import numpy as np

""" Kernel_Density_Estimation class."""
class Kernel_Density_Estimation(Density_Estimation):
    # Constructor
    def __init__(self, x_train, sigma):
        """ Constructs a kernel density estimation object.
            :param x_train: The sample points.
            :type x_train: numpy.ndarray
            :param sigma: The standard deviation of the Gaussian kernel.
            :type sigma: float
            :return: None
        """
        Density_Estimation.__init__(self, x_train)

        # Data fields
        self._sigma = sigma
    # Getters
    def get_sigma(self):
        """ Returns the standard deviation of the Gaussian kernel.
            :return: The standard deviation of the Gaussian kernel.
            :rtype: float
        """
        return self._sigma
    # Methods
    def gaussian_kernel(self, x):
        """ Computes the Gaussian kernel function.
            :param x: The input value.
            :type x: float
            :return: The value of the Gaussian kernel function at x.
            :rtype: float
        """
        return ((1 / np.sqrt(2 * np.pi)) * np.exp(-1/2 * (x ** 2)))
    def estimate_density(self, x_test):
        """ Estimates the density at a given point using Gaussian kernel density estimation (KDE).
            :param x_test: The query point.
            :type x_test: float
            :return: The estimated density at the query point.
            :rtype: float
        """
        x_train = self.get_x_train()
        n = len(x_train)
        sigma = self.get_sigma()

        # Computes the standardized distances from the query point to each training point
        standardized_distances = (x_test - x_train) / sigma

        # Computes all Gaussian kernel contributions
        kernel_contributions = self.gaussian_kernel(standardized_distances)

        # Computes the estimated density
        estimated_density = np.sum(kernel_contributions) / (n * sigma)

        return estimated_density