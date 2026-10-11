# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21

import numpy as np

""" Gaussian_Density class."""
class Gaussian_Density:
    # Constructor
    def __init__(self, mean, variance):
        """ Constructs a Gaussian density object.
            :param mean: The mean of the Gaussian distribution.
            :type mean: float
            :param variance: The variance of the Gaussian distribution.
            :type variance: float
            :return: None
        """
        # Data fields
        self._mean = mean
        self._variance = variance
    # Getters
    def get_mean(self):
        """ Returns the mean of the Gaussian distribution.
            :return: The mean of the Gaussian distribution.
            :rtype: float
        """
        return self._mean
    def get_variance(self):
        """ Returns the variance of the Gaussian distribution.
            :return: The variance of the Gaussian distribution.
            :rtype: float
        """
        return self._variance
    # Methods
    def generate_samples(self, n):
        """ Generates n iid samples from the Gaussian distribution.
            :param n: The number of samples.
            :type n: int
            :return: The generated sample points.
            :rtype: numpy.ndarray
        """
        standard_deviation = np.sqrt(self.get_variance())

        samples = np.random.normal(
            loc=self.get_mean(),
            scale=standard_deviation,
            size=n
        )
        return samples
    def density(self, x):
        """ Computes the density of the Gaussian distribution at a given point.
            :param x: The x values.
            :type x: float
            :return: The density of the Gaussian distribution at the given point.
            :rtype: float
        """
        standard_deviation = np.sqrt(self.get_variance())

        density = (
            1 / (standard_deviation * np.sqrt(2 * np.pi))
            * np.exp(
                -0.5
                * ((x - self.get_mean()) / standard_deviation) ** 2
            )
        )
        return density

