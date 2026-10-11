# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-23

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-30

import numpy as np

class Mixture_Density:
    # Constructor
    def __init__(self, weight1=0.4, weight2=0.6, f1_low=2.0, f1_high=3.0, f2_low=15.0, f2_high=20.0):
        """ Constructs a mixture density object.
            :param weight1: The weight of the first density function.
            :type weight1: float
            :param weight2: The weight of the second density function.
            :type weight2: float
            :param f1_low: The lower bound of the first density function.
            :type f1_low: float
            :param f1_high: The upper bound of the first density function.
            :type f1_high: float
            :param f2_low: The lower bound of the second density function.
            :type f2_low: float
            :param f2_high: The upper bound of the second density function.
            :type f2_high: float
            :return: None
        """

        # Data fields for f(x) = weight1 * f1(x) + weight2 * f2(x)
        self._weight1 = weight1
        self._weight2 = weight2
        self._f1_low = f1_low
        self._f1_high = f1_high
        self._f2_low = f2_low
        self._f2_high = f2_high
    # Getters
    def get_weight1(self):
        """ Returns the weight of the first density function.
            :return: The weight of the first density function.
            :rtype: float
        """
        return self._weight1
    def get_weight2(self):
        """ Returns the weight of the second density function.
            :return: The weight of the second density function.
            :rtype: float
        """
        return self._weight2
    def get_f1_low(self):
        """ Returns the lower bound of the first density function.
            :return: The lower bound of the first density function.
            :rtype: float
        """
        return self._f1_low
    def get_f1_high(self):
        """ Returns the upper bound of the first density function.
            :return: The upper bound of the first density function.
            :rtype: float
        """
        return self._f1_high
    def get_f2_low(self):
        """ Returns the lower bound of the second density function.
            :return: The lower bound of the second density function.
            :rtype: float
        """
        return self._f2_low
    def get_f2_high(self):
        """ Returns the upper bound of the second density function.
            :return: The upper bound of the second density function.
            :rtype: float
        """
        return self._f2_high
    # Methods
    def generate_samples(self, n):
        """ Generates n iid samples from the mixture distribution.
            :param n: The number of samples.
            :type n: int
            :return: The generated sample points.
            :rtype: numpy.ndarray
        """
        # Determines which distribution each sample comes from
        choices = np.random.random(n)

        samples = np.empty(n)

        # Generates samples from f1
        first = choices < self._weight1
        samples[first] = np.random.uniform(
            self.get_f1_low(),
            self.get_f1_high(),
            np.sum(first)
        )

        # Generates samples from f2
        second = ~first
        samples[second] = np.random.uniform(
            self.get_f2_low(),
            self.get_f2_high(),
            np.sum(second)
        )
        return samples
    def density(self, x):
        """ Computes the true mixture density at given x values.
            :param x: The x values.
            :type x: float
            :return: The density values.
            :rtype: float
        """
        x = np.asarray(x) # Converts x to a numpy array for element-wise operations

        density = np.zeros_like(x, dtype=float)

        # f1(x) = 1 / (f1_high - f1_low) for f1_low <= x <= f1_high, else 0
        f1_density = 1 / (self.get_f1_high() - self.get_f1_low())

        # f2(x) = 1 / (f2_high - f2_low) for f2_low <= x <= f2_high, else 0
        f2_density = 1 / (self.get_f2_high() - self.get_f2_low())

        # Computes the mixture density
        first = (x >= self.get_f1_low()) & (x <= self.get_f1_high())
        density[first] = self.get_weight1() * f1_density

        second = (x >= self.get_f2_low()) & (x <= self.get_f2_high())
        density[second] = self.get_weight2() * f2_density

        return density