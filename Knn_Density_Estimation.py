# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21

import numpy as np

from Density_Estimation import Density_Estimation

""" Knn_Density_Estimation class."""
class Knn_Density_Estimation(Density_Estimation):
    # Constructor
    def __init__(self, x_train: np.ndarray):
        """ Constructs a Knn density estimation object.
            :param x_train: The sample points.
            :type x_train: numpy.ndarray
            :return: None
        """
        Density_Estimation.__init__(self, x_train)
    # Methods
    def distance(self, x1, x2):
        """ Computes the Euclidean distance between two points.
            :param x1: The first point.
            :type x1: float
            :param x2: The second point.
            :type x2: float
            :return: The Euclidean distance between the two points.
            :rtype: float
        """
        return np.sqrt((x1 - x2) ** 2)
    def get_k_nearest_neighbors(self, x_test, k):
        """ Finds the k nearest neighbors of a test point.
            :param x_test: The test point.
            :type x_test: float
            :param k: The number of nearest neighbors.
            :type k: int
            :return: The distances of the k nearest neighbors.
            :rtype: numpy.ndarray
        """
        # Computes the distance from the test point to each training point
        differences = self.get_x_train() - x_test
        distances = np.sqrt(differences ** 2)

        # Sorts the distances from smallest to largest
        sorted_distances = np.sort(distances)

        # Returns the first k distances
        return sorted_distances[:k]
    def estimate_density(self, x_test, k):
        """ Estimates the density at a test point using the k-nearest neighbors.
            :param x_test: The test point.
            :type x_test: float
            :param k: The number of nearest neighbors.
            :type k: int
            :return: The estimated density at the test point.
            :rtype: float
        """
        # Finds the k nearest neighbors
        neighbors = self.get_k_nearest_neighbors(x_test, k)

        # Distance to the k-th nearest neighbor
        radius = neighbors[-1]

        # In 1D, the volume is the length of the interval
        volume = 2 * radius

        # Number of sample points
        num_of_samples = len(self.get_x_train())

        # Avoid division by zero
        if volume == 0:
            return np.inf  # Return infinity if the volume is zero

        # Computes the Knn density estimate
        density = k / (num_of_samples * volume)

        return density