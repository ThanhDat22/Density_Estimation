# Created by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-09-21

# Last modified by Thanh Dat Nguyen (tnrbf@umsystem.edu) on 2026-10-01

from Gaussian_Density import Gaussian_Density
from Knn_Density_Estimation import Knn_Density_Estimation
from Mixture_Density import Mixture_Density
from Kernel_Density_Estimation import Kernel_Density_Estimation

import numpy as np
import matplotlib.pyplot as plt

def plot_density(x_train, x_values, true_density, estimated_density, title):
    """ Plots the sample points, true density, and estimated density.
        :param x_train: The sample points.
        :type x_train: numpy.ndarray
        :param x_values: The x values for plotting.
        :type x_values: numpy.ndarray
        :param true_density: The true density values.
        :type true_density: numpy.ndarray
        :param estimated_density: The estimated density values.
        :type estimated_density: numpy.ndarray
        :param title: The title of the plot.
        :type title: str
        :return: None
    """
    # Creates a new figure
    plt.figure()

    # Plots the true density
    plt.plot(x_values, true_density, label='True (unknown) Density', color='blue')

    # Plots the estimated density
    plt.plot(x_values, estimated_density, label='Estimated Density', color='orange')

    # Plots the sample points
    plt.scatter(x_train, np.zeros_like(x_train), label='Sample Points', color='black', marker='o')

    # Sets the figure information
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
def plot_histogram(x_train, x_values, true_density, num_bins, title):
    """ Plots the histogram density estimate and true density.
        :param x_train: The sample points.
        :type x_train: numpy.ndarray
        :param x_values: The x values for plotting.
        :type x_values: numpy.ndarray
        :param true_density: The true density values.
        :type true_density: numpy.ndarray
        :param num_bins: The number of bins for the histogram.
        :type num_bins: int
        :param title: The title of the plot.
        :type title: str
        :return: None
    """
    # Creates a new figure
    plt.figure()

    # Plots the histogram density estimate
    plt.hist(
        x_train, 
        bins=num_bins, 
        density=True, 
        alpha=0.7, label='Histogram Density Estimate', 
        color='orange'
        )
    # Plots the true density
    plt.plot(x_values, true_density, label='True (unknown) Density', color='blue')

    # Sets the figure information
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
def Q1(cases, gaussian, x_values, true_density):
    """ Performs the density estimation experiment for a given case.
        :param cases: The case number list of number of sample points and neighbors (N, K).
        :type cases: list of tuples
        :param gaussian: The Gaussian density object.
        :type gaussian: Gaussian_Density
        :param x_values: The x values for plotting.
        :type x_values: numpy.ndarray
        :param true_density: The true density values.
        :type true_density: numpy.ndarray
        :return: None
    """
    
    for n, k in cases:
        # Generates N sample points
        x_train = gaussian.generate_samples(n)

        # Creates the Knn estimator
        knn_estimator = Knn_Density_Estimation(x_train)

        # Calculates the estimated density
        estimated_density = []

        for x in x_values:
            density = knn_estimator.estimate_density(x, k)
            estimated_density.append(density)
        estimated_density = np.array(estimated_density)

        # Plots the results
        print(f'Plotting Gaussian KNN Density Estimation for N={n}, K={k}')
        plot_density(
            x_train, 
            x_values, 
            true_density, 
            estimated_density, 
            f'KNN Density Estimation: \n'
            f'Gaussian \nK={k}, N={n}'
            )
        plt.savefig(f'Gaussian KNN_Density_Estimation_N{n}_K{k}.png')
        plt.close()
    print(40*'-')

def Q2(cases, mixture, x_values, true_density):
    """ Performs KNN density estimation on the mixture distribution.
        :param cases: The list of number of sample points and neighbors (N, K).
        :type cases: list of tuples
        :param mixture: The mixture density object.
        :type mixture: Mixture_Density
        :param x_values: The x values for plotting.
        :type x_values: numpy.ndarray
        :param true_density: The true density values.
        :type true_density: numpy.ndarray
        :return: None
    """
    for n, k in cases:
        # Generates N sample points
        x_train = mixture.generate_samples(n)

        # Creates the Knn estimator
        knn_estimator = Knn_Density_Estimation(x_train)

        # Calculates the estimated density
        estimated_density = []

        for x in x_values:
            density = knn_estimator.estimate_density(x, k)
            estimated_density.append(density)
        estimated_density = np.array(estimated_density)

        # Plots the results
        print(f'Plotting Mixture KNN Density Estimation for N={n}, K={k}')
        plot_density(
            x_train, 
            x_values, 
            true_density, 
            estimated_density, 
            f'KNN Density Estimation: \n'
            f'Mixture f(x) = 0.4f_1(x) + 0.6f_2(x) \n'
            f'K={k}, N={n}'
            )
        plt.savefig(f'Mixture KNN_Density_Estimation_N{n}_K{k}.png')
        plt.close()
    print(40*'-')
def Q3(N, sigma, mixture, x_values, true_density):
    """ Performs Gaussian kernel density estimation on the mixture distribution.
        :param N: The sample size.
        :type N: list of int
        :param sigma: The standard deviation (bandwidth) of the Gaussian kernel.
        :type sigma: float
        :param mixture: The mixture density object.
        :type mixture: Mixture_Density
        :param x_values: The x values for plotting.
        :type x_values: numpy.ndarray
        :param true_density: The true density values.
        :type true_density: numpy.ndarray
        :return: None
    """
    for n in N:
        # Generates N sample points
        x_train = mixture.generate_samples(n)

        # Creates the Gaussian KDE estimator
        kde_estimator = Kernel_Density_Estimation(x_train, sigma)

        # Calculates the estimated density
        estimated_density = []

        for x in x_values:
            density = kde_estimator.estimate_density(x)
            estimated_density.append(density)
        estimated_density = np.array(estimated_density)

        # Plots the results
        print(f'Plotting Gaussian Kernel Density Estimation for N={n}, sigma={sigma}')
        plot_density(
            x_train, 
            x_values, 
            true_density, 
            estimated_density, 
            f'Gaussian Kernel Density Estimation: \n'
            f'Mixture f(x) = 0.4f_1(x) + 0.6f_2(x) \n'
            f'Sample size={n}, sigma={sigma}'
        )
        plt.savefig(f'Gaussian_Kernel_Density_Estimation_N{n}_sigma{sigma}.png')
        plt.close()
    print(40*'-')
def Q4(N, bins, mixture, x_values, true_density):
    """ Performs histogram density estimation on the mixture distribution.
        :param N: The sample size.
        :type N: list of int
        :param bins: The number of bins for the histogram.
        :type bins: list of int
        :param mixture: The mixture density object.
        :type mixture: Mixture_Density
        :param x_values: The x values for plotting.
        :type x_values: numpy.ndarray
        :param true_density: The true density values.
        :type true_density: numpy.ndarray
        :return: None
    """
    for n in N:
        # Generates N sample points
        x_train = mixture.generate_samples(n)

        for num_bins in bins:
            # Plots the histogram density estimate
            print(f'Plotting histogram density estimation for N={n}, bins={num_bins}')
            plot_histogram(
                x_train, 
                x_values, 
                true_density, 
                num_bins, 
                f'Histogram Density Estimation: \n'
                f'Mixture f(x) = 0.4f_1(x) + 0.6f_2(x) \n'
                f'Sample size={n}, Number of bins={num_bins}'
            )
            plt.savefig(f'Histogram_Density_Estimation_N{n}_bins{num_bins}.png')
            plt.close()
    print(40*'-')
def main():
    # Sets the random seed for reproducibility
    np.random.seed(0) # For reproducibility

    # Each case is (N, K)
    cases = [(1, 1), (10, 2), (10, 10), (1000, 10), (1000, 100), (50000, 50000)]

    # Q1: Gaussian Distribution
    # Creates the Gaussian density object
    gaussian = Gaussian_Density(1.0, 2.0) # mean=1.0, variance=2.0

    # Generates 10,000 linearly spaced x values
    x_values_gaussian = np.linspace(-7, 9, 10000)

    # Calculates the true Gaussian density
    true_density_gaussian = gaussian.density(x_values_gaussian)

    # Calls the Q1 function
    print("Running Q1: Gaussian Density Estimation")
    Q1(cases, gaussian, x_values_gaussian, true_density_gaussian)

    # Q2: Mixture Distribution
    # Creates the mixture density object
    mixture = Mixture_Density(0.4, 0.6, 2.0, 3.0, 15.0, 20.0)

    # Generates 10,000 linearly spaced x values
    x_values_mixture = np.linspace(0, 22, 10000)

    # Calculates the true mixture density
    true_density_mixture = mixture.density(x_values_mixture)
    # Calls the Q2 function
    print("Running Q2: Mixture Density Estimation")
    Q2(cases, mixture, x_values_mixture, true_density_mixture)

    # Q3: Gaussian Kernel Density Estimation
    N = [1, 2, 5, 10, 50, 100, 1000] # Sample sizes
    sigma = 1.0 # Bandwidth for Gaussian kernel

    # Calls the Q3 function
    print("Running Q3: Gaussian Kernel Density Estimation")
    Q3(N, sigma, mixture, x_values_mixture, true_density_mixture)

    # Q4: Histogram Density Estimation
    N_histogram = [5, 10, 500] # Sample sizes
    bins = [2, 5, 20] # Number of bins

    # Calls the Q4 function
    print("Running Q4: Histogram Density Estimation")
    Q4(N_histogram, bins, mixture, x_values_mixture, true_density_mixture)

    print("Density estimation experiments completed. Please check the generated plots for results.")
    plt.show()

if __name__ == "__main__":
    main()