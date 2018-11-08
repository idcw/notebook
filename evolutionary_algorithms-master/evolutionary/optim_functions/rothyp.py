"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np

from .functions import Function


class Rothyp(Function):
    """    
    The Rotated Hyper-Ellipsoid function is continuous, convex and unimodal. 
    It is an extension of the Axis Parallel Hyper-Ellipsoid function, 
    also referred to as the Sum Squares function.
    d dimensional.
    Global minima at x = (0,..,0)
    (Source: https://www.sfu.ca/~ssurjano/rothyp.html)
    """

    def __init__(self, lower=-65.536, upper=65.536, pi_function=False, m_function=False, minimize=True):
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = None
        self.name = "Rothyp"
        super(self.__class__, self).__init__("Rothyp")

    def evaluate(self, population):
        """
        Returns the fitness of a population using the Rothyp function.
        Population has to be a numpy array for this method to work.
        """
        # Check the var type of the population
        super(self.__class__, self)._check(str(type(population)) == "<type 'numpy.ndarray'>",
                                           'The population has to be a numpy array')

        # Case of matrix
        if len(population.shape) == 2:
            return np.apply_along_axis(self.evaluate, 1, population)

        # Initialize vars
        res_sum = 0.0

        # Compute the sums on the population
        for i in range(len(population)):
            for j in range(i + 1):
                res_sum += population[j] ** 2

        # Return the function
        return res_sum if self.minimize else -(res_sum)

    def plot(self, d3=True, lower=-65.536, upper=65.536, samples=1000):
        """
        Makes a 2d plot using the parent class.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            super(self.__class__, self).plot3d(lower, upper, samples)
        else:
            super(self.__class__, self).plot(lower, upper, samples)
