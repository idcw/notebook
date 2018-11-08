"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np

from .functions import Function


class Easom(Function):
    """    
    The Easom function has several local minima. 
    It is unimodal, and the global minimum has a small area relative to the search space.
    2 dimensonal
    global minima at (pi,pi)
    (Source: https://www.sfu.ca/~ssurjano/ackley.html)
    """

    def __init__(self, lower=-100., upper=100., pi_function=False, m_function=False, minimize=True):
        """
        Initialize the Easom class
        """
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = 2
        self.name = "Easom"
        super(self.__class__, self).__init__("Easom")

    def evaluate(self, population):
        """
        Returns the fitness of a population using the Easom function.
        Population has to be a numpy array for this method to work.
        """
        # Check the var type of the population
        super(self.__class__, self)._check(str(type(population)) == "<type 'numpy.ndarray'>",
                                           'The population has to be a numpy array')
        # Case of matrix
        if len(population.shape) == 2:
            return np.apply_along_axis(self.evaluate, 1, population)

        # ensure population is 2 dimensional
        super(self.__class__, self)._check(len(population) == 2,
                                           self.name + ' function can only be evaluated with a chromosome size of 2')

        # Make sintax closer to the source
        x1 = population[0]
        x2 = population[1]

        # Return the function
        cost = -np.cos(x1) * np.cos(x2) * np.exp(-(x1 - np.pi) ** 2 - (x2 - np.pi) ** 2)
        return cost if self.minimize else -(cost)

    def plot(self, d3=True, lower=-100, upper=100, samples=1000):
        """
        Makes a 2d/3d (regarding the d3 var) plot using the parent classes.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            super(self.__class__, self).plot3d(lower, upper, samples)
        else:
            super(self.__class__, self).plot(lower, upper, samples)
