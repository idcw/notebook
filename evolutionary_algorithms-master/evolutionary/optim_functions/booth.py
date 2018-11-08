"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np

from .functions import Function


class Booth(Function):
    """
    Global minima at x* = (1,3)
    2 Dimensional
    (Source: https://www.sfu.ca/~ssurjano/griewank.html)
    """

    def __init__(self, lower=-10., upper=10., pi_function=False, m_function=False, minimize=True):
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = None
        self.name = "Booth"
        super(self.__class__, self).__init__("Booth")

    def evaluate(self, population):
        # Check the var type of the population
        super(self.__class__, self)._check(str(type(population)) == "<type 'numpy.ndarray'>",
                                           'The population has to be a numpy array')
        # Case of matrix
        if len(population.shape) == 2:
            return np.apply_along_axis(self.evaluate, 1, population)

        # Make sintax closer to the source
        x1 = float(population[0])
        x2 = float(population[1])

        # part1
        part1 = (x1 + 2 * x2 - 7) ** 2
        part2 = (2 * x1 + x2 - 5) ** 2

        # Return the value of the function
        return part1 + part2 if self.minimize else -(part1 + part2)

    def plot(self, d3=True, lower=-10, upper=10, samples=1000):
        """
        Makes a 2d plot using the parent class.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            super(self.__class__, self).plot3d(lower, upper, samples)
        else:
            print(self.name + " function is 2 dimensional, therefore it cannot have a 2d plot")
