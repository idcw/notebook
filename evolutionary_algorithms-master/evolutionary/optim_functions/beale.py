"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np

from .functions import Function


class Beale(Function):
    """
    The Beale function is multimodal, with sharp peaks at the corners of the input domain.
    2 Dimensional
    Global minima x = (3, 0.5)
    (Source: https://www.sfu.ca/~ssurjano/beale.html)
    """

    def __init__(self, lower=-4.5, upper=4.5, pi_function=False, m_function=False, minimize=True):
        """
        Initialize the function. 
        Call to its parent class and store the name of the optimization function.
        """
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = 2
        self.name = "Beale"
        super(self.__class__, self).__init__("Beale")

    def evaluate(self, population):

        # Case of matrix
        if len(population.shape) == 2:
            return np.apply_along_axis(self.evaluate, 1, population)

        # ensure population is 2 dimensiona
        super(self.__class__, self)._check(len(population) == 2,
                                           self.name + ' function can only be evaluated with a chromosome size of 2')

        # Make sintax closer to the source
        x1 = float(population[0])
        x2 = float(population[1])

        # Compute the three parts of the function
        part1 = (1.5 - x1 + x1 * x2) ** 2
        part2 = (2.225 - x1 + x1 * (x2 ** 2)) ** 2
        part3 = (2.625 - x1 + x1 * (x2 ** 3)) ** 2

        # Return the value of the function
        return part1 + part2 + part3 if self.minimize else -(part1 + part2 + part3)

    def plot(self, d3=True, lower=-4.5, upper=4.5, samples=1000):
        """
        Makes a 2d plot using the parent class.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            super(self.__class__, self).plot3d(lower, upper, samples)
        else:
            print(self.name + " function is 2 dimensional, therefore, it cannot be displayed in a 2d plot")
