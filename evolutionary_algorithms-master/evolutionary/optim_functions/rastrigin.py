"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: April - 2017
"""

import numpy as np

from .functions import Function


class Rastrigin(Function):
    """

    """

    def __init__(self, lower=-10., upper=10., pi_function=False, m_function=False, minimize=True):
        """
        Initialize the function.
        Call to its parent class and store the name of the optimization function.
        """
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = None
        self.name = "Rastrigin"
        self.pi_function = pi_function
        self.m_function = m_function
        if self.pi_function and self.m_function:
            super(self.__class__, self).__init__("pi-m-rastrigin")
        elif self.pi_function:
            super(self.__class__, self).__init__("pi-rastrigin")
        elif self.m_function:
            super(self.__class__, self).__init__("m-rastrigin")
        else:
            super(self.__class__, self).__init__("rastrigin")

    def evaluate(self, population):
        """
        """

        # Check the var type of the population
        super(self.__class__, self)._check(str(type(population)) == "<type 'numpy.ndarray'>",
                                           'The population has to be a numpy array')

        # Case of matrix
        if len(population.shape) == 2:
            return np.apply_along_axis(self.evaluate, 1, population)

        aux_population = population.copy()

        if self.m_function:
            aux_population = super(self.__class__, self).get_m_population(aux_population)

        aux_population = aux_population - np.ones(len(aux_population)) * np.pi if self.pi_function else aux_population

        # Calculate the sums over the population
        sum_1 = np.sum(aux_population ** 2)
        sum_2 = np.sum(np.cos(aux_population * np.pi * 2))

        # Return the value of the function
        res = 10 * len(aux_population) + sum_1 - 10 * sum_2
        return res if self.minimize else -res

    def plot(self, d3=True, samples=1000):
        """
        Makes a 2d plot using the parent class.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            super(self.__class__, self).plot3d(self.lower, self.upper, samples)
        else:
            super(self.__class__, self).plot(self.lower, self.upper, samples)
