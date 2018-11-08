"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np

from .functions import Function


class Forrester(Function):
    """    
    This function is a simple one-dimensional test function. 
    It is multimodal, with one global minimum, one local minimum and a zero-gradient inflection point. 
    (Source: https://www.sfu.ca/~ssurjano/forretal08.html)
    """

    def __init__(self, lower=-0., upper=1., pi_function=False, m_function=False, minimize=True):
        """
        """
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = 1
        self.name = "Forrester"
        super(self.__class__, self).__init__("Forrester")

    def evaluate(self, population):
        """
        """
        # Check the var type of the population
        super(self.__class__, self)._check(str(type(population)) == "<type 'numpy.ndarray'>",
                                           'The population has to be a numpy array')

        # Case of matrix
        if len(population.shape) == 2:
            return np.apply_along_axis(self.evaluate, 1, population)

        # ensure population is 2 dimensional
        super(self.__class__, self)._check(len(population) == 1,
                                           self.name + ' function can only be evaluated with a chromosome size of 1')

        # Make syntax closer to the source
        x = float(population[0])

        # Return its value
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4) if self.minimize else -((6 * x - 2) ** 2 * np.sin(12 * x - 4))

    def plot(self, d3=False, lower=0, upper=1, samples=1000):
        """
        Makes a 2d plot using the parent class.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            print(self.name + " function is 1 dimensional, therefore it cannot have a 3d plot")
        else:
            super(self.__class__, self).plot(lower, upper, samples)
