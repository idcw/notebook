"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np

from .functions import Function


class Matyas(Function):
    """    
    This function is a simple one-dimensional test function. 
    It is multimodal, with one global minimum, one local minimum and a zero-gradient inflection point. 
    2 dimensional
    (Source: https://www.sfu.ca/~ssurjano/matya.html)
    """

    def __init__(self, lower=-10., upper=10., pi_function=False, m_function=False, minimize=True):
        """
        Initialize the function. 
        Call to its parent class and store the name of the optimization function.
        """
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = 2
        self.name = "Matyas"
        super(self.__class__, self).__init__("Matyas")

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
        super(self.__class__, self)._check(len(population) == 2,
                                           self.name + ' function can only be evaluated with a chromosome size of 2')

        # Make syntax closer to the source
        x1 = float(population[0])
        x2 = float(population[1])

        # Return its value
        return 0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2 if self.minimize else -(
            0.26 * (x1 ** 2 + x2 ** 2) - 0.48 * x1 * x2)

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
