"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

from .functions import Function
import numpy as np


#################################################################
class Ackley(Function):
    """    
    The Ackley function is widely used for testing optimization algorithms.
    It is characterized by a nearly flat outer region, and a large hole at the centre. 
    The function poses a risk for optimization algorithms, particularly hillclimbing algorithms, 
    to be trapped in one of its many local minima. 
    (Source: https://www.sfu.ca/~ssurjano/ackley.html)
    """

    def __init__(self, lower=-32., upper=32., pi_function=False, m_function=False, a=False, b=False, c=False,
                 minimize=True):
        """
        Initialize the Ackley class with the values of a, b and c
        """
        self.a = 20 if not a else a
        self.b = 0.2 if not b else b
        self.c = 2 * np.pi if not c else c
        self.lower = lower
        self.upper = upper
        self.minimize = minimize
        self.dim = None
        self.name = "Ackley"
        self.pi_function = pi_function
        self.m_function = m_function
        if self.pi_function and self.m_function:
            super(self.__class__, self).__init__("pi-m-ackley")
        elif self.pi_function:
            super(self.__class__, self).__init__("pi-ackley")
        elif self.m_function:
            super(self.__class__, self).__init__("m-ackley")
        else:
            super(self.__class__, self).__init__("ackley")

    def evaluate(self, population):
        """
        Returns the fitness of a population using the Ackley function.
        Population has to be a numpy array for this method to work.
        """
        # Check the var type of the population
        assert str(type(population)) == "<type 'numpy.ndarray'>" and len(population) > 0

        if len(population.shape) > 1:
            return np.apply_along_axis(self.evaluate, 1, population)

        aux_population = population.copy()

        if self.m_function:
            aux_population = super(self.__class__, self).get_m_population(aux_population)

        aux_population = aux_population - np.ones(len(aux_population)) * np.pi if self.pi_function else aux_population

        # Initialize vars
        firstSum = 0.0
        secondSum = 0.0
        n = float(len(aux_population))

        # Compute the sums on the population
        for i in aux_population:
            firstSum += i ** 2.0
            secondSum += np.cos(self.c * i)

        # Return the function
        if self.minimize:
            return -self.a * np.exp(-self.b * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + self.a + np.e
        else:
            return -(-self.a * np.exp(-self.b * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + self.a + np.e)

    def plot(self, d3=True, samples=1000):
        """
        Makes a 2d/3d (regarding the d3 var) plot using the parent classes.
        It creates an array of samples between the upper and lower bounds
        and compute its fitness which will be plotted together.
        """
        if d3:
            super(self.__class__, self).plot3d(self.lower, self.upper, samples)
        else:
            super(self.__class__, self).plot(self.lower, self.upper, samples)
