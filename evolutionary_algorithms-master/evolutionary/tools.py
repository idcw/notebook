"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""
from __future__ import division
import ga_tools as ga_tools
import numpy as np


class Population(object):
    """
    Chromosome is an object that has the chromosomes and sigma.
    It is used in evolutionary algorithms.
    The chromosomes is the n-dimensional array that will be evaluated by the fitness function.
    Regarding the evolutionary-strategy followed sigma will be None or an array of values
    """

    def __init__(self, chromosomes=None, sigma=None, delta=None, alpha=None, s=None, space_s=None):
        """

        :param chromosomes:
        :param sigma: Parameter for evolutionary strategies
        :param delta: Paremeter to make a grid in the N-dimensional space and discretize the search space
        :param alpha: real part of the chromosomes in GGA(grid-based Genetic Algorithm)
        :param s: integer part of the chromosomes in GGA
        """
        self.chromosomes = chromosomes
        self.sigma = sigma
        self.s = s
        self.delta = delta
        self.alpha = alpha

    def gga_initialization(self, upper, lower, n_population, grid_intervals):
        """

        :param upper:
        :param lower:
        :param n_population:
        :return: this function returns the delta, alpha, s and space_s values for all the chromosomes of the population
        """

        # Get the delta values with the bounds of the problem
        self.delta = np.array((upper - lower) / grid_intervals)

        # Get the bounds of the discretized space
        upper_s = np.array(np.floor(upper / self.delta)) - 1
        lower_s = np.array(np.floor(lower / self.delta)) + 1

        # Get the points on the discretized space where the genes are
        self.s = np.array(
            [np.random.randint(lower_s[i, j], upper_s[i, j] + 1) for i in range(len(lower_s)) for j in
             range(len(lower_s[i]))]).reshape(self.delta.shape)

        # Randomly sample an alpha value
        self.alpha = np.array(
            [np.random.uniform(0, delta) for delta_array in self.delta for delta in delta_array]).reshape(self.s.shape)

        # Create the chromosome
        self.chromosomes = self.gga_chromosome()

        return upper_s.astype(int), lower_s.astype(int)

    def gga_chromosome(self, s=None, delta=None, alpha=None):
        """

        :param s:
        :param delta:
        :param alpha:
        :return: The function creates the chromosomes of the population of a grid-based genetic algorithm
         if not given it uses the parameters that the object has itself.
        """
        s = s if s is not None else self.s
        delta = delta if delta is not None else self.delta
        alpha = alpha if alpha is not None else self.alpha
        ga_tools.check(len(s) == len(delta) and len(delta) == len(alpha),
                       "Delta, Alpha and S must have the same number of elements (equal to the number of dimensions")
        return s * delta + alpha
