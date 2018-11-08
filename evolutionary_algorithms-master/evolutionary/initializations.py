"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""
from __future__ import division
import numpy as np


def uniform(n_population, lower, upper, N):
    """
    Function to initialize the population.
    Each member of the population is a candidate initialized randomly
    using the given uniform distribution.
    A number N of samples are generated between the lower(L) and upper(U) bounds.

    :param n_population: number of elements in the population
    :param lower: lower bounds
    :param upper: upper bounds
    :param N: N dimensions to solve the problem
    :return:
    """
    population = [np.random.uniform(lower[0], upper[0], N)]
    for i in range(1, n_population):
        population = np.concatenate((population, [np.random.uniform(lower[i], upper[i], N)]))
    return population


def permutation(n_population, N):
    """
    Function to initialize the population. Each member of the population
    is a candidate initialized randomly with a permutation of the number (N)
    :param n_population: number of elements in the population
    :param N: number of elements in the permutation
    :return:
    """
    population = [np.random.permutation(N)]
    for i in range(n_population - 1):
        population = np.concatenate((population, [np.random.permutation(N)]))
    return population
