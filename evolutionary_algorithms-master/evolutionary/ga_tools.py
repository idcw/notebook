"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""
import numpy as np

U = np.random.uniform

def check(assertion, message):
    """
    Test function that receives two vars.
    assertion is a boolean which should be true in order to avoid to throw an exception
    message is an string with the error message to show if the exception is thrown
    If it doesn't pass the assertion it raises an exception.
    """
    try:
        assert assertion
    except AssertionError as e:
        e.args += str(message)
        raise


def wheel_prob(fitness, minimize):
    """
    :param fitness:
    :param minimize:
    :return:
    """
    if minimize:
        # Normalize the fitness matrix so it is suitable for a minimization problem
        norm_fitness = np.absolute(fitness - np.max(fitness)) + np.min(fitness)
    else:
        # Normalize the fitness matrix so it is suitable for a minimization problem
        norm_fitness = fitness - np.min(fitness)

    # Compute the probabilities proportionaly to the fitness
    return norm_fitness / np.sum(norm_fitness)


def n_sort(fitness, n, minimize):
    """


    :param fitness: fitness value of each chromosome
    :param n: number of chromosomes that are going to be returned
    :param minimize: minimization problem or maximization (boolean)
    :return: return n chromosomes from a sorted array regarding if it is a minimization
             or maximization problem
    """
    if minimize:
        return fitness.argsort()[:n]
    else:
        return fitness.argsort()[-n:][::-1]

def geometric(dispersion):
    """
    :param dispersion: dispersion parameter of the geometric distribution
    :return: the function creates a geometrical distribution variable from a uniform distribution
    """

    psi = 1.0 - (dispersion / (1.0 + np.sqrt(1.0 + dispersion ** 2.0)))
    return np.floor(np.log(1.0 - U(0, 1)) / np.log(1.0 - psi))
