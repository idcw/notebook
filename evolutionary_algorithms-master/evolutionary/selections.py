"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""
from __future__ import division
import numpy as np
import ga_tools as ga_tools


def _random_n(N, chromosomes):
    # shuffle the parents to prevent any correlation
    shuffle = np.arange(len(chromosomes))
    np.random.shuffle(shuffle)
    return shuffle[:N]


def tournament(fitness, N=5, M=2, iterations=1, minimize=True):
    """
    parents is an array of chromosomes
    fitness is the chromosomes fitness
    N is the number of parents randomly sampled from parents
    M is the number of parents sampled with the wheel selection
    iterations is the number of times we sample M parents
    
    This function returns the M best genes (parents) out of the N randomly sampled from the population
    first it gets the parents with smallest fitness
    second it gets its indexes in the population
    finally it returns their genes
    """

    # Initialize the array that we will return
    indices = np.array([])
    for i in range(iterations):
        # Select a subgroup of parents
        random_parents = _random_n(N, fitness)
        idx = ga_tools.n_sort(fitness[random_parents], M, minimize)
        indices = np.append(indices, random_parents[idx])

    # Return the indices as an array of integers
    return indices.astype(np.int64)


def wheel(fitness, M, replacement=True, minimize=True):
    """
    The wheel selection method sample M parents from the population. Each chromosome has a
    probability to be sampled equivalent to be the value of its fitness.

    :param parents: array of chromosomes
    :param fitness: fitness value of each chromosome
    :param M: number of elements to sample
    :param replacement: sample with replacement or not (boolean)
    :param minimize: minimization problem or maximization (boolean)
    :return: the sampled chromosomes and its index in the global matrix of chromosomes
    """

    # Get the probabilities of each chromosome
    wheel_prob = ga_tools.wheel_prob(fitness, minimize)

    # print fitness[random_parents], wheel_prob, np.sum(wheel_prob)
    # Sample M indices from random_parents with the calculated probabilities
    indices = np.random.choice(np.arange(0, len(fitness)),
                               M, replace=replacement, p=wheel_prob)

    # Return the indices as an array of integers
    indices = indices.astype(np.int64)
    return indices
