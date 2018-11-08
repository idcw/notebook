"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

from __future__ import division
import numpy as np
import ga_tools as ga_tools


def shuffle(parents):
    # shuffle the parents to prevent any correlation
    shuffle = np.arange(len(parents))
    np.random.shuffle(shuffle)
    parents = parents[shuffle]

    # In case the length of the parents is not 2*n we remove the last element
    if len(parents) % 2 != 0:
        parents = parents[:-1]

    return parents


def one_point(parents, prob=1):
    """
    It recombines a pair of parents to generate their childrens.
    In order to do so it splits each parent in 2 halfs from the crossover point,
    then it combinates the halfs of each of them to generate two new children
    """

    # Check the input var and shuffle the elements
    ga_tools.check(len(parents) > 0, "The population cannot be an empty matrix")

    # Shuffle the array to avoid any relation between the individuals
    parents = shuffle(parents)

    # Iterate over the parents taking them two by two and store the generated children
    for i in range(0, len(parents), 2):
        # Apply the crossover function with probability prob
        if np.random.uniform(0, 1) <= prob:
            # Get the crossover point
            cp = np.random.randint(len(parents[i]))

            # R   ecombine to generate their children
            parents[i, cp:], parents[i + 1, cp:] = parents[i + 1, cp:], parents[i, cp:].copy()

    return parents


def one_point_gga(parents_s, parents_alpha, prob=1):
    """
    It recombines a pair of parents to generate their childrens.
    In order to do so it splits each parent in 2 halfs from the crossover point,
    then it combinates the halfs of each of them to generate two new children
    """

    # Check the input var and shuffle the elements
    ga_tools.check(len(parents_s) > 0, "The population S cannot be an empty matrix")
    ga_tools.check(len(parents_s) > 0, "The population Alpha cannot be an empty matrix")

    # Iterate over the parents taking them two by two and store the generated children
    for i in range(0, len(parents_s), 2):
        # Apply the crossover function with probability prob
        if np.random.uniform(0, 1) <= prob:
            # Get the crossover point
            cp = np.random.randint(len(parents_s[i]))

            # Recombine to generate their children
            parents_s[i, cp:], parents_s[i + 1, cp:] = parents_s[i + 1, cp:], parents_s[i, cp:].copy()
            parents_alpha[i, cp:], parents_alpha[i + 1, cp:] = parents_alpha[i + 1, cp:], parents_alpha[i, cp:].copy()

    return parents_s, parents_alpha

def one_point_permutation(parents, prob):
    """
    It recombines a pair of parents to generate their childrens.
    In order to do so it splits each parent in 2 halfs from the crossover point,
    then it combinates the halfs of each of them to generate two new children.
    Now the representation of each parent is a permutation of n elements, where n
    is len(parents[i]).
    """

    # Check the input var and shuffle the elements
    ga_tools.check(len(parents) > 0, "The population cannot be an empty matrix")

    # Shuffle the array to avoid any relation between the individuals
    parents = shuffle(parents)

    def add_parent(new_parent, parent):
        """
        Private auxiliary function to minimize the code.
        It appends to a given parents the values that it doesn't
        contain yet from the other parent's tail.
        """
        return np.hstack((new_parent, [x for x in parent if x not in new_parent]))

        # Iterate over the parents taking them two by two and store the generated children

    for i in range(0, len(parents), 2):

        # Apply the crossover function with probability prob
        if np.random.uniform(0, 1) <= prob:
            # Get a random crossover point
            crossover_point = np.random.randint(len(parents[i]))

            # split parents in 2 parts in the crossover point
            parent_1 = np.hsplit(parents[i], [crossover_point])
            parent_2 = np.hsplit(parents[i + 1], [crossover_point])

            # recombine to generate their children
            parents[i] = add_parent(parent_1[0], np.hstack((parent_2[1], parent_2[0])))
            parents[i + 1] = add_parent(parent_2[0], np.hstack((parent_1[1], parent_1[0])))

    # Return the generated children as a np array
    return parents


def two_point(parents, prob):
    """
    The following method recieves a pair of parents and the probability
    between [0,1] to apply the crossover function to them.
    It randomly sample 2 integers between 0 and the number of dimensions of the 
    parents. Then the two sub-arrays generated (one for each parent) are swapped
    generating 2 children.
    """

    # Check the input var and shuffle the elements
    ga_tools.check(len(parents) > 0, "The population cannot be an empty matrix")

    # Shuffle the array to avoid any relation between the individuals
    parents = shuffle(parents)

    # Iterate over the parents taking them two by two and store the generated children
    for i in range(0, len(parents), 2):

        # Apply the crossover function with probability prob
        if np.random.uniform(0, 1) <= prob:

            # Sample the 2 crossover points randomly
            cp1 = np.random.randint(len(parents[i]))
            cp2 = np.random.randint(len(parents[i]))

            # Avoid that cp1 and cp2 are equal
            while cp1 == cp2:
                cp2 = np.random.randint(len(parents[i]))

            # Swap if cp1 is bigger than cp2 (otherwise array slicing won't work)
            if cp1 > cp2:
                cp1, cp2 = cp2, cp1

            # Recombine to generate their children
            parents[i, cp1:cp2], parents[i + 1, cp1:cp2] = parents[i + 1, cp1:cp2], parents[i, cp1:cp2].copy()

    return parents


def blend(parents, prob, upper, lower, alpha=0.5):
    """
    The following method applies a blend crossover to a matrix of chromosomes.
    
    parents is the chromosomes matrix to apply the function
    prob is the probability to apply the recombination
    upper and lower are the upper and lower bounds of the population
    alpha is the parameter used for blend recombination
    """

    # Check the input var and shuffle the elements
    ga_tools.check(len(parents) > 0, "The population cannot be an empty matrix")

    # Shuffle the array to avoid any relation between the individuals
    parents = shuffle(parents)

    # In case the length of the parents is not 2*n we remove the last element
    if len(upper) % 2 != 0:
        upper = upper[:-1]
        lower = lower[:-1]

    # Iterate over the parents taking them two by two and store the generated children
    for i in range(0, len(parents), 2):

        # Recombine a pair of parents with probability prob
        if np.random.uniform(0, 1) <= prob:
            # Calculate the value of gamma
            gamma = (1. + 2. * alpha) * np.random.uniform(0, 1) - alpha
            # Generate the children and store them
            parents[i], parents[i + 1] = (1 - gamma) * parents[i] + gamma * parents[i + 1], \
                                         ((1 - gamma) * parents[i + 1] + gamma * parents[i]).copy()

    # Fix the chromosomes values that are out of bounds. First create a mask
    # set the masked chromosome values to 0, create a matrix with the value of the
    # bounds where the mask is 1 (true) and zero in the rest, at the end add the new
    # matrix to the parents 
    out = parents > upper
    parents[out] = 0
    parents += out.astype(int) * upper
    out = parents < lower
    parents[out] = 0
    parents += out.astype(int) * lower

    # Return the generated children
    return parents

    # def SPX(n, m, e=0.1):
    #
    #     O = 1/len(m) * np.sum(m, axis=0)
    #     print ("O", O)
    #     Y = (1+e) * (m - O)
    #     return Y
    #
    #
    # m = np.array([[1,1], [2,2], [3,3]])
    # print
    # print (SPX(2, m))
