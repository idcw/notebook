"""
    @Author: Manuel Lagunas
    @Personal_page: http://giga.cps.unizar.es/~mlagunas/
    @date: Feb - 2017
"""

import numpy as np
import matplotlib.pyplot as plt


class Function(object):
    """
    Python adaptation of a group of functionf by Manuel Lagunas.
    The functions are obtained from https://www.sfu.ca
    Feb 2017
    """

    def __init__(self, name):
        self.name = name

    def evaluate(self, population):
        # Returns the value of the function for a population
        raise NotImplementedError("Subclass must implement abstract method")

    def _check(self, assertion, message):
        """
        Test function that receives two vars.
        assertion is a boolean which should be true in order to avoid to throw an exception.
        message is an string with the error message to show if the exception is thrown
        If it doesn't pass the assertion it raises an exception.
        """
        if not assertion: raise Exception(message)

    def plot(self, lower, upper, samples):
        """
        2dplot the function considering a lower and upper bounds and the number of samples
        to sample between them.
        """
        # Import the neccessary libraries
        import matplotlib.pyplot as plt

        # Create the array of values between the bounds and calculate its fitness
        x = np.linspace(lower, upper, samples)
        fitness = np.empty(0)
        for i in x:
            fitness = np.append(fitness, self.evaluate(np.array([i])))

        # Plot the values and the fitness
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(x, fitness)
        fig.suptitle(self.name, fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Fitness")

    def plot3d(self, lower, upper, samples):
        """
        plot the surface generated by the given function between the lower and upper bounds.
        """

        # Import the necessary libraries
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

        # Create the three axis: x, y and fitness
        x = np.linspace(lower, upper, samples)
        y = np.linspace(lower, upper, samples)
        x, y = np.meshgrid(x, y)

        # Calculate the fitness
        fitness = np.empty([len(x), len(x[0])])
        for i in range(len(x)):
            for j in range(len(x[i])):
                fitness[i][j] = self.evaluate(np.array([x[i][j], y[i][j]]))

        # Plot the 3d surface
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(x, y, fitness, cmap=cm.jet, linewidth=0)

        # Name each axis and add a title
        fig.suptitle(self.name + " 3d", fontsize=14)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Fitness")
        ax.invert_yaxis()
        ax.invert_xaxis()
        # Make the 3d plot look like in www.sfu.ca/~ssurjano/
        ax.view_init(30, 25)

    def get_m_population(self, population):
        for i in range(len(population)):
            population[i] = population[i] * 2 ** (-(i + 1) + 1)
        return population
