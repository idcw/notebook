from __future__ import division
import numpy as np

from evolutionary import EAL, optim_functions as functions

seeds = np.array([82634, 16345, 12397, 84567, 34523, 65831, 40986, 8652, 12345, 98765, 19285, 97531,
                  52345, 12342, 8524, 13855, 10574, 10526, 342, 88529, 12385, 90874, 79432, 12953, 56372])

gga = EAL(
    goal=10 ** -4,
    minimization=False,
    n_dimensions=10,
    n_population=200,
    n_iterations=2000,
    n_children=200,
    xover_prob=0.8,
    mutat_prob=0.05,
    selection='tournament',
    crossover='one-point',
    mutation='gga-mutation',
    replacement='generational',
    grid_intervals=20,
    alpha_prob=0.9,
    control_alpha=10 ** -2,
    control_s=3,
    tournament_competitors=3,
    tournament_winners=1
)

gga.fit(ea_type="gga",
        problem=functions.Rastrigin, bounds=[-10, 10], pi_function=False, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Rastrigin, bounds=[-10, 10], pi_function=True, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Rastrigin, bounds=[-10, 10], pi_function=False, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Rastrigin, bounds=[-10, 10], pi_function=True, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Ackley, bounds=[-10, 10], pi_function=False, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Ackley, bounds=[-10, 10], pi_function=True, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Ackley, bounds=[-10, 10], pi_function=False, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Ackley, bounds=[-10, 10], pi_function=True, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Sphere, bounds=[-10, 10], pi_function=False, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Sphere, bounds=[-10, 10], pi_function=True, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Sphere, bounds=[-10, 10], pi_function=False, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Sphere, bounds=[-10, 10], pi_function=True, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Schwefel, bounds=[-500, 500], pi_function=False, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Schwefel, bounds=[-500, 500], pi_function=False, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Griewank, bounds=[-600, 600], pi_function=False, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Griewank, bounds=[-600, 600], pi_function=True, m_function=False,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Griewank, bounds=[-600, 600], pi_function=False, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)

gga.fit(ea_type="gga",
        problem=functions.Griewank, bounds=[-600, 600], pi_function=True, m_function=True,
        iter_log=-1,
        seeds=seeds,
        to_file=True)
