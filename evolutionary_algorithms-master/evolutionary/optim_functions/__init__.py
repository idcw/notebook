# Make possible the sintax:
# from optim_functions import module
from .ackley import Ackley
from .griewank import Griewank
from .beale import Beale
from .booth import Booth
from .rothyp import Rothyp
from .forrester import Forrester
from .matyas import Matyas
from .powell import Powell
from .zakharov import Zakharov
from .easom import Easom
from .schwefel import Schwefel
from .rastrigin import Rastrigin
from .sphere import Sphere


# Make possible the sintax:
# from optim_functions import *
__all__ = ["Ackley", "Griewank", "Beale", "Booth", "Rothyp", "Forrester", "Matyas", "Powell", "Zakharov", "Easom", "Schwefel", "Rastrigin", "Sphere"]
