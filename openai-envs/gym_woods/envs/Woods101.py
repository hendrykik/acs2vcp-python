import numpy as np

from gym_woods.envs import AbstractWoods


class Woods101(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('OOOOOOO'),
            list('O.....O'),
            list('O.O.O.O'),
            list('O.OFO.O'),
            list('OOOOOOO'),
        ]))
