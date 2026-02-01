import numpy as np

from gym_woods.envs import AbstractWoods


class Woods101demi(AbstractWoods):
    def __init__(self):
        super().__init__(np.asarray([
            list('OOOOOOO'),
            list('O.OFO.O'),
            list('O.O.O.O'),
            list('OO.O.OO'),
            list('O.O.O.O'),
            list('OOOOOOO'),
            list('O.O.O.O'),
            list('OO.O.OO'),
            list('O.O.O.O'),
            list('O.OFO.O'),
            list('OOOOOOO'),
        ]))
