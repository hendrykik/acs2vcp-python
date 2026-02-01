from gym_maze import Maze
import numpy as np


class Woods100(Maze):
    def __init__(self):
        super().__init__(np.asarray([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 9, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]))
