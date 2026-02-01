import random

import gym
import numpy as np

from gym_frozen_lake.common import LAKE_START, LAKE_PATH, LAKE_HOLE, LAKE_REWARD


class MazeObservationSpace(gym.Space):

    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 4
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def seed(self, seed=None):
        self.np_random.seed(seed)

    def sample(self):
        states = [LAKE_PATH, LAKE_HOLE, LAKE_REWARD]
        return tuple(random.choice(list(states)) for _ in range(self.n))

    @property
    def np_random(self):
        return np.random.RandomState()

    def contains(self, x):
        states = map(str, [LAKE_PATH, LAKE_HOLE, LAKE_REWARD, LAKE_START])
        return all(elem in states for elem in x)

    def to_jsonable(self, sample_n):
        return list(sample_n)

    def from_jsonable(self, sample_n):
        return tuple(sample_n)
